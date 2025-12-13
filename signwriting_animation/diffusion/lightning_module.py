# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import lightning as pl

from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure as PE_DTW
from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusionV2


def sanitize_btjc(x: torch.Tensor) -> torch.Tensor:
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    if hasattr(x, "tensor"):
        x = x.tensor
    if x.dim() == 5:
        x = x[:, :, 0]
    if x.dim() != 4:
        raise ValueError(f"Expected [B,T,J,C], got {tuple(x.shape)}")
    if x.shape[-1] != 3 and x.shape[-2] == 3:
        x = x.permute(0, 1, 3, 2)
    if x.shape[-1] != 3:
        raise ValueError(f"sanitize_btjc: last dim must be C=3, got {x.shape}")
    return x.contiguous().float()


def _btjc_to_tjc_list(x_btjc, mask_bt):
    x_btjc = sanitize_btjc(x_btjc)
    batch_size, seq_len, _, _ = x_btjc.shape
    mask_bt = (mask_bt > 0.5).float()
    seqs = []
    for b in range(batch_size):
        t = int(mask_bt[b].sum().item())
        t = max(0, min(t, seq_len))
        seqs.append(x_btjc[b, :t].contiguous())
    return seqs


@torch.no_grad()
def masked_dtw(pred_btjc, tgt_btjc, mask_bt):
    preds = _btjc_to_tjc_list(pred_btjc, mask_bt)
    tgts = _btjc_to_tjc_list(tgt_btjc, mask_bt)
    
    try:
        dtw_metric = PE_DTW()
    except:
        pred = sanitize_btjc(pred_btjc)
        tgt = sanitize_btjc(tgt_btjc)
        t_max = min(pred.size(1), tgt.size(1))
        return torch.mean((pred[:, :t_max] - tgt[:, :t_max]) ** 2)
    
    vals = []
    for p, g in zip(preds, tgts):
        if p.size(0) < 2 or g.size(0) < 2:
            continue
        pv = p.detach().cpu().numpy().astype("float32")[:, None, :, :]
        gv = g.detach().cpu().numpy().astype("float32")[:, None, :, :]
        vals.append(float(dtw_metric.get_distance(pv, gv)))
    if not vals:
        return torch.tensor(0.0, device=pred_btjc.device)
    return torch.tensor(vals, device=pred_btjc.device).mean()


def mean_frame_disp(x_btjc: torch.Tensor) -> float:
    x = sanitize_btjc(x_btjc)
    if x.size(1) < 2:
        return 0.0
    v = x[:, 1:] - x[:, :-1]
    return v.abs().mean().item()


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class _ConditionalWrapper(nn.Module):
    """
    包装模型，固定条件输入，只接受 (x, t) 参数
    这样可以兼容 GaussianDiffusion.p_sample_loop 的接口
    
    参考师姐的实现
    """
    def __init__(self, base_model: nn.Module, past_bjct: torch.Tensor, sign_img: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.past_bjct = past_bjct
        self.sign_img = sign_img
    
    def forward(self, x, t, **kwargs):
        """
        x: [B, J, C, T] (BJCT 格式，GaussianDiffusion 期望的)
        t: [B] timesteps
        """
        # 直接调用模型 forward
        return self.base_model(x, t, self.past_bjct, self.sign_img)


class LitDiffusion(pl.LightningModule):
    """
    修复版 Diffusion
    
    关键改动：使用 GaussianDiffusion.p_sample_loop 进行采样
    """

    def __init__(
        self,
        num_keypoints=178,
        num_dims=3,
        lr=1e-4,
        stats_path="/home/yayun/data/pose_data/mean_std_178_with_preprocess.pt",
        diffusion_steps=8,
        guidance_scale=0.0,  # CFG scale，0 表示不用 CFG
        residual_scale: float = 0.1,
        vel_weight: float = 0.5,
        acc_weight: float = 0.2,
        cond_drop_prob: float = 0.2,  # Condition dropout 概率，强迫模型学习 x_t
        t_zero_prob: float = 0.3,     # t=0 的概率（直接重建）
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.diffusion_steps = diffusion_steps
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.guidance_scale = guidance_scale
        self.cond_drop_prob = cond_drop_prob
        self.t_zero_prob = t_zero_prob
        self._step_count = 0

        # 加载统计信息
        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1, 1, -1, 3)
        std = stats["std"].float().view(1, 1, -1, 3)
        self.register_buffer("mean_pose", mean.clone())
        self.register_buffer("std_pose", std.clone())

        # 模型
        self.model = SignWritingToPoseDiffusionV2(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
            residual_scale=residual_scale,
        )

        # Cosine beta schedule
        betas = cosine_beta_schedule(diffusion_steps).numpy()
        
        # 使用 GaussianDiffusion
        # 回到预测 START_X（x0），因为预测 epsilon 需要更长的训练
        # 增加 diffusion steps 让采样更稳定
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.START_X,  # 预测 x0
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        self.lr = lr
        self.train_logs = {"loss": [], "mse": [], "vel": [], "acc": [], "disp": []}

    def normalize(self, x):
        return (x - self.mean_pose) / (self.std_pose + 1e-6)

    def unnormalize(self, x):
        return x * self.std_pose + self.mean_pose

    @staticmethod
    def btjc_to_bjct(x):
        """[B,T,J,C] -> [B,J,C,T]"""
        return x.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def bjct_to_btjc(x):
        """[B,J,C,T] -> [B,T,J,C]"""
        return x.permute(0, 3, 1, 2).contiguous()

    def training_step(self, batch, batch_idx):
        """
        Diffusion 训练
        """
        debug = self._step_count == 0 or self._step_count % 100 == 0

        cond_raw = batch["conditions"]
        gt_btjc = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(cond_raw["input_pose"])
        sign_img = cond_raw["sign_image"].float()
        
        # 获取 mask（关键修复！）
        # zero_pad_collator 会产生 MaskedTensor，需要提取 mask
        gt_mask = None
        past_mask = None
        
        if hasattr(batch["data"], "mask"):
            gt_mask = batch["data"].mask  # [B, T, J, C] or similar
        if hasattr(cond_raw["input_pose"], "mask"):
            past_mask = cond_raw["input_pose"].mask
        
        if debug:
            print(f"\n[MASK DEBUG]")
            print(f"  gt_mask: {gt_mask.shape if gt_mask is not None else 'None'}")
            print(f"  past_mask: {past_mask.shape if past_mask is not None else 'None'}")
            if gt_mask is not None:
                print(f"  gt_mask sum: {gt_mask.sum().item()}, total: {gt_mask.numel()}")
            if past_mask is not None:
                print(f"  past_mask sum: {past_mask.sum().item()}, total: {past_mask.numel()}")

        # Normalize
        gt_norm = self.normalize(gt_btjc)
        past_norm = self.normalize(past_btjc)

        B, T_future, J, C = gt_norm.shape
        device = gt_norm.device

        # 转换为 BJCT 格式（GaussianDiffusion 期望的）
        gt_bjct = self.btjc_to_bjct(gt_norm)  # [B, J, C, T]
        past_bjct = self.btjc_to_bjct(past_norm)  # [B, J, C, T_past]

        # 随机采样 timestep
        # 关键改动：一定概率用 t=0（直接重建），让模型学会保持运动
        if torch.rand(1).item() < self.t_zero_prob:
            t = torch.zeros(B, device=device, dtype=torch.long)
        else:
            t = torch.randint(1, self.diffusion_steps, (B,), device=device, dtype=torch.long)
        
        # 给 GT 加噪声
        noise = torch.randn_like(gt_bjct)
        x_t = self.diffusion.q_sample(gt_bjct, t, noise=noise)
        
        # Condition Dropout: 随机丢弃条件，强迫模型学习 x_t
        if torch.rand(1).item() < self.cond_drop_prob:
            past_input = torch.zeros_like(past_bjct)
            sign_input = torch.zeros_like(sign_img)
        else:
            past_input = past_bjct
            sign_input = sign_img
        
        # 模型预测 x0
        t_scaled = self.diffusion._scale_timesteps(t)
        pred_x0_bjct = self.model(x_t, t_scaled, past_input, sign_input)
        
        # 主 Loss：预测的 x0 vs 真实 x0（使用 mask！）
        if gt_mask is not None:
            # gt_mask 可能是 [B,T,J,C] 或 [B,T,J] 或 [B,T]
            # 需要转换为 BJCT 格式
            if gt_mask.dim() == 4:
                mask_bjct = gt_mask.permute(0, 2, 3, 1).float()  # [B,J,C,T]
            elif gt_mask.dim() == 3:
                # [B,T,J] -> [B,J,1,T]
                mask_bjct = gt_mask.permute(0, 2, 1).unsqueeze(2).float()
            elif gt_mask.dim() == 2:
                # [B,T] -> [B,1,1,T]
                mask_bjct = gt_mask.unsqueeze(1).unsqueeze(1).float()
            else:
                mask_bjct = None
            
            if mask_bjct is not None:
                # Masked MSE
                diff_sq = (pred_x0_bjct - gt_bjct) ** 2
                loss_main = (diff_sq * mask_bjct).sum() / (mask_bjct.sum() + 1e-8)
                if debug:
                    print(f"  [MASKED LOSS] mask coverage: {mask_bjct.mean().item():.4f}")
            else:
                loss_main = F.mse_loss(pred_x0_bjct, gt_bjct)
        else:
            loss_main = F.mse_loss(pred_x0_bjct, gt_bjct)
        
        # 转回 BTJC 计算辅助 loss
        pred_x0_btjc = self.bjct_to_btjc(pred_x0_bjct)
        gt_btjc_unnorm = self.unnormalize(gt_norm)
        pred_btjc_unnorm = self.unnormalize(pred_x0_btjc)
        
        loss_vel = torch.tensor(0.0, device=device)
        loss_acc = torch.tensor(0.0, device=device)
        loss_disp = torch.tensor(0.0, device=device)
        
        if pred_btjc_unnorm.size(1) > 1:
            v_pred = pred_btjc_unnorm[:, 1:] - pred_btjc_unnorm[:, :-1]
            v_gt = gt_btjc_unnorm[:, 1:] - gt_btjc_unnorm[:, :-1]
            loss_vel = F.mse_loss(v_pred, v_gt)
            
            if v_pred.size(1) > 1:
                a_pred = v_pred[:, 1:] - v_pred[:, :-1]
                a_gt = v_gt[:, 1:] - v_gt[:, :-1]
                loss_acc = F.mse_loss(a_pred, a_gt)
            
            # 关键：Displacement Ratio Loss
            # 使用 log 空间，让 ratio < 1 受到更大惩罚
            # log(0.01) ≈ -4.6, log(1) = 0, log(6) ≈ 1.8
            disp_pred = v_pred.abs().mean()
            disp_gt = v_gt.abs().mean()
            ratio = disp_pred / (disp_gt + 1e-8)
            
            # 简单记录 ratio，不作为 loss（多样本训练不需要强制）
            loss_disp = torch.tensor(0.0, device=device)
        
        # 权重配置（简化，类似师姐）
        mse_weight = 1.0
        disp_weight = 0.0  # 多样本训练不需要 disp loss
        loss = mse_weight * loss_main + self.vel_weight * loss_vel + self.acc_weight * loss_acc

        if debug:
            disp_pred = mean_frame_disp(pred_btjc_unnorm)
            disp_gt = mean_frame_disp(gt_btjc_unnorm)
            t_zero_count = (t == 0).sum().item()
            ratio_val = disp_pred / (disp_gt + 1e-8)
            print("\n" + "=" * 70)
            print(f"DIFFUSION TRAINING STEP {self._step_count} (Predict x0)")
            print("=" * 70)
            print(f"  t range: [{t.min().item()}, {t.max().item()}], t=0 count: {t_zero_count}/{B}")
            print(f"  loss weights: mse=1.0, vel={self.vel_weight}, acc={self.acc_weight}")
            print(f"  loss_main (x0): {loss_main.item():.6f}")
            print(f"  loss_vel: {loss_vel.item():.6f}")
            print(f"  loss_acc: {loss_acc.item():.6f}")
            print(f"  disp_pred: {disp_pred:.6f}, disp_gt: {disp_gt:.6f}, ratio: {ratio_val:.4f}")
            print(f"  TOTAL: {loss.item():.6f}")
            print("=" * 70)

        self.log_dict({
            "train/loss": loss,
            "train/mse": loss_main,
            "train/vel": loss_vel,
            "train/acc": loss_acc,
            "train/disp": loss_disp,
        }, prog_bar=True)

        self.train_logs["loss"].append(loss.item())
        self.train_logs["mse"].append(loss_main.item())
        self.train_logs["vel"].append(loss_vel.item())
        self.train_logs["acc"].append(loss_acc.item())
        self.train_logs["disp"].append(loss_disp.item())

        self._step_count += 1
        return loss

    @torch.no_grad()
    def sample(self, past_btjc, sign_img, future_len=20):
        """
        使用 GaussianDiffusion.p_sample_loop 进行采样（正确的方式）
        
        参考师姐的 _process_validation_batch 实现
        """
        self.eval()
        device = self.device

        past_raw = sanitize_btjc(past_btjc.to(device))
        past_norm = self.normalize(past_raw)
        past_bjct = self.btjc_to_bjct(past_norm)  # [B, J, C, T_past]

        B, J, C, _ = past_bjct.shape
        
        # 目标 shape: [B, J, C, T_future]
        target_shape = (B, J, C, future_len)
        
        # 包装模型，固定条件
        wrapped_model = _ConditionalWrapper(self.model, past_bjct, sign_img)
        
        # 使用 GaussianDiffusion 的 p_sample_loop（正确的采样！）
        # 注意：CAMDM 需要 model_kwargs 包含 'y' 键
        pred_bjct = self.diffusion.p_sample_loop(
            model=wrapped_model,
            shape=target_shape,
            clip_denoised=False,
            model_kwargs={"y": {}},  # CAMDM 需要这个键
            progress=False,
        )
        
        # 转回 BTJC
        pred_btjc = self.bjct_to_btjc(pred_bjct)
        
        return self.unnormalize(pred_btjc)

    @torch.no_grad()
    def sample_with_cfg(self, past_btjc, sign_img, future_len=20, guidance_scale=2.0):
        """
        使用 Classifier-Free Guidance 采样
        
        参考师姐的实现：
        chunk = uncond_chunk + guidance_scale * (cond_chunk - uncond_chunk)
        """
        self.eval()
        device = self.device

        past_raw = sanitize_btjc(past_btjc.to(device))
        past_norm = self.normalize(past_raw)
        past_bjct = self.btjc_to_bjct(past_norm)

        B, J, C, _ = past_bjct.shape
        target_shape = (B, J, C, future_len)
        
        # Conditional sampling
        wrapped_model_cond = _ConditionalWrapper(self.model, past_bjct, sign_img)
        cond_pred = self.diffusion.p_sample_loop(
            model=wrapped_model_cond,
            shape=target_shape,
            clip_denoised=False,
            model_kwargs={"y": {}},
            progress=False,
        )
        
        # Unconditional sampling (zero out conditions)
        uncond_past = torch.zeros_like(past_bjct)
        uncond_sign = torch.zeros_like(sign_img)
        wrapped_model_uncond = _ConditionalWrapper(self.model, uncond_past, uncond_sign)
        uncond_pred = self.diffusion.p_sample_loop(
            model=wrapped_model_uncond,
            shape=target_shape,
            clip_denoised=False,
            model_kwargs={"y": {}},
            progress=False,
        )
        
        # CFG combination
        pred_bjct = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
        
        pred_btjc = self.bjct_to_btjc(pred_bjct)
        return self.unnormalize(pred_btjc)

    @torch.no_grad()
    def sample_ddim(self, past_btjc, sign_img, future_len=20, ddim_steps=None):
        """
        使用 DDIM 采样（如果 GaussianDiffusion 支持）
        否则 fallback 到 p_sample_loop
        """
        self.eval()
        device = self.device

        past_raw = sanitize_btjc(past_btjc.to(device))
        past_norm = self.normalize(past_raw)
        past_bjct = self.btjc_to_bjct(past_norm)

        B, J, C, _ = past_bjct.shape
        target_shape = (B, J, C, future_len)
        
        wrapped_model = _ConditionalWrapper(self.model, past_bjct, sign_img)
        
        # 尝试使用 ddim_sample_loop
        if hasattr(self.diffusion, 'ddim_sample_loop'):
            pred_bjct = self.diffusion.ddim_sample_loop(
                model=wrapped_model,
                shape=target_shape,
                clip_denoised=False,
                model_kwargs={"y": {}},
                progress=False,
            )
        else:
            # Fallback to p_sample_loop
            pred_bjct = self.diffusion.p_sample_loop(
                model=wrapped_model,
                shape=target_shape,
                clip_denoised=False,
                model_kwargs={"y": {}},
                progress=False,
            )
        
        pred_btjc = self.bjct_to_btjc(pred_bjct)
        return self.unnormalize(pred_btjc)

    def on_train_start(self):
        self.mean_pose = self.mean_pose.to(self.device)
        self.std_pose = self.std_pose.to(self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer, "gradient_clip_val": 1.0}

    def on_train_end(self):
        try:
            import matplotlib.pyplot as plt
            out_dir = "logs/diffusion_real"
            os.makedirs(out_dir, exist_ok=True)

            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            ax[0, 0].plot(self.train_logs["loss"])
            ax[0, 0].set_title("loss")
            ax[0, 1].plot(self.train_logs["mse"])
            ax[0, 1].set_title("mse")
            ax[1, 0].plot(self.train_logs["vel"])
            ax[1, 0].set_title("vel")
            ax[1, 1].plot(self.train_logs["acc"])
            ax[1, 1].set_title("acc")
            plt.tight_layout()
            plt.savefig(f"{out_dir}/train_curve.png")
            print(f"[TRAIN CURVE] saved")
        except Exception as e:
            print(f"[TRAIN CURVE] failed: {e}")


# Alias
LitMinimal = LitDiffusion