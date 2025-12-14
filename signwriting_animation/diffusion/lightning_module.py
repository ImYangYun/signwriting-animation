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
    兼容 GaussianDiffusion.p_sample_loop
    """
    def __init__(self, base_model: nn.Module, past_bjct: torch.Tensor, sign_img: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.past_bjct = past_bjct
        self.sign_img = sign_img
    
    def forward(self, x, t, **kwargs):
        return self.base_model(x, t, self.past_bjct, self.sign_img)


class LitDiffusion(pl.LightningModule):
    """
    修复版 Diffusion Lightning 模块
    
    适配新的 SignWritingToPoseDiffusionV2（每帧独立解码）
    """

    def __init__(
        self,
        num_keypoints=178,
        num_dims=3,
        lr=1e-4,
        stats_path="/home/yayun/data/pose_data/mean_std_178_with_preprocess.pt",
        diffusion_steps=8,
        vel_weight: float = 1.0,
        acc_weight: float = 0.5,
        t_past: int = 40,
        t_future: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.diffusion_steps = diffusion_steps
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self._step_count = 0

        # 加载统计信息
        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1, 1, -1, 3)
        std = stats["std"].float().view(1, 1, -1, 3)
        self.register_buffer("mean_pose", mean.clone())
        self.register_buffer("std_pose", std.clone())

        # 模型（修复版：每帧独立解码，不需要 residual_scale）
        self.model = SignWritingToPoseDiffusionV2(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
            t_past=t_past,
            t_future=t_future,
        )

        # Cosine beta schedule + GaussianDiffusion
        betas = cosine_beta_schedule(diffusion_steps).numpy()
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.START_X,  # 预测 x0
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        self.lr = lr
        self.train_logs = {"loss": [], "mse": [], "vel": [], "acc": [], "disp_ratio": []}

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
        """Diffusion 训练（简化版）"""
        debug = self._step_count == 0 or self._step_count % 100 == 0

        cond_raw = batch["conditions"]
        gt_btjc = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(cond_raw["input_pose"])
        sign_img = cond_raw["sign_image"].float()

        # Normalize
        gt_norm = self.normalize(gt_btjc)
        past_norm = self.normalize(past_btjc)

        B, T_future, J, C = gt_norm.shape
        device = gt_norm.device

        # 转换为 BJCT 格式
        gt_bjct = self.btjc_to_bjct(gt_norm)
        past_bjct = self.btjc_to_bjct(past_norm)

        # 随机 timestep
        t = torch.randint(0, self.diffusion_steps, (B,), device=device, dtype=torch.long)
        
        # 加噪声
        noise = torch.randn_like(gt_bjct)
        x_t = self.diffusion.q_sample(gt_bjct, t, noise=noise)
        
        # 模型预测 x0
        pred_x0_bjct = self.model(x_t, t, past_bjct, sign_img)

        # MSE Loss
        loss_mse = F.mse_loss(pred_x0_bjct, gt_bjct)
        
        # Velocity Loss
        pred_vel = pred_x0_bjct[..., 1:] - pred_x0_bjct[..., :-1]
        gt_vel = gt_bjct[..., 1:] - gt_bjct[..., :-1]
        loss_vel = F.mse_loss(pred_vel, gt_vel)
        
        # Acceleration Loss
        loss_acc = torch.tensor(0.0, device=device)
        if pred_vel.size(-1) > 1:
            pred_acc = pred_vel[..., 1:] - pred_vel[..., :-1]
            gt_acc = gt_vel[..., 1:] - gt_vel[..., :-1]
            loss_acc = F.mse_loss(pred_acc, gt_acc)
        
        # Total Loss
        loss = loss_mse + self.vel_weight * loss_vel + self.acc_weight * loss_acc

        # 监控 disp_ratio（关键指标！）
        with torch.no_grad():
            pred_disp = pred_vel.abs().mean().item()
            gt_disp = gt_vel.abs().mean().item()
            disp_ratio = pred_disp / (gt_disp + 1e-8)

        if debug:
            print("\n" + "=" * 70)
            print(f"TRAINING STEP {self._step_count}")
            print("=" * 70)
            print(f"  t range: [{t.min().item()}, {t.max().item()}]")
            print(f"  loss_mse: {loss_mse.item():.6f}")
            print(f"  loss_vel: {loss_vel.item():.6f}")
            print(f"  loss_acc: {loss_acc.item():.6f}")
            print(f"  disp_ratio: {disp_ratio:.4f} (ideal=1.0)")
            print(f"  TOTAL: {loss.item():.6f}")
            print("=" * 70)

        self.log_dict({
            "train/loss": loss,
            "train/mse": loss_mse,
            "train/vel": loss_vel,
            "train/acc": loss_acc,
            "train/disp_ratio": disp_ratio,
        }, prog_bar=True)

        self.train_logs["loss"].append(loss.item())
        self.train_logs["mse"].append(loss_mse.item())
        self.train_logs["vel"].append(loss_vel.item())
        self.train_logs["acc"].append(loss_acc.item())
        self.train_logs["disp_ratio"].append(disp_ratio)

        self._step_count += 1
        return loss

    @torch.no_grad()
    def sample(self, past_btjc, sign_img, future_len=20):
        """使用 p_sample_loop 采样"""
        self.eval()
        device = self.device

        past_raw = sanitize_btjc(past_btjc.to(device))
        past_norm = self.normalize(past_raw)
        past_bjct = self.btjc_to_bjct(past_norm)

        B, J, C, _ = past_bjct.shape
        target_shape = (B, J, C, future_len)
        
        wrapped_model = _ConditionalWrapper(self.model, past_bjct, sign_img)
        
        pred_bjct = self.diffusion.p_sample_loop(
            model=wrapped_model,
            shape=target_shape,
            clip_denoised=False,
            model_kwargs={"y": {}},
            progress=False,
        )
        
        pred_btjc = self.bjct_to_btjc(pred_bjct)
        return self.unnormalize(pred_btjc)

    @torch.no_grad()
    def sample_with_cfg(self, past_btjc, sign_img, future_len=20, guidance_scale=2.0):
        """使用 Classifier-Free Guidance 采样"""
        self.eval()
        device = self.device

        past_raw = sanitize_btjc(past_btjc.to(device))
        past_norm = self.normalize(past_raw)
        past_bjct = self.btjc_to_bjct(past_norm)

        B, J, C, _ = past_bjct.shape
        target_shape = (B, J, C, future_len)
        
        # Conditional
        wrapped_cond = _ConditionalWrapper(self.model, past_bjct, sign_img)
        cond_pred = self.diffusion.p_sample_loop(
            model=wrapped_cond,
            shape=target_shape,
            clip_denoised=False,
            model_kwargs={"y": {}},
            progress=False,
        )
        
        # Unconditional
        uncond_past = torch.zeros_like(past_bjct)
        uncond_sign = torch.zeros_like(sign_img)
        wrapped_uncond = _ConditionalWrapper(self.model, uncond_past, uncond_sign)
        uncond_pred = self.diffusion.p_sample_loop(
            model=wrapped_uncond,
            shape=target_shape,
            clip_denoised=False,
            model_kwargs={"y": {}},
            progress=False,
        )
        
        # CFG
        pred_bjct = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
        
        pred_btjc = self.bjct_to_btjc(pred_bjct)
        return self.unnormalize(pred_btjc)

    @torch.no_grad()
    def sample_ddim(self, past_btjc, sign_img, future_len=20):
        """DDIM 采样（如果支持）"""
        self.eval()
        device = self.device

        past_raw = sanitize_btjc(past_btjc.to(device))
        past_norm = self.normalize(past_raw)
        past_bjct = self.btjc_to_bjct(past_norm)

        B, J, C, _ = past_bjct.shape
        target_shape = (B, J, C, future_len)
        
        wrapped_model = _ConditionalWrapper(self.model, past_bjct, sign_img)
        
        if hasattr(self.diffusion, 'ddim_sample_loop'):
            pred_bjct = self.diffusion.ddim_sample_loop(
                model=wrapped_model,
                shape=target_shape,
                clip_denoised=False,
                model_kwargs={"y": {}},
                progress=False,
            )
        else:
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
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_train_end(self):
        try:
            import matplotlib.pyplot as plt
            out_dir = "logs/diffusion"
            os.makedirs(out_dir, exist_ok=True)

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes[0, 0].plot(self.train_logs["loss"])
            axes[0, 0].set_title("Total Loss")
            axes[0, 1].plot(self.train_logs["mse"])
            axes[0, 1].set_title("MSE Loss")
            axes[1, 0].plot(self.train_logs["vel"])
            axes[1, 0].set_title("Velocity Loss")
            axes[1, 1].plot(self.train_logs["disp_ratio"])
            axes[1, 1].set_title("Displacement Ratio")
            axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"{out_dir}/train_curve.png")
            print(f"[TRAIN CURVE] saved to {out_dir}/train_curve.png")
        except Exception as e:
            print(f"[TRAIN CURVE] failed: {e}")


# Alias
LitMinimal = LitDiffusion