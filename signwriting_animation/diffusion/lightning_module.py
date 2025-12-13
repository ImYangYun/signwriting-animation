# pylint: disable=invalid-name,arguments-differ,too-many-locals,too-many-instance-attributes,too-many-arguments,too-many-positional-arguments
"""
真正的 Diffusion 版本 - 参考师姐论文 (T=8 步, cosine schedule, 预测 x0)
"""
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import lightning as pl

try:
    from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure as PE_DTW
    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    PE_DTW = None

from CAMDM.diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
)

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


class LitDiffusion(pl.LightningModule):
    """
    真正的 Diffusion 版本
    
    训练：对 GT 加噪声，模型预测 x0
    推理：从纯噪声开始，多步去噪
    """

    def __init__(
        self,
        num_keypoints=178,
        num_dims=3,
        lr=1e-4,
        stats_path="/home/yayun/data/pose_data/mean_std_178_with_preprocess.pt",
        diffusion_steps=8,  # 师姐用 T=8
        pred_target="x0",   # 预测 x0（不是 epsilon）
        guidance_scale=0.0,
        residual_scale: float = 0.1,
        vel_weight: float = 0.5,
        acc_weight: float = 0.2,
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

        # 模型
        self.model = SignWritingToPoseDiffusionV2(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
            residual_scale=residual_scale,
        )

        # Cosine beta schedule (师姐论文)
        betas = cosine_beta_schedule(diffusion_steps).numpy()
        
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.START_X,  # 预测 x0
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        self.lr = lr
        self.guidance_scale = float(guidance_scale)
        self.train_logs = {"loss": [], "mse": [], "vel": [], "acc": []}

    def normalize(self, x):
        return (x - self.mean_pose) / (self.std_pose + 1e-6)

    def unnormalize(self, x):
        return x * self.std_pose + self.mean_pose

    @staticmethod
    def btjc_to_bjct(x):
        return x.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def bjct_to_btjc(x):
        return x.permute(0, 3, 1, 2).contiguous()

    def _model_forward(self, x_t_btjc, t, past_btjc, sign_img):
        """
        模型前向：输入带噪声的 x_t，预测干净的 x0
        """
        x_t_bjct = self.btjc_to_bjct(x_t_btjc)
        past_bjct = self.btjc_to_bjct(past_btjc)
        
        # t 需要 scale
        t_scaled = self.diffusion._scale_timesteps(t)
        
        pred_bjct = self.model.forward(x_t_bjct, t_scaled, past_bjct, sign_img)
        pred_btjc = self.bjct_to_btjc(pred_bjct)
        
        return pred_btjc

    def training_step(self, batch, batch_idx):
        """
        真正的 Diffusion 训练：
        1. 随机采样 t
        2. 给 GT 加噪声得到 x_t
        3. 模型从 x_t 预测 x0
        4. Loss = MSE(pred_x0, gt)
        """
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

        # ===== 关键改动：真正的 Diffusion 训练 =====
        
        # 1. 随机采样 timestep t ∈ [0, T)
        t = torch.randint(0, self.diffusion_steps, (B,), device=device, dtype=torch.long)
        
        # 2. 给 GT 加噪声
        noise = torch.randn_like(gt_norm)
        x_t = self.diffusion.q_sample(gt_norm, t, noise=noise)
        
        # 3. 模型预测 x0
        pred_x0 = self._model_forward(x_t, t, past_norm, sign_img)
        
        # 4. 主 Loss：预测 x0
        loss_main = F.mse_loss(pred_x0, gt_norm)
        
        # ===== 辅助 Loss =====
        pred_raw = self.unnormalize(pred_x0)
        gt_raw = self.unnormalize(gt_norm)
        
        loss_vel = torch.tensor(0.0, device=device)
        loss_acc = torch.tensor(0.0, device=device)
        
        if pred_raw.size(1) > 1:
            v_pred = pred_raw[:, 1:] - pred_raw[:, :-1]
            v_gt = gt_raw[:, 1:] - gt_raw[:, :-1]
            loss_vel = F.mse_loss(v_pred, v_gt)
            
            if v_pred.size(1) > 1:
                a_pred = v_pred[:, 1:] - v_pred[:, :-1]
                a_gt = v_gt[:, 1:] - v_gt[:, :-1]
                loss_acc = F.mse_loss(a_pred, a_gt)
        
        loss = loss_main + self.vel_weight * loss_vel + self.acc_weight * loss_acc

        if debug:
            disp_pred = mean_frame_disp(pred_raw)
            disp_gt = mean_frame_disp(gt_raw)
            print("\n" + "=" * 70)
            print(f"DIFFUSION TRAINING STEP {self._step_count}")
            print("=" * 70)
            print(f"  t range: [{t.min().item()}, {t.max().item()}]")
            print(f"  loss_main: {loss_main.item():.6f}")
            print(f"  loss_vel: {loss_vel.item():.6f}")
            print(f"  loss_acc: {loss_acc.item():.6f}")
            print(f"  disp_pred: {disp_pred:.6f}, disp_gt: {disp_gt:.6f}")
            print(f"  TOTAL: {loss.item():.6f}")
            print("=" * 70)

        self.log_dict({
            "train/loss": loss,
            "train/mse": loss_main,
            "train/vel": loss_vel,
            "train/acc": loss_acc,
        }, prog_bar=True)

        self.train_logs["loss"].append(loss.item())
        self.train_logs["mse"].append(loss_main.item())
        self.train_logs["vel"].append(loss_vel.item())
        self.train_logs["acc"].append(loss_acc.item())

        self._step_count += 1
        return loss

    @torch.no_grad()
    def sample(self, past_btjc, sign_img, future_len=20):
        """
        改进的 Diffusion 采样：从 past_last 开始，而不是纯噪声
        
        这样模型可以基于历史位置进行去噪，避免完全静态的问题
        """
        self.eval()
        device = self.device

        past_raw = sanitize_btjc(past_btjc.to(device))
        past_norm = self.normalize(past_raw)

        B, _, J, C = past_norm.shape

        past_last = past_norm[:, -1:, :, :]  # [B, 1, J, C]
        past_last_expanded = past_last.expand(-1, future_len, -1, -1)  # [B, T, J, C]
        
        # 加上对应 t=T-1 的噪声水平
        alpha_bar_T = self.diffusion.alphas_cumprod[-1]
        noise = torch.randn(B, future_len, J, C, device=device)
        x_t = (
            torch.sqrt(torch.tensor(alpha_bar_T, device=device)) * past_last_expanded
            + torch.sqrt(torch.tensor(1 - alpha_bar_T, device=device)) * noise
        )

        # T 步去噪（从 T-1 到 0）
        for i in reversed(range(self.diffusion_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            # 模型预测 x0
            pred_x0 = self._model_forward(x_t, t, past_norm, sign_img)
            
            # 用 diffusion 的 p_sample 去噪一步
            # 这里我们手动实现简化版本
            if i > 0:
                # 计算 x_{t-1}
                alpha_bar_t = self.diffusion.alphas_cumprod[i]
                alpha_bar_t_prev = self.diffusion.alphas_cumprod[i - 1]
                
                # 简化的 DDPM 采样
                beta_t = 1 - alpha_bar_t / alpha_bar_t_prev
                
                # x_{t-1} = sqrt(alpha_bar_{t-1}) * pred_x0 + sqrt(1 - alpha_bar_{t-1}) * noise
                noise = torch.randn_like(x_t) if i > 1 else torch.zeros_like(x_t)
                x_t = (
                    torch.sqrt(torch.tensor(alpha_bar_t_prev, device=device)) * pred_x0
                    + torch.sqrt(torch.tensor(1 - alpha_bar_t_prev, device=device)) * noise
                )
            else:
                # 最后一步直接用 pred_x0
                x_t = pred_x0

        return self.unnormalize(x_t)

    @torch.no_grad()
    def sample_ddim(self, past_btjc, sign_img, future_len=20, ddim_steps=None):
        """
        改进的 DDIM 采样：从 past_last 开始
        """
        self.eval()
        device = self.device

        past_raw = sanitize_btjc(past_btjc.to(device))
        past_norm = self.normalize(past_raw)

        B, _, J, C = past_norm.shape

        if ddim_steps is None:
            ddim_steps = self.diffusion_steps
        
        step_size = self.diffusion_steps // ddim_steps
        timesteps = list(range(0, self.diffusion_steps, step_size))[::-1]

        # 改进：从 past_last 开始
        past_last = past_norm[:, -1:, :, :]
        past_last_expanded = past_last.expand(-1, future_len, -1, -1)
        
        alpha_bar_T = self.diffusion.alphas_cumprod[-1]
        noise = torch.randn(B, future_len, J, C, device=device)
        x_t = (
            torch.sqrt(torch.tensor(alpha_bar_T, device=device)) * past_last_expanded
            + torch.sqrt(torch.tensor(1 - alpha_bar_T, device=device)) * noise
        )

        for i, t_cur in enumerate(timesteps):
            t = torch.full((B,), t_cur, device=device, dtype=torch.long)
            
            # 模型预测 x0
            pred_x0 = self._model_forward(x_t, t, past_norm, sign_img)
            
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_bar_t = self.diffusion.alphas_cumprod[t_cur]
                alpha_bar_t_next = self.diffusion.alphas_cumprod[t_next]
                
                # DDIM 确定性采样
                x_t = (
                    torch.sqrt(torch.tensor(alpha_bar_t_next, device=device)) * pred_x0
                    + torch.sqrt(torch.tensor(1 - alpha_bar_t_next, device=device)) 
                    * (x_t - torch.sqrt(torch.tensor(alpha_bar_t, device=device)) * pred_x0)
                    / torch.sqrt(torch.tensor(1 - alpha_bar_t, device=device))
                )
            else:
                x_t = pred_x0

        return self.unnormalize(x_t)

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


LitMinimal = LitDiffusion