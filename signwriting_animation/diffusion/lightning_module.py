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
    """确保输入是 [B, T, J, C] 格式"""
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


def mean_frame_disp(x_btjc: torch.Tensor) -> float:
    """计算平均帧间位移"""
    x = sanitize_btjc(x_btjc)
    if x.size(1) < 2:
        return 0.0
    v = x[:, 1:] - x[:, :-1]
    return v.abs().mean().item()


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine beta schedule"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class _ConditionalWrapper(nn.Module):
    """
    包装模型，固定条件输入，只接受 (x, t) 参数
    兼容 GaussianDiffusion.p_sample_loop 的接口
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
    Diffusion 训练模块
    
    关键改动：
    1. 使用简化版模型（无 LayerNorm 的 OutputProcessMLP）
    2. 使用 GaussianDiffusion.p_sample_loop 进行采样
    """

    def __init__(
        self,
        num_keypoints=178,
        num_dims=3,
        lr=1e-4,
        stats_path="/home/yayun/data/pose_data/mean_std_178_with_preprocess.pt",
        diffusion_steps=8,
        vel_weight: float = 1.0,
        acc_weight: float = 0.0,
        use_mean_pool: bool = True,
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

        # 模型（使用修复后的版本）
        self.model = SignWritingToPoseDiffusionV2(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
            use_mean_pool=use_mean_pool,
        )

        # Cosine beta schedule
        betas = cosine_beta_schedule(diffusion_steps).numpy()
        
        # GaussianDiffusion - 预测 x0
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        self.lr = lr
        self.train_logs = {"loss": [], "mse": [], "vel": [], "disp_ratio": []}

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
        """Diffusion 训练步骤"""
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

        # 随机采样 timestep
        t = torch.randint(0, self.diffusion_steps, (B,), device=device, dtype=torch.long)
        
        # 给 GT 加噪声
        noise = torch.randn_like(gt_bjct)
        x_t = self.diffusion.q_sample(gt_bjct, t, noise=noise)
        
        # 模型预测 x0
        pred_x0_bjct = self.model(x_t, t, past_bjct, sign_img)
        
        # Loss: MSE
        loss_mse = F.mse_loss(pred_x0_bjct, gt_bjct)
        
        # Velocity loss
        pred_vel = pred_x0_bjct[..., 1:] - pred_x0_bjct[..., :-1]
        gt_vel = gt_bjct[..., 1:] - gt_bjct[..., :-1]
        loss_vel = F.mse_loss(pred_vel, gt_vel)
        
        # Total loss
        loss = loss_mse + self.vel_weight * loss_vel

        # 计算 displacement ratio 用于监控
        with torch.no_grad():
            pred_disp = pred_vel.abs().mean().item()
            gt_disp = gt_vel.abs().mean().item()
            disp_ratio = pred_disp / (gt_disp + 1e-8)

        if debug:
            print(f"\n[Step {self._step_count}] loss={loss.item():.4f}, mse={loss_mse.item():.4f}, "
                  f"vel={loss_vel.item():.4f}, disp_ratio={disp_ratio:.4f}")

        self.log_dict({
            "train/loss": loss,
            "train/mse": loss_mse,
            "train/vel": loss_vel,
            "train/disp_ratio": disp_ratio,
        }, prog_bar=True)

        self.train_logs["loss"].append(loss.item())
        self.train_logs["mse"].append(loss_mse.item())
        self.train_logs["vel"].append(loss_vel.item())
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