"""
Lightning module for Temporal Concat Diffusion Model
"""
import os
import torch
from torch import nn
import torch.nn.functional as F
import lightning as pl

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from CAMDM.diffusion.gaussian_diffusion import (
    GaussianDiffusion, ModelMeanType, ModelVarType, LossType
)

from signwriting_animation.diffusion.concat.concat_model import TemporalConcatDiffusion


def sanitize_btjc(x: torch.Tensor) -> torch.Tensor:
    """Sanitize pose tensor to BTJC format."""
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
    """Compute mean per-frame displacement."""
    x = sanitize_btjc(x_btjc)
    if x.size(1) < 2:
        return 0.0
    v = x[:, 1:] - x[:, :-1]
    return v.abs().mean().item()


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule for diffusion."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class _ConditionalWrapper(nn.Module):
    """Wrapper for GaussianDiffusion interface."""
    def __init__(self, base_model, past_bjct, sign_img):
        super().__init__()
        self.base_model = base_model
        self.past_bjct = past_bjct
        self.sign_img = sign_img

    def forward(self, x, t):
        return self.base_model(x, t, self.past_bjct, self.sign_img)


class LitConcatDiffusion(pl.LightningModule):
    """
    Lightning module for Temporal Concat Diffusion.
    
    和 LitDiffusion 的区别：使用 TemporalConcatDiffusion 模型
    """
    def __init__(
        self,
        num_keypoints: int = 178,
        num_dims: int = 3,
        lr: float = 1e-4,
        stats_path: str = "/home/yayun/data/pose_data/mean_std_178_with_preprocess.pt",
        diffusion_steps: int = 8,
        vel_weight: float = 1.0,
        acc_weight: float = 0.5,
        disp_weight: float = 1.0,
        t_past: int = 40,
        t_future: int = 20,
        num_latent_dims: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.diffusion_steps = diffusion_steps
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.disp_weight = disp_weight
        self._step_count = 0

        # Load normalization stats
        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1, 1, -1, 3)
        std = stats["std"].float().view(1, 1, -1, 3)
        self.register_buffer("mean_pose", mean.clone())
        self.register_buffer("std_pose", std.clone())

        # Create Temporal Concat model
        self.model = TemporalConcatDiffusion(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
            num_latent_dims=num_latent_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            t_past=t_past,
            t_future=t_future,
        )

        # Diffusion process
        betas = cosine_beta_schedule(diffusion_steps).numpy()
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        self.lr = lr
        self.train_logs = {"loss": [], "mse": [], "vel": [], "disp": [], "disp_ratio": []}

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

    def training_step(self, batch, _batch_idx):
        debug = self._step_count == 0 or self._step_count % 100 == 0

        cond_raw = batch["conditions"]
        gt_btjc = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(cond_raw["input_pose"])
        sign_img = cond_raw["sign_image"].float()

        gt_norm = self.normalize(gt_btjc)
        past_norm = self.normalize(past_btjc)

        batch_size = gt_norm.shape[0]
        device = gt_norm.device

        gt_bjct = self.btjc_to_bjct(gt_norm)
        past_bjct = self.btjc_to_bjct(past_norm)

        timestep = torch.randint(0, self.diffusion_steps, (batch_size,), device=device, dtype=torch.long)

        noise = torch.randn_like(gt_bjct)
        x_noisy = self.diffusion.q_sample(gt_bjct, timestep, noise=noise)

        # Model prediction
        pred_x0_bjct = self.model(x_noisy, timestep, past_bjct, sign_img)

        # Loss computation
        loss_mse = F.mse_loss(pred_x0_bjct, gt_bjct)

        pred_vel = pred_x0_bjct[..., 1:] - pred_x0_bjct[..., :-1]
        gt_vel = gt_bjct[..., 1:] - gt_bjct[..., :-1]
        loss_vel = F.mse_loss(pred_vel, gt_vel)

        loss_acc = torch.tensor(0.0, device=device)
        if pred_vel.size(-1) > 1:
            pred_acc = pred_vel[..., 1:] - pred_vel[..., :-1]
            gt_acc = gt_vel[..., 1:] - gt_vel[..., :-1]
            loss_acc = F.mse_loss(pred_acc, gt_acc)

        # Displacement loss
        pred_disp = pred_vel.abs().mean()
        gt_disp = gt_vel.abs().mean()
        loss_disp = torch.abs(pred_disp - gt_disp)
        disp_ratio = (pred_disp / (gt_disp + 1e-8)).item()

        loss = (loss_mse 
                + self.vel_weight * loss_vel 
                + self.acc_weight * loss_acc 
                + self.disp_weight * loss_disp)

        if debug:
            print("\n" + "=" * 70)
            print(f"TRAINING STEP {self._step_count} (Temporal Concat)")
            print("=" * 70)
            print(f"  t range: [{timestep.min().item()}, {timestep.max().item()}]")
            print(f"  loss_mse:  {loss_mse.item():.6f}")
            print(f"  loss_vel:  {loss_vel.item():.6f}")
            print(f"  loss_disp: {loss_disp.item():.6f}")
            print(f"  disp_ratio: {disp_ratio:.4f} (ideal=1.0)")
            print(f"  TOTAL: {loss.item():.6f}")
            print("=" * 70)

        self.log_dict({
            "train/loss": loss,
            "train/mse": loss_mse,
            "train/vel": loss_vel,
            "train/disp_loss": loss_disp,
            "train/disp_ratio": disp_ratio,
        }, prog_bar=True)

        self.train_logs["loss"].append(loss.item())
        self.train_logs["mse"].append(loss_mse.item())
        self.train_logs["vel"].append(loss_vel.item())
        self.train_logs["disp"].append(loss_disp.item())
        self.train_logs["disp_ratio"].append(disp_ratio)

        self._step_count += 1
        return loss

    @torch.no_grad()
    def sample(self, past_btjc, sign_img, future_len=20):
        self.eval()
        device = self.device

        past_raw = sanitize_btjc(past_btjc.to(device))
        past_norm = self.normalize(past_raw)
        past_bjct = self.btjc_to_bjct(past_norm)

        B, J, C, _ = past_bjct.shape
        target_shape = (B, J, C, future_len)

        wrapped = _ConditionalWrapper(self.model, past_bjct, sign_img)

        pred_bjct = self.diffusion.p_sample_loop(
            model=wrapped,
            shape=target_shape,
            clip_denoised=True,
            model_kwargs={"y": {}},
            progress=False,
        )

        pred_btjc = self.bjct_to_btjc(pred_bjct)
        return self.unnormalize(pred_btjc)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_train_end(self):
        out_dir = "logs/concat"
        os.makedirs(out_dir, exist_ok=True)

        if plt is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(self.train_logs["loss"])
        axes[0, 0].set_title("Total Loss")

        axes[0, 1].plot(self.train_logs["mse"])
        axes[0, 1].set_title("MSE Loss")

        axes[1, 0].plot(self.train_logs["disp"])
        axes[1, 0].set_title("Displacement Loss")

        axes[1, 1].plot(self.train_logs["disp_ratio"])
        axes[1, 1].set_title("Displacement Ratio (ideal=1.0)")
        axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{out_dir}/train_curve.png")
        print(f"[TRAIN CURVE] saved to {out_dir}/train_curve.png")