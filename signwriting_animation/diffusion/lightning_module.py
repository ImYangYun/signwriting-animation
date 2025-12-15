"""
PyTorch Lightning Module for V1 Model (Original CAMDM-based)

This Lightning module is adapted for the original SignWritingToPoseDiffusionV1 model
which returns (output, length_dist) tuple.
"""
# pylint: disable=invalid-name
# B, J, C, T are standard tensor dimension names in deep learning

import os
import torch
from torch import nn
import torch.nn.functional as F

import lightning as pl

from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure as PE_DTW
from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusionV2


def sanitize_btjc(x: torch.Tensor) -> torch.Tensor:
    """Sanitize pose tensor to ensure BTJC format [Batch, Time, Joints, Coords]."""
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
    """Convert batched BTJC tensor to list of TJC tensors using mask."""
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
    """Compute Dynamic Time Warping distance."""
    preds = _btjc_to_tjc_list(pred_btjc, mask_bt)
    tgts = _btjc_to_tjc_list(tgt_btjc, mask_bt)
    try:
        dtw_metric = PE_DTW()
    except (ImportError, RuntimeError):
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
    """Compute mean per-frame displacement (motion magnitude)."""
    x = sanitize_btjc(x_btjc)
    if x.size(1) < 2:
        return 0.0
    v = x[:, 1:] - x[:, :-1]
    return v.abs().mean().item()


def cosine_beta_schedule(timesteps, s=0.008):
    """Create cosine beta schedule for diffusion process."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class _ConditionalWrapperV1(nn.Module):
    """
    Wrapper for V1 model that only returns the pose output (not length_dist).
    
    GaussianDiffusion expects model(x, t) -> output
    But V1 model returns (output, length_dist)
    This wrapper extracts only the output.
    """
    def __init__(self, base_model: nn.Module, past_bjct: torch.Tensor, sign_img: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.past_bjct = past_bjct
        self.sign_img = sign_img
    
    def forward(self, x, t, **kwargs):  # pylint: disable=unused-argument
        """Forward with fixed conditions, return only pose output."""
        output, _length_dist = self.base_model(x, t, self.past_bjct, self.sign_img)
        return output  # Only return pose, ignore length_dist


class LitDiffusionV1(pl.LightningModule):  # pylint: disable=too-many-instance-attributes
    """
    PyTorch Lightning module for V1 (Original CAMDM-based) diffusion training.
    
    Differences from V2:
    - Model returns (output, length_dist) tuple
    - Uses all CAMDM components (MotionProcess, seq_encoder_factory, etc.)
    
    Args:
        num_keypoints: Number of pose keypoints
        num_dims: Dimensions per keypoint
        lr: Learning rate
        stats_path: Path to normalization statistics
        diffusion_steps: Number of diffusion timesteps
        vel_weight: Weight for velocity loss
        acc_weight: Weight for acceleration loss
        arch: Encoder architecture ("trans_enc", "trans_dec", or "gru")
        num_layers: Number of encoder layers
        ff_size: Feed-forward size
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_keypoints=178,
        num_dims=3,
        lr=1e-4,
        stats_path="/home/yayun/data/pose_data/mean_std_178_with_preprocess.pt",
        diffusion_steps=8,
        vel_weight: float = 1.0,
        acc_weight: float = 0.5,
        arch: str = "trans_enc",
        num_layers: int = 8,
        ff_size: int = 1024,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.diffusion_steps = diffusion_steps
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.arch = arch
        self._step_count = 0

        # Load normalization statistics
        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1, 1, -1, 3)
        std = stats["std"].float().view(1, 1, -1, 3)
        self.register_buffer("mean_pose", mean.clone())
        self.register_buffer("std_pose", std.clone())

        # Initialize V1 model (with all CAMDM components)
        self.model = SignWritingToPoseDiffusionV2(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
            arch=arch,
            num_layers=num_layers,
            ff_size=ff_size,
        )

        # Create Gaussian diffusion process
        betas = cosine_beta_schedule(diffusion_steps).numpy()
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        self.lr = lr
        self.train_logs = {"loss": [], "mse": [], "vel": [], "acc": [], "disp_ratio": []}

    def normalize(self, x):
        """Normalize pose to zero mean and unit variance."""
        return (x - self.mean_pose) / (self.std_pose + 1e-6)

    def unnormalize(self, x):
        """Denormalize pose back to original scale."""
        return x * self.std_pose + self.mean_pose

    @staticmethod
    def btjc_to_bjct(x):
        """Convert [B,T,J,C] to [B,J,C,T] format."""
        return x.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def bjct_to_btjc(x):
        """Convert [B,J,C,T] to [B,T,J,C] format."""
        return x.permute(0, 3, 1, 2).contiguous()

    def training_step(self, batch, _batch_idx):  # pylint: disable=arguments-differ,too-many-locals
        """Single training step for V1 diffusion model."""
        debug = self._step_count == 0 or self._step_count % 100 == 0

        # Extract data from batch
        cond_raw = batch["conditions"]
        gt_btjc = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(cond_raw["input_pose"])
        sign_img = cond_raw["sign_image"].float()

        # Normalize
        gt_norm = self.normalize(gt_btjc)
        past_norm = self.normalize(past_btjc)

        batch_size = gt_norm.shape[0]
        device = gt_norm.device

        # Convert to BJCT format
        gt_bjct = self.btjc_to_bjct(gt_norm)
        past_bjct = self.btjc_to_bjct(past_norm)

        # Sample random diffusion timestep
        timestep = torch.randint(0, self.diffusion_steps, (batch_size,), device=device, dtype=torch.long)

        # Forward diffusion: add noise
        noise = torch.randn_like(gt_bjct)
        x_noisy = self.diffusion.q_sample(gt_bjct, timestep, noise=noise)

        # Model prediction (V1 returns tuple)
        pred_x0_bjct, _length_dist = self.model(x_noisy, timestep, past_bjct, sign_img)

        # === Loss Computation ===
        loss_mse = F.mse_loss(pred_x0_bjct, gt_bjct)

        pred_vel = pred_x0_bjct[..., 1:] - pred_x0_bjct[..., :-1]
        gt_vel = gt_bjct[..., 1:] - gt_bjct[..., :-1]
        loss_vel = F.mse_loss(pred_vel, gt_vel)

        loss_acc = torch.tensor(0.0, device=device)
        if pred_vel.size(-1) > 1:
            pred_acc = pred_vel[..., 1:] - pred_vel[..., :-1]
            gt_acc = gt_vel[..., 1:] - gt_vel[..., :-1]
            loss_acc = F.mse_loss(pred_acc, gt_acc)

        loss = loss_mse + self.vel_weight * loss_vel + self.acc_weight * loss_acc

        # Displacement ratio monitoring
        with torch.no_grad():
            pred_disp = pred_vel.abs().mean().item()
            gt_disp = gt_vel.abs().mean().item()
            disp_ratio = pred_disp / (gt_disp + 1e-8)

        if debug:
            print("\n" + "=" * 70)
            print(f"TRAINING STEP {self._step_count} (V1 - arch={self.arch})")
            print("=" * 70)
            print(f"  t range: [{timestep.min().item()}, {timestep.max().item()}]")
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
        """Generate pose sequence using DDPM sampling (V1 version)."""
        self.eval()
        device = self.device

        past_raw = sanitize_btjc(past_btjc.to(device))
        past_norm = self.normalize(past_raw)
        past_bjct = self.btjc_to_bjct(past_norm)

        B, J, C, _ = past_bjct.shape
        target_shape = (B, J, C, future_len)

        # Wrap V1 model to only return pose output
        wrapped_model = _ConditionalWrapperV1(self.model, past_bjct, sign_img)

        pred_bjct = self.diffusion.p_sample_loop(
            model=wrapped_model,
            shape=target_shape,
            clip_denoised=False,
            model_kwargs={"y": {}},
            progress=False,
        )

        pred_btjc = self.bjct_to_btjc(pred_bjct)
        return self.unnormalize(pred_btjc)

    def on_train_start(self):  # pylint: disable=attribute-defined-outside-init
        """Move normalization buffers to correct device before training."""
        self.mean_pose = self.mean_pose.to(self.device)
        self.std_pose = self.std_pose.to(self.device)

    def configure_optimizers(self):
        """Configure AdamW optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_train_end(self):  # pylint: disable=import-outside-toplevel
        """Save training curves after training completes."""
        try:
            import matplotlib.pyplot as plt
            out_dir = "logs/diffusion_v1"
            os.makedirs(out_dir, exist_ok=True)

            _, axes = plt.subplots(2, 2, figsize=(10, 8))

            axes[0, 0].plot(self.train_logs["loss"])
            axes[0, 0].set_title(f"Total Loss (V1 - {self.arch})")

            axes[0, 1].plot(self.train_logs["mse"])
            axes[0, 1].set_title("MSE Loss")

            axes[1, 0].plot(self.train_logs["vel"])
            axes[1, 0].set_title("Velocity Loss")

            axes[1, 1].plot(self.train_logs["disp_ratio"])
            axes[1, 1].set_title("Displacement Ratio")
            axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.savefig(f"{out_dir}/train_curve_{self.arch}.png")
            print(f"[TRAIN CURVE] saved to {out_dir}/train_curve_{self.arch}.png")
        except Exception as error:  # pylint: disable=broad-exception-caught
            print(f"[TRAIN CURVE] failed: {error}")