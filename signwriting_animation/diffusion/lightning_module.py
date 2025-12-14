import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy
import lightning as pl

from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure as PE_DTW
from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusionV2


def sanitize_btjc(x: torch.Tensor) -> torch.Tensor:
    """
    Sanitize pose tensor to ensure BTJC format [Batch, Time, Joints, Coords].
    
    Handles various input formats and ensures output is [B, T, J, C=3].
    
    Args:
        x: Input tensor (may have masks, wrong dimensions, etc.)
        
    Returns:
        Sanitized tensor in BTJC format with C=3
        
    Raises:
        ValueError: If tensor cannot be converted to valid BTJC format
    """
    # Remove mask if present
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    if hasattr(x, "tensor"):
        x = x.tensor

    # Remove extra person dimension if present
    if x.dim() == 5:
        x = x[:, :, 0]

    # Check dimensionality
    if x.dim() != 4:
        raise ValueError(f"Expected [B,T,J,C], got {tuple(x.shape)}")

    # Swap dimensions if C and J are reversed
    if x.shape[-1] != 3 and x.shape[-2] == 3:
        x = x.permute(0, 1, 3, 2)

    # Final validation
    if x.shape[-1] != 3:
        raise ValueError(f"sanitize_btjc: last dim must be C=3, got {x.shape}")

    return x.contiguous().float()


def _btjc_to_tjc_list(x_btjc, mask_bt):
    """
    Convert batched BTJC tensor to list of TJC tensors using mask.
    
    Useful for metrics like DTW that require variable-length sequences.
    
    Args:
        x_btjc: Pose tensor [B, T, J, C]
        mask_bt: Binary mask [B, T] indicating valid frames
        
    Returns:
        List of tensors, each [T_i, J, C] for valid frames only
    """
    x_btjc = sanitize_btjc(x_btjc)
    batch_size, seq_len, _, _ = x_btjc.shape
    mask_bt = (mask_bt > 0.5).float()

    seqs = []
    for b in range(batch_size):
        # Get valid sequence length from mask
        t = int(mask_bt[b].sum().item())
        t = max(0, min(t, seq_len))
        seqs.append(x_btjc[b, :t].contiguous())
    return seqs


@torch.no_grad()
def masked_dtw(pred_btjc, tgt_btjc, mask_bt):
    """
    Compute Dynamic Time Warping distance between predictions and targets.
    
    DTW measures temporal alignment quality, crucial for evaluating motion
    sequences where timing matters.
    
    Args:
        pred_btjc: Predicted poses [B, T, J, C]
        tgt_btjc: Ground truth poses [B, T, J, C]
        mask_bt: Valid frame mask [B, T]
        
    Returns:
        Mean DTW distance across batch (scalar tensor)
    """
    # Convert to variable-length sequences
    preds = _btjc_to_tjc_list(pred_btjc, mask_bt)
    tgts = _btjc_to_tjc_list(tgt_btjc, mask_bt)

    # Try to use DTW metric, fall back to MSE if unavailable
    try:
        dtw_metric = PE_DTW()
    except:
        # Fallback: use simple MSE
        pred = sanitize_btjc(pred_btjc)
        tgt = sanitize_btjc(tgt_btjc)
        t_max = min(pred.size(1), tgt.size(1))
        return torch.mean((pred[:, :t_max] - tgt[:, :t_max]) ** 2)

    # Compute DTW for each sequence pair
    vals = []
    for p, g in zip(preds, tgts):
        # Skip sequences that are too short
        if p.size(0) < 2 or g.size(0) < 2:
            continue
    
        # Convert to numpy and add person dimension for DTW metric
        pv = p.detach().cpu().numpy().astype("float32")[:, None, :, :]
        gv = g.detach().cpu().numpy().astype("float32")[:, None, :, :]
        vals.append(float(dtw_metric.get_distance(pv, gv)))

    # Return mean or zero if no valid sequences
    if not vals:
        return torch.tensor(0.0, device=pred_btjc.device)
    return torch.tensor(vals, device=pred_btjc.device).mean()


def mean_frame_disp(x_btjc: torch.Tensor) -> float:
    """
    Compute mean per-frame displacement (motion magnitude).
    
    This measures how much motion is present in the sequence.
    Critical for detecting "static pose" collapse.
    
    Args:
        x_btjc: Pose sequence [B, T, J, C]
        
    Returns:
        Mean absolute displacement across all frames and joints
    """
    x = sanitize_btjc(x_btjc)
    if x.size(1) < 2:
        return 0.0

    # Compute frame-to-frame differences
    v = x[:, 1:] - x[:, :-1]
    return v.abs().mean().item()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Create cosine beta schedule for diffusion process.
    
    Cosine schedule is gentler than linear, leading to better sample quality.
    Reference: "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
    
    Args:
        timesteps: Number of diffusion timesteps
        s: Offset parameter for cosine schedule (default 0.008)
        
    Returns:
        Beta values [timesteps] for forward diffusion process
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)

    # Cosine schedule for alphas_cumprod
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    # Derive betas from alphas_cumprod
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    # Clip to valid range
    return torch.clip(betas, 0.0001, 0.9999)


class _ConditionalWrapper(nn.Module):
    """
    Wrapper to fix conditions for GaussianDiffusion sampling.

    GaussianDiffusion.p_sample_loop expects model(x, t), but our model needs
    additional conditions (past_motion, sign_image). This wrapper fixes those
    conditions so the model becomes compatible with the sampling loop.
    
    Args:
        base_model: The actual diffusion model
        past_bjct: Past motion context [B, J, C, T]
        sign_img: SignWriting image [B, 3, H, W]
    """
    def __init__(self, base_model: nn.Module, past_bjct: torch.Tensor, sign_img: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.past_bjct = past_bjct
        self.sign_img = sign_img

    def forward(self, x, t, **kwargs):
        """Forward with fixed conditions."""
        return self.base_model(x, t, self.past_bjct, self.sign_img)


class LitDiffusion(pl.LightningModule):
    """
    PyTorch Lightning module for SignWriting-to-Pose diffusion training.
    
    Handles:
    - Training with multiple loss terms (MSE, velocity, acceleration)
    - Inference with multiple sampling strategies (DDPM, DDIM, CFG)
    - Automatic pose normalization/denormalization
    - Comprehensive evaluation metrics
    
    Args:
        num_keypoints: Number of pose keypoints (default 178 for MediaPipe Holistic)
        num_dims: Dimensions per keypoint (default 3 for x,y,z)
        lr: Learning rate for AdamW optimizer
        stats_path: Path to normalization statistics (mean/std)
        diffusion_steps: Number of diffusion timesteps
        vel_weight: Weight for velocity loss term
        acc_weight: Weight for acceleration loss term
        t_past: Number of past frames for context
        t_future: Number of future frames to predict
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

        # Load normalization statistics (mean and std for each keypoint)
        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1, 1, -1, 3)
        std = stats["std"].float().view(1, 1, -1, 3)
        self.register_buffer("mean_pose", mean.clone())
        self.register_buffer("std_pose", std.clone())

        # Initialize diffusion model (V2: frame-independent decoding)
        self.model = SignWritingToPoseDiffusionV2(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
            t_past=t_past,
            t_future=t_future,
        )

        # Create Gaussian diffusion process with cosine beta schedule
        betas = cosine_beta_schedule(diffusion_steps).numpy()
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.START_X,  # Predict x0 (denoised pose)
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        self.lr = lr

        # Training logs for visualization
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

    def training_step(self, batch, batch_idx):
        """
        Single training step for diffusion model.
        
        Process:
        1. Normalize ground truth and conditions
        2. Sample random timestep t
        3. Add noise to ground truth: q(x_t | x_0)
        4. Predict x_0 from x_t: p(x_0 | x_t, conditions)
        5. Compute losses: MSE, velocity, acceleration
        
        Args:
            batch: Dictionary with 'data' (GT) and 'conditions'
            batch_idx: Batch index (unused)
            
        Returns:
            Total loss (scalar tensor)
        """
        debug = self._step_count == 0 or self._step_count % 100 == 0

        # Extract data from batch
        cond_raw = batch["conditions"]
        gt_btjc = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(cond_raw["input_pose"])
        sign_img = cond_raw["sign_image"].float()

        # Normalize to standard distribution
        gt_norm = self.normalize(gt_btjc)
        past_norm = self.normalize(past_btjc)

        B, T_future, J, C = gt_norm.shape
        device = gt_norm.device

        # Convert to BJCT format for model
        gt_bjct = self.btjc_to_bjct(gt_norm)
        past_bjct = self.btjc_to_bjct(past_norm)

        # Sample random diffusion timestep for each batch element
        t = torch.randint(0, self.diffusion_steps, (B,), device=device, dtype=torch.long)

        # Forward diffusion: add noise to ground truth
        noise = torch.randn_like(gt_bjct)
        x_t = self.diffusion.q_sample(gt_bjct, t, noise=noise)

        # Model prediction: denoise x_t to predict x_0
        pred_x0_bjct = self.model(x_t, t, past_bjct, sign_img)

        # === Loss Computation ===
        
        # 1. MSE Loss: position accuracy
        loss_mse = F.mse_loss(pred_x0_bjct, gt_bjct)

        # 2. Velocity Loss: motion smoothness and magnitude
        pred_vel = pred_x0_bjct[..., 1:] - pred_x0_bjct[..., :-1]
        gt_vel = gt_bjct[..., 1:] - gt_bjct[..., :-1]
        loss_vel = F.mse_loss(pred_vel, gt_vel)

        # 3. Acceleration Loss: motion dynamics
        loss_acc = torch.tensor(0.0, device=device)
        if pred_vel.size(-1) > 1:
            pred_acc = pred_vel[..., 1:] - pred_vel[..., :-1]
            gt_acc = gt_vel[..., 1:] - gt_vel[..., :-1]
            loss_acc = F.mse_loss(pred_acc, gt_acc)

        # Total weighted loss
        loss = loss_mse + self.vel_weight * loss_vel + self.acc_weight * loss_acc

        # === Displacement Ratio Monitoring ===
        # Critical metric: detects if model collapses to static poses
        # Ideal ratio = 1.0 (predicted motion matches GT motion)
        with torch.no_grad():
            pred_disp = pred_vel.abs().mean().item()
            gt_disp = gt_vel.abs().mean().item()
            disp_ratio = pred_disp / (gt_disp + 1e-8)

        # Debug logging
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

        # Log metrics to Lightning
        self.log_dict({
            "train/loss": loss,
            "train/mse": loss_mse,
            "train/vel": loss_vel,
            "train/acc": loss_acc,
            "train/disp_ratio": disp_ratio,
        }, prog_bar=True)

        # Store for training curve visualization
        self.train_logs["loss"].append(loss.item())
        self.train_logs["mse"].append(loss_mse.item())
        self.train_logs["vel"].append(loss_vel.item())
        self.train_logs["acc"].append(loss_acc.item())
        self.train_logs["disp_ratio"].append(disp_ratio)

        self._step_count += 1
        return loss

    @torch.no_grad()
    def sample(self, past_btjc, sign_img, future_len=20):
        """
        Generate pose sequence using DDPM sampling.
        
        Standard diffusion sampling from pure noise to clean pose.
        
        Args:
            past_btjc: Past motion context [B, T, J, C]
            sign_img: SignWriting condition [B, 3, H, W]
            future_len: Number of frames to generate
            
        Returns:
            Generated pose sequence [B, T, J, C] in original scale
        """
        self.eval()
        device = self.device

        # Normalize past motion
        past_raw = sanitize_btjc(past_btjc.to(device))
        past_norm = self.normalize(past_raw)
        past_bjct = self.btjc_to_bjct(past_norm)

        B, J, C, _ = past_bjct.shape
        target_shape = (B, J, C, future_len)

        # Wrap model with fixed conditions
        wrapped_model = _ConditionalWrapper(self.model, past_bjct, sign_img)

        # Run DDPM sampling loop
        pred_bjct = self.diffusion.p_sample_loop(
            model=wrapped_model,
            shape=target_shape,
            clip_denoised=False,
            model_kwargs={"y": {}},
            progress=False,
        )

        # Convert back to BTJC and denormalize
        pred_btjc = self.bjct_to_btjc(pred_bjct)
        return self.unnormalize(pred_btjc)

    @torch.no_grad()
    def sample_with_cfg(self, past_btjc, sign_img, future_len=20, guidance_scale=2.0):
        """
        Generate with Classifier-Free Guidance for better conditioning.
        
        CFG formula: pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
        Higher guidance_scale means stronger conditioning.
        
        Args:
            past_btjc: Past motion [B, T, J, C]
            sign_img: SignWriting image [B, 3, H, W]
            future_len: Sequence length to generate
            guidance_scale: Strength of conditioning (1.0 = no guidance)
            
        Returns:
            CFG-enhanced prediction [B, T, J, C]
        """
        self.eval()
        device = self.device

        # Normalize conditions
        past_raw = sanitize_btjc(past_btjc.to(device))
        past_norm = self.normalize(past_raw)
        past_bjct = self.btjc_to_bjct(past_norm)

        B, J, C, _ = past_bjct.shape
        target_shape = (B, J, C, future_len)

        # Sample with conditions
        wrapped_cond = _ConditionalWrapper(self.model, past_bjct, sign_img)
        cond_pred = self.diffusion.p_sample_loop(
            model=wrapped_cond,
            shape=target_shape,
            clip_denoised=False,
            model_kwargs={"y": {}},
            progress=False,
        )

        # Sample without conditions (zeros)
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

        # Apply CFG formula
        pred_bjct = uncond_pred + guidance_scale * (cond_pred - uncond_pred)

        # Convert and denormalize
        pred_btjc = self.bjct_to_btjc(pred_bjct)
        return self.unnormalize(pred_btjc)

    @torch.no_grad()
    def sample_ddim(self, past_btjc, sign_img, future_len=20):
        """
        Generate using DDIM (faster sampling, deterministic).
        
        DDIM allows fewer steps than DDPM while maintaining quality.
        Falls back to DDPM if DDIM not available.
        
        Args:
            past_btjc: Past motion [B, T, J, C]
            sign_img: SignWriting image [B, 3, H, W]
            future_len: Sequence length
            
        Returns:
            Generated sequence [B, T, J, C]
        """
        self.eval()
        device = self.device

        # Normalize conditions
        past_raw = sanitize_btjc(past_btjc.to(device))
        past_norm = self.normalize(past_raw)
        past_bjct = self.btjc_to_bjct(past_norm)

        B, J, C, _ = past_bjct.shape
        target_shape = (B, J, C, future_len)

        # Wrap model
        wrapped_model = _ConditionalWrapper(self.model, past_bjct, sign_img)

        # Try DDIM, fall back to DDPM if unavailable
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

        # Convert and denormalize
        pred_btjc = self.bjct_to_btjc(pred_bjct)
        return self.unnormalize(pred_btjc)

    def on_train_start(self):
        """Move normalization buffers to correct device before training."""
        self.mean_pose = self.mean_pose.to(self.device)
        self.std_pose = self.std_pose.to(self.device)

    def configure_optimizers(self):
        """Configure AdamW optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_train_end(self):
        """
        Save training curves after training completes.
        
        Creates a 2x2 plot showing:
        - Total loss over time
        - MSE loss over time
        - Velocity loss over time
        - Displacement ratio over time (with ideal=1.0 reference line)
        """
        try:
            import matplotlib.pyplot as plt
            out_dir = "logs/diffusion"
            os.makedirs(out_dir, exist_ok=True)

            # Create 2x2 subplot
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            # Plot each metric
            axes[0, 0].plot(self.train_logs["loss"])
            axes[0, 0].set_title("Total Loss")

            axes[0, 1].plot(self.train_logs["mse"])
            axes[0, 1].set_title("MSE Loss")

            axes[1, 0].plot(self.train_logs["vel"])
            axes[1, 0].set_title("Velocity Loss")

            axes[1, 1].plot(self.train_logs["disp_ratio"])
            axes[1, 1].set_title("Displacement Ratio")
            axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)  # Ideal line

            plt.tight_layout()
            plt.savefig(f"{out_dir}/train_curve.png")
            print(f"[TRAIN CURVE] saved to {out_dir}/train_curve.png")
        except Exception as e:
            print(f"[TRAIN CURVE] failed: {e}")


# Alias for backward compatibility
LitMinimal = LitDiffusion
