"""
Training with Past-only Dropout to force sign condition usage.

Key idea: 
- 30% of the time, drop past_motion completely
- Model is FORCED to use sign_image when past is unavailable
- This prevents the shortcut of "past -> future" ignoring sign

Based on successful unfrozen CLIP + contrastive loss version.

Usage:
    # Overfit test (32 samples, ~30 min)
    python train_past_dropout.py --mode overfit
    
    # Full dataset training (~30 hours)
    python train_past_dropout.py --mode full
    
    # Custom past dropout probability
    python train_past_dropout.py --mode overfit --past_drop_prob 0.5
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from torch.utils.data import DataLoader, Subset
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import unshift_hands
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder
from CAMDM.diffusion.gaussian_diffusion import (
    GaussianDiffusion, ModelMeanType, ModelVarType, LossType
)

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset


# ============================================================
# Model Components (Same as unfrozen CLIP version)
# ============================================================

class EmbedSignWritingUnfrozen(nn.Module):
    """SignWriting encoder with trainable CLIP."""
    
    def __init__(self, num_latent_dims: int, 
                 embedding_arch: str = 'openai/clip-vit-base-patch32',
                 freeze_clip: bool = False):
        super().__init__()
        self.model = CLIPModel.from_pretrained(embedding_arch)
        
        if freeze_clip:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.proj = None
        if (num_embedding_dims := self.model.visual_projection.out_features) != num_latent_dims:
            self.proj = nn.Linear(num_embedding_dims, num_latent_dims)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        embeddings_batch = self.model.get_image_features(pixel_values=image_batch)
        if self.proj is not None:
            embeddings_batch = self.proj(embeddings_batch)
        return embeddings_batch


class ContextEncoder(nn.Module):
    """Past motion context encoder."""
    
    def __init__(self, input_feats: int, latent_dim: int, 
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pose_encoder = nn.Linear(input_feats, latent_dim)
        self.pos_encoding = PositionalEncoding(latent_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=latent_dim * 4,
            dropout=dropout, activation="gelu", batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, T, J, C = x.shape
            x = x.reshape(B, T, J * C)
        x_emb = self.pose_encoder(x)
        x_emb = x_emb.permute(1, 0, 2)
        x_emb = self.pos_encoding(x_emb)
        x_enc = self.encoder(x_emb)
        x_enc = x_enc.permute(1, 0, 2)
        return x_enc.mean(dim=1)


class SignWritingToPoseDiffusionUnfrozen(nn.Module):
    """Diffusion model with unfrozen CLIP + Frame-Independent decoder."""
    
    def __init__(self, num_keypoints: int, num_dims_per_keypoint: int,
                 embedding_arch: str = 'openai/clip-vit-base-patch32',
                 num_latent_dims: int = 256, num_heads: int = 4, dropout: float = 0.1,
                 t_past: int = 40, t_future: int = 20, freeze_clip: bool = False):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
        self.t_past = t_past
        self.t_future = t_future

        input_feats = num_keypoints * num_dims_per_keypoint

        self.past_context_encoder = ContextEncoder(
            input_feats, num_latent_dims,
            num_layers=2, num_heads=num_heads, dropout=dropout,
        )
        self.embed_signwriting = EmbedSignWritingUnfrozen(
            num_latent_dims, embedding_arch, freeze_clip=freeze_clip
        )
        self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)
        self.time_embed = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)

        self.xt_frame_encoder = nn.Sequential(
            nn.Linear(input_feats, num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )
        self.output_pos_embed = nn.Embedding(512, num_latent_dims)

        decoder_input_dim = num_latent_dims * 3
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, input_feats),
        )

    def forward(self, x, timesteps, past_motion, signwriting_im_batch):
        B, J, C, T_future = x.shape
        device = x.device

        if past_motion.dim() == 4:
            if past_motion.shape[1] == J and past_motion.shape[2] == C:
                past_btjc = past_motion.permute(0, 3, 1, 2).contiguous()
            else:
                past_btjc = past_motion

        past_ctx = self.past_context_encoder(past_btjc)
        sign_emb = self.embed_signwriting(signwriting_im_batch)
        time_emb = self.time_embed(timesteps).squeeze(0)
        context = past_ctx + sign_emb + time_emb

        outputs = []
        for t in range(T_future):
            xt_frame = x[:, :, :, t].reshape(B, -1)
            xt_emb = self.xt_frame_encoder(xt_frame)
            pos_idx = torch.tensor([t], device=device)
            pos_emb = self.output_pos_embed(pos_idx).expand(B, -1)
            dec_input = torch.cat([context, xt_emb, pos_emb], dim=-1)
            out = self.decoder(dec_input)
            outputs.append(out)

        result = torch.stack(outputs, dim=0)
        result = result.permute(1, 0, 2)
        result = result.reshape(B, T_future, J, C)
        result = result.permute(0, 2, 3, 1).contiguous()
        return result


# ============================================================
# Utilities
# ============================================================

def sanitize_btjc(x):
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
    return x.contiguous().float()


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def mean_frame_disp(x_btjc: torch.Tensor) -> float:
    x = sanitize_btjc(x_btjc)
    if x.size(1) < 2:
        return 0.0
    v = x[:, 1:] - x[:, :-1]
    return v.abs().mean().item()


def tensor_to_pose(t_btjc: torch.Tensor, header, ref_pose: Pose, scale_to_ref: bool = True) -> Pose:
    """Convert tensor to Pose format."""
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc
    
    t_np = t.detach().cpu().numpy().astype(np.float32)
    arr = t_np[:, None, :, :]
    T = arr.shape[0]
    conf = np.ones((T, 1, arr.shape[2]), dtype=np.float32)
    fps = ref_pose.body.fps
    
    body = NumPyPoseBody(fps=fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)
    unshift_hands(pose_obj)
    
    if scale_to_ref:
        T_pred = t_np.shape[0]
        T_ref_total = ref_pose.body.data.shape[0]
        future_start = max(0, T_ref_total - T_pred)
        ref_arr = np.asarray(ref_pose.body.data[future_start:future_start+T_pred, 0], dtype=np.float32)
        
        def _var(a):
            center = a.mean(axis=(0, 1), keepdims=True)
            return float(((a - center) ** 2).mean())
        
        pose_data = pose_obj.body.data[:, 0, :, :]
        var_input = _var(pose_data)
        var_ref = _var(ref_arr)
        
        if var_input > 1e-8:
            scale = np.sqrt(var_ref / var_input)
            pose_obj.body.data = pose_obj.body.data * scale
        
        pose_data = pose_obj.body.data[:, 0, :, :].reshape(-1, 3)
        input_center = pose_data.mean(axis=0)
        ref_center = ref_arr.reshape(-1, 3).mean(axis=0)
        pose_obj.body.data = pose_obj.body.data + (ref_center - input_center)
    
    return pose_obj


# ============================================================
# Lightning Module with Past Dropout
# ============================================================

class LitDiffusionPastDropout(pl.LightningModule):
    """
    Lightning module with:
    - Unfrozen CLIP
    - Contrastive loss
    - Past-only Dropout (KEY NEW FEATURE!)
    """
    
    def __init__(self, 
                 num_keypoints=178, 
                 num_dims=3, 
                 lr=1e-4,
                 stats_path="/home/yayun/data/pose_data/mean_std_178_with_preprocess.pt",
                 diffusion_steps=8, 
                 vel_weight=1.0, 
                 acc_weight=0.5,
                 t_past=40, 
                 t_future=20,
                 freeze_clip=False,
                 contrastive_weight=0.5,
                 past_drop_prob=0.5):  # 50% dropout to force sign usage
        super().__init__()
        self.save_hyperparameters()

        self.diffusion_steps = diffusion_steps
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.contrastive_weight = contrastive_weight
        self.past_drop_prob = past_drop_prob  # NEW
        self._step_count = 0
        self._drop_count = 0  # Track how many times we dropped past

        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1, 1, -1, 3)
        std = stats["std"].float().view(1, 1, -1, 3)
        self.register_buffer("mean_pose", mean.clone())
        self.register_buffer("std_pose", std.clone())

        self.model = SignWritingToPoseDiffusionUnfrozen(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
            t_past=t_past,
            t_future=t_future,
            freeze_clip=freeze_clip,
        )

        betas = cosine_beta_schedule(diffusion_steps).numpy()
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        self.lr = lr

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

    def training_step(self, batch, batch_idx):
        cond_raw = batch["conditions"]
        gt_btjc = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(cond_raw["input_pose"])
        sign_img = cond_raw["sign_image"].float()

        gt_norm = self.normalize(gt_btjc)
        past_norm = self.normalize(past_btjc)

        batch_size = gt_norm.shape[0]
        device = gt_norm.device

        # ============================================================
        # KEY CHANGE: Past-only Dropout
        # With probability past_drop_prob, zero out past_motion
        # This FORCES the model to use sign_image when past is unavailable
        # ============================================================
        past_dropped = False
        if self.training and torch.rand(1).item() < self.past_drop_prob:
            past_norm = torch.zeros_like(past_norm)
            past_dropped = True
            self._drop_count += 1

        gt_bjct = self.btjc_to_bjct(gt_norm)
        past_bjct = self.btjc_to_bjct(past_norm)

        timestep = torch.randint(0, self.diffusion_steps, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(gt_bjct)
        x_noisy = self.diffusion.q_sample(gt_bjct, timestep, noise=noise)

        pred_x0_bjct = self.model(x_noisy, timestep, past_bjct, sign_img)

        # === Losses ===
        loss_mse = F.mse_loss(pred_x0_bjct, gt_bjct)
        
        pred_vel = pred_x0_bjct[..., 1:] - pred_x0_bjct[..., :-1]
        gt_vel = gt_bjct[..., 1:] - gt_bjct[..., :-1]
        loss_vel = F.mse_loss(pred_vel, gt_vel)

        loss_acc = torch.tensor(0.0, device=device)
        if pred_vel.size(-1) > 1:
            pred_acc = pred_vel[..., 1:] - pred_vel[..., :-1]
            gt_acc = gt_vel[..., 1:] - gt_vel[..., :-1]
            loss_acc = F.mse_loss(pred_acc, gt_acc)

        # === Contrastive Loss ===
        loss_contrastive = torch.tensor(0.0, device=device)
        if batch_size > 1 and self.contrastive_weight > 0:
            sign_embs = self.model.embed_signwriting(sign_img)
            sign_embs_norm = F.normalize(sign_embs, p=2, dim=-1)
            cos_sim = torch.mm(sign_embs_norm, sign_embs_norm.t())
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            off_diag_sim = cos_sim[mask]
            loss_contrastive = off_diag_sim.mean()
        
        loss = loss_mse + self.vel_weight * loss_vel + self.acc_weight * loss_acc + self.contrastive_weight * loss_contrastive

        # Displacement ratio
        with torch.no_grad():
            pred_disp = pred_vel.abs().mean().item()
            gt_disp = gt_vel.abs().mean().item()
            disp_ratio = pred_disp / (gt_disp + 1e-8)

        # Logging
        if self._step_count % 100 == 0:
            drop_str = " [PAST DROPPED]" if past_dropped else ""
            drop_rate = self._drop_count / (self._step_count + 1) * 100
            print(f"[Step {self._step_count}]{drop_str} loss={loss.item():.4f}, mse={loss_mse.item():.4f}, "
                  f"vel={loss_vel.item():.4f}, contrastive={loss_contrastive.item():.4f}, "
                  f"disp_ratio={disp_ratio:.4f}, drop_rate={drop_rate:.1f}%")

        self.log_dict({
            "train/loss": loss,
            "train/loss_mse": loss_mse,
            "train/loss_vel": loss_vel,
            "train/loss_contrastive": loss_contrastive,
            "train/disp_ratio": disp_ratio,
            "train/past_dropped": float(past_dropped),
        }, prog_bar=True)

        self._step_count += 1
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ============================================================
# Sign Influence Test
# ============================================================

@torch.no_grad()
def test_sign_influence(lit_model, dataset, device, num_samples=5):
    """Test if sign_image actually influences model output."""
    
    print("\n" + "=" * 70)
    print("TESTING SIGN INFLUENCE")
    print("=" * 70)
    
    lit_model.eval()
    
    # Select samples from different videos
    seen_poses = set()
    selected_indices = []
    for idx in range(len(dataset)):
        if len(selected_indices) >= num_samples:
            break
        record = dataset.records[idx]
        pose = record.get("pose", "")
        if pose not in seen_poses:
            seen_poses.add(pose)
            selected_indices.append(idx)
    
    print(f"Selected indices: {selected_indices}")
    
    # Load samples
    samples = []
    for idx in selected_indices:
        batch = zero_pad_collator([dataset[idx]])
        samples.append({
            "idx": idx,
            "past": sanitize_btjc(batch["conditions"]["input_pose"][:1]).to(device),
            "sign": batch["conditions"]["sign_image"][:1].float().to(device),
            "gt": sanitize_btjc(batch["data"][:1]).to(device),
        })
    
    # Helper function for DDPM sampling
    def sample_pose(past_norm, sign_img, future_len=20):
        past_bjct = lit_model.btjc_to_bjct(past_norm)
        B, J, C, _ = past_bjct.shape
        target_shape = (B, J, C, future_len)
        
        class _Wrapper(nn.Module):
            def __init__(self, model, past, sign):
                super().__init__()
                self.model, self.past, self.sign = model, past, sign
            def forward(self, x, t, **kwargs):
                return self.model(x, t, self.past, self.sign)
        
        wrapped = _Wrapper(lit_model.model, past_bjct, sign_img)
        torch.manual_seed(42)  # Fixed seed for reproducibility
        
        pred_bjct = lit_model.diffusion.p_sample_loop(
            model=wrapped,
            shape=target_shape,
            clip_denoised=True,
            model_kwargs={"y": {}},
            progress=False,
        )
        return lit_model.bjct_to_btjc(pred_bjct)
    
    # TEST 1: Same past_motion + Different sign_image
    print("\n--- TEST 1: Same past_motion + Different sign_image ---")
    base_past = lit_model.normalize(samples[0]["past"])
    
    predictions = []
    for s in samples:
        pred = sample_pose(base_past, s["sign"])
        pred_unnorm = lit_model.unnormalize(pred)
        predictions.append(pred_unnorm[0].cpu().numpy())
    
    predictions = np.array(predictions)
    diffs1 = []
    for i in range(len(predictions)):
        for j in range(i+1, len(predictions)):
            diff = np.sqrt(np.sum((predictions[i] - predictions[j])**2))
            diffs1.append(diff)
    avg_diff1 = np.mean(diffs1)
    print(f"  Average diff when swapping sign: {avg_diff1:.6f}")
    
    # TEST 2: Different past_motion + Same sign_image
    print("\n--- TEST 2: Different past_motion + Same sign_image ---")
    base_sign = samples[0]["sign"]
    
    predictions2 = []
    for s in samples:
        past_norm = lit_model.normalize(s["past"])
        pred = sample_pose(past_norm, base_sign)
        pred_unnorm = lit_model.unnormalize(pred)
        predictions2.append(pred_unnorm[0].cpu().numpy())
    
    predictions2 = np.array(predictions2)
    diffs2 = []
    for i in range(len(predictions2)):
        for j in range(i+1, len(predictions2)):
            diff = np.sqrt(np.sum((predictions2[i] - predictions2[j])**2))
            diffs2.append(diff)
    avg_diff2 = np.mean(diffs2)
    print(f"  Average diff when swapping past: {avg_diff2:.6f}")
    
    # Verdict
    ratio = avg_diff1 / (avg_diff2 + 1e-8)
    print(f"\n--- VERDICT ---")
    print(f"  Sign influence / Past influence = {ratio:.4f} ({ratio*100:.1f}%)")
    
    if ratio < 0.01:
        print("  ❌ Sign has NEGLIGIBLE influence (<1%)")
    elif ratio < 0.1:
        print("  ⚠️  Sign has WEAK influence (<10%)")
    elif ratio < 0.3:
        print("  ✓ Sign has MODERATE influence (10-30%)")
    else:
        print("  ✅ Sign has STRONG influence (>30%)")
    
    return ratio


# ============================================================
# Overfit Test
# ============================================================

def train_overfit(args):
    """Overfit test on small number of samples."""
    pl.seed_everything(42)

    # Configuration
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = f"logs/past_dropout_overfit_{args.num_samples}sample_p{int(args.past_drop_prob*100)}"

    NUM_SAMPLES = args.num_samples
    MAX_EPOCHS = args.epochs
    BATCH_SIZE = min(NUM_SAMPLES, 32)
    PAST_DROP_PROB = args.past_drop_prob

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f" OVERFIT TEST: PAST DROPOUT (prob={PAST_DROP_PROB})")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Samples: {NUM_SAMPLES}")
    print(f"  Epochs: {MAX_EPOCHS}")
    print(f"  Past Drop Prob: {PAST_DROP_PROB}")
    print(f"  Output: {out_dir}/")
    print("=" * 70)

    # Dataset
    print("\nLoading dataset...")
    full_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )

    # Select samples from different videos
    seen_poses = set()
    selected_indices = []
    for idx in range(len(full_ds)):
        if len(selected_indices) >= NUM_SAMPLES:
            break
        record = full_ds.records[idx]
        pose = record.get("pose", "")
        if pose not in seen_poses:
            seen_poses.add(pose)
            selected_indices.append(idx)

    train_ds = Subset(full_ds, selected_indices)
    print(f"Selected {len(train_ds)} samples from different videos")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=zero_pad_collator,
        num_workers=0,
    )

    # Get dimensions
    sample = full_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    num_joints = sample.shape[-2]
    num_dims = sample.shape[-1]
    future_len = sample.shape[0]

    print(f"Dimensions: J={num_joints}, D={num_dims}, T={future_len}")

    # Create Model
    lit_model = LitDiffusionPastDropout(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-4,
        diffusion_steps=8,
        vel_weight=1.0,
        acc_weight=0.5,
        t_past=40,
        t_future=future_len,
        freeze_clip=False,
        contrastive_weight=0.5,
        past_drop_prob=PAST_DROP_PROB,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{out_dir}/checkpoints",
        filename="best-epoch={epoch:03d}",
        save_top_k=1,
        monitor="train/disp_ratio",
        mode="max",
        save_last=True,
    )

    # Train
    print(f"\n{'='*70}")
    print("STARTING OVERFIT TRAINING...")
    print("="*70)

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        default_root_dir=out_dir,
        log_every_n_steps=10,
        enable_progress_bar=True,
        precision="16-mixed" if torch.cuda.is_available() else 32,
    )
    trainer.fit(lit_model, train_loader)

    # Test sign influence
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    sign_influence = test_sign_influence(lit_model, full_ds, device, num_samples=5)

    print(f"\n{'='*70}")
    print("OVERFIT TEST COMPLETE!")
    print("="*70)
    print(f"Sign Influence: {sign_influence*100:.1f}%")
    print(f"Checkpoint: {out_dir}/checkpoints/")
    print("="*70)

    return sign_influence


# ============================================================
# Full Dataset Training
# ============================================================

def train_full(args):
    """Train on full dataset."""
    pl.seed_everything(42)

    # Configuration
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = f"logs/full_past_dropout_p{int(args.past_drop_prob*100)}"

    MAX_EPOCHS = args.epochs
    BATCH_SIZE = 1024
    PAST_DROP_PROB = args.past_drop_prob

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f" FULL DATASET TRAINING: PAST DROPOUT (prob={PAST_DROP_PROB})")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Epochs: {MAX_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Past Drop Prob: {PAST_DROP_PROB}")
    print(f"  Output: {out_dir}/")
    print("=" * 70)

    # Dataset
    print("\nLoading full dataset...")
    train_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    print(f"Dataset loaded: {len(train_ds)} samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=zero_pad_collator,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
    )

    # Get dimensions
    sample = train_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    num_joints = sample.shape[-2]
    num_dims = sample.shape[-1]
    future_len = sample.shape[0]

    print(f"Dimensions: J={num_joints}, D={num_dims}, T={future_len}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Create Model
    lit_model = LitDiffusionPastDropout(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-4,
        diffusion_steps=8,
        vel_weight=1.0,
        acc_weight=0.5,
        t_past=40,
        t_future=future_len,
        freeze_clip=False,
        contrastive_weight=0.5,
        past_drop_prob=PAST_DROP_PROB,
    )

    total_params = sum(p.numel() for p in lit_model.parameters())
    trainable_params = sum(p.numel() for p in lit_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{out_dir}/checkpoints",
        filename="epoch{epoch:03d}-ratio{train/disp_ratio:.4f}",
        save_top_k=3,
        monitor="train/disp_ratio",
        mode="max",
        save_last=True,
        every_n_epochs=5,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Resume from checkpoint if exists
    resume_ckpt = None
    last_ckpt_path = f"{out_dir}/checkpoints/last.ckpt"
    if os.path.exists(last_ckpt_path):
        print(f"  Found existing checkpoint: {last_ckpt_path}")
        print(f"  Resuming training from checkpoint...")
        resume_ckpt = last_ckpt_path

    # Train
    print(f"\n{'='*70}")
    print("STARTING FULL DATASET TRAINING...")
    print("="*70)

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=out_dir,
        log_every_n_steps=50,
        enable_progress_bar=True,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
    )
    trainer.fit(lit_model, train_loader, ckpt_path=resume_ckpt)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"\nRun evaluation with:")
    print(f"  python test_checkpoints_v2.py --checkpoint {out_dir}/checkpoints/last.ckpt")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with Past Dropout")
    parser.add_argument("--mode", type=str, choices=["overfit", "full"], default="overfit",
                        help="Training mode: overfit (32 samples) or full (50K samples)")
    parser.add_argument("--num_samples", type=int, default=32,
                        help="Number of samples for overfit test (default: 32)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (default: 300 for overfit, 100 for full)")
    parser.add_argument("--past_drop_prob", type=float, default=0.5,
                        help="Past dropout probability (default: 0.5)")
    
    args = parser.parse_args()
    
    # Set default epochs
    if args.epochs is None:
        args.epochs = 300 if args.mode == "overfit" else 100
    
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    
    if args.mode == "overfit":
        train_overfit(args)
    else:
        train_full(args)