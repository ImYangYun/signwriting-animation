"""
FSW Encoder + Unfrozen CLIP Overfitting Experiment (Simplified)

Uses the original DynamicPosePredictionDataset and adds FSW extraction.
This ensures data loading is consistent with the working training script.

Usage:
    python train_fsw_overfit_simple.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
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

# Use original dataset!
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset

# SignWriting parsing
from signwriting.formats.swu_to_fsw import swu2fsw
from signwriting.formats.fsw_to_sign import fsw_to_sign


# ============================================================
# FSW Encoder (Simplified - just extract features)
# ============================================================

class SimpleFSWEncoder(nn.Module):
    """
    Simple FSW encoder: parse FSW → embed symbols → mean pool.
    """
    
    def __init__(self, num_latent_dims: int = 256):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        
        # Simple symbol embedding
        self.symbol_embed = nn.Embedding(1000, num_latent_dims)
        
        # Position encoder
        self.pos_encoder = nn.Linear(2, num_latent_dims)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_latent_dims * 2, num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )
        
    def parse_fsw(self, fsw_string: str):
        """Parse FSW string into symbols."""
        if not fsw_string or not fsw_string.strip():
            return [(0, 0.0, 0.0)]
        
        try:
            sign = fsw_to_sign(fsw_string)
            
            # Handle dict return type
            if isinstance(sign, dict):
                symbols_list = sign.get('symbols', [])
            else:
                symbols_list = getattr(sign, 'symbols', [])
            
            if not symbols_list:
                return [(0, 0.0, 0.0)]
            
            result = []
            for sym in symbols_list:
                if isinstance(sym, dict):
                    symbol_str = sym.get('symbol', 'S10000')
                    position = sym.get('position', (500, 500))
                else:
                    symbol_str = sym.symbol
                    position = sym.position
                
                # Get symbol ID (hash to embedding index)
                if symbol_str.startswith('S'):
                    symbol_str = symbol_str[1:]
                try:
                    symbol_id = int(symbol_str, 16) % 1000
                except:
                    symbol_id = 0
                
                # Normalize position
                x = (position[0] - 500) / 250.0
                y = (position[1] - 500) / 250.0
                x = max(-2.0, min(2.0, x))
                y = max(-2.0, min(2.0, y))
                
                result.append((symbol_id, x, y))
            
            return result if result else [(0, 0.0, 0.0)]
        except:
            return [(0, 0.0, 0.0)]
    
    def forward(self, fsw_batch: list):
        """Encode batch of FSW strings."""
        device = self.symbol_embed.weight.device
        batch_size = len(fsw_batch)
        
        embeddings = []
        for fsw in fsw_batch:
            symbols = self.parse_fsw(fsw)
            
            # Embed each symbol
            sym_embs = []
            for sym_id, x, y in symbols:
                sym_emb = self.symbol_embed(torch.tensor([sym_id], device=device))
                pos_emb = self.pos_encoder(torch.tensor([[x, y]], device=device))
                combined = torch.cat([sym_emb, pos_emb], dim=-1)
                sym_embs.append(combined)
            
            # Mean pool
            if sym_embs:
                pooled = torch.stack(sym_embs).mean(dim=0)
            else:
                pooled = torch.zeros(1, self.num_latent_dims * 2, device=device)
            
            embeddings.append(pooled)
        
        # Stack batch
        batch_emb = torch.cat(embeddings, dim=0)  # (B, D*2)
        output = self.output_proj(batch_emb)  # (B, D)
        
        return output


# ============================================================
# Utilities (from train_unfrozen_clip_full.py)
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
# Model Components
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


class SignWritingToPoseDiffusionWithFSW(nn.Module):
    """Diffusion model with CLIP + FSW encoder."""
    
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
        
        # CLIP encoder
        self.embed_signwriting = EmbedSignWritingUnfrozen(
            num_latent_dims, embedding_arch, freeze_clip=freeze_clip
        )
        
        # FSW encoder
        self.fsw_encoder = SimpleFSWEncoder(num_latent_dims)
        
        # Fusion: CLIP + FSW
        self.sign_fusion = nn.Sequential(
            nn.Linear(num_latent_dims * 2, num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
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

    def forward(self, x, timesteps, past_motion, signwriting_im_batch, fsw_strings=None):
        B, J, C, T_future = x.shape
        device = x.device

        if past_motion.dim() == 4:
            if past_motion.shape[1] == J and past_motion.shape[2] == C:
                past_btjc = past_motion.permute(0, 3, 1, 2).contiguous()
            else:
                past_btjc = past_motion

        past_ctx = self.past_context_encoder(past_btjc)
        
        # CLIP embedding
        clip_emb = self.embed_signwriting(signwriting_im_batch)
        
        # FSW embedding (if provided)
        if fsw_strings is not None:
            self.fsw_encoder = self.fsw_encoder.to(device)
            fsw_emb = self.fsw_encoder(fsw_strings)
            # Fuse CLIP + FSW
            sign_emb = self.sign_fusion(torch.cat([clip_emb, fsw_emb], dim=-1))
        else:
            sign_emb = clip_emb
        
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
# Lightning Module
# ============================================================

class LitDiffusionWithFSW(pl.LightningModule):
    """Lightning module with CLIP + FSW."""
    
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
                 use_fsw=True):
        super().__init__()
        self.save_hyperparameters()

        self.diffusion_steps = diffusion_steps
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.use_fsw = use_fsw
        self._step_count = 0

        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1, 1, -1, 3)
        std = stats["std"].float().view(1, 1, -1, 3)
        self.register_buffer("mean_pose", mean.clone())
        self.register_buffer("std_pose", std.clone())

        self.model = SignWritingToPoseDiffusionWithFSW(
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
        
        # Store FSW strings for current batch
        self.current_fsw_strings = None

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

        gt_bjct = self.btjc_to_bjct(gt_norm)
        past_bjct = self.btjc_to_bjct(past_norm)

        timestep = torch.randint(0, self.diffusion_steps, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(gt_bjct)
        x_noisy = self.diffusion.q_sample(gt_bjct, timestep, noise=noise)

        # Get FSW strings if available
        fsw_strings = self.current_fsw_strings if self.use_fsw else None
        
        pred_x0_bjct = self.model(x_noisy, timestep, past_bjct, sign_img, fsw_strings)

        # Losses
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

        # Displacement ratio
        with torch.no_grad():
            pred_disp = pred_vel.abs().mean().item()
            gt_disp = gt_vel.abs().mean().item()
            disp_ratio = pred_disp / (gt_disp + 1e-8)

        if self._step_count % 50 == 0:
            print(f"[Step {self._step_count}] loss={loss.item():.4f}, mse={loss_mse.item():.4f}, "
                  f"vel={loss_vel.item():.4f}, disp_ratio={disp_ratio:.4f}")

        self.log_dict({
            "train/loss": loss,
            "train/loss_mse": loss_mse,
            "train/loss_vel": loss_vel,
            "train/disp_ratio": disp_ratio,
        }, prog_bar=True)

        self._step_count += 1
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ============================================================
# Custom collate that extracts FSW
# ============================================================

def get_fsw_from_record(record):
    """Extract FSW string from record."""
    swu_text = record.get('text', '')
    if not swu_text:
        return ""
    
    swu_first = swu_text.split()[0] if ' ' in swu_text else swu_text
    
    try:
        fsw = swu2fsw(swu_first)
        return fsw
    except:
        return ""


def collate_with_fsw(batch, dataset):
    """Custom collate that adds FSW strings."""
    # Use original collator
    collated = zero_pad_collator(batch)
    
    # Extract FSW strings from records
    indices = [b.get('idx', i) for i, b in enumerate(batch)]
    fsw_strings = []
    for idx in indices:
        if hasattr(dataset, 'records') and idx < len(dataset.records):
            fsw = get_fsw_from_record(dataset.records[idx])
        else:
            fsw = ""
        fsw_strings.append(fsw)
    
    return collated, fsw_strings


def simple_collate(samples):
    """Simple collate for pre-processed cached samples."""
    batch_data = torch.stack([s['data'] for s in samples])
    batch_input = torch.stack([s['conditions']['input_pose'] for s in samples])
    batch_sign = torch.stack([s['conditions']['sign_image'] for s in samples])
    
    return {
        'data': batch_data,
        'conditions': {
            'input_pose': batch_input,
            'sign_image': batch_sign,
        }
    }


# ============================================================
# Main
# ============================================================

def train_overfit():
    """Overfitting experiment with FSW + CLIP encoder."""
    pl.seed_everything(42)

    # Config
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = "logs/fsw_overfit_simple"
    
    NUM_SAMPLES = 4
    MAX_EPOCHS = 500
    DIFFUSION_STEPS = 8
    LEARNING_RATE = 1e-4
    USE_FSW = True
    
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(" FSW + CLIP OVERFITTING (SIMPLIFIED)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Samples: {NUM_SAMPLES}")
    print(f"  Epochs: {MAX_EPOCHS}")
    print(f"  Use FSW: {USE_FSW}")
    print("=" * 70)

    # Use ORIGINAL dataset (this is the key difference!)
    print("\nLoading dataset (using original DynamicPosePredictionDataset)...")
    full_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    print(f"Full dataset size: {len(full_ds)}")
    
    # CACHE the samples to ensure we overfit to the SAME data each time!
    # (DynamicPosePredictionDataset randomly samples time windows, so we need to fix them)
    print("\nCaching fixed samples for overfitting...")
    cached_samples = []
    for i in range(NUM_SAMPLES):
        sample = full_ds[i]
        
        # Extract tensors properly
        data = sample['data']
        if hasattr(data, 'zero_filled'):
            data = data.zero_filled()
        if hasattr(data, 'tensor'):
            data = data.tensor
        
        input_pose = sample['conditions']['input_pose']
        if hasattr(input_pose, 'zero_filled'):
            input_pose = input_pose.zero_filled()
        if hasattr(input_pose, 'tensor'):
            input_pose = input_pose.tensor
        
        # Deep copy to prevent any mutation
        cached = {
            'data': data.clone().float(),
            'conditions': {
                'input_pose': input_pose.clone().float(),
                'sign_image': sample['conditions']['sign_image'].clone().float(),
            },
            'idx': i,
        }
        cached_samples.append(cached)
        print(f"  Cached sample {i}: data shape {cached['data'].shape}, input shape {cached['conditions']['input_pose'].shape}")
    
    # Check FSW for samples
    print("\nChecking FSW for samples:")
    fsw_strings_cache = []
    for i in range(NUM_SAMPLES):
        record = full_ds.records[i]
        fsw = get_fsw_from_record(record)
        fsw_strings_cache.append(fsw)
        print(f"  [{i}] FSW: {fsw[:60]}..." if len(fsw) > 60 else f"  [{i}] FSW: {fsw}")

    # Get dimensions
    sample = full_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    num_joints = sample.shape[-2]
    num_dims = sample.shape[-1]
    future_len = sample.shape[0]
    
    print(f"\nDimensions: J={num_joints}, D={num_dims}, T={future_len}")

    # Model
    print("\nInitializing model...")
    lit_model = LitDiffusionWithFSW(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=LEARNING_RATE,
        diffusion_steps=DIFFUSION_STEPS,
        vel_weight=1.0,
        acc_weight=0.5,
        t_past=40,
        t_future=future_len,
        freeze_clip=False,
        use_fsw=USE_FSW,
    )

    total_params = sum(p.numel() for p in lit_model.parameters())
    trainable_params = sum(p.numel() for p in lit_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Custom training loop to pass FSW strings
    print(f"\n{'='*70}")
    print("STARTING TRAINING...")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    
    optimizer = torch.optim.AdamW(lit_model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(MAX_EPOCHS):
        lit_model.train()
        
        # Use CACHED samples (not re-sampling from dataset!)
        batch = simple_collate(cached_samples)
        
        # Move to device
        batch["data"] = batch["data"].to(device)
        batch["conditions"]["input_pose"] = batch["conditions"]["input_pose"].to(device)
        batch["conditions"]["sign_image"] = batch["conditions"]["sign_image"].to(device)
        
        # Set FSW strings
        lit_model.current_fsw_strings = fsw_strings_cache
        
        # Forward
        optimizer.zero_grad()
        loss = lit_model.training_step(batch, 0)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lit_model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{MAX_EPOCHS}")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print("="*70)

    # Evaluation
    print("\nEvaluating...")
    lit_model.eval()
    
    results = []
    for idx in range(NUM_SAMPLES):
        # Use CACHED sample
        batch = simple_collate([cached_samples[idx]])
        
        cond = batch["conditions"]
        past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt_raw = sanitize_btjc(batch["data"][:1]).to(device)
        
        past_norm = lit_model.normalize(past_raw)
        gt_norm = lit_model.normalize(gt_raw)
        
        past_bjct = lit_model.btjc_to_bjct(past_norm)
        B, J, C, _ = past_bjct.shape
        T_future = gt_norm.shape[1]
        target_shape = (B, J, C, T_future)
        
        with torch.no_grad():
            # Normal inference
            class _Wrapper(nn.Module):
                def __init__(self, model, past, sign, fsw):
                    super().__init__()
                    self.model, self.past, self.sign, self.fsw = model, past, sign, fsw
                def forward(self, x, t, **kwargs):
                    return self.model(x, t, self.past, self.sign, self.fsw)
            
            wrapped = _Wrapper(lit_model.model, past_bjct, sign, [fsw_strings_cache[idx]])
            pred_bjct = lit_model.diffusion.p_sample_loop(
                model=wrapped,
                shape=target_shape,
                clip_denoised=True,
                model_kwargs={"y": {}},
                progress=False,
            )
            pred_norm = lit_model.bjct_to_btjc(pred_bjct)
            
            # Sign-Only (drop past)
            zeros_past = torch.zeros_like(past_bjct)
            wrapped_so = _Wrapper(lit_model.model, zeros_past, sign, [fsw_strings_cache[idx]])
            pred_so_bjct = lit_model.diffusion.p_sample_loop(
                model=wrapped_so,
                shape=target_shape,
                clip_denoised=True,
                model_kwargs={"y": {}},
                progress=False,
            )
            pred_so_norm = lit_model.bjct_to_btjc(pred_so_bjct)
        
        # Metrics
        pred_np = pred_norm[0].cpu().numpy()
        pred_so_np = pred_so_norm[0].cpu().numpy()
        gt_np = gt_norm[0].cpu().numpy()
        
        err_normal = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
        err_so = np.sqrt(((pred_so_np - gt_np) ** 2).sum(-1))
        
        pck_normal = (err_normal < 0.1).mean() * 100
        pck_so = (err_so < 0.1).mean() * 100
        
        ratio = mean_frame_disp(pred_norm) / (mean_frame_disp(gt_norm) + 1e-8)
        
        results.append({
            'idx': idx,
            'pck_normal': pck_normal,
            'pck_signonly': pck_so,
            'disp_ratio': ratio,
        })
        
        print(f"  [{idx}] Normal: {pck_normal:.1f}%, Sign-Only: {pck_so:.1f}%, Ratio: {ratio:.3f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    
    avg_normal = np.mean([r['pck_normal'] for r in results])
    avg_so = np.mean([r['pck_signonly'] for r in results])
    avg_ratio = np.mean([r['disp_ratio'] for r in results])
    
    print(f"  Average Normal PCK@0.1: {avg_normal:.1f}%")
    print(f"  Average Sign-Only PCK@0.1: {avg_so:.1f}%")
    print(f"  Average Disp Ratio: {avg_ratio:.3f}")
    print(f"  Gap: {avg_normal - avg_so:.1f}%")
    
    # Save checkpoint
    torch.save(lit_model.state_dict(), f"{out_dir}/model.pt")
    print(f"\nModel saved to: {out_dir}/model.pt")
    
    print("=" * 70)


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    train_overfit()