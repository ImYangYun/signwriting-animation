"""
4-Sample Overfit Test: Unfrozen CLIP Version

Goal: Test if unfreezing CLIP allows it to learn to distinguish SignWriting symbols.

Key change: CLIP parameters are trainable (not frozen).
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader, Subset
from pose_format.torch.masked.collator import zero_pad_collator
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder
from CAMDM.diffusion.gaussian_diffusion import (
    GaussianDiffusion, ModelMeanType, ModelVarType, LossType
)

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset


# ============================================================
# Modified EmbedSignWriting with freeze option
# ============================================================
class EmbedSignWritingUnfrozen(nn.Module):
    """
    SignWriting image encoder using CLIP vision model.
    Key change: CLIP is NOT frozen, so it learns to distinguish SignWriting.
    """
    
    def __init__(self, num_latent_dims: int, embedding_arch: str = 'openai/clip-vit-base-patch32',
                 freeze_clip: bool = False):
        super().__init__()
        self.model = CLIPModel.from_pretrained(embedding_arch)
        self.freeze_clip = freeze_clip
        
        # Freeze or unfreeze CLIP
        if freeze_clip:
            print("[EmbedSignWriting] CLIP is FROZEN")
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print("[EmbedSignWriting] CLIP is UNFROZEN (trainable)")
            # Optionally freeze some layers, only train later layers
            # For now, train everything
        
        self.proj = None
        if (num_embedding_dims := self.model.visual_projection.out_features) != num_latent_dims:
            self.proj = nn.Linear(num_embedding_dims, num_latent_dims)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        embeddings_batch = self.model.get_image_features(pixel_values=image_batch)
        if self.proj is not None:
            embeddings_batch = self.proj(embeddings_batch)
        return embeddings_batch


# ============================================================
# Modified ContextEncoder (same as before)
# ============================================================
class ContextEncoder(nn.Module):
    def __init__(self, input_feats: int, latent_dim: int, num_layers: int = 2, 
                 num_heads: int = 4, dropout: float = 0.1):
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
        context = x_enc.mean(dim=1)
        return context


# ============================================================
# Modified Model with unfrozen CLIP
# ============================================================
class SignWritingToPoseDiffusionUnfrozen(nn.Module):
    """
    Same architecture but with unfrozen CLIP.
    """
    
    def __init__(self, num_keypoints: int, num_dims_per_keypoint: int,
                 embedding_arch: str = 'openai/clip-vit-base-patch32',
                 num_latent_dims: int = 256, num_heads: int = 4, dropout: float = 0.1,
                 t_past: int = 40, t_future: int = 20,
                 freeze_clip: bool = False):  # ← Key parameter
        super().__init__()

        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
        self.t_past = t_past
        self.t_future = t_future
        self._forward_count = 0

        input_feats = num_keypoints * num_dims_per_keypoint

        # Past motion encoder
        self.past_context_encoder = ContextEncoder(
            input_feats, num_latent_dims,
            num_layers=2, num_heads=num_heads, dropout=dropout,
        )

        # SignWriting encoder - NOW WITH FREEZE OPTION
        self.embed_signwriting = EmbedSignWritingUnfrozen(
            num_latent_dims, embedding_arch, freeze_clip=freeze_clip
        )

        # Timestep encoder
        self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)
        self.time_embed = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)

        # Noisy frame encoder
        self.xt_frame_encoder = nn.Sequential(
            nn.Linear(input_feats, num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )

        # Position embedding
        self.output_pos_embed = nn.Embedding(512, num_latent_dims)

        # Decoder
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

        # Convert past_motion to BTJC
        if past_motion.dim() == 4:
            if past_motion.shape[1] == J and past_motion.shape[2] == C:
                past_btjc = past_motion.permute(0, 3, 1, 2).contiguous()
            else:
                past_btjc = past_motion

        # Encode conditions
        past_ctx = self.past_context_encoder(past_btjc)
        sign_emb = self.embed_signwriting(signwriting_im_batch)
        time_emb = self.time_embed(timesteps).squeeze(0)

        # Fuse conditions (addition)
        context = past_ctx + sign_emb + time_emb

        # Frame-independent decoding
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

        self._forward_count += 1
        return result


# ============================================================
# Lightning Module
# ============================================================
def sanitize_btjc(x):
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


class LitDiffusionUnfrozenCLIP(pl.LightningModule):
    """Lightning module with unfrozen CLIP."""
    
    def __init__(self, num_keypoints=178, num_dims=3, lr=1e-4,
                 stats_path="/home/yayun/data/pose_data/mean_std_178_with_preprocess.pt",
                 diffusion_steps=8, vel_weight=1.0, acc_weight=0.5,
                 t_past=40, t_future=20,
                 freeze_clip=False):  # ← Key parameter
        super().__init__()
        self.save_hyperparameters()

        self.diffusion_steps = diffusion_steps
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.freeze_clip = freeze_clip
        self._step_count = 0

        # Load normalization stats
        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1, 1, -1, 3)
        std = stats["std"].float().view(1, 1, -1, 3)
        self.register_buffer("mean_pose", mean.clone())
        self.register_buffer("std_pose", std.clone())

        # Create model with freeze option
        self.model = SignWritingToPoseDiffusionUnfrozen(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
            t_past=t_past,
            t_future=t_future,
            freeze_clip=freeze_clip,
        )

        # Diffusion
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

        gt_bjct = self.btjc_to_bjct(gt_norm)
        past_bjct = self.btjc_to_bjct(past_norm)

        timestep = torch.randint(0, self.diffusion_steps, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(gt_bjct)
        x_noisy = self.diffusion.q_sample(gt_bjct, timestep, noise=noise)

        pred_x0_bjct = self.model(x_noisy, timestep, past_bjct, sign_img)

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
            print(f"[Step {self._step_count}] loss={loss.item():.4f}, disp_ratio={disp_ratio:.4f}")

        self.log_dict({
            "train/loss": loss,
            "train/disp_ratio": disp_ratio,
        }, prog_bar=True)

        self._step_count += 1
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ============================================================
# Test Script
# ============================================================
def test_unfrozen_clip():
    """Test if unfreezing CLIP helps it learn to distinguish SignWriting."""
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    NUM_SAMPLES = 4
    MAX_EPOCHS = 200
    DIFFUSION_STEPS = 8
    FREEZE_CLIP = False  # ← KEY: Set to False to unfreeze CLIP
    
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = f"logs/unfrozen_clip_{NUM_SAMPLES}sample"
    # ============================================================
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 70)
    print("4-SAMPLE OVERFIT TEST: UNFROZEN CLIP")
    print("=" * 70)
    print(f"  NUM_SAMPLES: {NUM_SAMPLES}")
    print(f"  MAX_EPOCHS: {MAX_EPOCHS}")
    print(f"  DIFFUSION_STEPS: {DIFFUSION_STEPS}")
    print(f"  FREEZE_CLIP: {FREEZE_CLIP}")
    print(f"  Output: {out_dir}")
    
    # Load dataset
    full_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    # Use first N samples
    subset_ds = Subset(full_ds, list(range(NUM_SAMPLES)))
    train_loader = DataLoader(
        subset_ds, batch_size=NUM_SAMPLES, shuffle=True,
        collate_fn=zero_pad_collator, num_workers=0,
    )
    
    print(f"  Using {NUM_SAMPLES} samples for overfit test")
    
    # Get dimensions
    sample = full_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    num_joints = sample.shape[-2]
    num_dims = sample.shape[-1]
    future_len = sample.shape[0]
    
    print(f"  Dimensions: J={num_joints}, D={num_dims}, T={future_len}")
    
    # Create model
    lit_model = LitDiffusionUnfrozenCLIP(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-4,
        diffusion_steps=DIFFUSION_STEPS,
        vel_weight=1.0,
        t_past=40,
        t_future=future_len,
        freeze_clip=FREEZE_CLIP,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in lit_model.parameters())
    trainable_params = sum(p.numel() for p in lit_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{out_dir}/checkpoints",
        filename="best-{epoch}",
        save_top_k=1,
        monitor="train/loss",
        mode="min",
        save_last=True,
    )
    
    # Train
    print("\n" + "=" * 70)
    print("TRAINING...")
    print("=" * 70)
    
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        default_root_dir=out_dir,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    trainer.fit(lit_model, train_loader)
    
    # ============================================================
    # TEST: Check sign embedding similarity after training
    # ============================================================
    print("\n" + "=" * 70)
    print("CHECKING SIGN EMBEDDING SIMILARITY AFTER TRAINING")
    print("=" * 70)
    
    device = next(lit_model.parameters()).device
    lit_model.eval()
    
    sign_embeddings = []
    sample_indices = list(range(NUM_SAMPLES))
    
    for idx in sample_indices:
        batch = zero_pad_collator([full_ds[idx]])
        sign = batch["conditions"]["sign_image"][:1].float().to(device)
        
        with torch.no_grad():
            sign_emb = lit_model.model.embed_signwriting(sign)
        sign_embeddings.append(sign_emb)
        print(f"  idx={idx}: sign_emb norm = {sign_emb.norm().item():.4f}")
    
    print("\n  Pairwise cosine similarity:")
    for i in range(len(sign_embeddings)):
        for j in range(i+1, len(sign_embeddings)):
            cos_sim = F.cosine_similarity(sign_embeddings[i], sign_embeddings[j]).item()
            print(f"    idx {sample_indices[i]} vs {sample_indices[j]}: {cos_sim:.4f}")
    
    # ============================================================
    # TEST: Inference on training samples
    # ============================================================
    print("\n" + "=" * 70)
    print("INFERENCE ON TRAINING SAMPLES")
    print("=" * 70)
    
    results = []
    
    for idx in sample_indices:
        batch = zero_pad_collator([full_ds[idx]])
        cond = batch["conditions"]
        
        past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt_raw = sanitize_btjc(batch["data"][:1]).to(device)
        
        past_norm = lit_model.normalize(past_raw)
        gt_norm = lit_model.normalize(gt_raw)
        past_bjct = lit_model.btjc_to_bjct(past_norm)
        
        # DDPM sampling
        with torch.no_grad():
            B, J, C, _ = past_bjct.shape
            target_shape = (B, J, C, future_len)
            
            class _Wrapper(nn.Module):
                def __init__(self, model, past, sign):
                    super().__init__()
                    self.model, self.past, self.sign = model, past, sign
                def forward(self, x, t, **kwargs):
                    return self.model(x, t, self.past, self.sign)
            
            wrapped = _Wrapper(lit_model.model, past_bjct, sign)
            
            pred_bjct = lit_model.diffusion.p_sample_loop(
                model=wrapped,
                shape=target_shape,
                clip_denoised=True,
                model_kwargs={"y": {}},
                progress=False,
            )
            pred_btjc = lit_model.bjct_to_btjc(pred_bjct)
        
        # Compute metrics
        gt_unnorm = lit_model.unnormalize(gt_norm)
        pred_unnorm = lit_model.unnormalize(pred_btjc)
        
        gt_np = gt_unnorm[0].cpu().numpy()
        pred_np = pred_unnorm[0].cpu().numpy()
        
        gt_disp = np.sqrt(np.sum(np.diff(gt_np, axis=0)**2, axis=-1)).mean()
        pred_disp = np.sqrt(np.sum(np.diff(pred_np, axis=0)**2, axis=-1)).mean()
        ratio = pred_disp / (gt_disp + 1e-8)
        
        print(f"  Sample {idx}: GT_disp={gt_disp:.4f}, Pred_disp={pred_disp:.4f}, Ratio={ratio:.4f}")
        results.append(ratio)
    
    avg_ratio = np.mean(results)
    
    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"\n  Average Inference Disp Ratio: {avg_ratio:.4f} (ideal = 1.0)")
    
    if 0.8 <= avg_ratio <= 1.2:
        print("\n  ✅ SUCCESS! Unfrozen CLIP helps the model learn.")
        print("     → Proceed with full dataset training")
    else:
        print("\n  ❌ Still not working well.")
        print("     → May need other approaches (e.g., FSW text, different encoder)")
    
    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    test_unfrozen_clip()