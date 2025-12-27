"""
CLIP Ablation Study: Frozen vs Unfrozen (without contrastive learning)

Compares:
1. Frozen CLIP
2. Unfrozen CLIP (no contrastive)

Metrics:
- Embedding similarity (avg, min, max)
- Normal PCK
- Sign-Only PCK  
- Displacement Ratio

Usage:
    python train_clip_ablation.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as pl

from pose_format import Pose
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder
from CAMDM.diffusion.gaussian_diffusion import (
    GaussianDiffusion, ModelMeanType, ModelVarType, LossType
)

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset


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
# Model Components
# ============================================================

class EmbedSignWriting(nn.Module):
    """SignWriting encoder with CLIP."""
    
    def __init__(self, num_latent_dims: int, 
                 embedding_arch: str = 'openai/clip-vit-base-patch32',
                 freeze_clip: bool = True):
        super().__init__()
        self.model = CLIPModel.from_pretrained(embedding_arch)
        self.freeze_clip = freeze_clip
        
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


class SignWritingToPoseDiffusion(nn.Module):
    """Diffusion model with CLIP encoder (no FSW, no contrastive)."""
    
    def __init__(self, num_keypoints: int, num_dims_per_keypoint: int,
                 embedding_arch: str = 'openai/clip-vit-base-patch32',
                 num_latent_dims: int = 256, num_heads: int = 4, dropout: float = 0.1,
                 t_past: int = 40, t_future: int = 20, freeze_clip: bool = True):
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
        self.embed_signwriting = EmbedSignWriting(
            num_latent_dims, embedding_arch, freeze_clip=freeze_clip
        )
        
        self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)
        self.time_embed = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)

        self.xt_frame_encoder = nn.Sequential(
            nn.Linear(input_feats, num_latent_dims),
            nn.LayerNorm(num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )
        self.output_pos_embed = nn.Embedding(512, num_latent_dims)

        # Decoder
        decoder_input_dim = num_latent_dims * 3
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
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
        
        # CLIP embedding (no contrastive loss here)
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
# Lightning Module
# ============================================================

class LitDiffusionCLIP(pl.LightningModule):
    """Lightning module for CLIP ablation (no contrastive)."""
    
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
                 freeze_clip=True):
        super().__init__()
        self.save_hyperparameters()

        self.diffusion_steps = diffusion_steps
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self._step_count = 0

        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1, 1, -1, 3)
        std = stats["std"].float().view(1, 1, -1, 3)
        self.register_buffer("mean_pose", mean.clone())
        self.register_buffer("std_pose", std.clone())

        self.model = SignWritingToPoseDiffusion(
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

        if self._step_count % 500 == 0:
            print(f"[Step {self._step_count}] loss={loss.item():.4f}, mse={loss_mse.item():.4f}, "
                  f"vel={loss_vel.item():.4f}, disp_ratio={disp_ratio:.3f}")

        self._step_count += 1
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ============================================================
# Embedding Similarity Analysis
# ============================================================

def compute_embedding_similarity(lit_model, cached_samples, device):
    """Compute pairwise cosine similarity of CLIP embeddings."""
    lit_model.eval()
    
    embeddings = []
    with torch.no_grad():
        for sample in cached_samples:
            sign_img = sample['conditions']['sign_image'].unsqueeze(0).float().to(device)
            emb = lit_model.model.embed_signwriting(sign_img)
            embeddings.append(emb.cpu())
    
    embeddings = torch.cat(embeddings, dim=0)  # [N, D]
    
    # Normalize
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    
    # Pairwise cosine similarity
    sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
    
    # Get upper triangle (excluding diagonal)
    n = sim_matrix.size(0)
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    similarities = sim_matrix[mask]
    
    return {
        'avg_similarity': similarities.mean().item(),
        'min_similarity': similarities.min().item(),
        'max_similarity': similarities.max().item(),
        'std_similarity': similarities.std().item(),
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(lit_model, cached_samples, device):
    """Evaluate model on cached samples."""
    lit_model.eval()
    
    results = []
    for idx, sample in enumerate(cached_samples):
        batch = simple_collate([sample])
        
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
            class _Wrapper(nn.Module):
                def __init__(self, model, past, sign):
                    super().__init__()
                    self.model, self.past, self.sign = model, past, sign
                def forward(self, x, t, **kwargs):
                    return self.model(x, t, self.past, self.sign)
            
            # Normal inference
            wrapped = _Wrapper(lit_model.model, past_bjct, sign)
            pred_bjct = lit_model.diffusion.p_sample_loop(
                model=wrapped,
                shape=target_shape,
                clip_denoised=True,
                model_kwargs={"y": {}},
                progress=False,
            )
            pred_norm = lit_model.bjct_to_btjc(pred_bjct)
            
            # Sign-Only inference
            zeros_past = torch.zeros_like(past_bjct)
            wrapped_so = _Wrapper(lit_model.model, zeros_past, sign)
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
    
    return results


# ============================================================
# Main
# ============================================================

def run_ablation():
    """Run CLIP ablation study: Frozen vs Unfrozen."""
    pl.seed_everything(42)

    # Config
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = "logs/clip_ablation"

    NUM_SAMPLES = 32
    MAX_EPOCHS = 5000
    DIFFUSION_STEPS = 8
    LEARNING_RATE = 1e-4
    
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(" CLIP ABLATION: Frozen vs Unfrozen (no contrastive)")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    full_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    # Cache samples
    print(f"\nCaching {NUM_SAMPLES} samples...")
    cached_samples = []
    for i in range(NUM_SAMPLES):
        sample = full_ds[i]
        
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
        
        cached = {
            'data': data.clone().float(),
            'conditions': {
                'input_pose': input_pose.clone().float(),
                'sign_image': sample['conditions']['sign_image'].clone().float(),
            },
            'idx': i,
        }
        cached_samples.append(cached)

    # Get dimensions
    sample = full_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    num_joints = sample.shape[-2]
    num_dims = sample.shape[-1]
    future_len = sample.shape[0]
    
    print(f"Dimensions: J={num_joints}, D={num_dims}, T_future={future_len}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = {}
    
    # ========================================
    # Experiment 1: Frozen CLIP
    # ========================================
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: Frozen CLIP")
    print("="*70)
    
    lit_model_frozen = LitDiffusionCLIP(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=LEARNING_RATE,
        diffusion_steps=DIFFUSION_STEPS,
        vel_weight=1.0,
        acc_weight=0.5,
        t_past=40,
        t_future=future_len,
        freeze_clip=True,
    ).to(device)
    
    # Pre-training similarity
    print("\nComputing pre-training embedding similarity...")
    sim_frozen_pre = compute_embedding_similarity(lit_model_frozen, cached_samples, device)
    print(f"  Avg: {sim_frozen_pre['avg_similarity']:.4f}")
    print(f"  Min: {sim_frozen_pre['min_similarity']:.4f}")
    print(f"  Max: {sim_frozen_pre['max_similarity']:.4f}")
    
    # Training
    print("\nTraining Frozen CLIP...")
    optimizer = torch.optim.AdamW(lit_model_frozen.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(MAX_EPOCHS):
        lit_model_frozen.train()
        batch = simple_collate(cached_samples)
        batch["data"] = batch["data"].to(device)
        batch["conditions"]["input_pose"] = batch["conditions"]["input_pose"].to(device)
        batch["conditions"]["sign_image"] = batch["conditions"]["sign_image"].to(device)
        
        optimizer.zero_grad()
        loss = lit_model_frozen.training_step(batch, 0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lit_model_frozen.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"  Epoch {epoch}/{MAX_EPOCHS}, Loss: {loss.item():.4f}")
    
    # Post-training similarity (should be same for frozen)
    sim_frozen_post = compute_embedding_similarity(lit_model_frozen, cached_samples, device)
    
    # Evaluation
    print("\nEvaluating Frozen CLIP...")
    results_frozen = evaluate_model(lit_model_frozen, cached_samples, device)
    
    avg_normal_frozen = np.mean([r['pck_normal'] for r in results_frozen])
    avg_so_frozen = np.mean([r['pck_signonly'] for r in results_frozen])
    avg_ratio_frozen = np.mean([r['disp_ratio'] for r in results_frozen])
    
    all_results['frozen'] = {
        'similarity_pre': sim_frozen_pre,
        'similarity_post': sim_frozen_post,
        'pck_normal': avg_normal_frozen,
        'pck_signonly': avg_so_frozen,
        'disp_ratio': avg_ratio_frozen,
    }
    
    print(f"\n  Frozen CLIP Results:")
    print(f"    Similarity (post): avg={sim_frozen_post['avg_similarity']:.4f}, min={sim_frozen_post['min_similarity']:.4f}")
    print(f"    Normal PCK: {avg_normal_frozen:.1f}%")
    print(f"    Sign-Only PCK: {avg_so_frozen:.1f}%")
    print(f"    Disp Ratio: {avg_ratio_frozen:.3f}")
    
    torch.save(lit_model_frozen.state_dict(), f"{out_dir}/frozen_clip.pt")
    
    # ========================================
    # Experiment 2: Unfrozen CLIP (no contrastive)
    # ========================================
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: Unfrozen CLIP (no contrastive)")
    print("="*70)
    
    lit_model_unfrozen = LitDiffusionCLIP(
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
    ).to(device)
    
    # Pre-training similarity
    print("\nComputing pre-training embedding similarity...")
    sim_unfrozen_pre = compute_embedding_similarity(lit_model_unfrozen, cached_samples, device)
    print(f"  Avg: {sim_unfrozen_pre['avg_similarity']:.4f}")
    print(f"  Min: {sim_unfrozen_pre['min_similarity']:.4f}")
    print(f"  Max: {sim_unfrozen_pre['max_similarity']:.4f}")
    
    # Training
    print("\nTraining Unfrozen CLIP...")
    optimizer = torch.optim.AdamW(lit_model_unfrozen.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(MAX_EPOCHS):
        lit_model_unfrozen.train()
        batch = simple_collate(cached_samples)
        batch["data"] = batch["data"].to(device)
        batch["conditions"]["input_pose"] = batch["conditions"]["input_pose"].to(device)
        batch["conditions"]["sign_image"] = batch["conditions"]["sign_image"].to(device)
        
        optimizer.zero_grad()
        loss = lit_model_unfrozen.training_step(batch, 0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lit_model_unfrozen.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"  Epoch {epoch}/{MAX_EPOCHS}, Loss: {loss.item():.4f}")
    
    # Post-training similarity
    print("\nComputing post-training embedding similarity...")
    sim_unfrozen_post = compute_embedding_similarity(lit_model_unfrozen, cached_samples, device)
    print(f"  Avg: {sim_unfrozen_post['avg_similarity']:.4f}")
    print(f"  Min: {sim_unfrozen_post['min_similarity']:.4f}")
    print(f"  Max: {sim_unfrozen_post['max_similarity']:.4f}")
    
    # Evaluation
    print("\nEvaluating Unfrozen CLIP...")
    results_unfrozen = evaluate_model(lit_model_unfrozen, cached_samples, device)
    
    avg_normal_unfrozen = np.mean([r['pck_normal'] for r in results_unfrozen])
    avg_so_unfrozen = np.mean([r['pck_signonly'] for r in results_unfrozen])
    avg_ratio_unfrozen = np.mean([r['disp_ratio'] for r in results_unfrozen])
    
    all_results['unfrozen'] = {
        'similarity_pre': sim_unfrozen_pre,
        'similarity_post': sim_unfrozen_post,
        'pck_normal': avg_normal_unfrozen,
        'pck_signonly': avg_so_unfrozen,
        'disp_ratio': avg_ratio_unfrozen,
    }
    
    print(f"\n  Unfrozen CLIP Results:")
    print(f"    Similarity (pre):  avg={sim_unfrozen_pre['avg_similarity']:.4f}, min={sim_unfrozen_pre['min_similarity']:.4f}")
    print(f"    Similarity (post): avg={sim_unfrozen_post['avg_similarity']:.4f}, min={sim_unfrozen_post['min_similarity']:.4f}")
    print(f"    Normal PCK: {avg_normal_unfrozen:.1f}%")
    print(f"    Sign-Only PCK: {avg_so_unfrozen:.1f}%")
    print(f"    Disp Ratio: {avg_ratio_unfrozen:.3f}")
    
    torch.save(lit_model_unfrozen.state_dict(), f"{out_dir}/unfrozen_clip.pt")
    
    # ========================================
    # Summary
    # ========================================
    print(f"\n{'='*70}")
    print("SUMMARY: CLIP Ablation (32 samples, no contrastive)")
    print("="*70)
    
    print(f"\n{'Configuration':<25} {'Avg Sim':<10} {'Min Sim':<10} {'Normal PCK':<12} {'Sign-Only PCK':<14} {'Ratio':<8}")
    print("-" * 79)
    print(f"{'Frozen CLIP':<25} {all_results['frozen']['similarity_post']['avg_similarity']:<10.4f} "
          f"{all_results['frozen']['similarity_post']['min_similarity']:<10.4f} "
          f"{all_results['frozen']['pck_normal']:<12.1f} "
          f"{all_results['frozen']['pck_signonly']:<14.1f} "
          f"{all_results['frozen']['disp_ratio']:<8.3f}")
    print(f"{'Unfrozen CLIP':<25} {all_results['unfrozen']['similarity_post']['avg_similarity']:<10.4f} "
          f"{all_results['unfrozen']['similarity_post']['min_similarity']:<10.4f} "
          f"{all_results['unfrozen']['pck_normal']:<12.1f} "
          f"{all_results['unfrozen']['pck_signonly']:<14.1f} "
          f"{all_results['unfrozen']['disp_ratio']:<8.3f}")
    
    print(f"\n{'='*70}")
    print("Embedding Similarity Change (Unfrozen):")
    print(f"  Avg Similarity: {sim_unfrozen_pre['avg_similarity']:.4f} → {sim_unfrozen_post['avg_similarity']:.4f} "
          f"({sim_unfrozen_post['avg_similarity'] - sim_unfrozen_pre['avg_similarity']:+.4f})")
    print(f"  Min Similarity: {sim_unfrozen_pre['min_similarity']:.4f} → {sim_unfrozen_post['min_similarity']:.4f} "
          f"({sim_unfrozen_post['min_similarity'] - sim_unfrozen_pre['min_similarity']:+.4f})")
    print("="*70)
    
    # Save results
    import json
    with open(f"{out_dir}/results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {out_dir}/")
    print("Done!")


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    run_ablation()