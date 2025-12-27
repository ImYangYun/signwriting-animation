"""
Enhanced FSW Encoder + Stronger Decoder Training Script

Improvements over train_fsw_overfit_simple.py:
1. EnhancedFSWEncoder: Hierarchical symbol embedding + Transformer + Attention pooling
2. Stronger Decoder: Deeper MLP with LayerNorm
3. Cosine Annealing LR Schedule
4. Save poses for visualization

Usage:
    python train_fsw_enhanced.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as pl

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder
from CAMDM.diffusion.gaussian_diffusion import (
    GaussianDiffusion, ModelMeanType, ModelVarType, LossType
)

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting.formats.swu_to_fsw import swu2fsw
from signwriting.formats.fsw_to_sign import fsw_to_sign


# ============================================================
# Enhanced FSW Encoder
# ============================================================

class EnhancedFSWEncoder(nn.Module):
    """
    Enhanced FSW encoder with:
    1. Hierarchical symbol embedding (category + shape + variation)
    2. Learnable position encoding
    3. Transformer aggregation with attention pooling
    """
    
    def __init__(self, num_latent_dims: int = 256, max_symbols: int = 20):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.max_symbols = max_symbols
        
        # === Hierarchical Symbol Embedding ===
        # SignWriting符号结构: S + 基类(2位hex) + 变体(3位hex)
        # e.g., S10000 → category=1, shape=0, variation=000
        self.category_embed = nn.Embedding(64, num_latent_dims // 4)   # 第1位hex (0-3)
        self.shape_embed = nn.Embedding(256, num_latent_dims // 4)     # 第2位hex  
        self.variation_embed = nn.Embedding(4096, num_latent_dims // 2) # 后3位hex
        
        # === Position Encoding (learnable) ===
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, num_latent_dims),
            nn.LayerNorm(num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )
        
        # === Symbol Feature Fusion ===
        self.symbol_fusion = nn.Sequential(
            nn.Linear(num_latent_dims * 2, num_latent_dims),
            nn.LayerNorm(num_latent_dims),
            nn.GELU(),
        )
        
        # === Transformer for symbol interaction ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_latent_dims,
            nhead=4,
            dim_feedforward=num_latent_dims * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # === Attention Pooling ===
        self.attn_query = nn.Parameter(torch.randn(1, 1, num_latent_dims))
        
        # === Output projection ===
        self.output_proj = nn.Sequential(
            nn.Linear(num_latent_dims, num_latent_dims),
            nn.LayerNorm(num_latent_dims),
        )
        
    def parse_fsw(self, fsw_string: str):
        """Parse FSW into hierarchical symbol info."""
        if not fsw_string or not fsw_string.strip():
            return [(0, 0, 0, 0.0, 0.0)]  # (cat, shape, var, x, y)
        
        try:
            sign = fsw_to_sign(fsw_string)
            if isinstance(sign, dict):
                symbols_list = sign.get('symbols', [])
            else:
                symbols_list = getattr(sign, 'symbols', [])
            
            if not symbols_list:
                return [(0, 0, 0, 0.0, 0.0)]
            
            result = []
            for sym in symbols_list:
                if isinstance(sym, dict):
                    symbol_str = sym.get('symbol', 'S10000')
                    position = sym.get('position', (500, 500))
                else:
                    symbol_str = sym.symbol
                    position = sym.position
                
                # Parse symbol ID hierarchically
                if symbol_str.startswith('S'):
                    symbol_str = symbol_str[1:]
                
                try:
                    # S + 5 hex digits: e.g., "2e748"
                    # category = first digit (0-3 for hand/movement/dynamics/etc)
                    # shape = second digit
                    # variation = last 3 digits
                    full_id = int(symbol_str, 16)
                    category = (full_id >> 16) & 0xF  # top nibble
                    shape = (full_id >> 12) & 0xFF    # next byte-ish
                    variation = full_id & 0xFFF       # bottom 12 bits
                except:
                    category, shape, variation = 0, 0, 0
                
                # Normalize position
                x = (position[0] - 500) / 250.0
                y = (position[1] - 500) / 250.0
                x = max(-2.0, min(2.0, x))
                y = max(-2.0, min(2.0, y))
                
                result.append((category % 64, shape % 256, variation % 4096, x, y))
            
            return result if result else [(0, 0, 0, 0.0, 0.0)]
        except:
            return [(0, 0, 0, 0.0, 0.0)]
    
    def forward(self, fsw_batch: list):
        device = self.category_embed.weight.device
        batch_size = len(fsw_batch)
        
        batch_embeddings = []
        batch_masks = []
        
        for fsw in fsw_batch:
            symbols = self.parse_fsw(fsw)[:self.max_symbols]
            num_symbols = len(symbols)
            
            # Pad to max_symbols
            while len(symbols) < self.max_symbols:
                symbols.append((0, 0, 0, 0.0, 0.0))
            
            # Create tensors
            cats = torch.tensor([s[0] for s in symbols], device=device)
            shapes = torch.tensor([s[1] for s in symbols], device=device)
            vars_ = torch.tensor([s[2] for s in symbols], device=device)
            positions = torch.tensor([[s[3], s[4]] for s in symbols], device=device, dtype=torch.float)
            
            # Embed hierarchically
            cat_emb = self.category_embed(cats)      # [N, D/4]
            shape_emb = self.shape_embed(shapes)     # [N, D/4]
            var_emb = self.variation_embed(vars_)    # [N, D/2]
            
            symbol_emb = torch.cat([cat_emb, shape_emb, var_emb], dim=-1)  # [N, D]
            pos_emb = self.pos_encoder(positions)                          # [N, D]
            
            # Fuse symbol + position
            combined = self.symbol_fusion(torch.cat([symbol_emb, pos_emb], dim=-1))  # [N, D]
            
            batch_embeddings.append(combined)
            
            # Mask for padding
            mask = torch.zeros(self.max_symbols, device=device, dtype=torch.bool)
            mask[num_symbols:] = True
            batch_masks.append(mask)
        
        # Stack batch
        batch_emb = torch.stack(batch_embeddings)  # [B, N, D]
        batch_mask = torch.stack(batch_masks)      # [B, N]
        
        # Transformer with mask
        transformed = self.transformer(batch_emb, src_key_padding_mask=batch_mask)  # [B, N, D]
        
        # Attention pooling
        query = self.attn_query.expand(batch_size, -1, -1)  # [B, 1, D]
        
        # Simple attention
        scores = torch.bmm(query, transformed.transpose(1, 2))  # [B, 1, N]
        scores = scores.masked_fill(batch_mask.unsqueeze(1), float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(attn_weights, transformed).squeeze(1)  # [B, D]
        
        output = self.output_proj(pooled)
        return output


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


class SignWritingToPoseDiffusionEnhanced(nn.Module):
    """
    Enhanced Diffusion model with:
    - CLIP + Enhanced FSW encoder
    - Stronger decoder with LayerNorm
    """
    
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
        
        # Enhanced FSW encoder
        self.fsw_encoder = EnhancedFSWEncoder(num_latent_dims, max_symbols=20)
        
        # Fusion: CLIP + FSW (with gating)
        self.sign_fusion = nn.Sequential(
            nn.Linear(num_latent_dims * 2, num_latent_dims * 2),
            nn.LayerNorm(num_latent_dims * 2),
            nn.GELU(),
            nn.Linear(num_latent_dims * 2, num_latent_dims),
            nn.LayerNorm(num_latent_dims),
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

        # === ENHANCED DECODER: Deeper with LayerNorm ===
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

class LitDiffusionEnhanced(pl.LightningModule):
    """Enhanced Lightning module with CLIP + FSW + LR schedule."""
    
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
                 use_fsw=True,
                 past_dropout=0.3):
        super().__init__()
        self.save_hyperparameters()

        self.diffusion_steps = diffusion_steps
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.use_fsw = use_fsw
        self.past_dropout = past_dropout
        self._step_count = 0

        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1, 1, -1, 3)
        std = stats["std"].float().view(1, 1, -1, 3)
        self.register_buffer("mean_pose", mean.clone())
        self.register_buffer("std_pose", std.clone())

        self.model = SignWritingToPoseDiffusionEnhanced(
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

        # === PAST DROPOUT ===
        if self.training and self.past_dropout > 0:
            if torch.rand(1).item() < self.past_dropout:
                past_bjct = torch.zeros_like(past_bjct)

        timestep = torch.randint(0, self.diffusion_steps, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(gt_bjct)
        x_noisy = self.diffusion.q_sample(gt_bjct, timestep, noise=noise)

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

        if self._step_count % 100 == 0:
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
# Pose Saving Utilities
# ============================================================

def save_pose_for_visualization(lit_model, cached_samples, fsw_strings_cache, full_ds, out_dir, device, sample_idx=0):
    """Save generated poses for visualization."""
    
    lit_model.eval()
    
    # Get sample
    batch = simple_collate([cached_samples[sample_idx]])
    
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
        
        wrapped = _Wrapper(lit_model.model, past_bjct, sign, [fsw_strings_cache[sample_idx]])
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
        wrapped_so = _Wrapper(lit_model.model, zeros_past, sign, [fsw_strings_cache[sample_idx]])
        pred_so_bjct = lit_model.diffusion.p_sample_loop(
            model=wrapped_so,
            shape=target_shape,
            clip_denoised=True,
            model_kwargs={"y": {}},
            progress=False,
        )
        pred_so_norm = lit_model.bjct_to_btjc(pred_so_bjct)
    
    # Unnormalize
    pred_unnorm = lit_model.unnormalize(pred_norm)[0].cpu()      # [T, J, C]
    pred_so_unnorm = lit_model.unnormalize(pred_so_norm)[0].cpu()
    gt_unnorm = lit_model.unnormalize(gt_norm)[0].cpu()
    past_unnorm = lit_model.unnormalize(past_raw)[0].cpu()
    
    # Get original pose header
    record = full_ds.records[sample_idx]
    pose_path = os.path.join(full_ds.data_dir, record['pose'])
    with open(pose_path, 'rb') as f:
        ref_pose = Pose.read(f.read())
    
    # Get reduced header
    reduced_pose = reduce_holistic(ref_pose)
    header = reduced_pose.header
    
    def save_tensor_as_pose(tensor, filename, header, fps=25.0):
        """Save tensor [T, J, C] as .pose file"""
        data = tensor.numpy().astype(np.float32)
        # Add person dimension: [T, 1, J, C]
        data = data[:, np.newaxis, :, :]
        T = data.shape[0]
        conf = np.ones((T, 1, data.shape[2]), dtype=np.float32)
        
        body = NumPyPoseBody(fps=fps, data=data, confidence=conf)
        pose = Pose(header=header, body=body)
        
        with open(filename, 'wb') as f:
            pose.write(f)
        print(f"  Saved: {filename}")
    
    # Save all poses
    pose_dir = f"{out_dir}/poses"
    os.makedirs(pose_dir, exist_ok=True)
    
    save_tensor_as_pose(gt_unnorm, f"{pose_dir}/ground_truth.pose", header)
    save_tensor_as_pose(pred_unnorm, f"{pose_dir}/pred_normal.pose", header)
    save_tensor_as_pose(pred_so_unnorm, f"{pose_dir}/pred_sign_only.pose", header)
    save_tensor_as_pose(past_unnorm, f"{pose_dir}/past_context.pose", header)
    
    # Full sequences (past + future)
    full_pred = torch.cat([past_unnorm, pred_unnorm], dim=0)
    full_gt = torch.cat([past_unnorm, gt_unnorm], dim=0)
    full_so = torch.cat([torch.zeros_like(past_unnorm), pred_so_unnorm], dim=0)
    
    save_tensor_as_pose(full_pred, f"{pose_dir}/full_pred.pose", header)
    save_tensor_as_pose(full_gt, f"{pose_dir}/full_gt.pose", header)
    save_tensor_as_pose(full_so, f"{pose_dir}/full_sign_only.pose", header)
    
    # Also save the SignWriting image
    from PIL import Image
    sign_img = sign[0].cpu().permute(1, 2, 0).numpy()
    sign_img = ((sign_img - sign_img.min()) / (sign_img.max() - sign_img.min() + 1e-8) * 255).astype(np.uint8)
    Image.fromarray(sign_img).save(f"{pose_dir}/signwriting.png")
    
    print(f"\n✅ Poses saved to {pose_dir}/")
    print(f"   - ground_truth.pose: GT future frames")
    print(f"   - pred_normal.pose: Normal prediction (past + sign)")
    print(f"   - pred_sign_only.pose: Sign-only prediction (zeros + sign)")
    print(f"   - full_pred.pose: Past + predicted future")
    print(f"   - full_gt.pose: Past + GT future")
    print(f"   - signwriting.png: Input SignWriting image")
    
    return {
        'pred_normal': pred_unnorm,
        'pred_sign_only': pred_so_unnorm,
        'ground_truth': gt_unnorm,
        'past': past_unnorm,
    }


# ============================================================
# Main Training
# ============================================================

def train_enhanced():
    """Enhanced training with FSW encoder and pose saving."""
    pl.seed_everything(42)

    # Config
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = "logs/fsw_enhanced_32s_p50_5000"

    NUM_SAMPLES = 32
    MAX_EPOCHS = 5000
    DIFFUSION_STEPS = 8
    LEARNING_RATE = 1e-4
    USE_FSW = True
    PAST_DROPOUT = 0.5
    
    # LR Schedule config
    USE_LR_SCHEDULE = True
    LR_T0 = 500  # First restart period
    
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(" ENHANCED FSW + CLIP TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Samples: {NUM_SAMPLES}")
    print(f"  Epochs: {MAX_EPOCHS}")
    print(f"  Use FSW: {USE_FSW} (Enhanced Encoder)")
    print(f"  Past Dropout: {PAST_DROPOUT}")
    print(f"  LR Schedule: CosineAnnealingWarmRestarts (T0={LR_T0})")
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
    
    print(f"Full dataset size: {len(full_ds)}")
    
    # Cache samples
    print("\nCaching fixed samples for overfitting...")
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
    
    print(f"  Cached {NUM_SAMPLES} samples")
    
    # Extract FSW strings
    print("\nExtracting FSW strings...")
    fsw_strings_cache = []
    for i in range(NUM_SAMPLES):
        record = full_ds.records[i]
        fsw = get_fsw_from_record(record)
        fsw_strings_cache.append(fsw)
        if i < 5:
            print(f"  [{i}] FSW: {fsw[:50]}..." if len(fsw) > 50 else f"  [{i}] FSW: {fsw}")

    # Get dimensions
    sample = full_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    num_joints = sample.shape[-2]
    num_dims = sample.shape[-1]
    future_len = sample.shape[0]
    
    print(f"\nDimensions: J={num_joints}, D={num_dims}, T_future={future_len}")

    # Initialize model
    print("\nInitializing enhanced model...")
    lit_model = LitDiffusionEnhanced(
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
        past_dropout=PAST_DROPOUT,
    )

    total_params = sum(p.numel() for p in lit_model.parameters())
    trainable_params = sum(p.numel() for p in lit_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Training
    print(f"\n{'='*70}")
    print("STARTING TRAINING...")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    lit_model = lit_model.to(device)
    
    optimizer = torch.optim.AdamW(lit_model.parameters(), lr=LEARNING_RATE)
    
    # LR Scheduler
    if USE_LR_SCHEDULE:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=LR_T0, T_mult=2, eta_min=1e-6
        )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(MAX_EPOCHS):
        lit_model.train()
        
        batch = simple_collate(cached_samples)
        
        batch["data"] = batch["data"].to(device)
        batch["conditions"]["input_pose"] = batch["conditions"]["input_pose"].to(device)
        batch["conditions"]["sign_image"] = batch["conditions"]["sign_image"].to(device)
        
        lit_model.current_fsw_strings = fsw_strings_cache
        
        optimizer.zero_grad()
        loss = lit_model.training_step(batch, 0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lit_model.parameters(), 1.0)
        optimizer.step()
        
        if USE_LR_SCHEDULE:
            scheduler.step()
        
        # Save best
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(lit_model.state_dict(), f"{out_dir}/best_model.pt")
        
        if epoch % 500 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{MAX_EPOCHS}, LR: {current_lr:.2e}, Best Loss: {best_loss:.4f}")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print("="*70)

    # Evaluation
    print("\nEvaluating...")
    lit_model.eval()
    
    results = []
    for idx in range(NUM_SAMPLES):
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
            class _Wrapper(nn.Module):
                def __init__(self, model, past, sign, fsw):
                    super().__init__()
                    self.model, self.past, self.sign, self.fsw = model, past, sign, fsw
                def forward(self, x, t, **kwargs):
                    return self.model(x, t, self.past, self.sign, self.fsw)
            
            # Normal
            wrapped = _Wrapper(lit_model.model, past_bjct, sign, [fsw_strings_cache[idx]])
            pred_bjct = lit_model.diffusion.p_sample_loop(
                model=wrapped,
                shape=target_shape,
                clip_denoised=True,
                model_kwargs={"y": {}},
                progress=False,
            )
            pred_norm = lit_model.bjct_to_btjc(pred_bjct)
            
            # Sign-Only
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
    print(f"  Gap (Normal - SignOnly): {avg_normal - avg_so:.1f}%")
    print(f"  Ratio (SignOnly/Normal): {avg_so/avg_normal:.3f}")
    
    # Save final checkpoint
    torch.save(lit_model.state_dict(), f"{out_dir}/final_model.pt")
    print(f"\nModels saved to: {out_dir}/")
    
    # Save poses for visualization
    print("\n" + "="*70)
    print("SAVING POSES FOR VISUALIZATION...")
    print("="*70)
    
    # Save for first 3 samples
    for sample_idx in range(min(3, NUM_SAMPLES)):
        print(f"\nSample {sample_idx}:")
        save_pose_for_visualization(
            lit_model, cached_samples, fsw_strings_cache, 
            full_ds, f"{out_dir}/sample_{sample_idx}", device, sample_idx
        )
    
    # Save results to CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/results.csv", index=False)
    print(f"\nResults saved to: {out_dir}/results.csv")
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    train_enhanced()