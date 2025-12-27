"""
Enhanced FSW Encoder + Full Dataset Training Script

Based on train_fsw_enhanced.py but for full dataset training.

Usage:
    python train_fsw_full.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

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
# Enhanced FSW Encoder (same as before)
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
        
        self.category_embed = nn.Embedding(64, num_latent_dims // 4)
        self.shape_embed = nn.Embedding(256, num_latent_dims // 4)
        self.variation_embed = nn.Embedding(4096, num_latent_dims // 2)
        
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, num_latent_dims),
            nn.LayerNorm(num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )
        
        self.symbol_fusion = nn.Sequential(
            nn.Linear(num_latent_dims * 2, num_latent_dims),
            nn.LayerNorm(num_latent_dims),
            nn.GELU(),
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_latent_dims,
            nhead=4,
            dim_feedforward=num_latent_dims * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.attn_query = nn.Parameter(torch.randn(1, 1, num_latent_dims))
        
        self.output_proj = nn.Sequential(
            nn.Linear(num_latent_dims, num_latent_dims),
            nn.LayerNorm(num_latent_dims),
        )
        
    def parse_fsw(self, fsw_string: str):
        """Parse FSW into hierarchical symbol info."""
        if not fsw_string or not fsw_string.strip():
            return [(0, 0, 0, 0.0, 0.0)]
        
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
                
                if symbol_str.startswith('S'):
                    symbol_str = symbol_str[1:]
                
                try:
                    full_id = int(symbol_str, 16)
                    category = (full_id >> 16) & 0xF
                    shape = (full_id >> 12) & 0xFF
                    variation = full_id & 0xFFF
                except:
                    category, shape, variation = 0, 0, 0
                
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
            
            while len(symbols) < self.max_symbols:
                symbols.append((0, 0, 0, 0.0, 0.0))
            
            cats = torch.tensor([s[0] for s in symbols], device=device)
            shapes = torch.tensor([s[1] for s in symbols], device=device)
            vars_ = torch.tensor([s[2] for s in symbols], device=device)
            positions = torch.tensor([[s[3], s[4]] for s in symbols], device=device, dtype=torch.float)
            
            cat_emb = self.category_embed(cats)
            shape_emb = self.shape_embed(shapes)
            var_emb = self.variation_embed(vars_)
            
            symbol_emb = torch.cat([cat_emb, shape_emb, var_emb], dim=-1)
            pos_emb = self.pos_encoder(positions)
            
            combined = self.symbol_fusion(torch.cat([symbol_emb, pos_emb], dim=-1))
            
            batch_embeddings.append(combined)
            
            mask = torch.zeros(self.max_symbols, device=device, dtype=torch.bool)
            mask[num_symbols:] = True
            batch_masks.append(mask)
        
        batch_emb = torch.stack(batch_embeddings)
        batch_mask = torch.stack(batch_masks)
        
        transformed = self.transformer(batch_emb, src_key_padding_mask=batch_mask)
        
        query = self.attn_query.expand(batch_size, -1, -1)
        
        scores = torch.bmm(query, transformed.transpose(1, 2))
        scores = scores.masked_fill(batch_mask.unsqueeze(1), float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(attn_weights, transformed).squeeze(1)
        
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


# ============================================================
# Custom Collate Function for Full Dataset
# ============================================================

class FullDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper that includes FSW strings in the output."""
    
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.records = base_dataset.records
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        # Get FSW string
        record = self.records[idx]
        fsw = get_fsw_from_record(record)
        
        # Process tensors
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
        
        return {
            'data': data.float(),
            'input_pose': input_pose.float(),
            'sign_image': sample['conditions']['sign_image'].float(),
            'fsw': fsw,
            'idx': idx,
        }


def full_collate_fn(batch):
    """Collate function that handles FSW strings and variable length sequences."""
    TARGET_LEN = 20  # Force all sequences to 20 frames
    
    processed_data = []
    processed_input = []
    
    for s in batch:
        data = s['data']
        inp = s['input_pose']
        
        # Truncate or pad data to TARGET_LEN
        if data.shape[0] > TARGET_LEN:
            data = data[:TARGET_LEN]
        elif data.shape[0] < TARGET_LEN:
            # Pad with last frame
            pad_size = TARGET_LEN - data.shape[0]
            if data.dim() == 4:
                padding = data[-1:].repeat(pad_size, 1, 1, 1)
            else:
                padding = data[-1:].repeat(pad_size, 1, 1)
            data = torch.cat([data, padding], dim=0)
        
        # Same for input_pose (40 frames)
        INPUT_LEN = 40
        if inp.shape[0] > INPUT_LEN:
            inp = inp[:INPUT_LEN]
        elif inp.shape[0] < INPUT_LEN:
            pad_size = INPUT_LEN - inp.shape[0]
            if inp.dim() == 4:
                padding = inp[-1:].repeat(pad_size, 1, 1, 1)
            else:
                padding = inp[-1:].repeat(pad_size, 1, 1)
            inp = torch.cat([inp, padding], dim=0)
        
        processed_data.append(data)
        processed_input.append(inp)
    
    return {
        'data': torch.stack(processed_data),
        'conditions': {
            'input_pose': torch.stack(processed_input),
            'sign_image': torch.stack([s['sign_image'] for s in batch]),
        },
        'fsw_strings': [s['fsw'] for s in batch],
        'indices': [s['idx'] for s in batch],
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
    """Enhanced Diffusion model with CLIP + Enhanced FSW encoder."""
    
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
        
        self.fsw_encoder = EnhancedFSWEncoder(num_latent_dims, max_symbols=20)
        
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
        
        clip_emb = self.embed_signwriting(signwriting_im_batch)
        
        if fsw_strings is not None:
            self.fsw_encoder = self.fsw_encoder.to(device)
            fsw_emb = self.fsw_encoder(fsw_strings)
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
# Lightning Module for Full Dataset
# ============================================================

class LitDiffusionFull(pl.LightningModule):
    """Lightning module for full dataset training."""
    
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
        fsw_strings = batch.get("fsw_strings", None)

        gt_norm = self.normalize(gt_btjc)
        past_norm = self.normalize(past_btjc)

        batch_size = gt_norm.shape[0]
        device = gt_norm.device

        gt_bjct = self.btjc_to_bjct(gt_norm)
        past_bjct = self.btjc_to_bjct(past_norm)

        # === PAST DROPOUT ===
        if self.training and self.past_dropout > 0:
            # Per-sample dropout
            dropout_mask = torch.rand(batch_size, device=device) < self.past_dropout
            past_bjct = past_bjct.clone()
            past_bjct[dropout_mask] = 0

        timestep = torch.randint(0, self.diffusion_steps, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(gt_bjct)
        x_noisy = self.diffusion.q_sample(gt_bjct, timestep, noise=noise)

        fsw_input = fsw_strings if self.use_fsw else None
        pred_x0_bjct = self.model(x_noisy, timestep, past_bjct, sign_img, fsw_input)

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
                  f"vel={loss_vel.item():.4f}, disp_ratio={disp_ratio:.3f}")

        self.log_dict({
            "train/loss": loss,
            "train/loss_mse": loss_mse,
            "train/loss_vel": loss_vel,
            "train/disp_ratio": disp_ratio,
        }, prog_bar=True, batch_size=batch_size)

        self._step_count += 1
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1000, T_mult=2, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }


# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_model(lit_model, eval_loader, device, num_eval_samples=100):
    """Evaluate model on a subset of data."""
    lit_model.eval()
    
    results = []
    sample_count = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            if sample_count >= num_eval_samples:
                break
            
            cond = batch["conditions"]
            past_raw = sanitize_btjc(cond["input_pose"]).to(device)
            sign = cond["sign_image"].float().to(device)
            gt_raw = sanitize_btjc(batch["data"]).to(device)
            fsw_strings = batch.get("fsw_strings", None)
            
            past_norm = lit_model.normalize(past_raw)
            gt_norm = lit_model.normalize(gt_raw)
            
            past_bjct = lit_model.btjc_to_bjct(past_norm)
            B, J, C, _ = past_bjct.shape
            T_future = gt_norm.shape[1]
            target_shape = (B, J, C, T_future)
            
            class _Wrapper(nn.Module):
                def __init__(self, model, past, sign, fsw):
                    super().__init__()
                    self.model, self.past, self.sign, self.fsw = model, past, sign, fsw
                def forward(self, x, t, **kwargs):
                    return self.model(x, t, self.past, self.sign, self.fsw)
            
            # Normal inference
            wrapped = _Wrapper(lit_model.model, past_bjct, sign, fsw_strings)
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
            wrapped_so = _Wrapper(lit_model.model, zeros_past, sign, fsw_strings)
            pred_so_bjct = lit_model.diffusion.p_sample_loop(
                model=wrapped_so,
                shape=target_shape,
                clip_denoised=True,
                model_kwargs={"y": {}},
                progress=False,
            )
            pred_so_norm = lit_model.bjct_to_btjc(pred_so_bjct)
            
            # Compute metrics per sample
            for i in range(B):
                if sample_count >= num_eval_samples:
                    break
                
                pred_np = pred_norm[i].cpu().numpy()
                pred_so_np = pred_so_norm[i].cpu().numpy()
                gt_np = gt_norm[i].cpu().numpy()
                
                err_normal = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
                err_so = np.sqrt(((pred_so_np - gt_np) ** 2).sum(-1))
                
                pck_normal = (err_normal < 0.1).mean() * 100
                pck_so = (err_so < 0.1).mean() * 100
                
                ratio = mean_frame_disp(pred_norm[i:i+1]) / (mean_frame_disp(gt_norm[i:i+1]) + 1e-8)
                
                results.append({
                    'pck_normal': pck_normal,
                    'pck_signonly': pck_so,
                    'disp_ratio': ratio,
                })
                
                sample_count += 1
    
    return results


# ============================================================
# Main Training
# ============================================================

def train_full():
    """Full dataset training."""
    pl.seed_everything(42)

    # ========== CONFIG ==========
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = "logs/fsw_full_p30"

    BATCH_SIZE = 1024
    MAX_EPOCHS = 100
    DIFFUSION_STEPS = 8
    LEARNING_RATE = 1e-4
    USE_FSW = True
    PAST_DROPOUT = 0.3
    NUM_WORKERS = 8
    EVAL_EVERY = 10  # Evaluate every N epochs
    NUM_EVAL_SAMPLES = 100
    
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(" FULL DATASET TRAINING - Enhanced FSW + CLIP")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Max Epochs: {MAX_EPOCHS}")
    print(f"  Diffusion Steps: {DIFFUSION_STEPS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Use FSW: {USE_FSW}")
    print(f"  Past Dropout: {PAST_DROPOUT}")
    print(f"  Num Workers: {NUM_WORKERS}")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    print(f"Dataset size: {len(base_ds)}")
    
    # Wrap dataset
    train_ds = FullDatasetWrapper(base_ds)
    
    # DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=full_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    # For evaluation (smaller batch size)
    eval_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=full_collate_fn,
        pin_memory=True,
    )
    
    print(f"Batches per epoch: {len(train_loader)}")

    # Get dimensions
    sample = base_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    num_joints = sample.shape[-2]
    num_dims = sample.shape[-1]
    future_len = sample.shape[0]
    
    print(f"Dimensions: J={num_joints}, D={num_dims}, T_future={future_len}")

    # Initialize model
    print("\nInitializing model...")
    lit_model = LitDiffusionFull(
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader) * 10, T_mult=2, eta_min=1e-6
    )
    
    best_pck = 0
    
    for epoch in range(MAX_EPOCHS):
        lit_model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch["data"] = batch["data"].to(device)
            batch["conditions"]["input_pose"] = batch["conditions"]["input_pose"].to(device)
            batch["conditions"]["sign_image"] = batch["conditions"]["sign_image"].to(device)
            
            optimizer.zero_grad()
            loss = lit_model.training_step(batch, batch_idx)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lit_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{MAX_EPOCHS} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
        
        # Evaluation
        if (epoch + 1) % EVAL_EVERY == 0:
            print(f"\n  Evaluating on {NUM_EVAL_SAMPLES} samples...")
            results = evaluate_model(lit_model, eval_loader, device, NUM_EVAL_SAMPLES)
            
            avg_normal = np.mean([r['pck_normal'] for r in results])
            avg_so = np.mean([r['pck_signonly'] for r in results])
            avg_ratio = np.mean([r['disp_ratio'] for r in results])
            
            print(f"  Normal PCK: {avg_normal:.1f}%")
            print(f"  Sign-Only PCK: {avg_so:.1f}%")
            print(f"  Gap: {avg_normal - avg_so:.1f}%")
            print(f"  Disp Ratio: {avg_ratio:.3f}")
            
            # Save best model
            if avg_normal > best_pck:
                best_pck = avg_normal
                torch.save(lit_model.state_dict(), f"{out_dir}/best_model.pt")
                print(f"  ✓ Saved best model (PCK: {best_pck:.1f}%)")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': lit_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"{out_dir}/checkpoint_epoch{epoch+1}.pt")

    # Final evaluation
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print("="*70)
    
    results = evaluate_model(lit_model, eval_loader, device, NUM_EVAL_SAMPLES)
    
    avg_normal = np.mean([r['pck_normal'] for r in results])
    avg_so = np.mean([r['pck_signonly'] for r in results])
    avg_ratio = np.mean([r['disp_ratio'] for r in results])
    
    print(f"\n  Average Normal PCK@0.1: {avg_normal:.1f}%")
    print(f"  Average Sign-Only PCK@0.1: {avg_so:.1f}%")
    print(f"  Average Disp Ratio: {avg_ratio:.3f}")
    print(f"  Gap (Normal - SignOnly): {avg_normal - avg_so:.1f}%")
    print(f"  Ratio (SignOnly/Normal): {avg_so/avg_normal:.3f}")
    
    # Save final model
    torch.save(lit_model.state_dict(), f"{out_dir}/final_model.pt")
    print(f"\nModels saved to: {out_dir}/")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/eval_results.csv", index=False)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    train_full()