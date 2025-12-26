"""
FSW Encoder + Unfrozen CLIP Overfitting Experiment

Parse SignWriting FSW strings into structured features:
- Symbol IDs → Learned embeddings
- Positions → Positional encoding
- Combine with CLIP for richer representation

Usage:
    python train_fsw_overfit.py
"""
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader, Dataset
from pose_format import Pose
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.utils.generic import reduce_holistic
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder
from CAMDM.diffusion.gaussian_diffusion import (
    GaussianDiffusion, ModelMeanType, ModelVarType, LossType
)

# SignWriting parsing
from signwriting.formats.swu_to_fsw import swu2fsw
from signwriting.formats.fsw_to_sign import fsw_to_sign


# ============================================================
# FSW Encoder
# ============================================================

class FSWEncoder(nn.Module):
    """
    Encode Formal SignWriting (FSW) into dense vectors.
    
    FSW structure:
    - Each sign has multiple symbols
    - Each symbol has: symbol_id (e.g., S10000) + position (x, y)
    - Symbol ID encodes: category (hand, movement, face, etc.) + specific shape
    
    Encoding strategy:
    - Symbol ID → Learned embedding
    - Position → Normalized coordinates + positional encoding
    - Aggregate with attention pooling
    """
    
    def __init__(self, num_latent_dims: int = 256, max_symbols: int = 20):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.max_symbols = max_symbols
        
        # Symbol embedding: ~40000 possible symbols, but we use a hash trick
        # Symbol IDs are like S10000-S38b07, we embed the base + variation
        self.symbol_base_embed = nn.Embedding(100, num_latent_dims // 2)  # Category (first 2 digits after S)
        self.symbol_var_embed = nn.Embedding(1000, num_latent_dims // 2)  # Variation
        
        # Position encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )
        
        # Combine symbol + position
        self.combine = nn.Sequential(
            nn.Linear(num_latent_dims * 2, num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )
        
        # Attention pooling over symbols
        self.attn_query = nn.Parameter(torch.randn(1, 1, num_latent_dims))
        self.attn = nn.MultiheadAttention(num_latent_dims, num_heads=4, batch_first=True)
        
        # Final projection
        self.output_proj = nn.Linear(num_latent_dims, num_latent_dims)
        
    def parse_fsw(self, fsw_string: str):
        """
        Parse FSW string into list of (symbol_id, x, y).
        
        Returns:
            symbols: List of (base_id, var_id, x, y)
        """
        if not fsw_string or not fsw_string.strip():
            return [(0, 0, 0.0, 0.0)]  # Return dummy symbol instead of empty
        
        try:
            sign = fsw_to_sign(fsw_string)
            
            # Handle both dict and object return types
            if isinstance(sign, dict):
                symbols_list = sign.get('symbols', [])
            else:
                symbols_list = sign.symbols if hasattr(sign, 'symbols') else []
            
            if not symbols_list:
                return [(0, 0, 0.0, 0.0)]
            
            symbols = []
            for sym in symbols_list:
                # Handle both dict and object
                if isinstance(sym, dict):
                    symbol_str = sym.get('symbol', 'S10000')
                    position = sym.get('position', (500, 500))
                else:
                    symbol_str = sym.symbol
                    position = sym.position
                
                if symbol_str.startswith('S'):
                    symbol_str = symbol_str[1:]
                
                # Parse: symbol_str is like "10011" (5 hex digits)
                try:
                    full_id = int(symbol_str, 16)
                    base_id = (full_id >> 8) % 100   # Higher bits for category
                    var_id = full_id % 1000          # Lower bits for variation
                except:
                    base_id = 0
                    var_id = 0
                
                # Normalize position to [-1, 1]
                x = (position[0] - 500) / 250.0
                y = (position[1] - 500) / 250.0
                
                # Clamp to avoid extreme values
                x = max(-2.0, min(2.0, x))
                y = max(-2.0, min(2.0, y))
                
                symbols.append((base_id, var_id, x, y))
            
            return symbols if symbols else [(0, 0, 0.0, 0.0)]
        except Exception as e:
            # Return dummy symbol if parsing fails
            print(f"[WARNING] FSW parse failed for '{fsw_string[:30]}...': {e}")
            return [(0, 0, 0.0, 0.0)]
    
    def forward(self, fsw_batch: list):
        """
        Args:
            fsw_batch: List of FSW strings (batch_size,)
        
        Returns:
            embeddings: (batch_size, num_latent_dims)
        """
        device = self.symbol_base_embed.weight.device
        batch_size = len(fsw_batch)
        
        # Parse all FSW strings
        all_symbols = [self.parse_fsw(fsw) for fsw in fsw_batch]
        
        # Pad to max_symbols
        base_ids = torch.zeros(batch_size, self.max_symbols, dtype=torch.long, device=device)
        var_ids = torch.zeros(batch_size, self.max_symbols, dtype=torch.long, device=device)
        positions = torch.zeros(batch_size, self.max_symbols, 2, device=device)
        mask = torch.zeros(batch_size, self.max_symbols, dtype=torch.bool, device=device)
        
        for b, symbols in enumerate(all_symbols):
            for i, (base, var, x, y) in enumerate(symbols[:self.max_symbols]):
                base_ids[b, i] = base
                var_ids[b, i] = var
                positions[b, i, 0] = x
                positions[b, i, 1] = y
                mask[b, i] = True
        
        # Embed symbols
        base_emb = self.symbol_base_embed(base_ids)  # (B, S, D/2)
        var_emb = self.symbol_var_embed(var_ids)     # (B, S, D/2)
        symbol_emb = torch.cat([base_emb, var_emb], dim=-1)  # (B, S, D)
        
        # Encode positions
        pos_emb = self.pos_encoder(positions)  # (B, S, D)
        
        # Combine
        combined = self.combine(torch.cat([symbol_emb, pos_emb], dim=-1))  # (B, S, D)
        
        # Mask invalid symbols
        combined = combined * mask.unsqueeze(-1).float()
        
        # Attention pooling
        query = self.attn_query.expand(batch_size, -1, -1)  # (B, 1, D)
        
        # Create attention mask (True = ignore)
        attn_mask = ~mask  # (B, S)
        
        # Handle empty sequences
        all_masked = attn_mask.all(dim=1)
        if all_masked.any():
            attn_mask[all_masked, 0] = False  # Keep at least one
        
        pooled, _ = self.attn(query, combined, combined, key_padding_mask=attn_mask)
        pooled = pooled.squeeze(1)  # (B, D)
        
        output = self.output_proj(pooled)
        return output


class CombinedSignWritingEncoder(nn.Module):
    """
    Combine FSW structural encoding with CLIP visual encoding.
    
    FSW provides: explicit symbol IDs and positions
    CLIP provides: visual appearance features
    """
    
    def __init__(self, num_latent_dims: int = 256,
                 embedding_arch: str = 'openai/clip-vit-base-patch32',
                 freeze_clip: bool = False,
                 fsw_weight: float = 0.5):
        super().__init__()
        self.fsw_weight = fsw_weight
        
        # FSW encoder
        self.fsw_encoder = FSWEncoder(num_latent_dims)
        
        # CLIP encoder
        self.clip_model = CLIPModel.from_pretrained(embedding_arch)
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        clip_dim = self.clip_model.visual_projection.out_features
        self.clip_proj = nn.Linear(clip_dim, num_latent_dims) if clip_dim != num_latent_dims else nn.Identity()
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(num_latent_dims * 2, num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )
        
    def forward(self, sign_images: torch.Tensor, fsw_strings: list):
        """
        Args:
            sign_images: (B, 3, H, W) - SignWriting images for CLIP
            fsw_strings: List of FSW strings
        """
        # CLIP encoding
        clip_emb = self.clip_model.get_image_features(pixel_values=sign_images)
        clip_emb = self.clip_proj(clip_emb)
        
        # DEBUG: check CLIP output
        if torch.isnan(clip_emb).any():
            print(f"[ERROR] NaN in clip_emb!")
            clip_emb = torch.nan_to_num(clip_emb, nan=0.0)
        
        # FSW encoding - ensure FSW encoder is on same device
        self.fsw_encoder = self.fsw_encoder.to(clip_emb.device)
        fsw_emb = self.fsw_encoder(fsw_strings)
        
        # DEBUG: check FSW output
        if torch.isnan(fsw_emb).any():
            print(f"[ERROR] NaN in fsw_emb!")
            fsw_emb = torch.nan_to_num(fsw_emb, nan=0.0)
        
        # Weighted fusion
        combined = self.fusion(torch.cat([clip_emb, fsw_emb], dim=-1))
        
        return combined


# ============================================================
# Dataset with FSW
# ============================================================

class FSWPosePredictionDataset(Dataset):
    """Dataset that includes FSW strings along with pose data."""
    
    def __init__(self, data_dir: str, csv_path: str, 
                 num_past_frames: int = 40, num_future_frames: int = 20,
                 split: str = "train", indices: list = None):
        self.data_dir = data_dir
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        
        # Load CSV
        df = pd.read_csv(csv_path)
        if split:
            df = df[df['split'] == split]
        
        self.records = df.to_dict('records')
        
        if indices is not None:
            self.records = [self.records[i] for i in indices]
        
        # Load normalization stats
        stats_path = os.path.join(data_dir, "mean_std_178_with_preprocess.pt")
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu")
            self.mean = stats["mean"].float()
            self.std = stats["std"].float()
        else:
            self.mean = None
            self.std = None
    
    def __len__(self):
        return len(self.records)
    
    def _load_pose(self, record):
        """Load and preprocess pose file."""
        from pose_anonymization.data.normalization import unshift_hands
        
        pose_path = record['pose']
        if not os.path.isabs(pose_path):
            pose_path = os.path.join(self.data_dir, pose_path)
        
        with open(pose_path, 'rb') as f:
            pose = Pose.read(f)
        
        # Reduce to 178 keypoints
        pose = reduce_holistic(pose)
        if "POSE_WORLD_LANDMARKS" in [c.name for c in pose.header.components]:
            pose = pose.remove_components(["POSE_WORLD_LANDMARKS"])
        
        # Apply unshift_hands (same as original dataset)
        unshift_hands(pose)
        
        # Get frame range
        start_frame = int(record['start'] * pose.body.fps / 1000)
        end_frame = int(record['end'] * pose.body.fps / 1000)
        
        # Extract data
        data = pose.body.data[start_frame:end_frame, 0]  # (T, J, C)
        
        return torch.from_numpy(data.astype(np.float32))
    
    def _get_fsw(self, record):
        """Get FSW string from SWU text."""
        swu_text = record.get('text', '')
        if not swu_text:
            return ""
        
        # Take first sign if multiple
        swu_first = swu_text.split()[0] if ' ' in swu_text else swu_text
        
        try:
            fsw = swu2fsw(swu_first)
            return fsw
        except:
            return ""
    
    def _render_signwriting_image(self, fsw_string: str):
        """Render SignWriting to image for CLIP."""
        from signwriting.visualizer.visualize import signwriting_to_image
        from PIL import Image
        import torchvision.transforms as T
        
        try:
            if not fsw_string:
                # Return blank image
                img = Image.new('RGB', (224, 224), color='white')
            else:
                img = signwriting_to_image(fsw_string)
                img = img.convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), color='white')
        
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                       std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        
        return transform(img)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        
        # Load pose
        pose_data = self._load_pose(record)  # (T, J, C)
        
        # Handle NaN values
        if torch.isnan(pose_data).any():
            print(f"[WARNING] NaN in pose data for idx {idx}, replacing with 0")
            pose_data = torch.nan_to_num(pose_data, nan=0.0)
        
        T_total, J, C = pose_data.shape
        
        # Split into past and future
        split_point = min(self.num_past_frames, T_total - self.num_future_frames)
        split_point = max(split_point, 1)
        
        past = pose_data[:split_point]
        future = pose_data[split_point:split_point + self.num_future_frames]
        
        # Pad if needed
        if past.shape[0] < self.num_past_frames:
            pad = torch.zeros(self.num_past_frames - past.shape[0], J, C)
            past = torch.cat([pad, past], dim=0)
        
        if future.shape[0] < self.num_future_frames:
            pad = torch.zeros(self.num_future_frames - future.shape[0], J, C)
            future = torch.cat([future, pad], dim=0)
        
        # Get FSW
        fsw_string = self._get_fsw(record)
        
        # Render image
        sign_image = self._render_signwriting_image(fsw_string)
        
        return {
            'data': future,  # (T_future, J, C)
            'conditions': {
                'input_pose': past,  # (T_past, J, C)
                'sign_image': sign_image,  # (3, 224, 224)
                'fsw_string': fsw_string,  # str
            },
            'idx': idx,
        }


def fsw_collate_fn(batch):
    """Custom collate that handles FSW strings."""
    data = torch.stack([b['data'] for b in batch])
    past = torch.stack([b['conditions']['input_pose'] for b in batch])
    images = torch.stack([b['conditions']['sign_image'] for b in batch])
    fsw_strings = [b['conditions']['fsw_string'] for b in batch]
    indices = [b['idx'] for b in batch]
    
    return {
        'data': data,
        'conditions': {
            'input_pose': past,
            'sign_image': images,
            'fsw_string': fsw_strings,
        },
        'idx': indices,
    }


# ============================================================
# Model
# ============================================================

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


class FSWDiffusionModel(nn.Module):
    """Diffusion model with FSW + CLIP combined encoder."""
    
    def __init__(self, num_keypoints: int, num_dims_per_keypoint: int,
                 num_latent_dims: int = 256, num_heads: int = 4, dropout: float = 0.1,
                 t_past: int = 40, t_future: int = 20, freeze_clip: bool = False):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
        self.t_past = t_past
        self.t_future = t_future

        input_feats = num_keypoints * num_dims_per_keypoint

        # Past motion encoder
        self.past_context_encoder = ContextEncoder(
            input_feats, num_latent_dims,
            num_layers=2, num_heads=num_heads, dropout=dropout,
        )
        
        # Combined SignWriting encoder (FSW + CLIP)
        self.sign_encoder = CombinedSignWritingEncoder(
            num_latent_dims=num_latent_dims,
            freeze_clip=freeze_clip,
            fsw_weight=0.5,
        )
        
        # Time embedding
        self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)
        self.time_embed = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)

        # Frame-independent decoder
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

    def forward(self, x, timesteps, past_motion, sign_images, fsw_strings):
        """
        Args:
            x: noisy future poses (B, J, C, T)
            timesteps: diffusion timesteps (B,)
            past_motion: past poses (B, J, C, T_past)
            sign_images: SignWriting images for CLIP (B, 3, H, W)
            fsw_strings: list of FSW strings
        """
        B, J, C, T_future = x.shape
        device = x.device

        # Convert past to BTJC for encoder
        if past_motion.dim() == 4:
            if past_motion.shape[1] == J and past_motion.shape[2] == C:
                past_btjc = past_motion.permute(0, 3, 1, 2).contiguous()
            else:
                past_btjc = past_motion

        # Encode past motion
        past_ctx = self.past_context_encoder(past_btjc)
        
        # Encode SignWriting (FSW + CLIP combined)
        sign_emb = self.sign_encoder(sign_images, fsw_strings)
        
        # Time embedding
        time_emb = self.time_embed(timesteps).squeeze(0)
        
        # Combine context
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
        return result


# ============================================================
# Lightning Module
# ============================================================

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def mean_frame_disp(x_btjc: torch.Tensor) -> float:
    if x_btjc.dim() == 4:
        if x_btjc.size(1) < 2:
            return 0.0
        v = x_btjc[:, 1:] - x_btjc[:, :-1]
    return v.abs().mean().item()


def tensor_to_pose(t_btjc: torch.Tensor, header, ref_pose: Pose, scale_to_ref: bool = True) -> Pose:
    """Convert tensor to Pose format for visualization."""
    from pose_format.numpy.pose_body import NumPyPoseBody
    from pose_anonymization.data.normalization import unshift_hands
    
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


class LitFSWDiffusion(pl.LightningModule):
    """Lightning module for FSW + CLIP diffusion."""
    
    def __init__(self, 
                 num_keypoints=178, 
                 num_dims=3, 
                 lr=1e-4,
                 stats_path="/home/yayun/data/pose_data/mean_std_178_with_preprocess.pt",
                 diffusion_steps=8, 
                 vel_weight=1.0,
                 acc_weight=0.5,
                 contrastive_weight=0.5,
                 t_past=40, 
                 t_future=20,
                 freeze_clip=False):
        super().__init__()
        self.save_hyperparameters()

        self.diffusion_steps = diffusion_steps
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.contrastive_weight = contrastive_weight
        self._step_count = 0

        # Load normalization stats
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu")
            mean = stats["mean"].float().view(1, 1, -1, 3)
            std = stats["std"].float().view(1, 1, -1, 3)
        else:
            mean = torch.zeros(1, 1, num_keypoints, 3)
            std = torch.ones(1, 1, num_keypoints, 3)
        
        self.register_buffer("mean_pose", mean.clone())
        self.register_buffer("std_pose", std.clone())

        # Model
        self.model = FSWDiffusionModel(
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
        cond = batch["conditions"]
        gt_btjc = batch["data"].float()
        past_btjc = cond["input_pose"].float()
        sign_img = cond["sign_image"].float()
        fsw_strings = cond["fsw_string"]

        # DEBUG: Check for NaN in inputs
        if torch.isnan(gt_btjc).any():
            print(f"[ERROR] NaN in gt_btjc!")
            gt_btjc = torch.nan_to_num(gt_btjc, nan=0.0)
        if torch.isnan(past_btjc).any():
            print(f"[ERROR] NaN in past_btjc!")
            past_btjc = torch.nan_to_num(past_btjc, nan=0.0)
        if torch.isnan(sign_img).any():
            print(f"[ERROR] NaN in sign_img!")
            sign_img = torch.nan_to_num(sign_img, nan=0.0)

        # Normalize
        gt_norm = self.normalize(gt_btjc)
        past_norm = self.normalize(past_btjc)
        
        # DEBUG: Check after normalize
        if torch.isnan(gt_norm).any():
            print(f"[ERROR] NaN in gt_norm after normalize!")
            print(f"  gt_btjc range: [{gt_btjc.min():.4f}, {gt_btjc.max():.4f}]")
            print(f"  mean_pose range: [{self.mean_pose.min():.4f}, {self.mean_pose.max():.4f}]")
            print(f"  std_pose range: [{self.std_pose.min():.4f}, {self.std_pose.max():.4f}]")
            gt_norm = torch.nan_to_num(gt_norm, nan=0.0)

        batch_size = gt_norm.shape[0]
        device = gt_norm.device

        gt_bjct = self.btjc_to_bjct(gt_norm)
        past_bjct = self.btjc_to_bjct(past_norm)

        # Sample timestep and add noise
        timestep = torch.randint(0, self.diffusion_steps, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(gt_bjct)
        x_noisy = self.diffusion.q_sample(gt_bjct, timestep, noise=noise)

        # Predict x0
        pred_x0_bjct = self.model(x_noisy, timestep, past_bjct, sign_img, fsw_strings)

        # DEBUG: Check model output
        if torch.isnan(pred_x0_bjct).any():
            print(f"[ERROR] NaN in model output pred_x0_bjct!")
            print(f"  x_noisy has NaN: {torch.isnan(x_noisy).any()}")
            print(f"  timestep: {timestep}")
            print(f"  fsw_strings: {fsw_strings}")
            # Try to identify where NaN comes from
            with torch.no_grad():
                past_ctx = self.model.past_context_encoder(past_bjct.permute(0, 3, 1, 2))
                print(f"  past_ctx has NaN: {torch.isnan(past_ctx).any()}")
                sign_emb = self.model.sign_encoder(sign_img, fsw_strings)
                print(f"  sign_emb has NaN: {torch.isnan(sign_emb).any()}")
            pred_x0_bjct = torch.nan_to_num(pred_x0_bjct, nan=0.0)

        # Losses
        loss_mse = F.mse_loss(pred_x0_bjct, gt_bjct)
        
        pred_vel = pred_x0_bjct[..., 1:] - pred_x0_bjct[..., :-1]
        gt_vel = gt_bjct[..., 1:] - gt_bjct[..., :-1]
        loss_vel = F.mse_loss(pred_vel, gt_vel)

        # Acceleration loss
        loss_acc = torch.tensor(0.0, device=device)
        if pred_vel.size(-1) > 1:
            pred_acc = pred_vel[..., 1:] - pred_vel[..., :-1]
            gt_acc = gt_vel[..., 1:] - gt_vel[..., :-1]
            loss_acc = F.mse_loss(pred_acc, gt_acc)

        # Contrastive loss (for CLIP embeddings)
        loss_contrastive = torch.tensor(0.0, device=device)
        if batch_size > 1 and self.contrastive_weight > 0:
            # Get CLIP embeddings from the combined encoder
            clip_emb = self.model.sign_encoder.clip_model.get_image_features(pixel_values=sign_img)
            clip_emb_norm = F.normalize(clip_emb, p=2, dim=-1)
            cos_sim = torch.mm(clip_emb_norm, clip_emb_norm.t())
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            off_diag_sim = cos_sim[mask]
            loss_contrastive = off_diag_sim.mean()

        loss = (loss_mse + 
                self.vel_weight * loss_vel + 
                self.acc_weight * loss_acc + 
                self.contrastive_weight * loss_contrastive)

        # Displacement ratio
        with torch.no_grad():
            pred_disp = pred_vel.abs().mean().item()
            gt_disp = gt_vel.abs().mean().item()
            disp_ratio = pred_disp / (gt_disp + 1e-8)

        if self._step_count % 50 == 0:
            print(f"[Step {self._step_count}] loss={loss.item():.4f}, mse={loss_mse.item():.4f}, "
                  f"vel={loss_vel.item():.4f}, acc={loss_acc.item():.4f}, "
                  f"contrastive={loss_contrastive.item():.4f}, disp_ratio={disp_ratio:.4f}")

        self.log_dict({
            "train/loss": loss,
            "train/loss_mse": loss_mse,
            "train/loss_vel": loss_vel,
            "train/loss_acc": loss_acc,
            "train/loss_contrastive": loss_contrastive,
            "train/disp_ratio": disp_ratio,
        }, prog_bar=True)

        self._step_count += 1
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ============================================================
# Inference Wrapper
# ============================================================

class FSWInferenceWrapper(nn.Module):
    """Wrapper for inference with FSW + CLIP model."""
    
    def __init__(self, model, past_bjct, sign_img, fsw_strings):
        super().__init__()
        self.model = model
        self.past = past_bjct
        self.sign_img = sign_img
        self.fsw_strings = fsw_strings
    
    def forward(self, x, t, **kwargs):
        return self.model(x, t, self.past, self.sign_img, self.fsw_strings)


# ============================================================
# Main
# ============================================================

def train_overfit():
    """Overfitting experiment with FSW + CLIP encoder."""
    pl.seed_everything(42)

    # Config
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    out_dir = "logs/fsw_overfit"
    
    NUM_SAMPLES = 4
    MAX_EPOCHS = 500
    DIFFUSION_STEPS = 8
    LEARNING_RATE = 1e-4
    
    os.makedirs(out_dir, exist_ok=True)

    # ============================================================
    # DEBUG: Test FSW parsing first
    # ============================================================
    print("=" * 70)
    print(" DEBUG: Testing FSW parsing")
    print("=" * 70)
    
    test_swu = "𝠃𝤕𝤳񅹁𝣿𝣙񀟣𝣯𝣱񁯡𝣾𝤅񀦁𝣸𝤙"
    try:
        test_fsw = swu2fsw(test_swu)
        print(f"SWU: {test_swu}")
        print(f"FSW: {test_fsw}")
        
        test_sign = fsw_to_sign(test_fsw)
        print(f"Return type: {type(test_sign)}")
        
        # Handle both dict and object
        if isinstance(test_sign, dict):
            symbols_list = test_sign.get('symbols', [])
            print(f"Dict keys: {test_sign.keys()}")
        else:
            symbols_list = test_sign.symbols if hasattr(test_sign, 'symbols') else []
        
        print(f"Parsed {len(symbols_list)} symbols:")
        for sym in symbols_list:
            if isinstance(sym, dict):
                print(f"  {sym.get('symbol')} @ {sym.get('position')}")
            else:
                print(f"  {sym.symbol} @ {sym.position}")
        print("FSW parsing: OK ✓")
    except Exception as e:
        print(f"FSW parsing ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 70)

    print("=" * 70)
    print(" FSW + CLIP OVERFITTING EXPERIMENT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Samples: {NUM_SAMPLES}")
    print(f"  Epochs: {MAX_EPOCHS}")
    print(f"  Diffusion Steps: {DIFFUSION_STEPS}")
    print("=" * 70)

    # Dataset
    print("\nLoading dataset...")
    full_ds = FSWPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        split="train",
        indices=list(range(NUM_SAMPLES)),
    )
    
    print(f"Dataset size: {len(full_ds)}")
    
    # Check FSW parsing
    print("\nChecking FSW parsing:")
    for i in range(min(NUM_SAMPLES, len(full_ds))):
        sample = full_ds[i]
        fsw = sample['conditions']['fsw_string']
        print(f"  [{i}] FSW: {fsw[:50]}..." if len(fsw) > 50 else f"  [{i}] FSW: {fsw}")

    train_loader = DataLoader(
        full_ds, 
        batch_size=NUM_SAMPLES, 
        shuffle=True, 
        collate_fn=fsw_collate_fn,
        num_workers=0,
    )

    # Get dimensions
    sample = full_ds[0]
    num_joints = sample['data'].shape[1]
    num_dims = sample['data'].shape[2]
    future_len = sample['data'].shape[0]
    
    print(f"\nDimensions: J={num_joints}, D={num_dims}, T={future_len}")
    
    # DEBUG: Check data values
    print("\nDEBUG - Data check:")
    print(f"  data shape: {sample['data'].shape}")
    print(f"  data range: [{sample['data'].min():.4f}, {sample['data'].max():.4f}]")
    print(f"  data has NaN: {torch.isnan(sample['data']).any()}")
    print(f"  past shape: {sample['conditions']['input_pose'].shape}")
    print(f"  past has NaN: {torch.isnan(sample['conditions']['input_pose']).any()}")
    
    # Check stats file
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    if os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location="cpu")
        print(f"  stats mean shape: {stats['mean'].shape}")
        print(f"  stats std shape: {stats['std'].shape}")
        print(f"  stats std min: {stats['std'].min():.6f}")
        if stats['std'].min() < 1e-6:
            print(f"  [WARNING] Very small std values detected!")

    # Model
    print("\nInitializing FSW + CLIP model...")
    lit_model = LitFSWDiffusion(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=f"{data_dir}/mean_std_178_with_preprocess.pt",
        lr=LEARNING_RATE,
        diffusion_steps=DIFFUSION_STEPS,
        vel_weight=1.0,
        acc_weight=0.5,
        contrastive_weight=0.0,  # 0 for overfitting (only 4 samples)
        t_past=40,
        t_future=future_len,
        freeze_clip=False,
    )

    total_params = sum(p.numel() for p in lit_model.parameters())
    trainable_params = sum(p.numel() for p in lit_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Train
    print(f"\n{'='*70}")
    print("STARTING TRAINING...")
    print("="*70)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{out_dir}/checkpoints",
        filename="epoch{epoch:03d}",
        save_top_k=1,
        monitor="train/loss",
        mode="min",
        save_last=True,
    )

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

    # Evaluation
    print(f"\n{'='*70}")
    print("EVALUATION")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.eval()

    # Test on each sample
    results = []
    
    for idx in range(len(full_ds)):
        sample = full_ds[idx]
        
        gt_btjc = sample['data'].unsqueeze(0).to(device)
        past_btjc = sample['conditions']['input_pose'].unsqueeze(0).to(device)
        sign_img = sample['conditions']['sign_image'].unsqueeze(0).to(device)
        fsw_string = [sample['conditions']['fsw_string']]
        
        gt_norm = lit_model.normalize(gt_btjc)
        past_norm = lit_model.normalize(past_btjc)
        
        past_bjct = lit_model.btjc_to_bjct(past_norm)
        B, J, C, _ = past_bjct.shape
        T_future = gt_norm.shape[1]
        
        with torch.no_grad():
            # Normal inference
            wrapper = FSWInferenceWrapper(lit_model.model, past_bjct, sign_img, fsw_string)
            pred_bjct = lit_model.diffusion.p_sample_loop(
                model=wrapper,
                shape=(B, J, C, T_future),
                clip_denoised=True,
                model_kwargs={"y": {}},
                progress=False,
            )
            pred_norm = lit_model.bjct_to_btjc(pred_bjct)
            
            # Sign-Only (drop past)
            zeros_past = torch.zeros_like(past_bjct)
            wrapper_signonly = FSWInferenceWrapper(lit_model.model, zeros_past, sign_img, fsw_string)
            pred_signonly_bjct = lit_model.diffusion.p_sample_loop(
                model=wrapper_signonly,
                shape=(B, J, C, T_future),
                clip_denoised=True,
                model_kwargs={"y": {}},
                progress=False,
            )
            pred_signonly_norm = lit_model.bjct_to_btjc(pred_signonly_bjct)
        
        # Metrics
        gt_np = gt_norm[0].cpu().numpy()
        pred_np = pred_norm[0].cpu().numpy()
        pred_so_np = pred_signonly_norm[0].cpu().numpy()
        
        err_normal = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
        err_signonly = np.sqrt(((pred_so_np - gt_np) ** 2).sum(-1))
        
        pck_normal = (err_normal < 0.1).mean() * 100
        pck_signonly = (err_signonly < 0.1).mean() * 100
        
        disp_pred = mean_frame_disp(pred_norm)
        disp_gt = mean_frame_disp(gt_norm)
        ratio = disp_pred / (disp_gt + 1e-8)
        
        results.append({
            'idx': idx,
            'pck_normal': pck_normal,
            'pck_signonly': pck_signonly,
            'disp_ratio': ratio,
            'fsw': fsw_string[0][:30] + "..." if len(fsw_string[0]) > 30 else fsw_string[0],
        })
        
        print(f"  [{idx}] Normal PCK: {pck_normal:.1f}%, Sign-Only PCK: {pck_signonly:.1f}%, Ratio: {ratio:.3f}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    
    avg_pck_normal = np.mean([r['pck_normal'] for r in results])
    avg_pck_signonly = np.mean([r['pck_signonly'] for r in results])
    avg_ratio = np.mean([r['disp_ratio'] for r in results])
    
    print(f"  Average Normal PCK@0.1: {avg_pck_normal:.1f}%")
    print(f"  Average Sign-Only PCK@0.1: {avg_pck_signonly:.1f}%")
    print(f"  Average Disp Ratio: {avg_ratio:.3f}")
    print(f"  Gap (Normal - Sign-Only): {avg_pck_normal - avg_pck_signonly:.1f}%")
    
    # Save pose files for visualization
    print(f"\n{'='*70}")
    print("SAVING POSE FILES")
    print("="*70)
    
    pose_out_dir = f"{out_dir}/poses"
    os.makedirs(pose_out_dir, exist_ok=True)
    
    for idx in range(len(full_ds)):
        sample = full_ds[idx]
        record = full_ds.records[idx]
        
        gt_btjc = sample['data'].unsqueeze(0).to(device)
        past_btjc = sample['conditions']['input_pose'].unsqueeze(0).to(device)
        sign_img = sample['conditions']['sign_image'].unsqueeze(0).to(device)
        fsw_string = [sample['conditions']['fsw_string']]
        
        gt_norm = lit_model.normalize(gt_btjc)
        past_norm = lit_model.normalize(past_btjc)
        
        past_bjct = lit_model.btjc_to_bjct(past_norm)
        B, J, C, _ = past_bjct.shape
        T_future = gt_norm.shape[1]
        
        with torch.no_grad():
            # Normal inference
            wrapper = FSWInferenceWrapper(lit_model.model, past_bjct, sign_img, fsw_string)
            pred_bjct = lit_model.diffusion.p_sample_loop(
                model=wrapper,
                shape=(B, J, C, T_future),
                clip_denoised=True,
                model_kwargs={"y": {}},
                progress=False,
            )
            pred_norm = lit_model.bjct_to_btjc(pred_bjct)
            
            # Sign-Only
            zeros_past = torch.zeros_like(past_bjct)
            wrapper_signonly = FSWInferenceWrapper(lit_model.model, zeros_past, sign_img, fsw_string)
            pred_signonly_bjct = lit_model.diffusion.p_sample_loop(
                model=wrapper_signonly,
                shape=(B, J, C, T_future),
                clip_denoised=True,
                model_kwargs={"y": {}},
                progress=False,
            )
            pred_signonly_norm = lit_model.bjct_to_btjc(pred_signonly_bjct)
        
        # Unnormalize
        gt_unnorm = lit_model.unnormalize(gt_norm)
        pred_unnorm = lit_model.unnormalize(pred_norm)
        pred_signonly_unnorm = lit_model.unnormalize(pred_signonly_norm)
        
        # Load reference pose for header
        ref_path = record['pose']
        if not os.path.isabs(ref_path):
            ref_path = os.path.join(data_dir, ref_path)
        
        with open(ref_path, 'rb') as f:
            ref_pose = Pose.read(f)
        ref_pose = reduce_holistic(ref_pose)
        if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
            ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
        
        # Convert to pose and save
        gt_pose = tensor_to_pose(gt_unnorm, ref_pose.header, ref_pose)
        pred_pose = tensor_to_pose(pred_unnorm, ref_pose.header, ref_pose)
        pred_so_pose = tensor_to_pose(pred_signonly_unnorm, ref_pose.header, ref_pose)
        
        with open(f"{pose_out_dir}/sample{idx}_gt.pose", "wb") as f:
            gt_pose.write(f)
        with open(f"{pose_out_dir}/sample{idx}_pred.pose", "wb") as f:
            pred_pose.write(f)
        with open(f"{pose_out_dir}/sample{idx}_signonly.pose", "wb") as f:
            pred_so_pose.write(f)
        
        print(f"  Saved sample {idx}: gt, pred, signonly")
    
    print(f"\nPose files saved to: {pose_out_dir}/")
    
    print(f"\n{'='*70}")
    if avg_pck_normal > 90 and avg_pck_signonly > 50:
        print("✅ SUCCESS: FSW encoder helps with Sign-Only generation!")
    elif avg_pck_normal > 90:
        print("⚠️ PARTIAL: Overfitting works, but Sign-Only still needs improvement")
    else:
        print("❌ NEEDS MORE WORK: Overfitting not achieved")
    print("="*70)


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    train_overfit()