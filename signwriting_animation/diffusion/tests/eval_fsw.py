"""
Evaluate FSW Full Model on 32 Fixed Samples (v2)

Changes from v1:
- Save ALL samples with Sign-Only PCK > threshold
- Check left/right hand quality (detect collapsed joints)
- Print hand quality report for each sample

Usage:
    python eval_fsw_full_32sample_v2.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset
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
from signwriting.formats.swu_to_fsw import swu2fsw
from signwriting.formats.fsw_to_sign import fsw_to_sign


# ============================================================
# Config
# ============================================================
NUM_SAMPLES = 32
DATA_DIR = "/home/yayun/data/pose_data/"
CSV_PATH = "/home/yayun/data/signwriting-animation/data_fixed.csv"
STATS_PATH = f"{DATA_DIR}/mean_std_178_with_preprocess.pt"
MODEL_PATH = "logs/fsw_full_p30/final_model.pt"
OUT_DIR = "logs/fsw_full_32sample_eval_v2"

# ---- NEW: Save config ----
SAVE_TOP_N = 5              # Save top N pose sets (prioritize both_ok + high PCK)
HAND_STD_THRESHOLD = 0.02    # Left/right hand std below this = collapsed


def tensor_to_pose(t_btjc, header, ref_pose, scale_to_ref=True):
    """Convert tensor to Pose object."""
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc
    t_np = t.detach().cpu().numpy().astype(np.float32)
    arr = t_np[:, None, :, :]  # [T, 1, J, C]
    T = arr.shape[0]
    conf = np.ones((T, 1, arr.shape[2]), dtype=np.float32)
    body = NumPyPoseBody(fps=ref_pose.body.fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)
    
    # Fix hand positions (must be before scaling!)
    unshift_hands(pose_obj)
    
    if scale_to_ref:
        T_pred = t_np.shape[0]
        T_ref_total = ref_pose.body.data.shape[0]
        future_start = max(0, T_ref_total - T_pred)
        ref_arr = np.asarray(
            ref_pose.body.data[future_start:future_start+T_pred, 0], 
            dtype=np.float32
        )
        
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
# NEW: Hand quality check
# ============================================================
def check_hand_quality(gt_np, threshold=HAND_STD_THRESHOLD):
    """
    Check if left/right hand joints are collapsed.
    gt_np: (T, J, C) numpy array
    Returns: dict with left_ok, right_ok, left_std, right_std
    """
    left_hand = gt_np[:, 136:157, :2]   # (T, 21, 2)
    right_hand = gt_np[:, 157:178, :2]  # (T, 21, 2)
    
    # Per-frame std of joint positions, averaged across frames
    left_std = left_hand.std(axis=1).mean()
    right_std = right_hand.std(axis=1).mean()
    
    return {
        "left_ok": left_std > threshold,
        "right_ok": right_std > threshold,
        "left_std": float(left_std),
        "right_std": float(right_std),
        "both_ok": left_std > threshold and right_std > threshold,
    }


# ============================================================
# Enhanced FSW Encoder (same as training)
# ============================================================
class EnhancedFSWEncoder(nn.Module):
    def __init__(self, num_latent_dims: int = 256, max_symbols: int = 20):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.max_symbols = max_symbols
        
        self.category_embed = nn.Embedding(64, num_latent_dims // 4)
        self.shape_embed = nn.Embedding(256, num_latent_dims // 4)
        self.variation_embed = nn.Embedding(4096, num_latent_dims // 2)
        
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, num_latent_dims), nn.LayerNorm(num_latent_dims),
            nn.GELU(), nn.Linear(num_latent_dims, num_latent_dims),
        )
        
        self.symbol_fusion = nn.Sequential(
            nn.Linear(num_latent_dims * 2, num_latent_dims),
            nn.LayerNorm(num_latent_dims), nn.GELU(),
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_latent_dims, nhead=4, dim_feedforward=num_latent_dims * 4,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn_query = nn.Parameter(torch.randn(1, 1, num_latent_dims))
        self.output_proj = nn.Sequential(
            nn.Linear(num_latent_dims, num_latent_dims), nn.LayerNorm(num_latent_dims),
        )
        
    def parse_fsw(self, fsw_string: str):
        if not fsw_string or not fsw_string.strip():
            return [(0, 0, 0, 0.0, 0.0)]
        try:
            sign = fsw_to_sign(fsw_string)
            symbols_list = sign.get('symbols', []) if isinstance(sign, dict) else getattr(sign, 'symbols', [])
            if not symbols_list:
                return [(0, 0, 0, 0.0, 0.0)]
            
            result = []
            for sym in symbols_list:
                if isinstance(sym, dict):
                    symbol_str = sym.get('symbol', 'S10000')
                    position = sym.get('position', (500, 500))
                else:
                    symbol_str, position = sym.symbol, sym.position
                
                if symbol_str.startswith('S'):
                    symbol_str = symbol_str[1:]
                try:
                    full_id = int(symbol_str, 16)
                    category = (full_id >> 16) & 0xF
                    shape = (full_id >> 12) & 0xFF
                    variation = full_id & 0xFFF
                except:
                    category, shape, variation = 0, 0, 0
                
                x = max(-2.0, min(2.0, (position[0] - 500) / 250.0))
                y = max(-2.0, min(2.0, (position[1] - 500) / 250.0))
                result.append((category % 64, shape % 256, variation % 4096, x, y))
            return result if result else [(0, 0, 0, 0.0, 0.0)]
        except:
            return [(0, 0, 0, 0.0, 0.0)]
    
    def forward(self, fsw_batch: list):
        device = self.category_embed.weight.device
        batch_size = len(fsw_batch)
        
        batch_embeddings, batch_masks = [], []
        for fsw in fsw_batch:
            symbols = self.parse_fsw(fsw)[:self.max_symbols]
            num_symbols = len(symbols)
            while len(symbols) < self.max_symbols:
                symbols.append((0, 0, 0, 0.0, 0.0))
            
            cats = torch.tensor([s[0] for s in symbols], device=device)
            shapes = torch.tensor([s[1] for s in symbols], device=device)
            vars_ = torch.tensor([s[2] for s in symbols], device=device)
            positions = torch.tensor([[s[3], s[4]] for s in symbols], device=device, dtype=torch.float)
            
            symbol_emb = torch.cat([self.category_embed(cats), self.shape_embed(shapes), self.variation_embed(vars_)], dim=-1)
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
        return self.output_proj(pooled)


# ============================================================
# Model Components (same as training)
# ============================================================
class EmbedSignWritingUnfrozen(nn.Module):
    def __init__(self, num_latent_dims: int, embedding_arch: str = 'openai/clip-vit-base-patch32', freeze_clip: bool = False):
        super().__init__()
        self.model = CLIPModel.from_pretrained(embedding_arch)
        if freeze_clip:
            for param in self.model.parameters():
                param.requires_grad = False
        self.proj = None
        if self.model.visual_projection.out_features != num_latent_dims:
            self.proj = nn.Linear(self.model.visual_projection.out_features, num_latent_dims)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        embeddings_batch = self.model.get_image_features(pixel_values=image_batch)
        return self.proj(embeddings_batch) if self.proj else embeddings_batch


class ContextEncoder(nn.Module):
    def __init__(self, input_feats: int, latent_dim: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pose_encoder = nn.Linear(input_feats, latent_dim)
        self.pos_encoding = PositionalEncoding(latent_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=latent_dim * 4, dropout=dropout, activation="gelu", batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, T, J, C = x.shape
            x = x.reshape(B, T, J * C)
        x_emb = self.pose_encoder(x).permute(1, 0, 2)
        x_emb = self.pos_encoding(x_emb)
        x_enc = self.encoder(x_emb).permute(1, 0, 2)
        return x_enc.mean(dim=1)


class SignWritingToPoseDiffusionEnhanced(nn.Module):
    def __init__(self, num_keypoints: int, num_dims_per_keypoint: int, embedding_arch: str = 'openai/clip-vit-base-patch32',
                 num_latent_dims: int = 256, num_heads: int = 4, dropout: float = 0.1, t_past: int = 40, t_future: int = 20, freeze_clip: bool = False):
        super().__init__()
        self.num_keypoints, self.num_dims_per_keypoint = num_keypoints, num_dims_per_keypoint
        self.t_past, self.t_future = t_past, t_future
        input_feats = num_keypoints * num_dims_per_keypoint

        self.past_context_encoder = ContextEncoder(input_feats, num_latent_dims, num_layers=2, num_heads=num_heads, dropout=dropout)
        self.embed_signwriting = EmbedSignWritingUnfrozen(num_latent_dims, embedding_arch, freeze_clip=freeze_clip)
        self.fsw_encoder = EnhancedFSWEncoder(num_latent_dims, max_symbols=20)
        
        self.sign_fusion = nn.Sequential(
            nn.Linear(num_latent_dims * 2, num_latent_dims * 2), nn.LayerNorm(num_latent_dims * 2), nn.GELU(),
            nn.Linear(num_latent_dims * 2, num_latent_dims), nn.LayerNorm(num_latent_dims),
        )
        
        self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)
        self.time_embed = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)
        self.xt_frame_encoder = nn.Sequential(nn.Linear(input_feats, num_latent_dims), nn.LayerNorm(num_latent_dims), nn.GELU(), nn.Linear(num_latent_dims, num_latent_dims))
        self.output_pos_embed = nn.Embedding(512, num_latent_dims)

        self.decoder = nn.Sequential(
            nn.Linear(num_latent_dims * 3, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(768, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, input_feats),
        )

    def forward(self, x, timesteps, past_motion, signwriting_im_batch, fsw_strings=None):
        B, J, C, T_future = x.shape
        device = x.device

        past_btjc = past_motion.permute(0, 3, 1, 2).contiguous() if past_motion.dim() == 4 and past_motion.shape[1] == J else past_motion
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
            pos_emb = self.output_pos_embed(torch.tensor([t], device=device)).expand(B, -1)
            outputs.append(self.decoder(torch.cat([context, xt_emb, pos_emb], dim=-1)))

        result = torch.stack(outputs, dim=0).permute(1, 0, 2)
        return result.reshape(B, T_future, J, C).permute(0, 2, 3, 1).contiguous()


# ============================================================
# Utilities
# ============================================================
def sanitize_btjc(x):
    if hasattr(x, "zero_filled"): x = x.zero_filled()
    if hasattr(x, "tensor"): x = x.tensor
    if x.dim() == 5: x = x[:, :, 0]
    if x.dim() == 3: x = x.unsqueeze(0)
    if x.dim() != 4: raise ValueError(f"Expected [B,T,J,C], got {tuple(x.shape)}")
    if x.shape[-1] != 3 and x.shape[-2] == 3: x = x.permute(0, 1, 3, 2)
    return x.contiguous().float()

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def get_fsw_from_record(record):
    swu_text = record.get('text', '')
    if not swu_text: return ""
    swu_first = swu_text.split()[0] if ' ' in swu_text else swu_text
    try: return swu2fsw(swu_first)
    except: return ""


# ============================================================
# CachedDataset (same as training)
# ============================================================
class CachedDataset(Dataset):
    def __init__(self, base_ds, indices, collate_fn):
        self.samples, self.metadata = [], []
        print(f"Caching {len(indices)} samples...")
        for idx in indices:
            sample = base_ds[idx]
            batch = collate_fn([sample])
            
            past = batch["conditions"]["input_pose"]
            sign = batch["conditions"]["sign_image"]
            gt = batch["data"]
            
            if hasattr(past, "zero_filled"): past = past.zero_filled()
            if hasattr(past, "tensor"): past = past.tensor
            if hasattr(gt, "zero_filled"): gt = gt.zero_filled()
            if hasattr(gt, "tensor"): gt = gt.tensor
            
            past_t, gt_t = past[0], gt[0]
            if past_t.dim() == 4 and past_t.shape[1] == 1: past_t = past_t.squeeze(1)
            if gt_t.dim() == 4 and gt_t.shape[1] == 1: gt_t = gt_t.squeeze(1)
            
            record = base_ds.records[idx]
            self.samples.append({"past": past_t, "sign": sign[0], "gt": gt_t, "fsw": get_fsw_from_record(record)})
            self.metadata.append({"idx": idx, "pose_file": record.get("pose", "")})
        print(f"Cached {len(self.samples)} samples")
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
    def get_metadata(self, idx): return self.metadata[idx]


# ============================================================
# Main Evaluation
# ============================================================
def evaluate():
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print(f" EVALUATING FSW FULL MODEL ON {NUM_SAMPLES} FIXED SAMPLES (v2)")
    print(f" Save top {SAVE_TOP_N} candidates (prioritizing both hands OK)")
    print(f" Hand collapse threshold: std < {HAND_STD_THRESHOLD}")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    base_ds = DynamicPosePredictionDataset(
        data_dir=DATA_DIR, csv_path=CSV_PATH,
        num_past_frames=40, num_future_frames=20, with_metadata=True, split="train",
    )
    
    seen_poses, selected_indices = set(), []
    for idx in range(len(base_ds)):
        if len(selected_indices) >= NUM_SAMPLES: break
        pose = base_ds.records[idx].get("pose", "")
        if pose not in seen_poses:
            seen_poses.add(pose)
            selected_indices.append(idx)
    
    print(f"Selected {len(selected_indices)} unique samples")
    cached_ds = CachedDataset(base_ds, selected_indices, zero_pad_collator)
    
    sample_data = cached_ds.samples[0]["gt"]
    future_len, num_joints, num_dims = sample_data.shape
    print(f"Data shape: T={future_len}, J={num_joints}, C={num_dims}")
    
    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    
    stats = torch.load(STATS_PATH, map_location="cpu")
    mean_pose = stats["mean"].float().view(1, 1, -1, 3).to(device)
    std_pose = stats["std"].float().view(1, 1, -1, 3).to(device)
    
    model = SignWritingToPoseDiffusionEnhanced(
        num_keypoints=num_joints, num_dims_per_keypoint=num_dims,
        t_past=40, t_future=future_len, freeze_clip=False,
    ).to(device)
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
    new_state_dict = {(k[6:] if k.startswith('model.') else k): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    # Setup diffusion
    betas = cosine_beta_schedule(8).numpy()
    diffusion = GaussianDiffusion(
        betas=betas, model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL, loss_type=LossType.MSE, rescale_timesteps=False,
    )
    
    normalize = lambda x: (x - mean_pose) / (std_pose + 1e-6)
    unnormalize = lambda x: x * std_pose + mean_pose
    btjc_to_bjct = lambda x: x.permute(0, 2, 3, 1).contiguous()
    bjct_to_btjc = lambda x: x.permute(0, 3, 1, 2).contiguous()
    
    # Evaluate
    print("\n" + "=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)
    
    results_normal, results_signonly = [], []
    
    # ---- NEW: Collect candidates for saving ----
    save_candidates = []
    
    class ModelWrapper(nn.Module):
        def __init__(self, model, past, sign, fsw):
            super().__init__()
            self.model, self.past, self.sign, self.fsw = model, past, sign, fsw
        def forward(self, x, t, **kw):
            return self.model(x, t, self.past, self.sign, self.fsw)
    
    with torch.no_grad():
        for i in range(len(cached_ds)):
            sample = cached_ds.samples[i]
            meta = cached_ds.get_metadata(i)
            
            gt = sanitize_btjc(sample["gt"]).to(device)
            past = sanitize_btjc(sample["past"]).to(device)
            sign = sample["sign"].unsqueeze(0).float().to(device)
            fsw = [sample["fsw"]]
            
            past_norm = normalize(past)
            past_bjct = btjc_to_bjct(past_norm)
            B, J, C, T = 1, num_joints, num_dims, future_len
            
            # Normal inference
            wrapped_normal = ModelWrapper(model, past_bjct, sign, fsw)
            pred_bjct_normal = diffusion.p_sample_loop(wrapped_normal, (1, J, C, T), clip_denoised=True, model_kwargs={"y": {}}, progress=False)
            pred_normal = unnormalize(bjct_to_btjc(pred_bjct_normal))
            
            # Sign-Only inference
            zeros_past = torch.zeros_like(past_bjct)
            wrapped_signonly = ModelWrapper(model, zeros_past, sign, fsw)
            pred_bjct_signonly = diffusion.p_sample_loop(wrapped_signonly, (1, J, C, T), clip_denoised=True, model_kwargs={"y": {}}, progress=False)
            pred_signonly = unnormalize(bjct_to_btjc(pred_bjct_signonly))
            
            # Compute metrics
            gt_np = gt.cpu().numpy()[0]
            pred_normal_np = pred_normal.cpu().numpy()[0]
            pred_signonly_np = pred_signonly.cpu().numpy()[0]
            
            err_normal = np.sqrt(((pred_normal_np - gt_np) ** 2).sum(-1))
            err_signonly = np.sqrt(((pred_signonly_np - gt_np) ** 2).sum(-1))
            
            pck_normal = (err_normal < 0.1).mean() * 100
            pck_signonly = (err_signonly < 0.1).mean() * 100
            
            gt_disp = np.abs(np.diff(gt_np, axis=0)).mean()
            pred_disp = np.abs(np.diff(pred_normal_np, axis=0)).mean()
            ratio = pred_disp / (gt_disp + 1e-8)
            
            # ---- NEW: Hand quality check ----
            hand_info = check_hand_quality(gt_np)
            hand_tag = ""
            if hand_info["both_ok"]:
                hand_tag = "✅ BOTH_OK"
            elif hand_info["left_ok"]:
                hand_tag = "⚠️  RIGHT_COLLAPSED"
            elif hand_info["right_ok"]:
                hand_tag = "⚠️  LEFT_COLLAPSED"
            else:
                hand_tag = "❌ BOTH_COLLAPSED"
            
            results_normal.append({"idx": meta["idx"], "pck": pck_normal, "ratio": ratio})
            results_signonly.append({"idx": meta["idx"], "pck": pck_signonly})
            
            print(f"[{i+1}/{len(cached_ds)}] idx={meta['idx']:>5d}: "
                  f"Normal={pck_normal:5.1f}%, Sign-Only={pck_signonly:5.1f}%, Ratio={ratio:.2f}  "
                  f"LH_std={hand_info['left_std']:.2f} RH_std={hand_info['right_std']:.2f}  {hand_tag}")
            
            # ---- NEW: Collect ALL candidates (will sort & trim later) ----
            save_candidates.append({
                "sample_i": i,
                "idx": meta["idx"],
                "pose_file": meta["pose_file"],
                "gt": gt.cpu(),
                "pred_normal": pred_normal.cpu(),
                "pred_signonly": pred_signonly.cpu(),
                "pck_normal": pck_normal,
                "pck_signonly": pck_signonly,
                "ratio": ratio,
                "hand_info": hand_info,
            })
    
    # Summary
    avg_normal = np.mean([r["pck"] for r in results_normal])
    avg_signonly = np.mean([r["pck"] for r in results_signonly])
    avg_ratio = np.mean([r["ratio"] for r in results_normal])
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Average Normal PCK@0.1:    {avg_normal:.1f}%")
    print(f"  Average Sign-Only PCK@0.1: {avg_signonly:.1f}%")
    print(f"  Gap (Normal - Sign-Only):  {avg_normal - avg_signonly:.1f}%")
    print(f"  Average Disp Ratio:        {avg_ratio:.2f}")
    print("=" * 70)
    
    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON: OVERFITTING vs FULL DATA GENERALIZATION")
    print("=" * 70)
    print(f"{'Model':<35} {'Normal PCK':<12} {'Sign-Only':<12} {'Gap':<10}")
    print("-" * 70)
    print(f"{'32s Overfit (FSW+p0.5, trained)':<35} {'76.2%':<12} {'59.5%':<12} {'16.7%':<10}")
    print(f"{'FSW Full (p0.3, on same 32s)':<35} {f'{avg_normal:.1f}%':<12} {f'{avg_signonly:.1f}%':<12} {f'{avg_normal-avg_signonly:.1f}%':<10}")
    print("=" * 70)
    
    # ============================================================
    # NEW: Save candidates report & pose files
    # ============================================================
    print("\n" + "=" * 70)
    print(f"SAVE CANDIDATES (top {SAVE_TOP_N}, sorted by both_ok + Sign-Only PCK)")
    print("=" * 70)
    
    # Sort: both_ok first, then by sign-only PCK descending
    save_candidates.sort(key=lambda x: (-x["hand_info"]["both_ok"], -x["pck_signonly"]))
    
    both_ok_count = sum(1 for c in save_candidates if c["hand_info"]["both_ok"])
    print(f"  Total samples:    {len(save_candidates)}")
    print(f"  Both hands OK:    {both_ok_count}")
    print(f"  Hand issues:      {len(save_candidates) - both_ok_count}")
    print(f"  Will save top:    {SAVE_TOP_N}")
    
    print(f"\n{'Rank':<5} {'idx':<6} {'Normal':>7} {'SignOnly':>9} {'Ratio':>6} {'LH_std':>7} {'RH_std':>7} {'Hands':<18} {'Saved':<6}")
    print("-" * 80)
    
    saved_count = 0
    for rank, cand in enumerate(save_candidates):
        idx = cand["idx"]
        hi = cand["hand_info"]
        
        hand_tag = "✅ BOTH_OK" if hi["both_ok"] else ("⚠️ LH_BAD" if not hi["left_ok"] else "⚠️ RH_BAD")
        
        # Only save top SAVE_TOP_N
        saved = False
        if saved_count < SAVE_TOP_N:
            pose_file = cand["pose_file"]
            if not pose_file.startswith("/"):
                pose_file = DATA_DIR + pose_file
            
            try:
                with open(pose_file, "rb") as f:
                    ref_pose = Pose.read(f)
                ref_pose = reduce_holistic(ref_pose)
                if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
                    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
                
                for label, tensor_data in [("gt", cand["gt"]), ("pred_normal", cand["pred_normal"]), ("pred_signonly", cand["pred_signonly"])]:
                    pose_obj = tensor_to_pose(tensor_data, ref_pose.header, ref_pose, scale_to_ref=True)
                    out_path = f"{OUT_DIR}/idx{idx}_{label}.pose"
                    with open(out_path, "wb") as f:
                        pose_obj.write(f)
                
                saved = True
                saved_count += 1
            except Exception as e:
                print(f"  ❌ Failed to save idx={idx}: {e}")
        
        print(f"{rank+1:<5} {idx:<6} {cand['pck_normal']:6.1f}% {cand['pck_signonly']:8.1f}% {cand['ratio']:5.2f} {hi['left_std']:7.2f} {hi['right_std']:7.2f} {hand_tag:<18} {'✅' if saved else '❌'}")
    
    print(f"\n✅ Saved {saved_count}/{SAVE_TOP_N} sets of pose files to: {OUT_DIR}/")
    print(f"   Each set contains: idx{{N}}_gt.pose, idx{{N}}_pred_normal.pose, idx{{N}}_pred_signonly.pose")
    
    # ---- NEW: Recommend best sample for demo ----
    best_demo = None
    for cand in save_candidates:
        if cand["hand_info"]["both_ok"]:
            best_demo = cand
            break
    
    if best_demo:
        print(f"\n🏆 RECOMMENDED FOR DEMO: idx={best_demo['idx']}")
        print(f"   Normal={best_demo['pck_normal']:.1f}%, Sign-Only={best_demo['pck_signonly']:.1f}%")
        print(f"   Both hands OK (LH={best_demo['hand_info']['left_std']:.2f}, RH={best_demo['hand_info']['right_std']:.2f})")
    else:
        print(f"\n⚠️  No candidate with both hands OK found above threshold.")
        print(f"   Consider using overfitting experiment samples instead.")


if __name__ == "__main__":
    evaluate()