"""
Clean Overfit Test: FROZEN CLIP, NO contrastive, fixed data

目的：与 Unfrozen CLIP 对比，验证 freeze vs unfreeze 的影响
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset

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
# Configuration
# ============================================================
os.chdir("/home/yayun/data/signwriting-animation-fork")

DATA_DIR = '/home/yayun/data/pose_data/'
CSV_PATH = '/home/yayun/data/signwriting-animation/data_fixed.csv'
STATS_PATH = f"{DATA_DIR}/mean_std_178_with_preprocess.pt"
OUT_DIR = 'logs/overfit_frozen_32s'  # 改了输出目录

NUM_SAMPLES = 32
MAX_EPOCHS = 3000
DIFFUSION_STEPS = 8
LEARNING_RATE = 1e-4

# ============================================================
# Fixed Dataset
# ============================================================

class FixedDataset(Dataset):
    """预先缓存样本，确保训练和评估用完全相同的数据"""
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# Model Components - FROZEN CLIP
# ============================================================

class EmbedSignWritingFrozen(nn.Module):
    """FROZEN CLIP - 不训练 CLIP 参数"""
    def __init__(self, num_latent_dims: int, 
                 embedding_arch: str = 'openai/clip-vit-base-patch32'):
        super().__init__()
        self.model = CLIPModel.from_pretrained(embedding_arch)
        
        # === 关键：冻结 CLIP 所有参数 ===
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
    def __init__(self, num_keypoints: int, num_dims_per_keypoint: int,
                 num_latent_dims: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
        input_feats = num_keypoints * num_dims_per_keypoint

        self.past_context_encoder = ContextEncoder(input_feats, num_latent_dims)
        self.embed_signwriting = EmbedSignWritingFrozen(num_latent_dims)  # 用 Frozen 版本
        self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)
        self.time_embed = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)

        self.xt_frame_encoder = nn.Sequential(
            nn.Linear(input_feats, num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )
        self.output_pos_embed = nn.Embedding(512, num_latent_dims)

        self.decoder = nn.Sequential(
            nn.Linear(num_latent_dims * 3, 512),
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

        result = torch.stack(outputs, dim=0).permute(1, 0, 2)
        result = result.reshape(B, T_future, J, C).permute(0, 2, 3, 1).contiguous()
        return result


# ============================================================
# Utilities
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


def tensor_to_pose(t_btjc, header, ref_pose, scale_to_ref=True):
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
# Main
# ============================================================

def main():
    print("=" * 70)
    print("CLEAN OVERFIT TEST: FROZEN CLIP, NO contrastive, FIXED data")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/poses", exist_ok=True)
    
    # === Load Dataset ===
    print("\nLoading dataset...")
    full_ds = DynamicPosePredictionDataset(
        data_dir=DATA_DIR, csv_path=CSV_PATH,
        num_past_frames=40, num_future_frames=20,
        with_metadata=True, split='train',
    )
    print(f"Full dataset: {len(full_ds)} samples")
    
    # === 关键：预先缓存固定样本 ===
    print("\nCaching fixed samples...")
    seen_poses, train_indices = set(), []
    for idx in range(len(full_ds)):
        if len(train_indices) >= NUM_SAMPLES:
            break
        pose_path = full_ds.records[idx].get("pose", "")
        if pose_path not in seen_poses:
            seen_poses.add(pose_path)
            train_indices.append(idx)
    
    # 缓存样本
    cached_samples = []
    for idx in train_indices:
        sample = full_ds[idx]
        cached_samples.append(sample)
    
    print(f"Cached {len(cached_samples)} fixed samples")
    print(f"Indices: {train_indices}")
    
    # 保存 indices
    with open(f"{OUT_DIR}/train_indices.txt", "w") as f:
        for idx in train_indices:
            f.write(f"{idx}\n")
    
    # 使用固定数据集
    fixed_ds = FixedDataset(cached_samples)
    train_loader = DataLoader(fixed_ds, batch_size=NUM_SAMPLES, shuffle=True, collate_fn=zero_pad_collator)
    
    # Get dimensions
    sample = cached_samples[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    num_joints = sample.shape[-2]
    num_dims = sample.shape[-1]
    future_len = sample.shape[0]
    print(f"Dimensions: J={num_joints}, D={num_dims}, T_future={future_len}")
    
    # === Load Stats ===
    stats = torch.load(STATS_PATH, map_location="cpu")
    mean_pose = stats["mean"].float().view(1, 1, -1, 3).to(device)
    std_pose = stats["std"].float().view(1, 1, -1, 3).to(device)
    
    # === Create Model ===
    print("\nCreating model (FROZEN CLIP)...")
    model = SignWritingToPoseDiffusion(
        num_keypoints=num_joints,
        num_dims_per_keypoint=num_dims,
    ).to(device)
    
    betas = cosine_beta_schedule(DIFFUSION_STEPS).numpy()
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
    print(f"  (CLIP is FROZEN, so trainable << total)")
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Normalize helper
    def normalize(x):
        return (x - mean_pose) / (std_pose + 1e-6)
    
    def unnormalize(x):
        return x * std_pose + mean_pose
    
    # === Training ===
    print(f"\n{'='*70}")
    print("TRAINING (FROZEN CLIP, no contrastive loss)...")
    print("=" * 70)
    
    best_loss = float('inf')
    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            cond = batch["conditions"]
            gt_btjc = sanitize_btjc(batch["data"]).to(device)
            past_btjc = sanitize_btjc(cond["input_pose"]).to(device)
            sign_img = cond["sign_image"].float().to(device)
            
            gt_norm = normalize(gt_btjc)
            past_norm = normalize(past_btjc)
            
            B = gt_norm.shape[0]
            gt_bjct = gt_norm.permute(0, 2, 3, 1).contiguous()
            past_bjct = past_norm.permute(0, 2, 3, 1).contiguous()
            
            timestep = torch.randint(0, DIFFUSION_STEPS, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(gt_bjct)
            x_noisy = diffusion.q_sample(gt_bjct, timestep, noise=noise)
            
            pred_x0 = model(x_noisy, timestep, past_bjct, sign_img)
            
            # Simple MSE + velocity loss
            loss_mse = F.mse_loss(pred_x0, gt_bjct)
            
            pred_vel = pred_x0[..., 1:] - pred_x0[..., :-1]
            gt_vel = gt_bjct[..., 1:] - gt_bjct[..., :-1]
            loss_vel = F.mse_loss(pred_vel, gt_vel)
            
            loss = loss_mse + loss_vel
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                disp_ratio = pred_vel.abs().mean().item() / (gt_vel.abs().mean().item() + 1e-8)
        
        if epoch % 500 == 0 or epoch == MAX_EPOCHS - 1:
            print(f"[Epoch {epoch:5d}] loss={loss.item():.6f}, mse={loss_mse.item():.6f}, vel={loss_vel.item():.6f}, ratio={disp_ratio:.3f}")
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), f"{OUT_DIR}/best_model.pt")
    
    torch.save(model.state_dict(), f"{OUT_DIR}/final_model.pt")
    print(f"\nModels saved to {OUT_DIR}/")
    
    # === Evaluation ===
    print(f"\n{'='*70}")
    print("EVALUATION (on same fixed samples)...")
    print("=" * 70)
    
    model.eval()
    results = []
    
    for i, sample in enumerate(cached_samples):
        batch = zero_pad_collator([sample])
        past = sanitize_btjc(batch['conditions']['input_pose'][:1]).to(device)
        sign = batch['conditions']['sign_image'][:1].float().to(device)
        gt = sanitize_btjc(batch['data'][:1]).to(device)
        
        with torch.no_grad():
            past_norm = normalize(past)
            past_bjct = past_norm.permute(0, 2, 3, 1).contiguous()
            B, J, C, _ = past_bjct.shape
            
            class Wrapper(nn.Module):
                def __init__(self, m, p, s):
                    super().__init__()
                    self.m, self.p, self.s = m, p, s
                def forward(self, x, t, **kw):
                    return self.m(x, t, self.p, self.s)
            
            wrapped = Wrapper(model, past_bjct, sign)
            pred_bjct = diffusion.p_sample_loop(
                wrapped, (B, J, C, future_len), clip_denoised=True,
                model_kwargs={'y': {}}, progress=False
            )
            pred_norm = pred_bjct.permute(0, 3, 1, 2).contiguous()
            pred = unnormalize(pred_norm)
        
        # Metrics
        gt_disp = (gt[:, 1:] - gt[:, :-1]).abs().mean().item()
        pred_disp = (pred[:, 1:] - pred[:, :-1]).abs().mean().item()
        ratio = pred_disp / (gt_disp + 1e-8)
        
        diff = (pred - gt).cpu().numpy()[0]
        per_joint_err = np.sqrt((diff ** 2).sum(-1))
        pck = (per_joint_err < 0.1).mean() * 100
        
        results.append({'idx': train_indices[i], 'ratio': ratio, 'pck': pck})
        
        if i < 10:
            print(f"  [{i}] idx={train_indices[i]}: ratio={ratio:.2f}, PCK={pck:.1f}%")
        
        # Save pose files for first 5 samples
        if i < 5:
            ref_path = full_ds.records[train_indices[i]]['pose']
            if not ref_path.startswith('/'):
                ref_path = DATA_DIR + ref_path
            with open(ref_path, 'rb') as f:
                ref_pose = Pose.read(f)
            ref_pose = reduce_holistic(ref_pose)
            if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
                ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
            
            gt_pose = tensor_to_pose(gt, ref_pose.header, ref_pose)
            pred_pose = tensor_to_pose(pred, ref_pose.header, ref_pose)
            
            with open(f'{OUT_DIR}/poses/sample{i}_idx{train_indices[i]}_gt.pose', 'wb') as f:
                gt_pose.write(f)
            with open(f'{OUT_DIR}/poses/sample{i}_idx{train_indices[i]}_pred.pose', 'wb') as f:
                pred_pose.write(f)
    
    # Summary
    avg_ratio = np.mean([r['ratio'] for r in results])
    avg_pck = np.mean([r['pck'] for r in results])
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  Avg Ratio: {avg_ratio:.3f} (ideal = 1.0)")
    print(f"  Avg PCK@0.1: {avg_pck:.1f}%")
    print(f"\nPose files saved to: {OUT_DIR}/poses/")
    print("=" * 70)
    
    # Save summary
    with open(f"{OUT_DIR}/results.txt", "w") as f:
        f.write(f"Configuration: FROZEN CLIP, NO contrastive, 32 samples\n")
        f.write(f"Avg Ratio: {avg_ratio:.3f}\n")
        f.write(f"Avg PCK@0.1: {avg_pck:.1f}%\n")
        f.write(f"\nPer-sample results:\n")
        for r in results:
            f.write(f"  idx={r['idx']}: ratio={r['ratio']:.2f}, PCK={r['pck']:.1f}%\n")


if __name__ == "__main__":
    main()