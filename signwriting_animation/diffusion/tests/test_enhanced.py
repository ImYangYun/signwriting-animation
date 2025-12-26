"""
Past Dropout Training - Enhanced Contrastive Learning

改进：
1. contrastive_weight: 0.5 → 2.0
2. Contrastive-First: 前100 epochs 加重 contrastive，后面正常
3. 测试多个 contrastive weight 配置
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

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
# Config
# ============================================================
NUM_SAMPLES = 64
EPOCHS = 600
PAST_DROP_PROB = 0.3
BATCH_SIZE = 16  # 增大batch size，每batch有120对contrastive pairs

# 测试不同的 contrastive weight
CONTRASTIVE_WEIGHTS = [2.0]

DATA_DIR = "/home/yayun/data/pose_data/"
CSV_PATH = "/home/yayun/data/signwriting-animation/data_fixed.csv"
STATS_PATH = f"{DATA_DIR}/mean_std_178_with_preprocess.pt"


# ============================================================
# CachedDataset
# ============================================================
class CachedDataset(Dataset):
    def __init__(self, base_ds, indices, collate_fn):
        self.samples = []
        self.metadata = []
        
        print(f"Caching {len(indices)} samples...")
        for i, idx in enumerate(indices):
            sample = base_ds[idx]
            batch = collate_fn([sample])
            
            past = batch["conditions"]["input_pose"]
            sign = batch["conditions"]["sign_image"]
            gt = batch["data"]
            
            if hasattr(past, "zero_filled"): past = past.zero_filled()
            if hasattr(past, "tensor"): past = past.tensor
            if hasattr(gt, "zero_filled"): gt = gt.zero_filled()
            if hasattr(gt, "tensor"): gt = gt.tensor
            
            past_t = past[0]
            gt_t = gt[0]
            
            if past_t.dim() == 4 and past_t.shape[1] == 1:
                past_t = past_t.squeeze(1)
            if gt_t.dim() == 4 and gt_t.shape[1] == 1:
                gt_t = gt_t.squeeze(1)
            
            record = base_ds.records[idx]
            
            self.samples.append({
                "past": past_t,
                "sign": sign[0],
                "gt": gt_t,
            })
            self.metadata.append({
                "idx": idx,
                "pose_file": record.get("pose", ""),
            })
        
        print(f"Cached {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "data": s["gt"],
            "conditions": {
                "input_pose": s["past"],
                "sign_image": s["sign"],
            }
        }
    
    def get_metadata(self, idx):
        return self.metadata[idx]


# ============================================================
# Model Components
# ============================================================
class EmbedSignWritingUnfrozen(nn.Module):
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
    def __init__(self, num_keypoints: int, num_dims_per_keypoint: int,
                 embedding_arch: str = 'openai/clip-vit-base-patch32',
                 num_latent_dims: int = 256, num_heads: int = 4, dropout: float = 0.1,
                 t_past: int = 40, t_future: int = 20, freeze_clip: bool = False):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
        input_feats = num_keypoints * num_dims_per_keypoint

        self.past_context_encoder = ContextEncoder(input_feats, num_latent_dims, num_layers=2, num_heads=num_heads, dropout=dropout)
        self.embed_signwriting = EmbedSignWritingUnfrozen(num_latent_dims, embedding_arch, freeze_clip=freeze_clip)
        self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)
        self.time_embed = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)

        self.xt_frame_encoder = nn.Sequential(
            nn.Linear(input_feats, num_latent_dims), nn.GELU(), nn.Linear(num_latent_dims, num_latent_dims),
        )
        self.output_pos_embed = nn.Embedding(512, num_latent_dims)
        self.decoder = nn.Sequential(
            nn.Linear(num_latent_dims * 3, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, input_feats),
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
            outputs.append(self.decoder(dec_input))

        result = torch.stack(outputs, dim=0).permute(1, 0, 2)
        result = result.reshape(B, T_future, J, C).permute(0, 2, 3, 1).contiguous()
        return result


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

def tensor_to_pose(t_btjc, header, ref_pose, scale_to_ref=True):
    if t_btjc.dim() == 4: t = t_btjc[0]
    else: t = t_btjc
    
    t_np = t.detach().cpu().numpy().astype(np.float32)
    arr = t_np[:, None, :, :]
    T = arr.shape[0]
    conf = np.ones((T, 1, arr.shape[2]), dtype=np.float32)
    
    body = NumPyPoseBody(fps=ref_pose.body.fps, data=arr, confidence=conf)
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
        var_input, var_ref = _var(pose_data), _var(ref_arr)
        
        if var_input > 1e-8:
            pose_obj.body.data = pose_obj.body.data * np.sqrt(var_ref / var_input)
        
        pose_data = pose_obj.body.data[:, 0, :, :].reshape(-1, 3)
        input_center = pose_data.mean(axis=0)
        ref_center = ref_arr.reshape(-1, 3).mean(axis=0)
        pose_obj.body.data = pose_obj.body.data + (ref_center - input_center)
    
    return pose_obj


# ============================================================
# Lightning Module - Enhanced Contrastive
# ============================================================
class LitDiffusionEnhancedContrastive(pl.LightningModule):
    def __init__(self, num_keypoints=178, num_dims=3, lr=1e-4,
                 stats_path=STATS_PATH,
                 diffusion_steps=8, vel_weight=1.0, acc_weight=0.5,
                 t_past=40, t_future=20, 
                 contrastive_weight=2.0,  # 增强！
                 past_drop_prob=0.3,
                 contrastive_warmup_epochs=100):  # 前100 epochs 额外加重
        super().__init__()
        self.save_hyperparameters()
        self.diffusion_steps = diffusion_steps
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.contrastive_weight = contrastive_weight
        self.past_drop_prob = past_drop_prob
        self.contrastive_warmup_epochs = contrastive_warmup_epochs
        self._step_count = 0
        self._drop_count = 0

        stats = torch.load(stats_path, map_location="cpu")
        self.register_buffer("mean_pose", stats["mean"].float().view(1, 1, -1, 3))
        self.register_buffer("std_pose", stats["std"].float().view(1, 1, -1, 3))

        self.model = SignWritingToPoseDiffusionUnfrozen(
            num_keypoints=num_keypoints, num_dims_per_keypoint=num_dims,
            t_past=t_past, t_future=t_future, freeze_clip=False,
        )

        betas = cosine_beta_schedule(diffusion_steps).numpy()
        self.diffusion = GaussianDiffusion(
            betas=betas, model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL, loss_type=LossType.MSE, rescale_timesteps=False,
        )
        self.lr = lr

    def normalize(self, x): return (x - self.mean_pose) / (self.std_pose + 1e-6)
    def unnormalize(self, x): return x * self.std_pose + self.mean_pose
    @staticmethod
    def btjc_to_bjct(x): return x.permute(0, 2, 3, 1).contiguous()
    @staticmethod
    def bjct_to_btjc(x): return x.permute(0, 3, 1, 2).contiguous()

    def training_step(self, batch, batch_idx):
        gt_btjc = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(batch["conditions"]["input_pose"])
        sign_img = batch["conditions"]["sign_image"].float()

        gt_norm = self.normalize(gt_btjc)
        past_norm = self.normalize(past_btjc)

        batch_size, device = gt_norm.shape[0], gt_norm.device

        # Past Dropout
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

        loss_mse = F.mse_loss(pred_x0_bjct, gt_bjct)
        pred_vel = pred_x0_bjct[..., 1:] - pred_x0_bjct[..., :-1]
        gt_vel = gt_bjct[..., 1:] - gt_bjct[..., :-1]
        loss_vel = F.mse_loss(pred_vel, gt_vel)
        loss_acc = torch.tensor(0.0, device=device)
        if pred_vel.size(-1) > 1:
            loss_acc = F.mse_loss(pred_vel[..., 1:] - pred_vel[..., :-1], gt_vel[..., 1:] - gt_vel[..., :-1])

        # Enhanced Contrastive Loss
        loss_contrastive = torch.tensor(0.0, device=device)
        if batch_size > 1:
            sign_embs = F.normalize(self.model.embed_signwriting(sign_img), p=2, dim=-1)
            
            # InfoNCE-style loss (更强的对比学习)
            # 目标：让每个样本的 embedding 与其他样本尽量不同
            cos_sim = torch.mm(sign_embs, sign_embs.t())  # [B, B]
            
            # 方法1：简单版 - 最小化非对角线相似度
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            loss_contrastive = cos_sim[mask].mean()
            
            # 方法2：InfoNCE (可选，更强)
            # temperature = 0.1
            # cos_sim = cos_sim / temperature
            # labels = torch.arange(batch_size, device=device)
            # loss_contrastive = F.cross_entropy(cos_sim, labels)

        # Dynamic contrastive weight: 前 warmup epochs 额外加重
        current_epoch = self.current_epoch
        if current_epoch < self.contrastive_warmup_epochs:
            # 前100 epochs: contrastive weight × 2
            effective_contrastive_weight = self.contrastive_weight * 2.0
        else:
            effective_contrastive_weight = self.contrastive_weight

        loss = loss_mse + self.vel_weight * loss_vel + self.acc_weight * loss_acc + effective_contrastive_weight * loss_contrastive

        with torch.no_grad():
            disp_ratio = pred_vel.abs().mean().item() / (gt_vel.abs().mean().item() + 1e-8)
            avg_sim = cos_sim[mask].mean().item() if batch_size > 1 else 0.0

        if self._step_count % 100 == 0:
            drop_str = " [DROPPED]" if past_dropped else ""
            warmup_str = " [WARMUP]" if current_epoch < self.contrastive_warmup_epochs else ""
            print(f"[Epoch {current_epoch} Step {self._step_count}]{drop_str}{warmup_str} "
                  f"loss={loss.item():.4f}, mse={loss_mse.item():.4f}, "
                  f"contrast={loss_contrastive.item():.4f}, avg_sim={avg_sim:.4f}, disp_ratio={disp_ratio:.4f}")

        self.log_dict({
            "train/loss": loss, 
            "train/loss_mse": loss_mse, 
            "train/loss_contrastive": loss_contrastive,
            "train/avg_similarity": avg_sim,
            "train/disp_ratio": disp_ratio
        }, prog_bar=True)
        
        self._step_count += 1
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ============================================================
# Training Function
# ============================================================
def train_single(contrastive_weight, cached_ds, num_joints, num_dims, future_len):
    cw_str = f"cw{contrastive_weight:.1f}".replace(".", "")
    out_dir = f"logs/enhanced_contrastive_{NUM_SAMPLES}sample_p30_{cw_str}_bs{BATCH_SIZE}_ep{EPOCHS}"
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f" ENHANCED CONTRASTIVE TRAINING")
    print(f" Samples: {NUM_SAMPLES}, Batch: {BATCH_SIZE}, Dropout: {PAST_DROP_PROB}")
    print(f" Contrastive pairs per batch: {BATCH_SIZE * (BATCH_SIZE - 1) // 2}")
    print(f" Contrastive weight: {contrastive_weight} (warmup: ×2 for first 100 epochs)")
    print(f" Epochs: {EPOCHS}")
    print("=" * 70)

    train_loader = DataLoader(
        cached_ds, batch_size=BATCH_SIZE, 
        shuffle=True, collate_fn=zero_pad_collator, num_workers=0
    )

    lit_model = LitDiffusionEnhancedContrastive(
        num_keypoints=num_joints, num_dims=num_dims, stats_path=STATS_PATH,
        lr=1e-4, diffusion_steps=8, t_future=future_len, 
        contrastive_weight=contrastive_weight,
        past_drop_prob=PAST_DROP_PROB,
        contrastive_warmup_epochs=100,
    )

    # ============================================================
    # Check CLIP similarity BEFORE training
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model_temp = lit_model.to(device)
    
    print("\n" + "-" * 50)
    print("CLIP Embedding Similarity BEFORE Training:")
    avg_sim_before, max_sim_before, min_sim_before = compute_clip_similarity(lit_model_temp, cached_ds, device)
    print(f"  avg={avg_sim_before:.4f}, max={max_sim_before:.4f}, min={min_sim_before:.4f}")
    print("-" * 50)
    
    lit_model = lit_model.cpu()  # Move back for training

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{out_dir}/checkpoints", save_last=True,
        filename="best-epoch={epoch:03d}", save_top_k=1, monitor="train/loss", mode="min",
    )

    trainer = Trainer(
        max_epochs=EPOCHS, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
        callbacks=[checkpoint_callback], default_root_dir=out_dir,
        log_every_n_steps=1, precision="16-mixed" if torch.cuda.is_available() else 32,
    )
    trainer.fit(lit_model, train_loader)

    # ============================================================
    # Check CLIP similarity AFTER training
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    
    print("\n" + "-" * 50)
    print("CLIP Embedding Similarity AFTER Training:")
    avg_sim_after, max_sim_after, min_sim_after = compute_clip_similarity(lit_model, cached_ds, device)
    print(f"  avg={avg_sim_after:.4f}, max={max_sim_after:.4f}, min={min_sim_after:.4f}")
    print("-" * 50)

    # ============================================================
    # Test: Normal + Sign-Only
    # ============================================================
    print("\n" + "=" * 70)
    print(f"TESTING: contrastive_weight={contrastive_weight}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.eval()

    class W(nn.Module):
        def __init__(s, m, p, sg): super().__init__(); s.m, s.p, s.sg = m, p, sg
        def forward(s, x, t, **kw): return s.m(x, t, s.p, s.sg)

    # Test both Normal and Sign-Only
    results_normal = []
    results_signonly = []
    
    for cache_idx in range(min(10, len(cached_ds))):
        sample = cached_ds.samples[cache_idx]
        meta = cached_ds.get_metadata(cache_idx)
        
        gt = sanitize_btjc(sample["gt"]).to(device)
        past = sanitize_btjc(sample["past"]).to(device)
        sign = sample["sign"].unsqueeze(0).float().to(device)
        
        # Normal inference
        past_norm = lit_model.normalize(past)
        past_bjct = lit_model.btjc_to_bjct(past_norm)
        _, J, C, _ = past_bjct.shape
        
        with torch.no_grad():
            wrapped = W(lit_model.model, past_bjct, sign)
            pred_bjct = lit_model.diffusion.p_sample_loop(
                wrapped, (1, J, C, future_len), 
                clip_denoised=True, model_kwargs={"y": {}}, progress=False
            )
            pred_normal = lit_model.unnormalize(lit_model.bjct_to_btjc(pred_bjct))
        
        diff = (pred_normal - gt).cpu().numpy()[0]
        pck_normal = (np.sqrt((diff**2).sum(-1)) < 0.1).mean() * 100
        
        gt_disp = (gt[:, 1:] - gt[:, :-1]).abs().mean().item()
        pred_disp = (pred_normal[:, 1:] - pred_normal[:, :-1]).abs().mean().item()
        ratio_normal = pred_disp / (gt_disp + 1e-8)
        
        results_normal.append({"idx": meta["idx"], "pck": pck_normal, "ratio": ratio_normal})
        
        # Sign-Only inference
        past_zero = torch.zeros(1, 40, num_joints, num_dims, device=device)
        past_zero_norm = lit_model.normalize(past_zero)
        past_zero_bjct = lit_model.btjc_to_bjct(past_zero_norm)
        
        with torch.no_grad():
            wrapped = W(lit_model.model, past_zero_bjct, sign)
            pred_bjct = lit_model.diffusion.p_sample_loop(
                wrapped, (1, J, C, future_len), 
                clip_denoised=True, model_kwargs={"y": {}}, progress=False
            )
            pred_signonly = lit_model.unnormalize(lit_model.bjct_to_btjc(pred_bjct))
        
        diff = (pred_signonly - gt).cpu().numpy()[0]
        pck_signonly = (np.sqrt((diff**2).sum(-1)) < 0.1).mean() * 100
        
        pred_disp = (pred_signonly[:, 1:] - pred_signonly[:, :-1]).abs().mean().item()
        ratio_signonly = pred_disp / (gt_disp + 1e-8)
        
        results_signonly.append({"idx": meta["idx"], "pck": pck_signonly, "ratio": ratio_signonly})
        
        print(f"idx={meta['idx']}: Normal PCK={pck_normal:.1f}%, Sign-Only PCK={pck_signonly:.1f}%")

    avg_pck_normal = np.mean([r["pck"] for r in results_normal])
    avg_pck_signonly = np.mean([r["pck"] for r in results_signonly])
    avg_ratio_normal = np.mean([r["ratio"] for r in results_normal])
    avg_ratio_signonly = np.mean([r["ratio"] for r in results_signonly])
    
    print("-" * 60)
    print(f"Normal:    AVG PCK={avg_pck_normal:.1f}%, ratio={avg_ratio_normal:.2f}")
    print(f"Sign-Only: AVG PCK={avg_pck_signonly:.1f}%, ratio={avg_ratio_signonly:.2f}")
    print(f"Gap: {avg_pck_normal - avg_pck_signonly:.1f}%")
    
    # Diagnosis
    gap = avg_pck_normal - avg_pck_signonly
    sim_change = avg_sim_before - avg_sim_after
    
    print(f"\nCLIP Similarity Change: {avg_sim_before:.4f} → {avg_sim_after:.4f} (Δ={sim_change:+.4f})")
    
    if gap < 10:
        print("✅ Model learned SignWriting well!")
    elif gap < 20:
        print("⚠️  Partial SignWriting learning")
    else:
        print("❌ Still relies on past frames")
    
    if sim_change > 0.2:
        print("✅ Contrastive learning WORKED - embeddings more diverse")
    elif sim_change > 0.05:
        print("⚠️  Contrastive had small effect")
    else:
        print("❌ Contrastive learning failed - embeddings still similar")

    return {
        "contrastive_weight": contrastive_weight,
        "normal_pck": avg_pck_normal,
        "signonly_pck": avg_pck_signonly,
        "gap": gap,
        "normal_ratio": avg_ratio_normal,
        "signonly_ratio": avg_ratio_signonly,
        "sim_before": avg_sim_before,
        "sim_after": avg_sim_after,
        "sim_change": sim_change,
    }


# ============================================================
# Main
# ============================================================
def compute_clip_similarity(model, cached_ds, device):
    """计算当前CLIP embedding的平均相似度"""
    model.eval()
    all_embs = []
    with torch.no_grad():
        for i in range(len(cached_ds)):
            sign = cached_ds.samples[i]["sign"].unsqueeze(0).float().to(device)
            emb = model.model.embed_signwriting(sign)
            emb = F.normalize(emb, p=2, dim=-1)
            all_embs.append(emb)
    
    all_embs = torch.cat(all_embs, dim=0)
    sim_matrix = torch.mm(all_embs, all_embs.t())
    
    n = len(cached_ds)
    mask = ~torch.eye(n, dtype=torch.bool, device=device)
    avg_sim = sim_matrix[mask].mean().item()
    max_sim = sim_matrix[mask].max().item()
    min_sim = sim_matrix[mask].min().item()
    
    return avg_sim, max_sim, min_sim


def main():
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    pl.seed_everything(42)

    print("=" * 70)
    print(f" ENHANCED CONTRASTIVE LEARNING EXPERIMENT")
    print(f" Samples: {NUM_SAMPLES}, Epochs: {EPOCHS}")
    print(f" Contrastive weights to test: {CONTRASTIVE_WEIGHTS}")
    print("=" * 70)

    # Load dataset once
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

    # Train all configurations
    all_results = []
    for cw in CONTRASTIVE_WEIGHTS:
        result = train_single(cw, cached_ds, num_joints, num_dims, future_len)
        all_results.append(result)

    # Final summary
    print("\n" + "=" * 70)
    print(" FINAL SUMMARY")
    print("=" * 70)
    print(f"{'CW':<6} {'Normal':<10} {'SignOnly':<10} {'Gap':<8} {'Sim Before':<12} {'Sim After':<12} {'Δ Sim':<8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['contrastive_weight']:<6} {r['normal_pck']:<10.1f} {r['signonly_pck']:<10.1f} "
              f"{r['gap']:<8.1f} {r['sim_before']:<12.4f} {r['sim_after']:<12.4f} {r['sim_change']:+<8.4f}")
    
    print("-" * 70)
    print("Baseline (p30 cw=0.5, 32 samples): Normal=52.1%, Sign-Only=30.0%, Gap=22.1%")
    print("Original CLIP similarity: avg=0.88")
    print("=" * 70)


if __name__ == "__main__":
    main()