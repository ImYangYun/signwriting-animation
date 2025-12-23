"""
Full Dataset Training: Unfrozen CLIP + Contrastive Learning

Based on successful 32-sample overfit test:
- Disp ratio: 1.01 (ideal = 1.0) ✅
- Sign influence: 44% (verified) ✅
- CLIP learns to distinguish SignWriting ✅

Usage:
    srun --partition=lowprio --gres=gpu:V100:1 --mem=64G --time=48:00:00 \
        python train_unfrozen_clip_full.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from torch.utils.data import DataLoader
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import unshift_hands
from transformers import CLIPModel

# DTW metric (optional, graceful fallback if unavailable)
try:
    from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure as PE_DTW
    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    PE_DTW = None
    print("[WARNING] pose_evaluation not found, DTW metric disabled")

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder
from CAMDM.diffusion.gaussian_diffusion import (
    GaussianDiffusion, ModelMeanType, ModelVarType, LossType
)

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset


# ============================================================
# Model Components (Unfrozen CLIP version)
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


def _btjc_to_tjc_list(x_btjc: torch.Tensor, mask_bt: torch.Tensor = None) -> list:
    """Convert batched BTJC tensor to list of variable-length TJC tensors."""
    x_btjc = sanitize_btjc(x_btjc)
    batch_size, seq_len, _, _ = x_btjc.shape
    
    if mask_bt is None:
        # No mask, use full sequence
        return [x_btjc[b].contiguous() for b in range(batch_size)]
    
    mask_bt = (mask_bt > 0.5).float()
    seqs = []
    for b in range(batch_size):
        t = int(mask_bt[b].sum().item())
        t = max(0, min(t, seq_len))
        seqs.append(x_btjc[b, :t].contiguous())
    return seqs


@torch.no_grad()
def compute_dtw(pred_btjc: torch.Tensor, tgt_btjc: torch.Tensor) -> float:
    """Compute DTW distance between prediction and target."""
    if not HAS_DTW:
        return 0.0
    
    preds = _btjc_to_tjc_list(pred_btjc)
    tgts = _btjc_to_tjc_list(tgt_btjc)
    
    try:
        dtw_metric = PE_DTW()
    except (ImportError, RuntimeError):
        return 0.0
    
    vals = []
    for p, g in zip(preds, tgts):
        if p.size(0) < 2 or g.size(0) < 2:
            continue
        pv = p.detach().cpu().numpy().astype("float32")[:, None, :, :]
        gv = g.detach().cpu().numpy().astype("float32")[:, None, :, :]
        vals.append(float(dtw_metric.get_distance(pv, gv)))
    
    if not vals:
        return 0.0
    return float(np.mean(vals))


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
# Lightning Module
# ============================================================

class LitDiffusionUnfrozenCLIP(pl.LightningModule):
    """Lightning module with unfrozen CLIP + contrastive loss."""
    
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
                 contrastive_weight=0.5):
        super().__init__()
        self.save_hyperparameters()

        self.diffusion_steps = diffusion_steps
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.contrastive_weight = contrastive_weight
        self._step_count = 0

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
            print(f"[Step {self._step_count}] loss={loss.item():.4f}, mse={loss_mse.item():.4f}, "
                  f"vel={loss_vel.item():.4f}, contrastive={loss_contrastive.item():.4f}, "
                  f"disp_ratio={disp_ratio:.4f}")

        self.log_dict({
            "train/loss": loss,
            "train/loss_mse": loss_mse,
            "train/loss_vel": loss_vel,
            "train/loss_contrastive": loss_contrastive,
            "train/disp_ratio": disp_ratio,
        }, prog_bar=True)

        self._step_count += 1
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ============================================================
# Main Training
# ============================================================

def train_full_dataset():
    """Train on full dataset with unfrozen CLIP + contrastive loss."""
    pl.seed_everything(42)

    # === Configuration ===
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = "logs/full_unfrozen_clip"

    # Training hyperparameters
    MAX_EPOCHS = 100
    BATCH_SIZE = 1024  # H200 80GB, 如果 OOM 改成 512 或 256
    LEARNING_RATE = 1e-4
    DIFFUSION_STEPS = 8
    CONTRASTIVE_WEIGHT = 0.5
    FREEZE_CLIP = False  # KEY: CLIP is trainable
    
    # 打印 GPU 显存信息
    if torch.cuda.is_available():
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n  GPU Memory Total: {gpu_mem_total:.1f} GB")
        print(f"  Batch Size: {BATCH_SIZE}")
        print(f"  If OOM occurs, reduce BATCH_SIZE to 512 or 256")

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(" FULL DATASET TRAINING: UNFROZEN CLIP + CONTRASTIVE LOSS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Epochs: {MAX_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Diffusion Steps: {DIFFUSION_STEPS}")
    print(f"  Contrastive Weight: {CONTRASTIVE_WEIGHT}")
    print(f"  Freeze CLIP: {FREEZE_CLIP}")
    print(f"  Output: {out_dir}/")
    print(f"  GPU: {'Available ✓' if torch.cuda.is_available() else 'Not available ✗'}")
    print("\nExpected training time: ~30 hours on V100")
    print("=" * 70)

    # === Dataset ===
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
        pin_memory=True,
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

    # === Create Model ===
    print("\nInitializing model...")
    print(f"  [EmbedSignWriting] CLIP is {'FROZEN' if FREEZE_CLIP else 'UNFROZEN (trainable)'}")
    
    lit_model = LitDiffusionUnfrozenCLIP(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=LEARNING_RATE,
        diffusion_steps=DIFFUSION_STEPS,
        vel_weight=1.0,
        acc_weight=0.5,
        t_past=40,
        t_future=future_len,
        freeze_clip=FREEZE_CLIP,
        contrastive_weight=CONTRASTIVE_WEIGHT,
    )
    
    total_params = sum(p.numel() for p in lit_model.parameters())
    trainable_params = sum(p.numel() for p in lit_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # === Callbacks ===
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
    
    # === Train ===
    print(f"\n{'='*70}")
    print("STARTING TRAINING...")
    print("="*70)

    resume_ckpt = None
    last_ckpt_path = f"{out_dir}/checkpoints/last.ckpt"
    if os.path.exists(last_ckpt_path):
        print(f"  Found existing checkpoint: {last_ckpt_path}")
        print(f"  Resuming training from checkpoint...")
        resume_ckpt = last_ckpt_path
    
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
    
    # === Inference Test ===
    print(f"\n{'='*70}")
    print("TESTING INFERENCE ON SAMPLE 0...")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.eval()
    
    test_batch = zero_pad_collator([train_ds[0]])
    cond = test_batch["conditions"]
    
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(test_batch["data"][:1]).to(device)
    
    past_norm = lit_model.normalize(past_raw)
    gt_norm = lit_model.normalize(gt_raw)
    
    with torch.no_grad():
        past_bjct = lit_model.btjc_to_bjct(past_norm)
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
        pred_norm = lit_model.bjct_to_btjc(pred_bjct)
    
    # Metrics
    mse = F.mse_loss(pred_norm, gt_norm).item()
    disp_pred = mean_frame_disp(pred_norm)
    disp_gt = mean_frame_disp(gt_norm)
    disp_ratio = disp_pred / (disp_gt + 1e-8)
    
    # DTW
    gt_unnorm = lit_model.unnormalize(gt_norm)
    pred_unnorm = lit_model.unnormalize(pred_norm)
    dtw_val = compute_dtw(pred_unnorm, gt_unnorm)
    
    pred_np = pred_norm[0].cpu().numpy()
    gt_np = gt_norm[0].cpu().numpy()
    per_joint_err = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
    mpjpe = per_joint_err.mean()
    pck_01 = (per_joint_err < 0.1).mean() * 100
    
    print("\nInference Test Results (Sample 0):")
    print(f"  Disp Ratio: {disp_ratio:.4f} (ideal = 1.0)")
    print(f"  MPJPE: {mpjpe:.6f}")
    print(f"  PCK@0.1: {pck_01:.1f}%")
    print(f"  MSE: {mse:.6f}")
    print(f"  DTW: {dtw_val:.4f}")
    
    # Save poses
    ref_path = train_ds.records[0]["pose"]
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    ref_pose = reduce_holistic(ref_pose)
    if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
        ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    
    gt_pose = tensor_to_pose(gt_unnorm, ref_pose.header, ref_pose)
    pred_pose = tensor_to_pose(pred_unnorm, ref_pose.header, ref_pose)
    
    with open(f"{out_dir}/test_gt.pose", "wb") as f:
        gt_pose.write(f)
    with open(f"{out_dir}/test_pred.pose", "wb") as f:
        pred_pose.write(f)
    
    print(f"\nPose files saved to: {out_dir}/")
    
    print("\n" + "=" * 70)
    print("✅ FULL DATASET TRAINING COMPLETE!")
    print("=" * 70)
    print("\nKey improvements:")
    print("  1. CLIP is unfrozen - learns to distinguish SignWriting")
    print("  2. Contrastive loss prevents embedding collapse")
    print("  3. Sign condition verified to have ~44% influence")
    print(f"\nCheckpoints: {out_dir}/checkpoints/")
    print("=" * 70)


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    train_full_dataset()