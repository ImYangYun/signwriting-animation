"""
Sign-Only Training: Test if model can learn sign→motion mapping

Key idea:
- Phase 1: past_motion = 0 (100% dropout), only sign available
- This tests whether sign→motion mapping is even learnable

If Phase 1 works → sign is informative, just being overshadowed by past
If Phase 1 fails → sign→motion mapping is fundamentally too hard

Usage:
    python train_sign_only.py --mode overfit --epochs 300
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader, Subset
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
# Model Components (Same as before)
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

def tensor_to_pose(t_btjc: torch.Tensor, 
                   header, 
                   ref_pose: Pose, 
                   scale_to_ref: bool = True) -> Pose:
    """Convert tensor prediction to Pose format for visualization.
    
    CORRECT VERSION - from original train_full.py
    """
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc
    
    t_np = t.detach().cpu().numpy().astype(np.float32)
    
    # Create pose object
    arr = t_np[:, None, :, :]
    T = arr.shape[0]
    conf = np.ones((T, 1, arr.shape[2]), dtype=np.float32)
    fps = ref_pose.body.fps
    
    body = NumPyPoseBody(fps=fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)
    
    # Fix hand positions (must be before scaling!)
    unshift_hands(pose_obj)
    
    if scale_to_ref:
        # Get reference data
        T_pred = t_np.shape[0]
        T_ref_total = ref_pose.body.data.shape[0]
        future_start = max(0, T_ref_total - T_pred)
        ref_arr = np.asarray(
            ref_pose.body.data[future_start:future_start+T_pred, 0], 
            dtype=np.float32
        )
        
        # Scale to reference variance
        def _var(a):
            center = a.mean(axis=(0, 1), keepdims=True)
            return float(((a - center) ** 2).mean())
        
        pose_data = pose_obj.body.data[:, 0, :, :]
        var_input = _var(pose_data)
        var_ref = _var(ref_arr)
        
        if var_input > 1e-8:
            scale = np.sqrt(var_ref / var_input)
            pose_obj.body.data = pose_obj.body.data * scale
        
        # Align center (CORRECT VERSION)
        pose_data = pose_obj.body.data[:, 0, :, :].reshape(-1, 3)
        input_center = pose_data.mean(axis=0)
        ref_center = ref_arr.reshape(-1, 3).mean(axis=0)
        pose_obj.body.data = pose_obj.body.data + (ref_center - input_center)
    
    return pose_obj


# ============================================================
# Lightning Module - SIGN ONLY
# ============================================================

class LitDiffusionSignOnly(pl.LightningModule):
    """
    Sign-Only Training: past_motion is ALWAYS zero
    This tests if sign→motion mapping is learnable at all
    """
    
    def __init__(self, num_keypoints=178, num_dims=3, lr=1e-4,
                 stats_path="/home/yayun/data/pose_data/mean_std_178_with_preprocess.pt",
                 diffusion_steps=8, vel_weight=1.0, acc_weight=0.5,
                 t_past=40, t_future=20, contrastive_weight=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.diffusion_steps = diffusion_steps
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.contrastive_weight = contrastive_weight
        self._step_count = 0

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
        
        # ============================================================
        # KEY: past_motion is ALWAYS ZERO - only sign is available
        # ============================================================
        past_norm = torch.zeros_like(self.normalize(past_btjc))

        batch_size, device = gt_norm.shape[0], gt_norm.device
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
            loss_acc = F.mse_loss(pred_vel[..., 1:] - pred_vel[..., :-1], gt_vel[..., 1:] - gt_vel[..., :-1])

        # Contrastive
        loss_contrastive = torch.tensor(0.0, device=device)
        if batch_size > 1 and self.contrastive_weight > 0:
            sign_embs = F.normalize(self.model.embed_signwriting(sign_img), p=2, dim=-1)
            cos_sim = torch.mm(sign_embs, sign_embs.t())
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            loss_contrastive = cos_sim[mask].mean()

        loss = loss_mse + self.vel_weight * loss_vel + self.acc_weight * loss_acc + self.contrastive_weight * loss_contrastive

        with torch.no_grad():
            disp_ratio = pred_vel.abs().mean().item() / (gt_vel.abs().mean().item() + 1e-8)

        if self._step_count % 100 == 0:
            print(f"[Step {self._step_count}] [SIGN-ONLY] loss={loss.item():.4f}, mse={loss_mse.item():.4f}, "
                  f"vel={loss_vel.item():.4f}, disp_ratio={disp_ratio:.4f}")

        self.log_dict({"train/loss": loss, "train/loss_mse": loss_mse, "train/disp_ratio": disp_ratio}, prog_bar=True)
        self._step_count += 1
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ============================================================
# Main
# ============================================================

def train_overfit(args):
    pl.seed_everything(42)
    
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = f"logs/sign_only_overfit_{args.num_samples}sample"

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(" SIGN-ONLY TRAINING (past_motion = 0)")
    print("=" * 70)
    print(f"  Samples: {args.num_samples}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Output: {out_dir}/")
    print("=" * 70)
    print("\nThis tests: Can the model learn sign→motion WITHOUT past_motion?")
    print("=" * 70)

    # Dataset
    full_ds = DynamicPosePredictionDataset(
        data_dir=data_dir, csv_path=csv_path,
        num_past_frames=40, num_future_frames=20, with_metadata=True, split="train",
    )

    # Select samples from different videos
    seen_poses, selected_indices = set(), []
    for idx in range(len(full_ds)):
        if len(selected_indices) >= args.num_samples: break
        pose = full_ds.records[idx].get("pose", "")
        if pose not in seen_poses:
            seen_poses.add(pose)
            selected_indices.append(idx)
    
    print(f"Selected indices: {selected_indices[:10]}...")
    train_ds = Subset(full_ds, selected_indices)

    train_loader = DataLoader(train_ds, batch_size=min(args.num_samples, 32), shuffle=True, collate_fn=zero_pad_collator, num_workers=0)

    # Model
    sample = full_ds[0]["data"]
    if hasattr(sample, 'zero_filled'): sample = sample.zero_filled()
    if hasattr(sample, 'tensor'): sample = sample.tensor
    num_joints, num_dims, future_len = sample.shape[-2], sample.shape[-1], sample.shape[0]

    lit_model = LitDiffusionSignOnly(
        num_keypoints=num_joints, num_dims=num_dims, stats_path=stats_path,
        lr=1e-4, diffusion_steps=8, t_future=future_len, contrastive_weight=0.5,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{out_dir}/checkpoints", save_last=True,
        filename="best-epoch={epoch:03d}", save_top_k=1, monitor="train/disp_ratio", mode="max",
    )

    trainer = Trainer(
        max_epochs=args.epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
        callbacks=[checkpoint_callback], default_root_dir=out_dir,
        log_every_n_steps=10, precision="16-mixed" if torch.cuda.is_available() else 32,
    )
    trainer.fit(lit_model, train_loader)

    # ============================================================
    # Test: Generate poses using ONLY sign (no past)
    # ============================================================
    print("\n" + "=" * 70)
    print("TESTING SIGN-ONLY INFERENCE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.eval()

    def inference_sign_only(sign_img, future_len=20):
        # past is ZERO
        B = sign_img.shape[0]
        past_zero = torch.zeros(B, future_len, num_joints, num_dims, device=device)
        past_norm = lit_model.normalize(past_zero)
        past_bjct = lit_model.btjc_to_bjct(past_norm)
        _, J, C, _ = past_bjct.shape
        
        class W(nn.Module):
            def __init__(s, m, p, sg): super().__init__(); s.m, s.p, s.sg = m, p, sg
            def forward(s, x, t, **kw): return s.m(x, t, s.p, s.sg)
        
        wrapped = W(lit_model.model, past_bjct, sign_img)
        pred_bjct = lit_model.diffusion.p_sample_loop(wrapped, (B, J, C, future_len), clip_denoised=True, model_kwargs={"y": {}}, progress=False)
        return lit_model.unnormalize(lit_model.bjct_to_btjc(pred_bjct))

    # Test on training samples
    results = []
    for idx in selected_indices[:5]:
        batch = zero_pad_collator([full_ds[idx]])
        sign = batch["conditions"]["sign_image"][:1].float().to(device)
        gt = sanitize_btjc(batch["data"][:1]).to(device)  # Already unnormalized!
        
        with torch.no_grad():
            pred = inference_sign_only(sign)
        
        # gt is already unnormalized, don't double process!
        gt_disp = (gt[:, 1:] - gt[:, :-1]).abs().mean().item()
        pred_disp = (pred[:, 1:] - pred[:, :-1]).abs().mean().item()
        ratio = pred_disp / (gt_disp + 1e-8)
        
        diff = (pred - gt).cpu().numpy()[0]
        pck = (np.sqrt((diff**2).sum(-1)) < 0.1).mean() * 100
        
        print(f"idx={idx}: ratio={ratio:.2f}, PCK={pck:.1f}%")
        results.append({"idx": idx, "ratio": ratio, "pck": pck})

    avg_ratio = np.mean([r["ratio"] for r in results])
    avg_pck = np.mean([r["pck"] for r in results])
    
    print("-" * 60)
    print(f"AVG: ratio={avg_ratio:.2f}, PCK={avg_pck:.1f}%")
    print("=" * 70)

    # Save best sample pose
    best_idx = max(results, key=lambda x: x["pck"])["idx"]
    batch = zero_pad_collator([full_ds[best_idx]])
    sign = batch["conditions"]["sign_image"][:1].float().to(device)
    gt = sanitize_btjc(batch["data"][:1]).to(device)  # Already unnormalized!
    
    with torch.no_grad():
        pred = inference_sign_only(sign)
    # gt is already unnormalized, use directly

    ref_path = full_ds.records[best_idx]["pose"]
    if not ref_path.startswith("/"): ref_path = data_dir + ref_path
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    ref_pose = reduce_holistic(ref_pose)
    if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
        ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])

    gt_pose = tensor_to_pose(gt, ref_pose.header, ref_pose)
    pred_pose = tensor_to_pose(pred, ref_pose.header, ref_pose)

    with open(f"{out_dir}/test_gt.pose", "wb") as f: gt_pose.write(f)
    with open(f"{out_dir}/test_pred.pose", "wb") as f: pred_pose.write(f)
    print(f"\nPose saved: {out_dir}/test_*.pose")

    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    if avg_pck > 30:
        print("✅ Sign-Only learning WORKS! Sign contains useful information.")
        print("   → Problem is that past overshadows sign, need better fusion.")
    elif avg_pck > 10:
        print("⚠️  Sign-Only learning is WEAK. Sign has limited information.")
        print("   → May need better sign encoder or more data.")
    else:
        print("❌ Sign-Only learning FAILS. Sign→motion mapping too hard.")
        print("   → This is a fundamental limitation of the task/data.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="overfit")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    train_overfit(args)