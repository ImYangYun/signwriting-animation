# -*- coding: utf-8 -*-
"""
Minimal FluentPose-style loop for quick visualization & pose export
Author: yayun
"""
import os
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader, Subset

from pose_format import Pose
from pose_format.pose import PoseHeader
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.holistic import holistic_components
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, masked_dtw


def to_tjc_anyshape(x):
    """Convert tensor to [T,J,C] np.float32 (handles multiple possible shapes)."""
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    x = np.array(x)

    if x.ndim == 5:  # [1,T,1,J,C]
        x = x[0, :, 0, :, :]
    elif x.ndim == 4:
        if x.shape[0] == 1:  # [1,T,J,C]
            x = x[0]
        elif x.shape[-1] < 300:  # [1,J,C,T]
            x = np.transpose(x[0], (2, 0, 1))
    elif x.ndim != 3:
        raise ValueError(f"[to_tjc_anyshape] unexpected shape {x.shape}")
    return x.astype(np.float32)


def ensure_header(dataset):
    """Prefer dataset header if it has limbs, else fallback to holistic_components()."""
    header = getattr(dataset, "pose_header", None)
    if header is not None and any(len(c.limbs) > 0 for c in header.components):
        print("✅ Using header from dataset (with limbs).")
        return header
    print("⚠️ Dataset header missing or limbs empty — using holistic_components().")
    return PoseHeader(components=holistic_components())


def build_pose(btjc, header, fps=25.0):
    """Convert model/GT tensor to Pose() with proper shape."""
    tjc = to_tjc_anyshape(btjc)
    T, J, C = tjc.shape
    body = NumPyPoseBody(
        fps=fps,
        data=tjc[:, np.newaxis, :, :],
        confidence=np.ones((T, 1, J), dtype=np.float32)
    )
    return Pose(header, body)


def save_pose_and_video(pose_obj, out_prefix):
    """Save both .pose and .mp4 video using PoseVisualizer."""
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    pose_path = out_prefix + ".pose"
    with open(pose_path, "wb") as f:
        pose_obj.write(f)
    print(f"💾 Saved pose file: {pose_path}")

    mp4_path = out_prefix + ".mp4"
    viz = PoseVisualizer(pose_obj)
    viz.save_video(mp4_path)
    print(f"🎥 Saved video: {mp4_path}")


# ---------------------- dataloader ----------------------
def make_small_loader(data_dir, csv_path, n_samples=8, bs=2):
    """Load a small subset for quick testing."""
    dataset = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
        reduce_holistic=False,
    )
    subset = Subset(dataset, list(range(min(len(dataset), n_samples))))
    loader = DataLoader(
        subset,
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        collate_fn=zero_pad_collator,
    )
    print(f"[INFO] Loaded {len(subset)} samples for quick test.")
    return loader


# ---------------------- main loop ----------------------
if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_default_dtype(torch.float32)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "/data/yayun/signwriting-animation-fork/logs/test"
    os.makedirs(out_dir, exist_ok=True)

    # --- Data ---
    loader = make_small_loader(data_dir, csv_path, n_samples=8, bs=2)
    batch = next(iter(loader))
    print(f"[INFO] Batch shape: {batch['data'].shape}")

    # --- Tiny train (quiet) ---
    model = LitMinimal(log_dir=out_dir)
    trainer = pl.Trainer(
        max_steps=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=25,
        limit_train_batches=5,
        enable_checkpointing=False,
        enable_progress_bar=True,
        deterministic=True,
    )
    trainer.fit(model, loader, loader)

    # --- Inference ---
    model.eval()
    with torch.no_grad():
        cond = batch["conditions"]
        past = cond["input_pose"][:1].to(model.device)
        sign_img = cond["sign_image"][:1].to(model.device)
        gt = batch["data"][:1].to(model.device)

        print("[GEN] Generating future motion...")
        pred = model.generate_full_sequence(past, sign_img, target_len=20)

        # --- Evaluation ---
        mask = torch.ones(1, pred.size(1), device=pred.device)
        dtw_val = masked_dtw(pred, gt, mask).item()
        print(f"[EVAL] masked_dtw = {dtw_val:.4f}")

        # --- Pose export ---
        header = ensure_header(loader.dataset)
        gt_pose = build_pose(gt, header)
        pred_pose = build_pose(pred, header)

        save_pose_and_video(gt_pose, os.path.join(out_dir, "groundtruth"))
        save_pose_and_video(pred_pose, os.path.join(out_dir, "prediction"))

    print("\n✅ Finished. Results saved in:", os.path.abspath(out_dir))
