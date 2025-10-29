# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader, Subset

# --- pose-format imports ---
from pose_format import Pose
from pose_format.pose import PoseHeader
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.holistic import holistic_components
from pose_format.torch.masked.collator import zero_pad_collator

# --- project imports ---
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, masked_dtw


# ---------------------- Utility Functions ----------------------
def to_tjc_anyshape(x):
    """Ensure input is [T,J,C] np.float32."""
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


def build_pose(btjc, header, fps=25.0):
    """Convert tensor [1,T,J,C] to Pose()"""
    tjc = to_tjc_anyshape(btjc)
    T, J, C = tjc.shape
    body = NumPyPoseBody(
        fps=fps,
        data=tjc[:, np.newaxis, :, :],
        confidence=np.ones((T, 1, J), dtype=np.float32)
    )
    return Pose(header, body)


def save_pose_and_video(pose_obj, out_prefix):
    """Save both .pose and .mp4"""
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    pose_path = out_prefix + ".pose"
    with open(pose_path, "wb") as f:
        pose_obj.write(f)
    print(f"Saved: {pose_path}")

    mp4_path = out_prefix + ".mp4"
    viz = PoseVisualizer(pose_obj)
    viz.save_video(mp4_path)
    print(f"Saved: {mp4_path}")


# ---------------------- Data Loader ----------------------
def make_small_loader(data_dir, csv_path, n_samples=8, bs=2):
    """Small subset loader for quick testing"""
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
    print(f"[INFO] Loaded {len(subset)} samples.")
    return loader


# ---------------------- Main ----------------------
if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_default_dtype(torch.float32)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/test"
    os.makedirs(out_dir, exist_ok=True)

    # --- Load data ---
    loader = make_small_loader(data_dir, csv_path, n_samples=8, bs=2)
    batch = next(iter(loader))
    print(f"[INFO] Batch shape: {batch['data'].shape}")

    # --- Train small model ---
    model = LitMinimal(log_dir=out_dir)
    trainer = pl.Trainer(
        max_steps=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=20,
        enable_checkpointing=False,
        enable_progress_bar=True,
        deterministic=True,
        num_sanity_val_steps=0,  # no validation loop
    )
    trainer.fit(model, loader)

    # --- Inference ---
    model.eval()
    with torch.no_grad():
        cond = batch["conditions"]
        past = cond["input_pose"][:1].to(model.device)
        sign_img = cond["sign_image"][:1].to(model.device)
        gt = batch["data"][:1].to(model.device)

        print("[GEN] Generating future motion...")
        pred = model.generate_full_sequence(past, sign_img, target_len=20)

        mask = torch.ones(1, pred.size(1), device=pred.device)
        dtw_val = masked_dtw(pred, gt, mask).item()
        print(f"[EVAL] masked_dtw = {dtw_val:.4f}")

        # --- Use holistic header directly ---
        header = PoseHeader(version=1, dimensions=3, components=holistic_components())
        print(f"[INFO] Holistic header with {sum(len(c.limbs) for c in header.components)} limbs")

        # --- Build and Save ---
        gt_pose = build_pose(gt, header)
        pred_pose = build_pose(pred, header)

        save_pose_and_video(gt_pose, os.path.join(out_dir, "groundtruth"))
        print("✅ Groundtruth saved OK")

        save_pose_and_video(pred_pose, os.path.join(out_dir, "prediction"))
        print("✅ Prediction saved OK")

        print("GT has NaN?", np.isnan(gt_pose.body.data.filled(np.nan)).all())
        print("Pred has NaN?", np.isnan(pred_pose.body.data.filled(np.nan)).all())

        print(f"\n✅ Finished. Results saved in {os.path.abspath(out_dir)}")

    print(f"\n✅ Finished. Results saved in {os.path.abspath(out_dir)}")
