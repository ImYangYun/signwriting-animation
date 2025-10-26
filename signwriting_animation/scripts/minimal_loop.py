# -*- coding: utf-8 -*-
import os
import random
import torch
import numpy as np
import lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.holistic import holistic_skeleton
from pose_format.utils.openpose import openpose_skeleton
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, masked_dtw


def _to_plain_tensor(x):
    """Convert MaskedTensor or custom tensor to plain CPU tensor."""
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().cpu()

def _as_dense_cpu_btjc(x):
    if hasattr(x, "tensor"):
        x = x.tensor
    return x.detach().cpu()

def ensure_skeleton(header):
    """Ensure header has a skeleton for visualization."""
    if header is None:
        print("⚠️ [ensure_skeleton] header is None, cannot add skeleton.")
        return None
    if getattr(header, "skeleton", None):
        return header  # already has skeleton

    try:
        header.skeleton = holistic_skeleton()
        print("✅ Added holistic skeleton using pose-format utils.")
    except Exception as e:
        print(f"⚠️ holistic_skeleton() not available: {e}")
        try:
            header.skeleton = openpose_skeleton()
            print("✅ Added openpose skeleton as fallback.")
        except Exception as e2:
            print(f"❌ Failed to add any skeleton: {e2}")
    return header


def save_pose_files(gen_btjc_cpu, gt_btjc_cpu, header, data_dir, csv_path):
    """Save predicted and ground-truth pose sequences as .pose files."""
    try:
        os.makedirs("logs", exist_ok=True)
        header = ensure_skeleton(header)

        # Save predicted pose
        pose_pred = Pose(header, gen_btjc_cpu[0].numpy())
        with open("logs/prediction.pose", "wb") as f:
            pose_pred.write(f)

        # Save GT pose
        pose_gt = Pose(header, gt_btjc_cpu[0].numpy())
        with open("logs/groundtruth.pose", "wb") as f:
            pose_gt.write(f)

        print("✅ Saved prediction.pose & groundtruth.pose to logs/")
        return True
    except Exception as e:
        print(f"❌ Failed saving pose files: {e}")
        return False


def save_scatter_backup(seq_btjc, save_path, title="PRED"):
    """Fallback visualization if pose saving fails."""
    seq = _to_plain_tensor(seq_btjc)[0]
    T, J, C = seq.shape
    plt.figure(figsize=(5, 5))
    for t in range(0, T, max(1, T // 20)):
        plt.scatter(seq[t, :, 0], -seq[t, :, 1], label=f"t={t}")
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved scatter fallback: {save_path}")


# ============================== Main Training & Generation ==============================

def make_loader(data_dir, csv_path, split="train", bs=2, num_workers=2):
    dataset = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split=split,
        reduce_holistic=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        collate_fn=zero_pad_collator,
        num_workers=num_workers,
        pin_memory=False,
    )
    return loader


# ============================== MAIN ==============================

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_default_dtype(torch.float32)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    batch_size, num_workers = 2, 2

    train_loader = make_loader(data_dir, csv_path, "train", bs=batch_size, num_workers=num_workers)
    val_loader = train_loader

    print("\n" + "="*60)
    batch = next(iter(train_loader))
    print("[DATA DEBUG]")
    print(f"  data.shape        = {batch['data'].shape}")
    print(f"  target_mask.shape = {batch['conditions']['target_mask'].shape}")
    print(f"  input_pose.shape  = {batch['conditions']['input_pose'].shape}")
    print("="*60 + "\n")

    gt = _to_plain_tensor(batch["data"][0])
    gt = gt.numpy()
    frame_diff = np.abs(gt[1:] - gt[:-1]).mean()
    print(f"[DATA CHECK] mean|ΔGT| = {frame_diff:.6f}")
    if frame_diff < 1e-3:
        print("⚠️ Warning: this GT sample looks static (almost no motion). Try increasing target_count or max_scan.")

    # --- Training
    model = LitMinimal(log_dir="logs")
    trainer = pl.Trainer(
        max_steps=500,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=5,
        limit_train_batches=10,
        limit_val_batches=5,
        deterministic=True,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_loader, val_loader)

    # --- Generation
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        cond  = batch["conditions"]

        past_btjc = cond["input_pose"][:1].to(model.device)
        sign_img  = cond["sign_image"][:1].to(model.device)
        fut_gt    = batch["data"][:1].to(model.device)

        print("[GEN] Generating full sequence...")
        gen_btjc = model.generate_full_sequence(past_btjc, sign_img, target_len=20)

        gen_btjc_cpu = _as_dense_cpu_btjc(gen_btjc)
        fut_gt_cpu   = _as_dense_cpu_btjc(fut_gt)

        def frame_disp(x_btjc):
            x = x_btjc[0]
            return (x[1:, :, :2] - x[:-1, :, :2]).abs().mean().item() if x.size(0) > 1 else 0.0

        print(f"[GEN] Tf={gen_btjc_cpu.size(1)}, mean|Δpred|={frame_disp(gen_btjc_cpu):.6f}, mean|Δgt|={frame_disp(fut_gt_cpu):.6f}")

        try:
            mask_for_eval = torch.ones(1, gen_btjc.size(1), device=gen_btjc.device)
            dtw_val = masked_dtw(gen_btjc, fut_gt.to(gen_btjc.device), mask_for_eval).item()
            print(f"[EVAL] DTW (pred vs GT): {dtw_val:.4f}")
        except Exception as e:
            print(f"[EVAL] DTW failed: {e}")

        # --- Save & visualize
        header = None

        # ① Try to load header from data_dir
        for root, _, files in os.walk(data_dir):
            for name in files:
                if name.endswith(".pose"):
                    ref_pose_path = os.path.join(root, name)
                    try:
                        with open(ref_pose_path, "rb") as f:
                            pose = Pose.read(f)
                            header = pose.header
                            print(f"[HEADER] ✅ Loaded reference header from {ref_pose_path}")
                            break
                    except Exception:
                        continue
            if header:
                break

        header = ensure_skeleton(header)
        pose_saved = save_pose_files(gen_btjc_cpu, fut_gt_cpu, header, data_dir, csv_path)

        if not pose_saved:
            print("[FALLBACK] Using scatter backup...")
            save_scatter_backup(gen_btjc_cpu, "logs/scatter_pred.gif", "PRED")
            save_scatter_backup(fut_gt_cpu, "logs/scatter_gt.gif", "GT")

        # --- Convert to video (optional)
        try:
            for name in ["prediction", "groundtruth"]:
                pose_path = f"logs/{name}.pose"
                if os.path.exists(pose_path):
                    with open(pose_path, "rb") as f:
                        pose = Pose.read(f)
                        if not getattr(pose.header, "skeleton", None):
                            pose.header.skeleton = holistic_skeleton()
                        v = PoseVisualizer(pose)
                        v.save_video(f"logs/{name}.mp4", v.draw())
                        print(f"🎥 Saved visualization video: logs/{name}.mp4")
        except Exception as e:
            print(f"⚠️ Visualization with PoseVisualizer failed: {e}")
