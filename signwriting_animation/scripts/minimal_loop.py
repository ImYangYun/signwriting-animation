# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose import PoseHeader, PoseHeaderDimensions, PoseHeaderComponent
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.holistic import holistic_components
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import unnormalize_mean_std

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, masked_dtw


try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

def _patched_num_dims(self):
    """Override buggy num_dims(): force match to PoseHeaderDimensions.depth"""
    return self.dimensions.depth

PoseHeader.num_dims = _patched_num_dims
print("[PATCH] PoseHeader.num_dims() overridden to use dimensions.depth only")

def _to_plain_tensor(x):
    """Convert possibly masked/lightning tensor to plain CPU tensor."""
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().cpu()


def build_pose(tensor_btjc, header):
    """
    Convert [B,T,J,C] or [B,T,1,J,C] tensor to Pose object with verified shape.
    """
    arr = _to_plain_tensor(tensor_btjc)
    if arr.dim() == 5:  # [B,T,1,J,C]
        arr = arr[:, :, 0, :, :]
    if arr.dim() == 4:  # [B,T,J,C]
        arr = arr[0]
    if arr.dim() != 3:
        raise ValueError(f"[build_pose] Unexpected tensor shape: {arr.shape}")

    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr = arr[:, None, :, :]  # [T, P=1, J, C]
    conf = np.ones((arr.shape[0], arr.shape[1], arr.shape[2], 1), dtype=np.float32)

    print(f"[build_pose] Final data shape={arr.shape}, conf shape={conf.shape}")
    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    return Pose(header=header, body=body)


def safe_save_pose(pose_obj, out_path):
    """
    Save pose safely with validation (Fluent-Pose compatible).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    body = pose_obj.body.data
    T, P, J, C = body.shape
    hdr = pose_obj.header

    print(f"[SAVE] header.num_dims()={hdr.num_dims()} | depth={hdr.dimensions.depth} | body.C={C}")
    if hdr.dimensions.depth != C:
        print(f"[WARN] Header depth mismatch → forcing depth={C}")
        hdr.dimensions.depth = C

    with open(out_path, "wb") as f:
        pose_obj.write(f)
    print(f"💾 Saved pose: {out_path} | shape={body.shape}")


if __name__ == "__main__":
    pl.seed_everything(42)
    out_dir = "logs/test"
    os.makedirs(out_dir, exist_ok=True)

    print("[DATA] Loading dataset...")
    dataset = DynamicPosePredictionDataset(
        data_dir="/data/yayun/pose_data",
        csv_path="/data/yayun/signwriting-animation/data_fixed.csv",
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="test",
        reduce_holistic=True,  # ✅ reduce face for visualization stability
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=zero_pad_collator)
    batch = next(iter(loader))

    shape = batch["data"].shape
    print(f"[INFO] batch['data'].shape = {shape}")
    if len(shape) == 5:
        B, T, P, J, C = shape
    elif len(shape) == 4:
        B, T, J, C = shape
    else:
        raise ValueError(f"Unexpected data shape: {shape}")
    print(f"[INFO] Detected {J} joints, {C} dims")

    # ----------------------- Init model -----------------------
    print("[MODEL] Initializing model...")
    model = LitMinimal(num_keypoints=J, num_dims=C)
    model.eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

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

        try:
            gt = unnormalize_mean_std(gt)
            pred = unnormalize_mean_std(pred)
            print("[INFO] Unnormalized pose data for visualization.")
        except Exception as e:
            print(f"[WARN] Could not unnormalize (likely already unscaled): {e}")

        # ----------------------- Build header (Fluent-Pose style) -----------------------
        comps = [c for c in holistic_components()
                 if c.name in ["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]]
        for c in comps:
            c.format = "x y z"  # ✅ 最新 pose_format 要用 format，不是 point_format

        header = PoseHeader(
            version=1,
            dimensions=PoseHeaderDimensions(width=1, height=1, depth=3),
            components=comps,
        )

        gt_pose = build_pose(gt, header)
        pred_pose = build_pose(pred, header)

        print(f"GT range: [{np.nanmin(gt_pose.body.data):.4f}, {np.nanmax(gt_pose.body.data):.4f}]")
        print(f"Pred range: [{np.nanmin(pred_pose.body.data):.4f}, {np.nanmax(pred_pose.body.data):.4f}]")

        # ----------------------- Save -----------------------
        safe_save_pose(gt_pose, os.path.join(out_dir, "groundtruth.pose"))
        safe_save_pose(pred_pose, os.path.join(out_dir, "prediction.pose"))

        try:
            viz = PoseVisualizer(gt_pose)
            viz.save_video(os.path.join(out_dir, "groundtruth.mp4"), fps=25)
            viz = PoseVisualizer(pred_pose)
            viz.save_video(os.path.join(out_dir, "prediction.mp4"), fps=25)
            print("🎞️ Saved pose videos successfully.")
        except Exception as e:
            print(f"[WARN] Could not render video: {e}")

        print(f"\n✅ Finished. Results saved in {os.path.abspath(out_dir)}")
        print(f"Output files: {os.listdir(out_dir)}")
