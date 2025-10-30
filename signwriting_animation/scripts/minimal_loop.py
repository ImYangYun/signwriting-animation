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


def _to_plain_tensor(x):
    """Convert possibly masked/lightning tensor to plain CPU tensor."""
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().cpu()


def make_reduced_header(num_joints: int, num_dims: int = 3):
    """Fallback: create simplified header when holistic skeleton mismatched."""
    points = [f"joint_{i}" for i in range(num_joints)]
    limbs = [(i, i + 1) for i in range(num_joints - 1)]
    colors = [(255, 255, 255)] * len(limbs)
    component = PoseHeaderComponent(
        name="pose",
        points=points,
        limbs=limbs,
        colors=colors,
        point_format="x y z" if num_dims == 3 else "x y"
    )
    dims = PoseHeaderDimensions(width=1, height=1, depth=num_dims)
    return PoseHeader(version=1, dimensions=dims, components=[component])


def ensure_header_matches_body(header, body_array):
    """Verify header-body consistency; rebuild header if mismatch."""
    J = body_array.shape[2]
    total_joints = sum(len(c.points) for c in header.components)
    if total_joints != J:
        print(f"[WARN] Header joints ({total_joints}) != data joints ({J}) → rebuilding header")
        header = make_reduced_header(num_joints=J, num_dims=body_array.shape[-1])
    else:
        print(f"[INFO] Header matches {J} joints ✓")
    return header


def build_pose(tensor_btjc, header):
    """Convert [B,T,J,C] or [B,T,1,J,C] tensor to Pose object with verified shape."""
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

    print(f"[build_pose] Final data shape={arr.shape}, conf shape={conf.shape}, dtype={arr.dtype}")
    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    print(f"[DEBUG] build_pose → body.data.shape={body.data.shape}, header.num_dims={header.num_dims()}")
    return Pose(header=header, body=body)


def save_pose_and_video(pose_obj, out_prefix):
    """Save pose (.pose) and visualization (.mp4)."""
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    pose_path = out_prefix + ".pose"
    mp4_path = out_prefix + ".mp4"

    pose_obj.header.dimensions = PoseHeaderDimensions(width=1, height=1, depth=3)
    pose_obj.header = ensure_header_matches_body(pose_obj.header, pose_obj.body.data)

    print("Header dimensions:", pose_obj.header.dimensions)
    print("Header num_dims:", pose_obj.header.num_dims())
    print("Body data shape:", pose_obj.body.data.shape)

    try:
        with open(pose_path, "wb") as f:
            pose_obj.write(f)
        print(f"💾 Saved pose: {pose_path} | shape={pose_obj.body.data.shape}")
    except Exception as e:
        print(f"❌ Failed to save pose: {e}")
        return

    try:
        viz = PoseVisualizer(pose_obj)
        T = pose_obj.body.data.shape[0]
        viz.save_video(mp4_path, frames=range(T))
        print(f"🎞️ Saved video: {mp4_path}")
    except Exception as e:
        print(f"⚠️ Failed to save video: {e}")


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

        comps = holistic_components()
        comps = [
            c for c in comps
            if c.name in ["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]
        ]

        for c in comps:
            c.point_format = "x y z"

        header = PoseHeader(
            version=1,
            dimensions=PoseHeaderDimensions(width=1, height=1, depth=3),
            components=comps,
        )
        header = ensure_header_matches_body(header, _to_plain_tensor(gt))

        gt_pose = build_pose(gt, header)
        pred_pose = build_pose(pred, header)

        print(f"GT data range: [{np.nanmin(gt_pose.body.data)}, {np.nanmax(gt_pose.body.data)}]")
        print(f"Pred data range: [{np.nanmin(pred_pose.body.data)}, {np.nanmax(pred_pose.body.data)}]")
        print(f"GT NaN: {np.isnan(gt_pose.body.data).any()}, Pred NaN: {np.isnan(pred_pose.body.data).any()}")

        save_pose_and_video(gt_pose, os.path.join(out_dir, "groundtruth"))
        save_pose_and_video(pred_pose, os.path.join(out_dir, "prediction"))

        print(f"\n✅ Finished. Results saved in {os.path.abspath(out_dir)}")
        print(f"Output files: {os.listdir(out_dir)}")
