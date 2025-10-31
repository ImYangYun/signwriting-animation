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
    header = PoseHeader(version=1, dimensions=dims, components=[component])
    header.num_dims = lambda: num_dims
    return header


def ensure_header_matches_body(header, body_array):
    J = body_array.shape[2]
    total_joints = sum(len(c.points) for c in header.components)
    if total_joints != J:
        print(f"[WARN] Header joints ({total_joints}) != data joints ({J}) → rebuilding header skipped")
        header.components[0].points = [f"joint_{i}" for i in range(J)]
        header.components[0].limbs = [(i, i + 1) for i in range(J - 1)]
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


def safe_save_pose_verified(pose_obj, out_path, dataset_header=None):
    """
    Safely save a Pose object (.pose file) with automatic header validation and repair.
    Fixes the 'Header has 4 dimensions, but body has 3' issue by forcing exact dimensional sync.
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not isinstance(pose_obj, Pose):
        print(f"[ERROR] Provided object is not a Pose instance: {type(pose_obj)}")
        return

    body = pose_obj.body.data
    if not isinstance(body, np.ndarray):
        body = np.asarray(body)
    if body.ndim != 4:
        print(f"[ERROR] Unexpected body shape {body.shape} (expected [T, P, J, C])")
        return

    T, P, J, C = body.shape
    header = dataset_header if dataset_header is not None else pose_obj.header

    header_joint_count = sum(len(c.points) for c in header.components)
    header_dims = header.num_dims()
    print(f"[CHECK] Before save → header joints={header_joint_count}, body joints={J}")
    print(f"[CHECK] Header num_dims={header_dims}, Body num_dims={C}")

    if header_joint_count != J or pose_obj.header.num_dims() != C:
        print(f"[WARN] header ({header_joint_count} joints, {pose_obj.header.num_dims()} dims)"
            f" != body ({J} joints, {C} dims) → rebuilding header")

        points = [f"joint_{i}" for i in range(J)]
        limbs = [(i, i+1) for i in range(J-1)]
        colors = [(255, 255, 255)] * len(limbs)

        pfmt = "x y z" if C == 3 else "x y"

        component = PoseHeaderComponent(
            name="POSE_LANDMARKS",
            points=points,
            limbs=limbs,
            colors=colors,
            point_format="x y z",
            has_confidence=False,
        )

        new_header = PoseHeader(
            version=1,
            dimensions=PoseHeaderDimensions(width=1, height=1, depth=C),
            components=[component],
        )
        new_header.num_dims = lambda: C
        pose_obj = Pose(header=new_header, body=pose_obj.body)

        print(f"[FIX] Rebuilt header → {J} joints, depth={C}, header.num_dims={pose_obj.header.num_dims()}")

    try:
        with open(out_path, "wb") as f:
            pose_obj.write(f)
        with open(out_path, "rb") as f_check:
            Pose.read(f_check.read())
        print(f"💾 Saved + verified successfully: {out_path} | shape={body.shape}")
    except Exception as e:
        print(f"❌ Failed to save pose: {e}")


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
        comps = [c for c in comps if c.name in ["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]]
        for c in comps:
            c.point_format = "x y z "

        header = PoseHeader(
            version=1,
            dimensions=PoseHeaderDimensions(width=1, height=1, depth=3),
            components=comps,
        )

        gt_pose = build_pose(gt, header)
        pred_pose = build_pose(pred, header)

        print(f"GT data range: [{np.nanmin(gt_pose.body.data)}, {np.nanmax(gt_pose.body.data)}]")
        print(f"Pred data range: [{np.nanmin(pred_pose.body.data)}, {np.nanmax(pred_pose.body.data)}]")
        print(f"GT NaN: {np.isnan(gt_pose.body.data).any()}, Pred NaN: {np.isnan(pred_pose.body.data).any()}")

        safe_save_pose_verified(gt_pose, "logs/test/groundtruth.pose", dataset_header=header)
        safe_save_pose_verified(pred_pose, "logs/test/prediction.pose", dataset_header=header)

        print(f"\n✅ Finished. Results saved in {os.path.abspath(out_dir)}")
        print(f"Output files: {os.listdir(out_dir)}")
