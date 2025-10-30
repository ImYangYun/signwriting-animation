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
from pose_format.pose import PoseHeader, PoseHeaderDimensions
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.holistic import holistic_components
from pose_format.torch.masked.collator import zero_pad_collator

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


def build_pose(tensor_btjc, header):
    """Convert model output [B,T,J,C] or [B,T,1,J,C] → Pose object."""
    arr = _to_plain_tensor(tensor_btjc)
    if arr.dim() == 5:  # [B,T,1,J,C]
        arr = arr[:, :, 0, :, :]
    if arr.dim() == 4:  # [B,T,J,C]
        arr = arr[0]
    elif arr.dim() == 3:  # [T,J,C]
        pass
    else:
        raise ValueError(f"Unexpected pose tensor shape: {arr.shape}")

    # [T,1,J,C] for Pose format
    arr = arr[:, None, :, :]
    conf = np.ones_like(arr[..., :1], dtype=np.float32)  # confidence = 1
    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    return Pose(header=header, body=body)


def save_pose_and_video(pose_obj, out_prefix):
    """Save .pose + try to save .mp4 visual."""
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    pose_path = out_prefix + ".pose"
    mp4_path = out_prefix + ".mp4"

    try:
        pose_obj = pose_obj.remove_components([
            c.name for c in pose_obj.header.components if c.name != "POSE_LANDMARKS"
        ])
        print(f"[CLEAN] Components kept: {[c.name for c in pose_obj.header.components]}")
    except Exception as e:
        print(f"[WARN] Could not filter components: {e}")

    # Save .pose
    with open(pose_path, "wb") as f:
        pose_obj.write(f)
    print(f"💾 Saved pose: {pose_path}")

    try:
        viz = PoseVisualizer(pose_obj)
        T = pose_obj.body.data.shape[0]
        viz.save_video(mp4_path, frames=range(T))  # removed 'fps' to avoid TypeError
        print(f"🎞️ Saved video: {mp4_path}")
    except Exception as e:
        print(f"⚠️ Failed to save video: {e}")


# --------------------- Main ---------------------
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
        reduce_holistic=True,   # ✅ use reduced joints for visual stability
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=zero_pad_collator)
    batch = next(iter(loader))

    # auto-detect dimension (handles [B,T,1,J,C] or [B,T,J,C])
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

        comps = holistic_components()
        header = PoseHeader(
            version=1,
            dimensions=PoseHeaderDimensions(width=1, height=1, depth=3),
            components=comps,
        )
        print(f"[INFO] Holistic header built with {sum(len(c.limbs) for c in header.components)} limbs")

        gt_pose = build_pose(gt, header)
        pred_pose = build_pose(pred, header)

        print(f"GT data range: [{np.nanmin(gt_pose.body.data)}, {np.nanmax(gt_pose.body.data)}]")
        print(f"Pred data range: [{np.nanmin(pred_pose.body.data)}, {np.nanmax(pred_pose.body.data)}]")
        print(f"GT NaN: {np.isnan(gt_pose.body.data).any()}, Pred NaN: {np.isnan(pred_pose.body.data).any()}")

        save_pose_and_video(gt_pose, os.path.join(out_dir, "groundtruth"))
        save_pose_and_video(pred_pose, os.path.join(out_dir, "prediction"))

        print(f"\n✅ Finished. Results saved in {os.path.abspath(out_dir)}")
        print(f"Output files: {os.listdir(out_dir)}")
