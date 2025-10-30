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
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


# --------------------- Tensor utilities ---------------------
def _to_plain_tensor(x):
    """Convert possibly masked/lightning tensor to plain CPU tensor."""
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().cpu()


def build_pose(tensor_btjc, header):
    """
    Convert model output [1,T,J,C] → Pose(header, body)
    Creates confidence channel [T,1,J].
    """
    arr = _to_plain_tensor(tensor_btjc)
    if arr.dim() == 5:  # [B,1,T,J,C]
        arr = arr[0, 0]
    elif arr.dim() == 4:  # [B,T,J,C]
        arr = arr[0]
    elif arr.dim() == 3:
        pass
    else:
        raise ValueError(f"Unexpected tensor shape: {arr.shape}")

    # [T,J,C] → [T,1,J,C]
    data = arr.unsqueeze(1).numpy()
    conf = np.ones((data.shape[0], 1, data.shape[2]), dtype=np.float32)
    body = NumPyPoseBody(fps=25, data=data, confidence=conf)
    return Pose(header=header, body=body)


def save_pose_and_video(pose_obj, out_prefix):
    """
    Save both .pose file and .mp4 video.
    No fps arg (for old pose_format API).
    """
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    pose_path = out_prefix + ".pose"
    mp4_path  = out_prefix + ".mp4"

    with open(pose_path, "wb") as f:
        pose_obj.write(f)
    print(f"💾 Saved pose: {pose_path}")

    try:
        viz = PoseVisualizer(pose_obj)
        T = pose_obj.body.data.shape[0]
        viz.save_video(mp4_path, frames=range(T))
        print(f"🎞️ Saved video: {mp4_path}")
    except Exception as e:
        print(f"⚠️ Failed to save video: {e}")

    print(f"[INFO] Pose data shape: {pose_obj.body.data.shape}, "
          f"range=({pose_obj.body.data.min():.3f}, {pose_obj.body.data.max():.3f})")


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
        reduce_holistic=True,   # ✅ 已启用精简骨架
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=zero_pad_collator)

    batch = next(iter(loader))
    print(f"[INFO] Batch keys: {list(batch.keys())}")

    print("[MODEL] Initializing model...")
    model = LitMinimal(num_keypoints=586, num_dims=3)
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
            components=comps
        )
        print(f"[INFO] Holistic header built with {sum(len(c.limbs) for c in comps)} limbs")

        # Build poses
        gt_pose = build_pose(gt, header)
        pred_pose = build_pose(pred, header)

        # Sanity stats
        print(f"GT range: ({np.nanmin(gt_pose.body.data):.3f}, {np.nanmax(gt_pose.body.data):.3f})")
        print(f"Pred range: ({np.nanmin(pred_pose.body.data):.3f}, {np.nanmax(pred_pose.body.data):.3f})")
        print(f"GT NaN: {np.isnan(gt_pose.body.data).any()}, Pred NaN: {np.isnan(pred_pose.body.data).any()}")

        # Save results
        save_pose_and_video(gt_pose, os.path.join(out_dir, "groundtruth"))
        save_pose_and_video(pred_pose, os.path.join(out_dir, "prediction"))

        print(f"\n✅ Finished. Results saved in {os.path.abspath(out_dir)}")
        print(f"Output files: {os.listdir(out_dir)}")
