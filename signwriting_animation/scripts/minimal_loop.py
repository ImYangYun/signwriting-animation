# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import unnormalize_mean_std

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, masked_dtw


try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def _to_plain_tensor(x):
    """Convert MaskedTensor or Lightning tensor to plain CPU tensor."""
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().cpu()


def main():
    pl.seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/test"
    os.makedirs(out_dir, exist_ok=True)

    dataset = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="test",
        reduce_holistic=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=zero_pad_collator)
    batch = next(iter(loader))
    print(f"[INFO] Loaded one batch with shape: {batch['data'].shape}")

    first_pose_path = os.path.join(data_dir, dataset.records[0]["pose"])
    if not first_pose_path.endswith(".pose"):
        first_pose_path += ".pose"
    with open(first_pose_path, "rb") as f:
        header = Pose.read(f).header
    print(f"[INFO] Header loaded from {first_pose_path}")

    B, T, P, J, C = batch["conditions"]["input_pose"].shape
    print(f"[MODEL] Initializing with num_keypoints={J}, num_dims={C}")
    model = LitMinimal(num_keypoints=J, num_dims=C)
    model.eval().to(device)

    cond = batch["conditions"]
    past = cond["input_pose"][:1].to(device)
    sign_img = cond["sign_image"][:1].to(device)
    gt = batch["data"][:1].to(device)

    with torch.no_grad():
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

    gt_np = _to_plain_tensor(gt)[0].numpy()
    pred_np = _to_plain_tensor(pred)[0].numpy()

    gt_pose = Pose(
        header=header,
        body=NumPyPoseBody(fps=25, data=gt_np, confidence=np.ones_like(gt_np[..., :1], np.float32))
    )
    pred_pose = Pose(
        header=header,
        body=NumPyPoseBody(fps=25, data=pred_np, confidence=np.ones_like(pred_np[..., :1], np.float32))
    )

    try:
        with open(os.path.join(out_dir, "groundtruth.pose"), "wb") as f:
            gt_pose.write(f)
        with open(os.path.join(out_dir, "prediction.pose"), "wb") as f:
            pred_pose.write(f)
        print(f"💾 Saved .pose files to {out_dir}")
    except Exception as e:
        print(f"[WARN] Could not save pose files: {e}")

    np.save(os.path.join(out_dir, "gt.npy"), gt_np)
    np.save(os.path.join(out_dir, "pred.npy"), pred_np)
    print("💾 Saved .npy arrays")

    try:
        visualizer = PoseVisualizer(header)
        visualizer.save_animation(pred_pose.body.data, os.path.join(out_dir, "prediction.mp4"))
        print("🎥 Saved animation: prediction.mp4")
    except Exception as e:
        print(f"[WARN] Visualization failed: {e}")

    print(f"\n✅ Finished. Results saved in {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
