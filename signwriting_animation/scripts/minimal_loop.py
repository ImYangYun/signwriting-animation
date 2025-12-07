# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc, masked_dtw

import signwriting_animation.diffusion.lightning_module as LM
print(">>> USING LIGHTNING MODULE FROM:", LM.__file__)


def tensor_to_pose(t_btjc, header):
    """Convert tensor → Pose-format object (no transformation)."""
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    elif t_btjc.dim() == 3:
        t = t_btjc
    else:
        raise ValueError("Expected 3D or 4D tensor")

    print(f"[tensor_to_pose] input shape: {t.shape}")

    arr = t[:, None, :, :].cpu().numpy().astype(np.float32)
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)

    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    return Pose(header=header, body=body)


if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    out_dir = "logs/minimal_178"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178.pt"
    stats = torch.load(stats_path)
    print("mean shape:", stats["mean"].shape)
    print("std shape:", stats["std"].shape)

    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    base_ds.mean_std = torch.load(stats_path)

    small_ds = torch.utils.data.Subset(base_ds, [0, 1, 2, 3])
    loader = DataLoader(
        small_ds,
        batch_size=4,
        shuffle=True,
        collate_fn=zero_pad_collator,
    )

    num_joints = base_ds[0]["data"].shape[-2]
    num_dims = base_ds[0]["data"].shape[-1]
    print(f"[INFO] joints={num_joints}, dims={num_dims}")

    # Model
    model = LitMinimal(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-4,
    )

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_checkpointing=False,
        deterministic=True,
    )

    print("[TRAIN] Overfit 4 samples...")
    trainer.fit(model, loader, loader)

    # ============================================================
    # Load header from original pose file
    # ============================================================
    ref_path = base_ds.records[0]["pose"]
    ref_path = ref_path if os.path.isabs(ref_path) else os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)

    # Reduce to 178 joints (same as dataset)
    ref_reduced = reduce_holistic(ref_pose)
    ref_reduced = ref_reduced.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_reduced.header

    print(f"[HEADER] total joints: {header.total_points()}")

    # Save original reference pose for comparison
    out_original = os.path.join(out_dir, "original_ref.pose")
    if os.path.exists(out_original):
        os.remove(out_original)
    with open(out_original, "wb") as f:
        ref_reduced.write(f)
    print(f"[SAVE] Original reference pose saved to: {out_original}")

    # ============================================================
    # Inference
    # ============================================================
    model.eval()
    device = trainer.strategy.root_device
    model = model.to(device)
    model.mean_pose = model.mean_pose.to(device)
    model.std_pose = model.std_pose.to(device)

    with torch.no_grad():
        batch = next(iter(loader))
        cond = batch["conditions"]

        past = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt = sanitize_btjc(batch["data"][:1]).to(device)

        future_len = gt.size(1)
        print(f"[INFERENCE] future_len = {future_len}")

        # Generate prediction
        pred_norm = model.sample_autoregressive_fast(
            past_btjc=past,
            sign_img=sign,
            future_len=future_len,
            chunk=1,
        )

        # Unnormalize
        pred = model.unnormalize(pred_norm)

        print(f"\n[DATA CHECK]")
        print(f"GT shape: {gt.shape}")
        print(f"GT range: X[{gt[...,0].min():.4f}, {gt[...,0].max():.4f}], "
              f"Y[{gt[...,1].min():.4f}, {gt[...,1].max():.4f}], "
              f"Z[{gt[...,2].min():.4f}, {gt[...,2].max():.4f}]")
        print(f"PRED range: X[{pred[...,0].min():.4f}, {pred[...,0].max():.4f}], "
              f"Y[{pred[...,1].min():.4f}, {pred[...,1].max():.4f}], "
              f"Z[{pred[...,2].min():.4f}, {pred[...,2].max():.4f}]")

        # DTW evaluation
        mask_bt = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred, gt, mask_bt)
        print(f"[DTW] masked_dtw = {dtw_val:.4f}")

    # ============================================================
    # Save poses (NO transformation, direct conversion)
    # ============================================================
    
    # Convert to CPU for saving
    gt_cpu = gt.cpu()
    pred_cpu = pred.cpu()

    # Create pose objects
    pose_gt = tensor_to_pose(gt_cpu, header)
    pose_pred = tensor_to_pose(pred_cpu, header)

    # Save files
    out_gt = os.path.join(out_dir, "gt_178.pose")
    out_pred = os.path.join(out_dir, "pred_178.pose")

    for path in [out_gt, out_pred]:
        if os.path.exists(path):
            os.remove(path)

    with open(out_gt, "wb") as f:
        pose_gt.write(f)
    with open(out_pred, "wb") as f:
        pose_pred.write(f)

    print(f"\n[SAVE] Files saved:")
    print(f"  - GT:   {out_gt}")
    print(f"  - PRED: {out_pred}")
    print(f"  - REF:  {out_original}")
    print(f"\nOpen these files in pose viewer to compare!")