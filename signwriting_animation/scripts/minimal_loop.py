# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import reduce_holistic

from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset

from signwriting_animation.diffusion.lightning_module import LitMinimal
from signwriting_animation.diffusion.lightning_module import sanitize_btjc, masked_dtw


# --------------------------
# Tensor -> Pose
# --------------------------
def tensor_to_pose(t_btjc, header):
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc
    arr = np.ascontiguousarray(t.detach().cpu().numpy()[:, None, :, :], dtype=np.float32)
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)
    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    return Pose(header=header, body=body)


# --------------------------
# Inference (B): 30-frame
# --------------------------
@torch.no_grad()
def inference_full(model, past_btjc, sign_img, T):
    device = model.device
    model.eval()

    past_norm = model.normalize(past_btjc.to(device))
    sign = sign_img.to(device)

    B, Tp, J, C = past_norm.shape
    ts = torch.zeros(B, dtype=torch.long, device=device)

    # full T frames noise
    x_query = torch.randn((B, T, J, C), device=device) * 0.05
    x_query = model.normalize(x_query)

    pred_norm = model.forward(x_query, ts, past_norm[:, -T:], sign)
    return model.unnormalize(pred_norm)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    pl.seed_everything(42)
    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/overfit_178"
    os.makedirs(out_dir, exist_ok=True)

    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=60,
        num_future_frames=30,
        with_metadata=True,
        split="train",
        reduce_holistic=True
    )

    small_ds = torch.utils.data.Subset(base_ds, list(range(4)))
    loader = DataLoader(small_ds, batch_size=4, shuffle=True, collate_fn=zero_pad_collator)

    batch0 = next(iter(loader))
    B,T,P,J,C = batch0["data"].shape
    print(f"[INFO] Overfit set shape = {B,T,P,J,C}")

    # Model
    model = LitMinimal(
        num_keypoints=J,
        num_dims=C,
        lr=1e-4,
        stats_path=os.path.join(data_dir, "mean_std_178.pt")
    )

    # Train
    trainer = pl.Trainer(
        max_epochs=400,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_checkpointing=False,
        deterministic=True,
    )
    print("[TRAIN] Start overfit")
    trainer.fit(model, loader, loader)

    # Evaluation
    batch  = next(iter(loader))
    cond   = batch["conditions"]

    fut_raw  = sanitize_btjc(batch["data"][:1].to(model.device))
    past_raw = sanitize_btjc(cond["input_pose"][:1].to(model.device))
    sign_img = cond["sign_image"][:1].to(model.device)

    T = fut_raw.size(1)

    # Save GT
    gt = model.unnormalize(model.normalize(fut_raw))

    pose_path = base_ds.records[0]["pose"]
    src = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)
    with open(src, "rb") as f:
        ref_pose = Pose.read(f)
    header = reduce_holistic(ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])).header

    gt_pose = tensor_to_pose(gt, header)
    gt_pose.write(open(os.path.join(out_dir, "gt_178.pose"), "wb"))
    print("[SAVE] gt_178.pose saved")

    # Run inference
    gen_un = inference_full(model, past_raw, sign_img, T)

    # Motion check
    vel = gen_un[:,1:] - gen_un[:,:-1]
    print(f"[GEN MOTION] mean Δ = {vel.abs().mean().item():.6f}, std = {vel.std().item():.6f}")

    gen_pose = tensor_to_pose(gen_un, header)
    gen_pose.write(open(os.path.join(out_dir, "gen_178.pose"), "wb"))
    print("[SAVE] gen_178.pose saved")
