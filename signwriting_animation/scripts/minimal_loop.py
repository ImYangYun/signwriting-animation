# -*- coding: utf-8 -*-
"""
Final 178-joint minimal loop
Summary:
- Training (Lightning + CAMDM)
- Validation
- Future sampling (autoregressive)
- GT / Pred unnormalize
- Stable visualization
- Debug tools (range, motion, variance)
- Save .pose for gt / pred
"""

import os
import sys
import torch
import numpy as np
import lightning as pl
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

from torch.utils.data import DataLoader
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc


# ============================================================
# Utility: plain tensor
# ============================================================
def _plain(x):
    if hasattr(x, "tensor"): x = x.tensor
    if hasattr(x, "zero_filled"): x = x.zero_filled()
    return x.detach().cpu().contiguous().float()


# ============================================================
# Smooth predicted motion (GT 不平滑)
# ============================================================
def temporal_smooth(x, k=5):
    import torch.nn.functional as F
    if x.dim() == 4:
        x = x[0]
    T,J,C = x.shape
    x = x.permute(2,1,0).reshape(1, C*J, T)
    x = F.avg_pool1d(x, k, stride=1, padding=k//2)
    x = x.reshape(C, J, T).permute(2,1,0).contiguous()
    return x.unsqueeze(0)


# ============================================================
# Recenter both GT & Pred using same GT center
# ============================================================
def recenter_pair(gt, pr):
    if gt.dim() == 4: gt = gt[0]
    if pr.dim() == 4: pr = pr[0]

    gt = torch.nan_to_num(gt)
    pr = torch.nan_to_num(pr)

    # torso区域
    torso = gt[:, :33, :2].reshape(-1,2)
    center = torso.median(dim=0).values  # 使用中位数更鲁棒

    gt[..., :2] -= center
    pr[..., :2] -= center

    # 自适应缩放
    pts = torch.cat([gt[..., :2].reshape(-1,2),
                     pr[..., :2].reshape(-1,2)], dim=0)

    q02 = torch.quantile(pts, 0.02, dim=0)
    q98 = torch.quantile(pts, 0.98, dim=0)
    span = (q98 - q02).clamp(min=50.0)

    scale = min(450.0/span.max(), 5.0)
    gt[..., :2] *= scale
    pr[..., :2] *= scale

    gt[..., 0] += 256
    gt[..., 1] += 256
    pr[..., 0] += 256
    pr[..., 1] += 256

    return gt.unsqueeze(0), pr.unsqueeze(0)


# ============================================================
# Convert to Pose object
# ============================================================
def tensor_to_pose(x_btjc, header):
    if x_btjc.dim() == 4:
        x_btjc = x_btjc[0]
    arr = np.ascontiguousarray(x_btjc[:,None,:,:], dtype=np.float32)
    conf = np.ones((arr.shape[0],1,arr.shape[2],1), dtype=np.float32)
    return Pose(header=header, body=NumPyPoseBody(fps=25, data=arr, confidence=conf))


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    pl.seed_everything(42)
    torch.use_deterministic_algorithms(False)

    # -------------------------------
    # Paths
    # -------------------------------
    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    mean_std_178 = os.path.join(data_dir, "mean_std_178.pt")

    out_dir = "logs/minimal_178_final"
    os.makedirs(out_dir, exist_ok=True)

    BATCH_SIZE = 8
    MAX_EPOCHS = 50

    # -------------------------------
    # DataLoader
    # -------------------------------
    def make_loader(split, subset_size=16):     # 默认做一个16样本tiny dataset
        ds = DynamicPosePredictionDataset(
            data_dir=data_dir,
            csv_path=csv_path,
            num_past_frames=60,
            num_future_frames=30,
            with_metadata=True,
            split=split,
            reduce_holistic=True,
        )

        # ---- NEW: tiny subset ----
        if subset_size is not None:
            idx = list(range(min(subset_size, len(ds))))
            ds = torch.utils.data.Subset(ds, idx)

        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=(split=="train"),
            num_workers=0,
            collate_fn=zero_pad_collator,
            pin_memory=False,
            persistent_workers=False,
        )

    train_loader = make_loader("train", subset_size=16)
    val_loader   = make_loader("dev",   subset_size=16)

    print("[INFO] Train samples:", len(train_loader.dataset))
    print("[INFO] Val samples:", len(val_loader.dataset))


    # -------------------------------
    # Model: 178 joints CAMDM
    # -------------------------------
    model = LitMinimal(
        num_keypoints=178,
        num_dims=3,
        lr=1e-4,
        stats_path=mean_std_178,
        diffusion_steps=1000,
        pred_target="x0",
        guidance_scale=0.0,
    )


    # -------------------------------
    # Trainer
    # -------------------------------
    trainer = pl.Trainer(
        default_root_dir=out_dir,
        accelerator="gpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=10,
        accumulate_grad_batches=2,   # 💙 更稳定
        enable_checkpointing=True,
    )


    # -------------------------------
    # TRAIN
    # -------------------------------
    print("\n===== TRAINING =====")
    trainer.fit(model, train_loader, val_loader)
    print("===== TRAIN DONE =====\n")


    # -------------------------------
    # SAMPLE
    # -------------------------------
    print("===== SAMPLING FUTURE FRAMES =====")

    batch = next(iter(val_loader))
    cond = batch["conditions"]

    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(model.device)
    fut_raw  = sanitize_btjc(batch["data"][:1]).to(model.device)
    sign_img = cond["sign_image"][:1].to(model.device)

    # mask-based future length
    if "target_mask" in cond:
        mask = cond["target_mask"][:1]
        if mask.dim() == 4:
            mask = (mask.sum(dim=(2,3)) > 0).float()
        true_len = int(mask.sum().item())
    else:
        true_len = fut_raw.size(1)

    pred_norm = model.sample_autoregressive_diffusion(
        past_btjc=past_raw,
        sign_img=sign_img,
        future_len=true_len,
        chunk=1
    )

    # -------------------------------
    # Unnormalize (统一方式)
    # -------------------------------
    fut_un  = model.unnormalize(fut_raw)
    pred_un = model.unnormalize(pred_norm)

    fut_un  = _plain(fut_un)
    pred_un = _plain(pred_un)

    # -------------------------------
    # Visualization Stabilization
    # -------------------------------
    # 自适应 clamp
    p95 = torch.quantile(pred_un.abs(), 0.95)
    pred_un = torch.clamp(pred_un, -p95*2, p95*2)

    pred_s = temporal_smooth(pred_un)   # pred 平滑
    fut_s  = fut_un.unsqueeze(0)

    fut_vis, pred_vis = recenter_pair(fut_s, pred_s)


    # -------------------------------
    # Header caching
    # -------------------------------
    header_cache = os.path.join(out_dir, "header_178.pt")
    if not os.path.exists(header_cache):
        pose_path = batch["records"][0]["pose"]
        with open(os.path.join(data_dir, pose_path), "rb") as f:
            pose0 = Pose.read(f)
        header = reduce_holistic(pose0.remove_components(["POSE_WORLD_LANDMARKS"])).header
        torch.save(header, header_cache)
        print("[HEADER] Saved header_178.pt")
    else:
        header = torch.load(header_cache)


    # -------------------------------
    # Save pose files
    # -------------------------------
    out_pred = os.path.join(out_dir, "pred_178.pose")
    out_gt   = os.path.join(out_dir, "gt_178.pose")

    tensor_to_pose(fut_vis, header).write(open(out_gt, "wb"))
    tensor_to_pose(pred_vis, header).write(open(out_pred, "wb"))

    print("[SAVE] GT:", out_gt)
    print("[SAVE] Pred:", out_pred)


    # -------------------------------
    # Debug motion info
    # -------------------------------
    gt_m  = float((fut_un[:,1:]-fut_un[:,:-1]).abs().mean())
    pr_m  = float((pred_un[:,1:]-pred_un[:,:-1]).abs().mean())
    print(f"[DEBUG] GT motion avg:   {gt_m:.4f}")
    print(f"[DEBUG] Pred motion avg: {pr_m:.4f}")

    print("\n===== DONE =====")
