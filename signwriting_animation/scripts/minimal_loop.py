# -*- coding: utf-8 -*-
"""
Stable Minimal Loop for 178-joint CAMDM
---------------------------------------
Features:
- tiny subset (16 samples)
- diffusion_steps=200 → fast
- training + validation (Lightning)
- autoregressive sampling
- stable unnormalize
- stable visualization (center + scale)
- saves GT & Pred pose files
"""

import os
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
# Helpers
# ============================================================

def to_plain(x):
    if hasattr(x, "tensor"): x = x.tensor
    if hasattr(x, "zero_filled"): x = x.zero_filled()
    return x.detach().cpu().float().contiguous()


def smooth(x, k=5):
    """Temporal smoothing for predicted pose"""
    import torch.nn.functional as F
    x = x[0]
    T,J,C = x.shape
    x = x.permute(2,1,0).reshape(1,C*J,T)
    x = F.avg_pool1d(x, k, stride=1, padding=k//2)
    x = x.reshape(C,J,T).permute(2,1,0)
    return x.unsqueeze(0)


def recenter_pair(gt, pr):
    """Use GT torso median as reference → scale → recenter"""
    gt = gt[0]
    pr = pr[0]

    gt = torch.nan_to_num(gt)
    pr = torch.nan_to_num(pr)

    torso = gt[:, :33, :2].reshape(-1,2)
    center = torso.median(dim=0).values

    gt[..., :2] -= center
    pr[..., :2] -= center

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


def tensor_to_pose(x, header):
    """[1,T,J,C] → Pose()"""
    x = x[0]
    arr = np.ascontiguousarray(x[:,None,:,:], np.float32)
    conf = np.ones((arr.shape[0],1,arr.shape[2],1), np.float32)
    return Pose(header=header, body=NumPyPoseBody(fps=25, data=arr, confidence=conf))


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    stats_178 = f"{data_dir}/mean_std_178.pt"

    out_dir = "logs/minimal_178_final"
    os.makedirs(out_dir, exist_ok=True)

    BATCH_SIZE = 4
    SUBSET = 16   # fast debug
    EPOCHS = 20   # enough for movement test


    # ========================================================
    # Dataset (178 joints)
    # ========================================================
    def make_loader(split):
        ds = DynamicPosePredictionDataset(
            data_dir=data_dir,
            csv_path=csv_path,
            num_past_frames=60,
            num_future_frames=30,
            with_metadata=True,
            split=split,
            reduce_holistic=True,
        )

        # tiny dataset
        ds = torch.utils.data.Subset(ds, list(range(min(SUBSET, len(ds)))))

        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=(split=="train"),
            num_workers=0,
            collate_fn=zero_pad_collator,
            pin_memory=False,
        )

    train_loader = make_loader("train")
    val_loader   = make_loader("dev")

    print("[INFO] Train samples:", len(train_loader.dataset))
    print("[INFO] Val samples:", len(val_loader.dataset))


    # ========================================================
    # Model (CAMDM)
    # ========================================================
    model = LitMinimal(
        num_keypoints=178,
        num_dims=3,
        lr=1e-4,
        stats_path=stats_178,
        diffusion_steps=200,      # <<< SPEED-UP
        pred_target="x0",
        guidance_scale=0.0,
    )


    # ========================================================
    # Trainer
    # ========================================================
    trainer = pl.Trainer(
        default_root_dir=out_dir,
        accelerator="gpu",
        devices=1,
        max_epochs=EPOCHS,
        log_every_n_steps=10,
        num_sanity_val_steps=0,   # IMPORTANT: avoids infinite sanity loop
    )


    # ========================================================
    # TRAIN
    # ========================================================
    print("===== TRAINING =====")
    trainer.fit(model, train_loader, val_loader)
    print("===== TRAIN DONE =====\n")


    # ========================================================
    # SAMPLE FUTURE
    # ========================================================
    batch = next(iter(val_loader))
    cond = batch["conditions"]

    past = sanitize_btjc(cond["input_pose"][:1]).to(model.device)
    fut  = sanitize_btjc(batch["data"][:1]).to(model.device)
    img  = cond["sign_image"][:1].to(model.device)

    T = fut.size(1)

    pred_norm = model.sample_autoregressive_diffusion(
        past_btjc=past,
        sign_img=img,
        future_len=T,
        chunk=1
    )


    # ========================================================
    # Unnormalize
    # ========================================================
    fut_un  = to_plain(model.unnormalize(fut))
    pred_un = to_plain(model.unnormalize(pred_norm))

    pred_un = torch.clamp(pred_un, -2000, 2000)

    fut_s  = fut_un.unsqueeze(0)
    pred_s = smooth(pred_un)

    fut_vis, pred_vis = recenter_pair(fut_s, pred_s)


    # ========================================================
    # Header
    # ========================================================
    sample_path = batch["records"][0]["pose"]
    with open(os.path.join(data_dir, sample_path), "rb") as f:
        raw_pose = Pose.read(f)
    header = reduce_holistic(raw_pose.remove_components(["POSE_WORLD_LANDMARKS"])).header


    # ========================================================
    # SAVE
    # ========================================================
    out_gt   = f"{out_dir}/gt_178.pose"
    out_pred = f"{out_dir}/pred_178.pose"

    tensor_to_pose(fut_vis, header).write(open(out_gt, "wb"))
    tensor_to_pose(pred_vis, header).write(open(out_pred, "wb"))

    print("[SAVE]  GT  →", out_gt)
    print("[SAVE] PRED →", out_pred)


    # ========================================================
    # Debug info
    # ========================================================
    gt_m = float((fut_un[:,1:]-fut_un[:,:-1]).abs().mean())
    pr_m = float((pred_un[:,1:]-pred_un[:,:-1]).abs().mean())

    print(f"[DEBUG] GT motion:   {gt_m:.4f}")
    print(f"[DEBUG] Pred motion: {pr_m:.4f}")
    print("\n===== DONE =====")
