# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader
from pose_format import Pose
from pose_format.pose import PoseHeader
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.utils.generic import reduce_holistic

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc


# ============================================================
# Utility
# ============================================================

def _to_plain(x):
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().float().contiguous()


def temporal_smooth(x, k=3):
    import torch.nn.functional as F
    if x.dim() == 4:
        x = x[0]  # [T,J,C]
    T, J, C = x.shape
    x = x.permute(2,1,0).reshape(1, C*J, T)
    x = F.avg_pool1d(x, kernel_size=k, stride=1, padding=k//2)
    x = x.reshape(C, J, T).permute(2,1,0).contiguous()
    return x


def recenter_for_view(x, header):
    if x.dim() == 4:
        x = x[0]

    x = torch.nan_to_num(x, nan=0.0)

    torso_end = len(header.components[0].points)   # typically 8
    torso_xy = x[:, :torso_end, :2]

    ctr = torso_xy.reshape(-1,2).mean(dim=0)

    x[..., :2] -= ctr
    return x


def tensor_to_pose(t_btjc, header):
    t = t_btjc.detach().cpu()
    if t.dim() == 4:
        t = t[0]  # [T,J,C]

    arr = np.ascontiguousarray(t[:, None, :, :], dtype=np.float32)
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)
    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    return Pose(header=header, body=body)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    pl.seed_everything(1234)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/minimal_178"
    os.makedirs(out_dir, exist_ok=True)

    mean_std_path = f"{data_dir}/mean_std_178.pt"

    # ---------------- Dataset ----------------
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
        reduce_holistic=True,          # <-- ⭐ 关键：使用178
    )

    base_ds.mean_std = torch.load(mean_std_path)
    print(f"[NORM] mean_std loaded from {mean_std_path}")

    small_ds = torch.utils.data.Subset(base_ds, list(range(4)))
    loader = DataLoader(small_ds, batch_size=4, shuffle=True,
                        collate_fn=zero_pad_collator)

    batch0 = next(iter(loader))
    raw_shape = batch0["data"].shape
    print("[INFO] batch shape:", raw_shape)

    num_joints = raw_shape[-2]
    num_dims   = raw_shape[-1]

    # ---------------- Model ----------------
    model = LitMinimal(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=mean_std_path,
        lr=1e-4,
    )

    trainer = pl.Trainer(
        max_epochs=300,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_checkpointing=False,
        deterministic=True,
    )

    print("[TRAIN] Start overfitting 4 samples…")
    trainer.fit(model, loader, loader)

    # ---------------- Evaluation ----------------
    print("\n=== EVAL SAMPLE (diffusion sampling) ===")
    model.eval().to(trainer.strategy.root_device)

    with torch.no_grad():
        batch = next(iter(loader))
        cond = batch["conditions"]

        # Prepare inputs
        past = sanitize_btjc(cond["input_pose"][:1]).to(model.device)
        past = model.normalize(past)

        sign = cond["sign_image"][:1].float().to(model.device)
        gt   = sanitize_btjc(batch["data"][:1]).to(model.device)

        fut_len = gt.size(1)
        print(f"[SAMPLE] sampling future length = {fut_len}")

        # ---- sampling normalized ----
        pred_norm = model.sample_autoregressive_fast(
            past_btjc=past,
            sign_img=sign,
            future_len=fut_len,
            chunk=1,
        )
        print("pred_norm min/max:", pred_norm.min().item(), pred_norm.max().item())

        # ---- unnormalize ----
        gt_un   = gt
        pred_un = model.unnormalize(pred_norm)

        # ---- smooth ----
        gt_sm   = temporal_smooth(gt_un)
        pred_sm = temporal_smooth(pred_un)

    # ============================================================
    # 🔥 Build header FIRST — must happen BEFORE recenter + debug
    # ============================================================
    pose_path = base_ds.records[0]["pose"]
    ref_path = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)

    print(f"[REF] load reference pose from: {ref_path}")

    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)

    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    header = reduce_holistic(ref_pose).header

    print("[HEADER] components:", [c.name for c in header.components])
    print("[HEADER] limbs per component:", [len(c.limbs) for c in header.components])


    # ============================================================
    # 🔥 NOW debug (header exists)
    # ============================================================
    print("\n================ GT DEBUG ================")
    print("[1] GT raw shape:", gt.shape)
    print("[2] GT unnormalized shape:", gt_un.shape)
    print("[3] GT smooth shape:", gt_sm.shape)
    print("[4] GT smooth stats:",
        gt_sm.min().item(), gt_sm.max().item(),
        gt_sm.mean().item(), gt_sm.std().item())

    torso_end = len(header.components[0].points)
    print("[5] torso_end =", torso_end)

    torso_xy = gt_sm[:, :torso_end, :2]
    print("[6] torso_xy shape:", torso_xy.shape)
    print("[7] torso_xy min/max:",
        torso_xy.min().item(), torso_xy.max().item())

    # ============================================================
    # 🔥 recenter AFTER header and debug
    # ============================================================
    gt_rc   = recenter_for_view(gt_sm, header)
    pred_rc = recenter_for_view(pred_sm, header)


    # ---------------- Save pose files ----------------
    gt_pose   = tensor_to_pose(gt_rc.cpu(),   header)
    pred_pose = tensor_to_pose(pred_rc.cpu(), header)

    out_gt   = f"{out_dir}/gt_178.pose"
    out_pred = f"{out_dir}/pred_178.pose"

    for pth in [out_gt, out_pred]:
        if os.path.exists(pth):
            os.remove(pth)

    with open(out_gt, "wb") as f:
        gt_pose.write(f)
    with open(out_pred, "wb") as f:
        pred_pose.write(f)

    print(f"[SAVE] wrote: {out_gt}")
    print(f"[SAVE] wrote: {out_pred}")

    try:
        p = Pose.read(open(out_pred, "rb"))
        print("[CHECK] pred pose OK, limbs:", [len(c.limbs) for c in p.header.components])
    except Exception as e:
        print("[ERROR]", e)
