# -*- coding: utf-8 -*-
import os
import torch
import math
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc


# ============================================================
# Utility
# ============================================================
def _to_plain(x):
    if hasattr(x, "tensor"): x = x.tensor
    if hasattr(x, "zero_filled"): x = x.zero_filled()
    return x.detach().cpu().contiguous().float()


def temporal_smooth(x, k=5):
    import torch.nn.functional as F
    if x.dim() == 4: x = x[0]
    T, J, C = x.shape
    x = x.permute(2,1,0).reshape(1, C*J, T)
    x = F.avg_pool1d(x, kernel_size=k, stride=1, padding=k//2)
    x = x.reshape(C, J, T).permute(2,1,0)
    return x.contiguous()

def view_pose(x, scale=500.0):
    if x.dim() == 4:
        x = x[0]
    y = x.clone()
    y = torch.nan_to_num(y, nan=0.0)

    center = y.mean(dim=1, keepdim=True)
    y = y - center

    y[..., 1] = -y[..., 1]

    rad = math.radians(90)
    R = torch.tensor([
        [math.cos(rad), -math.sin(rad)],
        [math.sin(rad),  math.cos(rad)],
    ], dtype=y.dtype, device=y.device)
    y_xy = y[..., :2]
    y[..., :2] = torch.matmul(y_xy, R.T)

    y[..., :2] *= scale

    min_xy = y[..., :2].min(dim=1)[0].min(dim=0)[0]
    max_xy = y[..., :2].max(dim=1)[0].max(dim=0)[0]
    span = max_xy - min_xy
    canvas_center_x = 600
    canvas_center_y = 400
    offset_x = canvas_center_x - span[0] / 2
    offset_y = canvas_center_y - span[1] / 2

    y[..., 0] += offset_x
    y[..., 1] += offset_y

    return y.contiguous()



def tensor_to_pose(t_btjc, header):
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    elif t_btjc.dim() == 3:
        t = t_btjc
    else:
        raise ValueError(f"unexpected shape {t_btjc.shape}")

    print("[tensor_to_pose] final shape:", t.shape)

    arr = t[:, None, :, :].cpu().numpy().astype(np.float32)
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)

    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    return Pose(header=header, body=body)


# Main
if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/minimal_178"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178.pt"

    # ---------------- Dataset ----------------
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
        reduce_holistic=True,
    )
    base_ds.mean_std = torch.load(stats_path)

    small_ds = torch.utils.data.Subset(base_ds, [0,1,2,3])
    loader = DataLoader(small_ds, batch_size=4, shuffle=True,
                        collate_fn=zero_pad_collator)

    batch0 = next(iter(loader))
    num_joints = batch0["data"].shape[-2]
    num_dims   = batch0["data"].shape[-1]
    print(f"[INFO] joints={num_joints}, dims={num_dims}")

    # ---------------- Model ----------------
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

    print("[TRAIN] Overfit 4 samples…")
    trainer.fit(model, loader, loader)

    # Header from reference pose
    ref_path = base_ds.records[0]["pose"]
    ref_path = ref_path if os.path.isabs(ref_path) else os.path.join(data_dir, ref_path)

    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)

    header = reduce_holistic(ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])).header
    print("[HEADER] components:", [c.name for c in header.components])

    # DEBUG: check mean/std order
    # ==========================
    stats = torch.load(stats_path)
    print("\n[DEBUG] mean shape:", stats["mean"].shape)
    print("[DEBUG] std shape :", stats["std"].shape)

    # also check reduced header joint counts
    print("[DEBUG] reduced header joint counts:",
        [len(c.points) for c in header.components],
        "sum =", sum(len(c.points) for c in header.components))
    print("=====================================\n")

    # Inference
    model.eval()
    device = trainer.strategy.root_device

    model = model.to(device)
    model.mean_pose = model.mean_pose.to(device)
    model.std_pose  = model.std_pose.to(device)

    with torch.no_grad():
        batch = next(iter(loader))
        cond  = batch["conditions"]

        past = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt   = sanitize_btjc(batch["data"][:1]).to(device)

        future_len = gt.size(1)
        print("[SAMPLE] future_len =", future_len)

        # ---- 1) Sample normalized prediction ----
        pred_norm = model.sample_autoregressive_fast(
            past_btjc=past,
            sign_img=sign,
            future_len=future_len,
            chunk=1,
        )

        pred = model.unnormalize(pred_norm)

        pred_s = temporal_smooth(pred)
        gt_s   = temporal_smooth(gt)

        pred_f = view_pose(pred_s)
        gt_f   = view_pose(gt_s)

        print("gt_f shape:", gt_f.shape)
        print("pred_f shape:", pred_f.shape)
        print("gt_f min/max:",   gt_f.min().item(),   gt_f.max().item())
        print("pred_f min/max:", pred_f.min().item(), pred_f.max().item())

    pose_gt = tensor_to_pose(gt_f, header)
    pose_pr = tensor_to_pose(pred_f, header)

    out_gt = os.path.join(out_dir, "gt_178.pose")
    out_pr = os.path.join(out_dir, "pred_178.pose")

    for p in [out_gt, out_pr]:
        if os.path.exists(p):
            os.remove(p)

    with open(out_gt, "wb") as f:
        pose_gt.write(f)
    with open(out_pr, "wb") as f:
        pose_pr.write(f)

    print("[SAVE] GT & Pred pose saved ✔")
