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
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc, masked_dtw

# ----------- DEBUG: confirm which lightning_module is actually used -----------
import signwriting_animation.diffusion.lightning_module as LM
print(">>> USING LIGHTNING MODULE FROM:", LM.__file__)
# -------------------------------------------------------------------------------


def _to_plain(x):
    """Convert pose-format tensors to contiguous float32 CPU tensor."""
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().cpu().contiguous().float()


def temporal_smooth(x, k=5):
    """Simple temporal smoothing for visualization."""
    import torch.nn.functional as F
    if x.dim() == 4:
        x = x[0]

    T, J, C = x.shape
    x = x.permute(2,1,0).reshape(1, C*J, T)
    x = F.avg_pool1d(x, kernel_size=k, stride=1, padding=k//2)
    x = x.reshape(C, J, T).permute(2,1,0)
    return x.contiguous()


def rotate_to_upright(x):
    """
    Rotate Mediapipe coordinates to an upright human orientation.
    x: [1,T,J,3] or [T,J,3]
    """

    if x.dim() == 4:
        x = x[0]   # remove batch

    # rotation matrix: old(Z)->new(Y), old(Y)->new(-Z)
    R = torch.tensor([
        [1,  0,   0],   # X stays X
        [0,  0,   1],   # new Y = old Z
        [0, -1,   0],   # new Z = -old Y
    ], dtype=torch.float32, device=x.device)

    # apply rotation: [T,J,3] × [3,3] → [T,J,3]
    x_rot = torch.einsum("tjc,dc->tjd", x, R)
    return x_rot.unsqueeze(0)   # back to BTJC


def visualize_pose(tensor, scale=250.0, offset=(512, 384)):
    """Convert 3D pose → 2D viewer coordinates."""
    if tensor.dim() == 4:
        tensor = tensor[0]

    x = tensor.clone().float()
    center = x.mean(dim=1, keepdim=True)
    x = x - center

    x[..., 1] = -x[..., 1]  # flip Y
    x[..., :2] *= scale
    x[..., 0] += offset[0]
    x[..., 1] += offset[1]
    return x.contiguous()


def visualize_pose_for_viewer(btjc, scale=200.0, w=1024, h=768):
    if btjc.dim() == 4:
        x = btjc[0].clone()
    else:
        x = btjc.clone()

    # ---- 1. rotate +90 deg around Z ----
    R = torch.tensor([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1],
    ], dtype=x.dtype, device=x.device)
    x = torch.matmul(x, R)

    # ---- 2. remove Z component (fix flattening) ----
    x[..., 2] = 0

    # ---- 3. center ----
    x = x - x.mean(dim=1, keepdim=True)

    # ---- 4. flip Y ----
    x[..., 1] = -x[..., 1]

    # ---- 5. scale ----
    x[..., :2] *= scale

    # ---- 6. shift to center ----
    x[..., 0] += w / 2
    x[..., 1] += h / 2

    return x.unsqueeze(0)


def tensor_to_pose(t_btjc, header):
    """Convert tensor → Pose-format object."""
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    elif t_btjc.dim() == 3:
        t = t_btjc
    else:
        raise ValueError

    print("[tensor_to_pose] final shape:", t.shape)

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
    print("std min/max:", stats["std"].min(), stats["std"].max())


    # Dataset + reduction (178 joints)
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

    batch0 = next(iter(loader))
    raw = sanitize_btjc(batch0["data"][0:1]).clone().cpu()

    print("====== RAW DATA STATS ======")
    print("raw.min =", raw.min().item(), " raw.max =", raw.max().item())
    print("raw[0, :10] =", raw[0, :10])
    print("RAW shape:", raw.shape)

    num_joints = batch0["data"].shape[-2]
    num_dims   = batch0["data"].shape[-1]
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

    print("[TRAIN] Overfit 4 samples…")
    trainer.fit(model, loader, loader)

    # Load original header (reduced)
    ref_path = base_ds.records[0]["pose"]
    ref_path = ref_path if os.path.isabs(ref_path) else os.path.join(data_dir, ref_path)
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)

    # ---- correct sequence ----
    ref_p = reduce_holistic(ref_pose)
    ref_p = ref_p.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_p.header

    print("[CHECK HEADER] total joints:", header.total_points())

    print("[CHECK HEADER] total joints:", header.total_points())

    # ============================================================
    # Inference
    # ============================================================

    model.eval()
    device = trainer.strategy.root_device
    model = model.to(device)
    model.mean_pose = model.mean_pose.to(device)
    model.std_pose  = model.std_pose.to(device)

    with torch.no_grad():
        batch = next(iter(loader))
        cond  = batch["conditions"]

        raw_gt = batch["data"][0, 0]
        print("\n====== RAW GT FIRST FRAME (MaskedTensor) ======")
        print(type(raw_gt))

        if hasattr(raw_gt, "zero_filled"):
            dense = raw_gt.zero_filled()
            print("dense[:10] =", dense[:10])
            print("dense min/max =", dense.min(), dense.max())
            print("dense shape =", dense.shape)
        else:
            print("raw_gt[:10] =", raw_gt[:10])


        past = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt   = sanitize_btjc(batch["data"][:1]).to(device)

        future_len = gt.size(1)
        print("[SAMPLE] future_len =", future_len)

        # 1. Generate normalized prediction
        pred_norm = model.sample_autoregressive_fast(
            past_btjc=past,
            sign_img=sign,
            future_len=future_len,
            chunk=1,
        )

        # 2. unnormalize to BTJC
        pred = model.unnormalize(pred_norm)

        # 3. Smoothing (optional)
        #pred_s = temporal_smooth(pred)
        #gt_s   = temporal_smooth(gt)

        # 4. Visualization transform
        #pred_f = visualize_pose(pred, scale=250, offset=(500, 500))
        #gt_f   = visualize_pose(gt,  scale=250, offset=(500, 500))

        gt_f   = visualize_pose_for_viewer(gt)
        pred_f = visualize_pose_for_viewer(pred)

        print("gt_f shape:", gt_f.shape)
        print("pred_f shape:", pred_f.shape)

        # --- DTW evaluation ---
        mask_bt = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred, gt, mask_bt)
        print(f"[DTW] masked_dtw (unnormalized) = {dtw_val:.4f}")

    # ============================================================
    # Save .pose for viewer
    # ============================================================

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
