# -*- coding: utf-8 -*-
import os
import random
import torch
import numpy as np
import lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio  # for GIF export

from torch.utils.data import DataLoader
from pose_format import Pose
from pose_format.utils import holistic
from pose_format.pose import PoseHeader
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, masked_dtw


def _to_plain_tensor(x):
    """Turn MaskedTensor / custom batch tensor into a plain dense CPU torch.Tensor."""
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().cpu()

def _as_dense_cpu_btjc(x):
    """Same idea but do NOT zero_filled if not present; just detach+cpu."""
    if hasattr(x, "tensor"):
        x = x.tensor
    return x.detach().cpu()

def ensure_skeleton(header):
    """
    Make sure we have a PoseHeader with limb connectivity so we can draw stick figures.
    Priority:
      1) use header we loaded from dataset .pose file if it already has components
      2) try holistic.holistic_components()
      3) fallback manual 33-joint body skeleton
    """
    from pose_format.pose_header import PoseHeader, PoseHeaderComponent

    # Case 1: already have a header with components
    if header is not None and getattr(header, "components", None):
        print("ℹ Using existing header with components.")
        return header

    # Case 2: try holistic from pose_format
    try:
        components = holistic.holistic_components()
        header = PoseHeader(components=components)
        print("✅ Built header from holistic.py (with limbs).")
        return header
    except Exception as e:
        print(f"⚠ holistic import failed ({e}), using minimal fallback.")

    # Case 3: fallback manually define a body + simple limbs so we can draw
    components = [
        PoseHeaderComponent(
            name="pose",
            points=[f"p{i}" for i in range(33)],
            limbs=[(11,13),(13,15),(12,14),(14,16),(11,12),
                   (23,24),(23,25),(24,26),(25,27),(26,28),
                   (11,23),(12,24)],
            colors=[(255,0,0)]*12,
            point_format="XYZ",
        ),
        PoseHeaderComponent(name="face", points=[f"f{i}" for i in range(478)], point_format="XYZ"),
        PoseHeaderComponent(name="hand_left", points=[f"lh{i}" for i in range(21)], point_format="XYZ"),
        PoseHeaderComponent(name="hand_right", points=[f"rh{i}" for i in range(21)], point_format="XYZ"),
        PoseHeaderComponent(name="world", points=[f"w{i}" for i in range(33)], point_format="XYZ"),
    ]
    header = PoseHeader(version=0.1, components=components)
    print("✅ Built minimal fallback header with basic limbs.")
    return header


def save_skeleton_frames(seq_tjc, header, prefix="logs/prediction"):
    """
    把一个序列 (T, 33, 3) 画成多张PNG帧:
      logs/prediction_frame_000.png
      logs/prediction_frame_001.png
      ...
    而不是直接拼GIF，避免对 imageio / canvas API 的依赖。

    seq_tjc: [T, 33, 3]
    header: PoseHeader (拿 limbs 来画骨架)
    prefix: 文件前缀（不含 _frame_xxx.png）
    """
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    body_comp = header.components[0]
    limbs = getattr(body_comp, "limbs", [])

    T, J, C = seq_tjc.shape

    # 统一坐标范围，避免每帧缩放不同而抖动
    all_x = seq_tjc[:, :, 0].reshape(-1)
    all_y = seq_tjc[:, :, 1].reshape(-1)

    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())

    pad_x = (x_max - x_min) * 0.1 + 1e-5
    pad_y = (y_max - y_min) * 0.1 + 1e-5
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    for t in range(T):
        xs = seq_tjc[t, :, 0]  # [33]
        ys = seq_tjc[t, :, 1]  # [33]

        fig, ax = plt.subplots(figsize=(4,4))

        # 连骨架线
        for (a, b) in limbs:
            if a < len(xs) and b < len(xs):
                ax.plot([xs[a], xs[b]], [-ys[a], -ys[b]], linewidth=2)

        ax.scatter(xs, -ys, s=10)

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([-y_max, -y_min])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={t}")

        out_path = f"{prefix}_frame_{t:03d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"🖼️ Saved {T} frames to {prefix}_frame_###.png")


def visualize_prediction_vs_gt(gen_btjc_cpu, gt_btjc_cpu, header):
    """
    - 把预测和GT都规整成 [T,33,3]
    - 分别画成一堆PNG帧 (prediction_frame_xxx.png / groundtruth_frame_xxx.png)
    """
    try:
        os.makedirs("logs", exist_ok=True)
        header = ensure_skeleton(header)

        def to_tjc(tensor):
            """
            把各种形状的模型输出 (B,T,J,C) / (B,T,1,J,C) / (B,J,C,T)...
            统一成 [T,J,C] 的 numpy float32
            """
            x = tensor
            if hasattr(x, "tensor"):
                x = x.tensor
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu()

            print(f"[to_tjc] input shape: {x.shape}")

            if x.ndim == 5 and x.shape[2] == 1:
                print("[to_tjc] Detected dummy dimension at axis=2 -> squeeze it")
                x = x.squeeze(2)  # -> [B,T,J,C]

            if x.ndim == 4:
                if x.shape[1] < 200 and x.shape[2] > 200:
                    x = x[0]  # [T,J,C]
                elif x.shape[1] > 200 and x.shape[-1] < 50:
                    x = x[0].permute(2,0,1)  # [T,J,C]
                else:
                    if x.shape[0] == 1 and x.shape[2] > 200:
                        x = x[0]  # [T,J,C]
                    else:
                        raise ValueError(f"Can't infer time axis from 4D {x.shape}")

            elif x.ndim == 3:
                # 如果是 [J,C,T] 这种，把时间轴挪到前面
                if x.shape[0] > 200 and x.shape[-1] <= 50:
                    x = x.permute(2,0,1)  # -> [T,J,C]

            else:
                raise ValueError(f"❌ Unexpected tensor shape {x.shape}")

            x = np.array(x)
            if x.ndim != 3:
                raise ValueError(f"❌ to_tjc failed, got {x.shape}")

            print(f"[to_tjc] output shape: {x.shape}")
            return x.astype(np.float32)

        gen_np = to_tjc(gen_btjc_cpu)  # [T,586,3]
        gt_np  = to_tjc(gt_btjc_cpu)   # [T,586,3]

        min_T = min(gen_np.shape[0], gt_np.shape[0])
        if gen_np.shape[0] != gt_np.shape[0]:
            print(f"⚠️ Length mismatch: trimming to {min_T} frames")
        gen_np, gt_np = gen_np[:min_T], gt_np[:min_T]

        J_POSE = 33
        gen_pose_only = gen_np[:, :J_POSE, :]  # [T,33,3]
        gt_pose_only  = gt_np[:,  :J_POSE, :]  # [T,33,3]

        print(f"[POSE_ONLY] gen_pose_only.shape={gen_pose_only.shape}, gt_pose_only.shape={gt_pose_only.shape}")

        save_skeleton_frames(gen_pose_only, header, prefix="logs/prediction")
        save_skeleton_frames(gt_pose_only,  header, prefix="logs/groundtruth")

        print("✅ Saved skeleton frame sequences under logs/prediction_frame_###.png and logs/groundtruth_frame_###.png")
        return True

    except Exception as e:
        print(f"❌ visualize_prediction_vs_gt failed: {e}")
        return False

def save_scatter_backup(seq_btjc, save_path, title="PRED"):
    """
    Fallback quick plot if visualization totally fails.
    Just scatter some joints for ~20 frames.
    """
    if save_path.endswith(".gif"):
        save_path = save_path.replace(".gif", ".png")
    seq = _to_plain_tensor(seq_btjc)[0]  # first in batch
    T, J, C = seq.shape
    plt.figure(figsize=(5, 5))
    for t in range(0, T, max(1, T // 20)):
        plt.scatter(seq[t, :, 0], -seq[t, :, 1], s=10)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved scatter fallback: {save_path}")


def make_loader(data_dir, csv_path, split="train", bs=2, num_workers=2, reduce_holistic=False):
    dataset = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split=split,
        reduce_holistic=reduce_holistic,  # we keep full holistic joints
    )
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        collate_fn=zero_pad_collator,
        num_workers=num_workers,
        pin_memory=False,
    )
    return loader


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_default_dtype(torch.float32)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    batch_size, num_workers = 2, 2

    train_loader = make_loader(
        data_dir, csv_path,
        split="train",
        bs=batch_size,
        num_workers=num_workers,
        reduce_holistic=False
    )
    # you're reusing train_loader as val_loader; fine for debugging
    val_loader = train_loader

    print("\n" + "="*60)
    batch = next(iter(train_loader))
    print("[DATA DEBUG]")
    print(f"  data.shape        = {batch['data'].shape}")
    print(f"  target_mask.shape = {batch['conditions']['target_mask'].shape}")
    print(f"  input_pose.shape  = {batch['conditions']['input_pose'].shape}")
    print("="*60 + "\n")

    gt0 = _to_plain_tensor(batch["data"][0]).numpy()  # first element in batch
    frame_diff = np.abs(gt0[1:] - gt0[:-1]).mean()
    print(f"[DATA CHECK] mean|ΔGT| = {frame_diff:.6f}")

    model = LitMinimal(log_dir="logs")
    trainer = pl.Trainer(
        max_steps=500,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=5,
        limit_train_batches=10,
        limit_val_batches=5,
        deterministic=True,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_loader, val_loader)

    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        cond  = batch["conditions"]

        past_btjc = cond["input_pose"][:1].to(model.device)   # [1, Tp, J, C] masked-ish
        sign_img  = cond["sign_image"][:1].to(model.device)   # conditioning image(s)
        fut_gt    = batch["data"][:1].to(model.device)        # [1, Tf, J, C] future GT

        print("[GEN] Generating full sequence...")
        gen_btjc = model.generate_full_sequence(past_btjc, sign_img, target_len=20)

        gen_btjc_cpu = _as_dense_cpu_btjc(gen_btjc)
        fut_gt_cpu   = _as_dense_cpu_btjc(fut_gt)

        def frame_disp(x_btjc):
            # x_btjc expected ~ [B,T,J,C] OR [B,J,C,T] after our pipeline; we use [0] and diff in XY
            x = x_btjc[0]
            # guess "time" is dim 0; if time is not 0 we still just use indexing consistent
            # with how we computed mean|ΔGT| above (which also assumed time dim first).
            if x.dim() == 4 and x.shape[1] < 200 and x.shape[2] > 200:
                # [1,T,J,C]? then x is [T,J,C] after x = x_btjc[0], so already handled
                pass
            if x.size(0) > 1:
                return (x[1:, :, :2] - x[:-1, :, :2]).abs().mean().item()
            else:
                return 0.0

        print(f"[GEN] Tf={gen_btjc_cpu.size(1) if gen_btjc_cpu.dim()>=2 else '??'}, "
              f"mean|Δpred|={frame_disp(gen_btjc_cpu):.6f}, "
              f"mean|Δgt|={frame_disp(fut_gt_cpu):.6f}")

        try:
            mask_for_eval = torch.ones(1, gen_btjc.size(1), device=gen_btjc.device)
            dtw_val = masked_dtw(gen_btjc, fut_gt.to(gen_btjc.device), mask_for_eval).item()
            print(f"[EVAL] DTW (pred vs GT): {dtw_val:.4f}")
        except Exception as e:
            print(f"[EVAL] DTW failed: {e}")

        # ------------------
        # Load any header we can find from dataset pose files
        # ------------------
        header = None
        for root, _, files in os.walk(data_dir):
            for name in files:
                if name.endswith(".pose"):
                    try:
                        with open(os.path.join(root, name), "rb") as f:
                            pose_file = Pose.read(f)
                            header = pose_file.header
                            print(f"[HEADER] ✅ Loaded header from {name}")
                            break
                    except Exception:
                        continue
            if header:
                break

        header = ensure_skeleton(header)
        viz_ok = visualize_prediction_vs_gt(gen_btjc_cpu, fut_gt_cpu, header)

        if not viz_ok:
            # absolute fallback: simple scatter
            save_scatter_backup(gen_btjc_cpu, "logs/scatter_pred.png", "PRED")
            save_scatter_backup(fut_gt_cpu,  "logs/scatter_gt.png",   "GT")

