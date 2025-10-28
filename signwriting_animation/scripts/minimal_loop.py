# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import numpy.ma as ma
import lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils import holistic
from pose_format.pose import PoseHeader
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, masked_dtw


def _to_plain_tensor(x):
    """
    Convert possibly-masked / LightningBatch tensors to a plain dense CPU torch.Tensor.
    """
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().cpu()


def _as_dense_cpu_btjc(x):
    """
    Same idea but don't call zero_filled unless provided.
    """
    if hasattr(x, "tensor"):
        x = x.tensor
    return x.detach().cpu()


def ensure_skeleton_header(header):
    """
    Try to return a PoseHeader with all components (body, face, hands...).
    Priority:
    1. If we loaded header from dataset .pose, use it.
    2. Else try holistic.holistic_components()
    3. Else fallback minimal 33-joint body with some limbs.
    """
    from pose_format.pose_header import PoseHeader, PoseHeaderComponent

    if header is not None and getattr(header, "components", None):
        print("ℹ Using existing header with components.")
        return header

    try:
        components = holistic.holistic_components()
        header = PoseHeader(components=components)
        print("✅ Built header from holistic.holistic_components()")
        return header
    except Exception as e:
        print(f"⚠ holistic import failed ({e}), using minimal fallback header")

    # minimal fallback (body only). will NOT give hands/face, but at least draws torso.
    components = [
        PoseHeaderComponent(
            name="pose",
            points=[f"p{i}" for i in range(33)],
            limbs=[(11,13),(13,15),(12,14),(14,16),(11,12),
                   (23,24),(23,25),(24,26),(25,27),(26,28),
                   (11,23),(12,24)],
            colors=[(255,0,0)] * 12,
            point_format="XYZ",
        ),
    ]
    header = PoseHeader(version=0.1, components=components)
    print("✅ Built minimal fallback header with basic limbs ONLY (no face/hands)")
    return header


def to_tjc_anyshape(tensor_like):
    """
    Take model output / GT, which may be shaped:
      [1,T,J,C] or [1,T,1,J,C] or [1,J,C,T]
    and normalize to numpy float32 of shape [T,J,C].
    """
    x = tensor_like
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "detach"):
        x = x.detach().cpu()

    x = np.array(x)

    # Case A: [1, T, J, C]
    if x.ndim == 4 and x.shape[0] == 1 and x.shape[2] > 200:
        x = x[0]  # -> [T,J,C]

    # Case B: [1, T, 1, J, C]
    elif x.ndim == 5 and x.shape[0] == 1 and x.shape[2] == 1:
        x = x[0, :, 0, :, :]  # -> [T,J,C]

    # Case C: [1, J, C, T]  (e.g. 1,586,3,20)
    elif x.ndim == 4 and x.shape[0] == 1 and x.shape[1] > 200 and x.shape[-1] < 100:
        x = np.transpose(x[0], (2, 0, 1))  # -> [T,J,C]

    # Case D: already [T,J,C]
    elif x.ndim == 3 and x.shape[0] < 300 and x.shape[1] > 200:
        pass

    else:
        raise ValueError(f"[to_tjc_anyshape] Unexpected shape {x.shape}")

    if x.ndim != 3:
        raise ValueError(f"[to_tjc_anyshape] Final shape must be [T,J,C], got {x.shape}")

    return x.astype(np.float32)  # [T,J,C]


def build_pose_for_viz(seq_btjc_cpu, header, fps=25.0):
    """
    Turn GT or prediction sequence into a Pose object compatible with pose_format 0.10.5.
    pose_format 0.10.5 expects:
       NumPyPoseBody(fps, data=(T,P,V,C), confidence=(T,P,V))
    where P is number of people (we assume 1).

    seq_btjc_cpu: the raw sequence (torch or masked tensor)
    header: a full PoseHeader (the one loaded from dataset .pose, ideally with 586 joints)
    fps: guessed frame rate (25fps is common in sign datasets)

    returns: Pose(header, NumPyPoseBody)
    """

    # 1. normalize to [T,J,C]
    tjc = to_tjc_anyshape(seq_btjc_cpu)  # [T,J,C]
    T, J, C = tjc.shape
    print(f"[build_pose_for_viz] tjc.shape={tjc.shape} (T,J,C)")

    # 2. expand to [T,P,J,C] with P=1
    data_TPJC = tjc[:, np.newaxis, :, :]    # (T,1,J,C)

    # 3. confidence: shape (T,P,J). all 1s means fully confident
    confidence = np.ones((T, 1, J), dtype=np.float32)

    # 4. Make NumPyPoseBody
    #    According to pose_format 0.10.5 NumPyPoseBody.__init__,
    #    data is allowed to be normal ndarray. It will create mask from confidence (0=>mask True).
    body = NumPyPoseBody(
        fps=fps,
        data=data_TPJC,        # (T,1,J,3)
        confidence=confidence  # (T,1,J)
    )

    pose_obj = Pose(header, body)
    return pose_obj


def save_pose_files_and_frames(gen_btjc_cpu, gt_btjc_cpu, header_full, out_dir="logs", fps=25.0):
    """
    1. Build Pose() objects for prediction and GT with full 586-joint header.
    2. Save them to .pose files.
    3. Use PoseVisualizer to dump per-frame PNGs (GT & pred).
    4. If PoseVisualizer explodes, fallback to simple scatter plots
       for sanity.
    """
    os.makedirs(out_dir, exist_ok=True)

    header_full = ensure_skeleton_header(header_full)

    # Build Pose objects
    pred_pose_obj = build_pose_for_viz(gen_btjc_cpu, header_full, fps=fps)
    gt_pose_obj   = build_pose_for_viz(gt_btjc_cpu,  header_full, fps=fps)

    # Write .pose files (these should be loadable by PoseVisualizer and shareable with advisor)
    pred_pose_path = os.path.join(out_dir, "prediction.pose")
    gt_pose_path   = os.path.join(out_dir, "groundtruth.pose")

    with open(pred_pose_path, "wb") as f:
        pred_pose_obj.write(f)
    with open(gt_pose_path, "wb") as f:
        gt_pose_obj.write(f)

    print(f"✅ Wrote {pred_pose_path} and {gt_pose_path}")

    try:
        pred_pose_for_viz = pred_pose_obj
        gt_pose_for_viz   = gt_pose_obj

        viz_pred = PoseVisualizer(pred_pose_for_viz)
        viz_gt   = PoseVisualizer(gt_pose_for_viz)

        # save per-frame pngs
        viz_pred.save_frames(os.path.join(out_dir, "pred_vis_%03d.png"))
        viz_gt.save_frames(os.path.join(out_dir, "gt_vis_%03d.png"))

        print(f"🎉 Saved visualizer frames to {out_dir}/pred_vis_###.png and {out_dir}/gt_vis_###.png")
        return True

    except Exception as e:
        print(f"⚠ PoseVisualizer failed ({e}). Will fallback to simple skeleton scatter.")
        return False


def save_simple_body_frames(seq_btjc, header_for_limbs, prefix="logs/prediction_simple"):
    """
    Simpler fallback: draw only the first body component (33 joints) as a stick figure.
    This is our old manual renderer.

    seq_btjc: torch tensor like [1,T,J,C] / etc. We'll convert to [T,J,C] then slice first 33 joints.
    header_for_limbs: PoseHeader (ideally has components[0].limbs for the body)
    prefix: output png prefix
    """
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    tjc = to_tjc_anyshape(seq_btjc)  # [T,J,C]
    T, J, C = tjc.shape

    J_POSE = 33
    body_only = tjc[:, :J_POSE, :]  # [T,33,3]

    header_for_limbs = ensure_skeleton_header(header_for_limbs)
    body_comp = header_for_limbs.components[0]
    limbs = getattr(body_comp, "limbs", [])

    all_x = body_only[:, :, 0].reshape(-1)
    all_y = body_only[:, :, 1].reshape(-1)
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    pad_x = (x_max - x_min) * 0.1 + 1e-5
    pad_y = (y_max - y_min) * 0.1 + 1e-5
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    for t in range(T):
        xs = body_only[t, :, 0]
        ys = body_only[t, :, 1]

        fig, ax = plt.subplots(figsize=(4,4))
        # limbs
        for (a,b) in limbs:
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


def save_scatter_backup(seq_btjc, save_path, title="PRED"):
    """
    Ultra-minimal backup: just scatter some joints from ~20 frames.
    """
    if save_path.endswith(".gif"):
        save_path = save_path.replace(".gif", ".png")
    seq = _to_plain_tensor(seq_btjc)[0]  # first in batch
    T, J, C = seq.shape
    plt.figure(figsize=(5,5))
    for t in range(0, T, max(1, T // 20)):
        plt.scatter(seq[t, :, 0], -seq[t, :, 1], s=10)
    plt.title(title)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
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
        reduce_holistic=reduce_holistic,  # keep full holistic joints
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
    val_loader = train_loader  # just reuse for quick sanity

    print("\n" + "="*60)
    batch = next(iter(train_loader))
    print("[DATA DEBUG]")
    print(f"  data.shape        = {batch['data'].shape}")
    print(f"  target_mask.shape = {batch['conditions']['target_mask'].shape}")
    print(f"  input_pose.shape  = {batch['conditions']['input_pose'].shape}")
    print("="*60 + "\n")

    gt0 = _to_plain_tensor(batch["data"][0]).numpy()
    if gt0.shape[0] > 1:
        frame_diff = np.abs(gt0[1:] - gt0[:-1]).mean()
    else:
        frame_diff = 0.0
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

    # inference on one random batch
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        cond  = batch["conditions"]

        past_btjc = cond["input_pose"][:1].to(model.device)  # past motion
        sign_img  = cond["sign_image"][:1].to(model.device)  # sign image cond
        fut_gt    = batch["data"][:1].to(model.device)       # future GT

        print("[GEN] Generating future sequence...")
        gen_btjc = model.generate_full_sequence(past_btjc, sign_img, target_len=20)

        gen_btjc_cpu = _as_dense_cpu_btjc(gen_btjc)
        fut_gt_cpu   = _as_dense_cpu_btjc(fut_gt)

        def frame_disp_est(x_btjc):
            # rough movement metric in XY across time
            x = x_btjc[0]
            # try treat dim0 as T
            if x.dim() >= 3 and x.size(0) > 1:
                return (x[1:, :, :2] - x[:-1, :, :2]).abs().mean().item()
            return 0.0

        print(f"[GEN] mean|Δpred|={frame_disp_est(gen_btjc_cpu):.6f}, "
              f"mean|Δgt|={frame_disp_est(fut_gt_cpu):.6f}")

        # DTW for rough similarity
        try:
            mask_for_eval = torch.ones(1, gen_btjc.size(1), device=gen_btjc.device)
            dtw_val = masked_dtw(gen_btjc, fut_gt.to(gen_btjc.device), mask_for_eval).item()
            print(f"[EVAL] DTW (pred vs GT): {dtw_val:.4f}")
        except Exception as e:
            print(f"[EVAL] DTW failed: {e}")

        ################################
        # Try to load a real header (.pose from dataset)
        ################################
        header_loaded = None
        for root, _, files in os.walk(data_dir):
            for name in files:
                if name.endswith(".pose"):
                    try:
                        with open(os.path.join(root, name), "rb") as f:
                            pose_file = Pose.read(f)
                            header_loaded = pose_file.header
                            print(f"[HEADER] ✅ Loaded header from {name}")
                            break
                    except Exception:
                        continue
            if header_loaded:
                break

        header_loaded = ensure_skeleton_header(header_loaded)

        ok_full_vis = save_pose_files_and_frames(
            gen_btjc_cpu,
            fut_gt_cpu,
            header_loaded,
            out_dir="logs",
            fps=25.0
        )

        if not ok_full_vis:
            save_simple_body_frames(gen_btjc_cpu, header_loaded, prefix="logs/prediction_simple")
            save_simple_body_frames(fut_gt_cpu,  header_loaded, prefix="logs/groundtruth_simple")

        if not ok_full_vis:
            save_scatter_backup(gen_btjc_cpu, "logs/scatter_pred.png", "PRED")
            save_scatter_backup(fut_gt_cpu,  "logs/scatter_gt.png",   "GT")

    print("(END)")
