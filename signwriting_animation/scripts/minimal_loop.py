# -*- coding: utf-8 -*-
import os
import random
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
from pose_format.utils.generic import reduce_holistic
from pose_format.utils import holistic
from pose_format.pose import PoseHeader
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, masked_dtw


def _to_plain_tensor(x):
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().cpu()

def _as_dense_cpu_btjc(x):
    if hasattr(x, "tensor"):
        x = x.tensor
    return x.detach().cpu()

def build_holistic_header():
    """Build a full PoseHeader using holistic components"""
    try:
        components = holistic.holistic_components()
        header = PoseHeader(components=components)
        print(f"✅ Built holistic header with {len(components)} components")
        return header
    except Exception as e:
        print(f"❌ Failed to build header: {e}")
        return None

def ensure_skeleton(header):
    """Ensure header exists; rebuild if None or empty"""
    if header is None:
        print("[ensure_skeleton] No header found — building new holistic header")
        return build_holistic_header()
    if not getattr(header, "components", None):
        print("[ensure_skeleton] header missing components — rebuilding")
        return build_holistic_header()
    print("ℹUsing existing header with components.")
    return header


def save_pose_files(gen_btjc_cpu, gt_btjc_cpu, header):
    """
    Save predicted and ground-truth pose sequences as .pose files (component-safe, NumPyPoseBody).
    Handles multi-component headers and avoids NumPyPoseBody mask mismatch issues.
    """
    import numpy as np
    import numpy.ma as ma
    from pose_format.numpy.pose_body import NumPyPoseBody
    from pose_format import Pose
    import os

    try:
        os.makedirs("logs", exist_ok=True)
        header = ensure_skeleton(header)

        # ---------------------- helper: convert to [T,J,C] ----------------------
        def to_tjc(tensor):
            """Convert [B,T,J,C] → [T,J,C]."""
            x = tensor[0]
            if x.ndim == 4 and x.shape[1] == 1:
                x = x.squeeze(1)
            if x.ndim != 3:
                raise ValueError(f"Unexpected tensor shape {x.shape}")
            if x.shape[-1] == 3:
                pass
            elif x.shape[-2] == 3:
                x = x.permute(2, 0, 1)
            else:
                raise ValueError(f"❌ Unexpected 3D shape {x.shape}: cannot infer coordinate axis.")
            return x.numpy().astype(np.float32)

        gen_np = to_tjc(gen_btjc_cpu)
        gt_np  = to_tjc(gt_btjc_cpu)

        components = header.components
        split_sizes = [len(c.points) for c in components]
        print(f"[DEBUG] header components: {split_sizes}")

        gen_split = np.split(gen_np, np.cumsum(split_sizes)[:-1], axis=1)
        gt_split  = np.split(gt_np,  np.cumsum(split_sizes)[:-1], axis=1)

        class SafeNumPyPoseBody(NumPyPoseBody):
            def __init__(self, fps, data, confidence):
                self.fps = fps
                self.data = data
                self.confidence = confidence

        # ---------------------- build NumPyPoseBody from split ----------------------
        def make_pose_body(header, split_list):
            body_components = []
            for i, (component, data) in enumerate(zip(header.components, split_list)):
                expected_joints = len(component.points)
                arr = data.transpose(0, 2, 1)  # [T,C,J]
                if arr.shape[-1] != expected_joints:
                    fixed = np.zeros((arr.shape[0], arr.shape[1], expected_joints), dtype=arr.dtype)
                    fixed[..., :min(expected_joints, arr.shape[-1])] = arr[..., :min(expected_joints, arr.shape[-1])]
                    arr = fixed
                body_components.append(arr)

            # pad to same joint count
            max_joints = sum(len(c.points) for c in header.components)
            for i in range(len(body_components)):
                if body_components[i].shape[-1] < max_joints:
                    pad = np.zeros((body_components[i].shape[0], body_components[i].shape[1],
                                    max_joints - body_components[i].shape[-1]))
                    body_components[i] = np.concatenate([body_components[i], pad], axis=-1)

            # stack → [T,C,P,J]
            body = np.stack(body_components, axis=2)
            fps = getattr(header, "fps", 25)

            # create empty mask + confidence
            mask = np.zeros_like(body, dtype=bool)
            masked_body = ma.masked_array(body, mask=mask)
            confidence = np.ones((body.shape[0], body.shape[2], body.shape[3]), dtype=np.float32)

            print(f"[POSE SAVE] built NumPyPoseBody: data={body.shape}, confidence={confidence.shape}, fps={fps}")
            return SafeNumPyPoseBody(fps=fps, data=masked_body, confidence=confidence)

        pose_pred = Pose(header, make_pose_body(header, gen_split))
        pose_gt   = Pose(header, make_pose_body(header, gt_split))

        with open("logs/prediction.pose", "wb") as f:
            pose_pred.write(f)
        with open("logs/groundtruth.pose", "wb") as f:
            pose_gt.write(f)

        print("✅ Saved logs/prediction.pose & logs/groundtruth.pose (SafeNumPyPoseBody)")
        return True

    except Exception as e:
        print(f"❌ Failed saving pose files: {e}")
        return False

def save_scatter_backup(seq_btjc, save_path, title="PRED"):
    """Fallback visualization if pose saving fails."""
    if save_path.endswith(".gif"):
        save_path = save_path.replace(".gif", ".png")
    seq = _to_plain_tensor(seq_btjc)[0]
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
        reduce_holistic=reduce_holistic,  # ✅ 透传
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

    train_loader = make_loader(data_dir, csv_path, "train", bs=batch_size, num_workers=num_workers, reduce_holistic=False)
    val_loader = train_loader

    print("\n" + "="*60)
    batch = next(iter(train_loader))
    print("[DATA DEBUG]")
    print(f"  data.shape        = {batch['data'].shape}")
    print(f"  target_mask.shape = {batch['conditions']['target_mask'].shape}")
    print(f"  input_pose.shape  = {batch['conditions']['input_pose'].shape}")
    print("="*60 + "\n")

    # quick GT motion sanity check
    gt = _to_plain_tensor(batch["data"][0]).numpy()
    frame_diff = np.abs(gt[1:] - gt[:-1]).mean()
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

    # --- Generate
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        cond  = batch["conditions"]

        past_btjc = cond["input_pose"][:1].to(model.device)
        sign_img  = cond["sign_image"][:1].to(model.device)
        fut_gt    = batch["data"][:1].to(model.device)

        print("[GEN] Generating full sequence...")
        gen_btjc = model.generate_full_sequence(past_btjc, sign_img, target_len=20)

        gen_btjc_cpu = _as_dense_cpu_btjc(gen_btjc)
        fut_gt_cpu   = _as_dense_cpu_btjc(fut_gt)

        def frame_disp(x_btjc):
            x = x_btjc[0]
            return (x[1:, :, :2] - x[:-1, :, :2]).abs().mean().item() if x.size(0) > 1 else 0.0

        print(f"[GEN] Tf={gen_btjc_cpu.size(1)}, mean|Δpred|={frame_disp(gen_btjc_cpu):.6f}, mean|Δgt|={frame_disp(fut_gt_cpu):.6f}")

        try:
            mask_for_eval = torch.ones(1, gen_btjc.size(1), device=gen_btjc.device)
            dtw_val = masked_dtw(gen_btjc, fut_gt.to(gen_btjc.device), mask_for_eval).item()
            print(f"[EVAL] DTW (pred vs GT): {dtw_val:.4f}")
        except Exception as e:
            print(f"[EVAL] DTW failed: {e}")

        # save and visualize
        header = None
        for root, _, files in os.walk(data_dir):
            for name in files:
                if name.endswith(".pose"):
                    try:
                        with open(os.path.join(root, name), "rb") as f:
                            pose = Pose.read(f)
                            header = pose.header
                            print(f"[HEADER] ✅ Loaded header from {name}")
                            break
                    except Exception:
                        continue
            if header:
                break

        header = ensure_skeleton(header)
        pose_saved = save_pose_files(gen_btjc_cpu, fut_gt_cpu, header)

        if not pose_saved:
            save_scatter_backup(gen_btjc_cpu, "logs/scatter_pred.png", "PRED")
            save_scatter_backup(fut_gt_cpu, "logs/scatter_gt.png", "GT")
