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

def ensure_skeleton(header):
    """Ensure PoseHeader exists."""
    from pose_format.pose_header import PoseHeader, PoseHeaderComponent

    if header is not None and getattr(header, "components", None):
        print("ℹ Using existing header with components.")
        return header

    try:
        from pose_format.utils import holistic
        components = holistic.holistic_components()
        header = PoseHeader(components=components)
        print("✅ Built header from holistic.py (with limbs).")
        return header
    except Exception as e:
        print(f"⚠ holistic import failed ({e}), using minimal fallback.")

    components = [
        PoseHeaderComponent(
            name="pose",
            points=[f"p{i}" for i in range(33)],
            limbs=[(11,13),(13,15),(12,14),(14,16),(11,12),
                   (23,24),(23,25),(24,26),(25,27),(26,28),(11,23),(12,24)],
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


def save_pose_files(gen_btjc_cpu, gt_btjc_cpu, header):
    """
    Save predicted and ground-truth pose sequences as .pose files.
    For now we ONLY save the 'pose' (first 33 joints) component so that
    all arrays have consistent shape and PoseVisualizer can draw limbs.
    """

    try:
        os.makedirs("logs", exist_ok=True)
        header = ensure_skeleton(header)

        def to_tjc(tensor):
            x = tensor
            if hasattr(x, "tensor"):  # MaskedTensor
                x = x.tensor
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu()

            print(f"[to_tjc] input shape: {x.shape}")

            # Case 5D: [B,T,1,J,C]
            if x.ndim == 5 and x.shape[2] == 1:
                print("[to_tjc] Detected dummy dimension at axis=2 -> squeeze it")
                x = x.squeeze(2)  # -> [B,T,J,C]

            # Case 4D
            if x.ndim == 4:
                # [B,T,J,C]
                if x.shape[1] < 200 and x.shape[2] > 200:
                    x = x[0]  # -> [T,J,C]
                # [B,J,C,T]
                elif x.shape[1] > 200 and x.shape[-1] < 50:
                    x = x[0].permute(2,0,1)  # -> [T,J,C]
                else:
                    # could still be [B,T,J,C] after squeeze
                    if x.shape[0] == 1 and x.shape[2] > 200:
                        x = x[0]  # -> [T,J,C]
                    else:
                        raise ValueError(f"Can't infer time axis from 4D {x.shape}")

            # Case 3D
            elif x.ndim == 3:
                # [J,C,T] -> [T,J,C]
                if x.shape[0] > 200 and x.shape[-1] <= 50:
                    x = x.permute(2,0,1)

            else:
                raise ValueError(f"❌ Unexpected tensor shape {x.shape}")

            x = np.array(x)
            if x.ndim != 3:
                raise ValueError(f"❌ to_tjc failed, got {x.shape}")

            print(f"[to_tjc] output shape: {x.shape}")
            return x.astype(np.float32)

        # convert pred / gt
        gen_np = to_tjc(gen_btjc_cpu)  # [T,586,3]
        gt_np  = to_tjc(gt_btjc_cpu)   # [T,586,3]

        # frame align (T might differ, trim to shortest)
        min_T = min(gen_np.shape[0], gt_np.shape[0])
        if gen_np.shape[0] != gt_np.shape[0]:
            print(f"⚠️ Length mismatch: trimming to {min_T} frames")
        gen_np, gt_np = gen_np[:min_T], gt_np[:min_T]

        # ----------------------
        # now: only take first 33 joints (body/pose)
        # ----------------------
        J_POSE = 33
        gen_pose_only = gen_np[:, :J_POSE, :]  # [T,33,3]
        gt_pose_only  = gt_np[:,  :J_POSE, :]  # [T,33,3]

        print(f"[POSE-ONLY] gen_pose_only.shape={gen_pose_only.shape}, gt_pose_only.shape={gt_pose_only.shape}")

        # build a mini header that only has the first component ("pose")
        # (the header we loaded may have multiple components, but for saving we just
        #  keep the first one which should correspond to body with limbs)
        mini_header = None
        if hasattr(header, "components") and len(header.components) > 0:
            from pose_format.pose_header import PoseHeader, PoseHeaderComponent
            pose_comp = header.components[0]  # assume first is body/pose (33 joints)
            mini_header = PoseHeader(version=getattr(header,"version",0.1),
                                     components=[pose_comp])
        else:
            # fallback: rebuild minimal one
            mini_header = ensure_skeleton(None)
            mini_header.components = [mini_header.components[0]]

        # Now wrap into NumPyPoseBody format:
        # We need [T,C,P,J] masked array with confidence. For single component:
        #   T = time
        #   C = coords (3)
        #   P = 1 component
        #   J = joints (33)
        def build_single_body(seq_tjc):
            # seq_tjc: [T,33,3] -> [T,3,1,33]
            data_tcj = np.transpose(seq_tjc, (0,2,1))           # [T,3,33]
            data_tcpj = data_tcj[:, :, np.newaxis, :]           # [T,3,1,33]
            mask = np.zeros_like(data_tcpj, dtype=bool)
            masked_body = ma.masked_array(data_tcpj, mask=mask)
            confidence = np.ones((data_tcpj.shape[0], 1, data_tcpj.shape[3]), dtype=np.float32)
            fps = getattr(header, "fps", 25)
            return NumPyPoseBody(fps=fps, data=masked_body, confidence=confidence)

        pose_pred = Pose(mini_header, build_single_body(gen_pose_only))
        pose_gt   = Pose(mini_header, build_single_body(gt_pose_only))

        os.makedirs("logs", exist_ok=True)
        with open("logs/prediction.pose", "wb") as f:
            pose_pred.write(f)
        with open("logs/groundtruth.pose", "wb") as f:
            pose_gt.write(f)

        print("✅ Saved logs/prediction.pose & logs/groundtruth.pose (POSE-ONLY)")
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
