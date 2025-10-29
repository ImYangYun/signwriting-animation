# -*- coding: utf-8 -*-
"""
Minimal loop with proper pose visualization & pose file export (FluentPose-style)
Author: yayun
"""
import os
import torch
import numpy as np
import lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from torch.utils.data import DataLoader

# pose_format imports
from pose_format import Pose
from pose_format.pose import PoseHeader
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.utils.holistic import holistic_components

# project imports
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, masked_dtw


# ---------------------- utility helpers ----------------------
def _to_plain_tensor(x):
    """convert masked or lightning tensors to plain cpu tensor"""
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().cpu()


def to_tjc_anyshape(x):
    """convert tensor to [T,J,C] np.float32 (accept various shapes)"""
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    x = np.array(x)

    if x.ndim == 5:        # [1,T,1,J,C]
        x = x[0, :, 0, :, :]
    elif x.ndim == 4:
        if x.shape[0] == 1:      # [1,T,J,C]
            x = x[0]
        elif x.shape[-1] < 300:  # [1,J,C,T]
            x = np.transpose(x[0], (2, 0, 1))
    elif x.ndim == 3:
        pass
    else:
        raise ValueError(f"[to_tjc_anyshape] unexpected shape {x.shape}")

    if x.ndim != 3:
        raise ValueError(f"[to_tjc_anyshape] final shape must be [T,J,C], got {x.shape}")
    return x.astype(np.float32)


# ---------------------- pose reconstruction ----------------------
def ensure_header_from_dataset(dataset, fallback=True):
    """优先用 dataset.pose_header，若不存在再 fallback holistic_components()"""
    header = getattr(dataset, "pose_header", None)
    if header is not None:
        print("✅ Using header from dataset (matches joints & limbs)")
        return header
    if fallback:
        hdr = PoseHeader(components=holistic_components())
        print("⚠️ Dataset header missing, using fallback holistic_components()")
        return hdr
    raise ValueError("No header found in dataset and fallback=False")


def build_pose_from_btjc(btjc, header, fps=25.0):
    """convert model/GT tensor to Pose()"""
    tjc = to_tjc_anyshape(btjc)
    T, J, C = tjc.shape
    data = tjc[:, np.newaxis, :, :]
    conf = np.ones((T, 1, J), dtype=np.float32)
    body = NumPyPoseBody(fps=fps, data=data, confidence=conf)
    return Pose(header, body)


def render_pose_video(pose_obj, out_path, title="SEQ"):
    """render pose video with proper bounds and limbs"""
    data = pose_obj.body.data.filled(np.nan)
    if np.isnan(data).all():
        print("⚠ all-NaN pose, skip render")
        return

    xy = data[..., :2]
    x_min, x_max = np.nanmin(xy[..., 0]), np.nanmax(xy[..., 0])
    y_min, y_max = np.nanmin(xy[..., 1]), np.nanmax(xy[..., 1])
    pad_x, pad_y = (x_max - x_min) * 0.1 + 1e-5, (y_max - y_min) * 0.1 + 1e-5

    viz = PoseVisualizer(pose_obj)
    fig, ax = plt.subplots(figsize=(5, 5))

    def draw(i):
        ax.cla()
        ax.set_xlim([x_min - pad_x, x_max + pad_x])
        ax.set_ylim([y_min - pad_y, y_max + pad_y])
        ax.set_aspect("equal", "box")
        ax.axis("off")
        viz.draw(ax, frame_id=i)
        ax.set_title(f"{title} | frame {i}")

    anim = animation.FuncAnimation(fig, draw, frames=data.shape[0], interval=100)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        anim.save(out_path, writer="ffmpeg", fps=int(pose_obj.body.fps))
        print(f"✅ Saved video: {out_path}")
    except Exception as e:
        gif_path = out_path.replace(".mp4", ".gif")
        anim.save(gif_path, writer="pillow", fps=int(pose_obj.body.fps))
        print(f"⚠ ffmpeg failed ({e}), saved GIF instead: {gif_path}")
    plt.close(fig)


def save_pose_file(pose_obj, out_path):
    """save Pose object to .pose file"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pose_obj.write(f)
    print(f"💾 Saved pose file: {out_path}")


# ---------------------- dataloader ----------------------
def make_loader(data_dir, csv_path, split="train", bs=2, num_workers=2):
    dataset = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split=split,
        reduce_holistic=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=zero_pad_collator,
    )
    return loader


# ---------------------- main loop ----------------------
if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_default_dtype(torch.float32)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/minimal_viz"
    os.makedirs(out_dir, exist_ok=True)

    # --- Data ---
    train_loader = make_loader(data_dir, csv_path, split="train", bs=2, num_workers=2)
    batch = next(iter(train_loader))
    print(f"[INFO] batch data shape: {batch['data'].shape}")

    # --- Train tiny model ---
    model = LitMinimal(log_dir=out_dir)
    trainer = pl.Trainer(
        max_steps=500,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        limit_train_batches=10,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model, train_loader, train_loader)

    # --- Inference ---
    model.eval()
    with torch.no_grad():
        cond = batch["conditions"]
        past = cond["input_pose"][:1].to(model.device)
        sign_img = cond["sign_image"][:1].to(model.device)
        gt = batch["data"][:1].to(model.device)

        print("[GEN] generating...")
        pred = model.generate_full_sequence(past, sign_img, target_len=20)

        # --- Evaluation ---
        mask = torch.ones(1, pred.size(1), device=pred.device)
        dtw_val = masked_dtw(pred, gt, mask).item()
        print(f"[EVAL] masked_dtw={dtw_val:.4f}")

        # --- Pose reconstruction & export ---
        header = ensure_header_from_dataset(train_loader.dataset)
        gt_pose = build_pose_from_btjc(gt, header)
        pred_pose = build_pose_from_btjc(pred, header)

        # pose file outputs
        gt_pose_path = os.path.join(out_dir, "groundtruth.pose")
        pred_pose_path = os.path.join(out_dir, "prediction.pose")
        save_pose_file(gt_pose, gt_pose_path)
        save_pose_file(pred_pose, pred_pose_path)

        # video previews
        gt_vid_path = os.path.join(out_dir, "groundtruth.mp4")
        pred_vid_path = os.path.join(out_dir, "prediction.mp4")
        render_pose_video(gt_pose, gt_vid_path, title="Ground Truth")
        render_pose_video(pred_pose, pred_vid_path, title="Prediction")

    print("\n✅ Finished. Pose files and videos saved in:", os.path.abspath(out_dir))
