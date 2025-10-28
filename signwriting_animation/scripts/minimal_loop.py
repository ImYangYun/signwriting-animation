# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.pose import PoseHeader
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.utils import holistic

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

def to_tjc_anyshape(tensor_like):
    """
    Normalize model/GT sequence to [T, J, C] float32.
    Accepts common shapes we saw in this project.
    """
    x = tensor_like
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    x = np.array(x)

    # Case A: [1,T,J,C]
    if x.ndim == 4 and x.shape[0] == 1 and x.shape[1] < 300 and x.shape[2] > 200:
        # e.g. (1,20,586,3)
        x = x[0]  # -> [T,J,C]

    # Case B: [1,T,1,J,C]
    elif x.ndim == 5 and x.shape[0] == 1 and x.shape[2] == 1:
        # e.g. (1,20,1,586,3)
        x = x[0, :, 0, :, :]  # -> [T,J,C]

    # Case C: [1,J,C,T]
    elif x.ndim == 4 and x.shape[0] == 1 and x.shape[1] > 200 and x.shape[-1] < 300:
        # e.g. (1,586,3,20)
        x = np.transpose(x[0], (2, 0, 1))  # -> [T,J,C]

    # Case D: [T,J,C] already
    elif x.ndim == 3 and x.shape[0] < 300 and x.shape[1] > 200:
        pass

    else:
        raise ValueError(f"[to_tjc_anyshape] Unexpected shape {x.shape}")

    if x.ndim != 3:
        raise ValueError(f"[to_tjc_anyshape] Final shape must be [T,J,C], got {x.shape}")
    return x.astype(np.float32)

def ensure_full_header(header_from_dataset):
    """
    Prefer header from dataset pose file (with limbs),
    else fallback to holistic.holistic_components().
    """
    if header_from_dataset is not None and getattr(header_from_dataset, "components", None):
        print("ℹ Using existing header with components from dataset.")
        return header_from_dataset

    comps = holistic.holistic_components()
    hdr = PoseHeader(components=comps)
    print("✅ Built header via holistic.holistic_components() fallback")
    return hdr

def build_pose_from_sequence(seq_btjc_cpu, header_full, fps=25.0):
    """
    seq_btjc_cpu: torch-ish model output / GT
    header_full: PoseHeader with correct 586-point topology
    returns Pose(...)
    """
    tjc = to_tjc_anyshape(seq_btjc_cpu)  # [T,J,C]
    T, J, C = tjc.shape
    print(f"[build_pose_from_sequence] tjc.shape={tjc.shape} (T,J,C)")

    data_TPJC = tjc[:, np.newaxis, :, :]             # (T,1,J,C)
    confidence = np.ones((T, 1, J), dtype=np.float32)  # (T,1,J)

    body = NumPyPoseBody(
        fps=fps,
        data=data_TPJC,
        confidence=confidence
    )
    return Pose(header_full, body)

def drop_world_component(pose_obj):
    # remove the world "ghost person" component
    return pose_obj.remove_components("POSE_WORLD_LANDMARKS")

def render_pose_video(pose_obj, out_path, title_prefix="SEQ", fps_override=None):
    """
    Uses PoseVisualizer to draw skeleton with limbs from header.
    Keeps coordinates as-is (no extra normalization).
    """
    data_np = pose_obj.body.data.filled(np.nan)  # (T,1,J,3)
    if np.isnan(data_np).all():
        print(f"⚠ All NaN in pose for {title_prefix}, skip rendering.")
        return

    T = data_np.shape[0]

    # get 2D bounds for stable camera
    xy = data_np[..., :2]  # (T,1,J,2)
    x_min = np.nanmin(xy[..., 0]); x_max = np.nanmax(xy[..., 0])
    y_min = np.nanmin(xy[..., 1]); y_max = np.nanmax(xy[..., 1])

    pad_x = (x_max - x_min) * 0.1 + 1e-5
    pad_y = (y_max - y_min) * 0.1 + 1e-5
    x_min -= pad_x; x_max += pad_x
    y_min -= pad_y; y_max += pad_y

    viz = PoseVisualizer(pose_obj)

    fig, ax = plt.subplots(figsize=(5,5))

    def draw_frame(i):
        ax.cla()
        ax.set_aspect("equal","box")
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xticks([]); ax.set_yticks([])
        viz.draw(ax, frame_id=i)
        ax.set_title(f"{title_prefix} t={i}")
        return ax,

    anim = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=T,
        interval=200,
        blit=False
    )

    # pick fps
    fps = fps_override
    if fps is None:
        fps_attr = getattr(pose_obj.body, "fps", None)
        fps = fps_attr if (fps_attr is not None and fps_attr > 0) else 5

    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=int(round(fps)),
                        metadata=dict(artist='pose_format'),
                        bitrate=2400)
        anim.save(out_path, writer=writer)
        print(f"✅ Saved MP4: {out_path}")
    except Exception as e:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        anim.save(gif_path, writer='pillow', fps=int(round(fps)))
        print(f"⚠ ffmpeg failed ({e}), saved GIF instead: {gif_path}")

    plt.close(fig)


def make_loader(data_dir, csv_path, split="train", bs=2, num_workers=2, reduce_holistic=False):
    dataset = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split=split,
        reduce_holistic=reduce_holistic,
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


# ---------- main pipeline ----------
if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_default_dtype(torch.float32)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    batch_size, num_workers = 2, 2

    os.makedirs("logs", exist_ok=True)

    # WARNING: this call is what was blowing memory on the login node.
    # On a compute node it's fine. On login, you may comment everything
    # below this point and just run viz on pre-saved pose files.
    train_loader = make_loader(
        data_dir,
        csv_path,
        split="train",
        bs=batch_size,
        num_workers=num_workers,
        reduce_holistic=False
    )
    val_loader = train_loader

    # grab ONE batch up front, reuse it later so we don't call next(...) twice
    first_batch = next(iter(train_loader))

    print("\n" + "="*60)
    print("[DATA DEBUG]")
    print(f"  data.shape        = {first_batch['data'].shape}")
    print(f"  target_mask.shape = {first_batch['conditions']['target_mask'].shape}")
    print(f"  input_pose.shape  = {first_batch['conditions']['input_pose'].shape}")
    print("="*60 + "\n")

    gt0 = _to_plain_tensor(first_batch["data"][0]).numpy()
    frame_diff = np.abs(gt0[1:] - gt0[:-1]).mean() if gt0.shape[0] > 1 else 0.0
    print(f"[DATA CHECK] mean|ΔGT| = {frame_diff:.6f}")

    # tiny "training"
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

    # inference using the cached batch
    model.eval()
    with torch.no_grad():
        cond  = first_batch["conditions"]

        past_btjc = cond["input_pose"][:1].to(model.device)
        sign_img  = cond["sign_image"][:1].to(model.device)
        fut_gt    = first_batch["data"][:1].to(model.device)

        print("[GEN] Generating future sequence...")
        gen_btjc = model.generate_full_sequence(past_btjc, sign_img, target_len=20)

        gen_btjc_cpu = _as_dense_cpu_btjc(gen_btjc)
        fut_gt_cpu   = _as_dense_cpu_btjc(fut_gt)

        def frame_disp_est(x_btjc):
            x = x_btjc[0]
            if x.dim() >= 3 and x.size(0) > 1:
                return (x[1:, :, :2] - x[:-1, :, :2]).abs().mean().item()
            return 0.0

        print(f"[GEN] mean|Δpred|={frame_disp_est(gen_btjc_cpu):.6f}, "
              f"mean|Δgt|={frame_disp_est(fut_gt_cpu):.6f}")

        try:
            mask_for_eval = torch.ones(1, gen_btjc.size(1), device=gen_btjc.device)
            dtw_val = masked_dtw(gen_btjc, fut_gt.to(gen_btjc.device), mask_for_eval).item()
            print(f"[EVAL] DTW (pred vs GT): {dtw_val:.4f}")
        except Exception as e:
            print(f"[EVAL] DTW failed: {e}")

        # load header once from dataset pose files on disk
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

        header_full = ensure_full_header(header_loaded)

        # Build Pose objects for GT and Pred in pose_format
        gt_pose_full   = build_pose_from_sequence(fut_gt_cpu,  header_full, fps=25.0)
        pred_pose_full = build_pose_from_sequence(gen_btjc_cpu, header_full, fps=25.0)

        # Save .pose files (for sign.mt / offline inspection)
        with open("logs/groundtruth.pose", "wb") as f:
            gt_pose_full.write(f)
        with open("logs/prediction.pose", "wb") as f:
            pred_pose_full.write(f)
        print("✅ wrote logs/groundtruth.pose and logs/prediction.pose")

        # Remove POSE_WORLD_LANDMARKS for cleaner render
        gt_pose_trim   = drop_world_component(gt_pose_full)
        pred_pose_trim = drop_world_component(pred_pose_full)

        print("[TRIM] gt   full joints:",   gt_pose_full.body.data.shape,
              "-> trimmed:", gt_pose_trim.body.data.shape)
        print("[TRIM] pred full joints:", pred_pose_full.body.data.shape,
              "-> trimmed:", pred_pose_trim.body.data.shape)

        # Render MP4/GIF previews with PoseVisualizer
        render_pose_video(gt_pose_trim,   "logs/groundtruth_poseformat.mp4", title_prefix="GT")
        render_pose_video(pred_pose_trim, "logs/prediction_poseformat.mp4", title_prefix="PRED")

    print("(END)")
