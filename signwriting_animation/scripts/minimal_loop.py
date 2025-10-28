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
from pose_format.pose_header import PoseHeaderComponent
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
    x = tensor_like
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    x = np.array(x)

    # [1,T,J,C]
    if x.ndim == 4 and x.shape[0] == 1 and x.shape[2] > 200:
        x = x[0]  # -> [T,J,C]

    # [1,T,1,J,C]
    elif x.ndim == 5 and x.shape[0] == 1 and x.shape[2] == 1:
        x = x[0, :, 0, :, :]  # -> [T,J,C]

    # [1,J,C,T]  (1,586,3,20)
    elif x.ndim == 4 and x.shape[0] == 1 and x.shape[1] > 200 and x.shape[-1] < 100:
        x = np.transpose(x[0], (2,0,1))  # -> [T,J,C]

    # already [T,J,C]
    elif x.ndim == 3 and x.shape[0] < 300 and x.shape[1] > 200:
        pass
    else:
        raise ValueError(f"[to_tjc_anyshape] Unexpected shape {x.shape}")

    if x.ndim != 3:
        raise ValueError(f"[to_tjc_anyshape] Final shape must be [T,J,C], got {x.shape}")

    return x.astype(np.float32)  # [T,J,C]


def ensure_full_header(header_from_dataset):
    """
    We want a header that contains body, face, L/R hands, world... basically holistic.
    1. Prefer header we loaded from dataset (.pose)
    2. Otherwise fallback to holistic.holistic_components()
    """
    if header_from_dataset is not None and getattr(header_from_dataset, "components", None):
        print("ℹ Using existing header with components from dataset.")
        return header_from_dataset

    try:
        comps = holistic.holistic_components()
        hdr = PoseHeader(components=comps)
        print("✅ Built header via holistic.holistic_components()")
        return hdr
    except Exception as e:
        raise RuntimeError(f"Failed to build holistic header: {e}")


def drop_world_component(pose_obj):
    """
    Advisor said: drop POSE_WORLD_LANDMARKS (the 'ghost' person in corner).
    We assume last component is that world body (33 joints).
    We'll build a new Pose with only first N-1 components.
    """
    old_header = pose_obj.header
    comps = old_header.components
    sizes = [len(c.points) for c in comps]

    # Keep all except the last one
    keep_comps = comps[:-1]
    keep_sizes = sizes[:-1]
    keep_J = sum(keep_sizes)

    new_header = PoseHeader(
        version=old_header.version,
        components=keep_comps
    )

    body = pose_obj.body  # NumPyPoseBody or similar
    data_ma   = body.data            # masked array or ndarray (T,P,V,C)
    conf      = body.confidence      # (T,P,V)
    data_trim = data_ma[:, :, :keep_J, :]
    conf_trim = conf[:, :, :keep_J]

    new_body = NumPyPoseBody(
        fps=body.fps,
        data=data_trim,
        confidence=conf_trim
    )
    return Pose(new_header, new_body)


def build_pose_from_sequence(seq_btjc_cpu, header_full, fps=25.0):
    """
    seq_btjc_cpu: output or GT sequence (Torch tensor-ish)
    header_full:  PoseHeader (preferably from dataset)
    returns: Pose(header_full, NumPyPoseBody(...))
    """
    tjc = to_tjc_anyshape(seq_btjc_cpu)  # [T,J,C]
    T, J, C = tjc.shape
    print(f"[build_pose_from_sequence] tjc.shape={tjc.shape} (T,J,C)")

    data_TPJC = tjc[:, np.newaxis, :, :]        # (T,1,J,C)
    confidence = np.ones((T, 1, J), dtype=np.float32)  # (T,1,J)

    body = NumPyPoseBody(
        fps=fps,
        data=data_TPJC,
        confidence=confidence
    )
    return Pose(header_full, body)


def render_pose_video(pose_obj, out_path, title_prefix="SEQ", fps_override=None):
    """
    Use PoseVisualizer to draw the body/face/hands with nice colors, unnormalized.
    We call PoseVisualizer.draw(ax, frame_id=i) per frame and animate via matplotlib.
    """
    viz = PoseVisualizer(pose_obj)
    T = pose_obj.body.data.shape[0]

    # figure global XY limits so camera doesn't jump
    data_np = pose_obj.body.data.filled(np.nan)  # (T,1,J,C)
    xy = data_np[..., :2]                        # (T,1,J,2)
    x_min = np.nanmin(xy[...,0])
    x_max = np.nanmax(xy[...,0])
    y_min = np.nanmin(xy[...,1])
    y_max = np.nanmax(xy[...,1])
    pad_x = (x_max - x_min)*0.1 + 1e-5
    pad_y = (y_max - y_min)*0.1 + 1e-5
    x_min -= pad_x; x_max += pad_x
    y_min -= pad_y; y_max += pad_y

    fig, ax = plt.subplots(figsize=(5,5))

    def init():
        ax.cla()
        ax.set_aspect("equal","box")
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xticks([]); ax.set_yticks([])
        viz.draw(ax, frame_id=0)
        ax.set_title(f"{title_prefix} t=0")
        return ax,

    def update(i):
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
        update,
        init_func=init,
        frames=T,
        interval=200,  # ms between frames (~5fps visual pacing)
        blit=False
    )

    # fps for writer
    fps = fps_override
    if fps is None:
        fps = getattr(pose_obj.body, "fps", None)
        if fps is None or fps <= 0:
            fps = 5

    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=int(round(fps)), metadata=dict(artist='pose_format'), bitrate=2400)
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
        reduce_holistic=reduce_holistic,  # keep FULL holistic
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
    val_loader = train_loader  # quick sanity

    print("\n" + "="*60)
    batch = next(iter(train_loader))
    print("[DATA DEBUG]")
    print(f"  data.shape        = {batch['data'].shape}")
    print(f"  target_mask.shape = {batch['conditions']['target_mask'].shape}")
    print(f"  input_pose.shape  = {batch['conditions']['input_pose'].shape}")
    print("="*60 + "\n")

    gt0 = _to_plain_tensor(batch["data"][0]).numpy()
    frame_diff = np.abs(gt0[1:] - gt0[:-1]).mean() if gt0.shape[0] > 1 else 0.0
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

    # generate one sample
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        cond  = batch["conditions"]

        past_btjc = cond["input_pose"][:1].to(model.device)  # past seq
        sign_img  = cond["sign_image"][:1].to(model.device)  # gloss img
        fut_gt    = batch["data"][:1].to(model.device)       # future GT

        print("[GEN] Generating future sequence...")
        gen_btjc = model.generate_full_sequence(past_btjc, sign_img, target_len=20)

        gen_btjc_cpu = _as_dense_cpu_btjc(gen_btjc)
        fut_gt_cpu   = _as_dense_cpu_btjc(fut_gt)

        def frame_disp_est(x_btjc):
            x = x_btjc[0]
            return (x[1:, :, :2] - x[:-1, :, :2]).abs().mean().item() if (x.dim()>=3 and x.size(0)>1) else 0.0

        print(f"[GEN] mean|Δpred|={frame_disp_est(gen_btjc_cpu):.6f}, "
              f"mean|Δgt|={frame_disp_est(fut_gt_cpu):.6f}")

        try:
            mask_for_eval = torch.ones(1, gen_btjc.size(1), device=gen_btjc.device)
            dtw_val = masked_dtw(gen_btjc, fut_gt.to(gen_btjc.device), mask_for_eval).item()
            print(f"[EVAL] DTW (pred vs GT): {dtw_val:.4f}")
        except Exception as e:
            print(f"[EVAL] DTW failed: {e}")

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

        # ---- build Pose() objects (full holistic) ----
        pred_pose_full = build_pose_from_sequence(gen_btjc_cpu, header_full, fps=25.0)
        gt_pose_full   = build_pose_from_sequence(fut_gt_cpu,  header_full, fps=25.0)

        # ---- drop 'world' component for viz (advisor requested) ----
        pred_pose_trim = drop_world_component(pred_pose_full)
        gt_pose_trim   = drop_world_component(gt_pose_full)

        # ---- write .pose files (full holistic BEFORE drop) for reproducibility ----
        os.makedirs("logs", exist_ok=True)
        with open("logs/prediction.pose", "wb") as f:
            pred_pose_full.write(f)
        with open("logs/groundtruth.pose", "wb") as f:
            gt_pose_full.write(f)
        print("✅ wrote logs/prediction.pose and logs/groundtruth.pose")

        # ---- render mp4/gif using pose_format visualizer (trimmed = no ghost body) ----
        render_pose_video(gt_pose_trim,   "logs/groundtruth_poseformat.mp4", title_prefix="GT")
        render_pose_video(pred_pose_trim, "logs/prediction_poseformat.mp4", title_prefix="PRED")

    print("(END)")
