# -*- coding: utf-8 -*-
import os
import glob
import torch
from copy import deepcopy
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc
import sys, atexit
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
atexit.register(lambda: sys.stdout.flush())


np.set_printoptions(suppress=True, linewidth=120, precision=4)
torch.set_printoptions(sci_mode=False, precision=4)

# ---------- one-time flag to avoid repeated "SKIPPED (T=1)" ----------
_PRINTED_T1 = {"future_emb": False, "encoder_out": False}

def _maybe_print_t1(tag, T):
    if T <= 1 and not _PRINTED_T1[tag]:
        print(f"[DBG/model] {tag} time-std=SKIPPED (T=1)")
        _PRINTED_T1[tag] = True

# ---------- torso-aware normalization for visualization ----------
def _find_indices_by_name(header, names):
    name2idx = {}
    base = 0
    for comp in header.components:
        for i, n in enumerate(comp.points):
            name2idx[n] = base + i
        base += len(comp.points)
    idx = [name2idx[n] for n in names if n in name2idx]
    return idx

def tensor_to_pose(t_btjc, header=None, fps=25):
    # t: [T,J,C]
    t = t_btjc[0] if t_btjc.dim() == 4 else t_btjc
    arr = t.detach().cpu().numpy().astype(np.float32)

    center = np.median(arr[:, :, :2].reshape(-1, 2), axis=0)  # [2]
    arr[:, :, :2] -= center

    r = np.sqrt(arr[:, :, 0]**2 + arr[:, :, 1]**2).reshape(-1)
    scale = 120 / (np.percentile(r, 95) + 1e-6)
    arr[:, :, :2] *= scale
    if arr.shape[-1] > 2:
        arr[:, :, 2] *= scale

    arr[:, :, :2] += np.array([150.0, 150.0], dtype=np.float32)[None, None, :]

    arr4 = arr[:, None, :, :]  # [T,1,J,C]
    conf = np.ones((arr4.shape[0], 1, arr4.shape[2], 1), dtype=np.float32)

    body = NumPyPoseBody(fps=fps, data=arr4, confidence=conf)
    return Pose(header=header, body=body)

@torch.no_grad()
def inference_one_frame(model: LitMinimal, past_btjc: torch.Tensor, sign_img: torch.Tensor):
    return model.sample_autoregressive_diffusion(past_btjc=past_btjc, sign_img=sign_img,
                                                 future_len=1, chunk=1)

@torch.no_grad()
def autoregressive_generate(model: LitMinimal, past_btjc: torch.Tensor, sign_img: torch.Tensor, future_len: int):
    # 稳定些：多帧采样用较大的 chunk
    return model.sample_autoregressive_diffusion(past_btjc=past_btjc, sign_img=sign_img,
                                                 future_len=future_len, chunk=5)

# ---------------------- Main ----------------------
if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/overfit_178"
    os.makedirs(out_dir, exist_ok=True)

    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=60,
        num_future_frames=30,
        with_metadata=True,
        split="train",
        reduce_holistic=False,
    )
    small_ds = torch.utils.data.Subset(base_ds, list(range(10)))
    loader = DataLoader(
        small_ds,
        batch_size=2,
        shuffle=True,
        collate_fn=zero_pad_collator,
        num_workers=0,
        pin_memory=False
    )

    batch0 = next(iter(loader))
    B, T, P, J, C = batch0["data"].shape
    print(f"[INFO] Overfit set shape = {B, T, P, J, C}")

    # --------- Model ---------
    model = LitMinimal(
        num_keypoints=J,
        num_dims=C,
        lr=3e-4,
        stats_path=os.path.join(data_dir, "mean_std_586.pt"),
        diffusion_steps=200,
        beta_start=1e-4,
        beta_end=2e-2,
        pred_target="x0",
        guidance_scale=5.0,
    )

    model_cpu = model.eval()
    probe_btjc = sanitize_btjc(batch0["data"][:1])           # [1,T,J,C] on CPU
    std_probe = model_cpu.normalize(probe_btjc).float().std().item()
    status = "OK" if 0.5 <= std_probe <= 2.0 else "MISMATCH"
    print(f"[CHECK] GT_norm.std = {std_probe:.2f} → {status}")
    if status == "MISMATCH":
        factor = max(std_probe, 1e-3)
        with torch.no_grad():
            model_cpu.std_pose *= factor
        std_after = model_cpu.normalize(probe_btjc).float().std().item()
        print(f"[Calib] scaled std_pose by {factor:.3f} → recheck std={std_after:.2f}")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        limit_train_batches=10,
        limit_val_batches=2,
        enable_checkpointing=False,
        deterministic=True,
        enable_progress_bar=False,
        log_every_n_steps=1,
    )
    print("[TRAIN] Start overfit")
    trainer.fit(model, loader, loader)

 # ========================= Evaluation =========================
    print("\n=== Evaluation (586 → 178 reduce) ===")
    model.eval()
    batch = next(iter(loader))
    cond  = batch["conditions"]

    fut_raw  = sanitize_btjc(batch["data"][:1].to(model.device))       # [1, T, 586, 3]
    past_raw = sanitize_btjc(cond["input_pose"][:1].to(model.device))  # [1, 60, 586, 3]
    sign_img = cond["sign_image"][:1].to(model.device)
    T_future = fut_raw.size(1)

    pose_path = base_ds.records[0]["pose"]
    src = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)

    with open(src, "rb") as f:
        pose_raw = Pose.read(f)

    pose_raw = pose_raw.remove_components(["POSE_WORLD_LANDMARKS"])

    # ---------------------- Reduce to 178 ----------------------
    pose_reduced = reduce_holistic(pose_raw)
    header = pose_reduced.header
    index_map = header.reduction_index
    J_reduced = len(index_map)

    print("\n[HEADER-RAW 586]")
    print("components =", [c.name for c in pose_raw.header.components])
    print("joints per component =", [len(c.points) for c in pose_raw.header.components])
    print("total_joints =", sum(len(c.points) for c in pose_raw.header.components))

    print("\n[HEADER-REDUCED 178]")
    print("components =", [c.name for c in header.components])
    print("joints per component =", [len(c.points) for c in header.components])
    print("total_joints =", sum(len(c.points) for c in header.components))

    print("\n[INDEX_MAP] first 40 =", index_map[:40])
    print("[INDEX_MAP] len =", len(index_map))

    fut_reduced  = fut_raw[:, :, index_map, :]
    past_reduced = past_raw[:, :, index_map, :]

    # ---------------------- Save GT ----------------------
    with open(os.path.join(out_dir, "gt_178.pose"), "wb") as f:
        tensor_to_pose(fut_reduced, header).write(f)

    # ---------------------- Predictions ----------------------
    pred_full = inference_one_frame(model, past_raw, sign_img)
    pred_reduced = pred_full[:, :, index_map, :]

    gen_full = autoregressive_generate(model, past_raw, sign_img, T_future)
    gen_reduced = gen_full[:, :, index_map, :]

    with open(os.path.join(out_dir, "gen_178.pose"), "wb") as f:
        tensor_to_pose(gen_reduced, header).write(f)

    print("\n[DEBUG] Prediction saved to gen_178.pose")

    # ---------------------- Summary ----------------------
    def motion_stats(x):
        if x.size(1) <= 1: 
            return (0.0, 0.0)
        d = x[:, 1:] - x[:, :-1]
        return d.abs().mean().item(), d.std().item()

    print("\n=== MOTION SUMMARY ===")
    print("GT   motion:", motion_stats(fut_reduced))
    print("PRED motion:", motion_stats(pred_reduced))
    print("GEN  motion:", motion_stats(gen_reduced))
    print("=== END SUMMARY ===\n")
