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
    print("\n=== Evaluation (586 -> 178 reduce) ===")
    model.eval()
    batch = next(iter(loader))
    cond  = batch["conditions"]

    fut_raw  = sanitize_btjc(batch["data"][:1].to(model.device))        # [1,Tf,586,3]
    past_raw = sanitize_btjc(cond["input_pose"][:1].to(model.device))   # [1,Tp,586,3]
    sign_img = cond["sign_image"][:1].to(model.device)
    T_future = fut_raw.size(1)

    # ---------------------- Load header (full 586) ----------------------
    pose_path = base_ds.records[0]["pose"]
    src = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)

    from pose_format import Pose
    from pose_format.utils.generic import reduce_holistic

    with open(src, "rb") as f:
        pose_full = Pose.read(f)

    full_header = pose_full.header
    print("\n[DEBUG] Full 586 header:")
    print("components =", [c.name for c in full_header.components])
    print("joints per component =", [len(c.points) for c in full_header.components])
    print("total =", sum(len(c.points) for c in full_header.components))

    # 去掉 world，再做 holistic reduce → 178
    pose_no_world = pose_full.remove_components(["POSE_WORLD_LANDMARKS"])
    pose_reduced  = reduce_holistic(pose_no_world)
    header_178    = pose_reduced.header

    print("\n[DEBUG] Reduced 178 header:")
    print("components =", [c.name for c in header_178.components])
    print("joints per component =", [len(c.points) for c in header_178.components])
    print("total =", sum(len(c.points) for c in header_178.components))

    # ---------------------- build 586→178 index_map ----------------------
    def build_index_map(full_h, red_h):
        name2idx = {}
        base = 0
        for comp in full_h.components:
            for i, pt_name in enumerate(comp.points):
                # 用 (component_name, point_name) 唯一定位
                name2idx[(comp.name, pt_name)] = base + i
            base += len(comp.points)

        idx = []
        for comp in red_h.components:
            for pt_name in comp.points:
                idx.append(name2idx[(comp.name, pt_name)])
        return np.array(idx, dtype=np.int64)

    index_map = build_index_map(full_header, header_178)
    print("\n[INDEX MAP] first 30 =", index_map[:30])
    print("[INFO] J_reduced =", len(index_map))

    # 转成 torch index，方便切片
    idx = torch.as_tensor(index_map, device=fut_raw.device, dtype=torch.long)

    # ---------------------- Apply reduce to sequences ----------------------
    fut_178  = fut_raw.index_select(dim=2, index=idx)    # [1,Tf,178,3]
    past_178 = past_raw.index_select(dim=2, index=idx)   # 只用于统计 / 可视化，不喂给模型

    # ---------------------- Save GT ----------------------
    with open(os.path.join(out_dir, "gt_raw_178.pose"), "wb") as f:
        tensor_to_pose(fut_178, header_178).write(f)
    print("[SAVE] gt_raw_178.pose saved")

    with open(os.path.join(out_dir, "gt_178.pose"), "wb") as f:
        tensor_to_pose(fut_178, header_178).write(f)
    print("[SAVE] gt_178.pose saved")

    # points-only header
    header_points_only = deepcopy(header_178)
    for comp in header_points_only.components:
        comp.limbs = []

    with open(os.path.join(out_dir, "gt_points_only.pose"), "wb") as f:
        tensor_to_pose(fut_178, header_points_only).write(f)
    print("[SAVE] gt_points_only.pose saved")

    # ---------------------- Predictions (模型仍用 586) ----------------------
    # 1-frame
    pred_full = inference_one_frame(model, past_raw, sign_img)   # [1,1,586,3]
    pred_178  = pred_full.index_select(dim=2, index=idx)

    with open(os.path.join(out_dir, "pred_1frame_178.pose"), "wb") as f:
        tensor_to_pose(pred_178, header_178).write(f)
    print("[SAVE] pred_1frame_178.pose saved")

    # autoregressive multi-frame
    print("=== Inference B (autoregressive) ===")
    gen_full = autoregressive_generate(model, past_raw, sign_img, future_len=T_future)  # [1,Tf,586,3]
    gen_178  = gen_full.index_select(dim=2, index=idx)

    with open(os.path.join(out_dir, "gen_178.pose"), "wb") as f:
        tensor_to_pose(gen_178, header_178).write(f)
    print("[SAVE] gen_178.pose saved")

    # ---------------------- Summary ----------------------
    def motion_stats(x):
        if x.size(1) <= 1:
            return (0.0, 0.0)
        d = x[:, 1:] - x[:, :-1]
        return d.abs().mean().item(), d.std().item()

    print("\n=== Motion Summary (178 joints) ===")
    print("GT  motion:",  motion_stats(fut_178))
    print("PRED motion:", motion_stats(pred_178))
    print("GEN motion :", motion_stats(gen_178))
    print("================== END SUMMARY ====================\n")
