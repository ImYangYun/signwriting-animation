# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader
import atexit

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc


# ========================================================================
# Safe IO Handling
# ========================================================================
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

atexit.register(lambda: sys.stdout.flush())

np.set_printoptions(suppress=True, linewidth=120, precision=4)
torch.set_printoptions(sci_mode=False, precision=4)


# ========================================================================
# Utility Functions (from your old working pipeline)
# ========================================================================
def unnormalize_tensor_with_global_stats(tensor: torch.Tensor, mean_std: dict):
    """
    Unnormalize: x * std + mean
    """
    if torch.is_tensor(mean_std["mean"]):
        mean = mean_std["mean"].detach().float().to(tensor.device)
        std = mean_std["std"].detach().float().to(tensor.device)
    else:
        mean = torch.tensor(mean_std["mean"], dtype=torch.float32, device=tensor.device)
        std = torch.tensor(mean_std["std"], dtype=torch.float32, device=tensor.device)
    return tensor * std + mean


def _unwrap_mean_std(ms):
    if isinstance(ms, dict):
        return ms
    elif hasattr(ms, "mean") and hasattr(ms, "std"):
        return {"mean": ms.mean, "std": ms.std}
    elif isinstance(ms, (list, tuple)) and len(ms) == 2:
        return {"mean": ms[0], "std": ms[1]}
    else:
        raise ValueError(f"Unsupported mean_std format: {type(ms)}")


def temporal_smooth(x, k=5):
    """Apply avg_pool smoothing along time axis."""
    import torch.nn.functional as F
    if x.dim() == 5 and x.shape[2] == 1:
        x = x.squeeze(2)
    if x.dim() == 4:
        x = x[0]

    T, J, C = x.shape
    x2 = x.permute(2, 1, 0).reshape(1, C*J, T)
    x2 = F.avg_pool1d(x2, kernel_size=k, stride=1, padding=k // 2)
    x2 = x2.reshape(C, J, T).permute(2, 1, 0)
    return x2.contiguous()


def recenter_for_view(x, offset=(0, 0, 0)):
    """Your old version — the one that correctly re-centers poses."""
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    if x.dim() == 5 and x.shape[2] == 1:
        x = x.squeeze(2)
    if x.dim() == 4:
        x = x.mean(dim=0)

    x = torch.nan_to_num(x, nan=0.0)

    T, J, C = x.shape
    torso_end = min(33, J)
    torso_xy = x[:, :torso_end, :2]

    mask = torch.isfinite(torso_xy)
    valid_xy = torso_xy[mask].view(-1, 2) if mask.any() else torch.zeros(1, 2)
    mean_center = valid_xy.mean(dim=0, keepdim=True)
    median_center = valid_xy.median(dim=0).values.unsqueeze(0)
    center = 0.5 * (mean_center + median_center)

    x[..., :2] -= center

    flat_xy = x[..., :2].reshape(-1, 2)
    q02 = torch.quantile(flat_xy, 0.02, dim=0)
    q98 = torch.quantile(flat_xy, 0.98, dim=0)
    span = (q98 - q02).clamp(min=1.0)

    offset_x = 512 - span[0] / 2
    offset_y = 384 - span[1] / 2
    x[..., 0] += offset_x + offset[0]
    x[..., 1] += offset_y + offset[1]

    return x.contiguous().float()


def tensor_to_pose(t_btjc, header):
    """Convert [T,J,C] or [1,T,J,C] to Pose object."""
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc

    arr = np.ascontiguousarray(t[:, None, :, :], dtype=np.float32)
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)
    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    return Pose(header=header, body=body)


# ========================================================================
# Autoregressive helpers
# ========================================================================
@torch.no_grad()
def inference_one_frame(model, past, sign):
    return model.sample_autoregressive_diffusion(
        past_btjc=past, sign_img=sign, future_len=1, chunk=1
    )


@torch.no_grad()
def autoregressive_generate(model, past, sign, future_len):
    return model.sample_autoregressive_diffusion(
        past_btjc=past, sign_img=sign, future_len=future_len, chunk=5
    )


# ========================================================================
# Main
# ========================================================================
if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/overfit_178vis"
    os.makedirs(out_dir, exist_ok=True)

    # ================== DATASET (586 joints) ==================
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
    )

    batch0 = next(iter(loader))
    B, T, P, J, C = batch0["data"].shape
    print(f"[INFO] 586-joint shape → B={B}, T={T}, J={J}, C={C}")

    # ================== MODEL ==================
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
    model.eval()

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        deterministic=True,
        limit_train_batches=10,
        limit_val_batches=2,
        enable_progress_bar=False,
    )

    print("\n[TRAIN] Overfitting on subset...")
    #trainer.fit(model, loader, loader)

    # ========================================================================
    # Build 178-joint header (reduce_holistic)
    # ========================================================================
    print("\n[VIS] Building 178-joint skeleton...")

    pose_path = base_ds.records[0]["pose"]
    pose_abspath = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)

    with open(pose_abspath, "rb") as f:
        pose_full = Pose.read(f)

    pose_now = pose_full.remove_components(["POSE_WORLD_LANDMARKS"])
    pose_178 = reduce_holistic(pose_now)
    header_178 = pose_178.header

    print("[OK] 178 header ready")

    # ===== Build stable index_map (component_name, point_name) =====
    name2idx586 = {}
    base = 0
    for comp in pose_full.header.components:
        for p in comp.points:
            name2idx586[(comp.name, p)] = base
            base += 1

    index_map = []
    for comp in header_178.components:
        for p in comp.points:
            index_map.append(name2idx586[(comp.name, p)])

    idx_t = torch.tensor(index_map, dtype=torch.long, device=model.device)
    print("[OK] index_map length =", len(index_map))

    # ========================================================================
    # Inference
    # ========================================================================
    batch = next(iter(loader))
    cond = batch["conditions"]

    fut_raw  = sanitize_btjc(batch["data"][:1].to(model.device))
    past_raw = sanitize_btjc(cond["input_pose"][:1].to(model.device))
    sign_img = cond["sign_image"][:1].to(model.device)
    T_fut = fut_raw.size(1)

    # ========================================================================
    # ==== FULL VISUAL PIPELINE (your old working version) ====
    # ========================================================================

    # ----- 1) unnormalize -----
    mean_std = torch.load(os.path.join(data_dir, "mean_std_586.pt"))
    mean_std = _unwrap_mean_std(mean_std)

    fut_un  = unnormalize_tensor_with_global_stats(fut_raw, mean_std)
    pred_full = inference_one_frame(model, past_raw, sign_img)
    pred_un = unnormalize_tensor_with_global_stats(pred_full, mean_std)
    gen_full = autoregressive_generate(model, past_raw, sign_img, future_len=T_fut)
    gen_un   = unnormalize_tensor_with_global_stats(gen_full, mean_std)

    # ----- 2) clamp -----
    fut_un  = torch.clamp(fut_un,  -3,  3)
    pred_un = torch.clamp(pred_un, -3,  3)
    gen_un  = torch.clamp(gen_un,  -3,  3)

    # ----- 3) smooth -----
    fut_un  = temporal_smooth(fut_un)
    pred_un = temporal_smooth(pred_un)
    gen_un  = temporal_smooth(gen_un)

    # ----- 4) recenter -----
    fut_vis  = recenter_for_view(fut_un)
    pred_vis = recenter_for_view(pred_un)
    gen_vis  = recenter_for_view(gen_un)

    # ----- 5) reduce to 178 joints -----
    fut_178  = fut_vis.index_select(2, idx_t)
    pred_178 = pred_vis.index_select(2, idx_t)
    gen_178  = gen_vis.index_select(2, idx_t)

    # ================================
    # DEBUG SHAPES —— 必须加！！！ 
    # ================================
    def dbg(name, x):
        try:
            print(f"[DEBUG] {name} shape =", tuple(x.shape))
        except:
            print(f"[DEBUG] {name} shape = <unprintable>")

    dbg("fut_raw", fut_raw)
    dbg("fut_178", fut_178)

    dbg("pred_full", pred_full)
    dbg("pred_178", pred_178)

    dbg("gen_full", gen_full)
    dbg("gen_178", gen_178)

    # 如果你有 recenter_for_view，那也把 recenter 后的打印：
    try:
        fut_vis  = recenter_for_view(fut_178)
        pred_vis = recenter_for_view(pred_178)
        gen_vis  = recenter_for_view(gen_178)

        dbg("fut_vis", fut_vis)
        dbg("pred_vis", pred_vis)
        dbg("gen_vis", gen_vis)
    except Exception as e:
        print("[DEBUG] recenter failed:", e)


    # ========================================================================
    # Save Pose Files
    # ========================================================================
    out_gt   = os.path.join(out_dir, "gt_178.pose")
    out_pred = os.path.join(out_dir, "pred_178.pose")
    out_gen  = os.path.join(out_dir, "gen_178.pose")

    tensor_to_pose(fut_178, header_178).write(open(out_gt, "wb"))
    tensor_to_pose(pred_178, header_178).write(open(out_pred, "wb"))
    tensor_to_pose(gen_178, header_178).write(open(out_gen, "wb"))

    print(f"[SAVE] Saved -> {out_gt}")
    print(f"[SAVE] Saved -> {out_pred}")
    print(f"[SAVE] Saved -> {out_gen}")

    # ========================================================================
    # Motion Summary
    # ========================================================================
    def motion_stats(x):
        if x.size(1) <= 1:
            return 0.0, 0.0
        d = x[:, 1:] - x[:, :-1]
        return float(d.abs().mean()), float(d.std())

    print("\n=========== MOTION SUMMARY (178) ===========")
    print("GT   :", motion_stats(fut_178))
    print("PRED :", motion_stats(pred_178))
    print("GEN  :", motion_stats(gen_178))
    print("============================================\n")
