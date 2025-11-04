# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import lightning as pl
from torch.utils.data import Dataset, DataLoader
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import normalize_mean_std

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, masked_dtw


# ============================================================
#  Utility Functions
# ============================================================

def unnormalize_tensor_with_global_stats(tensor: torch.Tensor, mean_std: dict):
    """反归一化：x * std + mean"""
    if torch.is_tensor(mean_std["mean"]):
        mean = mean_std["mean"].detach().clone().float().to(tensor.device)
        std = mean_std["std"].detach().clone().float().to(tensor.device)
    else:
        mean = torch.tensor(mean_std["mean"], dtype=torch.float32, device=tensor.device)
        std = torch.tensor(mean_std["std"], dtype=torch.float32, device=tensor.device)
    return tensor * std + mean


def _unwrap_mean_std(ms):
    """兼容多种 mean/std 存储格式"""
    if isinstance(ms, dict):
        return ms
    elif hasattr(ms, "mean") and hasattr(ms, "std"):
        return {"mean": ms.mean, "std": ms.std}
    elif isinstance(ms, (list, tuple)) and len(ms) == 2:
        return {"mean": ms[0], "std": ms[1]}
    else:
        raise ValueError(f"Unsupported mean_std format: {type(ms)}")


def _to_plain(x):
    """Convert MaskedTensor or Lightning output to plain [B,T,J,C] tensor"""
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    if x.dim() == 5 and x.shape[2] == 1:
        x = x.squeeze(2)
    return x.detach().contiguous().float()


def temporal_smooth(x, k=5):
    """对时间维平滑，减少抖动 (x: [B,T,J,C] 或 [T,J,C])"""
    import torch.nn.functional as F
    if x.dim() == 5 and x.shape[2] == 1:
        x = x.squeeze(2)
    if x.dim() == 4:
        x = x[0]
    T, J, C = x.shape
    x = x.permute(2, 1, 0).reshape(1, C * J, T)
    x = F.avg_pool1d(x, kernel_size=k, stride=1, padding=k // 2)
    x = x.reshape(C, J, T).permute(2, 1, 0).contiguous()
    return x


def recenter_for_view(x, offset=(0, 0, 0)):
    if hasattr(x, "tensor"): x = x.tensor
    if hasattr(x, "zero_filled"): x = x.zero_filled()
    if x.dim() == 5 and x.shape[2] == 1: x = x.squeeze(2)
    if x.dim() == 4: x = x.mean(dim=0)  # [T,J,C]

    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    T, J, C = x.shape
    torso_end = min(33, J)
    torso_xy = x[:, :torso_end, :2]  # 仅 torso 区域

    mask = torch.isfinite(torso_xy)
    valid_xy = torso_xy[mask].view(-1, 2) if mask.any() else torch.zeros(1, 2)
    mean_center = valid_xy.mean(dim=0, keepdim=True)
    median_center = valid_xy.median(dim=0).values.unsqueeze(0)
    center = 0.5 * (mean_center + median_center)
    x[..., :2] -= center  # 平移到原点

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
    """将 [T,J,C] 或 [B,T,J,C] 转为 Pose 对象"""
    t = _to_plain(t_btjc)
    if t.dim() == 4:
        t = t[0]
    arr = np.ascontiguousarray(t[:, None, :, :], dtype=np.float32)  # [T,1,J,C]
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)
    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    return Pose(header=header, body=body)


# ============================================================
#  Main Script
# ============================================================

if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/overfit_178"
    os.makedirs(out_dir, exist_ok=True)

    mean_std_path = os.path.join(data_dir, "mean_std_178.pt")

    # ---------------------- Dataset ----------------------
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
        reduce_holistic=True,
    )

    base_ds.mean_std = torch.load(mean_std_path)
    print(f"[NORM] Loaded mean/std from {mean_std_path}")

    first = _to_plain(base_ds[0]["data"])
    print(f"[CHECK] raw data mean/std: {first.mean():.4f}, {first.std():.4f}")

    small_ds = torch.utils.data.Subset(base_ds, list(range(min(4, len(base_ds)))))
    loader = DataLoader(small_ds, batch_size=4, shuffle=True, collate_fn=zero_pad_collator)

    shape = next(iter(loader))["data"].shape
    print(f"[INFO] Overfit set shape: {tuple(shape)}")

    # ---------------------- Training ----------------------
    model = LitMinimal(num_keypoints=shape[-2], num_dims=shape[-1], lr=1e-4, log_dir=out_dir)
    trainer = pl.Trainer(
        max_epochs=500,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        limit_train_batches=1,
        limit_val_batches=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
        deterministic=True,
    )
    print(f"[TRAIN] Overfitting on 4 samples × {shape[-2]} joints × {shape[-1]} dims")
    trainer.fit(model, loader, loader)

# ------------------ Evaluation ------------------
if __name__ == "__main__":
    print("=== DEBUG EVAL START ===")
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        cond = batch["conditions"]
        past = cond["input_pose"][:1].to(model.device)
        sign = cond["sign_image"][:1].to(model.device)
        fut  = batch["data"][:1].to(model.device)
        mask = cond["target_mask"][:1].to(model.device)

        print("[CHECK] target_mask mean / sum:",
              mask.float().mean().item(), mask.float().sum().item())
        frame_density = mask.float().mean(dim=(2,3,4)).cpu().numpy().tolist()
        print("[CHECK] per-frame activation (avg over joints):", frame_density)
        sys.stdout.flush()

        print("=== USING TEACHER-FORCED PATH ===")
        ts = torch.zeros(fut.size(0), dtype=torch.long, device=fut.device)
        in_seq = fut.tensor.clone() if hasattr(fut, "tensor") else fut.clone()

        pred = model.forward(in_seq, ts, past, sign)
        print("[EVAL] pred (teacher-forced) mean/std:",
              pred.mean().item(), pred.std().item())

        pj_std = pred[0, :, :, :2].std(dim=0).mean(dim=1)
        print("[DBG] per-joint std head:", pj_std[:12].tolist())
        print("[DBG] avg joint std:", pj_std.mean().item())

        dtw_val = masked_dtw(pred, fut, mask).item()
        print(f"[EVAL] masked_dtw = {dtw_val:.4f}")

        # ---- plain tensors ----
        for name in ["fut", "pred"]:
            x = locals()[name]
            if hasattr(x, "tensor"):
                x = x.tensor
            if hasattr(x, "zero_filled"):
                x = x.zero_filled()
            if x.dim() == 5 and x.shape[2] == 1:
                x = x.squeeze(2)
            locals()[name] = x

        # ---- range check (normalized space) ----
        print(f"[CHECK fut range]  min={fut.min().item():.2f}, max={fut.max().item():.2f}, "
              f"mean={fut.mean().item():.2f}, std={fut.std().item():.2f}")
        print(f"[CHECK pred range] min={pred.min().item():.2f}, max={pred.max().item():.2f}, "
              f"mean={pred.mean().item():.2f}, std={pred.std().item():.2f}")

        # ---- clamp ----
        pred = torch.clamp(pred, -3, 3)
        print(f"[CHECK clamp] pred min={pred.min().item():.3f}, max={pred.max().item():.3f}")

        # ---- unnormalize ----
        mean_std = _unwrap_mean_std(base_ds.mean_std)
        fut_un  = unnormalize_tensor_with_global_stats(fut,  mean_std)
        pred_un = unnormalize_tensor_with_global_stats(pred, mean_std)
        print("[UNNORM] Applied FluentPose-style unnormalize ✅")
        print("[DEBUG unnorm] fut mean/std:", fut_un.mean().item(), fut_un.std().item())
        print("[DEBUG unnorm] pred mean/std:", pred_un.mean().item(), pred_un.std().item())

        # ---- temporal smooth ----
        fut_un  = temporal_smooth(fut_un)
        pred_un = temporal_smooth(pred_un)

        # ---- axis-wise stats ----
        def axis_stats(t):
            if t.dim() == 4: t = t[0]
            m = t.mean(dim=(0,1)); s = t.std(dim=(0,1))
            return [round(v, 3) for v in m.tolist()], [round(v, 3) for v in s.tolist()]
        m_gt, s_gt = axis_stats(fut_un)
        m_pr, s_pr = axis_stats(pred_un)
        print(f"[POST-UNNORM] GT axis mean={m_gt}, std={s_gt}")
        print(f"[POST-UNNORM] PR axis mean={m_pr}, std={s_pr}")

    # ---------------------- Save Pose ----------------------
    pose_path = base_ds.records[0]["pose"]
    ref_path = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)
    print(f"[REF] Using reference pose header from: {ref_path}")

    try:
        with open(ref_path, "rb") as f:
            ref_pose = Pose.read(f)
        ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
        header = reduce_holistic(ref_pose).header
        print("[HEADER] limb counts:", [len(c.limbs) for c in header.components])
    except Exception as e:
        print(f"[ERROR] Failed to load or reduce header: {e}")
        raise

    print("\n=== COMPONENT DIAG ===")
    print("[HEADER COMPONENTS]", [c.name for c in header.components])
    sys.stdout.flush()

    def component_slice(header, name):
        s = 0
        for comp in header.components:
            e = s + len(comp.points)
            if name.upper() in comp.name.upper():
                return s, e
            s = e
        return None

    def hand_diag(tag, x_btjc, sl):
        if sl is None:
            print(f"[WARN] {tag} hand slice not found in header")
            return
        x = x_btjc if x_btjc.dim()==3 else x_btjc[0]
        x = x[:, sl[0]:sl[1], :2]
        nan_ratio = torch.isnan(x).float().mean().item()
        big = (x.abs()>2000).float().mean().item()
        per_frame_std = torch.nanstd(x, dim=1).nanmean(dim=(0,1)).item()
        print(f"[{tag}] NaN%={nan_ratio:.4f}, |xy|>2000%={big:.4f}, frame-std≈{per_frame_std:.2f}")

    rhs = component_slice(header, "RIGHT_HAND_LANDMARKS")
    lhs = component_slice(header, "LEFT_HAND_LANDMARKS")
    hand_diag("GT RIGHT", fut_un, rhs)
    hand_diag("PR RIGHT", pred_un, rhs)
    hand_diag("GT LEFT",  fut_un, lhs)
    hand_diag("PR LEFT",  pred_un, lhs)

    # ---- recenter for visualization ----
    fut_for_save  = recenter_for_view(fut_un)
    pred_for_save = recenter_for_view(pred_un)

    try:
        gt_pose   = tensor_to_pose(fut_for_save,  header)
        pred_pose = tensor_to_pose(pred_for_save, header)
        print(f"[DEBUG] gt_pose frames={len(gt_pose.body.data)}, pred_pose frames={len(pred_pose.body.data)}")
    except Exception as e:
        print(f"[ERROR] Failed to create Pose objects: {e}")
        raise

    out_gt   = os.path.join(out_dir, "gt_178.pose")
    out_pred = os.path.join(out_dir, "pred_178.pose")
    for pth in [out_gt, out_pred]:
        if os.path.exists(pth):
            os.remove(pth)

    try:
        with open(out_gt, "wb") as f:
            gt_pose.write(f)
        with open(out_pred, "wb") as f:
            pred_pose.write(f)
        print(f"[SAVE] Pose files written → {out_dir}")
        print(f"[OK] ✅ Pose objects successfully written: {os.path.basename(out_gt)}, {os.path.basename(out_pred)}")
    except Exception as e:
        print(f"[ERROR] ❌ Failed to save pose files: {e}")
        raise

    try:
        p = Pose.read(open(out_pred, "rb"))
        print("[CHECK] Pose OK ✅ limbs:", [len(c.limbs) for c in p.header.components])
    except Exception as e:
        print(f"[ERROR] Pose verification failed: {e}")
