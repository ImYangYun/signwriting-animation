# -*- coding: utf-8 -*-
import os
import glob
import math
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc

np.set_printoptions(suppress=True, linewidth=120, precision=4)
torch.set_printoptions(sci_mode=False, precision=4)

# ======================== 可视化稳形配置 ========================
FLIP_Y  = True     # 图像坐标常见 y 轴向下 → 翻转更“直观”
SWAP_XY = False    # 若仍像旋转90°，再把它设 True 试试
ZERO_Z  = True     # 可视化置 z=0，避免“翻滚”错觉
TARGET_SCALE = 120.0
CANVAS_SHIFT = np.array([150.0, 150.0], dtype=np.float32)

TORSO_KEYWORDS = ("shoulder", "hip", "waist", "neck", "chest", "torso")

def _torso_indices(header):
    idx = []
    for comp_id, comp in enumerate(header.components):
        for j, name in enumerate(comp.points):
            low = name.lower()
            if any(k in low for k in TORSO_KEYWORDS):
                # 组件拼成全局下标：需要把之前组件的点数累加
                before = sum(len(hc.points) for hc in header.components[:comp_id])
                idx.append(before + j)
    return np.array(sorted(set(idx)), dtype=np.int64)

def _pca_angle_xy(arr_btjc, torso_idx):
    """对整段序列的躯干XY做 PCA，返回主轴相对 x 轴的平均角度（弧度，顺时针为负）。"""
    xy = arr_btjc[:, :, :2]  # [T,J,2]
    if torso_idx.size > 0:
        xy = xy[:, torso_idx, :]
    xy = xy.reshape(-1, 2)
    mu = np.nanmedian(xy, axis=0, keepdims=True)
    X = xy - mu
    cov = (X.T @ X) / max(1, X.shape[0]-1)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]  # 主特征向量
    angle = math.atan2(v[1], v[0])
    return angle

def _rotate_xy(arr_btjc, theta):
    """对 XY 平面整体旋转 -theta，使主轴横向。"""
    c, s = math.cos(-theta), math.sin(-theta)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)
    xy = arr_btjc[:, :, :2]
    T, J, _ = xy.shape
    xy2 = xy.reshape(-1, 2) @ R.T
    arr_btjc[:, :, :2] = xy2.reshape(T, J, 2)
    return arr_btjc

# ---------------------- robust tensor → Pose (center+scale+稳形，仅用于保存可视化) ----------------------
def tensor_to_pose(t_btjc, header):
    # t_btjc: [T,J,C] or [1,T,J,C]
    t = t_btjc[0] if t_btjc.dim() == 4 else t_btjc
    arr = t.detach().cpu().numpy().astype(np.float32)  # [T,J,C]

    # 0) 只做显示层面的坐标变换
    if SWAP_XY and arr.shape[-1] >= 2:
        arr[:, :, [0, 1]] = arr[:, :, [1, 0]]
    if FLIP_Y and arr.shape[-1] >= 2:
        arr[:, :, 1] = -arr[:, :, 1]
    if ZERO_Z and arr.shape[-1] >= 3:
        arr[:, :, 2] = 0.0

    # 1) 以“躯干关节”居中（没有就退化为全体点）
    torso_idx = _torso_indices(header)
    if torso_idx.size > 0:
        ctr_xy = np.nanmedian(arr[:, torso_idx, :2], axis=1, keepdims=True)  # [T,1,2]
    else:
        ctr_xy = np.nanmedian(arr[:, :, :2], axis=1, keepdims=True)
    arr[:, :, :2] -= ctr_xy

    # 2) 用整段序列躯干 PCA 求统一旋转角，减少“翻滚/歪头”
    try:
        theta = _pca_angle_xy(arr, torso_idx)
        arr = _rotate_xy(arr, theta)
    except Exception:
        pass  # 容错，不影响后续

    # 3) 统一缩放（95 分位半径）
    r = np.sqrt(arr[:, :, 0]**2 + arr[:, :, 1]**2)       # [T,J]
    s = np.percentile(r, 95) + 1e-6
    scale = TARGET_SCALE / s
    arr[:, :, :2] *= scale

    # 4) 平移到画布
    arr[:, :, :2] += CANVAS_SHIFT[None, None, :]

    arr4 = arr[:, None, :, :]  # [T,1,J,C]
    conf = np.ones((arr4.shape[0], 1, arr4.shape[2], 1), dtype=np.float32)
    body = NumPyPoseBody(fps=25, data=arr4, confidence=conf)
    return Pose(header=header, body=body)

@torch.no_grad()
def inference_one_frame(model: LitMinimal, past_btjc: torch.Tensor, sign_img: torch.Tensor):
    return model.sample_autoregressive_diffusion(
        past_btjc=past_btjc, sign_img=sign_img, future_len=1, chunk=1
    )

@torch.no_grad()
def autoregressive_generate(model: LitMinimal, past_btjc: torch.Tensor, sign_img: torch.Tensor, future_len: int):
    # chunk=1 更稳；若实现里忽略 future_len，这里下游有兜底
    return model.sample_autoregressive_diffusion(
        past_btjc=past_btjc, sign_img=sign_img, future_len=future_len, chunk=1
    )

# ---------------------- 推理安全夹（止爆，仅推理） ----------------------
def clip_to_data_range(model: LitMinimal, x_un: torch.Tensor, clip_sigma: float = 3.0):
    xn = model.normalize(x_un)
    xn = torch.clamp(xn, -clip_sigma, clip_sigma)
    return model.unnormalize(xn)

# ---------------------- Main ----------------------
if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/overfit_178"
    os.makedirs(out_dir, exist_ok=True)

    # Dataset（小样本过拟合）
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=60,
        num_future_frames=30,
        with_metadata=True,
        split="train",
        reduce_holistic=True,
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

    model = LitMinimal(
        num_keypoints=J,
        num_dims=C,
        lr=3e-4,
        stats_path=os.path.join(data_dir, "mean_std_178.pt"),
        diffusion_steps=200,
        beta_start=1e-4,
        beta_end=2e-2,
        pred_target="x0",          # <== 与日志里 target=eps 不一致会引爆，确保模型内部不覆盖
        guidance_scale=0.0,
    )

    # ===== mean/std 匹配检查（以及可选一键校准）=====
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

    try:
        if hasattr(model, "verbose"): model.verbose = False
        if hasattr(model, "model") and hasattr(model.model, "verbose"):
            model.model.verbose = False
    except Exception:
        pass

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
    print("\n=== Evaluation ===")
    model.eval()
    batch = next(iter(loader))
    cond  = batch["conditions"]

    fut_raw  = sanitize_btjc(batch["data"][:1].to(model.device))      # [1,30,178,3]
    past_raw = sanitize_btjc(cond["input_pose"][:1].to(model.device)) # [1,60,178,3]
    sign_img = cond["sign_image"][:1].to(model.device)
    T_future = fut_raw.size(1)

    # ---------------------- Header ----------------------
    pose_path = base_ds.records[0]["pose"]
    src = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)
    with open(src, "rb") as f:
        ref_pose = Pose.read(f)
    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    header = reduce_holistic(ref_pose).header
    print("[HEADER] limbs per component:", [len(c.limbs) for c in header.components])
    joint_counts = [len(c.points) for c in header.components]
    print("[HEADER] joints per component:", joint_counts, " total_joints=", sum(joint_counts))

    # --------- 保存 GT：raw 与 unnorm（仅可视化变换，不影响数值）---------
    with open(os.path.join(out_dir, "gt_raw_178.pose"), "wb") as f:
        tensor_to_pose(fut_raw, header).write(f)
    print("[SAVE] gt_raw_178.pose saved")

    gt_norm = model.normalize(fut_raw)
    gt_un   = model.unnormalize(gt_norm)
    with open(os.path.join(out_dir, "gt_178.pose"), "wb") as f:
        tensor_to_pose(gt_un, header).write(f)
    print("[SAVE] gt_178.pose saved")

    # ---------------------- 1-frame prediction ----------------------
    pred_un = inference_one_frame(model, past_raw, sign_img)             # [1,1,J,C]
    pred_un = clip_to_data_range(model, pred_un, clip_sigma=3.0)
    print(f"[DEBUG] pred_1frame: min={pred_un.min().item():.2f}, max={pred_un.max().item():.2f}")
    with open(os.path.join(out_dir, "pred_1frame_178.pose"), "wb") as f:
        tensor_to_pose(pred_un, header).write(f)
    print("[SAVE] pred_1frame_178.pose saved")

    print("\n=== SANITY CHECK (1-frame detailed) ===")
    pred_norm = model.normalize(pred_un)
    gt_norm_1 = model.normalize(fut_raw[:, :1])
    print(f"[DEBUG] pred_norm shape = {pred_norm.shape} (should be [1,1,J,C])")
    print(f"[GT_norm] min={gt_norm_1.min().item():.4f}, max={gt_norm_1.max().item():.4f}, std={gt_norm_1.std().item():.4f}")
    print(f"[PR_norm] min={pred_norm.min().item():.4f}, max={pred_norm.max().item():.4f}, std={pred_norm.std().item():.4f}")
    print(f"[UNNORM pred] min={pred_un.min().item():.4f}, max={pred_un.max().item():.4f}, std={pred_un.std().item():.4f}")
    print("=== SANITY CHECK END ===\n")

    # ---------------------- Autoregressive generation (multi frames) ----------------------
    print("=== Inference B (autoregressive, multi-frames) ===")
    T_pred = min(5, max(2, T_future))  # 强制>=2帧
    gen_un = autoregressive_generate(model, past_raw, sign_img, future_len=T_pred)
    # 兜底：有实现可能仍返回 1 帧，这里强制滚动拼接成 T_pred
    if gen_un.size(1) < T_pred:
        gen_list = [gen_un]
        cur = gen_un
        while sum(x.size(1) for x in gen_list) < T_pred:
            cur = autoregressive_generate(model, cur, sign_img, future_len=1)
            gen_list.append(cur)
        gen_un = torch.cat(gen_list, dim=1)[:, :T_pred]
    gen_un = clip_to_data_range(model, gen_un, clip_sigma=3.0)

    # 运动统计（这下不会再 T=1 了）
    vel = gen_un[:, 1:] - gen_un[:, :-1]
    print(f"[GEN MOTION] mean |Δ| = {vel.abs().mean().item():.6f}, std = {vel.std().item():.6f}")
    print(f"[DEBUG] gen_un shape = {gen_un.shape} (should be [1,{T_pred},178,3])")

    with open(os.path.join(out_dir, "gen_178.pose"), "wb") as f:
        tensor_to_pose(gen_un, header).write(f)
    print("[SAVE] gen_178.pose saved")

    # ========================= SUMMARY =========================
    print("\n==================== ACTION SUMMARY ====================")

    def motion_stats(x):
        if x.size(1) <= 1: return 0.0, 0.0
        d = x[:, 1:] - x[:, :-1]
        return d.abs().mean().item(), d.std().item()

    gt_motion   = motion_stats(gt_un)
    pred_motion = motion_stats(pred_un)   # 单帧仍为0，这是预期
    gen_motion  = motion_stats(gen_un)
    print(f"[Motion GT ] meanΔ={gt_motion[0]:.6f}, stdΔ={gt_motion[1]:.6f}")
    print(f"[Motion PRED] meanΔ={pred_motion[0]:.6f}, stdΔ={pred_motion[1]:.6f}")
    print(f"[Motion GEN ] meanΔ={gen_motion[0]:.6f}, stdΔ={gen_motion[1]:.6f}")

    def l2_error(a, b):
        d = ((a - b) ** 2).sum(dim=-1).sqrt()  # [1,T,J]
        return d.mean().item(), d[:, :5].mean().item(), d[:, -5:].mean().item()

    l2_full, l2_early, l2_late = l2_error(pred_un, fut_raw[:, :1])  # pred_un is 1 frame
    print(f"[L2 Error 1-frame] full={l2_full:.6f}, first5={l2_early:.6f}, last5={l2_late:.6f}")

    gt_center  = gt_un.mean().item()
    pred_center = pred_un.mean().item()
    gen_center  = gen_un.mean().item()
    print(f"[Drift] GT_center={gt_center:.4f}, Pred_center={pred_center:.4f}, Gen_center={gen_center:.4f}")

    gt_norm_stats = (gt_norm.mean().item(), gt_norm.std().item())
    pred_norm_stats = (pred_norm.mean().item(), pred_norm.std().item())
    print(f"[Norm GT ] mean={gt_norm_stats[0]:.4f}, std={gt_norm_stats[1]:.4f}")
    print(f"[Norm PRED] mean={pred_norm_stats[0]:.4f}, std={pred_norm_stats[1]:.4f}")

    def axis_stats(x):
        if x.dim() == 4: x = x[0]
        m = x.mean(dim=(0, 1))
        s = x.std(dim=(0, 1))
        return m.tolist(), s.tolist()

    gt_m, gt_s = axis_stats(gt_un)
    pr_m, pr_s = axis_stats(pred_un)
    gen_m, gen_s = axis_stats(gen_un)
    print(f"[XYZ GT ] mean={gt_m}, std={gt_s}")
    print(f"[XYZ PRED] mean={pr_m}, std={pr_s}")
    print(f"[XYZ GEN ] mean={gen_m}, std={gen_s}")
    print("================== END SUMMARY ====================\n")

    print("\n=== Saved files in", out_dir, "===")
    for p in sorted(glob.glob(os.path.join(out_dir, "*.pose"))):
        try:
            print("[POSE]", os.path.basename(p), "size=", os.path.getsize(p), "bytes")
        except Exception:
            print("[POSE]", os.path.basename(p))
    print("====================================\n")
