# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader
import sys, atexit
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc


# =============================================================================
# Safe IO Handling
# =============================================================================
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
atexit.register(lambda: sys.stdout.flush())

np.set_printoptions(suppress=True, linewidth=120, precision=4)
torch.set_printoptions(sci_mode=False, precision=4)


# =============================================================================
# Utility: Convert tensor to Pose (for visualization)
# =============================================================================
def tensor_to_pose(t_btjc, header, fps=25):
    """
    输入: [1,T,J,C] 或 [T,J,C]
    输出: Pose 对象，带 limbs，可被 pose_visualizer 画成人形
    """

    if t_btjc.dim() == 4:
        t = t_btjc[0]  # [T,J,C]
    else:
        t = t_btjc

    arr = t.detach().cpu().numpy().astype(np.float32)  # [T,J,C]

    # --------------------- 视觉中心平移 -----------------------
    center = np.median(arr[:, :, :2].reshape(-1, 2), axis=0)
    arr[:, :, :2] -= center

    # --------------------- 自适应缩放 -------------------------
    r = np.sqrt(arr[:, :, 0] ** 2 + arr[:, :, 1] ** 2).reshape(-1)
    scale = 120 / (np.percentile(r, 95) + 1e-6)
    arr[:, :, :2] *= scale
    if arr.shape[-1] > 2:
        arr[:, :, 2] *= scale

    # --------------------- 移到画面中央 -----------------------
    arr[:, :, :2] += np.array([150.0, 150.0], dtype=np.float32)

    # Pose 格式要求 [T,1,J,C]
    arr4 = arr[:, None, :, :]
    conf = np.ones((arr4.shape[0], 1, arr4.shape[2], 1), dtype=np.float32)

    return Pose(header=header, body=NumPyPoseBody(fps=fps, data=arr4, confidence=conf))


# =============================================================================
# Safe Inference Helpers
# =============================================================================
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


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/overfit_178vis"
    os.makedirs(out_dir, exist_ok=True)

    # ========================= DATASET =========================
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=60,
        num_future_frames=30,
        with_metadata=True,
        split="train",
        reduce_holistic=False,   # --- 保持 586 joints ---
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

    # ========================= MODEL =========================
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

    # ========================= TRAIN =========================
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

    print("[TRAIN] Overfitting on small subset...")
    trainer.fit(model, loader, loader)

    # ====================== REDUCE TO 178 ======================
    print("\n[VIS] Building 178-joint skeleton...")

    # 1) 加载第一条 pose 文件作为 header 参考
    pose_path = base_ds.records[0]["pose"]
    pose_abspath = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)

    with open(pose_abspath, "rb") as f:
        pose_full = Pose.read(f)

    # 2) 去掉 world component
    pose_now = pose_full.remove_components(["POSE_WORLD_LANDMARKS"])

    # 3) reduce_holistic 得到标准 178 结构（带 limbs）
    pose_178 = reduce_holistic(pose_now)
    header_178 = pose_178.header

    print("[OK] reduce_holistic → 178 joints ready")

    # ------------------ 建立 index_map（安全，不依赖顺序） --------------------
    name2index586 = {}
    base = 0
    for comp in pose_full.header.components:
        for p in comp.points:
            name2index586[(comp.name, p)] = base
            base += 1

    index_map = []
    for comp in header_178.components:
        for p in comp.points:
            index_map.append(name2index586[(comp.name, p)])

    idx_t = torch.tensor(index_map, dtype=torch.long, device=model.device)

    print("[OK] index_map built, length =", len(index_map))

    # ========================= GENERATION =========================
    batch = next(iter(loader))
    cond = batch["conditions"]

    fut_raw  = sanitize_btjc(batch["data"][:1].to(model.device))
    past_raw = sanitize_btjc(cond["input_pose"][:1].to(model.device))
    sign_img = cond["sign_image"][:1].to(model.device)

    T_fut = fut_raw.size(1)

    # ---- GT reduce
    fut_178 = fut_raw.index_select(2, idx_t)

    with open(os.path.join(out_dir, "gt_178.pose"), "wb") as f:
        tensor_to_pose(fut_178, header_178).write(f)

    # ---- Single-frame prediction
    pred_full = inference_one_frame(model, past_raw, sign_img)
    pred_178 = pred_full.index_select(2, idx_t)

    with open(os.path.join(out_dir, "pred_1frame_178.pose"), "wb") as f:
        tensor_to_pose(pred_178, header_178).write(f)

    # ---- Autoregressive generation
    gen_full = autoregressive_generate(model, past_raw, sign_img, future_len=T_fut)
    gen_178 = gen_full.index_select(2, idx_t)

    with open(os.path.join(out_dir, "gen_178.pose"), "wb") as f:
        tensor_to_pose(gen_178, header_178).write(f)

    # ========================= MOTION SUMMARY =========================
    def motion_stats(x):
        if x.size(1) <= 1:
            return (0.0, 0.0)
        d = x[:, 1:] - x[:, :-1]
        return d.abs().mean().item(), d.std().item()

    print("\n=========== MOTION SUMMARY (178) ===========")
    print("GT    :", motion_stats(fut_178))
    print("PRED  :", motion_stats(pred_178))
    print("GEN   :", motion_stats(gen_178))
    print("=============================================\n")
