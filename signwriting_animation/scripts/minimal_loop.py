# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import (
    LitMinimal,
    sanitize_btjc,
    masked_dtw,
)

# ---------------------------------------------
# 尝试导入 unshift_hands（可视化时用）
# ---------------------------------------------
try:
    from pose_anonymization.data.normalization import unshift_hands
    HAS_UNSHIFT = True
    print("[✓] Successfully imported unshift_hands")
except ImportError as e:
    print(f"[✗] Warning: Could not import unshift_hands: {e}")
    print("[!] PRED poses will have incorrect hand positions!")
    HAS_UNSHIFT = False


# ---------------------------------------------------------
# 工具：读回 .pose，简单检查分布
# ---------------------------------------------------------
def inspect_pose(path: str, name: str):
    """
    读回 .pose 文件，打印:
    - 数据 shape
    - 全局最小/最大值
    - 每帧骨架的平均方差
    """
    if not os.path.exists(path):
        print(f"\n[{name}] 文件不存在: {path}")
        return

    with open(path, "rb") as f:
        pose = Pose.read(f)

    data = pose.body.data  # [T, P, J, C]
    data_np = np.asarray(data, dtype=np.float32)
    T, P, J, C = data_np.shape
    data_tjc = data_np.reshape(T, P * J, C)

    center = data_tjc.mean(axis=1, keepdims=True)        # [T,1,C]
    var = ((data_tjc - center) ** 2).mean(axis=(1, 2))   # [T]

    print(f"\n[{name}] {path}")
    print(f"  shape: {data_np.shape}")
    print(f"  range: [{data_np.min():.4f}, {data_np.max():.4f}]")
    print(f"  per-frame var min/max: [{var.min():.6f}, {var.max():.6f}]")


# ---------------------------------------------------------
# 工具：tensor → Pose（带 unshift + 缩放 + 平移）
# ---------------------------------------------------------
def tensor_to_pose_complete(
    t_btjc: torch.Tensor,
    header,
    ref_pose: Pose,
    apply_unshift: bool = True,
    match_scale_to_ref: bool = True,
    align_center_to_ref: bool = True,
):
    """
    完整的 tensor → pose 转换：
    1. 使用 GT 的 FPS 和 confidence
    2. 可选调用 unshift_hands
    3. 可选根据 ref_pose 的空间方差对 PRED 做整体缩放
    4. 可选把 PRED 的全局中心平移到 ref_pose 的中心（只影响可视化）
    """
    # t_btjc: [B,T,J,C] 或 [T,J,C]
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    elif t_btjc.dim() == 3:
        t = t_btjc
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {t_btjc.dim()}D")

    t_np = t.detach().cpu().numpy().astype(np.float32)  # [T,J,C]

    print(
        f"\n[tensor_to_pose_complete] "
        f"(apply_unshift={apply_unshift}, match_scale_to_ref={match_scale_to_ref}, "
        f"align_center_to_ref={align_center_to_ref})"
    )
    print(f"  输入 shape: {t_np.shape}")
    print(f"  输入 range: [{t_np.min():.4f}, {t_np.max():.4f}]")

    arr = t_np[:, None, :, :]  # [T, 1, J, C]
    conf = ref_pose.body.confidence[: len(t_np)].copy()
    fps = ref_pose.body.fps

    body = NumPyPoseBody(fps=fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)

    print("  创建 Pose:")
    print(f"    fps: {fps}")
    print(f"    data shape: {pose_obj.body.data.shape}")
    print(f"    conf shape: {pose_obj.body.confidence.shape}")
    print(
        f"    data range: "
        f"[{pose_obj.body.data.min():.4f}, {pose_obj.body.data.max():.4f}]"
    )

    # 1) 手部反平移
    if apply_unshift and HAS_UNSHIFT:
        print("\n  调用 unshift_hands...")
        try:
            unshift_hands(pose_obj)
            print("    ✓ unshift 成功")
            print(
                f"    new range: "
                f"[{pose_obj.body.data.min():.4f}, {pose_obj.body.data.max():.4f}]"
            )
        except Exception as e:
            print(f"    ✗ unshift 失败: {e}")
    elif apply_unshift and not HAS_UNSHIFT:
        print("\n  ⚠️  预期 unshift_hands 但未导入成功")
    else:
        print("\n  ⚠️  本次不调用 unshift_hands，仅写入 raw 坐标")

    # 2) 根据 ref_pose 方差整体缩放
    T_pred = pose_obj.body.data.shape[0]
    ref_arr = np.asarray(ref_pose.body.data[:T_pred, 0], dtype=np.float32)   # [T,J,C]
    pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)  # [T,J,C]

    if match_scale_to_ref:
        try:
            def _var_tjc(a):
                center = a.mean(axis=1, keepdims=True)
                return float(((a - center) ** 2).mean())

            var_ref = _var_tjc(ref_arr)
            var_pred = _var_tjc(pred_arr)

            print(f"\n  [scale] ref_var={var_ref:.4f}, pred_var={var_pred:.4f}")
            if var_pred > 1e-8 and var_ref > 0:
                scale = float(np.sqrt((var_ref + 1e-6) / (var_pred + 1e-6)))
                print(f"  [scale] apply scale={scale:.3f}")
                pose_obj.body.data *= scale
                pred_arr = np.asarray(
                    pose_obj.body.data[:T_pred, 0], dtype=np.float32
                )
                print(
                    f"  scaled data range: "
                    f"[{pose_obj.body.data.min():.4f}, {pose_obj.body.data.max():.4f}]"
                )
            else:
                print("  [scale] var too small, skip scale")
        except Exception as e:
            print(f"  [scale] 计算缩放系数失败，跳过缩放: {e}")
            pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)

    # 3) 平移对齐中心
    if align_center_to_ref:
        try:
            ref_center = ref_arr.reshape(-1, 3).mean(axis=0)   # [3]
            pred_center = pred_arr.reshape(-1, 3).mean(axis=0) # [3]
            delta = ref_center - pred_center                   # [3]
            print(
                f"\n  [translate] ref_center={ref_center}, "
                f"pred_center={pred_center}"
            )
            print(f"  [translate] apply delta={delta}")
            pose_obj.body.data += delta  # broadcast 到 [T,1,J,C]
            print(
                f"  translated data range: "
                f"[{pose_obj.body.data.min():.4f}, {pose_obj.body.data.max():.4f}]"
            )
        except Exception as e:
            print(f"  [translate] 平移对齐失败，跳过平移: {e}")

    return pose_obj


# ---------------------------------------------------------
# 简单的“平均位移”度量：看帧间是否有运动
# ---------------------------------------------------------
def mean_frame_disp(x_btjc: torch.Tensor) -> float:
    x = sanitize_btjc(x_btjc)
    if x.size(1) < 2:
        return 0.0
    v = x[:, 1:] - x[:, :-1]
    return v.abs().mean().item()


# =========================================================
#                    MAIN
# =========================================================
if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    out_dir = "logs/minimal_178_aligned"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"

    print("\n" + "=" * 70)
    print("完全对齐版本 + 可视化缩放 & 平移修正（DIRECT 训练）")
    print("=" * 70)
    print("✅ LightningModule.unnormalize: 只做数值反归一化")
    print("✅ train_mode='direct'：直接预测 future pose")
    print("✅ tensor_to_pose: 调用 unshift_hands (若可用)")
    print("✅ 可视化时根据 ref_pose 方差 + 中心对 PRED 做 scale & translate")
    print("=" * 70 + "\n")

    # ---------------- Dataset ----------------
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )

    sample_0 = base_ds[0]

    class FixedSampleDataset(torch.utils.data.Dataset):
        def __init__(self, sample):
            self.sample = sample
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return self.sample

    train_ds = FixedSampleDataset(sample_0)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=zero_pad_collator,
    )

    # ---------------- Lightning Trainer ----------------
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=50,
    )

    num_joints = sample_0["data"].shape[-2]
    num_dims = sample_0["data"].shape[-1]

    # 关键：train_mode="direct"
    model = LitMinimal(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,
        diffusion_steps=50,
        beta_start=1e-4,
        beta_end=2e-2,
        pred_target="x0",
        train_mode="direct",
        vel_weight=0.5,
        acc_weight=0.25,
    )

    print("\n[训练中...]")
    trainer.fit(model, train_loader)

    # =================================================
    # INFERENCE: baseline vs direct prediction
    # =================================================
    print("\n" + "=" * 70)
    print("INFERENCE (direct mode)")
    print("=" * 70)

    model.eval()
    device = trainer.strategy.root_device
    model = model.to(device)

    with torch.no_grad():
        batch = next(iter(train_loader))
        cond = batch["conditions"]

        past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)  # [1,40,178,3]
        sign     = cond["sign_image"][:1].float().to(device)
        gt_raw   = sanitize_btjc(batch["data"][:1]).to(device)       # [1,20,178,3]

        future_len = gt_raw.size(1)

        # 1) Baseline: 静态平均骨架
        gt_mean_pose = gt_raw.mean(dim=1, keepdim=True)                 # [1,1,178,3]
        baseline     = gt_mean_pose.repeat(1, future_len, 1, 1)         # [1,20,178,3]
        mse_baseline = torch.mean((baseline - gt_raw) ** 2).item()
        disp_base    = mean_frame_disp(baseline)
        print(f"Baseline (static mean-pose) MSE: {mse_baseline:.4f}")
        print(f"Baseline mean frame-to-frame displacement: {disp_base:.6f}")

        # 2) Direct 预测
        pred_raw = model.predict_direct(past_raw, sign, future_len=future_len)

        mse_pred  = torch.mean((pred_raw - gt_raw) ** 2).item()
        disp_pred = mean_frame_disp(pred_raw)
        print(f"Direct prediction MSE: {mse_pred:.4f}")
        print(f"Direct mean frame-to-frame displacement: {disp_pred:.6f}")

        # 可选：DTW
        mask_bt = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred_raw, gt_raw, mask_bt)
        print(f"Direct prediction DTW: {dtw_val:.4f}")

        print(f"\nGT (raw):   [{gt_raw.min():.4f}, {gt_raw.max():.4f}]")
        print(f"PRED (raw): [{pred_raw.min():.4f}, {pred_raw.max():.4f}]")

    # =================================================
    # 保存到 .pose 做可视化
    # =================================================
    print("\n" + "=" * 70)
    print("加载参考 pose")
    print("=" * 70)

    ref_path = base_ds.records[0]["pose"]
    ref_path = ref_path if os.path.isabs(ref_path) else os.path.join(
        data_dir, ref_path
    )

    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)

    # 和训练一致：reduce_holistic + remove POSE_WORLD_LANDMARKS
    ref_pose = reduce_holistic(ref_pose)
    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_pose.header

    # 保存 GT（参考）
    out_gt = os.path.join(out_dir, "gt_reference.pose")
    with open(out_gt, "wb") as f:
        ref_pose.write(f)
    print(f"\n✓ GT (参考) 保存: {out_gt}")

    # 保存 PRED（unshift + scale + translate）
    print("\n" + "=" * 70)
    print("保存 PRED (unshift + scale + translate)")
    print("=" * 70)

    pose_pred = tensor_to_pose_complete(
        pred_raw,
        header,
        ref_pose,
        apply_unshift=True,
        match_scale_to_ref=True,
        align_center_to_ref=True,
    )
    out_pred = os.path.join(out_dir, "pred_complete.pose")
    with open(out_pred, "wb") as f:
        pose_pred.write(f)
    print(f"\n✓ PRED 保存: {out_pred}")

    # 回读检查
    print("\n" + "=" * 70)
    print("DEBUG: 读回 .pose 文件检查分布")
    print("=" * 70)

    inspect_pose(out_gt, "GT")
    inspect_pose(out_pred, "PRED")

    print("\n" + "=" * 70)
    print("✓ 完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print(f"  1. GT (参考):   {out_gt}")
    print(f"  2. PRED:        {out_pred}")
    print("\n在 sign.mt 中测试:")
    print("  1. 打开 gt_reference.pose")
    print("  2. 打开 pred_complete.pose")
    if not HAS_UNSHIFT:
        print("\n⚠️ 警告: unshift_hands 未成功导入，PRED 手部位置可能不正确")
