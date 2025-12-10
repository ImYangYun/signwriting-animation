# -*- coding: utf-8 -*-
"""
测试自回归 Direct 模式

关键检查点：
1. 训练时 frame-to-frame displacement 应该 > 0
2. 推理时预测应该有动态运动
"""
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
    LitMinimalAutoregressive,
    sanitize_btjc,
    masked_dtw,
    mean_frame_disp,
)

try:
    from pose_anonymization.data.normalization import unshift_hands
    HAS_UNSHIFT = True
    print("[✓] Successfully imported unshift_hands")
except ImportError as e:
    print(f"[✗] Warning: Could not import unshift_hands: {e}")
    HAS_UNSHIFT = False


def tensor_to_pose_complete(
    t_btjc: torch.Tensor,
    header,
    ref_pose: Pose,
    apply_unshift: bool = True,
    match_scale_to_ref: bool = True,
    align_center_to_ref: bool = True,
):
    """完整的 tensor → pose 转换"""
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    elif t_btjc.dim() == 3:
        t = t_btjc
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {t_btjc.dim()}D")

    t_np = t.detach().cpu().numpy().astype(np.float32)

    print(f"\n[tensor_to_pose_complete]")
    print(f"  输入 shape: {t_np.shape}")
    print(f"  输入 range: [{t_np.min():.4f}, {t_np.max():.4f}]")

    arr = t_np[:, None, :, :]
    conf = ref_pose.body.confidence[: len(t_np)].copy()
    fps = ref_pose.body.fps

    body = NumPyPoseBody(fps=fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)

    # 手部反平移
    if apply_unshift and HAS_UNSHIFT:
        try:
            unshift_hands(pose_obj)
            print("  ✓ unshift 成功")
        except Exception as e:
            print(f"  ✗ unshift 失败: {e}")

    # 缩放对齐
    T_pred = pose_obj.body.data.shape[0]
    ref_arr = np.asarray(ref_pose.body.data[:T_pred, 0], dtype=np.float32)
    pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)

    if match_scale_to_ref:
        def _var_tjc(a):
            center = a.mean(axis=1, keepdims=True)
            return float(((a - center) ** 2).mean())

        var_ref = _var_tjc(ref_arr)
        var_pred = _var_tjc(pred_arr)

        if var_pred > 1e-8 and var_ref > 0:
            scale = float(np.sqrt((var_ref + 1e-6) / (var_pred + 1e-6)))
            print(f"  [scale] apply scale={scale:.3f}")
            pose_obj.body.data *= scale
            pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)

    # 平移对齐
    if align_center_to_ref:
        ref_center = ref_arr.reshape(-1, 3).mean(axis=0)
        pred_center = pred_arr.reshape(-1, 3).mean(axis=0)
        delta = ref_center - pred_center
        print(f"  [translate] apply delta={delta}")
        pose_obj.body.data += delta

    return pose_obj


if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    out_dir = "logs/minimal_178_autoregressive"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"

    print("\n" + "=" * 70)
    print("自回归 Direct 模式测试")
    print("=" * 70)
    print("✅ 每步只预测 1 帧，滚动更新历史")
    print("✅ 添加 motion_loss 鼓励运动")
    print("=" * 70 + "\n")

    # Dataset
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

    # Trainer
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=50,
    )

    num_joints = sample_0["data"].shape[-2]
    num_dims = sample_0["data"].shape[-1]

    model = LitMinimalAutoregressive(
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
        motion_weight=0.1,
    )

    print("\n[训练中...]")
    trainer.fit(model, train_loader)

    # Inference
    print("\n" + "=" * 70)
    print("INFERENCE (autoregressive direct mode)")
    print("=" * 70)

    model.eval()
    device = trainer.strategy.root_device
    model = model.to(device)

    with torch.no_grad():
        batch = next(iter(train_loader))
        cond = batch["conditions"]

        past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt_raw = sanitize_btjc(batch["data"][:1]).to(device)

        future_len = gt_raw.size(1)

        # Baseline
        gt_mean_pose = gt_raw.mean(dim=1, keepdim=True)
        baseline = gt_mean_pose.repeat(1, future_len, 1, 1)
        mse_baseline = torch.mean((baseline - gt_raw) ** 2).item()
        disp_base = mean_frame_disp(baseline)
        print(f"Baseline (static) MSE: {mse_baseline:.4f}")
        print(f"Baseline frame-to-frame displacement: {disp_base:.6f}")

        # Direct prediction
        pred_raw = model.predict_direct(past_raw, sign, future_len=future_len)

        mse_pred = torch.mean((pred_raw - gt_raw) ** 2).item()
        disp_pred = mean_frame_disp(pred_raw)
        disp_gt = mean_frame_disp(gt_raw)
        
        print(f"\nDirect prediction MSE: {mse_pred:.4f}")
        print(f"Direct frame-to-frame displacement: {disp_pred:.6f}")
        print(f"GT frame-to-frame displacement: {disp_gt:.6f}")
        
        # 关键检查
        if disp_pred < 1e-6:
            print("\n⚠️ 警告: 预测仍然是静态的！displacement ≈ 0")
        elif disp_pred < disp_gt * 0.1:
            print("\n⚠️ 警告: 预测运动远小于 GT")
        else:
            print("\n✓ 预测有运动！")

        # DTW
        mask_bt = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred_raw, gt_raw, mask_bt)
        print(f"Direct prediction DTW: {dtw_val:.4f}")

        print(f"\nGT (raw):   [{gt_raw.min():.4f}, {gt_raw.max():.4f}]")
        print(f"PRED (raw): [{pred_raw.min():.4f}, {pred_raw.max():.4f}]")

    # 保存 .pose
    print("\n" + "=" * 70)
    print("保存 .pose 文件")
    print("=" * 70)

    ref_path = base_ds.records[0]["pose"]
    ref_path = ref_path if os.path.isabs(ref_path) else os.path.join(data_dir, ref_path)

    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)

    ref_pose = reduce_holistic(ref_pose)
    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_pose.header

    # GT
    out_gt = os.path.join(out_dir, "gt_reference.pose")
    with open(out_gt, "wb") as f:
        ref_pose.write(f)
    print(f"✓ GT 保存: {out_gt}")

    # PRED
    pose_pred = tensor_to_pose_complete(pred_raw, header, ref_pose)
    out_pred = os.path.join(out_dir, "pred_autoregressive.pose")
    with open(out_pred, "wb") as f:
        pose_pred.write(f)
    print(f"✓ PRED 保存: {out_pred}")

    print("\n" + "=" * 70)
    print("✓ 完成！")
    print("=" * 70)