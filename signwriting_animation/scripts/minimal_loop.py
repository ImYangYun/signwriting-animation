# -*- coding: utf-8 -*-
"""
真正的 Diffusion 测试 - T=8 步，cosine schedule，预测 x0
"""
import os
import sys
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset

from signwriting_animation.diffusion.lightning_module import LitDiffusion, sanitize_btjc, masked_dtw, mean_frame_disp

try:
    from pose_anonymization.data.normalization import unshift_hands
    HAS_UNSHIFT = True
except ImportError:
    HAS_UNSHIFT = False


def tensor_to_pose(t_btjc, header, ref_pose, gt_btjc=None, apply_scale=True):
    """转换 tensor 到 pose 格式"""
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc
    
    t_np = t.detach().cpu().numpy().astype(np.float32)

    gt_np = None
    if gt_btjc is not None:
        if gt_btjc.dim() == 4:
            gt_np = gt_btjc[0].detach().cpu().numpy().astype(np.float32)
        else:
            gt_np = gt_btjc.detach().cpu().numpy().astype(np.float32)

    arr = t_np[:, None, :, :]
    conf = ref_pose.body.confidence[:len(t_np)].copy()
    fps = ref_pose.body.fps
    
    body = NumPyPoseBody(fps=fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)
    
    if HAS_UNSHIFT:
        try:
            unshift_hands(pose_obj)
            print("  ✓ unshift 成功")
        except Exception as e:
            print(f"  ✗ unshift 失败: {e}")
    
    T_pred = pose_obj.body.data.shape[0]
    ref_arr = np.asarray(ref_pose.body.data[:T_pred, 0], dtype=np.float32)
    pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)
    
    if apply_scale and gt_np is not None:
        def _var(a):
            center = a.mean(axis=1, keepdims=True)
            return float(((a - center) ** 2).mean())
        
        var_gt_norm = _var(gt_np)
        var_ref = _var(ref_arr)
        
        if var_gt_norm > 1e-8:
            scale = np.sqrt(var_ref / var_gt_norm)
            print(f"  [scale] var_ref={var_ref:.2f}, var_gt_norm={var_gt_norm:.6f}")
            print(f"  [scale] normalized→pixel scale={scale:.2f}")
            pose_obj.body.data *= scale
            pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)

    ref_c = ref_arr.reshape(-1, 3).mean(axis=0)
    pred_c = pred_arr.reshape(-1, 3).mean(axis=0)
    delta = ref_c - pred_c
    print(f"  [translate] delta={delta}")
    pose_obj.body.data += delta
    
    print(f"  [final] range=[{pose_obj.body.data.min():.2f}, {pose_obj.body.data.max():.2f}]")
    
    return pose_obj


if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    out_dir = "logs/diffusion_real"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"

    print("\n" + "=" * 70)
    print("真正的 Diffusion 模型测试 (无残差版本)")
    print("=" * 70)
    print("参考师姐论文：T=8 步, cosine schedule, 预测 x0")
    print("关键改动：去掉残差，让模型直接预测 x0")
    print("训练：对 GT 加噪声 → 模型预测 x0")
    print("推理：纯噪声 → 8 步去噪 → 干净结果")
    print("=" * 70)

    # ===== 配置 =====
    SAMPLE_IDX = 50
    MAX_EPOCHS = 1000
    DIFFUSION_STEPS = 8  # 师姐用 T=8
    
    print(f"\n配置: 单样本 overfit (sample {SAMPLE_IDX})")
    print(f"Diffusion steps: {DIFFUSION_STEPS}")

    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    import random
    random.seed(12345)
    
    best_sample = base_ds[SAMPLE_IDX]
    print(f"样本 ID: {best_sample.get('id', 'unknown')}")
    
    # 单样本 Dataset
    class FixedSampleDataset(torch.utils.data.Dataset):
        def __init__(self, sample):
            self.sample = sample
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return self.sample
    
    train_ds = FixedSampleDataset(best_sample)
    train_loader = DataLoader(
        train_ds, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=zero_pad_collator,
    )
    
    # 获取维度信息
    sample_data = best_sample["data"]
    if hasattr(sample_data, 'zero_filled'):
        sample_data = sample_data.zero_filled()
    if hasattr(sample_data, 'tensor'):
        sample_data = sample_data.tensor
    
    num_joints = sample_data.shape[-2]
    num_dims = sample_data.shape[-1]
    print(f"关节数: {num_joints}, 维度: {num_dims}")

    # 创建真正的 Diffusion 模型
    model = LitDiffusion(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,
        diffusion_steps=DIFFUSION_STEPS,
        residual_scale=0.1,
        vel_weight=0.5,
        acc_weight=0.2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 训练
    print("\n" + "=" * 70)
    print("开始 Diffusion 训练")
    print("=" * 70)
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=10,
    )
    
    trainer.fit(model, train_loader)

    # Inference
    print("\n" + "=" * 70)
    print("DIFFUSION INFERENCE (8 步去噪)")
    print("=" * 70)
    
    model = model.to(device)
    model.eval()
    
    batch = next(iter(train_loader))
    cond = batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(batch["data"][:1]).to(device)
    
    future_len = gt_raw.size(1)
    
    with torch.no_grad():
        # Baseline
        baseline = gt_raw.mean(dim=1, keepdim=True).repeat(1, future_len, 1, 1)
        mse_base = torch.mean((baseline - gt_raw) ** 2).item()
        print(f"Baseline MSE: {mse_base:.4f}")
        
        # DDPM 采样（8 步）
        print("\n使用 DDPM 采样 (8 步)...")
        pred_raw_ddpm = model.sample(past_raw, sign, future_len)
        mse_ddpm = torch.mean((pred_raw_ddpm - gt_raw) ** 2).item()
        disp_ddpm = mean_frame_disp(pred_raw_ddpm)
        print(f"DDPM MSE: {mse_ddpm:.4f}, disp: {disp_ddpm:.6f}")
        
        # DDIM 采样（可以更快）
        print("\n使用 DDIM 采样...")
        pred_raw_ddim = model.sample_ddim(past_raw, sign, future_len)
        mse_ddim = torch.mean((pred_raw_ddim - gt_raw) ** 2).item()
        disp_ddim = mean_frame_disp(pred_raw_ddim)
        print(f"DDIM MSE: {mse_ddim:.4f}, disp: {disp_ddim:.6f}")
        
        disp_gt = mean_frame_disp(gt_raw)
        print(f"\nGT disp: {disp_gt:.6f}")
        
        # 用 DDPM 结果做评估
        pred_raw = pred_raw_ddpm
        
        # ===== 完整评估指标 =====
        print("\n" + "=" * 70)
        print("完整评估指标 (normalized 空间)")
        print("=" * 70)
        
        pred_np = pred_raw[0].cpu().numpy()
        gt_np = gt_raw[0].cpu().numpy()
        T, J, C = pred_np.shape
        
        # Position Errors
        mse = float(((pred_np - gt_np) ** 2).mean())
        mae = float(np.abs(pred_np - gt_np).mean())
        per_joint_error = np.sqrt(((pred_np - gt_np) ** 2).sum(axis=-1))
        mpjpe = float(per_joint_error.mean())
        fde = float(np.sqrt(((pred_np[-1] - gt_np[-1]) ** 2).sum(axis=-1)).mean())
        
        print(f"\n--- Position Errors (越低越好) ---")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  MPJPE: {mpjpe:.6f}")
        print(f"  FDE: {fde:.6f}")
        
        # Motion Match
        pred_vel = pred_np[1:] - pred_np[:-1]
        gt_vel = gt_np[1:] - gt_np[:-1]
        vel_mse = float(((pred_vel - gt_vel) ** 2).mean())
        pred_acc = pred_vel[1:] - pred_vel[:-1]
        gt_acc = gt_vel[1:] - gt_vel[:-1]
        acc_mse = float(((pred_acc - gt_acc) ** 2).mean())
        disp_ratio = disp_ddpm / (disp_gt + 1e-8)
        
        print(f"\n--- Motion Match ---")
        print(f"  disp_ratio: {disp_ratio:.4f} (理想=1.0)")
        print(f"  vel_mse: {vel_mse:.6f}")
        print(f"  acc_mse: {acc_mse:.6f}")
        
        # PCK
        print(f"\n--- PCK (越高越好) ---")
        for thresh in [0.05, 0.1, 0.2, 0.5]:
            pck = (per_joint_error < thresh).mean()
            print(f"  PCK@{thresh}: {pck:.2%}")
        
        # Joint Range
        pred_ranges = pred_np.max(axis=0) - pred_np.min(axis=0)
        gt_ranges = gt_np.max(axis=0) - gt_np.min(axis=0)
        ratio = (pred_ranges.sum(axis=1) + 1e-6) / (gt_ranges.sum(axis=1) + 1e-6)
        abnormal = np.where(ratio > 3.0)[0]
        
        print(f"\n--- Joint Range ---")
        print(f"  range_ratio_mean: {ratio.mean():.4f}")
        print(f"  range_ratio_max: {ratio.max():.4f}")
        print(f"  abnormal_joints (>3x): {len(abnormal)}")
        
        # DTW
        mask = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred_raw, gt_raw, mask)
        print(f"\n--- Trajectory ---")
        print(f"  DTW: {dtw_val:.4f}")
        
        print(f"\n预测范围: [{pred_raw.min():.4f}, {pred_raw.max():.4f}]")
        print(f"GT 范围: [{gt_raw.min():.4f}, {gt_raw.max():.4f}]")

    # 保存
    print("\n" + "=" * 70)
    print("保存文件")
    print("=" * 70)
    
    ref_path = base_ds.records[SAMPLE_IDX]["pose"]
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    
    ref_pose = reduce_holistic(ref_pose)
    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_pose.header
    
    # GT
    gt_pose = tensor_to_pose(gt_raw, header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
    out_gt = os.path.join(out_dir, f"gt_{SAMPLE_IDX}.pose")
    with open(out_gt, "wb") as f:
        gt_pose.write(f)
    print(f"\n✓ GT saved: {out_gt}")
    
    # Pred
    pred_pose = tensor_to_pose(pred_raw, header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
    out_pred = os.path.join(out_dir, f"pred_{SAMPLE_IDX}.pose")
    with open(out_pred, "wb") as f:
        pred_pose.write(f)
    print(f"✓ PRED saved: {out_pred}")
    
    print("\n" + "=" * 70)
    print("✓ 完成! Diffusion 版本测试结束")
    print("=" * 70)