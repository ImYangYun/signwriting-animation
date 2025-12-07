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
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc, masked_dtw

import signwriting_animation.diffusion.lightning_module as LM
print(">>> USING LIGHTNING MODULE FROM:", LM.__file__)


def tensor_to_pose(t_btjc, header):
    """Convert tensor → Pose-format object."""
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    elif t_btjc.dim() == 3:
        t = t_btjc
    else:
        raise ValueError("Expected 3D or 4D tensor")

    print(f"  [tensor_to_pose] shape: {t.shape}")
    
    # 检测零点
    zero_mask = (t.abs().sum(dim=-1) < 1e-6)
    num_zeros = zero_mask.sum().item()
    total = zero_mask.numel()
    print(f"  [tensor_to_pose] 零点: {num_zeros}/{total} ({100*num_zeros/total:.1f}%)")

    arr = t[:, None, :, :].cpu().numpy().astype(np.float32)
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)

    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    return Pose(header=header, body=body)


if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    out_dir = "logs/minimal_178"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178.pt"
    stats = torch.load(stats_path)

    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    base_ds.mean_std = torch.load(stats_path)

    small_ds = torch.utils.data.Subset(base_ds, [0, 1, 2, 3])
    loader = DataLoader(
        small_ds,
        batch_size=4,
        shuffle=True,
        collate_fn=zero_pad_collator,
    )

    num_joints = base_ds[0]["data"].shape[-2]
    num_dims = base_ds[0]["data"].shape[-1]
    print(f"[INFO] joints={num_joints}, dims={num_dims}")

    # Model
    model = LitMinimal(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-4,
    )

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_checkpointing=False,
        deterministic=True,
    )

    print("\n[TRAIN] Overfit 4 samples...")
    trainer.fit(model, loader, loader)

    # ============================================================
    # Load header
    # ============================================================
    ref_path = base_ds.records[0]["pose"]
    ref_path = ref_path if os.path.isabs(ref_path) else os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)

    ref_reduced = reduce_holistic(ref_pose)
    ref_reduced = ref_reduced.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_reduced.header

    print(f"\n[HEADER] total joints: {header.total_points()}")

    # ============================================================
    # Inference
    # ============================================================
    print("\n" + "="*70)
    print("INFERENCE")
    print("="*70)
    
    model.eval()
    device = trainer.strategy.root_device
    model = model.to(device)
    model.mean_pose = model.mean_pose.to(device)
    model.std_pose = model.std_pose.to(device)

    with torch.no_grad():
        batch = next(iter(loader))
        cond = batch["conditions"]

        past = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt = sanitize_btjc(batch["data"][:1]).to(device)

        future_len = gt.size(1)
        print(f"\n[1] 基本信息:")
        print(f"    future_len = {future_len}")
        print(f"    GT shape: {gt.shape}")

        # ============================================================
        # 诊断 PRED 的 unnormalize 问题
        # ============================================================
        print(f"\n[2] 生成 PRED（归一化空间）")
        
        pred_norm = model.sample_autoregressive_fast(
            past_btjc=past,
            sign_img=sign,
            future_len=future_len,
            chunk=1,
        )
        
        print(f"    pred_norm shape: {pred_norm.shape}")
        print(f"    pred_norm range: [{pred_norm.min():.4f}, {pred_norm.max():.4f}]")
        print(f"    pred_norm mean/std: {pred_norm.mean():.4f} / {pred_norm.std():.4f}")

        # ============================================================
        # 诊断模型的统计量
        # ============================================================
        print(f"\n[3] 检查模型的 mean 和 std:")
        print(f"    mean_pose shape: {model.mean_pose.shape}")
        print(f"    std_pose shape: {model.std_pose.shape}")
        print(f"    mean range: [{model.mean_pose.min():.4f}, {model.mean_pose.max():.4f}]")
        print(f"    std range: [{model.std_pose.min():.4f}, {model.std_pose.max():.4f}]")

        # 检查 std 的分布
        std_flat = model.std_pose.flatten()
        std_percentiles = {
            "min": std_flat.min().item(),
            "1%": torch.quantile(std_flat, 0.01).item(),
            "10%": torch.quantile(std_flat, 0.1).item(),
            "50%": torch.quantile(std_flat, 0.5).item(),
            "90%": torch.quantile(std_flat, 0.9).item(),
            "99%": torch.quantile(std_flat, 0.99).item(),
            "max": std_flat.max().item(),
        }
        
        print(f"\n    std 分布（百分位数）:")
        for k, v in std_percentiles.items():
            print(f"      {k:>4s}: {v:.6f}")

        std_near_zero = (model.std_pose < 1e-4).sum().item()
        std_very_small = (model.std_pose < 1e-2).sum().item()
        std_very_large = (model.std_pose > 100).sum().item()
        
        print(f"\n    异常 std 统计:")
        print(f"      接近0 (<1e-4): {std_near_zero}")
        print(f"      很小 (<0.01): {std_very_small}")
        print(f"      很大 (>100): {std_very_large}")

        # ============================================================
        # 测试 GT 的归一化循环
        # ============================================================
        print(f"\n[4] 测试 GT 的归一化→反归一化循环:")
        gt_test = gt.clone()
        gt_norm_test = model.normalize(gt_test)
        gt_recon = model.unnormalize(gt_norm_test)
        recon_error = (gt_test - gt_recon).abs().mean().item()
        recon_max_error = (gt_test - gt_recon).abs().max().item()
        
        print(f"    平均误差: {recon_error:.6f}")
        print(f"    最大误差: {recon_max_error:.6f}")
        
        if recon_error > 0.01:
            print("    ⚠️  重建误差过大！normalize/unnormalize 有问题")
        else:
            print("    ✓ GT 的归一化循环正常")

        # ============================================================
        # 修复 std（如果需要）
        # ============================================================
        needs_fix = False
        
        if std_near_zero > 0:
            print(f"\n⚠️  发现问题：有 {std_near_zero} 个接近0的 std 值")
            print(f"    这会导致反归一化时数值爆炸（x * std，当 std→0 时结果正常）")
            print(f"    但如果训练时 std 被错误缩放，会导致问题")
            needs_fix = True
        
        if std_percentiles["min"] < 1e-6:
            print(f"\n⚠️  发现问题：std 最小值 = {std_percentiles['min']:.2e}")
            print(f"    这个值太小，可能导致数值不稳定")
            needs_fix = True
        
        if needs_fix:
            print(f"\n[修复] 将 std clamp 到 [0.001, 100] 范围:")
            model.std_pose = torch.clamp(model.std_pose, min=1e-3, max=100)
            print(f"    修复后 std range: [{model.std_pose.min():.4f}, {model.std_pose.max():.4f}]")
            
            # 重新测试
            gt_norm_test2 = model.normalize(gt_test)
            gt_recon2 = model.unnormalize(gt_norm_test2)
            recon_error2 = (gt_test - gt_recon2).abs().mean().item()
            print(f"    修复后重建误差: {recon_error2:.6f}")

        # ============================================================
        # Unnormalize PRED
        # ============================================================
        print(f"\n[5] 反归一化 PRED:")
        pred = model.unnormalize(pred_norm)
        
        print(f"    PRED shape: {pred.shape}")
        print(f"    PRED range:")
        print(f"      X: [{pred[...,0].min():.4f}, {pred[...,0].max():.4f}]")
        print(f"      Y: [{pred[...,1].min():.4f}, {pred[...,1].max():.4f}]")
        print(f"      Z: [{pred[...,2].min():.4f}, {pred[...,2].max():.4f}]")
        
        print(f"\n    GT range (对比):")
        print(f"      X: [{gt[...,0].min():.4f}, {gt[...,0].max():.4f}]")
        print(f"      Y: [{gt[...,1].min():.4f}, {gt[...,1].max():.4f}]")
        print(f"      Z: [{gt[...,2].min():.4f}, {gt[...,2].max():.4f}]")
        
        # 检查 PRED 是否合理
        pred_range_ok = (
            pred.abs().max() < 10 and
            (pred[...,0].max() - pred[...,0].min()) < 5 and
            (pred[...,1].max() - pred[...,1].min()) < 5
        )
        
        if pred_range_ok:
            print(f"\n    ✓ PRED 数值范围正常")
        else:
            print(f"\n    ⚠️  PRED 数值范围异常！可能需要进一步调试")

        # DTW evaluation
        mask_bt = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred, gt, mask_bt)
        print(f"\n[6] DTW: {dtw_val:.4f}")

    print("="*70 + "\n")

    print("\n" + "="*70)
    print("详细检查 PRED 的关键点分布")
    print("="*70)

    pred_cpu = pred.cpu()
    pred_frame0 = pred_cpu[0, 0]  # 第一帧 [J, 3]

    # 按关键点组分析
    groups = {
        "Pose (身体)": (0, 33),      # MediaPipe Pose: 0-32
        "左手": (33, 54),              # 左手 21 个点: 33-53
        "右手": (54, 75),              # 右手 21 个点: 54-74  
        "面部": (75, 178),             # 面部 103 个点: 75-177
    }

    for name, (start, end) in groups.items():
        points = pred_frame0[start:end]
        
        # 计算范围
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        z_range = points[:, 2].max() - points[:, 2].min()
        
        # 计算中心
        center = points.mean(dim=0)
        
        # 计算标准差（衡量分散程度）
        std = points.std(dim=0)
        
        print(f"\n{name} ({start}-{end-1}, 共 {end-start} 个点):")
        print(f"  X range: {x_range:.2f}, Y range: {y_range:.2f}, Z range: {z_range:.2f}")
        print(f"  中心: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
        print(f"  标准差: [{std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f}]")
        
        # 检查是否有异常
        if x_range > 500 or y_range > 500:
            print(f"  ⚠️  范围异常大！")
        if std[0] > 100 or std[1] > 100:
            print(f"  ⚠️  标准差异常大，点分布很分散")

    print("\n" + "-"*70)
    print("对比 GT 的关键点分布:")
    print("-"*70)
    
    gt_cpu = gt.cpu()
    gt_frame0 = gt_cpu[0, 0]

    for name, (start, end) in groups.items():
        points = gt_frame0[start:end]
        std = points.std(dim=0)
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        print(f"{name}: X_range={x_range:.4f}, Y_range={y_range:.4f}, std=[{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

    print("="*70 + "\n")

    # ============================================================
    # 保存可视化文件
    # ============================================================
    print("\n" + "="*70)
    print("保存可视化文件")
    print("="*70)

    # 方案1：GT 从原始文件读取（最可靠）
    print("\n[1] GT - 从原始文件读取:")
    gt_file_path = base_ds.records[0]["pose"]
    gt_file_path = gt_file_path if os.path.isabs(gt_file_path) else os.path.join(data_dir, gt_file_path)
    
    with open(gt_file_path, "rb") as f:
        gt_from_file = Pose.read(f)
    
    gt_pose_obj = reduce_holistic(gt_from_file)
    gt_pose_obj = gt_pose_obj.remove_components(["POSE_WORLD_LANDMARKS"])
    
    out_gt = os.path.join(out_dir, "gt_final.pose")
    if os.path.exists(out_gt):
        os.remove(out_gt)
    
    with open(out_gt, "wb") as f:
        gt_pose_obj.write(f)
    
    print(f"  保存到: {out_gt}")
    print(f"  ✓ 这个文件应该显示正常的人体姿态")

    # 方案2：PRED 从模型输出转换
    print("\n[2] PRED - 从模型输出转换:")
    pred_cpu = pred.cpu()
    pose_pred = tensor_to_pose(pred_cpu, header)
    
    out_pred = os.path.join(out_dir, "pred_final.pose")
    if os.path.exists(out_pred):
        os.remove(out_pred)
    
    with open(out_pred, "wb") as f:
        pose_pred.write(f)
    
    print(f"  保存到: {out_pred}")
    
    if pred_range_ok:
        print(f"  ✓ PRED 数值正常，应该能显示合理的姿态")
    else:
        print(f"  ⚠️  PRED 数值异常，可能显示不正确")
        print(f"  建议：检查训练过程中的 std calibration")

    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    print(f"\n在 pose viewer 中打开:")
    print(f"  - {out_gt}")
    print(f"  - {out_pred}")
    print(f"\n对比两个文件，评估模型预测效果")