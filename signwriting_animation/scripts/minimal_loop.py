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


def check_normalization(stats_path):
    """检查归一化统计量"""
    print("\n" + "="*70)
    print("1. 检查归一化统计量")
    print("="*70)
    
    stats = torch.load(stats_path, map_location="cpu")
    mean = stats["mean"].float().view(1, 1, -1, 3)
    std = stats["std"].float().view(1, 1, -1, 3)
    
    print(f"\n[统计量形状]")
    print(f"  mean: {mean.shape}")
    print(f"  std:  {std.shape}")
    
    print(f"\n[统计量范围]")
    print(f"  mean: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  std:  [{std.min():.6f}, {std.max():.4f}]")
    
    print(f"\n[检查异常值]")
    mean_zeros = (mean.abs() < 1e-6).sum().item()
    std_small = (std < 1e-4).sum().item()
    print(f"  mean 接近零: {mean_zeros} / {mean.numel()}")
    print(f"  std < 0.0001: {std_small} / {std.numel()}")
    
    if std_small > 0:
        print(f"\n  ⚠️ 警告：有 {std_small} 个 std 值太小！")
        print(f"  std 最小值: {std.min():.8f}")
        small_indices = torch.where(std.view(-1, 3).min(dim=1)[0] < 1e-4)[0]
        if len(small_indices) > 0:
            print(f"  问题关键点索引: {small_indices[:10].tolist()}")
    else:
        print(f"  ✓ 所有 std 值正常")
    
    print(f"\n[模拟归一化/反归一化]")
    x_raw = torch.randn(1, 20, 178, 3) * 2 - 1
    print(f"  原始数据: [{x_raw.min():.4f}, {x_raw.max():.4f}]")
    
    x_norm = (x_raw - mean) / (std + 1e-6)
    print(f"  归一化后: [{x_norm.min():.4f}, {x_norm.max():.4f}]")
    
    x_unnorm = x_norm * std + mean
    print(f"  反归一化: [{x_unnorm.min():.4f}, {x_unnorm.max():.4f}]")
    
    diff = (x_raw - x_unnorm).abs().max()
    print(f"  重建误差: {diff:.8f}")
    
    if diff < 1e-5:
        print(f"  ✓ 归一化/反归一化正确")
    else:
        print(f"  ✗ 归一化/反归一化有误差")
    
    print("="*70 + "\n")


def tensor_to_pose(t_btjc, header):
    """修复版：正确处理 confidence"""
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    elif t_btjc.dim() == 3:
        t = t_btjc
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {t_btjc.dim()}D")
    
    t_np = t.cpu().numpy().astype(np.float32)
    
    print(f"\n[tensor_to_pose] 详细检查:")
    print(f"  Shape: {t_np.shape}")
    print(f"  数据范围:")
    print(f"    X: [{t_np[:, :, 0].min():.4f}, {t_np[:, :, 0].max():.4f}]")
    print(f"    Y: [{t_np[:, :, 1].min():.4f}, {t_np[:, :, 1].max():.4f}]")
    print(f"    Z: [{t_np[:, :, 2].min():.4f}, {t_np[:, :, 2].max():.4f}]")
    
    # 检查零点
    zero_mask = (np.abs(t_np).sum(axis=-1) < 1e-6)
    num_zeros = zero_mask.sum()
    total = zero_mask.size
    
    print(f"  零点: {num_zeros} / {total}")
    
    if num_zeros > 0:
        print(f"  ⚠️ 发现 {num_zeros} 个零点")
        zero_kps = np.where(zero_mask.any(axis=0))[0]
        print(f"  零点关键点索引: {zero_kps[:20].tolist()}")
    
    # 检查 NaN 和 Inf
    has_nan = np.isnan(t_np).any()
    has_inf = np.isinf(t_np).any()
    print(f"  NaN: {has_nan}, Inf: {has_inf}")
    
    if has_nan or has_inf:
        print(f"  ✗ 数据包含 NaN 或 Inf！")
        return None
    
    # arr: [T, 1, J, C]
    arr = t_np[:, None, :, :]
    
    # conf: [T, 1, J, 1]
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)
    zero_mask_expanded = zero_mask[:, None, :, None]
    conf[zero_mask_expanded] = 0.0
    
    num_visible = (conf > 0).sum()
    print(f"  可见点: {num_visible} / {conf.size}")
    print(f"  不可见点: {(conf == 0).sum()} / {conf.size}")
    
    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    
    return Pose(header=header, body=body)


def analyze_pose(pose_obj, name):
    """分析 pose 对象"""
    print(f"\n{'='*70}")
    print(f"分析: {name}")
    print('='*70)
    
    print(f"\n[Header]")
    print(f"  components: {len(pose_obj.header.components)}")
    print(f"  dimensions: {pose_obj.header.dimensions}")
    
    print(f"\n[Body]")
    print(f"  fps: {pose_obj.body.fps}")
    print(f"  data shape: {pose_obj.body.data.shape}")
    print(f"  confidence shape: {pose_obj.body.confidence.shape}")
    
    data = pose_obj.body.data
    conf = pose_obj.body.confidence
    
    print(f"\n[Data 范围]")
    print(f"  data: [{data.min():.4f}, {data.max():.4f}]")
    non_zero_data = data[data != 0]
    if len(non_zero_data) > 0:
        print(f"  非零数据: [{non_zero_data.min():.4f}, {non_zero_data.max():.4f}]")
    
    print(f"\n[Confidence 分布]")
    print(f"  范围: [{conf.min():.4f}, {conf.max():.4f}]")
    print(f"  =0 的点: {(conf == 0).sum()} / {conf.size}")
    print(f"  =1 的点: {(conf == 1).sum()} / {conf.size}")
    print(f"  (0,1) 的点: {((conf > 0) & (conf < 1)).sum()} / {conf.size}")
    
    # 唯一值
    unique_conf = np.unique(conf)
    print(f"  唯一 confidence 值: {unique_conf[:20].tolist()}")
    
    print(f"\n[零点分布]")
    zero_mask = (data == 0).all(axis=-1)  # [T, P, J]
    zero_per_frame = zero_mask.sum(axis=(1, 2))
    print(f"  每帧零点数: min={zero_per_frame.min()}, max={zero_per_frame.max()}, mean={zero_per_frame.mean():.1f}")


if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    out_dir = "logs/minimal_178_ultimate"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"

    print("\n" + "="*70)
    print("终极完整版诊断流程")
    print("="*70 + "\n")

    # 1. 检查归一化
    check_normalization(stats_path)

    # 2. Dataset
    print("="*70)
    print("2. 加载数据集")
    print("="*70)
    
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )

    print(f"  总样本数: {len(base_ds)}")
    
    sample_0 = base_ds[0]
    
    class FixedSampleDataset(torch.utils.data.Dataset):
        def __init__(self, sample):
            self.sample = sample
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return self.sample
    
    train_ds = FixedSampleDataset(sample_0)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=zero_pad_collator)
    
    print(f"  使用样本: index=0 (固定)")
    print("="*70 + "\n")

    # 3. 训练配置
    print("="*70)
    print("3. 训练配置")
    print("="*70)
    print("  max_epochs: 100")
    print("  lr: 1e-3")
    print("  diffusion_steps: 50")
    print("="*70 + "\n")

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=50,
    )

    num_joints = sample_0["data"].shape[-2]
    num_dims = sample_0["data"].shape[-1]

    model = LitMinimal(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,
        diffusion_steps=50,
        beta_start=1e-4,
        beta_end=2e-2,
        pred_target="x0",
    )

    print(f"\n模型统计量:")
    print(f"  mean_pose: {model.mean_pose.shape}")
    print(f"  std_pose: {model.std_pose.shape}")
    print()

    # 4. 训练
    print("="*70)
    print("4. 开始训练")
    print("="*70 + "\n")
    
    trainer.fit(model, train_loader)

    # 5. Inference
    print("\n" + "="*70)
    print("5. INFERENCE")
    print("="*70)

    model.eval()
    device = trainer.strategy.root_device
    model = model.to(device)

    with torch.no_grad():
        batch = next(iter(train_loader))
        cond = batch["conditions"]

        past = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt = sanitize_btjc(batch["data"][:1]).to(device)

        future_len = gt.size(1)
        
        print(f"\n[采样] diffusion_steps=50, future_len={future_len}")
        
        pred_norm = model.sample_autoregressive_fast(
            past_btjc=past,
            sign_img=sign,
            future_len=future_len,
            chunk=20,
        )

        print(f"\n[反归一化]")
        print(f"  归一化空间: [{pred_norm.min():.4f}, {pred_norm.max():.4f}]")
        
        pred = model.unnormalize(pred_norm)
        
        print(f"  原始空间:   [{pred.min():.4f}, {pred.max():.4f}]")
        print(f"  GT 范围:    [{gt.min():.4f}, {gt.max():.4f}]")
        
        # 检查异常值
        pred_cpu = pred.cpu()
        zero_count = (pred_cpu.abs().sum(dim=-1) < 1e-6).sum().item()
        total_count = pred_cpu.shape[0] * pred_cpu.shape[1] * pred_cpu.shape[2]
        print(f"\n  PRED 零点: {zero_count} / {total_count}")
        
        has_nan = torch.isnan(pred).any().item()
        has_inf = torch.isinf(pred).any().item()
        print(f"  PRED NaN: {has_nan}, Inf: {has_inf}")

        mask_bt = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred, gt, mask_bt)
        print(f"\n  DTW: {dtw_val:.4f}")

    # 6. 保存文件
    print("\n" + "="*70)
    print("6. 保存文件")
    print("="*70)

    ref_path = base_ds.records[0]["pose"]
    ref_path = ref_path if os.path.isabs(ref_path) else os.path.join(data_dir, ref_path)

    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)

    ref_reduced = reduce_holistic(ref_pose)
    ref_reduced = ref_reduced.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_reduced.header

    # GT
    print("\n[保存 GT]")
    gt_pose_obj = reduce_holistic(ref_pose)
    gt_pose_obj = gt_pose_obj.remove_components(["POSE_WORLD_LANDMARKS"])
    out_gt = os.path.join(out_dir, "gt_final.pose")
    with open(out_gt, "wb") as f:
        gt_pose_obj.write(f)
    print(f"  {out_gt}")

    # PRED (原始 confidence)
    print("\n[保存 PRED - 自动 confidence]")
    pose_pred = tensor_to_pose(pred, header)
    
    if pose_pred is None:
        print("  ✗ PRED 包含 NaN/Inf，无法保存")
    else:
        out_pred = os.path.join(out_dir, "pred_auto_conf.pose")
        with open(out_pred, "wb") as f:
            pose_pred.write(f)
        print(f"  {out_pred}")

    # 7. 对比分析
    print("\n" + "="*70)
    print("7. 对比 GT 和 PRED")
    print("="*70)
    
    analyze_pose(gt_pose_obj, "GT")
    
    if pose_pred is not None:
        analyze_pose(pose_pred, "PRED (自动 confidence)")

    # 8. 生成用 GT confidence 的 PRED
    if pose_pred is not None:
        print("\n" + "="*70)
        print("8. 生成用 GT confidence 的 PRED")
        print("="*70)
        
        # 用 GT 的前 20 帧 confidence
        gt_conf_20frames = gt_pose_obj.body.confidence[:20].copy()
        
        print(f"\n[GT confidence (前20帧)]")
        print(f"  shape: {gt_conf_20frames.shape}")
        print(f"  range: [{gt_conf_20frames.min():.4f}, {gt_conf_20frames.max():.4f}]")
        print(f"  唯一值: {np.unique(gt_conf_20frames)[:10].tolist()}")
        
        new_body = NumPyPoseBody(
            fps=25,
            data=pose_pred.body.data,
            confidence=gt_conf_20frames
        )
        
        pred_with_gt_conf = Pose(header=header, body=new_body)
        
        out_pred_gt_conf = os.path.join(out_dir, "pred_with_gt_conf.pose")
        with open(out_pred_gt_conf, "wb") as f:
            pred_with_gt_conf.write(f)
        
        print(f"\n✓ 保存: {out_pred_gt_conf}")
        
        analyze_pose(pred_with_gt_conf, "PRED (GT confidence)")

    # 9. 总结
    print("\n" + "="*70)
    print("✓ 完成！")
    print("="*70)
    
    print(f"\n生成的文件:")
    print(f"  1. GT:                    {out_gt}")
    if pose_pred is not None:
        print(f"  2. PRED (自动 conf):      {out_pred}")
        print(f"  3. PRED (GT conf):        {out_pred_gt_conf}")
    
    print(f"\n测试步骤:")
    print(f"  1. 在 sign.mt 中打开 GT - 应该能正常显示")
    print(f"  2. 打开 pred_auto_conf.pose - 如果不能显示，看看是否是 confidence 问题")
    print(f"  3. 打开 pred_with_gt_conf.pose - 如果这个能显示，说明问题就是 confidence")
    print(f"\n关键对比:")
    print(f"  - GT 和 PRED (自动) 的 confidence 分布")
    print(f"  - GT 的 confidence 可能不是简单的 0/1")