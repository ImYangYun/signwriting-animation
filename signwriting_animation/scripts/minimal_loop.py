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
    """
    Convert tensor → Pose-format object.
    
    Args:
        t_btjc: Tensor of shape [B, T, J, C] or [T, J, C]
        header: Pose header
    
    Returns:
        Pose object
    """
    import numpy as np
    from pose_format.numpy.pose_body import NumPyPoseBody
    from pose_format import Pose
    
    # 处理维度
    if t_btjc.dim() == 4:
        # [B, T, J, C] -> [T, J, C]
        t = t_btjc[0]
    elif t_btjc.dim() == 3:
        # [T, J, C]
        t = t_btjc
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {t_btjc.dim()}D")
    
    print(f"  [tensor_to_pose] input shape: {t.shape}")  # 应该是 [T, J, C]
    
    # 检测零点
    zero_mask = (t.abs().sum(dim=-1) < 1e-6)
    num_zeros = zero_mask.sum().item()
    total = zero_mask.numel()
    print(f"  [tensor_to_pose] 零点: {num_zeros}/{total} ({100*num_zeros/total:.1f}%)")
    
    # 转换为 numpy
    t_np = t.cpu().numpy().astype(np.float32)
    print(f"  [tensor_to_pose] numpy shape: {t_np.shape}")
    
    # 检查数据范围
    print(f"  [tensor_to_pose] 数据范围:")
    print(f"    X: [{t_np[:, :, 0].min():.4f}, {t_np[:, :, 0].max():.4f}]")
    print(f"    Y: [{t_np[:, :, 1].min():.4f}, {t_np[:, :, 1].max():.4f}]")
    print(f"    Z: [{t_np[:, :, 2].min():.4f}, {t_np[:, :, 2].max():.4f}]")
    
    # NumPyPoseBody 期望的格式: [frames, people, points, dims]
    # 我们有: [frames, points, dims]
    # 需要添加 people 维度
    arr = t_np[:, None, :, :]  # [T, 1, J, C]
    print(f"  [tensor_to_pose] arr shape after adding people dim: {arr.shape}")
    
    # 置信度
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)
    print(f"  [tensor_to_pose] conf shape: {conf.shape}")
    
    # 创建 body
    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    
    # 验证
    print(f"  [tensor_to_pose] body.data.shape: {body.data.shape}")
    print(f"  [tensor_to_pose] 第一帧第一个点: {body.data[0, 0, 0]}")
    
    return Pose(header=header, body=body)


# ============================================================
# 在 minimal_loop.py 的保存部分替换
# ============================================================

print("\n[2] PRED:")

# 🔍 保存前再次验证
print(f"  pred shape: {pred.shape}")
print(f"  pred[0, 0, 0]: {pred[0, 0, 0]}")  # 第一帧第一个点
print(f"  pred[0, 0, 1]: {pred[0, 0, 1]}")  # 第一帧第二个点

pose_pred = tensor_to_pose(pred, header)

out_pred = os.path.join(out_dir, "pred_final.pose")
with open(out_pred, "wb") as f:
    pose_pred.write(f)

print(f"  保存到: {out_pred}")

# 🔍 验证保存后的文件
print(f"\n  验证保存的文件:")
with open(out_pred, "rb") as f:
    verify_pose = Pose.read(f)

print(f"    读回的 shape: {verify_pose.body.data.shape}")
print(f"    第一帧第一个点: {verify_pose.body.data[0, 0, 0]}")
print(f"    数据范围:")
print(f"      X: [{verify_pose.body.data[:, :, :, 0].min():.4f}, {verify_pose.body.data[:, :, :, 0].max():.4f}]")
print(f"      Y: [{verify_pose.body.data[:, :, :, 1].min():.4f}, {verify_pose.body.data[:, :, :, 1].max():.4f}]")
print(f"      Z: [{verify_pose.body.data[:, :, :, 2].min():.4f}, {verify_pose.body.data[:, :, :, 2].max():.4f}]")


if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    out_dir = "logs/minimal_178_fixed"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    stats = torch.load(stats_path)

    print("\n" + "="*70)
    print("最终修复版本")
    print("="*70)
    print("归一化策略：")
    print("  - DataLoader: 返回原始数据（不归一化）")
    print("  - LightningModule: 使用全局统计量归一化")
    print("  - 结果: 只归一化一次，避免重复压缩")
    print("="*70 + "\n")

    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )

    num_samples = min(200, len(base_ds))
    max_epochs = 20
    print(f"[INFO] 训练配置:")
    print(f"  - 样本数: {num_samples} / {len(base_ds)}")
    print(f"  - Epochs: {max_epochs}")
    print(f"  - Batch size: 8")
    print(f"  - 每个epoch: {num_samples // 8} batches")
    print(f"  - 总训练步数: {(num_samples // 8) * max_epochs}")
    print(f"  - 预计时间: ~30-40 分钟")
    print()

    train_indices = list(range(num_samples))
    train_ds = torch.utils.data.Subset(base_ds, train_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        collate_fn=zero_pad_collator,
    )

    val_indices = list(range(num_samples, min(num_samples + 20, len(base_ds))))
    if len(val_indices) == 0:
        val_indices = list(range(max(0, num_samples - 20), num_samples))
    val_ds = torch.utils.data.Subset(base_ds, val_indices)
    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=zero_pad_collator,
    )

    print("\n" + "="*70)
    print("验证 DataLoader 输出")
    print("="*70)

    test_batch = next(iter(train_loader))
    test_data = test_batch["data"]
    test_data_dense = test_data.tensor if hasattr(test_data, "tensor") else test_data

    print(f"DataLoader 输出统计:")
    print(f"  Shape: {test_data_dense.shape}")
    print(f"  Min: {test_data_dense.min().item():.4f}")
    print(f"  Max: {test_data_dense.max().item():.4f}")
    print(f"  Mean: {test_data_dense.mean().item():.4f}")
    print(f"  Std: {test_data_dense.std().item():.4f}")

    if abs(test_data_dense.mean().item()) < 0.1 and abs(test_data_dense.std().item() - 1.0) < 0.2:
        print("\n❌ 错误：数据已被归一化！")
        raise RuntimeError("DataLoader 不应该返回归一化的数据")
    else:
        print("\n✓ 正确：数据是原始范围（未归一化）")

    print("="*70 + "\n")

    num_joints = base_ds[0]["data"].shape[-2]
    num_dims = base_ds[0]["data"].shape[-1]
    print(f"[INFO] joints={num_joints}, dims={num_dims}")

    # Model
    model = LitMinimal(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=5e-5,
        diffusion_steps=100,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        deterministic=False,
        log_every_n_steps=5,
    )

    print("\n[TRAIN] 开始训练...")
    #trainer.fit(model, train_loader, val_loader)

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
    print("INFERENCE - 最终修复版本")
    print("="*70)

    model.eval()
    device = trainer.strategy.root_device
    model = model.to(device)
    model.mean_pose = model.mean_pose.to(device)
    model.std_pose = model.std_pose.to(device)

    with torch.no_grad():
        print("\n[DEBUG] 创建 inference loader...")
        inference_loader = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=zero_pad_collator,
        )
        
        batch = next(iter(inference_loader))
        cond = batch["conditions"]

        past = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt = sanitize_btjc(batch["data"][:1]).to(device)

        future_len = gt.size(1)
        print(f"\n[1] 基本信息:")
        print(f"    future_len = {future_len}")
        print(f"    GT shape: {gt.shape}")
        
        print(f"\n[2] 输入数据范围检查:")
        print(f"    GT range: [{gt.min():.4f}, {gt.max():.4f}]")
        print(f"    GT mean: {gt.mean():.4f}")
        print(f"    GT std: {gt.std():.4f}")
        
        print(f"\n[2.1] GT 第一帧前 5 个关键点:")
        gt_frame0 = gt[0, 0]
        for i in range(5):
            x, y, z = gt_frame0[i]
            print(f"      关键点 {i}: x={x:.4f}, y={y:.4f}, z={z:.4f}")
        
        if abs(gt.mean().item()) < 0.1:
            print("    ❌ 输入数据已被归一化！")
        else:
            print("    ✓ 输入数据是原始范围")

        # 生成 PRED
        print(f"\n[3] 生成 PRED:")
        pred_norm = model.sample_autoregressive_fast(
            past_btjc=past,
            sign_img=sign,
            future_len=future_len,
            chunk=20,
        )
        
        print(f"    pred_norm shape: {pred_norm.shape}")
        print(f"    pred_norm range: [{pred_norm.min():.4f}, {pred_norm.max():.4f}]")
        print(f"    pred_norm mean: {pred_norm.mean():.4f}")
        print(f"    pred_norm std: {pred_norm.std():.4f}")
        
        print(f"\n[3.1] pred_norm 第一帧前 5 个关键点:")
        pred_norm_frame0 = pred_norm[0, 0]
        for i in range(5):
            x, y, z = pred_norm_frame0[i]
            print(f"      关键点 {i}: x={x:.4f}, y={y:.4f}, z={z:.4f}")

        print(f"\n[4] 反归一化 PRED:")
        pred = model.unnormalize(pred_norm)
        
        print(f"    PRED range:")
        print(f"      X: [{pred[...,0].min():.4f}, {pred[...,0].max():.4f}]")
        print(f"      Y: [{pred[...,1].min():.4f}, {pred[...,1].max():.4f}]")
        print(f"      Z: [{pred[...,2].min():.4f}, {pred[...,2].max():.4f}]")
        
        print(f"\n    GT range (对比):")
        print(f"      X: [{gt[...,0].min():.4f}, {gt[...,0].max():.4f}]")
        print(f"      Y: [{gt[...,1].min():.4f}, {gt[...,1].max():.4f}]")
        print(f"      Z: [{gt[...,2].min():.4f}, {gt[...,2].max():.4f}]")
        
        print(f"\n[4.1] PRED 第一帧前 5 个关键点:")
        pred_frame0 = pred[0, 0]
        for i in range(5):
            x, y, z = pred_frame0[i]
            print(f"      关键点 {i}: x={x:.4f}, y={y:.4f}, z={z:.4f}")
        
        print(f"\n[4.2] GT 第一帧前 5 个关键点 (对比):")
        for i in range(5):
            x, y, z = gt_frame0[i]
            print(f"      关键点 {i}: x={x:.4f}, y={y:.4f}, z={z:.4f}")
        
        pred_x_range = pred[...,0].max() - pred[...,0].min()
        gt_x_range = gt[...,0].max() - gt[...,0].min()
        range_ratio = pred_x_range / gt_x_range
        
        print(f"\n    范围比率 (PRED/GT): X={range_ratio:.4f}")
        if 0.5 < range_ratio < 2.0:
            print(f"    ✓ PRED 数值范围正常")

        print(f"\n[4.3] 关键点唯一性检查:")
        unique_points = torch.unique(pred_frame0, dim=0)
        print(f"    PRED 唯一点: {len(unique_points)} / {pred_frame0.shape[0]}")
        
        if len(unique_points) < 10:
            print(f"    ❌ 几乎所有点都一样！")
        else:
            print(f"    ✓ 关键点有多样性")
        
        zero_mask = (pred_frame0.abs().sum(dim=-1) < 1e-6)
        num_zeros = zero_mask.sum().item()
        print(f"\n[4.4] 零点: {num_zeros} / {pred_frame0.shape[0]}")

        mask_bt = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred, gt, mask_bt)
        print(f"\n[5] DTW: {dtw_val:.4f}")

    print("="*70 + "\n")

    print("\n" + "="*70)
    print("关键点分布检查")
    print("="*70)

    groups = {
        "Pose": (0, 33),
        "左手": (33, 54),
        "右手": (54, 75),
        "面部": (75, 178),
    }

    print("\nPRED:")
    for name, (start, end) in groups.items():
        points = pred_frame0[start:end]
        x_r = points[:, 0].max() - points[:, 0].min()
        y_r = points[:, 1].max() - points[:, 1].min()
        z_r = points[:, 2].max() - points[:, 2].min()
        print(f"  {name}: X={x_r:.4f}, Y={y_r:.4f}, Z={z_r:.4f}")

    print("\nGT:")
    for name, (start, end) in groups.items():
        points = gt_frame0[start:end]
        x_r = points[:, 0].max() - points[:, 0].min()
        y_r = points[:, 1].max() - points[:, 1].min()
        z_r = points[:, 2].max() - points[:, 2].min()
        print(f"  {name}: X={x_r:.4f}, Y={y_r:.4f}, Z={z_r:.4f}")

    print("="*70 + "\n")

    print("\n" + "="*70)
    print("保存文件")
    print("="*70)

    print("\n[1] GT:")
    gt_file_path = base_ds.records[0]["pose"]
    gt_file_path = gt_file_path if os.path.isabs(gt_file_path) else os.path.join(data_dir, gt_file_path)

    with open(gt_file_path, "rb") as f:
        gt_from_file = Pose.read(f)
    
    gt_pose_obj = reduce_holistic(gt_from_file)
    gt_pose_obj = gt_pose_obj.remove_components(["POSE_WORLD_LANDMARKS"])

    out_gt = os.path.join(out_dir, "gt_final.pose")
    with open(out_gt, "wb") as f:
        gt_pose_obj.write(f)

    print(f"  保存: {out_gt}")

    print("\n[2] PRED:")
    pose_pred = tensor_to_pose(pred, header)

    out_pred = os.path.join(out_dir, "pred_final.pose")
    with open(out_pred, "wb") as f:
        pose_pred.write(f)

    print(f"  保存: {out_pred}")

    print("\n" + "="*70)
    print("✓ 完成！")
    print("="*70)
    print(f"\n在 pose viewer 中打开:")
    print(f"  - GT:   {out_gt}")
    print(f"  - PRED: {out_pred}")