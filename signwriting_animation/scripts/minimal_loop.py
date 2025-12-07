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

    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
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
    print("INFERENCE (Fixed - No runtime std clamp)")
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
        # 生成 PRED
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
        # 检查模型统计量
        # ============================================================
        print(f"\n[3] 模型的 mean 和 std:")
        print(f"    mean range: [{model.mean_pose.min():.4f}, {model.mean_pose.max():.4f}]")
        print(f"    std range: [{model.std_pose.min():.4f}, {model.std_pose.max():.4f}]")

        std_flat = model.std_pose.flatten()
        print(f"\n    std 分布:")
        print(f"      min: {std_flat.min().item():.6f}")
        print(f"      50%: {torch.quantile(std_flat, 0.5).item():.6f}")
        print(f"      max: {std_flat.max().item():.6f}")

        # ============================================================
        # 验证 GT normalize/unnormalize
        # ============================================================
        print(f"\n[4] 验证 GT 的归一化循环:")
        gt_test = gt.clone()
        gt_norm_test = model.normalize(gt_test)
        gt_recon = model.unnormalize(gt_norm_test)
        recon_error = (gt_test - gt_recon).abs().mean().item()
        
        print(f"    平均误差: {recon_error:.8f}")
        
        if recon_error > 1e-4:
            print("    ⚠️  重建误差过大！")
        else:
            print("    ✓ 归一化循环正确")

        # ============================================================
        # ❌ 不要 clamp std！这是关键修复
        # ============================================================
        # 之前的代码在这里做了 std clamp，导致训练/推理不一致
        # 现在完全移除这个步骤

        # ============================================================
        # Unnormalize PRED
        # ============================================================
        print(f"\n[5] 反归一化 PRED:")
        pred = model.unnormalize(pred_norm)
        
        print(f"    PRED range:")
        print(f"      X: [{pred[...,0].min():.4f}, {pred[...,0].max():.4f}]")
        print(f"      Y: [{pred[...,1].min():.4f}, {pred[...,1].max():.4f}]")
        print(f"      Z: [{pred[...,2].min():.4f}, {pred[...,2].max():.4f}]")
        
        print(f"\n    GT range (对比):")
        print(f"      X: [{gt[...,0].min():.4f}, {gt[...,0].max():.4f}]")
        print(f"      Y: [{gt[...,1].min():.4f}, {gt[...,1].max():.4f}]")
        print(f"      Z: [{gt[...,2].min():.4f}, {gt[...,2].max():.4f}]")
        
        # 检查范围是否匹配
        pred_x_range = pred[...,0].max() - pred[...,0].min()
        gt_x_range = gt[...,0].max() - gt[...,0].min()
        range_ratio = pred_x_range / gt_x_range
        
        print(f"\n    范围比率 (PRED/GT):")
        print(f"      X: {range_ratio:.4f}")
        
        if 0.5 < range_ratio < 2.0:
            print(f"    ✓ PRED 数值范围正常（与 GT 接近）")
        else:
            print(f"    ⚠️  PRED 数值范围异常（比率应接近 1.0）")

        # DTW evaluation
        mask_bt = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred, gt, mask_bt)
        print(f"\n[6] DTW: {dtw_val:.4f}")

    print("="*70 + "\n")

    # ============================================================
    # 详细检查关键点分布
    # ============================================================
    print("\n" + "="*70)
    print("详细检查 PRED 的关键点分布")
    print("="*70)

    pred_cpu = pred.cpu()
    pred_frame0 = pred_cpu[0, 0]

    groups = {
        "Pose (身体)": (0, 33),
        "左手": (33, 54),
        "右手": (54, 75),
        "面部": (75, 178),
    }

    for name, (start, end) in groups.items():
        points = pred_frame0[start:end]
        
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        
        print(f"\n{name}:")
        print(f"  X range: {x_range:.4f}, Y range: {y_range:.4f}")

    print("\n" + "-"*70)
    print("对比 GT:")
    print("-"*70)
    
    gt_cpu = gt.cpu()
    gt_frame0 = gt_cpu[0, 0]

    for name, (start, end) in groups.items():
        points = gt_frame0[start:end]
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        print(f"{name}: X_range={x_range:.4f}, Y_range={y_range:.4f}")

    print("="*70 + "\n")

    # ============================================================
    # 保存文件
    # ============================================================
    print("\n" + "="*70)
    print("保存可视化文件")
    print("="*70)

    # GT
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
    
    print(f"  保存到: {out_gt}")

    print("\n[2] PRED (直接保存，不缩放):")
    pose_pred = tensor_to_pose(pred, header)
    
    out_pred = os.path.join(out_dir, "pred_final.pose")
    with open(out_pred, "wb") as f:
        pose_pred.write(f)
    
    print(f"  保存到: {out_pred}")

    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    print(f"\n在 pose viewer 中打开:")
    print(f"  - {out_gt}")
    print(f"  - {out_pred}")