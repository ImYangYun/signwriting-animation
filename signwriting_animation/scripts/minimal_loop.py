# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc, masked_dtw

# ----------- DEBUG: confirm which lightning_module is actually used -----------
import signwriting_animation.diffusion.lightning_module as LM
print(">>> USING LIGHTNING MODULE FROM:", LM.__file__)
# -------------------------------------------------------------------------------


def _to_plain(x):
    """Convert pose-format tensors to contiguous float32 CPU tensor."""
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().cpu().contiguous().float()


def temporal_smooth(x, k=5):
    """Simple temporal smoothing for visualization."""
    import torch.nn.functional as F
    if x.dim() == 4:
        x = x[0]

    T, J, C = x.shape
    x = x.permute(2,1,0).reshape(1, C*J, T)
    x = F.avg_pool1d(x, kernel_size=k, stride=1, padding=k//2)
    x = x.reshape(C, J, T).permute(2,1,0)
    return x.contiguous()


import torch

def visualize_xy_plane_fixed(btjc, name="pose"):
    """
    使用 XY 平面，但修正 Z 轴的异常缩放
    
    根据诊断结果：
    - 肩宽主要在 XY 平面（0.08 单位）
    - 但脖子的 Z 分量异常大（1.7 单位）
    - 需要将 Z 轴缩小约 10 倍来匹配 XY 平面的尺度
    """
    if btjc.dim() == 4:
        x = btjc[0].clone()  # [T, J, 3]
    else:
        x = btjc.clone()
    
    T, J, C = x.shape
    print(f"\n====== {name} 可视化 (XY平面, Z轴修正) ======")
    print(f"原始 shape: {x.shape}")
    print(f"原始 X range: [{x[...,0].min():.4f}, {x[...,0].max():.4f}]")
    print(f"原始 Y range: [{x[...,1].min():.4f}, {x[...,1].max():.4f}]")
    print(f"原始 Z range: [{x[...,2].min():.4f}, {x[...,2].max():.4f}]")
    
    # ===============================================
    # 关键修复：将 Z 轴缩放到与 XY 相同的尺度
    # ===============================================
    # 根据诊断：肩宽在 XY 平面约 0.08，在 Z 轴约 0.15
    # 但脖子在 Z 轴是 1.7，在 XY 平面是 0.28
    # Z 轴被拉伸了约 6 倍
    
    # 计算 XY 和 Z 的尺度比例
    xy_std = x[0, :, :2].std()
    z_std = x[0, :, 2].std()
    
    if z_std > 1e-6:
        z_scale_factor = xy_std / z_std
    else:
        z_scale_factor = 1.0
    
    print(f"XY std: {xy_std:.4f}, Z std: {z_std:.4f}")
    print(f"Z轴缩放因子: {z_scale_factor:.4f}")
    
    # 应用 Z 轴缩放
    x_corrected = x.clone()
    x_corrected[..., 2] = x_corrected[..., 2] * z_scale_factor
    
    print(f"修正后 Z range: [{x_corrected[...,2].min():.4f}, {x_corrected[...,2].max():.4f}]")
    
    # ===============================================
    # 现在使用修正后的数据进行 XY 平面可视化
    # ===============================================
    
    # 1. 提取 XY 平面
    xy = x_corrected[..., :2].clone()  # [T, J, 2]
    
    # 2. 找到有效点
    valid_mask = (xy[0].abs().sum(dim=-1) > 1e-6)
    print(f"有效关键点: {valid_mask.sum()}/{J}")
    
    if valid_mask.sum() == 0:
        print("⚠️ 没有有效点")
        result = torch.zeros_like(x)
        result[..., :2] = xy
        return result.unsqueeze(0)
    
    # 3. 中心化
    xy_valid = xy[0, valid_mask]
    center = xy_valid.mean(dim=0)  # [2]
    xy = xy - center
    print(f"中心点 (XY): [{center[0]:.4f}, {center[1]:.4f}]")
    
    # 4. 计算缩放因子
    dist = torch.norm(xy[0, valid_mask], dim=-1)
    k = max(1, int(len(dist) * 0.95))
    scale_ref = torch.topk(dist, k, largest=False)[0].max()
    
    if scale_ref < 1e-3:
        scale_ref = 1.0
    
    # 目标：让身体占据约 400 像素
    scale_factor = 200 / scale_ref
    xy = xy * scale_factor
    print(f"缩放参考: {scale_ref:.4f}, 缩放因子: {scale_factor:.2f}")
    
    # 5. 翻转 Y 轴（让头在上方）
    xy[..., 1] = -xy[..., 1]
    
    # 6. 平移到屏幕中心
    xy[..., 0] += 512
    xy[..., 1] += 384
    
    # 7. 重新组装成 3D 坐标
    result = torch.zeros_like(x)
    result[..., 0] = xy[..., 0]  # 显示 X
    result[..., 1] = xy[..., 1]  # 显示 Y
    result[..., 2] = x_corrected[..., 2] * scale_factor  # 修正后的 Z
    
    print(f"最终 X range: [{result[...,0].min():.1f}, {result[...,0].max():.1f}]")
    print(f"最终 Y range: [{result[...,1].min():.1f}, {result[...,1].max():.1f}]")
    
    # 检查超出屏幕
    out_x = ((result[..., 0] < 0) | (result[..., 0] > 1024)).sum()
    out_y = ((result[..., 1] < 0) | (result[..., 1] > 768)).sum()
    if out_x > 0 or out_y > 0:
        print(f"⚠️ 超出屏幕: X={out_x}, Y={out_y}")
    
    return result.unsqueeze(0)


def visualize_with_rotation_test(btjc, name="pose"):
    """
    测试不同的坐标轴组合，找到正确的视角
    """
    if btjc.dim() == 4:
        x = btjc[0].clone()
    else:
        x = btjc.clone()
    
    print(f"\n====== {name} 旋转测试 ======")
    
    # 测试 6 种主要视角
    views = {
        "XY平面 (原始)": (0, 1, 2),      # X→X, Y→Y, Z→Z
        "XZ平面 (俯视)": (0, 2, 1),      # X→X, Z→Y, Y→Z
        "YZ平面 (侧视)": (1, 2, 0),      # Y→X, Z→Y, X→Z
        "XY平面 (Y翻转)": (0, 1, 2, True),  # X→X, -Y→Y, Z→Z
        "XZ平面 (Z翻转)": (0, 2, 1, True),  # X→X, -Z→Y, Y→Z
        "ZY平面 (旋转90°)": (2, 1, 0),   # Z→X, Y→Y, X→Z
    }
    
    for view_name, indices in views.items():
        flip_y = len(indices) == 4 and indices[3]
        i, j, k = indices[:3]
        
        # 重新排列坐标轴
        x_view = x.clone()
        x_view_new = torch.zeros_like(x_view)
        x_view_new[..., 0] = x_view[..., i]
        x_view_new[..., 1] = x_view[..., j] if not flip_y else -x_view[..., j]
        x_view_new[..., 2] = x_view[..., k]
        
        # 计算第一帧的空间分布
        valid = (x_view_new[0].abs().sum(dim=-1) > 1e-6)
        if valid.sum() > 0:
            std_x = x_view_new[0, valid, 0].std().item()
            std_y = x_view_new[0, valid, 1].std().item()
            aspect_ratio = std_x / std_y if std_y > 1e-6 else 0
            
            print(f"{view_name:20s} | X_std={std_x:.3f}, Y_std={std_y:.3f}, "
                  f"ratio={aspect_ratio:.2f}")

# 专门为上半身手语数据设计的诊断函数

def analyze_upper_body_structure(btjc, name="pose"):
    """
    分析上半身关键点的空间关系
    
    MediaPipe Holistic 上半身关键点:
    - 0: 鼻子
    - 7, 8: 左耳、右耳
    - 11, 12: 左肩、右肩
    - 13, 14: 左肘、右肘
    - 15, 16: 左手腕、右手腕
    - 33-53: 左手21个点
    - 54-74: 右手21个点
    """
    
    if btjc.dim() == 4:
        x = btjc[0, 0].clone()  # 只看第一帧 [J, 3]
    else:
        x = btjc[0].clone()
    
    print(f"\n{'='*60}")
    print(f"{name} 上半身结构分析")
    print(f"{'='*60}")
    
    # 定义关键点
    keypoints = {
        'nose': 0,
        'left_ear': 7,
        'right_ear': 8,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
    }
    
    print("\n关键点坐标 (原始):")
    for name_key, idx in keypoints.items():
        if idx < x.shape[0]:
            coord = x[idx]
            print(f"  {name_key:15s} [{idx:3d}]: X={coord[0]:7.4f}, Y={coord[1]:7.4f}, Z={coord[2]:7.4f}")
    
    # 计算关键向量
    print("\n关键向量分析:")
    
    # 1. 肩膀向量 (左肩 → 右肩) - 应该是水平的
    shoulder_vec = x[keypoints['right_shoulder']] - x[keypoints['left_shoulder']]
    shoulder_len = shoulder_vec.norm().item()
    print(f"  肩膀向量 (L→R): X={shoulder_vec[0]:7.4f}, Y={shoulder_vec[1]:7.4f}, Z={shoulder_vec[2]:7.4f}")
    print(f"  肩宽: {shoulder_len:.4f}")
    
    shoulder_abs = shoulder_vec.abs()
    shoulder_main = torch.argmax(shoulder_abs).item()
    axis_names = ['X', 'Y', 'Z']
    print(f"  → 肩膀主要沿 {axis_names[shoulder_main]} 轴 (值={shoulder_vec[shoulder_main]:.4f})")
    
    # 2. 脖子向量 (肩膀中心 → 鼻子) - 应该是垂直的
    shoulder_center = (x[keypoints['left_shoulder']] + x[keypoints['right_shoulder']]) / 2
    neck_vec = x[keypoints['nose']] - shoulder_center
    neck_len = neck_vec.norm().item()
    print(f"\n  脖子向量 (shoulder→nose): X={neck_vec[0]:7.4f}, Y={neck_vec[1]:7.4f}, Z={neck_vec[2]:7.4f}")
    print(f"  脖子长度: {neck_len:.4f}")
    
    neck_abs = neck_vec.abs()
    neck_main = torch.argmax(neck_abs).item()
    print(f"  → 脖子主要沿 {axis_names[neck_main]} 轴 (值={neck_vec[neck_main]:.4f})")
    
    # 3. 头部宽度向量 (左耳 → 右耳) - 应该和肩膀平行
    ear_vec = x[keypoints['right_ear']] - x[keypoints['left_ear']]
    ear_len = ear_vec.norm().item()
    print(f"\n  头部向量 (L_ear→R_ear): X={ear_vec[0]:7.4f}, Y={ear_vec[1]:7.4f}, Z={ear_vec[2]:7.4f}")
    print(f"  头宽: {ear_len:.4f}")
    
    # 4. 手臂向量
    left_arm_vec = x[keypoints['left_wrist']] - x[keypoints['left_shoulder']]
    right_arm_vec = x[keypoints['right_wrist']] - x[keypoints['right_shoulder']]
    print(f"\n  左臂向量 (shoulder→wrist): X={left_arm_vec[0]:7.4f}, Y={left_arm_vec[1]:7.4f}, Z={left_arm_vec[2]:7.4f}")
    print(f"  右臂向量 (shoulder→wrist): X={right_arm_vec[0]:7.4f}, Y={right_arm_vec[1]:7.4f}, Z={right_arm_vec[2]:7.4f}")
    
    # 5. 判断正确的坐标系
    print(f"\n{'='*60}")
    print("坐标系判断:")
    print(f"{'='*60}")
    
    # 理想情况：肩膀应该是水平的，脖子应该是垂直的
    if shoulder_main == neck_main:
        print("⚠️  异常：肩膀和脖子在同一个轴上！")
    
    # 判断当前坐标系
    if shoulder_main == 0 and neck_main == 1:
        print("✓ 当前配置：X=水平(左右), Y=垂直(上下)")
        print("  → 这是标准正视图")
        if neck_vec[1] > 0:
            print("  → 头在上方 ✓")
            orientation = "upright"
        else:
            print("  → 头在下方，需要翻转Y轴")
            orientation = "flip_y"
    
    elif shoulder_main == 0 and neck_main == 2:
        print("✗ 肩膀在X轴，脖子在Z轴")
        print("  → 这是俯视/仰视图")
        print("  → 建议：将 Z 映射为 Y (垂直方向)")
        orientation = "use_xz_plane"
    
    elif shoulder_main == 1 and neck_main == 0:
        print("✗ 肩膀在Y轴，脖子在X轴")
        print("  → 坐标系旋转了90度")
        print("  → 建议：交换 X 和 Y")
        orientation = "swap_xy"
    
    elif shoulder_main == 1 and neck_main == 2:
        print("✗ 肩膀在Y轴，脖子在Z轴")
        print("  → 建议：将 Y 映射为 X, Z 映射为 Y")
        orientation = "use_yz_plane"
    
    elif shoulder_main == 2 and neck_main == 0:
        print("✗ 肩膀在Z轴，脖子在X轴")
        print("  → 建议：将 Z 映射为 X, X 映射为 Y")
        orientation = "use_zx_plane"
    
    elif shoulder_main == 2 and neck_main == 1:
        print("✗ 肩膀在Z轴，脖子在Y轴")
        print("  → 建议：将 Z 映射为 X")
        orientation = "use_zy_plane"
    
    else:
        print("⚠️  无法判断，使用默认配置")
        orientation = "default"
    
    print(f"\n推荐配置: {orientation}")
    print(f"{'='*60}\n")
    
    return orientation, shoulder_main, neck_main


def comprehensive_diagnosis(btjc, name="pose"):
    """
    全面诊断数据的空间分布，找出正确的可视化方法
    """
    if btjc.dim() == 4:
        x = btjc[0, 0].clone()  # 只看第一帧 [J, 3]
    else:
        x = btjc[0].clone()
    
    print(f"\n{'='*70}")
    print(f"{name} 全面诊断")
    print(f"{'='*70}")
    
    # 关键点定义
    keypoints = {
        'nose': 0,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_wrist': 15,
        'right_wrist': 16,
    }
    
    print("\n[1] 关键点的原始坐标:")
    for kp_name, idx in keypoints.items():
        coord = x[idx]
        print(f"  {kp_name:15s}: X={coord[0]:7.4f}, Y={coord[1]:7.4f}, Z={coord[2]:7.4f}")
    
    # 计算关键距离
    left_shoulder = x[keypoints['left_shoulder']]
    right_shoulder = x[keypoints['right_shoulder']]
    nose = x[keypoints['nose']]
    left_wrist = x[keypoints['left_wrist']]
    right_wrist = x[keypoints['right_wrist']]
    
    shoulder_center = (left_shoulder + right_shoulder) / 2
    
    # 各个轴上的距离
    print("\n[2] 肩宽在各个轴上的分量:")
    shoulder_diff = right_shoulder - left_shoulder
    print(f"  ΔX = {shoulder_diff[0]:7.4f}")
    print(f"  ΔY = {shoulder_diff[1]:7.4f}")
    print(f"  ΔZ = {shoulder_diff[2]:7.4f}")
    shoulder_width = shoulder_diff.norm().item()
    print(f"  总肩宽 = {shoulder_width:.4f}")
    
    print("\n[3] 脖子长度在各个轴上的分量:")
    neck_diff = nose - shoulder_center
    print(f"  ΔX = {neck_diff[0]:7.4f}")
    print(f"  ΔY = {neck_diff[1]:7.4f}")
    print(f"  ΔZ = {neck_diff[2]:7.4f}")
    neck_length = neck_diff.norm().item()
    print(f"  总脖子长度 = {neck_length:.4f}")
    
    print("\n[4] 左臂长度在各个轴上的分量:")
    left_arm_diff = left_wrist - left_shoulder
    print(f"  ΔX = {left_arm_diff[0]:7.4f}")
    print(f"  ΔY = {left_arm_diff[1]:7.4f}")
    print(f"  ΔZ = {left_arm_diff[2]:7.4f}")
    left_arm_length = left_arm_diff.norm().item()
    print(f"  总左臂长度 = {left_arm_length:.4f}")
    
    # 人体比例检查
    print("\n[5] 人体比例检查:")
    print(f"  脖子长度 / 肩宽 = {neck_length / shoulder_width:.2f}")
    print(f"    (正常人体应该是 0.3-0.5)")
    print(f"  左臂长度 / 肩宽 = {left_arm_length / shoulder_width:.2f}")
    print(f"    (正常人体应该是 1.5-2.0)")
    
    # 测试所有可能的平面投影
    print("\n[6] 测试所有平面投影的人体比例:")
    
    planes = {
        'XY平面': (0, 1),
        'XZ平面': (0, 2),
        'YZ平面': (1, 2),
    }
    
    best_plane = None
    best_score = float('inf')
    
    for plane_name, (axis1, axis2) in planes.items():
        # 在这个平面上重新计算比例
        shoulder_2d = torch.tensor([shoulder_diff[axis1], shoulder_diff[axis2]])
        neck_2d = torch.tensor([neck_diff[axis1], neck_diff[axis2]])
        arm_2d = torch.tensor([left_arm_diff[axis1], left_arm_diff[axis2]])
        
        shoulder_w = shoulder_2d.norm().item()
        neck_l = neck_2d.norm().item()
        arm_l = arm_2d.norm().item()
        
        if shoulder_w > 1e-6:
            neck_ratio = neck_l / shoulder_w
            arm_ratio = arm_l / shoulder_w
        else:
            neck_ratio = 999
            arm_ratio = 999
        
        # 计算与理想比例的偏差
        neck_error = abs(neck_ratio - 0.4)  # 理想脖子/肩宽 = 0.4
        arm_error = abs(arm_ratio - 1.8)    # 理想手臂/肩宽 = 1.8
        total_error = neck_error + arm_error
        
        print(f"\n  {plane_name}:")
        print(f"    肩宽: {shoulder_w:.4f}")
        print(f"    脖子/肩宽: {neck_ratio:.2f} (误差: {neck_error:.2f})")
        print(f"    手臂/肩宽: {arm_ratio:.2f} (误差: {arm_error:.2f})")
        print(f"    总误差: {total_error:.2f}")
        
        if total_error < best_score:
            best_score = total_error
            best_plane = (plane_name, axis1, axis2)
    
    print(f"\n[7] 推荐使用: {best_plane[0]} (轴 {best_plane[1]}, {best_plane[2]})")
    print(f"    总误差最小: {best_score:.2f}")
    
    print(f"\n{'='*70}\n")
    
    return best_plane


def tensor_to_pose(t_btjc, header):
    """Convert tensor → Pose-format object."""
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    elif t_btjc.dim() == 3:
        t = t_btjc
    else:
        raise ValueError

    print("[tensor_to_pose] final shape:", t.shape)

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
    print("mean shape:", stats["mean"].shape)
    print("std shape:", stats["std"].shape)
    print("std min/max:", stats["std"].min(), stats["std"].max())


    # Dataset + reduction (178 joints)
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

    batch0 = next(iter(loader))
    raw = sanitize_btjc(batch0["data"][0:1]).clone().cpu()

    print("====== RAW DATA STATS ======")
    print("raw.min =", raw.min().item(), " raw.max =", raw.max().item())
    print("raw[0, :10] =", raw[0, :10])
    print("RAW shape:", raw.shape)

    num_joints = batch0["data"].shape[-2]
    num_dims   = batch0["data"].shape[-1]
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

    print("[TRAIN] Overfit 4 samples…")
    trainer.fit(model, loader, loader)

    # Load original header (reduced)
    ref_path = base_ds.records[0]["pose"]
    ref_path = ref_path if os.path.isabs(ref_path) else os.path.join(data_dir, ref_path)
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)

    original_reduced = reduce_holistic(ref_pose)
    original_reduced = original_reduced.remove_components(["POSE_WORLD_LANDMARKS"])

    # 保存这个原始 pose
    out_original = os.path.join(out_dir, "original_ref.pose")
    if os.path.exists(out_original):
        os.remove(out_original)
    with open(out_original, "wb") as f:
        original_reduced.write(f)

    print("[TEST] 保存了原始参考 pose 到 original_ref.pose")
    print("      请先在 viewer 中打开这个文件，确认它显示是否正常")
    # ---- correct sequence ----
    ref_p = reduce_holistic(ref_pose)
    ref_p = ref_p.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_p.header

    print("[CHECK HEADER] total joints:", header.total_points())

    print("[CHECK HEADER] total joints:", header.total_points())

    # ============================================================
    # Inference
    # ============================================================

    model.eval()
    device = trainer.strategy.root_device
    model = model.to(device)
    model.mean_pose = model.mean_pose.to(device)
    model.std_pose  = model.std_pose.to(device)

    with torch.no_grad():
        batch = next(iter(loader))
        cond  = batch["conditions"]

        raw_gt = batch["data"][0, 0]
        print("\n====== RAW GT FIRST FRAME (MaskedTensor) ======")
        print(type(raw_gt))

        if hasattr(raw_gt, "zero_filled"):
            dense = raw_gt.zero_filled()
            print("dense[:10] =", dense[:10])
            print("dense min/max =", dense.min(), dense.max())
            print("dense shape =", dense.shape)
        else:
            print("raw_gt[:10] =", raw_gt[:10])


        past = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt   = sanitize_btjc(batch["data"][:1]).to(device)

        future_len = gt.size(1)
        print("[SAMPLE] future_len =", future_len)

        # 1. Generate normalized prediction
        pred_norm = model.sample_autoregressive_fast(
            past_btjc=past,
            sign_img=sign,
            future_len=future_len,
            chunk=1,
        )

        # 2. unnormalize to BTJC
        pred = model.unnormalize(pred_norm)

        # 3. Smoothing (optional)
        #pred_s = temporal_smooth(pred)
        #gt_s   = temporal_smooth(gt)

        # 4. Visualization transform
        #pred_f = visualize_pose(pred, scale=250, offset=(500, 500))
        #gt_f   = visualize_pose(gt,  scale=250, offset=(500, 500))

        #visualize_with_rotation_test(gt, "GT")

        #orientation, shoulder_axis, neck_axis = analyze_upper_body_structure(gt, "GT")

        best_plane_info = comprehensive_diagnosis(gt, "GT")
        print(f"\n最佳平面: {best_plane_info}")
        
        gt_f = visualize_xy_plane_fixed(gt, "GT")
        pred_f = visualize_xy_plane_fixed(pred, "PRED")

        print("gt_f shape:", gt_f.shape)
        print("pred_f shape:", pred_f.shape)

        # --- DTW evaluation ---
        mask_bt = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred, gt, mask_bt)
        print(f"[DTW] masked_dtw (unnormalized) = {dtw_val:.4f}")

    # ============================================================
    # Save .pose for viewer
    # ============================================================

    pose_gt = tensor_to_pose(gt_f, header)
    pose_pr = tensor_to_pose(pred_f, header)

    out_gt = os.path.join(out_dir, "gt_178.pose")
    out_pr = os.path.join(out_dir, "pred_178.pose")

    for p in [out_gt, out_pr]:
        if os.path.exists(p):
            os.remove(p)

    with open(out_gt, "wb") as f:
        pose_gt.write(f)
    with open(out_pr, "wb") as f:
        pose_pr.write(f)

    print("[SAVE] GT & Pred pose saved ✔")
