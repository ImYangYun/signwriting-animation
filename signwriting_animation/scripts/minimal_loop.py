# -*- coding: utf-8 -*-
import os
import torch
import torch.nn.functional as F
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import unshift_hands

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import (
    LitDiffusion, 
    sanitize_btjc, 
    masked_dtw, 
    mean_frame_disp,
)


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
    
    unshift_hands(pose_obj)
    print("  ✓ unshift 成功")

    T_pred = pose_obj.body.data.shape[0]
    T_ref_total = ref_pose.body.data.shape[0]
    
    future_start = max(0, T_ref_total - T_pred)
    ref_arr = np.asarray(ref_pose.body.data[future_start:future_start+T_pred, 0], dtype=np.float32)
    pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)
    
    print(f"  [alignment] ref 用原始文件的帧 {future_start}-{future_start+T_pred-1}")
    
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
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = "logs/4sample_test_fixed"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("4-Sample Overfit Test (Fixed Version)")
    print("=" * 70)

    # 配置
    NUM_SAMPLES = 4
    MAX_EPOCHS = 500
    BATCH_SIZE = 4

    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    print(f"数据集大小: {len(base_ds)}")

    # 选 4 个样本
    class SubsetDataset(torch.utils.data.Dataset):
        def __init__(self, base, indices):
            self.samples = [base[i] for i in indices]
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    train_ds = SubsetDataset(base_ds, list(range(NUM_SAMPLES)))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=zero_pad_collator)

    # 获取维度
    sample = train_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    
    num_joints, num_dims, future_len = sample.shape[-2], sample.shape[-1], sample.shape[0]
    print(f"J={num_joints}, D={num_dims}, T_future={future_len}")

    # 创建模型
    model = LitDiffusion(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,
        diffusion_steps=8,
        vel_weight=1.0,
        t_past=40,
        t_future=future_len,
    )

    # 训练
    print("\n开始训练...")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=50,
    )
    trainer.fit(model, train_loader)

    # Inference
    print("\n" + "=" * 70)
    print("Inference")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_batch = zero_pad_collator([train_ds[0]])
    cond = test_batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(test_batch["data"][:1]).to(device)

    with torch.no_grad():
        pred_raw = model.sample(past_raw, sign, future_len)
        
        # 评估（在归一化空间）
        mse = F.mse_loss(pred_raw, gt_raw).item()
        disp_pred = mean_frame_disp(pred_raw)
        disp_gt = mean_frame_disp(gt_raw)
        disp_ratio = disp_pred / (disp_gt + 1e-8)
        
        # DTW
        mask = torch.ones(1, future_len, device=device)
        dtw = masked_dtw(pred_raw, gt_raw, mask)
        if isinstance(dtw, torch.Tensor):
            dtw = dtw.item()
        
        # MPJPE, PCK（在归一化空间）
        pred_np = pred_raw[0].cpu().numpy()
        gt_np = gt_raw[0].cpu().numpy()
        per_joint_err = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
        mpjpe = per_joint_err.mean()
        pck_01 = (per_joint_err < 0.1).mean() * 100
        pck_02 = (per_joint_err < 0.2).mean() * 100

    print(f"""
结果（归一化空间）:
  MSE: {mse:.6f}
  MPJPE: {mpjpe:.6f}
  PCK@0.1: {pck_01:.1f}%
  PCK@0.2: {pck_02:.1f}%
  DTW: {dtw:.6f}
  Disp GT: {disp_gt:.6f}
  Disp Pred: {disp_pred:.6f}
  Disp Ratio: {disp_ratio:.4f}
""")

    # 保存 Pose 文件
    print("=" * 70)
    print("保存 Pose 文件...")
    print("=" * 70)
    
    ref_path = base_ds.records[0]["pose"]
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    ref_pose = reduce_holistic(ref_pose)
    if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
        ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    
    # ✅ 关键修改：GT 也经过 tensor_to_pose 转换
    print("\n转换 GT (使用 tensor_to_pose):")
    gt_pose = tensor_to_pose(gt_raw, ref_pose.header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
    with open(f"{out_dir}/gt.pose", "wb") as f:
        gt_pose.write(f)
    
    print("\n转换 Pred (使用 tensor_to_pose):")
    pred_pose = tensor_to_pose(pred_raw, ref_pose.header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
    with open(f"{out_dir}/pred.pose", "wb") as f:
        pred_pose.write(f)
    
    print(f"\n✓ 保存到 {out_dir}/")

    # 验证转换后的差异
    print("\n" + "=" * 70)
    print("验证像素空间的差异:")
    print("=" * 70)
    
    gt_data = gt_pose.body.data[:, 0, :, :]
    pred_data = pred_pose.body.data[:, 0, :, :]
    
    pixel_diff = np.abs(gt_data - pred_data)
    pixel_mpjpe = np.sqrt(((gt_data - pred_data) ** 2).sum(-1)).mean()
    
    print(f"  像素空间 MPJPE: {pixel_mpjpe:.2f} 像素")
    print(f"  平均差异: {pixel_diff.mean():.2f} 像素")
    print(f"  最大差异: {pixel_diff.max():.2f} 像素")
    
    # 每帧差异
    per_frame_diff = pixel_diff.mean(axis=(1, 2))
    print(f"\n  每帧平均差异 (前10帧):")
    for i in range(min(10, len(per_frame_diff))):
        print(f"    Frame {i}: {per_frame_diff[i]:.2f} 像素")

    # ========================================================================
    # 手部运动范围诊断
    # ========================================================================
    print("\n" + "=" * 70)
    print("手部运动范围诊断")
    print("=" * 70)
    
    # MediaPipe Holistic 关键点索引
    # 0-32: Pose (33个点)
    # 33-53: Face contour (21个点) 
    # 54-86: Left hand (33个点)
    # 87-119: Right hand (33个点)
    LEFT_HAND_START = 54
    LEFT_HAND_END = 87
    RIGHT_HAND_START = 87
    RIGHT_HAND_END = 120
    
    # 提取左右手
    gt_left_hand = gt_data[:, LEFT_HAND_START:LEFT_HAND_END, :]
    gt_right_hand = gt_data[:, RIGHT_HAND_START:RIGHT_HAND_END, :]
    pred_left_hand = pred_data[:, LEFT_HAND_START:LEFT_HAND_END, :]
    pred_right_hand = pred_data[:, RIGHT_HAND_START:RIGHT_HAND_END, :]
    
    def calc_movement_stats(hand_data, name):
        """计算手部运动统计"""
        # hand_data: [T, 33, 3]
        center = hand_data.mean(axis=(0, 1))  # [3]
        centered = hand_data - center
        variance = (centered ** 2).mean()
        std = np.sqrt(variance)
        
        # 计算每帧的最大位移
        frame_disps = []
        for t in range(1, len(hand_data)):
            disp = np.sqrt(((hand_data[t] - hand_data[t-1]) ** 2).sum(axis=-1)).mean()
            frame_disps.append(disp)
        mean_disp = np.mean(frame_disps) if frame_disps else 0
        
        # X, Y, Z 方向的范围
        x_range = hand_data[:, :, 0].max() - hand_data[:, :, 0].min()
        y_range = hand_data[:, :, 1].max() - hand_data[:, :, 1].min()
        z_range = hand_data[:, :, 2].max() - hand_data[:, :, 2].min()
        
        print(f"\n{name}:")
        print(f"  中心位置: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
        print(f"  Variance: {variance:.2f}")
        print(f"  Std: {std:.2f}")
        print(f"  平均帧间位移: {mean_disp:.2f} 像素")
        print(f"  X范围: {x_range:.2f} 像素")
        print(f"  Y范围: {y_range:.2f} 像素")
        print(f"  Z范围: {z_range:.2f}")
        
        return variance, mean_disp, (x_range, y_range, z_range)
    
    # 左手分析
    print("\n左手运动范围:")
    gt_var_l, gt_disp_l, gt_range_l = calc_movement_stats(gt_left_hand, "  GT 左手")
    pred_var_l, pred_disp_l, pred_range_l = calc_movement_stats(pred_left_hand, "  Pred 左手")
    
    print(f"\n  左手对比:")
    print(f"    Variance 比率 (Pred/GT): {pred_var_l / (gt_var_l + 1e-8):.4f}")
    print(f"    位移比率 (Pred/GT): {pred_disp_l / (gt_disp_l + 1e-8):.4f}")
    print(f"    X范围比率: {pred_range_l[0] / (gt_range_l[0] + 1e-8):.4f}")
    print(f"    Y范围比率: {pred_range_l[1] / (gt_range_l[1] + 1e-8):.4f}")
    
    # 右手分析
    print("\n右手运动范围:")
    gt_var_r, gt_disp_r, gt_range_r = calc_movement_stats(gt_right_hand, "  GT 右手")
    pred_var_r, pred_disp_r, pred_range_r = calc_movement_stats(pred_right_hand, "  Pred 右手")
    
    print(f"\n  右手对比:")
    print(f"    Variance 比率 (Pred/GT): {pred_var_r / (gt_var_r + 1e-8):.4f}")
    print(f"    位移比率 (Pred/GT): {pred_disp_r / (gt_disp_r + 1e-8):.4f}")
    print(f"    X范围比率: {pred_range_r[0] / (gt_range_r[0] + 1e-8):.4f}")
    print(f"    Y范围比率: {pred_range_r[1] / (gt_range_r[1] + 1e-8):.4f}")
    
    # 右手各手指运动分析
    print("\n右手各手指运动分析:")
    finger_names = ["手腕", "拇指", "食指", "中指", "无名指", "小指"]
    finger_ranges = [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 21)]
    
    for finger_name, (start, end) in zip(finger_names, finger_ranges):
        gt_finger = gt_right_hand[:, start:end, :]
        pred_finger = pred_right_hand[:, start:end, :]
        
        gt_disp = np.sqrt(np.diff(gt_finger, axis=0) ** 2).sum(axis=-1).mean()
        pred_disp = np.sqrt(np.diff(pred_finger, axis=0) ** 2).sum(axis=-1).mean()
        
        ratio = pred_disp / (gt_disp + 1e-8)
        print(f"  {finger_name}: GT={gt_disp:.2f}px, Pred={pred_disp:.2f}px, 比率={ratio:.4f}")
    
    # 诊断结论
    print("\n诊断结论:")
    if pred_var_r / (gt_var_r + 1e-8) < 0.8:
        print("  ⚠️ Pred 的手部运动方差明显小于 GT")
        print("     → 模型学到的运动范围偏小，虽然位置准确但'活跃度'不够")
        motion_issue = True
    elif abs(pred_var_r / (gt_var_r + 1e-8) - 1.0) < 0.1:
        print("  ✓ Pred 和 GT 的运动方差接近")
        motion_issue = False
    else:
        print(f"  Pred 和 GT 方差比率: {pred_var_r / (gt_var_r + 1e-8):.4f}")
        motion_issue = abs(pred_var_r / (gt_var_r + 1e-8) - 1.0) > 0.2

    # 结论
    print("\n" + "=" * 70)
    passed_normalized = disp_ratio > 0.5 and pck_01 > 50
    passed_pixel = pixel_mpjpe < 5.0  # 5像素以内认为是完美过拟合
    
    if passed_normalized and passed_pixel:
        print("🎉 4-Sample Overfit 测试完美通过！")
        print(f"   归一化空间: MPJPE={mpjpe:.6f}, PCK@0.1={pck_01:.1f}%")
        print(f"   像素空间: MPJPE={pixel_mpjpe:.2f} 像素")
        
        if motion_issue:
            print(f"\n   ⚠️ 但手部运动范围不匹配:")
            print(f"      右手 Variance 比率: {pred_var_r / (gt_var_r + 1e-8):.4f}")
            print(f"      建议:")
            print(f"      1. 增加 vel_weight 到 5.0")
            print(f"      2. 添加手部运动专门损失")
            print(f"      3. 检查数据归一化是否过度压缩了手部运动")
        else:
            print(f"   ✓ 手部运动范围也匹配！")
    else:
        print("⚠️ 测试未通过")
        if not passed_normalized:
            print("   归一化空间指标不达标")
        if not passed_pixel:
            print(f"   像素空间差异过大: {pixel_mpjpe:.2f} > 5.0 像素")
    print("=" * 70)