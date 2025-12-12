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
from signwriting_animation.diffusion.lightning_module import LitResidual, sanitize_btjc, masked_dtw, mean_frame_disp

try:
    from pose_anonymization.data.normalization import unshift_hands
    HAS_UNSHIFT = True
except ImportError:
    HAS_UNSHIFT = False


def diagnose_model(model, past_norm, sign_img, device):
    """诊断模型对输入的敏感度"""
    print("\n" + "=" * 70)
    print("诊断：模型对输入的敏感度")
    print("=" * 70)
    
    model.eval()
    
    with torch.no_grad():
        # 测试1：相同输入
        pred1 = model._predict_frames(past_norm, sign_img, 5)
        pred2 = model._predict_frames(past_norm, sign_img, 5)
        diff = (pred1 - pred2).abs().max().item()
        print(f"1. 相同输入两次预测差异: {diff:.8f}")
        
        # 测试2：不同 past
        past_shifted = past_norm.clone()
        past_shifted[:, :, :, 0] += 0.1
        pred_orig = model._predict_frames(past_norm, sign_img, 5)
        pred_shifted = model._predict_frames(past_shifted, sign_img, 5)
        diff = (pred_orig - pred_shifted).abs().mean().item()
        print(f"2. 不同 past 的输出差异: {diff:.6f}")
        if diff < 1e-6:
            print("   ❌ 模型忽略了 past!")
        else:
            print("   ✓ 模型对 past 敏感")
        
        # 测试3：帧间差异
        pred = model._predict_frames(past_norm, sign_img, 10)
        disp = mean_frame_disp(pred)
        print(f"3. 预测帧间 displacement: {disp:.6f}")
        if disp < 1e-6:
            print("   ❌ 预测是静态的!")
        else:
            print("   ✓ 预测有运动")


def tensor_to_pose(t_btjc, header, ref_pose, gt_btjc=None, apply_scale=False, max_scale=500.0, fix_abnormal_joints=True):
    """
    转换 tensor 到 pose 格式
    
    新增：gt_btjc - 如果提供，用 GT 来约束异常关节
    """
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc
    
    t_np = t.detach().cpu().numpy().astype(np.float32)
    
    # 获取 GT 用于约束（如果提供）
    gt_np = None
    if gt_btjc is not None:
        if gt_btjc.dim() == 4:
            gt_np = gt_btjc[0].detach().cpu().numpy().astype(np.float32)
        else:
            gt_np = gt_btjc.detach().cpu().numpy().astype(np.float32)
    
    # 修复异常关节
    if fix_abnormal_joints:
        T, J, C = t_np.shape
        
        # 右手手指关节 (159-177)
        hand_joints = list(range(153, min(178, J)))
        
        for j in hand_joints:
            # 计算该关节的运动范围
            pred_range = t_np[:, j].max(axis=0) - t_np[:, j].min(axis=0)
            
            if gt_np is not None:
                # 使用 GT 的范围作为参考
                gt_range = gt_np[:, j].max(axis=0) - gt_np[:, j].min(axis=0)
                gt_mean = gt_np[:, j].mean(axis=0)
                
                for c in range(C):
                    if pred_range[c] > gt_range[c] * 2.0:  # 超过 GT 范围的 2 倍
                        # 将该关节 clamp 到 GT 范围
                        max_dev = gt_range[c] * 1.5
                        t_np[:, j, c] = np.clip(t_np[:, j, c], 
                                                gt_mean[c] - max_dev, 
                                                gt_mean[c] + max_dev)
            else:
                # 没有 GT 时，用整体范围约束
                overall_mean = t_np.mean(axis=(0, 1))
                overall_std = t_np.std(axis=(0, 1))
                
                for c in range(C):
                    if pred_range[c] > overall_std[c] * 6:  # 超过 6 倍标准差
                        joint_mean = t_np[:, j, c].mean()
                        max_dev = overall_std[c] * 3
                        t_np[:, j, c] = np.clip(t_np[:, j, c],
                                                joint_mean - max_dev,
                                                joint_mean + max_dev)
        
        print(f"  ✓ 异常关节修复完成")
    
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
    
    if apply_scale:
        def _var(a):
            center = a.mean(axis=1, keepdims=True)
            return float(((a - center) ** 2).mean())
        
        var_ref = _var(ref_arr)
        var_pred = _var(pred_arr)
        
        print(f"  [scale] var_ref={var_ref:.2f}, var_pred={var_pred:.6f}")
        
        if var_pred > 1e-8:
            scale = np.sqrt((var_ref + 1e-6) / (var_pred + 1e-6))
            scale = np.clip(scale, 1.0, max_scale)
            print(f"  [scale] apply scale={scale:.2f}")
            pose_obj.body.data *= scale
            pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)
    
    # 平移对齐
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
    out_dir = "logs/minimal_residual"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"

    print("\n" + "=" * 70)
    print("残差预测模型测试")
    print("=" * 70)
    print("核心思想: output = past_last + scale * delta")
    print("模型被强制学习'变化量'而不是'绝对位置'")
    print("=" * 70)

    # ===== 配置 =====
    SAMPLE_IDX = 1000 # 使用样本 50
    MAX_EPOCHS = 1000
    
    print(f"\n配置: 单样本 overfit (sample {SAMPLE_IDX}), epochs={MAX_EPOCHS}")

    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    # 重置随机种子
    import random
    random.seed(12345)
    
    # 获取单样本
    best_sample = base_ds[SAMPLE_IDX]
    print(f"样本 ID: {best_sample.get('id', 'unknown')}")
    
    # 计算样本的 disp
    data = best_sample["data"]
    if hasattr(data, 'zero_filled'):
        data = data.zero_filled()
    if hasattr(data, 'tensor'):
        data = data.tensor
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if data.ndim == 4:
        data = data[0]
    
    gt_disp_raw = float(np.abs(data[1:] - data[:-1]).mean()) if data.shape[0] > 1 else 0
    print(f"样本 future disp (raw): {gt_disp_raw:.4f}")
    
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

    model = LitResidual(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,
        train_mode="ar",
        vel_weight=1.0,
        acc_weight=0.5,
        residual_scale=0.3,
        hand_reg_weight=2.0,
    )

    # 训练前诊断
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    batch = next(iter(train_loader))
    cond = batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    past_norm = model.normalize(past_raw)
    
    print("\n训练前诊断:")
    diagnose_model(model, past_norm, sign, device)

    # 训练
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=10,
    )
    
    trainer.fit(model, train_loader)

    # 训练后诊断
    model = model.to(device)
    model.eval()
    
    batch = next(iter(train_loader))
    cond = batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(batch["data"][:1]).to(device)
    past_norm = model.normalize(past_raw)
    
    print("\n训练后诊断:")
    diagnose_model(model, past_norm, sign, device)

    # Inference
    print("\n" + "=" * 70)
    print("INFERENCE")
    print("=" * 70)
    
    future_len = gt_raw.size(1)
    
    with torch.no_grad():
        # Baseline
        baseline = gt_raw.mean(dim=1, keepdim=True).repeat(1, future_len, 1, 1)
        mse_base = torch.mean((baseline - gt_raw) ** 2).item()
        print(f"Baseline MSE: {mse_base:.4f}")
        
        # 非自回归预测
        pred_raw_direct = model.predict_direct(past_raw, sign, future_len, use_autoregressive=False)
        mse_direct = torch.mean((pred_raw_direct - gt_raw) ** 2).item()
        disp_direct = mean_frame_disp(pred_raw_direct)
        print(f"\nDirect (non-AR) MSE: {mse_direct:.4f}, disp: {disp_direct:.6f}")
        
        # 自回归预测
        pred_raw_ar = model.predict_direct(past_raw, sign, future_len, use_autoregressive=True)
        mse_ar = torch.mean((pred_raw_ar - gt_raw) ** 2).item()
        disp_ar = mean_frame_disp(pred_raw_ar)
        print(f"Autoregressive MSE: {mse_ar:.4f}, disp: {disp_ar:.6f}")
        
        disp_gt = mean_frame_disp(gt_raw)
        print(f"GT disp: {disp_gt:.6f}")
        
        # ===== 完整评估指标 =====
        print("\n" + "=" * 70)
        print("完整评估指标 (normalized 空间)")
        print("=" * 70)
        
        pred_np = pred_raw_ar[0].cpu().numpy()  # [T, J, C]
        gt_np = gt_raw[0].cpu().numpy()
        T, J, C = pred_np.shape
        
        # Position Errors
        mse = float(((pred_np - gt_np) ** 2).mean())
        mae = float(np.abs(pred_np - gt_np).mean())
        per_joint_error = np.sqrt(((pred_np - gt_np) ** 2).sum(axis=-1))  # [T, J]
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
        disp_ratio = disp_ar / (disp_gt + 1e-8)
        
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
        
        if len(abnormal) > 0:
            print(f"  异常关节: {abnormal[:10].tolist()}")
        
        # DTW
        mask = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred_raw_ar, gt_raw, mask)
        print(f"\n--- Trajectory ---")
        print(f"  DTW: {dtw_val:.4f}")
        
        print(f"\n预测范围: [{pred_raw_ar.min():.4f}, {pred_raw_ar.max():.4f}]")
        print(f"GT 范围: [{gt_raw.min():.4f}, {gt_raw.max():.4f}]")
        
        if disp_ar < 1e-6:
            print("\n❌ 预测仍然是静态的!")
        elif disp_ar < disp_gt * 0.1:
            print("\n⚠️ 预测运动太小")
        else:
            print("\n✓ 预测有运动!")

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
    
    out_gt = os.path.join(out_dir, "gt.pose")
    with open(out_gt, "wb") as f:
        ref_pose.write(f)
    print(f"✓ GT: {out_gt}")
    
    # 传入 gt_raw 来约束异常关节
    pose_pred = tensor_to_pose(pred_raw_ar, header, ref_pose, gt_btjc=gt_raw)
    out_pred = os.path.join(out_dir, "pred.pose")
    with open(out_pred, "wb") as f:
        pose_pred.write(f)
    print(f"✓ PRED: {out_pred}")
    
    print("\n✓ 完成!")