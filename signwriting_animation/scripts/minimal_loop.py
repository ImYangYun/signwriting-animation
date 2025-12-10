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


def tensor_to_pose(t_btjc, header, ref_pose, apply_scale=True, max_scale=10.0):
    """
    转换 tensor 到 pose 格式
    
    改进：
    1. 限制最大缩放系数，避免异常放大
    2. 使用 percentile 而不是 variance 来计算缩放
    """
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc
    
    t_np = t.detach().cpu().numpy().astype(np.float32)
    arr = t_np[:, None, :, :]
    conf = ref_pose.body.confidence[:len(t_np)].copy()
    fps = ref_pose.body.fps
    
    body = NumPyPoseBody(fps=fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)
    
    if HAS_UNSHIFT:
        try:
            unshift_hands(pose_obj)
        except:
            pass
    
    T_pred = pose_obj.body.data.shape[0]
    ref_arr = np.asarray(ref_pose.body.data[:T_pred, 0], dtype=np.float32)
    pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)
    
    if apply_scale:
        # 使用 IQR (四分位距) 来计算缩放，更稳健
        def _iqr_scale(a):
            flat = a.reshape(-1)
            q75, q25 = np.percentile(flat, [75, 25])
            return q75 - q25
        
        iqr_ref = _iqr_scale(ref_arr)
        iqr_pred = _iqr_scale(pred_arr)
        
        if iqr_pred > 1e-6:
            scale = iqr_ref / iqr_pred
            # 限制缩放范围
            scale = np.clip(scale, 1.0 / max_scale, max_scale)
            print(f"  [scale] IQR ref={iqr_ref:.2f}, pred={iqr_pred:.2f}, scale={scale:.2f}")
            pose_obj.body.data *= scale
            pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)
    
    # 平移对齐
    ref_c = ref_arr.reshape(-1, 3).mean(axis=0)
    pred_c = pred_arr.reshape(-1, 3).mean(axis=0)
    pose_obj.body.data += (ref_c - pred_c)
    
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
        train_ds, batch_size=1, shuffle=False, collate_fn=zero_pad_collator,
    )

    num_joints = sample_0["data"].shape[-2]
    num_dims = sample_0["data"].shape[-1]

    model = LitResidual(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,
        train_mode="direct",
        vel_weight=1.0,  # 增加速度损失权重
        acc_weight=0.5,
        residual_scale=0.1,  # delta 的缩放
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
        max_epochs=1000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=50,
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
        
        # 详细诊断：检查每个关节的预测范围
        print("\n--- 关节范围诊断 ---")
        pred_np = pred_raw_ar[0].cpu().numpy()  # [T, J, C]
        gt_np = gt_raw[0].cpu().numpy()
        
        # 计算每个关节的运动范围
        pred_ranges = pred_np.max(axis=0) - pred_np.min(axis=0)  # [J, C]
        gt_ranges = gt_np.max(axis=0) - gt_np.min(axis=0)
        
        # 找出异常关节（预测范围远大于 GT）
        ratio = (pred_ranges.sum(axis=1) + 1e-6) / (gt_ranges.sum(axis=1) + 1e-6)
        abnormal = np.where(ratio > 3.0)[0]  # 超过 3 倍的关节
        
        if len(abnormal) > 0:
            print(f"⚠️ 发现 {len(abnormal)} 个异常关节 (预测范围 > 3x GT):")
            for j in abnormal[:10]:
                print(f"  Joint {j}: pred_range={pred_ranges[j].sum():.4f}, gt_range={gt_ranges[j].sum():.4f}, ratio={ratio[j]:.1f}x")
        else:
            print("✓ 所有关节范围正常")
        
        print(f"\n预测范围: [{pred_raw_ar.min():.4f}, {pred_raw_ar.max():.4f}]")
        print(f"GT 范围: [{gt_raw.min():.4f}, {gt_raw.max():.4f}]")
        
        # DTW
        mask = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred_raw_ar, gt_raw, mask)
        print(f"DTW: {dtw_val:.4f}")
        
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
    
    ref_path = base_ds.records[0]["pose"]
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
    
    pose_pred = tensor_to_pose(pred_raw_ar, header, ref_pose)
    out_pred = os.path.join(out_dir, "pred.pose")
    with open(out_pred, "wb") as f:
        pose_pred.write(f)
    print(f"✓ PRED: {out_pred}")
    
    print("\n✓ 完成!")