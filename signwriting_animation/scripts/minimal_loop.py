# -*- coding: utf-8 -*-
"""快速 overfit 验证 - 验证 GT 保存是否正确"""
import os
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader, Subset

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import unshift_hands
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitResidual, sanitize_btjc, mean_frame_disp


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
            pose_obj.body.data *= scale
            pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)
    
    # 平移对齐
    ref_c = ref_arr.reshape(-1, 3).mean(axis=0)
    pred_c = pred_arr.reshape(-1, 3).mean(axis=0)
    delta = ref_c - pred_c
    pose_obj.body.data += delta
    
    return pose_obj


if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    out_dir = "logs/overfit_gt_test"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"

    print("\n" + "=" * 70)
    print("快速 Overfit 验证 - 验证 GT 保存")
    print("=" * 70)

    # 单样本 overfit
    SAMPLE_IDX = 50
    BATCH_SIZE = 1
    MAX_EPOCHS = 1000

    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    # 单样本
    train_ds = Subset(base_ds, [SAMPLE_IDX])
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=zero_pad_collator,
        num_workers=0,
        pin_memory=False,
    )
    
    # 获取维度信息
    sample_data = base_ds[0]["data"]
    if hasattr(sample_data, 'zero_filled'):
        sample_data = sample_data.zero_filled()
    if hasattr(sample_data, 'tensor'):
        sample_data = sample_data.tensor
    
    num_joints = sample_data.shape[-2]
    num_dims = sample_data.shape[-1]
    print(f"关节数: {num_joints}, 维度: {num_dims}")
    print(f"训练样本: {SAMPLE_IDX}")
    print(f"Epochs: {MAX_EPOCHS}")

    # 模型
    model = LitResidual(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,  # overfit 用大学习率
        train_mode="ar",
        vel_weight=0.1,
        acc_weight=0.1,
        residual_scale=1.0,
        hand_reg_weight=2.0,
    )

    # 训练
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        default_root_dir=out_dir,
        log_every_n_steps=1,
        enable_checkpointing=False,
    )
    
    trainer.fit(model, train_loader)

    # ===== 测试并保存 =====
    print("\n" + "=" * 70)
    print("保存可视化样本")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    sample = base_ds[SAMPLE_IDX]
    batch = zero_pad_collator([sample])
    
    cond = batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(batch["data"][:1]).to(device)
    
    print(f"gt_raw shape: {gt_raw.shape}")
    print(f"gt_raw range: [{gt_raw.min().item():.4f}, {gt_raw.max().item():.4f}]")
    
    with torch.no_grad():
        pred_raw = model.predict_direct(past_raw, sign, gt_raw.size(1), use_autoregressive=True)
    
    print(f"pred_raw shape: {pred_raw.shape}")
    print(f"pred_raw range: [{pred_raw.min().item():.4f}, {pred_raw.max().item():.4f}]")
    
    # 计算指标
    mse = torch.mean((pred_raw - gt_raw) ** 2).item()
    disp_pred = mean_frame_disp(pred_raw)
    disp_gt = mean_frame_disp(gt_raw)
    ratio = disp_pred / (disp_gt + 1e-8)
    print(f"\nMSE: {mse:.6f}")
    print(f"disp_ratio: {ratio:.4f}")
    
    # 获取参考 pose
    ref_path = base_ds.records[SAMPLE_IDX]["pose"]
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    
    ref_pose = reduce_holistic(ref_pose)
    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_pose.header
    
    print(f"\nref_pose frames: {ref_pose.body.data.shape[0]}")
    print(f"ref_pose range: [{ref_pose.body.data.min():.4f}, {ref_pose.body.data.max():.4f}]")
    
    # GT - 关键修复：gt_btjc=gt_raw
    gt_pose = tensor_to_pose(gt_raw, header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
    out_gt = os.path.join(out_dir, f"gt_{SAMPLE_IDX}.pose")
    with open(out_gt, "wb") as f:
        gt_pose.write(f)
    print(f"\n✓ GT saved: {out_gt}")
    print(f"  GT pose frames: {gt_pose.body.data.shape[0]}")
    print(f"  GT pose range: [{gt_pose.body.data.min():.4f}, {gt_pose.body.data.max():.4f}]")
    
    # Pred
    pred_pose = tensor_to_pose(pred_raw, header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
    out_pred = os.path.join(out_dir, f"pred_{SAMPLE_IDX}.pose")
    with open(out_pred, "wb") as f:
        pred_pose.write(f)
    print(f"\n✓ PRED saved: {out_pred}")
    print(f"  PRED pose frames: {pred_pose.body.data.shape[0]}")
    print(f"  PRED pose range: [{pred_pose.body.data.min():.4f}, {pred_pose.body.data.max():.4f}]")
    
    print("\n" + "=" * 70)
    print("✓ 完成! 检查 GT 和 PRED 的 frames 和 range 应该相近")
    print("=" * 70)