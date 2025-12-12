# -*- coding: utf-8 -*-
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
from signwriting_animation.diffusion.lightning_module import LitResidual, sanitize_btjc, masked_dtw, mean_frame_disp


def tensor_to_pose(t_btjc, header, ref_pose, gt_btjc=None, apply_scale=True, fix_abnormal_joints=True):
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
    
    if fix_abnormal_joints and gt_np is not None:
        T, J, C = t_np.shape
        hand_joints = list(range(136, min(178, J)))
        
        for j in hand_joints:
            pred_range = t_np[:, j].max(axis=0) - t_np[:, j].min(axis=0)
            gt_range = gt_np[:, j].max(axis=0) - gt_np[:, j].min(axis=0)
            gt_mean = gt_np[:, j].mean(axis=0)
            
            for c in range(C):
                if pred_range[c] > gt_range[c] * 2.0:
                    max_dev = gt_range[c] * 1.5
                    t_np[:, j, c] = np.clip(t_np[:, j, c], 
                                            gt_mean[c] - max_dev, 
                                            gt_mean[c] + max_dev)
    
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
    out_dir = "logs/multi_sample"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"

    print("\n" + "=" * 70)
    print("多样本训练 (带 Validation)")
    print("=" * 70)

    NUM_TRAIN = 500
    NUM_VAL = 100
    BATCH_SIZE = 8
    MAX_EPOCHS = 100
    
    print(f"\n配置:")
    print(f"  训练样本: 0 ~ {NUM_TRAIN-1}")
    print(f"  验证样本: {NUM_TRAIN} ~ {NUM_TRAIN + NUM_VAL - 1}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {MAX_EPOCHS}")

    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    print(f"数据集总样本数: {len(base_ds)}")
    
    # 训练集
    train_indices = list(range(NUM_TRAIN))
    train_ds = Subset(base_ds, train_indices)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=zero_pad_collator,
        num_workers=4,
        pin_memory=True,
    )
    
    # 验证集
    val_indices = list(range(NUM_TRAIN, NUM_TRAIN + NUM_VAL))
    val_ds = Subset(base_ds, val_indices)
    
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=zero_pad_collator,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"训练集大小: {len(train_ds)}, 每 epoch {len(train_loader)} 个 batch")
    print(f"验证集大小: {len(val_ds)}, 每次 {len(val_loader)} 个 batch")
    
    # 获取维度信息
    sample_data = base_ds[0]["data"]
    if hasattr(sample_data, 'zero_filled'):
        sample_data = sample_data.zero_filled()
    if hasattr(sample_data, 'tensor'):
        sample_data = sample_data.tensor
    
    num_joints = sample_data.shape[-2]
    num_dims = sample_data.shape[-1]
    print(f"关节数: {num_joints}, 维度: {num_dims}")

    # 模型
    model = LitResidual(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-4,
        train_mode="ar",
        vel_weight=1.0,
        acc_weight=0.5,
        residual_scale=0.3,
        hand_reg_weight=2.0,
    )

    # Callbacks
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir,
        filename="best-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=20,
        mode="min",
    )

    # 训练
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        default_root_dir=out_dir,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
    )
    
    # 训练，传入 train_loader 和 val_loader
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\n✓ 最佳模型: {checkpoint_callback.best_model_path}")

    # ===== 测试最佳模型 =====
    print("\n" + "=" * 70)
    print("加载最佳模型进行测试")
    print("=" * 70)
    
    # 加载最佳模型
    best_model = LitResidual.load_from_checkpoint(checkpoint_callback.best_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = best_model.to(device)
    best_model.eval()
    
    # 多样本测试
    test_indices = [50, 100, 200, 300, 400, 550, 650, 750, 850, 950]
    results = []
    
    print("\n泛化测试:")
    print("-" * 60)
    
    for idx in test_indices:
        if idx >= len(base_ds):
            continue
            
        sample = base_ds[idx]
        batch = zero_pad_collator([sample])
        
        cond = batch["conditions"]
        past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt_raw = sanitize_btjc(batch["data"][:1]).to(device)
        
        future_len = gt_raw.size(1)
        
        with torch.no_grad():
            pred_raw = best_model.predict_direct(past_raw, sign, future_len, use_autoregressive=True)
            
            mse = torch.mean((pred_raw - gt_raw) ** 2).item()
            disp_pred = mean_frame_disp(pred_raw)
            disp_gt = mean_frame_disp(gt_raw)
            ratio = disp_pred / (disp_gt + 1e-8)
            
            # DTW
            mask = torch.ones(1, future_len, device=device)
            dtw = masked_dtw(pred_raw, gt_raw, mask).item()
            
            pred_np = pred_raw[0].cpu().numpy()
            gt_np = gt_raw[0].cpu().numpy()
            per_joint_error = np.sqrt(((pred_np - gt_np) ** 2).sum(axis=-1))
            pck = (per_joint_error < 0.1).mean()
            
            in_train = idx < NUM_TRAIN
            results.append({
                'idx': idx,
                'mse': mse,
                'pck': pck,
                'ratio': ratio,
                'dtw': dtw,
                'in_train': in_train
            })
            
            status = "训练" if in_train else "测试"
            print(f"  Sample {idx:4d} ({status}): MSE={mse:.4f}, PCK@0.1={pck:.2%}, DTW={dtw:.4f}, disp_ratio={ratio:.2f}")
    
    # 汇总
    print("-" * 60)
    train_results = [r for r in results if r['in_train']]
    test_results = [r for r in results if not r['in_train']]
    
    if train_results:
        avg_pck_train = np.mean([r['pck'] for r in train_results])
        avg_ratio_train = np.mean([r['ratio'] for r in train_results])
        avg_dtw_train = np.mean([r['dtw'] for r in train_results])
        print(f"训练集: 平均 PCK@0.1={avg_pck_train:.2%}, DTW={avg_dtw_train:.4f}, disp_ratio={avg_ratio_train:.2f}")
    
    if test_results:
        avg_pck_test = np.mean([r['pck'] for r in test_results])
        avg_ratio_test = np.mean([r['ratio'] for r in test_results])
        avg_dtw_test = np.mean([r['dtw'] for r in test_results])
        print(f"测试集: 平均 PCK@0.1={avg_pck_test:.2%}, DTW={avg_dtw_test:.4f}, disp_ratio={avg_ratio_test:.2f}")

    # ===== 保存可视化样本 =====
    print("\n" + "=" * 70)
    print("保存可视化样本")
    print("=" * 70)
    
    # 选一个测试集样本
    vis_idx = 550
    sample = base_ds[vis_idx]
    batch = zero_pad_collator([sample])
    
    cond = batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(batch["data"][:1]).to(device)
    
    with torch.no_grad():
        pred_raw = best_model.predict_direct(past_raw, sign, gt_raw.size(1), use_autoregressive=True)
    
    # 获取参考 pose
    ref_path = base_ds.records[vis_idx]["pose"]
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    
    ref_pose = reduce_holistic(ref_pose)
    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_pose.header
    
    # GT
    out_gt = os.path.join(out_dir, "test_gt.pose")
    with open(out_gt, "wb") as f:
        ref_pose.write(f)
    print(f"✓ GT: {out_gt}")
    
    # Pred
    pose_pred = tensor_to_pose(pred_raw, header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
    out_pred = os.path.join(out_dir, "test_pred.pose")
    with open(out_pred, "wb") as f:
        pose_pred.write(f)
    print(f"✓ PRED: {out_pred}")
    
    print("\n" + "=" * 70)
    print("✓ 训练完成!")
    print("=" * 70)