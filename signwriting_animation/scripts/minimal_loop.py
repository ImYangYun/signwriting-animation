# -*- coding: utf-8 -*-
"""
4-Sample Overfit 测试（真实数据 + MeanPool + Diffusion）

参考 AmitMY 的建议：
"train on like 4 examples from the dataset (the loss will go down fast), 
and then take these samples, and run the inference loop"

测试目的：
- 验证 Diffusion 架构和流程是否正确
- 如果 4 样本能 overfit，说明问题只是数据量

配置：
- NUM_SAMPLES = 4
- USE_MEAN_POOL = True（参考师姐）
- COND_DROP_PROB = 0.0（overfit 不用 dropout）
"""
import os
import sys
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset

from lightning_module import LitDiffusion, sanitize_btjc, masked_dtw, mean_frame_disp

try:
    from pose_anonymization.data.normalization import unshift_hands
    HAS_UNSHIFT = True
except ImportError:
    HAS_UNSHIFT = False


def check_pose_file_quality(pose_path, num_future_frames=20):
    """
    检查 pose 文件质量
    返回: (is_good, issues, stats)
    """
    issues = []
    stats = {}
    
    try:
        with open(pose_path, "rb") as f:
            pose = Pose.read(f)
        
        pose = reduce_holistic(pose)
        if "POSE_WORLD_LANDMARKS" in [c.name for c in pose.header.components]:
            pose = pose.remove_components(["POSE_WORLD_LANDMARKS"])
        
        data = np.array(pose.body.data[:, 0])
        T, J, C = data.shape
        
        stats['total_frames'] = T
        stats['joints'] = J
        
        if T > num_future_frames:
            future_start = T - num_future_frames
            data = data[future_start:]
            T = num_future_frames
        
        stats['checked_frames'] = T
        
    except Exception as e:
        return False, [f"读取失败: {str(e)}"], {}
    
    # 检查手部坐标为 0
    hand_joints = list(range(136, min(178, J)))
    zero_frames = []
    for t in range(T):
        for j in hand_joints:
            if j < J:
                pos = data[t, j]
                if np.abs(pos[0]) < 5.0 and np.abs(pos[1]) < 5.0:
                    zero_frames.append(t)
                    break
    
    if zero_frames:
        issues.append(f"手部坐标为0: 帧{zero_frames}")
    stats['zero_frames'] = len(zero_frames)
    
    # 检查异常跳变
    large_jumps = []
    frame_disps = []
    
    for t in range(1, T):
        diff = np.abs(data[t] - data[t-1])
        frame_disp = diff.mean()
        max_jump = diff.max()
        frame_disps.append(frame_disp)
        
        if max_jump > 100:
            large_jumps.append((t-1, t, max_jump))
    
    if large_jumps:
        issues.append(f"异常跳变(>100px): {len(large_jumps)}处")
    stats['large_jumps'] = len(large_jumps)
    
    # 检查运动幅度
    if frame_disps:
        stats['min_frame_disp'] = min(frame_disps)
        stats['max_frame_disp'] = max(frame_disps)
        stats['mean_frame_disp'] = np.mean(frame_disps)
        
        static_frames = sum(1 for d in frame_disps if d < 0.5)
        stats['static_frames'] = static_frames
        
        if stats['mean_frame_disp'] < 0.5:
            issues.append(f"运动幅度太小: {stats['mean_frame_disp']:.2f}px")
        
        if static_frames > (T - 1) * 0.5:
            issues.append(f"静态帧过多: {static_frames}/{T-1}")
        
        if stats['mean_frame_disp'] > 0:
            jump_ratio = stats['max_frame_disp'] / stats['mean_frame_disp']
            stats['jump_ratio'] = jump_ratio
            if jump_ratio > 20:
                issues.append(f"运动不均匀: max/mean={jump_ratio:.1f}")
    
    is_good = len(issues) == 0
    return is_good, issues, stats


def find_good_samples(dataset, data_dir, num_to_check=200, top_k=4):
    """
    筛选好样本
    """
    print("\n" + "=" * 70)
    print("筛选高质量样本")
    print("=" * 70)
    
    good_samples = []
    num_to_check = min(num_to_check, len(dataset))
    print(f"检查前 {num_to_check} 个样本...")
    
    for i in range(num_to_check):
        try:
            record = dataset.records[i]
            pose_path = record["pose"]
            if not os.path.isabs(pose_path):
                pose_path = os.path.join(data_dir, pose_path)
            
            is_good, issues, stats = check_pose_file_quality(pose_path)
            
            if is_good:
                good_samples.append({
                    'idx': i,
                    'pose_file': os.path.basename(pose_path),
                    'stats': stats
                })
                
            if (i + 1) % 50 == 0:
                print(f"  已检查 {i+1}/{num_to_check}, 好样本: {len(good_samples)}")
                
        except Exception as e:
            pass
    
    print(f"\n✓ 好样本: {len(good_samples)}/{num_to_check}")
    
    # 按运动幅度排序
    good_samples.sort(key=lambda x: x['stats'].get('mean_frame_disp', 0), reverse=True)
    
    # 找理想样本
    ideal_samples = [s for s in good_samples 
                     if s['stats'].get('jump_ratio', 100) < 10
                     and s['stats'].get('mean_frame_disp', 0) > 0.5]
    
    print(f"\n推荐样本 (运动均匀 >0.5px): {len(ideal_samples)} 个")
    for i, s in enumerate(ideal_samples[:top_k]):
        stats = s['stats']
        print(f"  {i+1}. 样本 {s['idx']}: mean_disp={stats.get('mean_frame_disp', 0):.2f}px")
    
    recommended = [s['idx'] for s in ideal_samples[:top_k]]
    if not recommended and good_samples:
        print(f"  (使用任意好样本)")
        recommended = [s['idx'] for s in good_samples[:top_k]]
    
    return recommended, good_samples


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
    
    if HAS_UNSHIFT:
        try:
            unshift_hands(pose_obj)
        except Exception as e:
            pass
    
    T_pred = pose_obj.body.data.shape[0]
    T_ref_total = ref_pose.body.data.shape[0]
    
    future_start = max(0, T_ref_total - T_pred)
    ref_arr = np.asarray(ref_pose.body.data[future_start:future_start+T_pred, 0], dtype=np.float32)
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

    ref_c = ref_arr.reshape(-1, 3).mean(axis=0)
    pred_c = pred_arr.reshape(-1, 3).mean(axis=0)
    delta = ref_c - pred_c
    pose_obj.body.data += delta
    
    return pose_obj


class MultiSampleDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices
        self.samples = [base_dataset[i] for i in indices]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    out_dir = "logs/4sample_overfit"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"

    print("\n" + "=" * 70)
    print("4-Sample Overfit 测试 (MeanPool + Diffusion)")
    print("=" * 70)
    print("参考 AmitMY 建议：4 个真实样本 overfit 测试")
    print("=" * 70)

    # ===== 配置 (4-sample overfit) =====
    NUM_SAMPLES = 4
    MAX_EPOCHS = 200
    DIFFUSION_STEPS = 8
    BATCH_SIZE = 4          # batch = 全部样本
    LR = 1e-3
    
    # 关键配置
    USE_MEAN_POOL = True    # ✅ 使用 MeanPool 模式（参考师姐）
    VEL_WEIGHT = 0.0        # overfit 先不用辅助 loss
    ACC_WEIGHT = 0.0
    COND_DROP_PROB = 0.0    # overfit 不用 dropout
    T_ZERO_PROB = 0.0
    
    print(f"\n配置 (4-Sample Overfit):")
    print(f"  NUM_SAMPLES: {NUM_SAMPLES}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  MAX_EPOCHS: {MAX_EPOCHS}")
    print(f"  DIFFUSION_STEPS: {DIFFUSION_STEPS}")
    print(f"  USE_MEAN_POOL: {USE_MEAN_POOL} ← 参考师姐")
    print(f"  COND_DROP_PROB: {COND_DROP_PROB} ← overfit 不用 dropout")

    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    print(f"\n数据集大小: {len(base_ds)}")

    # 选择 4 个好样本
    recommended, good_samples = find_good_samples(base_ds, data_dir, num_to_check=500, top_k=NUM_SAMPLES)
    
    if len(recommended) >= NUM_SAMPLES:
        selected_indices = recommended[:NUM_SAMPLES]
    else:
        selected_indices = list(range(NUM_SAMPLES))
    
    print(f"\n✓ 选择的 {NUM_SAMPLES} 个样本: {selected_indices}")
    
    # 检查选中样本
    print(f"\n选中样本质量:")
    for idx in selected_indices:
        record = base_ds.records[idx]
        pose_path = record["pose"]
        if not os.path.isabs(pose_path):
            pose_path = os.path.join(data_dir, pose_path)
        
        is_good, issues, stats = check_pose_file_quality(pose_path)
        status = "✓" if is_good else "✗"
        mean_disp = stats.get('mean_frame_disp', 0)
        print(f"  {status} 样本 {idx}: mean_disp={mean_disp:.2f}px")

    # 创建 DataLoader
    train_ds = MultiSampleDataset(base_ds, selected_indices)
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=zero_pad_collator,
    )
    
    # 获取维度信息
    sample_data = train_ds[0]["data"]
    if hasattr(sample_data, 'zero_filled'):
        sample_data = sample_data.zero_filled()
    if hasattr(sample_data, 'tensor'):
        sample_data = sample_data.tensor
    
    num_joints = sample_data.shape[-2]
    num_dims = sample_data.shape[-1]
    print(f"\n关节数: {num_joints}, 维度: {num_dims}")


    model = LitDiffusion(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=LR,
        diffusion_steps=DIFFUSION_STEPS,
        residual_scale=0.1,
        vel_weight=VEL_WEIGHT,
        acc_weight=ACC_WEIGHT,
        cond_drop_prob=COND_DROP_PROB,
        t_zero_prob=T_ZERO_PROB,
        use_mean_pool=USE_MEAN_POOL,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 训练
    print("\n" + "=" * 70)
    print("开始 4-Sample Overfit 训练")
    print("=" * 70)
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )
    
    trainer.fit(model, train_loader)

    print("\n" + "=" * 70)
    print("INFERENCE (p_sample_loop 去噪)")
    print("=" * 70)
    
    model = model.to(device)
    model.eval()

    for sample_idx, test_idx in enumerate(selected_indices):
        print(f"\n--- 测试样本 {sample_idx+1}/{NUM_SAMPLES}: idx={test_idx} ---")
        
        test_sample = train_ds[sample_idx]
        test_batch = zero_pad_collator([test_sample])
        
        cond = test_batch["conditions"]
        past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt_raw = sanitize_btjc(test_batch["data"][:1]).to(device)
        
        future_len = gt_raw.size(1)
        
        with torch.no_grad():
            # p_sample_loop 采样
            pred_raw = model.sample(past_raw, sign, future_len)
            
            mse = torch.mean((pred_raw - gt_raw) ** 2).item()
            disp_pred = mean_frame_disp(pred_raw)
            disp_gt = mean_frame_disp(gt_raw)
            ratio = disp_pred / (disp_gt + 1e-8)
            
            print(f"  MSE: {mse:.6f}")
            print(f"  disp_pred: {disp_pred:.6f}, disp_gt: {disp_gt:.6f}")
            print(f"  disp_ratio: {ratio:.4f} {'✓' if ratio > 0.3 else '✗'}")
    
    # ===== 详细评估第一个样本 =====
    print("\n" + "=" * 70)
    print("详细评估第一个样本")
    print("=" * 70)
    
    test_idx = selected_indices[0]
    test_sample = train_ds[0]
    test_batch = zero_pad_collator([test_sample])
    
    cond = test_batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(test_batch["data"][:1]).to(device)
    future_len = gt_raw.size(1)
    
    with torch.no_grad():
        pred_raw = model.sample(past_raw, sign, future_len)
        
        pred_np = pred_raw[0].cpu().numpy()
        gt_np = gt_raw[0].cpu().numpy()
        T, J, C = pred_np.shape
        
        # Position Errors
        mse = float(((pred_np - gt_np) ** 2).mean())
        mae = float(np.abs(pred_np - gt_np).mean())
        per_joint_error = np.sqrt(((pred_np - gt_np) ** 2).sum(axis=-1))
        mpjpe = float(per_joint_error.mean())
        
        print(f"\n--- Position Errors ---")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  MPJPE: {mpjpe:.6f}")
        
        # Motion Match
        disp_pred = mean_frame_disp(pred_raw)
        disp_gt = mean_frame_disp(gt_raw)
        disp_ratio = disp_pred / (disp_gt + 1e-8)
        
        print(f"\n--- Motion Match ---")
        print(f"  disp_ratio: {disp_ratio:.4f} (理想=1.0)")
        
        # PCK
        print(f"\n--- PCK ---")
        for thresh in [0.05, 0.1, 0.2, 0.5]:
            pck = (per_joint_error < thresh).mean()
            print(f"  PCK@{thresh}: {pck:.2%}")
        
        # DTW
        mask = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred_raw, gt_raw, mask)
        print(f"\n--- Trajectory ---")
        print(f"  DTW: {dtw_val:.4f}")
        
        # Velocity MSE
        pred_vel = pred_np[1:] - pred_np[:-1]
        gt_vel = gt_np[1:] - gt_np[:-1]
        vel_mse = float(((pred_vel - gt_vel) ** 2).mean())
        print(f"  Velocity MSE: {vel_mse:.6f}")
    
    # ===== 保存结果 =====
    print("\n" + "=" * 70)
    print("保存文件")
    print("=" * 70)
    
    ref_path = base_ds.records[test_idx]["pose"]
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    
    ref_pose = reduce_holistic(ref_pose)
    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_pose.header
    
    # GT
    T_total = ref_pose.body.data.shape[0]
    gt_data = ref_pose.body.data[-20:]
    gt_conf = ref_pose.body.confidence[-20:]
    gt_body = NumPyPoseBody(fps=ref_pose.body.fps, data=gt_data, confidence=gt_conf)
    gt_pose = Pose(header=header, body=gt_body)
    
    out_gt = os.path.join(out_dir, f"gt_{test_idx}.pose")
    with open(out_gt, "wb") as f:
        gt_pose.write(f)
    print(f"✓ GT saved: {out_gt}")
    
    # Pred
    pred_pose = tensor_to_pose(pred_raw, header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
    out_pred = os.path.join(out_dir, f"pred_{test_idx}.pose")
    with open(out_pred, "wb") as f:
        pred_pose.write(f)
    print(f"✓ PRED saved: {out_pred}")
    
    # ===== 判断测试结果 =====
    print("\n" + "=" * 70)
    print("4-Sample Overfit 测试结果")
    print("=" * 70)
    
    final_loss = model.train_logs["loss"][-1] if model.train_logs["loss"] else float('inf')
    
    print(f"\n最终 Loss: {final_loss:.6f}")
    print(f"disp_ratio: {disp_ratio:.4f}")
    
    if final_loss < 0.01 and disp_ratio > 0.3:
        print("\n✅ 测试 PASS!")
        print("   - 训练 loss 收敛 (< 0.01)")
        print("   - 预测有运动 (disp_ratio > 0.3)")
        print("   → 架构和 Diffusion 流程正确，问题只是数据量！")
    elif final_loss < 0.01:
        print("\n⚠️ 部分 PASS")
        print("   - 训练 loss 收敛")
        print(f"   - 但 disp_ratio={disp_ratio:.4f} 偏低")
        print("   → 可能需要调整采样参数或增加训练")
    else:
        print("\n❌ 测试 FAIL")
        print(f"   - 训练 loss={final_loss:.6f} 未充分收敛")
        print("   → 检查模型架构或训练配置")
    
    print("\n" + "=" * 70)
    print("✓ 完成!")
    print("=" * 70)