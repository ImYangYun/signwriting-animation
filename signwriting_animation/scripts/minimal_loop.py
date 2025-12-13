# -*- coding: utf-8 -*-
"""
真正的 Diffusion 测试 - T=8 步，cosine schedule，预测 x0
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

from signwriting_animation.diffusion.lightning_module import LitDiffusion, sanitize_btjc, masked_dtw, mean_frame_disp

try:
    from pose_anonymization.data.normalization import unshift_hands
    HAS_UNSHIFT = True
except ImportError:
    HAS_UNSHIFT = False



def check_pose_file_quality(pose_path, num_future_frames=20):
    """
    直接读取 pose 文件检查质量（pixel 空间）
    
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
        
        data = np.array(pose.body.data[:, 0])  # [T, J, C]
        T, J, C = data.shape
        
        stats['total_frames'] = T
        stats['joints'] = J
        
        # 只检查后 num_future_frames 帧
        if T > num_future_frames:
            future_start = T - num_future_frames
            data = data[future_start:]
            T = num_future_frames
        
        stats['checked_frames'] = T
        
    except Exception as e:
        return False, [f"读取失败: {str(e)}"], {}
    
    # 检查 1: 手部坐标为 0（追踪丢失）
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
    
    # 检查 2: 异常跳变
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
    
    # 检查 3: 运动幅度
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


def find_good_samples(dataset, data_dir, num_to_check=200, top_k=10):
    """
    筛选好样本，返回推荐的样本索引
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
    
    # 找理想样本（运动均匀）
    ideal_samples = [s for s in good_samples 
                     if s['stats'].get('jump_ratio', 100) < 10
                     and s['stats'].get('mean_frame_disp', 0) > 1.0]
    
    print("\n推荐样本 (运动均匀，无大跳变):")
    for i, s in enumerate(ideal_samples[:top_k]):
        stats = s['stats']
        print(f"  {i+1}. 样本 {s['idx']}: mean_disp={stats.get('mean_frame_disp', 0):.2f}px, "
              f"jump_ratio={stats.get('jump_ratio', 0):.1f}")
    
    recommended = [s['idx'] for s in ideal_samples[:top_k]]
    if not recommended and good_samples:
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
            print("  ✓ unshift 成功")
        except Exception as e:
            print(f"  ✗ unshift 失败: {e}")
    
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
    out_dir = "logs/diffusion_real"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"

    print("\n" + "=" * 70)
    print("真正的 Diffusion 模型测试 (带样本质量检查)")
    print("=" * 70)
    print("参考师姐论文：T=8 步, cosine schedule, 预测 x0")
    print("=" * 70)

    # ===== 配置 =====
    AUTO_SELECT_SAMPLE = True   # True: 自动筛选好样本, False: 使用 SAMPLE_IDX
    SAMPLE_IDX = 50             # 如果 AUTO_SELECT_SAMPLE=False，使用这个
    MAX_EPOCHS = 1000
    DIFFUSION_STEPS = 8
    
    print(f"\n配置:")
    print(f"  AUTO_SELECT_SAMPLE: {AUTO_SELECT_SAMPLE}")
    print(f"  SAMPLE_IDX: {SAMPLE_IDX}")
    print(f"  MAX_EPOCHS: {MAX_EPOCHS}")
    print(f"  DIFFUSION_STEPS: {DIFFUSION_STEPS}")

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

    if AUTO_SELECT_SAMPLE:
        recommended, good_samples = find_good_samples(base_ds, data_dir, num_to_check=200)
        if recommended:
            SAMPLE_IDX = recommended[0]
            print(f"\n✓ 自动选择样本: {SAMPLE_IDX}")
        else:
            print(f"\n⚠ 没有找到理想样本，使用默认: {SAMPLE_IDX}")

    record = base_ds.records[SAMPLE_IDX]
    pose_path = record["pose"]
    if not os.path.isabs(pose_path):
        pose_path = os.path.join(data_dir, pose_path)
    
    is_good, issues, stats = check_pose_file_quality(pose_path)
    
    print(f"\n" + "=" * 70)
    print(f"选中样本 {SAMPLE_IDX} 的质量检查")
    print("=" * 70)
    print(f"文件: {os.path.basename(pose_path)}")
    print(f"质量: {'✓ 良好' if is_good else '✗ 有问题'}")
    if issues:
        print(f"问题: {issues}")
    print(f"统计:")
    print(f"  帧数: {stats.get('checked_frames', 0)}")
    print(f"  平均帧间位移: {stats.get('mean_frame_disp', 0):.2f} px")
    print(f"  最大帧间位移: {stats.get('max_frame_disp', 0):.2f} px")
    print(f"  跳变比例 (max/mean): {stats.get('jump_ratio', 0):.1f}")
    print(f"  零坐标帧: {stats.get('zero_frames', 0)}")
    print(f"  异常跳变: {stats.get('large_jumps', 0)}")

    # 加载样本
    import random
    random.seed(12345)
    
    best_sample = base_ds[SAMPLE_IDX]
    print(f"\n样本 ID: {best_sample.get('id', 'unknown')}")
    
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

    # 创建真正的 Diffusion 模型
    model = LitDiffusion(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,
        diffusion_steps=DIFFUSION_STEPS,
        residual_scale=0.1,
        vel_weight=0.5,
        acc_weight=0.2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 训练
    print("\n" + "=" * 70)
    print("开始 Diffusion 训练")
    print("=" * 70)
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=10,
    )
    
    trainer.fit(model, train_loader)

    # Inference
    print("\n" + "=" * 70)
    print("DIFFUSION INFERENCE (8 步去噪)")
    print("=" * 70)
    
    model = model.to(device)
    model.eval()
    
    batch = next(iter(train_loader))
    cond = batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(batch["data"][:1]).to(device)
    
    future_len = gt_raw.size(1)
    
    with torch.no_grad():
        # Baseline
        baseline = gt_raw.mean(dim=1, keepdim=True).repeat(1, future_len, 1, 1)
        mse_base = torch.mean((baseline - gt_raw) ** 2).item()
        print(f"Baseline MSE: {mse_base:.4f}")
        
        # 使用 p_sample_loop 采样（正确的方式）
        print("\n使用 p_sample_loop 采样...")
        pred_raw = model.sample(past_raw, sign, future_len)
        mse_pred = torch.mean((pred_raw - gt_raw) ** 2).item()
        disp_pred = mean_frame_disp(pred_raw)
        print(f"DDPM MSE: {mse_pred:.4f}, disp: {disp_pred:.6f}")
        
        # 也可以用 DDIM（如果支持）
        print("\n使用 DDIM 采样...")
        pred_raw_ddim = model.sample_ddim(past_raw, sign, future_len)
        mse_ddim = torch.mean((pred_raw_ddim - gt_raw) ** 2).item()
        disp_ddim = mean_frame_disp(pred_raw_ddim)
        print(f"DDIM MSE: {mse_ddim:.4f}, disp: {disp_ddim:.6f}")
        
        disp_gt = mean_frame_disp(gt_raw)
        print(f"\nGT disp: {disp_gt:.6f}")
        print(f"disp_ratio (DDPM): {disp_pred / (disp_gt + 1e-8):.4f}")
        print(f"disp_ratio (DDIM): {disp_ddim / (disp_gt + 1e-8):.4f}")
        
        # ===== 完整评估指标 =====
        print("\n" + "=" * 70)
        print("完整评估指标 (normalized 空间)")
        print("=" * 70)
        
        pred_np = pred_raw[0].cpu().numpy()
        gt_np = gt_raw[0].cpu().numpy()
        T, J, C = pred_np.shape
        
        # Position Errors
        mse = float(((pred_np - gt_np) ** 2).mean())
        mae = float(np.abs(pred_np - gt_np).mean())
        per_joint_error = np.sqrt(((pred_np - gt_np) ** 2).sum(axis=-1))
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
        disp_ratio = disp_pred / (disp_gt + 1e-8)
        
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
        
        # DTW
        mask = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred_raw, gt_raw, mask)
        print(f"\n--- Trajectory ---")
        print(f"  DTW: {dtw_val:.4f}")
        
        print(f"\n预测范围: [{pred_raw.min():.4f}, {pred_raw.max():.4f}]")
        print(f"GT 范围: [{gt_raw.min():.4f}, {gt_raw.max():.4f}]")

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
    
    # GT
    gt_pose = tensor_to_pose(gt_raw, header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
    out_gt = os.path.join(out_dir, f"gt_{SAMPLE_IDX}.pose")
    with open(out_gt, "wb") as f:
        gt_pose.write(f)
    print(f"\n✓ GT saved: {out_gt}")
    
    # Pred
    pred_pose = tensor_to_pose(pred_raw, header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
    out_pred = os.path.join(out_dir, f"pred_{SAMPLE_IDX}.pose")
    with open(out_pred, "wb") as f:
        pred_pose.write(f)
    print(f"✓ PRED saved: {out_pred}")
    
    # ===== 分析保存后的 pose 文件 =====
    print("\n" + "=" * 70)
    print("分析保存的 pose 文件 (pixel 空间)")
    print("=" * 70)
    
    with open(out_gt, "rb") as f:
        saved_gt = Pose.read(f)
    with open(out_pred, "rb") as f:
        saved_pred = Pose.read(f)
    
    gt_data = np.array(saved_gt.body.data[:, 0])
    pred_data = np.array(saved_pred.body.data[:, 0])
    
    print(f"\nGT 逐帧位移 (pixel):")
    for t in range(1, min(10, len(gt_data))):
        d = np.abs(gt_data[t] - gt_data[t-1]).mean()
        print(f"  帧 {t-1}→{t}: {d:.2f} px")
    
    print(f"\nPRED 逐帧位移 (pixel):")
    for t in range(1, min(10, len(pred_data))):
        d = np.abs(pred_data[t] - pred_data[t-1]).mean()
        print(f"  帧 {t-1}→{t}: {d:.2f} px")
    
    gt_disp_px = np.abs(gt_data[1:] - gt_data[:-1]).mean()
    pred_disp_px = np.abs(pred_data[1:] - pred_data[:-1]).mean()
    
    print(f"\n总结 (pixel 空间):")
    print(f"  GT 平均帧间位移: {gt_disp_px:.2f} px")
    print(f"  PRED 平均帧间位移: {pred_disp_px:.2f} px")
    print(f"  disp_ratio (pixel): {pred_disp_px / (gt_disp_px + 1e-8):.4f}")
    
    print("\n" + "=" * 70)
    print("✓ 完成! Diffusion 版本测试结束")
    print("=" * 70)