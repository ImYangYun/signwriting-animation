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
    
    # 找理想样本（运动均匀，无异常跳变）
    # 注意：这个数据集运动量普遍很小，降低阈值
    ideal_samples = [s for s in good_samples 
                     if s['stats'].get('jump_ratio', 100) < 10
                     and s['stats'].get('mean_frame_disp', 0) > 0.5]
    
    print(f"\n推荐样本 (运动均匀 >0.5px，无大跳变): {len(ideal_samples)} 个")
    for i, s in enumerate(ideal_samples[:top_k]):
        stats = s['stats']
        print(f"  {i+1}. 样本 {s['idx']}: mean_disp={stats.get('mean_frame_disp', 0):.2f}px, "
              f"jump_ratio={stats.get('jump_ratio', 0):.1f}")
    
    recommended = [s['idx'] for s in ideal_samples[:top_k]]
    if not recommended and good_samples:
        # Fallback: 用任何好样本
        print(f"  (使用任意好样本)")
        recommended = [s['idx'] for s in good_samples[:top_k]]
    
    return recommended, good_samples


# ============================================================
# tensor_to_pose
# ============================================================

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
    
    # 关键修复：gt/pred 是 future 部分，对应原始文件的后 T_pred 帧
    # 不是前 T_pred 帧！
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


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    out_dir = "logs/diffusion_real"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"

    print("\n" + "=" * 70)
    print("真正的 Diffusion 模型测试 (x0 预测 + CFG 采样)")
    print("=" * 70)
    print("使用 CFG (Classifier-Free Guidance) 采样")
    print("=" * 70)

    # ===== 配置 =====
    AUTO_SELECT_SAMPLE = True   # True: 自动筛选好样本, False: 使用 SAMPLE_IDX
    SAMPLE_IDX = 50             # 如果 AUTO_SELECT_SAMPLE=False，使用这个
    NUM_SAMPLES = 1             # 单样本过拟合！
    MAX_EPOCHS = 2000           # 增加 epoch
    DIFFUSION_STEPS = 8         # 师姐用 T=8
    BATCH_SIZE = 1              # 单样本
    
    # 新参数：让模型学习 x_t
    VEL_WEIGHT = 5.0            # 高 velocity weight
    ACC_WEIGHT = 2.0
    COND_DROP_PROB = 0.2        # 20% 丢弃条件
    T_ZERO_PROB = 0.3           # 30% 用 t=0 直接重建
    
    print(f"\n配置:")
    print(f"  AUTO_SELECT_SAMPLE: {AUTO_SELECT_SAMPLE}")
    print(f"  NUM_SAMPLES: {NUM_SAMPLES} (单样本过拟合)")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  MAX_EPOCHS: {MAX_EPOCHS}")
    print(f"  DIFFUSION_STEPS: {DIFFUSION_STEPS}")
    print(f"  VEL_WEIGHT: {VEL_WEIGHT}, ACC_WEIGHT: {ACC_WEIGHT}")
    print(f"  COND_DROP_PROB: {COND_DROP_PROB}, T_ZERO_PROB: {T_ZERO_PROB}")

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

    # ===== 样本选择 =====
    if AUTO_SELECT_SAMPLE:
        recommended, good_samples = find_good_samples(base_ds, data_dir, num_to_check=500)  # 增加搜索数量
        if len(recommended) >= NUM_SAMPLES:
            selected_indices = recommended[:NUM_SAMPLES]
            print(f"\n✓ 自动选择 {NUM_SAMPLES} 个好样本: {selected_indices}")
        else:
            # 不够好样本，用所有好样本 + 一些其他样本
            selected_indices = recommended + list(range(len(recommended), NUM_SAMPLES))
            print(f"\n⚠ 好样本不足，选择: {selected_indices}")
    else:
        # 手动选择，从 SAMPLE_IDX 开始取 NUM_SAMPLES 个
        selected_indices = list(range(SAMPLE_IDX, SAMPLE_IDX + NUM_SAMPLES))
        print(f"\n手动选择样本: {selected_indices}")
    
    # 检查选中样本的质量
    print(f"\n" + "=" * 70)
    print(f"选中样本质量检查")
    print("=" * 70)
    
    for idx in selected_indices[:5]:  # 只显示前5个
        record = base_ds.records[idx]
        pose_path = record["pose"]
        if not os.path.isabs(pose_path):
            pose_path = os.path.join(data_dir, pose_path)
        
        is_good, issues, stats = check_pose_file_quality(pose_path)
        status = "✓" if is_good else "✗"
        mean_disp = stats.get('mean_frame_disp', 0)
        print(f"  {status} 样本 {idx}: mean_disp={mean_disp:.2f}px, issues={issues if issues else 'None'}")

    # 加载样本
    import random
    random.seed(12345)
    
    # 多样本 Dataset
    class MultiSampleDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, indices):
            self.base_dataset = base_dataset
            self.indices = indices
            self.samples = [base_dataset[i] for i in indices]
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    train_ds = MultiSampleDataset(base_ds, selected_indices)
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,  # 多样本时 shuffle
        collate_fn=zero_pad_collator,
    )
    
    print(f"\n训练集大小: {len(train_ds)}, Batch size: {BATCH_SIZE}")
    
    # 获取维度信息（用第一个样本）
    sample_data = train_ds[0]["data"]
    if hasattr(sample_data, 'zero_filled'):
        sample_data = sample_data.zero_filled()
    if hasattr(sample_data, 'tensor'):
        sample_data = sample_data.tensor
    
    num_joints = sample_data.shape[-2]
    num_dims = sample_data.shape[-1]
    print(f"关节数: {num_joints}, 维度: {num_dims}")

    # 创建真正的 Diffusion 模型
    # 注意：现在用 Epsilon 模式
    model = LitDiffusion(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,
        diffusion_steps=DIFFUSION_STEPS,
        residual_scale=0.1,
        vel_weight=VEL_WEIGHT,
        acc_weight=ACC_WEIGHT,
        cond_drop_prob=COND_DROP_PROB,
        t_zero_prob=T_ZERO_PROB,
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

    # Inference - 用第一个样本测试
    print("\n" + "=" * 70)
    print("DIFFUSION INFERENCE (8 步去噪)")
    print("=" * 70)
    
    model = model.to(device)
    model.eval()
    
    # 用第一个样本做测试
    test_sample = train_ds[0]
    test_batch = zero_pad_collator([test_sample])
    
    cond = test_batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(test_batch["data"][:1]).to(device)
    
    future_len = gt_raw.size(1)
    test_idx = selected_indices[0]
    print(f"测试样本: {test_idx}")
    
    with torch.no_grad():
        # Baseline
        baseline = gt_raw.mean(dim=1, keepdim=True).repeat(1, future_len, 1, 1)
        mse_base = torch.mean((baseline - gt_raw) ** 2).item()
        print(f"Baseline MSE: {mse_base:.4f}")
        
        # 1. 标准 p_sample_loop 采样
        print("\n1. 使用 p_sample_loop 采样...")
        pred_raw = model.sample(past_raw, sign, future_len)
        mse_pred = torch.mean((pred_raw - gt_raw) ** 2).item()
        disp_pred = mean_frame_disp(pred_raw)
        print(f"   MSE: {mse_pred:.4f}, disp: {disp_pred:.6f}")
        
        # 2. CFG 采样
        print("\n2. 使用 CFG 采样 (guidance_scale=2.0)...")
        pred_raw_cfg = model.sample_with_cfg(past_raw, sign, future_len, guidance_scale=2.0)
        mse_cfg = torch.mean((pred_raw_cfg - gt_raw) ** 2).item()
        disp_cfg = mean_frame_disp(pred_raw_cfg)
        print(f"   MSE: {mse_cfg:.4f}, disp: {disp_cfg:.6f}")
        
        # 3. 从 GT 加噪声测试（验证去噪能力）
        print("\n3. 从 GT+噪声 开始去噪（验证去噪能力）...")
        gt_norm = model.normalize(gt_raw)
        gt_bjct = model.btjc_to_bjct(gt_norm)
        
        # 加少量噪声 (t=2，较小的噪声)
        t_test = torch.tensor([2], device=device)
        noise = torch.randn_like(gt_bjct)
        x_noisy = model.diffusion.q_sample(gt_bjct, t_test, noise=noise)
        
        # 用模型预测
        past_norm = model.normalize(past_raw)
        past_bjct = model.btjc_to_bjct(past_norm)
        t_scaled = model.diffusion._scale_timesteps(t_test)
        pred_x0 = model.model(x_noisy, t_scaled, past_bjct, sign)
        
        pred_btjc = model.bjct_to_btjc(pred_x0)
        pred_denoise = model.unnormalize(pred_btjc)
        
        mse_denoise = torch.mean((pred_denoise - gt_raw) ** 2).item()
        disp_denoise = mean_frame_disp(pred_denoise)
        print(f"   MSE: {mse_denoise:.4f}, disp: {disp_denoise:.6f}")
        
        disp_gt = mean_frame_disp(gt_raw)
        print(f"\nGT disp: {disp_gt:.6f}")
        print(f"disp_ratio (标准): {disp_pred / (disp_gt + 1e-8):.4f}")
        print(f"disp_ratio (CFG): {disp_cfg / (disp_gt + 1e-8):.4f}")
        print(f"disp_ratio (去噪): {disp_denoise / (disp_gt + 1e-8):.4f}")
        
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
    
    ref_path = base_ds.records[test_idx]["pose"]
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    
    ref_pose = reduce_holistic(ref_pose)
    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_pose.header
    
    # GT：直接从 ref_pose 取后 20 帧
    T_total = ref_pose.body.data.shape[0]
    gt_data = ref_pose.body.data[-20:]
    gt_conf = ref_pose.body.confidence[-20:]
    gt_body = NumPyPoseBody(fps=ref_pose.body.fps, data=gt_data, confidence=gt_conf)
    gt_pose = Pose(header=header, body=gt_body)
    
    out_gt = os.path.join(out_dir, f"gt_{test_idx}.pose")
    with open(out_gt, "wb") as f:
        gt_pose.write(f)
    print(f"\n✓ GT saved: {out_gt}")
    
    # Pred
    pred_pose = tensor_to_pose(pred_raw, header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
    out_pred = os.path.join(out_dir, f"pred_{test_idx}.pose")
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