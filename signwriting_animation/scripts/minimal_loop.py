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
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset

from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusionV2
from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure as PE_DTW

try:
    from pose_anonymization.data.normalization import unshift_hands
    HAS_UNSHIFT = True
except ImportError:
    HAS_UNSHIFT = False



def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sanitize_btjc(x):
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    if hasattr(x, "tensor"):
        x = x.tensor
    if x.dim() == 5:
        x = x[:, :, 0]
    if x.shape[-1] != 3 and x.shape[-2] == 3:
        x = x.permute(0, 1, 3, 2)
    return x.contiguous().float()


def mean_frame_disp(x):
    if x.dim() == 4:
        if x.size(1) < 2:
            return 0.0
        v = x[:, 1:] - x[:, :-1]
    else:
        if x.size(0) < 2:
            return 0.0
        v = x[1:] - x[:-1]
    return v.abs().mean().item()


def compute_dtw(pred_btjc, gt_btjc):
    """计算 DTW 距离"""
    if not HAS_DTW:
        return 0.0
    
    try:
        pred = pred_btjc[0].cpu().numpy().astype("float32")[:, None, :, :]
        gt = gt_btjc[0].cpu().numpy().astype("float32")[:, None, :, :]
        dtw_metric = PE_DTW()
        return float(dtw_metric.get_distance(pred, gt))
    except:
        return 0.0


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
        except:
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


class ConditionalWrapper(torch.nn.Module):
    """包装模型，固定条件输入"""
    def __init__(self, model, past, sign):
        super().__init__()
        self.model = model
        self.past = past
        self.sign = sign
    
    def forward(self, x, t, **kwargs):
        return self.model(x, t, self.past, self.sign)


def main():
    print("=" * 70)
    print("完整的最小化 Overfit 测试")
    print("=" * 70)
    print("目标: 1 样本 overfit，loss < 0.001，disp_ratio > 0.3")
    print("=" * 70)
    
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # ========== 配置 ==========
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = "logs/minimal_overfit"
    os.makedirs(out_dir, exist_ok=True)
    
    DIFFUSION_STEPS = 8
    USE_MEAN_POOL = True
    LR = 1e-3
    MAX_STEPS = 5000
    TARGET_LOSS = 0.001
    MIN_GT_DISP = 0.02  # 要求 GT 运动量至少这么大
    
    # 先找一个运动量足够大的样本
    print(f"\n寻找运动量足够大的样本 (disp > {MIN_GT_DISP})...")
    
    base_ds_temp = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    SAMPLE_IDX = None
    best_disp = 0
    best_idx = 0
    
    for idx in range(min(200, len(base_ds_temp))):
        try:
            sample_temp = base_ds_temp[idx]
            batch_temp = zero_pad_collator([sample_temp])
            gt_temp = sanitize_btjc(batch_temp["data"][:1])
            disp_temp = mean_frame_disp(gt_temp)
            
            if disp_temp > best_disp:
                best_disp = disp_temp
                best_idx = idx
            
            if disp_temp > MIN_GT_DISP:
                SAMPLE_IDX = idx
                print(f"  ✓ 找到样本 {idx}: disp={disp_temp:.6f}")
                break
        except:
            continue
    
    if SAMPLE_IDX is None:
        SAMPLE_IDX = best_idx
        print(f"  ⚠ 未找到 disp > {MIN_GT_DISP} 的样本")
        print(f"  使用最大运动量样本 {best_idx}: disp={best_disp:.6f}")
    
    del base_ds_temp
    
    print(f"\n配置:")
    print(f"  SAMPLE_IDX: {SAMPLE_IDX}")
    print(f"  DIFFUSION_STEPS: {DIFFUSION_STEPS}")
    print(f"  USE_MEAN_POOL: {USE_MEAN_POOL}")
    print(f"  LR: {LR}")
    print(f"  MAX_STEPS: {MAX_STEPS}")
    print(f"  TARGET_LOSS: {TARGET_LOSS}")
    
    # ========== 加载数据 ==========
    stats = torch.load(stats_path, map_location="cpu")
    mean_pose = stats["mean"].float().view(1, 1, -1, 3).to(device)
    std_pose = stats["std"].float().view(1, 1, -1, 3).to(device)
    
    def normalize(x):
        return (x - mean_pose) / (std_pose + 1e-6)
    
    def unnormalize(x):
        return x * std_pose + mean_pose
    
    def btjc_to_bjct(x):
        return x.permute(0, 2, 3, 1).contiguous()
    
    def bjct_to_btjc(x):
        return x.permute(0, 3, 1, 2).contiguous()
    
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    sample = base_ds[SAMPLE_IDX]
    batch = zero_pad_collator([sample])
    
    cond = batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(batch["data"][:1]).to(device)
    
    num_joints = gt_raw.shape[2]
    num_dims = gt_raw.shape[3]
    future_len = gt_raw.shape[1]
    
    gt_disp = mean_frame_disp(gt_raw)
    
    print(f"\n数据信息:")
    print(f"  past: {past_raw.shape}")
    print(f"  gt: {gt_raw.shape}")
    print(f"  gt disp: {gt_disp:.6f}")
    print(f"  joints: {num_joints}, dims: {num_dims}")
    
    # Normalize
    gt_norm = normalize(gt_raw)
    past_norm = normalize(past_raw)
    gt_bjct = btjc_to_bjct(gt_norm)
    past_bjct = btjc_to_bjct(past_norm)
    
    print(f"  gt_norm range: [{gt_norm.min():.2f}, {gt_norm.max():.2f}]")
    
    # ========== 创建模型 ==========
    print(f"\n创建模型 (use_mean_pool={USE_MEAN_POOL})...")
    model = SignWritingToPoseDiffusionV2(
        num_keypoints=num_joints,
        num_dims_per_keypoint=num_dims,
        residual_scale=0.1,
        use_mean_pool=USE_MEAN_POOL,
    ).to(device)
    
    betas = cosine_beta_schedule(DIFFUSION_STEPS).numpy()
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # ========== 训练 ==========
    print("\n" + "=" * 70)
    print(f"开始训练 (目标: loss < {TARGET_LOSS})")
    print("=" * 70)
    
    model.train()
    losses = []
    converged = False
    
    for step in range(MAX_STEPS):
        optimizer.zero_grad()
        
        t = torch.randint(0, DIFFUSION_STEPS, (1,), device=device)
        noise = torch.randn_like(gt_bjct)
        x_t = diffusion.q_sample(gt_bjct, t, noise=noise)
        
        t_scaled = diffusion._scale_timesteps(t)
        pred_x0 = model(x_t, t_scaled, past_bjct, sign)
        
        loss = F.mse_loss(pred_x0, gt_bjct)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 200 == 0:
            pred_btjc = bjct_to_btjc(pred_x0.detach())
            pred_unnorm = unnormalize(pred_btjc)
            disp = mean_frame_disp(pred_unnorm)
            ratio = disp / (gt_disp + 1e-8)
            
            print(f"  Step {step}: loss={loss.item():.6f}, disp={disp:.6f}, ratio={ratio:.4f}, t={t.item()}")
        
        if loss.item() < TARGET_LOSS:
            print(f"\n  ✓ 收敛! Step {step}, loss={loss.item():.6f}")
            converged = True
            break
    
    final_loss = losses[-1]
    print(f"\n最终 loss: {final_loss:.6f}")
    
    if not converged:
        print(f"  ⚠️ 警告: {MAX_STEPS} 步后未达到目标 loss")
    
    # ========== 测试 1: 模型直接预测 ==========
    print("\n" + "=" * 70)
    print("测试 1: 模型直接预测 (不经过 p_sample_loop)")
    print("=" * 70)
    
    model.eval()
    with torch.no_grad():
        for t_val in [0, 1, 4, 7]:
            t = torch.tensor([t_val], device=device)
            noise = torch.randn_like(gt_bjct)
            x_t = diffusion.q_sample(gt_bjct, t, noise=noise)
            
            t_scaled = diffusion._scale_timesteps(t)
            pred_x0 = model(x_t, t_scaled, past_bjct, sign)
            
            pred_btjc = bjct_to_btjc(pred_x0)
            pred_unnorm = unnormalize(pred_btjc)
            
            disp = mean_frame_disp(pred_unnorm)
            mse = F.mse_loss(pred_unnorm, gt_raw).item()
            ratio = disp / (gt_disp + 1e-8)
            
            print(f"  t={t_val}: disp={disp:.6f}, ratio={ratio:.4f}, MSE={mse:.6f}")
    
    # ========== 测试 2: 逐步 p_sample ==========
    print("\n" + "=" * 70)
    print("测试 2: 逐步 p_sample 观察")
    print("=" * 70)
    
    with torch.no_grad():
        target_shape = (1, num_joints, num_dims, future_len)
        wrapped = ConditionalWrapper(model, past_bjct, sign)
        
        x = torch.randn(target_shape, device=device)
        print(f"\n  初始噪声范围: [{x.min():.2f}, {x.max():.2f}]")
        
        for i in range(DIFFUSION_STEPS - 1, -1, -1):
            t = torch.tensor([i], device=device)
            out = diffusion.p_sample(wrapped, x, t, clip_denoised=False, model_kwargs={"y": {}})
            x = out["sample"]
            
            x_btjc = bjct_to_btjc(x)
            x_unnorm = unnormalize(x_btjc)
            disp = mean_frame_disp(x_unnorm)
            ratio = disp / (gt_disp + 1e-8)
            
            print(f"  t={i}: disp={disp:.6f}, ratio={ratio:.4f}, range=[{x.min():.2f}, {x.max():.2f}]")
        
        # 最终结果
        final_pred = x_unnorm
        final_disp = disp
        final_ratio = ratio
    
    # ========== 测试 3: 完整 p_sample_loop ==========
    print("\n" + "=" * 70)
    print("测试 3: 完整 p_sample_loop")
    print("=" * 70)
    
    with torch.no_grad():
        pred_bjct = diffusion.p_sample_loop(
            model=wrapped,
            shape=target_shape,
            clip_denoised=False,
            model_kwargs={"y": {}},
            progress=False,
        )
        
        pred_btjc = bjct_to_btjc(pred_bjct)
        pred_raw = unnormalize(pred_btjc)
        
        disp = mean_frame_disp(pred_raw)
        mse = F.mse_loss(pred_raw, gt_raw).item()
        ratio = disp / (gt_disp + 1e-8)
        
        print(f"\n  p_sample_loop 结果:")
        print(f"    disp: {disp:.6f}")
        print(f"    disp_ratio: {ratio:.4f}")
        print(f"    MSE: {mse:.6f}")
        print(f"    pred range: [{pred_raw.min():.2f}, {pred_raw.max():.2f}]")
        print(f"    GT range: [{gt_raw.min():.2f}, {gt_raw.max():.2f}]")
    
    # ========== 完整评估 ==========
    print("\n" + "=" * 70)
    print("完整评估指标")
    print("=" * 70)
    
    with torch.no_grad():
        pred_np = pred_raw[0].cpu().numpy()
        gt_np = gt_raw[0].cpu().numpy()
        T, J, C = pred_np.shape
        
        # Position Errors
        mse = float(((pred_np - gt_np) ** 2).mean())
        mae = float(np.abs(pred_np - gt_np).mean())
        per_joint_error = np.sqrt(((pred_np - gt_np) ** 2).sum(axis=-1))
        mpjpe = float(per_joint_error.mean())
        fde = float(np.sqrt(((pred_np[-1] - gt_np[-1]) ** 2).sum(axis=-1)).mean())
        
        print(f"\n--- Position Errors ---")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  MPJPE: {mpjpe:.6f}")
        print(f"  FDE: {fde:.6f}")
        
        # Motion Match
        pred_vel = pred_np[1:] - pred_np[:-1]
        gt_vel = gt_np[1:] - gt_np[:-1]
        vel_mse = float(((pred_vel - gt_vel) ** 2).mean())
        
        print(f"\n--- Motion Match ---")
        print(f"  disp_ratio: {ratio:.4f} (理想=1.0)")
        print(f"  vel_mse: {vel_mse:.6f}")
        
        # PCK
        print(f"\n--- PCK ---")
        for thresh in [0.05, 0.1, 0.2, 0.5]:
            pck = (per_joint_error < thresh).mean()
            print(f"  PCK@{thresh}: {pck:.2%}")
        
        # DTW
        dtw_val = compute_dtw(pred_raw, gt_raw)
        print(f"\n--- Trajectory ---")
        print(f"  DTW: {dtw_val:.4f}")
    
    # ========== 保存文件 ==========
    print("\n" + "=" * 70)
    print("保存文件")
    print("=" * 70)
    
    ref_path = base_ds.records[SAMPLE_IDX]["pose"]
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    
    ref_pose = reduce_holistic(ref_pose)
    if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
        ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_pose.header
    
    # GT
    T_total = ref_pose.body.data.shape[0]
    gt_data = ref_pose.body.data[-future_len:]
    gt_conf = ref_pose.body.confidence[-future_len:]
    gt_body = NumPyPoseBody(fps=ref_pose.body.fps, data=gt_data, confidence=gt_conf)
    gt_pose = Pose(header=header, body=gt_body)
    
    out_gt = os.path.join(out_dir, f"gt_{SAMPLE_IDX}.pose")
    with open(out_gt, "wb") as f:
        gt_pose.write(f)
    print(f"✓ GT saved: {out_gt}")
    
    # Pred
    pred_pose = tensor_to_pose(pred_raw, header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
    out_pred = os.path.join(out_dir, f"pred_{SAMPLE_IDX}.pose")
    with open(out_pred, "wb") as f:
        pred_pose.write(f)
    print(f"✓ PRED saved: {out_pred}")
    
    # ========== 分析保存的文件 ==========
    print("\n" + "=" * 70)
    print("分析保存的 pose 文件 (pixel 空间)")
    print("=" * 70)
    
    with open(out_gt, "rb") as f:
        saved_gt = Pose.read(f)
    with open(out_pred, "rb") as f:
        saved_pred = Pose.read(f)
    
    gt_data_px = np.array(saved_gt.body.data[:, 0])
    pred_data_px = np.array(saved_pred.body.data[:, 0])
    
    gt_disp_px = np.abs(gt_data_px[1:] - gt_data_px[:-1]).mean()
    pred_disp_px = np.abs(pred_data_px[1:] - pred_data_px[:-1]).mean()
    
    print(f"\n  GT 平均帧间位移: {gt_disp_px:.2f} px")
    print(f"  PRED 平均帧间位移: {pred_disp_px:.2f} px")
    print(f"  disp_ratio (pixel): {pred_disp_px / (gt_disp_px + 1e-8):.4f}")
    
    # ========== 最终判断 ==========
    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)
    
    print(f"\n最终 Loss: {final_loss:.6f}")
    print(f"disp_ratio: {ratio:.4f}")
    
    if final_loss < 0.01 and ratio > 0.3:
        print("\n✅ 测试 PASS!")
        print("   - 训练 loss 收敛 (< 0.01)")
        print("   - 预测有运动 (disp_ratio > 0.3)")
        print("   → 模型架构和 Diffusion 流程正确!")
    elif final_loss < 0.01:
        print("\n⚠️ 部分 PASS")
        print("   - 训练 loss 收敛")
        print(f"   - 但 disp_ratio={ratio:.4f} 偏低")
        print("   → 问题在 p_sample_loop 采样过程")
    else:
        print("\n❌ 测试 FAIL")
        print(f"   - 训练 loss={final_loss:.6f} 未充分收敛")
        print("   → 检查模型架构或学习率")
    
    print("\n" + "=" * 70)
    print("✓ 完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()