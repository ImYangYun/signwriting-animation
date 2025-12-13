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


class ConditionalWrapper(torch.nn.Module):
    def __init__(self, model, past, sign):
        super().__init__()
        self.model = model
        self.past = past
        self.sign = sign
    
    def forward(self, x, t, **kwargs):
        return self.model(x, t, self.past, self.sign)


def test_scheme_1_disp_loss(gt_raw, past_raw, sign, gt_bjct, past_bjct, 
                            model, diffusion, device, max_steps=2000):
    """方案 1: 加 Displacement Loss"""
    print("\n" + "=" * 70)
    print("方案 1: 加 Displacement Loss")
    print("=" * 70)
    
    # 重置模型
    model_1 = SignWritingToPoseDiffusionV2(
        num_keypoints=gt_raw.shape[2],
        num_dims_per_keypoint=gt_raw.shape[3],
        residual_scale=0.1,
        use_mean_pool=True,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model_1.parameters(), lr=1e-3)
    model_1.train()
    
    gt_disp = mean_frame_disp(gt_raw)
    
    for step in range(max_steps):
        optimizer.zero_grad()
        
        t = torch.randint(0, 8, (1,), device=device)
        noise = torch.randn_like(gt_bjct)
        x_t = diffusion.q_sample(gt_bjct, t, noise=noise)
        
        t_scaled = diffusion._scale_timesteps(t)
        pred_x0 = model_1(x_t, t_scaled, past_bjct, sign)
        
        # 主 loss: MSE
        loss_mse = F.mse_loss(pred_x0, gt_bjct)
        
        # Displacement loss
        pred_btjc = pred_x0.permute(0, 3, 1, 2)  # [B,J,C,T] -> [B,T,J,C]
        gt_btjc_local = gt_bjct.permute(0, 3, 1, 2)
        
        pred_vel = pred_btjc[:, 1:] - pred_btjc[:, :-1]
        gt_vel = gt_btjc_local[:, 1:] - gt_btjc_local[:, :-1]
        
        pred_disp = pred_vel.abs().mean()
        gt_disp_local = gt_vel.abs().mean()
        
        # 强制运动量匹配
        loss_disp = torch.abs(pred_disp - gt_disp_local)
        
        # 总 loss
        loss = loss_mse + 10.0 * loss_disp  # disp loss 权重大一些
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_1.parameters(), 1.0)
        optimizer.step()
        
        if step % 500 == 0:
            print(f"  Step {step}: mse={loss_mse.item():.6f}, disp_loss={loss_disp.item():.6f}, "
                  f"pred_disp={pred_disp.item():.6f}, gt_disp={gt_disp_local.item():.6f}")
    
    # 测试
    model_1.eval()
    with torch.no_grad():
        wrapped = ConditionalWrapper(model_1, past_bjct, sign)
        target_shape = (1, gt_raw.shape[2], gt_raw.shape[3], gt_raw.shape[1])
        
        pred_bjct = diffusion.p_sample_loop(
            model=wrapped, shape=target_shape,
            clip_denoised=False, model_kwargs={"y": {}}, progress=False,
        )
        pred_btjc = pred_bjct.permute(0, 3, 1, 2)
        
        # Unnormalize（用原始方式）
        stats_path = "/home/yayun/data/pose_data/mean_std_178_with_preprocess.pt"
        stats = torch.load(stats_path, map_location=device)
        mean_pose = stats["mean"].float().view(1, 1, -1, 3).to(device)
        std_pose = stats["std"].float().view(1, 1, -1, 3).to(device)
        pred_unnorm = pred_btjc * std_pose + mean_pose
        
        disp = mean_frame_disp(pred_unnorm)
        ratio = disp / (gt_disp + 1e-8)
        
        print(f"\n  结果: disp={disp:.6f}, ratio={ratio:.4f}")
    
    return ratio, model_1


def test_scheme_2_small_std(gt_raw, past_raw, sign, device, max_steps=2000):
    """方案 2: 用更小的 std normalize"""
    print("\n" + "=" * 70)
    print("方案 2: 调整 Normalize（用更小的 std）")
    print("=" * 70)
    
    # 用更小的 std（只用位置的 std，不用全局）
    # 或者直接用固定的小 std
    small_std = 50.0  # 而不是 ~200
    
    gt_mean = gt_raw.mean()
    gt_norm_small = (gt_raw - gt_mean) / small_std
    past_norm_small = (past_raw - gt_mean) / small_std
    
    gt_bjct = gt_norm_small.permute(0, 2, 3, 1).contiguous()
    past_bjct = past_norm_small.permute(0, 2, 3, 1).contiguous()
    
    print(f"  小 std normalize:")
    print(f"    std = {small_std}")
    print(f"    gt_norm range: [{gt_norm_small.min():.2f}, {gt_norm_small.max():.2f}]")
    print(f"    gt_norm disp: {mean_frame_disp(gt_norm_small):.6f}")
    
    model_2 = SignWritingToPoseDiffusionV2(
        num_keypoints=gt_raw.shape[2],
        num_dims_per_keypoint=gt_raw.shape[3],
        residual_scale=0.1,
        use_mean_pool=True,
    ).to(device)
    
    betas = cosine_beta_schedule(8).numpy()
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )
    
    optimizer = torch.optim.AdamW(model_2.parameters(), lr=1e-3)
    model_2.train()
    
    gt_disp = mean_frame_disp(gt_raw)
    
    for step in range(max_steps):
        optimizer.zero_grad()
        
        t = torch.randint(0, 8, (1,), device=device)
        noise = torch.randn_like(gt_bjct)
        x_t = diffusion.q_sample(gt_bjct, t, noise=noise)
        
        t_scaled = diffusion._scale_timesteps(t)
        pred_x0 = model_2(x_t, t_scaled, past_bjct, sign)
        
        loss = F.mse_loss(pred_x0, gt_bjct)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_2.parameters(), 1.0)
        optimizer.step()
        
        if step % 500 == 0:
            pred_btjc = pred_x0.permute(0, 3, 1, 2)
            pred_unnorm = pred_btjc * small_std + gt_mean
            disp = mean_frame_disp(pred_unnorm)
            ratio = disp / (gt_disp + 1e-8)
            print(f"  Step {step}: loss={loss.item():.6f}, disp={disp:.6f}, ratio={ratio:.4f}")
    
    # 测试
    model_2.eval()
    with torch.no_grad():
        wrapped = ConditionalWrapper(model_2, past_bjct, sign)
        target_shape = (1, gt_raw.shape[2], gt_raw.shape[3], gt_raw.shape[1])
        
        pred_bjct = diffusion.p_sample_loop(
            model=wrapped, shape=target_shape,
            clip_denoised=False, model_kwargs={"y": {}}, progress=False,
        )
        pred_btjc = pred_bjct.permute(0, 3, 1, 2)
        pred_unnorm = pred_btjc * small_std + gt_mean
        
        disp = mean_frame_disp(pred_unnorm)
        ratio = disp / (gt_disp + 1e-8)
        
        print(f"\n  结果: disp={disp:.6f}, ratio={ratio:.4f}")
    
    return ratio, model_2


def test_scheme_3_velocity(gt_raw, past_raw, sign, device, max_steps=2000):
    """方案 3: 预测 Velocity 而不是位置"""
    print("\n" + "=" * 70)
    print("方案 3: 预测 Velocity（帧间差）")
    print("=" * 70)
    
    # 计算 velocity
    gt_vel = gt_raw[:, 1:] - gt_raw[:, :-1]  # [B, T-1, J, C]
    
    # Normalize velocity（用 velocity 自己的 std）
    vel_mean = gt_vel.mean()
    vel_std = gt_vel.std() + 1e-6
    gt_vel_norm = (gt_vel - vel_mean) / vel_std
    
    print(f"  Velocity normalize:")
    print(f"    vel_std = {vel_std.item():.6f}")
    print(f"    gt_vel_norm range: [{gt_vel_norm.min():.2f}, {gt_vel_norm.max():.2f}]")
    print(f"    gt_vel_norm disp: {mean_frame_disp(gt_vel_norm):.6f}")
    
    # 模型输出维度变了：T-1 帧
    future_len = gt_vel.shape[1]  # 19 instead of 20
    
    gt_vel_bjct = gt_vel_norm.permute(0, 2, 3, 1).contiguous()
    
    # Past 还是用位置（normalize）
    stats_path = "/home/yayun/data/pose_data/mean_std_178_with_preprocess.pt"
    stats = torch.load(stats_path, map_location=device)
    mean_pose = stats["mean"].float().view(1, 1, -1, 3).to(device)
    std_pose = stats["std"].float().view(1, 1, -1, 3).to(device)
    past_norm = (past_raw - mean_pose) / (std_pose + 1e-6)
    past_bjct = past_norm.permute(0, 2, 3, 1).contiguous()
    
    model_3 = SignWritingToPoseDiffusionV2(
        num_keypoints=gt_raw.shape[2],
        num_dims_per_keypoint=gt_raw.shape[3],
        residual_scale=0.1,
        use_mean_pool=True,
    ).to(device)
    
    betas = cosine_beta_schedule(8).numpy()
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )
    
    optimizer = torch.optim.AdamW(model_3.parameters(), lr=1e-3)
    model_3.train()
    
    gt_disp = mean_frame_disp(gt_raw)
    
    for step in range(max_steps):
        optimizer.zero_grad()
        
        t = torch.randint(0, 8, (1,), device=device)
        noise = torch.randn_like(gt_vel_bjct)
        x_t = diffusion.q_sample(gt_vel_bjct, t, noise=noise)
        
        t_scaled = diffusion._scale_timesteps(t)
        pred_vel_bjct = model_3(x_t, t_scaled, past_bjct, sign)
        
        loss = F.mse_loss(pred_vel_bjct, gt_vel_bjct)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_3.parameters(), 1.0)
        optimizer.step()
        
        if step % 500 == 0:
            print(f"  Step {step}: loss={loss.item():.6f}")
    
    # 测试
    model_3.eval()
    with torch.no_grad():
        wrapped = ConditionalWrapper(model_3, past_bjct, sign)
        target_shape = (1, gt_raw.shape[2], gt_raw.shape[3], future_len)
        
        pred_vel_bjct = diffusion.p_sample_loop(
            model=wrapped, shape=target_shape,
            clip_denoised=False, model_kwargs={"y": {}}, progress=False,
        )
        pred_vel_btjc = pred_vel_bjct.permute(0, 3, 1, 2)
        
        # Unnormalize velocity
        pred_vel_unnorm = pred_vel_btjc * vel_std + vel_mean
        
        # 从 velocity 重建位置
        # 用 past 的最后一帧作为起点
        start_pos = past_raw[:, -1:, :, :]  # [B, 1, J, C]
        pred_pos = [start_pos]
        for t in range(future_len):
            next_pos = pred_pos[-1] + pred_vel_unnorm[:, t:t+1]
            pred_pos.append(next_pos)
        pred_pos = torch.cat(pred_pos[1:], dim=1)  # [B, T-1, J, C]
        
        disp = mean_frame_disp(pred_pos)
        ratio = disp / (gt_disp + 1e-8)
        
        print(f"\n  结果: disp={disp:.6f}, ratio={ratio:.4f}")
    
    return ratio, model_3


def main():
    print("=" * 70)
    print("三种方案对比测试")
    print("=" * 70)
    
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 加载数据
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    # 用样本 36（之前测试过的）
    sample = base_ds[36]
    batch = zero_pad_collator([sample])
    
    cond = batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(batch["data"][:1]).to(device)
    
    gt_disp = mean_frame_disp(gt_raw)
    print(f"\n样本 36:")
    print(f"  gt shape: {gt_raw.shape}")
    print(f"  gt disp (pixel): {gt_disp:.6f}")
    
    # 原始 normalize
    stats = torch.load(stats_path, map_location=device)
    mean_pose = stats["mean"].float().view(1, 1, -1, 3).to(device)
    std_pose = stats["std"].float().view(1, 1, -1, 3).to(device)
    
    gt_norm = (gt_raw - mean_pose) / (std_pose + 1e-6)
    past_norm = (past_raw - mean_pose) / (std_pose + 1e-6)
    
    gt_bjct = gt_norm.permute(0, 2, 3, 1).contiguous()
    past_bjct = past_norm.permute(0, 2, 3, 1).contiguous()
    
    print(f"  gt_norm disp: {mean_frame_disp(gt_norm):.6f}")
    print(f"  std mean: {std_pose.mean().item():.2f}")
    
    # Diffusion
    betas = cosine_beta_schedule(8).numpy()
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )
    
    # 测试三种方案
    results = {}
    
    # 方案 1
    ratio_1, _ = test_scheme_1_disp_loss(
        gt_raw, past_raw, sign, gt_bjct, past_bjct,
        None, diffusion, device, max_steps=2000
    )
    results['方案1_DispLoss'] = ratio_1
    
    # 方案 2
    ratio_2, _ = test_scheme_2_small_std(
        gt_raw, past_raw, sign, device, max_steps=2000
    )
    results['方案2_SmallStd'] = ratio_2
    
    # 方案 3
    ratio_3, _ = test_scheme_3_velocity(
        gt_raw, past_raw, sign, device, max_steps=2000
    )
    results['方案3_Velocity'] = ratio_3
    
    # 总结
    print("\n" + "=" * 70)
    print("结果对比")
    print("=" * 70)
    print(f"\n{'方案':<20} | {'disp_ratio':<12} | {'评价'}")
    print("-" * 50)
    for name, ratio in results.items():
        if ratio > 0.5:
            eval_str = "✓ 好"
        elif ratio > 0.3:
            eval_str = "○ 可以"
        else:
            eval_str = "✗ 差"
        print(f"{name:<20} | {ratio:<12.4f} | {eval_str}")
    
    best = max(results, key=results.get)
    print(f"\n最佳方案: {best} (ratio={results[best]:.4f})")
    
    print("\n" + "=" * 70)
    print("完成")
    print("=" * 70)


if __name__ == "__main__":
    main()