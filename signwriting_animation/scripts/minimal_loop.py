# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as pl

from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusionV2
from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType

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


class ConditionalWrapper(nn.Module):
    def __init__(self, model, past, sign):
        super().__init__()
        self.model = model
        self.past = past
        self.sign = sign
    
    def forward(self, x, t, **kwargs):
        return self.model(x, t, self.past, self.sign)


def compute_velocity_loss(pred_bjct, gt_bjct):
    """计算 velocity loss (在 BJCT 格式下)"""
    # BJCT: [B, J, C, T] -> velocity 在 T 维度
    pred_vel = pred_bjct[:, :, :, 1:] - pred_bjct[:, :, :, :-1]
    gt_vel = gt_bjct[:, :, :, 1:] - gt_bjct[:, :, :, :-1]
    return F.mse_loss(pred_vel, gt_vel)


def main():
    print("=" * 70)
    print("修复 Diffusion：加入 Velocity Loss")
    print("=" * 70)
    
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 加载数据
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir, csv_path=csv_path,
        num_past_frames=40, num_future_frames=20,
        with_metadata=True, split="train",
    )
    
    sample = base_ds[36]
    batch = zero_pad_collator([sample])
    
    cond = batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(batch["data"][:1]).to(device)
    
    stats = torch.load(stats_path, map_location=device)
    mean_pose = stats["mean"].float().view(1, 1, -1, 3).to(device)
    std_pose = stats["std"].float().view(1, 1, -1, 3).to(device)
    
    gt_norm = (gt_raw - mean_pose) / (std_pose + 1e-6)
    past_norm = (past_raw - mean_pose) / (std_pose + 1e-6)
    
    gt_bjct = gt_norm.permute(0, 2, 3, 1).contiguous()
    past_bjct = past_norm.permute(0, 2, 3, 1).contiguous()
    
    gt_disp = mean_frame_disp(gt_raw)
    
    print(f"\n数据:")
    print(f"  GT pixel disp: {gt_disp:.6f}")
    
    # 创建 Diffusion
    DIFFUSION_STEPS = 8
    betas = cosine_beta_schedule(DIFFUSION_STEPS).numpy()
    
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )
    
    # 创建模型
    model = SignWritingToPoseDiffusionV2(
        num_keypoints=gt_raw.shape[2],
        num_dims_per_keypoint=gt_raw.shape[3],
        residual_scale=0.1,
        use_mean_pool=True,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # ========== 训练（带 Velocity Loss）==========
    print("\n" + "=" * 70)
    print("训练 (3000 步, MSE + Velocity Loss)")
    print("=" * 70)
    
    VEL_WEIGHT = 1.0  # velocity loss 权重
    
    model.train()
    for step in range(3000):
        optimizer.zero_grad()
        
        t = torch.randint(0, DIFFUSION_STEPS, (1,), device=device)
        noise = torch.randn_like(gt_bjct)
        x_t = diffusion.q_sample(gt_bjct, t, noise=noise)
        
        t_scaled = diffusion._scale_timesteps(t)
        pred_x0 = model(x_t, t_scaled, past_bjct, sign)
        
        # MSE Loss
        loss_mse = F.mse_loss(pred_x0, gt_bjct)
        
        # Velocity Loss
        loss_vel = compute_velocity_loss(pred_x0, gt_bjct)
        
        # 总 Loss
        loss = loss_mse + VEL_WEIGHT * loss_vel
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 500 == 0:
            pred_btjc = pred_x0.permute(0, 3, 1, 2)
            pred_unnorm = pred_btjc * std_pose + mean_pose
            disp = mean_frame_disp(pred_unnorm)
            ratio = disp / (gt_disp + 1e-8)
            print(f"  Step {step}: mse={loss_mse.item():.6f}, vel={loss_vel.item():.6f}, "
                  f"disp={disp:.6f}, ratio={ratio:.4f}, t={t.item()}")
    
    print(f"\n训练完成!")
    
    # ========== 测试 ==========
    print("\n" + "=" * 70)
    print("测试：模型在不同输入下的预测")
    print("=" * 70)
    
    model.eval()
    with torch.no_grad():
        target_shape = (1, gt_raw.shape[2], gt_raw.shape[3], gt_raw.shape[1])
        
        # 测试 1: 从 GT+noise 预测
        print("\n从 GT+noise 预测:")
        for t_val in [0, 4, 7]:
            t = torch.tensor([t_val], device=device)
            noise = torch.randn_like(gt_bjct)
            x_t = diffusion.q_sample(gt_bjct, t, noise=noise)
            
            t_scaled = diffusion._scale_timesteps(t)
            pred_x0 = model(x_t, t_scaled, past_bjct, sign)
            
            pred_btjc = pred_x0.permute(0, 3, 1, 2)
            pred_unnorm = pred_btjc * std_pose + mean_pose
            disp = mean_frame_disp(pred_unnorm)
            ratio = disp / (gt_disp + 1e-8)
            
            print(f"  t={t_val}: disp={disp:.6f}, ratio={ratio:.4f}")
        
        # 测试 2: 从纯噪声预测
        print("\n从纯噪声预测:")
        x_noise = torch.randn(target_shape, device=device)
        for t_val in [7, 4, 0]:
            t = torch.tensor([t_val], device=device)
            t_scaled = diffusion._scale_timesteps(t)
            pred_x0 = model(x_noise, t_scaled, past_bjct, sign)
            
            pred_btjc = pred_x0.permute(0, 3, 1, 2)
            pred_unnorm = pred_btjc * std_pose + mean_pose
            disp = mean_frame_disp(pred_unnorm)
            ratio = disp / (gt_disp + 1e-8)
            
            print(f"  t={t_val}: disp={disp:.6f}, ratio={ratio:.4f}")
        
        # 测试 3: p_sample_loop
        print("\np_sample_loop:")
        wrapped = ConditionalWrapper(model, past_bjct, sign)
        
        pred_bjct = diffusion.p_sample_loop(
            model=wrapped, shape=target_shape,
            clip_denoised=False, model_kwargs={"y": {}}, progress=False,
        )
        
        pred_btjc = pred_bjct.permute(0, 3, 1, 2)
        pred_unnorm = pred_btjc * std_pose + mean_pose
        
        disp = mean_frame_disp(pred_unnorm)
        mse = F.mse_loss(pred_unnorm, gt_raw).item()
        ratio = disp / (gt_disp + 1e-8)
        
        print(f"  disp={disp:.6f}, ratio={ratio:.4f}, MSE={mse:.6f}")
        
        # 逐步观察
        print("\n逐步 p_sample:")
        x = torch.randn(target_shape, device=device)
        for i in range(DIFFUSION_STEPS - 1, -1, -1):
            t = torch.tensor([i], device=device)
            out = diffusion.p_sample(wrapped, x, t, clip_denoised=False, model_kwargs={"y": {}})
            x = out["sample"]
            pred_x0 = out.get("pred_xstart", None)
            
            if pred_x0 is not None:
                pred_btjc = pred_x0.permute(0, 3, 1, 2)
                pred_unnorm = pred_btjc * std_pose + mean_pose
                pred_disp = mean_frame_disp(pred_unnorm)
                pred_ratio = pred_disp / (gt_disp + 1e-8)
            else:
                pred_ratio = -1
            
            x_btjc = x.permute(0, 3, 1, 2)
            x_unnorm = x_btjc * std_pose + mean_pose
            sample_disp = mean_frame_disp(x_unnorm)
            sample_ratio = sample_disp / (gt_disp + 1e-8)
            
            print(f"  t={i}: pred_x0_ratio={pred_ratio:.4f}, sample_ratio={sample_ratio:.4f}")
    
    # ========== 最终评估 ==========
    print("\n" + "=" * 70)
    print("最终评估")
    print("=" * 70)
    
    final_ratio = ratio
    
    if final_ratio > 0.5:
        print(f"\n✅ 成功! disp_ratio={final_ratio:.4f}")
        print("   Velocity loss 解决了问题!")
    elif final_ratio > 0.3:
        print(f"\n⚠️ 部分成功: disp_ratio={final_ratio:.4f}")
        print("   可能需要调整 velocity loss 权重")
    else:
        print(f"\n❌ 仍然失败: disp_ratio={final_ratio:.4f}")
        print("   需要其他方案")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()