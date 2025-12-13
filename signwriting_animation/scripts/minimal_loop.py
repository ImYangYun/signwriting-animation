# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as pl

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset


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


def main():
    print("=" * 70)
    print("Debug Diffusion 采样过程")
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
    gt_norm_disp = mean_frame_disp(gt_norm)
    
    print(f"\n数据:")
    print(f"  GT pixel disp: {gt_disp:.6f}")
    print(f"  GT norm disp: {gt_norm_disp:.6f}")
    
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
    
    # 打印 diffusion 参数
    print(f"\nDiffusion 参数:")
    print(f"  num_timesteps: {diffusion.num_timesteps}")
    print(f"  betas: {diffusion.betas}")
    print(f"  alphas_cumprod: {diffusion.alphas_cumprod}")
    print(f"  sqrt_alphas_cumprod: {diffusion.sqrt_alphas_cumprod}")
    print(f"  sqrt_one_minus_alphas_cumprod: {diffusion.sqrt_one_minus_alphas_cumprod}")
    
    # 创建模型
    model = SignWritingToPoseDiffusionV2(
        num_keypoints=gt_raw.shape[2],
        num_dims_per_keypoint=gt_raw.shape[3],
        residual_scale=0.1,
        use_mean_pool=True,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # ========== 训练 ==========
    print("\n" + "=" * 70)
    print("训练 (3000 步)")
    print("=" * 70)
    
    model.train()
    for step in range(3000):
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
        
        if step % 500 == 0:
            pred_btjc = pred_x0.permute(0, 3, 1, 2)
            pred_unnorm = pred_btjc * std_pose + mean_pose
            disp = mean_frame_disp(pred_unnorm)
            ratio = disp / (gt_disp + 1e-8)
            print(f"  Step {step}: loss={loss.item():.6f}, disp={disp:.6f}, ratio={ratio:.4f}, t={t.item()}")
    
    print(f"\n训练完成!")
    
    # ========== Debug 1: 模型在不同 t 的直接预测 ==========
    print("\n" + "=" * 70)
    print("Debug 1: 模型在不同 t 的直接预测 (从 GT+noise)")
    print("=" * 70)
    
    model.eval()
    with torch.no_grad():
        for t_val in range(DIFFUSION_STEPS):
            t = torch.tensor([t_val], device=device)
            
            # 从 GT 加噪
            noise = torch.randn_like(gt_bjct)
            x_t = diffusion.q_sample(gt_bjct, t, noise=noise)
            
            # 模型预测
            t_scaled = diffusion._scale_timesteps(t)
            pred_x0 = model(x_t, t_scaled, past_bjct, sign)
            
            # 评估
            pred_btjc = pred_x0.permute(0, 3, 1, 2)
            pred_unnorm = pred_btjc * std_pose + mean_pose
            
            disp = mean_frame_disp(pred_unnorm)
            mse = F.mse_loss(pred_unnorm, gt_raw).item()
            ratio = disp / (gt_disp + 1e-8)
            
            # x_t 的信息
            x_t_btjc = x_t.permute(0, 3, 1, 2)
            x_t_unnorm = x_t_btjc * std_pose + mean_pose
            x_t_disp = mean_frame_disp(x_t_unnorm)
            
            print(f"  t={t_val}: pred_disp={disp:.6f}, ratio={ratio:.4f}, "
                  f"mse={mse:.6f}, x_t_disp={x_t_disp:.6f}")
    
    # ========== Debug 2: 从纯噪声开始的模型预测 ==========
    print("\n" + "=" * 70)
    print("Debug 2: 从纯噪声开始的模型预测")
    print("=" * 70)
    
    with torch.no_grad():
        target_shape = (1, gt_raw.shape[2], gt_raw.shape[3], gt_raw.shape[1])
        
        # 纯噪声
        x_noise = torch.randn(target_shape, device=device)
        print(f"  纯噪声 range: [{x_noise.min():.2f}, {x_noise.max():.2f}]")
        
        for t_val in range(DIFFUSION_STEPS - 1, -1, -1):
            t = torch.tensor([t_val], device=device)
            t_scaled = diffusion._scale_timesteps(t)
            
            pred_x0 = model(x_noise, t_scaled, past_bjct, sign)
            
            pred_btjc = pred_x0.permute(0, 3, 1, 2)
            pred_unnorm = pred_btjc * std_pose + mean_pose
            
            disp = mean_frame_disp(pred_unnorm)
            ratio = disp / (gt_disp + 1e-8)
            
            print(f"  t={t_val}: pred_disp={disp:.6f}, ratio={ratio:.4f}, "
                  f"pred_range=[{pred_x0.min():.2f}, {pred_x0.max():.2f}]")
    
    # ========== Debug 3: 手动 p_sample 每一步 ==========
    print("\n" + "=" * 70)
    print("Debug 3: 手动 p_sample 每一步详细分析")
    print("=" * 70)
    
    with torch.no_grad():
        wrapped = ConditionalWrapper(model, past_bjct, sign)
        
        x = torch.randn(target_shape, device=device)
        print(f"\n  初始噪声: range=[{x.min():.2f}, {x.max():.2f}]")
        
        for i in range(DIFFUSION_STEPS - 1, -1, -1):
            t = torch.tensor([i], device=device)
            
            # 获取 p_sample 的详细输出
            out = diffusion.p_sample(
                wrapped, x, t, 
                clip_denoised=False, 
                model_kwargs={"y": {}}
            )
            
            x_prev = out["sample"]
            pred_xstart = out.get("pred_xstart", None)
            
            # 分析
            x_btjc = x_prev.permute(0, 3, 1, 2)
            x_unnorm = x_btjc * std_pose + mean_pose
            sample_disp = mean_frame_disp(x_unnorm)
            
            if pred_xstart is not None:
                pred_btjc = pred_xstart.permute(0, 3, 1, 2)
                pred_unnorm = pred_btjc * std_pose + mean_pose
                pred_disp = mean_frame_disp(pred_unnorm)
            else:
                pred_disp = -1
            
            sample_ratio = sample_disp / (gt_disp + 1e-8)
            pred_ratio = pred_disp / (gt_disp + 1e-8) if pred_disp >= 0 else -1
            
            print(f"  t={i}: sample_disp={sample_disp:.6f} (ratio={sample_ratio:.4f}), "
                  f"pred_x0_disp={pred_disp:.6f} (ratio={pred_ratio:.4f}), "
                  f"range=[{x_prev.min():.2f}, {x_prev.max():.2f}]")
            
            x = x_prev
        
        # 最终结果
        final_btjc = x.permute(0, 3, 1, 2)
        final_unnorm = final_btjc * std_pose + mean_pose
        final_disp = mean_frame_disp(final_unnorm)
        final_ratio = final_disp / (gt_disp + 1e-8)
        
        print(f"\n  最终: disp={final_disp:.6f}, ratio={final_ratio:.4f}")
    
    # ========== Debug 4: 检查 p_sample 内部计算 ==========
    print("\n" + "=" * 70)
    print("Debug 4: 手动实现 p_sample 检查每个步骤")
    print("=" * 70)
    
    with torch.no_grad():
        x = torch.randn(target_shape, device=device)
        
        for i in range(DIFFUSION_STEPS - 1, -1, -1):
            t = torch.tensor([i], device=device)
            t_scaled = diffusion._scale_timesteps(t)
            
            # Step 1: 模型预测 x_0
            model_output = model(x, t_scaled, past_bjct, sign)
            pred_x0 = model_output  # 因为是 START_X mode
            
            # Step 2: 计算 mean
            # p_mean_variance 会用 pred_x0 计算 mean
            # mean = sqrt_recip_alphas_cumprod * x - sqrt_recipm1_alphas_cumprod * pred_x0
            # 或者对于 START_X: mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x
            
            posterior_mean_coef1 = diffusion.posterior_mean_coef1[i]
            posterior_mean_coef2 = diffusion.posterior_mean_coef2[i]
            
            mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x
            
            # Step 3: 计算 variance
            posterior_variance = diffusion.posterior_variance[i]
            posterior_log_variance = diffusion.posterior_log_variance_clipped[i]
            
            # Step 4: 采样
            if i > 0:
                noise = torch.randn_like(x)
                x_prev = mean + torch.sqrt(torch.tensor(posterior_variance, device=device)) * noise
            else:
                # t=0 时不加噪声
                x_prev = mean
            
            # 分析
            pred_btjc = pred_x0.permute(0, 3, 1, 2)
            pred_unnorm = pred_btjc * std_pose + mean_pose
            pred_disp = mean_frame_disp(pred_unnorm)
            
            mean_btjc = mean.permute(0, 3, 1, 2)
            mean_unnorm = mean_btjc * std_pose + mean_pose
            mean_disp = mean_frame_disp(mean_unnorm)
            
            x_prev_btjc = x_prev.permute(0, 3, 1, 2)
            x_prev_unnorm = x_prev_btjc * std_pose + mean_pose
            x_prev_disp = mean_frame_disp(x_prev_unnorm)
            
            print(f"  t={i}:")
            print(f"    coef1={posterior_mean_coef1:.4f}, coef2={posterior_mean_coef2:.4f}, var={posterior_variance:.6f}")
            print(f"    pred_x0 disp={pred_disp:.6f} (ratio={pred_disp/gt_disp:.4f})")
            print(f"    mean disp={mean_disp:.6f} (ratio={mean_disp/gt_disp:.4f})")
            print(f"    x_prev disp={x_prev_disp:.6f} (ratio={x_prev_disp/gt_disp:.4f})")
            
            x = x_prev
    
    # ========== Debug 5: 直接用 pred_x0 作为最终输出（跳过采样） ==========
    print("\n" + "=" * 70)
    print("Debug 5: 跳过采样，直接用最终 pred_x0")
    print("=" * 70)
    
    with torch.no_grad():
        x = torch.randn(target_shape, device=device)
        
        # 迭代到最后
        for i in range(DIFFUSION_STEPS - 1, -1, -1):
            t = torch.tensor([i], device=device)
            out = diffusion.p_sample(wrapped, x, t, clip_denoised=False, model_kwargs={"y": {}})
            x = out["sample"]
            last_pred_x0 = out.get("pred_xstart", None)
        
        # 用 p_sample_loop 的最终 sample
        sample_btjc = x.permute(0, 3, 1, 2)
        sample_unnorm = sample_btjc * std_pose + mean_pose
        sample_disp = mean_frame_disp(sample_unnorm)
        
        # 用最后一步的 pred_x0
        if last_pred_x0 is not None:
            pred_btjc = last_pred_x0.permute(0, 3, 1, 2)
            pred_unnorm = pred_btjc * std_pose + mean_pose
            pred_disp = mean_frame_disp(pred_unnorm)
        else:
            pred_disp = -1
        
        print(f"  p_sample_loop 最终 sample: disp={sample_disp:.6f}, ratio={sample_disp/gt_disp:.4f}")
        print(f"  最后一步的 pred_x0: disp={pred_disp:.6f}, ratio={pred_disp/gt_disp:.4f}")
        
        # 直接用模型预测 t=0
        t = torch.tensor([0], device=device)
        t_scaled = diffusion._scale_timesteps(t)
        direct_pred = model(x, t_scaled, past_bjct, sign)
        
        direct_btjc = direct_pred.permute(0, 3, 1, 2)
        direct_unnorm = direct_btjc * std_pose + mean_pose
        direct_disp = mean_frame_disp(direct_unnorm)
        
        print(f"  直接用 t=0 预测: disp={direct_disp:.6f}, ratio={direct_disp/gt_disp:.4f}")
    
    # ========== 总结 ==========
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"""
    关键发现：
    1. 模型从 GT+noise 预测时有运动 (Debug 1)
    2. 从纯噪声预测时运动量如何？ (Debug 2)
    3. p_sample 每步的 pred_x0 vs mean vs sample 变化 (Debug 3, 4)
    4. 最终 sample vs pred_x0 的差异 (Debug 5)

    如果 pred_x0 一直有运动，但 sample 逐渐变静态：
    → 问题在 posterior mean 的混合过程

    如果 pred_x0 本身就变静态：
    → 问题在模型对纯噪声输入的处理
    """)


if __name__ == "__main__":
    main()