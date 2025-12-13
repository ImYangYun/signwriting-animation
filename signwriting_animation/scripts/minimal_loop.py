# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
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


class SimpleRegressionModel(nn.Module):
    """
    简单的回归模型：直接从 past + sign 预测 future
    不用 Diffusion，不用 timestep embedding
    """
    def __init__(self, num_joints=178, num_dims=3, past_frames=40, future_frames=20, latent_dim=256):
        super().__init__()
        self.num_joints = num_joints
        self.num_dims = num_dims
        self.past_frames = past_frames
        self.future_frames = future_frames
        
        # Past motion encoder
        self.past_encoder = nn.Sequential(
            nn.Linear(past_frames * num_joints * num_dims, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        
        # Sign image encoder (简化)
        self.sign_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # [1, H, W] -> [32, H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> [64, H/4, W/4]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> [64, 1, 1]
            nn.Flatten(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, future_frames * num_joints * num_dims),
        )
    
    def forward(self, past_motion, sign_image):
        """
        past_motion: [B, T_past, J, C]
        sign_image: [B, 1, H, W]
        return: [B, T_future, J, C]
        """
        B = past_motion.shape[0]
        
        # Encode past
        past_flat = past_motion.reshape(B, -1)
        past_feat = self.past_encoder(past_flat)
        
        # Encode sign
        sign_feat = self.sign_encoder(sign_image)
        
        # Decode
        combined = torch.cat([past_feat, sign_feat], dim=-1)
        output = self.decoder(combined)
        
        # Reshape
        output = output.reshape(B, self.future_frames, self.num_joints, self.num_dims)
        
        return output


class TransformerRegressionModel(nn.Module):
    """
    Transformer 回归模型：更强的架构
    """
    def __init__(self, num_joints=178, num_dims=3, past_frames=40, future_frames=20, 
                 latent_dim=256, num_layers=4, num_heads=4):
        super().__init__()
        self.num_joints = num_joints
        self.num_dims = num_dims
        self.future_frames = future_frames
        
        # Input projection
        self.input_proj = nn.Linear(num_joints * num_dims, latent_dim)
        
        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, past_frames + future_frames, latent_dim) * 0.02)
        
        # Sign encoder
        self.sign_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, latent_dim),
        )
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, 
            dim_feedforward=latent_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(latent_dim, num_joints * num_dims)
        
        # Learnable future queries
        self.future_queries = nn.Parameter(torch.randn(1, future_frames, latent_dim) * 0.02)
    
    def forward(self, past_motion, sign_image):
        """
        past_motion: [B, T_past, J, C]
        sign_image: [B, 1, H, W]
        """
        B, T_past, J, C = past_motion.shape
        
        # Project past motion
        past_flat = past_motion.reshape(B, T_past, -1)  # [B, T, J*C]
        past_emb = self.input_proj(past_flat)  # [B, T, D]
        
        # Sign embedding as first token
        sign_emb = self.sign_encoder(sign_image).unsqueeze(1)  # [B, 1, D]
        
        # Future queries
        future_q = self.future_queries.expand(B, -1, -1)  # [B, T_future, D]
        
        # Concatenate: [sign, past, future_queries]
        seq = torch.cat([sign_emb, past_emb, future_q], dim=1)  # [B, 1+T_past+T_future, D]
        
        # Add positional encoding
        seq = seq + self.pos_enc[:, :seq.shape[1], :]
        
        # Transformer (with causal mask for future)
        # 简化：不用 mask，让模型自己学
        out = self.transformer(seq)
        
        # 取最后 T_future 个输出
        future_out = out[:, -self.future_frames:, :]  # [B, T_future, D]
        
        # Project to poses
        output = self.output_proj(future_out)  # [B, T_future, J*C]
        output = output.reshape(B, self.future_frames, J, C)
        
        return output


def test_regression_model(model_class, model_name, gt_raw, past_raw, sign, 
                          mean_pose, std_pose, device, max_steps=3000):
    """测试回归模型"""
    print(f"\n" + "=" * 70)
    print(f"测试: {model_name}")
    print("=" * 70)
    
    # Normalize
    gt_norm = (gt_raw - mean_pose) / (std_pose + 1e-6)
    past_norm = (past_raw - mean_pose) / (std_pose + 1e-6)
    
    gt_disp = mean_frame_disp(gt_raw)
    gt_norm_disp = mean_frame_disp(gt_norm)
    
    print(f"  GT pixel disp: {gt_disp:.6f}")
    print(f"  GT norm disp: {gt_norm_disp:.6f}")
    
    # 创建模型
    model = model_class(
        num_joints=gt_raw.shape[2],
        num_dims=gt_raw.shape[3],
        past_frames=past_raw.shape[1],
        future_frames=gt_raw.shape[1],
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    
    for step in range(max_steps):
        optimizer.zero_grad()
        
        # 直接预测（不用 Diffusion）
        pred_norm = model(past_norm, sign)
        
        # MSE Loss
        loss_mse = F.mse_loss(pred_norm, gt_norm)
        
        # Velocity Loss
        pred_vel = pred_norm[:, 1:] - pred_norm[:, :-1]
        gt_vel = gt_norm[:, 1:] - gt_norm[:, :-1]
        loss_vel = F.mse_loss(pred_vel, gt_vel)
        
        # 总 loss
        loss = loss_mse + 1.0 * loss_vel
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 500 == 0:
            pred_unnorm = pred_norm * std_pose + mean_pose
            disp = mean_frame_disp(pred_unnorm)
            ratio = disp / (gt_disp + 1e-8)
            
            pred_norm_disp = mean_frame_disp(pred_norm)
            
            print(f"  Step {step}: loss={loss.item():.6f}, "
                  f"pred_norm_disp={pred_norm_disp:.6f}, ratio={ratio:.4f}")
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        pred_norm = model(past_norm, sign)
        pred_unnorm = pred_norm * std_pose + mean_pose
        
        disp = mean_frame_disp(pred_unnorm)
        mse = F.mse_loss(pred_unnorm, gt_raw).item()
        ratio = disp / (gt_disp + 1e-8)
        
        # Velocity match
        pred_vel = pred_unnorm[:, 1:] - pred_unnorm[:, :-1]
        gt_vel_raw = gt_raw[:, 1:] - gt_raw[:, :-1]
        vel_mse = F.mse_loss(pred_vel, gt_vel_raw).item()
        
        print(f"\n  最终结果:")
        print(f"    MSE: {mse:.6f}")
        print(f"    disp: {disp:.6f}")
        print(f"    disp_ratio: {ratio:.4f}")
        print(f"    vel_mse: {vel_mse:.6f}")
        
        # PCK
        pred_np = pred_unnorm[0].cpu().numpy()
        gt_np = gt_raw[0].cpu().numpy()
        per_joint_error = np.sqrt(((pred_np - gt_np) ** 2).sum(axis=-1))
        
        print(f"\n    PCK@0.05: {(per_joint_error < 0.05).mean():.2%}")
        print(f"    PCK@0.1: {(per_joint_error < 0.1).mean():.2%}")
        print(f"    PCK@0.2: {(per_joint_error < 0.2).mean():.2%}")
    
    return ratio, model


def test_velocity_regression(gt_raw, past_raw, sign, mean_pose, std_pose, device, max_steps=3000):
    """直接预测 velocity 的回归模型"""
    print(f"\n" + "=" * 70)
    print(f"测试: Velocity Regression")
    print("=" * 70)
    
    # Normalize past
    past_norm = (past_raw - mean_pose) / (std_pose + 1e-6)
    
    # 计算 GT velocity (pixel 空间)
    gt_vel = gt_raw[:, 1:] - gt_raw[:, :-1]  # [B, 19, J, C]
    
    # Velocity 的 normalize（用 velocity 自己的 std）
    vel_mean = gt_vel.mean()
    vel_std = gt_vel.std() + 1e-6
    gt_vel_norm = (gt_vel - vel_mean) / vel_std
    
    gt_disp = mean_frame_disp(gt_raw)
    
    print(f"  GT pixel disp: {gt_disp:.6f}")
    print(f"  Velocity std: {vel_std.item():.6f}")
    print(f"  GT vel_norm range: [{gt_vel_norm.min():.2f}, {gt_vel_norm.max():.2f}]")
    
    # 创建模型（预测 19 帧 velocity）
    model = TransformerRegressionModel(
        num_joints=gt_raw.shape[2],
        num_dims=gt_raw.shape[3],
        past_frames=past_raw.shape[1],
        future_frames=gt_vel.shape[1],  # 19
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    
    for step in range(max_steps):
        optimizer.zero_grad()
        
        pred_vel_norm = model(past_norm, sign)
        loss = F.mse_loss(pred_vel_norm, gt_vel_norm)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 500 == 0:
            # Unnormalize velocity
            pred_vel = pred_vel_norm * vel_std + vel_mean
            
            # 重建位置
            start_pos = past_raw[:, -1:, :, :]
            positions = [start_pos]
            for t in range(pred_vel.shape[1]):
                next_pos = positions[-1] + pred_vel[:, t:t+1]
                positions.append(next_pos)
            pred_pos = torch.cat(positions[1:], dim=1)
            
            disp = mean_frame_disp(pred_pos)
            ratio = disp / (gt_disp + 1e-8)
            
            print(f"  Step {step}: loss={loss.item():.6f}, disp={disp:.6f}, ratio={ratio:.4f}")
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        pred_vel_norm = model(past_norm, sign)
        pred_vel = pred_vel_norm * vel_std + vel_mean
        
        # 重建位置
        start_pos = past_raw[:, -1:, :, :]
        positions = [start_pos]
        for t in range(pred_vel.shape[1]):
            next_pos = positions[-1] + pred_vel[:, t:t+1]
            positions.append(next_pos)
        pred_pos = torch.cat(positions[1:], dim=1)  # [B, 19, J, C]
        
        # 对比需要用 GT 的后 19 帧
        gt_19 = gt_raw[:, 1:]  # 跳过第一帧
        
        disp = mean_frame_disp(pred_pos)
        mse = F.mse_loss(pred_pos, gt_19).item()
        ratio = disp / (gt_disp + 1e-8)
        
        print(f"\n  最终结果:")
        print(f"    MSE: {mse:.6f}")
        print(f"    disp: {disp:.6f}")
        print(f"    disp_ratio: {ratio:.4f}")
    
    return ratio, model


def main():
    print("=" * 70)
    print("方案 4: 直接 Regression（不用 Diffusion）")
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
    
    # 先找一个运动量大的样本
    print("\n寻找运动量大的样本...")
    best_idx = 36
    best_disp = 0
    
    for idx in range(min(200, len(base_ds))):
        try:
            s = base_ds[idx]
            b = zero_pad_collator([s])
            gt = sanitize_btjc(b["data"][:1])
            disp = mean_frame_disp(gt)
            if disp > best_disp:
                best_disp = disp
                best_idx = idx
        except:
            continue
    
    print(f"  最大运动样本: {best_idx}, disp={best_disp:.6f}")
    
    sample = base_ds[best_idx]
    batch = zero_pad_collator([sample])
    
    cond = batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(batch["data"][:1]).to(device)
    
    stats = torch.load(stats_path, map_location=device)
    mean_pose = stats["mean"].float().view(1, 1, -1, 3).to(device)
    std_pose = stats["std"].float().view(1, 1, -1, 3).to(device)
    
    print(f"\n样本 {best_idx}:")
    print(f"  past: {past_raw.shape}")
    print(f"  gt: {gt_raw.shape}")
    print(f"  sign: {sign.shape}")
    
    # 详细分析
    gt_np = gt_raw[0].cpu().numpy()
    
    # 全体关节
    all_disp = np.abs(gt_np[1:] - gt_np[:-1]).mean()
    print(f"\n  运动量分析 (pixel 空间):")
    print(f"    全体关节 mean disp: {all_disp:.4f} px")
    
    # 手部关节 (136-178)
    hand_disp = np.abs(gt_np[1:, 136:178] - gt_np[:-1, 136:178]).mean()
    print(f"    手部关节 mean disp: {hand_disp:.4f} px")
    
    # 身体关节 (0-33)
    body_disp = np.abs(gt_np[1:, :33] - gt_np[:-1, :33]).mean()
    print(f"    身体关节 mean disp: {body_disp:.4f} px")
    
    # 第一帧和最后一帧的差异
    total_motion = np.abs(gt_np[-1] - gt_np[0]).mean()
    print(f"    首尾帧差异: {total_motion:.4f} px")
    
    results = {}
    
    # Sanity Check: 直接输出 GT
    print("\n" + "=" * 70)
    print("Sanity Check: 直接输出 GT（验证评估代码）")
    print("=" * 70)
    sanity_disp = mean_frame_disp(gt_raw)
    gt_disp = sanity_disp
    print(f"  GT disp: {sanity_disp:.6f}")
    print(f"  GT disp_ratio: {sanity_disp / (sanity_disp + 1e-8):.4f} (应该=1.0)")
    results['Sanity_GT'] = 1.0
    
    # Baseline: 输出 past 的最后一帧（完全静态）
    print("\n" + "=" * 70)
    print("Baseline: 输出 past 最后一帧（完全静态）")
    print("=" * 70)
    static_pred = past_raw[:, -1:].expand(-1, gt_raw.shape[1], -1, -1)
    static_disp = mean_frame_disp(static_pred)
    static_mse = F.mse_loss(static_pred, gt_raw).item()
    print(f"  Static pred disp: {static_disp:.6f}")
    print(f"  Static pred MSE: {static_mse:.6f}")
    print(f"  这就是'预测静态均值'的 baseline")
    results['Static_Baseline'] = static_disp / (gt_disp + 1e-8)
    
    # 测试 1: 简单 MLP 回归
    ratio_1, _ = test_regression_model(
        SimpleRegressionModel, "Simple MLP",
        gt_raw, past_raw, sign, mean_pose, std_pose, device,
        max_steps=3000
    )
    results['Simple_MLP'] = ratio_1
    
    # 测试 2: Transformer 回归
    ratio_2, _ = test_regression_model(
        TransformerRegressionModel, "Transformer",
        gt_raw, past_raw, sign, mean_pose, std_pose, device,
        max_steps=3000
    )
    results['Transformer'] = ratio_2
    
    # 测试 3: Velocity 回归
    ratio_3, _ = test_velocity_regression(
        gt_raw, past_raw, sign, mean_pose, std_pose, device,
        max_steps=3000
    )
    results['Velocity_Reg'] = ratio_3
    
    # 总结
    print("\n" + "=" * 70)
    print("结果对比")
    print("=" * 70)
    print(f"\n{'模型':<20} | {'disp_ratio':<12} | {'评价'}")
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
    print(f"\n最佳: {best} (ratio={results[best]:.4f})")
    
    if results[best] < 0.3:
        print(f"\n⚠️ 所有方案都失败！")
        print(f"可能的原因:")
        print(f"  1. 数据本身运动量太小")
        print(f"  2. 需要更强的运动约束")
        print(f"  3. 需要换更大运动的样本测试")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()