# -*- coding: utf-8 -*-
"""
修正测试 - 对比之前成功的 Regression 设置

之前成功: past 有内容，模型从 past 预测 future
现在测试中: past=0, x_t=0 —— 这当然会失败！

同时修复 TimestepEmbedder 问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("=" * 70)
print("修正测试")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess
from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType

K = 178
D = 3
T_future = 20
T_past = 40
latent_dim = 256

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

# ============================================================
print("\n" + "=" * 70)
print("问题分析 1: TimestepEmbedder")
print("=" * 70)

pos_enc = PositionalEncoding(latent_dim, dropout=0.0)  # 关闭 dropout
timestep_emb = TimestepEmbedder(latent_dim, pos_enc).to(device)

# 测试不同 t
print("\n师姐的 TimestepEmbedder 输出:")
for t_val in [0, 1, 4, 7, 100, 500, 999]:
    t = torch.tensor([t_val]).to(device)
    emb = timestep_emb(t)
    print(f"  t={t_val}: shape={emb.shape}, mean={emb.mean():.4f}, std={emb.std():.4f}, range=[{emb.min():.2f}, {emb.max():.2f}]")

# 检查 t 范围的影响
t_0 = torch.tensor([0]).to(device)
t_7 = torch.tensor([7]).to(device)
t_100 = torch.tensor([100]).to(device)
t_999 = torch.tensor([999]).to(device)

emb_0 = timestep_emb(t_0)
emb_7 = timestep_emb(t_7)
emb_100 = timestep_emb(t_100)
emb_999 = timestep_emb(t_999)

print(f"\n不同 t 之间的差异:")
print(f"  t=0 vs t=7: {(emb_0 - emb_7).abs().mean().item():.4f}")
print(f"  t=0 vs t=100: {(emb_0 - emb_100).abs().mean().item():.4f}")
print(f"  t=0 vs t=999: {(emb_0 - emb_999).abs().mean().item():.4f}")

print("""
⚠️ 注意：你的 Diffusion 只有 8 步，所以 t 的范围是 0-7
   t=0 vs t=7 的差异只有 0.07，这可能是正常的（因为范围太小）
   
   但是！rescale_timesteps=False 时，t 不会被缩放到 0-1000
   所以 t=0 和 t=7 的 embedding 差异确实很小
""")

# ============================================================
print("\n" + "=" * 70)
print("问题分析 2: Diffusion timestep 缩放")
print("=" * 70)

DIFFUSION_STEPS = 8
betas = cosine_beta_schedule(DIFFUSION_STEPS).numpy()

diffusion = GaussianDiffusion(
    betas=betas,
    model_mean_type=ModelMeanType.START_X,
    model_var_type=ModelVarType.FIXED_SMALL,
    loss_type=LossType.MSE,
    rescale_timesteps=False,  # 不缩放！
)

print(f"num_timesteps: {diffusion.num_timesteps}")
print(f"rescale_timesteps: False")

# 测试 _scale_timesteps
for t_val in [0, 4, 7]:
    t = torch.tensor([t_val]).to(device)
    t_scaled = diffusion._scale_timesteps(t)
    print(f"  t={t_val} -> scaled={t_scaled.item()}")

print("""
⚠️ 因为 rescale_timesteps=False，t 直接传入模型
   t 的范围只有 0-7，这对 TimestepEmbedder 来说差异太小了！
   
   解决方案：设置 rescale_timesteps=True
   这样 t 会被缩放到 0-1000 范围
""")

# ============================================================
print("\n" + "=" * 70)
print("修正测试: Regression (用有内容的 past)")
print("=" * 70)

from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusionV2

# 创建有内容的数据
# GT: 有明显运动
gt = torch.zeros(1, K, D, T_future).to(device)
for t_idx in range(T_future):
    gt[:, :, 0, t_idx] = t_idx * 0.5  # x 方向线性移动

gt_disp = (gt[:, :, :, 1:] - gt[:, :, :, :-1]).abs().mean().item()
print(f"GT displacement: {gt_disp:.4f}")

# past: 也有内容（和 GT 连续）
past = torch.zeros(1, K, D, T_past).to(device)
for t_idx in range(T_past):
    past[:, :, 0, t_idx] = (t_idx - T_past) * 0.5  # 从负值开始

past_disp = (past[:, :, :, 1:] - past[:, :, :, :-1]).abs().mean().item()
print(f"Past displacement: {past_disp:.4f}")

sign = torch.randn(1, 3, 224, 224).to(device)

# 创建模型
model = SignWritingToPoseDiffusionV2(
    num_keypoints=K,
    num_dims_per_keypoint=D,
    residual_scale=0.1,
    use_mean_pool=True,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("\nRegression 训练 (有内容的 past)...")
model.train()

for step in range(2000):
    optimizer.zero_grad()
    
    # Regression: t=0, x_t 可以是零或者 past 的最后帧
    t = torch.tensor([0]).to(device)
    
    # 用 past 的最后一帧作为 x_t 的起点（更合理）
    x_t = past[:, :, :, -1:].expand(-1, -1, -1, T_future).clone()
    
    pred = model(x_t, t, past, sign)
    
    # Loss: MSE + velocity
    loss_mse = F.mse_loss(pred, gt)
    
    pred_vel = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_vel = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    loss_vel = F.mse_loss(pred_vel, gt_vel)
    
    loss = loss_mse + loss_vel
    loss.backward()
    optimizer.step()
    
    if step % 400 == 0:
        pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
        ratio = pred_disp / gt_disp if gt_disp > 0 else 0
        print(f"  Step {step}: loss={loss.item():.6f}, pred_disp={pred_disp:.4f}, ratio={ratio:.4f}")

# 测试
model.eval()
with torch.no_grad():
    pred = model(x_t, t, past, sign)
    pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
    ratio = pred_disp / gt_disp if gt_disp > 0 else 0
    
    print(f"\nRegression 最终: pred_disp={pred_disp:.4f}, ratio={ratio:.4f}")
    
    if ratio > 0.5:
        print("✓ Regression 成功！模型架构没问题")
    else:
        print("⚠️ Regression 失败，需要进一步检查")

# ============================================================
print("\n" + "=" * 70)
print("修正测试: Diffusion (rescale_timesteps=True)")
print("=" * 70)

# 重新创建 Diffusion，启用 timestep 缩放
diffusion_scaled = GaussianDiffusion(
    betas=betas,
    model_mean_type=ModelMeanType.START_X,
    model_var_type=ModelVarType.FIXED_SMALL,
    loss_type=LossType.MSE,
    rescale_timesteps=True,  # 启用缩放！
)

print(f"rescale_timesteps: True")

# 测试缩放后的 t
for t_val in [0, 4, 7]:
    t = torch.tensor([t_val]).to(device)
    t_scaled = diffusion_scaled._scale_timesteps(t)
    print(f"  t={t_val} -> scaled={t_scaled.item():.1f}")

# 创建新模型
model2 = SignWritingToPoseDiffusionV2(
    num_keypoints=K,
    num_dims_per_keypoint=D,
    residual_scale=0.1,
    use_mean_pool=True,
).to(device)

optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

print("\nDiffusion 训练 (rescale_timesteps=True)...")
model2.train()

for step in range(2000):
    optimizer2.zero_grad()
    
    t = torch.randint(0, DIFFUSION_STEPS, (1,), device=device)
    noise = torch.randn_like(gt)
    x_t = diffusion_scaled.q_sample(gt, t, noise=noise)
    
    # 使用缩放后的 t
    t_scaled = diffusion_scaled._scale_timesteps(t)
    pred = model2(x_t, t_scaled, past, sign)
    
    loss = F.mse_loss(pred, gt)
    loss.backward()
    optimizer2.step()
    
    if step % 400 == 0:
        pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
        ratio = pred_disp / gt_disp if gt_disp > 0 else 0
        print(f"  Step {step}: loss={loss.item():.6f}, pred_disp={pred_disp:.4f}, ratio={ratio:.4f}, t={t.item()}, t_scaled={t_scaled.item():.1f}")

# 测试
model2.eval()
with torch.no_grad():
    t = torch.tensor([0]).to(device)
    t_scaled = diffusion_scaled._scale_timesteps(t)
    x_t = diffusion_scaled.q_sample(gt, t, noise=torch.randn_like(gt))
    
    pred = model2(x_t, t_scaled, past, sign)
    pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
    ratio = pred_disp / gt_disp if gt_disp > 0 else 0
    
    print(f"\nDiffusion (rescale) 最终: pred_disp={pred_disp:.4f}, ratio={ratio:.4f}")

# ============================================================
print("\n" + "=" * 70)
print("总结")
print("=" * 70)

print("""
发现的问题:

1. TimestepEmbedder 对小范围 t (0-7) 不敏感
   - 因为 rescale_timesteps=False，t 直接传入
   - t 的范围只有 0-7，embedding 差异很小
   
2. 之前测试中 Regression 失败是因为:
   - past = zeros (没有条件信息)
   - x_t = zeros (没有输入信息)
   - 模型没有任何信息来源
   
解决方案:

1. 设置 rescale_timesteps=True
   - 这样 t 会被缩放到 0-1000
   - TimestepEmbedder 能产生更明显的差异

2. 确保 past 有内容
   - Regression/Diffusion 都需要条件信息
""")