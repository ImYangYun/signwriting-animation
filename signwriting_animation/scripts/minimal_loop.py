# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("=" * 70)
print("Minimal Loop - 验证修复")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusionV2
from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType

K = 178  # keypoints
D = 3    # dims
T_past = 40
T_future = 20
latent_dim = 256
DIFFUSION_STEPS = 8

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

print("\n创建测试数据...")

# GT: 线性运动 (x 坐标随时间增加)
gt_bjct = torch.zeros(1, K, D, T_future).to(device)
for t_idx in range(T_future):
    gt_bjct[:, :, 0, t_idx] = t_idx * 0.5  # x 随时间增加

gt_disp = (gt_bjct[:, :, :, 1:] - gt_bjct[:, :, :, :-1]).abs().mean().item()
print(f"GT displacement: {gt_disp:.4f}")

# Past: 延续 GT 的运动模式
past_bjct = torch.zeros(1, K, D, T_past).to(device)
for t_idx in range(T_past):
    past_bjct[:, :, 0, t_idx] = (t_idx - T_past) * 0.5

# Sign image (随机)
sign_img = torch.randn(1, 3, 224, 224).to(device)

# ============================================================
print("\n" + "=" * 70)
print("测试 1: Regression 单样本 Overfit")
print("=" * 70)

model_reg = SignWritingToPoseDiffusionV2(
    num_keypoints=K,
    num_dims_per_keypoint=D,
    use_mean_pool=True,
).to(device)
model_reg.verbose = False

optimizer_reg = torch.optim.AdamW(model_reg.parameters(), lr=1e-3)

print("\n训练 Regression...")
for step in range(2001):
    optimizer_reg.zero_grad()
    
    # Regression: t=0, x_t = gt (无噪声)
    t = torch.tensor([0]).to(device)
    pred = model_reg(gt_bjct, t, past_bjct, sign_img)
    
    # Loss
    loss_mse = F.mse_loss(pred, gt_bjct)
    pred_vel = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_vel = gt_bjct[:, :, :, 1:] - gt_bjct[:, :, :, :-1]
    loss_vel = F.mse_loss(pred_vel, gt_vel)
    
    loss = loss_mse + loss_vel
    loss.backward()
    optimizer_reg.step()
    
    if step % 400 == 0:
        pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
        ratio = pred_disp / gt_disp
        print(f"  Step {step}: loss={loss.item():.4f}, pred_disp={pred_disp:.4f}, ratio={ratio:.4f}")

# 最终测试
model_reg.eval()
with torch.no_grad():
    t = torch.tensor([0]).to(device)
    pred = model_reg(gt_bjct, t, past_bjct, sign_img)
    pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
    ratio_reg = pred_disp / gt_disp
    print(f"\n✓ Regression 最终: pred_disp={pred_disp:.4f}, ratio={ratio_reg:.4f}")

# ============================================================
print("\n" + "=" * 70)
print("测试 2: Diffusion 单样本 Overfit")
print("=" * 70)

betas = cosine_beta_schedule(DIFFUSION_STEPS).numpy()
diffusion = GaussianDiffusion(
    betas=betas,
    model_mean_type=ModelMeanType.START_X,
    model_var_type=ModelVarType.FIXED_SMALL,
    loss_type=LossType.MSE,
    rescale_timesteps=False,
)

model_diff = SignWritingToPoseDiffusionV2(
    num_keypoints=K,
    num_dims_per_keypoint=D,
    use_mean_pool=True,
).to(device)
model_diff.verbose = False

optimizer_diff = torch.optim.AdamW(model_diff.parameters(), lr=1e-3)

print("\n训练 Diffusion...")
for step in range(2001):
    optimizer_diff.zero_grad()
    
    # 随机 timestep
    t = torch.randint(0, DIFFUSION_STEPS, (1,), device=device)
    
    # 加噪声
    noise = torch.randn_like(gt_bjct)
    x_t = diffusion.q_sample(gt_bjct, t, noise=noise)
    
    # 预测 x0
    pred = model_diff(x_t, t, past_bjct, sign_img)
    
    # Loss
    loss_mse = F.mse_loss(pred, gt_bjct)
    pred_vel = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_vel = gt_bjct[:, :, :, 1:] - gt_bjct[:, :, :, :-1]
    loss_vel = F.mse_loss(pred_vel, gt_vel)
    
    loss = loss_mse + loss_vel
    loss.backward()
    optimizer_diff.step()
    
    if step % 400 == 0:
        # 用 t=0 测试
        with torch.no_grad():
            t_test = torch.tensor([0]).to(device)
            x_t_test = diffusion.q_sample(gt_bjct, t_test, noise=torch.randn_like(gt_bjct))
            pred_test = model_diff(x_t_test, t_test, past_bjct, sign_img)
            pred_disp = (pred_test[:, :, :, 1:] - pred_test[:, :, :, :-1]).abs().mean().item()
            ratio = pred_disp / gt_disp
        print(f"  Step {step}: loss={loss.item():.4f}, pred_disp={pred_disp:.4f}, ratio={ratio:.4f}, t={t.item()}")

# 最终测试
model_diff.eval()
with torch.no_grad():
    t = torch.tensor([0]).to(device)
    x_t = diffusion.q_sample(gt_bjct, t, noise=torch.randn_like(gt_bjct))
    pred = model_diff(x_t, t, past_bjct, sign_img)
    pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
    ratio_diff = pred_disp / gt_disp
    print(f"\n✓ Diffusion 最终: pred_disp={pred_disp:.4f}, ratio={ratio_diff:.4f}")

# ============================================================
print("\n" + "=" * 70)
print("测试 3: p_sample_loop 采样")
print("=" * 70)

class ConditionalWrapper(nn.Module):
    def __init__(self, model, past, sign):
        super().__init__()
        self.model = model
        self.past = past
        self.sign = sign
    
    def forward(self, x, t, **kwargs):
        return self.model(x, t, self.past, self.sign)

wrapped = ConditionalWrapper(model_diff, past_bjct, sign_img)

print("\n使用 p_sample_loop 采样...")
with torch.no_grad():
    sampled = diffusion.p_sample_loop(
        model=wrapped,
        shape=(1, K, D, T_future),
        clip_denoised=False,
        model_kwargs={"y": {}},
        progress=False,
    )
    
    sampled_disp = (sampled[:, :, :, 1:] - sampled[:, :, :, :-1]).abs().mean().item()
    ratio_sample = sampled_disp / gt_disp
    print(f"✓ 采样结果: sampled_disp={sampled_disp:.4f}, ratio={ratio_sample:.4f}")

# ============================================================
print("\n" + "=" * 70)
print("📊 结果汇总")
print("=" * 70)

print(f"""
| 测试 | ratio | 结果 |
|------|-------|------|
| Regression | {ratio_reg:.4f} | {'✅ 成功' if ratio_reg > 0.5 else '❌ 失败'} |
| Diffusion | {ratio_diff:.4f} | {'✅ 成功' if ratio_diff > 0.5 else '❌ 失败'} |
| p_sample_loop | {ratio_sample:.4f} | {'✅ 成功' if ratio_sample > 0.5 else '❌ 失败'} |
""")

if ratio_reg > 0.5 and ratio_diff > 0.5:
    print("🎉 修复成功！模型可以正常学习运动！")
else:
    print("⚠️ 还有问题，需要进一步调试")