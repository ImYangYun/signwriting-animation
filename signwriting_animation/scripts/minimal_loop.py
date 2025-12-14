# -*- coding: utf-8 -*-
"""
深入调试：为什么训练过程中模型坍缩到输出常数？

观察：
- Step 0: ratio=1.4 (有运动)
- Step 1000+: ratio=0 (静态)

模型学会了输出常数，这是 MSE loss 的"捷径"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("=" * 70)
print("深入调试：训练坍缩问题")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusionV2

K = 178
D = 3
T_future = 20
T_past = 40
latent_dim = 256

# ============================================================
print("\n" + "=" * 70)
print("关于 rescale_timesteps 报错的说明")
print("=" * 70)

print("""
报错原因：
  rescale_timesteps=True 时，t 被缩放为浮点数 (0.0, 500.0, 875.0)
  但 TimestepEmbedder 用 pe[timesteps] 做索引，需要整数！

解决方案：
  1. 不用 rescale_timesteps，保持 t 为整数 0-7
  2. 或者修改 TimestepEmbedder 来处理浮点数
  3. 或者手动缩放后转为整数: t_scaled = (t * 1000 / 8).long()

目前我们先不管这个问题，专注于为什么模型会坍缩到输出常数。
""")

# ============================================================
print("\n" + "=" * 70)
print("实验 1: 观察训练过程中各层的变化")
print("=" * 70)

# 创建有内容的数据
gt = torch.zeros(1, K, D, T_future).to(device)
for t_idx in range(T_future):
    gt[:, :, 0, t_idx] = t_idx * 0.5

gt_disp = (gt[:, :, :, 1:] - gt[:, :, :, :-1]).abs().mean().item()
print(f"GT displacement: {gt_disp:.4f}")

past = torch.zeros(1, K, D, T_past).to(device)
for t_idx in range(T_past):
    past[:, :, 0, t_idx] = (t_idx - T_past) * 0.5

sign = torch.randn(1, 3, 224, 224).to(device)

# 简单的 x_t (用 past 最后帧)
x_t = past[:, :, :, -1:].expand(-1, -1, -1, T_future).clone()
t = torch.tensor([0]).to(device)

# 创建模型
model = SignWritingToPoseDiffusionV2(
    num_keypoints=K,
    num_dims_per_keypoint=D,
    residual_scale=0.1,
    use_mean_pool=True,
).to(device)
model.verbose = False

# 记录初始参数
initial_pose_proj_weight = model.pose_projection.out_proj.weight.clone().detach()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("\n训练并观察各层...")

for step in range(2001):
    optimizer.zero_grad()
    
    pred = model(x_t, t, past, sign)
    
    # Loss
    loss_mse = F.mse_loss(pred, gt)
    pred_vel = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_vel = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    loss_vel = F.mse_loss(pred_vel, gt_vel)
    
    loss = loss_mse + loss_vel
    loss.backward()
    
    # 记录梯度信息
    if step % 200 == 0:
        pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
        ratio = pred_disp / gt_disp
        
        # 检查各层梯度
        grad_info = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_info[name] = param.grad.abs().mean().item()
        
        # 只打印关键层
        key_grads = {k: v for k, v in grad_info.items() if 'pose_projection' in k or 'future_motion' in k}
        
        print(f"\nStep {step}: loss={loss.item():.4f}, mse={loss_mse.item():.4f}, vel={loss_vel.item():.4f}")
        print(f"  pred_disp={pred_disp:.6f}, ratio={ratio:.4f}")
        print(f"  pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
        
        # 检查 pose_projection 输出层的权重变化
        current_weight = model.pose_projection.out_proj.weight
        weight_change = (current_weight - initial_pose_proj_weight).abs().mean().item()
        print(f"  pose_projection weight change: {weight_change:.6f}")
        
        # 检查输出的每一帧
        pred_per_frame = pred[0, 0, 0, :].detach().cpu().numpy()
        print(f"  pred frame values (kp0, dim0): {pred_per_frame[:5]}...")
    
    optimizer.step()

# ============================================================
print("\n" + "=" * 70)
print("实验 2: 用更强的 velocity loss")
print("=" * 70)

model2 = SignWritingToPoseDiffusionV2(
    num_keypoints=K,
    num_dims_per_keypoint=D,
    residual_scale=0.1,
    use_mean_pool=True,
).to(device)
model2.verbose = False

optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

print("\n训练 (velocity_weight=10)...")

for step in range(2001):
    optimizer2.zero_grad()
    
    pred = model2(x_t, t, past, sign)
    
    loss_mse = F.mse_loss(pred, gt)
    pred_vel = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_vel = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    loss_vel = F.mse_loss(pred_vel, gt_vel)
    
    # 更强的 velocity loss
    loss = loss_mse + 10.0 * loss_vel
    
    loss.backward()
    optimizer2.step()
    
    if step % 400 == 0:
        pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
        ratio = pred_disp / gt_disp
        print(f"  Step {step}: loss={loss.item():.4f}, pred_disp={pred_disp:.4f}, ratio={ratio:.4f}")

# 最终测试
model2.eval()
with torch.no_grad():
    pred = model2(x_t, t, past, sign)
    pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
    ratio = pred_disp / gt_disp
    print(f"\n最终 (vel_weight=10): pred_disp={pred_disp:.4f}, ratio={ratio:.4f}")

# ============================================================
print("\n" + "=" * 70)
print("实验 3: 用 displacement loss 直接约束")
print("=" * 70)

model3 = SignWritingToPoseDiffusionV2(
    num_keypoints=K,
    num_dims_per_keypoint=D,
    residual_scale=0.1,
    use_mean_pool=True,
).to(device)
model3.verbose = False

optimizer3 = torch.optim.AdamW(model3.parameters(), lr=1e-3)

print("\n训练 (with displacement loss)...")

for step in range(2001):
    optimizer3.zero_grad()
    
    pred = model3(x_t, t, past, sign)
    
    loss_mse = F.mse_loss(pred, gt)
    
    pred_vel = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_vel = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    loss_vel = F.mse_loss(pred_vel, gt_vel)
    
    # Displacement loss: 鼓励输出有运动
    pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
    gt_disp_tensor = torch.tensor(gt_disp).to(device)
    loss_disp = F.mse_loss(pred_disp, gt_disp_tensor)
    
    loss = loss_mse + loss_vel + 10.0 * loss_disp
    
    loss.backward()
    optimizer3.step()
    
    if step % 400 == 0:
        pred_disp_val = pred_disp.item()
        ratio = pred_disp_val / gt_disp
        print(f"  Step {step}: loss={loss.item():.4f}, disp_loss={loss_disp.item():.4f}, pred_disp={pred_disp_val:.4f}, ratio={ratio:.4f}")

# 最终测试
model3.eval()
with torch.no_grad():
    pred = model3(x_t, t, past, sign)
    pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
    ratio = pred_disp / gt_disp
    print(f"\n最终 (with disp_loss): pred_disp={pred_disp:.4f}, ratio={ratio:.4f}")

# ============================================================
print("\n" + "=" * 70)
print("实验 4: 检查是否是 MeanPool 的问题")
print("=" * 70)

print("测试 Concat 模式 (use_mean_pool=False)...")

model4 = SignWritingToPoseDiffusionV2(
    num_keypoints=K,
    num_dims_per_keypoint=D,
    residual_scale=0.1,
    use_mean_pool=False,  # 不用 MeanPool!
).to(device)
model4.verbose = False

optimizer4 = torch.optim.AdamW(model4.parameters(), lr=1e-3)

for step in range(2001):
    optimizer4.zero_grad()
    
    pred = model4(x_t, t, past, sign)
    
    loss_mse = F.mse_loss(pred, gt)
    pred_vel = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_vel = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    loss_vel = F.mse_loss(pred_vel, gt_vel)
    
    loss = loss_mse + loss_vel
    loss.backward()
    optimizer4.step()
    
    if step % 400 == 0:
        pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
        ratio = pred_disp / gt_disp
        print(f"  Step {step}: loss={loss.item():.4f}, pred_disp={pred_disp:.4f}, ratio={ratio:.4f}")

# 最终测试
model4.eval()
with torch.no_grad():
    pred = model4(x_t, t, past, sign)
    pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
    ratio = pred_disp / gt_disp
    print(f"\n最终 (Concat模式): pred_disp={pred_disp:.4f}, ratio={ratio:.4f}")

# ============================================================
print("\n" + "=" * 70)
print("实验 5: 极简模型 - 只用 OutputProcessMLP")
print("=" * 70)

class SimpleModel(nn.Module):
    """极简模型：直接从 past 预测 future"""
    def __init__(self):
        super().__init__()
        input_dim = K * D * T_past
        hidden_dim = 512
        output_dim = K * D * T_future
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, past):
        B = past.shape[0]
        x = past.reshape(B, -1)
        y = self.net(x)
        return y.reshape(B, K, D, T_future)

simple_model = SimpleModel().to(device)
optimizer5 = torch.optim.AdamW(simple_model.parameters(), lr=1e-3)

print("\n训练极简模型...")

for step in range(2001):
    optimizer5.zero_grad()
    
    pred = simple_model(past)
    
    loss_mse = F.mse_loss(pred, gt)
    pred_vel = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_vel = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    loss_vel = F.mse_loss(pred_vel, gt_vel)
    
    loss = loss_mse + loss_vel
    loss.backward()
    optimizer5.step()
    
    if step % 400 == 0:
        pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
        ratio = pred_disp / gt_disp
        print(f"  Step {step}: loss={loss.item():.4f}, pred_disp={pred_disp:.4f}, ratio={ratio:.4f}")

# 最终测试
simple_model.eval()
with torch.no_grad():
    pred = simple_model(past)
    pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
    ratio = pred_disp / gt_disp
    print(f"\n最终 (极简模型): pred_disp={pred_disp:.4f}, ratio={ratio:.4f}")

# ============================================================
print("\n" + "=" * 70)
print("总结")
print("=" * 70)

print("""
实验结果对比:
1. 标准训练 (vel_weight=1) - ratio=?
2. 强 velocity loss (vel_weight=10) - ratio=?
3. 加 displacement loss - ratio=?
4. Concat 模式 - ratio=?
5. 极简模型 - ratio=?

如果极简模型成功但复杂模型失败，问题在模型架构
如果所有模型都失败，问题可能在数据或 loss 设计
""")