# -*- coding: utf-8 -*-
"""
逐层检查：时间差异在哪里丢失？

策略：每一层后检查不同时间步的输出是否有差异
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("逐层检查时间差异")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from CAMDM.network.models import MotionProcess

K = 178
D = 3
T_future = 20
latent_dim = 256

# 创建有时间差异的输入
x_bjct = torch.zeros(1, K, D, T_future).to(device)
for t_idx in range(T_future):
    x_bjct[:, :, 0, t_idx] = t_idx * 0.5  # x 随时间增加

print(f"\n输入 x_bjct: shape={x_bjct.shape}")
print(f"  帧间差异: {(x_bjct[:, :, :, 1:] - x_bjct[:, :, :, :-1]).abs().mean().item():.4f}")

# ============================================================
print("\n" + "=" * 70)
print("检查 1: MotionProcess")
print("=" * 70)

motion_proc = MotionProcess(K * D, latent_dim).to(device)

# MotionProcess 期望 [B, J, C, T] 输入，输出 [T, B, D]
motion_out = motion_proc(x_bjct)
print(f"MotionProcess 输出: shape={motion_out.shape}")

# 检查不同时间步的差异
frame_diffs = []
for t in range(1, T_future):
    diff = (motion_out[t] - motion_out[t-1]).abs().mean().item()
    frame_diffs.append(diff)
print(f"  帧间差异: mean={sum(frame_diffs)/len(frame_diffs):.6f}, min={min(frame_diffs):.6f}, max={max(frame_diffs):.6f}")

# 检查第一帧和最后一帧
diff_first_last = (motion_out[0] - motion_out[-1]).abs().mean().item()
print(f"  第一帧 vs 最后帧: {diff_first_last:.6f}")

# ============================================================
print("\n" + "=" * 70)
print("检查 2: MotionProcess + future_time_proj")
print("=" * 70)

future_time_proj = nn.Sequential(
    nn.Linear(1, latent_dim),
    nn.SiLU(),
    nn.Linear(latent_dim, latent_dim)
).to(device)

B = 1
t = torch.linspace(0, 1, steps=T_future, device=device).view(T_future, 1, 1)
t_latent = future_time_proj(t).expand(-1, B, -1)

# 加上时间编码
motion_with_time = motion_out + 0.1 * t_latent

print(f"加时间编码后: shape={motion_with_time.shape}")
frame_diffs = []
for t_idx in range(1, T_future):
    diff = (motion_with_time[t_idx] - motion_with_time[t_idx-1]).abs().mean().item()
    frame_diffs.append(diff)
print(f"  帧间差异: mean={sum(frame_diffs)/len(frame_diffs):.6f}")

diff_first_last = (motion_with_time[0] - motion_with_time[-1]).abs().mean().item()
print(f"  第一帧 vs 最后帧: {diff_first_last:.6f}")

# ============================================================
print("\n" + "=" * 70)
print("检查 3: 通过 Transformer")
print("=" * 70)

encoder_layer = nn.TransformerEncoderLayer(
    d_model=latent_dim, nhead=4,
    dim_feedforward=1024, dropout=0.0,
    activation="gelu", batch_first=False,
)
transformer = nn.TransformerEncoder(encoder_layer, num_layers=8).to(device)

# 只用 future motion 作为输入
trans_out = transformer(motion_with_time)

print(f"Transformer 输出: shape={trans_out.shape}")
frame_diffs = []
for t_idx in range(1, T_future):
    diff = (trans_out[t_idx] - trans_out[t_idx-1]).abs().mean().item()
    frame_diffs.append(diff)
print(f"  帧间差异: mean={sum(frame_diffs)/len(frame_diffs):.6f}")

diff_first_last = (trans_out[0] - trans_out[-1]).abs().mean().item()
print(f"  第一帧 vs 最后帧: {diff_first_last:.6f}")

# ============================================================
print("\n" + "=" * 70)
print("检查 4: 通过简单 MLP 输出")
print("=" * 70)

output_mlp = nn.Sequential(
    nn.Linear(latent_dim, 512),
    nn.GELU(),
    nn.Linear(512, 256),
    nn.GELU(),
    nn.Linear(256, K * D),
).to(device)

mlp_out = output_mlp(trans_out)  # [T, B, K*D]
mlp_out = mlp_out.reshape(T_future, 1, K, D).permute(1, 2, 3, 0)  # [B, K, D, T]

print(f"MLP 输出: shape={mlp_out.shape}")
output_disp = (mlp_out[:, :, :, 1:] - mlp_out[:, :, :, :-1]).abs().mean().item()
print(f"  帧间差异: {output_disp:.6f}")

# ============================================================
print("\n" + "=" * 70)
print("检查 5: 完整序列 [time + sign + past_ctx + x_t]")
print("=" * 70)

# 模拟完整输入
time_emb = torch.randn(1, 1, latent_dim).to(device)
sign_emb = torch.randn(1, 1, latent_dim).to(device)
past_ctx = torch.randn(1, 1, latent_dim).to(device)

# 拼接: [time(1), sign(1), past_ctx(1), x_t(20)] = 23 tokens
xseq = torch.cat([time_emb, sign_emb, past_ctx, motion_with_time], dim=0)
print(f"完整序列: shape={xseq.shape}")

# 通过 Transformer
trans_out_full = transformer(xseq)

# 取最后 T_future 帧
output_tokens = trans_out_full[-T_future:]  # [20, 1, D]

print(f"取最后 {T_future} 帧: shape={output_tokens.shape}")
frame_diffs = []
for t_idx in range(1, T_future):
    diff = (output_tokens[t_idx] - output_tokens[t_idx-1]).abs().mean().item()
    frame_diffs.append(diff)
print(f"  帧间差异: mean={sum(frame_diffs)/len(frame_diffs):.6f}")

# 通过 MLP
final_out = output_mlp(output_tokens)
final_out = final_out.reshape(T_future, 1, K, D).permute(1, 2, 3, 0)

final_disp = (final_out[:, :, :, 1:] - final_out[:, :, :, :-1]).abs().mean().item()
print(f"\n最终输出帧间差异: {final_disp:.6f}")

# ============================================================
print("\n" + "=" * 70)
print("检查 6: 训练后会发生什么？")
print("=" * 70)

# 创建一个最小模型
class MinimalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_proc = MotionProcess(K * D, latent_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=4,
            dim_feedforward=1024, dropout=0.0,
            activation="gelu", batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.output_mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, K * D),
        )
    
    def forward(self, x):
        # x: [B, K, D, T]
        B, _, _, T = x.shape
        
        # MotionProcess
        h = self.motion_proc(x)  # [T, B, latent_dim]
        
        # 加时间编码
        t = torch.linspace(0, 1, steps=T, device=x.device).view(T, 1, 1)
        t_latent = self.time_proj(t).expand(-1, B, -1)
        h = h + 0.1 * t_latent
        
        # Transformer
        h = self.transformer(h)  # [T, B, latent_dim]
        
        # Output MLP
        y = self.output_mlp(h)  # [T, B, K*D]
        y = y.reshape(T, B, K, D).permute(1, 2, 3, 0)  # [B, K, D, T]
        
        return y

model = MinimalModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# GT
gt = torch.zeros(1, K, D, T_future).to(device)
for t_idx in range(T_future):
    gt[:, :, 0, t_idx] = t_idx * 0.5

gt_disp = (gt[:, :, :, 1:] - gt[:, :, :, :-1]).abs().mean().item()
print(f"GT displacement: {gt_disp:.4f}")

print("\n训练最小模型...")
model.train()
for step in range(2001):
    optimizer.zero_grad()
    
    pred = model(gt)  # 直接用 GT 作为输入（Regression）
    
    loss_mse = F.mse_loss(pred, gt)
    pred_vel = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_vel = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    loss_vel = F.mse_loss(pred_vel, gt_vel)
    
    loss = loss_mse + loss_vel
    loss.backward()
    optimizer.step()
    
    if step % 400 == 0:
        pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
        ratio = pred_disp / gt_disp
        print(f"  Step {step}: loss={loss.item():.4f}, ratio={ratio:.4f}")

model.eval()
with torch.no_grad():
    pred = model(gt)
    pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
    ratio = pred_disp / gt_disp
    print(f"\n最终: ratio={ratio:.4f}")

if ratio > 0.5:
    print("✅ 最小模型成功！")
else:
    print("❌ 最小模型也失败，问题在 MotionProcess 或 Transformer")