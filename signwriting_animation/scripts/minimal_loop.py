# -*- coding: utf-8 -*-
"""
诊断 OutputProcessMLP 的问题

发现：
- Transformer + 简单 Linear 输出 = ✅ 成功
- Transformer + OutputProcessMLP = ❌ 失败
- 直接用 OutputProcessMLP = ❌ 失败

问题在 OutputProcessMLP！
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("诊断 OutputProcessMLP")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K = 178
D = 3
T_future = 20
T_past = 40
latent_dim = 256

# 创建数据
gt = torch.zeros(1, K, D, T_future).to(device)
for t_idx in range(T_future):
    gt[:, :, 0, t_idx] = t_idx * 0.5

gt_disp = (gt[:, :, :, 1:] - gt[:, :, :, :-1]).abs().mean().item()

past = torch.zeros(1, K, D, T_past).to(device)
for t_idx in range(T_past):
    past[:, :, 0, t_idx] = (t_idx - T_past) * 0.5

def train_and_test(model, name, num_steps=2000):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for step in range(num_steps):
        optimizer.zero_grad()
        pred = model(past)
        loss_mse = F.mse_loss(pred, gt)
        pred_vel = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        gt_vel = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        loss_vel = F.mse_loss(pred_vel, gt_vel)
        loss = loss_mse + loss_vel
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
            ratio = pred_disp / gt_disp
            print(f"  Step {step}: loss={loss.item():.4f}, ratio={ratio:.4f}")
    
    model.eval()
    with torch.no_grad():
        pred = model(past)
        pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
        ratio = pred_disp / gt_disp
        print(f"  {name} 最终: ratio={ratio:.4f}")
        return ratio

# ============================================================
print("\n" + "=" * 70)
print("你的 OutputProcessMLP (有问题的版本)")
print("=" * 70)

class ResidualBlockOriginal(nn.Module):
    """你的原版"""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return self.ln(x + residual * 0.5)  # ⚠️ 问题可能在这里！

class OutputProcessMLPOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 1024
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlockOriginal(hidden_dim) for _ in range(6)])
        self.out_proj = nn.Linear(hidden_dim, K * D)

    def forward(self, x):
        # x: [T, B, latent_dim]
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h)

class TestModelOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(K * D * T_past, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.output_pos = nn.Embedding(T_future, latent_dim)
        self.output_mlp = OutputProcessMLPOriginal()
    
    def forward(self, past):
        x = past.reshape(1, -1)
        h = self.encoder(x)
        h_future = []
        for t in range(T_future):
            pos = self.output_pos(torch.tensor([t], device=device))
            h_future.append(h + pos)
        h_future = torch.stack(h_future, dim=0)
        y = self.output_mlp(h_future)
        y = y.reshape(T_future, 1, K, D).permute(1, 2, 3, 0)
        return y

print("\n测试原版 OutputProcessMLP...")
model_orig = TestModelOriginal().to(device)
ratio_orig = train_and_test(model_orig, "Original")

# ============================================================
print("\n" + "=" * 70)
print("修复 1: 移除 LayerNorm")
print("=" * 70)

class ResidualBlockNoLN(nn.Module):
    """移除 LayerNorm"""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x + residual  # 不用 LayerNorm

class OutputProcessMLPNoLN(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 1024
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlockNoLN(hidden_dim) for _ in range(6)])
        self.out_proj = nn.Linear(hidden_dim, K * D)

    def forward(self, x):
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h)

class TestModelNoLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(K * D * T_past, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.output_pos = nn.Embedding(T_future, latent_dim)
        self.output_mlp = OutputProcessMLPNoLN()
    
    def forward(self, past):
        x = past.reshape(1, -1)
        h = self.encoder(x)
        h_future = []
        for t in range(T_future):
            pos = self.output_pos(torch.tensor([t], device=device))
            h_future.append(h + pos)
        h_future = torch.stack(h_future, dim=0)
        y = self.output_mlp(h_future)
        y = y.reshape(T_future, 1, K, D).permute(1, 2, 3, 0)
        return y

print("\n测试移除 LayerNorm...")
model_no_ln = TestModelNoLN().to(device)
ratio_no_ln = train_and_test(model_no_ln, "NoLayerNorm")

# ============================================================
print("\n" + "=" * 70)
print("修复 2: 减少层数 (6 -> 2)")
print("=" * 70)

class OutputProcessMLPShallow(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 1024
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlockOriginal(hidden_dim) for _ in range(2)])  # 只用 2 层
        self.out_proj = nn.Linear(hidden_dim, K * D)

    def forward(self, x):
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h)

class TestModelShallow(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(K * D * T_past, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.output_pos = nn.Embedding(T_future, latent_dim)
        self.output_mlp = OutputProcessMLPShallow()
    
    def forward(self, past):
        x = past.reshape(1, -1)
        h = self.encoder(x)
        h_future = []
        for t in range(T_future):
            pos = self.output_pos(torch.tensor([t], device=device))
            h_future.append(h + pos)
        h_future = torch.stack(h_future, dim=0)
        y = self.output_mlp(h_future)
        y = y.reshape(T_future, 1, K, D).permute(1, 2, 3, 0)
        return y

print("\n测试减少层数...")
model_shallow = TestModelShallow().to(device)
ratio_shallow = train_and_test(model_shallow, "ShallowMLP")

# ============================================================
print("\n" + "=" * 70)
print("修复 3: 用简单 MLP (不用 ResidualBlock)")
print("=" * 70)

class OutputProcessMLPSimple(nn.Module):
    """用简单 MLP 替代 ResidualBlock"""
    def __init__(self):
        super().__init__()
        hidden_dim = 512
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K * D),
        )

    def forward(self, x):
        # x: [T, B, latent_dim]
        return self.net(x)

class TestModelSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(K * D * T_past, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.output_pos = nn.Embedding(T_future, latent_dim)
        self.output_mlp = OutputProcessMLPSimple()
    
    def forward(self, past):
        x = past.reshape(1, -1)
        h = self.encoder(x)
        h_future = []
        for t in range(T_future):
            pos = self.output_pos(torch.tensor([t], device=device))
            h_future.append(h + pos)
        h_future = torch.stack(h_future, dim=0)
        y = self.output_mlp(h_future)
        y = y.reshape(T_future, 1, K, D).permute(1, 2, 3, 0)
        return y

print("\n测试简单 MLP...")
model_simple = TestModelSimple().to(device)
ratio_simple = train_and_test(model_simple, "SimpleMLP")

# ============================================================
print("\n" + "=" * 70)
print("修复 4: 参考师姐的 OutputProcessMLP")
print("=" * 70)

class OutputProcessMLPSister(nn.Module):
    """参考师姐的实现: 3层 MLP"""
    def __init__(self):
        super().__init__()
        hidden_dim = 512
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, K * D),
        )

    def forward(self, x):
        return self.net(x)

class TestModelSister(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(K * D * T_past, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.output_pos = nn.Embedding(T_future, latent_dim)
        self.output_mlp = OutputProcessMLPSister()
    
    def forward(self, past):
        x = past.reshape(1, -1)
        h = self.encoder(x)
        h_future = []
        for t in range(T_future):
            pos = self.output_pos(torch.tensor([t], device=device))
            h_future.append(h + pos)
        h_future = torch.stack(h_future, dim=0)
        y = self.output_mlp(h_future)
        y = y.reshape(T_future, 1, K, D).permute(1, 2, 3, 0)
        return y

print("\n测试师姐风格 MLP...")
model_sister = TestModelSister().to(device)
ratio_sister = train_and_test(model_sister, "SisterMLP")

# ============================================================
print("\n" + "=" * 70)
print("📊 结果汇总")
print("=" * 70)

print(f"""
| 模型 | ratio | 结果 |
|------|-------|------|
| Original (6层 ResidualBlock + LN) | {ratio_orig:.4f} | {'✅' if ratio_orig > 0.5 else '❌'} |
| NoLayerNorm (移除 LN) | {ratio_no_ln:.4f} | {'✅' if ratio_no_ln > 0.5 else '❌'} |
| ShallowMLP (2层) | {ratio_shallow:.4f} | {'✅' if ratio_shallow > 0.5 else '❌'} |
| SimpleMLP (无 Residual) | {ratio_simple:.4f} | {'✅' if ratio_simple > 0.5 else '❌'} |
| SisterMLP (师姐风格) | {ratio_sister:.4f} | {'✅' if ratio_sister > 0.5 else '❌'} |
""")

print("""
修复建议:
1. 如果 NoLayerNorm 成功 → 问题在 LayerNorm
2. 如果 ShallowMLP 成功 → 问题在层数太多
3. 如果 SimpleMLP 成功 → 问题在 ResidualBlock 结构
4. 如果 SisterMLP 成功 → 直接换成师姐的实现
""")