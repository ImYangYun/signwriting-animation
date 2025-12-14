# -*- coding: utf-8 -*-
"""
逐步简化模型，定位问题在哪一层

极简 MLP 成功 (ratio=1.0)
Transformer 模型失败 (ratio=0.0)

逐步添加组件，找到导致失败的那一层
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("逐步简化定位问题")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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
print(f"GT displacement: {gt_disp:.4f}")

past = torch.zeros(1, K, D, T_past).to(device)
for t_idx in range(T_past):
    past[:, :, 0, t_idx] = (t_idx - T_past) * 0.5

def train_and_test(model, name, num_steps=2000):
    """训练并测试模型"""
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
print("模型 1: 极简 MLP (baseline)")
print("=" * 70)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K * D * T_past, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, K * D * T_future),
        )
    
    def forward(self, past):
        x = past.reshape(1, -1)
        y = self.net(x)
        return y.reshape(1, K, D, T_future)

model1 = SimpleMLP().to(device)
ratio1 = train_and_test(model1, "SimpleMLP")

# ============================================================
print("\n" + "=" * 70)
print("模型 2: MLP + 时间维度分开处理")
print("=" * 70)

class MLPWithTime(nn.Module):
    """每个时间步单独预测"""
    def __init__(self):
        super().__init__()
        # past -> hidden
        self.encoder = nn.Sequential(
            nn.Linear(K * D * T_past, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        # hidden -> 每帧输出
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, K * D),
        )
    
    def forward(self, past):
        x = past.reshape(1, -1)
        h = self.encoder(x)  # [1, latent_dim]
        
        outputs = []
        for t in range(T_future):
            # 每帧用相同的 hidden (应该输出常数!)
            out = self.decoder(h)  # [1, K*D]
            outputs.append(out)
        
        y = torch.stack(outputs, dim=-1)  # [1, K*D, T]
        return y.reshape(1, K, D, T_future)

model2 = MLPWithTime().to(device)
ratio2 = train_and_test(model2, "MLPWithTime")

# ============================================================
print("\n" + "=" * 70)
print("模型 3: MLP + 时间编码")
print("=" * 70)

class MLPWithTimeEncoding(nn.Module):
    """每个时间步有不同的时间编码"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(K * D * T_past, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        # 时间编码
        self.time_embed = nn.Embedding(T_future, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),  # hidden + time
            nn.ReLU(),
            nn.Linear(512, K * D),
        )
    
    def forward(self, past):
        x = past.reshape(1, -1)
        h = self.encoder(x)  # [1, latent_dim]
        
        outputs = []
        for t in range(T_future):
            t_emb = self.time_embed(torch.tensor([t], device=device))  # [1, latent_dim]
            h_t = torch.cat([h, t_emb], dim=-1)  # [1, latent_dim*2]
            out = self.decoder(h_t)
            outputs.append(out)
        
        y = torch.stack(outputs, dim=-1)
        return y.reshape(1, K, D, T_future)

model3 = MLPWithTimeEncoding().to(device)
ratio3 = train_and_test(model3, "MLPWithTimeEncoding")

# ============================================================
print("\n" + "=" * 70)
print("模型 4: 用 Transformer 但简化输出")
print("=" * 70)

class SimpleTransformer(nn.Module):
    """Transformer encoder + 简单线性输出"""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(K * D, latent_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=4,
            dim_feedforward=512, dropout=0.1,
            activation="gelu", batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 简单线性输出
        self.output_proj = nn.Linear(latent_dim, K * D)
    
    def forward(self, past):
        # past: [1, K, D, T_past] -> [T_past, 1, K*D]
        x = past.permute(3, 0, 1, 2).reshape(T_past, 1, K * D)
        
        # 编码
        h = self.input_proj(x)  # [T_past, 1, latent_dim]
        h = self.transformer(h)  # [T_past, 1, latent_dim]
        
        # 取最后 T_future 帧的特征 (用最后帧重复)
        h_last = h[-1:]  # [1, 1, latent_dim]
        h_future = h_last.expand(T_future, -1, -1)  # [T_future, 1, latent_dim]
        
        # 输出
        y = self.output_proj(h_future)  # [T_future, 1, K*D]
        y = y.reshape(T_future, 1, K, D).permute(1, 2, 3, 0)  # [1, K, D, T_future]
        
        return y

model4 = SimpleTransformer().to(device)
ratio4 = train_and_test(model4, "SimpleTransformer")

# ============================================================
print("\n" + "=" * 70)
print("模型 5: Transformer + 位置编码输出")
print("=" * 70)

class TransformerWithPosOutput(nn.Module):
    """Transformer + 输出时加位置编码"""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(K * D, latent_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=4,
            dim_feedforward=512, dropout=0.1,
            activation="gelu", batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 输出位置编码
        self.output_pos = nn.Embedding(T_future, latent_dim)
        self.output_proj = nn.Linear(latent_dim, K * D)
    
    def forward(self, past):
        x = past.permute(3, 0, 1, 2).reshape(T_past, 1, K * D)
        
        h = self.input_proj(x)
        h = self.transformer(h)
        h_last = h[-1]  # [1, latent_dim]
        
        # 为每个输出时间步加位置编码
        outputs = []
        for t in range(T_future):
            pos = self.output_pos(torch.tensor([t], device=device))  # [1, latent_dim]
            h_t = h_last + pos
            out = self.output_proj(h_t)  # [1, K*D]
            outputs.append(out)
        
        y = torch.stack(outputs, dim=0)  # [T_future, 1, K*D]
        y = y.reshape(T_future, 1, K, D).permute(1, 2, 3, 0)
        
        return y

model5 = TransformerWithPosOutput().to(device)
ratio5 = train_and_test(model5, "TransformerWithPosOutput")

# ============================================================
print("\n" + "=" * 70)
print("模型 6: 测试你的 OutputProcessMLP")
print("=" * 70)

# 复制你的 OutputProcessMLP
class ResidualBlock(nn.Module):
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
        return self.ln(x + residual * 0.5)

class OutputProcessMLP(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 1024
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(6)])
        self.out_proj = nn.Linear(hidden_dim, K * D)

    def forward(self, x):
        # x: [T, B, latent_dim]
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h)  # [T, B, K*D]

class TransformerWithOutputMLP(nn.Module):
    """Transformer + 你的 OutputProcessMLP"""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(K * D, latent_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=4,
            dim_feedforward=512, dropout=0.1,
            activation="gelu", batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.output_pos = nn.Embedding(T_future, latent_dim)
        self.output_mlp = OutputProcessMLP()
    
    def forward(self, past):
        x = past.permute(3, 0, 1, 2).reshape(T_past, 1, K * D)
        
        h = self.input_proj(x)
        h = self.transformer(h)
        h_last = h[-1]  # [1, latent_dim]
        
        # 构建输出序列
        h_future = []
        for t in range(T_future):
            pos = self.output_pos(torch.tensor([t], device=device))
            h_future.append(h_last + pos)
        
        h_future = torch.stack(h_future, dim=0)  # [T_future, 1, latent_dim]
        
        y = self.output_mlp(h_future)  # [T_future, 1, K*D]
        y = y.reshape(T_future, 1, K, D).permute(1, 2, 3, 0)
        
        return y

model6 = TransformerWithOutputMLP().to(device)
ratio6 = train_and_test(model6, "TransformerWithOutputMLP")

# ============================================================
print("\n" + "=" * 70)
print("模型 7: 直接测试 OutputProcessMLP (不用 Transformer)")
print("=" * 70)

class DirectOutputMLP(nn.Module):
    """直接用 OutputProcessMLP，不经过 Transformer"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(K * D * T_past, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.output_pos = nn.Embedding(T_future, latent_dim)
        self.output_mlp = OutputProcessMLP()
    
    def forward(self, past):
        x = past.reshape(1, -1)
        h = self.encoder(x)  # [1, latent_dim]
        
        h_future = []
        for t in range(T_future):
            pos = self.output_pos(torch.tensor([t], device=device))
            h_future.append(h + pos)
        
        h_future = torch.stack(h_future, dim=0)  # [T_future, 1, latent_dim]
        
        y = self.output_mlp(h_future)  # [T_future, 1, K*D]
        y = y.reshape(T_future, 1, K, D).permute(1, 2, 3, 0)
        
        return y

model7 = DirectOutputMLP().to(device)
ratio7 = train_and_test(model7, "DirectOutputMLP")

# ============================================================
print("\n" + "=" * 70)
print("📊 结果汇总")
print("=" * 70)

print(f"""
| 模型 | ratio | 结果 |
|------|-------|------|
| 1. SimpleMLP | {ratio1:.4f} | {'✅' if ratio1 > 0.5 else '❌'} |
| 2. MLPWithTime (无时间区分) | {ratio2:.4f} | {'✅' if ratio2 > 0.5 else '❌'} |
| 3. MLPWithTimeEncoding | {ratio3:.4f} | {'✅' if ratio3 > 0.5 else '❌'} |
| 4. SimpleTransformer | {ratio4:.4f} | {'✅' if ratio4 > 0.5 else '❌'} |
| 5. TransformerWithPosOutput | {ratio5:.4f} | {'✅' if ratio5 > 0.5 else '❌'} |
| 6. TransformerWithOutputMLP | {ratio6:.4f} | {'✅' if ratio6 > 0.5 else '❌'} |
| 7. DirectOutputMLP | {ratio7:.4f} | {'✅' if ratio7 > 0.5 else '❌'} |
""")

# 分析
if ratio7 < 0.5 and ratio3 > 0.5:
    print("⚠️ OutputProcessMLP 有问题！用简单 MLP 可以，但 OutputProcessMLP 不行")
elif ratio4 < 0.5 and ratio3 > 0.5:
    print("⚠️ Transformer 有问题！即使用简单输出也失败")
elif ratio6 < 0.5 and ratio5 > 0.5:
    print("⚠️ OutputProcessMLP 和 Transformer 结合有问题")