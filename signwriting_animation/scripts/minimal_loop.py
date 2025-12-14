# -*- coding: utf-8 -*-
"""
模型完整诊断 - 逐个部分测试

合并了所有测试，一次运行全部检查
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("=" * 70)
print("模型完整诊断")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 导入组件
from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess, seq_encoder_factory

# 基本参数
B = 2
K = 178
D = 3
T_future = 20
T_past = 40
latent_dim = 256

results = {}  # 存储测试结果

# ============================================================
print("\n" + "=" * 70)
print("Test 1: MotionProcess")
print("=" * 70)

motion_process = MotionProcess(K * D, latent_dim).to(device)

x1 = torch.randn(B, K, D, T_future).to(device) * 0.01
x2 = torch.randn(B, K, D, T_future).to(device) * 100

out1 = motion_process(x1)
out2 = motion_process(x2)

diff = (out1 - out2).abs().mean().item()
print(f"输入 x1 范围: [{x1.min():.4f}, {x1.max():.4f}]")
print(f"输入 x2 范围: [{x2.min():.4f}, {x2.max():.4f}]")
print(f"输出差异: {diff:.4f}")

results['MotionProcess'] = '✓' if diff > 0.1 else '⚠️'
print(f"结果: {results['MotionProcess']} {'对输入敏感' if diff > 0.1 else '对输入不敏感!'}")

# ============================================================
print("\n" + "=" * 70)
print("Test 2: TimestepEmbedder")
print("=" * 70)

pos_enc = PositionalEncoding(latent_dim, dropout=0.1)
timestep_emb = TimestepEmbedder(latent_dim, pos_enc).to(device)

t_0 = torch.tensor([0]).to(device)
t_7 = torch.tensor([7]).to(device)

emb_0 = timestep_emb(t_0)
emb_7 = timestep_emb(t_7)

diff = (emb_0 - emb_7).abs().mean().item()
print(f"t=0 emb: mean={emb_0.mean():.4f}, std={emb_0.std():.4f}")
print(f"t=7 emb: mean={emb_7.mean():.4f}, std={emb_7.std():.4f}")
print(f"差异: {diff:.4f}")

results['TimestepEmbedder'] = '✓' if diff > 0.1 else '⚠️'
print(f"结果: {results['TimestepEmbedder']} {'对 t 敏感' if diff > 0.1 else '对 t 不敏感!'}")

# ============================================================
print("\n" + "=" * 70)
print("Test 3: ContextEncoder (MeanPool)")
print("=" * 70)

class ContextEncoder(nn.Module):
    def __init__(self, input_feats, latent_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.pose_encoder = nn.Linear(input_feats, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads,
            dim_feedforward=latent_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        if x.dim() == 4:
            B, T, J, C = x.shape
            x = x.reshape(B, T, J * C)
        x_emb = self.pose_encoder(x)
        x_enc = self.encoder(x_emb)
        return x_enc.mean(dim=1).unsqueeze(0)

context_enc = ContextEncoder(K * D, latent_dim).to(device)

past1 = torch.randn(B, T_past, K, D).to(device)
past2 = torch.randn(B, T_past, K, D).to(device) * 10

ctx1 = context_enc(past1)
ctx2 = context_enc(past2)

diff = (ctx1 - ctx2).abs().mean().item()
print(f"输出 shape: {ctx1.shape}")
print(f"不同输入差异: {diff:.4f}")

results['ContextEncoder'] = '✓' if diff > 0.1 else '⚠️'
print(f"结果: {results['ContextEncoder']} {'对输入敏感' if diff > 0.1 else '对输入不敏感!'}")

# ============================================================
print("\n" + "=" * 70)
print("Test 4: OutputProcessMLP")
print("=" * 70)

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
    def __init__(self, num_latent_dims, num_keypoints, num_dims_per_keypoint, hidden_dim=1024, num_layers=6):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
        self.in_proj = nn.Linear(num_latent_dims, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hidden_dim, num_keypoints * num_dims_per_keypoint)

    def forward(self, x):
        T, B, D = x.shape
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        y = self.out_proj(h)
        return y.reshape(T, B, self.num_keypoints, self.num_dims_per_keypoint)

output_mlp = OutputProcessMLP(latent_dim, K, D).to(device)

# 有时间变化的输入
input_motion = torch.randn(T_future, B, latent_dim).to(device)
for t in range(T_future):
    input_motion[t] += t * 0.5

out_motion = output_mlp(input_motion)
motion_disp = (out_motion[1:] - out_motion[:-1]).abs().mean().item()

# 常数输入
input_const = torch.randn(1, B, latent_dim).to(device).expand(T_future, -1, -1).clone()
out_const = output_mlp(input_const)
const_disp = (out_const[1:] - out_const[:-1]).abs().mean().item()

print(f"有时间变化输入 -> 输出帧间差异: {motion_disp:.6f}")
print(f"常数输入 -> 输出帧间差异: {const_disp:.6f}")

results['OutputProcessMLP'] = '✓' if motion_disp > const_disp * 1.5 else '⚠️'
print(f"结果: {results['OutputProcessMLP']} {'能传递时间变化' if motion_disp > const_disp * 1.5 else '不能传递时间变化!'}")

# ============================================================
print("\n" + "=" * 70)
print("Test 5: Transformer 对 future 部分敏感性")
print("=" * 70)

seq_encoder = seq_encoder_factory(
    arch="trans_enc", latent_dim=latent_dim, ff_size=1024,
    num_layers=8, num_heads=4, dropout=0.2, activation="gelu"
).to(device)

seq_len = 3 + T_future  # time + sign + past_ctx + future

xseq_base = torch.randn(seq_len, B, latent_dim).to(device)
xseq_modified = xseq_base.clone()
xseq_modified[3:] = torch.randn(T_future, B, latent_dim).to(device) * 10

enc_base = seq_encoder(xseq_base)
enc_modified = seq_encoder(xseq_modified)

diff_future = (enc_base[-T_future:] - enc_modified[-T_future:]).abs().mean().item()
print(f"只改变 future 部分，输出差异: {diff_future:.4f}")

results['Transformer_future'] = '✓' if diff_future > 0.1 else '⚠️'
print(f"结果: {results['Transformer_future']} {'对 future 敏感' if diff_future > 0.1 else '对 future 不敏感!'}")

# ============================================================
print("\n" + "=" * 70)
print("Test 6: 梯度分析 - 各部分对输出的贡献")
print("=" * 70)

seq_encoder_grad = seq_encoder_factory(
    arch="trans_enc", latent_dim=latent_dim, ff_size=1024,
    num_layers=8, num_heads=4, dropout=0.0, activation="gelu"
).to(device)

xseq_grad = torch.randn(seq_len, B, latent_dim, device=device, requires_grad=True)
output = seq_encoder_grad(xseq_grad)
loss = output[-T_future:].sum()
loss.backward()

grad = xseq_grad.grad
grad_time = grad[0].abs().mean().item()
grad_sign = grad[1].abs().mean().item()
grad_past = grad[2].abs().mean().item()
grad_future = grad[3:].abs().mean().item()

total_grad = grad_time + grad_sign + grad_past + grad_future

print(f"各部分梯度占比:")
print(f"  time: {grad_time/total_grad:.1%}")
print(f"  sign: {grad_sign/total_grad:.1%}")
print(f"  past: {grad_past/total_grad:.1%}")
print(f"  future/x_t: {grad_future/total_grad:.1%}")

results['Gradient_xt'] = '✓' if grad_future/total_grad > 0.2 else '⚠️'
print(f"结果: {results['Gradient_xt']} {'x_t 梯度足够' if grad_future/total_grad > 0.2 else 'x_t 梯度太小!'}")

# ============================================================
print("\n" + "=" * 70)
print("Test 7: 🔥 完整模型 - 是否使用 x_t")
print("=" * 70)

try:
    from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusionV2
    
    model = SignWritingToPoseDiffusionV2(
        num_keypoints=K, num_dims_per_keypoint=D,
        residual_scale=0.1, use_mean_pool=True,
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        past_fixed = torch.randn(B, K, D, T_past).to(device)
        sign_fixed = torch.randn(B, 3, 224, 224).to(device)
        t_fixed = torch.tensor([4, 4]).to(device)
        
        x_t_1 = torch.randn(B, K, D, T_future).to(device) * 0.1
        x_t_2 = torch.randn(B, K, D, T_future).to(device) * 10
        
        out_1 = model(x_t_1, t_fixed, past_fixed, sign_fixed)
        out_2 = model(x_t_2, t_fixed, past_fixed, sign_fixed)
        
        diff = (out_1 - out_2).abs().mean().item()
        
        print(f"x_t_1 范围: [{x_t_1.min():.2f}, {x_t_1.max():.2f}]")
        print(f"x_t_2 范围: [{x_t_2.min():.2f}, {x_t_2.max():.2f}]")
        print(f"输出差异: {diff:.6f}")
        
        results['Model_uses_xt'] = '✓' if diff > 0.01 else '⚠️⚠️⚠️'
        print(f"结果: {results['Model_uses_xt']} {'模型使用 x_t' if diff > 0.01 else '模型忽略 x_t!'}")

except Exception as e:
    print(f"测试失败: {e}")
    results['Model_uses_xt'] = '❌'

# ============================================================
print("\n" + "=" * 70)
print("Test 8: 🔥 完整模型 - 是否使用 timestep t")
print("=" * 70)

try:
    model.eval()
    with torch.no_grad():
        x_t_fixed = torch.randn(B, K, D, T_future).to(device)
        past_fixed = torch.randn(B, K, D, T_past).to(device)
        sign_fixed = torch.randn(B, 3, 224, 224).to(device)
        
        t_0 = torch.tensor([0, 0]).to(device)
        t_7 = torch.tensor([7, 7]).to(device)
        
        out_t0 = model(x_t_fixed, t_0, past_fixed, sign_fixed)
        out_t7 = model(x_t_fixed, t_7, past_fixed, sign_fixed)
        
        diff = (out_t0 - out_t7).abs().mean().item()
        
        print(f"t=0 vs t=7 输出差异: {diff:.6f}")
        
        results['Model_uses_t'] = '✓' if diff > 0.01 else '⚠️'
        print(f"结果: {results['Model_uses_t']} {'模型使用 t' if diff > 0.01 else '模型忽略 t!'}")

except Exception as e:
    print(f"测试失败: {e}")
    results['Model_uses_t'] = '❌'

# ============================================================
print("\n" + "=" * 70)
print("Test 9: 🔥 单样本 Overfit (Diffusion)")
print("=" * 70)

try:
    from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
    
    def cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    DIFFUSION_STEPS = 8
    betas = cosine_beta_schedule(DIFFUSION_STEPS).numpy()
    
    diffusion = GaussianDiffusion(
        betas=betas, model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL, loss_type=LossType.MSE,
    )
    
    model = SignWritingToPoseDiffusionV2(
        num_keypoints=K, num_dims_per_keypoint=D,
        residual_scale=0.1, use_mean_pool=True,
    ).to(device)
    
    # GT 有明显运动
    gt = torch.zeros(1, K, D, T_future).to(device)
    for t_idx in range(T_future):
        gt[:, :, 0, t_idx] = t_idx * 0.5
    
    gt_disp = (gt[:, :, :, 1:] - gt[:, :, :, :-1]).abs().mean().item()
    print(f"GT displacement: {gt_disp:.4f}")
    
    past = torch.zeros(1, K, D, T_past).to(device)
    sign = torch.randn(1, 3, 224, 224).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("\nDiffusion 训练 1500 步...")
    model.train()
    for step in range(1500):
        optimizer.zero_grad()
        
        t = torch.randint(0, DIFFUSION_STEPS, (1,), device=device)
        noise = torch.randn_like(gt)
        x_t = diffusion.q_sample(gt, t, noise=noise)
        
        pred = model(x_t, t, past, sign)
        loss = F.mse_loss(pred, gt)
        loss.backward()
        optimizer.step()
        
        if step % 300 == 0:
            pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
            ratio = pred_disp / gt_disp
            print(f"  Step {step}: loss={loss.item():.6f}, pred_disp={pred_disp:.4f}, ratio={ratio:.4f}")
    
    # 测试
    model.eval()
    with torch.no_grad():
        t = torch.tensor([0]).to(device)
        x_t = diffusion.q_sample(gt, t, noise=torch.randn_like(gt))
        pred = model(x_t, t, past, sign)
        
        pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
        diffusion_ratio = pred_disp / gt_disp
        
        print(f"\nDiffusion 最终: pred_disp={pred_disp:.4f}, ratio={diffusion_ratio:.4f}")
        
        results['Diffusion_overfit'] = '✓' if diffusion_ratio > 0.5 else '⚠️'

except Exception as e:
    print(f"测试失败: {e}")
    diffusion_ratio = 0
    results['Diffusion_overfit'] = '❌'

# ============================================================
print("\n" + "=" * 70)
print("Test 10: 🔥 单样本 Overfit (Regression 对比)")
print("=" * 70)

try:
    model_reg = SignWritingToPoseDiffusionV2(
        num_keypoints=K, num_dims_per_keypoint=D,
        residual_scale=0.1, use_mean_pool=True,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model_reg.parameters(), lr=1e-3)
    
    print("Regression 训练 1500 步 (不用 Diffusion)...")
    model_reg.train()
    
    for step in range(1500):
        optimizer.zero_grad()
        
        t = torch.tensor([0]).to(device)  # 固定 t=0
        x_t = torch.zeros(1, K, D, T_future).to(device)  # 零输入
        
        pred = model_reg(x_t, t, past, sign)
        
        loss_mse = F.mse_loss(pred, gt)
        pred_vel = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        gt_vel = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        loss_vel = F.mse_loss(pred_vel, gt_vel)
        
        loss = loss_mse + loss_vel
        loss.backward()
        optimizer.step()
        
        if step % 300 == 0:
            pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
            ratio = pred_disp / gt_disp
            print(f"  Step {step}: loss={loss.item():.6f}, pred_disp={pred_disp:.4f}, ratio={ratio:.4f}")
    
    # 测试
    model_reg.eval()
    with torch.no_grad():
        pred = model_reg(x_t, t, past, sign)
        pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
        regression_ratio = pred_disp / gt_disp
        
        print(f"\nRegression 最终: pred_disp={pred_disp:.4f}, ratio={regression_ratio:.4f}")
        
        results['Regression_overfit'] = '✓' if regression_ratio > 0.5 else '⚠️'

except Exception as e:
    print(f"测试失败: {e}")
    regression_ratio = 0
    results['Regression_overfit'] = '❌'

# ============================================================
print("\n" + "=" * 70)
print("📊 测试结果汇总")
print("=" * 70)

print("\n组件测试:")
for key in ['MotionProcess', 'TimestepEmbedder', 'ContextEncoder', 'OutputProcessMLP', 'Transformer_future', 'Gradient_xt']:
    if key in results:
        print(f"  {key}: {results[key]}")

print("\n完整模型测试:")
for key in ['Model_uses_xt', 'Model_uses_t']:
    if key in results:
        print(f"  {key}: {results[key]}")

print("\nOverfit 测试:")
print(f"  Diffusion: {results.get('Diffusion_overfit', '?')} (ratio={diffusion_ratio:.4f})")
print(f"  Regression: {results.get('Regression_overfit', '?')} (ratio={regression_ratio:.4f})")

# ============================================================
print("\n" + "=" * 70)
print("🔍 诊断结论")
print("=" * 70)

if results.get('Model_uses_xt') == '⚠️⚠️⚠️':
    print("""
⚠️ 模型忽略了 x_t 输入!

可能原因:
1. Transformer 学会了主要依赖条件 tokens (time, sign, past)
2. x_t 通过 MotionProcess 后信息被压缩
3. 模型发现忽略 x_t 也能最小化 MSE loss
""")

if results.get('Diffusion_overfit') == '⚠️' and results.get('Regression_overfit') == '✓':
    print("""
⚠️ Regression 成功但 Diffusion 失败!

说明模型架构本身没问题，问题在 Diffusion 的使用方式:
1. MSE loss 让模型找到了 "输出均值" 的捷径
2. 需要额外的 loss (velocity, displacement) 强制学习运动
3. 或者换用 EPSILON mode
""")

if results.get('Regression_overfit') == '⚠️':
    print("""
⚠️ Regression 也失败了!

说明模型架构本身可能有问题:
1. 检查 OutputProcessMLP 是否正确传递时间信息
2. 检查各组件的连接方式
3. 检查梯度流是否通畅
""")

print("\n" + "=" * 70)