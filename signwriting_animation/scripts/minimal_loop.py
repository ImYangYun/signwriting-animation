# -*- coding: utf-8 -*-
"""
修复版模型：Transformer 只处理 past，future 用解码方式生成

问题：当 future tokens 一起通过 Transformer 时，self-attention 会"平均化"它们

解决：
1. Transformer 只编码 [time, sign, past]
2. future 的每帧用 cross-attention 或简单解码生成
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("修复版模型测试")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from CAMDM.network.models import MotionProcess
from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from transformers import CLIPModel

K = 178
D = 3
T_past = 40
T_future = 20
latent_dim = 256

# ============================================================
# 修复版模型
# ============================================================

class EmbedSignWriting(nn.Module):
    def __init__(self, num_latent_dims):
        super().__init__()
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        if (num_embedding_dims := self.model.visual_projection.out_features) != num_latent_dims:
            self.proj = nn.Linear(num_embedding_dims, num_latent_dims)
        else:
            self.proj = None

    def forward(self, image_batch):
        emb = self.model.get_image_features(pixel_values=image_batch)
        if self.proj:
            emb = self.proj(emb)
        return emb[None, ...]


class FixedDiffusionModel(nn.Module):
    """
    修复版 Diffusion 模型
    
    关键改动：
    1. Transformer 只编码 context [time, sign, past_ctx]
    2. future 的每帧独立解码（加位置编码区分）
    3. 用 x_t 作为额外输入，通过 cross-attention 或 concat
    """
    def __init__(self, num_keypoints, num_dims_per_keypoint, num_latent_dims=256):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
        input_feats = num_keypoints * num_dims_per_keypoint
        
        # 编码器
        self.past_encoder = nn.Sequential(
            nn.Linear(input_feats * T_past, 512),
            nn.GELU(),
            nn.Linear(512, num_latent_dims),
        )
        
        self.xt_encoder = nn.Sequential(
            nn.Linear(input_feats, num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )
        
        self.sign_encoder = EmbedSignWriting(num_latent_dims)
        self.time_embed = nn.Embedding(1000, num_latent_dims)
        
        # 输出位置编码（区分不同时间步！）
        self.output_pos = nn.Embedding(T_future, num_latent_dims)
        
        # 解码器：为每个时间步独立解码
        # 输入: context + x_t[t] + pos[t]
        self.decoder = nn.Sequential(
            nn.Linear(num_latent_dims * 3, 512),  # context + x_t + pos
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, input_feats),
        )
    
    def forward(self, x_t, timesteps, past_motion, sign_img):
        """
        x_t: [B, K, D, T_future] - 带噪声的输入
        timesteps: [B] - diffusion timestep
        past_motion: [B, K, D, T_past] - 历史
        sign_img: [B, 3, H, W] - 条件图像
        """
        B = x_t.shape[0]
        device = x_t.device
        
        # 编码 context
        past_flat = past_motion.reshape(B, -1)  # [B, K*D*T_past]
        past_emb = self.past_encoder(past_flat)  # [B, latent_dim]
        
        sign_emb = self.sign_encoder(sign_img).squeeze(0)  # [B, latent_dim]
        time_emb = self.time_embed(timesteps.clamp(0, 999))  # [B, latent_dim]
        
        # 融合 context
        context = past_emb + sign_emb + time_emb  # [B, latent_dim]
        
        # 对每个时间步独立解码
        outputs = []
        for t in range(T_future):
            # x_t 的第 t 帧
            xt_frame = x_t[:, :, :, t].reshape(B, -1)  # [B, K*D]
            xt_emb = self.xt_encoder(xt_frame)  # [B, latent_dim]
            
            # 位置编码
            pos_emb = self.output_pos(torch.tensor([t], device=device)).expand(B, -1)  # [B, latent_dim]
            
            # 拼接并解码
            dec_input = torch.cat([context, xt_emb, pos_emb], dim=-1)  # [B, latent_dim*3]
            out = self.decoder(dec_input)  # [B, K*D]
            outputs.append(out)
        
        # Stack: [T, B, K*D] -> [B, K, D, T]
        result = torch.stack(outputs, dim=0)  # [T, B, K*D]
        result = result.permute(1, 0, 2)  # [B, T, K*D]
        result = result.reshape(B, T_future, K, D)  # [B, T, K, D]
        result = result.permute(0, 2, 3, 1)  # [B, K, D, T]
        
        return result


# ============================================================
# 测试
# ============================================================

gt_bjct = torch.zeros(1, K, D, T_future).to(device)
for t_idx in range(T_future):
    gt_bjct[:, :, 0, t_idx] = t_idx * 0.5

gt_disp = (gt_bjct[:, :, :, 1:] - gt_bjct[:, :, :, :-1]).abs().mean().item()
print(f"\nGT displacement: {gt_disp:.4f}")

past_bjct = torch.zeros(1, K, D, T_past).to(device)
for t_idx in range(T_past):
    past_bjct[:, :, 0, t_idx] = (t_idx - T_past) * 0.5

sign_img = torch.randn(1, 3, 224, 224).to(device)

# ============================================================
print("\n" + "=" * 70)
print("测试 1: Regression (修复版模型)")
print("=" * 70)

model = FixedDiffusionModel(K, D).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("\n训练 Regression...")
model.train()
for step in range(2001):
    optimizer.zero_grad()
    
    t = torch.tensor([0]).to(device)
    pred = model(gt_bjct, t, past_bjct, sign_img)
    
    loss_mse = F.mse_loss(pred, gt_bjct)
    pred_vel = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_vel = gt_bjct[:, :, :, 1:] - gt_bjct[:, :, :, :-1]
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
    t = torch.tensor([0]).to(device)
    pred = model(gt_bjct, t, past_bjct, sign_img)
    pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
    ratio_reg = pred_disp / gt_disp
    print(f"\n✓ Regression 最终: ratio={ratio_reg:.4f}")

# ============================================================
print("\n" + "=" * 70)
print("测试 2: Diffusion (修复版模型)")
print("=" * 70)

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
    betas=betas,
    model_mean_type=ModelMeanType.START_X,
    model_var_type=ModelVarType.FIXED_SMALL,
    loss_type=LossType.MSE,
    rescale_timesteps=False,
)

model2 = FixedDiffusionModel(K, D).to(device)
optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

print("\n训练 Diffusion...")
model2.train()
for step in range(2001):
    optimizer2.zero_grad()
    
    t = torch.randint(0, DIFFUSION_STEPS, (1,), device=device)
    noise = torch.randn_like(gt_bjct)
    x_t = diffusion.q_sample(gt_bjct, t, noise=noise)
    
    pred = model2(x_t, t, past_bjct, sign_img)
    
    loss_mse = F.mse_loss(pred, gt_bjct)
    pred_vel = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_vel = gt_bjct[:, :, :, 1:] - gt_bjct[:, :, :, :-1]
    loss_vel = F.mse_loss(pred_vel, gt_vel)
    
    loss = loss_mse + loss_vel
    loss.backward()
    optimizer2.step()
    
    if step % 400 == 0:
        with torch.no_grad():
            t_test = torch.tensor([0]).to(device)
            x_test = diffusion.q_sample(gt_bjct, t_test, noise=torch.randn_like(gt_bjct))
            pred_test = model2(x_test, t_test, past_bjct, sign_img)
            pred_disp = (pred_test[:, :, :, 1:] - pred_test[:, :, :, :-1]).abs().mean().item()
            ratio = pred_disp / gt_disp
        print(f"  Step {step}: loss={loss.item():.4f}, ratio={ratio:.4f}")

model2.eval()
with torch.no_grad():
    t = torch.tensor([0]).to(device)
    x_t = diffusion.q_sample(gt_bjct, t, noise=torch.randn_like(gt_bjct))
    pred = model2(x_t, t, past_bjct, sign_img)
    pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
    ratio_diff = pred_disp / gt_disp
    print(f"\n✓ Diffusion 最终: ratio={ratio_diff:.4f}")

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

wrapped = ConditionalWrapper(model2, past_bjct, sign_img)

print("\n采样...")
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
    print(f"✓ 采样结果: ratio={ratio_sample:.4f}")

# ============================================================
print("\n" + "=" * 70)
print("📊 结果汇总")
print("=" * 70)

print(f"""
| 测试 | ratio | 结果 |
|------|-------|------|
| Regression | {ratio_reg:.4f} | {'✅' if ratio_reg > 0.5 else '❌'} |
| Diffusion | {ratio_diff:.4f} | {'✅' if ratio_diff > 0.5 else '❌'} |
| p_sample_loop | {ratio_sample:.4f} | {'✅' if ratio_sample > 0.5 else '❌'} |
""")

if ratio_reg > 0.5 and ratio_diff > 0.5:
    print("🎉 修复成功！")
    print("\n关键改动：")
    print("  1. 不让 future tokens 通过 Transformer self-attention")
    print("  2. 每帧独立解码，用位置编码区分")
    print("  3. x_t 的每帧单独编码并参与解码")