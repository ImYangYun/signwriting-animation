# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("自包含测试 - 内置修复后的模型")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess, seq_encoder_factory
from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from transformers import CLIPModel

# ============================================================
# 修复后的模型定义（直接写在这里，不依赖外部文件）
# ============================================================

class ContextEncoder(nn.Module):
    """MeanPool 上下文编码器"""
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
        if x.dim() == 3:
            x = x.permute(1, 0, 2)
        elif x.dim() == 4:
            B, T, J, C = x.shape
            x = x.reshape(B, T, J * C)
        x_emb = self.pose_encoder(x)
        x_enc = self.encoder(x_emb)
        context = x_enc.mean(dim=1)
        return context.unsqueeze(0)


class OutputProcessMLP(nn.Module):
    """
    ⚠️ 修复版：简单 MLP，无 LayerNorm！
    """
    def __init__(self, num_latent_dims, num_keypoints, num_dims_per_keypoint, hidden_dim=512):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
        
        # 简单 3 层 MLP（师姐风格，无 LayerNorm！）
        self.net = nn.Sequential(
            nn.Linear(num_latent_dims, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_keypoints * num_dims_per_keypoint),
        )

    def forward(self, x):
        T, B, D = x.shape
        y = self.net(x)
        return y.reshape(T, B, self.num_keypoints, self.num_dims_per_keypoint)


class EmbedSignWriting(nn.Module):
    def __init__(self, num_latent_dims, embedding_arch='openai/clip-vit-base-patch32'):
        super().__init__()
        self.model = CLIPModel.from_pretrained(embedding_arch)
        self.proj = None
        if (num_embedding_dims := self.model.visual_projection.out_features) != num_latent_dims:
            self.proj = nn.Linear(num_embedding_dims, num_latent_dims)

    def forward(self, image_batch):
        embeddings_batch = self.model.get_image_features(pixel_values=image_batch)
        if self.proj is not None:
            embeddings_batch = self.proj(embeddings_batch)
        return embeddings_batch[None, ...]


class FixedModel(nn.Module):
    """
    修复后的 Diffusion 模型
    """
    def __init__(self, num_keypoints, num_dims_per_keypoint, num_latent_dims=256,
                 ff_size=1024, num_layers=8, num_heads=4, dropout=0.2,
                 use_mean_pool=True):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
        self.use_mean_pool = use_mean_pool

        input_feats = num_keypoints * num_dims_per_keypoint
        
        self.future_motion_process = MotionProcess(input_feats, num_latent_dims)
        self.past_motion_process = MotionProcess(input_feats, num_latent_dims)
        self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)

        self.embed_signwriting = EmbedSignWriting(num_latent_dims)
        self.embed_timestep = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)

        if use_mean_pool:
            self.past_context_encoder = ContextEncoder(input_feats, num_latent_dims)
            print(f"✓ 使用 MeanPool 模式")
        else:
            self.past_context_encoder = None
            print(f"✓ 使用 Concat 模式")

        self.seqEncoder = seq_encoder_factory(
            arch="trans_enc", latent_dim=num_latent_dims,
            ff_size=ff_size, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout, activation="gelu"
        )

        # ⚠️ 修复版 OutputProcessMLP（无 LayerNorm）
        self.pose_projection = OutputProcessMLP(
            num_latent_dims, num_keypoints, num_dims_per_keypoint
        )

        self.future_time_proj = nn.Sequential(
            nn.Linear(1, num_latent_dims),
            nn.SiLU(),
            nn.Linear(num_latent_dims, num_latent_dims)
        )

    def forward(self, x, timesteps, past_motion, signwriting_im_batch):
        batch_size, num_keypoints, num_dims_per_keypoint, num_frames = x.shape

        if past_motion.dim() == 4:
            if past_motion.shape[1] == num_keypoints and past_motion.shape[2] == num_dims_per_keypoint:
                pass
            elif past_motion.shape[2] == num_keypoints and past_motion.shape[3] == num_dims_per_keypoint:
                past_motion = past_motion.permute(0, 2, 3, 1).contiguous()

        T_past = past_motion.shape[-1]
        T_future = num_frames
        B = batch_size

        time_emb = self.embed_timestep(timesteps)
        signwriting_emb = self.embed_signwriting(signwriting_im_batch)
        future_motion_emb = self.future_motion_process(x)

        t = torch.linspace(0, 1, steps=T_future, device=x.device).view(T_future, 1, 1)
        t_latent = self.future_time_proj(t).expand(-1, B, -1)
        future_motion_emb = future_motion_emb + 0.1 * t_latent

        if self.use_mean_pool:
            past_btjc = past_motion.permute(0, 3, 1, 2).contiguous()
            past_context = self.past_context_encoder(past_btjc)
            xseq = torch.cat([time_emb, signwriting_emb, past_context, future_motion_emb], dim=0)
        else:
            past_motion_emb = self.past_motion_process(past_motion)
            xseq = torch.cat([time_emb, signwriting_emb, past_motion_emb, future_motion_emb], dim=0)
        
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqEncoder(xseq)
        output = output[-T_future:]
        
        result = self.pose_projection(output)
        result = result.permute(1, 2, 3, 0).contiguous()

        return result


# ============================================================
# 测试配置
# ============================================================
K = 178
D = 3
T_past = 40
T_future = 20
DIFFUSION_STEPS = 8

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

print("\n创建测试数据...")

gt_bjct = torch.zeros(1, K, D, T_future).to(device)
for t_idx in range(T_future):
    gt_bjct[:, :, 0, t_idx] = t_idx * 0.5

gt_disp = (gt_bjct[:, :, :, 1:] - gt_bjct[:, :, :, :-1]).abs().mean().item()
print(f"GT displacement: {gt_disp:.4f}")

past_bjct = torch.zeros(1, K, D, T_past).to(device)
for t_idx in range(T_past):
    past_bjct[:, :, 0, t_idx] = (t_idx - T_past) * 0.5

sign_img = torch.randn(1, 3, 224, 224).to(device)

# ============================================================
print("\n" + "=" * 70)
print("测试 1: Regression (修复后的模型)")
print("=" * 70)

model_reg = FixedModel(
    num_keypoints=K,
    num_dims_per_keypoint=D,
    use_mean_pool=True,
).to(device)

optimizer_reg = torch.optim.AdamW(model_reg.parameters(), lr=1e-3)

print("\n训练 Regression...")
model_reg.train()
for step in range(2001):
    optimizer_reg.zero_grad()
    
    t = torch.tensor([0]).to(device)
    pred = model_reg(gt_bjct, t, past_bjct, sign_img)
    
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

model_reg.eval()
with torch.no_grad():
    t = torch.tensor([0]).to(device)
    pred = model_reg(gt_bjct, t, past_bjct, sign_img)
    pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
    ratio_reg = pred_disp / gt_disp
    print(f"\n✓ Regression 最终: pred_disp={pred_disp:.4f}, ratio={ratio_reg:.4f}")

# ============================================================
print("\n" + "=" * 70)
print("测试 2: Diffusion (修复后的模型)")
print("=" * 70)

betas = cosine_beta_schedule(DIFFUSION_STEPS).numpy()
diffusion = GaussianDiffusion(
    betas=betas,
    model_mean_type=ModelMeanType.START_X,
    model_var_type=ModelVarType.FIXED_SMALL,
    loss_type=LossType.MSE,
    rescale_timesteps=False,
)

model_diff = FixedModel(
    num_keypoints=K,
    num_dims_per_keypoint=D,
    use_mean_pool=True,
).to(device)

optimizer_diff = torch.optim.AdamW(model_diff.parameters(), lr=1e-3)

print("\n训练 Diffusion...")
model_diff.train()
for step in range(2001):
    optimizer_diff.zero_grad()
    
    t = torch.randint(0, DIFFUSION_STEPS, (1,), device=device)
    noise = torch.randn_like(gt_bjct)
    x_t = diffusion.q_sample(gt_bjct, t, noise=noise)
    
    pred = model_diff(x_t, t, past_bjct, sign_img)
    
    loss_mse = F.mse_loss(pred, gt_bjct)
    pred_vel = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_vel = gt_bjct[:, :, :, 1:] - gt_bjct[:, :, :, :-1]
    loss_vel = F.mse_loss(pred_vel, gt_vel)
    
    loss = loss_mse + loss_vel
    loss.backward()
    optimizer_diff.step()
    
    if step % 400 == 0:
        with torch.no_grad():
            t_test = torch.tensor([0]).to(device)
            x_t_test = diffusion.q_sample(gt_bjct, t_test, noise=torch.randn_like(gt_bjct))
            pred_test = model_diff(x_t_test, t_test, past_bjct, sign_img)
            pred_disp = (pred_test[:, :, :, 1:] - pred_test[:, :, :, :-1]).abs().mean().item()
            ratio = pred_disp / gt_disp
        print(f"  Step {step}: loss={loss.item():.4f}, pred_disp={pred_disp:.4f}, ratio={ratio:.4f}")

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
    print("🎉 修复成功！现在请替换你的 models.py 文件")
    print("   cp models_fixed.py /path/to/signwriting_animation/diffusion/core/models.py")
else:
    print("⚠️ 还有问题，需要进一步调试")