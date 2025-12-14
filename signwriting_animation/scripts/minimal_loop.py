# -*- coding: utf-8 -*-
"""
诊断 train() vs eval() 差异

现象：
- model.train(): ratio ≈ 0.3-0.4
- model.eval(): ratio = 0.0000

可能原因：Dropout 在训练时引入随机性，eval 时关闭导致坍缩
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("诊断 train() vs eval() 差异")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess, seq_encoder_factory
from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from transformers import CLIPModel

# ============================================================
# 模型定义（完全无 Dropout 版本）
# ============================================================

class ContextEncoderNoDropout(nn.Module):
    def __init__(self, input_feats, latent_dim, num_layers=2, num_heads=4):
        super().__init__()
        self.pose_encoder = nn.Linear(input_feats, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.0,  # 无 Dropout!
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
    def __init__(self, num_latent_dims, num_keypoints, num_dims_per_keypoint, hidden_dim=512):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
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


class PositionalEncodingNoDropout(nn.Module):
    """无 Dropout 的位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [T, B, D]
        return x + self.pe[:x.size(0)]


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


class FixedModelNoDropout(nn.Module):
    """完全无 Dropout 的模型"""
    def __init__(self, num_keypoints, num_dims_per_keypoint, num_latent_dims=256,
                 ff_size=1024, num_layers=8, num_heads=4):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint

        input_feats = num_keypoints * num_dims_per_keypoint
        
        self.future_motion_process = MotionProcess(input_feats, num_latent_dims)
        self.past_motion_process = MotionProcess(input_feats, num_latent_dims)
        
        # 无 Dropout 的位置编码
        self.sequence_pos_encoder = PositionalEncodingNoDropout(num_latent_dims)

        self.embed_signwriting = EmbedSignWriting(num_latent_dims)
        
        # TimestepEmbedder 用原版（它内部也用 PositionalEncoding）
        # 我们用简单的 embedding 替代
        self.embed_timestep = nn.Embedding(1000, num_latent_dims)

        # 无 Dropout 的 ContextEncoder
        self.past_context_encoder = ContextEncoderNoDropout(input_feats, num_latent_dims)
        print(f"✓ 使用 MeanPool 模式 (无 Dropout)")

        # 无 Dropout 的 Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_latent_dims, nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=0.0,  # 无 Dropout!
            activation="gelu", batch_first=False,
        )
        self.seqEncoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        print(f"✓ Transformer (无 Dropout)")

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

        # Timestep embedding (简化版)
        time_emb = self.embed_timestep(timesteps.clamp(0, 999)).unsqueeze(0)  # [1, B, D]
        
        signwriting_emb = self.embed_signwriting(signwriting_im_batch)
        future_motion_emb = self.future_motion_process(x)

        t = torch.linspace(0, 1, steps=T_future, device=x.device).view(T_future, 1, 1)
        t_latent = self.future_time_proj(t).expand(-1, B, -1)
        future_motion_emb = future_motion_emb + 0.1 * t_latent

        past_btjc = past_motion.permute(0, 3, 1, 2).contiguous()
        past_context = self.past_context_encoder(past_btjc)
        xseq = torch.cat([time_emb, signwriting_emb, past_context, future_motion_emb], dim=0)
        
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqEncoder(xseq)
        output = output[-T_future:]
        
        result = self.pose_projection(output)
        result = result.permute(1, 2, 3, 0).contiguous()

        return result


# ============================================================
# 测试
# ============================================================
K = 178
D = 3
T_past = 40
T_future = 20

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
print("测试: 无 Dropout 模型")
print("=" * 70)

model = FixedModelNoDropout(
    num_keypoints=K,
    num_dims_per_keypoint=D,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("\n训练...")
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

# 测试 train 模式
print("\n测试 (train 模式):")
model.train()
with torch.no_grad():
    t = torch.tensor([0]).to(device)
    pred = model(gt_bjct, t, past_bjct, sign_img)
    pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
    ratio_train = pred_disp / gt_disp
    print(f"  ratio (train mode): {ratio_train:.4f}")

# 测试 eval 模式
print("\n测试 (eval 模式):")
model.eval()
with torch.no_grad():
    t = torch.tensor([0]).to(device)
    pred = model(gt_bjct, t, past_bjct, sign_img)
    pred_disp = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean().item()
    ratio_eval = pred_disp / gt_disp
    print(f"  ratio (eval mode): {ratio_eval:.4f}")

print("\n" + "=" * 70)
print("📊 结论")
print("=" * 70)

if abs(ratio_train - ratio_eval) < 0.1:
    print(f"✅ train 和 eval 一致: train={ratio_train:.4f}, eval={ratio_eval:.4f}")
    if ratio_eval > 0.5:
        print("🎉 修复成功！")
    else:
        print("⚠️ 还有其他问题导致运动丢失")
else:
    print(f"❌ train 和 eval 不一致: train={ratio_train:.4f}, eval={ratio_eval:.4f}")
    print("   问题确认是 Dropout 导致的！")