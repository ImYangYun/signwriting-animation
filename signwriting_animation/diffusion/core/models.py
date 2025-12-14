import torch
import torch.nn as nn
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder


class EmbedSignWriting(nn.Module):
    """SignWriting 图像编码器（使用 CLIP）"""
    def __init__(self, num_latent_dims: int, embedding_arch: str = 'openai/clip-vit-base-patch32'):
        super().__init__()
        self.model = CLIPModel.from_pretrained(embedding_arch)
        self.proj = None
        if (num_embedding_dims := self.model.visual_projection.out_features) != num_latent_dims:
            self.proj = nn.Linear(num_embedding_dims, num_latent_dims)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        embeddings_batch = self.model.get_image_features(pixel_values=image_batch)
        if self.proj is not None:
            embeddings_batch = self.proj(embeddings_batch)
        return embeddings_batch


class ContextEncoder(nn.Module):
    """Past motion 上下文编码器"""
    def __init__(self, input_feats: int, latent_dim: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pose_encoder = nn.Linear(input_feats, latent_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, J, C] or [B, T, J*C]
        returns: [B, D] (mean pooled context)
        """
        if x.dim() == 4:
            B, T, J, C = x.shape
            x = x.reshape(B, T, J * C)
        
        x_emb = self.pose_encoder(x)      # [B, T, D]
        x_enc = self.encoder(x_emb)       # [B, T, D]
        context = x_enc.mean(dim=1)       # Mean pooling: [B, D]
        return context


class SignWritingToPoseDiffusionV2(nn.Module):
    """
    Diffusion 模型 - 修复版
    
    关键架构改动：
    - 不让 future tokens 通过 Transformer self-attention（会被平均化！）
    - 每帧独立解码，用位置编码区分不同时间步
    - x_t 的每帧单独编码并参与解码
    """
    
    def __init__(self,
                 num_keypoints: int,
                 num_dims_per_keypoint: int,
                 embedding_arch: str = 'openai/clip-vit-base-patch32',
                 num_latent_dims: int = 256,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 cond_mask_prob: float = 0,
                 t_past: int = 40,
                 t_future: int = 20):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
        self.cond_mask_prob = cond_mask_prob
        self.t_past = t_past
        self.t_future = t_future
        self._forward_count = 0

        input_feats = num_keypoints * num_dims_per_keypoint
        
        # === 条件编码器 ===
        # Past motion 编码
        self.past_context_encoder = ContextEncoder(
            input_feats, num_latent_dims,
            num_layers=2, num_heads=num_heads, dropout=dropout
        )
        
        # SignWriting 图像编码
        self.embed_signwriting = EmbedSignWriting(num_latent_dims, embedding_arch)
        
        # Timestep 编码
        self.time_embed = nn.Embedding(1000, num_latent_dims)
        
        # === x_t 帧编码器 ===
        # 每帧独立编码
        self.xt_frame_encoder = nn.Sequential(
            nn.Linear(input_feats, num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )
        
        # === 输出位置编码 ===
        # 用于区分不同时间步的输出
        self.output_pos_embed = nn.Embedding(t_future, num_latent_dims)
        
        # === 解码器 ===
        # 输入: context + x_t[t] + pos[t]
        # 输出: 该帧的预测
        decoder_input_dim = num_latent_dims * 3
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, input_feats),
        )
        
        print(f"✓ SignWritingToPoseDiffusionV2 初始化")
        print(f"  - 每帧独立解码（避免 Transformer 平均化）")
        print(f"  - t_past={t_past}, t_future={t_future}")

    def forward(self,
                x: torch.Tensor,
                timesteps: torch.Tensor,
                past_motion: torch.Tensor,
                signwriting_im_batch: torch.Tensor):
        """
        输入：
            x: 带噪声的 motion [B, J, C, T_future] (BJCT 格式)
            timesteps: diffusion timestep [B]
            past_motion: 历史帧 [B, J, C, T_past] (BJCT 格式)
            signwriting_im_batch: 条件图像 [B, 3, H, W]
            
        输出：
            预测的 x0 [B, J, C, T_future]
        """
        B, J, C, T_future = x.shape
        device = x.device
        
        debug = self._forward_count == 0
        
        # === 处理 past_motion 格式 ===
        # 支持 BJCT 或 BTJC 输入
        if past_motion.dim() == 4:
            if past_motion.shape[1] == J and past_motion.shape[2] == C:
                # BJCT -> BTJC
                past_btjc = past_motion.permute(0, 3, 1, 2).contiguous()
            else:
                # 已经是 BTJC
                past_btjc = past_motion
        
        # === 编码条件 ===
        # Past context
        past_ctx = self.past_context_encoder(past_btjc)  # [B, D]
        
        # SignWriting
        sign_emb = self.embed_signwriting(signwriting_im_batch)  # [B, D]
        
        # Timestep
        time_emb = self.time_embed(timesteps.clamp(0, 999))  # [B, D]
        
        # 融合 context
        context = past_ctx + sign_emb + time_emb  # [B, D]
        
        # === 每帧独立解码 ===
        outputs = []
        for t in range(T_future):
            # x_t 的第 t 帧: [B, J, C] -> [B, J*C]
            xt_frame = x[:, :, :, t].reshape(B, -1)
            xt_emb = self.xt_frame_encoder(xt_frame)  # [B, D]
            
            # 位置编码
            pos_idx = torch.tensor([t], device=device)
            pos_emb = self.output_pos_embed(pos_idx).expand(B, -1)  # [B, D]
            
            # 拼接并解码
            dec_input = torch.cat([context, xt_emb, pos_emb], dim=-1)  # [B, D*3]
            out = self.decoder(dec_input)  # [B, J*C]
            outputs.append(out)
        
        # Stack: [T, B, J*C] -> [B, J, C, T]
        result = torch.stack(outputs, dim=0)  # [T, B, J*C]
        result = result.permute(1, 0, 2)  # [B, T, J*C]
        result = result.reshape(B, T_future, J, C)  # [B, T, J, C]
        result = result.permute(0, 2, 3, 1).contiguous()  # [B, J, C, T]
        
        if debug:
            disp = (result[:, :, :, 1:] - result[:, :, :, :-1]).abs().mean().item()
            print(f"[FORWARD] result shape={result.shape}, disp={disp:.6f}")
        
        self._forward_count += 1
        return result

    def interface(self, x, timesteps, y):
        """Diffusion training interface (兼容 CAMDM)"""
        batch_size = x.shape[0]
        signwriting_image = y['sign_image']
        past_motion = y['input_pose']

        # Classifier-free guidance dropout
        if self.cond_mask_prob > 0:
            keep_batch_idx = torch.rand(batch_size, device=past_motion.device) < (1 - self.cond_mask_prob)
            past_motion = past_motion * keep_batch_idx.view((batch_size, 1, 1, 1))

        return self.forward(x, timesteps, past_motion, signwriting_image)


# Alias
SignWritingToPoseDiffusion = SignWritingToPoseDiffusionV2