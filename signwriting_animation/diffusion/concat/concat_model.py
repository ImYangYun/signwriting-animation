"""
Temporal Concat Model - 复用原有组件，只改变融合方式

和 Frame-Independent 的区别：
- 不再 mean pool past context
- 把 past 和 noisy future concat 在时间维度
- 用 Transformer 处理整个序列
- 输出只取 future 部分
"""
import torch
import torch.nn as nn
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder


class EmbedSignWriting(nn.Module):
    """SignWriting image encoder using CLIP - 和原来一样"""
    
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


class TemporalConcatDiffusion(nn.Module):
    """
    Temporal Concat Diffusion Model
    
    和 Frame-Independent 的关键区别：
    1. 不用 ContextEncoder mean pool
    2. 直接 project past 和 noisy future 到同一空间
    3. Concat 在时间维度: [past, noisy_future] → [B, T_total, D]
    4. Transformer 处理整个序列（past 是干净的锚点）
    5. 输出只取 future 部分
    
    这样模型必须通过 attention 从 past 学习，而不是直接从 noisy frame 复制
    """
    
    def __init__(self,
                 num_keypoints: int = 178,
                 num_dims_per_keypoint: int = 3,
                 embedding_arch: str = 'openai/clip-vit-base-patch32',
                 num_latent_dims: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 t_past: int = 40,
                 t_future: int = 20):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
        self.t_past = t_past
        self.t_future = t_future
        self.t_total = t_past + t_future
        self.num_latent_dims = num_latent_dims
        self._forward_count = 0

        input_feats = num_keypoints * num_dims_per_keypoint  # 178 * 3 = 534

        # === 复用的组件 ===
        
        # SignWriting encoder - 和原来一样
        self.embed_signwriting = EmbedSignWriting(num_latent_dims, embedding_arch)
        
        # Timestep encoder - 和原来一样
        self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)
        self.time_embed = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)

        # === 新的组件 ===
        
        # Pose frame projection (past 和 future 用同一个)
        self.pose_proj = nn.Sequential(
            nn.Linear(input_feats, num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )
        
        # Learnable positional encoding for full sequence
        self.pos_encoding = PositionalEncoding(num_latent_dims, dropout)
        
        # Segment embedding: 区分 past (0) 和 future (1)
        self.segment_embed = nn.Embedding(2, num_latent_dims)
        
        # Transformer encoder (处理整个序列)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_latent_dims,
            nhead=num_heads,
            dim_feedforward=num_latent_dims * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # [B, T, D] 格式
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(num_latent_dims),
            nn.Linear(num_latent_dims, input_feats),
        )

    def forward(self,
                x: torch.Tensor,
                timesteps: torch.Tensor,
                past_motion: torch.Tensor,
                signwriting_im_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Predict clean x0 from noisy input.
        
        Args:
            x: Noisy motion [B, J, C, T_future] in BJCT format
            timesteps: Diffusion timestep [B]
            past_motion: Historical frames [B, J, C, T_past] in BJCT format
            signwriting_im_batch: SignWriting images [B, 3, H, W]
        
        Returns:
            Predicted x0 [B, J, C, T_future] in BJCT format
        """
        B, J, C, T_future = x.shape
        device = x.device
        debug = self._forward_count == 0

        # === 1. 转换格式 BJCT → BTJC → BT(J*C) ===
        # past: [B, J, C, T_past] → [B, T_past, J*C]
        past_btjc = past_motion.permute(0, 3, 1, 2).reshape(B, self.t_past, -1)
        # noisy future: [B, J, C, T_future] → [B, T_future, J*C]
        noisy_btjc = x.permute(0, 3, 1, 2).reshape(B, T_future, -1)
        
        # === 2. Project poses to latent space ===
        past_emb = self.pose_proj(past_btjc)    # [B, T_past, D]
        noisy_emb = self.pose_proj(noisy_btjc)  # [B, T_future, D]
        
        # === 3. Concat in time dimension ===
        # 关键！past 是干净的，future 是 noisy 的
        full_seq = torch.cat([past_emb, noisy_emb], dim=1)  # [B, T_total, D]
        
        # === 4. Add positional encoding ===
        full_seq = full_seq.permute(1, 0, 2)  # [T, B, D] for PositionalEncoding
        full_seq = self.pos_encoding(full_seq)
        full_seq = full_seq.permute(1, 0, 2)  # [B, T, D]
        
        # === 5. Add segment embedding ===
        segment_ids = torch.cat([
            torch.zeros(B, self.t_past, dtype=torch.long, device=device),
            torch.ones(B, T_future, dtype=torch.long, device=device),
        ], dim=1)  # [B, T_total]
        full_seq = full_seq + self.segment_embed(segment_ids)
        
        # === 6. Add time embedding (broadcast to all positions) ===
        time_emb = self.time_embed(timesteps).squeeze(0)  # [B, D]
        full_seq = full_seq + time_emb.unsqueeze(1)
        
        # === 7. Add sign embedding (broadcast to all positions) ===
        sign_emb = self.embed_signwriting(signwriting_im_batch)  # [B, D]
        full_seq = full_seq + sign_emb.unsqueeze(1)
        
        # === 8. Transformer ===
        # 这里 Transformer 可以 attend past（干净）和 future（noisy）
        # 模型必须从 past 学习，而不是直接复制 noisy frame
        out = self.transformer(full_seq)  # [B, T_total, D]
        
        # === 9. 只取 future 部分输出 ===
        future_out = out[:, self.t_past:, :]  # [B, T_future, D]
        
        # === 10. Output projection ===
        result = self.output_proj(future_out)  # [B, T_future, J*C]
        
        # === 11. Reshape back to BJCT ===
        result = result.reshape(B, T_future, J, C)
        result = result.permute(0, 2, 3, 1).contiguous()  # [B, J, C, T]

        if debug:
            disp = (result[:, :, :, 1:] - result[:, :, :, :-1]).abs().mean().item()
            print(f"[FORWARD] Temporal Concat: disp={disp:.6f}")

        self._forward_count += 1
        return result


# Quick test
if __name__ == "__main__":
    print("Testing TemporalConcatDiffusion...")
    
    model = TemporalConcatDiffusion(
        num_keypoints=178,
        num_dims_per_keypoint=3,
        num_latent_dims=256,
        num_heads=4,
        num_layers=4,
        t_past=40,
        t_future=20,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward
    B = 2
    x_noisy = torch.randn(B, 178, 3, 20)  # [B, J, C, T_future]
    timestep = torch.randint(0, 8, (B,))
    past = torch.randn(B, 178, 3, 40)      # [B, J, C, T_past]
    sign_img = torch.randn(B, 3, 224, 224)
    
    out = model(x_noisy, timestep, past, sign_img)
    print(f"Input shape:  {x_noisy.shape}")
    print(f"Past shape:   {past.shape}")
    print(f"Output shape: {out.shape}")
    print("✅ Forward pass successful!")