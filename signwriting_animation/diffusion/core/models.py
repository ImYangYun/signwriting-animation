import torch
import torch.nn as nn
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess, seq_encoder_factory


class SignWritingToPoseDiffusionV2(nn.Module):
    """
    Diffusion 版本 - 直接预测 x0（不用残差）
    
    关键改动：去掉残差，让模型直接预测绝对位置
    这样 diffusion 采样才能正常工作
    """
    
    def __init__(self,
                 num_keypoints: int,
                 num_dims_per_keypoint: int,
                 embedding_arch: str = 'openai/clip-vit-base-patch32',
                 num_latent_dims: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 activation: str = "gelu",
                 arch: str = "trans_enc",
                 cond_mask_prob: float = 0,
                 residual_scale: float = 0.1):  # 保留参数但不使用
        super().__init__()
        self.verbose = False
        self.cond_mask_prob = cond_mask_prob
        self._forward_count = 0
        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
        # 不再使用 residual_scale
        self.use_residual = False  # 关闭残差

        input_feats = num_keypoints * num_dims_per_keypoint
        
        # Motion processors
        self.future_motion_process = MotionProcess(input_feats, num_latent_dims)
        self.past_motion_process = MotionProcess(input_feats, num_latent_dims)
        self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)

        # Global conditions
        self.embed_signwriting = EmbedSignWriting(num_latent_dims, embedding_arch)
        self.embed_timestep = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)

        # Sequence encoder
        self.seqEncoder = seq_encoder_factory(
            arch=arch,
            latent_dim=num_latent_dims,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation
        )

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=num_latent_dims,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=False
            )
            for _ in range(3)
        ])
        self.cross_lns = nn.ModuleList([
            nn.LayerNorm(num_latent_dims)
            for _ in range(3)
        ])

        # past_last encoder (用于条件注入，但不做残差)
        self.past_last_encoder = nn.Sequential(
            nn.Linear(input_feats, num_latent_dims),
            nn.LayerNorm(num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )

        # Output projection - 直接输出 x0
        self.pose_projection = OutputProcessMLP(
            num_latent_dims, num_keypoints, num_dims_per_keypoint
        )

        # Time projection for future frames
        self.future_time_proj = nn.Sequential(
            nn.Linear(1, num_latent_dims),
            nn.SiLU(),
            nn.Linear(num_latent_dims, num_latent_dims)
        )
        self.future_after_time_ln = nn.LayerNorm(num_latent_dims)

    def forward(self,
                x: torch.Tensor,
                timesteps: torch.Tensor,
                past_motion: torch.Tensor,
                signwriting_im_batch: torch.Tensor):
        """
        直接预测 x0（不用残差）
        
        输入：
            x: 带噪声的 motion [B, J, C, T]
            timesteps: diffusion timestep
            past_motion: 历史帧 [B, J, C, T_past]
            signwriting_im_batch: 条件图像
            
        输出：
            预测的干净 x0 [B, J, C, T]
        """
        batch_size, num_keypoints, num_dims_per_keypoint, num_frames = x.shape
        
        debug = (self._forward_count == 0) or (self._forward_count % 100 == 0)
        
        if debug:
            print(f"\n[FORWARD #{self._forward_count}] (No Residual Mode)")
            print(f"  x: {x.shape}, range=[{x.min():.2f}, {x.max():.2f}]")
            print(f"  past_motion: {past_motion.shape}")

        # Format check
        if past_motion.dim() == 4:
            if past_motion.shape[1] == num_keypoints and past_motion.shape[2] == num_dims_per_keypoint:
                pass  # [B, J, C, T] format
            elif past_motion.shape[2] == num_keypoints and past_motion.shape[3] == num_dims_per_keypoint:
                past_motion = past_motion.permute(0, 2, 3, 1).contiguous()
            else:
                raise ValueError(f"Cannot interpret past_motion shape: {past_motion.shape}")

        # 获取 past 的最后一帧作为残差基础
        past_last = past_motion[..., -1]  # [B, J, C]
        past_last_flat = past_last.reshape(batch_size, -1)  # [B, J*C]
        
        if debug:
            print(f"  past_last mean: {past_last.mean().item():.4f}")

        # Embeddings
        time_emb = self.embed_timestep(timesteps)
        signwriting_emb = self.embed_signwriting(signwriting_im_batch)
        past_last_emb = self.past_last_encoder(past_last_flat)  # [B, D]
        
        # 处理输入的带噪声 motion
        future_motion_emb = self.future_motion_process(x)  # [T, B, D]
        past_motion_emb = self.past_motion_process(past_motion)  # [T_past, B, D]

        Tf = future_motion_emb.size(0)
        B = future_motion_emb.size(1)
        
        # Time encoding for future frames
        t = torch.linspace(0, 1, steps=Tf, device=future_motion_emb.device).view(Tf, 1, 1)
        t_latent = self.future_time_proj(t).expand(-1, B, -1)

        future_motion_emb = future_motion_emb + 0.1 * t_latent
        future_motion_emb = self.future_after_time_ln(future_motion_emb)

        # past_last_emb 作为条件
        past_last_emb_expanded = past_last_emb.unsqueeze(0).expand(Tf, -1, -1)  # [Tf, B, D]

        # Condition fusion - 大幅降低权重，让模型更依赖 x_t
        time_cond = time_emb.repeat(Tf, 1, 1)
        sign_cond = signwriting_emb.repeat(Tf, 1, 1)

        # 条件权重极低，强迫模型从 x_t 获取信息
        cond_sum = (0.1 * time_cond 
                   + 0.1 * sign_cond 
                   + 0.1 * past_last_emb_expanded)  # 总共 0.3，远小于 x_t 的 1.0
        
        # x_t (future_motion_emb) 保持权重 1.0
        xseq = future_motion_emb + cond_sum
        xseq = self.sequence_pos_encoder(xseq)

        # Cross-attention with past
        for cross_attn, cross_ln in zip(self.cross_attn_layers, self.cross_lns):
            attn_out, _ = cross_attn(
                query=xseq,
                key=past_motion_emb,
                value=past_motion_emb
            )
            xseq = cross_ln(xseq + attn_out)

        # Sequence encoding
        output = self.seqEncoder(xseq)[-num_frames:]
        result = self.pose_projection(output)  # [T, B, J, C]
        result = result.permute(1, 2, 3, 0).contiguous()  # [B, J, C, T]

        if debug:
            print(f"  result: {result.shape}, range=[{result.min():.4f}, {result.max():.4f}]")
            # 计算帧间差异
            if result.size(-1) > 1:
                disp = (result[..., 1:] - result[..., :-1]).abs().mean().item()
                print(f"  result disp: {disp:.6f}")

        self._forward_count += 1
        return result

    def interface(self, x, timesteps, y):
        batch_size = x.shape[0]
        signwriting_image = y['sign_image']
        past_motion = y['input_pose']

        # CFG
        keep_batch_idx = torch.rand(batch_size, device=past_motion.device) < (1 - self.cond_mask_prob)
        past_motion = past_motion * keep_batch_idx.view((batch_size, 1, 1, 1))

        return self.forward(x, timesteps, past_motion, signwriting_image)


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
    def __init__(self,
                 num_latent_dims: int,
                 num_keypoints: int,
                 num_dims_per_keypoint: int,
                 hidden_dim: int = 1024,
                 num_layers: int = 6):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint

        self.in_proj = nn.Linear(num_latent_dims, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, num_keypoints * num_dims_per_keypoint)

    def forward(self, x):
        T, B, D = x.shape
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        y = self.out_proj(h)
        return y.reshape(T, B, self.num_keypoints, self.num_dims_per_keypoint)


class EmbedSignWriting(nn.Module):
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
        return embeddings_batch[None, ...]


# Alias
SignWritingToPoseDiffusion = SignWritingToPoseDiffusionV2