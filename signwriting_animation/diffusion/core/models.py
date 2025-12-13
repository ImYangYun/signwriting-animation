import torch
import torch.nn as nn
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess, seq_encoder_factory


class ContextEncoder(nn.Module):
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
        x: [T, B, J*C] (seq_first) 或 [B, T, J, C]
        returns: [1, B, D] (mean pooled, seq_first format)
        """
        if x.dim() == 3:
            # [T, B, D] -> [B, T, D]
            x = x.permute(1, 0, 2)
        elif x.dim() == 4:
            # [B, T, J, C] -> [B, T, J*C]
            B, T, J, C = x.shape
            x = x.reshape(B, T, J * C)
        
        x_emb = self.pose_encoder(x)      # [B, T, D]
        x_enc = self.encoder(x_emb)       # [B, T, D]
        context = x_enc.mean(dim=1)       # Mean pooling: [B, D]
        return context.unsqueeze(0)       # [1, B, D] (seq_first format)


class SignWritingToPoseDiffusionV2(nn.Module):
    """
    Diffusion 版本 - 直接预测 x0（不用残差）
    
    支持两种条件注入模式：
    - Concat 模式: [time(1), sign(1), past(40), x_t(20)] = 62 tokens
    - MeanPool 模式: [time(1), sign_ctx(1), past_ctx(1), x_t(20)] = 23 tokens (参考师姐)
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
                 residual_scale: float = 0.1,
                 use_mean_pool: bool = False):
        super().__init__()
        self.verbose = False
        self.cond_mask_prob = cond_mask_prob
        self._forward_count = 0
        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
        self.use_residual = False
        self.use_mean_pool = use_mean_pool

        input_feats = num_keypoints * num_dims_per_keypoint
        
        # Motion processors
        self.future_motion_process = MotionProcess(input_feats, num_latent_dims)
        self.past_motion_process = MotionProcess(input_feats, num_latent_dims)
        self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)

        # Global conditions
        self.embed_signwriting = EmbedSignWriting(num_latent_dims, embedding_arch)
        self.embed_timestep = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)

        # MeanPool
        if use_mean_pool:
            self.past_context_encoder = ContextEncoder(
                input_feats, num_latent_dims,
                num_layers=2, num_heads=num_heads, dropout=dropout
            )
            print(f"✓ 使用 MeanPool 模式 (参考师姐)")
        else:
            self.past_context_encoder = None
            print(f"✓ 使用 Concat 模式")

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

        self.past_last_encoder = nn.Sequential(
            nn.Linear(input_feats, num_latent_dims),
            nn.LayerNorm(num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )

        # Output projection
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
        Concat 版本 - 参考师姐实现
        
        输入：
            x: 带噪声的 motion [B, J, C, T]
            timesteps: diffusion timestep
            past_motion: 历史帧 [B, J, C, T_past]
            signwriting_im_batch: 条件图像
            
        输出：
            预测的干净 x0 [B, J, C, T]
            
        序列结构: [time(1), sign(1), past(T_past), x_t(T_future)]
        """
        batch_size, num_keypoints, num_dims_per_keypoint, num_frames = x.shape
        
        debug = (self._forward_count == 0) or (self._forward_count % 100 == 0)
        
        if debug:
            print(f"\n[FORWARD #{self._forward_count}] (Concat Mode)")
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

        T_past = past_motion.shape[-1]
        T_future = num_frames

        # Embeddings
        time_emb = self.embed_timestep(timesteps)  # [1, B, D]
        signwriting_emb = self.embed_signwriting(signwriting_im_batch)  # [1, B, D]
        
        # 处理输入的带噪声 motion
        future_motion_emb = self.future_motion_process(x)  # [T_future, B, D]

        if not self.use_mean_pool:
            past_motion_emb = self.past_motion_process(past_motion)  # [T_past, B, D]

        B = future_motion_emb.size(1)
        D = future_motion_emb.size(2)
        
        # Time encoding for future frames
        t = torch.linspace(0, 1, steps=T_future, device=future_motion_emb.device).view(T_future, 1, 1)
        t_latent = self.future_time_proj(t).expand(-1, B, -1)
        future_motion_emb = future_motion_emb + 0.1 * t_latent
        future_motion_emb = self.future_after_time_ln(future_motion_emb)

        # ========== 根据模式选择条件注入方式 ==========
        if self.use_mean_pool:
            # MeanPool 模式（参考师姐的 DisfluentContextEncoder）
            # past_motion: [B, J, C, T] -> mean pooled context [1, B, D]
            past_btjc = past_motion.permute(0, 3, 1, 2).contiguous()  # [B, T, J, C]
            past_context = self.past_context_encoder(past_btjc)  # [1, B, D]
            
            # 序列结构: [time(1), sign(1), past_ctx(1), x_t(T_future)] = 3 + T_future tokens
            xseq = torch.cat([
                time_emb,           # [1, B, D]
                signwriting_emb,    # [1, B, D]
                past_context,       # [1, B, D] - mean pooled!
                future_motion_emb,  # [T_future, B, D]
            ], dim=0)
            
            if debug:
                print(f"  MeanPool: time(1) + sign(1) + past_ctx(1) + x_t({T_future}) = {xseq.shape[0]} tokens")
        else:
            # Concat 模式（原来的方式）
            # 序列结构: [time(1), sign(1), past(T_past), x_t(T_future)]
            xseq = torch.cat([
                time_emb,           # [1, B, D]
                signwriting_emb,    # [1, B, D]
                past_motion_emb,    # [T_past, B, D]
                future_motion_emb,  # [T_future, B, D]
            ], dim=0)
            
            if debug:
                print(f"  Concat: time(1) + sign(1) + past({T_past}) + x_t({T_future}) = {xseq.shape[0]} tokens")
        
        # Positional encoding
        xseq = self.sequence_pos_encoder(xseq)

        # Sequence encoding (Transformer)
        output = self.seqEncoder(xseq)
        
        # 取最后 T_future 帧（对应 x_t 的位置）
        output = output[-T_future:]  # [T_future, B, D]
        
        # 直接预测 x0
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