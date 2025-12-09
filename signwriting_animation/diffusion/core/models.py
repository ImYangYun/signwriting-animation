import torch
import torch.nn as nn
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess, seq_encoder_factory


class SignWritingToPoseDiffusion(nn.Module):
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
                 cond_mask_prob: float = 0):
        super().__init__()
        self.verbose = False
        self.cond_mask_prob = cond_mask_prob
        self._forward_count = 0  # 计数器

        # local conditions
        input_feats = num_keypoints * num_dims_per_keypoint
        self.future_motion_process = MotionProcess(input_feats, num_latent_dims)
        self.past_motion_process = MotionProcess(input_feats, num_latent_dims)
        self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)

        # global conditions
        self.embed_signwriting = EmbedSignWriting(num_latent_dims, embedding_arch)
        self.embed_timestep = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)

        self.seqEncoder = seq_encoder_factory(arch=arch,
                                              latent_dim=num_latent_dims,
                                              ff_size=ff_size,
                                              num_layers=num_layers,
                                              num_heads=num_heads,
                                              dropout=dropout,
                                              activation=activation)

        # cross-attention to fuse past motion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=num_latent_dims,
            num_heads=4,
            dropout=0.1,
            batch_first=False
        )
        self.cross_ln = nn.LayerNorm(num_latent_dims)

        self.pose_projection = OutputProcessMLP(num_latent_dims, num_keypoints, num_dims_per_keypoint)

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
        Forward pass with selective debug output.
        Only prints on first call and every 100th call.
        """
        batch_size, num_keypoints, num_dims_per_keypoint, num_frames = x.shape
        
        # 🎯 只在第一次和每 100 次输出
        debug_this_call = (self._forward_count == 0) or (self._forward_count % 100 == 0)
        
        if debug_this_call:
            print(f"\n[FORWARD #{self._forward_count}]")
            print(f"  x: {x.shape}")
            print(f"  past_motion: {past_motion.shape}")
        
        # Format check
        if past_motion.dim() == 4:
            if past_motion.shape[1] == num_keypoints and past_motion.shape[2] == num_dims_per_keypoint:
                if debug_this_call:
                    print(f"  ✓ past_motion format correct")
            elif past_motion.shape[2] == num_keypoints and past_motion.shape[3] == num_dims_per_keypoint:
                if debug_this_call:
                    print(f"  ⚠️ permuting past_motion")
                past_motion = past_motion.permute(0, 2, 3, 1).contiguous()
            else:
                raise ValueError(f"Cannot interpret past_motion shape: {past_motion.shape}")

        time_emb = self.embed_timestep(timesteps)
        signwriting_emb = self.embed_signwriting(signwriting_im_batch)
        
        past_motion_emb = self.past_motion_process(past_motion)
        future_motion_emb = self.future_motion_process(x)

        Tf = future_motion_emb.size(0)
        B  = future_motion_emb.size(1)
        t = torch.linspace(0, 1, steps=Tf, device=future_motion_emb.device).view(Tf, 1, 1)
        t_latent = self.future_time_proj(t).expand(-1, B, -1)

        future_motion_emb = future_motion_emb + 0.1 * t_latent
        future_motion_emb = self.future_after_time_ln(future_motion_emb)

        time_cond = time_emb.repeat(Tf, 1, 1)
        sign_cond = signwriting_emb.repeat(Tf, 1, 1)

        xseq = future_motion_emb + 0.3 * time_cond + 0.3 * sign_cond
        xseq = self.sequence_pos_encoder(xseq)
        
        attn_out, _ = self.cross_attn(
            query=future_motion_emb,
            key=past_motion_emb,
            value=past_motion_emb
        )

        future_motion_emb = self.cross_ln(future_motion_emb + attn_out)
        xseq = xseq + future_motion_emb

        output = self.seqEncoder(xseq)[-num_frames:]
        output = self.pose_projection(output)
        result = output.permute(1, 2, 3, 0).contiguous()
        
        if debug_this_call:
            print(f"  result: {result.shape}, range=[{result.min():.4f}, {result.max():.4f}]")
        
        self._forward_count += 1
        return result

    def interface(self,
                  x: torch.Tensor,
                  timesteps: torch.Tensor,
                  y: dict):
        batch_size, num_keypoints, num_dims_per_keypoint, num_frames = x.shape

        signwriting_image = y['sign_image']
        past_motion = y['input_pose']

        # CFG on past motion
        keep_batch_idx = torch.rand(batch_size, device=past_motion.device) < (1 - self.cond_mask_prob)
        past_motion = past_motion * keep_batch_idx.view((batch_size, 1, 1, 1))

        return self.forward(x, timesteps, past_motion, signwriting_image)


class ResidualBlock(nn.Module):
    """Small residual MLP block."""
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
    """Strong MLP with residual blocks."""
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
            ResidualBlock(hidden_dim)
            for _ in range(num_layers)
        ])

        self.out_proj = nn.Linear(hidden_dim,
                                  num_keypoints * num_dims_per_keypoint)

    def forward(self, x):
        """
        Args:
            x: [T, B, D]
        Returns:
            [T, B, J, C]
        """
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