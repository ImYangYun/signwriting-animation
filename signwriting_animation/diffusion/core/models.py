"""
SignWriting-to-Pose Diffusion Model V2 - Improved with CAMDM Components

This version adds CAMDM components to V2's frame-independent decoder:
- ✅ V2's frame-independent decoding (prevents motion collapse)
- ✅ PositionalEncoding from CAMDM (better temporal modeling)
- ✅ TimestepEmbedder from CAMDM (better timestep representation)

Progressive improvement strategy:
1. V2 baseline (current) - works well
2. V2 + PositionalEncoding (this version, Step 1)
3. V2 + PositionalEncoding + TimestepEmbedder (this version, Step 2)
"""

import torch
import torch.nn as nn
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder


class EmbedSignWriting(nn.Module):
    """SignWriting image encoder using CLIP."""
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
    """
    Past motion context encoder with PositionalEncoding.
    
    IMPROVEMENT over V2 baseline:
    - Added PositionalEncoding (from CAMDM) for better temporal awareness
    - Transformer can now distinguish frame order in past motion
    
    Args:
        input_feats: Input feature dimension
        latent_dim: Latent dimension
        num_layers: Number of Transformer layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_positional_encoding: Whether to use CAMDM's PositionalEncoding
    """
    def __init__(self, 
                 input_feats: int, 
                 latent_dim: int, 
                 num_layers: int = 2, 
                 num_heads: int = 4, 
                 dropout: float = 0.1,
                 use_positional_encoding: bool = True):
        super().__init__()
        self.use_positional_encoding = use_positional_encoding
        
        # Project pose features to latent space
        self.pose_encoder = nn.Linear(input_feats, latent_dim)
        
        # CAMDM Component: PositionalEncoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(latent_dim, dropout)
            print("  ✓ ContextEncoder: Using CAMDM PositionalEncoding")
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,  # CAMDM uses [T, B, D] format
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode past motion with positional encoding.
        
        Args:
            x: Past poses [B, T, J, C] or [B, T, J*C]
        Returns:
            context: Mean-pooled context [B, D]
        """
        # Reshape to [B, T, J*C] if needed
        if x.dim() == 4:
            B, T, J, C = x.shape
            x = x.reshape(B, T, J * C)
        
        # Encode to latent space
        x_emb = self.pose_encoder(x)      # [B, T, D]
        
        # Convert to CAMDM format [T, B, D] for PositionalEncoding
        x_emb = x_emb.permute(1, 0, 2)    # [T, B, D]
        
        # Apply PositionalEncoding if enabled
        if self.use_positional_encoding:
            x_emb = self.pos_encoding(x_emb)  # Adds temporal information
        
        # Apply Transformer encoder
        x_enc = self.encoder(x_emb)       # [T, B, D]
        
        # Convert back to [B, T, D] and mean pool
        x_enc = x_enc.permute(1, 0, 2)    # [B, T, D]
        context = x_enc.mean(dim=1)       # [B, D]
        return context


class SignWritingToPoseDiffusionV2(nn.Module):
    """
    V2 Improved: Frame-Independent Decoder + CAMDM Components
    
    Improvements over V2 baseline:
    1. ✅ PositionalEncoding in ContextEncoder (better temporal modeling)
    2. ✅ TimestepEmbedder (optional, better timestep representation)
    
    Architecture:
    - ContextEncoder with PositionalEncoding
    - SignWriting CLIP encoder
    - TimestepEmbedder (optional) or simple Embedding
    - Frame-independent decoder (V2's key innovation, preserved)
    
    Args:
        num_keypoints: Number of pose keypoints
        num_dims_per_keypoint: Dimensions per keypoint
        embedding_arch: CLIP model architecture
        num_latent_dims: Latent dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        cond_mask_prob: Condition masking probability for CFG
        t_past: Number of past frames
        t_future: Number of future frames
        use_positional_encoding: Use CAMDM PositionalEncoding
        use_timestep_embedder: Use CAMDM TimestepEmbedder
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
                 t_future: int = 20,
                 use_positional_encoding: bool = True,
                 use_timestep_embedder: bool = True):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
        self.cond_mask_prob = cond_mask_prob
        self.t_past = t_past
        self.t_future = t_future
        self.use_positional_encoding = use_positional_encoding
        self.use_timestep_embedder = use_timestep_embedder
        self._forward_count = 0

        input_feats = num_keypoints * num_dims_per_keypoint
        
        # === Condition Encoders ===
        # IMPROVEMENT 1: ContextEncoder with PositionalEncoding
        self.past_context_encoder = ContextEncoder(
            input_feats, num_latent_dims,
            num_layers=2, num_heads=num_heads, dropout=dropout,
            use_positional_encoding=use_positional_encoding
        )
        
        # SignWriting encoder (unchanged)
        self.embed_signwriting = EmbedSignWriting(num_latent_dims, embedding_arch)
        
        # IMPROVEMENT 2: TimestepEmbedder (optional)
        if use_timestep_embedder:
            # Use CAMDM's TimestepEmbedder with PositionalEncoding
            self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)
            self.time_embed = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)
            print("  ✓ Using CAMDM TimestepEmbedder")
        else:
            # Use simple embedding (V2 baseline)
            self.time_embed = nn.Embedding(1000, num_latent_dims)
            print("  ✓ Using simple Embedding for timestep")
        
        # === Noisy Frame Encoder (unchanged from V2) ===
        self.xt_frame_encoder = nn.Sequential(
            nn.Linear(input_feats, num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )
        
        # === Output Positional Embeddings (unchanged from V2) ===
        self.output_pos_embed = nn.Embedding(t_future, num_latent_dims)
        
        # === Frame Decoder (V2's key innovation, preserved!) ===
        decoder_input_dim = num_latent_dims * 3
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, input_feats),
        )
        
        print(f"✓ SignWritingToPoseDiffusionV2Improved initialized")
        print(f"  - Frame-independent decoding (V2 innovation, preserved)")
        print(f"  - PositionalEncoding: {use_positional_encoding}")
        print(f"  - TimestepEmbedder: {use_timestep_embedder}")

    def forward(self,
                x: torch.Tensor,
                timesteps: torch.Tensor,
                past_motion: torch.Tensor,
                signwriting_im_batch: torch.Tensor):
        """
        Forward pass with CAMDM improvements.
        
        Args:
            x: Noisy motion [B, J, C, T_future]
            timesteps: Diffusion timestep [B]
            past_motion: Historical frames [B, J, C, T_past]
            signwriting_im_batch: Condition images [B, 3, H, W]
        
        Returns:
            Predicted x0 [B, J, C, T_future]
        """
        B, J, C, T_future = x.shape
        device = x.device
        
        debug = self._forward_count == 0
        
        # === Handle past_motion format ===
        if past_motion.dim() == 4:
            if past_motion.shape[1] == J and past_motion.shape[2] == C:
                past_btjc = past_motion.permute(0, 3, 1, 2).contiguous()
            else:
                past_btjc = past_motion
        
        # === Encode Conditions ===
        # IMPROVEMENT 1: ContextEncoder with PositionalEncoding
        past_ctx = self.past_context_encoder(past_btjc)  # [B, D]
        
        # SignWriting embedding
        sign_emb = self.embed_signwriting(signwriting_im_batch)  # [B, D]
        
        # IMPROVEMENT 2: TimestepEmbedder or simple Embedding
        if self.use_timestep_embedder:
            # TimestepEmbedder returns [1, B, D], need to squeeze
            time_emb = self.time_embed(timesteps)  # [1, B, D]
            time_emb = time_emb.squeeze(0)         # [B, D]
        else:
            # Simple embedding
            time_emb = self.time_embed(timesteps.clamp(0, 999))  # [B, D]
        
        # Fuse conditions
        context = past_ctx + sign_emb + time_emb  # [B, D]
        
        # === V2's Frame-Independent Decoding (PRESERVED!) ===
        outputs = []
        for t in range(T_future):
            # Extract frame t
            xt_frame = x[:, :, :, t].reshape(B, -1)
            xt_emb = self.xt_frame_encoder(xt_frame)
            
            # Add positional embedding
            pos_idx = torch.tensor([t], device=device)
            pos_emb = self.output_pos_embed(pos_idx).expand(B, -1)
            
            # Decode
            dec_input = torch.cat([context, xt_emb, pos_emb], dim=-1)
            out = self.decoder(dec_input)
            outputs.append(out)
        
        # Stack and reshape
        result = torch.stack(outputs, dim=0)
        result = result.permute(1, 0, 2)
        result = result.reshape(B, T_future, J, C)
        result = result.permute(0, 2, 3, 1).contiguous()
        
        if debug:
            disp = (result[:, :, :, 1:] - result[:, :, :, :-1]).abs().mean().item()
            print(f"[FORWARD] V2Improved: disp={disp:.6f}")
        
        self._forward_count += 1
        return result

    def interface(self, x, timesteps, y):
        """Diffusion training interface."""
        batch_size = x.shape[0]
        signwriting_image = y['sign_image']
        past_motion = y['input_pose']

        if self.cond_mask_prob > 0:
            keep_batch_idx = torch.rand(batch_size, device=past_motion.device) < (1 - self.cond_mask_prob)
            past_motion = past_motion * keep_batch_idx.view((batch_size, 1, 1, 1))

        return self.forward(x, timesteps, past_motion, signwriting_image)


# Factory function for easy switching
def create_v2_model(version='improved', **kwargs):
    """
    Factory function to create different V2 versions.
    
    Args:
        version: 'baseline', 'with_pos', 'with_timestep', 'improved'
        **kwargs: Model arguments
    
    Returns:
        Model instance
    """
    if version == 'baseline':
        # V2 baseline: no CAMDM components
        return SignWritingToPoseDiffusionV2(
            use_positional_encoding=False,
            use_timestep_embedder=False,
            **kwargs
        )
    elif version == 'with_pos':
        # V2 + PositionalEncoding only
        return SignWritingToPoseDiffusionV2(
            use_positional_encoding=True,
            use_timestep_embedder=False,
            **kwargs
        )
    elif version == 'with_timestep':
        # V2 + TimestepEmbedder only
        return SignWritingToPoseDiffusionV2(
            use_positional_encoding=False,
            use_timestep_embedder=True,
            **kwargs
        )
    elif version == 'improved':
        # V2 + both improvements
        return SignWritingToPoseDiffusionV2(
            use_positional_encoding=True,
            use_timestep_embedder=True,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown version: {version}")