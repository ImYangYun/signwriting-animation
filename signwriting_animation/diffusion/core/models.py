import torch
import torch.nn as nn
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder


class EmbedSignWriting(nn.Module):
    """
    SignWriting image encoder using CLIP vision model.
    
    Encodes SignWriting symbol images into latent embeddings that condition
    the pose generation process.
    
    Args:
        num_latent_dims: Dimension of output latent embeddings
        embedding_arch: Pretrained CLIP model identifier
    """
    def __init__(self, num_latent_dims: int, embedding_arch: str = 'openai/clip-vit-base-patch32'):
        super().__init__()
        # Load pretrained CLIP model for vision encoding
        self.model = CLIPModel.from_pretrained(embedding_arch)
        self.proj = None
        
        # Add projection layer if CLIP output dimension differs from target
        if (num_embedding_dims := self.model.visual_projection.out_features) != num_latent_dims:
            self.proj = nn.Linear(num_embedding_dims, num_latent_dims)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Encode SignWriting images to latent embeddings.
        
        Args:
            image_batch: Batch of images [B, 3, H, W]
            
        Returns:
            embeddings_batch: Latent embeddings [B, D]
        """
        # Extract image features using CLIP vision encoder
        embeddings_batch = self.model.get_image_features(pixel_values=image_batch)
        
        # Project to target dimension if needed
        if self.proj is not None:
            embeddings_batch = self.proj(embeddings_batch)
        return embeddings_batch


class ContextEncoder(nn.Module):
    """
    Past motion context encoder using Transformer.
    
    Encodes historical pose sequences into a single context vector using
    self-attention and mean pooling.
    
    Args:
        input_feats: Input feature dimension (J*C for flattened poses)
        latent_dim: Latent dimension for Transformer
        num_layers: Number of Transformer encoder layers
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(self, input_feats: int, latent_dim: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Project pose features to latent space
        self.pose_encoder = nn.Linear(input_feats, latent_dim)
        
        # Transformer encoder for temporal modeling
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
        Encode past motion sequence to context vector.
        
        Args:
            x: Past poses [B, T, J, C] or [B, T, J*C]
            
        Returns:
            context: Mean-pooled context vector [B, D]
        """
        # Reshape to [B, T, J*C] if needed
        if x.dim() == 4:
            B, T, J, C = x.shape
            x = x.reshape(B, T, J * C)
        
        # Encode to latent space
        x_emb = self.pose_encoder(x)      # [B, T, D]
        
        # Apply Transformer encoder
        x_enc = self.encoder(x_emb)       # [B, T, D]
        
        # Mean pooling over time to get single context vector
        context = x_enc.mean(dim=1)       # [B, D]
        return context


class SignWritingToPoseDiffusionV2(nn.Module):
    """
    Diffusion model for SignWriting-to-Pose generation (V2 - Fixed version).
    
    Key architectural improvements:
    - Frame-independent decoding: Each future frame is decoded separately to prevent
      Transformer self-attention from averaging them together
    - Each frame receives: context + noisy_frame_encoding + positional_embedding
    - This preserves motion dynamics and prevents collapse to static poses
    
    Args:
        num_keypoints: Number of pose keypoints (e.g., 178 for MediaPipe Holistic)
        num_dims_per_keypoint: Dimensions per keypoint (typically 3 for x,y,z)
        embedding_arch: CLIP model architecture for SignWriting encoding
        num_latent_dims: Latent dimension for all encoders/decoders
        num_heads: Number of attention heads in context encoder
        dropout: Dropout probability
        cond_mask_prob: Probability of masking conditions during training (for CFG)
        t_past: Number of past frames for context
        t_future: Number of future frames to predict
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
        
        # === Condition Encoders ===
        # Encode past motion history into context
        self.past_context_encoder = ContextEncoder(
            input_feats, num_latent_dims,
            num_layers=2, num_heads=num_heads, dropout=dropout
        )
        
        # Encode SignWriting symbol image
        self.embed_signwriting = EmbedSignWriting(num_latent_dims, embedding_arch)
        
        # Encode diffusion timestep
        self.time_embed = nn.Embedding(1000, num_latent_dims)
        
        # === Noisy Frame Encoder ===
        # Encode each noisy frame x_t independently
        self.xt_frame_encoder = nn.Sequential(
            nn.Linear(input_feats, num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )
        
        # === Output Positional Embeddings ===
        # Distinguish different time steps in the output sequence
        self.output_pos_embed = nn.Embedding(t_future, num_latent_dims)
        
        # === Frame Decoder ===
        # Decode: context + x_t[t] + pos[t] -> predicted frame
        decoder_input_dim = num_latent_dims * 3
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, input_feats),
        )
        
        print(f"✓ SignWritingToPoseDiffusionV2 initialized")
        print(f"  - Frame-independent decoding (prevents Transformer averaging)")
        print(f"  - t_past={t_past}, t_future={t_future}")

    def forward(self,
                x: torch.Tensor,
                timesteps: torch.Tensor,
                past_motion: torch.Tensor,
                signwriting_im_batch: torch.Tensor):
        """
        Forward pass for diffusion model.
        
        Args:
            x: Noisy motion [B, J, C, T_future] in BJCT format
            timesteps: Diffusion timestep [B]
            past_motion: Historical frames [B, J, C, T_past] in BJCT format
            signwriting_im_batch: Condition images [B, 3, H, W]
            
        Returns:
            Predicted x0 (denoised motion) [B, J, C, T_future]
        """
        B, J, C, T_future = x.shape
        device = x.device
        
        # Debug logging on first forward pass
        debug = self._forward_count == 0
        
        # === Handle past_motion format ===
        # Support both BJCT and BTJC input formats
        if past_motion.dim() == 4:
            if past_motion.shape[1] == J and past_motion.shape[2] == C:
                # BJCT -> BTJC conversion
                past_btjc = past_motion.permute(0, 3, 1, 2).contiguous()
            else:
                # Already in BTJC format
                past_btjc = past_motion
        
        # === Encode Conditions ===
        # Past motion context
        past_ctx = self.past_context_encoder(past_btjc)  # [B, D]
        
        # SignWriting symbol embedding
        sign_emb = self.embed_signwriting(signwriting_im_batch)  # [B, D]
        
        # Diffusion timestep embedding
        time_emb = self.time_embed(timesteps.clamp(0, 999))  # [B, D]
        
        # Fuse all conditions into single context vector
        context = past_ctx + sign_emb + time_emb  # [B, D]
        
        # === Frame-Independent Decoding ===
        # Decode each future frame independently to preserve motion dynamics
        outputs = []
        for t in range(T_future):
            # Extract frame t from noisy input: [B, J, C] -> [B, J*C]
            xt_frame = x[:, :, :, t].reshape(B, -1)
            xt_emb = self.xt_frame_encoder(xt_frame)  # [B, D]
            
            # Add positional embedding for temporal information
            pos_idx = torch.tensor([t], device=device)
            pos_emb = self.output_pos_embed(pos_idx).expand(B, -1)  # [B, D]
            
            # Concatenate and decode
            dec_input = torch.cat([context, xt_emb, pos_emb], dim=-1)  # [B, D*3]
            out = self.decoder(dec_input)  # [B, J*C]
            outputs.append(out)
        
        # Stack outputs: [T, B, J*C] -> [B, J, C, T]
        result = torch.stack(outputs, dim=0)  # [T, B, J*C]
        result = result.permute(1, 0, 2)  # [B, T, J*C]
        result = result.reshape(B, T_future, J, C)  # [B, T, J, C]
        result = result.permute(0, 2, 3, 1).contiguous()  # [B, J, C, T]
        
        # Debug: Check if model produces motion (not static poses)
        if debug:
            disp = (result[:, :, :, 1:] - result[:, :, :, :-1]).abs().mean().item()
            print(f"[FORWARD] result shape={result.shape}, disp={disp:.6f}")
        
        self._forward_count += 1
        return result

    def interface(self, x, timesteps, y):
        """
        Diffusion training interface (compatible with CAMDM framework).
        
        Args:
            x: Noisy motion [B, J, C, T]
            timesteps: Diffusion timesteps [B]
            y: Dictionary with 'sign_image' and 'input_pose'
            
        Returns:
            Predicted x0 [B, J, C, T]
        """
        batch_size = x.shape[0]
        signwriting_image = y['sign_image']
        past_motion = y['input_pose']

        # Classifier-free guidance: randomly drop conditions during training
        if self.cond_mask_prob > 0:
            keep_batch_idx = torch.rand(batch_size, device=past_motion.device) < (1 - self.cond_mask_prob)
            past_motion = past_motion * keep_batch_idx.view((batch_size, 1, 1, 1))

        return self.forward(x, timesteps, past_motion, signwriting_image)


# Alias for backward compatibility
SignWritingToPoseDiffusion = SignWritingToPoseDiffusionV2