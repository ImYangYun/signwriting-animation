"""
SignWriting-to-Pose Diffusion Model V1 (Original CAMDM-based version)

This is the original model architecture that uses all CAMDM components.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess, seq_encoder_factory


class DistributionPredictionModel(nn.Module):
    """
    Predicts a probability distribution (Gaussian) over sequence length.
    
    Used in the original model to predict future sequence length,
    though this may not be needed if sequence length is fixed.
    
    Args:
        input_size: Input feature dimension
    """
    def __init__(self, input_size: int):
        super().__init__()
        self.fc_mu = nn.Linear(input_size, 1)
        self.fc_var = nn.Linear(input_size, 1)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor [B, D]
            
        Returns:
            q: Normal distribution with predicted mean and std
        """
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        std = torch.exp(0.5 * log_var)
        q = torch.distributions.Normal(mu, std)
        return q


class EmbedSignWriting(nn.Module):
    """
    SignWriting image encoder using CLIP.
    """
    def __init__(self, num_latent_dims: int, embedding_arch: str = 'openai/clip-vit-base-patch32'):
        super().__init__()
        self.model = CLIPModel.from_pretrained(embedding_arch)
        self.proj = None

        if (num_embedding_dims := self.model.visual_projection.out_features) != num_latent_dims:
            self.proj = nn.Linear(num_embedding_dims, num_latent_dims)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_batch: [batch_size, 3, 224, 224]
        Returns:
            embeddings_batch: [1, batch_size, num_latent_dims]
        """
        embeddings_batch = self.model.get_image_features(pixel_values=image_batch)

        if self.proj is not None:
            embeddings_batch = self.proj(embeddings_batch)

        return embeddings_batch[None, ...]


class OutputProcessMLP(nn.Module):
    """
    Output process for the Sign Language Pose Diffusion model: project to pose space.
    
    Obtained module from https://github.com/sign-language-processing/fluent-pose-synthesis
    """
    def __init__(self,
                 num_latent_dims: int,
                 num_keypoints: int,
                 num_dims_per_keypoint: int,
                 hidden_dim: int = 512):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(num_latent_dims, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_keypoints * num_dims_per_keypoint)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes a sequence of latent vectors into keypoint motion data using a multi-layer perceptron (MLP).

        Args:
            x (Tensor):
                Input latent tensor.
                Shape: [num_frames, batch_size, num_latent_dims].

        Returns:
            Tensor:
                Decoded keypoint motion.
                Shape: [batch_size, num_keypoints, num_dims_per_keypoint, num_frames].
        """
        num_frames, batch_size, num_latent_dims = x.shape
        x = self.mlp(x)  # use MLP instead of single linear layer
        x = x.reshape(num_frames, batch_size, self.num_keypoints, self.num_dims_per_keypoint)
        x = x.permute(1, 2, 3, 0)
        return x


class SignWritingToPoseDiffusionV1(nn.Module):
    """
    Original SignWriting-to-Pose Diffusion Model (V1).
    
    This version uses all CAMDM components:
    - MotionProcess for motion encoding
    - seq_encoder_factory for flexible encoder architecture
    - PositionalEncoding for temporal information
    - TimestepEmbedder for timestep conditioning
    - DistributionPredictionModel for length prediction
    
    Args:
        num_keypoints: Number of pose keypoints
        num_dims_per_keypoint: Dimensions per keypoint (typically 3 for x,y,z)
        embedding_arch: CLIP model architecture
        num_latent_dims: Latent dimension
        ff_size: Feed-forward size in Transformer
        num_layers: Number of Transformer layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        activation: Activation function
        arch: Encoder architecture ("trans_enc", "trans_dec", or "gru")
        cond_mask_prob: Probability of masking conditions (for CFG)
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
                 cond_mask_prob: float = 0):
        super().__init__()

        self.cond_mask_prob = cond_mask_prob
        self.arch = arch

        # Local conditions: process motion sequences
        input_feats = num_keypoints * num_dims_per_keypoint
        self.future_motion_process = MotionProcess(input_feats, num_latent_dims)
        self.past_motion_process = MotionProcess(input_feats, num_latent_dims)
        self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)

        # Global conditions: image and timestep
        self.embed_signwriting = EmbedSignWriting(num_latent_dims, embedding_arch)
        self.embed_timestep = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)

        # Sequence encoder (can be trans_enc, trans_dec, or gru)
        self.seqEncoder = seq_encoder_factory(arch=arch,
                                              latent_dim=num_latent_dims,
                                              ff_size=ff_size,
                                              num_layers=num_layers,
                                              num_heads=num_heads,
                                              dropout=dropout,
                                              activation=activation)

        # Output projection and length predictor
        self.pose_projection = OutputProcessMLP(num_latent_dims, num_keypoints, num_dims_per_keypoint)
        self.length_predictor = DistributionPredictionModel(num_latent_dims)
        self.global_norm = nn.LayerNorm(num_latent_dims)

        print(f"✓ SignWritingToPoseDiffusionV1 initialized")
        print(f"  - Architecture: {arch}")
        print(f"  - Using all CAMDM components")
        print(f"  - MotionProcess for past/future motion")
        print(f"  - {num_layers} layers, {num_heads} heads")

    def forward(self,
                x: torch.Tensor,
                timesteps: torch.Tensor,
                past_motion: torch.Tensor,
                signwriting_im_batch: torch.Tensor):
        """
        Forward pass for V1 model.

        Args:
            x: Noisy input tensor [B, J, C, T_future] (BJCT format)
            timesteps: Diffusion timesteps [B]
            past_motion: Historical frames [B, J, C, T_past] (BJCT format)
            signwriting_im_batch: Condition images [B, 3, H, W]

        Returns:
            output: Predicted denoised motion [B, J, C, T_future]
            length_dist: Predicted length distribution (may be None if not used)
        """
        batch_size, num_keypoints, num_dims_per_keypoint, num_frames = x.shape

        # Encode all conditions
        time_emb = self.embed_timestep(timesteps)  # [1, B, D]
        signwriting_emb = self.embed_signwriting(signwriting_im_batch)  # [1, B, D]
        
        # MotionProcess expects [B, J, C, T] format
        past_motion_emb = self.past_motion_process(past_motion)  # [T_past, B, D]
        future_motion_emb = self.future_motion_process(x)  # [T_future, B, D]

        # Concatenate all embeddings into one sequence
        # Shape: [1 + 1 + T_past + T_future, B, D]
        xseq = torch.cat((time_emb,
                          signwriting_emb,
                          past_motion_emb,
                          future_motion_emb), axis=0)

        # Add positional encoding
        xseq = self.sequence_pos_encoder(xseq)
        
        # Process through sequence encoder
        output = self.seqEncoder(xseq)[-num_frames:]  # Take last T_future frames
        
        # Project to pose space
        output = self.pose_projection(output)  # [B, J, C, T]
        
        # Predict sequence length (global context)
        global_latent = self.global_norm(xseq.mean(0))  # [B, D]
        length_dist = self.length_predictor(global_latent)
        
        return output, length_dist

    def interface(self, x, timesteps, y):
        """
        Interface for diffusion training (compatible with CAMDM).
        
        Args:
            x: Noisy motion [B, J, C, T]
            timesteps: Diffusion timesteps [B]
            y: Dictionary with 'sign_image' and 'input_pose'
            
        Returns:
            output: Predicted x0 [B, J, C, T]
            length_dist: Length distribution
        """
        batch_size = x.shape[0]
        signwriting_image = y['sign_image']
        past_motion = y['input_pose']

        # Classifier-free guidance: randomly drop conditions during training
        if self.cond_mask_prob > 0:
            keep_batch_idx = torch.rand(batch_size, device=past_motion.device) < (1 - self.cond_mask_prob)
            past_motion = past_motion * keep_batch_idx.view((batch_size, 1, 1, 1))

        return self.forward(x, timesteps, past_motion, signwriting_image)


# Alias
SignWritingToPoseDiffusion = SignWritingToPoseDiffusionV1