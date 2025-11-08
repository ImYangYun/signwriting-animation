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
                 dropout: float = 0.05,
                 activation: str = "gelu",
                 arch: str = "trans_enc",
                 cond_mask_prob: float = 0,
                 mean: torch.Tensor = None,
                 std: torch.Tensor = None):
        """
        Generates pose sequences conditioned on SignWriting images and past motion using a diffusion model.

        Args:
            num_keypoints (int):
                Number of keypoints in the pose representation.

            num_dims_per_keypoint (int):
                Number of spatial dimensions per keypoint (e.g., 2 for 2D, 3 for 3D).

            embedding_arch (str):
                Architecture used for extracting image embeddings (e.g., CLIP variants).

            num_latent_dims (int):
                Dimensionality of the latent representation used by the model.

            ff_size (int):
                Size of the feed-forward network in the Transformer blocks.

            num_layers (int):
                Number of Transformer encoder/decoder layers.

            num_heads (int):
                Number of attention heads in each multi-head attention block.

            dropout (float):
                Dropout rate applied during training.

            activation (str):
                Activation function used in the Transformer (e.g., "gelu", "relu").

            arch (str):
                Architecture type used in the diffusion model. Options: "trans_enc", "trans_dec", or "gru".

            cond_mask_prob (float):
                Probability of masking conditional inputs for classifier-free guidance (CFG).
        """
        super().__init__()

        self.cond_mask_prob = cond_mask_prob

        if mean is not None and std is not None:
            self.mean = mean.detach().clone().float()
            self.std  = std.detach().clone().float()
            print(f"[DBG] Using dataset mean/std (mean≈{self.mean.mean():.2f}, std≈{self.std.mean():.2f})")
        else:
            self.mean = torch.tensor([502.02, 299.43, 230.21], dtype=torch.float32)
            self.std  = torch.tensor([203.70, 152.87,   0.13], dtype=torch.float32)
            print("[WARN] No dataset mean/std passed → using fallback stats "
                f"(mean≈{self.mean.mean():.2f}, std≈{self.std.mean():.2f})")

        if self.std.mean() > 200 and self.mean.mean() > 300:
            print("[WARN] mean/std look unusually large — double check you are not mixing FluentPose stats.")

        # local conditions
        input_feats = num_keypoints * num_dims_per_keypoint
        self.future_motion_process = MotionProcess(input_feats, num_latent_dims)
        self.past_motion_process = MotionProcess(input_feats, num_latent_dims)
        self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)

        # global conditions
        self.embed_signwriting = EmbedSignWriting(num_latent_dims, embedding_arch)
        self.embed_timestep = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)

        orig_forward = self.embed_timestep.forward

        def safe_forward(timesteps):
            try:
                return orig_forward(timesteps)
            except RuntimeError as e:
                if "permute" in str(e) or "sparse_coo" in str(e):
                    pe = self.sequence_pos_encoder.pe
                    if isinstance(pe, torch.Tensor) and pe.dim() == 4:
                        pe = pe.squeeze(0).squeeze(0)
                    sel = pe[timesteps]
                    if sel.dim() == 4:
                        sel = sel.squeeze(0).squeeze(0)
                    elif sel.dim() == 2:
                        sel = sel.unsqueeze(1)
                    out = self.embed_timestep.time_embed(sel)
                    if out.dim() == 2:
                        out = out.unsqueeze(0)  # ensure [1, B, D]
                    return out.permute(1, 0, 2)
                raise e

        self.embed_timestep.forward = safe_forward

        self.seqEncoder = seq_encoder_factory(arch=arch,
                                              latent_dim=num_latent_dims,
                                              ff_size=ff_size,
                                              num_layers=num_layers,
                                              num_heads=num_heads,
                                              dropout=dropout,
                                              activation=activation)

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
        Performs classifier-free guidance by running a forward pass of the diffusion model.
        """
        if x.dim() == 5 and x.shape[2] == 1:
            # [B, T, 1, J, C] → [B, T, J, C]
            x = x.squeeze(2)
        if x.dim() == 4 and x.shape[-1] < x.shape[-2]:
            # [B, J, C, T] → [B, T, J, C]
            x = x.permute(0, 3, 1, 2).contiguous()
            print(f"[DBG/reshape] Permuted x to [B, T, J, C]: {x.shape}")
        batch_size, num_keypoints, num_dims_per_keypoint, num_frames = x.shape

        def _stat(name, tensor):
            if tensor is None:
                return
            if torch.isnan(tensor).any():
                print(f"[NaN DETECTED] {name} has NaN")
            else:
                print(f"[DBG/model] {name} mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}, "
                    f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, shape={tuple(tensor.shape)}")
        # ======================= DEBUG PATCH END ==========================

        _stat("input_x", x)
        _stat("past_motion", past_motion)
        _stat("signwriting_im_batch", signwriting_im_batch)

        # ---- 1. Embeddings ----
        time_emb = self.embed_timestep(timesteps)                 # [1, B, D]
        signwriting_emb = self.embed_signwriting(signwriting_im_batch)  # [1, B, D]
        time_emb = time_emb.expand(-1, signwriting_emb.size(1), -1)

        _stat("time_emb", time_emb)
        _stat("signwriting_emb", signwriting_emb)

        # ---- 2. Local motion embeddings ----
        past_motion_emb = self.past_motion_process(past_motion)   # [Tp, B, D]
        future_motion_emb = self.future_motion_process(x)         # [Tf, B, D]
        _stat("past_motion_emb", past_motion_emb)
        _stat("future_motion_emb", future_motion_emb)

        # ---- 3. Add continuous time embedding to future ----
        Tf = future_motion_emb.size(0)
        B  = future_motion_emb.size(1)
        t  = torch.linspace(0, 1, steps=Tf, device=future_motion_emb.device).view(Tf, 1, 1)
        t_latent = self.future_time_proj(t)                       # [Tf,1,D]
        t_latent = t_latent.expand(-1, B, -1)                     # [Tf,B,D]
        future_motion_emb = future_motion_emb + 2.0 * t_latent
        future_motion_emb = self.future_after_time_ln(future_motion_emb)
        _stat("future_motion_emb + time", future_motion_emb)

        time_cond = time_emb.repeat(Tf, 1, 1)         # [Tf,B,D]
        sign_cond = signwriting_emb.repeat(Tf, 1, 1)  # [Tf,B,D]
        xseq = future_motion_emb + 1.0 * (time_cond + sign_cond)
        xseq = self.sequence_pos_encoder(xseq)
        _stat("xseq_after_posenc", xseq)

        output = self.seqEncoder(xseq)
        output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        output = torch.clamp(output, -10, 10)
        _stat("encoder_out", output)

        output = self.pose_projection(output)         # [B,J,C,T]
        _stat("pose_projection_output", output)

        with torch.no_grad():
            def _motion_diag(name, t):
                if t is None or t.dim() < 2:
                    print(f"[MOTION] {name}: skipped (shape={None if t is None else tuple(getattr(t, 'shape', []))})")
                    return
                t = torch.nan_to_num(t.float(), nan=0.0, posinf=0.0, neginf=0.0)
                if t.size(0) < 2:
                    print(f"[MOTION] {name}: skipped (T={t.size(0)})")
                    return
                diff = t[1:] - t[:-1]
                mag = diff.abs().mean().item()
                std = diff.std(unbiased=False).item()
                print(f"[MOTION] {name}: Δmean={mag:.6f}, Δstd={std:.6f}")

            _motion_diag("future_motion_emb", future_motion_emb)
            _motion_diag("xseq_after_posenc", xseq)
            _motion_diag("encoder_out", output.transpose(0, 1))

            if output.dim() == 4:
                pose_diff = output[..., 1:] - output[..., :-1]
                pose_mag = pose_diff.abs().mean().item()
                pose_std = pose_diff.std().item()
                print(f"[MOTION] pose_projection_output: Δmean={pose_mag:.6f}, Δstd={pose_std:.6f}")
 
        return output

    def interface(self,
                  x: torch.Tensor,
                  timesteps: torch.Tensor,
                  y: dict):
        """
        Performs classifier-free guidance by running a forward pass of the diffusion model
        in either conditional or unconditional mode. Extracts conditioning inputs from `y` and
        applies random masking to simulate unconditional sampling.

        Args:
            x (Tensor):
                The noisy input tensor at the current diffusion step, denoted as x_t in the CAMDM paper.
                Shape: [batch_size, num_past_frames, num_keypoints, num_dims_per_keypoint].

            timesteps (Tensor):
                Diffusion timesteps for each sample in the batch.
                Shape: [batch_size], dtype: int.

            y (dict):
                Dictionary of conditioning inputs. Must contain:
                    - 'sign_image': Tensor of shape [batch_size, 3, 224, 224]
                    - 'input_pose': Tensor of shape [batch_size, num_keypoints, num_dims_per_keypoint, num_past_frames]

        Returns:
            Tensor:
                The predicted denoised motion at the current timestep.
                Shape: [batch_size, num_past_frames, num_keypoints, num_dims_per_keypoint].
        """
        batch_size, num_past_frames, num_keypoints, num_dims_per_keypoint = x.shape

        signwriting_image = y['sign_image']
        past_motion = y['input_pose']

        # CFG on past motion
        keep_batch_idx = torch.rand(batch_size, device=past_motion.device) < (1 - self.cond_mask_prob)
        past_motion = past_motion * keep_batch_idx.view((batch_size, 1, 1, 1))

        return self.forward(x, timesteps, past_motion, signwriting_image)


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

        self.ln = nn.LayerNorm(num_latent_dims)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(num_latent_dims, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_keypoints * num_dims_per_keypoint)
        )
        with torch.no_grad():
            last = self.mlp[-1]
            nn.init.xavier_uniform_(last.weight, gain=0.01)
            nn.init.zeros_(last.bias)
        
        self.scale = 1.0

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
        x = self.ln(x)
        x = self.mlp(x)
        x = torch.tanh(x * self.scale) * 3.0
        x = x.reshape(num_frames, batch_size, self.num_keypoints, self.num_dims_per_keypoint)
        x_center = x.mean(dim=(2, 3), keepdim=True)
        x = x - x_center
        if not self.training:
            x = x + 1e-3 * torch.randn_like(x)

        x = x.permute(1, 2, 3, 0)
        return x


class EmbedSignWriting(nn.Module):
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
        # image_batch should be in the format [B, 3, H, W], where H=W=224.
        embeddings_batch = self.model.get_image_features(pixel_values=image_batch)

        if self.proj is not None:
            embeddings_batch = self.proj(embeddings_batch)

        return embeddings_batch[None, ...]
