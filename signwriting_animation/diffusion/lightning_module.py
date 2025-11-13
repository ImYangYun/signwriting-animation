import os
import csv
import torch
import lightning as pl
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure as PE_DTW


def _to_dense(x):
    """
    Convert a potentially sparse or masked tensor to a dense, contiguous float32 tensor.
        - pose-format MaskedTensor (via .zero_filled())
        - sparse tensors (.to_dense())
        - dtype casting to float32
    Args:
        x (torch.Tensor or MaskedTensor): Input tensor of arbitrary type.
    Returns:
        torch.Tensor: Dense float32 tensor with contiguous memory layout.
    """
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    if getattr(x, "is_sparse", False):
        x = x.to_dense()
    if x.dtype != torch.float32:
        x = x.float()
    return x.contiguous()

def sanitize_btjc(x):
    """Ensure tensor is [B,T,J,C]. Handle sparse or [B,T,P,J,C] inputs."""
    x = _to_dense(x)
    if x.dim() == 5:  # [B,T,P,J,C]
        x = x[:, :, 0, ...]
    if x.dim() != 4:
        raise ValueError(f"Expected [B,T,J,C], got {tuple(x.shape)}")
    return x

def btjc_to_bjct(x):  # [B,T,J,C] -> [B,J,C,T]
    """Permute tensor from [B, T, J, C] → [B, J, C, T]."""
    return x.permute(0, 2, 3, 1).contiguous()

def bjct_to_btjc(x):  # [B,J,C,T] -> [B,T,J,C]
    """Permute tensor from [B, J, C, T] → [B, T, J, C]."""
    return x.permute(0, 3, 1, 2).contiguous()

def masked_mse(pred_btjc, tgt_btjc, mask_bt):
    """
    Compute mean squared error over valid (masked) frames.
    Args:
        pred_btjc (torch.Tensor): Predicted poses [B, T, J, C].
        tgt_btjc (torch.Tensor): Target poses [B, T, J, C].
        mask_bt (torch.Tensor): Binary mask [B, T] where 1 indicates valid frames.
    Returns:
        torch.Tensor: Scalar loss value (float).
    """
    pred = sanitize_btjc(pred_btjc)
    tgt  = sanitize_btjc(tgt_btjc)

    Tm = min(pred.size(1), tgt.size(1), mask_bt.size(1))
    pred = pred[:, :Tm]
    tgt  = tgt[:,  :Tm]
    m4 = mask_bt[:, :Tm].float()[:, :, None, None]   # [B,T,1,1]

    diff2 = (pred - tgt) ** 2                        # [B,T,J,C]
    num = (diff2 * m4).sum()
    den = (m4.sum() * pred.size(2) * pred.size(3)).clamp_min(1.0)
    return num / den

def _btjc_to_tjc_list(x_btjc, mask_bt):
    """
    Convert batched [B,T,J,C] tensor into list of variable-length [T,J,C] sequences.
    Uses mask to trim valid frames for each sample.
    Args:
        x_btjc (torch.Tensor): Pose tensor [B, T, J, C].
        mask_bt (torch.Tensor): Frame validity mask [B, T].
    Returns:
        list[torch.Tensor]: List of [T, J, C] tensors (one per batch sample).
    """
    x_btjc = sanitize_btjc(x_btjc)
    B, T, J, C = x_btjc.shape
    seqs = []
    mask_bt = (mask_bt > 0.5).float()
    for b in range(B):
        t = int(mask_bt[b].sum().item())
        t = max(0, min(t, T))
        seqs.append(x_btjc[b, :t].contiguous())
    return seqs

@torch.no_grad()
def masked_dtw(pred_btjc, tgt_btjc, mask_bt):
    preds = _btjc_to_tjc_list(pred_btjc, mask_bt)
    tgts  = _btjc_to_tjc_list(tgt_btjc,  mask_bt)
    dtw_metric = PE_DTW()

    vals = []
    for p, g in zip(preds, tgts):
        if p.size(0) < 2 or g.size(0) < 2:
            continue
        pv = p.detach().cpu().numpy().astype("float32")  # [T,J,C]
        gv = g.detach().cpu().numpy().astype("float32")  # [T,J,C]
        pv = pv[:, None, :, :]  # (T, 1, J, C)
        gv = gv[:, None, :, :]  # (T, 1, J, C)
        vals.append(float(dtw_metric.get_distance(pv, gv)))

    if not vals:
        return torch.tensor(0.0, device=pred_btjc.device)
    return torch.tensor(vals, device=pred_btjc.device).mean()


class LitMinimal(pl.LightningModule):

    def __init__(self, num_keypoints=178, num_dims=3, lr=1e-4, log_dir="logs",
                 stats_path="/data/yayun/pose_data/mean_std_178.pt",
                 data_dir="/data/yayun/pose_data",
                 csv_path="/data/yayun/signwriting-animation/data_fixed.csv"):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.log_dir = log_dir

        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float()   # [178,3]
        std  = stats["std"].float()    # [178,3]

        # reshape to broadcast with [B,T,J,C]
        mean = mean.view(1,1,-1,3)
        std  = std.view(1,1,-1,3)

        self.register_buffer("mean_pose", mean)
        self.register_buffer("std_pose", std)

        print(f"[Loaded 178 stats] mean={mean.mean():.4f}, std={std.mean():.4f}")

        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
            mean=self.mean_pose,
            std=self.std_pose,
        )

    def normalize_pose(self, x):
        """Global normalization"""
        x = sanitize_btjc(x)
        return (x - self.mean_pose) / (self.std_pose + 1e-6)

    def unnormalize_pose(self, x):
        """Global unnorm"""
        return x * self.std_pose + self.mean_pose

    def forward(self, x_btjc, timesteps, past_btjc, sign_img):
        x_bjct = btjc_to_bjct(sanitize_btjc(x_btjc))
        past_bjct = btjc_to_bjct(sanitize_btjc(past_btjc))
        out_bjct = self.model.forward(x_bjct, timesteps, past_bjct, sign_img)
        return bjct_to_btjc(out_bjct)

    def training_step(self, batch, _):
        fut  = self.normalize_pose(batch["data"])
        past = self.normalize_pose(batch["conditions"]["input_pose"])
        mask = batch["conditions"]["target_mask"].float()
        sign = batch["conditions"]["sign_image"].float()

        B, T = fut.size(0), fut.size(1)
        ts = torch.zeros(B, dtype=torch.long, device=fut.device)
        t_ramp = torch.linspace(0,1,T,device=fut.device).view(1,T,1,1)

        # noise injection
        velocity = fut[:,1:] - fut[:,:-1]
        temporal_noise = 0.6*velocity.mean(1,keepdim=True) + 0.4*torch.cat([velocity, velocity[:,-1:]],1).mean(1, keepdim=True)
        noise = 0.1*torch.randn_like(fut) + 0.1*temporal_noise

        in_seq = 0.05*noise + 0.05*t_ramp + 0.4*past[:, -T:] + 0.5*fut

        pred = self.forward(in_seq, ts, past[:, -T:], sign)

        # loss
        mask4 = mask[:,:,None,None]
        loss_pos = ((pred - fut)**2 * mask4).mean()

        vel_pred = pred[:,1:] - pred[:,:-1]
        vel_gt   = fut[:,1:] - fut[:,:-1]
        loss_vel = ((vel_pred - vel_gt)**2 * mask[:,1:,None,None]).mean()

        loss = 1.0*loss_pos + 0.1*loss_vel

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        fut  = self.normalize_pose(batch["data"])
        past = self.normalize_pose(batch["conditions"]["input_pose"])
        sign = batch["conditions"]["sign_image"].float()
        mask = batch["conditions"]["target_mask"].float()
        ts = torch.zeros(fut.size(0), dtype=torch.long, device=fut.device)

        T = fut.size(1)
        noise = 0.1*torch.randn_like(fut)
        in_seq = 0.1*noise + 0.1 + 0.3*past[:, -T:] + 0.5*fut

        pred = self.forward(in_seq, ts, past[:, -T:], sign)

        loss = ((pred - fut)**2 * mask[:,:,None,None]).mean()

        dtw_unorm = masked_dtw(self.unnormalize_pose(pred), self.unnormalize_pose(fut), mask)
        self.log("val/dtw_unorm", dtw_unorm, prog_bar=True)
        self.log("val/loss", loss, prog_bar=True)

        return {"val_loss": loss, "val_dtw": dtw_unorm}

    @torch.no_grad()
    def generate_full_sequence(self, past_btjc, sign_img, target_len=30):
        ctx  = self.normalize_pose(past_btjc)
        sign = sign_img.to(self.device)
        B,_,J,C = ctx.shape

        outs = []
        for b in range(B):
            preds = []
            for t in range(target_len):
                t_norm = torch.tensor([[t/target_len]], device=self.device)
                x_query = torch.zeros((1,1,J,C), device=self.device)
                pred_t = self.forward(x_query, t_norm, ctx[b:b+1], sign[b:b+1])
                preds.append(pred_t)
            outs.append(torch.cat(preds,1))

        return self.unnormalize_pose(torch.cat(outs,0))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
