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
    """
    Minimal Lightning module with global mean/std normalization.
    Uses precomputed global statistics for consistent scale across samples.
    """

    def __init__(self, num_keypoints=586, num_dims=3, lr=1e-4, log_dir="logs",
                stats_path="/data/yayun/pose_data/mean_std_178.pt",
                data_dir="/data/yayun/pose_data",
                csv_path="/data/yayun/signwriting-animation/data_fixed.csv"):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.log_dir = log_dir
        self.train_losses, self.val_losses, self.val_dtws = [], [], []

        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu")
            mean = stats["mean"].float()
            std  = stats["std"].float()
            print(f"[Loaded mean/std] mean={mean.mean():.4f}, std={std.mean():.4f}")
        else:
            print(f"[WARN] stats not found → computing global mean/std from {csv_path}")
            from pose_format import Pose
            import pandas as pd, random

            df = pd.read_csv(csv_path)
            df = df[df["split"] == "train"].reset_index(drop=True)
            records = df.to_dict(orient="records")
            random.shuffle(records)
            sample_size = min(500, len(records))
            records = records[:sample_size]
            print(f"[INFO] Using {sample_size} samples to estimate mean/std")

            sum_all = torch.zeros(3)
            sum_sq_all = torch.zeros(3)
            count = 0

            for i, rec in enumerate(records):
                pose_path = os.path.join(data_dir, rec["pose"])
                if not os.path.exists(pose_path):
                    continue
                try:
                    with open(pose_path, "rb") as f:
                        p = Pose.read(f)
                    arr = torch.tensor(p.body.data, dtype=torch.float32)  # [T,P,J,C]
                    arr = arr.view(-1, arr.shape[-1])                     # [T*P*J, C]
                    sum_all += arr.sum(dim=0)
                    sum_sq_all += (arr ** 2).sum(dim=0)
                    count += arr.shape[0]
                except Exception as e:
                    print(f"[WARN] Failed on {pose_path}: {e}")
                    continue

                if (i + 1) % 50 == 0 or (i + 1) == sample_size:
                    print(f"[INFO] Processed {i+1}/{sample_size} files...")

            mean = sum_all / count
            std  = torch.sqrt(sum_sq_all / count - mean ** 2).clamp_min(1e-6)
            torch.save({"mean": mean, "std": std}, stats_path)
            print(f"[Saved new mean/std] → {stats_path}")

        self.register_buffer("mean_pose", mean)
        self.register_buffer("std_pose", std)
        print(f"[LitMinimal] Using mean={mean.tolist()} std={std.tolist()}")

        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
            mean=self.mean_pose,
            std=self.std_pose
        )

    def normalize_pose(self, x_btjc):
        x = sanitize_btjc(x_btjc)
        mean = self.mean_pose.view(1, 1, 1, -1)
        std  = self.std_pose.view(1, 1, 1, -1)
        return (x - mean) / (std + 1e-6)

    def unnormalize_pose(self, x_btjc):
        x = sanitize_btjc(x_btjc)
        mean = self.mean_pose.view(1, 1, 1, -1)
        std  = self.std_pose.view(1, 1, 1, -1)
        x = x * std + mean

        try:
            from pose_anonymization.data.normalization import unshift_hands
            from pose_format.pose import Pose
            dummy = Pose(header=None, body=None)
            dummy.body.data = x[0].detach().cpu().numpy()
            unshift_hands(dummy)
            x[0] = torch.tensor(dummy.body.data, device=x.device)
        except Exception as e:
            print(f"[WARN] unshift_hands failed: {e}")

        return x


    def forward(self, x_btjc, timesteps, past_btjc, sign_img):
        x_bjct    = btjc_to_bjct(sanitize_btjc(x_btjc))
        past_bjct = btjc_to_bjct(sanitize_btjc(past_btjc))
        if timesteps.dtype != torch.long:
            timesteps = timesteps.long()
        out_bjct  = self.model.forward(x_bjct, timesteps, past_bjct, sign_img)
        pred_btjc = bjct_to_btjc(out_bjct)
        return pred_btjc

    def training_step(self, batch, _):
        cond = batch["conditions"]
        fut  = self.normalize_pose(batch["data"])
        past = self.normalize_pose(cond["input_pose"])
        mask = (cond["target_mask"].float().sum(dim=(2, 3)) > 0).float() \
            if cond["target_mask"].dim() == 4 else cond["target_mask"].float()
        sign = cond["sign_image"].float()

        B, T = fut.size(0), fut.size(1)
        ts = torch.zeros(B, dtype=torch.long, device=fut.device)
        t_ramp = torch.linspace(0, 1, steps=T, device=fut.device).view(1, T, 1, 1)

        velocity = fut[:, 1:] - fut[:, :-1]
        temporal_noise = torch.cat([velocity, velocity[:, -1:]], dim=1)
        temporal_noise = 0.6 * velocity.mean(dim=1, keepdim=True) + 0.4 * temporal_noise.mean(dim=1, keepdim=True)
        noise = 0.1 * torch.randn_like(fut) + 0.1 * temporal_noise
        if fut.size(2) > 150:
            hand_face_mask = torch.ones_like(fut)
            hand_face_mask[:, :, 130:, :] = 1.2
            noise = noise * hand_face_mask

        past = past[:, -T:, :, :]
        in_seq = 0.05 * noise + 0.05 * t_ramp + 0.4 * past + 0.5 * fut

        pred = self.forward(in_seq, ts, past, sign)

        B, T, J, C = pred.shape
        w = torch.ones(B, T, J, 1, device=pred.device)
        w[:, :, :33, :] = 0.4      # face
        w[:, :, 133:154, :] = 1.8  # left hand
        w[:, :, 154:175, :] = 1.8 # right hand
        w[:, :, 175:, :] = 0.8

        mask_4d = mask[:, :, None, None].float()
        loss_pos = ((pred - fut)**2 * w * mask_4d).mean()

        vel_pred = pred[:, 1:] - pred[:, :-1]
        vel_gt   = fut[:, 1:] - fut[:, :-1]
        vel_mask = mask[:, 1:]
        loss_vel = masked_mse(vel_pred, vel_gt, vel_mask)

        acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
        acc_gt   = vel_gt[:, 1:] - vel_gt[:, :-1]
        loss_acc = masked_mse(acc_pred, acc_gt, vel_mask[:, 1:])

        loss = 1.0 * loss_pos + 0.1 * loss_vel + 0.05 * loss_acc

        self.log("train/loss", loss, prog_bar=True)
        return loss


    def validation_step(self, batch, _):
        cond = batch["conditions"]
        fut  = self.normalize_pose(batch["data"])
        past = self.normalize_pose(cond["input_pose"])
        mask = (cond["target_mask"].float().sum(dim=(2,3)) > 0).float() \
            if cond["target_mask"].dim() == 4 else cond["target_mask"].float()
        sign = cond["sign_image"].float()
        ts   = torch.zeros(fut.size(0), dtype=torch.long, device=fut.device)
        T = fut.size(1)
        t_ramp = torch.linspace(0, 1, steps=T, device=fut.device).view(1, T, 1, 1)

        velocity = fut[:, 1:] - fut[:, :-1]
        temporal_noise = torch.cat([velocity, velocity[:, -1:]], dim=1)
        temporal_noise = 0.6 * velocity.mean(dim=1, keepdim=True) + 0.4 * temporal_noise.mean(dim=1, keepdim=True)
        noise = 0.1 * torch.randn_like(fut) + 0.1 * temporal_noise
        if fut.size(2) > 150:
            hand_face_mask = torch.ones_like(fut)
            hand_face_mask[:, :, 130:, :] = 1.2
            hand_face_mask[:, :, :33, :] = 1.3
            noise = noise * hand_face_mask
        past = past[:, -T:, :, :]
        in_seq = 0.1 * noise + 0.1 * t_ramp + 0.3 * past + 0.5 * fut

        pred = self.forward(in_seq, ts, past, sign)
        B, T, J, C = pred.shape
        w = torch.ones(B, T, J, 1, device=pred.device)
        w[:, :, :33, :] = 0.4
        w[:, :, 133:154, :] = 1.3
        w[:, :, 154:175, :] = 1.3
        w[:, :, 175:, :] = 0.8

        mask_4d = mask[:, :, None, None].float()
        loss_pos = ((pred - fut)**2 * w * mask_4d).mean()

        vel_pred = pred[:, 1:] - pred[:, :-1]
        vel_gt   = fut[:, 1:] - fut[:, :-1]
        vel_mask = mask[:, 1:]
        loss_vel = masked_mse(vel_pred, vel_gt, vel_mask)

        acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
        acc_gt   = vel_gt[:, 1:] - vel_gt[:, :-1]
        loss_acc = masked_mse(acc_pred, acc_gt, vel_mask[:, 1:])

        loss = 1.0 * loss_pos + 0.3 * loss_vel + 0.1 * loss_acc

        # === DTW metrics ===
        dtw_norm  = masked_dtw(pred, fut, mask)
        dtw_unorm = masked_dtw(self.unnormalize_pose(pred), self.unnormalize_pose(fut), mask)
        dtw_face  = masked_dtw(pred[:, :, :33, :], fut[:, :, :33, :], mask)
        dtw_handR = masked_dtw(pred[:, :, 154:175, :], fut[:, :, 154:175, :], mask)

        self.log("val/dtw_norm", dtw_norm, prog_bar=False)
        self.log("val/dtw_unorm", dtw_unorm, prog_bar=True)
        self.log("val/dtw_face", dtw_face, prog_bar=False)
        self.log("val/dtw_handR", dtw_handR, prog_bar=False)
        self.log("val/loss", loss, prog_bar=True)

        return {"val_loss": loss, "val_dtw": dtw_unorm}


    @torch.no_grad()
    def generate_full_sequence(self, past_btjc, sign_img, target_mask=None, target_len=None):
        print("[GEN/full] ENTER generate_full_sequence", flush=True)
        self.eval()
        ctx  = self.normalize_pose(past_btjc).to(self.device)
        sign = sign_img.to(self.device)
        B, _, J, C = ctx.shape

        if target_len is not None:
            if isinstance(target_len, (int, float)):
                tf_list = [int(target_len)] * B
            elif torch.is_tensor(target_len):
                tf_list = target_len.view(-1).to(torch.long).cpu().tolist()
            else:
                tf_list = [int(x) for x in target_len]
        else:
            assert target_mask is not None, "Need target_len or target_mask"
            mask_bt = (target_mask.float().sum(dim=(2,3)) > 0).float() if target_mask.dim() == 4 else target_mask.float()
            tf_list = mask_bt.sum(dim=1).to(torch.long).view(-1).cpu().tolist()

        outs = []
        for b in range(B):
            Tf = max(1, int(tf_list[b]))
            preds = []
            for t_idx in range(Tf):
                t_norm = torch.tensor([[t_idx / Tf]], device=self.device)
                x_query = 0.05 * torch.randn((1, 1, J, C), device=self.device)
                pred_t = self.forward(x_query, t_norm, ctx[b:b+1], sign[b:b+1])
                preds.append(pred_t)
            pred_seq = torch.cat(preds, dim=1)
            outs.append(pred_seq)
        out = torch.cat(outs, dim=0)
        return self.unnormalize_pose(out)  # 🟩 restore to real coordinates

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)