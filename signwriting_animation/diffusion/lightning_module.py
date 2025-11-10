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

    def __init__(self, num_keypoints=586, num_dims=3, lr=1e-3, log_dir="logs",
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
        """
        Normalize tensor [B,T,J,C] per frame to preserve motion.
        Previously used global mean/std, which removed temporal variation.
        """
        mean = x_btjc.mean(dim=(2, 3), keepdim=True)
        std = x_btjc.std(dim=(2, 3), keepdim=True)
        return (x_btjc - mean) / (std + 1e-6)


    def unnormalize_pose(self, x_btjc):
        x_btjc = x_btjc * self.std_pose + self.mean_pose
        try:
            from pose_anonymization.data.normalization import unshift_hands
            from pose_format.pose import Pose
            dummy = Pose(header=None, body=None)
            dummy.body.data = x_btjc[0].detach().cpu().numpy()
            unshift_hands(dummy)
            x_btjc[0] = torch.tensor(dummy.body.data, device=x_btjc.device)
        except Exception as e:
            print(f"[WARN] unshift_hands failed: {e}")
        return x_btjc


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
        fut  = self.normalize_pose(sanitize_btjc(batch["data"]))
        past = self.normalize_pose(sanitize_btjc(cond["input_pose"]))
        mask = (cond["target_mask"].float().sum(dim=(2, 3)) > 0).float() \
            if cond["target_mask"].dim() == 4 else cond["target_mask"].float()
        sign = cond["sign_image"].float()

        B, T = fut.size(0), fut.size(1)
        ts = torch.zeros(B, dtype=torch.long, device=fut.device)
        t_ramp = torch.linspace(0, 1, steps=T, device=fut.device).view(1, T, 1, 1)

        # === temporal-dependent noise ===
        temporal_noise = torch.randn_like(fut[:, 1:] - fut[:, :-1])
        temporal_noise = torch.cat([temporal_noise[:, :1], temporal_noise], dim=1)
        noise = 0.3 * torch.randn_like(fut) + 0.3 * temporal_noise
        if fut.size(2) > 150:
            hand_face_mask = torch.ones_like(fut)
            hand_face_mask[:, :, 130:, :] = 0.5
            noise = noise * hand_face_mask

        past = past[:, -T:, :, :]
        in_seq = 0.8 * noise + 0.6 * t_ramp + 0.05 * past + 0.05 * fut

        pred = self.forward(in_seq, ts, past, sign)
        loss_pos = masked_mse(pred, fut, mask)

        if T > 1:
            vel_mask = mask[:, 1:]
            vel_pred = pred[:, 1:] - pred[:, :-1]
            vel_gt   = fut[:, 1:] - fut[:, :-1]
            loss_vel = masked_mse(vel_pred, vel_gt, vel_mask)

            # === cosine motion direction loss ===
            cos_sim = torch.nn.functional.cosine_similarity(
                vel_pred.flatten(2), vel_gt.flatten(2), dim=2
            ).mean()
            motion_loss = (1 - cos_sim)  # encourage same direction
            loss = loss_pos + 0.8 * loss_vel + 0.2 * motion_loss
        else:
            loss_vel = torch.tensor(0.0, device=fut.device)
            motion_loss = torch.tensor(0.0, device=fut.device)
            loss = loss_pos

        gt_std = fut[..., :2].std()
        pred_std = pred[..., :2].std()
        scale_loss = ((pred_std / (gt_std + 1e-6) - 1.0) ** 2)
        loss = loss + 0.05 * scale_loss

        def torso_center(btjc):
            torso_end = min(33, btjc.size(2))
            return btjc[..., :torso_end, :2].mean(dim=(1, 2))

        c_gt = torso_center(fut)
        c_pr = torso_center(pred)
        center_loss = ((c_pr - c_gt) ** 2).mean()
        loss = loss + 0.02 * center_loss

        if pred.size(2) >= 175:
            right = pred[:, :, 154:175, :2].mean(dim=(1, 2))
            left  = pred[:, :, 133:154, :2].mean(dim=(1, 2))
            hand_sep = ((right - left).pow(2).sum(dim=-1) + 1e-6).sqrt().mean()
            loss = loss + 0.005 * (1.0 / hand_sep)

        self.log_dict({
            "train/loss": loss,
            "train/loss_pos": loss_pos,
            "train/loss_vel": loss_vel,
            "train/motion_loss": motion_loss,
            "train/scale_loss": scale_loss,
            "train/center_loss": center_loss,
        }, prog_bar=True, on_step=True)

        self.train_losses.append(loss.item())
        return loss


    def validation_step(self, batch, _):
        cond = batch["conditions"]
        fut  = self.normalize_pose(sanitize_btjc(batch["data"]))
        past = self.normalize_pose(sanitize_btjc(cond["input_pose"]))
        mask = (cond["target_mask"].float().sum(dim=(2,3)) > 0).float() \
            if cond["target_mask"].dim() == 4 else cond["target_mask"].float()
        sign = cond["sign_image"].float()
        ts   = torch.zeros(fut.size(0), dtype=torch.long, device=fut.device)

        T = fut.size(1)
        t_ramp = torch.linspace(0, 1, steps=T, device=fut.device).view(1, T, 1, 1)

        # === temporal noise same as training ===
        temporal_noise = torch.randn_like(fut[:, 1:] - fut[:, :-1])
        temporal_noise = torch.cat([temporal_noise[:, :1], temporal_noise], dim=1)
        noise = 0.3 * torch.randn_like(fut) + 0.3 * temporal_noise
        if fut.size(2) > 150:
            hand_face_mask = torch.ones_like(fut)
            hand_face_mask[:, :, 130:, :] = 0.5
            noise = noise * hand_face_mask

        past = past[:, -T:, :, :]
        in_seq = 0.8 * noise + 0.6 * t_ramp + 0.05 * past + 0.05 * fut
        print("[VAL DEBUG] pred frame-wise std (before forward):", fut.std(dim=(0,2,3)).detach().cpu().numpy())

        pred = self.forward(in_seq, ts, past, sign)
        loss_pos = masked_mse(pred, fut, mask)

        if T > 1:
            vel_mask = mask[:, 1:]
            vel_pred = pred[:, 1:] - pred[:, :-1]
            vel_gt   = fut[:, 1:] - fut[:, :-1]
            loss_vel = masked_mse(vel_pred, vel_gt, vel_mask)
            cos_sim = torch.nn.functional.cosine_similarity(
                vel_pred.flatten(2), vel_gt.flatten(2), dim=2
            ).mean()
            motion_loss = (1 - cos_sim)
            loss = loss_pos + 0.8 * loss_vel + 0.2 * motion_loss
        else:
            loss_vel = torch.tensor(0.0, device=fut.device)
            motion_loss = torch.tensor(0.0, device=fut.device)
            loss = loss_pos

        def torso_center(btjc):
            torso_end = min(33, btjc.size(2))
            return btjc[..., :torso_end, :2].mean(dim=(1, 2))

        c_gt = torso_center(fut)
        c_pr = torso_center(pred)
        center_loss = ((c_pr - c_gt) ** 2).mean()
        loss = loss + 0.02 * center_loss

        dtw = masked_dtw(self.unnormalize_pose(pred), self.unnormalize_pose(fut), mask)

        self.val_losses.append(loss.item())
        self.val_dtws.append(dtw.item())

        self.log_dict({
            "val/loss": loss,
            "val/loss_pos": loss_pos,
            "val/vel_loss": loss_vel,
            "val/motion_loss": motion_loss,
            "val/center_loss": center_loss,
            "val/dtw": dtw
        }, prog_bar=True)

        motion_magnitude = (pred[:, 1:] - pred[:, :-1]).abs().mean().item()
        msg = f"[DEBUG MOTION] avg frame delta: {motion_magnitude:.6f}"
        self.print(msg)
        self.log("val/motion_delta", motion_magnitude, prog_bar=True)

        pred_std_per_frame = pred.std(dim=(0,2,3)).detach().cpu().numpy()
        fut_std_per_frame  = fut.std(dim=(0,2,3)).detach().cpu().numpy()
        print("[VAL DEBUG] pred frame-wise std:", pred_std_per_frame)
        print("[VAL DEBUG] fut  frame-wise std:", fut_std_per_frame)

        return {"val_loss": loss, "val_dtw": dtw}


    @torch.no_grad()
    def generate_full_sequence(self, past_btjc, sign_img, target_mask=None, target_len=None):
        print("[GEN/full] ENTER generate_full_sequence", flush=True)
        self.eval()
        ctx  = self.normalize_pose(sanitize_btjc(past_btjc)).to(self.device)  # 🟩 normalize context
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