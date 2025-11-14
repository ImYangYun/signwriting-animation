import torch
import lightning as pl
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure as PE_DTW


# -----------------------------------------------------
# Utilities
# -----------------------------------------------------
def sanitize_btjc(x):
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    if x.dim() == 5:  # [B,T,P,J,C]
        x = x[:, :, 0]
    return x.float().contiguous()


def masked_mse(pred, gt, mask_bt):
    B,T,J,C = pred.shape
    m4 = mask_bt[:, :, None, None].float()
    return ((pred - gt)**2 * m4).sum() / (m4.sum() * J * C + 1e-6)


@torch.no_grad()
def masked_dtw(pred_btjc, tgt_btjc, mask_bt):
    metric = PE_DTW()
    B,T,J,C = pred_btjc.shape
    vals = []

    for b in range(B):
        t = int(mask_bt[b].sum().item())
        if t < 2:
            continue

        p = pred_btjc[b,:t].detach().cpu().numpy().astype("float32")
        g = tgt_btjc[b,:t].detach().cpu().numpy().astype("float32")

        # DTW expects shape (T,1,J,C)
        p = p[:,None,:,:]
        g = g[:,None,:,:]
        vals.append(metric.get_distance(p, g))

    if not vals:
        return torch.tensor(0.0, device=pred_btjc.device)
    return torch.tensor(vals, device=pred_btjc.device).mean()


class LitMinimal(pl.LightningModule):

    def __init__(self,
                 num_keypoints=178,
                 num_dims=3,
                 lr=1e-4,
                 stats_path="/data/yayun/pose_data/mean_std_178.pt"):
        super().__init__()
        self.save_hyperparameters()

        # ---- Load dataset-wide statistics ----
        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1,1,-1,3)   # [1,1,J,C]
        std  = stats["std"].float().view(1,1,-1,3)

        self.register_buffer("mean_pose", mean)
        self.register_buffer("std_pose",  std)

        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
        )

        print("[LitMinimal] CAMDM deterministic x0-prediction loaded ✔")

    # normalization helpers
    def normalize(self, x):
        return (x - self.mean_pose) / (self.std_pose + 1e-6)

    def unnormalize(self, x):
        return x * self.std_pose + self.mean_pose

    def forward(self, x_btjc, ts, past_btjc, sign_img):
        """
        x_btjc:   [B,T,J,C]  →  model query frames
        past_btjc: [B,T,J,C] → past motion (teacher forcing)
        ts:       [B]        → diffusion timestep index
        sign_img: [B,3,224,224]
        """

        # ---- permute to BJCT for CAMDM core ----
        x_bjct    = x_btjc.permute(0, 2, 3, 1).contiguous()
        past_bjct = past_btjc.permute(0, 2, 3, 1).contiguous()

        # ---- run CAMDM core ----
        out_bjct = self.model.forward(
            x_bjct, 
            ts, 
            past_bjct,
            sign_img
        )  # [B,J,C,T]

        # ---- back to BTJC ----
        pred_btjc = out_bjct.permute(0, 3, 1, 2).contiguous()

        # ======================================
        # DEBUG (only during validation/inference)
        # ======================================
        if not self.training:
            try:
                # 1) check NaN/Inf
                if torch.isnan(pred_btjc).any() or torch.isinf(pred_btjc).any():
                    print("[DBG] pred contains NaN/Inf !!!", flush=True)

                # 2) output stats
                std = pred_btjc.float().std().item()
                mean = pred_btjc.float().mean().item()
                print(f"[DBG] pred std={std:.6f}, mean={mean:.6f}", flush=True)

                # 3) motion check
                if pred_btjc.size(1) > 1:
                    vel = pred_btjc[:,1:] - pred_btjc[:,:-1]
                    vel_mean = vel.abs().mean().item()
                    print(f"[DBG] velocity mean={vel_mean:.6f}", flush=True)
                else:
                    print("[DBG] velocity skipped (T=1)", flush=True)

            except Exception as e:
                print("[DBG ERROR]", e, flush=True)

        return pred_btjc


    # TRAINING STEP（MSE only, for stable overfit）
    def training_step(self, batch, _):
        cond = batch["conditions"]

        gt   = self.normalize(sanitize_btjc(batch["data"]))
        past = self.normalize(sanitize_btjc(cond["input_pose"]))
        sign = cond["sign_image"].float()

        # mask
        if cond["target_mask"].dim() == 4:
            mask = (cond["target_mask"].sum((2,3)) > 0).float()
        else:
            mask = cond["target_mask"].float()

        B,T,J,C = gt.shape
        past = past[:, -T:]
        ts = torch.zeros(B, dtype=torch.long, device=self.device)

        pred = self.forward(past, ts, past, sign)

        with torch.no_grad():
            fut = self.model.future_motion_process(
                past.permute(0,2,3,1)  # BJCT
            )  # [T,B,D]
            fut_std = fut.float().std(dim=0).mean().item()
            self.log("debug/future_emb_time_std", fut_std, prog_bar=False)

            enc = self.model.seqEncoder(
                self.model.sequence_pos_encoder(fut)
            )  # [T,B,D]
            enc_std = enc.float().std(dim=0).mean().item()
            self.log("debug/encoder_out_time_std", enc_std, prog_bar=False)
        # ======================================================

        loss = masked_mse(pred, gt, mask)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    # -----------------------------------------------------
    def validation_step(self, batch, _):
        cond = batch["conditions"]

        gt   = self.normalize(sanitize_btjc(batch["data"]))
        past = self.normalize(sanitize_btjc(cond["input_pose"]))
        sign = cond["sign_image"].float()

        # mask
        if cond["target_mask"].dim() == 4:
            mask = (cond["target_mask"].sum((2,3)) > 0).float()
        else:
            mask = cond["target_mask"].float()

        B,T,J,C = gt.shape
        past = past[:, -T:]
        ts = torch.zeros(B, dtype=torch.long, device=self.device)

        pred = self.forward(past, ts, past, sign)

        with torch.no_grad():
            fut = self.model.future_motion_process(
                past.permute(0,2,3,1)
            )  
            fut_std = fut.float().std(dim=0).mean().item()
            self.log("debug_val/future_emb_time_std", fut_std, prog_bar=False)

            enc = self.model.seqEncoder(
                self.model.sequence_pos_encoder(fut)
            )
            enc_std = enc.float().std(dim=0).mean().item()
            self.log("debug_val/encoder_out_time_std", enc_std, prog_bar=False)

        loss = masked_mse(pred, gt, mask)

        dtw_val = masked_dtw(self.unnormalize(pred),
                            self.unnormalize(gt),
                            mask)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/dtw", dtw_val, prog_bar=True)

        return loss

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
