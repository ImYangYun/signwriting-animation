import os
import csv
import torch
import lightning as pl

from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from pose_evaluation.metrics.dtw import dtw_mje as PE_DTW


def _to_dense(x):
    """
    Safely convert a batch tensor to dense float32:
    - If it's a pose-format MaskedTensor, call .zero_filled() to obtain a dense tensor.
    - If it's a sparse tensor, densify it.
    - Cast to float32 and make memory contiguous.
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
    return x.permute(0, 2, 3, 1).contiguous()

def bjct_to_btjc(x):  # [B,J,C,T] -> [B,T,J,C]
    return x.permute(0, 3, 1, 2).contiguous()

def masked_mse(pred_btjc, tgt_btjc, mask_bt):
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


# ---- 仅使用 pose-evaluation 的 DTW-MJE ----

def _btjc_to_tjc_list(x_btjc, mask_bt):
    """BTJC + [B,T] mask -> list[[t,J,C]]，按 mask 截去 padding。"""
    x_btjc = sanitize_btjc(x_btjc)
    B, T, J, C = x_btjc.shape
    seqs = []
    for b in range(B):
        t = int(mask_bt[b].sum().item())
        t = max(0, min(t, T))
        seqs.append(x_btjc[b, :t].contiguous())  # [t,J,C]
    return seqs

@torch.no_grad()
def masked_dtw(pred_btjc, tgt_btjc, mask_bt):
    preds = _btjc_to_tjc_list(pred_btjc, mask_bt)
    tgts  = _btjc_to_tjc_list(tgt_btjc,  mask_bt)
    vals = []
    for p, g in zip(preds, tgts):
        pv = p.detach().cpu().numpy().astype("float32")
        gv = g.detach().cpu().numpy().astype("float32")
        vals.append(float(PE_DTW(pv, gv)))
    return torch.tensor(vals, device=pred_btjc.device).mean()


# ------------------ Lightning module ------------------

class LitMinimal(pl.LightningModule):
    """
    Minimal Lightning module:
    - Forward: SignWritingToPoseDiffusion (expects BJCT)
    - Loss: masked MSE; plus DTW in validation as a sanity metric
    - No checkpointing; meant for quick end-to-end checks.
    """
    def __init__(self, num_keypoints=586, num_dims=3, lr=1e-3, log_dir="logs"):
        super().__init__()
        self.save_hyperparameters()

        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints, num_dims_per_keypoint=num_dims
        )
        self.lr = lr
        self.log_dir = log_dir
        self.train_losses, self.val_losses, self.val_dtws = [], [], []

    # —— 关键：把布局转换“藏”在 forward 里（导师建议） ——
    def forward(self, x_btjc, timesteps, past_btjc, sign_img):
        x_bjct    = btjc_to_bjct(sanitize_btjc(x_btjc))
        past_bjct = btjc_to_bjct(sanitize_btjc(past_btjc))
        out_bjct  = self.model.forward(x_bjct, timesteps, past_bjct, sign_img)
        pred_btjc = bjct_to_btjc(out_bjct)
        return pred_btjc

    def _make_mask_bt(self, raw_mask):
        mask = raw_mask.float()
        if mask.dim() == 5:   # [B,T,P,J,C] -> [B,T]
            mask = (mask.abs().sum(dim=(2, 3, 4)) > 0).float()
        elif mask.dim() == 4: # [B,T,J,C] -> [B,T]
            mask = (mask.abs().sum(dim=(2, 3)) > 0).float()
        elif mask.dim() == 3: # [B,T,C]   -> [B,T]
            mask = (mask.abs().sum(dim=2) > 0).float()
        return mask

    def training_step(self, batch, _):
        cond = batch["conditions"]
        fut  = sanitize_btjc(batch["data"])               # [B,Tf,J,C]
        past = sanitize_btjc(cond["input_pose"])          # [B,Tp,J,C]
        mask = self._make_mask_bt(cond["target_mask"])    # [B,Tf]
        sign = cond["sign_image"].float()                 # [B,3,224,224]
        ts   = torch.zeros(fut.size(0), dtype=torch.long, device=fut.device)

        pred = self.forward(fut, ts, past, sign)
        loss = masked_mse(pred, fut, mask)

        self.train_losses.append(loss.item())
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, _):
        cond = batch["conditions"]
        fut  = sanitize_btjc(batch["data"])
        past = sanitize_btjc(cond["input_pose"])
        mask = self._make_mask_bt(cond["target_mask"])
        sign = cond["sign_image"].float()
        ts   = torch.zeros(fut.size(0), dtype=torch.long, device=fut.device)

        pred = self.forward(fut, ts, past, sign)
        loss = masked_mse(pred, fut, mask)
        dtw  = masked_dtw(pred, fut, mask)

        self.val_losses.append(loss.item())
        self.val_dtws.append(dtw.item())
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/dtw",  dtw,  prog_bar=False)

    @torch.no_grad()
    def generate_autoregressive(self, past_btjc, sign_img, future_steps):
        """给定 past（BTJC）与 sign 图像，逐帧生成 future_steps 帧，返回 [B,Tf,J,C]。"""
        self.eval()
        ctx  = sanitize_btjc(past_btjc).to(self.device)   # [B,Tp,J,C]
        sign = sign_img.to(self.device)
        B, Tp, J, C = ctx.shape

        preds = []
        for _ in range(future_steps):
            seed = ctx[:, -1:, ...]                       # [B,1,J,C]
            ts   = torch.zeros(B, dtype=torch.long, device=self.device)
            nxt  = self.forward(seed, ts, ctx, sign)[:, :1, ...]  # 取 1 帧
            preds.append(nxt)
            ctx = torch.cat([ctx, nxt], dim=1)
            if ctx.size(1) > Tp:
                ctx = ctx[:, 1:, ...]
        return torch.cat(preds, dim=1)                    # [B,Tf,J,C]

    def on_fit_end(self):
        os.makedirs(self.log_dir, exist_ok=True)
        csv_path = os.path.join(self.log_dir, "minimal_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "train_loss", "val_loss", "val_dtw"])
            max_len = max(len(self.train_losses), len(self.val_losses), len(self.val_dtws))
            for i in range(max_len):
                tr  = self.train_losses[i] if i < len(self.train_losses) else ""
                vl  = self.val_losses[i]  if i < len(self.val_losses)  else ""
                dv  = self.val_dtws[i]    if i < len(self.val_dtws)    else ""
                w.writerow([i + 1, tr, vl, dv])

        import matplotlib.pyplot as plt
        plt.figure()
        if self.train_losses: plt.plot(self.train_losses, label="train_loss")
        if self.val_losses:   plt.plot(self.val_losses,   label="val_loss")
        if self.val_dtws:     plt.plot(self.val_dtws,     label="val_dtw")
        plt.xlabel("steps")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "minimal_curves.png"))
        plt.close()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

