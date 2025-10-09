# signwriting_animation/scripts/minimal_loop.py
import os
import csv
import torch
import lightning as pl
from torch.utils.data import DataLoader, Dataset
from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from pose_evaluation.metrics import dtw_mje as PE_DTW


def _to_dense(x):
    """
    Make a batch tensor dense float32:
    - pose-format MaskedTensor -> .zero_filled()
    - sparse -> to_dense()
    - cast to float32, contiguous
    """
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    if getattr(x, "is_sparse", False):
        x = x.to_dense()
    if x.dtype != torch.float32:
        x = x.float()
    return x.contiguous()

def sanitize_btjc(x):
    """Ensure tensor is [B,T,J,C]. Handle [B,T,P,J,C] (take first person)."""
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


def masked_mse(pred, tgt, mask_bt):
    """
    pred/tgt: [B,T,J,C]
    mask_bt : [B,T]  (1 valid, 0 padding)
    """
    pred, tgt = sanitize_btjc(pred), sanitize_btjc(tgt)
    Tm = min(pred.size(1), tgt.size(1), mask_bt.size(1))
    pred = pred[:, :Tm]
    tgt  = tgt[:,  :Tm]
    m4 = mask_bt[:, :Tm].float()[:, :, None, None]     # [B,T,1,1]
    diff2 = (pred - tgt) ** 2                          # [B,T,J,C]
    num = (diff2 * m4).sum()
    den = (m4.sum() * pred.size(2) * pred.size(3)).clamp_min(1.0)
    return num / den


def _btjc_to_tjc_list(x_btjc, mask_bt):
    """BTJC + [B,T] mask -> list of per-sample [t,J,C] sequences (trim padded time)."""
    x_btjc = sanitize_btjc(x_btjc)
    B, T, J, C = x_btjc.shape
    seqs = []
    for b in range(B):
        t = int(mask_bt[b].sum().item())
        t = max(0, min(t, T))
        seqs.append(x_btjc[b, :t].contiguous())  # [t,J,C]
    return seqs


def compute_dtw_metric(pred_btjc, tgt_btjc, mask_bt):
    """Use pose-evaluation's DTW-MJE on CPU; return mean over batch (torch scalar)."""
    pred_list = _btjc_to_tjc_list(pred_btjc, mask_bt)
    tgt_list  = _btjc_to_tjc_list(tgt_btjc,  mask_bt)

    vals = []
    for p, g in zip(pred_list, tgt_list):
        pv = p.detach().cpu().numpy().astype("float32")  # [t,J,C]
        gv = g.detach().cpu().numpy().astype("float32")
        vals.append(float(PE_DTW(pv, gv)))
    return torch.tensor(vals, device=pred_btjc.device).mean()


# ============================== filtered dataset ==============================

class FilteredDataset(Dataset):
    """Only keep valid samples; quick scan to build a small valid index set."""
    def __init__(self, base: Dataset, target_count=200, max_scan=5000):
        self.base = base
        self.idx = []
        N = len(base)
        for i in range(min(N, max_scan)):
            try:
                it = base[i]
                if isinstance(it, dict) and "data" in it and "conditions" in it:
                    self.idx.append(i)
                if len(self.idx) >= target_count:
                    break
            except Exception:
                continue
        if not self.idx:
            self.idx = [0]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]

class LitMinimal(pl.LightningModule):
    """
    Minimal Lightning module (name unchanged):
    - Forward accepts BTJC; internal BJCT↔BTJC permutations are handled here.
    - Training loss: masked MSE; Validation: masked MSE + DTW (pose-eval).
    """
    def __init__(self, num_keypoints=586, num_dims=3, lr=1e-3, log_dir="logs"):
        super().__init__()
        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints, num_dims_per_keypoint=num_dims
        )
        self.lr = lr
        self.log_dir = log_dir
        self.train_losses, self.val_losses, self.val_dtws = [], [], []

    def forward(self, x_btjc, timesteps, past_btjc, sign_img):
        out_bjct = self.model.forward(
            btjc_to_bjct(x_btjc),
            timesteps,
            btjc_to_bjct(past_btjc),
            sign_img,
        )
        return bjct_to_btjc(out_bjct)

    @staticmethod
    def _to_mask_bt(raw_mask):
        mask = raw_mask.float()
        if mask.dim() == 5:
            mask = (mask.abs().sum(dim=(2, 3, 4)) > 0).float()
        elif mask.dim() == 4:
            mask = (mask.abs().sum(dim=(2, 3)) > 0).float()
        elif mask.dim() == 3:
            mask = (mask.abs().sum(dim=2) > 0).float()
        return mask

    def training_step(self, batch, _):
        cond = batch["conditions"]
        fut  = sanitize_btjc(batch["data"])
        past = sanitize_btjc(cond["input_pose"])
        mask = self._to_mask_bt(cond["target_mask"])
        sign_img  = cond["sign_image"].float()
        timesteps = torch.zeros(fut.size(0), dtype=torch.long, device=fut.device)

        pred = self.forward(fut, timesteps, past, sign_img)
        loss = masked_mse(pred, fut, mask)

        self.train_losses.append(loss.item())
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, _):
        cond = batch["conditions"]
        fut  = sanitize_btjc(batch["data"])
        past = sanitize_btjc(cond["input_pose"])
        mask = self._to_mask_bt(cond["target_mask"])
        sign_img  = cond["sign_image"].float()
        timesteps = torch.zeros(fut.size(0), dtype=torch.long, device=fut.device)

        pred = self.forward(fut, timesteps, past, sign_img)
        loss = masked_mse(pred, fut, mask)
        dtw  = compute_dtw_metric(pred, fut, mask).to(loss.device)

        self.val_losses.append(loss.item())
        self.val_dtws.append(dtw.item())
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/dtw",  dtw,  prog_bar=False)

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
        plt.xlabel("steps"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "minimal_curves.png")); plt.close()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ============================== dataloader builder ==============================

def make_loader(data_dir, csv_path, split, bs, num_workers):
    """
    Build a DataLoader with:
    - Dataset returning MaskedTensor.
    - zero_pad_collator to align sequences along time dimension.
    - FilteredDataset to pick a small, valid subset for a minimal run.
    """
    base = DynamicPosePredictionDataset(
        data_dir=data_dir, csv_path=csv_path, with_metadata=True, split=split
    )
    ds = FilteredDataset(base, target_count=200, max_scan=3000)
    print(f"[DEBUG] split={split} | batch_size={bs} | len(ds)={len(ds)}")
    return DataLoader(
        ds, batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=zero_pad_collator
    )


# ============================== main ==============================

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_default_dtype(torch.float32)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data.csv"

    batch_size = 4
    num_workers = 2

    train_loader = make_loader(data_dir, csv_path, "train", bs=batch_size, num_workers=num_workers)
    val_loader = train_loader

    model = LitMinimal(log_dir="logs")

    trainer = pl.Trainer(
        max_steps=1000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=5,
        limit_train_batches=10,
        limit_val_batches=5,
        check_val_every_n_epoch=1,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_loader, val_loader)
