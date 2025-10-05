# signwriting_animation/scripts/minimal_loop.py
import os
import csv
import torch
import lightning as pl
from torch.utils.data import DataLoader, Dataset
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion


# ============================== helper functions ==============================

def _to_dense(x):
    if x.is_sparse:
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
        raise ValueError(f"Expected 4D tensor [B,T,J,C], got {tuple(x.shape)}")
    return x


def btjc_to_bjct(x):
    return x.permute(0, 2, 3, 1).contiguous()


def bjct_to_btjc(x):
    return x.permute(0, 3, 1, 2).contiguous()


def masked_mse(pred, tgt, mask):
    """
    pred/tgt: [B,T,J,C], mask: [B,T] (1 有效, 0 padding)
    自动对齐时间长度，避免维度不匹配。
    """
    pred, tgt = sanitize_btjc(pred), sanitize_btjc(tgt)

    # 对齐时间长度
    B, T1, J, C = pred.shape
    B2, T2, J2, C2 = tgt.shape
    assert B == B2 and J == J2 and C == C2, "shape mismatch"
    Tm = min(T1, T2, mask.size(1))

    pred = pred[:, :Tm]
    tgt = tgt[:, :Tm]
    mask = mask[:, :Tm].float()[:, :, None, None]

    diff2 = (pred - tgt) ** 2
    num = (diff2 * mask).sum()
    den = (mask.sum() * J * C).clamp_min(1.0)
    return num / den



# ============================== filtered dataset ==============================

class FilteredDataset(Dataset):
    """Only keep valid samples."""
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


# ============================== Lightning module ==============================

class LitMinimal(pl.LightningModule):
    def __init__(self, num_keypoints=586, num_dims=3, lr=1e-3, log_dir="logs"):
        super().__init__()
        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints, num_dims_per_keypoint=num_dims
        )
        self.lr = lr
        self.log_dir = log_dir
        self.train_losses, self.val_losses, self.val_dtws = [], [], []

    def forward(self, x_bjct, timesteps, past_bjct, sign_img):
        return self.model.forward(x_bjct, timesteps, past_bjct, sign_img)

    def training_step(self, batch, _):
        cond = batch["conditions"]
        fut = sanitize_btjc(batch["data"])
        past = sanitize_btjc(cond["input_pose"])
        mask = cond["target_mask"].float().mean(dim=(2, 3), keepdim=False)  # [B,T]
        sign_img = cond["sign_image"].float()

        x_bjct = btjc_to_bjct(fut)
        past_bjct = btjc_to_bjct(past)
        timesteps = torch.zeros(x_bjct.size(0), dtype=torch.long, device=x_bjct.device)
        out = self.forward(x_bjct, timesteps, past_bjct, sign_img)
        pred = bjct_to_btjc(out)
        loss = masked_mse(pred, fut, mask)

        self.train_losses.append(loss.item())
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, _):
        cond = batch["conditions"]
        fut = sanitize_btjc(batch["data"])
        past = sanitize_btjc(cond["input_pose"])
        mask = cond["target_mask"].float().mean(dim=(2, 3), keepdim=False)
        sign_img = cond["sign_image"].float()

        x_bjct = btjc_to_bjct(fut)
        past_bjct = btjc_to_bjct(past)
        timesteps = torch.zeros(x_bjct.size(0), dtype=torch.long, device=x_bjct.device)
        out = self.forward(x_bjct, timesteps, past_bjct, sign_img)
        pred = bjct_to_btjc(out)
        loss = masked_mse(pred, fut, mask)

        self.val_losses.append(loss.item())
        self.log("val/loss", loss, prog_bar=True)

    def on_fit_end(self):
        os.makedirs(self.log_dir, exist_ok=True)
        csv_path = os.path.join(self.log_dir, "minimal_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "train_loss", "val_loss"])
            for i in range(max(len(self.train_losses), len(self.val_losses))):
                tr = self.train_losses[i] if i < len(self.train_losses) else ""
                vl = self.val_losses[i] if i < len(self.val_losses) else ""
                w.writerow([i + 1, tr, vl])

        import matplotlib.pyplot as plt
        plt.figure()
        if self.train_losses:
            plt.plot(self.train_losses, label="train_loss")
        if self.val_losses:
            plt.plot(self.val_losses, label="val_loss")
        plt.xlabel("steps")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "minimal_curves.png"))
        plt.close()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ============================== dataloader builder ==============================

def make_loader(data_dir, csv_path, split, bs, num_workers):
    base = DynamicPosePredictionDataset(data_dir=data_dir, csv_path=csv_path, with_metadata=True, split=split)
    ds = FilteredDataset(base, target_count=200, max_scan=5000)
    print(f"[DEBUG] split={split} | batch_size={bs} | len(ds)={len(ds)}")
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=zero_pad_collator)


# ============================== main ==============================

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_default_dtype(torch.float32)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data.csv"  # ← full dataset

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

