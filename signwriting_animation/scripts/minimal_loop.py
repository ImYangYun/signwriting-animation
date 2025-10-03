import os
import lightning as pl
import torch
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import (
    DynamicPosePredictionDataset,
    get_num_workers,
)
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion


# ------------------------ losses & metrics ------------------------

def masked_mse(pred, target, mask):
    mask = mask.float()                    # [B,T]
    diff2 = (pred - target) ** 2           # [B,T,J,C]
    m = mask[:, :, None, None]             # [B,T,1,1]
    num = (diff2 * m).sum()
    den = (m.sum() * pred.size(2) * pred.size(3)).clamp_min(1.0)
    return num / den


def simple_dtw(a, b):
    a = a.detach().cpu()
    b = b.detach().cpu()
    T, D = a.shape
    Tp, Dp = b.shape
    assert D == Dp, "DTW dims mismatch"
    dist = torch.cdist(a, b)
    dp = torch.full((T + 1, Tp + 1), float("inf"))
    dp[0, 0] = 0.0
    for i in range(1, T + 1):
        for j in range(1, Tp + 1):
            cost = dist[i - 1, j - 1]
            dp[i, j] = float(cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]))
    return torch.tensor(dp[T, Tp])


def chunked_dtw_mean(pred_seq, tgt_seq, max_len=160, chunk=40):
    T = min(pred_seq.shape[0], max_len)
    pred_seq = pred_seq[:T]
    tgt_seq = tgt_seq[:T]
    if T <= 1:
        return torch.tensor(0.0)
    vals = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        if e - s < 2:
            continue
        vals.append(simple_dtw(pred_seq[s:e], tgt_seq[s:e]))
    return torch.stack(vals).mean() if vals else torch.tensor(0.0)


# ------------------------ small filtered dataset ------------------------

class FilteredDataset(Dataset):
    """
    Wrap a base dataset; only keep up to `target_count` valid samples,
    scanning at most `max_scan` items. This prevents huge memory usage.
    """
    def __init__(self, base: Dataset, min_past: int, min_future: int,
                 target_count: int = 64, max_scan: Optional[int] = 2000):
        self.base = base
        self.idx = []
        N = len(base)
        scan_limit = N if max_scan is None else min(N, max_scan)
        for i in range(scan_limit):
            try:
                it = base[i]
                pm, fm = it.get("past_mask"), it.get("future_mask")
                if int(pm.sum().item()) < min_past:
                    continue
                if int(fm.sum().item()) < min_future:
                    continue
                self.idx.append(i)
                if target_count is not None and len(self.idx) >= target_count:
                    break
            except Exception:
                # skip broken items
                continue
        if not self.idx:
            raise RuntimeError("FilteredDataset got 0 valid samples; "
                               "try lowering thresholds or increasing max_scan.")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]


# ------------------------ lightning module ------------------------

class LitMinimal(pl.LightningModule):
    def __init__(self, num_keypoints: int, num_dims: int, lr=1e-3):
        super().__init__()
        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints, num_dims_per_keypoint=num_dims
        )
        self.lr = lr

    def forward(self, past, past_mask, **kwargs):
        return self.model(past_motion=past, past_mask=past_mask, return_dict=True)

    def training_step(self, batch, _):
        out = self(batch["past_pose"], batch["past_mask"])
        loss = masked_mse(out["pred_future"], batch["future_pose"], batch["future_mask"])
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, _):
        out = self(batch["past_pose"], batch["past_mask"])
        pred = out["pred_future"]
        loss = masked_mse(pred, batch["future_pose"], batch["future_mask"])

        # cheap DTW on first sample
        b0 = 0
        tf = int(batch["future_mask"][b0].sum().item())
        tf = min(tf, pred.shape[1])
        pf = pred[b0, :tf].reshape(tf, -1)
        gt = batch["future_pose"][b0, :tf].reshape(tf, -1)
        dtw = chunked_dtw_mean(pf, gt)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/dtw", dtw, prog_bar=True)

        # length prediction (if model provides it)
        if "pred_len" in out:
            pred_len = out["pred_len"].squeeze(-1) if out["pred_len"].dim() > 1 else out["pred_len"]
            len_mae = (pred_len - batch["future_mask"].sum(dim=1)).abs().mean()
            self.log("val/len_mae", len_mae, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ------------------------ dataloader builder ------------------------

def make_loader(
    data_dir,
    csv_path,
    split,
    bs,
    num_workers,
    num_past=40,
    num_future=20,
    target_count=64,
    max_scan=2000,
):
    base = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=num_past,
        num_future_frames=num_future,
        with_metadata=True,
        split=split,
    )
    ds = FilteredDataset(
        base, min_past=num_past, min_future=num_future,
        target_count=target_count, max_scan=max_scan
    )

    print(f"[DEBUG] split={split} | batch_size={bs} | num_workers=0 "
          f"| target_count={target_count} | max_scan={max_scan} | len(ds)={len(ds)}")

    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(split == "train"),
        num_workers=0,              # force 0 to reduce memory pressure
        pin_memory=False,
        persistent_workers=False,
        collate_fn=zero_pad_collator,
    )


# ------------------------ main ------------------------

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    # default to your current layout; can be overridden by env vars
    data_dir = os.getenv("DATA_DIR", "/data/yayun/signwriting-animation/raw_poses")
    csv_path = os.getenv("CSV_PATH", "/data/yayun/signwriting-animation/data.csv")

    # small, safe defaults
    batch_size = 2
    num_workers = 0
    num_keypoints, num_dims = 586, 3
    num_past, num_future = 40, 20
    target_count, max_scan = 64, 2000

    train_loader = make_loader(
        data_dir, csv_path, "train",
        bs=batch_size, num_workers=num_workers,
        num_past=num_past, num_future=num_future,
        target_count=target_count, max_scan=max_scan
    )
    val_loader = make_loader(
        data_dir, csv_path, "dev",
        bs=batch_size, num_workers=num_workers,
        num_past=num_past, num_future=num_future,
        target_count=target_count, max_scan=max_scan
    )

    # sanity grab one batch to ensure we won't OOM at first iteration
    try:
        first_batch = next(iter(train_loader))
        shapes = {k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in first_batch.items()}
        print("[DEBUG] first train batch keys:", shapes)
    except Exception as e:
        print("[DEBUG] failed to get first batch:", repr(e))
        raise

    model = LitMinimal(num_keypoints=num_keypoints, num_dims=num_dims, lr=1e-3)
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        log_every_n_steps=5,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)
