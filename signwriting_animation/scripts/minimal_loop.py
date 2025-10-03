import os
import lightning as pl
import torch
from torch.utils.data import DataLoader
from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset, get_num_workers
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion

def masked_mse(pred, target, mask):
    mask = mask.float()
    diff2 = (pred - target) ** 2          # [B,T,J,C]
    m = mask[:, :, None, None]            # [B,T,1,1]
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

        # DTW on first sample to keep it cheap
        b0 = 0
        tf = int(batch["future_mask"][b0].sum().item())
        tf = min(tf, pred.shape[1])
        pf = pred[b0, :tf].reshape(tf, -1)
        gt = batch["future_pose"][b0, :tf].reshape(tf, -1)
        dtw = chunked_dtw_mean(pf, gt)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/dtw", dtw, prog_bar=True)

        # length prediction (if available)
        if "pred_len" in out:
            pred_len = out["pred_len"].squeeze(-1) if out["pred_len"].dim() > 1 else out["pred_len"]
            len_mae = (pred_len - batch["future_mask"].sum(dim=1)).abs().mean()
            self.log("val/len_mae", len_mae, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def make_loader(data_dir, csv_path, split, bs, num_workers, num_past=40, num_future=20):
    ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=num_past,
        num_future_frames=num_future,
        with_metadata=True,
        split=split,
    )
    return DataLoader(
        ds, batch_size=bs, shuffle=(split == "train"),
        num_workers=num_workers, collate_fn=zero_pad_collator
    )


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    data_dir = os.getenv("DATA_DIR", "/data/yayun/raw_poses")
    csv_path = os.getenv("CSV_PATH", "/data/yayun/signwriting-animation/data.csv")
    num_workers = min(2, get_num_workers())

    num_keypoints, num_dims = 586, 3

    train_loader = make_loader(data_dir, csv_path, "train", bs=8, num_workers=num_workers)
    val_loader   = make_loader(data_dir, csv_path, "dev",   bs=8, num_workers=num_workers)

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
