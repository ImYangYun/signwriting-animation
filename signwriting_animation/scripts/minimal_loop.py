import os
import torch
import lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion


# ------------------------ utils ------------------------

def masked_mse(pred, target, mask):
    mask = mask.float()  # [B, T]
    diff2 = (pred - target) ** 2  # [B, T, J, C]
    m = mask[:, :, None, None]  # [B, T, 1, 1]
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


def _to_dense_btjc(t):
    """Convert pose tensor (possibly sparse or 5D) -> [B, T, J, C] dense"""
    if isinstance(t, torch.Tensor) and t.is_sparse:
        t = t.to_dense()

    if t.dim() == 5 and t.size(2) == 1:  # [B, T, 1, J, C]
        t = t.squeeze(2)
    elif t.dim() == 5 and t.size(2) > 1:
        # Multi-person → take first person
        t = t[:, :, 0, ...]
    elif t.dim() != 4:
        raise RuntimeError(f"Unexpected pose tensor shape {tuple(t.shape)}")

    return t.contiguous()


# ------------------------ filtered dataset ------------------------

class FilteredDataset(Dataset):
    def __init__(self, base: Dataset, target_count: int = 1, max_scan: Optional[int] = 200):
        self.base = base
        self.idx = []
        N = len(base)
        scan_limit = N if max_scan is None else min(N, max_scan)
        for i in range(scan_limit):
            try:
                it = base[i]
                data = it.get("data", None)
                if data is not None:
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


# ------------------------ lightning module ------------------------

class LitMinimal(pl.LightningModule):
    def __init__(self, num_keypoints: int, num_dims: int, lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints, num_dims_per_keypoint=num_dims
        )
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x, timesteps, past_motion, sign_img):
        return self.model.forward(x, timesteps, past_motion, sign_img)

    def training_step(self, batch, _):
        cond = batch["conditions"]
        fut = _to_dense_btjc(batch["data"])  # [B, T_f, J, C]
        past = _to_dense_btjc(cond["input_pose"])  # [B, T_p, J, C]
        tgt_mask = cond["target_mask"].float()  # [B, T_f]
        sign_img = cond["sign_image"]  # [B, 3, 224, 224]

        # 转换为 [B, J, C, T]
        x = fut.permute(0, 2, 3, 1).contiguous()
        past_motion = past.permute(0, 2, 3, 1).contiguous()
        timesteps = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        out_bjct = self.model.forward(x, timesteps, past_motion, sign_img)
        pred = out_bjct.permute(0, 3, 1, 2).contiguous()  # -> [B, T, J, C]

        loss = masked_mse(pred, fut, tgt_mask)
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, _):
        cond = batch["conditions"]
        fut = _to_dense_btjc(batch["data"])
        past = _to_dense_btjc(cond["input_pose"])
        tgt_mask = cond["target_mask"].float()
        sign_img = cond["sign_image"]

        x = fut.permute(0, 2, 3, 1).contiguous()
        past_motion = past.permute(0, 2, 3, 1).contiguous()
        timesteps = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        out_bjct = self.model.forward(x, timesteps, past_motion, sign_img)
        pred = out_bjct.permute(0, 3, 1, 2).contiguous()

        loss = masked_mse(pred, fut, tgt_mask)

        b0 = 0
        tf = int(tgt_mask[b0].sum().item())
        tf = min(tf, pred.shape[1])
        pf = pred[b0, :tf].reshape(tf, -1)
        gt = fut[b0, :tf].reshape(tf, -1)
        dtw = chunked_dtw_mean(pf, gt)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/dtw", dtw, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# ------------------------ dataloader builder ------------------------

def make_loader(data_dir, csv_path, split, bs, num_workers, target_count=1, max_scan=1000):
    base = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        with_metadata=True,
        split=split,
    )
    ds = FilteredDataset(base, target_count=target_count, max_scan=max_scan)
    print(f"[DEBUG] split={split} | batch_size={bs} | len(ds)={len(ds)}")
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers,
                      pin_memory=False, collate_fn=zero_pad_collator)


# ------------------------ main ------------------------

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    data_dir = os.getenv("DATA_DIR", "/data/yayun/pose_data")
    csv_path = os.getenv("CSV_PATH", "/data/yayun/signwriting-animation/mini_data.csv")

    batch_size = 1
    num_keypoints, num_dims = 586, 3
    target_count, max_scan = 1, 100

    train_loader = make_loader(data_dir, csv_path, "train", bs=batch_size, num_workers=0,
                               target_count=target_count, max_scan=max_scan)
    val_loader = train_loader

    # Debug print first batch
    first_batch = next(iter(train_loader))
    print("[DEBUG] keys:", list(first_batch.keys()))
    print("[DEBUG] cond keys:", list(first_batch["conditions"].keys()))
    for k, v in first_batch["conditions"].items():
        if isinstance(v, torch.Tensor):
            print(f"[DEBUG] cond[{k}] shape:", v.shape, "sparse:", v.is_sparse)
    if isinstance(first_batch["data"], torch.Tensor):
        print("[DEBUG] data shape:", first_batch["data"].shape, "sparse:", first_batch["data"].is_sparse)

    model = LitMinimal(num_keypoints=num_keypoints, num_dims=num_dims)
    trainer = pl.Trainer(
        max_steps=600,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1, 
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_loader, val_loader)

