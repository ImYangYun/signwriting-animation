# signwriting_animation/scripts/minimal_loop.py
import os
from typing import Optional
import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion


# ------------------------ losses & metrics ------------------------

def masked_mse(pred, target, mask):
    """
    pred/target: [B, T, J, C]
    mask:       [B, T] (1=valid, 0=pad)
    """
    mask = mask.float()
    diff2 = (pred - target) ** 2           # [B,T,J,C]
    m = mask[:, :, None, None]             # [B,T,1,1]
    num = (diff2 * m).sum()
    den = (m.sum() * pred.size(2) * pred.size(3)).clamp_min(1.0)
    return num / den


def simple_dtw(a, b):
    """
    a,b: [T, D]
    """
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
    """
    pred_seq/tgt_seq: [T, D]
    """
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
    从基础 dataset 里预扫描，最多保留 target_count 条“有效”样本（输入/目标长度满足阈值）。
    这样可以避免一次性遍历全数据。
    """
    def __init__(self, base: Dataset, min_past: int, min_future: int,
                 target_count: int = 1, max_scan: Optional[int] = 200):
        self.base = base
        self.idx = []
        N = len(base)
        scan_limit = N if max_scan is None else min(N, max_scan)
        for i in range(scan_limit):
            try:
                item = base[i]
                cond = item.get("conditions", {})
                past_mask = cond.get("input_mask", None)
                fut_mask  = cond.get("target_mask", None)

                # 这两个 mask 在你的 dataloader 里是 1D (T)，通过 collator 后会变成 [T] 或 [B,T]。
                def _sum(x):
                    return int(torch.as_tensor(x).sum().item())

                if past_mask is not None and _sum(past_mask) < min_past:
                    continue
                if fut_mask is not None and _sum(fut_mask) < min_future:
                    continue

                self.idx.append(i)
                if target_count is not None and len(self.idx) >= target_count:
                    break
            except Exception:
                continue

        if not self.idx:
            # 兜底：至少保留第一条，保证 loop 能跑起来
            self.idx = [0]

    def __len__(self): return len(self.idx)
    def __getitem__(self, i): return self.base[self.idx[i]]


# ------------------------ lightning module ------------------------

class LitMinimal(pl.LightningModule):
    def __init__(self, num_keypoints: int, num_dims: int, lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints, num_dims_per_keypoint=num_dims
        )
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch, _):
        # --- unpack from your dataset format ---
        fut      = batch["data"]                      # [B, T_f, J, C]
        cond     = batch["conditions"]
        past     = cond["input_pose"]                 # [B, T_p, J, C]
        tgt_mask = cond["target_mask"].float()        # [B, T_f]
        sign_img = cond["sign_image"]                 # [B, 3, 224, 224]

        # --- permute to model's expected layout [B, J, C, T] ---
        x           = fut.permute(0, 2, 3, 1).contiguous()   # [B, J, C, T_f]
        past_motion = past.permute(0, 2, 3, 1).contiguous()  # [B, J, C, T_p]
        timesteps   = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # --- model forward (returns [B, J, C, T_f]) ---
        out_bjct = self.model.forward(x, timesteps, past_motion, sign_img)
        pred     = out_bjct.permute(0, 3, 1, 2).contiguous()  # -> [B, T_f, J, C]

        loss = masked_mse(pred, fut, tgt_mask)
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, _):
        fut      = batch["data"]
        cond     = batch["conditions"]
        past     = cond["input_pose"]
        tgt_mask = cond["target_mask"].float()
        sign_img = cond["sign_image"]

        x           = fut.permute(0, 2, 3, 1).contiguous()
        past_motion = past.permute(0, 2, 3, 1).contiguous()
        timesteps   = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        out_bjct = self.model.forward(x, timesteps, past_motion, sign_img)
        pred     = out_bjct.permute(0, 3, 1, 2).contiguous()

        loss = masked_mse(pred, fut, tgt_mask)

        # DTW on first sample
        b0 = 0
        tf = int(tgt_mask[b0].sum().item())
        tf = min(tf, pred.shape[1])
        pf = pred[b0, :tf].reshape(tf, -1)
        gt = fut[b0,  :tf].reshape(tf, -1)
        dtw = chunked_dtw_mean(pf, gt)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/dtw",  dtw,  prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# ------------------------ dataloader builder ------------------------

def make_loader(
    data_dir,
    csv_path,
    split,
    bs,
    num_workers,
    num_past=10,
    num_future=5,
    target_count=1,
    max_scan=1000,
):
    base = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=num_past,
        num_future_frames=num_future,
        with_metadata=True,    # 保留即可，不影响训练
        split=split,
    )
    ds = FilteredDataset(
        base, min_past=num_past, min_future=num_future,
        target_count=target_count, max_scan=max_scan
    )

    print(f"[DEBUG] split={split} | batch_size={bs} | len(ds)={len(ds)}")

    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,              # 单样本过拟合不需要 shuffle
        num_workers=0,              # 省内存
        pin_memory=False,
        persistent_workers=False,
        collate_fn=zero_pad_collator,
    )


# ------------------------ main ------------------------

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    data_dir = os.getenv("DATA_DIR", "/data/yayun/pose_data")
    csv_path = os.getenv("CSV_PATH", "/data/yayun/signwriting-animation/mini_data.csv")

    # 过拟合设置（尽量保守，先跑通）
    batch_size = 1
    num_workers = 0
    num_keypoints, num_dims = 586, 3
    num_past, num_future = 10, 5
    target_count, max_scan = 1, 1000

    # 构建 loader（train/val 用同一条样本）
    train_loader = make_loader(
        data_dir, csv_path, "train",
        bs=batch_size, num_workers=num_workers,
        num_past=num_past, num_future=num_future,
        target_count=target_count, max_scan=max_scan
    )
    val_loader = train_loader

    # 预取一个 batch，打印结构
    first = next(iter(train_loader))
    print("[DEBUG] top-level keys:", list(first.keys()))
    if "conditions" in first:
        print("[DEBUG] condition keys:", list(first["conditions"].keys()))
        print("[DEBUG] shapes -> data:", tuple(first["data"].shape),
              "input_pose:", tuple(first["conditions"]["input_pose"].shape),
              "target_mask:", tuple(first["conditions"]["target_mask"].shape),
              "sign_image:", tuple(first["conditions"]["sign_image"].shape))

    model = LitMinimal(num_keypoints=num_keypoints, num_dims=num_dims, lr=1e-3, weight_decay=0.0)
    trainer = pl.Trainer(
        max_steps=600,
        accelerator="auto",
        devices=1,
        log_every_n_steps=5,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,   # 先关 sanity check，加快起跑
    )
    trainer.fit(model, train_loader, val_loader)


