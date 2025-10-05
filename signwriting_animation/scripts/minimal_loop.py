# signwriting_animation/scripts/minimal_loop.py
import os
import csv
import math
from typing import Optional, Dict, Any, Tuple

import torch
import lightning as pl
from torch.utils.data import DataLoader, Dataset
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion


# ============================== helpers: tensor & mask sanitation ==============================

def _to_dense(x: torch.Tensor) -> torch.Tensor:
    """If sparse -> dense; always contiguous float32."""
    if x.is_sparse:
        x = x.to_dense()
    if x.dtype != torch.float32:
        x = x.float()
    return x.contiguous()


def btjc_to_bjct(x_btjc: torch.Tensor) -> torch.Tensor:
    """[B,T,J,C] -> [B,J,C,T]"""
    return x_btjc.permute(0, 2, 3, 1).contiguous()


def bjct_to_btjc(x_bjct: torch.Tensor) -> torch.Tensor:
    """[B,J,C,T] -> [B,T,J,C]"""
    return x_bjct.permute(0, 3, 1, 2).contiguous()


def sanitize_btjc(x: torch.Tensor) -> torch.Tensor:
    """
    Accept tensors like:
      - [B,T,J,C] (dense or sparse)
      - [B,T,1,J,C] / [B,T,P,J,C] (multi-person) -> 取第一个人
    Return dense [B,T,J,C].
    """
    x = _to_dense(x)

    if x.dim() == 5:  # [B,T,P,J,C]
        # 取第一个人（P>=1）
        x = x[:, :, 0, ...]
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor [B,T,J,C], got {tuple(x.shape)} (ndim={x.dim()})")
    return x.contiguous()


def ensure_non_empty_time(x_btjc: torch.Tensor) -> torch.Tensor:
    """
    If T==0 -> pad a single zero frame. Keep [B,T,J,C].
    """
    B, T, J, C = x_btjc.shape
    if T > 0:
        return x_btjc
    pad = torch.zeros((B, 1, J, C), dtype=x_btjc.dtype, device=x_btjc.device)
    return pad


def infer_mask_from_btjc(x_btjc: torch.Tensor) -> torch.Tensor:
    """
    推断 [B,T] 的有效帧：只要该帧有任意关节/通道非零，就视为有效。
    """
    B, T, J, C = x_btjc.shape
    if T == 0:
        # 如果上游还没补，补一帧零并返回 [B,1] 全零
        x_btjc = ensure_non_empty_time(x_btjc)
        B, T, J, C = x_btjc.shape
    # [B,T,J,C] -> [B,T]
    valid = (x_btjc.abs().sum(dim=(2, 3)) > 0).float()
    return valid


def to_bt_mask_strict(x: torch.Tensor, target_T: Optional[int] = None) -> torch.Tensor:
    """
    期望把任何 mask 变成严格的 [B,T]。
    支持输入：
      - [B,T]
      - [B,1,T] / [B,T,1]
      - [B,T,J,C]（会降维到 [B,T]）
      - [B,T,K]（K>1，按最后一维求“是否存在非零”）
    如果 target_T 给定，会在末尾/截断到该长度。
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("mask must be a tensor")

    # 稀疏/精度统一
    if x.is_sparse:
        x = x.to_dense()
    if x.dtype != torch.float32:
        x = x.float()

    if x.dim() == 4:
        # 直接当做姿态，推断 [B,T]
        m = infer_mask_from_btjc(x)

    elif x.dim() == 3:
        # 常见三种： [B,1,T] / [B,T,1] / [B,T,K]
        B, A, T = x.shape
        if A == 1 and T >= 1:               # [B,1,T]
            m = x.squeeze(1)
        elif T == 1 and A >= 1:             # [B,T,1] 其实是 [B,?,1]，但约定第二维是 T
            m = x.squeeze(2)
            if m.dim() != 2:
                # 万一不是 [B,T,1] 这种规范的，就当成最后一维聚合
                m = (x.abs().sum(dim=2) > 0).float()
        else:
            # [B,T,K]（K>1） → 对最后一维聚合成 [B,T]
            m = (x.abs().sum(dim=2) > 0).float()

    elif x.dim() == 2:
        # 已经是 [B,T]
        m = x

    elif x.dim() == 1:
        # [T] → 补 B 维
        m = x.unsqueeze(0)

    else:
        raise ValueError(f"to_bt_mask_strict(): unsupported ndim={x.dim()}, shape={tuple(x.shape)}")

    # 对齐长度
    if target_T is not None and m.size(1) != target_T:
        B = m.size(0)
        T = m.size(1)
        if T > target_T:
            m = m[:, :target_T]
        else:
            pad = torch.zeros((B, target_T - T), dtype=m.dtype, device=m.device)
            m = torch.cat([m, pad], dim=1)

    return m.contiguous()


# ============================== losses & metrics ==============================

def masked_mse(pred_btjc: torch.Tensor, tgt_btjc: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
    """
    pred/tgt: [B,T,J,C], mask: [B,T] (1 有效, 0 padding)
    """
    pred = sanitize_btjc(pred_btjc)
    tgt = sanitize_btjc(tgt_btjc)

    # 对齐时间长度
    B, T, J, C = pred.shape
    B2, T2, J2, C2 = tgt.shape
    Tm = min(T, T2)
    pred = pred[:, :Tm]
    tgt = tgt[:, :Tm]

    m = to_bt_mask_strict(mask_bt, target_T=Tm)  # [B,Tm]
    m4 = m[:, :, None, None]  # [B,T,1,1]
    diff2 = (pred - tgt) ** 2
    num = (diff2 * m4).sum()
    den = (m4.sum() * J * C).clamp_min(1.0)
    return num / den


def simple_dtw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    简单 DTW（O(T^2)），CPU 上跑；a,b: [T, D]
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


def chunked_dtw_mean(pred_seq_btjc: torch.Tensor, tgt_seq_btjc: torch.Tensor,
                     max_len: int = 160, chunk: int = 40) -> torch.Tensor:
    """
    把 [B,T,J,C] 里的第一个样本拉平做 DTW（分块求平均），只做一个样本以省时
    """
    pred = sanitize_btjc(pred_seq_btjc)[0]  # [T,J,C]
    tgt = sanitize_btjc(tgt_seq_btjc)[0]

    T = min(pred.shape[0], tgt.shape[0], max_len)
    pred = pred[:T].reshape(T, -1)
    tgt = tgt[:T].reshape(T, -1)
    if T <= 1:
        return torch.tensor(0.0)

    vals = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        if e - s < 2:
            continue
        vals.append(simple_dtw(pred[s:e], tgt[s:e]))
    return torch.stack(vals).mean() if vals else torch.tensor(0.0)


# ============================== filtered dataset ==============================

class FilteredDataset(Dataset):
    """
    只保留最多 target_count 条“拿得到 data 和 conditions”的样本，最多扫描 max_scan 行。
    """
    def __init__(self, base: Dataset, target_count: int = 1, max_scan: Optional[int] = 200):
        self.base = base
        self.idx = []
        N = len(base)
        scan_limit = N if max_scan is None else min(N, max_scan)
        for i in range(scan_limit):
            try:
                it = base[i]
                if isinstance(it, dict) and ("data" in it) and ("conditions" in it):
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
    def __init__(self, num_keypoints: int, num_dims: int, lr=1e-3, weight_decay=0.0, log_dir="logs"):
        super().__init__()
        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints, num_dims_per_keypoint=num_dims
        )
        self.lr = lr
        self.weight_decay = weight_decay

        # for curves
        self.log_dir = log_dir
        self.train_losses = []
        self.val_losses = []
        self.val_dtws = []

    # -------- utilities for one batch --------
    def _prepare_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
          fut_btjc: [B,T,J,C]
          past_btjc:[B,Tp,J,C]
          tgt_mask_bt: [B,T]
          sign_img: [B,3,224,224] (float32)
        均已经过兜底，不会出现 T==0 或 mask 非 2D。
        """
        cond = batch["conditions"]
        fut = sanitize_btjc(batch["data"])
        past = sanitize_btjc(cond["input_pose"])

        # 兜底: 确保时间轴非空
        fut = ensure_non_empty_time(fut)
        past = ensure_non_empty_time(past)

        # mask（可能来自数据，也可能缺失/形状不对）
        if "target_mask" in cond and isinstance(cond["target_mask"], torch.Tensor):
            tgt_mask = to_bt_mask_strict(cond["target_mask"], target_T=fut.size(1))
        else:
            tgt_mask = infer_mask_from_btjc(fut)

        # sign image
        sign_img = cond.get("sign_image")
        if not isinstance(sign_img, torch.Tensor):
            raise ValueError("conditions['sign_image'] must be a tensor [B,3,224,224]")
        sign_img = sign_img.float().contiguous()

        return fut, past, tgt_mask, sign_img

    # -------- Lightning steps --------
    def forward(self, x_bjct, timesteps, past_bjct, sign_img):
        return self.model.forward(x_bjct, timesteps, past_bjct, sign_img)

    def training_step(self, batch, _):
        fut, past, tgt_mask, sign_img = self._prepare_batch(batch)

        # CAMDM 接口: [B,J,C,T]
        x_bjct = btjc_to_bjct(fut)
        past_bjct = btjc_to_bjct(past)
        timesteps = torch.zeros(x_bjct.size(0), dtype=torch.long, device=x_bjct.device)

        out_bjct = self.forward(x_bjct, timesteps, past_bjct, sign_img)  # [B,J,C,Tf]
        pred = bjct_to_btjc(out_bjct)  # [B,T,J,C]

        loss = masked_mse(pred, fut, tgt_mask)
        self.train_losses.append(float(loss.detach().cpu()))
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, _):
        fut, past, tgt_mask, sign_img = self._prepare_batch(batch)

        x_bjct = btjc_to_bjct(fut)
        past_bjct = btjc_to_bjct(past)
        timesteps = torch.zeros(x_bjct.size(0), dtype=torch.long, device=x_bjct.device)

        out_bjct = self.forward(x_bjct, timesteps, past_bjct, sign_img)
        pred = bjct_to_btjc(out_bjct)

        loss = masked_mse(pred, fut, tgt_mask)
        self.val_losses.append(float(loss.detach().cpu()))

        dtw = chunked_dtw_mean(pred, fut)
        self.val_dtws.append(float(dtw.detach().cpu()))

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/dtw", dtw, prog_bar=True)

    def on_fit_end(self):
        # 保存曲线数据
        os.makedirs(self.log_dir, exist_ok=True)
        csv_path = os.path.join(self.log_dir, "minimal_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "train_loss", "val_loss", "val_dtw"])
            steps = max(len(self.train_losses), len(self.val_losses))
            for i in range(steps):
                tr = self.train_losses[i] if i < len(self.train_losses) else ""
                vl = self.val_losses[i] if i < len(self.val_losses) else ""
                vd = self.val_dtws[i] if i < len(self.val_dtws) else ""
                w.writerow([i + 1, tr, vl, vd])

        # 画图
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            if self.train_losses:
                plt.plot(self.train_losses, label="train_loss")
            if self.val_losses:
                plt.plot(self.val_losses, label="val_loss")
            if self.val_dtws:
                plt.plot(self.val_dtws, label="val_dtw")
            plt.xlabel("steps")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, "minimal_curves.png"))
            plt.close()
        except Exception:
            pass  # 没装 matplotlib 也不影响训练

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# ============================== dataloader builder ==============================

def make_loader(data_dir: str, csv_path: str, split: str,
                bs: int, num_workers: int, target_count=1, max_scan=1000):
    base = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        with_metadata=True,
        split=split,
    )
    ds = FilteredDataset(base, target_count=target_count, max_scan=max_scan)
    print(f"[DEBUG] split={split} | batch_size={bs} | len(ds)={len(ds)}")
    return DataLoader(
        ds, batch_size=bs, shuffle=False, num_workers=num_workers,
        pin_memory=False, persistent_workers=False,
        collate_fn=zero_pad_collator
    )


# ============================== main ==============================

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    pl.seed_everything(42, workers=True)

    # 路径（环境变量优先）
    data_dir = os.getenv("DATA_DIR", "/data/yayun/pose_data")  # 你的 raw_poses 根目录
    # 可换回大 CSV：/data/yayun/signwriting-animation/data.csv
    csv_path = os.getenv("CSV_PATH", "/data/yayun/signwriting-animation/mini_data.csv")

    # 过拟合设置（稳定）
    batch_size = 1
    num_workers = 0
    num_keypoints, num_dims = 586, 3
    target_count, max_scan = 1, 200  # 最多扫 200 行，拿到 1 条能用的样本

    train_loader = make_loader(data_dir, csv_path, "train",
                               bs=batch_size, num_workers=num_workers,
                               target_count=target_count, max_scan=max_scan)
    val_loader = train_loader  # 过拟合同一条

    # 先看一眼第一批，确保维度稳定
    fb = next(iter(train_loader))
    print("[DEBUG] keys:", list(fb.keys()))
    print("[DEBUG] cond keys:", list(fb["conditions"].keys()))
    for k, v in fb["conditions"].items():
        if isinstance(v, torch.Tensor):
            print(f"[DEBUG] cond[{k}] ->", tuple(v.shape), "sparse:", v.is_sparse)
    if isinstance(fb["data"], torch.Tensor):
        print("[DEBUG] data ->", tuple(fb["data"].shape), "sparse:", fb["data"].is_sparse)

    model = LitMinimal(num_keypoints=num_keypoints, num_dims=num_dims, log_dir="logs")

    trainer = pl.Trainer(
        max_steps=5,
        limit_train_batches=1,       # 每轮只跑 1 个 batch
        limit_val_batches=1,         # 验证也只跑 1 个 batch
        val_check_interval=1.0, 
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,       # 每步都记
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,    # 避免额外 sanity 验证再次触发随机取样
    )
    trainer.fit(model, train_loader, val_loader)
