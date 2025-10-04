# signwriting_animation/scripts/minimal_loop.py
import os
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import lightning as pl
from torch.utils.data import DataLoader
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion


# =====================
# dtype & utils
# =====================
def set_global_float32():
    torch.set_default_dtype(torch.float32)
    # 让所有新建张量默认 float32（避免 numpy -> double 带来的混用）
    torch.set_default_tensor_type(torch.FloatTensor)


def btjc_to_bjct(x: torch.Tensor) -> torch.Tensor:
    """
    dataset 输出: [B, T, J, C]  -> 模型期望: [B, J, C, T]
    并统一成 float32 连续内存。
    """
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor [B,T,J,C], got {x.shape} (ndim={x.ndim})")
    return x.permute(0, 2, 3, 1).contiguous().float()


def sanitize_btjc(x: torch.Tensor) -> torch.Tensor:
    """确保是 dense float32 连续内存（有些场景会返回稀疏/非连续 Tensor）"""
    if x.is_sparse:
        x = x.to_dense()
    if x.dtype != torch.float32:
        x = x.float()
    if not x.is_contiguous():
        x = x.contiguous()
    return x


def masked_mse_btjc(pred_btjc: torch.Tensor, tgt_btjc: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
    """
    pred/tgt: [B, T, J, C], mask: [B, T]  -> 有效帧上的 MSE
    """
    pred_btjc = sanitize_btjc(pred_btjc)
    tgt_btjc = sanitize_btjc(tgt_btjc)
    mask_bt = mask_bt.float()

    diff2 = (pred_btjc - tgt_btjc) ** 2  # [B,T,J,C]
    m = mask_bt[:, :, None, None]        # [B,T,1,1]
    num = (diff2 * m).sum()
    den = (m.sum() * pred_btjc.size(2) * pred_btjc.size(3)).clamp_min(1.0)
    return num / den


def simple_dtw(a_td: torch.Tensor, b_td: torch.Tensor) -> torch.Tensor:
    """
    朴素 DTW（CPU，上小样本 & 片段即可）
    a/b: [T, D]
    """
    a = a_td.detach().cpu().float()
    b = b_td.detach().cpu().float()
    T, D = a.shape
    Tp, Dp = b.shape
    if D != Dp:
        raise ValueError(f"DTW feature dims mismatch: {D} vs {Dp}")

    dist = torch.cdist(a, b)  # [T, Tp]
    dp = torch.full((T + 1, Tp + 1), float("inf"))
    dp[0, 0] = 0.0
    for i in range(1, T + 1):
        for j in range(1, Tp + 1):
            cost = float(dist[i - 1, j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return torch.tensor(dp[T, Tp], dtype=torch.float32)


def chunked_dtw_mean(btjc_pred: torch.Tensor, btjc_gt: torch.Tensor, mask_bt: torch.Tensor,
                     max_len: int = 160, chunk: int = 40) -> torch.Tensor:
    """
    把 [B,T,J,C] 的第一个样本拉直成 [T,D]，做分片 DTW 并平均。
    """
    b0 = 0
    T = int(mask_bt[b0].sum().item())
    T = min(T, btjc_pred.shape[1], max_len)
    if T <= 1:
        return torch.tensor(0.0)

    # [T, D] 其中 D = J*C
    pf_td = btjc_pred[b0, :T].reshape(T, -1)
    gt_td = btjc_gt[b0, :T].reshape(T, -1)

    vals = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        if e - s < 2:
            continue
        vals.append(simple_dtw(pf_td[s:e], gt_td[s:e]))
    return torch.stack(vals).mean() if vals else torch.tensor(0.0)


# =====================
# Data
# =====================
def make_loader(data_dir: str, csv_path: str, split: str,
                bs: int, num_past: int, num_future: int) -> DataLoader:
    ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=num_past,
        num_future_frames=num_future,
        with_metadata=True,
        split=split,
    )
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(split == "train"),
        num_workers=0,                 # 小批量 + 省内存
        pin_memory=False,
        persistent_workers=False,
        collate_fn=zero_pad_collator,
    )


# =====================
# Lightning Module
# =====================
class LitSWTDiffusion(pl.LightningModule):
    def __init__(self, num_keypoints: int, num_dims: int, lr: float = 1e-3):
        super().__init__()
        # 你的自定义模型
        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims
        )
        self.lr = lr

        # 运行中曲线缓存（同时我们也会写 CSV）
        self.train_loss_hist = []
        self.val_loss_hist = []
        self.val_dtw_hist = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def _forward_pose(self, fut_btjc: torch.Tensor, past_bjct: torch.Tensor, sign_b3hw: torch.Tensor) -> torch.Tensor:
        """
        用你模型的 interface：输入:
          - x: 未来段（噪声/占位，最简我们直接用 GT 形状的 0 张量）
          - timesteps: 这里先给 0（最小可运行版本）
          - y: dict 里放 sign_image / input_pose（注意形状）
        输出:
          - 预测的未来段（[B,J,C,Tf]）
        """
        B, T, J, C = fut_btjc.shape

        # 模型期望: x: [B,J,C,Tf]，past: [B,J,C,Tp]，image: [B,3,224,224]
        x_bjct = btjc_to_bjct(torch.zeros_like(fut_btjc))     # 这里用 0 噪声做占位
        past_bjct = sanitize_btjc(past_bjct)
        sign_b3hw = sign_b3hw.float()

        timesteps = torch.zeros((B,), dtype=torch.long, device=x_bjct.device)
        y = {
            "sign_image": sign_b3hw,
            "input_pose": past_bjct,
        }
        pred_bjct = self.model.interface(x_bjct, timesteps, y)  # [B,J,C,Tf]
        # 回到 [B,T,J,C]
        return pred_bjct.permute(0, 3, 1, 2).contiguous()

    def training_step(self, batch: Dict, _):
        # 取出 batch
        fut_btjc = sanitize_btjc(batch["data"])  # [B,Tf,J,C]
        cond = batch["conditions"]
        past_btjc = sanitize_btjc(cond["input_pose"])  # [B,Tp,J,C]
        # 变换 past -> [B,J,C,Tp]
        past_bjct = btjc_to_bjct(past_btjc)
        sign_b3hw = cond["sign_image"].float()  # [B,3,224,224]
        fut_mask_bt = cond["target_mask"].float()  # [B,Tf]

        # 前向
        pred_btjc = self._forward_pose(fut_btjc, past_bjct, sign_b3hw)
        loss = masked_mse_btjc(pred_btjc, fut_btjc, fut_mask_bt)

        self.train_loss_hist.append(float(loss.detach().cpu()))
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: Dict, _):
        fut_btjc = sanitize_btjc(batch["data"])
        cond = batch["conditions"]
        past_bjct = btjc_to_bjct(sanitize_btjc(cond["input_pose"]))
        sign_b3hw = cond["sign_image"].float()
        fut_mask_bt = cond["target_mask"].float()

        pred_btjc = self._forward_pose(fut_btjc, past_bjct, sign_b3hw)
        loss = masked_mse_btjc(pred_btjc, fut_btjc, fut_mask_bt)
        dtw = chunked_dtw_mean(pred_btjc, fut_btjc, fut_mask_bt, max_len=160, chunk=40)

        self.val_loss_hist.append(float(loss.detach().cpu()))
        self.val_dtw_hist.append(float(dtw.detach().cpu()))
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/dtw", dtw, prog_bar=True)


# =====================
# CSV logger (简单易取)
# =====================
class SimpleCSVLogger(pl.Callback):
    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # header
        if not self.csv_path.exists():
            self.csv_path.write_text("step,split,loss,dtw\n")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        loss = float(pl_module.train_loss_hist[-1]) if pl_module.train_loss_hist else float("nan")
        with self.csv_path.open("a") as f:
            f.write(f"{step},train,{loss},\n")

    def on_validation_epoch_end(self, trainer, pl_module):
        step = trainer.global_step
        loss = pl_module.val_loss_hist[-1] if pl_module.val_loss_hist else float("nan")
        dtw = pl_module.val_dtw_hist[-1] if pl_module.val_dtw_hist else float("nan")
        with self.csv_path.open("a") as f:
            f.write(f"{step},val,{loss},{dtw}\n")


if __name__ == "__main__":
    set_global_float32()
    pl.seed_everything(42, workers=True)

    # --- 路径：可用环境变量覆盖 ---
    DATA_DIR = os.getenv("DATA_DIR", "/data/yayun/pose_data")  # 你的原始 pose 根目录
    CSV_PATH = os.getenv("CSV_PATH", "/data/yayun/signwriting-animation/mini_data.csv")  # 也可切 data.csv

    # --- 形状参数（与你的数据保持一致）---
    NUM_KEYPOINTS, NUM_DIMS = 586, 3
    NUM_PAST, NUM_FUTURE = 10, 5   # mini 版本先小窗口
    BATCH_SIZE = 1

    # --- dataloaders ---
    train_loader = make_loader(DATA_DIR, CSV_PATH, "train", BATCH_SIZE, NUM_PAST, NUM_FUTURE)
    # 用 train 里的同一条做 sanity-val：如果你有 dev split，建议切成 dev
    val_loader = make_loader(DATA_DIR, CSV_PATH, "dev",   BATCH_SIZE, NUM_PAST, NUM_FUTURE)

    # --- 模型 & 记录 ---
    model = LitSWTDiffusion(num_keypoints=NUM_KEYPOINTS, num_dims=NUM_DIMS, lr=1e-3)
    csv_logger = SimpleCSVLogger(csv_path="logs/minimal_metrics.csv")

    # --- Trainer ---
    trainer = pl.Trainer(
        max_steps=int(os.getenv("MAX_STEPS", "600")),   # 你之前用 600，先保持
        accelerator="auto",
        devices=1,
        precision="32-true",    # 👈 强制全程 float32（更稳）
        log_every_n_steps=5,
        enable_checkpointing=False,
        deterministic=True,
        callbacks=[csv_logger],
    )

    # --- 先拿一批做形状检查（快速 fail fast）---
    try:
        bt = next(iter(train_loader))
        _ = bt["data"].shape
    except Exception as e:
        raise RuntimeError(f"First batch failed: {repr(e)}")

    # --- Train ---
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("\n[Done] Metrics CSV saved to: logs/minimal_metrics.csv")
    print("     Columns: step,split,loss,dtw")

