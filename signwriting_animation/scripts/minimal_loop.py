# signwriting_animation/scripts/minimal_loop.py
import os
from typing import Dict

import torch
import lightning as pl
from torch.utils.data import DataLoader
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion

# --- 可视化与日志 ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor


# ------------------------ losses & metrics ------------------------

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: [B, T, J, C]; mask: [B, T]  (1 有效, 0 pad)
    """
    mask = mask.float()                      # [B,T]
    diff2 = (pred - target) ** 2             # [B,T,J,C]
    m = mask[:, :, None, None]               # [B,T,1,1]
    num = (diff2 * m).sum()
    den = (m.sum() * pred.size(2) * pred.size(3)).clamp_min(1.0)
    return num / den


def simple_dtw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    朴素 DTW，CPU，O(T^2)
    a: [T, D], b: [T', D]  -> 返回标量 tensor
    """
    a = a.detach().cpu()
    b = b.detach().cpu()
    T, D = a.shape
    Tp, Dp = b.shape
    assert D == Dp, "DTW dims mismatch"
    dist = torch.cdist(a, b)  # [T, Tp]
    dp = torch.full((T + 1, Tp + 1), float("inf"))
    dp[0, 0] = 0.0
    for i in range(1, T + 1):
        for j in range(1, Tp + 1):
            cost = dist[i - 1, j - 1]
            dp[i, j] = float(cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]))
    return dp[T, Tp]  # 直接返回 tensor，避免警告


def chunked_dtw_mean(pred_seq: torch.Tensor, tgt_seq: torch.Tensor, max_len: int = 160, chunk: int = 40) -> torch.Tensor:
    """
    分片 DTW，限制长度后分段求 DTW 再平均；pred_seq/tgt_seq: [T, D]
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


# ------------------------ dataloader builder ------------------------

def make_loader(
    data_dir: str,
    csv_path: str,
    split: str,
    batch_size: int,
    num_workers: int,
    num_past: int = 40,
    num_future: int = 20,
    target_count=None,
    max_scan=None,
):
    """
    注意：这里直接使用你 Dataset 的输出，不再额外变换。
    """
    ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=num_past,
        num_future_frames=num_future,
        with_metadata=True,
        split=split,
    )

    # 如果你想小样本快速实验，可裁剪：
    if target_count is not None:
        from torch.utils.data import Subset
        N = len(ds)
        keep = min(N, int(target_count))
        ds = Subset(ds, list(range(keep)))

    print(f"[DEBUG] split={split} | len(ds)={len(ds)} | bs={batch_size} | workers={num_workers}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=zero_pad_collator,
    )


# ------------------------ small viz utils ------------------------

def _btjc_to_np_xy(seq_btjc: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    seq_btjc: [B, T, J, C]  只取 batch=0，返回若干关节的 (x,y) 序列。
    你可以根据真实关键点映射调整 pick_joints 的索引。
    """
    x = seq_btjc[0]  # [T, J, C]
    x = x.detach().cpu().float().numpy()
    pick_joints = {
        "wrist_L": 0,
        "wrist_R": 1,
    }
    curves = {}
    for name, j in pick_joints.items():
        if j < x.shape[1]:
            curves[name] = x[:, j, :2]  # (T, 2)
    return curves


class PlotPredictions(Callback):
    """
    每隔 every_n_epochs 在 val 上画一张 pred vs GT 的 2D 轨迹图
    """
    def __init__(self, log_dir: str, every_n_epochs: int = 5):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(os.path.join(log_dir, "figs"), exist_ok=True)
        self.every = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every != 0:
            return
        val_loader = trainer.val_dataloaders[0]
        batch = next(iter(val_loader))
        cond = batch["conditions"]

        # 统一 float32
        fut      = batch["data"].float()               # [B,T,J,C]
        past     = cond["input_pose"].float()          # [B,Tp,J,C]
        sign_img = cond["sign_image"].float()          # [B,3,224,224]

        # 模型需要 [B,J,C,T]
        x_bjct    = fut.permute(0, 2, 3, 1).contiguous()
        past_bjct = past.permute(0, 2, 3, 1).contiguous()
        t         = torch.zeros(x_bjct.size(0), dtype=torch.long, device=x_bjct.device)

        pl_module.eval()
        with torch.no_grad():
            out_bjct = pl_module.model.forward(x_bjct, t, past_bjct, sign_img)  # [B,J,C,Tf]
            pred     = out_bjct.permute(0, 3, 1, 2).contiguous()               # [B,Tf,J,C]

        gt_curves   = _btjc_to_np_xy(fut)
        pred_curves = _btjc_to_np_xy(pred)

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        for name, arr in gt_curves.items():
            ax.plot(arr[:, 0], arr[:, 1], linestyle='--', label=f'{name} GT')
        for name, arr in pred_curves.items():
            ax.plot(arr[:, 0], arr[:, 1], label=f'{name} Pred')
        ax.set_title(f"Epoch {trainer.current_epoch+1} – 2D Trajectories (sample 0)")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.invert_yaxis()
        ax.legend(loc='best')
        fig.tight_layout()

        png_path = os.path.join(self.log_dir, "figs", f"traj_epoch_{trainer.current_epoch+1:04d}.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

        # 同步到 TensorBoard
        if isinstance(trainer.logger, TensorBoardLogger):
            from PIL import Image
            img = np.array(Image.open(png_path)).transpose(2, 0, 1)  # [C,H,W]
            trainer.logger.experiment.add_image("viz/trajectory", img, global_step=trainer.global_step)


# ------------------------ lightning module ------------------------

class LitMinimal(pl.LightningModule):
    def __init__(self, num_keypoints: int, num_dims: int, lr=2e-4, weight_decay=0.0):
        super().__init__()
        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims
        )
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, fut_btjc, past_btjc, sign_img):
        # 模型 forward 期望: x=[B,J,C,T], past=[B,J,C,Tp], images=[B,3,224,224]
        x_bjct    = fut_btjc.permute(0, 2, 3, 1).contiguous()
        past_bjct = past_btjc.permute(0, 2, 3, 1).contiguous()
        t         = torch.zeros(x_bjct.size(0), dtype=torch.long, device=x_bjct.device)
        out_bjct  = self.model.forward(x_bjct, t, past_bjct, sign_img)     # [B,J,C,Tf]
        pred      = out_bjct.permute(0, 3, 1, 2).contiguous()              # -> [B,Tf,J,C]
        return pred

    def training_step(self, batch, _):
        cond = batch["conditions"]
        fut      = batch["data"].float()
        past     = cond["input_pose"].float()
        tgt_mask = cond["target_mask"].float()
        sign_img = cond["sign_image"].float()

        pred = self(fut, past, sign_img)
        loss = masked_mse(pred, fut, tgt_mask)

        B = fut.size(0)
        self.log("train/loss", loss, prog_bar=True, on_step=True, batch_size=B)
        return loss

    def validation_step(self, batch, _):
        cond = batch["conditions"]
        fut      = batch["data"].float()
        past     = cond["input_pose"].float()
        tgt_mask = cond["target_mask"].float()
        sign_img = cond["sign_image"].float()

        pred = self(fut, past, sign_img)
        loss = masked_mse(pred, fut, tgt_mask)

        # 计算 DTW（batch 里第 0 条）
        b0  = 0
        Tf  = min(pred.shape[1], fut.shape[1])
        pf  = pred[b0, :Tf].reshape(Tf, -1)
        gt  = fut[b0, :Tf].reshape(Tf, -1)
        dtw = chunked_dtw_mean(pf, gt)

        B = fut.size(0)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=B)
        self.log("val/dtw",  dtw,  prog_bar=True, on_epoch=True, batch_size=B)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# ------------------------ main ------------------------

if __name__ == "__main__":
    # 统一 float32，避免 Double/Float 冲突
    torch.set_default_dtype(torch.float32)
    pl.seed_everything(42, workers=True)

    # 指向你的数据（mini/全量二选一）
    DATA_DIR = os.getenv("DATA_DIR", "/data/yayun/pose_data")
    #CSV_PATH = os.getenv("CSV_PATH", "/data/yayun/signwriting-animation/data.csv")
    # 若想快速 sanity：改成 mini_data.csv
    CSV_PATH = os.getenv("CSV_PATH", "/data/yayun/signwriting-animation/mini_data.csv")

    # 训练规模（可以先小，再放开）
    BATCH_SIZE   = int(os.getenv("BATCH_SIZE", "4"))
    NUM_WORKERS  = int(os.getenv("NUM_WORKERS", "2"))
    NUM_PAST     = int(os.getenv("NUM_PAST", "40"))
    NUM_FUTURE   = int(os.getenv("NUM_FUTURE", "20"))
    NUM_KEYPOINTS, NUM_DIMS = 586, 3

    # 小样本快速测试可设置：TARGET_COUNT=64；全量则为 None
    TARGET_COUNT = None
    MAX_SCAN     = None

    # DataLoader
    train_loader = make_loader(
        DATA_DIR, CSV_PATH, "train",
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        num_past=NUM_PAST, num_future=NUM_FUTURE,
        target_count=TARGET_COUNT, max_scan=MAX_SCAN
    )
    val_loader = make_loader(
        DATA_DIR, CSV_PATH, "dev",
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        num_past=NUM_PAST, num_future=NUM_FUTURE,
        target_count=TARGET_COUNT, max_scan=MAX_SCAN
    )

    # 日志与回调
    tb_logger  = TensorBoardLogger(save_dir="outputs", name="runs")
    csv_logger = CSVLogger(save_dir="outputs", name="csv")
    ckpt_cb = ModelCheckpoint(
        dirpath="outputs/checkpoints",
        filename="ep{epoch:03d}-valloss{val/loss:.5f}",
        save_top_k=2, monitor="val/loss", mode="min"
    )
    lr_cb  = LearningRateMonitor(logging_interval="step")
    viz_cb = PlotPredictions(log_dir="outputs", every_n_epochs=5)

    # 模型 & Trainer
    model = LitMinimal(num_keypoints=NUM_KEYPOINTS, num_dims=NUM_DIMS, lr=2e-4, weight_decay=0.0)

    trainer = pl.Trainer(
        max_epochs=5,                 # 先跑几轮看趋势；也可改 max_steps
        accelerator="auto",
        devices=1,
        precision="32-true",
        log_every_n_steps=10,
        enable_checkpointing=True,
        deterministic=True,
        num_sanity_val_steps=1,
        logger=[tb_logger, csv_logger],
        callbacks=[ckpt_cb, lr_cb, viz_cb],
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_loader, val_loader)

