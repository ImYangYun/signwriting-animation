import os
import math
import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion

# ------------------------ utils: shape + losses + metrics ------------------------

def to_btjc(x: torch.Tensor) -> torch.Tensor:
    """
    Robustly coerce tensor into [B, T, J, C].
    - squeeze 前置的 size==1 维直到 <=4 维
    - 若是 3 维 [T,J,C]，补 batch 维
    - 最终必须是 4 维
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.to(torch.float32)

    # 优先挤掉前部多余的 1 维，避免误伤真正的 T==1
    while x.ndim > 4 and x.shape[0] == 1:
        x = x.squeeze(0)
    while x.ndim > 4 and x.shape[1] == 1:
        x = x.squeeze(1)
    while x.ndim > 4 and x.shape[0] == 1:
        x = x.squeeze(0)

    if x.ndim == 3:  # [T,J,C] -> [1,T,J,C]
        x = x.unsqueeze(0)

    if x.ndim != 4:
        raise ValueError(f"to_btjc(): still not 4D after squeeze, got shape={tuple(x.shape)}")

    return x


def btjc_to_bjct(x: torch.Tensor) -> torch.Tensor:
    """[B,T,J,C] -> [B,J,C,T]"""
    x = to_btjc(x)
    return x.permute(0, 2, 3, 1).contiguous()


def to_bt_mask(x: torch.Tensor) -> torch.Tensor:
    """
    将各种形态的 mask 规范到 [B,T] 的 float32。
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.to(torch.float32)

    # squeeze 前置 1 维直到 <=2 维
    while x.ndim > 2 and x.shape[0] == 1:
        x = x.squeeze(0)
    while x.ndim > 2 and x.shape[1] == 1:
        x = x.squeeze(1)
    while x.ndim > 2 and x.shape[0] == 1:
        x = x.squeeze(0)

    if x.ndim == 1:
        x = x.unsqueeze(0)

    if x.ndim != 2:
        raise ValueError(f"to_bt_mask(): still not 2D after squeeze, got shape={tuple(x.shape)}")

    return x


def masked_mse(pred_btjc: torch.Tensor, tgt_btjc: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
    """
    pred/tgt: [B,T,J,C], mask: [B,T] (1=valid, 0=pad)
    """
    pred = to_btjc(pred_btjc)
    tgt = to_btjc(tgt_btjc)
    m = to_bt_mask(mask_bt)  # [B,T]
    diff2 = (pred - tgt) ** 2             # [B,T,J,C]
    num = (diff2 * m[:, :, None, None]).sum()
    den = (m.sum() * pred.size(2) * pred.size(3)).clamp_min(1.0)
    return num / den


def simple_dtw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Very small O(T^2) DTW (CPU). a: [T,D], b:[T',D]
    返回标量 tensor
    """
    a = a.detach().cpu().to(torch.float32)
    b = b.detach().cpu().to(torch.float32)
    T, D = a.shape
    Tp, Dp = b.shape
    if D != Dp:
        raise ValueError(f"DTW dims mismatch: {D} vs {Dp}")
    dist = torch.cdist(a, b)  # [T,Tp]
    dp = torch.full((T + 1, Tp + 1), float("inf"))
    dp[0, 0] = 0.0
    for i in range(1, T + 1):
        for j in range(1, Tp + 1):
            cost = dist[i - 1, j - 1].item()
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return torch.tensor(dp[T, Tp], dtype=torch.float32)


def chunked_dtw_mean(pred_seq_t_d: torch.Tensor, tgt_seq_t_d: torch.Tensor,
                     max_len: int = 160, chunk: int = 40) -> torch.Tensor:
    """
    限长 + 分块 DTW 的均值。输入 [T,D]。
    """
    T = min(pred_seq_t_d.shape[0], max_len)
    a = pred_seq_t_d[:T]
    b = tgt_seq_t_d[:T]
    if T <= 1:
        return torch.tensor(0.0, dtype=torch.float32)
    vals = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        if e - s < 2:
            continue
        vals.append(simple_dtw(a[s:e], b[s:e]))
    return torch.stack(vals).mean() if vals else torch.tensor(0.0, dtype=torch.float32)


# ------------------------ a tiny filtered dataset (optional but helpful) ------------------------

class FilteredDataset(Dataset):
    """
    从底层 dataset 中筛出少量“长度足够”的样本，避免一次性扫全量导致内存/IO 压力。
    """
    def __init__(self, base: Dataset, min_past: int, min_future: int,
                 target_count: int = 1, max_scan: Optional[int] = 1000):
        self.base = base
        self.idx = []
        N = len(base)
        lim = N if max_scan is None else min(N, max_scan)
        for i in range(lim):
            try:
                it = base[i]
                pm = to_bt_mask(it["conditions"]["input_mask"])
                fm = to_bt_mask(it["conditions"]["target_mask"])
                if int(pm.sum().item()) < min_past:
                    continue
                if int(fm.sum().item()) < min_future:
                    continue
                self.idx.append(i)
                if target_count and len(self.idx) >= target_count:
                    break
            except Exception:
                continue
        if not self.idx:
            # 兜底：至少保留第 0 个，避免完全找不到
            self.idx = [0]

    def __len__(self): return len(self.idx)
    def __getitem__(self, i): return self.base[self.idx[i]]


def make_loader(
    data_dir: str,
    csv_path: str,
    split: str,
    bs: int,
    num_past: int = 10,
    num_future: int = 5,
    target_count: int = 1,
    max_scan: int = 1000,
):
    base = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=num_past,
        num_future_frames=num_future,
        with_metadata=False,   # 简化 batch 结构，够用
        split=split,
    )
    ds = FilteredDataset(
        base, min_past=num_past, min_future=num_future,
        target_count=target_count, max_scan=max_scan
    )
    print(f"[DEBUG] split={split} | len(ds)={len(ds)}")
    return DataLoader(
        ds, batch_size=bs, shuffle=False,
        num_workers=0, pin_memory=False, persistent_workers=False,
        collate_fn=zero_pad_collator,
    )


# ------------------------ Lightning module ------------------------

class LitMinimal(pl.LightningModule):
    def __init__(self, num_keypoints: int, num_dims: int, lr=1e-3, out_dir: str = "./outputs"):
        super().__init__()
        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints, num_dims_per_keypoint=num_dims
        )
        self.lr = lr
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # histories
        self.train_losses = []
        self.val_losses = []
        self.val_dtws = []

    def forward(self, x_bjct, past_bjct, sign_img):
        # 你的模型 forward 接口（来自 models.py）
        B = x_bjct.size(0)
        timesteps = torch.zeros(B, dtype=torch.long, device=x_bjct.device)
        out_bjct = self.model(
            x=x_bjct,
            timesteps=timesteps,
            past_motion=past_bjct,
            signwriting_im_batch=sign_img,
        )  # 期望 [B,J,C,T_f]
        return out_bjct

    def training_step(self, batch: Dict[str, Any], _):
        cond = batch["conditions"]
        # 规范形状
        past_btjc   = to_btjc(cond["input_pose"])   # [B,Tp,J,C]
        fut_btjc    = to_btjc(batch["data"])        # [B,Tf,J,C]
        in_mask_bt  = to_bt_mask(cond["input_mask"])
        out_mask_bt = to_bt_mask(cond["target_mask"])
        sign_img    = cond["sign_image"].to(torch.float32)  # [B,3,224,224]

        # [B,T,J,C] -> [B,J,C,T]
        past_bjct = btjc_to_bjct(past_btjc)
        fut_bjct  = btjc_to_bjct(fut_btjc)

        pred_bjct = self.forward(fut_bjct, past_bjct, sign_img)  # [B,J,C,T_f]
        pred_btjc = pred_bjct.permute(0, 3, 1, 2).contiguous()   # [B,T_f,J,C]

        loss = masked_mse(pred_btjc, fut_btjc, out_mask_bt)
        self.train_losses.append(float(loss.detach().cpu().item()))
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], _):
        cond = batch["conditions"]
        past_btjc   = to_btjc(cond["input_pose"])
        fut_btjc    = to_btjc(batch["data"])
        out_mask_bt = to_bt_mask(cond["target_mask"])
        sign_img    = cond["sign_image"].to(torch.float32)

        past_bjct = btjc_to_bjct(past_btjc)
        fut_bjct  = btjc_to_bjct(fut_btjc)

        with torch.no_grad():
            pred_bjct = self.forward(fut_bjct, past_bjct, sign_img)
            pred_btjc = pred_bjct.permute(0, 3, 1, 2).contiguous()

        loss = masked_mse(pred_btjc, fut_btjc, out_mask_bt)
        self.val_losses.append(float(loss.detach().cpu().item()))
        self.log("val/loss", loss, prog_bar=True)

        # 取第一个样本做 DTW
        b0 = 0
        Tf = int(out_mask_bt[b0].sum().item())
        Tf = min(Tf, pred_btjc.shape[1])
        pred_flat = pred_btjc[b0, :Tf].reshape(Tf, -1)
        gt_flat   = fut_btjc[b0, :Tf].reshape(Tf, -1)
        dtw = chunked_dtw_mean(pred_flat, gt_flat)
        self.val_dtws.append(float(dtw.detach().cpu().item()))
        self.log("val/dtw", dtw, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_fit_end(self):
        # 画图
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Loss 曲线
            plt.figure()
            if self.train_losses:
                plt.plot(self.train_losses, label="train_loss")
            if self.val_losses:
                plt.plot([None]*(len(self.train_losses)-len(self.val_losses)) + self.val_losses, label="val_loss")
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, "loss_curve.png"))
            plt.close()

            # DTW 曲线
            if self.val_dtws:
                plt.figure()
                plt.plot(self.val_dtws, label="val_dtw")
                plt.xlabel("val step")
                plt.ylabel("dtw")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, "dtw_curve.png"))
                plt.close()
        except Exception as e:
            print(f"[WARN] plotting failed: {e}")


# ------------------------ main ------------------------

if __name__ == "__main__":
    # 统一默认 dtype，避免 Double/Float 混用
    torch.set_default_dtype(torch.float32)
    pl.seed_everything(42, workers=True)

    # 数据路径（可用环境变量覆盖）
    data_dir = os.getenv("DATA_DIR", "/data/yayun/pose_data")  # 你的 raw_poses 所在上级目录
    # 优先用 mini_data；没有就用 data.csv
    csv_default = "/data/yayun/signwriting-animation/mini_data.csv"
    if not os.path.exists(csv_default):
        csv_default = "/data/yayun/signwriting-animation/data.csv"
    csv_path = os.getenv("CSV_PATH", csv_default)

    # 过拟合/小步验证配置
    batch_size = 1
    num_keypoints, num_dims = 586, 3
    num_past, num_future = 10, 5
    target_count, max_scan = 1, 1000

    # dataloader（train/val 同一条样本，方便 sanity check）
    train_loader = make_loader(
        data_dir, csv_path, "train",
        bs=batch_size, num_past=num_past, num_future=num_future,
        target_count=target_count, max_scan=max_scan
    )
    val_loader = train_loader

    # 先拉一个 batch 看看维度
    first_batch = next(iter(train_loader))
    shapes = {}
    for k, v in first_batch.items():
        if isinstance(v, dict):
            shapes[k] = {kk: (vv.shape if hasattr(vv, "shape") else type(vv)) for kk, vv in v.items()}
        else:
            shapes[k] = v.shape if hasattr(v, "shape") else type(v)
    print("[DEBUG] first batch shapes:", shapes)

    # 模型 + 训练
    out_dir = os.getenv("OUT_DIR", "./outputs")
    model = LitMinimal(num_keypoints=num_keypoints, num_dims=num_dims, lr=1e-3, out_dir=out_dir)

    trainer = pl.Trainer(
        max_steps=600,
        accelerator="auto",
        devices=1,
        log_every_n_steps=5,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,  # 避免额外的 sanity loop 扰动
    )
    trainer.fit(model, train_loader, val_loader)

    print(f"[INFO] curves saved under: {out_dir}")

