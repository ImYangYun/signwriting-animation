import os
import random
from typing import Optional
import pytest
import torch
import lightning as pl
from torch.utils.data import Dataset, DataLoader
from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import (
    DynamicPosePredictionDataset,
    get_num_workers,
)
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    MSE over valid (non-padded) frames only.
    pred/target: [B, T, ...]; mask: [B, T] with 1 valid, 0 pad
    """
    diff = (pred - target) ** 2
    while diff.dim() > 2:
        diff = diff.mean(dim=-1)  # reduce pose dims -> [B, T]
    diff = diff * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def simple_dtw(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Tiny CPU DTW (O(T^2)) for validation hook.
    a: [T, D], b: [T', D]
    """
    a = a.detach().cpu()
    b = b.detach().cpu()
    T, D = a.shape
    Tp, Dp = b.shape
    assert D == Dp, "DTW feature dims must match."
    dist = torch.cdist(a, b)  # [T, Tp]
    dp = torch.full((T + 1, Tp + 1), float("inf"))
    dp[0, 0] = 0.0
    for i in range(1, T + 1):
        for j in range(1, Tp + 1):
            cost = float(dist[i - 1, j - 1].item())
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[T, Tp].item())


def chunked_dtw_mean(pred_seq: torch.Tensor, tgt_seq: torch.Tensor,
                     max_len: int = 160, chunk: int = 40) -> float:
    """
    Chunked DTW (fluent-pose style):
    - Cap sequence length to max_len
    - Split into chunks and average per-chunk DTW
    """
    T = min(pred_seq.shape[0], max_len)
    pred_seq = pred_seq[:T]
    tgt_seq = tgt_seq[:T]
    if T <= 1:
        return 0.0
    vals = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        if e - s < 2:
            continue
        vals.append(simple_dtw(pred_seq[s:e], tgt_seq[s:e]))
    return float(sum(vals) / len(vals)) if vals else 0.0


class FilteredDataset(Dataset):
    """
    Wrap a base dataset; pre-scan to keep only valid indices (missing files/too-short pruned).
    This avoids None-batches that Lightning can't handle.
    """
    def __init__(self, base: Dataset, min_past: int, min_future: int,
                 target_count: int = 64, max_scan: Optional[int] = None):
        self.base = base
        self.min_past = min_past
        self.min_future = min_future
        self.valid_idx = []

        N = len(base)
        scan_limit = N if max_scan is None else min(N, max_scan)
        for i in range(scan_limit):
            try:
                item = base[i]
                past_mask = item.get("past_mask", None)
                future_mask = item.get("future_mask", None)
                if past_mask is not None and int(past_mask.sum().item()) < min_past:
                    continue
                if future_mask is not None and int(future_mask.sum().item()) < min_future:
                    continue
                self.valid_idx.append(i)
                if len(self.valid_idx) >= target_count:
                    break
            except Exception:
                continue  # skip broken rows

        if len(self.valid_idx) == 0:
            raise RuntimeError("No valid samples found after filtering.")

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        return self.base[self.valid_idx[idx]]


def make_loader(data_dir, csv_path, num_past, num_future, batch_size, seed: int = 42):
    base = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=num_past,
        num_future_frames=num_future,
        with_metadata=True,
        split="train",
    )
    ds = FilteredDataset(base, min_past=num_past, min_future=num_future,
                         target_count=64, max_scan=2000)
    # Fixed shuffle order via generator for full determinism
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, generator=g,
        collate_fn=zero_pad_collator,
        num_workers=min(2, get_num_workers()),
        pin_memory=False,
    )


class LitSWTPoseWithDTW(pl.LightningModule):
    """
    Lightning wrapper for short overfitting/sanity training on real data with DTW tracking.
    Keeps in-memory histories for pytest assertions.
    """
    def __init__(self, base_model: torch.nn.Module, lr: float = 2e-4):
        super().__init__()
        self.model = base_model
        self.lr = lr
        self.loss_hist = []
        self.len_mae_hist = []
        self.val_dtw_hist = []  

    def forward(self, past, past_mask):
        return self.model(past_motion=past, past_mask=past_mask, return_dict=True)

    def training_step(self, batch, batch_idx):
        past = batch["past_pose"]
        future = batch["future_pose"]
        past_mask = batch["past_mask"].float()
        future_mask = batch["future_mask"].float()

        out = self(past, past_mask)
        pred_future = out["pred_future"]
        pred_len = out["pred_len"].squeeze(-1) if out["pred_len"].dim() > 1 else out["pred_len"]
        target_len = future_mask.sum(dim=1)

        pose_loss = masked_mse(pred_future, future, future_mask)
        len_mae = (pred_len - target_len).abs().mean()
        loss = pose_loss + 0.01 * len_mae

        # Store histories for pytest assertions
        self.loss_hist.append(float(loss.detach().cpu().item()))
        self.len_mae_hist.append(float(len_mae.detach().cpu().item()))
        self.log_dict({"train/loss": loss, "train/len_mae": len_mae}, prog_bar=False, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


@pytest.mark.parametrize("batch_size, max_steps", [(2, 60)])
def test_real_data_overfitting_lightning_dtw(batch_size, max_steps):
    """
    Build loader/model, run a short Lightning training, then assert:
    - total loss decreases
    - length-MAE improves or is small
    - DTW improves (soft) on a fixed baseline batch
    """
    pl.seed_everything(42, workers=True)

    # (Optional) Stronger determinism—uncomment if you need bitwise reproducibility:
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # or ":4096:2"
    # torch.use_deterministic_algorithms(True)

    data_dir = os.getenv("DATA_DIR", "/scratch/yayun/pose_data")
    csv_path = os.getenv("CSV_PATH", "/data/yayun/signwriting-animation/data.csv")
    if not os.path.exists(data_dir):
        pytest.skip(f"DATA_DIR not found: {data_dir}")
    if not os.path.exists(csv_path):
        pytest.skip(f"CSV_PATH not found: {csv_path}")

    num_past, num_future = 40, 20
    num_keypoints, num_dims = 586, 3

    # Deterministic, filtered loader with fixed shuffle
    loader = make_loader(data_dir, csv_path, num_past, num_future, batch_size, seed=42)

    # Fixed baseline batch for DTW before/after training
    try:
        baseline_batch = next(iter(loader))
    except StopIteration:
        pytest.skip("No valid batches after filtering.")
        return

    device = "gpu" if torch.cuda.is_available() else "cpu"
    base_model = SignWritingToPoseDiffusion(
        num_keypoints=num_keypoints,
        num_dims_per_keypoint=num_dims,
    )
    lit = LitSWTPoseWithDTW(base_model)

    # Initial DTW on baseline
    lit.eval()
    with torch.no_grad():
        past0 = baseline_batch["past_pose"]
        fut0 = baseline_batch["future_pose"]
        past_mask0 = baseline_batch["past_mask"].float()
        fut_mask0 = baseline_batch["future_mask"].float()
        out0 = lit(past0, past_mask0)
        b0 = 0
        tf0 = int(fut_mask0[b0].sum().item())
        pf0 = out0["pred_future"][b0, :tf0].reshape(tf0, -1)
        gt0 = fut0[b0, :tf0].reshape(tf0, -1)
        dtw_init = chunked_dtw_mean(pf0, gt0)
        lit.val_dtw_hist.append(dtw_init)

    # Train shortly (deterministic=True pairs with seed_everything)
    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator=device,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        deterministic=True,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
    )
    trainer.fit(lit, train_dataloaders=loader)

    # DTW again on the SAME baseline batch
    lit.eval()
    with torch.no_grad():
        past1 = baseline_batch["past_pose"]
        fut1 = baseline_batch["future_pose"]
        past_mask1 = baseline_batch["past_mask"].float()
        fut_mask1 = baseline_batch["future_mask"].float()
        out1 = lit(past1, past_mask1)
        b0 = 0
        tf1 = int(fut_mask1[b0].sum().item())
        pf1 = out1["pred_future"][b0, :tf1].reshape(tf1, -1)
        gt1 = fut1[b0, :tf1].reshape(tf1, -1)
        dtw_end = chunked_dtw_mean(pf1, gt1)
        lit.val_dtw_hist.append(dtw_end)

    lh = lit.loss_hist
    assert len(lh) > 0, "No training steps executed."
    k = max(1, len(lh) // 10)
    start_loss = sum(lh[:k]) / k
    end_loss = sum(lh[-k:]) / k
    assert (end_loss < start_loss * 0.8) or (end_loss < start_loss - 0.05), \
        f"Loss did not decrease enough: {start_loss:.4f} -> {end_loss:.4f}"

    lmh = lit.len_mae_hist
    k2 = max(1, len(lmh) // 10)
    start_len = sum(lmh[:k2]) / k2
    end_len = sum(lmh[-k2:]) / k2
    assert (end_len <= start_len * 0.85) or (end_len <= 2.0), \
        f"Length MAE not improved/enough: {start_len:.3f} -> {end_len:.3f}"

    assert len(lit.val_dtw_hist) >= 2, "DTW history missing."
    s, e = lit.val_dtw_hist[0], lit.val_dtw_hist[-1]
    assert (e <= s * 0.9) or (e <= 1e-4), f"DTW not improved/enough: {s:.4f} -> {e:.4f}"
