import os
import csv
import torch
import numpy as np
import lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from torch.utils.data import DataLoader, Dataset
from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, masked_dtw


def _to_plain_tensor(x):
    """Convert MaskedTensor or custom tensor to plain CPU tensor."""
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().cpu()


def visualize_pose_sequence(seq_btjc, save_path="logs/free_run_vis.png", step=5):
    """
    Visualize a pose sequence (e.g., model prediction).
    seq_btjc: [1,T,J,C]
    """
    seq = _to_plain_tensor(seq_btjc)[0]  # [T,J,C]
    T, J, C = seq.shape
    plt.figure(figsize=(5, 5))
    for t in range(0, T, step):
        pose = seq[t]
        plt.scatter(pose[:, 0], -pose[:, 1], s=8)
    plt.title("Predicted Pose Trajectory (sampled frames)")
    plt.axis("equal")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def _get_pose_header_from_loader(loader):
    ds = loader.dataset
    base = getattr(ds, "base", None) or getattr(ds, "dataset", None)
    # 多解一层常见包装（如 FilteredDataset -> base）
    base = getattr(base, "base", None) or base
    if base is not None and hasattr(base, "header"):
        return base.header
    return getattr(ds, "header", None)


def unnormalize_btjc(x_btjc, header):
    """
    Inverse normalization using pose-format header if available.
    x_btjc: [B,T,J,C] float CPU tensor.
    If no info exists, returns input as-is.
    """
    x = x_btjc.detach().float().cpu()
    if header is None or not hasattr(header, "normalization_info") or header.normalization_info is None:
        return x
    ni = header.normalization_info
    # Try mean/std pattern
    if hasattr(ni, "mean") and hasattr(ni, "std") and ni.mean is not None and ni.std is not None:
        mean = torch.tensor(ni.mean, dtype=x.dtype).view(1, 1, 1, -1)   # [1,1,1,C]
        std  = torch.tensor(ni.std,  dtype=x.dtype).view(1, 1, 1, -1)
        return x * std + mean
    # Try scale/translation pattern
    if hasattr(ni, "scale") and hasattr(ni, "translation") and ni.scale is not None and ni.translation is not None:
        scale = torch.tensor(ni.scale, dtype=x.dtype).view(1, 1, 1, -1)
        trans = torch.tensor(ni.translation, dtype=x.dtype).view(1, 1, 1, -1)
        return x * scale + trans
    return x


def visualize_with_pose_format(pred_btjc, gt_btjc, header, save_path="logs/free_run_posefmt.gif", fps=10):
    """
    Render side-by-side animation using pose-format skeleton edges & colors.
    pred_btjc, gt_btjc: [1,T,J,C] CPU tensors.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    pred = pred_btjc[0].numpy()  # [T,J,C]
    gt   = gt_btjc[0].numpy()    # [T,J,C]
    T, J, C = pred.shape

    edges, colors = [], []
    if header is not None and hasattr(header, "skeleton") and header.skeleton is not None:
        sk = header.skeleton
        if hasattr(sk, "edges") and sk.edges is not None:
            edges = [(int(a), int(b)) for a, b in sk.edges]
        if hasattr(sk, "edge_colors") and sk.edge_colors is not None:
            colors = [tuple(c) for c in sk.edge_colors]
    if not edges:
        edges = [(i, i + 1) for i in range(min(J - 1, 20))]
        colors = [(0.5, 0.5, 0.5)] * len(edges)

    # Axis limits from both sequences
    xy = np.concatenate([pred[..., :2].reshape(-1, 2), gt[..., :2].reshape(-1, 2)], axis=0)
    x_min, y_min = xy.min(axis=0); x_max, y_max = xy.max(axis=0)
    pad = 0.05 * max(x_max - x_min, y_max - y_min, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax_pred, ax_gt = axes
    for ax, title in [(ax_pred, "Predicted (unnormalized)"), (ax_gt, "Ground Truth (unnormalized)")]:
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.axis("off")

    # line artists
    pred_lines = [ax_pred.plot([], [], lw=2, color=colors[i % len(colors)])[0] for i in range(len(edges))]
    gt_lines   = [ax_gt.plot([],   [], lw=2, color=colors[i % len(colors)])[0] for i in range(len(edges))]

    def _set_lines(lines, pose_xy):
        for k, (a, b) in enumerate(edges):
            xa, ya = pose_xy[a, 0], pose_xy[a, 1]
            xb, yb = pose_xy[b, 0], pose_xy[b, 1]
            lines[k].set_data([xa, xb], [ya, yb])

    def _init():
        _set_lines(pred_lines, pred[0, :, :2])
        _set_lines(gt_lines,   gt[0,   :, :2])
        return pred_lines + gt_lines

    def _update(tidx):
        _set_lines(pred_lines, pred[tidx, :, :2])
        _set_lines(gt_lines,   gt[tidx,   :, :2])
        return pred_lines + gt_lines

    ani = animation.FuncAnimation(fig, _update, frames=T, init_func=_init,
                                  interval=1000 // max(1, fps), blit=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"[viz] saved pose-format-style animation to {save_path}")

class FilteredDataset(Dataset):
    """Subset of valid samples for minimal overfit test."""
    def __init__(self, base: Dataset, target_count=4, max_scan=500):
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


def _as_dense_cpu_btjc(x):
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    if getattr(x, "is_sparse", False):
        x = x.to_dense()
    x = x.detach().float().cpu().contiguous()
    if x.dim() == 5:
        x = x[:, :, 0, ...]
    return x


def make_loader(data_dir, csv_path, split, bs, num_workers):
    base = DynamicPosePredictionDataset(
        data_dir=data_dir, csv_path=csv_path, with_metadata=True, split=split
    )
    ds = FilteredDataset(base, target_count=4, max_scan=1000)
    print(f"[DEBUG] split={split} | batch_size={bs} | len(ds)={len(ds)}")
    return DataLoader(
        ds, batch_size=bs, shuffle=True, num_workers=num_workers, collate_fn=zero_pad_collator
    )


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_default_dtype(torch.float32)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data.csv"

    batch_size = 2
    num_workers = 2

    train_loader = make_loader(data_dir, csv_path, "train", bs=batch_size, num_workers=num_workers)
    val_loader = train_loader  # same small subset for validation

    model = LitMinimal(log_dir="logs")

    trainer = pl.Trainer(
        max_steps=500,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=5,
        limit_train_batches=10,
        limit_val_batches=5,
        check_val_every_n_epoch=1,
        deterministic=True,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )

    trainer.fit(model, train_loader, val_loader)

    try:
        model.on_fit_end()
    except Exception as e:
        print("[WARN] on_fit_end() failed:", e)

    # ============ Inference on 1 example for visual check ============
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        cond  = batch["conditions"]

        past_btjc = cond["input_pose"][:1].to(model.device)
        sign_img  = cond["sign_image"][:1].to(model.device)
        fut_gt    = batch["data"][:1].to(model.device)
        mask_bt   = cond["target_mask"][:1].to(model.device)

        print("[GEN] using generate_full_sequence", flush=True)
        gen_btjc = model.generate_full_sequence(
            past_btjc=past_btjc,
            sign_img=sign_img,
            target_mask=mask_bt,
        )

        gen_btjc_cpu = _as_dense_cpu_btjc(gen_btjc)  # [1,T,J,C]
        fut_gt_cpu   = _as_dense_cpu_btjc(fut_gt)    # [1,T,J,C]

        def frame_disp_cpu(x_btjc):  # x: [1,T,J,C]
            x = x_btjc[0]
            if x.size(0) < 2: return 0.0
            dx = x[1:, :, :2] - x[:-1, :, :2]
            return dx.abs().mean().item()

        Tf = gen_btjc_cpu.size(1)
        mv_pred = frame_disp_cpu(gen_btjc_cpu)
        mv_gt   = frame_disp_cpu(fut_gt_cpu)
        print(f"[GEN] Tf={Tf}, mean|Δpred|={mv_pred:.4f}, mean|Δgt|={mv_gt:.4f}", flush=True)

        try:
            dtw_free = masked_dtw(gen_btjc, fut_gt, mask_bt).item()
            print(f"[Free-run] DTW: {dtw_free:.4f}", flush=True)
        except Exception as e:
            print("[Free-run] DTW eval skipped:", e)

        os.makedirs("logs", exist_ok=True)
        torch.save({"gen": gen_btjc_cpu[0], "gt": fut_gt_cpu[0]}, "logs/free_run.pt")
        print("Saved free-run sequences to logs/free_run.pt", flush=True)

        header = _get_pose_header_from_loader(train_loader)
        gen_unnorm = unnormalize_btjc(gen_btjc_cpu, header)  # [1,T,J,C]
        gt_unnorm  = unnormalize_btjc(fut_gt_cpu,  header)   # [1,T,J,C]

        # —— 修复过的调用行（只修这里） ——
        visualize_with_pose_format(
            pred_btjc=gen_unnorm,
            gt_btjc=gt_unnorm,
            header=header,
            save_path="logs/free_run_posefmt.gif",
            fps=10,
        )

        # 可选：简单散点动画（原样保留）
        fig, ax = plt.subplots(figsize=(5, 5))
        sc_pred = ax.scatter([], [], s=15, c="red",  label="Predicted", animated=True)
        sc_gt   = ax.scatter([], [], s=15, c="blue", label="GT",        alpha=0.35, animated=True)
        ax.legend(loc="upper right", frameon=False)
        ax.axis("equal")

        xy = torch.cat([
            gen_btjc_cpu[..., :2].reshape(-1, 2),
            fut_gt_cpu[...,  :2].reshape(-1, 2)
        ], dim=0).numpy()
        x_min, y_min = xy.min(axis=0); x_max, y_max = xy.max(axis=0)
        pad = 0.05 * max(x_max - x_min, y_max - y_min, 1e-6)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

        def _init():
            sc_pred.set_offsets(np.empty((0, 2)))
            sc_gt.set_offsets(np.empty((0, 2)))
            return sc_pred, sc_gt

        def _update(f):
            ax.set_title(f"Free-run prediction  |  frame {f+1}/{len(gen_btjc_cpu[0])}")
            sc_pred.set_offsets(gen_btjc_cpu[0, f, :, :2].numpy())
            sc_gt.set_offsets(  fut_gt_cpu[0,  f, :, :2].numpy())
            return sc_pred, sc_gt

        ani = animation.FuncAnimation(
            fig, _update, frames=max(1, len(gen_btjc_cpu[0])),
            init_func=_init, interval=100, blit=True
        )
        ani.save("logs/free_run_anim.gif", writer="pillow", fps=10)
        plt.close(fig)
        print("Saved free-run animation to logs/free_run_anim.gif", flush=True)
