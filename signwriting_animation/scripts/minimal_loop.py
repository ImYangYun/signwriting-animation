# -*- coding: utf-8 -*-
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
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, masked_dtw


def _to_plain_tensor(x):
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().cpu()

def _as_dense_cpu_btjc(x):
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    if getattr(x, "is_sparse", False):
        x = x.to_dense()
    x = x.detach().float().cpu().contiguous()
    if x.dim() == 5:  # [B,T,P,J,C] -> 取 P=0
        x = x[:, :, 0, ...]
    return x

def visualize_pose_sequence(seq_btjc, save_path="logs/free_run_vis.png", step=5):
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

# ============== Header 获取 ==============
def _probe_header_from_csv(csv_path, data_dir):
    """从 CSV 中任取一条 .pose 读 header。若你的列名不是 'pose' 请改这里。"""
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get("pose", "")
                if isinstance(p, str) and p.lower().endswith(".pose"):
                    full = p if os.path.isabs(p) else os.path.join(data_dir, p)
                    if not os.path.exists(full):
                        continue
                    with open(full, "rb") as pf:
                        return Pose.read(pf).header
    except Exception as e:
        print("[probe] failed to read header from csv:", e)
    return None

def _get_pose_header_from_loader(loader):
    ds = loader.dataset
    for _ in range(3):
        if hasattr(ds, "header") and ds.header is not None:
            return ds.header
        base = getattr(ds, "base", None) or getattr(ds, "dataset", None)
        if base is None:
            break
        ds = base
    try:
        sample = loader.dataset[0]
        if isinstance(sample, dict):
            meta = sample.get("metadata", {}) or {}
            return meta.get("pose_header") or meta.get("header")
    except Exception:
        pass
    return None

# ============== 诊断 + 可视化辅助 ==============
def _header_joint_count(h):
    if h is None:
        return None
    try:
        # 新版 pose-format
        return sum(getattr(c, "points", 0) or 0 for c in h.components)
    except Exception:
        try:
            # 旧版字段名
            return sum(len(c.joints) for c in h.components)
        except Exception:
            return None

def _dump_stats(tag, x_btjc):
    x = x_btjc[0, :, :, :2]
    print(f"[{tag}] shape={tuple(x_btjc.shape)} "
          f"min={x.min().item():.6f} max={x.max().item():.6f} "
          f"nan={torch.isnan(x).any().item()} "
          f"frameΔ={(x[1:]-x[:-1]).abs().mean().item():.6f}", flush=True)

def normalize_for_viz(x_btjc, eps=1e-8):
    """把坐标线性拉到 [0.1, 0.9] 区间，仅用于画图，不影响评估。"""
    x = x_btjc.clone()
    x2 = x[..., :2]
    x_min = x2.amin(dim=(1, 2, 3), keepdim=True)
    x_max = x2.amax(dim=(1, 2, 3), keepdim=True)
    scale = (x_max - x_min).clamp_min(eps)
    x2 = (x2 - x_min) / scale
    x2 = 0.1 + 0.8 * x2
    x[..., :2] = x2
    # 清 NaN
    x[torch.isnan(x)] = 0.0
    return x

def save_scatter(seq_btjc, path, label):
    """完全独立于 header 的散点动图兜底。"""
    x = seq_btjc[0, :, :, :2].cpu().numpy()
    x = x - x.mean(axis=(0, 1), keepdims=True)
    M = float(np.abs(x).max())
    if M < 1e-8:
        M = 1.0
    x = x / M
    T = x.shape[0]
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter([], [], s=15)
    ax.axis("equal")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)

    def _init():
        sc.set_offsets(np.empty((0, 2)))
        return sc,

    def _update(f):
        ax.set_title(f"{label}: frame {f+1}/{T}")
        sc.set_offsets(x[f])
        return sc,

    ani = animation.FuncAnimation(fig, _update, frames=T, init_func=_init, interval=100, blit=True)
    ani.save(path, writer="pillow", fps=10)
    plt.close(fig)
    print(f"[scatter] saved -> {path}")

# ============== pose-format 构造 & 可视化 ==============
def btjc_to_pose(x_btjc, header, fps=25, conf_btj=None):
    """
    x_btjc: [1,T,J,C] -> [T,1,J,C]
    conf:   [1,T,J]   -> [T,1,J]
    """
    x = x_btjc[0].detach().cpu().numpy().astype(np.float32)     # [T,J,C]
    x = x[:, np.newaxis, :, :]                                  # [T,1,J,C]
    if conf_btj is None:
        conf = np.ones((x.shape[0], x.shape[2]), dtype=np.float32)
    else:
        conf = conf_btj[0].detach().cpu().numpy().astype(np.float32)
    conf = conf[:, np.newaxis, :]                               # [T,1,J]
    body = NumPyPoseBody(fps=fps, data=x, confidence=conf)
    return Pose(header, body)

def save_with_pose_visualizer(pred_btjc, gt_btjc, header, fps=12):
    pred_pose = btjc_to_pose(pred_btjc, header, fps=fps)
    gt_pose   = btjc_to_pose(gt_btjc,   header, fps=fps)
    # 2D 可视化移除 3D 组件
    try:
        pred_pose = pred_pose.remove_components(["POSE_WORLD_LANDMARKS"])
        gt_pose   = gt_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    except Exception:
        pass
    os.makedirs("logs", exist_ok=True)
    viz_pred = PoseVisualizer(pred_pose)
    viz_gt   = PoseVisualizer(gt_pose)
    viz_pred.save_gif("logs/free_run_posefmt_pred.gif", viz_pred.draw())
    viz_gt.save_gif("logs/free_run_posefmt_gt.gif",   viz_gt.draw())
    print("[viz] saved -> logs/free_run_posefmt_pred.gif / _gt.gif")

# ------------------------
# Dataset / Loader：保持原样
# ------------------------
class FilteredDataset(Dataset):
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
    def __len__(self): return len(self.idx)
    def __getitem__(self, i): return self.base[self.idx[i]]

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
        max_steps=2000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=5,
        limit_train_batches=20,
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

    # -------- Inference on 1 example for visual check --------
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

        # ------- 诊断 -------
        def frame_disp_cpu(x_btjc):
            x = x_btjc[0]
            if x.size(0) < 2: return 0.0
            dx = x[1:, :, :2] - x[:-1, :, :2]
            return dx.abs().mean().item()
        Tf = gen_btjc_cpu.size(1)
        print(f"[GEN] Tf={Tf}, mean|Δpred|={frame_disp_cpu(gen_btjc_cpu):.6f}, "
              f"mean|Δgt|={frame_disp_cpu(fut_gt_cpu):.6f}", flush=True)

        try:
            dtw_free = masked_dtw(gen_btjc, fut_gt, mask_bt).item()
            print(f"[Free-run] DTW: {dtw_free:.4f}", flush=True)
        except Exception as e:
            print("[Free-run] DTW eval skipped:", e)

        os.makedirs("logs", exist_ok=True)
        torch.save({"gen": gen_btjc_cpu[0], "gt": fut_gt_cpu[0]}, "logs/free_run.pt")
        print("Saved free-run sequences to logs/free_run.pt", flush=True)

        # ===== 可视化部分 =====
        # 1) header：优先 loader，其次 CSV
        header = _get_pose_header_from_loader(train_loader)
        if header is None:
            header = _probe_header_from_csv(csv_path, data_dir)

        # 2) 打印数值与 header-J 对齐情况
        _dump_stats("PRED", gen_btjc_cpu)
        _dump_stats("GT",   fut_gt_cpu)
        expJ = _header_joint_count(header)
        print("[HEADER] expected_J =", expJ, "| actual_J =", gen_btjc_cpu.size(2))

        # 3) 若 J 不匹配，重探 header（避免只出一个点）
        if expJ is not None and expJ != gen_btjc_cpu.size(2):
            print("[WARN] header J mismatch, re-probing header from CSV...")
            header = _probe_header_from_csv(csv_path, data_dir)
            print("[HEADER] reprobe expected_J =", _header_joint_count(header))

        # 4) 仅用于画图的归一化（把坐标拉到[0.1,0.9]）
        gen_viz = normalize_for_viz(gen_btjc_cpu)
        gt_viz  = normalize_for_viz(fut_gt_cpu)

        # 5) 用 pose-format 画 gif；并行导出散点兜底
        try:
            save_with_pose_visualizer(gen_viz, gt_viz, header, fps=12)
        except Exception as e:
            print("[viz] PoseVisualizer failed:", e)

        save_scatter(gen_btjc_cpu, "logs/free_run_scatter_pred.gif", "PRED")
        save_scatter(fut_gt_cpu,  "logs/free_run_scatter_gt.gif",   "GT")

