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
    """原有简易散点轨迹，保留。"""
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
    """尽量从 dataloader/dataset 链条里拿 header；拿不到返回 None。"""
    ds = loader.dataset
    # 解最多 3 层包装（FilteredDataset -> base -> dataset）
    for _ in range(3):
        if hasattr(ds, "header") and ds.header is not None:
            return ds.header
        base = getattr(ds, "base", None) or getattr(ds, "dataset", None)
        if base is None:
            break
        ds = base
    # 从一个样本的 metadata 兜底
    try:
        sample = loader.dataset[0]
        if isinstance(sample, dict):
            meta = sample.get("metadata", {}) or {}
            return meta.get("pose_header") or meta.get("header")
    except Exception:
        pass
    return None

def _probe_header_from_csv(csv_path, data_dir):
    """你的 CSV 有 'pose' 列：直接读第一条 .pose 拿 header。"""
    from pose_format import Pose as _Pose
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get("pose", "")
                if isinstance(p, str) and p.lower().endswith(".pose"):
                    full = p if os.path.isabs(p) else os.path.join(data_dir, p)
                    if not os.path.exists(full):
                        print(f"[probe] not found: {full}")
                        continue
                    with open(full, "rb") as pf:
                        print(f"[probe] header from CSV 'pose': {full}")
                        return _Pose.read(pf).header
    except Exception as e:
        print("[probe] failed to read header from csv:", e)
    return None

def unnormalize_btjc(x_btjc, header):
    """根据 header.normalization_info 做反归一化。"""
    x = x_btjc.detach().float().cpu()
    if header is None or not hasattr(header, "normalization_info") or header.normalization_info is None:
        return x
    ni = header.normalization_info
    if getattr(ni, "mean", None) is not None and getattr(ni, "std", None) is not None:
        mean = torch.tensor(ni.mean, dtype=x.dtype).view(1, 1, 1, -1)
        std  = torch.tensor(ni.std,  dtype=x.dtype).view(1, 1, 1, -1)
        return x * std + mean
    if getattr(ni, "scale", None) is not None and getattr(ni, "translation", None) is not None:
        scale = torch.tensor(ni.scale, dtype=x.dtype).view(1, 1, 1, -1)
        trans = torch.tensor(ni.translation, dtype=x.dtype).view(1, 1, 1, -1)
        return x * scale + trans
    return x

def print_header_info(header):
    """调试：打印 header 关键字段。"""
    if header is None:
        print("[header] is None")
        return
    sk = getattr(header, "skeleton", None)
    comps = getattr(header, "components", None) or []
    print("[header] has_skeleton =", sk is not None)
    if sk is not None:
        n_edges = len(getattr(sk, "edges", []) or [])
        num_j = getattr(sk, "num_joints", None)
        print(f"[header] skeleton edges: {n_edges}, num_joints: {num_j}")
    print(f"[header] components: {len(comps)}",
          [getattr(c, "name", f"comp{i}") for i, c in enumerate(comps)])


# =========================
# pose-format 官方可视化（Pred/GT 各导一张 GIF）
# =========================
def btjc_to_pose(x_btjc, header, fps=25):
    """x_btjc: [1,T,J,C]（已 unnormalize, CPU）→ Pose（NumPy 后端）"""
    x = x_btjc[0].detach().cpu().numpy()                        # [T,J,C]
    conf = np.ones((x.shape[0], x.shape[1]), dtype=np.float32)  # [T,J]
    body = NumPyPoseBody(fps=fps, data=x, confidence=conf)
    return Pose(header, body)

def save_pose_gifs_with_pose_format(pred_btjc, gt_btjc, header,
                                    out_dir="logs", stem="free_run_posefmt", fps=12):
    os.makedirs(out_dir, exist_ok=True)
    pred_pose = btjc_to_pose(pred_btjc, header, fps=fps)
    gt_pose   = btjc_to_pose(gt_btjc,   header, fps=fps)
    viz_pred = PoseVisualizer(pred_pose)
    viz_gt   = PoseVisualizer(gt_pose)
    pred_path = os.path.join(out_dir, f"{stem}_pred.gif")
    gt_path   = os.path.join(out_dir, f"{stem}_gt.gif")
    viz_pred.save_gif(pred_path, viz_pred.draw())
    viz_gt.save_gif(gt_path,     viz_gt.draw())
    print(f"[viz] pose-format GIF saved:\n  - {pred_path}\n  - {gt_path}")

def align_to_header_joints(x_btjc, header, ignore_world=True):
    """
    仅用于【可视化】的关节数对齐：
    - 从 header.skeleton 或 header.components 推断需要的 J_h
    - 预测 J > J_h 时截断到前 J_h；预测 J < J_h 返回 None（交给 fallback）
    - ignore_world=True 时会忽略 '...WORLD...' 组件（常见于 MediaPipe 的 POSE_WORLD_LANDMARKS）
    """
    x_btjc = x_btjc.clone()
    J_pred = x_btjc.size(2)

    # 1) 直接用全局 skeleton.num_joints
    sk = getattr(header, "skeleton", None)
    if sk is not None and hasattr(sk, "num_joints") and sk.num_joints:
        J_h = int(sk.num_joints)
    else:
        # 2) 聚合 components：offset + component 内的关节数/最大下标
        J_h = None
        comps = getattr(header, "components", None) or []
        max_end = 0
        for comp in comps:
            name = (getattr(comp, "name", "") or "").upper()
            if ignore_world and "WORLD" in name:
                continue
            off = int(getattr(comp, "offset", 0))
            skc = getattr(comp, "skeleton", None)
            if skc is None:
                continue
            # 优先 num_joints；没有就从 edges 估计
            if hasattr(skc, "num_joints") and skc.num_joints:
                end = off + int(skc.num_joints)
            else:
                end = off
                for a, b in (getattr(skc, "edges", None) or []):
                    end = max(end, off + int(a) + 1, off + int(b) + 1)
            max_end = max(max_end, end)
        if max_end > 0:
            J_h = max_end

    if J_h is None:
        return None  # 还是无法推断，交给 fallback

    if J_pred == J_h:
        return x_btjc
    if J_pred > J_h:
        return x_btjc[:, :, :J_h, :]
    # 预测比 header 少，没法补点
    return None


# =========================
# fallback 可视化（无 header 时兜底）
# =========================
def visualize_pose_components_fallback(pred_btjc, gt_btjc, save_path="logs/free_run_posefmt_fallback.gif",
                                       fps=12, show_points=True):
    pred = pred_btjc[0].numpy()
    gt   = gt_btjc[0].numpy()
    T, J, _ = pred.shape
    n = max(0, min(J - 1, 20))
    edges = [(i, i + 1) for i in range(n)]
    xy = np.concatenate([pred[..., :2].reshape(-1, 2), gt[..., :2].reshape(-1, 2)], axis=0)
    x_min, y_min = xy.min(axis=0); x_max, y_max = xy.max(axis=0)
    pad = 0.05 * max(x_max - x_min, y_max - y_min, 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True)
    axp, axg = axes
    for ax, title in [(axp, "Predicted (unnormalized) – fallback"),
                      (axg, "Ground Truth (unnormalized) – fallback")]:
        ax.set_title(title); ax.set_aspect("equal")
        ax.set_xlim(x_min - pad, x_max + pad); ax.set_ylim(y_min - pad, y_max + pad); ax.axis("off")
    def _mk(ax):
        lines = [ax.plot([], [], lw=2, color=(0.5, 0.5, 0.5, 1.0))[0] for _ in edges]
        pts = ax.scatter([], [], s=10) if show_points else None
        return lines, pts
    pln, pp = _mk(axp); gln, gp = _mk(axg)
    def _set(lines, pts, pose_xy):
        for k, (a, b) in enumerate(edges):
            xa, ya = pose_xy[a, 0], pose_xy[a, 1]
            xb, yb = pose_xy[b, 0], pose_xy[b, 1]
            lines[k].set_data([xa, xb], [ya, yb])
        if pts is not None: pts.set_offsets(pose_xy)
    def _init():
        _set(pln, pp, pred[0, :, :2]); _set(gln, gp, gt[0, :, :2])
        return pln + gln + ([pp] if pp is not None else []) + ([gp] if gp is not None else [])
    def _update(t):
        _set(pln, pp, pred[t, :, :2]); _set(gln, gp, gt[t, :, :2])
        return pln + gln + ([pp] if pp is not None else []) + ([gp] if gp is not None else [])
    ani = animation.FuncAnimation(fig, _update, frames=T, init_func=_init,
                                  interval=max(1, int(1000 / max(1, fps))), blit=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ani.save(save_path, writer="pillow", fps=fps); plt.close(fig)
    print(f"[viz] fallback animation saved to {save_path}")


# =========================
# Dataset / Loader（保持原样）
# =========================
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


# =========================
# main（训练 & 可视化）—— 训练配置等保持不变
# =========================
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

        def frame_disp_cpu(x_btjc):
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

        # ---- header（先 loader，后 CSV）----
        header = _get_pose_header_from_loader(train_loader)
        if header is None or getattr(header, "skeleton", None) is None:
            header = _probe_header_from_csv(csv_path, data_dir)
        print_header_info(header)

        gen_unnorm = unnormalize_btjc(gen_btjc_cpu, header)  # [1,T,J,C]
        gt_unnorm  = unnormalize_btjc(fut_gt_cpu,  header)   # [1,T,J,C]

        gen_aligned = align_to_header_joints(gen_unnorm, header) if header is not None else None
        gt_aligned  = align_to_header_joints(gt_unnorm,  header) if header is not None else None

        def _has_drawable_topology(h):
            return (h is not None) and (
                getattr(h, "skeleton", None) is not None or
                (len(getattr(h, "components", None) or []) > 0)
            )

        gen_aligned = align_to_header_joints(gen_unnorm, header) if header is not None else None
        gt_aligned  = align_to_header_joints(gt_unnorm,  header) if header is not None else None

        if _has_drawable_topology(header) and gen_aligned is not None and gt_aligned is not None:
            save_pose_gifs_with_pose_format(
                gen_aligned, gt_aligned, header,
                out_dir="logs", stem="free_run_posefmt", fps=12
            )
        else:
            visualize_pose_components_fallback(
                gen_unnorm, gt_unnorm, save_path="logs/free_run_posefmt_fallback.gif",
                fps=12, show_points=True
            )

        fig, ax = plt.subplots(figsize=(5, 5))
        sc_pred = ax.scatter([], [], s=15, c="red",  label="Predicted", animated=True)
        sc_gt   = ax.scatter([], [], s=15, c="blue", label="GT", alpha=0.35, animated=True)
        ax.legend(loc="upper right", frameon=False); ax.axis("equal")
        xy = torch.cat([gen_btjc_cpu[..., :2].reshape(-1, 2),
                        fut_gt_cpu[...,  :2].reshape(-1, 2)], dim=0).numpy()
        x_min, y_min = xy.min(axis=0); x_max, y_max = xy.max(axis=0)
        pad = 0.05 * max(x_max - x_min, y_max - y_min, 1e-6)
        ax.set_xlim(x_min - pad, x_max + pad); ax.set_ylim(y_min - pad, y_max + pad)

        def _init_scatter():
            sc_pred.set_offsets(np.empty((0, 2))); sc_gt.set_offsets(np.empty((0, 2)))
            return sc_pred, sc_gt
        def _update_scatter(f):
            ax.set_title(f"Free-run prediction  |  frame {f+1}/{len(gen_btjc_cpu[0])}")
            sc_pred.set_offsets(gen_btjc_cpu[0, f, :, :2].numpy())
            sc_gt.set_offsets(  fut_gt_cpu[0,  f, :, :2].numpy())
            return sc_pred, sc_gt
        ani = animation.FuncAnimation(
            fig, _update_scatter, frames=max(1, len(gen_btjc_cpu[0])),
            init_func=_init_scatter, interval=100, blit=True
        )
        ani.save("logs/free_run_anim.gif", writer="pillow", fps=10)
        plt.close(fig)
        print("Saved free-run animation to logs/free_run_anim.gif", flush=True)
