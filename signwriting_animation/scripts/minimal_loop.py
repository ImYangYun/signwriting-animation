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


def _parse_components_from_header(header):
    """
    返回一个组件列表 comps，每个元素形如：
      {
        "name": "body"/"face"/"lh"/"rh"/...，
        "edges": [(a,b), ...],            # 已加上全局 joint offset
        "edge_colors": [(r,g,b,a) or None] * len(edges),
        "j_range": (start, end),          # 该组件在全局 joint 下标范围（左闭右开）
      }
    如果 header 只有一个 skeleton，则作为一个组件返回。
    """
    comps = []
    if header is None:
        return comps

    def _safe_edges_and_colors(sk):
        e = [(int(a), int(b)) for a, b in (getattr(sk, "edges", []) or [])]
        c = getattr(sk, "edge_colors", None)
        if c and len(c) == len(e):
            c = [tuple(cc) for cc in c]
        else:
            c = [None] * len(e)
        return e, c

    # 情况 1：单一 skeleton
    sk = getattr(header, "skeleton", None)
    if sk is not None and getattr(sk, "edges", None):
        e, c = _safe_edges_and_colors(sk)
        j_end = getattr(header, "num_joints", None)
        if j_end is None:
            j_end = (max([max(a, b) for a, b in e]) + 1) if e else 0
        comps.append({
            "name": getattr(sk, "name", "pose").lower(),
            "edges": e,
            "edge_colors": c,
            "j_range": (0, j_end),
        })
        return comps

    # 情况 2：多组件（常见于 body/face/hands）
    header_comps = getattr(header, "components", None) or []
    for comp in header_comps:
        comp_sk = getattr(comp, "skeleton", None)
        if comp_sk is None:
            continue
        offset = int(getattr(comp, "offset", 0))
        e, c = _safe_edges_and_colors(comp_sk)
        e = [(a + offset, b + offset) for a, b in e]
        name = getattr(comp, "name", None) or getattr(comp_sk, "name", None) or f"comp@{offset}"
        j_end = (max([max(a, b) for a, b in e]) + 1) if e else offset
        comps.append({
            "name": name.lower(),
            "edges": e,
            "edge_colors": c,
            "j_range": (offset, j_end),
        })
    return comps


def _edge_colors_or_palette(edge_colors, num_edges, palette="tab20"):
    """header 提供就用；没有就用 matplotlib 调色盘循环。"""
    if edge_colors and len(edge_colors) == num_edges and all(c is not None for c in edge_colors):
        return edge_colors
    cmap = plt.get_cmap(palette)
    return [cmap(i % cmap.N) for i in range(num_edges)]


def _get_pose_header_from_loader(loader):
    ds = loader.dataset
    base = getattr(ds, "base", None) or getattr(ds, "dataset", None)
    base = getattr(base, "base", None) or base
    # 1) 直接拿数据集上的 header
    if base is not None and hasattr(base, "header") and base.header is not None:
        return base.header
    if hasattr(ds, "header") and ds.header is not None:
        return ds.header
    # 2) 兜底：从一个样本里找
    try:
        sample = ds[0] if len(ds) > 0 else None
        if isinstance(sample, dict):
            meta = sample.get("metadata", {}) or {}
            # 常见字段名：pose_header / header
            if "pose_header" in meta and meta["pose_header"] is not None:
                return meta["pose_header"]
            if "header" in meta and meta["header"] is not None:
                return meta["header"]
    except Exception:
        pass
    return None


def unnormalize_btjc(x_btjc, header):
    """
    Inverse normalization using pose-format header if available.
    x_btjc: [B,T,J,C] float CPU tensor.
    """
    x = x_btjc.detach().float().cpu()
    if header is None or not hasattr(header, "normalization_info") or header.normalization_info is None:
        return x
    ni = header.normalization_info
    # mean/std
    if getattr(ni, "mean", None) is not None and getattr(ni, "std", None) is not None:
        mean = torch.tensor(ni.mean, dtype=x.dtype).view(1, 1, 1, -1)
        std  = torch.tensor(ni.std,  dtype=x.dtype).view(1, 1, 1, -1)
        return x * std + mean
    # scale/translation
    if getattr(ni, "scale", None) is not None and getattr(ni, "translation", None) is not None:
        scale = torch.tensor(ni.scale, dtype=x.dtype).view(1, 1, 1, -1)
        trans = torch.tensor(ni.translation, dtype=x.dtype).view(1, 1, 1, -1)
        return x * scale + trans
    return x


def visualize_pose_components(
    pred_btjc, gt_btjc, header,
    save_path="logs/free_run_posefmt.gif",
    fps=12, show_points=True, trail=0, annotate_indices=True
):
    """
    Pred vs GT 并排动画：按组件（body/face/hands...）分色；支持残影与关节编号标注。
    传入 pred_btjc / gt_btjc 应为 UNnormalized 的 [1,T,J,C] CPU tensor。
    """
    pred = pred_btjc[0].numpy()  # [T,J,C]
    gt   = gt_btjc[0].numpy()
    T, J, _ = pred.shape

    # 组件解析
    comps = _parse_components_from_header(header)
    used_fallback = False
    if not comps:
        used_fallback = True
        n = max(0, min(J - 1, 20))
        edges = [(i, i + 1) for i in range(n)]
        comps = [{
            "name": "fallback",
            "edges": edges,
            "edge_colors": [(0.5, 0.5, 0.5, 1.0)] * len(edges),
            "j_range": (0, J),
        }]

    # 坐标范围
    xy = np.concatenate([pred[..., :2].reshape(-1, 2), gt[..., :2].reshape(-1, 2)], axis=0)
    x_min, y_min = xy.min(axis=0); x_max, y_max = xy.max(axis=0)
    pad = 0.05 * max(x_max - x_min, y_max - y_min, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True)
    ax_pred, ax_gt = axes
    for ax, title in [(ax_pred, "Predicted (unnormalized)"), (ax_gt, "Ground Truth (unnormalized)")]:
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.axis("off")

    # artists 容器
    artists_pred, artists_gt = [], []
    legend_handles = []
    # 文字标注对象（可选）
    idx_text_pred = idx_text_gt = None
    if annotate_indices:
        idx_text_pred = [ax_pred.text(0, 0, "", fontsize=6, ha="center", va="center") for _ in range(J)]
        idx_text_gt   = [ax_gt.text(0, 0, "", fontsize=6, ha="center", va="center") for _ in range(J)]

    for comp in comps:
        name  = comp["name"]
        edges = comp["edges"]
        colors = _edge_colors_or_palette(comp.get("edge_colors"), len(edges))

        # 当前帧线
        pred_lines = [ax_pred.plot([], [], lw=2, color=colors[i])[0] for i in range(len(edges))]
        gt_lines   = [ax_gt.plot([],   [], lw=2, color=colors[i])[0] for i in range(len(edges))]

        # 残影
        pred_trails, gt_trails = [], []
        if trail > 0:
            alpha_step = 0.6 / max(1, trail)
            for k in range(trail):
                fade = 0.4 + alpha_step * (trail - 1 - k)
                pred_trails.append([ax_pred.plot([], [], lw=1, color=(*colors[i][:3], fade))[0]
                                    for i in range(len(edges))])
                gt_trails.append([ax_gt.plot([], [], lw=1, color=(*colors[i][:3], fade))[0]
                                  for i in range(len(edges))])

        # 关节点散点
        pred_pts = gt_pts = None
        if show_points:
            pred_pts = ax_pred.scatter([], [], s=10, label=name)
            gt_pts   = ax_gt.scatter([], [], s=10)

        artists_pred.append((pred_lines, pred_trails, pred_pts, edges))
        artists_gt.append((gt_lines,   gt_trails,   gt_pts,   edges))

        # 图例（左图）
        handle = ax_pred.scatter([], [], s=20, color=colors[0], label=name) if show_points \
                 else ax_pred.plot([], [], lw=2, color=colors[0], label=name)[0]
        legend_handles.append(handle)

    if legend_handles:
        ax_pred.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=9)

    # ------- 内部工具 -------

    def _set_current_lines(line_list, edges, pose_xy):
        for k, (a, b) in enumerate(edges):
            xa, ya = pose_xy[a, 0], pose_xy[a, 1]
            xb, yb = pose_xy[b, 0], pose_xy[b, 1]
            line_list[k].set_data([xa, xb], [ya, yb])

    def _set_trails(trails_lists, edges, pose_xy_hist):  # pose_xy_hist: [trail, J, 2]
        if not trails_lists: return
        trail_len = min(len(trails_lists), len(pose_xy_hist))
        for t in range(trail_len):
            for k, (a, b) in enumerate(edges):
                xa, ya = pose_xy_hist[t][a, 0], pose_xy_hist[t][a, 1]
                xb, yb = pose_xy_hist[t][b, 0], pose_xy_hist[t][b, 1]
                trails_lists[t][k].set_data([xa, xb], [ya, yb])

    def _set_points(scatter_obj, pose_xy):
        if scatter_obj is not None:
            scatter_obj.set_offsets(pose_xy)

    def _set_indices(text_objs, pose_xy):
        if text_objs is None: return
        for j, txt in enumerate(text_objs):
            x, y = pose_xy[j, 0], pose_xy[j, 1]
            txt.set_position((x, y))
            txt.set_text(str(j))

    pred_xy = pred[:, :, :2]
    gt_xy   = gt[:,   :, :2]

    def _init():
        for (pl, ptrails, ppts, pedges), (gl, gtrails, gpts, gedges) in zip(artists_pred, artists_gt):
            _set_current_lines(pl, pedges, pred_xy[0])
            _set_current_lines(gl, gedges, gt_xy[0])
            _set_points(ppts, pred_xy[0])
            _set_points(gpts, gt_xy[0])
        _set_indices(idx_text_pred, pred_xy[0])
        _set_indices(idx_text_gt,   gt_xy[0])
        return sum([[*ap[0], *sum(ap[1], []), ap[2]] for ap in artists_pred], []) + \
               sum([[*ag[0], *sum(ag[1], []), ag[2]] for ag in artists_gt], []) + \
               (idx_text_pred or []) + (idx_text_gt or [])

    def _update(t):
        for (pl, ptrails, ppts, pedges), (gl, gtrails, gpts, gedges) in zip(artists_pred, artists_gt):
            _set_current_lines(pl, pedges, pred_xy[t])
            _set_current_lines(gl, gedges, gt_xy[t])
            _set_points(ppts, pred_xy[t])
            _set_points(gpts, gt_xy[t])
            if trail > 0 and t > 0:
                hist_idx = list(range(max(0, t - trail), t))[::-1]
                _set_trails(ptrails, pedges, pred_xy[hist_idx])
                _set_trails(gtrails, gedges, gt_xy[hist_idx])
        _set_indices(idx_text_pred, pred_xy[t])
        _set_indices(idx_text_gt,   gt_xy[t])
        return sum([[*ap[0], *sum(ap[1], []), ap[2]] for ap in artists_pred], []) + \
               sum([[*ag[0], *sum(ag[1], []), ag[2]] for ag in artists_gt], []) + \
               (idx_text_pred or []) + (idx_text_gt or [])

    ani = animation.FuncAnimation(
        fig, _update, frames=T, init_func=_init,
        interval=max(1, int(1000 / max(1, fps))), blit=True
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)
    comp_names = ",".join([c["name"] for c in comps])
    print(f"[viz] components={len(comps)} [{comp_names}], fallback={used_fallback} -> {save_path}")


# -------------------- dataset & training (保持不变) --------------------

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

        # —— 新版可视化（注意括号缩进） ——
        visualize_pose_components(
            gen_unnorm, gt_unnorm, header,
            save_path="logs/free_run_posefmt.gif",
            fps=12, show_points=True, trail=3, annotate_indices=False
        )

        # 可选：简单散点动画（保留）
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

        def _init_scatter():
            sc_pred.set_offsets(np.empty((0, 2)))
            sc_gt.set_offsets(np.empty((0, 2)))
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
