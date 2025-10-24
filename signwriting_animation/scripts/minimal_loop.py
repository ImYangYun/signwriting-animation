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
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import unnormalize_mean_std
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
    if x.dim() == 5:  # [B,T,P,J,C] -> [B,T,J,C]
        x = x[:, :, 0, ...]
    return x


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

def _probe_header_from_csv(csv_path, data_dir):
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
                        pose = Pose.read(pf)
                        print(f"[HEADER] Loaded header from: {full}")
                        print("          Components:", [c.name for c in pose.header.components])
                        return pose.header
    except Exception as e:
        print(f"[HEADER] Failed to probe CSV: {e}")
    return None


# ==================== Pose Saving ====================
def btjc_to_pose(x_btjc, header, fps=12):
    """Convert [B,T,J,C] → Pose object for saving"""
    x = x_btjc[0].detach().cpu().numpy().astype(np.float32)  # [T,J,C]
    T, J, C = x.shape
    x = x[:, np.newaxis, :, :]  # [T,1,J,C]
    conf = np.ones((T, 1, J), dtype=np.float32)
    body = NumPyPoseBody(fps=fps, data=x, confidence=conf)
    return Pose(header, body)

def save_pose_files(pred_btjc, gt_btjc, header, data_dir, csv_path):
    """
    Save predicted and ground truth poses to .pose files for visualization.
    Includes unnormalization and correct component selection for sign.mt.
    """
    from pose_format import Pose

    # ======== 1. Header sourcing ========
    if header is None:
        print("[POSE SAVE] Loading header from CSV...")
        header = _probe_header_from_csv(csv_path, data_dir)

    if header is None:
        print("[POSE SAVE] ❌ No header found, skipping .pose export")
        return False

    try:
        pred_pose = btjc_to_pose(pred_btjc, header)
        gt_pose   = btjc_to_pose(gt_btjc, header)

        pred_pose = unnormalize_mean_std(pred_pose)
        gt_pose   = unnormalize_mean_std(gt_pose)

        for label, pose_obj in [("pred", pred_pose), ("gt", gt_pose)]:
            data = pose_obj.body.data
            print(
                f"[POSE SAVE] After unnormalize ({label}): "
                f"mean={data.mean():.3f}, std={data.std():.3f}, "
                f"min={data.min():.3f}, max={data.max():.3f}"
            )

        try:
            pred_pose = pred_pose.remove_components(["POSE_WORLD_LANDMARKS"])
            gt_pose   = gt_pose.remove_components(["POSE_WORLD_LANDMARKS"])
        except Exception:
            pass

        header_names = [c.name for c in header.components]
        keep_components = ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]
        try:
            if hasattr(pred_pose, "select_components"):
                available = [c for c in keep_components if c in [comp.name for comp in pred_pose.header.components]]
                if available:
                    pred_pose = pred_pose.select_components(available)
                    gt_pose = gt_pose.select_components(available)
                    print(f"[POSE SAVE] ✅ Kept components: {available}")
                else:
                    print("[POSE SAVE] ⚠️ No target components found, kept all.")
            else:
                print("[POSE SAVE] ℹ️ select_components() not available, skipping filtering.")
        except Exception as e:
            print(f"[POSE SAVE] ⚠️ Skipped component filtering ({e})")

        for pose_obj, label in [(pred_pose, "pred"), (gt_pose, "gt")]:
            data = pose_obj.body.data
            min_v, max_v = np.min(data), np.max(data)
            print(f"[POSE SAVE] {label}: range=({min_v:.3f}, {max_v:.3f})")
            if np.abs(data).max() < 0.05:  # too small
                print(f"[POSE SAVE] ⚠️ {label} coords very small — scaling ×100")
                pose_obj.body.data = data * 100.0

        os.makedirs("logs", exist_ok=True)
        pred_path = "logs/prediction.pose"
        gt_path   = "logs/groundtruth.pose"

        with open(pred_path, "wb") as f:
            pred_pose.write(f)
        with open(gt_path, "wb") as f:
            gt_pose.write(f)

        print("\n" + "="*65)
        print("✅ Saved .pose files for visualization:")
        print(f"   • {pred_path}")
        print(f"   • {gt_path}\n")
        print("To visualize locally:")
        print("   visualize_pose -i logs/prediction.pose -o logs/prediction.mp4")
        print("   visualize_pose -i logs/groundtruth.pose -o logs/groundtruth.mp4\n")
        print("Or drag the .pose files into https://sign.mt")
        print("="*65 + "\n")

        return True
    except Exception as e:
        print(f"[POSE SAVE] ❌ Error during save: {e}")
        import traceback; traceback.print_exc()
        return False

def save_scatter_backup(seq_btjc, path, label):
    try:
        x = seq_btjc[0, :, :, :2].cpu().numpy()
        x = x - x.mean(axis=(0, 1), keepdims=True)
        M = float(np.abs(x).max()) or 1.0
        x = x / M
        T = x.shape[0]

        fig, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter([], [], s=15)
        ax.axis("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)

        def _init(): sc.set_offsets(np.empty((0, 2))); return sc,
        def _update(f):
            ax.set_title(f"{label}: frame {f+1}/{T}")
            sc.set_offsets(x[f])
            return sc,

        ani = animation.FuncAnimation(fig, _update, frames=T, init_func=_init, interval=100, blit=True)
        ani.save(path, writer="pillow", fps=10)
        plt.close(fig)
        print(f"[BACKUP] Saved scatter: {path}")
    except Exception as e:
        print(f"[BACKUP] ❌ Failed scatter: {e}")


# ==================== Dataset Wrappers ====================
class FilteredDataset(Dataset):
    def __init__(self, base: Dataset, target_count=4, max_scan=500, min_frames=15):
        self.base = base
        self.idx = []
        N = len(base)
        for i in range(min(N, max_scan)):
            try:
                it = base[i]
                if isinstance(it, dict) and "data" in it and "conditions" in it:
                    data = it["data"]
                    if hasattr(data, "zero_filled"):
                        data = data.zero_filled()
                    if data.shape[1] >= min_frames:
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
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=num_workers, collate_fn=zero_pad_collator)


# ==================== Main ====================
if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_default_dtype(torch.float32)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data.csv"
    batch_size, num_workers = 2, 2

    train_loader = make_loader(data_dir, csv_path, "train", bs=batch_size, num_workers=num_workers)
    val_loader = train_loader

    print("\n" + "="*60)
    batch = next(iter(train_loader))
    print("[DATA DEBUG]")
    print(f"  data.shape        = {batch['data'].shape}")
    print(f"  target_mask.shape = {batch['conditions']['target_mask'].shape}")
    print(f"  input_pose.shape  = {batch['conditions']['input_pose'].shape}")
    print("="*60 + "\n")

    model = LitMinimal(log_dir="logs")
    trainer = pl.Trainer(
        max_steps=500,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=5,
        limit_train_batches=10,
        limit_val_batches=5,
        deterministic=True,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_loader, val_loader)

    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        cond  = batch["conditions"]

        past_btjc = cond["input_pose"][:1].to(model.device)
        sign_img  = cond["sign_image"][:1].to(model.device)
        fut_gt    = batch["data"][:1].to(model.device)

        print("[GEN] Generating full sequence...")
        gen_btjc = model.generate_full_sequence(past_btjc, sign_img, target_len=20)

        gen_btjc_cpu = _as_dense_cpu_btjc(gen_btjc)
        fut_gt_cpu   = _as_dense_cpu_btjc(fut_gt)

        def frame_disp(x_btjc):
            x = x_btjc[0]
            return (x[1:, :, :2] - x[:-1, :, :2]).abs().mean().item() if x.size(0) > 1 else 0.0
        print(f"[GEN] Tf={gen_btjc_cpu.size(1)}, mean|Δpred|={frame_disp(gen_btjc_cpu):.6f}, mean|Δgt|={frame_disp(fut_gt_cpu):.6f}")

        try:
            mask_for_eval = torch.ones(1, gen_btjc.size(1), device=gen_btjc.device)
            dtw_val = masked_dtw(gen_btjc, fut_gt.to(gen_btjc.device), mask_for_eval).item()
            print(f"[EVAL] DTW (pred vs GT): {dtw_val:.4f}")
        except Exception as e:
            print(f"[EVAL] DTW failed: {e}")

        os.makedirs("logs", exist_ok=True)
        torch.save({"gen": gen_btjc_cpu[0], "gt": fut_gt_cpu[0]}, "logs/free_run.pt")

        header = None

        for root, _, files in os.walk(data_dir):
            for name in files:
                if name.endswith(".pose"):
                    ref_pose_path = os.path.join(root, name)
                    try:
                        with open(ref_pose_path, "rb") as f:
                            pose = Pose.read(f)
                            header = pose.header
                            print(f"[HEADER] ✅ Auto-loaded reference header from {ref_pose_path}")
                            print("          Components:", [c.name for c in header.components])
                            break
                    except Exception as e:
                        print(f"[HEADER] Skipped invalid file: {ref_pose_path} ({e})")
                        continue
            if header is not None:
                break

        # ② fallback：如果上面没找到，就用 loader / CSV 探测
        if header is None:
            header = _get_pose_header_from_loader(train_loader)
        if header is None:
            header = _probe_header_from_csv(csv_path, data_dir)

        if header:
            print("[HEADER DEBUG]")
            for c in header.components:
                print(f"  - {c.name} ({getattr(c, 'points', 'unknown')} points)")
        else:
            print("[HEADER DEBUG] ❌ None (pose header missing!)")

        # === SAVE POSES FOR VISUALIZATION ===
        pose_saved = save_pose_files(gen_btjc_cpu, fut_gt_cpu, header, data_dir, csv_path)
        if not pose_saved:
            print("[FALLBACK] Using scatter backup...")
            save_scatter_backup(gen_btjc_cpu, "logs/scatter_pred.gif", "PRED")
            save_scatter_backup(fut_gt_cpu, "logs/scatter_gt.gif", "GT")
