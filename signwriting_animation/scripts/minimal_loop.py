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

# ============== Header 获取 ==============
def _get_pose_header_from_loader(loader):
    """从 DataLoader 中获取 pose header"""
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
                        header = Pose.read(pf).header
                        print(f"[HEADER] Successfully loaded from: {full}")
                        return header
    except Exception as e:
        print(f"[HEADER] Failed to probe from CSV: {e}")
    return None

def btjc_to_pose(x_btjc, header, fps=25):
    """
    将 [B,T,J,C] tensor 转换为 Pose 对象
    x_btjc: [B,T,J,C] 只取第一个样本 -> [T,1,J,C]
    """
    x = x_btjc[0].detach().cpu().numpy().astype(np.float32)  # [T,J,C]
    T, J, C = x.shape
    
    # Reshape to [T,1,J,C] - pose-format expects people dimension
    x = x[:, np.newaxis, :, :]  # [T,1,J,C]
    
    # Create confidence (all ones for now)
    conf = np.ones((T, 1, J), dtype=np.float32)
    
    # Create body
    body = NumPyPoseBody(fps=fps, data=x, confidence=conf)
    
    return Pose(header, body)

def save_pose_files(pred_btjc, gt_btjc, header, data_dir, csv_path):
    """保存预测和GT为 .pose 文件，用于在 sign.mt 上可视化"""
    
    # 确保 header 存在
    if header is None:
        print("[POSE SAVE] Trying to load header from CSV...")
        header = _probe_header_from_csv(csv_path, data_dir)
    
    if header is None:
        print("[POSE SAVE] ❌ No header available, cannot save .pose files")
        return False
    
    try:
        pred_pose = btjc_to_pose(pred_btjc, header, fps=25)
        gt_pose = btjc_to_pose(gt_btjc, header, fps=25)

        try:
            pred_pose = pred_pose.remove_components(["POSE_WORLD_LANDMARKS"])
            gt_pose = gt_pose.remove_components(["POSE_WORLD_LANDMARKS"])
        except Exception:
            pass  # 如果没有这个组件就跳过
        
        # 保存文件
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/prediction.pose", "wb") as f:
            pred_pose.write(f)
        
        with open("logs/groundtruth.pose", "wb") as f:
            gt_pose.write(f)
        
        print("\n" + "="*60)
        print("✅ Saved .pose files for visualization!")
        print("   Prediction:   logs/prediction.pose")
        print("   Ground Truth: logs/groundtruth.pose")
        print("\nTo visualize:")
        print("1. Go to https://sign.mt")
        print("2. Drag and drop the .pose files")
        print("3. Or use: visualize_pose logs/prediction.pose")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"[POSE SAVE] ❌ Failed to save .pose files: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============== 简单的散点图备份（如果 .pose 保存失败） ==============
def save_scatter_backup(seq_btjc, path, label):
    """仅作为备份的简单散点图"""
    try:
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
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

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
        print(f"[BACKUP] Saved scatter plot: {path}")
    except Exception as e:
        print(f"[BACKUP] Failed to save scatter: {e}")

# ------------------------
# Dataset / Loader
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
    
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self, i):
        return self.base[self.idx[i]]

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

    # ============ 数据诊断 ============
    print("\n" + "="*60)
    print("[DATA DEBUG] Checking data shapes...")
    batch = next(iter(train_loader))
    print(f"[DATA] batch['data'].shape = {batch['data'].shape}")
    print(f"[DATA] target_mask.shape = {batch['conditions']['target_mask'].shape}")
    print(f"[DATA] input_pose.shape = {batch['conditions']['input_pose'].shape}")
    
    # 检查mask的有效长度
    mask = batch['conditions']['target_mask']
    if hasattr(mask, 'zero_filled'):
        mask = mask.zero_filled()
    mask_float = mask.float()
    if mask_float.dim() == 5:
        mask_bt = (mask_float.sum(dim=(2,3,4)) > 0).float()
    elif mask_float.dim() == 4:
        mask_bt = (mask_float.sum(dim=(2,3)) > 0).float()
    else:
        mask_bt = mask_float
    print(f"[DATA] mask valid lengths per sample: {mask_bt.sum(dim=1)}")
    print("="*60 + "\n")

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
            target_len=20,  # 强制20帧
        )

        gen_btjc_cpu = _as_dense_cpu_btjc(gen_btjc)  # [1,T,J,C]
        fut_gt_cpu   = _as_dense_cpu_btjc(fut_gt)    # [1,T,J,C]

        def frame_disp_cpu(x_btjc):
            x = x_btjc[0]
            if x.size(0) < 2: return 0.0
            dx = x[1:, :, :2] - x[:-1, :, :2]
            return dx.abs().mean().item()        
        Tf = gen_btjc_cpu.size(1)
        print(f"[GEN] Tf={Tf}, mean|Δpred|={frame_disp_cpu(gen_btjc_cpu):.6f}, "
              f"mean|Δgt|={frame_disp_cpu(fut_gt_cpu):.6f}", flush=True)

        try:
            mask_for_eval = torch.ones(1, Tf, device=gen_btjc.device)
            dtw_free = masked_dtw(gen_btjc, gen_btjc_cpu.to(gen_btjc.device), mask_for_eval).item()
            print(f"[Free-run] DTW: {dtw_free:.4f}", flush=True)
        except Exception as e:
            print("[Free-run] DTW eval skipped:", e)

        os.makedirs("logs", exist_ok=True)
        torch.save({"gen": gen_btjc_cpu[0], "gt": fut_gt_cpu[0]}, "logs/free_run.pt")
        print("Saved free-run sequences to logs/free_run.pt", flush=True)

        header = _get_pose_header_from_loader(train_loader)
        pose_saved = save_pose_files(gen_btjc_cpu, fut_gt_cpu, header, data_dir, csv_path)

        if not pose_saved:
            print("[FALLBACK] Saving scatter plots as backup...")
            save_scatter_backup(gen_btjc_cpu, "logs/scatter_pred.gif", "PRED")
            save_scatter_backup(fut_gt_cpu, "logs/scatter_gt.gif", "GT")

