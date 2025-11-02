# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import lightning as pl
from torch.utils.data import Dataset, DataLoader
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import normalize_mean_std

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, masked_dtw


def unnormalize_tensor_with_global_stats(tensor: torch.Tensor, mean_std: dict):
    if torch.is_tensor(mean_std["mean"]):
        mean = mean_std["mean"].detach().clone().float().to(tensor.device)
        std = mean_std["std"].detach().clone().float().to(tensor.device)
    else:
        mean = torch.tensor(mean_std["mean"], dtype=torch.float32, device=tensor.device)
        std = torch.tensor(mean_std["std"], dtype=torch.float32, device=tensor.device)
    return tensor * std + mean

class FilteredSmallDataset(Dataset):
    """从大 dataset 中挑出 N 条合法样本（非空 pose）用于过拟合测试"""
    def __init__(self, base_ds, num_samples=4, max_scan=500):
        self.base = base_ds
        self.valid_idx = []
        for i in range(min(len(base_ds), max_scan)):
            try:
                sample = base_ds[i]
                if isinstance(sample, dict) and "data" in sample:
                    if sample["data"].shape[1] > 5:
                        self.valid_idx.append(i)
                if len(self.valid_idx) >= num_samples:
                    break
            except Exception:
                continue
        if len(self.valid_idx) < num_samples:
            print(f"[WARN] only {len(self.valid_idx)} valid samples found")
        print(f"[INIT] Selected {len(self.valid_idx)} samples for overfit test.")

    def __len__(self): return len(self.valid_idx)
    def __getitem__(self, i): return self.base[self.valid_idx[i]]


def _to_plain(x):
    if hasattr(x, "tensor"): x = x.tensor
    if hasattr(x, "zero_filled"): x = x.zero_filled()
    return x.detach().cpu().contiguous().float()


def center_and_scale_pose(tensor, scale=1.0, offset=(500, 500, 0)):
    """居中、翻转Y轴以适配PoseViewer (不再放大scale)"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    center = tensor.mean(dim=1, keepdim=True)
    tensor = tensor - center
    tensor[..., 1] = -tensor[..., 1]
    tensor[..., 0] += offset[0]
    tensor[..., 1] += offset[1]
    return tensor


def temporal_smooth(x, k=5):
    """对时间维进行平滑，减少抖动 (x: [T, J, C])"""
    import torch.nn.functional as F

    # 如果输入是 [B,T,J,C]，只取 batch 0
    if x.dim() == 4:
        x = x[0]
    # 如果输入是 [T,1,J,C]（多了一维P）
    if x.dim() == 5 and x.shape[2] == 1:
        x = x.squeeze(2)

    T, J, C = x.shape
    x = x.contiguous().float()  # 确保内存连续

    # [T,J,C] → [1, C*J, T]
    x = x.permute(2, 1, 0).reshape(1, C * J, T)

    # 在时间维上平均池化平滑
    x = F.avg_pool1d(x, kernel_size=k, stride=1, padding=k // 2)

    # [1, C*J, T] → [T, J, C]
    x = x.reshape(C, J, T).permute(2, 1, 0).contiguous()
    return x


def tensor_to_pose(t_btjc, header):
    """将 [T,J,C] 或 [B,T,J,C] tensor 转为 Pose 对象"""
    t = _to_plain(t_btjc)
    if t.dim() == 4:
        t = t[0]
    arr = np.ascontiguousarray(t[:, None, :, :], dtype=np.float32)  # [T,1,J,C]
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)
    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    return Pose(header=header, body=body)


# ============================================================
#  Main pipeline
# ============================================================
if __name__ == "__main__":
    pl.seed_everything(42)
    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/overfit_178"
    os.makedirs(out_dir, exist_ok=True)

    mean_std_path = os.path.join(data_dir, "mean_std.pt")

    # ---------------------- Dataset ----------------------
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
        reduce_holistic=True,
    )

    base_ds.mean_std = torch.load(mean_std_path)
    print(f"[NORM] Loaded mean/std from {mean_std_path}")

    first = _to_plain(base_ds[0]["data"])
    print(f"[CHECK] raw data mean/std: {first.mean():.4f}, {first.std():.4f}")

    small_ds = torch.utils.data.Subset(base_ds, list(range(min(4, len(base_ds)))))
    loader = DataLoader(small_ds, batch_size=4, shuffle=True, collate_fn=zero_pad_collator)

    shape = next(iter(loader))["data"].shape
    print(f"[INFO] Overfit set shape: {tuple(shape)}")

    # ---------------------- Training ----------------------
    model = LitMinimal(num_keypoints=shape[-2], num_dims=shape[-1], lr=1e-3, log_dir=out_dir)
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        limit_train_batches=1,
        limit_val_batches=1,
        log_every_n_steps=1,
        enable_checkpointing=False,
        deterministic=True,
    )
    print(f"[TRAIN] Overfitting on 4 samples × {shape[-2]} joints × {shape[-1]} dims")
    trainer.fit(model, loader, loader)

    # =====================================================
    #  Inference + postprocess
    # =====================================================
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        cond = batch["conditions"]
        past = cond["input_pose"][:1].to(model.device)
        sign = cond["sign_image"][:1].to(model.device)
        fut  = batch["data"][:1].to(model.device)
        mask = cond["target_mask"][:1].to(model.device)

        pred = model.generate_full_sequence(past_btjc=past, sign_img=sign, target_mask=mask)
        dtw_val = masked_dtw(pred, fut, mask).item()
        print(f"[EVAL] masked_dtw = {dtw_val:.4f}")

        # -- Unnormalize --
        fut_un  = unnormalize_tensor_with_global_stats(fut, base_ds.mean_std)
        pred_un = unnormalize_tensor_with_global_stats(pred, base_ds.mean_std)
        print("[UNNORM] Applied FluentPose-style unnormalize ✅")

        # -- Convert to plain tensors --
        for x_name in ["fut_un", "pred_un"]:
            x = locals()[x_name]
            if hasattr(x, "tensor"):
                x = x.tensor
            if hasattr(x, "zero_filled"):
                x = x.zero_filled()
            locals()[x_name] = x

        # -- Remove person dim if present --
        if fut_un.dim() == 5 and fut_un.shape[2] == 1:
            fut_un = fut_un.squeeze(2)
        if pred_un.dim() == 5 and pred_un.shape[2] == 1:
            pred_un = pred_un.squeeze(2)

        # -- Center / scale / smooth --
        fut_un  = center_and_scale_pose(fut_un)
        pred_un = center_and_scale_pose(pred_un)
        pred_un = temporal_smooth(pred_un)
        print("[POST] Center + scale + smooth ✅")
        print(f"[DEBUG] fut_un shape={fut_un.shape}, pred_un shape={pred_un.shape}")

    # =====================================================
    #  Save & visualize
    # =====================================================
    ref_path = os.path.join(data_dir, base_ds.records[0]["pose"])
    print(f"[REF] Using reference pose header from: {ref_path}")

    try:
        with open(ref_path, "rb") as f:
            ref_pose = Pose.read(f)
        ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
        header = reduce_holistic(ref_pose).header
        print("[HEADER] limb counts:", [len(c.limbs) for c in header.components])
    except Exception as e:
        print(f"[ERROR] Failed to load or reduce header: {e}")
        raise

    try:
        gt_pose = tensor_to_pose(fut_un, header)
        pred_pose = tensor_to_pose(pred_un, header)
        print(f"[DEBUG] gt_pose frames={len(gt_pose.body.data)}, pred_pose frames={len(pred_pose.body.data)}")
    except Exception as e:
        print(f"[ERROR] Failed to create Pose objects: {e}")
        raise

    out_gt = os.path.join(out_dir, "gt_178.pose")
    out_pred = os.path.join(out_dir, "pred_178.pose")
    for f in [out_gt, out_pred, os.path.join(out_dir, "gt.mp4"), os.path.join(out_dir, "pred.mp4")]:
        if os.path.exists(f):
            os.remove(f)

    try:
        with open(out_gt, "wb") as f:
            gt_pose.write(f)
        with open(out_pred, "wb") as f:
            pred_pose.write(f)
        print(f"[SAVE] Pose files written → {out_dir}")
        print(f"[OK] ✅ Pose objects successfully written: {os.path.basename(out_gt)}, {os.path.basename(out_pred)}")
    except Exception as e:
        print(f"[ERROR] ❌ Failed to save pose files: {e}")
        raise

    try:
        p = Pose.read(open(out_pred, "rb"))
        print("[CHECK] Pose OK ✅ limbs:", [len(c.limbs) for c in p.header.components])
    except Exception as e:
        print(f"[ERROR] Pose verification failed: {e}")

    try:
        viz = PoseVisualizer(gt_pose.header)
        viz.save_video(gt_pose.body, os.path.join(out_dir, "gt.mp4"), fps=25)
        viz.save_video(pred_pose.body, os.path.join(out_dir, "pred.mp4"), fps=25)
        print("✅ Visualization videos saved successfully!")
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")

