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
from pose_anonymization.data.normalization import normalize_mean_std, unnormalize_mean_std

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, masked_dtw


# ---------------------- Dataset wrapper ----------------------
class FilteredSmallDataset(Dataset):
    """从大 dataset 中挑出 N 条合法样本（非空 pose）用于过拟合测试"""
    def __init__(self, base_ds, num_samples=4, max_scan=500):
        self.base = base_ds
        self.valid_idx = []
        for i in range(min(len(base_ds), max_scan)):
            try:
                sample = base_ds[i]
                if isinstance(sample, dict) and "data" in sample:
                    if sample["data"].shape[1] > 5:  # 至少 5 帧
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


# ---------------------- Helper functions ----------------------
def _to_plain(x):
    if hasattr(x, "tensor"): x = x.tensor
    if hasattr(x, "zero_filled"): x = x.zero_filled()
    return x.detach().cpu().contiguous().float()

def tensor_to_pose(t_btjc, header):
    t = _to_plain(t_btjc)
    if t.dim() == 5: t = t[:, :, 0, :, :]
    if t.dim() == 4:  # [B,T,J,C]
        t = t[0]
    arr = np.ascontiguousarray(t[:, None, :, :], dtype=np.float32)  # [T,1,J,C]
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)
    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    return Pose(header=header, body=body)

def visualize_pose(pose_obj, out_mp4, title="Motion"):
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    viz = PoseVisualizer(pose_obj)
    viz.save_video(pose_obj.body, out_mp4, fps=25)
    print(f"[VIZ] Saved → {out_mp4}")

def get_reduced_header(ref_pose_path):
    with open(ref_pose_path, "rb") as f:
        pose = Pose.read(f)
    if any(c.name == "POSE_WORLD_LANDMARKS" for c in pose.header.components):
        pose = pose.remove_components(["POSE_WORLD_LANDMARKS"])
    pose = reduce_holistic(pose)
    return pose.header


# ---------------------- Main pipeline ----------------------
if __name__ == "__main__":
    pl.seed_everything(42)
    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/overfit_178"
    os.makedirs(out_dir, exist_ok=True)

    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
        reduce_holistic=True,
    )

    small_ds = torch.utils.data.Subset(base_ds, list(range(min(4, len(base_ds)))))
    print(f"[DEBUG] Using subset of {len(small_ds)} samples for overfit test")

    loader = DataLoader(
        small_ds,
        batch_size=4,
        shuffle=True,
        collate_fn=zero_pad_collator,
    )

    batch = next(iter(loader))
    shape = batch["data"].shape

    if len(shape) == 5:
        B, T, P, J, C = shape
        print(f"[INFO] Overfit set: {B}×{T} frames | P={P}, J={J}, C={C}")
    else:
        B, T, J, C = shape
        print(f"[INFO] Overfit set: {B}×{T} frames | J={J}, C={C}")

    model = LitMinimal(num_keypoints=J, num_dims=C, lr=1e-3, log_dir=out_dir)
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        limit_train_batches=1,  # 只用1个batch = 4样本
        limit_val_batches=1,
        log_every_n_steps=1,
        enable_checkpointing=False,
        deterministic=True,
    )

    print(f"[TRAIN] Overfitting on 4 samples × {J} joints × {C} dims")
    trainer.fit(model, loader, loader)

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        cond = batch["conditions"]
        past = cond["input_pose"][:1].to(model.device)
        sign = cond["sign_image"][:1].to(model.device)
        fut  = batch["data"][:1].to(model.device)
        mask = cond["target_mask"][:1].to(model.device)

        pred = model.generate_full_sequence(past_btjc=past, sign_img=sign, target_mask=mask)
        print(f"[GEN] pred shape: {tuple(pred.shape)}, fut shape: {tuple(fut.shape)}")

        dtw_val = masked_dtw(pred, fut, mask).item()
        print(f"[EVAL] masked_dtw = {dtw_val:.4f}")

        try:
            # 尝试对 Pose 对象/标准结构使用官方 unnormalize
            fut_un  = unnormalize_mean_std(fut)
            pred_un = unnormalize_mean_std(pred)
            print("[UNNORM] Applied unnormalize_mean_std ✅")
        except Exception as e:
            # 若输入为 Tensor，则直接 fallback + 手动放大
            print(f"[WARN] Unnormalize fallback (tensor only): {e}")
            fut_un, pred_un = fut.detach().cpu(), pred.detach().cpu()
            scale = 200.0
            fut_un  = fut_un * scale
            pred_un = pred_un * scale
            print(f"[SCALE] multiplied coordinates by {scale}")

        print(f"[UNNORM] fut range=({fut_un.min():.3f},{fut_un.max():.3f}) pred range=({pred_un.min():.3f},{pred_un.max():.3f})")


    ref_path = os.path.join(data_dir, base_ds.records[0]["pose"])
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)

    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS", "FACE_LANDMARKS"])
    ref_pose_reduced = reduce_holistic(ref_pose)
    header = ref_pose_reduced.header

    gt_pose   = tensor_to_pose(fut_un,  header)
    pred_pose = tensor_to_pose(pred_un, header)
    print(f"[POSE] gt shape={gt_pose.body.data.shape}, pred shape={pred_pose.body.data.shape}")

    out_gt, out_pred = os.path.join(out_dir, "gt_178.pose"), os.path.join(out_dir, "pred_178.pose")
    with open(out_gt, "wb") as f: gt_pose.write(f)
    with open(out_pred, "wb") as f: pred_pose.write(f)
    print(f"[SAVE] Reduced 178-joint pose files written → {out_dir}")

    try:
        _ = Pose.read(open(out_pred, "rb"))
        print("[CHECK] Pose file verified: structure OK ✅")
    except Exception as e:
        print(f"[ERROR] Pose read-back failed: {e}")

    visualize_pose(gt_pose,   os.path.join(out_dir, "gt.mp4"),   "Ground Truth")
    visualize_pose(pred_pose, os.path.join(out_dir, "pred.mp4"), "Prediction")
    print("✅ Done! Overfit sanity check (178 joints) complete.")
