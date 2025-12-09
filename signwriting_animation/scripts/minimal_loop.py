# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc, masked_dtw

import signwriting_animation.diffusion.lightning_module as LM
print(">>> USING LIGHTNING MODULE FROM:", LM.__file__)


def tensor_to_pose(t_btjc, header):
    """
    Convert tensor → Pose-format object.
    """
    import numpy as np
    from pose_format.numpy.pose_body import NumPyPoseBody
    from pose_format import Pose
    
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    elif t_btjc.dim() == 3:
        t = t_btjc
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {t_btjc.dim()}D")
    
    t_np = t.cpu().numpy().astype(np.float32)
    
    # 坐标轴修正
    t_np_fixed = np.stack([
        t_np[:, :, 1],  # Y → X
        t_np[:, :, 2],  # Z → Y
        t_np[:, :, 0]   # X → Z
    ], axis=-1)
    
    arr = t_np_fixed[:, None, :, :]
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)
    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    
    return Pose(header=header, body=body)


if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    out_dir = "logs/minimal_178_fixed"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"

    print("\n" + "="*70)
    print("Overfit 实验 - 真正固定单个样本")
    print("="*70)

    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )

    print(f"[INFO] 总样本数: {len(base_ds)}")
    print(f"[OVERFIT] 固定使用样本 index=0\n")
    
    # 🔧 创建固定样本的 Dataset
    sample_0 = base_ds[0]
    
    class FixedSampleDataset(torch.utils.data.Dataset):
        """每次都返回同一个样本"""
        def __init__(self, sample):
            self.sample = sample
        
        def __len__(self):
            return 1
        
        def __getitem__(self, idx):
            return self.sample
    
    train_ds = FixedSampleDataset(sample_0)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=zero_pad_collator,
    )

    print("[TRAINER CONFIG]")
    print("  max_epochs: 1000")
    print("  lr: 1e-3")
    print("  diffusion_steps: 200")
    print("  样本: 固定第 0 个样本（真正的 overfit）\n")

    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        deterministic=False,
        log_every_n_steps=50,
    )

    num_joints = sample_0["data"].shape[-2]
    num_dims = sample_0["data"].shape[-1]
    print(f"[INFO] joints={num_joints}, dims={num_dims}\n")

    # Model
    model = LitMinimal(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,
        diffusion_steps=50,  # 增加 steps
        beta_start=1e-4,
        beta_end=2e-2,
        pred_target="x0",
    )

    print("[TRAIN] 开始训练...")
    print("  预期: loss 应该快速降到 < 0.01")
    print("="*70 + "\n")
    
    trainer.fit(model, train_loader)

    # ============================================================
    # Inference
    # ============================================================
    print("\n" + "="*70)
    print("INFERENCE")
    print("="*70)

    model.eval()
    device = trainer.strategy.root_device
    model = model.to(device)

    with torch.no_grad():
        batch = next(iter(train_loader))
        cond = batch["conditions"]

        past = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt = sanitize_btjc(batch["data"][:1]).to(device)

        future_len = gt.size(1)

        pred_norm = model.sample_autoregressive_fast(
            past_btjc=past,
            sign_img=sign,
            future_len=future_len,
            chunk=20,
        )

        pred = model.unnormalize(pred_norm)

        # DTW
        mask_bt = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred, gt, mask_bt)
        print(f"\nDTW: {dtw_val:.4f}")
        print(f"预期 overfit 成功: DTW < 0.1")

    # ============================================================
    # 保存文件
    # ============================================================
    ref_path = base_ds.records[0]["pose"]
    ref_path = ref_path if os.path.isabs(ref_path) else os.path.join(data_dir, ref_path)

    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)

    ref_reduced = reduce_holistic(ref_pose)
    ref_reduced = ref_reduced.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_reduced.header

    # GT
    gt_pose_obj = reduce_holistic(ref_pose)
    gt_pose_obj = gt_pose_obj.remove_components(["POSE_WORLD_LANDMARKS"])
    out_gt = os.path.join(out_dir, "gt_final.pose")
    with open(out_gt, "wb") as f:
        gt_pose_obj.write(f)

    # PRED
    pose_pred = tensor_to_pose(pred, header)
    out_pred = os.path.join(out_dir, "pred_final.pose")
    with open(out_pred, "wb") as f:
        pose_pred.write(f)

    print(f"\n✓ 文件已保存:")
    print(f"  GT:   {out_gt}")
    print(f"  PRED: {out_pred}")