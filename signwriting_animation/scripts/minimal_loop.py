# -*- coding: utf-8 -*-
"""
完全修复版：
1. 修复 confidence shape (去掉多余维度)
2. 修复 FPS (使用 GT 的 FPS)
3. 保留连续的 confidence 值
4. 对比数据范围
"""
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


def tensor_to_pose_fixed(t_btjc, header, gt_body):
    """
    修复所有问题：
    1. Confidence shape 正确 (3D 不是 4D)
    2. 使用 GT 的 FPS
    3. 从 GT 学习 confidence
    """
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    elif t_btjc.dim() == 3:
        t = t_btjc
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {t_btjc.dim()}D")
    
    t_np = t.cpu().numpy().astype(np.float32)
    
    print(f"\n[tensor_to_pose_fixed] 修复版:")
    print(f"  PRED data shape: {t_np.shape}")
    print(f"  PRED data range: [{t_np.min():.4f}, {t_np.max():.4f}]")
    
    # arr: [T, 1, J, C]
    arr = t_np[:, None, :, :]
    
    # 🔧 修复 1: Confidence shape 正确 - 3D 不是 4D
    # 错误：conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), ...)  # 4D
    # 正确：conf = np.ones((arr.shape[0], 1, arr.shape[2]), ...)     # 3D
    
    # 🔧 修复 2: 从 GT 学习 confidence 的模式
    # GT 的前 20 帧 confidence
    gt_conf_20 = gt_body.confidence[:20]  # shape: (20, 1, 178)
    
    print(f"\n  GT confidence (前20帧):")
    print(f"    shape: {gt_conf_20.shape}")
    print(f"    range: [{gt_conf_20.min():.4f}, {gt_conf_20.max():.4f}]")
    print(f"    唯一值数量: {len(np.unique(gt_conf_20))}")
    
    # 使用 GT 的 confidence
    conf = gt_conf_20.copy()
    
    print(f"\n  PRED confidence (使用GT的):")
    print(f"    shape: {conf.shape}")
    print(f"    =0: {(conf == 0).sum()} / {conf.size}")
    print(f"    =1: {(conf == 1).sum()} / {conf.size}")
    print(f"    (0,1): {((conf > 0) & (conf < 1)).sum()} / {conf.size}")
    
    # 🔧 修复 3: 使用 GT 的 FPS
    fps = gt_body.fps
    print(f"\n  使用 GT 的 FPS: {fps}")
    
    body = NumPyPoseBody(fps=fps, data=arr, confidence=conf)
    
    print(f"\n  最终 body:")
    print(f"    fps: {body.fps}")
    print(f"    data shape: {body.data.shape}")
    print(f"    conf shape: {body.confidence.shape}")
    
    return Pose(header=header, body=body)


def analyze_data_range(gt_body, pred_tensor):
    """分析数据范围差异"""
    print(f"\n" + "="*70)
    print("数据范围分析")
    print("="*70)
    
    gt_data = gt_body.data
    pred_np = pred_tensor[0].cpu().numpy() if pred_tensor.dim() == 4 else pred_tensor.cpu().numpy()
    
    print(f"\n[GT (文件中)]")
    print(f"  shape: {gt_data.shape}")
    print(f"  range: [{gt_data.min():.4f}, {gt_data.max():.4f}]")
    print(f"  非零 range: [{gt_data[gt_data != 0].min():.4f}, {gt_data[gt_data != 0].max():.4f}]")
    
    print(f"\n[PRED (归一化空间)]")
    print(f"  shape: {pred_np.shape}")
    print(f"  range: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
    
    print(f"\n⚠️ 注意：")
    print(f"  GT 文件中的数据是原始像素坐标 (range ~600)")
    print(f"  PRED 是归一化后的坐标 (range ~2)")
    print(f"  这是正常的 - 我们的 PRED 已经经过 unnormalize")
    print(f"  但 unnormalize 后的范围应该和训练时的 GT 一致")


if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    out_dir = "logs/minimal_178_fixed_all"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"

    print("\n" + "="*70)
    print("完全修复版")
    print("="*70 + "\n")

    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )

    sample_0 = base_ds[0]
    
    class FixedSampleDataset(torch.utils.data.Dataset):
        def __init__(self, sample):
            self.sample = sample
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return self.sample
    
    train_ds = FixedSampleDataset(sample_0)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=zero_pad_collator)

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=50,
    )

    num_joints = sample_0["data"].shape[-2]
    num_dims = sample_0["data"].shape[-1]

    model = LitMinimal(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,
        diffusion_steps=50,
        beta_start=1e-4,
        beta_end=2e-2,
        pred_target="x0",
    )

    trainer.fit(model, train_loader)

    # Inference
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

        print(f"\nGT (训练时):   [{gt.min():.4f}, {gt.max():.4f}]")
        print(f"PRED (生成):   [{pred.min():.4f}, {pred.max():.4f}]")

        mask_bt = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred, gt, mask_bt)
        print(f"DTW: {dtw_val:.4f}")

    # 加载 GT 文件
    ref_path = base_ds.records[0]["pose"]
    ref_path = ref_path if os.path.isabs(ref_path) else os.path.join(data_dir, ref_path)

    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)

    ref_reduced = reduce_holistic(ref_pose)
    ref_reduced = ref_reduced.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_reduced.header

    gt_pose_obj = reduce_holistic(ref_pose)
    gt_pose_obj = gt_pose_obj.remove_components(["POSE_WORLD_LANDMARKS"])
    
    # 分析数据范围
    analyze_data_range(gt_pose_obj.body, pred)

    # 保存 GT
    out_gt = os.path.join(out_dir, "gt_final.pose")
    with open(out_gt, "wb") as f:
        gt_pose_obj.write(f)
    print(f"\n✓ GT 保存: {out_gt}")

    # 保存 PRED (完全修复)
    print("\n" + "="*70)
    print("保存 PRED (完全修复版)")
    print("="*70)
    
    pose_pred = tensor_to_pose_fixed(pred, header, gt_pose_obj.body)
    out_pred = os.path.join(out_dir, "pred_fully_fixed.pose")
    with open(out_pred, "wb") as f:
        pose_pred.write(f)
    print(f"\n✓ PRED 保存: {out_pred}")

    print("\n" + "="*70)
    print("✓ 完成！")
    print("="*70)
    print(f"\n生成的文件:")
    print(f"  1. GT:                {out_gt}")
    print(f"  2. PRED (完全修复):   {out_pred}")
    print(f"\n修复内容:")
    print(f"  ✅ Confidence shape: (20, 1, 178) - 去掉多余维度")
    print(f"  ✅ FPS: 使用 GT 的 {gt_pose_obj.body.fps}")
    print(f"  ✅ Confidence 值: 使用 GT 的连续值")
    print(f"\n在 sign.mt 中打开 pred_fully_fixed.pose")
    print(f"应该能正常显示了！")