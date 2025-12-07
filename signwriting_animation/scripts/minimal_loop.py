# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc, masked_dtw

# ----------- DEBUG: confirm which lightning_module is actually used -----------
import signwriting_animation.diffusion.lightning_module as LM
print(">>> USING LIGHTNING MODULE FROM:", LM.__file__)
# -------------------------------------------------------------------------------


def _to_plain(x):
    """Convert pose-format tensors to contiguous float32 CPU tensor."""
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().cpu().contiguous().float()


def temporal_smooth(x, k=5):
    """Simple temporal smoothing for visualization."""
    import torch.nn.functional as F
    if x.dim() == 4:
        x = x[0]

    T, J, C = x.shape
    x = x.permute(2,1,0).reshape(1, C*J, T)
    x = F.avg_pool1d(x, kernel_size=k, stride=1, padding=k//2)
    x = x.reshape(C, J, T).permute(2,1,0)
    return x.contiguous()


def debug_and_visualize(btjc, name="pose"):
    """诊断 + 可视化一体函数（支持 deterministic mode）"""
    if btjc.dim() == 4:
        x = btjc[0].clone()
    else:
        x = btjc.clone()
    
    T, J, C = x.shape
    print(f"\n====== {name} 诊断 ======")
    print(f"Shape: {x.shape}")
    print(f"X range: [{x[...,0].min():.4f}, {x[...,0].max():.4f}]")
    print(f"Y range: [{x[...,1].min():.4f}, {x[...,1].max():.4f}]")
    print(f"Z range: [{x[...,2].min():.4f}, {x[...,2].max():.4f}]")
    
    # 检查是否所有点都在同一位置
    x_std = x[0, :, 0].std()
    y_std = x[0, :, 1].std()
    print(f"First frame std: X={x_std:.6f}, Y={y_std:.6f}")
    
    if x_std < 0.01 and y_std < 0.01:
        print("⚠️  警告：所有关键点几乎在同一位置！")
    
    # ===== 可视化变换 =====
    x0 = x[0]  # [J, 3]
    
    # 找到有效点
    valid_mask = (x0.abs().sum(dim=-1) > 1e-6)
    if valid_mask.sum() == 0:
        print("⚠️  错误：没有有效的关键点！")
        return x.unsqueeze(0)
    
    x_valid = x0[valid_mask]
    print(f"有效关键点数: {valid_mask.sum()}/{J}")
    
    # 1. 中心化：使用均值（deterministic-safe）
    center = x_valid.mean(dim=0)  # [3]
    x = x - center
    print(f"中心点: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    
    # 2. 翻转Y轴（让人物正立）
    x[..., 1] = -x[..., 1]
    
    # 3. 自适应缩放：基于有效点的距离分布
    dist = torch.norm(x[0, valid_mask, :2], dim=-1)
    # 使用90百分位数来排除异常点
    k = int(len(dist) * 0.9)
    scale_ref = torch.topk(dist, k, largest=False)[0].max()
    
    if scale_ref < 1e-3:
        scale_ref = 1.0
    
    scale_factor = 250 / scale_ref  # 让90%的点在250像素内
    x[..., :2] *= scale_factor
    
    print(f"缩放参考距离: {scale_ref:.4f}, 缩放因子: {scale_factor:.2f}")
    
    # 4. 平移到屏幕中心（pose-viewer 默认分辨率 1024x768）
    x[..., 0] += 512
    x[..., 1] += 384
    
    print(f"变换后 X range: [{x[...,0].min():.1f}, {x[...,0].max():.1f}]")
    print(f"变换后 Y range: [{x[...,1].min():.1f}, {x[...,1].max():.1f}]")
    
    # 检查是否有点超出屏幕
    out_of_bounds = (x[..., 0] < 0) | (x[..., 0] > 1024) | (x[..., 1] < 0) | (x[..., 1] > 768)
    if out_of_bounds.any():
        print(f"⚠️  警告：{out_of_bounds.sum().item()} 个点超出屏幕范围")
    
    return x.unsqueeze(0)


def tensor_to_pose(t_btjc, header):
    """Convert tensor → Pose-format object."""
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    elif t_btjc.dim() == 3:
        t = t_btjc
    else:
        raise ValueError

    print("[tensor_to_pose] final shape:", t.shape)

    arr = t[:, None, :, :].cpu().numpy().astype(np.float32)
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)

    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    return Pose(header=header, body=body)


if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    out_dir = "logs/minimal_178"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178.pt"
    stats = torch.load(stats_path)
    print("mean shape:", stats["mean"].shape)
    print("std shape:", stats["std"].shape)
    print("std min/max:", stats["std"].min(), stats["std"].max())


    # Dataset + reduction (178 joints)
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    base_ds.mean_std = torch.load(stats_path)

    small_ds = torch.utils.data.Subset(base_ds, [0, 1, 2, 3])
    loader = DataLoader(
        small_ds,
        batch_size=4,
        shuffle=True,
        collate_fn=zero_pad_collator,
    )

    batch0 = next(iter(loader))
    raw = sanitize_btjc(batch0["data"][0:1]).clone().cpu()

    print("====== RAW DATA STATS ======")
    print("raw.min =", raw.min().item(), " raw.max =", raw.max().item())
    print("raw[0, :10] =", raw[0, :10])
    print("RAW shape:", raw.shape)

    num_joints = batch0["data"].shape[-2]
    num_dims   = batch0["data"].shape[-1]
    print(f"[INFO] joints={num_joints}, dims={num_dims}")

    # Model
    model = LitMinimal(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-4,
    )

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_checkpointing=False,
        deterministic=True,
    )

    print("[TRAIN] Overfit 4 samples…")
    trainer.fit(model, loader, loader)

    # Load original header (reduced)
    ref_path = base_ds.records[0]["pose"]
    ref_path = ref_path if os.path.isabs(ref_path) else os.path.join(data_dir, ref_path)
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)

    # ---- correct sequence ----
    ref_p = reduce_holistic(ref_pose)
    ref_p = ref_p.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_p.header

    print("[CHECK HEADER] total joints:", header.total_points())

    print("[CHECK HEADER] total joints:", header.total_points())

    # ============================================================
    # Inference
    # ============================================================

    model.eval()
    device = trainer.strategy.root_device
    model = model.to(device)
    model.mean_pose = model.mean_pose.to(device)
    model.std_pose  = model.std_pose.to(device)

    with torch.no_grad():
        batch = next(iter(loader))
        cond  = batch["conditions"]

        raw_gt = batch["data"][0, 0]
        print("\n====== RAW GT FIRST FRAME (MaskedTensor) ======")
        print(type(raw_gt))

        if hasattr(raw_gt, "zero_filled"):
            dense = raw_gt.zero_filled()
            print("dense[:10] =", dense[:10])
            print("dense min/max =", dense.min(), dense.max())
            print("dense shape =", dense.shape)
        else:
            print("raw_gt[:10] =", raw_gt[:10])


        past = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt   = sanitize_btjc(batch["data"][:1]).to(device)

        future_len = gt.size(1)
        print("[SAMPLE] future_len =", future_len)

        # 1. Generate normalized prediction
        pred_norm = model.sample_autoregressive_fast(
            past_btjc=past,
            sign_img=sign,
            future_len=future_len,
            chunk=1,
        )

        # 2. unnormalize to BTJC
        pred = model.unnormalize(pred_norm)

        # 3. Smoothing (optional)
        #pred_s = temporal_smooth(pred)
        #gt_s   = temporal_smooth(gt)

        # 4. Visualization transform
        #pred_f = visualize_pose(pred, scale=250, offset=(500, 500))
        #gt_f   = visualize_pose(gt,  scale=250, offset=(500, 500))

        # 替换你的可视化部分
        gt_f   = debug_and_visualize(gt, "GT")
        pred_f = debug_and_visualize(pred, "PRED")

        print("gt_f shape:", gt_f.shape)
        print("pred_f shape:", pred_f.shape)

        # --- DTW evaluation ---
        mask_bt = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred, gt, mask_bt)
        print(f"[DTW] masked_dtw (unnormalized) = {dtw_val:.4f}")

    # ============================================================
    # Save .pose for viewer
    # ============================================================

    pose_gt = tensor_to_pose(gt_f, header)
    pose_pr = tensor_to_pose(pred_f, header)

    out_gt = os.path.join(out_dir, "gt_178.pose")
    out_pr = os.path.join(out_dir, "pred_178.pose")

    for p in [out_gt, out_pr]:
        if os.path.exists(p):
            os.remove(p)

    with open(out_gt, "wb") as f:
        pose_gt.write(f)
    with open(out_pr, "wb") as f:
        pose_pr.write(f)

    print("[SAVE] GT & Pred pose saved ✔")
