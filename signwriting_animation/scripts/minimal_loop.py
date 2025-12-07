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


def visualize_gt_correctly(btjc, name="pose"):
    """
    专门为你的 178 关键点数据设计的可视化函数
    假设数据已经过中心化，Z轴是深度
    """
    if btjc.dim() == 4:
        x = btjc[0].clone()  # [T, J, 3]
    else:
        x = btjc.clone()
    
    T, J, C = x.shape
    print(f"\n====== {name} 可视化 ======")
    print(f"Shape: {x.shape}")
    print(f"原始 X range: [{x[...,0].min():.4f}, {x[...,0].max():.4f}]")
    print(f"原始 Y range: [{x[...,1].min():.4f}, {x[...,1].max():.4f}]")
    print(f"原始 Z range: [{x[...,2].min():.4f}, {x[...,2].max():.4f}]")
    
    # ============================================
    # 策略 1: 直接使用 X, Y，忽略 Z
    # ============================================
    
    # 1. 找到有效点（排除全0点）
    valid_mask = (x[0].abs().sum(dim=-1) > 1e-6)
    print(f"有效关键点: {valid_mask.sum()}/{J}")
    
    if valid_mask.sum() == 0:
        print("⚠️ 没有有效点！")
        return x.unsqueeze(0)
    
    # 2. 只处理 XY 平面（忽略 Z）
    xy = x[..., :2].clone()  # [T, J, 2]
    
    # 3. 使用第一帧的有效点计算中心
    xy_valid = xy[0, valid_mask]
    center = xy_valid.mean(dim=0)  # [2]
    print(f"中心点 (XY): [{center[0]:.4f}, {center[1]:.4f}]")
    
    # 4. 中心化
    xy = xy - center
    
    # 5. 计算缩放因子（使用95百分位避免异常值）
    dist = torch.norm(xy[0, valid_mask], dim=-1)
    k = max(1, int(len(dist) * 0.95))
    scale_ref = torch.topk(dist, k, largest=False)[0].max()
    
    if scale_ref < 1e-3:
        scale_ref = 1.0
    
    # 目标：让身体宽度约为 400 像素
    scale_factor = 200 / scale_ref
    xy = xy * scale_factor
    
    print(f"缩放参考: {scale_ref:.4f}, 缩放因子: {scale_factor:.2f}")
    
    # 6. 翻转 Y 轴（让人正立）
    xy[..., 1] = -xy[..., 1]
    
    # 7. 平移到屏幕中心
    xy[..., 0] += 512
    xy[..., 1] += 384
    
    # 8. 重新组装成 3D（Z 保持不变或清零）
    result = torch.zeros_like(x)
    result[..., :2] = xy
    result[..., 2] = x[..., 2]  # 保留原始 Z（如果需要的话）
    
    print(f"最终 X range: [{result[...,0].min():.1f}, {result[...,0].max():.1f}]")
    print(f"最终 Y range: [{result[...,1].min():.1f}, {result[...,1].max():.1f}]")
    
    # 检查是否有点超出屏幕
    out_x = ((result[..., 0] < 0) | (result[..., 0] > 1024)).sum()
    out_y = ((result[..., 1] < 0) | (result[..., 1] > 768)).sum()
    if out_x > 0 or out_y > 0:
        print(f"⚠️ 超出屏幕: X={out_x}, Y={out_y}")
    
    return result.unsqueeze(0)


def visualize_with_rotation_test(btjc, name="pose"):
    """
    测试不同的坐标轴组合，找到正确的视角
    """
    if btjc.dim() == 4:
        x = btjc[0].clone()
    else:
        x = btjc.clone()
    
    print(f"\n====== {name} 旋转测试 ======")
    
    # 测试 6 种主要视角
    views = {
        "XY平面 (原始)": (0, 1, 2),      # X→X, Y→Y, Z→Z
        "XZ平面 (俯视)": (0, 2, 1),      # X→X, Z→Y, Y→Z
        "YZ平面 (侧视)": (1, 2, 0),      # Y→X, Z→Y, X→Z
        "XY平面 (Y翻转)": (0, 1, 2, True),  # X→X, -Y→Y, Z→Z
        "XZ平面 (Z翻转)": (0, 2, 1, True),  # X→X, -Z→Y, Y→Z
        "ZY平面 (旋转90°)": (2, 1, 0),   # Z→X, Y→Y, X→Z
    }
    
    for view_name, indices in views.items():
        flip_y = len(indices) == 4 and indices[3]
        i, j, k = indices[:3]
        
        # 重新排列坐标轴
        x_view = x.clone()
        x_view_new = torch.zeros_like(x_view)
        x_view_new[..., 0] = x_view[..., i]
        x_view_new[..., 1] = x_view[..., j] if not flip_y else -x_view[..., j]
        x_view_new[..., 2] = x_view[..., k]
        
        # 计算第一帧的空间分布
        valid = (x_view_new[0].abs().sum(dim=-1) > 1e-6)
        if valid.sum() > 0:
            std_x = x_view_new[0, valid, 0].std().item()
            std_y = x_view_new[0, valid, 1].std().item()
            aspect_ratio = std_x / std_y if std_y > 1e-6 else 0
            
            print(f"{view_name:20s} | X_std={std_x:.3f}, Y_std={std_y:.3f}, "
                  f"ratio={aspect_ratio:.2f}")


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

        visualize_with_rotation_test(gt, "GT")
        # 替换你的可视化部分
        gt_f = visualize_gt_correctly(gt, "GT")
        pred_f = visualize_gt_correctly(pred, "PRED")

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
