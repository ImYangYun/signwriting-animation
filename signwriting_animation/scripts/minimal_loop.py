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

try:
    from pose_anonymization.data.normalization import unshift_hands
    HAS_UNSHIFT = True
    print("[✓] Successfully imported unshift_hands")
except ImportError as e:
    print(f"[✗] Warning: Could not import unshift_hands: {e}")
    print("[!] PRED poses will have incorrect hand positions!")
    HAS_UNSHIFT = False


def tensor_to_pose_complete(t_btjc, header, ref_pose, apply_unshift: bool = True):
    """
    完整的 tensor → pose 转换：
    1. 正确的 confidence shape (3D)
    2. 使用 GT 的 FPS 和 confidence
    3. 可选地调用 unshift_hands
    """
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    elif t_btjc.dim() == 3:
        t = t_btjc
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {t_btjc.dim()}D")

    t_np = t.cpu().numpy().astype(np.float32)

    print(f"\n[tensor_to_pose_complete] (apply_unshift={apply_unshift})")
    print(f"  输入 shape: {t_np.shape}")
    print(f"  输入 range: [{t_np.min():.4f}, {t_np.max():.4f}]")

    # 创建 pose 对象
    arr = t_np[:, None, :, :]  # [T, 1, J, C]
    conf = ref_pose.body.confidence[:len(t_np)].copy()  # 使用 GT 的 confidence
    fps = ref_pose.body.fps  # 使用 GT 的 FPS

    body = NumPyPoseBody(fps=fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)

    print(f"  创建 Pose:")
    print(f"    fps: {fps}")
    print(f"    data shape: {pose_obj.body.data.shape}")
    print(f"    conf shape: {pose_obj.body.confidence.shape}")
    print(f"    data range: [{pose_obj.body.data.min():.4f}, {pose_obj.body.data.max():.4f}]")

    # 是否调用 unshift_hands
    if apply_unshift and HAS_UNSHIFT:
        print(f"\n  调用 unshift_hands...")
        try:
            unshift_hands(pose_obj)
            print(f"    ✓ unshift 成功")
            print(f"    new range: [{pose_obj.body.data.min():.4f}, {pose_obj.body.data.max():.4f}]")
        except Exception as e:
            print(f"    ✗ unshift 失败: {e}")
    elif apply_unshift and not HAS_UNSHIFT:
        print(f"\n  ⚠️  预期 unshift_hands 但未导入成功")
    else:
        print(f"\n  ⚠️  本次不调用 unshift_hands，仅写入 raw 坐标")

    return pose_obj


def inspect_pose(path: str, name: str):
    """
    读回 .pose 文件，打印:
    - 数据 shape
    - 全局最小/最大值
    - 每帧骨架的方差（关节在质心附近的分散程度）
    """
    if not os.path.exists(path):
        print(f"\n[{name}] 文件不存在: {path}")
        return

    with open(path, "rb") as f:
        pose = Pose.read(f)

    data = pose.body.data  # [T, P, J, C]，numpy
    data_np = np.asarray(data, dtype=np.float32)
    T, P, J, C = data_np.shape

    # 把 person 和 joint 合并，只看一帧内的空间分布
    data_tpjc = data_np.reshape(T, P * J, C)
    center = data_tpjc.mean(axis=1, keepdims=True)  # [T, 1, C]
    var = ((data_tpjc - center) ** 2).mean(axis=(1, 2))  # [T]

    print(f"\n[{name}] {path}")
    print(f"  shape: {data_np.shape}")
    print(f"  range: [{data_np.min():.4f}, {data_np.max():.4f}]")
    print(f"  per-frame var min/max: [{var.min():.6f}, {var.max():.6f}]")


if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    out_dir = "logs/minimal_178_aligned"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"

    print("\n" + "="*70)
    print("完全对齐版本")
    print("="*70)
    print("  ✅ LightningModule.unnormalize: 只做数值反归一化")
    print("  ✅ tensor_to_pose: 调用 unshift_hands")
    print("  ✅ Confidence: 3D shape, GT 的连续值")
    print("  ✅ FPS: 使用 GT 的 FPS")
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
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=False, collate_fn=zero_pad_collator
    )

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

    print("\n[训练中...]")
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

        print(f"\n[采样] diffusion_steps=50, future_len={future_len}")

        # 这里 sample_autoregressive_fast 已在内部做过 unnormalize
        pred = model.sample_autoregressive_fast(
            past_btjc=past,
            sign_img=sign,
            future_len=future_len,
            chunk=20,
        )

        print(f"\nGT (raw):   [{gt.min():.4f}, {gt.max():.4f}]")
        print(f"PRED (raw): [{pred.min():.4f}, {pred.max():.4f}]")

        mask_bt = torch.ones(1, future_len, device=device)
        dtw_val = masked_dtw(pred, gt, mask_bt)
        print(f"DTW: {dtw_val:.4f}")

    # 加载 GT
    print("\n" + "="*70)
    print("加载参考 pose")
    print("="*70)

    ref_path = base_ds.records[0]["pose"]
    ref_path = ref_path if os.path.isabs(ref_path) else os.path.join(data_dir, ref_path)

    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)

    ref_pose = reduce_holistic(ref_pose)
    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])

    header = ref_pose.header
    n_header_joints = ref_pose.body.data.shape[2] 
    print(f"\nheader joints: {n_header_joints}")
    print(f"pred joints:   {pred.shape[-2]}")

    out_gt = os.path.join(out_dir, "gt_reference.pose")
    with open(out_gt, "wb") as f:
        ref_pose.write(f)
    print(f"\n✓ GT (参考) 保存: {out_gt}")

    print("\n" + "="*70)
    print("保存 PRED（完整流程，有 unshift_hands）")
    print("="*70)

    pose_pred = tensor_to_pose_complete(pred, header, ref_pose, apply_unshift=True)
    out_pred = os.path.join(out_dir, "pred_complete.pose")
    with open(out_pred, "wb") as f:
        pose_pred.write(f)
    print(f"\n✓ PRED 保存: {out_pred}")

    # 保存 PRED（不做 unshift_hands 的版本，用于 AB 测试）
    print("\n" + "="*70)
    print("保存 PRED（no_unshift 版本）")
    print("="*70)

    pose_pred_no = tensor_to_pose_complete(pred, header, ref_pose, apply_unshift=False)
    out_pred_no = os.path.join(out_dir, "pred_no_unshift.pose")
    with open(out_pred_no, "wb") as f:
        pose_pred_no.write(f)
    print(f"\n✓ PRED (no_unshift) 保存: {out_pred_no}")

    # 读回三个 pose 做数值检查
    print("\n" + "="*70)
    print("DEBUG: 读回 .pose 文件检查分布")
    print("="*70)

    inspect_pose(out_gt, "GT")
    inspect_pose(out_pred, "PRED_with_unshift")
    inspect_pose(out_pred_no, "PRED_no_unshift")

    # 最终总结
    print("\n" + "="*70)
    print("✓ 完成！")
    print("="*70)
    print(f"\n生成的文件:")
    print(f"  1. GT (参考):          {out_gt}")
    print(f"  2. PRED (unshift):     {out_pred}")
    print(f"  3. PRED (no_unshift):  {out_pred_no}")

    print(f"\n数据流程:")
    print(f"  训练时:")
    print(f"    原始 pose → pre_process (shift_hands) → normalize → 训练")
    print(f"  Inference:")
    print(f"    模型输出 → unnormalize（已在 sample_autoregressive_fast 内部完成）"
          f"→ (可选)unshift_hands → 保存")

    print(f"\n在 sign.mt 中测试:")
    print(f"  1. 打开 gt_reference.pose")
    print(f"  2. 打开 pred_complete.pose (有 unshift)")
    print(f"  3. 打开 pred_no_unshift.pose (无 unshift)")

    if not HAS_UNSHIFT:
        print(f"\n⚠️  警告:")
        print(f"  unshift_hands 未成功导入")
        print(f"  PRED 的手部位置可能不正确")
        print(f"  请确保 pose_anonymization 包可用")
