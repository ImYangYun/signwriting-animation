"""
V2 Clean Test - 超级干净版本

目标：确保GT和pred来自完全相同的pipeline，没有任何隐藏的变换

关键原则：
1. GT和pred都从dataloader来（同样的帧区间）
2. 两者经过完全相同的处理流程
3. tensor_to_pose只做必要的格式转换，不做scale
4. 打印每一步的数据统计，方便debug
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset


def print_tensor_stats(name, t):
    """打印tensor的统计信息，用于debug"""
    if isinstance(t, torch.Tensor):
        t_np = t.detach().cpu().numpy()
    else:
        t_np = t
    
    print(f"  {name}:")
    print(f"    shape: {t_np.shape}")
    print(f"    range: [{t_np.min():.4f}, {t_np.max():.4f}]")
    print(f"    mean: {t_np.mean():.4f}")
    print(f"    std: {t_np.std():.4f}")
    if len(t_np.shape) >= 2:
        # 计算帧间位移
        if t_np.shape[0] > 1 or (len(t_np.shape) > 1 and t_np.shape[1] > 1):
            # 假设第一个或第二个维度是时间
            axis = 0 if t_np.shape[0] > 1 else 1
            disp = np.abs(np.diff(t_np, axis=axis)).mean()
            print(f"    frame_disp: {disp:.4f}")


def simple_tensor_to_pose(t_btjc, header, ref_pose):
    """
    最简单的tensor转pose，只做格式转换，不做任何scale或shift！
    
    这样可以保证数据不被扭曲。
    """
    if t_btjc.dim() == 4:
        t = t_btjc[0]  # [T, J, C]
    else:
        t = t_btjc
    
    t_np = t.detach().cpu().numpy().astype(np.float32)
    
    # 只做格式转换
    arr = t_np[:, None, :, :]  # [T, 1, J, C]
    
    T = arr.shape[0]
    conf = np.ones((T, 1, arr.shape[2]), dtype=np.float32)
    fps = ref_pose.body.fps
    
    body = NumPyPoseBody(fps=fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)
    
    return pose_obj


def test_improved_clean():
    """干净的测试函数"""
    
    # Import
    from signwriting_animation.diffusion.core.models import create_v2_model
    from signwriting_animation.diffusion.lightning_module import (
        LitDiffusion, sanitize_btjc, mean_frame_disp
    )
    
    print("=" * 70)
    print("V2 CLEAN TEST")
    print("=" * 70)
    
    # Config
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    
    NUM_SAMPLES = 4
    MAX_EPOCHS = 500
    BATCH_SIZE = 4
    
    out_dir = "logs/v2_clean_test"
    os.makedirs(out_dir, exist_ok=True)
    
    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    class SubsetDataset(torch.utils.data.Dataset):
        def __init__(self, base, indices):
            self.samples = [base[i] for i in indices]
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]
    
    train_ds = SubsetDataset(base_ds, list(range(NUM_SAMPLES)))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=zero_pad_collator)
    
    # Get dimensions
    sample = train_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    num_joints = sample.shape[-2]
    num_dims = sample.shape[-1]
    future_len = sample.shape[0]
    
    print(f"\nDataset: {NUM_SAMPLES} samples, J={num_joints}, D={num_dims}, T={future_len}")
    
    # Create model (improved version)
    model_kwargs = {
        'num_keypoints': num_joints,
        'num_dims_per_keypoint': num_dims,
        't_past': 40,
        't_future': future_len,
    }
    
    custom_model = create_v2_model('improved', **model_kwargs)
    
    lit_model = LitDiffusion(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,
        diffusion_steps=8,
        vel_weight=1.0,
        t_past=40,
        t_future=future_len,
    )
    lit_model.model = custom_model
    
    # Train
    print(f"\nTraining improved model for {MAX_EPOCHS} epochs...")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=50,
        enable_progress_bar=False,
    )
    trainer.fit(lit_model, train_loader)
    
    # Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.eval()
    
    # Get test batch
    test_batch = zero_pad_collator([train_ds[0]])
    cond = test_batch["conditions"]
    
    # ========================================
    # 关键：跟踪每一步的数据
    # ========================================
    
    print("\n" + "=" * 70)
    print("DATA PIPELINE TRACKING")
    print("=" * 70)
    
    # Step 1: 原始数据从dataloader
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(test_batch["data"][:1]).to(device)
    
    print("\n[Step 1] Raw data from dataloader:")
    print_tensor_stats("past_raw", past_raw)
    print_tensor_stats("gt_raw", gt_raw)
    
    # Step 2: Normalize
    past_norm = lit_model.normalize(past_raw)
    gt_norm = lit_model.normalize(gt_raw)
    
    print("\n[Step 2] After normalize:")
    print_tensor_stats("past_norm", past_norm)
    print_tensor_stats("gt_norm", gt_norm)
    
    # Step 3: Model inference (in normalized space)
    with torch.no_grad():
        past_bjct = lit_model.btjc_to_bjct(past_norm)
        
        B, J, C, _ = past_bjct.shape
        target_shape = (B, J, C, future_len)
        
        class _ConditionalWrapper(torch.nn.Module):
            def __init__(self, base_model, past_bjct, sign_img):
                super().__init__()
                self.base_model = base_model
                self.past_bjct = past_bjct
                self.sign_img = sign_img
            
            def forward(self, x, t, **kwargs):
                return self.base_model(x, t, self.past_bjct, self.sign_img)
        
        wrapped_model = _ConditionalWrapper(lit_model.model, past_bjct, sign)
        
        pred_bjct = lit_model.diffusion.p_sample_loop(
            model=wrapped_model,
            shape=target_shape,
            clip_denoised=False,
            model_kwargs={"y": {}},
            progress=False,
        )
        
        pred_norm = lit_model.bjct_to_btjc(pred_bjct)
    
    print("\n[Step 3] Model prediction (normalized space):")
    print_tensor_stats("pred_norm", pred_norm)
    
    # Step 4: Compute metrics in normalized space
    print("\n[Step 4] Metrics in NORMALIZED space:")
    mse = F.mse_loss(pred_norm, gt_norm).item()
    disp_pred = mean_frame_disp(pred_norm)
    disp_gt = mean_frame_disp(gt_norm)
    disp_ratio = disp_pred / (disp_gt + 1e-8)
    
    pred_np = pred_norm[0].cpu().numpy()
    gt_np = gt_norm[0].cpu().numpy()
    per_joint_err = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
    mpjpe = per_joint_err.mean()
    pck_01 = (per_joint_err < 0.1).mean() * 100
    
    print(f"  MSE: {mse:.6f}")
    print(f"  MPJPE: {mpjpe:.6f}")
    print(f"  PCK@0.1: {pck_01:.1f}%")
    print(f"  Disp GT: {disp_gt:.4f}")
    print(f"  Disp Pred: {disp_pred:.4f}")
    print(f"  Disp Ratio: {disp_ratio:.4f}")
    
    # Step 5: Unnormalize for visualization
    gt_unnorm = lit_model.unnormalize(gt_norm)
    pred_unnorm = lit_model.unnormalize(pred_norm)
    
    print("\n[Step 5] After unnormalize:")
    print_tensor_stats("gt_unnorm", gt_unnorm)
    print_tensor_stats("pred_unnorm", pred_unnorm)
    
    # Step 6: Verify unnormalize is reversible
    gt_renorm = lit_model.normalize(gt_unnorm)
    diff = (gt_norm - gt_renorm).abs().max().item()
    print(f"\n[Step 6] Verify normalize/unnormalize reversible:")
    print(f"  Max diff after round-trip: {diff:.10f}")
    if diff < 1e-5:
        print("  ✅ normalize/unnormalize is reversible!")
    else:
        print("  ⚠️ WARNING: normalize/unnormalize has precision loss!")
    
    # Step 7: Save poses (NO scaling, NO shifting - just format conversion)
    print("\n[Step 7] Saving poses (minimal processing)...")
    
    ref_path = base_ds.records[0]["pose"]
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    ref_pose = reduce_holistic(ref_pose)
    if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
        ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    
    # 使用最简单的转换，不做任何scale或shift
    gt_pose = simple_tensor_to_pose(gt_unnorm, ref_pose.header, ref_pose)
    pred_pose = simple_tensor_to_pose(pred_unnorm, ref_pose.header, ref_pose)
    
    with open(f"{out_dir}/gt.pose", "wb") as f:
        gt_pose.write(f)
    with open(f"{out_dir}/pred.pose", "wb") as f:
        pred_pose.write(f)
    
    # Step 8: Verify saved poses
    print("\n[Step 8] Verify saved poses:")
    gt_saved = gt_pose.body.data[:, 0, :, :]
    pred_saved = pred_pose.body.data[:, 0, :, :]
    
    print_tensor_stats("gt_pose (saved)", gt_saved)
    print_tensor_stats("pred_pose (saved)", pred_saved)
    
    # Compare GT and pred in pose file
    pose_diff = np.abs(gt_saved - pred_saved).mean()
    print(f"\n  Mean diff between gt and pred pose: {pose_diff:.4f}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nNormalized space metrics (for paper):")
    print(f"  Disp Ratio: {disp_ratio:.4f} (should be ~1.0)")
    print(f"  MPJPE: {mpjpe:.6f}")
    print(f"  PCK@0.1: {pck_01:.1f}%")
    
    print(f"\nPose files saved to: {out_dir}/")
    print(f"  - gt.pose: GT from dataloader (帧区间由dataloader决定)")
    print(f"  - pred.pose: Model prediction (同样的帧区间)")
    
    print("\n" + "=" * 70)
    print("✅ CLEAN TEST COMPLETE!")
    print("=" * 70)
    
    return {
        'disp_ratio': disp_ratio,
        'mpjpe': mpjpe,
        'pck_01': pck_01,
    }


if __name__ == "__main__":
    pl.seed_everything(42)
    test_improved_clean()