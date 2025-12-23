"""
4-Sample Overfit Test with Displacement Loss
验证 disp_loss 是否能改善 inference 时的 displacement ratio
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader, Subset

from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from signwriting_animation.diffusion.lightning_disp import (
    LitDiffusion, sanitize_btjc, mean_frame_disp
)


def test_4sample_overfit():
    """4-sample overfit test with displacement loss."""
    pl.seed_everything(42)

    # ============================================================
    # CONFIGURATION
    # ============================================================
    NUM_SAMPLES = 4
    MAX_EPOCHS = 100  # 足够 overfit
    DIFFUSION_STEPS = 8
    BATCH_SIZE = 4
    DISP_WEIGHT = 1.0  # displacement loss 权重
    
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = f"logs/4sample_disp{DISP_WEIGHT}"
    # ============================================================
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 70)
    print("4-SAMPLE OVERFIT TEST (with Displacement Loss)")
    print("=" * 70)
    print(f"  NUM_SAMPLES: {NUM_SAMPLES}")
    print(f"  MAX_EPOCHS: {MAX_EPOCHS}")
    print(f"  DIFFUSION_STEPS: {DIFFUSION_STEPS}")
    print(f"  DISP_WEIGHT: {DISP_WEIGHT}")
    print(f"  Output: {out_dir}")
    print("=" * 70)

    # Load dataset
    full_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    # 取前 4 个样本
    train_ds = Subset(full_ds, list(range(NUM_SAMPLES)))
    print(f"\nUsing {len(train_ds)} samples for overfit test")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=zero_pad_collator,
    )

    # Get dimensions
    sample = full_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    num_joints = sample.shape[-2]
    num_dims = sample.shape[-1]
    future_len = sample.shape[0]

    print(f"Dimensions: J={num_joints}, D={num_dims}, T={future_len}")

    # Create model with displacement loss
    model = SignWritingToPoseDiffusion(
        num_keypoints=num_joints,
        num_dims_per_keypoint=num_dims,
        t_past=40,
        t_future=future_len,
    )

    lit_model = LitDiffusion(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-4,
        diffusion_steps=DIFFUSION_STEPS,
        vel_weight=1.0,
        acc_weight=0.5,
        disp_weight=DISP_WEIGHT,  # NEW: displacement loss
        t_past=40,
        t_future=future_len,
    )
    lit_model.model = model

    # Train
    print(f"\n{'='*70}")
    print("TRAINING...")
    print("="*70)
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        default_root_dir=out_dir,
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_checkpointing=False,  # 4-sample 不需要存 checkpoint
    )
    trainer.fit(lit_model, train_loader)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE! Now testing inference...")
    print("="*70)

    # ====== INFERENCE TEST ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.eval()

    inference_results = []
    
    for idx in range(NUM_SAMPLES):
        test_batch = zero_pad_collator([full_ds[idx]])
        cond = test_batch["conditions"]

        past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt_raw = sanitize_btjc(test_batch["data"][:1]).to(device)

        past_norm = lit_model.normalize(past_raw)
        gt_norm = lit_model.normalize(gt_raw)

        # DDPM sampling (same as inference)
        with torch.no_grad():
            past_bjct = lit_model.btjc_to_bjct(past_norm)
            B, J, C, _ = past_bjct.shape
            target_shape = (B, J, C, future_len)

            class _Wrapper(torch.nn.Module):
                def __init__(self, model, past, sign):
                    super().__init__()
                    self.model, self.past, self.sign = model, past, sign
                def forward(self, x, t, **kwargs):
                    return self.model(x, t, self.past, self.sign)

            wrapped = _Wrapper(lit_model.model, past_bjct, sign)

            pred_bjct = lit_model.diffusion.p_sample_loop(
                model=wrapped,
                shape=target_shape,
                clip_denoised=True,
                model_kwargs={"y": {}},
                progress=False,
            )
            pred_norm = lit_model.bjct_to_btjc(pred_bjct)

        # Compute metrics
        pred_disp = mean_frame_disp(pred_norm)
        gt_disp = mean_frame_disp(gt_norm)
        disp_ratio = pred_disp / (gt_disp + 1e-8)
        
        mse = F.mse_loss(pred_norm, gt_norm).item()
        
        inference_results.append({
            'idx': idx,
            'gt_disp': gt_disp,
            'pred_disp': pred_disp,
            'ratio': disp_ratio,
            'mse': mse,
        })
        
        print(f"  Sample {idx}: GT_disp={gt_disp:.4f}, Pred_disp={pred_disp:.4f}, Ratio={disp_ratio:.4f}")

    # ====== SUMMARY ======
    print("\n" + "=" * 70)
    print("INFERENCE RESULTS SUMMARY")
    print("=" * 70)
    
    avg_ratio = np.mean([r['ratio'] for r in inference_results])
    avg_mse = np.mean([r['mse'] for r in inference_results])
    
    print(f"\n  Average Inference Disp Ratio: {avg_ratio:.4f} (ideal = 1.0)")
    print(f"  Average MSE: {avg_mse:.6f}")
    
    # Final judgment
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if 0.8 <= avg_ratio <= 1.2:
        print(f"\n✅ SUCCESS! Disp_loss is working! (ratio={avg_ratio:.4f})")
        print("   → 可以跑 full dataset 训练")
    elif 0.6 <= avg_ratio <= 1.4:
        print(f"\n⚠️  PARTIAL SUCCESS (ratio={avg_ratio:.4f})")
        print("   → 考虑调整 disp_weight 或继续优化")
    else:
        print(f"\n❌ NOT WORKING (ratio={avg_ratio:.4f})")
        print("   → 需要其他方法（如 Temporal Concat）")
    
    # Compare with baseline
    print("\n" + "-" * 70)
    print("对比 (8步 + clip=True, 无 disp_loss):")
    print("  之前 5 样本测试: ratio ≈ 1.25, PCK ≈ 49%")
    print(f"  当前 4 样本 overfit: ratio = {avg_ratio:.4f}")
    print("-" * 70)
    
    print("\n✅ Test complete!")
    return avg_ratio


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    test_4sample_overfit()