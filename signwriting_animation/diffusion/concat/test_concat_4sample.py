"""
4-Sample Overfit Test for Temporal Concat Model
先验证架构是否能 work
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader, Subset

from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.concat.concat_lightning import (
    LitConcatDiffusion, sanitize_btjc, mean_frame_disp
)


def test_concat_4sample():
    """4-sample overfit test for Temporal Concat model."""
    pl.seed_everything(42)

    # ============================================================
    # CONFIGURATION
    # ============================================================
    NUM_SAMPLES = 4
    MAX_EPOCHS = 200
    DIFFUSION_STEPS = 8
    BATCH_SIZE = 4
    
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = "logs/concat_4sample"
    # ============================================================
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 70)
    print("4-SAMPLE OVERFIT TEST (Temporal Concat Model)")
    print("=" * 70)
    print(f"  NUM_SAMPLES: {NUM_SAMPLES}")
    print(f"  MAX_EPOCHS: {MAX_EPOCHS}")
    print(f"  DIFFUSION_STEPS: {DIFFUSION_STEPS}")
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

    # Create Temporal Concat model
    # 注意参数名要和 concat_lightning.py 里的 LitConcatDiffusion 匹配！
    lit_model = LitConcatDiffusion(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-4,
        diffusion_steps=DIFFUSION_STEPS,
        vel_weight=1.0,
        acc_weight=0.5,
        disp_weight=1.0,
        t_past=40,
        t_future=future_len,
        num_latent_dims=256,  # 修正：原来写的 d_model
        num_heads=4,          # 修正：原来写的 nhead
        num_layers=4,         # 修正：原来是 6
    )

    print(f"\nModel parameters: {sum(p.numel() for p in lit_model.parameters()):,}")

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
        enable_checkpointing=False,
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
    print("INFERENCE RESULTS SUMMARY (Temporal Concat)")
    print("=" * 70)
    
    avg_ratio = np.mean([r['ratio'] for r in inference_results])
    avg_mse = np.mean([r['mse'] for r in inference_results])
    
    print(f"\n  Average Inference Disp Ratio: {avg_ratio:.4f} (ideal = 1.0)")
    print(f"  Average MSE: {avg_mse:.6f}")
    
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if 0.8 <= avg_ratio <= 1.2:
        print(f"\n✅ SUCCESS! Temporal Concat is working! (ratio={avg_ratio:.4f})")
        print("   → 可以跑 full dataset 训练")
    elif 0.6 <= avg_ratio <= 1.4:
        print(f"\n⚠️  PARTIAL SUCCESS (ratio={avg_ratio:.4f})")
        print("   → 考虑调整超参数")
    else:
        print(f"\n❌ NOT WORKING (ratio={avg_ratio:.4f})")
        print("   → 需要检查架构")
    
    print("\n" + "-" * 70)
    print("对比:")
    print("  Frame-Independent (无 disp_loss): ratio ≈ 1.53")
    print("  Frame-Independent (+ disp_loss): 待测试")
    print(f"  Temporal Concat (4-sample): ratio = {avg_ratio:.4f}")
    print("-" * 70)
    
    print("\n✅ Test complete!")
    return avg_ratio


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    test_concat_4sample()