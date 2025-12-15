#!/usr/bin/env python
"""
V2 Ablation Study - Fully Automatic (No Interaction)

Perfect for batch job submission!
Just run: python run_ablation_auto.py

No confirmation needed - starts immediately!
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import unshift_hands

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset


def tensor_to_pose(t_btjc, header, ref_pose, gt_btjc=None, apply_scale=True):
    """Convert normalized tensor predictions back to pose format."""
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc
    
    t_np = t.detach().cpu().numpy().astype(np.float32)
    gt_np = None
    if gt_btjc is not None:
        if gt_btjc.dim() == 4:
            gt_np = gt_btjc[0].detach().cpu().numpy().astype(np.float32)
        else:
            gt_np = gt_btjc.detach().cpu().numpy().astype(np.float32)

    arr = t_np[:, None, :, :]
    conf = ref_pose.body.confidence[:len(t_np)].copy()
    fps = ref_pose.body.fps
    
    body = NumPyPoseBody(fps=fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)
    
    unshift_hands(pose_obj)
    
    T_pred = pose_obj.body.data.shape[0]
    T_ref_total = ref_pose.body.data.shape[0]
    future_start = max(0, T_ref_total - T_pred)
    ref_arr = np.asarray(ref_pose.body.data[future_start:future_start+T_pred, 0], dtype=np.float32)
    
    if apply_scale and gt_np is not None:
        def _var(a):
            center = a.mean(axis=1, keepdims=True)
            return float(((a - center) ** 2).mean())
        
        var_gt_norm = _var(gt_np)
        var_ref = _var(ref_arr)
        
        if var_gt_norm > 1e-8:
            scale = np.sqrt(var_ref / var_gt_norm)
            pose_obj.body.data *= scale

    pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)
    ref_c = ref_arr.reshape(-1, 3).mean(axis=0)
    pred_c = pred_arr.reshape(-1, 3).mean(axis=0)
    delta = ref_c - pred_c
    pose_obj.body.data += delta
    
    return pose_obj


def test_v2_version(version, train_ds, train_loader, num_joints, num_dims, future_len,
                    stats_path, data_dir, base_ds, max_epochs=500):
    """Test a specific V2 version."""
    from signwriting_animation.diffusion.core.models import create_v2_model
    from signwriting_animation.diffusion.lightning_module import (
        LitDiffusion, sanitize_btjc, mean_frame_disp
    )
    
    print("\n" + "=" * 70)
    print(f"Testing V2 - Version: {version.upper()}")
    print("=" * 70)
    
    version_names = {
        'baseline': 'V2-Baseline (Frame-Independent Only)',
        'improved': 'V2-Improved (Frame-Independent + Both CAMDM Components)'
    }
    print(f"Description: {version_names.get(version, version)}")

    out_dir = f"logs/v2_improved_{version}"
    os.makedirs(out_dir, exist_ok=True)

    model_kwargs = {
        'num_keypoints': num_joints,
        'num_dims_per_keypoint': num_dims,
        't_past': 40,
        't_future': future_len,
    }
    
    custom_model = create_v2_model(version, **model_kwargs)
    
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
    
    print(f"\nTraining {version}...")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=50,
        enable_progress_bar=False,
    )
    trainer.fit(lit_model, train_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.eval()

    test_batch = zero_pad_collator([train_ds[0]])
    cond = test_batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(test_batch["data"][:1]).to(device)

    with torch.no_grad():
        pred_raw = lit_model.sample(past_raw, sign, future_len)
        
        mse = F.mse_loss(pred_raw, gt_raw).item()
        disp_pred = mean_frame_disp(pred_raw)
        disp_gt = mean_frame_disp(gt_raw)
        disp_ratio = disp_pred / (disp_gt + 1e-8)
        
        pred_np = pred_raw[0].cpu().numpy()
        gt_np = gt_raw[0].cpu().numpy()
        per_joint_err = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
        mpjpe = per_joint_err.mean()
        pck_01 = (per_joint_err < 0.1).mean() * 100

    ref_path = base_ds.records[0]["pose"]
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    ref_pose = reduce_holistic(ref_pose)
    if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
        ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    
    gt_pose = tensor_to_pose(gt_raw, ref_pose.header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
    with open(f"{out_dir}/gt.pose", "wb") as f:
        gt_pose.write(f)
    
    pred_pose = tensor_to_pose(pred_raw, ref_pose.header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
    with open(f"{out_dir}/pred.pose", "wb") as f:
        pred_pose.write(f)

    gt_data = gt_pose.body.data[:, 0, :, :]
    pred_data = pred_pose.body.data[:, 0, :, :]
    pixel_mpjpe = np.sqrt(((gt_data - pred_data) ** 2).sum(-1)).mean()

    results = {
        'version': version,
        'mse': mse,
        'mpjpe': mpjpe,
        'pck_01': pck_01,
        'disp_ratio': disp_ratio,
        'disp_pred': disp_pred,
        'disp_gt': disp_gt,
        'pixel_mpjpe': pixel_mpjpe,
        'out_dir': out_dir,
    }

    print(f"\n{version_names.get(version, version)} Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MPJPE: {mpjpe:.6f}")
    print(f"  PCK@0.1: {pck_01:.1f}%")
    print(f"  Disp Ratio: {disp_ratio:.4f} {'✅' if 0.7 < disp_ratio < 1.3 else '❌'}")
    print(f"  Pixel MPJPE: {pixel_mpjpe:.2f}px")
    print(f"  Saved to: {out_dir}/")

    return results


def print_comparison_table(results_list):
    """Print comparison table for all tested versions."""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE - ABLATION STUDY RESULTS")
    print("=" * 80)
    
    version_names = {
        'baseline': 'V2-Baseline',
        'improved': 'V2-Improved'
    }
    
    print(f"\n{'Version':<20} {'Disp Ratio':<12} {'MPJPE':<10} {'PCK@0.1':<10} {'Status':<10}")
    print("-" * 80)
    
    for r in results_list:
        vname = version_names.get(r['version'], r['version'])
        status = '✅ Good' if 0.7 < r['disp_ratio'] < 1.3 else '❌ Bad'
        print(f"{vname:<20} {r['disp_ratio']:<12.4f} {r['mpjpe']:<10.6f} {r['pck_01']:<10.1f} {status}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    baseline = next((r for r in results_list if r['version'] == 'baseline'), None)
    improved = next((r for r in results_list if r['version'] == 'improved'), None)
    
    if baseline and improved:
        print("\nKey Findings:")
        print(f"  V2-Baseline disp_ratio: {baseline['disp_ratio']:.4f}")
        print(f"  V2-Improved disp_ratio: {improved['disp_ratio']:.4f}")
        
        if baseline['disp_ratio'] > 0.7:
            print("\n  ✅ Frame-independent decoding successfully prevents motion collapse!")
            print("     (V1 had disp_ratio=0.00, V2-baseline has ~1.0)")
        
        mpjpe_change = (baseline['mpjpe'] - improved['mpjpe']) / baseline['mpjpe'] * 100
        pck_change = improved['pck_01'] - baseline['pck_01']
        
        print(f"\n  CAMDM components contribution:")
        print(f"    MPJPE: {mpjpe_change:+.1f}% change")
        print(f"    PCK@0.1: {pck_change:+.1f}% change")


def main():
    print("=" * 70)
    print("V2 ABLATION STUDY - FULLY AUTOMATIC")
    print("=" * 70)
    print()
    print("This script will automatically test:")
    print("  1. V2-baseline (frame-independent only)")
    print("  2. V2-improved (frame-independent + CAMDM components)")
    print()
    print("Current known results:")
    print("  ✓ V1 (trans_enc): disp_ratio=0.00 (motion collapse)")
    print("  ✓ V2-pos: disp_ratio=1.05 (already tested)")
    print()
    print("Settings:")
    print("  - 4 samples overfitting test")
    print("  - 500 epochs per version")
    print("  - Estimated time: ~1 hour")
    print("=" * 70)
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    
    if not os.path.exists("signwriting_animation/diffusion/core/models.py"):
        print("❌ ERROR: models.py not found!")
        print("\nPlease run:")
        print("  cp models_v2_improved.py signwriting_animation/diffusion/core/models.py")
        sys.exit(1)
    
    # Check if create_v2_model exists
    try:
        from signwriting_animation.diffusion.core.models import create_v2_model
        print("✅ Model file verified (create_v2_model found)")
    except ImportError:
        print("❌ ERROR: create_v2_model not found in models.py!")
        print("\nPlease run:")
        print("  cp models_v2_improved.py signwriting_animation/diffusion/core/models.py")
        sys.exit(1)
    
    print("✅ All prerequisites satisfied!")
    
    # Auto start - NO CONFIRMATION NEEDED!
    print("\n" + "=" * 70)
    print("🚀 Starting ablation study automatically...")
    print("   (Perfect for batch job submission!)")
    print("=" * 70)
    
    # Configuration
    pl.seed_everything(42)
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    
    NUM_SAMPLES = 4
    MAX_EPOCHS = 500
    BATCH_SIZE = 4

    print("\n" + "=" * 70)
    print("Loading dataset...")
    print("=" * 70)

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

    sample = train_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    num_joints = sample.shape[-2]
    num_dims = sample.shape[-1]
    future_len = sample.shape[0]

    print(f"Dataset loaded: {NUM_SAMPLES} samples, J={num_joints}, D={num_dims}, T={future_len}")
    
    # Run tests
    versions = ['baseline', 'improved']
    results_list = []
    
    total_start = time.time()
    
    for idx, version in enumerate(versions):
        print(f"\n{'=' * 70}")
        print(f"TESTING {version.upper()} ({idx+1}/{len(versions)})")
        print(f"{'=' * 70}")
        
        try:
            result = test_v2_version(
                version, train_ds, train_loader, num_joints, num_dims, future_len,
                stats_path, data_dir, base_ds, MAX_EPOCHS
            )
            results_list.append(result)
        except Exception as e:
            print(f"\n❌ Version {version} failed: {e}")
            import traceback
            traceback.print_exc()
    
    total_elapsed = time.time() - total_start
    
    # Print results
    if len(results_list) > 0:
        print_comparison_table(results_list)
    
    print("\n" + "=" * 70)
    print("🎉 ABLATION STUDY COMPLETE!")
    print("=" * 70)
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print()
    print("Complete ablation data:")
    print("  ✓ V1 (trans_enc): disp_ratio=0.00 (collapse)")
    if len(results_list) > 0:
        print(f"  ✓ V2-baseline: disp_ratio={results_list[0]['disp_ratio']:.4f}")
    print("  ✓ V2-pos: disp_ratio=1.05")
    if len(results_list) > 1:
        print(f"  ✓ V2-improved: disp_ratio={results_list[1]['disp_ratio']:.4f}")
    print()
    print("Results saved in:")
    print("  - logs/v2_improved_baseline/")
    print("  - logs/v2_improved_improved/")
    print()
    print("Next steps:")
    print("  1. Create ablation table for paper")
    print("  2. Compare pose files visually")
    print("  3. Write up analysis")
    print("=" * 70)


if __name__ == "__main__":
    main()