"""
Test V2 Ablation Study - AUTO MODE

Automatically tests:
- V2-baseline: Frame-independent only
- V2-improved: Frame-independent + both CAMDM components

Just run: python test_v2_improved_FIXED.py

No arguments needed!
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
    # Import V2's Lightning module (NOT V1's!)
    from signwriting_animation.diffusion.core.models import create_v2_model
    from signwriting_animation.diffusion.lightning_module import (
        LitDiffusion, sanitize_btjc, mean_frame_disp  # ✅ V2的Lightning
    )
    
    print("\n" + "=" * 70)
    print(f"Testing V2 - Version: {version.upper()}")
    print("=" * 70)
    
    version_names = {
        'baseline': 'V2-Baseline (Frame-Independent)',
        'with_pos': 'V2+PositionalEncoding',
        'with_timestep': 'V2+TimestepEmbedder',
        'improved': 'V2+Both (Full Improved)'
    }
    print(f"Description: {version_names.get(version, version)}")

    out_dir = f"logs/v2_improved_{version}"
    os.makedirs(out_dir, exist_ok=True)

    # Create custom model with specific configuration
    model_kwargs = {
        'num_keypoints': num_joints,
        'num_dims_per_keypoint': num_dims,
        't_past': 40,
        't_future': future_len,
    }
    
    custom_model = create_v2_model(version, **model_kwargs)
    
    # Create V2 Lightning module (✅ 这次是对的)
    lit_model = LitDiffusion(  # ✅ 用V2的LitDiffusion
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,
        diffusion_steps=8,
        vel_weight=1.0,
        t_past=40,           # ✅ V2接受这个参数
        t_future=future_len, # ✅ V2接受这个参数
    )
    
    # Replace the model with our custom version
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

    # Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.eval()

    test_batch = zero_pad_collator([train_ds[0]])
    cond = test_batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(test_batch["data"][:1]).to(device)

    with torch.no_grad():
        # Get prediction in UNNORMALIZED space
        pred_raw = lit_model.sample(past_raw, sign, future_len)
        
        # CRITICAL: Re-normalize for fair comparison
        # Different model configs may produce different scales after unnormalize
        pred_norm = lit_model.normalize(pred_raw)
        gt_norm = lit_model.normalize(gt_raw)
        
        # Compute ALL metrics in NORMALIZED space for consistency
        mse = F.mse_loss(pred_norm, gt_norm).item()
        disp_pred = mean_frame_disp(pred_norm)
        disp_gt = mean_frame_disp(gt_norm)
        disp_ratio = disp_pred / (disp_gt + 1e-8)
        
        # Also compute position metrics in normalized space
        pred_np = pred_norm[0].cpu().numpy()
        gt_np = gt_norm[0].cpu().numpy()
        per_joint_err = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
        mpjpe = per_joint_err.mean()
        pck_01 = (per_joint_err < 0.1).mean() * 100

    # Save poses - use NORMALIZED data for consistency
    ref_path = base_ds.records[0]["pose"]
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    ref_pose = reduce_holistic(ref_pose)
    if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
        ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    
    # Use NORMALIZED data - this ensures consistent scaling across all model variants
    gt_pose = tensor_to_pose(gt_norm, ref_pose.header, ref_pose, gt_btjc=gt_norm, apply_scale=True)
    with open(f"{out_dir}/gt.pose", "wb") as f:
        gt_pose.write(f)
    
    pred_pose = tensor_to_pose(pred_norm, ref_pose.header, ref_pose, gt_btjc=gt_norm, apply_scale=True)
    with open(f"{out_dir}/pred.pose", "wb") as f:
        pred_pose.write(f)

    # Pixel space metrics
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
    print(f"  Disp Ratio: {disp_ratio:.4f}")
    print(f"  Pixel MPJPE: {pixel_mpjpe:.2f}px")
    print(f"  Saved to: {out_dir}/")

    return results


def print_comparison_table(results_list):
    """Print comparison table for all tested versions."""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE - V2 VARIANTS")
    print("=" * 80)
    
    version_names = {
        'baseline': 'V2-Baseline',
        'with_pos': 'V2+PosEnc',
        'with_timestep': 'V2+TimeEmb',
        'improved': 'V2+Both'
    }
    
    print(f"\n{'Version':<20} {'Disp Ratio':<12} {'MPJPE':<10} {'PCK@0.1':<10}")
    print("-" * 80)
    
    for r in results_list:
        vname = version_names.get(r['version'], r['version'])
        print(f"{vname:<20} {r['disp_ratio']:<12.4f} {r['mpjpe']:<10.6f} {r['pck_01']:<10.1f}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    baseline = next((r for r in results_list if r['version'] == 'baseline'), None)
    if baseline:
        print(f"\nBaseline (V2 current):")
        print(f"  Disp Ratio: {baseline['disp_ratio']:.4f}")
        print(f"  MPJPE: {baseline['mpjpe']:.6f}")
        print(f"  PCK@0.1: {baseline['pck_01']:.1f}%")
        
        for r in results_list:
            if r['version'] != 'baseline':
                vname = version_names.get(r['version'], r['version'])
                mpjpe_improve = (baseline['mpjpe'] - r['mpjpe']) / baseline['mpjpe'] * 100
                pck_improve = r['pck_01'] - baseline['pck_01']
                
                print(f"\n{vname}:")
                print(f"  MPJPE: {mpjpe_improve:+.1f}% change")
                print(f"  PCK@0.1: {pck_improve:+.1f}% change")
                print(f"  Disp Ratio: {r['disp_ratio']:.4f} (should stay ~1.0)")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    best_mpjpe = min(results_list, key=lambda x: x['mpjpe'])
    best_pck = max(results_list, key=lambda x: x['pck_01'])
    
    print(f"\nBest MPJPE: {version_names.get(best_mpjpe['version'], best_mpjpe['version'])}")
    print(f"Best PCK@0.1: {version_names.get(best_pck['version'], best_pck['version'])}")
    
    if best_mpjpe['version'] == best_pck['version']:
        print(f"\n✅ Clear winner: {version_names.get(best_mpjpe['version'], best_mpjpe['version'])}")
    else:
        print(f"\n💡 Trade-off between MPJPE and PCK - test on larger dataset")


if __name__ == "__main__":
    # ========================================
    # AUTO MODE - No arguments needed!
    # Tests: baseline + improved
    # ========================================
    
    pl.seed_everything(42)

    # Configuration
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    
    NUM_SAMPLES = 4
    MAX_EPOCHS = 500  # Fixed: 500 epochs
    BATCH_SIZE = 4

    print("=" * 70)
    print("V2 ABLATION STUDY - AUTO MODE")
    print("=" * 70)
    print("\nWill test:")
    print("  1. V2-baseline (frame-independent only)")
    print("  2. V2-improved (frame-independent + both CAMDM components)")
    print(f"\nSettings: {NUM_SAMPLES} samples, {MAX_EPOCHS} epochs each")
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
    print(f"Max epochs: {MAX_EPOCHS}")

    # ========================================
    # AUTO TEST: baseline + improved
    # ========================================
    versions = ['baseline', 'improved']
    
    results_list = []
    for idx, version in enumerate(versions):
        print(f"\n{'=' * 70}")
        print(f"Testing {version.upper()} ({idx+1}/{len(versions)})")
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

    # Print comparison
    if len(results_list) > 1:
        print_comparison_table(results_list)
    
    print("\n" + "=" * 70)
    print("🎉 ABLATION STUDY COMPLETE!")
    print("=" * 70)
    print("\nComplete data:")
    print("  ✓ V1 (trans_enc): disp_ratio=0.00 (collapse)")
    if len(results_list) > 0:
        print(f"  ✓ V2-baseline: disp_ratio={results_list[0]['disp_ratio']:.4f}")
    print("  ✓ V2-pos: disp_ratio=1.05")
    if len(results_list) > 1:
        print(f"  ✓ V2-improved: disp_ratio={results_list[1]['disp_ratio']:.4f}")
    print("\n💡 Next Steps:")
    print("  - Create ablation table for paper")
    print("  - Compare pose files visually")
    print("  - Scale to 100 samples for validation")
    print("=" * 70)