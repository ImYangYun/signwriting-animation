"""
Test: Does sign_image actually influence model output?

Method: Fix past_motion, swap sign_image, check if output differs.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Subset
from pose_format.torch.masked.collator import zero_pad_collator

# Import from test script (reuse model definitions)
sys.path.insert(0, "/home/yayun/data/signwriting-animation-fork")
os.chdir("/home/yayun/data/signwriting-animation-fork")

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from test_unfrozen_clip import (
    LitDiffusionUnfrozenCLIP, 
    sanitize_btjc,
)


def test_sign_influence(checkpoint_path: str):
    """Test if sign_image actually influences model output."""
    
    print("=" * 70)
    print("TEST: Does sign_image influence model output?")
    print("=" * 70)
    print(f"  Checkpoint: {checkpoint_path}")
    
    # Load model
    print("\n[1] Loading model...")
    lit_model = LitDiffusionUnfrozenCLIP.load_from_checkpoint(
        checkpoint_path,
        map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    lit_model.eval()
    device = next(lit_model.parameters()).device
    print(f"  Device: {device}")
    
    # Load dataset
    print("\n[2] Loading dataset...")
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    
    full_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    # Select samples from different videos
    seen_poses = set()
    selected_indices = []
    for idx in range(len(full_ds)):
        if len(selected_indices) >= 5:
            break
        record = full_ds.records[idx]
        pose = record.get("pose", "")
        if pose not in seen_poses:
            seen_poses.add(pose)
            selected_indices.append(idx)
    
    print(f"  Selected indices: {selected_indices}")
    
    # Load samples
    samples = []
    for idx in selected_indices:
        batch = zero_pad_collator([full_ds[idx]])
        samples.append({
            "idx": idx,
            "past": sanitize_btjc(batch["conditions"]["input_pose"][:1]).to(device),
            "sign": batch["conditions"]["sign_image"][:1].float().to(device),
            "gt": sanitize_btjc(batch["data"][:1]).to(device),
        })
    
    # ============================================================
    # TEST 1: Same past_motion + Different sign_image
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 1: Same past_motion + Different sign_image")
    print("=" * 70)
    
    # Use sample 0's past_motion
    base_past = samples[0]["past"]
    base_past_norm = lit_model.normalize(base_past)
    base_past_bjct = lit_model.btjc_to_bjct(base_past_norm)
    
    print(f"\n  Using past_motion from sample idx={samples[0]['idx']}")
    print("  Swapping in different sign_images...\n")
    
    predictions = []
    
    for i, s in enumerate(samples):
        sign_img = s["sign"]
        
        # DDPM sampling
        with torch.no_grad():
            B, J, C, _ = base_past_bjct.shape
            T_future = 20
            target_shape = (B, J, C, T_future)
            
            class _Wrapper(nn.Module):
                def __init__(self, model, past, sign):
                    super().__init__()
                    self.model, self.past, self.sign = model, past, sign
                def forward(self, x, t, **kwargs):
                    return self.model(x, t, self.past, self.sign)
            
            wrapped = _Wrapper(lit_model.model, base_past_bjct, sign_img)
            
            # Set seed for reproducibility
            torch.manual_seed(42)
            
            pred_bjct = lit_model.diffusion.p_sample_loop(
                model=wrapped,
                shape=target_shape,
                clip_denoised=True,
                model_kwargs={"y": {}},
                progress=False,
            )
            pred_btjc = lit_model.bjct_to_btjc(pred_bjct)
            pred_unnorm = lit_model.unnormalize(pred_btjc)
        
        predictions.append(pred_unnorm[0].cpu().numpy())
        
        # Compute displacement
        pred_np = pred_unnorm[0].cpu().numpy()
        disp = np.sqrt(np.sum(np.diff(pred_np, axis=0)**2, axis=-1)).mean()
        print(f"  sign from idx={s['idx']:4d}: pred_disp={disp:.6f}")
    
    # Compare predictions pairwise
    print("\n  Pairwise prediction differences (L2 norm):")
    predictions = np.array(predictions)
    
    diffs = []
    for i in range(len(predictions)):
        for j in range(i+1, len(predictions)):
            diff = np.sqrt(np.sum((predictions[i] - predictions[j])**2))
            diffs.append(diff)
            idx_i = samples[i]["idx"]
            idx_j = samples[j]["idx"]
            print(f"    pred(sign={idx_i}) vs pred(sign={idx_j}): diff={diff:.6f}")
    
    avg_diff = np.mean(diffs)
    max_diff = np.max(diffs)
    
    print(f"\n  Average diff: {avg_diff:.6f}")
    print(f"  Max diff: {max_diff:.6f}")
    
    # ============================================================
    # TEST 2: Different past_motion + Same sign_image
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 2: Different past_motion + Same sign_image (baseline)")
    print("=" * 70)
    
    # Use sample 0's sign_image
    base_sign = samples[0]["sign"]
    
    print(f"\n  Using sign_image from sample idx={samples[0]['idx']}")
    print("  Swapping in different past_motions...\n")
    
    predictions2 = []
    
    for i, s in enumerate(samples):
        past_raw = s["past"]
        past_norm = lit_model.normalize(past_raw)
        past_bjct = lit_model.btjc_to_bjct(past_norm)
        
        with torch.no_grad():
            B, J, C, _ = past_bjct.shape
            T_future = 20
            target_shape = (B, J, C, T_future)
            
            wrapped = _Wrapper(lit_model.model, past_bjct, base_sign)
            
            torch.manual_seed(42)
            
            pred_bjct = lit_model.diffusion.p_sample_loop(
                model=wrapped,
                shape=target_shape,
                clip_denoised=True,
                model_kwargs={"y": {}},
                progress=False,
            )
            pred_btjc = lit_model.bjct_to_btjc(pred_bjct)
            pred_unnorm = lit_model.unnormalize(pred_btjc)
        
        predictions2.append(pred_unnorm[0].cpu().numpy())
        
        pred_np = pred_unnorm[0].cpu().numpy()
        disp = np.sqrt(np.sum(np.diff(pred_np, axis=0)**2, axis=-1)).mean()
        print(f"  past from idx={s['idx']:4d}: pred_disp={disp:.6f}")
    
    print("\n  Pairwise prediction differences (L2 norm):")
    predictions2 = np.array(predictions2)
    
    diffs2 = []
    for i in range(len(predictions2)):
        for j in range(i+1, len(predictions2)):
            diff = np.sqrt(np.sum((predictions2[i] - predictions2[j])**2))
            diffs2.append(diff)
            idx_i = samples[i]["idx"]
            idx_j = samples[j]["idx"]
            print(f"    pred(past={idx_i}) vs pred(past={idx_j}): diff={diff:.6f}")
    
    avg_diff2 = np.mean(diffs2)
    max_diff2 = np.max(diffs2)
    
    print(f"\n  Average diff: {avg_diff2:.6f}")
    print(f"  Max diff: {max_diff2:.6f}")
    
    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    print(f"\n  Test 1 (swap sign): avg_diff = {avg_diff:.6f}")
    print(f"  Test 2 (swap past): avg_diff = {avg_diff2:.6f}")
    
    ratio = avg_diff / (avg_diff2 + 1e-8)
    print(f"\n  Ratio (sign_influence / past_influence) = {ratio:.4f}")
    
    if avg_diff < 0.001:
        print("\n  ❌ sign_image has NEGLIGIBLE influence")
        print("     Model is ignoring the sign condition!")
    elif ratio < 0.1:
        print("\n  ⚠️  sign_image has WEAK influence")
        print("     Past motion dominates, sign provides little information.")
    elif ratio < 0.5:
        print("\n  ✓ sign_image has MODERATE influence")
        print("     Both conditions are being used.")
    else:
        print("\n  ✅ sign_image has STRONG influence")
        print("     Model is using sign condition effectively!")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, 
                        default="logs/unfrozen_clip_contrastive_diffvideos_8sample/checkpoints/last.ckpt")
    args = parser.parse_args()
    
    test_sign_influence(args.checkpoint)