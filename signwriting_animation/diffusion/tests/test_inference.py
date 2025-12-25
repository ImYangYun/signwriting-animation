"""
Test inference quality for p=0.5 past dropout checkpoint.
Check if train/inference mismatch causes issues.

Usage:
    python test_inference_p50.py
"""
import os
import torch
import torch.nn as nn
import numpy as np

os.chdir("/home/yayun/data/signwriting-animation-fork")

from test_droupout import (
    LitDiffusionPastDropout, 
    DynamicPosePredictionDataset,
    sanitize_btjc,
    tensor_to_pose,
    Pose,
)
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.utils.generic import reduce_holistic


def inference(lit_model, past, sign, future_len=20):
    """Run DDPM inference."""
    past_norm = lit_model.normalize(past)
    past_bjct = lit_model.btjc_to_bjct(past_norm)
    B, J, C, _ = past_bjct.shape
    
    class Wrapper(nn.Module):
        def __init__(self, m, p, s):
            super().__init__()
            self.m, self.p, self.s = m, p, s
        def forward(self, x, t, **kw):
            return self.m(x, t, self.p, self.s)
    
    wrapped = Wrapper(lit_model.model, past_bjct, sign)
    pred_bjct = lit_model.diffusion.p_sample_loop(
        wrapped, (B, J, C, future_len), clip_denoised=True,
        model_kwargs={'y': {}}, progress=False
    )
    return lit_model.unnormalize(lit_model.bjct_to_btjc(pred_bjct))


def main():
    # Configuration
    ckpt_path = 'logs/past_dropout_overfit_32sample_p50/checkpoints/last.ckpt'
    data_dir = '/home/yayun/data/pose_data/'
    csv_path = '/home/yayun/data/signwriting-animation/data_fixed.csv'
    out_dir = 'logs/past_dropout_overfit_32sample_p50'
    
    print("=" * 60)
    print("INFERENCE TEST: p=0.5 Past Dropout Checkpoint")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading checkpoint: {ckpt_path}")
    lit_model = LitDiffusionPastDropout.load_from_checkpoint(ckpt_path)
    lit_model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lit_model = lit_model.to(device)
    print(f"Device: {device}")
    
    # Load dataset
    print(f"\nLoading dataset...")
    ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split='train',
    )
    print(f"Dataset size: {len(ds)}")
    
    # Test indices (same as sign influence test)
    test_indices = [0, 34, 264, 321, 336]
    
    print(f"\n{'='*60}")
    print("TESTING INFERENCE ON 5 SAMPLES")
    print("=" * 60)
    print(f"{'idx':>5} | {'GT disp':>10} | {'Pred disp':>10} | {'Ratio':>8} | {'PCK@0.1':>8}")
    print("-" * 60)
    
    results = []
    
    for idx in test_indices:
        batch = zero_pad_collator([ds[idx]])
        past = sanitize_btjc(batch['conditions']['input_pose'][:1]).to(device)
        sign = batch['conditions']['sign_image'][:1].float().to(device)
        gt = sanitize_btjc(batch['data'][:1]).to(device)
        
        with torch.no_grad():
            pred = inference(lit_model, past, sign)
        
        gt_unnorm = lit_model.unnormalize(lit_model.normalize(gt))
        
        # Compute metrics
        gt_disp = (gt_unnorm[:, 1:] - gt_unnorm[:, :-1]).abs().mean().item()
        pred_disp = (pred[:, 1:] - pred[:, :-1]).abs().mean().item()
        ratio = pred_disp / (gt_disp + 1e-8)
        
        # MPJPE & PCK
        diff = (pred - gt_unnorm).cpu().numpy()[0]
        per_joint_err = np.sqrt((diff ** 2).sum(-1))
        mpjpe = per_joint_err.mean()
        pck = (per_joint_err < 0.1).mean() * 100
        
        print(f"{idx:5d} | {gt_disp:10.4f} | {pred_disp:10.4f} | {ratio:8.2f} | {pck:7.1f}%")
        results.append({'idx': idx, 'ratio': ratio, 'mpjpe': mpjpe, 'pck': pck, 
                        'gt_disp': gt_disp, 'pred_disp': pred_disp})
    
    # Summary
    avg_ratio = np.mean([r['ratio'] for r in results])
    avg_pck = np.mean([r['pck'] for r in results])
    avg_mpjpe = np.mean([r['mpjpe'] for r in results])
    
    print("-" * 60)
    print(f"{'AVG':>5} | {'-':>10} | {'-':>10} | {avg_ratio:8.2f} | {avg_pck:7.1f}%")
    print(f"\nAverage MPJPE: {avg_mpjpe:.4f}")
    
    # Diagnosis
    print(f"\n{'='*60}")
    print("DIAGNOSIS")
    print("=" * 60)
    
    if 0.8 <= avg_ratio <= 1.2:
        print(f"✅ Disp Ratio ({avg_ratio:.2f}) is GOOD (0.8-1.2)")
    else:
        print(f"⚠️  Disp Ratio ({avg_ratio:.2f}) is outside ideal range (0.8-1.2)")
    
    if avg_pck > 50:
        print(f"✅ PCK@0.1 ({avg_pck:.1f}%) is GOOD (>50%)")
    elif avg_pck > 30:
        print(f"⚠️  PCK@0.1 ({avg_pck:.1f}%) is MODERATE (30-50%)")
    else:
        print(f"❌ PCK@0.1 ({avg_pck:.1f}%) is LOW (<30%)")
    
    # Save pose files for visualization
    print(f"\n{'='*60}")
    print("SAVING POSE FILES")
    print("=" * 60)
    
    # Use idx=0 for pose output
    batch = zero_pad_collator([ds[0]])
    past = sanitize_btjc(batch['conditions']['input_pose'][:1]).to(device)
    sign = batch['conditions']['sign_image'][:1].float().to(device)
    gt = sanitize_btjc(batch['data'][:1]).to(device)
    
    with torch.no_grad():
        pred = inference(lit_model, past, sign)
    gt_unnorm = lit_model.unnormalize(lit_model.normalize(gt))
    
    # Load reference pose for header
    ref_path = ds.records[0]['pose']
    if not ref_path.startswith('/'):
        ref_path = data_dir + ref_path
    with open(ref_path, 'rb') as f:
        ref_pose = Pose.read(f)
    ref_pose = reduce_holistic(ref_pose)
    
    # Convert and save
    gt_pose = tensor_to_pose(gt_unnorm, ref_pose.header, ref_pose)
    pred_pose = tensor_to_pose(pred, ref_pose.header, ref_pose)
    
    os.makedirs(out_dir, exist_ok=True)
    
    gt_path = f'{out_dir}/test_idx0_gt.pose'
    pred_path = f'{out_dir}/test_idx0_pred.pose'
    
    with open(gt_path, 'wb') as f:
        gt_pose.write(f)
    with open(pred_path, 'wb') as f:
        pred_pose.write(f)
    
    print(f"GT saved: {gt_path}")
    print(f"Pred saved: {pred_path}")
    
    # Also save best PCK sample
    best_idx = max(results, key=lambda x: x['pck'])['idx']
    if best_idx != 0:
        batch = zero_pad_collator([ds[best_idx]])
        past = sanitize_btjc(batch['conditions']['input_pose'][:1]).to(device)
        sign = batch['conditions']['sign_image'][:1].float().to(device)
        gt = sanitize_btjc(batch['data'][:1]).to(device)
        
        with torch.no_grad():
            pred = inference(lit_model, past, sign)
        gt_unnorm = lit_model.unnormalize(lit_model.normalize(gt))
        
        ref_path = ds.records[best_idx]['pose']
        if not ref_path.startswith('/'):
            ref_path = data_dir + ref_path
        with open(ref_path, 'rb') as f:
            ref_pose = Pose.read(f)
        ref_pose = reduce_holistic(ref_pose)
        
        gt_pose = tensor_to_pose(gt_unnorm, ref_pose.header, ref_pose)
        pred_pose = tensor_to_pose(pred, ref_pose.header, ref_pose)
        
        gt_path = f'{out_dir}/test_idx{best_idx}_gt.pose'
        pred_path = f'{out_dir}/test_idx{best_idx}_pred.pose'
        
        with open(gt_path, 'wb') as f:
            gt_pose.write(f)
        with open(pred_path, 'wb') as f:
            pred_pose.write(f)
        
        print(f"Best PCK sample (idx={best_idx}):")
        print(f"  GT saved: {gt_path}")
        print(f"  Pred saved: {pred_path}")
    
    print(f"\n{'='*60}")
    print("TEST COMPLETE!")
    print("=" * 60)
    
    # Final recommendation
    print("\nRECOMMENDATION:")
    if avg_ratio >= 0.8 and avg_ratio <= 1.2 and avg_pck > 40:
        print("✅ Results look good! Safe to run full dataset training with p=0.5")
    else:
        print("⚠️  Consider adjusting dropout rate or checking model further")


if __name__ == "__main__":
    main()