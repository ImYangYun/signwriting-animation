"""
Inference Test - Fixed version (no command line args)

Tests both p=0.3 and p=0.5 checkpoints for comparison.

Usage:
    python test_inference_fixed.py
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
)
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import unshift_hands


# ============================================================
# Configuration - EDIT HERE
# ============================================================
CHECKPOINTS = [
    ("p30", "logs/past_dropout_overfit_32sample_p30/checkpoints/last.ckpt"),
    ("p50", "logs/past_dropout_overfit_32sample_p50/checkpoints/last.ckpt"),
]

DATA_DIR = '/home/yayun/data/pose_data/'
CSV_PATH = '/home/yayun/data/signwriting-animation/data_fixed.csv'
# ============================================================


def tensor_to_pose(t_btjc: torch.Tensor, 
                   header, 
                   ref_pose: Pose, 
                   scale_to_ref: bool = True) -> Pose:
    """Convert tensor prediction to Pose format for visualization.
    
    CORRECT VERSION - from original train_full.py
    """
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc
    
    t_np = t.detach().cpu().numpy().astype(np.float32)
    
    # Create pose object
    arr = t_np[:, None, :, :]
    T = arr.shape[0]
    conf = np.ones((T, 1, arr.shape[2]), dtype=np.float32)
    fps = ref_pose.body.fps
    
    body = NumPyPoseBody(fps=fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)
    
    # Fix hand positions (must be before scaling!)
    unshift_hands(pose_obj)
    
    if scale_to_ref:
        # Get reference data
        T_pred = t_np.shape[0]
        T_ref_total = ref_pose.body.data.shape[0]
        future_start = max(0, T_ref_total - T_pred)
        ref_arr = np.asarray(
            ref_pose.body.data[future_start:future_start+T_pred, 0], 
            dtype=np.float32
        )
        
        # Scale to reference variance
        def _var(a):
            center = a.mean(axis=(0, 1), keepdims=True)
            return float(((a - center) ** 2).mean())
        
        pose_data = pose_obj.body.data[:, 0, :, :]
        var_input = _var(pose_data)
        var_ref = _var(ref_arr)
        
        if var_input > 1e-8:
            scale = np.sqrt(var_ref / var_input)
            pose_obj.body.data = pose_obj.body.data * scale
        
        # Align center (CORRECT VERSION)
        pose_data = pose_obj.body.data[:, 0, :, :].reshape(-1, 3)
        input_center = pose_data.mean(axis=0)
        ref_center = ref_arr.reshape(-1, 3).mean(axis=0)
        pose_obj.body.data = pose_obj.body.data + (ref_center - input_center)
    
    return pose_obj


def inference(lit_model, past, sign, future_len=20):
    """Run DDPM inference - returns UNNORMALIZED prediction."""
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
    pred_norm = lit_model.bjct_to_btjc(pred_bjct)
    return lit_model.unnormalize(pred_norm)  # Return UNNORMALIZED


def compute_metrics(pred, gt):
    """Compute metrics on UNNORMALIZED data."""
    # Displacement ratio
    gt_disp = (gt[:, 1:] - gt[:, :-1]).abs().mean().item()
    pred_disp = (pred[:, 1:] - pred[:, :-1]).abs().mean().item()
    ratio = pred_disp / (gt_disp + 1e-8)
    
    # MPJPE & PCK (per-frame, per-joint error)
    diff = (pred - gt).cpu().numpy()[0]  # [T, J, 3]
    per_joint_err = np.sqrt((diff ** 2).sum(-1))  # [T, J]
    mpjpe = per_joint_err.mean()
    pck_01 = (per_joint_err < 0.1).mean() * 100
    
    # Jitter (acceleration)
    gt_vel = gt[:, 1:] - gt[:, :-1]
    pred_vel = pred[:, 1:] - pred[:, :-1]
    if gt_vel.shape[1] > 1:
        gt_acc = (gt_vel[:, 1:] - gt_vel[:, :-1]).abs().mean().item()
        pred_acc = (pred_vel[:, 1:] - pred_vel[:, :-1]).abs().mean().item()
        jitter_ratio = pred_acc / (gt_acc + 1e-8)
    else:
        jitter_ratio = 1.0
    
    return {
        'ratio': ratio,
        'mpjpe': mpjpe,
        'pck': pck_01,
        'jitter': jitter_ratio,
        'gt_disp': gt_disp,
        'pred_disp': pred_disp,
    }


def test_checkpoint(name, ckpt_path, ds, train_indices, device):
    """Test a single checkpoint."""
    print(f"\n{'='*70}")
    print(f"TESTING: {name}")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 70)
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return None
    
    # Load model
    lit_model = LitDiffusionPastDropout.load_from_checkpoint(ckpt_path)
    lit_model.eval()
    lit_model = lit_model.to(device)
    
    print(f"{'idx':>5} | {'GT disp':>10} | {'Pred disp':>10} | {'Ratio':>8} | {'PCK@0.1':>8} | {'Jitter':>8}")
    print("-" * 70)
    
    results = []
    for idx in train_indices[:5]:
        batch = zero_pad_collator([ds[idx]])
        past = sanitize_btjc(batch['conditions']['input_pose'][:1]).to(device)
        sign = batch['conditions']['sign_image'][:1].float().to(device)
        gt = sanitize_btjc(batch['data'][:1]).to(device)  # Already unnormalized!
        
        with torch.no_grad():
            pred = inference(lit_model, past, sign)
        
        metrics = compute_metrics(pred, gt)
        
        print(f"{idx:5d} | {metrics['gt_disp']:10.4f} | {metrics['pred_disp']:10.4f} | "
              f"{metrics['ratio']:8.2f} | {metrics['pck']:7.1f}% | {metrics['jitter']:8.2f}")
        results.append({'idx': idx, **metrics})
    
    # Summary
    avg_ratio = np.mean([r['ratio'] for r in results])
    avg_pck = np.mean([r['pck'] for r in results])
    avg_jitter = np.mean([r['jitter'] for r in results])
    avg_mpjpe = np.mean([r['mpjpe'] for r in results])
    
    print("-" * 70)
    print(f"{'AVG':>5} | {'-':>10} | {'-':>10} | {avg_ratio:8.2f} | {avg_pck:7.1f}% | {avg_jitter:8.2f}")
    print(f"Average MPJPE: {avg_mpjpe:.4f}")
    
    # Save best pose
    out_dir = os.path.dirname(os.path.dirname(ckpt_path))
    best = max(results, key=lambda x: x['pck'])
    best_idx = best['idx']
    
    batch = zero_pad_collator([ds[best_idx]])
    past = sanitize_btjc(batch['conditions']['input_pose'][:1]).to(device)
    sign = batch['conditions']['sign_image'][:1].float().to(device)
    gt = sanitize_btjc(batch['data'][:1]).to(device)
    
    with torch.no_grad():
        pred = inference(lit_model, past, sign)
    
    # Load reference pose
    ref_path = ds.records[best_idx]['pose']
    if not ref_path.startswith('/'): ref_path = DATA_DIR + ref_path
    with open(ref_path, 'rb') as f:
        ref_pose = Pose.read(f)
    ref_pose = reduce_holistic(ref_pose)
    if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
        ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    
    gt_pose = tensor_to_pose(gt, ref_pose.header, ref_pose)
    pred_pose = tensor_to_pose(pred, ref_pose.header, ref_pose)
    
    gt_path = f'{out_dir}/test_idx{best_idx}_gt_fixed.pose'
    pred_path = f'{out_dir}/test_idx{best_idx}_pred_fixed.pose'
    
    with open(gt_path, 'wb') as f: gt_pose.write(f)
    with open(pred_path, 'wb') as f: pred_pose.write(f)
    
    print(f"\nPose saved (best idx={best_idx}, PCK={best['pck']:.1f}%):")
    print(f"  GT: {gt_path}")
    print(f"  Pred: {pred_path}")
    
    return {
        'name': name,
        'avg_ratio': avg_ratio,
        'avg_pck': avg_pck,
        'avg_jitter': avg_jitter,
        'avg_mpjpe': avg_mpjpe,
    }


def main():
    print("=" * 70)
    print("INFERENCE TEST (Fixed Version)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    ds = DynamicPosePredictionDataset(
        data_dir=DATA_DIR, csv_path=CSV_PATH,
        num_past_frames=40, num_future_frames=20,
        with_metadata=True, split='train',
    )
    print(f"Dataset size: {len(ds)}")
    
    # Get training sample indices (same logic as train_overfit)
    seen_poses, train_indices = set(), []
    for idx in range(len(ds)):
        if len(train_indices) >= 32: break
        pose = ds.records[idx].get("pose", "")
        if pose not in seen_poses:
            seen_poses.add(pose)
            train_indices.append(idx)
    
    print(f"Training indices: {train_indices[:10]}...")
    
    # Test all checkpoints
    all_results = []
    for name, ckpt_path in CHECKPOINTS:
        result = test_checkpoint(name, ckpt_path, ds, train_indices, device)
        if result:
            all_results.append(result)
    
    # Final comparison
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Model':<10} | {'Ratio':>8} | {'PCK@0.1':>8} | {'Jitter':>8} | {'MPJPE':>8}")
        print("-" * 70)
        for r in all_results:
            print(f"{r['name']:<10} | {r['avg_ratio']:8.2f} | {r['avg_pck']:7.1f}% | {r['avg_jitter']:8.2f} | {r['avg_mpjpe']:8.4f}")
        
        # Winner
        best = max(all_results, key=lambda x: x['avg_pck'])
        print("-" * 70)
        print(f"Best by PCK: {best['name']} ({best['avg_pck']:.1f}%)")
        
        best_ratio = min(all_results, key=lambda x: abs(x['avg_ratio'] - 1.0))
        print(f"Best by Ratio: {best_ratio['name']} ({best_ratio['avg_ratio']:.2f})")
    
    print(f"\n{'='*70}")
    print("TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()