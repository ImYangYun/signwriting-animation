"""
Classifier-Free Guidance (CFG) Evaluation - Fixed Version

Uses a wrapper model to inject CFG into the existing diffusion p_sample_loop,
instead of reimplementing the diffusion loop.

Usage:
    python eval_cfg_fixed.py --checkpoint logs/full_unfrozen_clip/checkpoints/last.ckpt
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.data import DataLoader
from pose_format.torch.masked.collator import zero_pad_collator

# Import from training script
sys.path.insert(0, os.getcwd())
from train_unfrozen_clip_full import (
    LitDiffusionUnfrozenCLIP,
    DynamicPosePredictionDataset,
    sanitize_btjc,
    mean_frame_disp,
)


class CFGModelWrapper(nn.Module):
    """
    Wrapper that applies CFG at each denoising step.
    
    This wraps the original model and applies CFG logic in forward(),
    so we can use the existing p_sample_loop without modification.
    """
    
    def __init__(self, model, past_cond, past_uncond, sign_cond, sign_uncond, 
                 guidance_scale=1.0, mode='sign'):
        """
        Args:
            model: The original diffusion model
            past_cond: Conditional past motion [B, J, C, T]
            past_uncond: Unconditional past motion (zeros) [B, J, C, T]
            sign_cond: Conditional sign embedding [B, ...]
            sign_uncond: Unconditional sign embedding (zeros) [B, ...]
            guidance_scale: CFG scale (1.0 = no guidance)
            mode: 'sign' to amplify sign, 'past' to amplify past
        """
        super().__init__()
        self.model = model
        self.past_cond = past_cond
        self.past_uncond = past_uncond
        self.sign_cond = sign_cond
        self.sign_uncond = sign_uncond
        self.guidance_scale = guidance_scale
        self.mode = mode
    
    def forward(self, x, t, **kwargs):
        """
        CFG forward pass.
        
        pred = pred_uncond + s * (pred_cond - pred_uncond)
        """
        if self.guidance_scale == 1.0:
            # No guidance, just normal forward
            return self.model(x, t, self.past_cond, self.sign_cond)
        
        if self.mode == 'sign':
            # Amplify sign's influence
            # cond: with sign, uncond: without sign
            pred_cond = self.model(x, t, self.past_cond, self.sign_cond)
            pred_uncond = self.model(x, t, self.past_cond, self.sign_uncond)
        elif self.mode == 'past':
            # Amplify past's influence
            # cond: with past, uncond: without past
            pred_cond = self.model(x, t, self.past_cond, self.sign_cond)
            pred_uncond = self.model(x, t, self.past_uncond, self.sign_cond)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # CFG formula
        pred = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)
        return pred


class NormalModelWrapper(nn.Module):
    """Simple wrapper for normal inference."""
    def __init__(self, model, past, sign):
        super().__init__()
        self.model, self.past, self.sign = model, past, sign
    
    def forward(self, x, t, **kwargs):
        return self.model(x, t, self.past, self.sign)


def compute_metrics(pred_norm, gt_norm):
    """Compute evaluation metrics."""
    pred_np = pred_norm[0].cpu().numpy()
    gt_np = gt_norm[0].cpu().numpy()
    
    # Per-joint error
    per_joint_err = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
    mpjpe = per_joint_err.mean()
    pck_01 = (per_joint_err < 0.1).mean() * 100
    pck_005 = (per_joint_err < 0.05).mean() * 100
    
    # Displacement
    pred_disp = mean_frame_disp(pred_norm)
    gt_disp = mean_frame_disp(gt_norm)
    disp_ratio = pred_disp / (gt_disp + 1e-8)
    
    # MSE
    mse = F.mse_loss(pred_norm, gt_norm).item()
    
    return {
        'mpjpe': mpjpe,
        'pck_01': pck_01,
        'pck_005': pck_005,
        'disp_ratio': disp_ratio,
        'gt_disp': gt_disp,
        'pred_disp': pred_disp,
        'mse': mse,
    }


@torch.no_grad()
def evaluate_sample(lit_model, batch, device, guidance_scales=[1.5, 2.0, 3.0]):
    """Evaluate one sample with different methods and guidance scales."""
    
    cond = batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(batch["data"][:1]).to(device)
    
    # Normalize
    past_norm = lit_model.normalize(past_raw)
    gt_norm = lit_model.normalize(gt_raw)
    
    past_bjct = lit_model.btjc_to_bjct(past_norm)
    B, J, C, _ = past_bjct.shape
    T_future = gt_norm.shape[1]
    target_shape = (B, J, C, T_future)
    
    # Create unconditional inputs
    zeros_past = torch.zeros_like(past_bjct)
    zeros_sign = torch.zeros_like(sign)
    
    results = {}
    
    # 1. Normal inference (past + sign)
    wrapped = NormalModelWrapper(lit_model.model, past_bjct, sign)
    pred_bjct = lit_model.diffusion.p_sample_loop(
        model=wrapped,
        shape=target_shape,
        clip_denoised=True,
        model_kwargs={"y": {}},
        progress=False,
    )
    pred_norm = lit_model.bjct_to_btjc(pred_bjct)
    results['normal'] = compute_metrics(pred_norm, gt_norm)
    
    # 2. Sign-Only inference (no past)
    wrapped = NormalModelWrapper(lit_model.model, zeros_past, sign)
    pred_bjct = lit_model.diffusion.p_sample_loop(
        model=wrapped,
        shape=target_shape,
        clip_denoised=True,
        model_kwargs={"y": {}},
        progress=False,
    )
    pred_norm = lit_model.bjct_to_btjc(pred_bjct)
    results['sign_only'] = compute_metrics(pred_norm, gt_norm)
    
    # 3. CFG amplifying Sign (with past, amplify sign's effect)
    for scale in guidance_scales:
        wrapped = CFGModelWrapper(
            model=lit_model.model,
            past_cond=past_bjct,
            past_uncond=past_bjct,  # Keep past the same
            sign_cond=sign,
            sign_uncond=zeros_sign,
            guidance_scale=scale,
            mode='sign'
        )
        pred_bjct = lit_model.diffusion.p_sample_loop(
            model=wrapped,
            shape=target_shape,
            clip_denoised=True,
            model_kwargs={"y": {}},
            progress=False,
        )
        pred_norm = lit_model.bjct_to_btjc(pred_bjct)
        results[f'cfg_sign_{scale}'] = compute_metrics(pred_norm, gt_norm)
    
    # 4. CFG amplifying Past (with sign, amplify past's effect)
    for scale in guidance_scales:
        wrapped = CFGModelWrapper(
            model=lit_model.model,
            past_cond=past_bjct,
            past_uncond=zeros_past,
            sign_cond=sign,
            sign_uncond=sign,  # Keep sign the same
            guidance_scale=scale,
            mode='past'
        )
        pred_bjct = lit_model.diffusion.p_sample_loop(
            model=wrapped,
            shape=target_shape,
            clip_denoised=True,
            model_kwargs={"y": {}},
            progress=False,
        )
        pred_norm = lit_model.bjct_to_btjc(pred_bjct)
        results[f'cfg_past_{scale}'] = compute_metrics(pred_norm, gt_norm)
    
    # 5. Sign-Only + CFG Sign (no past, but amplify sign)
    # This tests if CFG can help Sign-Only inference
    for scale in guidance_scales:
        wrapped = CFGModelWrapper(
            model=lit_model.model,
            past_cond=zeros_past,  # No past
            past_uncond=zeros_past,
            sign_cond=sign,
            sign_uncond=zeros_sign,
            guidance_scale=scale,
            mode='sign'
        )
        pred_bjct = lit_model.diffusion.p_sample_loop(
            model=wrapped,
            shape=target_shape,
            clip_denoised=True,
            model_kwargs={"y": {}},
            progress=False,
        )
        pred_norm = lit_model.bjct_to_btjc(pred_bjct)
        results[f'signonly_cfg_{scale}'] = compute_metrics(pred_norm, gt_norm)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                        default='logs/full_unfrozen_clip/checkpoints/last.ckpt',
                        help='Path to checkpoint')
    parser.add_argument('--data_dir', type=str, 
                        default='/home/yayun/data/pose_data/',
                        help='Data directory')
    parser.add_argument('--csv_path', type=str,
                        default='/home/yayun/data/signwriting-animation/data_fixed.csv',
                        help='CSV path')
    parser.add_argument('--n_samples', type=int, default=20,
                        help='Number of samples to evaluate')
    parser.add_argument('--output', type=str, default='cfg_results_fixed.csv',
                        help='Output CSV path')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    lit_model = LitDiffusionUnfrozenCLIP.load_from_checkpoint(
        args.checkpoint,
        map_location=device
    )
    lit_model = lit_model.to(device)
    lit_model.eval()
    
    # Load dataset
    print("Loading dataset...")
    test_ds = DynamicPosePredictionDataset(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="test",
    )
    print(f"Dataset size: {len(test_ds)}")
    
    # Sample indices
    np.random.seed(42)
    sample_indices = np.random.choice(len(test_ds), min(args.n_samples, len(test_ds)), replace=False)
    sample_indices = sorted(sample_indices)
    
    # Evaluate
    all_results = []
    guidance_scales = [1.5, 2.0, 3.0]
    
    print(f"\nEvaluating {len(sample_indices)} samples...")
    print("="*70)
    
    for idx in tqdm(sample_indices):
        batch = zero_pad_collator([test_ds[idx]])
        results = evaluate_sample(lit_model, batch, device, guidance_scales)
        
        for method, metrics in results.items():
            row = {'idx': idx, 'method': method}
            row.update(metrics)
            all_results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Summary statistics
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    summary = df.groupby('method').agg({
        'pck_01': ['mean', 'std'],
        'pck_005': 'mean',
        'disp_ratio': ['mean', 'std'],
        'mpjpe': 'mean',
    }).round(2)
    
    print(summary)
    
    # Detailed comparison
    print("\n" + "-"*70)
    print("Key Comparisons:")
    print("-"*70)
    
    methods_order = ['normal', 'sign_only'] + \
                    [f'cfg_sign_{s}' for s in guidance_scales] + \
                    [f'cfg_past_{s}' for s in guidance_scales] + \
                    [f'signonly_cfg_{s}' for s in guidance_scales]
    
    for method in methods_order:
        if method in df['method'].values:
            m = df[df['method'] == method]
            print(f"{method:20s}: PCK@0.1 = {m['pck_01'].mean():5.1f}% ± {m['pck_01'].std():4.1f}, "
                  f"ratio = {m['disp_ratio'].mean():.2f}")
    
    # Gap analysis
    print("\n" + "-"*70)
    print("Gap Analysis (vs Normal):")
    print("-"*70)
    
    normal_pck = df[df['method'] == 'normal']['pck_01'].mean()
    sign_only_pck = df[df['method'] == 'sign_only']['pck_01'].mean()
    
    print(f"{'Baseline:':<20s}")
    print(f"  normal:            PCK = {normal_pck:.1f}%")
    print(f"  sign_only:         PCK = {sign_only_pck:.1f}% (gap = {sign_only_pck - normal_pck:+.1f}%)")
    
    print(f"\n{'CFG on Sign (with past):':<30s}")
    for s in guidance_scales:
        method = f'cfg_sign_{s}'
        if method in df['method'].values:
            method_pck = df[df['method'] == method]['pck_01'].mean()
            print(f"  {method:18s}: PCK = {method_pck:.1f}% (vs normal: {method_pck - normal_pck:+.1f}%)")
    
    print(f"\n{'CFG on Past (with sign):':<30s}")
    for s in guidance_scales:
        method = f'cfg_past_{s}'
        if method in df['method'].values:
            method_pck = df[df['method'] == method]['pck_01'].mean()
            print(f"  {method:18s}: PCK = {method_pck:.1f}% (vs normal: {method_pck - normal_pck:+.1f}%)")
    
    print(f"\n{'Sign-Only + CFG:':<30s}")
    for s in guidance_scales:
        method = f'signonly_cfg_{s}'
        if method in df['method'].values:
            method_pck = df[df['method'] == method]['pck_01'].mean()
            print(f"  {method:18s}: PCK = {method_pck:.1f}% (vs sign_only: {method_pck - sign_only_pck:+.1f}%)")
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")
    
    # Also save summary
    summary_path = args.output.replace('.csv', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("CFG Evaluation Results (Fixed)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Normal:    PCK@0.1 = {normal_pck:.1f}%\n")
        f.write(f"Sign-Only: PCK@0.1 = {sign_only_pck:.1f}%\n\n")
        f.write(summary.to_string())
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    main()