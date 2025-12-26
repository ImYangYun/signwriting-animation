"""
Classifier-Free Guidance (CFG) Evaluation

Test if CFG can improve Sign-Only generation by amplifying
the influence of SignWriting conditioning.

Usage:
    python eval_cfg.py --checkpoint logs/full_unfrozen_clip/checkpoints/last.ckpt
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


class CFGSampler:
    """Classifier-Free Guidance sampler for diffusion model."""
    
    def __init__(self, lit_model, device):
        self.lit_model = lit_model
        self.model = lit_model.model
        self.diffusion = lit_model.diffusion
        self.device = device
        
    @torch.no_grad()
    def sample_normal(self, past_bjct, sign_img, target_shape):
        """Normal sampling with both conditions."""
        
        class Wrapper(nn.Module):
            def __init__(self, model, past, sign):
                super().__init__()
                self.model, self.past, self.sign = model, past, sign
            def forward(self, x, t, **kwargs):
                return self.model(x, t, self.past, self.sign)
        
        wrapped = Wrapper(self.model, past_bjct, sign_img)
        
        pred_bjct = self.diffusion.p_sample_loop(
            model=wrapped,
            shape=target_shape,
            clip_denoised=True,
            model_kwargs={"y": {}},
            progress=False,
        )
        return pred_bjct
    
    @torch.no_grad()
    def sample_sign_only(self, sign_img, target_shape):
        """Sign-Only sampling: past motion = zeros."""
        B, J, C, T = target_shape
        zeros_past = torch.zeros(B, J, C, 40, device=self.device)  # T_past = 40
        
        class Wrapper(nn.Module):
            def __init__(self, model, past, sign):
                super().__init__()
                self.model, self.past, self.sign = model, past, sign
            def forward(self, x, t, **kwargs):
                return self.model(x, t, self.past, self.sign)
        
        wrapped = Wrapper(self.model, zeros_past, sign_img)
        
        pred_bjct = self.diffusion.p_sample_loop(
            model=wrapped,
            shape=target_shape,
            clip_denoised=True,
            model_kwargs={"y": {}},
            progress=False,
        )
        return pred_bjct
    
    @torch.no_grad()
    def sample_cfg(self, past_bjct, sign_img, target_shape, guidance_scale=2.0):
        """
        CFG sampling: amplify the difference between cond and uncond.
        
        pred = pred_uncond + s * (pred_cond - pred_uncond)
             = (1-s) * pred_uncond + s * pred_cond
        
        When s > 1, this amplifies the effect of having past motion,
        which should make the model rely more on SignWriting for the 
        "content" while using past for smoothness.
        
        Or we can flip it: use Sign as the condition to amplify.
        """
        B, J, C, T = target_shape
        zeros_past = torch.zeros(B, J, C, 40, device=self.device)
        
        # Custom p_sample_loop with CFG
        # Start from pure noise
        x_t = torch.randn(target_shape, device=self.device)
        
        indices = list(range(self.diffusion.num_timesteps))[::-1]
        
        for i in indices:
            t = torch.tensor([i] * B, device=self.device)
            
            # Conditional prediction (with past)
            pred_cond = self.model(x_t, t, past_bjct, sign_img)
            
            # Unconditional prediction (without past, Sign-Only)
            pred_uncond = self.model(x_t, t, zeros_past, sign_img)
            
            # CFG: amplify the effect of past motion
            # This makes the model produce smoother motion while
            # still being guided by SignWriting
            pred_x0 = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            
            # Get x_{t-1} from predicted x_0
            if i > 0:
                # Use diffusion to get previous timestep
                # Simplified: just use the predicted x0 with some noise
                alpha_bar = self.diffusion.alphas_cumprod[i]
                alpha_bar_prev = self.diffusion.alphas_cumprod[i-1] if i > 0 else 1.0
                
                # Compute x_{t-1}
                beta = 1 - alpha_bar / alpha_bar_prev
                noise = torch.randn_like(x_t) if i > 1 else 0
                
                # x_{t-1} = sqrt(alpha_bar_prev) * pred_x0 + sqrt(1-alpha_bar_prev) * noise
                x_t = (
                    torch.sqrt(torch.tensor(alpha_bar_prev, device=self.device)) * pred_x0 +
                    torch.sqrt(torch.tensor(1 - alpha_bar_prev, device=self.device)) * noise
                )
            else:
                x_t = pred_x0
        
        return x_t
    
    @torch.no_grad()
    def sample_cfg_sign_amplified(self, past_bjct, sign_img, target_shape, guidance_scale=2.0):
        """
        Alternative CFG: amplify SignWriting's influence.
        
        Uses empty/random sign embedding as unconditional.
        NOTE: This requires the model to have seen zero sign embeddings during training,
        which may not be the case. This is more experimental.
        """
        B, J, C, T = target_shape
        
        # Create "unconditional" sign embedding (zeros or random)
        # This is a bit hacky since we didn't train with sign dropout
        zeros_sign = torch.zeros_like(sign_img)
        
        x_t = torch.randn(target_shape, device=self.device)
        indices = list(range(self.diffusion.num_timesteps))[::-1]
        
        for i in indices:
            t = torch.tensor([i] * B, device=self.device)
            
            # With sign
            pred_cond = self.model(x_t, t, past_bjct, sign_img)
            
            # Without sign (experimental)
            pred_uncond = self.model(x_t, t, past_bjct, zeros_sign)
            
            # Amplify sign's influence
            pred_x0 = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            
            if i > 0:
                alpha_bar = self.diffusion.alphas_cumprod[i]
                alpha_bar_prev = self.diffusion.alphas_cumprod[i-1] if i > 0 else 1.0
                noise = torch.randn_like(x_t) if i > 1 else 0
                x_t = (
                    torch.sqrt(torch.tensor(alpha_bar_prev, device=self.device)) * pred_x0 +
                    torch.sqrt(torch.tensor(1 - alpha_bar_prev, device=self.device)) * noise
                )
            else:
                x_t = pred_x0
        
        return x_t


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


def evaluate_sample(sampler, lit_model, batch, guidance_scales=[1.0, 1.5, 2.0, 3.0]):
    """Evaluate one sample with different methods and guidance scales."""
    device = sampler.device
    
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
    
    results = {}
    
    # 1. Normal inference
    pred_bjct = sampler.sample_normal(past_bjct, sign, target_shape)
    pred_norm = lit_model.bjct_to_btjc(pred_bjct)
    results['normal'] = compute_metrics(pred_norm, gt_norm)
    
    # 2. Sign-Only inference
    pred_bjct = sampler.sample_sign_only(sign, target_shape)
    pred_norm = lit_model.bjct_to_btjc(pred_bjct)
    results['sign_only'] = compute_metrics(pred_norm, gt_norm)
    
    # 3. CFG with different scales (amplify past motion effect)
    for scale in guidance_scales:
        if scale == 1.0:
            continue  # Same as normal
        pred_bjct = sampler.sample_cfg(past_bjct, sign, target_shape, guidance_scale=scale)
        pred_norm = lit_model.bjct_to_btjc(pred_bjct)
        results[f'cfg_past_{scale}'] = compute_metrics(pred_norm, gt_norm)
    
    # 4. CFG amplifying sign (experimental)
    for scale in [1.5, 2.0]:
        pred_bjct = sampler.sample_cfg_sign_amplified(past_bjct, sign, target_shape, guidance_scale=scale)
        pred_norm = lit_model.bjct_to_btjc(pred_bjct)
        results[f'cfg_sign_{scale}'] = compute_metrics(pred_norm, gt_norm)
    
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
    parser.add_argument('--output', type=str, default='cfg_results.csv',
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
    
    # Create sampler
    sampler = CFGSampler(lit_model, device)
    
    # Evaluate
    all_results = []
    guidance_scales = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    print(f"\nEvaluating {len(sample_indices)} samples...")
    print("="*70)
    
    for idx in tqdm(sample_indices):
        batch = zero_pad_collator([test_ds[idx]])
        results = evaluate_sample(sampler, lit_model, batch, guidance_scales)
        
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
    
    methods_order = ['normal', 'sign_only'] + [f'cfg_past_{s}' for s in guidance_scales if s != 1.0] + ['cfg_sign_1.5', 'cfg_sign_2.0']
    
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
    for method in methods_order:
        if method in df['method'].values and method != 'normal':
            method_pck = df[df['method'] == method]['pck_01'].mean()
            gap = normal_pck - method_pck
            print(f"{method:20s}: Gap = {gap:+5.1f}%")
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")
    
    # Also save summary
    summary_path = args.output.replace('.csv', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("CFG Evaluation Results\n")
        f.write("="*70 + "\n\n")
        f.write(summary.to_string())
        f.write("\n\n")
        f.write("Key Comparisons:\n")
        f.write("-"*70 + "\n")
        for method in methods_order:
            if method in df['method'].values:
                m = df[df['method'] == method]
                f.write(f"{method:20s}: PCK@0.1 = {m['pck_01'].mean():5.1f}% ± {m['pck_01'].std():4.1f}, "
                        f"ratio = {m['disp_ratio'].mean():.2f}\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    main()