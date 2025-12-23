"""
50-sample evaluation for thesis results.
Config: 8 steps + clip_denoised=True
"""
import os
import torch
import torch.nn.functional as F
import numpy as np

from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitDiffusion, sanitize_btjc, mean_frame_disp


def compute_disp_ratio_numpy(pred_data, gt_data):
    pred_disp = np.sqrt(np.sum(np.diff(pred_data, axis=0)**2, axis=-1)).mean()
    gt_disp = np.sqrt(np.sum(np.diff(gt_data, axis=0)**2, axis=-1)).mean()
    return pred_disp / (gt_disp + 1e-8), pred_disp, gt_disp


def evaluate():
    DIFFUSION_STEPS = 8
    CLIP_DENOISED = True
    NUM_SAMPLES = 50

    SAMPLE_INDICES = list(range(0, 5000, 100))[:NUM_SAMPLES]
    
    ckpt_path = "logs/full/checkpoints/last-v1.ckpt"
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = f"logs/full/eval_{NUM_SAMPLES}samples_step{DIFFUSION_STEPS}_clip{CLIP_DENOISED}"
    # ============================================================
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"THESIS EVALUATION: {NUM_SAMPLES} samples")
    print("=" * 70)
    print(f"  Diffusion steps: {DIFFUSION_STEPS}")
    print(f"  clip_denoised: {CLIP_DENOISED}")
    print(f"  Output: {out_dir}")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Load dataset
    test_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    print(f"  Dataset size: {len(test_ds)}")
    print(f"  Sample indices: {SAMPLE_INDICES[:5]}...{SAMPLE_INDICES[-5:]}")
    
    # Get dimensions
    sample = test_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    
    num_joints = sample.shape[-2]
    num_dims = sample.shape[-1]
    future_len = sample.shape[0]
    
    # Create model
    from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
    
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
        t_past=40,
        t_future=future_len,
    )
    lit_model.model = model
    lit_model.load_state_dict(checkpoint['state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.eval()
    
    print(f"  Model loaded on: {device}")
    print("\n" + "=" * 70)
    print("RUNNING EVALUATION...")
    print("=" * 70)
    
    results = []
    
    for i, idx in enumerate(SAMPLE_INDICES):
        if idx >= len(test_ds):
            print(f"  Skipping idx={idx} (out of range)")
            continue
            
        test_batch = zero_pad_collator([test_ds[idx]])
        cond = test_batch["conditions"]
        
        past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt_raw = sanitize_btjc(test_batch["data"][:1]).to(device)
        
        past_norm = lit_model.normalize(past_raw)
        gt_norm = lit_model.normalize(gt_raw)
        
        # DDPM sampling
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
                clip_denoised=CLIP_DENOISED,
                model_kwargs={"y": {}},
                progress=False,
            )
            pred_norm = lit_model.bjct_to_btjc(pred_bjct)
        
        # Unnormalize
        gt_unnorm = lit_model.unnormalize(gt_norm)
        pred_unnorm = lit_model.unnormalize(pred_norm)
        
        # Metrics
        pred_np = pred_unnorm[0].cpu().numpy()
        gt_np = gt_unnorm[0].cpu().numpy()
        ratio, pred_disp, gt_disp = compute_disp_ratio_numpy(pred_np, gt_np)
        
        per_joint_err = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
        mpjpe = per_joint_err.mean()
        pck_01 = (per_joint_err < 0.1).mean() * 100
        pck_005 = (per_joint_err < 0.05).mean() * 100
        mse = F.mse_loss(pred_unnorm, gt_unnorm).item()
        
        results.append({
            'idx': idx,
            'gt_disp': gt_disp,
            'pred_disp': pred_disp,
            'ratio': ratio,
            'mpjpe': mpjpe,
            'pck_01': pck_01,
            'pck_005': pck_005,
            'mse': mse,
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(SAMPLE_INDICES)} samples...")
    
    # ====== SUMMARY ======
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    avg_ratio = np.mean([r['ratio'] for r in results])
    std_ratio = np.std([r['ratio'] for r in results])
    avg_mpjpe = np.mean([r['mpjpe'] for r in results])
    std_mpjpe = np.std([r['mpjpe'] for r in results])
    avg_pck_01 = np.mean([r['pck_01'] for r in results])
    std_pck_01 = np.std([r['pck_01'] for r in results])
    avg_pck_005 = np.mean([r['pck_005'] for r in results])
    avg_mse = np.mean([r['mse'] for r in results])
    
    print(f"\nSamples evaluated: {len(results)}")
    print(f"\nDisplacement Ratio: {avg_ratio:.4f} ± {std_ratio:.4f}")
    print(f"MPJPE:              {avg_mpjpe:.6f} ± {std_mpjpe:.6f}")
    print(f"PCK@0.1:            {avg_pck_01:.1f}% ± {std_pck_01:.1f}%")
    print(f"PCK@0.05:           {avg_pck_005:.1f}%")
    print(f"MSE:                {avg_mse:.6f}")
    
    # Ratio distribution
    ratios = [r['ratio'] for r in results]
    print(f"\nRatio distribution:")
    print(f"  Min:    {min(ratios):.4f}")
    print(f"  Max:    {max(ratios):.4f}")
    print(f"  Median: {np.median(ratios):.4f}")
    
    # Count by ratio range
    ratio_good = sum(1 for r in ratios if 0.8 <= r <= 1.2)
    ratio_ok = sum(1 for r in ratios if 0.5 <= r <= 1.5)
    print(f"\n  Ratio in [0.8, 1.2]: {ratio_good}/{len(results)} ({100*ratio_good/len(results):.1f}%)")
    print(f"  Ratio in [0.5, 1.5]: {ratio_ok}/{len(results)} ({100*ratio_ok/len(results):.1f}%)")
    
    # Save detailed results
    import csv
    csv_path = f"{out_dir}/results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results saved to: {csv_path}")
    
    # LaTeX table format
    print("\n" + "=" * 70)
    print("LATEX TABLE (copy to thesis)")
    print("=" * 70)
    print(f"""
\\begin{{table}}[h]
\\centering
\\caption{{Quantitative evaluation results (n={len(results)})}}
\\begin{{tabular}}{{lc}}
\\toprule
Metric & Value \\\\
\\midrule
Displacement Ratio & {avg_ratio:.2f} $\\pm$ {std_ratio:.2f} \\\\
MPJPE & {avg_mpjpe:.4f} $\\pm$ {std_mpjpe:.4f} \\\\
PCK@0.1 & {avg_pck_01:.1f}\\% $\\pm$ {std_pck_01:.1f}\\% \\\\
PCK@0.05 & {avg_pck_005:.1f}\\% \\\\
MSE & {avg_mse:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")
    
    print("=" * 70)
    print("✅ Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    evaluate()