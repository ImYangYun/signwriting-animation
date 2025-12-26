"""
Comparative evaluation across multiple checkpoints.
Uses the same samples for fair comparison.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import unshift_hands

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitDiffusion, sanitize_btjc


# ============================================================
# CONFIGURATION - Edit these
# ============================================================
CONFIGS = {
    "frozen_8step": {
        "ckpt": "logs/full/checkpoints/last-v1.ckpt",
        "steps": 8,
        "name": "Frozen CLIP (8 steps)",
    },
    "frozen_50step": {
        "ckpt": "logs/full/checkpoints/last-v2.ckpt",
        "steps": 50,
        "name": "Frozen CLIP (50 steps)",
    },
    "unfrozen_8step": {
        "ckpt": "logs/full_unfrozen_clip/checkpoints/last.ckpt",
        "steps": 8,
        "name": "Unfrozen CLIP (8 steps)",
    },
}

NUM_SAMPLES = 50
CLIP_DENOISED = True
SAMPLE_INDICES = list(range(0, 5000, 100))[:NUM_SAMPLES]

data_dir = "/home/yayun/data/pose_data/"
csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
out_dir = "logs/eval_comparison"
# ============================================================


def compute_disp_ratio_numpy(pred_data, gt_data):
    pred_disp = np.sqrt(np.sum(np.diff(pred_data, axis=0)**2, axis=-1)).mean()
    gt_disp = np.sqrt(np.sum(np.diff(gt_data, axis=0)**2, axis=-1)).mean()
    return pred_disp / (gt_disp + 1e-8), pred_disp, gt_disp


def load_model(ckpt_path, diffusion_steps, num_joints, num_dims, future_len, device):
    """Load a model from checkpoint."""
    from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
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
        diffusion_steps=diffusion_steps,
        vel_weight=1.0,
        t_past=40,
        t_future=future_len,
    )
    lit_model.model = model
    lit_model.load_state_dict(checkpoint['state_dict'])
    lit_model = lit_model.to(device)
    lit_model.eval()
    
    return lit_model


def evaluate_sample(lit_model, test_ds, idx, device, clip_denoised=True):
    """Evaluate a single sample."""
    test_batch = zero_pad_collator([test_ds[idx]])
    cond = test_batch["conditions"]
    
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(test_batch["data"][:1]).to(device)
    
    past_norm = lit_model.normalize(past_raw)
    gt_norm = lit_model.normalize(gt_raw)
    future_len = gt_raw.shape[1]
    
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
            clip_denoised=clip_denoised,
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
    
    return {
        'idx': idx,
        'gt_disp': gt_disp,
        'pred_disp': pred_disp,
        'disp_ratio': ratio,
        'mpjpe': mpjpe,
        'pck_01': pck_01,
        'pck_005': pck_005,
        'mse': mse,
    }


def main():
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 80)
    print("COMPARATIVE EVALUATION")
    print("=" * 80)
    print(f"Configs to test: {list(CONFIGS.keys())}")
    print(f"Samples: {NUM_SAMPLES} (indices: {SAMPLE_INDICES[:3]}...{SAMPLE_INDICES[-3:]})")
    print(f"clip_denoised: {CLIP_DENOISED}")
    print()
    
    # Load dataset
    test_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    print(f"Dataset size: {len(test_ds)}")
    
    # Get dimensions from first sample
    sample = test_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    
    num_joints = sample.shape[-2]
    num_dims = sample.shape[-1]
    future_len = sample.shape[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Filter valid indices
    valid_indices = [idx for idx in SAMPLE_INDICES if idx < len(test_ds)]
    print(f"Valid indices: {len(valid_indices)}")
    
    # Store results for each config
    all_results = {}
    
    # Evaluate each config
    for config_key, config in CONFIGS.items():
        print("\n" + "=" * 80)
        print(f"EVALUATING: {config['name']}")
        print(f"  Checkpoint: {config['ckpt']}")
        print(f"  Diffusion steps: {config['steps']}")
        print("=" * 80)
        
        # Check if checkpoint exists
        if not os.path.exists(config['ckpt']):
            print(f"  ⚠️  Checkpoint not found! Skipping...")
            continue
        
        # Load model
        lit_model = load_model(
            config['ckpt'], 
            config['steps'], 
            num_joints, 
            num_dims, 
            future_len, 
            device
        )
        print(f"  Model loaded!")
        
        results = []
        for i, idx in enumerate(valid_indices):
            result = evaluate_sample(lit_model, test_ds, idx, device, CLIP_DENOISED)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(valid_indices)}...")
        
        all_results[config_key] = results
        
        # Quick summary
        avg_pck = np.mean([r['pck_01'] for r in results])
        avg_ratio = np.mean([r['disp_ratio'] for r in results])
        print(f"  → PCK@0.1: {avg_pck:.1f}%, Ratio: {avg_ratio:.2f}")
        
        # Free memory
        del lit_model
        torch.cuda.empty_cache()
    
    # ============================================================
    # COMPARATIVE ANALYSIS
    # ============================================================
    print("\n" + "=" * 80)
    print("COMPARATIVE RESULTS")
    print("=" * 80)
    
    # Summary table
    print("\n{:<25} {:>12} {:>12} {:>12} {:>12}".format(
        "Config", "PCK@0.1", "PCK@0.05", "Ratio", "MPJPE"
    ))
    print("-" * 80)
    
    summary_data = {}
    for config_key, results in all_results.items():
        if not results:
            continue
        
        avg_pck_01 = np.mean([r['pck_01'] for r in results])
        std_pck_01 = np.std([r['pck_01'] for r in results])
        avg_pck_005 = np.mean([r['pck_005'] for r in results])
        avg_ratio = np.mean([r['disp_ratio'] for r in results])
        std_ratio = np.std([r['disp_ratio'] for r in results])
        avg_mpjpe = np.mean([r['mpjpe'] for r in results])
        std_mpjpe = np.std([r['mpjpe'] for r in results])
        
        summary_data[config_key] = {
            'pck_01': (avg_pck_01, std_pck_01),
            'pck_005': avg_pck_005,
            'ratio': (avg_ratio, std_ratio),
            'mpjpe': (avg_mpjpe, std_mpjpe),
        }
        
        name = CONFIGS[config_key]['name']
        print("{:<25} {:>5.1f}±{:<5.1f} {:>11.1f}% {:>5.2f}±{:<5.2f} {:>5.4f}±{:<.4f}".format(
            name,
            avg_pck_01, std_pck_01,
            avg_pck_005,
            avg_ratio, std_ratio,
            avg_mpjpe, std_mpjpe
        ))
    
    # Per-sample comparison
    print("\n" + "-" * 80)
    print("PER-SAMPLE COMPARISON (first 10 samples)")
    print("-" * 80)
    
    config_keys = list(all_results.keys())
    if len(config_keys) >= 2:
        header = "{:<8}".format("idx")
        for ck in config_keys:
            header += " {:>15}".format(ck[:15])
        print(header)
        
        for i in range(min(10, len(valid_indices))):
            idx = valid_indices[i]
            row = "{:<8}".format(idx)
            for ck in config_keys:
                if i < len(all_results[ck]):
                    pck = all_results[ck][i]['pck_01']
                    ratio = all_results[ck][i]['disp_ratio']
                    row += " {:>6.1f}% r={:<.2f}".format(pck, ratio)
                else:
                    row += " {:>15}".format("N/A")
            print(row)
    
    # ============================================================
    # LATEX OUTPUT
    # ============================================================
    print("\n" + "=" * 80)
    print("LATEX TABLE (copy to thesis)")
    print("=" * 80)
    
    latex = """
\\begin{table}[htbp]
\\centering
\\caption{Full dataset training results with different configurations (n=""" + str(len(valid_indices)) + """).}
\\label{tab:full_training_comparison}
\\begin{tabular}{lccccc}
\\toprule
\\textbf{Configuration} & \\textbf{Diff. Steps} & \\textbf{PCK@0.1} & \\textbf{PCK@0.05} & \\textbf{Ratio} & \\textbf{MPJPE} \\\\
\\midrule
"""
    
    for config_key in config_keys:
        if config_key not in summary_data:
            continue
        s = summary_data[config_key]
        name = CONFIGS[config_key]['name']
        steps = CONFIGS[config_key]['steps']
        
        latex += "{} & {} & {:.1f}\\% $\\pm$ {:.1f} & {:.1f}\\% & {:.2f} $\\pm$ {:.2f} & {:.4f} \\\\\n".format(
            name.replace("(", "").replace(")", "").replace(" steps", ""),
            steps,
            s['pck_01'][0], s['pck_01'][1],
            s['pck_005'],
            s['ratio'][0], s['ratio'][1],
            s['mpjpe'][0]
        )
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    print(latex)
    
    # Save to CSV
    import csv
    for config_key, results in all_results.items():
        if not results:
            continue
        csv_file = f"{out_dir}/{config_key}_results.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved: {csv_file}")
    
    print("\n" + "=" * 80)
    print("✅ Comparative evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    main()