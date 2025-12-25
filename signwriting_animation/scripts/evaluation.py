"""
50-sample evaluation for thesis results.
Config: 8 steps + clip_denoised=True
Now also saves pose files for best/worst samples.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import unshift_hands

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitDiffusion, sanitize_btjc, mean_frame_disp


def tensor_to_pose(t_btjc, header, ref_pose, scale_to_ref=True):
    """Convert tensor to pose format."""
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc
    
    t_np = t.detach().cpu().numpy().astype(np.float32)
    arr = t_np[:, None, :, :]
    T = arr.shape[0]
    conf = np.ones((T, 1, arr.shape[2]), dtype=np.float32)
    fps = ref_pose.body.fps
    
    body = NumPyPoseBody(fps=fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)
    unshift_hands(pose_obj)
    
    if scale_to_ref:
        T_pred = t_np.shape[0]
        T_ref_total = ref_pose.body.data.shape[0]
        future_start = max(0, T_ref_total - T_pred)
        ref_arr = np.asarray(ref_pose.body.data[future_start:future_start+T_pred, 0], dtype=np.float32)
        
        def _var(a):
            center = a.mean(axis=(0, 1), keepdims=True)
            return float(((a - center) ** 2).mean())
        
        pose_data = pose_obj.body.data[:, 0, :, :]
        var_input = _var(pose_data)
        var_ref = _var(ref_arr)
        
        if var_input > 1e-8:
            scale = np.sqrt(var_ref / var_input)
            pose_obj.body.data = pose_obj.body.data * scale
        
        pose_data = pose_obj.body.data[:, 0, :, :].reshape(-1, 3)
        input_center = pose_data.mean(axis=0)
        ref_center = ref_arr.reshape(-1, 3).mean(axis=0)
        pose_obj.body.data = pose_obj.body.data + (ref_center - input_center)
    
    return pose_obj


def compute_disp_ratio_numpy(pred_data, gt_data):
    pred_disp = np.sqrt(np.sum(np.diff(pred_data, axis=0)**2, axis=-1)).mean()
    gt_disp = np.sqrt(np.sum(np.diff(gt_data, axis=0)**2, axis=-1)).mean()
    return pred_disp / (gt_disp + 1e-8), pred_disp, gt_disp


def evaluate():
    DIFFUSION_STEPS = 8
    CLIP_DENOISED = True
    NUM_SAMPLES = 50
    SAVE_POSE_FILES = True
    NUM_POSE_TO_SAVE = 10

    SAMPLE_INDICES = list(range(0, 5000, 100))[:NUM_SAMPLES]
    
    ckpt_path = "logs/full_unfrozen_clip/checkpoints/last.ckpt"
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = f"logs/full/eval_{NUM_SAMPLES}samples_step{DIFFUSION_STEPS}_clip{CLIP_DENOISED}"
    pose_dir = f"{out_dir}/poses"
    # ============================================================
    
    os.makedirs(out_dir, exist_ok=True)
    if SAVE_POSE_FILES:
        os.makedirs(pose_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"THESIS EVALUATION: {NUM_SAMPLES} samples")
    print("=" * 70)
    print(f"  Diffusion steps: {DIFFUSION_STEPS}")
    print(f"  clip_denoised: {CLIP_DENOISED}")
    print(f"  Save pose files: {SAVE_POSE_FILES}")
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
    # 存储用于保存 pose 的数据
    pose_data_cache = {}
    
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
            'disp_ratio': ratio,
            'mpjpe': mpjpe,
            'pck_01': pck_01,
            'pck_005': pck_005,
            'mse': mse,
        })
        
        # Cache data for pose saving
        if SAVE_POSE_FILES:
            pose_data_cache[idx] = {
                'gt_unnorm': gt_unnorm.cpu(),
                'pred_unnorm': pred_unnorm.cpu(),
                'ratio': ratio,
                'pck': pck_01,
            }
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(SAMPLE_INDICES)} samples...")
    
    # ====== SAVE POSE FILES ======
    if SAVE_POSE_FILES and len(results) > 0:
        print("\n" + "=" * 70)
        print("SAVING POSE FILES")
        print("=" * 70)
        
        # Sort by ratio distance from 1.0
        sorted_results = sorted(results, key=lambda x: abs(x['disp_ratio'] - 1.0))
        
        # Best samples (closest to 1.0)
        best_samples = sorted_results[:NUM_POSE_TO_SAVE // 2]
        # Worst samples (furthest from 1.0)
        worst_samples = sorted_results[-(NUM_POSE_TO_SAVE // 2):]
        
        samples_to_save = best_samples + worst_samples
        
        print(f"\nSaving {len(samples_to_save)} pose files:")
        print(f"  Best {len(best_samples)}: {[s['idx'] for s in best_samples]}")
        print(f"  Worst {len(worst_samples)}: {[s['idx'] for s in worst_samples]}")
        
        for r in samples_to_save:
            idx = r['idx']
            ratio = r['disp_ratio']
            pck = r['pck_01']
            
            if idx not in pose_data_cache:
                continue
            
            cache = pose_data_cache[idx]
            
            # Get reference pose
            ref_path = test_ds.records[idx]["pose"]
            if not os.path.isabs(ref_path):
                ref_path = os.path.join(data_dir, ref_path)
            
            with open(ref_path, "rb") as f:
                ref_pose = Pose.read(f)
            ref_pose = reduce_holistic(ref_pose)
            if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
                ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
            
            gt_pose = tensor_to_pose(cache['gt_unnorm'], ref_pose.header, ref_pose)
            pred_pose = tensor_to_pose(cache['pred_unnorm'], ref_pose.header, ref_pose)
            
            # Label
            label = "good" if 0.7 <= ratio <= 1.3 else "bad"
            
            gt_filename = f"{pose_dir}/idx{idx}_{label}_ratio{ratio:.2f}_pck{pck:.0f}_gt.pose"
            pred_filename = f"{pose_dir}/idx{idx}_{label}_ratio{ratio:.2f}_pck{pck:.0f}_pred.pose"
            
            with open(gt_filename, "wb") as f:
                gt_pose.write(f)
            with open(pred_filename, "wb") as f:
                pred_pose.write(f)
            
            print(f"  Saved idx={idx}: ratio={ratio:.2f}, pck={pck:.1f}%, label={label}")
    
    # ====== SUMMARY ======
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    avg_ratio = np.mean([r['disp_ratio'] for r in results])
    std_ratio = np.std([r['disp_ratio'] for r in results])
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
    ratios = [r['disp_ratio'] for r in results]
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
    csv_file = f"{out_dir}/results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results saved to: {csv_file}")
    
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
    
    if SAVE_POSE_FILES:
        print(f"\nPose files saved to: {pose_dir}/")
    
    print("=" * 70)
    print("✅ Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    evaluate()