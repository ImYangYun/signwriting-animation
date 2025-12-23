"""
Generate pose files for selected samples from 50-sample evaluation.
Based on results.csv to pick good/bad examples.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import unshift_hands

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitDiffusion, sanitize_btjc


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


def generate_pose_files():
    """Generate pose files for selected samples."""

    # ============================================================
    # CONFIGURATION
    # ============================================================
    ckpt_path = "logs/full/checkpoints/last-v1.ckpt"  # 8-step model
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    results_csv = "logs/full/eval_50samples_step8_clipTrue/results.csv"
    out_dir = "logs/full/pose_samples"
    
    CLIP_DENOISED = True
    # ============================================================
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 70)
    print("GENERATE POSE FILES FOR SELECTED SAMPLES")
    print("=" * 70)
    
    # Load results.csv to find good/bad samples
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        print(f"\nLoaded {len(df)} samples from {results_csv}")
        print(f"Columns: {list(df.columns)}")
        
        # Find best samples (ratio closest to 1.0)
        df['ratio_diff'] = abs(df['disp_ratio'] - 1.0)
        df_sorted = df.sort_values('ratio_diff')
        
        print("\n--- Best samples (ratio closest to 1.0) ---")
        print(df_sorted.head(10)[['idx', 'disp_ratio', 'pck_01', 'mpjpe']])
        
        print("\n--- Worst samples (ratio furthest from 1.0) ---")
        print(df_sorted.tail(5)[['idx', 'disp_ratio', 'pck_01', 'mpjpe']])
        
        # Select samples: 5 best + 2 worst for comparison
        best_indices = df_sorted.head(5)['idx'].tolist()
        worst_indices = df_sorted.tail(2)['idx'].tolist()
        selected_indices = best_indices + worst_indices
        
        print(f"\nSelected indices: {selected_indices}")
    else:
        print(f"WARNING: {results_csv} not found, using default indices")
        # Default: uniformly sampled
        selected_indices = [0, 500, 1000, 2000, 3000, 4000, 4500]
    
    # Load model
    print(f"\nLoading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset
    test_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    
    sample = test_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    
    num_joints = sample.shape[-2]
    num_dims = sample.shape[-1]
    future_len = sample.shape[0]
    
    print(f"Dimensions: J={num_joints}, D={num_dims}, T={future_len}")
    
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
        diffusion_steps=8,
        vel_weight=1.0,
        t_past=40,
        t_future=future_len,
    )
    lit_model.model = model
    lit_model.load_state_dict(checkpoint['state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.eval()
    
    print(f"Model loaded on: {device}")
    
    # Generate pose files
    print("\n" + "=" * 70)
    print("GENERATING POSE FILES")
    print("=" * 70)
    
    results = []
    
    for i, idx in enumerate(selected_indices):
        print(f"\n--- Sample {i}: idx={idx} ---")
        
        if idx >= len(test_ds):
            print(f"  WARNING: idx {idx} out of range, skipping")
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
        
        # Compute metrics
        pred_np = pred_unnorm[0].cpu().numpy()
        gt_np = gt_unnorm[0].cpu().numpy()
        
        pred_disp = np.sqrt(np.sum(np.diff(pred_np, axis=0)**2, axis=-1)).mean()
        gt_disp = np.sqrt(np.sum(np.diff(gt_np, axis=0)**2, axis=-1)).mean()
        ratio = pred_disp / (gt_disp + 1e-8)
        
        per_joint_err = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
        mpjpe = per_joint_err.mean()
        pck_01 = (per_joint_err < 0.1).mean() * 100
        
        print(f"  GT disp: {gt_disp:.4f}, Pred disp: {pred_disp:.4f}, Ratio: {ratio:.4f}")
        print(f"  MPJPE: {mpjpe:.6f}, PCK@0.1: {pck_01:.1f}%")
        
        results.append({
            'idx': idx,
            'ratio': ratio,
            'pck': pck_01,
            'mpjpe': mpjpe,
        })
        
        # Save pose files
        ref_path = test_ds.records[idx]["pose"]
        if not os.path.isabs(ref_path):
            ref_path = os.path.join(data_dir, ref_path)
        
        with open(ref_path, "rb") as f:
            ref_pose = Pose.read(f)
        ref_pose = reduce_holistic(ref_pose)
        if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
            ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
        
        gt_pose = tensor_to_pose(gt_unnorm, ref_pose.header, ref_pose)
        pred_pose = tensor_to_pose(pred_unnorm, ref_pose.header, ref_pose)
        
        # Label as good/bad based on ratio
        label = "good" if 0.7 <= ratio <= 1.3 else "bad"
        
        gt_filename = f"{out_dir}/idx{idx}_{label}_ratio{ratio:.2f}_gt.pose"
        pred_filename = f"{out_dir}/idx{idx}_{label}_ratio{ratio:.2f}_pred.pose"
        
        with open(gt_filename, "wb") as f:
            gt_pose.write(f)
        with open(pred_filename, "wb") as f:
            pred_pose.write(f)
        
        print(f"  Saved: {os.path.basename(gt_filename)}, {os.path.basename(pred_filename)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'idx':>6} | {'Ratio':>6} | {'PCK@0.1':>8} | {'Label':>5}")
    print("-" * 40)
    for r in results:
        label = "good" if 0.7 <= r['ratio'] <= 1.3 else "bad"
        print(f"{r['idx']:>6} | {r['ratio']:>6.2f} | {r['pck']:>7.1f}% | {label:>5}")
    
    print(f"\nPose files saved to: {out_dir}/")
    print("✅ Complete!")


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    generate_pose_files()