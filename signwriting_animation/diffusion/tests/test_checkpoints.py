"""
Improved checkpoint test - checks both normalized and unnormalized spaces.
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
    """Compute displacement ratio using numpy (same method as pose file analysis)."""
    pred_disp = np.sqrt(np.sum(np.diff(pred_data, axis=0)**2, axis=-1)).mean()
    gt_disp = np.sqrt(np.sum(np.diff(gt_data, axis=0)**2, axis=-1)).mean()
    return pred_disp / (gt_disp + 1e-8), pred_disp, gt_disp


def compute_disp_ratio_torch(pred, gt):
    """Compute displacement ratio using torch (same as training)."""
    pred_disp = mean_frame_disp(pred)
    gt_disp = mean_frame_disp(gt)
    return pred_disp / (gt_disp + 1e-8), pred_disp, gt_disp


def test_checkpoint():
    """Test checkpoint with detailed diagnostics."""
    
    print("=" * 70)
    print("CHECKPOINT EVALUATION V2 (with space comparison)")
    print("=" * 70)
    
    # Configuration
    ckpt_path = "logs/full/checkpoints/last-v2.ckpt"
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = "logs/full/eval"
    num_samples = 5
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\nLoading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Fixed seed for reproducibility
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
    
    # Get dimensions
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
        diffusion_steps=50,
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
    
    # Check normalization stats
    print("\n" + "=" * 70)
    print("NORMALIZATION STATS")
    print("=" * 70)
    print(f"Mean shape: {lit_model.mean_pose.shape}")
    print(f"Std shape: {lit_model.std_pose.shape}")
    print(f"Mean range: [{lit_model.mean_pose.min():.4f}, {lit_model.mean_pose.max():.4f}]")
    print(f"Std range: [{lit_model.std_pose.min():.4f}, {lit_model.std_pose.max():.4f}]")
    
    # ====== DDPM SAMPLING TEST ======
    print("\n" + "=" * 70)
    print("DDPM SAMPLING TEST (comparing normalized vs unnormalized)")
    print("=" * 70)
    
    results = []
    
    # Store indices for reproducibility
    test_indices = list(range(num_samples))
    
    for i, idx in enumerate(test_indices):
        print(f"\n--- Sample {i} (dataset idx={idx}) ---")
        
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
                clip_denoised=False,
                model_kwargs={"y": {}},
                progress=False,
            )
            pred_norm = lit_model.bjct_to_btjc(pred_bjct)
        
        # Unnormalize
        gt_unnorm = lit_model.unnormalize(gt_norm)
        pred_unnorm = lit_model.unnormalize(pred_norm)
        
        # Compute metrics in BOTH spaces
        
        # 1. Normalized space (torch)
        ratio_norm_torch, pred_d_norm, gt_d_norm = compute_disp_ratio_torch(pred_norm, gt_norm)
        
        # 2. Unnormalized space (torch)
        ratio_unnorm_torch, pred_d_unnorm, gt_d_unnorm = compute_disp_ratio_torch(pred_unnorm, gt_unnorm)
        
        # 3. Unnormalized space (numpy - same as pose file analysis)
        pred_np = pred_unnorm[0].cpu().numpy()
        gt_np = gt_unnorm[0].cpu().numpy()
        ratio_unnorm_np, pred_d_np, gt_d_np = compute_disp_ratio_numpy(pred_np, gt_np)
        
        # MSE and other metrics
        mse_norm = F.mse_loss(pred_norm, gt_norm).item()
        mse_unnorm = F.mse_loss(pred_unnorm, gt_unnorm).item()
        
        per_joint_err = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
        mpjpe = per_joint_err.mean()
        pck_01 = (per_joint_err < 0.1).mean() * 100
        
        print(f"  Normalized space:")
        print(f"    GT disp: {gt_d_norm:.4f}, Pred disp: {pred_d_norm:.4f}, Ratio: {ratio_norm_torch:.4f}")
        print(f"  Unnormalized space (torch):")
        print(f"    GT disp: {gt_d_unnorm:.4f}, Pred disp: {pred_d_unnorm:.4f}, Ratio: {ratio_unnorm_torch:.4f}")
        print(f"  Unnormalized space (numpy):")
        print(f"    GT disp: {gt_d_np:.4f}, Pred disp: {pred_d_np:.4f}, Ratio: {ratio_unnorm_np:.4f}")
        print(f"  MPJPE: {mpjpe:.6f}, PCK@0.1: {pck_01:.1f}%")
        
        results.append({
            'idx': idx,
            'ratio_norm': ratio_norm_torch,
            'ratio_unnorm_torch': ratio_unnorm_torch,
            'ratio_unnorm_np': ratio_unnorm_np,
            'gt_disp_norm': gt_d_norm,
            'pred_disp_norm': pred_d_norm,
            'gt_disp_unnorm': gt_d_unnorm,
            'pred_disp_unnorm': pred_d_unnorm,
            'mpjpe': mpjpe,
            'pck': pck_01,
            'mse_norm': mse_norm,
            'mse_unnorm': mse_unnorm,
        })
        
        # Save pose files for first 3 samples
        if i < 3:
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
            
            with open(f"{out_dir}/sample{i}_gt.pose", "wb") as f:
                gt_pose.write(f)
            with open(f"{out_dir}/sample{i}_pred.pose", "wb") as f:
                pred_pose.write(f)
            print(f"  Saved: {out_dir}/sample{i}_gt.pose, sample{i}_pred.pose")
    
    # ====== SUMMARY ======
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    avg_ratio_norm = np.mean([r['ratio_norm'] for r in results])
    avg_ratio_unnorm_torch = np.mean([r['ratio_unnorm_torch'] for r in results])
    avg_ratio_unnorm_np = np.mean([r['ratio_unnorm_np'] for r in results])
    avg_mpjpe = np.mean([r['mpjpe'] for r in results])
    avg_pck = np.mean([r['pck'] for r in results])
    
    print(f"\nAverage Disp Ratio (normalized, torch):   {avg_ratio_norm:.4f}")
    print(f"Average Disp Ratio (unnormalized, torch): {avg_ratio_unnorm_torch:.4f}")
    print(f"Average Disp Ratio (unnormalized, numpy): {avg_ratio_unnorm_np:.4f}")
    print(f"Average MPJPE: {avg_mpjpe:.6f}")
    print(f"Average PCK@0.1: {avg_pck:.1f}%")
    
    # ====== DIAGNOSIS ======
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    print(f"\nNormalized space ratio:   {avg_ratio_norm:.4f}")
    print(f"Unnormalized space ratio: {avg_ratio_unnorm_np:.4f}")
    
    if abs(avg_ratio_norm - avg_ratio_unnorm_np) > 0.5:
        print("\n⚠️  Large discrepancy between normalized and unnormalized space!")
        print("   This suggests the normalization stats may not match the data.")
        print("   The UNNORMALIZED ratio is more reliable for visual quality.")
    
    # Use unnormalized ratio for final judgment
    if 0.8 <= avg_ratio_unnorm_np <= 1.2:
        print(f"\n✅ Model is working well! (unnorm ratio={avg_ratio_unnorm_np:.4f})")
    elif 0.7 <= avg_ratio_unnorm_np <= 1.3:
        print(f"\n⚠️  Model is okay but could be better (unnorm ratio={avg_ratio_unnorm_np:.4f})")
    else:
        print(f"\n❌ Model needs improvement (unnorm ratio={avg_ratio_unnorm_np:.4f})")
    
    print(f"\nPose files saved to: {out_dir}/")
    print("✅ Testing complete!")


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    test_checkpoint()