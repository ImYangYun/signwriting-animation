"""
Test inference using the epoch 29 checkpoint.
Quick validation to see if 29 epochs is sufficient.

Includes both:
1. Single-step test (same as training) - to verify model learned correctly
2. Full DDPM sampling test - to check actual generation quality
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

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


def test_single_step(lit_model, test_ds, device, num_samples=5):
    """
    Test using SINGLE-STEP prediction (same as training).
    
    This tests whether the model learned to denoise correctly,
    without the complexity of full DDPM sampling.
    """
    print("\n" + "=" * 70)
    print("TEST 1: SINGLE-STEP PREDICTION (Same as Training)")
    print("=" * 70)
    print("This tests the model's denoising ability at different noise levels.")
    print("If disp_ratio ≈ 1.0 here but not in DDPM sampling, the issue is sampling.\n")
    
    lit_model.eval()
    
    # Test at different timesteps
    timesteps_to_test = [0, 3, 7]  # Low, medium, high noise
    
    all_results = {t: [] for t in timesteps_to_test}
    
    for idx in range(num_samples):
        test_batch = zero_pad_collator([test_ds[idx]])
        cond = test_batch["conditions"]
        
        gt_btjc = sanitize_btjc(test_batch["data"][:1]).to(device)
        past_btjc = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign_img = cond["sign_image"][:1].float().to(device)
        
        gt_norm = lit_model.normalize(gt_btjc)
        past_norm = lit_model.normalize(past_btjc)
        
        gt_bjct = lit_model.btjc_to_bjct(gt_norm)
        past_bjct = lit_model.btjc_to_bjct(past_norm)
        
        with torch.no_grad():
            for t_val in timesteps_to_test:
                timestep = torch.tensor([t_val], device=device, dtype=torch.long)
                noise = torch.randn_like(gt_bjct)
                x_noisy = lit_model.diffusion.q_sample(gt_bjct, timestep, noise=noise)
                
                # Single-step prediction (same as training)
                pred_x0_bjct = lit_model.model(x_noisy, timestep, past_bjct, sign_img)
                
                # Compute disp_ratio same as training
                pred_vel = pred_x0_bjct[..., 1:] - pred_x0_bjct[..., :-1]
                gt_vel = gt_bjct[..., 1:] - gt_bjct[..., :-1]
                
                pred_disp = pred_vel.abs().mean().item()
                gt_disp = gt_vel.abs().mean().item()
                disp_ratio = pred_disp / (gt_disp + 1e-8)
                
                # MSE
                mse = F.mse_loss(pred_x0_bjct, gt_bjct).item()
                
                all_results[t_val].append({
                    'disp_ratio': disp_ratio,
                    'mse': mse
                })
    
    # Print results
    print(f"Results across {num_samples} samples:\n")
    print(f"{'Timestep':<12} {'Avg Disp Ratio':<18} {'Avg MSE':<15} {'Status'}")
    print("-" * 60)
    
    for t_val in timesteps_to_test:
        avg_disp = np.mean([r['disp_ratio'] for r in all_results[t_val]])
        avg_mse = np.mean([r['mse'] for r in all_results[t_val]])
        
        if 0.9 <= avg_disp <= 1.1:
            status = "✅ Good"
        elif 0.8 <= avg_disp <= 1.2:
            status = "⚠️ Okay"
        else:
            status = "❌ Bad"
        
        print(f"t={t_val:<10} {avg_disp:<18.4f} {avg_mse:<15.6f} {status}")
    
    return all_results


def test_ddpm_sampling(lit_model, test_ds, device, future_len, num_samples=5):
    """
    Test using FULL DDPM SAMPLING (actual inference).
    
    This is how the model will be used in practice - generating
    from pure noise through iterative denoising.
    """
    print("\n" + "=" * 70)
    print("TEST 2: FULL DDPM SAMPLING (Actual Inference)")
    print("=" * 70)
    print("This tests the full generation pipeline from pure noise.\n")
    
    lit_model.eval()
    results = []
    
    for idx in range(num_samples):
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
        
        # Compute metrics
        mse = F.mse_loss(pred_norm, gt_norm).item()
        disp_pred = mean_frame_disp(pred_norm)
        disp_gt = mean_frame_disp(gt_norm)
        disp_ratio = disp_pred / (disp_gt + 1e-8)
        
        pred_np = pred_norm[0].cpu().numpy()
        gt_np = gt_norm[0].cpu().numpy()
        per_joint_err = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
        mpjpe = per_joint_err.mean()
        pck_01 = (per_joint_err < 0.1).mean() * 100
        
        results.append({
            'disp_ratio': disp_ratio,
            'mpjpe': mpjpe,
            'pck': pck_01,
            'mse': mse
        })
        
        print(f"Sample {idx}: Disp Ratio={disp_ratio:.4f}, MPJPE={mpjpe:.6f}, PCK@0.1={pck_01:.1f}%")
    
    # Summary
    print("\n" + "-" * 40)
    avg_disp = np.mean([r['disp_ratio'] for r in results])
    avg_mpjpe = np.mean([r['mpjpe'] for r in results])
    avg_pck = np.mean([r['pck'] for r in results])
    avg_mse = np.mean([r['mse'] for r in results])
    
    print(f"Average Disp Ratio: {avg_disp:.4f} (ideal=1.0)")
    print(f"Average MPJPE: {avg_mpjpe:.6f}")
    print(f"Average PCK@0.1: {avg_pck:.1f}%")
    print(f"Average MSE: {avg_mse:.6f}")
    
    return results


def test_checkpoint():
    """Test the epoch 29 checkpoint with both single-step and DDPM sampling."""
    
    print("=" * 70)
    print("CHECKPOINT EVALUATION")
    print("=" * 70)
    
    # Load checkpoint
    ckpt_path = "logs/full/checkpoints/last-v1.ckpt"
    
    print(f"\nLoading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Create model
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    
    # Load a test sample to get dimensions
    # Use fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
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
    
    # Create LitDiffusion model
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
    
    # Load weights
    lit_model.load_state_dict(checkpoint['state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.eval()
    
    print(f"Model loaded on: {device}")
    
    # ====== TEST 1: Single-step (same as training) ======
    single_step_results = test_single_step(lit_model, test_ds, device, num_samples=5)
    
    # ====== TEST 2: Full DDPM sampling ======
    ddpm_results = test_ddpm_sampling(lit_model, test_ds, device, future_len, num_samples=5)
    
    # ====== DIAGNOSIS ======
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    # Check single-step at t=0 (minimal noise)
    avg_single_t0 = np.mean([r['disp_ratio'] for r in single_step_results[0]])
    avg_ddpm = np.mean([r['disp_ratio'] for r in ddpm_results])
    
    print(f"\nSingle-step (t=0) disp_ratio: {avg_single_t0:.4f}")
    print(f"DDPM sampling disp_ratio:     {avg_ddpm:.4f}")
    
    if 0.9 <= avg_single_t0 <= 1.1 and not (0.9 <= avg_ddpm <= 1.1):
        print("\n⚠️  DIAGNOSIS: Model learned well, but DDPM sampling is broken!")
        print("   Possible causes:")
        print("   - diffusion_steps=8 is too few (try 50-100)")
        print("   - Noise schedule mismatch")
        print("   - p_sample_loop implementation issue")
    elif not (0.9 <= avg_single_t0 <= 1.1):
        print("\n❌ DIAGNOSIS: Model hasn't learned properly yet.")
        print("   - Continue training for more epochs")
        print("   - Check learning rate and loss weights")
    else:
        print("\n✅ DIAGNOSIS: Model is working well!")
    
    # ====== SAVE SAMPLE FOR VISUALIZATION ======
    print("\n" + "=" * 70)
    print("SAVING SAMPLE FOR VISUALIZATION")
    print("=" * 70)
    
    test_batch = zero_pad_collator([test_ds[0]])
    cond = test_batch["conditions"]
    
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(test_batch["data"][:1]).to(device)
    
    past_norm = lit_model.normalize(past_raw)
    gt_norm = lit_model.normalize(gt_raw)
    
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
    
    # Get reference pose
    ref_path = test_ds.records[0]["pose"]
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    ref_pose = reduce_holistic(ref_pose)
    if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
        ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    
    gt_unnorm = lit_model.unnormalize(gt_norm)
    pred_unnorm = lit_model.unnormalize(pred_norm)
    
    gt_pose = tensor_to_pose(gt_unnorm, ref_pose.header, ref_pose)
    pred_pose = tensor_to_pose(pred_unnorm, ref_pose.header, ref_pose)
    
    os.makedirs("logs/full", exist_ok=True)
    
    with open("logs/full/epoch29_test_gt.pose", "wb") as f:
        gt_pose.write(f)
    with open("logs/full/epoch29_test_pred.pose", "wb") as f:
        pred_pose.write(f)
    
    print("Saved: logs/full/epoch29_test_gt.pose")
    print("Saved: logs/full/epoch29_test_pred.pose")
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    test_checkpoint()