"""
Test inference using the epoch 29 checkpoint.
Quick validation to see if 29 epochs is sufficient.
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


def test_checkpoint():
    """Test the epoch 29 checkpoint."""
    
    print("=" * 70)
    print("TESTING EPOCH 29 CHECKPOINT")
    print("=" * 70)
    
    # Load checkpoint
    ckpt_path = "logs/full/checkpoints/last.ckpt"
    
    print(f"\nLoading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Create model
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    
    # Load a test sample to get dimensions
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
    
    # Test on first 5 samples
    print("\n" + "=" * 70)
    print("RUNNING INFERENCE ON 5 SAMPLES")
    print("=" * 70)
    
    results = []
    
    for idx in range(5):
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
        
        print(f"\nSample {idx}:")
        print(f"  Disp Ratio: {disp_ratio:.4f}")
        print(f"  MPJPE: {mpjpe:.6f}")
        print(f"  PCK@0.1: {pck_01:.1f}%")
        print(f"  MSE: {mse:.6f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (5 samples)")
    print("=" * 70)
    
    avg_disp = np.mean([r['disp_ratio'] for r in results])
    avg_mpjpe = np.mean([r['mpjpe'] for r in results])
    avg_pck = np.mean([r['pck'] for r in results])
    avg_mse = np.mean([r['mse'] for r in results])
    
    print(f"Average Disp Ratio: {avg_disp:.4f} (ideal=1.0)")
    print(f"Average MPJPE: {avg_mpjpe:.6f}")
    print(f"Average PCK@0.1: {avg_pck:.1f}%")
    print(f"Average MSE: {avg_mse:.6f}")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if 0.95 <= avg_disp <= 1.10 and avg_mpjpe < 0.1 and avg_pck > 80:
        print("✅ EXCELLENT! 29 epochs is sufficient!")
        print("   The model generalizes well to unseen data.")
    elif 0.90 <= avg_disp <= 1.15 and avg_mpjpe < 0.15 and avg_pck > 70:
        print("✓ GOOD! 29 epochs is acceptable.")
        print("  More training might improve slightly, but not critical.")
    else:
        print("⚠ NEEDS MORE TRAINING")
        print("  Consider training to 50 epochs or using A100 for full training.")
    
    print("=" * 70)
    
    # Save first sample for visualization
    print("\nSaving first sample for visualization...")
    
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
        import os
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
    
    import os
    os.makedirs("logs/full", exist_ok=True)
    
    with open("logs/full/epoch29_test_gt.pose", "wb") as f:
        gt_pose.write(f)
    with open("logs/full/epoch29_test_pred.pose", "wb") as f:
        pred_pose.write(f)
    
    print("Saved: logs/full/epoch29_test_gt.pose")
    print("Saved: logs/full/epoch29_test_pred.pose")
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    import os
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    test_checkpoint()