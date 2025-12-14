import os
import torch
import torch.nn.functional as F
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import unshift_hands

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import (
    LitDiffusion, 
    sanitize_btjc, 
    masked_dtw, 
    mean_frame_disp,
)


def tensor_to_pose(t_btjc, header, ref_pose, gt_btjc=None, apply_scale=True):
    """
    Convert normalized tensor predictions back to pose format for visualization.
    
    This function handles:
    1. Conversion from tensor to NumPy
    2. Adding confidence scores
    3. unshift_hands operation (inverse of shift_hands preprocessing)
    4. Scaling from normalized space to pixel space
    5. Translation to align with reference pose
    
    Args:
        t_btjc: Prediction tensor [T, J, C] or [B, T, J, C]
        header: Pose header from reference pose file
        ref_pose: Reference pose for alignment and scaling
        gt_btjc: Ground truth tensor for computing scale (optional)
        apply_scale: Whether to apply scaling (default True)
        
    Returns:
        Pose object ready for visualization/saving
    """
    # Handle batch dimension if present
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc
    
    # Convert to NumPy
    t_np = t.detach().cpu().numpy().astype(np.float32)

    # Extract GT for scaling if provided
    gt_np = None
    if gt_btjc is not None:
        if gt_btjc.dim() == 4:
            gt_np = gt_btjc[0].detach().cpu().numpy().astype(np.float32)
        else:
            gt_np = gt_btjc.detach().cpu().numpy().astype(np.float32)

    # Create pose body with confidence scores from reference
    arr = t_np[:, None, :, :]  # Add person dimension
    conf = ref_pose.body.confidence[:len(t_np)].copy()
    fps = ref_pose.body.fps
    
    body = NumPyPoseBody(fps=fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)
    
    # Reverse shift_hands preprocessing
    unshift_hands(pose_obj)
    print("  ✓ unshift succeeded")

    # Determine alignment range in reference pose
    T_pred = pose_obj.body.data.shape[0]
    T_ref_total = ref_pose.body.data.shape[0]
    
    future_start = max(0, T_ref_total - T_pred)
    ref_arr = np.asarray(ref_pose.body.data[future_start:future_start+T_pred, 0], dtype=np.float32)
    pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)
    
    print(f"  [alignment] using ref frames {future_start}-{future_start+T_pred-1}")
    
    # Scale from normalized space to pixel space
    if apply_scale and gt_np is not None:
        def _var(a):
            """Compute variance after removing center"""
            center = a.mean(axis=1, keepdims=True)
            return float(((a - center) ** 2).mean())
        
        # Compute variances
        var_gt_norm = _var(gt_np)
        var_ref = _var(ref_arr)
        
        # Scale = sqrt(var_ref / var_gt_norm)
        if var_gt_norm > 1e-8:
            scale = np.sqrt(var_ref / var_gt_norm)
            print(f"  [scale] var_ref={var_ref:.2f}, var_gt_norm={var_gt_norm:.6f}")
            print(f"  [scale] normalized→pixel scale={scale:.2f}")
            pose_obj.body.data *= scale
            pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)

    # Translate to align centers
    ref_c = ref_arr.reshape(-1, 3).mean(axis=0)
    pred_c = pred_arr.reshape(-1, 3).mean(axis=0)
    delta = ref_c - pred_c
    print(f"  [translate] delta={delta}")
    pose_obj.body.data += delta
    
    print(f"  [final] range=[{pose_obj.body.data.min():.2f}, {pose_obj.body.data.max():.2f}]")
    
    return pose_obj


if __name__ == "__main__":
    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Paths
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = "logs/4sample_test_fixed"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("4-Sample Overfit Test (Fixed Version)")
    print("=" * 70)

    # Configuration
    NUM_SAMPLES = 4      # Small number for overfit test
    MAX_EPOCHS = 500     # Enough to fully overfit
    BATCH_SIZE = 4       # All samples in one batch

    # Load full dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    print(f"Total dataset size: {len(base_ds)}")

    # Create subset with only 4 samples
    class SubsetDataset(torch.utils.data.Dataset):
        """Simple wrapper to select subset of samples."""
        def __init__(self, base, indices):
            self.samples = [base[i] for i in indices]
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    train_ds = SubsetDataset(base_ds, list(range(NUM_SAMPLES)))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=zero_pad_collator)

    # Get dimensions from sample
    sample = train_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    
    num_joints, num_dims, future_len = sample.shape[-2], sample.shape[-1], sample.shape[0]
    print(f"J={num_joints}, D={num_dims}, T_future={future_len}")

    # Create model
    model = LitDiffusion(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,              # Higher LR for faster overfitting
        diffusion_steps=8,
        vel_weight=1.0,
        t_past=40,
        t_future=future_len,
    )

    # Train until convergence
    print("\nStarting training...")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,  # Don't need checkpoints for overfit test
        log_every_n_steps=50,
    )
    trainer.fit(model, train_loader)

    # ========================================================================
    # Inference and Evaluation
    # ========================================================================
    print("\n" + "=" * 70)
    print("Inference")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Get first sample for inference
    test_batch = zero_pad_collator([train_ds[0]])
    cond = test_batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(test_batch["data"][:1]).to(device)

    # Run inference
    with torch.no_grad():
        pred_raw = model.sample(past_raw, sign, future_len)
        
        # === Evaluation in Normalized Space ===
        # MSE: overall position accuracy
        mse = F.mse_loss(pred_raw, gt_raw).item()
        
        # Frame displacement: detect static poses
        disp_pred = mean_frame_disp(pred_raw)
        disp_gt = mean_frame_disp(gt_raw)
        disp_ratio = disp_pred / (disp_gt + 1e-8)
        
        # DTW: temporal alignment quality
        mask = torch.ones(1, future_len, device=device)
        dtw = masked_dtw(pred_raw, gt_raw, mask)
        if isinstance(dtw, torch.Tensor):
            dtw = dtw.item()
        
        # MPJPE and PCK: position accuracy metrics
        pred_np = pred_raw[0].cpu().numpy()
        gt_np = gt_raw[0].cpu().numpy()
        per_joint_err = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
        mpjpe = per_joint_err.mean()
        pck_01 = (per_joint_err < 0.1).mean() * 100  # % of joints within 0.1 units
        pck_02 = (per_joint_err < 0.2).mean() * 100  # % of joints within 0.2 units

    # Print normalized space results
    print(f"""
Results (Normalized Space):
  MSE: {mse:.6f}
  MPJPE: {mpjpe:.6f}
  PCK@0.1: {pck_01:.1f}%
  PCK@0.2: {pck_02:.1f}%
  DTW: {dtw:.6f}
  Disp GT: {disp_gt:.6f}
  Disp Pred: {disp_pred:.6f}
  Disp Ratio: {disp_ratio:.4f}
""")

    # ========================================================================
    # Save Pose Files for Visualization
    # ========================================================================
    print("=" * 70)
    print("Saving Pose Files...")
    print("=" * 70)
    
    # Load reference pose file
    ref_path = base_ds.records[0]["pose"]
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    ref_pose = reduce_holistic(ref_pose)
    
    # Remove world landmarks if present (they're redundant)
    if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
        ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    
    # Convert GT using same process as Pred for fair comparison
    print("\nConverting GT (using tensor_to_pose):")
    gt_pose = tensor_to_pose(gt_raw, ref_pose.header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
    with open(f"{out_dir}/gt.pose", "wb") as f:
        gt_pose.write(f)
    
    print("\nConverting Pred (using tensor_to_pose):")
    pred_pose = tensor_to_pose(pred_raw, ref_pose.header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
    with open(f"{out_dir}/pred.pose", "wb") as f:
        pred_pose.write(f)
    
    print(f"\n✓ Saved to {out_dir}/")

    # ========================================================================
    # Pixel Space Verification
    # ========================================================================
    print("\n" + "=" * 70)
    print("Pixel Space Verification:")
    print("=" * 70)
    
    gt_data = gt_pose.body.data[:, 0, :, :]
    pred_data = pred_pose.body.data[:, 0, :, :]
    
    # Overall pixel-space metrics
    pixel_diff = np.abs(gt_data - pred_data)
    pixel_mpjpe = np.sqrt(((gt_data - pred_data) ** 2).sum(-1)).mean()
    
    print(f"  Pixel space MPJPE: {pixel_mpjpe:.2f} pixels")
    print(f"  Mean difference: {pixel_diff.mean():.2f} pixels")
    print(f"  Max difference: {pixel_diff.max():.2f} pixels")
    
    # Per-frame breakdown
    per_frame_diff = pixel_diff.mean(axis=(1, 2))
    print(f"\n  Per-frame mean difference (first 10 frames):")
    for i in range(min(10, len(per_frame_diff))):
        print(f"    Frame {i}: {per_frame_diff[i]:.2f} pixels")

    # ========================================================================
    # Hand Motion Range Diagnosis
    # ========================================================================
    print("\n" + "=" * 70)
    print("Hand Motion Range Diagnosis")
    print("=" * 70)
    
    # MediaPipe Holistic keypoint indices
    # 0-32: Pose (33 points)
    # 33-53: Face contour (21 points) 
    # 54-86: Left hand (33 points)
    # 87-119: Right hand (33 points)
    LEFT_HAND_START = 54
    LEFT_HAND_END = 87
    RIGHT_HAND_START = 87
    RIGHT_HAND_END = 120
    
    # Extract hand keypoints
    gt_left_hand = gt_data[:, LEFT_HAND_START:LEFT_HAND_END, :]
    gt_right_hand = gt_data[:, RIGHT_HAND_START:RIGHT_HAND_END, :]
    pred_left_hand = pred_data[:, LEFT_HAND_START:LEFT_HAND_END, :]
    pred_right_hand = pred_data[:, RIGHT_HAND_START:RIGHT_HAND_END, :]
    
    def calc_movement_stats(hand_data, name):
        """
        Calculate comprehensive motion statistics for hand.
        
        Args:
            hand_data: Hand keypoints [T, 33, 3]
            name: Identifier for printing
            
        Returns:
            variance, mean_displacement, (x_range, y_range, z_range)
        """
        # Compute variance (spread of hand positions)
        center = hand_data.mean(axis=(0, 1))  # [3]
        centered = hand_data - center
        variance = (centered ** 2).mean()
        std = np.sqrt(variance)
        
        # Compute frame-to-frame displacement
        frame_disps = []
        for t in range(1, len(hand_data)):
            disp = np.sqrt(((hand_data[t] - hand_data[t-1]) ** 2).sum(axis=-1)).mean()
            frame_disps.append(disp)
        mean_disp = np.mean(frame_disps) if frame_disps else 0
        
        # Compute spatial extent in each dimension
        x_range = hand_data[:, :, 0].max() - hand_data[:, :, 0].min()
        y_range = hand_data[:, :, 1].max() - hand_data[:, :, 1].min()
        z_range = hand_data[:, :, 2].max() - hand_data[:, :, 2].min()
        
        print(f"\n{name}:")
        print(f"  Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
        print(f"  Variance: {variance:.2f}")
        print(f"  Std: {std:.2f}")
        print(f"  Mean frame displacement: {mean_disp:.2f} pixels")
        print(f"  X range: {x_range:.2f} pixels")
        print(f"  Y range: {y_range:.2f} pixels")
        print(f"  Z range: {z_range:.2f}")
        
        return variance, mean_disp, (x_range, y_range, z_range)
    
    # Analyze left hand
    print("\nLeft Hand Motion Range:")
    gt_var_l, gt_disp_l, gt_range_l = calc_movement_stats(gt_left_hand, "  GT Left")
    pred_var_l, pred_disp_l, pred_range_l = calc_movement_stats(pred_left_hand, "  Pred Left")
    
    print(f"\n  Left Hand Comparison:")
    print(f"    Variance ratio (Pred/GT): {pred_var_l / (gt_var_l + 1e-8):.4f}")
    print(f"    Displacement ratio (Pred/GT): {pred_disp_l / (gt_disp_l + 1e-8):.4f}")
    print(f"    X range ratio: {pred_range_l[0] / (gt_range_l[0] + 1e-8):.4f}")
    print(f"    Y range ratio: {pred_range_l[1] / (gt_range_l[1] + 1e-8):.4f}")
    
    # Analyze right hand
    print("\nRight Hand Motion Range:")
    gt_var_r, gt_disp_r, gt_range_r = calc_movement_stats(gt_right_hand, "  GT Right")
    pred_var_r, pred_disp_r, pred_range_r = calc_movement_stats(pred_right_hand, "  Pred Right")
    
    print(f"\n  Right Hand Comparison:")
    print(f"    Variance ratio (Pred/GT): {pred_var_r / (gt_var_r + 1e-8):.4f}")
    print(f"    Displacement ratio (Pred/GT): {pred_disp_r / (gt_disp_r + 1e-8):.4f}")
    print(f"    X range ratio: {pred_range_r[0] / (gt_range_r[0] + 1e-8):.4f}")
    print(f"    Y range ratio: {pred_range_r[1] / (gt_range_r[1] + 1e-8):.4f}")
    
    # Analyze individual finger movements
    print("\nRight Hand Finger Analysis:")
    finger_names = ["Wrist", "Thumb", "Index", "Middle", "Ring", "Pinky"]
    finger_ranges = [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 21)]
    
    for finger_name, (start, end) in zip(finger_names, finger_ranges):
        gt_finger = gt_right_hand[:, start:end, :]
        pred_finger = pred_right_hand[:, start:end, :]
        
        # Compute mean displacement for this finger
        gt_disp = np.sqrt(np.diff(gt_finger, axis=0) ** 2).sum(axis=-1).mean()
        pred_disp = np.sqrt(np.diff(pred_finger, axis=0) ** 2).sum(axis=-1).mean()
        
        ratio = pred_disp / (gt_disp + 1e-8)
        print(f"  {finger_name}: GT={gt_disp:.2f}px, Pred={pred_disp:.2f}px, ratio={ratio:.4f}")
    
    # Diagnostic conclusion
    print("\nDiagnostic Conclusion:")
    if pred_var_r / (gt_var_r + 1e-8) < 0.8:
        print("  ⚠️ Pred hand motion variance significantly lower than GT")
        print("     → Model learned compressed motion range (position accurate but less 'active')")
        motion_issue = True
    elif abs(pred_var_r / (gt_var_r + 1e-8) - 1.0) < 0.1:
        print("  ✓ Pred and GT motion variance match well")
        motion_issue = False
    else:
        print(f"  Pred/GT variance ratio: {pred_var_r / (gt_var_r + 1e-8):.4f}")
        motion_issue = abs(pred_var_r / (gt_var_r + 1e-8) - 1.0) > 0.2

    # ========================================================================
    # Final Verdict
    # ========================================================================
    print("\n" + "=" * 70)
    passed_normalized = disp_ratio > 0.5 and pck_01 > 50
    passed_pixel = pixel_mpjpe < 5.0  # Within 5 pixels = perfect overfit
    
    if passed_normalized and passed_pixel:
        print("🎉 4-Sample Overfit Test PASSED!")
        print(f"   Normalized space: MPJPE={mpjpe:.6f}, PCK@0.1={pck_01:.1f}%")
        print(f"   Pixel space: MPJPE={pixel_mpjpe:.2f} pixels")
        
        if motion_issue:
            print(f"\n   ⚠️ But hand motion range mismatch detected:")
            print(f"      Right hand variance ratio: {pred_var_r / (gt_var_r + 1e-8):.4f}")
            print(f"      Recommendations:")
            print(f"      1. Increase vel_weight to 5.0")
            print(f"      2. Add dedicated hand motion loss")
            print(f"      3. Check if normalization over-compresses hand motion")
        else:
            print(f"   ✓ Hand motion range also matches!")
    else:
        print("⚠️ Test FAILED")
        if not passed_normalized:
            print("   Normalized space metrics insufficient")
        if not passed_pixel:
            print(f"   Pixel space difference too large: {pixel_mpjpe:.2f} > 5.0 pixels")
    print("=" * 70)