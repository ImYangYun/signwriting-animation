"""
4-Sample Overfit Test for V1 Model (Original CAMDM-based)

This script tests whether the original V1 model can successfully overfit
to 4 samples. The goal is to determine if V1 had the same motion collapse
issues as V2, or if it worked fine all along.

Test procedure:
1. Train V1 model on exactly 4 samples
2. Test with different encoder architectures (trans_enc, trans_dec, gru)
3. Compare displacement ratios and visual quality

Critical questions to answer:
- Does trans_enc in V1 also produce static poses?
- Does trans_dec or gru in V1 avoid the problem?
- Was V2's frame-independent decoding necessary, or over-engineering?
"""

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
from signwriting_animation.diffusion.lightning_module_v1 import (
    LitDiffusionV1,
    sanitize_btjc,
    masked_dtw,
    mean_frame_disp,
)


def tensor_to_pose(t_btjc, header, ref_pose, gt_btjc=None, apply_scale=True):
    """Convert normalized tensor predictions back to pose format."""
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc
    
    t_np = t.detach().cpu().numpy().astype(np.float32)

    gt_np = None
    if gt_btjc is not None:
        if gt_btjc.dim() == 4:
            gt_np = gt_btjc[0].detach().cpu().numpy().astype(np.float32)
        else:
            gt_np = gt_btjc.detach().cpu().numpy().astype(np.float32)

    arr = t_np[:, None, :, :]
    conf = ref_pose.body.confidence[:len(t_np)].copy()
    fps = ref_pose.body.fps
    
    body = NumPyPoseBody(fps=fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)
    
    unshift_hands(pose_obj)
    print("  ✓ unshift succeeded")

    T_pred = pose_obj.body.data.shape[0]
    T_ref_total = ref_pose.body.data.shape[0]
    
    future_start = max(0, T_ref_total - T_pred)
    ref_arr = np.asarray(ref_pose.body.data[future_start:future_start+T_pred, 0], dtype=np.float32)
    pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)
    
    print(f"  [alignment] using ref frames {future_start}-{future_start+T_pred-1}")
    
    if apply_scale and gt_np is not None:
        def _var(a):
            center = a.mean(axis=1, keepdims=True)
            return float(((a - center) ** 2).mean())
        
        var_gt_norm = _var(gt_np)
        var_ref = _var(ref_arr)
        
        if var_gt_norm > 1e-8:
            scale = np.sqrt(var_ref / var_gt_norm)
            print(f"  [scale] var_ref={var_ref:.2f}, var_gt_norm={var_gt_norm:.6f}")
            print(f"  [scale] normalized→pixel scale={scale:.2f}")
            pose_obj.body.data *= scale
            pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)

    ref_c = ref_arr.reshape(-1, 3).mean(axis=0)
    pred_c = pred_arr.reshape(-1, 3).mean(axis=0)
    delta = ref_c - pred_c
    print(f"  [translate] delta={delta}")
    pose_obj.body.data += delta
    
    print(f"  [final] range=[{pose_obj.body.data.min():.2f}, {pose_obj.body.data.max():.2f}]")
    
    return pose_obj


if __name__ == "__main__":
    pl.seed_everything(42)

    # Paths
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"

    print("=" * 70)
    print("4-Sample Overfit Test - V1 (Original CAMDM-based)")
    print("=" * 70)

    # Configuration
    NUM_SAMPLES = 4
    MAX_EPOCHS = 500
    BATCH_SIZE = 4

    # Test different architectures
    ARCHITECTURES = [
        "trans_enc",  # Standard Transformer Encoder (may have averaging)
        # "trans_dec",  # Transformer Decoder with causal mask (may be better)
        # "gru",        # GRU (sequential, no averaging)
    ]

    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    print(f"Total dataset size: {len(base_ds)}")

    class SubsetDataset(torch.utils.data.Dataset):
        def __init__(self, base, indices):
            self.samples = [base[i] for i in indices]
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    train_ds = SubsetDataset(base_ds, list(range(NUM_SAMPLES)))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=zero_pad_collator)

    # Get dimensions
    sample = train_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    
    num_joints, num_dims, future_len = sample.shape[-2], sample.shape[-1], sample.shape[0]
    print(f"J={num_joints}, D={num_dims}, T_future={future_len}")

    # Test each architecture
    for arch in ARCHITECTURES:
        print("\n" + "=" * 70)
        print(f"Testing Architecture: {arch.upper()}")
        print("=" * 70)

        out_dir = f"logs/4sample_test_v1_{arch}"
        os.makedirs(out_dir, exist_ok=True)

        # Create V1 model with specific architecture
        model = LitDiffusionV1(
            num_keypoints=num_joints,
            num_dims=num_dims,
            stats_path=stats_path,
            lr=1e-3,
            diffusion_steps=8,
            vel_weight=1.0,
            arch=arch,  # KEY: Test different architectures
            num_layers=8,
            ff_size=1024,
        )

        # Training
        print(f"\nTraining with {arch}...")
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            enable_checkpointing=False,
            log_every_n_steps=50,
        )
        trainer.fit(model, train_loader)

        # Inference
        print("\n" + "=" * 70)
        print(f"Inference - {arch}")
        print("=" * 70)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        test_batch = zero_pad_collator([train_ds[0]])
        cond = test_batch["conditions"]
        past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt_raw = sanitize_btjc(test_batch["data"][:1]).to(device)

        with torch.no_grad():
            pred_raw = model.sample(past_raw, sign, future_len)
            
            # Evaluate in normalized space
            mse = F.mse_loss(pred_raw, gt_raw).item()
            disp_pred = mean_frame_disp(pred_raw)
            disp_gt = mean_frame_disp(gt_raw)
            disp_ratio = disp_pred / (disp_gt + 1e-8)
            
            # DTW
            mask = torch.ones(1, future_len, device=device)
            dtw = masked_dtw(pred_raw, gt_raw, mask)
            if isinstance(dtw, torch.Tensor):
                dtw = dtw.item()
            
            # MPJPE, PCK
            pred_np = pred_raw[0].cpu().numpy()
            gt_np = gt_raw[0].cpu().numpy()
            per_joint_err = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
            mpjpe = per_joint_err.mean()
            pck_01 = (per_joint_err < 0.1).mean() * 100
            pck_02 = (per_joint_err < 0.2).mean() * 100

        print(f"""
Results ({arch} - Normalized Space):
  MSE: {mse:.6f}
  MPJPE: {mpjpe:.6f}
  PCK@0.1: {pck_01:.1f}%
  PCK@0.2: {pck_02:.1f}%
  DTW: {dtw:.6f}
  Disp GT: {disp_gt:.6f}
  Disp Pred: {disp_pred:.6f}
  Disp Ratio: {disp_ratio:.4f}
""")

        # Critical diagnosis
        print("\n" + "=" * 70)
        print(f"CRITICAL DIAGNOSIS - {arch}")
        print("=" * 70)
        
        if disp_ratio < 0.3:
            print(f"  ❌ SEVERE MOTION COLLAPSE (disp_ratio={disp_ratio:.4f})")
            print(f"     → {arch} produces static poses")
            print(f"     → V1 has the same problem as V2!")
        elif disp_ratio < 0.7:
            print(f"  ⚠️  MILD MOTION REDUCTION (disp_ratio={disp_ratio:.4f})")
            print(f"     → {arch} reduces motion range")
            print(f"     → May still have averaging issues")
        else:
            print(f"  ✓ GOOD MOTION PRESERVATION (disp_ratio={disp_ratio:.4f})")
            print(f"     → {arch} preserves motion well")
            print(f"     → V1 with {arch} may not need V2's fix!")

        # Save pose files
        print("\n" + "=" * 70)
        print(f"Saving Pose Files - {arch}...")
        print("=" * 70)
        
        ref_path = base_ds.records[0]["pose"]
        if not os.path.isabs(ref_path):
            ref_path = os.path.join(data_dir, ref_path)
        
        with open(ref_path, "rb") as f:
            ref_pose = Pose.read(f)
        ref_pose = reduce_holistic(ref_pose)
        if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
            ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
        
        gt_pose = tensor_to_pose(gt_raw, ref_pose.header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
        with open(f"{out_dir}/gt.pose", "wb") as f:
            gt_pose.write(f)
        
        pred_pose = tensor_to_pose(pred_raw, ref_pose.header, ref_pose, gt_btjc=gt_raw, apply_scale=True)
        with open(f"{out_dir}/pred.pose", "wb") as f:
            pred_pose.write(f)
        
        print(f"\n✓ Saved to {out_dir}/")

        # Pixel space verification
        gt_data = gt_pose.body.data[:, 0, :, :]
        pred_data = pred_pose.body.data[:, 0, :, :]
        
        pixel_mpjpe = np.sqrt(((gt_data - pred_data) ** 2).sum(-1)).mean()
        print(f"\nPixel space MPJPE: {pixel_mpjpe:.2f} pixels")

    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print("\nPlease check the displacement ratios for each architecture:")
    print("- If trans_enc has low disp_ratio: V1 also has motion collapse")
    print("- If trans_dec/gru have high disp_ratio: V1 was fine, just used wrong arch")
    print("- If all have low disp_ratio: The problem is elsewhere (data/training)")
    print("\nNext steps:")
    print("1. Compare V1 vs V2 results")
    print("2. Determine if V2's frame-independent decoding was necessary")
    print("3. Decide whether to keep V2 or revert to V1 with better arch")
    print("=" * 70)