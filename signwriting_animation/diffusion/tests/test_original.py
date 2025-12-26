"""
Test Original Unfrozen CLIP Model (NO past dropout)

This is the model that showed 44% sign influence in test_sign_influence.py
We want to see actual generated poses from this model.

Checkpoint: logs/unfrozen_clip_contrastive_diffvideos_32sample/checkpoints/last.ckpt
"""
import os
import torch
import torch.nn as nn
import numpy as np

os.chdir("/home/yayun/data/signwriting-animation-fork")

# Import from the original training script
from test_clip import (
    LitDiffusionUnfrozenCLIP,
    DynamicPosePredictionDataset,
    sanitize_btjc,
)
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import unshift_hands


# ============================================================
# Configuration
# ============================================================
# Original unfrozen CLIP model (NO past dropout, 44% sign influence)
CHECKPOINT = "logs/unfrozen_clip_contrastive_diffvideos_32sample/checkpoints/last.ckpt"

# Alternative: try full dataset unfrozen CLIP if above doesn't exist
CHECKPOINT_ALT = "logs/full_unfrozen_clip/checkpoints/last.ckpt"

DATA_DIR = '/home/yayun/data/pose_data/'
CSV_PATH = '/home/yayun/data/signwriting-animation/data_fixed.csv'
OUT_DIR = 'logs/unfrozen_clip_test_poses'
# ============================================================


def tensor_to_pose(t_btjc: torch.Tensor, 
                   header, 
                   ref_pose: Pose, 
                   scale_to_ref: bool = True) -> Pose:
    """Convert tensor to Pose - CORRECT VERSION."""
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
        ref_arr = np.asarray(
            ref_pose.body.data[future_start:future_start+T_pred, 0], 
            dtype=np.float32
        )
        
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


def inference(lit_model, past, sign, future_len=20):
    """Run DDPM inference."""
    past_norm = lit_model.normalize(past)
    past_bjct = lit_model.btjc_to_bjct(past_norm)
    B, J, C, _ = past_bjct.shape
    
    class Wrapper(nn.Module):
        def __init__(self, m, p, s):
            super().__init__()
            self.m, self.p, self.s = m, p, s
        def forward(self, x, t, **kw):
            return self.m(x, t, self.p, self.s)
    
    wrapped = Wrapper(lit_model.model, past_bjct, sign)
    pred_bjct = lit_model.diffusion.p_sample_loop(
        wrapped, (B, J, C, future_len), clip_denoised=True,
        model_kwargs={'y': {}}, progress=False
    )
    pred_norm = lit_model.bjct_to_btjc(pred_bjct)
    return lit_model.unnormalize(pred_norm)


def compute_metrics(pred, gt):
    """Compute metrics."""
    gt_disp = (gt[:, 1:] - gt[:, :-1]).abs().mean().item()
    pred_disp = (pred[:, 1:] - pred[:, :-1]).abs().mean().item()
    ratio = pred_disp / (gt_disp + 1e-8)
    
    diff = (pred - gt).cpu().numpy()[0]
    per_joint_err = np.sqrt((diff ** 2).sum(-1))
    mpjpe = per_joint_err.mean()
    pck = (per_joint_err < 0.1).mean() * 100
    
    gt_vel = gt[:, 1:] - gt[:, :-1]
    pred_vel = pred[:, 1:] - pred[:, :-1]
    if gt_vel.shape[1] > 1:
        gt_acc = (gt_vel[:, 1:] - gt_vel[:, :-1]).abs().mean().item()
        pred_acc = (pred_vel[:, 1:] - pred_vel[:, :-1]).abs().mean().item()
        jitter = pred_acc / (gt_acc + 1e-8)
    else:
        jitter = 1.0
    
    return {'ratio': ratio, 'mpjpe': mpjpe, 'pck': pck, 'jitter': jitter}


def main():
    print("=" * 70)
    print("TEST: Original Unfrozen CLIP (NO Past Dropout)")
    print("This model showed 44% sign influence")
    print("=" * 70)
    
    # Find checkpoint
    ckpt_path = CHECKPOINT
    if not os.path.exists(ckpt_path):
        print(f"Primary checkpoint not found: {ckpt_path}")
        ckpt_path = CHECKPOINT_ALT
        if not os.path.exists(ckpt_path):
            print(f"Alternative checkpoint not found: {ckpt_path}")
            # List available checkpoints
            print("\nAvailable logs:")
            for d in os.listdir("logs"):
                if "unfrozen" in d.lower() or "clip" in d.lower():
                    print(f"  {d}")
            return
    
    print(f"\nCheckpoint: {ckpt_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    lit_model = LitDiffusionUnfrozenCLIP.load_from_checkpoint(ckpt_path)
    lit_model.eval()
    lit_model = lit_model.to(device)
    
    # Load dataset
    print("Loading dataset...")
    ds = DynamicPosePredictionDataset(
        data_dir=DATA_DIR, csv_path=CSV_PATH,
        num_past_frames=40, num_future_frames=20,
        with_metadata=True, split='train',
    )
    print(f"Dataset size: {len(ds)}")
    
    # Get training indices (same logic as original overfit)
    seen_poses, train_indices = set(), []
    for idx in range(len(ds)):
        if len(train_indices) >= 32: break
        pose = ds.records[idx].get("pose", "")
        if pose not in seen_poses:
            seen_poses.add(pose)
            train_indices.append(idx)
    
    print(f"Training indices: {train_indices[:10]}...")
    
    # Test
    print(f"\n{'='*70}")
    print("INFERENCE TEST")
    print("=" * 70)
    print(f"{'idx':>5} | {'Ratio':>8} | {'PCK@0.1':>8} | {'Jitter':>8} | {'MPJPE':>8}")
    print("-" * 70)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    
    for i, idx in enumerate(train_indices[:5]):
        batch = zero_pad_collator([ds[idx]])
        past = sanitize_btjc(batch['conditions']['input_pose'][:1]).to(device)
        sign = batch['conditions']['sign_image'][:1].float().to(device)
        gt = sanitize_btjc(batch['data'][:1]).to(device)
        
        with torch.no_grad():
            pred = inference(lit_model, past, sign)
        
        metrics = compute_metrics(pred, gt)
        
        print(f"{idx:5d} | {metrics['ratio']:8.2f} | {metrics['pck']:7.1f}% | {metrics['jitter']:8.2f} | {metrics['mpjpe']:8.4f}")
        results.append({'idx': idx, **metrics})
        
        # Save pose files for each sample
        ref_path = ds.records[idx]['pose']
        if not ref_path.startswith('/'): ref_path = DATA_DIR + ref_path
        with open(ref_path, 'rb') as f:
            ref_pose = Pose.read(f)
        ref_pose = reduce_holistic(ref_pose)
        if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
            ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
        
        gt_pose = tensor_to_pose(gt, ref_pose.header, ref_pose)
        pred_pose = tensor_to_pose(pred, ref_pose.header, ref_pose)
        
        with open(f'{OUT_DIR}/sample{i}_idx{idx}_gt.pose', 'wb') as f:
            gt_pose.write(f)
        with open(f'{OUT_DIR}/sample{i}_idx{idx}_pred.pose', 'wb') as f:
            pred_pose.write(f)
    
    # Summary
    avg_ratio = np.mean([r['ratio'] for r in results])
    avg_pck = np.mean([r['pck'] for r in results])
    avg_jitter = np.mean([r['jitter'] for r in results])
    avg_mpjpe = np.mean([r['mpjpe'] for r in results])
    
    print("-" * 70)
    print(f"{'AVG':>5} | {avg_ratio:8.2f} | {avg_pck:7.1f}% | {avg_jitter:8.2f} | {avg_mpjpe:8.4f}")
    
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print("=" * 70)
    
    if 0.8 <= avg_ratio <= 1.2:
        print(f"✅ Ratio ({avg_ratio:.2f}) is GOOD")
    else:
        print(f"⚠️  Ratio ({avg_ratio:.2f}) is outside ideal range")
    
    if avg_pck > 50:
        print(f"✅ PCK ({avg_pck:.1f}%) is GOOD")
    elif avg_pck > 30:
        print(f"⚠️  PCK ({avg_pck:.1f}%) is MODERATE")
    else:
        print(f"❌ PCK ({avg_pck:.1f}%) is LOW")
    
    if avg_jitter < 1.5:
        print(f"✅ Jitter ({avg_jitter:.2f}) is GOOD")
    else:
        print(f"⚠️  Jitter ({avg_jitter:.2f}) is HIGH")
    
    print(f"\n{'='*70}")
    print(f"POSE FILES SAVED: {OUT_DIR}/")
    print("=" * 70)
    print("Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith('.pose'):
            print(f"  {f}")
    
    print(f"\n{'='*70}")
    print("TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()