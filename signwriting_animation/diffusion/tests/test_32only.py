"""
Test: Does Past Dropout training help model learn sign? (32 samples)

Method: Use Past Dropout checkpoint, but inference with past=0 (Sign-Only)

If Sign-Only PCK ≈ Normal PCK: Model learned SignWriting well
If Sign-Only PCK << Normal PCK: Model still relies on past frames
"""
import os
import torch
import torch.nn as nn
import numpy as np

os.chdir("/home/yayun/data/signwriting-animation-fork")

from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset


def sanitize_btjc(x):
    if hasattr(x, "zero_filled"): x = x.zero_filled()
    if hasattr(x, "tensor"): x = x.tensor
    if x.dim() == 5: x = x[:, :, 0]
    if x.dim() == 3: x = x.unsqueeze(0)
    if x.dim() != 4: raise ValueError(f"Expected [B,T,J,C], got {tuple(x.shape)}")
    if x.shape[-1] != 3 and x.shape[-2] == 3: x = x.permute(0, 1, 3, 2)
    return x.contiguous().float()


def test_checkpoint(ckpt_path, num_samples=32):
    DATA_DIR = "/home/yayun/data/pose_data/"
    CSV_PATH = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    
    print("=" * 70)
    print(f"Testing: {ckpt_path}")
    print(f"Num samples: {num_samples}")
    print("=" * 70)
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found!")
        return None
    
    # Load model - import from the training script
    from test_32samples import LitDiffusionPastDropout
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    lit_model = LitDiffusionPastDropout.load_from_checkpoint(ckpt_path, map_location=device)
    lit_model.eval()
    lit_model = lit_model.to(device)
    print("Model loaded!")
    
    # Load samples
    base_ds = DynamicPosePredictionDataset(
        data_dir=DATA_DIR, csv_path=CSV_PATH,
        num_past_frames=40, num_future_frames=20, 
        with_metadata=True, split="train",
    )
    
    seen_poses, selected_indices = set(), []
    for idx in range(len(base_ds)):
        if len(selected_indices) >= num_samples: break
        pose = base_ds.records[idx].get("pose", "")
        if pose not in seen_poses:
            seen_poses.add(pose)
            selected_indices.append(idx)
    
    print(f"Testing on {len(selected_indices)} samples")
    
    # Cache samples
    samples = []
    for idx in selected_indices:
        sample = base_ds[idx]
        batch = zero_pad_collator([sample])
        
        past = batch["conditions"]["input_pose"]
        sign = batch["conditions"]["sign_image"]
        gt = batch["data"]
        
        if hasattr(past, "zero_filled"): past = past.zero_filled()
        if hasattr(past, "tensor"): past = past.tensor
        if hasattr(gt, "zero_filled"): gt = gt.zero_filled()
        if hasattr(gt, "tensor"): gt = gt.tensor
        
        past_t = past[0]
        gt_t = gt[0]
        if past_t.dim() == 4 and past_t.shape[1] == 1:
            past_t = past_t.squeeze(1)
        if gt_t.dim() == 4 and gt_t.shape[1] == 1:
            gt_t = gt_t.squeeze(1)
        
        samples.append({
            "idx": idx,
            "past": past_t,
            "sign": sign[0],
            "gt": gt_t,
        })
    
    future_len = samples[0]["gt"].shape[0]
    num_joints = samples[0]["gt"].shape[1]
    num_dims = samples[0]["gt"].shape[2]
    
    print(f"Data shape: T={future_len}, J={num_joints}, C={num_dims}")
    
    class W(nn.Module):
        def __init__(self, m, p, sg): 
            super().__init__()
            self.m, self.p, self.sg = m, p, sg
        def forward(self, x, t, **kw): 
            return self.m(x, t, self.p, self.sg)
    
    # ============================================================
    # Test 1: Normal inference (with past)
    # ============================================================
    print("\n" + "-" * 50)
    print("TEST 1: Normal inference (with past)")
    print("-" * 50)
    
    results_normal = []
    for s in samples[:10]:  # Test first 10 samples for speed
        gt = sanitize_btjc(s["gt"]).to(device)
        past = sanitize_btjc(s["past"]).to(device)
        sign = s["sign"].unsqueeze(0).float().to(device)
        
        past_norm = lit_model.normalize(past)
        past_bjct = lit_model.btjc_to_bjct(past_norm)
        _, J, C, _ = past_bjct.shape
        
        with torch.no_grad():
            wrapped = W(lit_model.model, past_bjct, sign)
            pred_bjct = lit_model.diffusion.p_sample_loop(
                wrapped, (1, J, C, future_len), 
                clip_denoised=True, model_kwargs={"y": {}}, progress=False
            )
            pred = lit_model.unnormalize(lit_model.bjct_to_btjc(pred_bjct))
        
        diff = (pred - gt).cpu().numpy()[0]
        pck = (np.sqrt((diff**2).sum(-1)) < 0.1).mean() * 100
        
        gt_disp = (gt[:, 1:] - gt[:, :-1]).abs().mean().item()
        pred_disp = (pred[:, 1:] - pred[:, :-1]).abs().mean().item()
        ratio = pred_disp / (gt_disp + 1e-8)
        
        print(f"  idx={s['idx']}: PCK={pck:.1f}%, ratio={ratio:.2f}")
        results_normal.append({"idx": s["idx"], "pck": pck, "ratio": ratio})
    
    avg_pck_normal = np.mean([r["pck"] for r in results_normal])
    avg_ratio_normal = np.mean([r["ratio"] for r in results_normal])
    print(f"  AVG: PCK={avg_pck_normal:.1f}%, ratio={avg_ratio_normal:.2f}")
    
    # ============================================================
    # Test 2: Sign-Only inference (past = 0)
    # ============================================================
    print("\n" + "-" * 50)
    print("TEST 2: Sign-Only inference (past = 0)")
    print("-" * 50)
    
    results_signonly = []
    for s in samples[:10]:  # Test first 10 samples
        gt = sanitize_btjc(s["gt"]).to(device)
        sign = s["sign"].unsqueeze(0).float().to(device)
        
        # KEY: past = zeros
        past_zero = torch.zeros(1, 40, num_joints, num_dims, device=device)
        past_norm = lit_model.normalize(past_zero)
        past_bjct = lit_model.btjc_to_bjct(past_norm)
        _, J, C, _ = past_bjct.shape
        
        with torch.no_grad():
            wrapped = W(lit_model.model, past_bjct, sign)
            pred_bjct = lit_model.diffusion.p_sample_loop(
                wrapped, (1, J, C, future_len), 
                clip_denoised=True, model_kwargs={"y": {}}, progress=False
            )
            pred = lit_model.unnormalize(lit_model.bjct_to_btjc(pred_bjct))
        
        diff = (pred - gt).cpu().numpy()[0]
        pck = (np.sqrt((diff**2).sum(-1)) < 0.1).mean() * 100
        
        gt_disp = (gt[:, 1:] - gt[:, :-1]).abs().mean().item()
        pred_disp = (pred[:, 1:] - pred[:, :-1]).abs().mean().item()
        ratio = pred_disp / (gt_disp + 1e-8)
        
        print(f"  idx={s['idx']}: PCK={pck:.1f}%, ratio={ratio:.2f}")
        results_signonly.append({"idx": s["idx"], "pck": pck, "ratio": ratio})
    
    avg_pck_signonly = np.mean([r["pck"] for r in results_signonly])
    avg_ratio_signonly = np.mean([r["ratio"] for r in results_signonly])
    print(f"  AVG: PCK={avg_pck_signonly:.1f}%, ratio={avg_ratio_signonly:.2f}")
    
    return {
        "normal_pck": avg_pck_normal,
        "normal_ratio": avg_ratio_normal,
        "signonly_pck": avg_pck_signonly,
        "signonly_ratio": avg_ratio_signonly,
    }


def main():
    print("\n" + "=" * 70)
    print("PAST DROPOUT SIGN-ONLY TEST (32 samples)")
    print("=" * 70)
    
    checkpoints = [
        ("p30", "logs/past_dropout_fixed_32sample_p30/checkpoints/last.ckpt"),
        ("p70", "logs/past_dropout_fixed_32sample_p70/checkpoints/last.ckpt"),
    ]
    
    results = {}
    for name, ckpt_path in checkpoints:
        print(f"\n\n{'#' * 70}")
        print(f"# {name.upper()}")
        print(f"{'#' * 70}")
        result = test_checkpoint(ckpt_path, num_samples=32)
        if result:
            results[name] = result
    
    # ============================================================
    # Final Summary
    # ============================================================
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Config':<10} {'Normal PCK':<15} {'Sign-Only PCK':<15} {'Δ PCK':<10} {'Ratio (N/S)':<15}")
    print("-" * 65)
    
    for name, r in results.items():
        delta = r["signonly_pck"] - r["normal_pck"]
        delta_str = f"{delta:+.1f}%"
        ratio_str = f"{r['normal_ratio']:.2f} / {r['signonly_ratio']:.2f}"
        print(f"{name:<10} {r['normal_pck']:<15.1f} {r['signonly_pck']:<15.1f} {delta_str:<10} {ratio_str:<15}")
    
    print("-" * 65)
    print(f"Baseline (pure Sign-Only, no Past Dropout): PCK≈10%")
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    for name, r in results.items():
        gap = r["normal_pck"] - r["signonly_pck"]
        print(f"\n{name}:")
        if gap < 5:
            print(f"  ✅ Sign-Only ≈ Normal (gap={gap:.1f}%)")
            print(f"     → Model learned SignWriting well, not relying on past")
        elif gap < 15:
            print(f"  ⚠️  Sign-Only slightly worse (gap={gap:.1f}%)")
            print(f"     → Model uses both past and SignWriting")
        else:
            print(f"  ❌ Sign-Only much worse (gap={gap:.1f}%)")
            print(f"     → Model still relies heavily on past frames")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()