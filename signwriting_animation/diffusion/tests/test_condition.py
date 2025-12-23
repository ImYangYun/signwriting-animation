"""
Test: Does the model actually use sign_image and past_motion as conditions?

Hypothesis:
- If sign_image matters: different sign images → different outputs
- If past_motion matters: different past motions → different outputs
- If conditions are ignored: outputs will be similar regardless of conditions

This test swaps conditions between samples and measures output differences.
"""
import os
import torch
import numpy as np

from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitDiffusion, sanitize_btjc


def compute_output_difference(pred1, pred2):
    """Compute normalized difference between two predictions."""
    diff = (pred1 - pred2).abs().mean().item()
    magnitude = (pred1.abs().mean().item() + pred2.abs().mean().item()) / 2
    return diff, diff / (magnitude + 1e-8)  # absolute diff, relative diff


def test_condition_influence():
    """Test whether sign_image and past_motion actually influence the output."""
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    ckpt_path = "logs/full/checkpoints/last-v1.ckpt"
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    
    CLIP_DENOISED = True
    DIFFUSION_STEPS = 8
    
    # Use samples with different characteristics
    TEST_INDICES = [100, 500, 1000, 2000, 3000]
    # ============================================================
    
    print("=" * 70)
    print("CONDITION INFLUENCE TEST")
    print("=" * 70)
    print("Testing whether sign_image and past_motion affect model output")
    print()
    
    # Load model
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
    
    sample = test_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    
    num_joints = sample.shape[-2]
    num_dims = sample.shape[-1]
    future_len = sample.shape[0]
    
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
    
    print(f"Model loaded on: {device}")
    print(f"Test indices: {TEST_INDICES}")
    
    # ============================================================
    # Load all test samples
    # ============================================================
    print("\n" + "=" * 70)
    print("LOADING TEST SAMPLES")
    print("=" * 70)
    
    samples = {}
    for idx in TEST_INDICES:
        batch = zero_pad_collator([test_ds[idx]])
        cond = batch["conditions"]
        
        past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt_raw = sanitize_btjc(batch["data"][:1]).to(device)
        
        past_norm = lit_model.normalize(past_raw)
        past_bjct = lit_model.btjc_to_bjct(past_norm)
        
        samples[idx] = {
            'past_bjct': past_bjct,
            'sign': sign,
            'gt': gt_raw,
        }
        print(f"  Loaded idx={idx}")
    
    # ============================================================
    # Helper: Generate prediction
    # ============================================================
    def generate(past_bjct, sign_img):
        """Generate prediction with given conditions."""
        torch.manual_seed(42)  # Same noise for fair comparison
        
        with torch.no_grad():
            B, J, C, _ = past_bjct.shape
            target_shape = (B, J, C, future_len)
            
            class _Wrapper(torch.nn.Module):
                def __init__(self, model, past, sign):
                    super().__init__()
                    self.model, self.past, self.sign = model, past, sign
                def forward(self, x, t, **kwargs):
                    return self.model(x, t, self.past, self.sign)
            
            wrapped = _Wrapper(lit_model.model, past_bjct, sign_img)
            
            pred_bjct = lit_model.diffusion.p_sample_loop(
                model=wrapped,
                shape=target_shape,
                clip_denoised=CLIP_DENOISED,
                model_kwargs={"y": {}},
                progress=False,
            )
            pred_btjc = lit_model.bjct_to_btjc(pred_bjct)
        
        return lit_model.unnormalize(pred_btjc)
    
    # ============================================================
    # TEST 1: Does SIGN IMAGE matter?
    # Fix past_motion from sample A, vary sign_image from A vs B
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 1: SIGN IMAGE INFLUENCE")
    print("=" * 70)
    print("Fix past_motion, vary sign_image")
    print()
    
    sign_diffs = []
    
    for i, idx_a in enumerate(TEST_INDICES[:-1]):
        idx_b = TEST_INDICES[i + 1]
        
        past_a = samples[idx_a]['past_bjct']
        sign_a = samples[idx_a]['sign']
        sign_b = samples[idx_b]['sign']
        
        # Same past, different sign
        pred_aa = generate(past_a, sign_a)  # past_a + sign_a
        pred_ab = generate(past_a, sign_b)  # past_a + sign_b
        
        abs_diff, rel_diff = compute_output_difference(pred_aa, pred_ab)
        sign_diffs.append(rel_diff)
        
        print(f"  past={idx_a}, sign={idx_a} vs sign={idx_b}")
        print(f"    Absolute diff: {abs_diff:.6f}")
        print(f"    Relative diff: {rel_diff*100:.2f}%")
    
    avg_sign_influence = np.mean(sign_diffs) * 100
    print(f"\n  >>> Average sign_image influence: {avg_sign_influence:.2f}%")
    
    # ============================================================
    # TEST 2: Does PAST MOTION matter?
    # Fix sign_image from sample A, vary past_motion from A vs B
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 2: PAST MOTION INFLUENCE")
    print("=" * 70)
    print("Fix sign_image, vary past_motion")
    print()
    
    past_diffs = []
    
    for i, idx_a in enumerate(TEST_INDICES[:-1]):
        idx_b = TEST_INDICES[i + 1]
        
        past_a = samples[idx_a]['past_bjct']
        past_b = samples[idx_b]['past_bjct']
        sign_a = samples[idx_a]['sign']
        
        # Same sign, different past
        pred_aa = generate(past_a, sign_a)  # past_a + sign_a
        pred_ba = generate(past_b, sign_a)  # past_b + sign_a
        
        abs_diff, rel_diff = compute_output_difference(pred_aa, pred_ba)
        past_diffs.append(rel_diff)
        
        print(f"  sign={idx_a}, past={idx_a} vs past={idx_b}")
        print(f"    Absolute diff: {abs_diff:.6f}")
        print(f"    Relative diff: {rel_diff*100:.2f}%")
    
    avg_past_influence = np.mean(past_diffs) * 100
    print(f"\n  >>> Average past_motion influence: {avg_past_influence:.2f}%")
    
    # ============================================================
    # TEST 3: Baseline - Same conditions, same output?
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 3: REPRODUCIBILITY CHECK")
    print("=" * 70)
    print("Same conditions, same seed → should be identical")
    print()
    
    idx = TEST_INDICES[0]
    pred1 = generate(samples[idx]['past_bjct'], samples[idx]['sign'])
    pred2 = generate(samples[idx]['past_bjct'], samples[idx]['sign'])
    
    abs_diff, rel_diff = compute_output_difference(pred1, pred2)
    print(f"  Same conditions twice:")
    print(f"    Absolute diff: {abs_diff:.6f}")
    print(f"    Relative diff: {rel_diff*100:.4f}%")
    
    # ============================================================
    # TEST 4: Different random seed
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 4: RANDOM SEED INFLUENCE")
    print("=" * 70)
    print("Same conditions, different seed → how much variation?")
    print()
    
    seed_diffs = []
    for idx in TEST_INDICES[:3]:
        torch.manual_seed(42)
        pred_seed1 = generate(samples[idx]['past_bjct'], samples[idx]['sign'])
        
        torch.manual_seed(123)
        pred_seed2 = generate(samples[idx]['past_bjct'], samples[idx]['sign'])
        
        abs_diff, rel_diff = compute_output_difference(pred_seed1, pred_seed2)
        seed_diffs.append(rel_diff)
        print(f"  idx={idx}: relative diff = {rel_diff*100:.2f}%")
    
    avg_seed_influence = np.mean(seed_diffs) * 100
    print(f"\n  >>> Average seed influence: {avg_seed_influence:.2f}%")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Sign image influence:   {avg_sign_influence:.2f}%
    Past motion influence:  {avg_past_influence:.2f}%
    Random seed influence:  {avg_seed_influence:.2f}%
    
    INTERPRETATION:
    - If sign/past influence << seed influence: conditions are IGNORED
    - If sign/past influence >> seed influence: conditions are USED
    - If sign/past influence ≈ seed influence: weak conditioning
    """)
    
    # Diagnosis
    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    if avg_sign_influence < 5 and avg_past_influence < 5:
        print("❌ Model is likely IGNORING both conditions!")
        print("   → Output is mainly determined by diffusion noise")
    elif avg_sign_influence < avg_seed_influence * 0.5:
        print("⚠️  Sign image has WEAK influence")
        print("   → CLIP embedding may not be effectively used")
    elif avg_past_influence < avg_seed_influence * 0.5:
        print("⚠️  Past motion has WEAK influence")
        print("   → Context encoder may not be effectively used")
    else:
        print("✅ Both conditions appear to influence output")
        print("   → Problem may be elsewhere (capacity, training, etc.)")
    
    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    test_condition_influence()