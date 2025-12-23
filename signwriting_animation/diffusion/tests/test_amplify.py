"""
Test: Amplify sign embedding at inference time.

If sign_image has weak influence, maybe we can boost it by scaling
the sign embedding before it's used by the model.

This is a quick hack that doesn't require retraining.
"""
import os
import torch
import numpy as np

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import unshift_hands

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitDiffusion, sanitize_btjc


def tensor_to_pose(t_btjc, header, ref_pose):
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
    return pose_obj


def test_sign_amplification():
    """Test if amplifying sign embedding improves results."""
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    ckpt_path = "logs/full/checkpoints/last-v1.ckpt"
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = "logs/full/sign_amplify_test"
    
    CLIP_DENOISED = True
    DIFFUSION_STEPS = 8
    
    # Test with different amplification factors
    AMPLIFY_FACTORS = [1.0, 2.0, 5.0, 10.0]
    
    # Test samples
    TEST_INDICES = [3300, 2800, 900]  # Good samples from previous eval
    # ============================================================
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 70)
    print("SIGN EMBEDDING AMPLIFICATION TEST")
    print("=" * 70)
    
    # Load model
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
    print(f"Amplification factors: {AMPLIFY_FACTORS}")
    
    # Check model structure
    print("\n" + "=" * 70)
    print("MODEL STRUCTURE CHECK")
    print("=" * 70)
    print("Model components:")
    for name, module in lit_model.model.named_children():
        print(f"  {name}: {type(module).__name__}")
    
    # Verify required components exist
    required = ['sign_encoder', 'time_embed', 'context_encoder', 'xt_encoder', 'pos_embed', 'decoder']
    missing = [r for r in required if not hasattr(lit_model.model, r)]
    if missing:
        print(f"\n❌ Missing components: {missing}")
        print("Cannot proceed with amplification test.")
        print("Please check the model architecture.")
        return
    else:
        print("\n✅ All required components found!")
    
    # ============================================================
    # First: Check sign embedding similarity
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 1: CHECK SIGN EMBEDDING SIMILARITY")
    print("=" * 70)
    
    sign_embeddings = []
    for idx in TEST_INDICES:
        batch = zero_pad_collator([test_ds[idx]])
        sign = batch["conditions"]["sign_image"][:1].float().to(device)
        
        with torch.no_grad():
            sign_emb = lit_model.model.sign_encoder(sign)  # [B, D]
        sign_embeddings.append(sign_emb)
        print(f"  idx={idx}: sign_emb norm = {sign_emb.norm().item():.4f}")
    
    # Compute pairwise cosine similarity
    print("\n  Pairwise cosine similarity:")
    for i in range(len(sign_embeddings)):
        for j in range(i+1, len(sign_embeddings)):
            cos_sim = torch.nn.functional.cosine_similarity(
                sign_embeddings[i], sign_embeddings[j]
            ).item()
            print(f"    idx {TEST_INDICES[i]} vs {TEST_INDICES[j]}: {cos_sim:.4f}")
    
    # ============================================================
    # Custom forward with amplified sign embedding
    # ============================================================
    class AmplifiedSignModel(torch.nn.Module):
        """Wrapper that amplifies sign embedding."""
        def __init__(self, base_model, amplify_factor):
            super().__init__()
            self.base_model = base_model
            self.amplify_factor = amplify_factor
        
        def forward(self, x_noisy, timestep, past, sign_img):
            # Get original embeddings
            sign_emb = self.base_model.sign_encoder(sign_img)
            
            # AMPLIFY sign embedding
            sign_emb = sign_emb * self.amplify_factor
            
            # Continue with rest of forward pass
            # (Need to replicate the model's forward logic with modified sign_emb)
            t_emb = self.base_model.time_embed(timestep)
            context = self.base_model.context_encoder(past)
            
            B, J, C, T = x_noisy.shape
            x_flat = x_noisy.permute(0, 3, 1, 2).reshape(B * T, J * C)
            xt_emb = self.base_model.xt_encoder(x_flat)
            
            pos_idx = torch.arange(T, device=x_noisy.device).unsqueeze(0).expand(B, -1)
            pos_emb = self.base_model.pos_embed(pos_idx).reshape(B * T, -1)
            
            context_exp = context.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            t_emb_exp = t_emb.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            sign_emb_exp = sign_emb.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            
            combined = torch.cat([context_exp, xt_emb, t_emb_exp, sign_emb_exp, pos_emb], dim=-1)
            out = self.base_model.decoder(combined)
            out = out.reshape(B, T, J, C).permute(0, 2, 3, 1)
            
            return out
    
    # ============================================================
    # Test with different amplification factors
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2: TEST AMPLIFICATION FACTORS")
    print("=" * 70)
    
    results = []
    
    for idx in TEST_INDICES:
        print(f"\n--- Sample idx={idx} ---")
        
        batch = zero_pad_collator([test_ds[idx]])
        cond = batch["conditions"]
        
        past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
        sign = cond["sign_image"][:1].float().to(device)
        gt_raw = sanitize_btjc(batch["data"][:1]).to(device)
        
        past_norm = lit_model.normalize(past_raw)
        gt_norm = lit_model.normalize(gt_raw)
        past_bjct = lit_model.btjc_to_bjct(past_norm)
        
        gt_np = gt_raw[0].cpu().numpy()
        gt_disp = np.sqrt(np.sum(np.diff(gt_np, axis=0)**2, axis=-1)).mean()
        
        # Get reference pose for saving
        ref_path = test_ds.records[idx]["pose"]
        if not os.path.isabs(ref_path):
            ref_path = os.path.join(data_dir, ref_path)
        with open(ref_path, "rb") as f:
            ref_pose = Pose.read(f)
        ref_pose = reduce_holistic(ref_pose)
        if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
            ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
        
        for amp_factor in AMPLIFY_FACTORS:
            torch.manual_seed(42)  # Same noise for fair comparison
            
            # Create amplified model
            amp_model = AmplifiedSignModel(lit_model.model, amp_factor)
            
            with torch.no_grad():
                B, J, C, _ = past_bjct.shape
                target_shape = (B, J, C, future_len)
                
                class _Wrapper(torch.nn.Module):
                    def __init__(self, model, past, sign):
                        super().__init__()
                        self.model, self.past, self.sign = model, past, sign
                    def forward(self, x, t, **kwargs):
                        return self.model(x, t, self.past, self.sign)
                
                wrapped = _Wrapper(amp_model, past_bjct, sign)
                
                pred_bjct = lit_model.diffusion.p_sample_loop(
                    model=wrapped,
                    shape=target_shape,
                    clip_denoised=CLIP_DENOISED,
                    model_kwargs={"y": {}},
                    progress=False,
                )
                pred_btjc = lit_model.bjct_to_btjc(pred_bjct)
            
            pred_unnorm = lit_model.unnormalize(pred_btjc)
            pred_np = pred_unnorm[0].cpu().numpy()
            
            pred_disp = np.sqrt(np.sum(np.diff(pred_np, axis=0)**2, axis=-1)).mean()
            ratio = pred_disp / (gt_disp + 1e-8)
            
            gt_unnorm = lit_model.unnormalize(gt_norm)
            gt_np_eval = gt_unnorm[0].cpu().numpy()
            per_joint_err = np.sqrt(((pred_np - gt_np_eval) ** 2).sum(-1))
            pck = (per_joint_err < 0.1).mean() * 100
            
            print(f"  amp={amp_factor:.1f}x: ratio={ratio:.3f}, PCK={pck:.1f}%")
            
            results.append({
                'idx': idx,
                'amp_factor': amp_factor,
                'ratio': ratio,
                'pck': pck,
                'pred_disp': pred_disp,
                'gt_disp': gt_disp,
            })
            
            # Save pose files
            pred_pose = tensor_to_pose(pred_unnorm, ref_pose.header, ref_pose)
            filename = f"{out_dir}/idx{idx}_amp{amp_factor:.0f}x_ratio{ratio:.2f}_pred.pose"
            with open(filename, "wb") as f:
                pred_pose.write(f)
        
        # Also save GT
        gt_pose = tensor_to_pose(gt_unnorm, ref_pose.header, ref_pose)
        with open(f"{out_dir}/idx{idx}_gt.pose", "wb") as f:
            gt_pose.write(f)
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY BY AMPLIFICATION FACTOR")
    print("=" * 70)
    
    for amp in AMPLIFY_FACTORS:
        amp_results = [r for r in results if r['amp_factor'] == amp]
        avg_ratio = np.mean([r['ratio'] for r in amp_results])
        avg_pck = np.mean([r['pck'] for r in amp_results])
        print(f"  amp={amp:.1f}x: avg_ratio={avg_ratio:.3f}, avg_PCK={avg_pck:.1f}%")
    
    print(f"\nPose files saved to: {out_dir}/")
    print("=" * 70)
    print("✅ Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    os.chdir("/home/yayun/data/signwriting-animation-fork")
    test_sign_amplification()