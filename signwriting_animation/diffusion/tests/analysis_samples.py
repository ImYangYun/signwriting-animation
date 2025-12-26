"""
分析高 PCK vs 低 PCK 样本的差异

目标：
1. SignWriting 图片特征（复杂度、大小）
2. CLIP embedding 相似度
3. Pose 动作幅度
4. 保存图片便于可视化
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

os.chdir("/home/yayun/data/signwriting-animation-fork")

from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from transformers import CLIPModel, CLIPProcessor


def analyze_samples():
    DATA_DIR = "/home/yayun/data/pose_data/"
    CSV_PATH = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    OUT_DIR = "logs/sample_analysis"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("SAMPLE ANALYSIS: Why some samples have low PCK?")
    print("=" * 70)
    
    # Load dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=DATA_DIR, csv_path=CSV_PATH,
        num_past_frames=40, num_future_frames=20, 
        with_metadata=True, split="train",
    )
    
    # Select 32 unique samples (same as training)
    seen_poses, selected_indices = set(), []
    for idx in range(len(base_ds)):
        if len(selected_indices) >= 32: break
        pose = base_ds.records[idx].get("pose", "")
        if pose not in seen_poses:
            seen_poses.add(pose)
            selected_indices.append(idx)
    
    print(f"Analyzing {len(selected_indices)} samples")
    
    # Load CLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    clip_model.eval()
    
    # Collect data
    samples_data = []
    
    for i, idx in enumerate(selected_indices):
        sample = base_ds[idx]
        batch = zero_pad_collator([sample])
        
        sign_img = batch["conditions"]["sign_image"]
        gt = batch["data"]
        
        if hasattr(gt, "zero_filled"): gt = gt.zero_filled()
        if hasattr(gt, "tensor"): gt = gt.tensor
        
        gt_t = gt[0]
        if gt_t.dim() == 4 and gt_t.shape[1] == 1:
            gt_t = gt_t.squeeze(1)
        
        # Get CLIP embedding
        with torch.no_grad():
            sign_tensor = sign_img.float().to(device)
            clip_emb = clip_model.get_image_features(pixel_values=sign_tensor)
            clip_emb = F.normalize(clip_emb, p=2, dim=-1)
        
        # Calculate motion magnitude
        if gt_t.dim() == 3:  # [T, J, C]
            displacement = (gt_t[1:] - gt_t[:-1]).abs().mean().item()
            max_displacement = (gt_t[1:] - gt_t[:-1]).abs().max().item()
        else:
            displacement = 0
            max_displacement = 0
        
        # SignWriting image stats
        sign_np = sign_img[0].permute(1, 2, 0).numpy()  # [H, W, C]
        sign_np = ((sign_np + 1) / 2 * 255).astype(np.uint8)  # Denormalize
        
        # Count non-white pixels (complexity proxy)
        if sign_np.shape[2] == 3:
            gray = sign_np.mean(axis=2)
        else:
            gray = sign_np[:, :, 0]
        non_white = (gray < 250).sum()
        
        record = base_ds.records[idx]
        
        samples_data.append({
            "idx": idx,
            "list_idx": i,
            "clip_emb": clip_emb.cpu(),
            "displacement": displacement,
            "max_displacement": max_displacement,
            "non_white_pixels": non_white,
            "sign_image": sign_np,
            "pose_file": record.get("pose", ""),
            "fsw": record.get("fsw", ""),
        })
        
        # Save SignWriting image
        img = Image.fromarray(sign_np)
        img.save(f"{OUT_DIR}/sign_idx{idx}.png")
    
    print(f"Saved SignWriting images to {OUT_DIR}/")
    
    # ============================================================
    # Analysis 1: Motion magnitude
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Motion Magnitude")
    print("=" * 70)
    
    sorted_by_motion = sorted(samples_data, key=lambda x: x["displacement"])
    
    print(f"\n{'idx':<8} {'Displacement':<15} {'Max Disp':<12} {'NonWhite':<12} {'Pose File':<30}")
    print("-" * 80)
    
    for s in sorted_by_motion[:10]:
        pose_short = s["pose_file"][-30:] if len(s["pose_file"]) > 30 else s["pose_file"]
        print(f"{s['idx']:<8} {s['displacement']:<15.4f} {s['max_displacement']:<12.4f} {s['non_white_pixels']:<12} ...{pose_short}")
    
    print("\n... (lowest motion samples shown)")
    
    # ============================================================
    # Analysis 2: CLIP embedding similarity
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: CLIP Embedding Similarity")
    print("=" * 70)
    
    # Stack all embeddings
    all_embs = torch.cat([s["clip_emb"] for s in samples_data], dim=0)
    
    # Compute pairwise cosine similarity
    sim_matrix = torch.mm(all_embs, all_embs.t()).numpy()
    
    # Find most similar pairs (excluding self)
    n = len(samples_data)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append({
                "i": samples_data[i]["idx"],
                "j": samples_data[j]["idx"],
                "sim": sim_matrix[i, j],
            })
    
    pairs_sorted = sorted(pairs, key=lambda x: x["sim"], reverse=True)
    
    print("\nMost SIMILAR pairs (potential confusion):")
    print(f"{'idx_i':<8} {'idx_j':<8} {'Similarity':<12}")
    print("-" * 30)
    for p in pairs_sorted[:10]:
        print(f"{p['i']:<8} {p['j']:<8} {p['sim']:<12.4f}")
    
    print("\nMost DIFFERENT pairs:")
    print(f"{'idx_i':<8} {'idx_j':<8} {'Similarity':<12}")
    print("-" * 30)
    for p in pairs_sorted[-10:]:
        print(f"{p['i']:<8} {p['j']:<8} {p['sim']:<12.4f}")
    
    # Average similarity
    avg_sim = np.mean([p["sim"] for p in pairs])
    max_sim = max([p["sim"] for p in pairs])
    min_sim = min([p["sim"] for p in pairs])
    
    print(f"\nSimilarity stats: avg={avg_sim:.4f}, max={max_sim:.4f}, min={min_sim:.4f}")
    
    # ============================================================
    # Analysis 3: Known problematic samples
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Known Problematic Samples (low PCK)")
    print("=" * 70)
    
    # From previous experiments
    low_pck_indices = [34, 264]
    high_pck_indices = [0, 321, 336]
    
    print("\nLow PCK samples:")
    for target_idx in low_pck_indices:
        s = next((x for x in samples_data if x["idx"] == target_idx), None)
        if s:
            print(f"\n  idx={target_idx}:")
            print(f"    Displacement: {s['displacement']:.4f}")
            print(f"    Max displacement: {s['max_displacement']:.4f}")
            print(f"    Non-white pixels: {s['non_white_pixels']}")
            print(f"    FSW: {s['fsw'][:50] if s['fsw'] else 'N/A'}...")
            
            # Find most similar other sample
            s_idx = s["list_idx"]
            sims = sim_matrix[s_idx].copy()
            sims[s_idx] = -1  # Exclude self
            most_similar_idx = np.argmax(sims)
            most_similar_sample = samples_data[most_similar_idx]
            print(f"    Most similar to: idx={most_similar_sample['idx']} (sim={sims[most_similar_idx]:.4f})")
    
    print("\nHigh PCK samples:")
    for target_idx in high_pck_indices:
        s = next((x for x in samples_data if x["idx"] == target_idx), None)
        if s:
            print(f"\n  idx={target_idx}:")
            print(f"    Displacement: {s['displacement']:.4f}")
            print(f"    Max displacement: {s['max_displacement']:.4f}")
            print(f"    Non-white pixels: {s['non_white_pixels']}")
            print(f"    FSW: {s['fsw'][:50] if s['fsw'] else 'N/A'}...")
    
    # ============================================================
    # Analysis 4: Correlation between features and (hypothesized) difficulty
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Feature Correlations")
    print("=" * 70)
    
    # Group by motion magnitude
    low_motion = [s for s in samples_data if s["displacement"] < np.median([x["displacement"] for x in samples_data])]
    high_motion = [s for s in samples_data if s["displacement"] >= np.median([x["displacement"] for x in samples_data])]
    
    print(f"\nLow motion samples (displacement < median):")
    print(f"  Count: {len(low_motion)}")
    print(f"  Indices: {[s['idx'] for s in low_motion]}")
    
    print(f"\nHigh motion samples (displacement >= median):")
    print(f"  Count: {len(high_motion)}")
    print(f"  Indices: {[s['idx'] for s in high_motion]}")
    
    # ============================================================
    # Save similarity matrix visualization
    # ============================================================
    plt.figure(figsize=(12, 10))
    plt.imshow(sim_matrix, cmap='RdYlBu_r', vmin=0.5, vmax=1.0)
    plt.colorbar(label='Cosine Similarity')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.title('CLIP Embedding Similarity Matrix')
    
    # Add sample idx labels
    idx_labels = [s["idx"] for s in samples_data]
    plt.xticks(range(len(idx_labels)), idx_labels, rotation=90, fontsize=6)
    plt.yticks(range(len(idx_labels)), idx_labels, fontsize=6)
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/clip_similarity_matrix.png", dpi=150)
    print(f"\nSaved similarity matrix to {OUT_DIR}/clip_similarity_matrix.png")
    
    # ============================================================
    # Create comparison figure for high vs low PCK
    # ============================================================
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Top row: low PCK samples
    low_samples = [s for s in samples_data if s["idx"] in low_pck_indices]
    for i, s in enumerate(low_samples[:5]):
        if i < 5:
            axes[0, i].imshow(s["sign_image"])
            axes[0, i].set_title(f"LOW PCK\nidx={s['idx']}\ndisp={s['displacement']:.3f}")
            axes[0, i].axis('off')
    
    # Fill remaining with high PCK
    high_samples = [s for s in samples_data if s["idx"] in high_pck_indices]
    for i, s in enumerate(high_samples[:5]):
        if i < 5:
            axes[1, i].imshow(s["sign_image"])
            axes[1, i].set_title(f"HIGH PCK\nidx={s['idx']}\ndisp={s['displacement']:.3f}")
            axes[1, i].axis('off')
    
    # Hide unused subplots
    for i in range(len(low_samples), 5):
        axes[0, i].axis('off')
    for i in range(len(high_samples), 5):
        axes[1, i].axis('off')
    
    plt.suptitle("Low PCK vs High PCK Samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/high_vs_low_pck.png", dpi=150)
    print(f"Saved comparison figure to {OUT_DIR}/high_vs_low_pck.png")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
可能导致低 PCK 的因素：
    
1. 【动作幅度小】
   - 某些样本的 ground truth 动作本身很小
   - 模型可能学到了"不动"的 shortcut
   - 检查: displacement 值

2. 【CLIP embedding 相似度高】
   - 不同 SignWriting 的 embedding 太接近
   - CLIP 无法区分细微差异
   - 检查: similarity matrix 中的高相似度对

3. 【SignWriting 复杂度】
   - 复杂的符号可能更难编码
   - 简单符号（少 non-white pixels）可能更容易学

4. 【数据质量】
   - pose 文件标注可能有问题
   - SignWriting 和 pose 对应可能不准确

下一步:
1. 查看 logs/sample_analysis/ 中的图片
2. 对比 high/low PCK 样本的 SignWriting
3. 检查 similarity matrix 中是否有模式
""")
    
    print("=" * 70)
    print(f"All outputs saved to: {OUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    analyze_samples()