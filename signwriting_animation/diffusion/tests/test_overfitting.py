import torch
import torch.optim as optim
from transformers import CLIPProcessor
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from signwriting_evaluation.metrics.clip import signwriting_to_clip_image

def make_sample(signwriting_str, value, device, clip_processor,
                batch_size=1, num_past_frames=10, num_keypoints=21, num_dims=3):
    x = torch.full((batch_size, num_past_frames, num_keypoints, num_dims), value, device=device)
    timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
    past_motion = torch.full((batch_size, num_keypoints, num_dims, num_past_frames), value, device=device)
    pil_img = signwriting_to_clip_image(signwriting_str)
    signwriting_im_batch = clip_processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    return x, timesteps, past_motion, signwriting_im_batch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    num_keypoints = 21
    num_dims_per_keypoint = 3
    num_past_frames = 10

    model = SignWritingToPoseDiffusion(
        num_keypoints=num_keypoints,
        num_dims_per_keypoint=num_dims_per_keypoint,
        embedding_arch="openai/clip-vit-base-patch32",
        num_latent_dims=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.2,
        activation="gelu",
        arch="trans_enc",
        cond_mask_prob=0
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    sample_configs = [
        ("AS14c20S27106M518x529S14c20481x471S27106503x489", 0.),  # zeros
        ("AS18701S1870aS2e734S20500M518x533S1870a489x515S18701482x490S20500508x496S2e734500x468", 1.) # ones
    ]

    samples = []
    for sw, val in sample_configs:
        x, timesteps, past_motion, sw_img = make_sample(sw, val, device, clip_processor,
                                                        num_past_frames=num_past_frames,
                                                        num_keypoints=num_keypoints,
                                                        num_dims=num_dims_per_keypoint)
        # --- 这里打印每个sample的shape ---
        print(f"\nSample for target={val}")
        print("  x.shape:", x.shape)
        print("  timesteps.shape:", timesteps.shape)
        print("  past_motion.shape:", past_motion.shape)
        print("  sw_img.shape:", sw_img.shape)
        samples.append((x, timesteps, past_motion, sw_img, val))

    # 训练
    model.train()
    for epoch in range(100):
        epoch_loss = 0
        for x, timesteps, past_motion, sw_img, val in samples:
            # --- 这里加 permute ---
            x_perm = x.permute(0, 1, 3, 2)  # [1, 10, 21, 3] -> [1, 10, 3, 21]
            past_motion_perm = past_motion.permute(0, 2, 1, 3)  # [1, 21, 3, 10] -> [1, 3, 21, 10]
            print("\n[FORWARD] x_perm.shape:", x_perm.shape)
            print("[FORWARD] past_motion_perm.shape:", past_motion_perm.shape)
            print("[FORWARD] sw_img.shape:", sw_img.shape)
            optimizer.zero_grad()
            # 注意这里传入 permute 后的新变量
            output = model(x_perm, timesteps, past_motion_perm, sw_img)
            target = torch.full_like(output, val)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss / len(samples):.6f}")

    model.eval()
    with torch.no_grad():
        for idx, (x, timesteps, past_motion, sw_img, val) in enumerate(samples):
            # --- 同样加 permute ---
            x_perm = x.permute(0, 1, 3, 2)
            past_motion_perm = past_motion.permute(0, 2, 1, 3)
            print(f"\n[EVAL] Sample {idx + 1} (target={val})")
            print("[EVAL] x_perm.shape:", x_perm.shape)
            print("[EVAL] past_motion_perm.shape:", past_motion_perm.shape)
            print("[EVAL] sw_img.shape:", sw_img.shape)
            output = model(x_perm, timesteps, past_motion_perm, sw_img)
            rounded = torch.round(output)
            print("Prediction after round:\n", rounded.cpu().numpy())
            print("Target:\n", torch.full_like(output, val).cpu().numpy())


if __name__ == "__main__":
    main()
