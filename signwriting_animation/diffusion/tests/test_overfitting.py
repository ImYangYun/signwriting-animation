import torch
import torch.optim as optim
from transformers import CLIPProcessor
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from signwriting_evaluation.metrics.clip import signwriting_to_clip_image

def make_sample(signwriting_str, value, device, clip_processor,
                batch_size=1, num_past_frames=10, num_keypoints=21, num_dims=3):
    # x: [B, num_past_frames, num_keypoints, num_dims]
    x = torch.full((batch_size, num_past_frames, num_keypoints, num_dims), value, device=device)
    # reshape to [num_past_frames, batch_size, num_keypoints * num_dims] == [10, 1, 63]
    x = x.permute(1, 0, 2, 3).reshape(num_past_frames, batch_size, num_keypoints * num_dims)
    # timesteps: [B]
    timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
    # past_motion: [B, num_keypoints, num_dims, num_past_frames]
    past_motion = torch.full((batch_size, num_keypoints, num_dims, num_past_frames), value, device=device)
    # [B, num_keypoints, num_dims, num_past_frames] -> [num_past_frames, batch, num_keypoints * num_dims]
    past_motion = past_motion.permute(3, 0, 1, 2).reshape(num_past_frames, batch_size, num_keypoints * num_dims)
    # signwriting image batch: [B, 3, 224, 224]
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
        samples.append((x, timesteps, past_motion, sw_img, val))

    # 训练
    model.train()
    for epoch in range(100):
        epoch_loss = 0
        for x, timesteps, past_motion, sw_img, val in samples:
            optimizer.zero_grad()
            # forward: x, timesteps, past_motion, signwriting_im_batch
            output = model(x, timesteps, past_motion, sw_img)
            # loss与目标的比较，目标和x一致
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
            output = model(x, timesteps, past_motion, sw_img)
            rounded = torch.round(output)
            print(f"\nSample {idx + 1} (target={val}):")
            print("Prediction after round:\n", rounded.cpu().numpy())
            print("Target:\n", torch.full_like(output, val).cpu().numpy())

if __name__ == "__main__":
    main()
