import torch
import torch.optim as optim
from transformers import CLIPProcessor
import matplotlib.pyplot as plt

from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from signwriting_evaluation.metrics.clip import signwriting_to_clip_image

def make_sample(signwriting_str, value, device, clip_processor,
                batch_size=1, num_past_frames=10, num_keypoints=21, num_dims=3):
    x = torch.full((batch_size, num_keypoints, num_dims, num_past_frames), value, device=device)
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
        num_latent_dims=64,
        ff_size=128,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        #activation="gelu",
        #arch="trans_enc",
        cond_mask_prob=0
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    sample_configs = [
        ("AS14c20S27106M518x529S14c20481x471S27106503x489", 0.)  # zeros
        #("AS18701S1870aS2e734S20500M518x533S1870a489x515S18701482x490S20500508x496S2e734500x468", 1.) # ones
    ]

    samples = []
    for sw, val in sample_configs:
        x, timesteps, past_motion, sw_img = make_sample(
            sw, val, device, clip_processor,
            num_past_frames=num_past_frames,
            num_keypoints=num_keypoints,
            num_dims=num_dims_per_keypoint)
        samples.append((x, timesteps, past_motion, sw_img, val))

    # Training
    num_epochs = 1000
    losses = []
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x, timesteps, past_motion, sw_img, val in samples:
            optimizer.zero_grad()
            output = model(x, timesteps, past_motion, sw_img)
            target = torch.full_like(output, val)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(samples)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

    # Plot loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Overfit Loss Curve")
    plt.tight_layout()
    plt.savefig("overfit_loss_curve.png")
    print("Saved loss curve to overfit_loss_curve.png")

    # Evaluation
    model.eval()
    with torch.no_grad():
    for idx, (x, timesteps, past_motion, sw_img, val) in enumerate(samples):
        output = model(x, timesteps, past_motion, sw_img)
        rounded = torch.round(output)
        target = torch.full_like(output, val)
        print(f"\n[EVAL] Sample {idx + 1} (target={val})")
        print("Output min/max:", output.min().item(), output.max().item())
        print("Rounded unique:", rounded.unique())
        print("Target unique:", target.unique())
        print("Prediction after round:\n", rounded.cpu().numpy())
        print("Target:\n", target.cpu().numpy())
        # Assert output matches target after rounding (allclose is tolerant of tiny float errors)
        assert torch.allclose(rounded, target, atol=1e-1), f"Sample {idx+1} did not overfit!"

    print("All overfit sanity checks passed!")

if __name__ == "__main__":
    main()
