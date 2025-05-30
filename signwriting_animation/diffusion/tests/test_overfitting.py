import torch
import torch.optim as optim
from transformers import CLIPProcessor
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from signwriting_evaluation.metrics.clip import signwriting_to_clip_image

def get_toy_samples(clip_processor, device="cpu"):
    seq_len = 10
    num_keypoints = 21
    num_dims_per_keypoint = 3

    signwriting_strings = [
        "AS14c20S27106M518x529S14c20481x471S27106503x489",
        "AS18701S1870aS2e734S20500M518x533S1870a489x515S18701482x490S20500508x496S2e734500x468"
    ]

    sw_imgs = []
    for s in signwriting_strings:
        pil_img = signwriting_to_clip_image(s)
        sw_img = clip_processor(images=pil_img, return_tensors="pt").pixel_values.squeeze(0)
        sw_imgs.append(sw_img.to(device))

    # pose target
    pose_zeros = torch.zeros(seq_len, num_keypoints, num_dims_per_keypoint, device=device)
    pose_ones = torch.ones(seq_len, num_keypoints, num_dims_per_keypoint, device=device)

    return [
    (pose_zeros, sw_imgs[0], pose_zeros),  
    (pose_ones, sw_imgs[1], pose_ones)
    ]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    num_keypoints = 21
    num_dims_per_keypoint = 3
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

    samples = get_toy_samples(clip_processor, device=device)

    model.train()
    for epoch in range(100):
        epoch_loss = 0
        for sw_img, past_pose, future_pose in samples:
            optimizer.zero_grad()
            output = model(past_pose, sw_img)
            loss = loss_fn(output, future_pose)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss/len(samples):.6f}")

    model.eval()
    with torch.no_grad():
        for idx, (sw_img, past_pose, future_pose) in enumerate(samples):
            output = model(past_pose, sw_img)
            rounded = torch.round(output)
            print(f"\nSample {idx+1}:")
            print("Prediction after round:\n", rounded.cpu().numpy())
            print("Target:\n", future_pose.cpu().numpy())

if __name__ == "__main__":
    main()
