import pytest
import torch
from torch.utils.data import DataLoader
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset, get_num_workers
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from pose_format.torch.masked.collator import zero_pad_collator
import matplotlib.pyplot as plt

@pytest.mark.parametrize("batch_size", [4])
def test_length_prediction_on_real_data(batch_size):
    data_dir = "/scratch/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data.csv"

    dataset = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split='train'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=zero_pad_collator,
        num_workers=get_num_workers(),
        pin_memory=False,
    )

    model = SignWritingToPoseDiffusion(
        num_keypoints=586,
        num_dims_per_keypoint=3,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch = next(iter(dataloader))
    input_pose = batch["conditions"]["input_pose"].to(device).to(torch.float32)
    sign_image = batch["conditions"]["sign_image"].to(device).to(torch.float32)
    noisy_x = batch["data"].to(device).to(torch.float32)
    target_lengths = batch["length_target"].to(device).to(torch.float32).squeeze(-1)

    if input_pose.dim() == 5:
        input_pose = input_pose.squeeze(2).permute(0, 2, 3, 1)
    if noisy_x.dim() == 5:
        noisy_x = noisy_x.squeeze(2).permute(0, 2, 3, 1).contiguous()

    timesteps = torch.randint(0, 1000, (input_pose.shape[0],), device=device)

    # === Micro-training loop ===
    optimizer = torch.optim.Adam(model.length_predictor.parameters(), lr=1e-3)
    losses = []

    model.train()
    for step in range(100):
        optimizer.zero_grad()
        _, length_pred_dist = model(noisy_x, timesteps, input_pose, sign_image)
        nll = length_pred_dist.nll(target_lengths)
        loss = nll.mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 10 == 0:
            print(f"[Step {step}] NLL Loss: {loss.item():.4f}")

    # === Evaluation ===
    model.eval()
    with torch.no_grad():
        _, length_pred_dist = model(noisy_x, timesteps, input_pose, sign_image)
        global_latent = model.global_norm(model.sequence_pos_encoder(
            torch.cat([
                model.embed_timestep(timesteps),
                model.embed_signwriting(sign_image),
                model.past_motion_process(input_pose),
                model.future_motion_process(noisy_x)
            ], dim=0)
        ).mean(0))

    print("Global latent stats: min", global_latent.min().item(), "max", global_latent.max().item())
    print("Sample mean (length):", length_pred_dist.mean.mean().item())

    pred_lengths = length_pred_dist.mean.squeeze(-1)
    abs_diff = (pred_lengths - target_lengths).abs()
    nll = length_pred_dist.nll(target_lengths)
    samples = length_pred_dist.sample().squeeze(-1)

    print("\n=== Length Prediction Test ===")
    print("Target lengths:      ", [round(float(x), 2) for x in target_lengths])
    print("Predicted means:     ", [round(float(x), 2) for x in pred_lengths])
    print("Sampled lengths:     ", [round(float(x), 2) for x in samples])
    print("Absolute differences:", [round(float(x), 2) for x in abs_diff])
    print("Mean NLL:            ", round(float(nll.mean()), 4))

    # === Assertions ===
    relative_error = abs_diff / target_lengths.clamp(min=1.0)
    max_relative_error = 0.3
    assert torch.all(relative_error < max_relative_error), "Relative length prediction error too large."

    # === Plot prediction vs target ===
    plt.figure(figsize=(6, 6))
    plt.scatter(target_lengths.cpu(), pred_lengths.cpu(), c='blue', label='Prediction')
    plt.plot([0, max(target_lengths.max(), pred_lengths.max())], 
             [0, max(target_lengths.max(), pred_lengths.max())], 
             color='red', linestyle='--', label='Ideal')
    plt.xlabel("Target Length")
    plt.ylabel("Predicted Length")
    plt.title("Length Prediction vs Target")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("length_prediction_scatter.png")
    plt.close()

    # === Plot loss ===
    plt.plot(losses)
    plt.title("Training Loss (NLL)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("length_nll_loss.png")



