import pytest
import torch
from torch.utils.data import DataLoader
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset, get_num_workers
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from pose_format.torch.masked.collator import zero_pad_collator


@pytest.mark.parametrize("batch_size", [4])
def test_length_prediction_on_real_data(batch_size):
    """
    Unit test for probabilistic length prediction module using DiagonalGaussianDistribution.
    Checks that:
        - predicted mean is close to target
        - distribution sampling works
        - NLL is non-negative
    """
    data_dir = "/scratch/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data.csv"

    # Load dataset
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

    # Load model
    model = SignWritingToPoseDiffusion(
        num_keypoints=586,
        num_dims_per_keypoint=3,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Prepare batch
    batch = next(iter(dataloader))
    input_pose = batch["conditions"]["input_pose"].to(device).to(torch.float32)
    sign_image = batch["conditions"]["sign_image"].to(device).to(torch.float32)
    noisy_x = batch["data"].to(device).to(torch.float32)
    target_lengths = batch["length_target"].to(device).to(torch.float32)

    # Expected input_pose: [B, T, 1, K, D]
    if input_pose.dim() == 5:
        input_pose = input_pose.squeeze(2)                # [B, T, K, D]
        input_pose = input_pose.permute(0, 2, 3, 1)       # -> [B, K, D, T]
    else:
        raise ValueError(f"Unexpected input_pose shape: {input_pose.shape}")


    # Fix noisy_x to [B, K, D, T]
    # noisy_x: [B, T, 1, K, D]?
    if noisy_x.dim() == 5:
        noisy_x = noisy_x.squeeze(2).permute(0, 2, 3, 1).contiguous()  # -> [B, K, D, T]
    else:
        raise ValueError(f"Unexpected noisy_x shape: {noisy_x.shape}")

    timesteps = torch.randint(0, 1000, (input_pose.shape[0],), device=device)

    with torch.no_grad():
        _, length_pred_dist = model(noisy_x, timesteps, input_pose, sign_image)

    pred_lengths = length_pred_dist.mean.squeeze(-1)
    abs_diff = (pred_lengths - target_lengths).abs()
    nll = length_pred_dist.nll(target_lengths)

    # === Additional distribution checks ===
    samples = length_pred_dist.sample().squeeze(-1) 
    assert samples.shape == pred_lengths.shape
    assert pred_lengths.shape == target_lengths.shape
    assert torch.all(nll >= 0), "Negative NLL encountered."

    print("pred_lengths shape:", pred_lengths.shape)
    print("samples shape:", samples.shape)
    print("target_lengths shape:", target_lengths.shape)

    # === Logging ===
    print("\n=== Length Prediction Test ===")
    print("Target lengths:      ", [round(float(x), 2) for x in target_lengths])
    print("Predicted means:     ", [round(float(x), 2) for x in pred_lengths])
    print("Sampled lengths:     ", [round(float(x), 2) for x in samples])
    print("Absolute differences:", [round(float(x), 2) for x in abs_diff])
    print("Mean NLL:            ", round(float(nll.mean()), 4))

    # === Main assertion ===
    max_allowed_diff = 10.0
    assert torch.all(abs_diff < max_allowed_diff), "Length prediction error too large."



