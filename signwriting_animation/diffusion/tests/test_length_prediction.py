import pytest
import torch
from torch.utils.data import DataLoader
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset, get_num_workers
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from pose_format.torch.masked.collator import zero_pad_collator


@pytest.mark.parametrize("batch_size", [4])
def test_length_prediction_on_real_data(batch_size):
    """
    Pytest-style test for probabilistic length prediction using DiagonalGaussianDistribution.
    Ensures that predicted mean length is close to target, and prints NLL.
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

    # Initialize model
    model = SignWritingToPoseDiffusion(
        num_keypoints=21,
        num_dims_per_keypoint=3,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load batch
    batch = next(iter(dataloader))
    input_pose = batch["conditions"]["input_pose"].to(device)

    # Fix input_pose from [T, 1, K, D] to [1, T, K, D]
    if input_pose.dim() == 4 and input_pose.shape[1] == 1:
        input_pose = input_pose.permute(1, 0, 2, 3).contiguous()
    elif input_pose.dim() != 4:
        raise ValueError(f"Unexpected shape for input_pose: {input_pose.shape}")

    sign_image = batch["conditions"]["sign_image"].to(device)
    noisy_x = batch["data"].to(device)
    timesteps = torch.randint(0, 1000, (input_pose.shape[0],)).to(device)

    # Check and reshape noisy_x
    if noisy_x.dim() == 5 and noisy_x.shape[2] == 1:
        noisy_x = noisy_x.squeeze(2)
    elif noisy_x.dim() != 4:
        raise ValueError(f"Unexpected shape for noisy_x: {noisy_x.shape}")

    with torch.no_grad():
        _, length_pred_dist = model(noisy_x, timesteps, input_pose, sign_image)

    target_lengths = batch["length_target"].to(device)

    # Use predicted mean as the estimate
    pred_lengths = length_pred_dist.mean.squeeze(-1)
    abs_diff = (pred_lengths - target_lengths).abs()
    nll = length_pred_dist.nll(target_lengths)

    print("\n=== Length Prediction Test ===")
    print("Target lengths:     ", [round(float(x), 2) for x in target_lengths])
    print("Predicted means:    ", [round(float(x), 2) for x in pred_lengths])
    print("Absolute differences:", [round(float(x), 2) for x in abs_diff])
    print("Mean NLL:           ", round(float(nll.mean()), 4))

    max_allowed_diff = 10.0
    assert (abs_diff < max_allowed_diff).all(), "Length prediction difference too large!"



