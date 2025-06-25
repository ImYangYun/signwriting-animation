import pytest
import torch
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset, get_num_workers
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from pose_format.torch.masked.collator import zero_pad_collator
from torch.utils.data import DataLoader


@pytest.mark.parametrize("batch_size", [4])
def test_length_prediction_on_real_data(batch_size):
    """
    Pytest-style test for length prediction on real data using SignWritingToPoseDiffusion.
    Asserts that the predicted length is reasonably close to the target length.
    """
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
        num_keypoints=21,
        num_dims_per_keypoint=3,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch = next(iter(dataloader))

    input_pose = batch["conditions"]["input_pose"].to(device)
    sign_image = batch["conditions"]["sign_image"].to(device)
    noisy_x = batch["data"].to(device)
    timesteps = torch.randint(0, 1000, (input_pose.shape[0],)).to(device)

    noisy_x = noisy_x.permute(0, 2, 3, 1).contiguous()
    input_pose = input_pose.permute(0, 2, 3, 1).contiguous()
    
    model.eval()
    with torch.no_grad():
        pose_out, length_pred = model(noisy_x, timesteps, input_pose, sign_image)

    target_lengths = batch["length_target"].to(device).squeeze(-1)
    pred_lengths = length_pred.squeeze(-1)

    abs_diff = (pred_lengths - target_lengths).abs()

    print("\n=== Length Prediction Test ===")
    print("Target lengths:", [round(float(x), 2) for x in target_lengths])
    print("Predicted lengths:", [round(float(x), 2) for x in pred_lengths])
    print("Absolute differences:", [round(float(x), 2) for x in abs_diff])

    max_allowed_diff = 10.0
    assert (abs_diff < max_allowed_diff).all(), "Length prediction difference too large!"



