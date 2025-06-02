import torch
from torch.utils.data import DataLoader, Dataset
from lightning import seed_everything, Trainer, LightningModule
from torch import optim
from transformers import CLIPProcessor
import numpy as np

from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from signwriting_evaluation.metrics.clip import signwriting_to_clip_image

# 固定随机种子
seed_everything(42)

# -------- Dataset Definition --------
class OverfitSamplesDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# -------- Make Sample --------
def make_sample(signwriting_str, pose_val, past_motion_val, device, clip_processor,
                batch_size=1, num_past_frames=10, num_keypoints=21, num_dims=3):
    """
    Create a dummy sample: specific pose and past_motion values, with SignWriting image embedding.
    """
    x = torch.full((batch_size, num_keypoints, num_dims, num_past_frames), pose_val, device=device)
    timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
    past_motion = torch.full((batch_size, num_keypoints, num_dims, num_past_frames), past_motion_val, device=device)
    pil_img = signwriting_to_clip_image(signwriting_str)
    sw_img = clip_processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    # 这里 target 设为 pose_val（你可根据需求也加 past_motion_val 作分析）
    return x, timesteps, past_motion, sw_img, torch.tensor(pose_val, dtype=torch.float32, device=device)

# -------- Lightning Module --------
class LightningOverfitModel(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x, timesteps, past_motion, sw_img):
        return self.model(x, timesteps, past_motion, sw_img)

    def training_step(self, batch, batch_idx):
        x, timesteps, past_motion, sw_img, target = batch
        output = self(x, timesteps, past_motion, sw_img)
        # target shape needs to match output
        target_tensor = torch.full_like(output, target.item())
        loss = self.loss_fn(output, target_tensor)
        if batch_idx == 0 and self.current_epoch % 200 == 0:
            self.print(f"Output min/max: {output.min().item()}, {output.max().item()}")
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-4)

# -------- Main Overfitting Test --------
def run_overfit_lightning():
    # Always use cpu for testing (github CI-friendly, deterministic)
    device = torch.device("cpu")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Model hyperparameters
    num_keypoints = 21
    num_dims_per_keypoint = 3
    num_past_frames = 10

    # Init model
    model = SignWritingToPoseDiffusion(
        num_keypoints=num_keypoints,
        num_dims_per_keypoint=num_dims_per_keypoint,
        embedding_arch="openai/clip-vit-base-patch32",
        num_latent_dims=32,
        ff_size=64,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        cond_mask_prob=0
    ).to(device)

    # Use four overfit samples as suggested by your advisor
    sample_configs = [
        ("M518x529S14c20481x471S27106503x489", 0, -1),
        ("M518x529S14c20481x471S27106503x489", 1, 1),
        ("M518x533S1870a489x515S18701482x490", 0, 0),
        ("M518x533S1870a489x515S18701482x490", 1, 2),
    ]
    samples = [
        make_sample(sw, pose_val, past_motion_val, device, clip_processor,
                    num_past_frames=num_past_frames,
                    num_keypoints=num_keypoints,
                    num_dims=num_dims_per_keypoint)
        for (sw, pose_val, past_motion_val) in sample_configs
    ]

    dataset = OverfitSamplesDataset(samples)
    # Don't shuffle for deterministic behavior
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    lightning_model = LightningOverfitModel(model)

    # Train
    trainer = Trainer(max_epochs=1000, log_every_n_steps=1, enable_checkpointing=False)
    trainer.fit(lightning_model, dataloader)

    # Evaluation: check output is close to target for all four cases
    model.eval()
    print("\n[EVAL] Overfit sanity check:")
    with torch.no_grad():
        for idx, (x, timesteps, past_motion, sw_img, target) in enumerate(samples):
            output = model(x, timesteps, past_motion, sw_img)
            rounded = torch.round(output)
            target_tensor = torch.full_like(output, target.item())
            print(f"\nSample {idx+1}")
            print("Output min/max:", output.min().item(), output.max().item())
            print("Rounded unique:", rounded.unique())
            print("Target unique:", target_tensor.unique())
            print("Prediction after round:\n", rounded.cpu().numpy())
            print("Target:\n", target_tensor.cpu().numpy())
            assert torch.allclose(rounded, target_tensor, atol=1e-1), f"Sample {idx+1} did not overfit!"

    print("All overfit sanity checks passed!")

if __name__ == "__main__":
    run_overfit_lightning()
