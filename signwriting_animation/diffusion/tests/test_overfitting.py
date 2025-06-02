import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from transformers import CLIPProcessor
import lightning as L
from lightning import seed_everything

from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from signwriting_evaluation.metrics.clip import signwriting_to_clip_image

# Set global random seed for reproducibility
seed_everything(42)

def make_sample(signwriting_str, value, device, clip_processor,
                batch_size=1, num_past_frames=10, num_keypoints=21, num_dims=3):
    """
    Create a dummy sample: all-zero or all-one pose, with SignWriting image embedding.
    """
    x = torch.full((batch_size, num_keypoints, num_dims, num_past_frames), value, device=device)
    timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
    past_motion = torch.full((batch_size, num_keypoints, num_dims, num_past_frames), value, device=device)
    pil_img = signwriting_to_clip_image(signwriting_str)
    sw_img = clip_processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    return x, timesteps, past_motion, sw_img, torch.tensor(value, device=device)

# Custom dataset for overfit samples
class OverfitPoseDataset(Dataset):
    def __init__(self, sample_configs, device, clip_processor, num_past_frames=10, num_keypoints=21, num_dims=3):
        self.samples = []
        for sw, val in sample_configs:
            s = make_sample(
                sw, val, device, clip_processor,
                num_past_frames=num_past_frames,
                num_keypoints=num_keypoints,
                num_dims=num_dims)
            self.samples.append(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class LightningOverfitModel(L.LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.lr = lr
    
    print("x.shape:", x.shape)
    def forward(self, x, timesteps, past_motion, sw_img):
        return self.model(x, timesteps, past_motion, sw_img)

    def training_step(self, batch, batch_idx):
        x, timesteps, past_motion, sw_img, val = batch
        output = self(x, timesteps, past_motion, sw_img)
        target = torch.full_like(output, val)
        loss = self.loss_fn(output, target)
        if batch_idx == 0 and self.current_epoch % 200 == 0:
            self.print(f"Output min/max: {output.min().item()}, {output.max().item()}")
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

def run_overfit_lightning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    num_keypoints = 21
    num_dims_per_keypoint = 3
    num_past_frames = 10

    # Prepare 4 different overfit samples for better check
    sample_configs = [
        # All zeros
        ("AS14c20S27106M518x529S14c20481x471S27106503x489", 0.0),
        # All ones
        ("AS18701S1870aS2e734S20500M518x533S1870a489x515S18701482x490S20500508x496S2e734500x468", 1.0),
        # All 0.5
        ("AS1f720S20500M520x540S20500507x461S1f720482x483", 0.5),
        # All -1 (to test a negative value)
        ("AS15250S2a20bM515x525S15250482x481S2a20b505x488", -1.0)
    ]
    dataset = OverfitPoseDataset(
        sample_configs, device, clip_processor,
        num_past_frames=num_past_frames,
        num_keypoints=num_keypoints,
        num_dims=num_dims_per_keypoint
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

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

    lightning_model = LightningOverfitModel(model)

    # Trainer: use 2000 epochs for better overfit
    trainer = L.Trainer(
        max_epochs=2000,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu"
    )
    trainer.fit(lightning_model, dataloader)

    # Manual loss curve (optional)
    # Save checkpoint or evaluate
    lightning_model.eval()
    with torch.no_grad():
        for idx, (x, timesteps, past_motion, sw_img, val) in enumerate(dataset):
            output = lightning_model(x, timesteps, past_motion, sw_img)
            rounded = torch.round(output)
            target = torch.full_like(output, val)
            print(f"\n[EVAL] Sample {idx + 1} (target={val.item()})")
            print("Output min/max:", output.min().item(), output.max().item())
            print("Rounded unique:", rounded.unique())
            print("Target unique:", target.unique())
            print("Prediction after round:\n", rounded.cpu().numpy())
            print("Target:\n", target.cpu().numpy())
            assert torch.allclose(rounded, target, atol=1e-1), f"Sample {idx+1} did not overfit!"
    print("All overfit sanity checks passed!")

if __name__ == "__main__":
    run_overfit_lightning()

