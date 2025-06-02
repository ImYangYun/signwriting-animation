import torch
import lightning as pl  # 新版 lightning
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from signwriting_evaluation.metrics.clip import signwriting_to_clip_image

# Set reproducibility seed
pl.seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def make_sample(signwriting_str, value, device, clip_processor,
                num_past_frames=10, num_keypoints=21, num_dims=3):
    """
    Create a dummy sample: all-zero or all-one pose, with SignWriting image embedding.
    No batch dimension here!
    """
    x = torch.full((num_keypoints, num_dims, num_past_frames), value, device=device)
    timesteps = torch.tensor(0, dtype=torch.long, device=device)
    past_motion = torch.full((num_keypoints, num_dims, num_past_frames), value, device=device)
    pil_img = signwriting_to_clip_image(signwriting_str)
    sw_img = clip_processor(images=pil_img, return_tensors="pt").pixel_values.squeeze(0).to(device)
    val_tensor = torch.tensor(value, dtype=torch.float32, device=device)
    return x, timesteps, past_motion, sw_img, val_tensor

# ==== 构造4个不同的样本（2个0, 2个1，可自定义SW字符串）====
sample_configs = [
    ("AS14c20S27106M518x529S14c20481x471S27106503x489", 0.0),
    ("AS18701S1870aS2e734S20500M518x533S1870a489x515S18701482x490S20500508x496S2e734500x468", 1.0),
    ("AS14c20S27106M518x529S14c20481x471S27106503x489", 0.0),
    ("AS18701S1870aS2e734S20500M518x533S1870a489x515S18701482x490S20500508x496S2e734500x468", 1.0),
]

samples = [
    make_sample(sw, val, device, clip_processor)
    for sw, val in sample_configs
]

dataloader = DataLoader(samples, batch_size=4, shuffle=True)

# ====== Lightning 模型封装 ======
class LightningOverfitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x, timesteps, past_motion, sw_img):
        return self.model(x, timesteps, past_motion, sw_img)

    def training_step(self, batch, batch_idx):
        x, timesteps, past_motion, sw_img, val = batch
        # x: [B, 21, 3, 10] ...
        output = self(x, timesteps, past_motion, sw_img)
        target = torch.full_like(output, val.view(-1, 1, 1, 1))
        loss = self.loss_fn(output, target)
        if batch_idx == 0 and self.current_epoch % 200 == 0:
            self.print(f"Output min/max: {output.min().item()}, {output.max().item()}")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

# ==== 构建待测模型（与训练脚本一致，超参数可自定义）====
num_keypoints = 21
num_dims_per_keypoint = 3
num_past_frames = 10

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

# ==== Lightning训练器 ====
trainer = pl.Trainer(
    max_epochs=1000,
    log_every_n_steps=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
)

# ==== 开始训练（overfitting）====
trainer.fit(lightning_model, dataloader)

# ==== 训练结束后，手动检查是否完美拟合 ====
print("\nEvaluating overfit sanity...")
lightning_model.eval()
with torch.no_grad():
    for idx, (x, timesteps, past_motion, sw_img, val) in enumerate(samples):
        # 手动加 batch 维
        x_b = x.unsqueeze(0)
        t_b = timesteps.unsqueeze(0)
        p_b = past_motion.unsqueeze(0)
        s_b = sw_img.unsqueeze(0)
        output = lightning_model(x_b, t_b, p_b, s_b)
        rounded = torch.round(output)
        target = torch.full_like(output, val)
        print(f"[EVAL] Sample {idx+1} (target={val.item()})")
        print("Output min/max:", output.min().item(), output.max().item())
        print("Rounded unique:", rounded.unique())
        print("Prediction after round:\n", rounded.cpu().numpy())
        print("Target:\n", target.cpu().numpy())

        assert torch.allclose(rounded, target, atol=1e-1), f"Sample {idx+1} did not overfit!"

print("All overfit sanity checks passed!")
