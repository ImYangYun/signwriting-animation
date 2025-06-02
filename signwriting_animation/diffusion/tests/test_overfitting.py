import torch
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPProcessor

from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from signwriting_evaluation.metrics.clip import signwriting_to_clip_image

device = torch.device("cpu")

clip_model_name = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# 生成四个极端样本，覆盖 SW / past_motion 组合
sample_configs = [
    # (SignWriting字符串, pose取值, past_motion取值)
    ("M518x529S14c20481x471S27106503x489", 0.0, -1.0),
    ("M518x529S14c20481x471S27106503x489", 1.0,  1.0),
    ("M518x533S1870a489x515S18701482x490", 0.0,  0.0),
    ("M518x533S1870a489x515S18701482x490", 1.0,  2.0),
]

def make_sample(signwriting_str, val, past_val, device, clip_processor,
                batch_size=1, num_past_frames=10, num_keypoints=21, num_dims=3):
    # 这里 val 表示 future，past_val 表示 past_motion
    x = torch.full((batch_size, num_keypoints, num_dims, num_past_frames), val, device=device)
    timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
    past_motion = torch.full((batch_size, num_keypoints, num_dims, num_past_frames), past_val, device=device)
    pil_img = signwriting_to_clip_image(signwriting_str)
    sw_img = clip_processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    return x, timesteps, past_motion, sw_img, val

samples = [
    make_sample(sw, val, past_val, device, clip_processor)
    for sw, val, past_val in sample_configs
]

class OverfitLightningModel(pl.LightningModule):
    def __init__(self, diffusion_model):
        super().__init__()
        self.model = diffusion_model
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x, timesteps, past_motion, sw_img):
        return self.model(x, timesteps, past_motion, sw_img)

    def training_step(self, batch, batch_idx):
        x, timesteps, past_motion, sw_img, val = batch
        output = self(x, timesteps, past_motion, sw_img)
        target = torch.full_like(output, val)
        loss = self.loss_fn(output, target)
        if batch_idx == 0 and self.current_epoch % 50 == 0:
            self.print(f"Epoch {self.current_epoch}: output min/max: {output.min().item()}, {output.max().item()}")
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-4)

# 注意：Lightning 期望 DataLoader 里返回的 batch 结构
# 这里直接用 list of tuples，Lightning 会自动打包为 batch
dataloader = DataLoader(samples, batch_size=4, shuffle=False)

num_keypoints = 21
num_dims_per_keypoint = 3
num_past_frames = 10

diffusion_model = SignWritingToPoseDiffusion(
    num_keypoints=num_keypoints,
    num_dims_per_keypoint=num_dims_per_keypoint,
    embedding_arch=clip_model_name,
    num_latent_dims=32,   
    ff_size=64,
    num_layers=2,
    num_heads=2,
    dropout=0.0,
    cond_mask_prob=0
)

pl.seed_everything(42)
model = OverfitLightningModel(diffusion_model)
trainer = pl.Trainer(
    max_epochs=1000,  # 可先设小一点，收敛快就行
    log_every_n_steps=1,
    accelerator="cpu",
    devices=1,
    enable_checkpointing=False
)
trainer.fit(model, dataloader)

# evaluation
diffusion_model.eval()
with torch.no_grad():
    for idx, (x, timesteps, past_motion, sw_img, val) in enumerate(samples):
        output = diffusion_model(x, timesteps, past_motion, sw_img)
        rounded = torch.round(output)
        target = torch.full_like(output, val)
        print(f"[EVAL] Sample {idx+1} (target={val})")
        print("Output min/max:", output.min().item(), output.max().item())
        print("Rounded unique:", rounded.unique())
        print("Prediction after round:\n", rounded.cpu().numpy())
        print("Target:\n", target.cpu().numpy())
        assert torch.allclose(rounded, target, atol=1e-1), f"Sample {idx+1} did not overfit!"

print("All overfit sanity checks passed!")

