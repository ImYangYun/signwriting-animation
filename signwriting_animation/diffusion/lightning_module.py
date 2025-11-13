import torch
import lightning as pl
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure as PE_DTW


# -----------------------------------------------------
# Utilities
# -----------------------------------------------------
def sanitize_btjc(x):
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    if x.dim() == 5:  # [B,T,P,J,C]
        x = x[:, :, 0]
    return x.float().contiguous()


def masked_mse(pred, gt, mask_bt):
    B,T,J,C = pred.shape
    m4 = mask_bt[:, :, None, None].float()
    return ((pred - gt)**2 * m4).sum() / (m4.sum() * J * C + 1e-6)


@torch.no_grad()
def masked_dtw(pred_btjc, tgt_btjc, mask_bt):
    metric = PE_DTW()
    B,T,J,C = pred_btjc.shape
    vals = []

    for b in range(B):
        t = int(mask_bt[b].sum().item())
        if t < 2:
            continue

        p = pred_btjc[b,:t].detach().cpu().numpy().astype("float32")
        g = tgt_btjc[b,:t].detach().cpu().numpy().astype("float32")

        # DTW expects shape (T,1,J,C)
        p = p[:,None,:,:]
        g = g[:,None,:,:]
        vals.append(metric.get_distance(p, g))

    if not vals:
        return torch.tensor(0.0, device=pred_btjc.device)
    return torch.tensor(vals, device=pred_btjc.device).mean()


class LitMinimal(pl.LightningModule):

    def __init__(self,
                 num_keypoints=178,
                 num_dims=3,
                 lr=1e-4,
                 stats_path="/data/yayun/pose_data/mean_std_178.pt"):
        super().__init__()
        self.save_hyperparameters()

        # ---- Load dataset-wide statistics ----
        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1,1,-1,3)   # [1,1,J,C]
        std  = stats["std"].float().view(1,1,-1,3)

        self.register_buffer("mean_pose", mean)
        self.register_buffer("std_pose",  std)

        # ---- Initialize CAMDM-based model ----
        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
            mean=self.mean_pose,
            std=self.std_pose,
        )

        print("[LitMinimal] CAMDM deterministic x0-prediction loaded ✔")

    # -----------------------------------------------------
    # normalization helpers
    # -----------------------------------------------------
    def normalize(self, x):
        return (x - self.mean_pose) / (self.std_pose + 1e-6)

    def unnormalize(self, x):
        return x * self.std_pose + self.mean_pose

    # -----------------------------------------------------
    # forward wrapper
    # -----------------------------------------------------
    def forward(self, x_btjc, ts, past_btjc, sign_img):
        x_bjct    = x_btjc.permute(0,2,3,1)
        past_bjct = past_btjc.permute(0,2,3,1)
        out_bjct  = self.model.forward(x_bjct, ts, past_bjct, sign_img)
        return out_bjct.permute(0,3,1,2).contiguous()

    # -----------------------------------------------------
    # TRAINING STEP（MSE only, for stable overfit）
    # -----------------------------------------------------
    def training_step(self, batch, _):
        cond = batch["conditions"]

        gt   = self.normalize(sanitize_btjc(batch["data"]))
        past = self.normalize(sanitize_btjc(cond["input_pose"]))
        sign = cond["sign_image"].float()

        # mask
        if cond["target_mask"].dim() == 4:
            mask = (cond["target_mask"].sum((2,3)) > 0).float()
        else:
            mask = cond["target_mask"].float()

        B,T,J,C = gt.shape
        past = past[:, -T:]   # ensure aligned windows

        ts = torch.zeros(B, dtype=torch.long, device=self.device)

        # ---- teacher forcing ----
        pred = self.forward(past, ts, past, sign)

        loss = masked_mse(pred, gt, mask)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    # -----------------------------------------------------
    # VALIDATION STEP（MSE + DTW）
    # -----------------------------------------------------
    def validation_step(self, batch, _):
        cond = batch["conditions"]

        gt   = self.normalize(sanitize_btjc(batch["data"]))
        past = self.normalize(sanitize_btjc(cond["input_pose"]))
        sign = cond["sign_image"].float()

        # mask
        if cond["target_mask"].dim() == 4:
            mask = (cond["target_mask"].sum((2,3)) > 0).float()
        else:
            mask = cond["target_mask"].float()

        B,T,J,C = gt.shape
        past = past[:, -T:]

        ts = torch.zeros(B, dtype=torch.long, device=self.device)

        pred = self.forward(past, ts, past, sign)

        loss = masked_mse(pred, gt, mask)

        # ---- DTW on unnormalized sequences ----
        dtw_val = masked_dtw(self.unnormalize(pred),
                             self.unnormalize(gt),
                             mask)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/dtw", dtw_val, prog_bar=True)

        return loss

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
