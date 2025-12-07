# pylint: disable=invalid-name,arguments-differ,too-many-locals,too-many-instance-attributes,too-many-arguments,too-many-positional-arguments
import torch
from torch import nn
import numpy as np
import lightning as pl
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure as PE_DTW
from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion


def _to_dense(x):
    """Convert MaskedTensor / sparse tensor to dense contiguous float32."""
    if hasattr(x, "tensor"):
        x = x.tensor
    elif hasattr(x, "zero_filled"):
        x = x.zero_filled()
    
    if getattr(x, "is_sparse", False):
        x = x.to_dense()
    if x.dtype != torch.float32:
        x = x.float()
    return x.contiguous()


def sanitize_btjc(x):
    x = _to_dense(x)
    
    # Strip P dimension if exists [B,T,P,J,C] → [B,T,J,C]
    if x.dim() == 5:
        x = x[:, :, 0]
    
    if x.dim() != 4:
        raise ValueError(f"Expected [B,T,J,C], got {tuple(x.shape)}")  
    return x


def masked_mse(pred_btjc, tgt_btjc, mask_bt):
    """
    Stable masked MSE (trim to min length, avoid denominator errors).
    """
    pred = sanitize_btjc(pred_btjc)
    tgt  = sanitize_btjc(tgt_btjc)

    t_max = min(pred.size(1), tgt.size(1), mask_bt.size(1))
    pred = pred[:, :t_max]
    tgt  = tgt[:,  :t_max]
    m4 = mask_bt[:, :t_max].float()[:, :, None, None]   # [B,T,1,1]
    diff2 = (pred - tgt) ** 2

    num = (diff2 * m4).sum()
    den = (m4.sum() * pred.size(2) * pred.size(3)).clamp_min(1.0)
    return num / den


def _btjc_to_tjc_list(x_btjc, mask_bt):
    """
    Convert batched sequence [B,T,J,C] into list of valid-length [T,J,C].
    """
    x_btjc = sanitize_btjc(x_btjc)
    batch_size, seq_len, _, _ = x_btjc.shape
    mask_bt = (mask_bt > 0.5).float()

    seqs = []
    for b in range(batch_size):
        t = int(mask_bt[b].sum().item())
        t = max(0, min(t, seq_len))
        seqs.append(x_btjc[b, :t].contiguous())
    return seqs


@torch.no_grad()
def masked_dtw(pred_btjc, tgt_btjc, mask_bt):
    """
    Stable DTW using trimmed sequences.
    This is the SAME version used in your successful PR.
    """
    preds = _btjc_to_tjc_list(pred_btjc, mask_bt)
    tgts  = _btjc_to_tjc_list(tgt_btjc,  mask_bt)
    dtw_metric = PE_DTW()

    vals = []
    for p, g in zip(preds, tgts):
        if p.size(0) < 2 or g.size(0) < 2:
            continue

        pv = p.detach().cpu().numpy().astype("float32")   # (T,J,C)
        gv = g.detach().cpu().numpy().astype("float32")   # (T,J,C)

        pv = pv[:, None, :, :]   # → (T,1,J,C)
        gv = gv[:, None, :, :]

        vals.append(float(dtw_metric.get_distance(pv, gv)))

    if not vals:
        return torch.tensor(0.0, device=pred_btjc.device)

    return torch.tensor(vals, device=pred_btjc.device).mean()


#  --------------------- LitMinimal Model ---------------------

class LitMinimal(pl.LightningModule):
    """
    Final Unified Version of Your LightningModule
    ---------------------------------------------
    Features:
        ✓ Stable sanitize / masked_mse / masked_dtw (from your best PR)
        ✓ CAMDM diffusion core (GaussianDiffusion)
        ✓ Global mean/std normalization for 178 joints
        ✓ Velocity + acceleration auxiliary loss
        ✓ Autoregressive fast sampling (p_sample_loop)
        ✓ Fully compatible with your minimal loop pipeline

    This is the version recommended for your new PR.
    """

    def __init__(
        self,
        num_keypoints=178,
        num_dims=3,
        lr=1e-4,
        stats_path="/home/yayun/data/pose_data/mean_std_178_with_preprocess.pt",
        diffusion_steps=200,
        beta_start=1e-4,
        beta_end=2e-2,
        pred_target="x0",
        guidance_scale=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.verbose = False

        self.mean_pose = None
        self.std_pose = None
        self._std_calibrated = False

        # ---------------- Load global mean/std ----------------
        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1, 1, -1, 3)
        std  = stats["std"].float().view(1, 1, -1, 3)

        # mean_pose
        if hasattr(self, "mean_pose"):
            delattr(self, "mean_pose")
        if "mean_pose" in self._buffers:
            del self._buffers["mean_pose"]

        self.register_buffer("mean_pose", mean.clone())

        # std_pose
        if hasattr(self, "std_pose"):
            delattr(self, "std_pose")
        if "std_pose" in self._buffers:
            del self._buffers["std_pose"]

        self.register_buffer("std_pose", std.clone())


        # ---------------- Load SignWriting → Pose model ----------------
        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
        )
        print("[LitMinimal] CAMDM model loaded ✔")

        # ---------------- Diffusion setup ----------------
        self.pred_target = pred_target.lower()
        model_mean_type = ModelMeanType.EPSILON if self.pred_target == "eps" else ModelMeanType.START_X

        betas = np.linspace(beta_start, beta_end, diffusion_steps, dtype=np.float64)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False
        )

        self.lr = lr
        self.guidance_scale = float(guidance_scale)
        self._std_calibrated = False
        print(f"[LitMinimal] diffusion ready (target={self.pred_target}, T={diffusion_steps})")

    #  Normalization
    def normalize(self, x):
        return (x - self.mean_pose) / (self.std_pose + 1e-6)

    def unnormalize(self, x):
        return x * self.std_pose + self.mean_pose

    #  Format: [B,T,J,C] ↔ [B,J,C,T]
    @staticmethod
    def btjc_to_bjct(x):
        return x.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def bjct_to_btjc(x):
        return x.permute(0, 3, 1, 2).contiguous()

    def _forward_bjct(self, x_bjct, t_long, past_btjc, sign_img):
        past_bjct = self.btjc_to_bjct(past_btjc)
        return self.model.forward(x_bjct, t_long, past_bjct, sign_img)

    #  Diffusion training step
    def _diffuse_once(self, x0_btjc, t_long, cond):
        x0_bjct = self.btjc_to_bjct(x0_btjc)
        noise   = torch.randn_like(x0_bjct)
        x_t     = self.diffusion.q_sample(x0_bjct, t_long, noise=noise)

        t_scaled = getattr(self.diffusion, "_scale_timesteps")(t_long)

        pred_bjct = self._forward_bjct(
            x_bjct=x_t,
            t_long=t_scaled,
            past_btjc=cond["input_pose"],
            sign_img=cond["sign_image"]
        )


        target = noise if self.pred_target == "eps" else x0_bjct
        return pred_bjct, target

    #  Training Step
    def training_step(self, batch, _):
        cond_raw  = batch["conditions"]
        gt_btjc   = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(cond_raw["input_pose"])
        sign_img  = cond_raw["sign_image"].float()

        gt   = self.normalize(gt_btjc)
        past = self.normalize(past_btjc)[:, -gt.size(1):]
        cond = {"input_pose": past, "sign_image": sign_img}

        B = gt.size(0)
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)

        pred_bjct, target_bjct = self._diffuse_once(gt, t, cond)
        pred_bjct = torch.clamp(pred_bjct, -3.0, 3.0)

        loss_main = torch.nn.functional.mse_loss(pred_bjct, target_bjct)

        # velocity / acceleration
        pred_btjc = self.bjct_to_btjc(pred_bjct)

        loss_vel = torch.tensor(0.0, device=self.device)
        loss_acc = torch.tensor(0.0, device=self.device)

        if pred_btjc.size(1) > 1:
            v_pred = pred_btjc[:, 1:] - pred_btjc[:, :-1]
            v_gt   = gt[:, 1:]         - gt[:, :-1]
            loss_vel = torch.nn.functional.l1_loss(v_pred, v_gt)

            if v_pred.size(1) > 1:
                a_pred = v_pred[:, 1:] - v_pred[:, :-1]
                a_gt   = v_gt[:, 1:]   - v_gt[:, :-1]
                loss_acc = torch.nn.functional.l1_loss(a_pred, a_gt)

        loss = loss_main + 0.5 * loss_vel + 0.25 * loss_acc

        self.log_dict({
            "train/loss": loss,
            "train/mse": loss_main,
            "train/vel": loss_vel,
            "train/acc": loss_acc,
        }, prog_bar=True)
        # ==== PATCH 5: debug 检查 ====
        if pred_bjct.abs().max() > 10:
            print("⚠ WARNING: pred_norm explosion →", pred_bjct.abs().max().item())
        return loss

    #  Validation Step
    @torch.no_grad()
    def validation_step(self, batch, _):
        cond_raw  = batch["conditions"]
        gt_btjc   = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(cond_raw["input_pose"])
        mask_bt   = cond_raw.get("target_mask", None)
        sign_img  = cond_raw["sign_image"].float()

        # normalize
        gt   = self.normalize(gt_btjc)
        past = self.normalize(past_btjc)[:, -gt.size(1):]

        cond = {"input_pose": past, "sign_image": sign_img}

        B = gt.size(0)
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)

        # NOTE: _diffuse_once has safe timestep now
        pred_bjct, _ = self._diffuse_once(gt, t, cond)

        # -- FIX: convert GT to BJCT for correct loss --
        gt_bjct = self.btjc_to_bjct(gt)
        loss_main = torch.nn.functional.mse_loss(pred_bjct, gt_bjct)

        self.log("val/mse", loss_main, prog_bar=True)

        # Only evaluate DTW when predicting x0
        if self.pred_target == "x0":
            pred_btjc = self.bjct_to_btjc(pred_bjct)

            if mask_bt is None:
                mask_use = torch.ones(gt.shape[:2], device=gt.device)
            elif mask_bt.dim() == 2:
                mask_use = mask_bt.float()
            else:
                mask_use = (mask_bt.sum((2,3)) > 0).float()

            # unnormalize sequences
            pred_btjc_u = self.unnormalize(pred_btjc)
            gt_btjc_u   = self.unnormalize(gt)

            dtw_val = masked_dtw(pred_btjc_u, gt_btjc_u, mask_use)
            self.log("val/dtw", dtw_val, prog_bar=True)

        return loss_main

    @torch.no_grad()
    def sample_autoregressive_fast(
        self, past_btjc, sign_img, future_len=20, chunk=1, guidance_scale=None
    ):
        self.eval()
        device = self.device

        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        past_norm = self.normalize(past_btjc.to(device))
        sign = sign_img.to(device)

        B, Tp, J, C = past_norm.shape
        frames = []
        remain = int(future_len)
        cur_hist = past_norm.clone()

        class _Wrapper(nn.Module):
            def __init__(self, mdl):
                super().__init__()
                self.m = mdl
            def forward(self, x, t, **kw):
                return self.m.interface(x, t, kw["y"])

        wrapped = _Wrapper(self.model)

        while remain > 0:
            n = min(chunk, remain)
            bjct_shape = (B, J, C, n)

            cond = {
                "input_pose": self.btjc_to_bjct(cur_hist),
                "sign_image": sign,
            }

            x_bjct = self.diffusion.p_sample_loop(
                wrapped,
                shape=bjct_shape,
                model_kwargs={"y": cond},
                clip_denoised=True,
                progress=False,
            )
            x_bjct = torch.clamp(x_bjct, -3.0, 3.0)
            x_btjc = self.bjct_to_btjc(x_bjct)
            frames.append(x_btjc)

            cur_hist = torch.cat([cur_hist, x_btjc], dim=1)
            if cur_hist.size(1) > Tp:
                cur_hist = cur_hist[:, -Tp:]

            remain -= n

        return torch.cat(frames, dim=1)

    def on_train_start(self):
        self.mean_pose = self.mean_pose.to(self.device)
        self.std_pose  = self.std_pose.to(self.device)
        print(f"[on_train_start] stats moved to {self.device}")

    def on_predict_start(self):
        self.mean_pose = self.mean_pose.to(self.device)
        self.std_pose  = self.std_pose.to(self.device)
        print(f"[on_predict_start] stats moved to {self.device}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "gradient_clip_val": 1.0,      # ==== PATCH 4 ====
        }
