# pylint: disable=invalid-name,arguments-differ,too-many-locals,too-many-instance-attributes,too-many-arguments,too-many-positional-arguments
import torch
from torch import nn
import numpy as np
import lightning as pl
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure as PE_DTW
from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion


def sanitize_btjc(x):
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    if hasattr(x, "tensor"):
        x = x.tensor
    if x.dim() == 5:
        x = x[:, :, 0]
    if x.dim() != 4:
        raise ValueError(f"Expected [B,T,J,C], got {tuple(x.shape)}")
    if x.shape[-1] != 3 and x.shape[-2] == 3:
        x = x.permute(0, 1, 3, 2)
    if x.shape[-1] != 3:
        raise ValueError(f"sanitize_btjc: last dim must be C=3, got {x.shape}")
    return x.contiguous().float()


def masked_mse(pred_btjc, tgt_btjc, mask_bt):
    pred = sanitize_btjc(pred_btjc)
    tgt  = sanitize_btjc(tgt_btjc)
    t_max = min(pred.size(1), tgt.size(1), mask_bt.size(1))
    pred = pred[:, :t_max]
    tgt  = tgt[:,  :t_max]
    m4 = mask_bt[:, :t_max].float()[:, :, None, None]
    diff2 = (pred - tgt) ** 2
    num = (diff2 * m4).sum()
    den = (m4.sum() * pred.size(2) * pred.size(3)).clamp_min(1.0)
    return num / den


def _btjc_to_tjc_list(x_btjc, mask_bt):
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
    preds = _btjc_to_tjc_list(pred_btjc, mask_bt)
    tgts  = _btjc_to_tjc_list(tgt_btjc,  mask_bt)
    dtw_metric = PE_DTW()
    vals = []
    for p, g in zip(preds, tgts):
        if p.size(0) < 2 or g.size(0) < 2:
            continue
        pv = p.detach().cpu().numpy().astype("float32")
        gv = g.detach().cpu().numpy().astype("float32")
        pv = pv[:, None, :, :]
        gv = gv[:, None, :, :]
        vals.append(float(dtw_metric.get_distance(pv, gv)))
    if not vals:
        return torch.tensor(0.0, device=pred_btjc.device)
    return torch.tensor(vals, device=pred_btjc.device).mean()


class LitMinimal(pl.LightningModule):
    """
    Optimized version with selective debug output.
    
    ✅ 修复：unnormalize 现在只做数值反归一化
    ⚠️ 重要：保存 pose 时必须调用 unshift_hands(pose_obj)
    """

    def __init__(
        self,
        num_keypoints=178,
        num_dims=3,
        lr=1e-4,
        stats_path="/home/yayun/data/pose_data/mean_std_178_with_preprocess.pt",
        diffusion_steps=200,
        beta_start=1e-4,
        beta_end=1e-3,
        pred_target="x0",
        guidance_scale=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.mean_pose = None
        self.std_pose = None
        self._step_count = 0

        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1, 1, -1, 3)
        std  = stats["std"].float().view(1, 1, -1, 3)

        if hasattr(self, "mean_pose"):
            delattr(self, "mean_pose")
        if "mean_pose" in self._buffers:
            del self._buffers["mean_pose"]
        self.register_buffer("mean_pose", mean.clone())

        if hasattr(self, "std_pose"):
            delattr(self, "std_pose")
        if "std_pose" in self._buffers:
            del self._buffers["std_pose"]
        self.register_buffer("std_pose", std.clone())

        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
        )

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
        self.w_vel = 50.0
        self.w_acc = 20.0
        self.train_logs = {
            "loss": [],
            "mse": [],
            "vel": [],
            "acc": []
        }

    def normalize(self, x):
        return (x - self.mean_pose) / (self.std_pose + 1e-6)

    def unnormalize(self, x):
        return x * self.std_pose + self.mean_pose

    @staticmethod
    def btjc_to_bjct(x):
        return x.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def bjct_to_btjc(x):
        return x.permute(0, 3, 1, 2).contiguous()

    def _diffuse_once(self, x0_btjc, t_long, cond):
        x0_bjct = self.btjc_to_bjct(x0_btjc)
        noise = torch.randn_like(x0_bjct)
        x_t = self.diffusion.q_sample(x0_bjct, t_long, noise=noise)
        t_scaled = getattr(self.diffusion, "_scale_timesteps")(t_long)
        past_bjct = self.btjc_to_bjct(cond["input_pose"])
        pred_bjct = self.model.forward(x_t, t_scaled, past_bjct, cond["sign_image"])
        target = noise if self.pred_target == "eps" else x0_bjct
        return pred_bjct, target

    def training_step(self, batch, batch_idx):
        debug_this_step = (
            self._step_count == 0
            or self._step_count % 100 == 0
            or self._step_count % 1000 == 0
        )

        if debug_this_step:
            print(f"\n{'='*70}")
            print(f"TRAINING STEP {self._step_count}")
            print(f"{'='*70}")

        cond_raw  = batch["conditions"]
        gt_btjc   = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(cond_raw["input_pose"])
        sign_img  = cond_raw["sign_image"].float()

        if debug_this_step:
            print(f"gt: {gt_btjc.shape}, range=[{gt_btjc.min():.4f}, {gt_btjc.max():.4f}]")
            print(f"past: {past_btjc.shape}")

        # --- normalize ---
        gt   = self.normalize(gt_btjc)
        past = self.normalize(past_btjc)
        cond = {"input_pose": past, "sign_image": sign_img}

        B = gt.size(0)
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)

        # --- diffusion training step ---
        pred_bjct, target_bjct = self._diffuse_once(gt, t, cond)
        loss_main = torch.nn.functional.mse_loss(pred_bjct, target_bjct)

        pred_btjc = self.bjct_to_btjc(pred_bjct)
        gt_raw   = self.unnormalize(gt)
        pred_raw = self.unnormalize(pred_btjc)

        loss_vel = torch.tensor(0.0, device=self.device)
        loss_acc = torch.tensor(0.0, device=self.device)

        if pred_raw.size(1) > 1:
            v_pred = pred_raw[:, 1:] - pred_raw[:, :-1]
            v_gt   = gt_raw[:, 1:]   - gt_raw[:, :-1]
            loss_vel = torch.nn.functional.l1_loss(v_pred, v_gt)
            if v_pred.size(1) > 1:
                a_pred = v_pred[:, 1:] - v_pred[:, :-1]
                a_gt   = v_gt[:, 1:]   - v_gt[:, :-1]
                loss_acc = torch.nn.functional.l1_loss(a_pred, a_gt)

        loss = loss_main + self.w_vel * loss_vel + self.w_acc * loss_acc

        if debug_this_step:
            mean_d_pred = v_pred.abs().mean().item() if pred_raw.size(1) > 1 else 0.0
            mean_d_gt   = v_gt.abs().mean().item()   if gt_raw.size(1) > 1 else 0.0
            print(f"loss_main: {loss_main.item():.6f}")
            print(f"loss_vel: {loss_vel.item():.6f} (w={self.w_vel})")
            print(f"loss_acc: {loss_acc.item():.6f} (w={self.w_acc})")
            print(f"TOTAL: {loss.item():.6f}")
            print(f"mean |Δ pred|: {mean_d_pred:.6f}, mean |Δ gt|: {mean_d_gt:.6f}")
            print(f"{'='*70}\n")

        self.log_dict(
            {
                "train/loss": loss,
                "train/mse": loss_main,
                "train/vel": loss_vel,
                "train/acc": loss_acc,
            },
            prog_bar=True,
        )

        self.train_logs["loss"].append(loss.item())
        self.train_logs["mse"].append(loss_main.item())
        self.train_logs["vel"].append(loss_vel.item())
        self.train_logs["acc"].append(loss_acc.item())

        self._step_count += 1
        return loss

    @torch.no_grad()
    def validation_step(self, batch, _):
        cond_raw  = batch["conditions"]
        gt_btjc   = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(cond_raw["input_pose"])
        mask_bt   = cond_raw.get("target_mask", None)
        sign_img  = cond_raw["sign_image"].float()

        gt   = self.normalize(gt_btjc)
        past = self.normalize(past_btjc)
        cond = {"input_pose": past, "sign_image": sign_img}

        B = gt.size(0)
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)

        pred_bjct, _ = self._diffuse_once(gt, t, cond)
        gt_bjct = self.btjc_to_bjct(gt)

        Tpred = pred_bjct.size(-1)
        Tgt   = gt_bjct.size(-1)
        Tmin  = min(Tpred, Tgt)

        pred_bjct = pred_bjct[..., :Tmin]
        gt_bjct   = gt_bjct[..., :Tmin]

        loss_main = torch.nn.functional.mse_loss(pred_bjct, gt_bjct)
        self.log("val/mse", loss_main, prog_bar=True)

        if self.pred_target == "x0":
            pred_btjc = self.bjct_to_btjc(pred_bjct)
            if mask_bt is None:
                mask_use = torch.ones(gt.shape[:2], device=gt.device)
            elif mask_bt.dim() == 2:
                mask_use = mask_bt.float()
            else:
                mask_use = (mask_bt.sum((2, 3)) > 0).float()
            pred_btjc_u = self.unnormalize(pred_btjc)
            gt_btjc_u   = self.unnormalize(gt[:, :Tmin])
            dtw_val = masked_dtw(pred_btjc_u, gt_btjc_u, mask_use[:, :Tmin])
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

        past_raw = past_btjc.to(device)
        B, Tp, J, C = past_raw.shape

        cur_hist = past_raw.clone()
        frames_out = []
        sign = sign_img.to(device)

        class _Wrapper(nn.Module):
            def __init__(self, mdl):
                super().__init__()
                self.m = mdl
            def forward(self, x, t, **kw):
                return self.m.interface(x, t, kw["y"])

        wrapped = _Wrapper(self.model)

        remain = future_len
        while remain > 0:
            n = min(chunk, remain)
            cur_hist_norm = self.normalize(cur_hist)
            cond = {
                "input_pose": self.btjc_to_bjct(cur_hist_norm),
                "sign_image": sign,
            }
            shape = (B, J, C, n)
            x_bjct = self.diffusion.p_sample_loop(
                wrapped,
                shape=shape,
                model_kwargs={"y": cond},
                clip_denoised=False,
                progress=False
            )
            x_btjc_norm = self.bjct_to_btjc(x_bjct)
            x_btjc_raw = self.unnormalize(x_btjc_norm)
            frames_out.append(x_btjc_raw)
            cur_hist = torch.cat([cur_hist, x_btjc_raw], dim=1)
            if cur_hist.size(1) > Tp:
                cur_hist = cur_hist[:, -Tp:]
            remain -= n

        return torch.cat(frames_out, dim=1)

    def on_train_start(self):
        self.mean_pose = self.mean_pose.to(self.device)
        self.std_pose  = self.std_pose.to(self.device)

    def on_predict_start(self):
        self.mean_pose = self.mean_pose.to(self.device)
        self.std_pose  = self.std_pose.to(self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "gradient_clip_val": 1.0,
        }
    
    def on_train_end(self):
        import matplotlib.pyplot as plt
        import os

        out_dir = "logs/minimal_178_fixed"
        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        ax[0,0].plot(self.train_logs["loss"])
        ax[0,0].set_title("train/loss")
        ax[0,1].plot(self.train_logs["mse"])
        ax[0,1].set_title("train/mse")
        ax[1,0].plot(self.train_logs["vel"])
        ax[1,0].set_title("train/vel")
        ax[1,1].plot(self.train_logs["acc"])
        ax[1,1].set_title("train/acc")
        plt.tight_layout()
        out_path = f"{out_dir}/train_curve.png"
        plt.savefig(out_path)
        print(f"[TRAIN CURVE] saved → {out_path}")
