import torch
import torch.nn as nn
import numpy as np
import lightning as pl
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from CAMDM.diffusion.respace import SpacedDiffusion, space_timesteps
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
                 stats_path="/data/yayun/pose_data/mean_std_178.pt",
                 diffusion_steps: int = 200,
                 beta_start: float = 1e-4,
                 beta_end: float = 2e-2,
                 pred_target: str = "x0",
                 guidance_scale: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        self.verbose = False

        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1, 1, -1, 3)  # [1,1,J,C]
        std  = stats["std"].float().view(1, 1, -1, 3)
        self.register_buffer("mean_pose", mean)
        self.register_buffer("std_pose",  std)

        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
        )
        print("[LitMinimal] CAMDM core loaded ✔")

        self.pred_target = pred_target.lower()  # "x0" or "eps"
        model_mean_type = (ModelMeanType.EPSILON
                           if self.pred_target == "eps"
                           else ModelMeanType.START_X)

        betas = np.linspace(beta_start, beta_end, diffusion_steps, dtype=np.float64)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False
        )

        self.guidance_scale = float(guidance_scale)
        self.lr = lr
        print(f"[LitMinimal] ✅ diffusion ready (target={self.pred_target}, steps={diffusion_steps})")

    # ---------- 工具 ----------
    def normalize(self, x):   return (x - self.mean_pose) / (self.std_pose + 1e-6)
    def unnormalize(self, x): return x * self.std_pose + self.mean_pose

    @staticmethod
    def btjc_to_bjct(x): return x.permute(0, 2, 3, 1).contiguous()
    @staticmethod
    def bjct_to_btjc(x): return x.permute(0, 3, 1, 2).contiguous()

    def _forward_bjct(self, x_bjct, t_long, past_btjc, sign_img):
        past_bjct = self.btjc_to_bjct(past_btjc)                # [B,J,C,Tp]
        out_bjct  = self.model.forward(x_bjct, t_long, past_bjct, sign_img)  # [B,J,C,T]
        return out_bjct

    def _diffuse_once(self, x0_btjc, t_long, cond):
        x0_bjct = self.btjc_to_bjct(x0_btjc)  # [B,J,C,T]
        noise   = torch.randn_like(x0_bjct)
        x_t     = self.diffusion.q_sample(x0_bjct, t_long, noise=noise)

        pred_bjct = self._forward_bjct(
            x_bjct=x_t,
            t_long=self.diffusion._scale_timesteps(t_long),
            past_btjc=cond["input_pose"],
            sign_img=cond["sign_image"]
        )

        if self.pred_target == "eps":
            target_bjct = noise            # 训练预测 ε
        else:
            target_bjct = x0_bjct          # 训练预测 x0

        return pred_bjct, target_bjct

    # ---------- 训练 ----------
    def training_step(self, batch, _):
        cond_raw  = batch["conditions"]
        gt_btjc   = sanitize_btjc(batch["data"])                  # [B,30,J,C]
        past_btjc = sanitize_btjc(cond_raw["input_pose"])         # [B,60,J,C]
        sign_img  = cond_raw["sign_image"].float()

        if not hasattr(self, "_std_calibrated"):
            with torch.no_grad():
                raw_gt = sanitize_btjc(batch["data"]).to(self.device)
                tmp = (raw_gt - self.mean_pose) / (self.std_pose + 1e-6)
                cur = tmp.float().std().item()
                factor = max(cur, 1e-3)
                self.std_pose[:] = self.std_pose * factor
                print(f"[Calib] normalized std was {cur:.3f} → scaled std_pose by {factor:.3f}")
                self._std_calibrated = True

        gt   = self.normalize(gt_btjc)
        past = self.normalize(past_btjc)[:, -gt.size(1):]         # [B,30,J,C]

        cond = {"input_pose": past, "sign_image": sign_img}

        B = gt.size(0)
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device, dtype=torch.long)

        pred_bjct, target_bjct = self._diffuse_once(gt, t, cond)
        loss_main = torch.nn.functional.mse_loss(pred_bjct, target_bjct)

        pred_btjc   = self.bjct_to_btjc(pred_bjct)
        x0_target_btjc = gt if self.pred_target == "x0" else gt
        loss_vel = torch.tensor(0.0, device=self.device)
        loss_acc = torch.tensor(0.0, device=self.device)
        if pred_btjc.size(1) > 1:
            v_pred = pred_btjc[:, 1:] - pred_btjc[:, :-1]
            v_tgt  = x0_target_btjc[:, 1:] - x0_target_btjc[:, :-1]
            loss_vel = torch.nn.functional.l1_loss(v_pred, v_tgt)
            if v_pred.size(1) > 1:
                a_pred = v_pred[:, 1:] - v_pred[:, :-1]
                a_tgt  = v_tgt[:, 1:]  - v_tgt[:, :-1]
                loss_acc = torch.nn.functional.l1_loss(a_pred, a_tgt)

        loss = loss_main + 0.5 * loss_vel + 0.25 * loss_acc

        self.log_dict({
            "train/loss": loss,
            "train/mse": loss_main,
            "train/vel": loss_vel,
            "train/acc": loss_acc,
        }, prog_bar=True)

        return loss

    # ---------- 验证 ----------
    @torch.no_grad()
    def validation_step(self, batch, _):
        cond_raw  = batch["conditions"]
        gt_btjc   = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(cond_raw["input_pose"])
        mask_bt   = cond_raw.get("target_mask", None)
        sign_img  = cond_raw["sign_image"].float()

        gt   = self.normalize(gt_btjc)
        past = self.normalize(past_btjc)[:, -gt.size(1):]
        cond = {"input_pose": past, "sign_image": sign_img}

        B = gt.size(0)
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device, dtype=torch.long)

        pred_bjct, target_bjct = self._diffuse_once(gt, t, cond)
        loss_main = torch.nn.functional.mse_loss(pred_bjct, target_bjct)
        self.log("val/mse", loss_main, prog_bar=True)

        if self.pred_target == "x0":
            x0_pred_btjc = self.bjct_to_btjc(pred_bjct)
            # 掩码处理（兼容 [B,T] 或 [B,T,J,C]）
            if mask_bt is None:
                mask_bt_use = torch.ones(gt.shape[:2], device=gt.device)
            elif mask_bt.dim() == 2:
                mask_bt_use = mask_bt.float()
            else:
                mask_bt_use = (mask_bt.sum((2, 3)) > 0).float()

            dtw_val = masked_dtw(self.unnormalize(x0_pred_btjc),
                                 self.unnormalize(gt),
                                 mask_bt_use)
            self.log("val/dtw", dtw_val, prog_bar=True)

        return loss_main


    @torch.no_grad()
    def sample_autoregressive_fast(
        self,
        past_btjc,
        sign_img,
        future_len: int = 30,
        chunk: int = 1,
        guidance_scale: float = None,
    ):
        self.eval()
        device = self.device

        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        # ------------ Normalize -------------
        past_norm = self.normalize(past_btjc.to(device))
        sign = sign_img.to(device)

        B, Tp, J, C = past_norm.shape
        frames = []
        remain = int(future_len)
        cur_hist = past_norm.clone()

        # ------------ Wrapper for interface() -------------
        class _Wrapper(nn.Module):
            def __init__(self, mdl):
                super().__init__()
                self.m = mdl

            def forward(self, x, t, **kwargs):
                cond = kwargs["y"]
                return self.m.interface(x, t, cond)

        wrapped = _Wrapper(self.model).to(device)

        # ------------ Loop -------------
        while remain > 0:
            n = min(chunk, remain)
            shape_bjct = (B, J, C, n)

            cond = {
            "input_pose": self.btjc_to_bjct(cur_hist),
            "sign_image": sign,
        }

            x_bjct = self.diffusion.p_sample_loop(
                model=wrapped,
                shape=shape_bjct,
                model_kwargs={"y": cond},
                clip_denoised=False,
                progress=False,
            )

            x_btjc = self.bjct_to_btjc(x_bjct)

            frames.append(x_btjc)

            # update window
            cur_hist = torch.cat([cur_hist, x_btjc], dim=1)
            if cur_hist.size(1) > Tp:
                cur_hist = cur_hist[:, -Tp:, :]

            remain -= n

        pred_norm = torch.cat(frames, dim=1)
        return pred_norm
    
    def on_train_start(self):
        self.mean_pose = self.mean_pose.to(self.device)
        self.std_pose  = self.std_pose.to(self.device)
        print(f"[on_train_start] mean/std moved to {self.device}")

    def on_predict_start(self):
        self.mean_pose = self.mean_pose.to(self.device)
        self.std_pose  = self.std_pose.to(self.device)
        print(f"[on_predict_start] mean/std moved to {self.device}")


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
