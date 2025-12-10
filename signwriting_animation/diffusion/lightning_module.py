# pylint: disable=invalid-name,arguments-differ,too-many-locals,too-many-instance-attributes,too-many-arguments,too-many-positional-arguments
import os
import torch
from torch import nn
import numpy as np
import lightning as pl
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure as PE_DTW
from CAMDM.diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
)
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion


def sanitize_btjc(x: torch.Tensor) -> torch.Tensor:
    """Convert pose-format tensors to [B,T,J,C] float32."""
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
    tgts = _btjc_to_tjc_list(tgt_btjc, mask_bt)
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


def mean_frame_disp(x_btjc: torch.Tensor) -> float:
    x = sanitize_btjc(x_btjc)
    if x.size(1) < 2:
        return 0.0
    v = x[:, 1:] - x[:, :-1]
    return v.abs().mean().item()


class LitMinimalAutoregressive(pl.LightningModule):
    """
    修复版 - 自回归 Direct 模式
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
        train_mode: str = "direct",
        vel_weight: float = 0.5,
        acc_weight: float = 0.25,
        motion_weight: float = 0.1,
        history_len: int = 40,  # 保持的历史长度
    ):
        super().__init__()
        self.save_hyperparameters()

        assert train_mode in {"diffusion", "direct"}
        self.train_mode = train_mode
        self.vel_weight = float(vel_weight)
        self.acc_weight = float(acc_weight)
        self.motion_weight = float(motion_weight)
        self.history_len = history_len

        self.mean_pose = None
        self.std_pose = None
        self._step_count = 0

        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1, 1, -1, 3)
        std = stats["std"].float().view(1, 1, -1, 3)

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
        model_mean_type = (
            ModelMeanType.EPSILON
            if self.pred_target == "eps"
            else ModelMeanType.START_X
        )

        betas = np.linspace(beta_start, beta_end, diffusion_steps, dtype=np.float64)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        self.lr = lr
        self.guidance_scale = float(guidance_scale)
        self.train_logs = {"loss": [], "mse": [], "vel": [], "acc": [], "motion": []}

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

    def _predict_single_frame(self, past_norm_btjc, sign_img, debug=False):
        """
        预测下一帧（单帧）
        """
        B, T_past, J, C = past_norm_btjc.shape
        device = past_norm_btjc.device

        x0_btjc = torch.zeros(B, 1, J, C, device=device, dtype=past_norm_btjc.dtype)
        x0_bjct = self.btjc_to_bjct(x0_btjc)
        t_long = torch.zeros(B, dtype=torch.long, device=device)
        t_scaled = getattr(self.diffusion, "_scale_timesteps")(t_long)

        past_bjct = self.btjc_to_bjct(past_norm_btjc)
        pred_bjct = self.model.forward(x0_bjct, t_scaled, past_bjct, sign_img)
        pred_btjc = self.bjct_to_btjc(pred_bjct)
        
        if debug:
            print(f"    [_predict_single_frame] past: {past_norm_btjc.shape}, pred: {pred_btjc.shape}")
            print(f"    past[-1] mean: {past_norm_btjc[:, -1].mean().item():.4f}")
            print(f"    pred mean: {pred_btjc.mean().item():.4f}")
        
        return pred_btjc

    def _direct_predict_autoregressive(self, past_norm_btjc, sign_img, future_len, debug=False):
        """
        自回归预测多帧 - 修复版
        
        关键：每次预测后，把预测结果加入历史，滑动窗口
        """
        B, T_past, J, C = past_norm_btjc.shape
        device = past_norm_btjc.device
        
        predictions = []
        # 重要：clone 历史，避免原地修改
        current_history = past_norm_btjc.clone()
        
        if debug:
            print(f"\n[_direct_predict_autoregressive] future_len={future_len}")
            print(f"  initial history shape: {current_history.shape}")
            print(f"  initial history[-1] mean: {current_history[:, -1].mean().item():.4f}")
        
        for t in range(future_len):
            # 预测下一帧
            pred_frame = self._predict_single_frame(current_history, sign_img, debug=(debug and t < 3))
            predictions.append(pred_frame.clone())  # 重要：clone
            
            if debug and t < 3:
                print(f"  step {t}: pred_frame mean={pred_frame.mean().item():.4f}")
            
            # 关键修复：滑动窗口更新历史
            # 去掉最老的一帧，加入新预测的帧
            current_history = torch.cat([
                current_history[:, 1:],  # 去掉第一帧
                pred_frame               # 加入预测帧
            ], dim=1)
            
            if debug and t < 3:
                print(f"  step {t}: new history[-1] mean={current_history[:, -1].mean().item():.4f}")
        
        result = torch.cat(predictions, dim=1)
        
        if debug:
            print(f"  final result shape: {result.shape}")
            # 检查帧间差异
            if result.size(1) > 1:
                diff = (result[:, 1:] - result[:, :-1]).abs().mean().item()
                print(f"  frame-to-frame diff: {diff:.6f}")
        
        return result

    def training_step(self, batch, batch_idx):
        debug_this_step = (
            self._step_count == 0
            or self._step_count % 100 == 0
        )

        cond_raw = batch["conditions"]
        gt_btjc = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(cond_raw["input_pose"])
        sign_img = cond_raw["sign_image"].float()

        if debug_this_step:
            print("\n" + "=" * 70)
            print(f"TRAINING STEP {self._step_count}")
            print("=" * 70)
            print(f"gt:   {gt_btjc.shape}, range=[{gt_btjc.min():.4f}, {gt_btjc.max():.4f}]")
            print(f"past: {past_btjc.shape}, range=[{past_btjc.min():.4f}, {past_btjc.max():.4f}]")

        gt = self.normalize(gt_btjc)
        past = self.normalize(past_btjc)

        if self.train_mode == "diffusion":
            B = gt.size(0)
            t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)
            cond = {"input_pose": past, "sign_image": sign_img}

            pred_bjct, target_bjct = self._diffuse_once(gt, t, cond)
            loss_main = torch.nn.functional.mse_loss(pred_bjct, target_bjct)

            pred_btjc = self.bjct_to_btjc(pred_bjct)
            gt_raw = self.unnormalize(gt)
            pred_raw = self.unnormalize(pred_btjc)

        else:
            B, T_future, J, C = gt.shape
            
            # 随机选择一个时间步
            t_idx = torch.randint(0, T_future, (1,)).item()
            
            # 构建历史：past + gt[:t_idx]
            if t_idx == 0:
                history = past
            else:
                history = torch.cat([past, gt[:, :t_idx]], dim=1)
                # 只保留最后 history_len 帧
                if history.size(1) > self.history_len:
                    history = history[:, -self.history_len:]
            
            # 预测下一帧
            pred_frame = self._predict_single_frame(history, sign_img)
            gt_frame = gt[:, t_idx:t_idx+1]
            
            loss_main = torch.nn.functional.mse_loss(pred_frame, gt_frame)
            
            # 每隔一段时间做完整自回归预测来计算 vel/acc
            if self._step_count % 10 == 0:
                with torch.no_grad():
                    pred_norm_full = self._direct_predict_autoregressive(
                        past, sign_img, T_future, debug=debug_this_step
                    )
                pred_raw = self.unnormalize(pred_norm_full)
            else:
                # 简化：用单帧重复
                pred_raw = self.unnormalize(pred_frame.repeat(1, T_future, 1, 1))
            
            gt_raw = self.unnormalize(gt)

        # 计算 smooth losses
        loss_vel = torch.tensor(0.0, device=self.device)
        loss_acc = torch.tensor(0.0, device=self.device)
        loss_motion = torch.tensor(0.0, device=self.device)

        if pred_raw.size(1) > 1:
            v_pred = pred_raw[:, 1:] - pred_raw[:, :-1]
            v_gt = gt_raw[:, 1:] - gt_raw[:, :-1]
            loss_vel = torch.nn.functional.l1_loss(v_pred, v_gt)
            
            motion_magnitude = v_pred.abs().mean()
            gt_motion_magnitude = v_gt.abs().mean()
            loss_motion = torch.relu(gt_motion_magnitude - motion_magnitude)
            
            if v_pred.size(1) > 1:
                a_pred = v_pred[:, 1:] - v_pred[:, :-1]
                a_gt = v_gt[:, 1:] - v_gt[:, :-1]
                loss_acc = torch.nn.functional.l1_loss(a_pred, a_gt)

        loss = (
            loss_main
            + self.vel_weight * loss_vel
            + self.acc_weight * loss_acc
            + self.motion_weight * loss_motion
        )

        if debug_this_step:
            print(f"loss_main: {loss_main.item():.6f}")
            print(f"loss_vel:  {loss_vel.item():.6f} (w={self.vel_weight:.1f})")
            print(f"loss_acc:  {loss_acc.item():.6f} (w={self.acc_weight:.1f})")
            print(f"loss_motion: {loss_motion.item():.6f} (w={self.motion_weight:.2f})")
            print(f"pred frame-to-frame disp: {mean_frame_disp(pred_raw):.6f}")
            print(f"gt frame-to-frame disp: {mean_frame_disp(gt_raw):.6f}")
            print(f"TOTAL: {loss.item():.6f}")
            print("=" * 70 + "\n")

        self.log_dict(
            {
                "train/loss": loss,
                "train/mse": loss_main,
                "train/vel": loss_vel,
                "train/acc": loss_acc,
                "train/motion": loss_motion,
            },
            prog_bar=True,
        )
        self.train_logs["loss"].append(loss.item())
        self.train_logs["mse"].append(loss_main.item())
        self.train_logs["vel"].append(loss_vel.item())
        self.train_logs["acc"].append(loss_acc.item())
        self.train_logs["motion"].append(loss_motion.item())

        self._step_count += 1
        return loss

    @torch.no_grad()
    def validation_step(self, batch, _):
        cond_raw = batch["conditions"]
        gt_btjc = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(cond_raw["input_pose"])
        mask_bt = cond_raw.get("target_mask", None)
        sign_img = cond_raw["sign_image"].float()

        gt = self.normalize(gt_btjc)
        past = self.normalize(past_btjc)
        
        B, T_future, J, C = gt.shape

        if self.train_mode == "direct":
            pred_norm = self._direct_predict_autoregressive(past, sign_img, T_future)
            loss_main = torch.nn.functional.mse_loss(pred_norm, gt)
            
            pred_raw = self.unnormalize(pred_norm)
            gt_raw = self.unnormalize(gt)
            
            disp = mean_frame_disp(pred_raw)
            self.log("val/frame_disp", disp, prog_bar=True)
        else:
            cond = {"input_pose": past, "sign_image": sign_img}
            t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)
            pred_bjct, _ = self._diffuse_once(gt, t, cond)
            gt_bjct = self.btjc_to_bjct(gt)
            
            Tmin = min(pred_bjct.size(-1), gt_bjct.size(-1))
            pred_bjct = pred_bjct[..., :Tmin]
            gt_bjct = gt_bjct[..., :Tmin]
            
            loss_main = torch.nn.functional.mse_loss(pred_bjct, gt_bjct)
            pred_raw = self.unnormalize(self.bjct_to_btjc(pred_bjct))
            gt_raw = self.unnormalize(gt[:, :Tmin])

        self.log("val/mse", loss_main, prog_bar=True)

        if mask_bt is None:
            mask_use = torch.ones(gt_raw.shape[:2], device=gt_raw.device)
        elif mask_bt.dim() == 2:
            mask_use = mask_bt.float()
        else:
            mask_use = (mask_bt.sum((2, 3)) > 0).float()
        
        Tmin = min(pred_raw.size(1), gt_raw.size(1), mask_use.size(1))
        dtw_val = masked_dtw(pred_raw[:, :Tmin], gt_raw[:, :Tmin], mask_use[:, :Tmin])
        self.log("val/dtw", dtw_val, prog_bar=True)

        return loss_main

    @torch.no_grad()
    def predict_direct(self, past_btjc, sign_img, future_len=20, debug=True):
        """
        自回归推理 - 修复版，添加 debug 输出
        """
        self.eval()
        device = self.device

        past_raw = sanitize_btjc(past_btjc.to(device))
        past_norm = self.normalize(past_raw)
        
        print(f"\n[predict_direct] past_norm shape: {past_norm.shape}")
        print(f"[predict_direct] past_norm range: [{past_norm.min():.4f}, {past_norm.max():.4f}]")
        
        pred_norm = self._direct_predict_autoregressive(
            past_norm, sign_img.to(device), future_len, debug=debug
        )
        
        pred_raw = self.unnormalize(pred_norm)
        
        print(f"[predict_direct] pred_raw shape: {pred_raw.shape}")
        print(f"[predict_direct] pred_raw range: [{pred_raw.min():.4f}, {pred_raw.max():.4f}]")
        
        return pred_raw

    @torch.no_grad()
    def sample_autoregressive_fast(
        self, past_btjc, sign_img, future_len=20, chunk=1, guidance_scale=None
    ):
        assert self.train_mode == "diffusion", "Only valid in diffusion mode"

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
                progress=False,
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
        self.std_pose = self.std_pose.to(self.device)

    def on_predict_start(self):
        self.mean_pose = self.mean_pose.to(self.device)
        self.std_pose = self.std_pose.to(self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer, "gradient_clip_val": 1.0}

    def on_train_end(self):
        try:
            import matplotlib.pyplot as plt

            out_dir = "logs/minimal_178_autoregressive"
            os.makedirs(out_dir, exist_ok=True)

            fig, ax = plt.subplots(2, 3, figsize=(15, 8))
            ax[0, 0].plot(self.train_logs["loss"])
            ax[0, 0].set_title("train/loss")
            ax[0, 1].plot(self.train_logs["mse"])
            ax[0, 1].set_title("train/mse")
            ax[0, 2].plot(self.train_logs["motion"])
            ax[0, 2].set_title("train/motion")
            ax[1, 0].plot(self.train_logs["vel"])
            ax[1, 0].set_title("train/vel")
            ax[1, 1].plot(self.train_logs["acc"])
            ax[1, 1].set_title("train/acc")
            plt.tight_layout()
            out_path = f"{out_dir}/train_curve.png"
            plt.savefig(out_path)
            print(f"[TRAIN CURVE] saved → {out_path}")
        except Exception as e:
            print(f"[TRAIN CURVE] plotting failed: {e}")


LitMinimal = LitMinimalAutoregressive