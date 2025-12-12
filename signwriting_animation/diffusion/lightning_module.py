# pylint: disable=invalid-name,arguments-differ,too-many-locals,too-many-instance-attributes,too-many-arguments,too-many-positional-arguments
import os
import torch
from torch import nn
import numpy as np
import lightning as pl


from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure as PE_DTW
from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusionV2

def sanitize_btjc(x: torch.Tensor) -> torch.Tensor:
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
    
    try:
        dtw_metric = PE_DTW()
    except:
        pred = sanitize_btjc(pred_btjc)
        tgt = sanitize_btjc(tgt_btjc)
        t_max = min(pred.size(1), tgt.size(1))
        return torch.mean((pred[:, :t_max] - tgt[:, :t_max]) ** 2)
    
    vals = []
    for p, g in zip(preds, tgts):
        if p.size(0) < 2 or g.size(0) < 2:
            continue
        pv = p.detach().cpu().numpy().astype("float32")[:, None, :, :]
        gv = g.detach().cpu().numpy().astype("float32")[:, None, :, :]
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


class LitResidual(pl.LightningModule):
    """
    使用残差预测模型的 Lightning Module
    
    支持多样本训练 + validation
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
        vel_weight: float = 1.0,
        acc_weight: float = 0.5,
        residual_scale: float = 0.1,
        hand_reg_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hand_reg_weight = hand_reg_weight
        
        self.right_hand_joints = list(range(157, 178))
        self.left_hand_joints = list(range(136, 157))

        assert train_mode in {"diffusion", "direct", "ar"}
        self.train_mode = train_mode
        self.vel_weight = float(vel_weight)
        self.acc_weight = float(acc_weight)

        self._step_count = 0

        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1, 1, -1, 3)
        std = stats["std"].float().view(1, 1, -1, 3)

        self.register_buffer("mean_pose", mean.clone())
        self.register_buffer("std_pose", std.clone())

        self.model = SignWritingToPoseDiffusionV2(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
            residual_scale=residual_scale,
        )

        self.pred_target = pred_target.lower()
        model_mean_type = (
            ModelMeanType.EPSILON if self.pred_target == "eps" else ModelMeanType.START_X
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
        self.train_logs = {"loss": [], "mse": [], "vel": [], "acc": []}

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

    def _predict_frames(self, past_norm_btjc, sign_img, num_frames):
        """预测 num_frames 帧"""
        B, T_past, J, C = past_norm_btjc.shape
        device = past_norm_btjc.device

        x_btjc = torch.zeros(B, num_frames, J, C, device=device, dtype=past_norm_btjc.dtype)
        x_bjct = self.btjc_to_bjct(x_btjc)
        
        t_long = torch.zeros(B, dtype=torch.long, device=device)
        t_scaled = getattr(self.diffusion, "_scale_timesteps")(t_long)

        past_bjct = self.btjc_to_bjct(past_norm_btjc)
        pred_bjct = self.model.forward(x_bjct, t_scaled, past_bjct, sign_img)
        pred_btjc = self.bjct_to_btjc(pred_bjct)
        
        return pred_btjc

    def _predict_autoregressive(self, past_norm_btjc, sign_img, future_len, chunk_size=5):
        """自回归预测"""
        B, T_past, J, C = past_norm_btjc.shape
        device = past_norm_btjc.device
        
        predictions = []
        current_history = past_norm_btjc.clone()
        
        remaining = future_len
        while remaining > 0:
            n = min(chunk_size, remaining)
            pred_chunk = self._predict_frames(current_history, sign_img, n)
            predictions.append(pred_chunk.clone())
            
            current_history = torch.cat([current_history, pred_chunk], dim=1)
            current_history = current_history[:, -T_past:]
            
            remaining -= n
        
        return torch.cat(predictions, dim=1)

    def training_step(self, batch, batch_idx):
        debug = self._step_count == 0 or self._step_count % 500 == 0

        cond_raw = batch["conditions"]
        gt_btjc = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(cond_raw["input_pose"])
        sign_img = cond_raw["sign_image"].float()

        if debug:
            print("\n" + "=" * 70)
            print(f"TRAINING STEP {self._step_count}")
            print("=" * 70)

        gt = self.normalize(gt_btjc)
        past = self.normalize(past_btjc)

        B, T_future, J, C = gt.shape

        # Direct 预测
        pred_norm = self._predict_frames(past, sign_img, T_future)
        loss_main = torch.nn.functional.mse_loss(pred_norm, gt)

        pred_raw = self.unnormalize(pred_norm)
        gt_raw = self.unnormalize(gt)

        # Velocity & acceleration losses
        loss_vel = torch.tensor(0.0, device=self.device)
        loss_acc = torch.tensor(0.0, device=self.device)

        if pred_raw.size(1) > 1:
            v_pred = pred_raw[:, 1:] - pred_raw[:, :-1]
            v_gt = gt_raw[:, 1:] - gt_raw[:, :-1]
            loss_vel = torch.nn.functional.mse_loss(v_pred, v_gt)
            
            if v_pred.size(1) > 1:
                a_pred = v_pred[:, 1:] - v_pred[:, :-1]
                a_gt = v_gt[:, 1:] - v_gt[:, :-1]
                loss_acc = torch.nn.functional.mse_loss(a_pred, a_gt)

        # 手指正则化损失
        loss_hand = torch.tensor(0.0, device=self.device)
        if self.hand_reg_weight > 0:
            abnormal_joints = list(range(136, 178))  # 所有手部关节
            if J > max(abnormal_joints):
                pred_hand = pred_norm[:, :, abnormal_joints, :]
                gt_hand = gt[:, :, abnormal_joints, :]
                loss_hand = torch.nn.functional.mse_loss(pred_hand, gt_hand)

        loss = (loss_main 
                + self.vel_weight * loss_vel 
                + self.acc_weight * loss_acc
                + self.hand_reg_weight * loss_hand)

        if debug:
            disp_pred = mean_frame_disp(pred_raw)
            disp_gt = mean_frame_disp(gt_raw)
            disp_ratio_val = disp_pred / (disp_gt + 1e-8)
            print(f"loss_main: {loss_main.item():.6f}")
            print(f"loss_vel: {loss_vel.item():.6f}, loss_acc: {loss_acc.item():.6f}")
            print(f"loss_hand: {loss_hand.item():.6f}")
            print(f"disp: pred={disp_pred:.6f}, gt={disp_gt:.6f}, ratio={disp_ratio_val:.4f}")
            print(f"TOTAL: {loss.item():.6f}")
            print("=" * 70)

        self.log_dict({
            "train/loss": loss,
            "train/mse": loss_main,
            "train/vel": loss_vel,
            "train/acc": loss_acc,
            "train/hand": loss_hand,
        }, prog_bar=True)

        self._step_count += 1
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        cond_raw = batch["conditions"]
        gt_btjc = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(cond_raw["input_pose"])
        sign_img = cond_raw["sign_image"].float()
        
        gt = self.normalize(gt_btjc)
        past = self.normalize(past_btjc)
        
        B, T_future, J, C = gt.shape
        
        # Direct 预测
        pred_norm = self._predict_frames(past, sign_img, T_future)
        loss_main = torch.nn.functional.mse_loss(pred_norm, gt)
        
        # 计算指标
        pred_raw = self.unnormalize(pred_norm)
        gt_raw = self.unnormalize(gt)
        
        # disp ratio
        disp_pred = mean_frame_disp(pred_raw)
        disp_gt = mean_frame_disp(gt_raw)
        disp_ratio = disp_pred / (disp_gt + 1e-8)
        
        # PCK
        pred_np = pred_raw.cpu().numpy()
        gt_np = gt_raw.cpu().numpy()
        per_joint_error = np.sqrt(((pred_np - gt_np) ** 2).sum(axis=-1))
        pck_01 = (per_joint_error < 0.1).mean()
        
        self.log_dict({
            "val/loss": loss_main,
            "val/disp_ratio": disp_ratio,
            "val/pck_01": pck_01,
        }, prog_bar=True, sync_dist=True)
        
        return loss_main

    @torch.no_grad()
    def predict_direct(self, past_btjc, sign_img, future_len=20, use_autoregressive=True):
        """推理"""
        self.eval()
        device = self.device

        past_raw = sanitize_btjc(past_btjc.to(device))
        past_norm = self.normalize(past_raw)

        if use_autoregressive:
            pred_norm = self._predict_autoregressive(past_norm, sign_img.to(device), future_len)
        else:
            pred_norm = self._predict_frames(past_norm, sign_img.to(device), future_len)

        return self.unnormalize(pred_norm)

    def on_train_start(self):
        self.mean_pose = self.mean_pose.to(self.device)
        self.std_pose = self.std_pose.to(self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
            "gradient_clip_val": 1.0,
        }

    def on_train_end(self):
        try:
            import matplotlib.pyplot as plt
            out_dir = self.trainer.default_root_dir or "logs/multi_sample"
            os.makedirs(out_dir, exist_ok=True)

            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            ax[0, 0].plot(self.train_logs["loss"])
            ax[0, 0].set_title("loss")
            ax[0, 1].plot(self.train_logs["mse"])
            ax[0, 1].set_title("mse")
            ax[1, 0].plot(self.train_logs["vel"])
            ax[1, 0].set_title("vel")
            ax[1, 1].plot(self.train_logs["acc"])
            ax[1, 1].set_title("acc")
            plt.tight_layout()
            plt.savefig(f"{out_dir}/train_curve.png")
            print(f"[TRAIN CURVE] saved to {out_dir}/train_curve.png")
        except Exception as e:
            print(f"[TRAIN CURVE] failed: {e}")


# Alias
LitMinimal = LitResidual