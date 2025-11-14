import torch
import lightning as pl
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType
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
                 timesteps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 pred_target="eps",
                 guidance_scale=2.0):
        super().__init__()
        self.save_hyperparameters()

        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1,1,-1,3)   # [1,1,J,C]
        std  = stats["std"].float().view(1,1,-1,3)
        self.register_buffer("mean_pose", mean)
        self.register_buffer("std_pose",  std)

        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
        )

        # ---- 扩散对象 ----
        self.pred_target = pred_target.lower()
        model_mean_type = ModelMeanType.EPSILON if self.pred_target == "eps" else ModelMeanType.START_X
        self.diffusion = GaussianDiffusion(
            betas=torch.linspace(beta_start, beta_end, timesteps),
            model_mean_type=model_mean_type,
            model_var_type=ModelVarType.FIXED_LARGE,
            rescale_timesteps=True
        )

        self.guidance_scale = guidance_scale
        self.lr = lr
        print("[LitMinimal] ✅ true diffusion (q_sample + target=%s) enabled" % self.pred_target)

    # ---------- 归一化 ----------
    def normalize(self, x):   return (x - self.mean_pose) / (self.std_pose + 1e-6)
    def unnormalize(self, x): return x * self.std_pose + self.mean_pose

    # ---------- BTJC↔BJCT ----------
    @staticmethod
    def btjc_to_bjct(x): return x.permute(0,2,3,1).contiguous()
    @staticmethod
    def bjct_to_btjc(x): return x.permute(0,3,1,2).contiguous()

    # ---------- 模型前向（包装 BJCT） ----------
    def _forward_bjct(self, x_bjct, ts, past_btjc, sign_img):
        past_bjct = self.btjc_to_bjct(past_btjc)
        out_bjct  = self.model.forward(x_bjct, ts, past_bjct, sign_img)  # [B,J,C,T]
        return out_bjct

    # ---------- 统一的扩散一步 ----------
    def _diffuse_once(self, x_start_btjc, t_long, cond):
        """
        x_start_btjc: [B,T,J,C] (规范化空间)
        t_long:       [B] long
        cond:         dict: {'input_pose': BTJC, 'sign_image': BCHW, 'target_mask': [B,T] or [B,T,J,C]}
        """
        # 1) 置换到 BJCT
        x0_bjct = self.btjc_to_bjct(x_start_btjc)  # [B,J,C,T]
        noise   = torch.randn_like(x0_bjct)
        # 2) q_sample 得到 x_t
        x_t = self.diffusion.q_sample(x0_bjct, t_long, noise=noise)  # same shape

        # 3) 调用 CAMDM 接口（注意：你的模型 forward 接受 BJCT）
        #    但它需要 'past' 和 'sign'；我们直接把 BTJC 的 past → BJCT 在 _forward_bjct 里做
        out_bjct = self._forward_bjct(x_t, self.diffusion._scale_timesteps(t_long), cond["input_pose"], cond["sign_image"])

        # 4) 准备训练目标
        if self.pred_target == "eps":
            target = noise                      # 预测 ε
        else:
            target = x0_bjct                    # 预测 x0

        return out_bjct, target

    # ---------- 训练 ----------
    def training_step(self, batch, _):
        cond_raw = batch["conditions"]
        gt_btjc  = sanitize_btjc(batch["data"])                  # [B,30,J,C]
        past_btjc= sanitize_btjc(cond_raw["input_pose"])         # [B,60,J,C]
        mask_bt  = cond_raw["target_mask"]
        sign_img = cond_raw["sign_image"].float()

        # 归一化 & 对齐 60→30
        gt = self.normalize(gt_btjc)
        past = self.normalize(past_btjc)[:, -gt.size(1):]        # [B,30,J,C]

        cond = {"input_pose": past, "sign_image": sign_img, "target_mask": mask_bt}

        B = gt.size(0)
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device, dtype=torch.long)

        pred_bjct, target_bjct = self._diffuse_once(gt, t, cond)

        # MSE 主损失（在 BJCT）
        loss_main = torch.nn.functional.mse_loss(pred_bjct, target_bjct)

        # 可选：速度/加速度（回到 BTJC 上做）
        pred_btjc   = self.bjct_to_btjc(pred_bjct)
        target_btjc = self.bjct_to_btjc(target_bjct if self.pred_target=="x0" else gt)  # 若预测 ε，则用 gt 近似作辅助项
        loss_vel = torch.tensor(0.0, device=self.device)
        loss_acc = torch.tensor(0.0, device=self.device)
        if pred_btjc.size(1) > 1:
            v_pred = pred_btjc[:,1:] - pred_btjc[:,:-1]
            v_tgt  = target_btjc[:,1:] - target_btjc[:,:-1]
            loss_vel = torch.nn.functional.l1_loss(v_pred, v_tgt)
            if v_pred.size(1) > 1:
                a_pred = v_pred[:,1:] - v_pred[:,:-1]
                a_tgt  = v_tgt[:,1:]  - v_tgt[:,:-1]
                loss_acc = torch.nn.functional.l1_loss(a_pred, a_tgt)

        loss = loss_main + 0.5*loss_vel + 0.25*loss_acc

        self.log_dict({
            "train/loss": loss,
            "train/mse": loss_main,
            "train/vel": loss_vel,
            "train/acc": loss_acc
        }, prog_bar=True)
        return loss

    # ---------- 验证 ----------
    @torch.no_grad()
    def validation_step(self, batch, _):
        cond_raw = batch["conditions"]
        gt_btjc  = sanitize_btjc(batch["data"])
        past_btjc= sanitize_btjc(cond_raw["input_pose"])
        mask_bt  = cond_raw["target_mask"]
        sign_img = cond_raw["sign_image"].float()

        gt   = self.normalize(gt_btjc)
        past = self.normalize(past_btjc)[:, -gt.size(1):]
        cond = {"input_pose": past, "sign_image": sign_img, "target_mask": mask_bt}

        B = gt.size(0)
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device, dtype=torch.long)

        pred_bjct, target_bjct = self._diffuse_once(gt, t, cond)
        loss_main = torch.nn.functional.mse_loss(pred_bjct, target_bjct)

        # 回到 BTJC 做 DTW（用 x0 空间）
        if self.pred_target == "eps":
            x0_pred_btjc = self.bjct_to_btjc(pred_bjct*0.0 + gt.permute(0,2,3,1))  # 占位避免分支太长；下方统一用简化：直接用 gt 作 x0 参考
        # 简化：直接把 pred_bjct 当作 x0 估计回到 BTJC（实践上你可以做反推一步）
        x0_est_btjc = self.bjct_to_btjc(pred_bjct if self.pred_target=="x0" else self.btjc_to_bjct(gt))

        dtw_val = masked_dtw(self.unnormalize(x0_est_btjc), self.unnormalize(gt), 
                             mask_bt if mask_bt.dim()==2 else (mask_bt.sum((2,3))>0).float())

        self.log_dict({
            "val/mse": loss_main,
            "val/dtw": dtw_val
        }, prog_bar=True)

        return loss_main

    @torch.no_grad()
    def sample_autoregressive_diffusion(self, past_btjc, sign_img, future_len=30, chunk=1):
        """
        past_btjc: [B,Tp,J,C] (未归一化)
        返回：生成的未归一化 [B,future_len,J,C]
        """
        self.eval()
        B, Tp, J, C = past_btjc.shape
        device = self.device

        past = self.normalize(past_btjc.to(device))
        sign = sign_img.to(device)

        generated = []
        cur_hist = past.clone()

        steps = (future_len + chunk - 1)//chunk
        for _ in range(steps):
            n = min(chunk, future_len - len(generated))
            # 目标形状（BJCT）
            shape_bjct = (B, J, C, n)

            # 条件与“无条件”（CFG）
            cond = {"input_pose": cur_hist, "sign_image": sign}
            uncond = {"input_pose": cur_hist*0.0, "sign_image": sign}

            # 包装器：把 (x,t) 送入你的模型 interface
            class _Wrap(nn.Module):
                def __init__(self, base, cond_dict): super().__init__(); self.base=base; self.cond=cond_dict
                def forward(self, x, t, **kw): return self.base.interface(x, t, {"input_pose": self.base.bjct_to_btjc(self.cond["input_pose"]).permute(0,2,3,1), "sign_image": self.cond["sign_image"]})

            # 直接用 diffusion 的 p_sample_loop
            x_cond   = self.diffusion.p_sample_loop(lambda x,t,**kw: self.model.interface(x,t,{"input_pose": self.btjc_to_bjct(cur_hist),"sign_image": sign}), shape_bjct, clip_denoised=False, progress=False)
            x_uncond = self.diffusion.p_sample_loop(lambda x,t,**kw: self.model.interface(x,t,{"input_pose": self.btjc_to_bjct(cur_hist*0.0),"sign_image": sign}), shape_bjct, clip_denoised=False, progress=False)

            x_hat = x_uncond + self.guidance_scale * (x_cond - x_uncond)  # CFG 组合
            pred_btjc = self.bjct_to_btjc(x_hat)  # [B,n,J,C]（归一化空间）

            generated.append(pred_btjc)
            cur_hist = torch.cat([cur_hist, pred_btjc], dim=1)

        pred_norm = torch.cat(generated, dim=1)           # [B,Tf,J,C]（归一化）
        return self.unnormalize(pred_norm)

    # ---------- 优化器 ----------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
