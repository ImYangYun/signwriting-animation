# -*- coding: utf-8 -*-
"""
MINIMAL LOOP — FAST PREDICTOR VERSION (方案一)
NO diffusion, NO p_sample_loop, NO autoregressive.
Speed: training <10min, sampling <5s.
"""

import os
import torch
import numpy as np
import lightning as pl
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

from torch.utils.data import DataLoader
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.utils.generic import reduce_holistic

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import sanitize_btjc


# ============================================================
# FAST PREDICTOR (NO DIFFUSION)
# ============================================================
class FastFuturePredictor(pl.LightningModule):
    """
    使用 CAMDM 的 encoder + cross-attention，但不 diffusion。
    直接一次性预测未来序列 → 速度极快。
    Training loss = MSE + VEL.
    """

    def __init__(self, num_keypoints=178, num_dims=3, lr=1e-4,
                 stats_path="/data/yayun/pose_data/mean_std_178.pt"):
        super().__init__()
        self.save_hyperparameters()

        stats = torch.load(stats_path)
        self.register_buffer("mean_pose", stats["mean"].view(1,1,1,3).float())
        self.register_buffer("std_pose",  stats["std"].view(1,1,1,3).float())

        # 使用 CAMDM 的 base encoder
        from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
        self.backbone = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
        )

        # 一个简单的 MLP 预测 future sequence
        hidden = num_keypoints * num_dims
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden)
        )

        self.lr = lr

    # ---------------------- norm ----------------------
    def normalize(self, x):
        return (x - self.mean_pose) / (self.std_pose + 1e-6)

    def unnormalize(self, x):
        return x * self.std_pose + self.mean_pose

    # ---------------------- forward ----------------------
    def forward(self, past_btjc, future_len, sign_img):
        """
        输入：
            past_btjc: [B, Tp, J, C]
        输出：
            pred: [B, Tf, J, C]
        """
        B,Tp,J,C = past_btjc.shape

        # backbone encode
        enc = self.backbone.encoder_only(
            self.normalize(past_btjc),       # 输入
            sign_img                         # 图片
        )                                     # 输出 shape: [B, J*C]

        x = self.mlp(enc)                     # [B, J*C]
        x = x.view(B, 1, J, C)                # 只预测一帧

        # repeat → 生成 full future sequence（non-diffusion forward）
        x = x.repeat(1, future_len, 1, 1)

        return self.unnormalize(x)

    # ---------------------- training ----------------------
    def training_step(self, batch, _):
        cond = batch["conditions"]
        fut  = sanitize_btjc(batch["data"]).float()
        past = sanitize_btjc(cond["input_pose"]).float()
        sign = cond["sign_image"].float()

        B,Tf,J,C = fut.shape

        pred = self.forward(past, Tf, sign)

        loss_pos = torch.nn.functional.mse_loss(pred, fut)

        vel_gt   = fut[:,1:] - fut[:,:-1]
        vel_pred = pred[:,1:] - pred[:,:-1]
        loss_vel = torch.nn.functional.l1_loss(vel_pred, vel_gt)

        loss = loss_pos + 0.3 * loss_vel

        self.log("train/loss", loss, prog_bar=True)
        return loss

    # ---------------------- val ----------------------
    def validation_step(self, batch, _):
        cond = batch["conditions"]
        fut  = sanitize_btjc(batch["data"]).float()
        past = sanitize_btjc(cond["input_pose"]).float()
        sign = cond["sign_image"].float()

        B,Tf,J,C = fut.shape
        pred = self.forward(past, Tf, sign)

        loss = torch.nn.functional.mse_loss(pred, fut)
        self.log("val/loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)



# ============================================================
# Helper functions
# ============================================================
def _plain(x):
    if hasattr(x,"tensor"): x = x.tensor
    if hasattr(x,"zero_filled"): x = x.zero_filled()
    return x.detach().cpu().float().contiguous()

def temporal_smooth(x, k=5):
    import torch.nn.functional as F
    if x.dim()==4: x=x[0]
    T,J,C = x.shape
    x = x.permute(2,1,0).reshape(1, C*J, T)
    x = F.avg_pool1d(x, k, stride=1, padding=k//2)
    x = x.reshape(C,J,T).permute(2,1,0).contiguous()
    return x.unsqueeze(0)

def recenter_pair(gt, pr):
    if gt.dim()==4: gt = gt[0]
    if pr.dim()==4: pr = pr[0]
    gt = torch.nan_to_num(gt)
    pr = torch.nan_to_num(pr)

    torso = gt[:, :33, :2].reshape(-1,2)
    center = torso.median(dim=0).values
    gt[..., :2] -= center
    pr[..., :2] -= center

    pts = torch.cat([gt[..., :2].reshape(-1,2),
                     pr[..., :2].reshape(-1,2)], dim=0)

    q02 = torch.quantile(pts, 0.02, dim=0)
    q98 = torch.quantile(pts, 0.98, dim=0)
    scale = 450.0 / (q98-q02).clamp(min=50.0).max()

    gt[..., :2] *= scale
    pr[..., :2] *= scale
    gt[...,0]+=256; gt[...,1]+=256
    pr[...,0]+=256; pr[...,1]+=256
    return gt.unsqueeze(0), pr.unsqueeze(0)

def tensor_to_pose(x_btjc, header):
    if x_btjc.dim()==4: x_btjc=x_btjc[0]
    arr = np.ascontiguousarray(x_btjc[:,None,:,:], dtype=np.float32)
    conf = np.ones((arr.shape[0],1,arr.shape[2],1), dtype=np.float32)
    return Pose(header=header, body=NumPyPoseBody(fps=25, data=arr, confidence=conf))



# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    pl.seed_everything(42)
    torch.use_deterministic_algorithms(False)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    mean_std_178 = os.path.join(data_dir, "mean_std_178.pt")

    out_dir = "logs/minimal_178_predictor"
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------
    # Dataset
    # -------------------------------
    def make_loader(split, subset=16):
        ds = DynamicPosePredictionDataset(
            data_dir=data_dir,
            csv_path=csv_path,
            num_past_frames=60,
            num_future_frames=30,
            with_metadata=True,
            split=split,
            reduce_holistic=True,
        )
        if subset is not None:
            ds = torch.utils.data.Subset(ds, list(range(min(len(ds),subset))))

        return DataLoader(
            ds, batch_size=4, shuffle=(split=="train"),
            num_workers=0, collate_fn=zero_pad_collator
        )

    train_loader = make_loader("train")
    val_loader   = make_loader("dev")

    print("[INFO] train:", len(train_loader.dataset))
    print("[INFO] val:", len(val_loader.dataset))

    # -------------------------------
    # Model
    # -------------------------------
    model = FastFuturePredictor(
        num_keypoints=178,
        num_dims=3,
        stats_path=mean_std_178,
    )

    trainer = pl.Trainer(
        default_root_dir=out_dir,
        accelerator="gpu",
        devices=1,
        max_epochs=20,
        log_every_n_steps=5,
        enable_checkpointing=False,
    )

    # -------------------------------
    # TRAIN
    # -------------------------------
    print("\n===== TRAINING =====")
    trainer.fit(model, train_loader, val_loader)
    print("===== TRAIN DONE =====")


    # -------------------------------
    # SAMPLE
    # -------------------------------
    print("\n===== SAMPLING =====")

    batch = next(iter(val_loader))
    cond = batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(model.device)
    fut_raw  = sanitize_btjc(batch["data"][:1]).to(model.device)
    sign_img = cond["sign_image"][:1].to(model.device)

    Tf = fut_raw.size(1)
    pred = model.forward(past_raw, Tf, sign_img)

    # -------------------------------
    # VIS
    # -------------------------------
    fut_un  = _plain(fut_raw)
    pred_un = _plain(pred)

    pred_s  = temporal_smooth(pred_un)
    fut_s   = fut_un.unsqueeze(0)

    fut_vis, pred_vis = recenter_pair(fut_s, pred_s)

    pose_path = batch["pose_path"][0]
    with open(os.path.join(data_dir, pose_path),"rb") as f:
        pose0 = Pose.read(f)
    header = reduce_holistic(pose0.remove_components(["POSE_WORLD_LANDMARKS"])).header

    out_gt   = os.path.join(out_dir, "gt_178.pose")
    out_pred = os.path.join(out_dir, "pred_178.pose")

    tensor_to_pose(fut_vis, header).write(open(out_gt,"wb"))
    tensor_to_pose(pred_vis, header).write(open(out_pred,"wb"))

    print("[SAVE] GT:", out_gt)
    print("[SAVE] Pred:", out_pred)
    print("\n===== DONE =====")
