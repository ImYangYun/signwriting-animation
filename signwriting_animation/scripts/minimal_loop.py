# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import lightning as pl
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

from torch.utils.data import DataLoader
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def _plain(x):
    if hasattr(x, "tensor"): x = x.tensor
    if hasattr(x, "zero_filled"): x = x.zero_filled()
    return x.detach().cpu().float().contiguous()

def temporal_smooth(x, k=5):
    import torch.nn.functional as F
    if x.dim() == 4:
        x = x[0]
    T,J,C = x.shape
    x = x.permute(2,1,0).reshape(1, C*J, T)
    x = F.avg_pool1d(x, k, stride=1, padding=k//2)
    x = x.reshape(C, J, T).permute(2,1,0).contiguous()
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

    pts = torch.cat([
        gt[..., :2].reshape(-1,2),
        pr[..., :2].reshape(-1,2)
    ], dim=0)

    q02 = torch.quantile(pts, 0.02, dim=0)
    q98 = torch.quantile(pts, 0.98, dim=0)
    span = (q98 - q02).clamp(min=50.0)

    scale = min(450.0 / span.max(), 5.0)
    gt[..., :2] *= scale
    pr[..., :2] *= scale

    gt[...,0] += 256; gt[...,1] += 256
    pr[...,0] += 256; pr[...,1] += 256

    return gt.unsqueeze(0), pr.unsqueeze(0)


def tensor_to_pose(x_btjc, header):
    if x_btjc.dim()==4:
        x_btjc = x_btjc[0]
    arr = np.ascontiguousarray(x_btjc[:,None,:,:], dtype=np.float32)
    conf = np.ones((arr.shape[0],1,arr.shape[2],1), dtype=np.float32)
    return Pose(header=header,
                body=NumPyPoseBody(fps=25, data=arr, confidence=conf))


# ---------------------------------------------------------
# PATCH: final fast autoregressive sampling
# ---------------------------------------------------------
@torch.no_grad()
def sample_autoregressive_fast(self, past_btjc, sign_img,
                               future_len: int = 30, chunk: int = 1,
                               guidance_scale: float = None):

    self.eval()
    device = self.device

    if guidance_scale is None:
        guidance_scale = self.guidance_scale

    # Normalize past
    past_norm = self.normalize(past_btjc.to(device))
    sign = sign_img.to(device)
    B, Tp, J, C = past_norm.shape

    # Wrapper for CAMDM
    class _Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x, t, **kwargs):
            y = kwargs["y"]
            return self.model.interface(x, t, y)

    wrapped = _Wrapper(self.model).to(device)

    frames = []
    remain = int(future_len)
    cur_hist = past_norm.clone()     # [B,Tp,J,C]

    while remain > 0:
        n = min(chunk, remain)
        shape_bjct = (B, J, C, n)

        cond_dict = {
            "sign_image": sign,
            "input_pose": self.btjc_to_bjct(cur_hist),
        }
        uncond_dict = {
            "sign_image": torch.zeros_like(sign),
            "input_pose": torch.zeros_like(cond_dict["input_pose"]),
        }

        # conditional
        x_cond = self.diffusion.p_sample_loop(
            model=wrapped, shape=shape_bjct,
            model_kwargs={"y": cond_dict},
            clip_denoised=False, progress=False,
        )

        # unconditional guidance
        if guidance_scale and guidance_scale > 0:
            x_uncond = self.diffusion.p_sample_loop(
                model=wrapped, shape=shape_bjct,
                model_kwargs={"y": uncond_dict},
                clip_denoised=False, progress=False,
            )
            x_bjct = x_uncond + guidance_scale * (x_cond - x_uncond)
        else:
            x_bjct = x_cond

        x_btjc_norm = self.bjct_to_btjc(x_bjct)
        frames.append(x_btjc_norm)

        cur_hist = torch.cat([cur_hist, x_btjc_norm], dim=1)
        if cur_hist.size(1) > Tp:
            cur_hist = cur_hist[:, -Tp:, :]

        remain -= n

    pred_norm = torch.cat(frames, dim=1)
    return pred_norm   # 注意：返回 normalized（让外面去 unnormalize）

# attach patch to model class
LitMinimal.sample_autoregressive_fast = sample_autoregressive_fast


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    pl.seed_everything(42)
    torch.use_deterministic_algorithms(False)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    mean_std_178 = os.path.join(data_dir, "mean_std_178.pt")

    out_dir = "logs/minimal_178_final"
    os.makedirs(out_dir, exist_ok=True)

    BATCH_SIZE = 4
    MAX_EPOCHS = 20

    # -------------------------------
    # Dataset tiny subset
    # -------------------------------
    def make_loader(split):
        ds = DynamicPosePredictionDataset(
            data_dir=data_dir,
            csv_path=csv_path,
            num_past_frames=60,
            num_future_frames=30,
            with_metadata=True,
            split=split,
            reduce_holistic=True,
        )

        # tiny dataset
        idx = list(range(min(16, len(ds))))
        ds = torch.utils.data.Subset(ds, idx)

        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=(split=="train"),
            num_workers=0,
            collate_fn=zero_pad_collator,
            pin_memory=False,
        )

    train_loader = make_loader("train")
    val_loader   = make_loader("dev")

    print("[INFO] Train samples:", len(train_loader.dataset))
    print("[INFO] Val samples:", len(val_loader.dataset))


    # -------------------------------
    # Model
    # -------------------------------
    model = LitMinimal(
        num_keypoints=178,
        num_dims=3,
        lr=1e-4,
        stats_path=mean_std_178,
        diffusion_steps=200,
        pred_target="x0",
        guidance_scale=0.0,
    )

    trainer = pl.Trainer(
        default_root_dir=out_dir,
        accelerator="gpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=5,
    )

    print("\n===== TRAINING =====")
    trainer.fit(model, train_loader, val_loader)
    print("===== TRAIN DONE =====")


    # -------------------------------
    # Sampling
    # -------------------------------
    print("\n===== SAMPLING =====")

    batch = next(iter(val_loader))
    cond = batch["conditions"]

    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(model.device)
    fut_raw  = sanitize_btjc(batch["data"][:1]).to(model.device)
    sign_img = cond["sign_image"][:1].to(model.device)

    # determine future length
    if "target_mask" in cond:
        mask = cond["target_mask"][:1]
        if mask.dim()==4:
            mask = (mask.sum((2,3))>0).float()
        true_len = int(mask.sum().item())
    else:
        true_len = fut_raw.size(1)

    # --- GENERATE ---
    pred_norm = model.sample_autoregressive_fast(
        past_btjc=past_raw,
        sign_img=sign_img,
        future_len=true_len,
        chunk=1
    )

    # unnormalize
    fut_un  = _plain(model.unnormalize(fut_raw))
    pred_un = _plain(model.unnormalize(pred_norm))

    # smooth + recenter
    pred_s  = temporal_smooth(pred_un)
    fut_s   = fut_un.unsqueeze(0)

    fut_vis, pred_vis = recenter_pair(fut_s, pred_s)

    # load header
    pose_path = batch["records"][0]["pose"]
    with open(os.path.join(data_dir, pose_path), "rb") as f:
        pose0 = Pose.read(f)
    header = reduce_holistic(
        pose0.remove_components(["POSE_WORLD_LANDMARKS"])
    ).header

    # save
    out_gt   = os.path.join(out_dir, "gt_178.pose")
    out_pred = os.path.join(out_dir, "pred_178.pose")

    tensor_to_pose(fut_vis, header).write(open(out_gt, "wb"))
    tensor_to_pose(pred_vis, header).write(open(out_pred, "wb"))

    print("[SAVE] GT:", out_gt)
    print("[SAVE] Pred:", out_pred)
    print("===== DONE =====")
