# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from torch.utils.data import DataLoader
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc


# ----------------------------- helpers -----------------------------

def _plain(x):
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().cpu().float().contiguous()

def temporal_smooth(x, k=5):
    import torch.nn.functional as F
    if x.dim() == 4:
        x = x[0]
    T, J, C = x.shape
    x = x.permute(2,1,0).reshape(1, C*J, T)
    x = F.avg_pool1d(x, k, stride=1, padding=k//2)
    x = x.reshape(C,J,T).permute(2,1,0)
    return x.unsqueeze(0)

def recenter_pair(gt, pr):
    if gt.dim() == 4: gt = gt[0]
    if pr.dim() == 4: pr = pr[0]

    gt = torch.nan_to_num(gt)
    pr = torch.nan_to_num(pr)

    torso = gt[:, :33, :2].reshape(-1,2)
    center = torso.median(dim=0).values
    gt[..., :2] -= center
    pr[..., :2] -= center

    pts = torch.cat([gt[..., :2].reshape(-1,2), pr[..., :2].reshape(-1,2)], dim=0)
    q02 = torch.quantile(pts, 0.02, dim=0)
    q98 = torch.quantile(pts, 0.98, dim=0)

    scale = 450.0 / (q98 - q02).clamp(min=50.0).max()
    gt[..., :2] *= scale
    pr[..., :2] *= scale

    gt[...,0]+=256; gt[...,1]+=256
    pr[...,0]+=256; pr[...,1]+=256

    return gt.unsqueeze(0), pr.unsqueeze(0)

def tensor_to_pose(x_btjc, header):
    if x_btjc.dim() == 4:
        x_btjc = x_btjc[0]
    arr = np.ascontiguousarray(x_btjc[:,None,:,:], dtype=np.float32)
    conf = np.ones((arr.shape[0],1,arr.shape[2],1), dtype=np.float32)
    return Pose(header=header, body=NumPyPoseBody(fps=25, data=arr, confidence=conf))


# ----------------------------- main -----------------------------
if __name__ == "__main__":

    # === paths ===
    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    ckpt_path = "/data/yayun/.../last.ckpt"     # ← 你训练好的 checkpoint

    out_dir = "logs/minimal_eval"
    os.makedirs(out_dir, exist_ok=True)

    # === dataset: tiny subset ===
    ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=60,
        num_future_frames=30,
        with_metadata=True,
        split="dev",
        reduce_holistic=True,
    )
    ds = torch.utils.data.Subset(ds, [0])   # 只抽 1 个样本即可

    loader = DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=zero_pad_collator
    )

    # === load Lightning model ===
    model = LitMinimal.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to("cuda")

    print("Loaded model ✔")

    # === inference ===
    batch = next(iter(loader))
    cond = batch["conditions"]

    past_raw = sanitize_btjc(cond["input_pose"]).to("cuda")
    fut_raw  = sanitize_btjc(batch["data"]).to("cuda")
    sign_img = cond["sign_image"].to("cuda")

    future_len = fut_raw.size(1)

    with torch.no_grad():
        pred_norm = model.sample_autoregressive_fast(
            past_btjc=past_raw,
            sign_img=sign_img,
            future_len=future_len,
            chunk=1
        )

    # === unnormalize & postprocess ===
    fut_un  = _plain(model.unnormalize(fut_raw))
    pred_un = _plain(model.unnormalize(pred_norm))

    pred_s  = temporal_smooth(pred_un)
    fut_s   = fut_un.unsqueeze(0)

    fut_vis, pred_vis = recenter_pair(fut_s, pred_s)

    # === save pose ===
    pose_path = batch["pose_path"][0]
    with open(os.path.join(data_dir, pose_path), "rb") as f:
        pose0 = Pose.read(f)
    header = reduce_holistic(pose0.remove_components(["POSE_WORLD_LANDMARKS"])).header

    out_gt   = os.path.join(out_dir, "gt_178.pose")
    out_pred = os.path.join(out_dir, "pred_178.pose")

    tensor_to_pose(fut_vis, header).write(open(out_gt, "wb"))
    tensor_to_pose(pred_vis, header).write(open(out_pred, "wb"))

    print("[SAVE] GT:", out_gt)
    print("[SAVE] Pred:", out_pred)
    print("Done ✔")
