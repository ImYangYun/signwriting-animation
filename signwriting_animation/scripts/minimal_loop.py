# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader
import atexit

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.utils.generic import reduce_holistic
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc


try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass
atexit.register(lambda: sys.stdout.flush())

np.set_printoptions(suppress=True, linewidth=120, precision=4)
torch.set_printoptions(sci_mode=False, precision=4)


# ========================================================================
# Helper functions
# ========================================================================
def unnormalize_tensor_with_global_stats(t, mean_std):
    mean = torch.tensor(mean_std["mean"], dtype=torch.float32, device=t.device)
    std = torch.tensor(mean_std["std"], dtype=torch.float32, device=t.device)
    return t * std + mean


def temporal_smooth(x, k=5):
    import torch.nn.functional as F
    if x.dim() == 4:
        x = x[0]
    T, J, C = x.shape
    x2 = x.permute(2, 1, 0).reshape(1, C*J, T)
    x2 = F.avg_pool1d(x2, kernel_size=k, stride=1, padding=k//2)
    return x2.reshape(C, J, T).permute(2, 1, 0).contiguous()


def recenter_for_view(x):
    if x.dim() == 4:
        x = x[0]
    x = torch.nan_to_num(x, nan=0.0)
    T, J, C = x.shape

    torso = x[:, :33, :2]
    center = torso.reshape(-1, 2).median(dim=0).values
    x[..., :2] -= center

    flat = x[..., :2].reshape(-1, 2)
    q02 = torch.quantile(flat, 0.02, dim=0)
    q98 = torch.quantile(flat, 0.98, dim=0)
    span = (q98 - q02).clamp(min=1.0)

    ox = 512 - span[0] / 2
    oy = 384 - span[1] / 2
    x[..., 0] += ox
    x[..., 1] += oy
    return x


def tensor_to_pose(t_btjc, header):
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc
    arr = np.ascontiguousarray(t[:, None, :, :], dtype=np.float32)
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)
    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    return Pose(header=header, body=body)


# ========================================================================
# Main
# ========================================================================
if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/overfit_178vis"
    os.makedirs(out_dir, exist_ok=True)

    # ============= Load 586-joint dataset ======================
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=60,
        num_future_frames=30,
        with_metadata=True,
        split="train",
        reduce_holistic=False,
    )

    small_ds = torch.utils.data.Subset(base_ds, list(range(10)))
    loader = DataLoader(small_ds, batch_size=1, shuffle=False, collate_fn=zero_pad_collator)

    batch = next(iter(loader))
    cond = batch["conditions"]

    past_raw = sanitize_btjc(cond["input_pose"][:1])
    fut_raw  = sanitize_btjc(batch["data"][:1])

    print("\n================ ENTERING DEBUG MODE ================\n")
    print("past_raw shape =", past_raw.shape)
    print("fut_raw shape  =", fut_raw.shape)

    # ====================================================================
    # HEADER DEBUG (586 → 178)
    # ====================================================================
    print("\n================ HEADER DEBUG ================\n")

    pose_path = base_ds.records[0]["pose"]
    pose_abspath = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)
    with open(pose_abspath, "rb") as f:
        pose_full = Pose.read(f)

    print("[HEADER] Full components:")
    for c in pose_full.header.components:
        print(f" - {c.name:30s} points={len(c.points):3d}")

    # Build 178 header
    pose_now = pose_full.remove_components(["POSE_WORLD_LANDMARKS"])
    pose_178 = reduce_holistic(pose_now)
    header_178 = pose_178.header

    print("\n[HEADER] Reduced 178 components:")
    for c in header_178.components:
        print(f" - {c.name:30s} points={len(c.points):3d}")

    # Build index_map
    name2idx586 = {}
    base = 0
    for comp in pose_full.header.components:
        for p in comp.points:
            name2idx586[(comp.name, p)] = base
            base += 1

    index_map = []
    for comp in header_178.components:
        for p in comp.points:
            index_map.append(name2idx586[(comp.name, p)])
    idx_t = torch.tensor(index_map, dtype=torch.long)

    print("\n[CHECK] index_map length =", len(index_map))
    print("================ END HEADER DEBUG ================\n")

    # ====================================================================
    #  Model (just for inference)
    # ====================================================================
    model = LitMinimal(
        num_keypoints=586,
        num_dims=3,
        lr=3e-4,
        stats_path=os.path.join(data_dir, "mean_std_586.pt"),
        diffusion_steps=20,
        pred_target="x0",
    )
    model.eval()

    sign_img = cond["sign_image"][:1]
    pred_full = model.sample_autoregressive_diffusion(
        past_btjc=past_raw, sign_img=sign_img, future_len=fut_raw.size(1), chunk=1
    )

    # ====================================================================
    print("\n[STEP] Unnormalize 586 → Reduce to 178")

    mean_std = torch.load(os.path.join(data_dir, "mean_std_586.pt"))
    fut_un_full  = unnormalize_tensor_with_global_stats(fut_raw,  mean_std)    # [1,T,586,3]
    pred_un_full = unnormalize_tensor_with_global_stats(pred_full, mean_std)   # [1,T,586,3]

    print("fut_un_full  shape =", fut_un_full.shape)
    print("pred_un_full shape =", pred_un_full.shape)

    fut_un_178  = fut_un_full.index_select(2, idx_t)      # [1,T,178,3]
    pred_un_178 = pred_un_full.index_select(2, idx_t)     # [1,T,178,3]

    print("fut_un_178  shape =", fut_un_178.shape)
    print("pred_un_178 shape =", pred_un_178.shape)

    fut_un_178  = torch.clamp(fut_un_178,  -3,  3)
    pred_un_178 = torch.clamp(pred_un_178, -3,  3)

    fut_un_178  = temporal_smooth(fut_un_178)
    pred_un_178 = temporal_smooth(pred_un_178)

    fut_vis  = recenter_for_view(fut_un_178)
    pred_vis = recenter_for_view(pred_un_178)

    # ===== clamp =====
    # ===== 3) clamp =====
    fut_un_178  = torch.clamp(fut_un_178,  -3,  3)
    pred_un_178 = torch.clamp(pred_un_178, -3,  3)

    # ===== 4) smooth =====
    fut_un_178  = temporal_smooth(fut_un_178)
    pred_un_178 = temporal_smooth(pred_un_178)

    # ===== 5) recenter =====
    fut_vis  = recenter_for_view(fut_un_178)
    pred_vis = recenter_for_view(pred_un_178)

    # ====================================================================
    # SAVE pose files
    # ====================================================================
    out_gt   = os.path.join(out_dir, "gt_178.pose")
    out_pred = os.path.join(out_dir, "pred_178.pose")

    tensor_to_pose(fut_vis, header_178).write(open(out_gt, "wb"))
    tensor_to_pose(pred_vis, header_178).write(open(out_pred, "wb"))

    print(f"\n[SAVE] Saved -> {out_gt}")
    print(f"[SAVE] Saved -> {out_pred}\n")

    print("=============== DONE ===============\n")
