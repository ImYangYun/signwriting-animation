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


def prepare_for_visualization(x):
    """
    Stable visualization for 178-joint reduced holistic data.
    - x: [1,T,J,3] or [T,J,3]
    Output: [T,J,3]
    """
    # --- Make dense ---
    if x.dim() == 4:
        x = x[0]  # [T,J,3]
    x = torch.nan_to_num(x, nan=0.0)

    T, J, C = x.shape

    BODY_START, BODY_END = 0, 8
    FACE_START, FACE_END = 8, 8+128
    LH_START, LH_END = FACE_END, FACE_END+21
    RH_START, RH_END = LH_END, LH_END+21

    stable = torch.cat([
        x[:, BODY_START:BODY_END, :2],   # 8 body joints
        x[:, LH_START:LH_END, :2],       # left hand
        x[:, RH_START:RH_END, :2],       # right hand
    ], dim=1)  # [T,8+21+21,2] = [T,50,2]

    stable_xy = stable.reshape(-1, 2)

    min_xy = stable_xy.min(dim=0).values
    max_xy = stable_xy.max(dim=0).values
    span = (max_xy - min_xy).max().clamp(min=1e-6)

    scale = 350.0 / span
    x[..., :2] *= scale

    body_xy = x[:, BODY_START:BODY_END, :2].reshape(-1, 2)
    center = body_xy.mean(dim=0)

    x[..., 0] -= center[0]
    x[..., 1] -= center[1]

    x[..., 0] += 256
    x[..., 1] += 256

    return x.contiguous()


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

    # ====== FULL 586 HEADER DUMP ======
    print("\n=== FULL 586 HEADER DUMP ===")
    for comp in pose_full.header.components:
        for p in comp.points:
            print(f"{comp.name}::{p}")

    print("\n=== EXTRA DEBUG: 586 HEADER DUMP (first 200 joints) ===")

    joint_i = 0
    for comp in pose_full.header.components:
        print(f"\n--- Component: {comp.name} (num={len(comp.points)}) ---")
        for p in comp.points:
            print(f"{joint_i:3d}: {comp.name:25s} {p}")
            joint_i += 1
            if joint_i >= 200:
                break
        if joint_i >= 200:
            break

    print("=== END EXTRA DEBUG ===\n")

    # ====== REDUCED 178 HEADER DUMP ======
    print("\n=== REDUCED 178 HEADER DUMP ===")
    for comp in header_178.components:
        for p in comp.points:
            print(f"{comp.name}::{p}")
    print("================ END HEADER DEBUG ================\n")

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
    print("\n=== INDEX MAP (586→178) ===")
    for i,(comp,p) in enumerate([(c.name,p) for c in header_178.components for p in c.points]):
        print(f"{i:3d}: {comp:25s} {p:30s} -> 586 index {index_map[i]}")

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

    fut_vis  = prepare_for_visualization(fut_un_178)
    pred_vis = prepare_for_visualization(pred_un_178)
    print("fut_vis shape =", fut_vis.shape)
    print("pred_vis shape =", pred_vis.shape)

    # ====================================================================
    # SAVE pose files
    out_gt   = os.path.join(out_dir, "gt_178.pose")
    out_pred = os.path.join(out_dir, "pred_178.pose")

    tensor_to_pose(fut_vis.unsqueeze(0), header_178).write(open(out_gt, "wb"))
    tensor_to_pose(pred_vis.unsqueeze(0), header_178).write(open(out_pred, "wb"))

    print(f"[SAVE] Saved -> {out_gt}")
    print(f"[SAVE] Saved -> {out_pred}")

    print("=============== DONE ===============\n")
