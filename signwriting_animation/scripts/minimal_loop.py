# -*- coding: utf-8 -*-
import os, sys
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc

# -------------------------------------------------------------------
# Pretty print
# -------------------------------------------------------------------
try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass
np.set_printoptions(suppress=True, linewidth=180, precision=4)
torch.set_printoptions(sci_mode=False, precision=4)



def _to_plain(x):
    if hasattr(x, "tensor"): x = x.tensor
    if hasattr(x, "zero_filled"): x = x.zero_filled()
    return x.detach().cpu().contiguous().float()


def unnormalize_tensor(t, mean_std):
    mean = torch.tensor(mean_std["mean"], dtype=torch.float32, device=t.device)
    std  = torch.tensor(mean_std["std"], dtype=torch.float32, device=t.device)
    return t * std + mean


def prepare_for_visualization_178(x):
    """x: [T,J,C]"""
    if x.dim() == 4: x = x[0]
    x = torch.nan_to_num(x, nan=0.0)

    torso = x[:, :8, :2].reshape(-1, 2)
    span = (torso.max(0).values - torso.min(0).values).max().clamp(min=1e-6)
    scale = 450.0 / span
    x[..., :2] *= scale

    center = torso.mean(dim=0)
    x[..., 0] -= center[0]
    x[..., 1] -= center[1]

    x[..., 0] += 256
    x[..., 1] += 256
    return x


def tensor_to_pose(x, header):
    if x.dim() == 4: x = x[0]
    arr = np.ascontiguousarray(x[:, None, :, :], dtype=np.float32)
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)
    return Pose(header=header, body=NumPyPoseBody(fps=25, data=arr, confidence=conf))


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"

    out_dir = "logs/minimal_debug_178"
    os.makedirs(out_dir, exist_ok=True)

    print("==============================================")
    print("        Loading Dataset")
    print("==============================================")

    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=60,
        num_future_frames=30,
        with_metadata=True,
        split="train",
        reduce_holistic=False,
    )

    small_ds = torch.utils.data.Subset(base_ds, list(range(4)))
    loader = DataLoader(small_ds, batch_size=4, shuffle=True, collate_fn=zero_pad_collator)
    batch0 = next(iter(loader))

    print("[DS] Example batch shape:", batch0["data"].shape)


    # -------------------------------------------------------------------
    # Build 178 header
    # -------------------------------------------------------------------
    pose_path = base_ds.records[0]["pose"]
    pose_abspath = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)
    print(f"[LOAD] reference pose from {pose_abspath}")

    with open(pose_abspath, "rb") as f:
        pose_full = Pose.read(f)

    pose_no_world = pose_full.remove_components(["POSE_WORLD_LANDMARKS"])
    pose_178 = reduce_holistic(pose_no_world)
    header_178 = pose_178.header

    print("[HEADER] Components:", [c.name for c in header_178.components])


    # -------------------------------------------------------------------
    # Build 586→178 index map
    # -------------------------------------------------------------------
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
    print(f"[MAP] index_map length = {len(index_map)}")


    # -------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------
    print("==============================================")
    print("        Loading Model")
    print("==============================================")

    model = LitMinimal(
        num_keypoints=586,
        num_dims=3,
        stats_path=os.path.join(data_dir, "mean_std_586.pt"),
        lr=1e-4,
        diffusion_steps=100,
        pred_target="x0",
    )

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        deterministic=True,
        log_every_n_steps=1,
    )

    trainer.fit(model, loader, loader)
    print("======== TRAIN DONE ========")


    # -------------------------------------------------------------------
    # Sample (autoregressive)
    # -------------------------------------------------------------------
    print("======== SAMPLING ========")

    batch = next(iter(loader))
    cond = batch["conditions"]

    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(model.device)
    fut_raw  = sanitize_btjc(batch["data"][:1]).to(model.device)
    sign_img = cond["sign_image"][:1].to(model.device)

    pred_586 = model.sample_autoregressive_diffusion(
        past_btjc=past_raw,
        sign_img=sign_img,
        future_len=fut_raw.size(1),
        chunk=1
    )


    # -------------------------------------------------------------------
    # Unnormalize + reduce
    # -------------------------------------------------------------------
    ms = torch.load(os.path.join(data_dir, "mean_std_586.pt"))
    fut_un  = unnormalize_tensor(fut_raw,  ms)
    pred_un = unnormalize_tensor(pred_586, ms)

    fut_un  = _to_plain(fut_un)
    pred_un = _to_plain(pred_un)

    fut_178  = fut_un.index_select(2, idx_t)
    pred_178 = pred_un.index_select(2, idx_t)

    print("[SHAPE] fut_178:", fut_178.shape, "pred_178:", pred_178.shape)


    # -------------------------------------------------------------------
    # Debug: Range / collapse / joint-wise stats
    # -------------------------------------------------------------------
    print("==============================================")
    print("        DEBUG: RANGE CHECK")
    print("==============================================")

    print("[GT]   min/max:", float(fut_178.min()), float(fut_178.max()))
    print("[PRED] min/max:", float(pred_178.min()), float(pred_178.max()))

    print("[GT]   std:", float(fut_178.std()))
    print("[PRED] std:", float(pred_178.std()))

    print("[PER-JOINT STD HEAD]", torch.std(pred_178[:, :, :20, :2], dim=(0,1)))


    # -------------------------------------------------------------------
    # Visualization fix
    # -------------------------------------------------------------------
    fut_vis  = prepare_for_visualization_178(fut_178)
    pred_vis = prepare_for_visualization_178(pred_178)

    print("[VIS GT]   xy range:", float(fut_vis[..., :2].min()), float(fut_vis[..., :2].max()))
    print("[VIS PRED] xy range:", float(pred_vis[..., :2].min()), float(pred_vis[..., :2].max()))


    # -------------------------------------------------------------------
    # Save pose
    # -------------------------------------------------------------------
    out_gt   = os.path.join(out_dir, "gt_178.pose")
    out_pred = os.path.join(out_dir, "pred_178.pose")

    tensor_to_pose(fut_vis.unsqueeze(0),  header_178).write(open(out_gt, "wb"))
    tensor_to_pose(pred_vis.unsqueeze(0), header_178).write(open(out_pred, "wb"))

    print("[SAVE] Written:", out_gt)
    print("[SAVE] Written:", out_pred)


    # -------------------------------------------------------------------
    # Verify pose
    # -------------------------------------------------------------------
    try:
        P = Pose.read(open(out_pred, "rb"))
        print("[CHECK] pose OK, limbs:", [len(c.limbs) for c in P.header.components])
    except Exception as e:
        print("[ERROR] Saved pose corrupted:", e)

    print("=============== ALL DONE ================")
