# -*- coding: utf-8 -*-
import os, sys
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc


# ---------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------
try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass
np.set_printoptions(suppress=True, linewidth=180, precision=4)
torch.set_printoptions(sci_mode=False, precision=4)



# ---------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------
def _plain(x):
    """Make tensor CPU float, remove MaskedTensor wrappers."""
    if hasattr(x, "tensor"): x = x.tensor
    if hasattr(x, "zero_filled"): x = x.zero_filled()
    return x.detach().cpu().contiguous().float()


def unnormalize_tensor(t, mean_std):
    """t: [B,T,J,C]"""
    mean = torch.tensor(mean_std["mean"], dtype=torch.float32, device=t.device)
    std  = torch.tensor(mean_std["std"], dtype=torch.float32, device=t.device)
    return t * std + mean


def prepare_for_visualization_178(x):
    """Avoid collapse, center body, scale XY."""
    if x.dim() == 4:
        x = x[0]
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
    return x.contiguous()


def tensor_to_pose(x_btjc, header):
    """Convert tensor [1,T,J,C] or [T,J,C] to Pose object."""
    if x_btjc.dim() == 4:
        x_btjc = x_btjc[0]
    arr = np.ascontiguousarray(x_btjc[:, None, :, :], dtype=np.float32)
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)
    return Pose(header=header, body=NumPyPoseBody(fps=25, data=arr, confidence=conf))



# ---------------------------------------------------------------
# DEBUG TOOL (强力版)
# ---------------------------------------------------------------
def pose_debug(gt_btjc, pred_btjc, header, out_dir="debug_178"):
    """Dump full diagnostics: stats, histograms, component ranges."""
    os.makedirs(out_dir, exist_ok=True)
    print("\n================ DEBUG TOOL ================")

    gt = _plain(gt_btjc)[0] if gt_btjc.dim()==4 else _plain(gt_btjc)
    pr = _plain(pred_btjc)[0] if pred_btjc.dim()==4 else _plain(pred_btjc)

    def desc(x):
        return {
            "min": float(x.min()), "max": float(x.max()),
            "mean": float(x.mean()), "std": float(x.std()),
            "nan%": float(torch.isnan(x).float().mean())*100,
            "inf%": float(torch.isinf(x).float().mean())*100,
        }

    print("\n-- GLOBAL RANGE --")
    print("GT:", desc(gt))
    print("PR:", desc(pr))

    # Joint variance
    gv = torch.var(gt[..., :2], dim=(0,1)).numpy()
    pv = torch.var(pr[..., :2], dim=(0,1)).numpy()
    print("\n-- JOINT VARIANCE --")
    print("GT var min/max:", gv.min(), gv.max())
    print("PR var min/max:", pv.min(), pv.max())

    # Save plots
    def plot_vec(vec, path):
        plt.figure(figsize=(10,4))
        plt.plot(vec)
        plt.title(os.path.basename(path))
        plt.savefig(path)
        plt.close()
    plot_vec(gv, f"{out_dir}/gt_joint_var.png")
    plot_vec(pv, f"{out_dir}/pred_joint_var.png")

    # Frame-wise motion
    gdiff = (gt[1:]-gt[:-1]).norm(dim=-1).mean(dim=1).numpy()
    pdiff = (pr[1:]-pr[:-1]).norm(dim=-1).mean(dim=1).numpy()
    print("\n-- FRAME MOTION --")
    print("GT motion avg:", gdiff.mean())
    print("PR motion avg:", pdiff.mean())
    plot_vec(gdiff, f"{out_dir}/gt_motion.png")
    plot_vec(pdiff, f"{out_dir}/pred_motion.png")

    # Component ranges
    print("\n-- COMPONENT RANGES --")
    start = 0
    for comp in header.components:
        end = start + len(comp.points)
        seg_gt = gt[:, start:end, :2].reshape(-1, 2)
        seg_pr = pr[:, start:end, :2].reshape(-1, 2)
        print(f"{comp.name:25s} | GT std={float(seg_gt.std()):.3f} | PR std={float(seg_pr.std()):.3f}")
        start = end

    # Histograms
    def save_hist(x, path):
        x = x[..., :2].reshape(-1)
        plt.hist(x.numpy(), bins=80)
        plt.title(os.path.basename(path))
        plt.savefig(path)
        plt.close()
    save_hist(gt, f"{out_dir}/GT_hist.png")
    save_hist(pr, f"{out_dir}/Pred_hist.png")

    print("\n=============== END DEBUG TOOL ===============\n")



# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/minimal_loop_178"
    os.makedirs(out_dir, exist_ok=True)

    print("================================================")
    print("                Loading Dataset")
    print("================================================")

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



    # ---------------------------------------------------------------
    # Build 178 header
    # ---------------------------------------------------------------
    pose_path = base_ds.records[0]["pose"]
    pose_abspath = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)
    print(f"[LOAD] reference pose from {pose_abspath}")

    with open(pose_abspath, "rb") as f:
        pose_full = Pose.read(f)

    pose_no_world = pose_full.remove_components(["POSE_WORLD_LANDMARKS"])
    pose_178 = reduce_holistic(pose_no_world)
    header_178 = pose_178.header

    print("[HEADER] Components:", [c.name for c in header_178.components])


    # ---------------------------------------------------------------
    # Build index map 586→178
    # ---------------------------------------------------------------
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


    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    print("================================================")
    print("                Loading Model")
    print("================================================")

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



    # ---------------------------------------------------------------
    # SAMPLING
    # ---------------------------------------------------------------
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



    # ---------------------------------------------------------------
    # Unnormalize + Reduce to 178
    # ---------------------------------------------------------------
    ms = torch.load(os.path.join(data_dir, "mean_std_586.pt"))

    fut_un  = unnormalize_tensor(fut_raw,  ms)
    pred_un = unnormalize_tensor(pred_586, ms)

    fut_un  = _plain(fut_un)
    pred_un = _plain(pred_un)

    fut_178  = fut_un.index_select(2, idx_t)
    pred_178 = pred_un.index_select(2, idx_t)



    # ---------------------------------------------------------------
    # DEBUG before visualization
    # ---------------------------------------------------------------
    print("\n[DEBUG] Running pose_debug() ...")
    pose_debug(fut_178, pred_178, header_178, out_dir=os.path.join(out_dir, "debug"))
    print("[DEBUG] Done.")

    # ======================================================
    # Debug: Diagnose 586 → 178 mapping + range + collapse
    # ======================================================

    print("===== DEBUG POSE RANGE =====")
    print("[GT]   min/max:", float(fut_178.min()), float(fut_178.max()))
    print("[PRED] min/max:", float(pred_178.min()), float(pred_178.max()))

    print("[GT]   std:", float(fut_178.std()))
    print("[PRED] std:", float(pred_178.std()))

    # --- debug per-joint motion ---
    pj_gt = fut_178[0, :, :, :2].std(dim=0).mean(dim=1)
    pj_pr = pred_178[0, :, :, :2].std(dim=0).mean(dim=1)
    print("[GT]   joint motion head:", pj_gt[:10].tolist())
    print("[PRED] joint motion head:", pj_pr[:10].tolist())

    # --- detect collapse ---
    xy = pred_178[..., :2].reshape(-1, 2)
    span_pred = (xy.max(0).values - xy.min(0).values).tolist()
    print("[COLLAPSE CHECK] pred xy span:", span_pred)



    # ---------------------------------------------------------------
    # Visualization fix
    # ---------------------------------------------------------------
    fut_vis  = prepare_for_visualization_178(fut_178)
    pred_vis = prepare_for_visualization_178(pred_178)

    # ---------------------------------------------------------------
    # SAVE POSE FILES
    # ---------------------------------------------------------------
    out_gt   = os.path.join(out_dir, "gt_178.pose")
    out_pred = os.path.join(out_dir, "pred_178.pose")

    tensor_to_pose(fut_vis.unsqueeze(0), header_178).write(open(out_gt, "wb"))
    tensor_to_pose(pred_vis.unsqueeze(0), header_178).write(open(out_pred, "wb"))

    print("[SAVE] GT ->", out_gt)
    print("[SAVE] Pred ->", out_pred)
    print("================== DONE ==================")
