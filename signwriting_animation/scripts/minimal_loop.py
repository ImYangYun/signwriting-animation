# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import reduce_holistic

from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset

from signwriting_animation.diffusion.lightning_module import LitMinimal
from signwriting_animation.diffusion.lightning_module import sanitize_btjc, masked_dtw


# ============================================================
# Utility: convert [T,J,C] → Pose object
# ============================================================
def tensor_to_pose(t_btjc, header):
    """
    t_btjc: [1,T,J,C] or [T,J,C]
    """
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc
    arr = np.ascontiguousarray(t.detach().cpu().numpy()[:, None, :, :], dtype=np.float32)
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)
    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    return Pose(header=header, body=body)


# ============================================================
# Autoregressive inference method B
# ============================================================
@torch.no_grad()
def autoregressive_generate(model, past_btjc, sign_img, future_len=30):
    """
    Past_btjc : [1,Tp,J,C] (unnormalized)
    sign_img  : [1,3,224,224]
    Return    : [1, future_len, J, C] (unnormalized)
    """
    device = model.device
    model.eval()

    past_norm = model.normalize(past_btjc.to(device))
    sign      = sign_img.to(device)

    B, Tp, J, C = past_norm.shape
    ts = torch.zeros(B, dtype=torch.long, device=device)

    generated = []

    for t in range(future_len):
        # Small random query for diffusion x_t
        x_query = torch.randn((B,1,J,C), device=device) * 0.05
        x_query = model.normalize(x_query)

        pred_next = model.forward(x_query, ts, past_norm, sign)  # [B,1,J,C]
        generated.append(pred_next)
        past_norm = torch.cat([past_norm, pred_next], dim=1)

    gen_norm = torch.cat(generated, dim=1)
    return model.unnormalize(gen_norm)


if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/overfit_178"
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------- Dataset ----------------------
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=60,
        num_future_frames=30,
        with_metadata=True,
        split="train",
        reduce_holistic=True,
    )

    small_ds = torch.utils.data.Subset(base_ds, list(range(4)))
    loader = DataLoader(small_ds, batch_size=4, shuffle=True, collate_fn=zero_pad_collator)

    batch0 = next(iter(loader))
    B,T,P,J,C = batch0["data"].shape
    print(f"[INFO] Overfit set shape = {B,T,P,J,C}")

    # ---------------------- Model ----------------------
    model = LitMinimal(
        num_keypoints=J,
        num_dims=C,
        lr=1e-4,
        stats_path=os.path.join(data_dir, "mean_std_178.pt")
    )


    # ---------------------- Train ----------------------
    trainer = pl.Trainer(
        max_epochs=500,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        limit_train_batches=1,
        limit_val_batches=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
        deterministic=True,
    )
    print("[TRAIN] Start overfit 4 samples")
    trainer.fit(model, loader, loader)


    print("\n=== Evaluation ===")
    model.eval()
    batch  = next(iter(loader))
    cond   = batch["conditions"]

    fut_raw  = sanitize_btjc(batch["data"][:1].to(model.device))
    past_raw = sanitize_btjc(cond["input_pose"][:1].to(model.device))
    sign     = cond["sign_image"][:1].to(model.device)
    mask_raw = cond["target_mask"][:1].to(model.device)

    gt_norm = model.normalize(fut_raw)
    past_norm = model.normalize(past_raw)

    # Align lengths
    T = gt_norm.size(1)
    past_norm = past_norm[:, -T:]

    # Mask
    if mask_raw.dim() == 4:
        mask_bt = (mask_raw.sum((2,3)) > 0).float()
    else:
        mask_bt = mask_raw.float()

    ts = torch.zeros(1, dtype=torch.long, device=model.device)
    pred_norm = model.forward(past_norm, ts, past_norm, sign)

    # Sanity Check (inside minimal loop)
    # ============================================================
    print("\n=== SANITY CHECK ===")

    # ---- 1. Check NaN / Inf / value ranges ----
    def check_numeric(name, t):
        t_cpu = t.detach().cpu()
        print(f"[{name}] shape={tuple(t_cpu.shape)}")
        print(f"  min={t_cpu.min():.4f}, max={t_cpu.max():.4f}, mean={t_cpu.mean():.4f}, std={t_cpu.std():.4f}")
        print(f"  NaN%={(torch.isnan(t_cpu).float().mean()*100):.4f}%, "
              f"Inf%={(torch.isinf(t_cpu).float().mean()*100):.4f}%")

    check_numeric("GT_norm", gt_norm)
    check_numeric("PRED_norm", pred_norm)

    # ---- 2. Temporal change check (is there motion?) ----
    if pred_norm.size(1) > 1:
        vel = pred_norm[:, 1:] - pred_norm[:, :-1]
        print(f"[MOTION] mean |Δpred_norm| = {vel.abs().mean().item():.6f}")
        print(f"[MOTION] std(Δpred_norm)   = {vel.std().item():.6f}")
    else:
        print("[MOTION] skipped (T=1)")

    # ---- 3. Per-joint STD check (is output collapsing?) ----
    std_per_joint = pred_norm[0].std(dim=0).mean(dim=1)  # shape [J]
    print(f"[JOINT STD] mean std across joints = {std_per_joint.mean().item():.6f}")
    print(f"[JOINT STD] first 10 joints std     = {[round(v,4) for v in std_per_joint[:10].tolist()]}")
    if pred_norm.size(1) > 1:
        vel_pred = pred_norm[:, 1:] - pred_norm[:, :-1]
        vel_gt   = gt_norm[:, 1:]  - gt_norm[:, :-1]

        print(f"[VEL norm] |pred_vel| mean = {vel_pred.abs().mean().item():.6f}")
        print(f"[VEL norm] |gt_vel|   mean = {vel_gt.abs().mean().item():.6f}")
    else:
        print("[VEL norm] skipped (T=1)")
    print("=== SANITY CHECK END ===\n")

    # Unnormalize
    gt_un   = model.unnormalize(gt_norm)
    pred_un = model.unnormalize(pred_norm)

    if pred_un.size(1) > 1:
        vel_pred_un = pred_un[:, 1:] - pred_un[:, :-1]
        vel_gt_un   = gt_un[:, 1:]  - gt_un[:, :-1]

        print(f"[VEL unnorm] |pred_vel| mean = {vel_pred_un.abs().mean().item():.3f}")
        print(f"[VEL unnorm] |gt_vel|   mean = {vel_gt_un.abs().mean().item():.3f}")
    else:
        print("[VEL unnorm] skipped (T=1)")
    # Save only valid frames
    T_valid = int(mask_bt[0].sum().item())
    gt_un   = gt_un[:, :T_valid]
    pred_un = pred_un[:, :T_valid]

    print(f"[SAVE] Valid frames = {T_valid}")

    # ============================================================
    # Load header from reference pose file
    # ============================================================
    pose_path = base_ds.records[0]["pose"]
    src = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)

    with open(src, "rb") as f:
        ref_pose = Pose.read(f)
    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    header = reduce_holistic(ref_pose).header

    # ============================================================
    # Save gt / pred
    # ============================================================
    gt_pose   = tensor_to_pose(gt_un, header)
    pred_pose = tensor_to_pose(pred_un, header)

    out_gt   = os.path.join(out_dir, "gt_178.pose")
    out_pred = os.path.join(out_dir, "pred_178.pose")

    for p in [out_gt, out_pred]:
        if os.path.exists(p):
            os.remove(p)

    gt_pose.write(open(out_gt, "wb"))
    pred_pose.write(open(out_pred, "wb"))

    print("[SAVE] GT and Pred saved.")


    # ============================================================
    # B-version autoregressive generation
    # ============================================================
    print("\n=== Inference B (autoregressive) ===")

    gen_un = autoregressive_generate(
        model=model,
        past_btjc=past_raw,     # unnormalized
        sign_img=sign,
        future_len=T_valid
    )

    gen_pose = tensor_to_pose(gen_un, header)
    out_gen = os.path.join(out_dir, "gen_178.pose")
    if os.path.exists(out_gen):
        os.remove(out_gen)
    gen_pose.write(open(out_gen, "wb"))

    print("[SAVE] gen_178.pose saved.")
