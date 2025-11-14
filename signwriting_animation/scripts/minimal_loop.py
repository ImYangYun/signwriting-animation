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
# Utility: convert BTJC → Pose
# ============================================================
def tensor_to_pose(t_btjc, header):
    if t_btjc.dim() == 4:
        t = t_btjc[0]              # [T,J,C]
    else:
        t = t_btjc

    arr = np.ascontiguousarray(t.detach().cpu().numpy()[:, None, :, :], dtype=np.float32)
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)
    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    return Pose(header=header, body=body)


# ============================================================
# Autoregressive generator (Method B)
# ============================================================
@torch.no_grad()
def autoregressive_generate(model, past_btjc, sign_img, future_len):
    """
    past_btjc: [1,Tp,J,C] (unnormalized)
    sign_img : [1,3,224,224]
    returns  : [1,future_len,J,C] (unnormalized)
    """
    device = model.device
    model.eval()

    past_norm = model.normalize(past_btjc.to(device))
    sign      = sign_img.to(device)

    B, Tp, J, C = past_norm.shape
    ts = torch.zeros(B, dtype=torch.long, device=device)

    frames = []

    for t in range(future_len):
        # model always predicts only 1 frame
        x_query = torch.randn((B,1,J,C), device=device) * 0.05
        x_query = model.normalize(x_query)

        pred_next = model.forward(x_query, ts, past_norm, sign)   # [B,1,J,C]
        frames.append(pred_next)

        past_norm = torch.cat([past_norm, pred_next], dim=1)

    gen_norm = torch.cat(frames, dim=1)
    return model.unnormalize(gen_norm)


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir  = "logs/overfit_178"
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- Dataset ----------------
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=60,
        num_future_frames=30,
        with_metadata=True,
        split="train",
        reduce_holistic=True,    # 178 joints
    )

    small_ds = torch.utils.data.Subset(base_ds, list(range(4)))
    loader = DataLoader(small_ds, batch_size=4, shuffle=True, collate_fn=zero_pad_collator)

    # read shapes
    batch0 = next(iter(loader))
    B,T,P,J,C = batch0["data"].shape
    print(f"[INFO] Overfit set shape = {B,T,P,J,C} (expected [4,30,1,178,3])")


    # ---------------- Model ----------------
    model = LitMinimal(
        num_keypoints=J,
        num_dims=C,
        lr=1e-4,
        stats_path=os.path.join(data_dir, "mean_std_178.pt"),
    )

    # ---------------- Train ----------------
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        limit_train_batches=1,
        limit_val_batches=1,
        log_every_n_steps=1,
        enable_checkpointing=False,
        deterministic=True,
    )
    print("[TRAIN] Start overfit 4 samples")
    trainer.fit(model, loader, loader)


    # ============================================================
    #                   Evaluation
    # ============================================================
    print("\n=== Evaluation ===")
    model.eval()

    batch = next(iter(loader))
    cond = batch["conditions"]

    # raw BTJC
    fut_raw  = sanitize_btjc(batch["data"][:1].to(model.device))
    past_raw = sanitize_btjc(cond["input_pose"][:1].to(model.device))
    sign_img = cond["sign_image"][:1].to(model.device)
    mask_raw = cond["target_mask"][:1].to(model.device)

    # normalize
    gt_norm   = model.normalize(fut_raw)
    past_norm = model.normalize(past_raw)

    # align past to same T as future
    T = gt_norm.size(1)
    past_norm = past_norm[:, -T:]

    # mask
    if mask_raw.dim() == 4:
        mask_bt = (mask_raw.sum((2,3)) > 0).float()
    else:
        mask_bt = mask_raw.float()

    # ============================================================
    #  Teacher-forcing sanity check (1 frame prediction)
    # ============================================================
    print("\n=== SANITY CHECK (1-frame) ===")
    ts = torch.zeros(1, dtype=torch.long, device=model.device)

    x_query = torch.randn((1,1,J,C), device=model.device) * 0.05
    x_query = model.normalize(x_query)

    pred_norm = model.forward(x_query, ts, past_norm, sign_img)
    print(f"[DEBUG] pred_norm shape = {pred_norm.shape} (should be [1,1,J,C])")

    # ---- numerics ----
    def check(name, t):
        t_cpu = t.detach().cpu()
        print(f"[{name}] shape={tuple(t_cpu.shape)}")
        print(f"  min={t_cpu.min():.4f}, max={t_cpu.max():.4f}, mean={t_cpu.mean():.4f}, std={t_cpu.std():.4f}")
        print(f"  NaN%={(torch.isnan(t_cpu).float().mean()*100):.3f}%")

    check("GT_norm", gt_norm)
    check("PRED_norm", pred_norm)

    # ---- motion & collapse check ----
    pred_un = model.unnormalize(pred_norm)
    gt_un   = model.unnormalize(gt_norm)

    print(f"[UNNORM] pred  min={pred_un.min():.1f}, max={pred_un.max():.1f}, std={pred_un.std():.1f}")

    print("=== SANITY CHECK END ===\n")

    # ============================================================
    # Save GT & pred_1frame
    # ============================================================
    pose_path = base_ds.records[0]["pose"]
    src = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)
    with open(src, "rb") as f:
        ref_pose = Pose.read(f)

    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    header = reduce_holistic(ref_pose).header

    # save
    gt_pose   = tensor_to_pose(gt_un, header)
    pred_pose = tensor_to_pose(pred_un, header)

    gt_pose.write(open(os.path.join(out_dir,"gt_178.pose"), "wb"))
    pred_pose.write(open(os.path.join(out_dir,"pred_1frame_178.pose"), "wb"))
    print("[SAVE] GT and pred_1frame_178.pose saved.")


    # ============================================================
    #          Autoregressive Prediction (full 30 frames)
    # ============================================================
    print("\n=== Inference B (autoregressive) ===")

    gen_un = autoregressive_generate(
        model=model,
        past_btjc=past_raw,
        sign_img=sign_img,
        future_len=T
    )

    print(f"[DEBUG] gen_un shape = {gen_un.shape} (should be [1,30,178,3])")

    gen_pose = tensor_to_pose(gen_un, header)
    gen_pose.write(open(os.path.join(out_dir,"gen_178.pose"), "wb"))
    print("[SAVE] gen_178.pose saved.")
