# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc, masked_dtw


# Convert tensor → Pose (only used for saving .pose)
def tensor_to_pose(t_btjc, header):
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    arr = t.detach().cpu().numpy().astype(np.float32)

    center_offset = np.array([150.0, 150.0, 0.0], dtype=np.float32)
    if arr.shape[-1] >= 2:
        arr[..., :3] += center_offset

    arr = arr[:, None, :, :].astype(np.float32)
    conf = np.ones((arr.shape[0], 1, arr.shape[2], 1), dtype=np.float32)
    body = NumPyPoseBody(fps=25, data=arr, confidence=conf)
    return Pose(header=header, body=body)


@torch.no_grad()
def inference_one_frame(model, past_btjc, sign_img):
    device = model.device
    model.eval()

    past_norm = model.normalize(past_btjc.to(device))
    sign      = sign_img.to(device)

    B, Tp, J, C = past_norm.shape
    ts = torch.zeros(B, dtype=torch.long, device=device)

    x_query = model.normalize(torch.randn((B, 1, J, C), device=device) * 0.05)
    pred_norm = model.forward(x_query, ts, past_norm, sign)
    return model.unnormalize(pred_norm)


@torch.no_grad()
def autoregressive_generate(model, past_btjc, sign_img, future_len):
    device = model.device
    model.eval()

    past_norm = model.normalize(past_btjc.to(device))
    sign      = sign_img.to(device)

    B, Tp, J, C = past_norm.shape
    ts = torch.zeros(B, dtype=torch.long, device=device)

    frames = []
    for t in range(future_len):
        x_query = model.normalize(torch.randn((B, 1, J, C), device=device) * 0.05)
        pred_next = model.forward(x_query, ts, past_norm, sign)
        frames.append(pred_next)
        past_norm = torch.cat([past_norm, pred_next], dim=1)

    gen_norm = torch.cat(frames, dim=1)
    return model.unnormalize(gen_norm)


# Main
if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/overfit_178"
    os.makedirs(out_dir, exist_ok=True)

    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=60,
        num_future_frames=30,
        with_metadata=True,
        split="train",
        reduce_holistic=True,
    )

    small_ds = torch.utils.data.Subset(base_ds, list(range(10)))
    loader = DataLoader(small_ds, batch_size=2, shuffle=True, collate_fn=zero_pad_collator)

    batch0 = next(iter(loader))
    B, T, P, J, C = batch0["data"].shape
    print(f"[INFO] Overfit set shape = {B, T, P, J, C}")

    # Model
    model = LitMinimal(
        num_keypoints=J,
        num_dims=C,
        lr=1e-4,
        stats_path=os.path.join(data_dir, "mean_std_178.pt"),
    )

    # Train
    trainer = pl.Trainer(
        max_epochs=800,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_checkpointing=False,
        deterministic=True,
    )
    print("[TRAIN] Start overfit")
    trainer.fit(model, loader, loader)

    # Evaluation
    print("\n=== Evaluation ===")
    model.eval()
    batch = next(iter(loader))
    cond  = batch["conditions"]

    fut_raw  = sanitize_btjc(batch["data"][:1].to(model.device))      # [1,30,178,3]
    past_raw = sanitize_btjc(cond["input_pose"][:1].to(model.device)) # [1,60,178,3]
    sign_img = cond["sign_image"][:1].to(model.device)
    T = fut_raw.size(1)

    # ---------------------- Header ----------------------
    pose_path = base_ds.records[0]["pose"]
    src = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)
    with open(src, "rb") as f:
        ref_pose = Pose.read(f)
    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    header = reduce_holistic(ref_pose).header
    print("[HEADER] limbs per component:", [len(c.limbs) for c in header.components])


    # ---------------------- GT (unnorm) ----------------------
    gt_norm = model.normalize(fut_raw)
    gt_un   = model.unnormalize(gt_norm)

    gt_pose = tensor_to_pose(gt_un, header)
    gt_pose.write(open(os.path.join(out_dir, "gt_178.pose"), "wb"))
    print("[SAVE] gt_178.pose saved")


    # ============================================================
    # 1-frame prediction
    # ============================================================
    pred_un = inference_one_frame(model, past_raw, sign_img)  # [1,1,J,C]
    print(f"[DEBUG] pred_1frame: min={pred_un.min().item():.2f}, max={pred_un.max().item():.2f}")

    pred_pose = tensor_to_pose(pred_un, header)
    pred_pose.write(open(os.path.join(out_dir, "pred_1frame_178.pose"), "wb"))
    print("[SAVE] pred_1frame_178.pose saved")

    print("\n=== SANITY CHECK (1-frame detailed) ===")
    pred_norm = model.normalize(pred_un)
    gt_norm_1 = model.normalize(fut_raw[:, :1])

    print(f"[DEBUG] pred_norm shape = {pred_norm.shape} (should be [1,1,J,C])")
    print(f"[GT_norm] min={gt_norm_1.min().item():.4f}, max={gt_norm_1.max().item():.4f}, std={gt_norm_1.std().item():.4f}")
    print(f"[PR_norm] min={pred_norm.min().item():.4f}, max={pred_norm.max().item():.4f}, std={pred_norm.std().item():.4f}")
    print(f"[UNNORM pred] min={pred_un.min().item():.4f}, max={pred_un.max().item():.4f}, std={pred_un.std().item():.4f}")
    print("=== SANITY CHECK END ===\n")


    print("=== Inference B (autoregressive, 30 frames) ===")
    gen_un = model.sample_autoregressive_diffusion(past_btjc=past_raw, sign_img=sign_img, future_len=T, chunk=1)

    if gen_un.size(1) > 1:
        vel = gen_un[:, 1:] - gen_un[:, :-1]
        print(f"[GEN MOTION] mean |Δ| = {vel.abs().mean().item():.6f}, std = {vel.std().item():.6f}")
    else:
        print("[GEN MOTION] skipped (T=1)")

    print(f"[DEBUG] gen_un shape = {gen_un.shape} (should be [1,30,178,3])")

    gen_pose = tensor_to_pose(gen_un, header)
    gen_pose.write(open(os.path.join(out_dir, "gen_178.pose"), "wb"))
    print("[SAVE] gen_178.pose saved")


    # ============================================================
    # SUMMARY BLOCK (1–5 full action analysis)
    # ============================================================
    print("\n==================== ACTION SUMMARY ====================")

    # 1. Motion amplitude (GT vs PRED vs GEN)
    def motion_stats(x):
        if x.size(1) <= 1:
            return 0.0, 0.0
        d = x[:, 1:] - x[:, :-1]   # [1,T-1,J,C]
        return d.abs().mean().item(), d.std().item()

    gt_motion  = motion_stats(gt_un)
    pred_motion = motion_stats(pred_un)
    gen_motion = motion_stats(gen_un)

    print(f"[Motion GT ] meanΔ={gt_motion[0]:.6f}, stdΔ={gt_motion[1]:.6f}")
    print(f"[Motion PRED] meanΔ={pred_motion[0]:.6f}, stdΔ={pred_motion[1]:.6f}")
    print(f"[Motion GEN ] meanΔ={gen_motion[0]:.6f}, stdΔ={gen_motion[1]:.6f}")


    # 2. L2 errors
    def l2_error(a, b):
        d = ((a - b) ** 2).sum(dim=-1).sqrt()  # [1,T,J]
        full = d.mean().item()
        first5 = d[:, :5].mean().item()
        last5  = d[:, -5:].mean().item()
        return full, first5, last5

    l2_full, l2_early, l2_late = l2_error(pred_un, fut_raw[:, :1])  # pred_un is 1 frame
    print(f"[L2 Error 1-frame] full={l2_full:.6f}, first5={l2_early:.6f}, last5={l2_late:.6f}")


    # 3. Drift check
    gt_center  = gt_un.mean().item()
    pred_center = pred_un.mean().item()
    gen_center  = gen_un.mean().item()
    print(f"[Drift] GT_center={gt_center:.4f}, Pred_center={pred_center:.4f}, Gen_center={gen_center:.4f}")


    # 4. Norm-space mismatch
    gt_norm_stats = (gt_norm.mean().item(), gt_norm.std().item())
    pred_norm_stats = (pred_norm.mean().item(), pred_norm.std().item())
    print(f"[Norm GT ] mean={gt_norm_stats[0]:.4f}, std={gt_norm_stats[1]:.4f}")
    print(f"[Norm PRED] mean={pred_norm_stats[0]:.4f}, std={pred_norm_stats[1]:.4f}")


    # 5. XYZ dimension stats
    def axis_stats(x):
        if x.dim() == 4:
            x = x[0]
        m = x.mean(dim=(0,1))
        s = x.std(dim=(0,1))
        return m.tolist(), s.tolist()

    gt_m, gt_s = axis_stats(gt_un)
    pr_m, pr_s = axis_stats(pred_un)
    gen_m, gen_s = axis_stats(gen_un)

    print(f"[XYZ GT ] mean={gt_m}, std={gt_s}")
    print(f"[XYZ PRED] mean={pr_m}, std={pr_s}")
    print(f"[XYZ GEN ] mean={gen_m}, std={gen_s}")

    print("================== END SUMMARY ====================\n")
