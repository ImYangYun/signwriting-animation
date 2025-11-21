# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader
from copy import deepcopy
import sys, atexit

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc


# ===========================================
# stdout safe
# ===========================================
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
atexit.register(lambda: sys.stdout.flush())

np.set_printoptions(suppress=True, linewidth=120, precision=4)
torch.set_printoptions(sci_mode=False, precision=4)


# ===========================================
# Pose visualizer helper
# ===========================================
def tensor_to_pose(t_btjc, header, fps=25):
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc

    arr = t.detach().cpu().numpy().astype(np.float32)

    # ---- visual centering ----
    center = np.median(arr[:, :, :2].reshape(-1, 2), axis=0)
    arr[:, :, :2] -= center

    r = np.sqrt(arr[:, :, 0]**2 + arr[:, :, 1]**2).reshape(-1)
    scale = 120 / (np.percentile(r, 95) + 1e-6)
    arr[:, :, :2] *= scale
    if arr.shape[-1] > 2:
        arr[:, :, 2] *= scale

    arr[:, :, :2] += np.array([150,150],dtype=np.float32)[None,None,:]

    arr4 = arr[:, None, :, :]
    conf = np.ones((arr4.shape[0],1,arr4.shape[2],1),dtype=np.float32)

    body = NumPyPoseBody(fps=fps, data=arr4, confidence=conf)
    return Pose(header=header, body=body)


@torch.no_grad()
def inference_one_frame(model, past_btjc, sign_img):
    return model.sample_autoregressive_diffusion(
        past_btjc=past_btjc,
        sign_img=sign_img,
        future_len=1,
        chunk=1
    )

@torch.no_grad()
def autoregressive_generate(model, past_btjc, sign_img, future_len):
    return model.sample_autoregressive_diffusion(
        past_btjc=past_btjc,
        sign_img=sign_img,
        future_len=future_len,
        chunk=5
    )

###############################################################
#                         MAIN START
###############################################################
if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/overfit_586"
    os.makedirs(out_dir, exist_ok=True)

    # ==========================================================
    # Dataset
    # ==========================================================
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=60,
        num_future_frames=30,
        with_metadata=True,
        split="train",
        reduce_holistic=False,  # FULL 586
    )

    small_ds = torch.utils.data.Subset(base_ds, list(range(10)))
    loader = DataLoader(
        small_ds,
        batch_size=2,
        shuffle=True,
        collate_fn=zero_pad_collator,
        num_workers=0,
        pin_memory=False,
    )

    batch0 = next(iter(loader))
    B,T,P,J,C = batch0["data"].shape
    print(f"[INFO] Overfit set = (B={B},T={T},P={P},J={J},C={C})")


    # Model
    model = LitMinimal(
        num_keypoints=J,
        num_dims=C,
        lr=3e-4,
        stats_path=os.path.join(data_dir, "mean_std_586.pt"),
        diffusion_steps=200,
        beta_start=1e-4,
        beta_end=2e-2,
        pred_target="x0",
        guidance_scale=5.0,
    )

    model_cpu = model.eval()
    probe_btjc = sanitize_btjc(batch0["data"][:1])
    std_probe = model_cpu.normalize(probe_btjc).float().std().item()
    print(f"[CHECK] initial normalized-std = {std_probe:.4f}")

    if not (0.5 <= std_probe <= 2.0):
        print("[WARN] std mismatch → scaling std_pose")
        factor = max(std_probe,1e-3)
        with torch.no_grad():
            model_cpu.std_pose *= factor
        std_after = model_cpu.normalize(probe_btjc).float().std().item()
        print(f"[Calib] new normalized-std = {std_after:.4f}")


    # TRAIN (small overfit)
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        deterministic=True,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_train_batches=10,
        limit_val_batches=2,
        log_every_n_steps=1,
    )

    print("\n[TRAIN] Begin overfit 586(no-world)…")
    trainer.fit(model, loader, loader)


    # ==========================================================
    # VALIDATION —— use Lightning's DTW
    # ==========================================================
    print("\n================ EVALUATION: Lightning Validation ================")
    trainer.validate(model, loader)


    # ==========================================================
    # MANUAL EVALUATION: reduce to no-world for visualization
    # ==========================================================
    batch = next(iter(loader))
    cond = batch["conditions"]

    fut_raw  = sanitize_btjc(batch["data"][:1].to(model.device))
    past_raw = sanitize_btjc(cond["input_pose"][:1].to(model.device))
    sign_img = cond["sign_image"][:1].to(model.device)
    T_future = fut_raw.size(1)

    # load full header
    pose_path = base_ds.records[0]["pose"]
    pose_src = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)
    with open(pose_src,"rb") as f:
        pose_full = Pose.read(f)

    full_header = pose_full.header
    print("\n[HEADER] FULL 586:")
    for comp in full_header.components:
        print(f"{comp.name}: {len(comp.points)}")
    print(f"Total joints = {sum(len(c.points) for c in full_header.components)}")


    # ==========================================================
    # BUILD no-world header
    # ==========================================================
    header_now = deepcopy(full_header)
    header_now.components = [
        c for c in header_now.components
        if c.name != "POSE_WORLD_LANDMARKS"
    ]

    print("\n[HEADER] no-world:")
    for c in header_now.components:
        print(f"{c.name}: {len(c.points)}")

    keep_J = sum(len(c.points) for c in header_now.components)
    print(f"[INFO] joints kept (no-world) = {keep_J}")

    # build index map
    name2idx = {}
    idx = 0
    for comp in full_header.components:
        for p in comp.points:
            name2idx[(comp.name,p)] = idx
            idx += 1

    index_map = []
    for comp in header_now.components:
        for p in comp.points:
            index_map.append(name2idx[(comp.name,p)])

    index_map = np.asarray(index_map,dtype=np.int64)
    idx_t = torch.as_tensor(index_map, device=fut_raw.device)

    print("\n[DEBUG] index_map first 30:", index_map[:30])
    print("[DEBUG] index_map len:", len(index_map))


    # ==========================================================
    # REDUCE fut/past/preds to no-world
    # ==========================================================
    fut_now  = fut_raw.index_select(2, idx_t)
    past_now = past_raw.index_select(2, idx_t)


    # ==========================================================
    # SAVE GT
    # ==========================================================
    gt_out = os.path.join(out_dir,"gt_586_noworld.pose")
    with open(gt_out,"wb") as f:
        tensor_to_pose(fut_now, header_now).write(f)
    print("[SAVE]", gt_out)


    # ==========================================================
    # 1-FRAME PRED
    # ==========================================================
    pred_full = inference_one_frame(model, past_raw, sign_img)
    pred_now  = pred_full.index_select(2, idx_t)

    pred_out = os.path.join(out_dir,"pred_1frame_586_noworld.pose")
    with open(pred_out,"wb") as f:
        tensor_to_pose(pred_now, header_now).write(f)
    print("[SAVE]", pred_out)


    # ==========================================================
    # AUTOREGRESSIVE GEN
    # ==========================================================
    gen_full = autoregressive_generate(model, past_raw, sign_img, T_future)
    gen_now = gen_full.index_select(2, idx_t)

    gen_out = os.path.join(out_dir,"gen_586_noworld.pose")
    with open(gen_out,"wb") as f:
        tensor_to_pose(gen_now, header_now).write(f)
    print("[SAVE]", gen_out)


    # ==========================================================
    # MOTION DEBUG
    # ==========================================================
    def motion_stats(x):
        if x.size(1) <= 1:
            return (0.0,0.0)
        d = x[:,1:] - x[:,:-1]
        return d.abs().mean().item(), d.std().item()

    print("\n============ MOTION SUMMARY (586-no-world) ============")
    print("GT   motion:", motion_stats(fut_now))
    print("PRED motion:", motion_stats(pred_now))
    print("GEN  motion:", motion_stats(gen_now))
    print("========================================================\n")

