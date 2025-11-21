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
from pose_format.utils.generic import reduce_holistic

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc


# ---------- IO safe ----------
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
atexit.register(lambda: sys.stdout.flush())

np.set_printoptions(suppress=True, linewidth=120, precision=4)
torch.set_printoptions(sci_mode=False, precision=4)


def tensor_to_pose(t_btjc, header, fps=25):
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc

    arr = t.detach().cpu().numpy().astype(np.float32)

    # ------ Visual centering & scaling ------
    center = np.median(arr[:, :, :2].reshape(-1, 2), axis=0)
    arr[:, :, :2] -= center

    r = np.sqrt(arr[:, :, 0]**2 + arr[:, :, 1]**2).reshape(-1)
    scale = 120 / (np.percentile(r, 95) + 1e-6)
    arr[:, :, :2] *= scale
    if arr.shape[-1] > 2:
        arr[:, :, 2] *= scale

    arr[:, :, :2] += np.array([150.0, 150.0], dtype=np.float32)[None, None, :]

    arr4 = arr[:, None, :, :]
    conf = np.ones((arr4.shape[0], 1, arr4.shape[2], 1), dtype=np.float32)

    body = NumPyPoseBody(fps=fps, data=arr4, confidence=conf)
    return Pose(header=header, body=body)


@torch.no_grad()
def inference_one_frame(model, past_btjc, sign_img):
    return model.sample_autoregressive_diffusion(
        past_btjc=past_btjc, sign_img=sign_img, future_len=1, chunk=1
    )


@torch.no_grad()
def autoregressive_generate(model, past_btjc, sign_img, future_len):
    return model.sample_autoregressive_diffusion(
        past_btjc=past_btjc, sign_img=sign_img, future_len=future_len, chunk=5
    )


if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir = "logs/overfit_178vis"
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- Dataset ----------------
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=60,
        num_future_frames=30,
        with_metadata=True,
        split="train",
        reduce_holistic=False,  # KEEP full 586 for training
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
    B, T, P, J, C = batch0["data"].shape
    print(f"[INFO] Overfit set shape = B={B}, T={T}, P={P}, J={J}, C={C}")

    # ---------------- Model ----------------
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
    probe = sanitize_btjc(batch0["data"][:1])
    sp = model_cpu.normalize(probe).float().std().item()
    print(f"[CHECK] normalized std = {sp:.4f}")

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        limit_train_batches=10,
        limit_val_batches=2,
        enable_checkpointing=False,
        deterministic=True,
        enable_progress_bar=False,
        log_every_n_steps=1,
    )

    print("\n[TRAIN] Begin overfit training (586)")
    trainer.fit(model, loader, loader)

    # VALIDATION (Lightning’s DTW)
    print("\n================ VALIDATION ================")
    trainer.validate(model, loader)

    #            REDUCE to 178 joints FOR VISUALIZATION
    print("\n================ BUILD 178 SKELETON ================")

    # load header from sample file
    pose_path = base_ds.records[0]["pose"]
    pose_src = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)
    with open(pose_src, "rb") as f:
        pose_full = Pose.read(f)

    print("[HEADER] full:", sum(len(c.points) for c in pose_full.header.components))

    # remove world then reduce to 178
    pose_no_world = pose_full.remove_components(["POSE_WORLD_LANDMARKS"])
    pose_red = reduce_holistic(pose_no_world)
    header_178 = pose_red.header

    print("[HEADER] 178 components:")
    for c in header_178.components:
        print(c.name, len(c.points))
    print("Total:", sum(len(c.points) for c in header_178.components))

    # ---- Build index_map: 586 → 178 ----
    name2idx = {}
    base = 0
    for comp in pose_full.header.components:
        for p in comp.points:
            name2idx[(comp.name, p)] = base
            base += 1

    index_map = []
    for comp in header_178.components:
        for p in comp.points:
            index_map.append(name2idx[(comp.name, p)])

    index_map = np.array(index_map, dtype=np.int64)
    print("[DEBUG] index_map length =", len(index_map))
    print("[DEBUG] first 30:", index_map[:30])

    idx_t = torch.as_tensor(index_map, device=model.device)


    #               Apply reduce & Save 178 Poses
    batch = next(iter(loader))
    cond = batch["conditions"]

    fut_raw = sanitize_btjc(batch["data"][:1].to(model.device))
    past_raw = sanitize_btjc(cond["input_pose"][:1].to(model.device))
    sign_img = cond["sign_image"][:1].to(model.device)
    T_fut = fut_raw.size(1)

    # Reduce GT
    fut_178 = fut_raw.index_select(2, idx_t)

    with open(os.path.join(out_dir, "gt_178.pose"), "wb") as f:
        tensor_to_pose(fut_178, header_178).write(f)

    # Reduce prediction (1-frame)
    pred_full = inference_one_frame(model, past_raw, sign_img)
    pred_178 = pred_full.index_select(2, idx_t)
    with open(os.path.join(out_dir, "pred_1frame_178.pose"), "wb") as f:
        tensor_to_pose(pred_178, header_178).write(f)

    # Reduce autoregressive generation
    gen_full = autoregressive_generate(model, past_raw, sign_img, future_len=T_fut)
    gen_178 = gen_full.index_select(2, idx_t)
    with open(os.path.join(out_dir, "gen_178.pose"), "wb") as f:
        tensor_to_pose(gen_178, header_178).write(f)

    ################################################################
    # MOTION SUMMARY
    def motion_stats(x):
        if x.size(1) <= 1: return (0.0, 0.0)
        d = x[:, 1:] - x[:, :-1]
        return d.abs().mean().item(), d.std().item()

    print("\n=========== MOTION SUMMARY (178 joints) ===========")
    print("GT   :", motion_stats(fut_178))
    print("PRED :", motion_stats(pred_178))
    print("GEN  :", motion_stats(gen_178))
    print("====================================================\n")
