# -*- coding: utf-8 -*-
import os
import glob
import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.utils.generic import reduce_holistic

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, sanitize_btjc

np.set_printoptions(suppress=True, linewidth=120, precision=4)
torch.set_printoptions(sci_mode=False, precision=4)

# ---------------------- robust tensor → Pose (center+scale over whole sequence) ----------------------
def tensor_to_pose(t_btjc, header, fps=25):
    t = t_btjc[0] if t_btjc.dim() == 4 else t_btjc          # [T,J,C]
    arr = t.detach().cpu().numpy().astype(np.float32)       # [T,J,C]

    xy = arr[:, :, :2].reshape(-1, 2)
    ctr_xy = np.median(xy, axis=0, keepdims=True)           # [1,2]
    arr[:, :, :2] -= ctr_xy

    r = np.sqrt(arr[:, :, 0]**2 + arr[:, :, 1]**2).reshape(-1)
    s = np.percentile(r, 95) + 1e-6
    scale = 120.0 / s
    arr[:, :, :2] *= scale
    if arr.shape[-1] >= 3:
        arr[:, :, 2] *= scale

    # shift to canvas center
    arr[:, :, :2] += np.array([150.0, 150.0], dtype=np.float32)[None, None, :]

    arr4 = arr[:, None, :, :]                                # [T,1,J,C]
    conf = np.ones((arr4.shape[0], 1, arr4.shape[2], 1), dtype=np.float32)
    body = NumPyPoseBody(fps=fps, data=arr4, confidence=conf)
    return Pose(header=header, body=body)

# ---------------------- header helpers（保证与 dataloader 一致） ----------------------
def load_header_178_from_file(pose_path: str):
    with open(pose_path, "rb") as f:
        ref_pose = Pose.read(f)
    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    return reduce_holistic(ref_pose).header

def dump_header_info(header):
    comp_names = [c.name for c in header.components]
    pts_counts = [len(c.points) for c in header.components]
    limb_counts = [len(c.limbs) for c in header.components]
    print(f"[HEADER] components={comp_names}")
    print(f"[HEADER] joints per component={pts_counts}  total_joints= {sum(pts_counts)}")
    print(f"[HEADER] limbs   per component={limb_counts}")
    first10 = []
    for c in header.components:
        first10.extend(c.points)
        if len(first10) >= 10: break
    print("[HEADER] first 10 point names:", first10[:10])

def assert_limb_indices_valid(header, J: int):
    for ci, c in enumerate(header.components):
        for (a, b) in c.limbs:
            assert 0 <= a < J and 0 <= b < J, f"Invalid limb ({a},{b}) in comp#{ci} for J={J}"
    print("[CHECK] All limb indices < J ✅")

def points_only_pose(t_btjc, header, save_path, fps=25):
    backups = [list(c.limbs) for c in header.components]
    for c in header.components:
        c.limbs.clear()
    p = tensor_to_pose(t_btjc, header, fps=fps)
    with open(save_path, "wb") as f:
        p.write(f)
    for c, limbs in zip(header.components, backups):
        c.limbs[:] = limbs

# ---------------------- sampling wrappers ----------------------
@torch.no_grad()
def inference_one_frame(model: LitMinimal, past_btjc: torch.Tensor, sign_img: torch.Tensor):
    return model.sample_autoregressive_diffusion(past_btjc=past_btjc, sign_img=sign_img, future_len=1, chunk=1)

@torch.no_grad()
def autoregressive_generate(model: LitMinimal, past_btjc: torch.Tensor, sign_img: torch.Tensor, future_len: int):
    chunk = max(2, min(5, future_len))
    return model.sample_autoregressive_diffusion(
        past_btjc=past_btjc, sign_img=sign_img, future_len=future_len, chunk=chunk
    )

# ---------------------- Main ----------------------
if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"
    out_dir  = "logs/overfit_178"
    os.makedirs(out_dir, exist_ok=True)

    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=60,
        num_future_frames=30,
        with_metadata=True,
        split="train",
        reduce_holistic=True,  # 和 header 路径严格一致
    )
    small_ds = torch.utils.data.Subset(base_ds, list(range(10)))
    loader = DataLoader(
        small_ds,
        batch_size=2,
        shuffle=True,
        collate_fn=zero_pad_collator,
        num_workers=0,
        pin_memory=False
    )

    batch0 = next(iter(loader))
    B, T, P, J, C = batch0["data"].shape
    print(f"[INFO] Overfit set shape = {B, T, P, J, C}")

    model = LitMinimal(
        num_keypoints=J,
        num_dims=C,
        lr=3e-4,
        stats_path=os.path.join(data_dir, "mean_std_178.pt"),
        diffusion_steps=200,
        beta_start=1e-4,
        beta_end=2e-2,
        pred_target="x0",
        guidance_scale=0.0,
    )

    try:
        if hasattr(model, "verbose"): model.verbose = False
        if hasattr(model, "model") and hasattr(model.model, "verbose"):
            model.model.verbose = False
        for attr in ("dbg", "log", "logger"):
            if hasattr(model, attr) and callable(getattr(model, attr)):
                setattr(model, attr, lambda *a, **k: None)
    except Exception:
        pass

    model_cpu = model.eval()
    probe_btjc = sanitize_btjc(batch0["data"][:1])           # [1,T,J,C]
    std_probe = model_cpu.normalize(probe_btjc).float().std().item()
    status = "OK" if 0.5 <= std_probe <= 2.0 else "MISMATCH"
    print(f"[CHECK] GT_norm.std = {std_probe:.2f} → {status}")
    if status == "MISMATCH":
        factor = max(std_probe, 1e-3)
        with torch.no_grad():
            model_cpu.std_pose *= factor
        std_after = model_cpu.normalize(probe_btjc).float().std().item()
        print(f"[Calib] scaled std_pose by {factor:.3f} → recheck std={std_after:.2f}")

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
    print("[TRAIN] Start overfit")
    trainer.fit(model, loader, loader)

    # ========================= Evaluation =========================
    print("\n=== Evaluation ===")
    model.eval()
    batch = next(iter(loader))
    cond  = batch["conditions"]

    fut_raw  = sanitize_btjc(batch["data"][:1].to(model.device))      # [1,30,J,3]
    past_raw = sanitize_btjc(cond["input_pose"][:1].to(model.device)) # [1,60,J,3]
    sign_img = cond["sign_image"][:1].to(model.device)
    T_future = fut_raw.size(1)

    # ---------------------- Header：严格与 dataloader 同源 ----------------------
    pose_path = base_ds.records[0]["pose"]
    src = pose_path if os.path.isabs(pose_path) else os.path.join(data_dir, pose_path)
    header_178 = load_header_178_from_file(src)
    dump_header_info(header_178)

    # 验证 J 与 limb 索引
    J_from_header = sum(len(c.points) for c in header_178.components)
    assert J_from_header == J, f"Header J={J_from_header} but batch J={J} (mismatch!)"
    assert_limb_indices_valid(header_178, J)

    # ---------- GT：只画点（排查顺序/坐标），再画带 limb 的 ----------
    points_only_pose(fut_raw, header_178, os.path.join(out_dir, "gt_points_only.pose"))
    print("[SAVE] gt_points_only.pose saved")

    gt_norm = model.normalize(fut_raw)
    gt_un   = model.unnormalize(gt_norm)
    with open(os.path.join(out_dir, "gt_178.pose"), "wb") as f:
        tensor_to_pose(gt_un, header_178).write(f)
    print("[SAVE] gt_178.pose saved")

    # ---------- 1-frame prediction ----------
    pred_un = inference_one_frame(model, past_raw, sign_img)  # [1,1,J,C]
    print(f"[DEBUG] pred_1frame: min={pred_un.min().item():.2f}, max={pred_un.max().item():.2f}")
    with open(os.path.join(out_dir, "pred_1frame_178.pose"), "wb") as f:
        tensor_to_pose(pred_un, header_178).write(f)
    print("[SAVE] pred_1frame_178.pose saved")

    print("\n=== SANITY CHECK (1-frame detailed) ===")
    pred_norm = model.normalize(pred_un)
    gt_norm_1 = model.normalize(fut_raw[:, :1])
    print(f"[DEBUG] pred_norm shape = {pred_norm.shape} (should be [1,1,J,C])")
    print(f"[GT_norm] min={gt_norm_1.min().item():.4f}, max={gt_norm_1.max().item():.4f}, std={gt_norm_1.std().item():.4f}")
    print(f"[PR_norm] min={pred_norm.min().item():.4f}, max={pred_norm.max().item():.4f}, std={pred_norm.std().item():.4f}")
    print(f"[UNNORM pred] min={pred_un.min().item():.4f}, max={pred_un.max().item():.4f}, std={pred_un.std().item():.4f}")
    print("=== SANITY CHECK END ===\n")

    # ---------- Autoregressive generation (multi-frames) ----------
    print("=== Inference B (autoregressive, multi-frames) ===")
    gen_un = autoregressive_generate(model, past_raw, sign_img, future_len=T_future)  # [1,T,J,3]
    if gen_un.size(1) > 1:
        vel = gen_un[:, 1:] - gen_un[:, :-1]
        print(f"[GEN MOTION] mean |Δ| = {vel.abs().mean().item():.6f}, std = {vel.std().item():.6f}")
    else:
        print("[GEN MOTION] skipped (T=1)")
    print(f"[DEBUG] gen_un shape = {tuple(gen_un.shape)} (should be [1,{T_future},{J},{C}])")
    with open(os.path.join(out_dir, "gen_178.pose"), "wb") as f:
        tensor_to_pose(gen_un, header_178).write(f)
    print("[SAVE] gen_178.pose saved")

    # ========================= SUMMARY =========================
    print("\n==================== ACTION SUMMARY ====================")

    def motion_stats(x):
        if x.size(1) <= 1: return 0.0, 0.0
        d = x[:, 1:] - x[:, :-1]
        return d.abs().mean().item(), d.std().item()

    gt_motion   = motion_stats(gt_un)
    pred_motion = motion_stats(pred_un)   # 这里T=1，mean/std 为 0 属于正常现象
    gen_motion  = motion_stats(gen_un)
    print(f"[Motion GT ] meanΔ={gt_motion[0]:.6f}, stdΔ={gt_motion[1]:.6f}")
    print(f"[Motion PRED] meanΔ={pred_motion[0]:.6f}, stdΔ={pred_motion[1]:.6f}")
    print(f"[Motion GEN ] meanΔ={gen_motion[0]:.6f}, stdΔ={gen_motion[1]:.6f}")

    def l2_error(a, b):
        d = ((a - b) ** 2).sum(dim=-1).sqrt()  # [1,T,J]
        return d.mean().item(), d[:, :5].mean().item(), d[:, -5:].mean().item()

    l2_full, l2_early, l2_late = l2_error(pred_un, fut_raw[:, :1])  # pred_un is 1 frame
    print(f"[L2 Error 1-frame] full={l2_full:.6f}, first5={l2_early:.6f}, last5={l2_late:.6f}")

    gt_center  = gt_un.mean().item()
    pred_center = pred_un.mean().item()
    gen_center  = gen_un.mean().item()
    print(f"[Drift] GT_center={gt_center:.4f}, Pred_center={pred_center:.4f}, Gen_center={gen_center:.4f}")

    gt_norm_stats   = (gt_norm.mean().item(), gt_norm.std().item())
    pred_norm_stats = (pred_norm.mean().item(), pred_norm.std().item())
    print(f"[Norm GT ] mean={gt_norm_stats[0]:.4f}, std={gt_norm_stats[1]:.4f}")
    print(f"[Norm PRED] mean={pred_norm_stats[0]:.4f}, std={pred_norm_stats[1]:.4f}")

    def axis_stats(x):
        if x.dim() == 4: x = x[0]
        m = x.mean(dim=(0, 1))
        s = x.std(dim=(0, 1))
        return m.tolist(), s.tolist()

    gt_m, gt_s   = axis_stats(gt_un)
    pr_m, pr_s   = axis_stats(pred_un)
    gen_m, gen_s = axis_stats(gen_un)
    print(f"[XYZ GT ] mean={gt_m}, std={gt_s}")
    print(f"[XYZ PRED] mean={pr_m}, std={pr_s}")
    print(f"[XYZ GEN ] mean={gen_m}, std={gen_s}")
    print("================== END SUMMARY ====================\n")

    print("\n=== Saved files in", out_dir, "===")
    for p in sorted(glob.glob(os.path.join(out_dir, "*.pose"))):
        try:
            print("[POSE]", os.path.basename(p), "size=", os.path.getsize(p), "bytes")
        except Exception:
            print("[POSE]", os.path.basename(p))
    print("====================================\n")
