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
from signwriting_animation.diffusion.lightning_module import (
    LitMinimalAutoregressive,
    sanitize_btjc,
    masked_dtw,
    mean_frame_disp,
)

try:
    from pose_anonymization.data.normalization import unshift_hands
    HAS_UNSHIFT = True
except ImportError:
    HAS_UNSHIFT = False


def diagnose_model_sensitivity(model, past_norm, sign_img, device):
    """
    诊断：模型是否对输入敏感？
    """
    print("\n" + "=" * 70)
    print("诊断：模型对输入的敏感度")
    print("=" * 70)
    
    model.eval()
    B, T, J, C = past_norm.shape
    
    # 测试 1：相同输入，输出是否相同
    with torch.no_grad():
        pred1 = model._predict_single_frame(past_norm, sign_img)
        pred2 = model._predict_single_frame(past_norm, sign_img)
        diff_same = (pred1 - pred2).abs().max().item()
        print(f"1. 相同输入两次预测的差异: {diff_same:.8f}")
        if diff_same > 1e-6:
            print("   ⚠️ 有随机性！检查是否有 dropout")
    
    # 测试 2：不同 past，输出是否不同
    with torch.no_grad():
        past_shifted = past_norm.clone()
        past_shifted[:, :, :, 0] += 0.1  # X 坐标 +0.1
        
        pred_orig = model._predict_single_frame(past_norm, sign_img)
        pred_shifted = model._predict_single_frame(past_shifted, sign_img)
        diff_past = (pred_orig - pred_shifted).abs().mean().item()
        print(f"2. 不同 past 的输出差异: {diff_past:.6f}")
        if diff_past < 1e-6:
            print("   ❌ 模型完全忽略了 past_motion！")
        else:
            print("   ✓ 模型对 past 敏感")
    
    # 测试 3：滑动窗口后，输出是否变化
    with torch.no_grad():
        # 原始 past
        pred_t0 = model._predict_single_frame(past_norm, sign_img)
        
        # 滑动一帧：去掉 past[0]，加入 pred_t0
        new_past = torch.cat([past_norm[:, 1:], pred_t0], dim=1)
        pred_t1 = model._predict_single_frame(new_past, sign_img)
        
        diff_ar = (pred_t0 - pred_t1).abs().mean().item()
        print(f"3. 自回归第 0 帧 vs 第 1 帧差异: {diff_ar:.6f}")
        if diff_ar < 1e-6:
            print("   ❌ 自回归预测相同！问题在于模型架构")
        else:
            print("   ✓ 自回归产生不同输出")
    
    # 测试 4：检查 past 的哪部分被使用
    with torch.no_grad():
        # 只改变 past 的最后一帧
        past_last_changed = past_norm.clone()
        past_last_changed[:, -1] += 0.1
        
        pred_orig = model._predict_single_frame(past_norm, sign_img)
        pred_last = model._predict_single_frame(past_last_changed, sign_img)
        diff_last = (pred_orig - pred_last).abs().mean().item()
        print(f"4. 只改 past 最后一帧的输出差异: {diff_last:.6f}")
        
        # 只改变 past 的第一帧
        past_first_changed = past_norm.clone()
        past_first_changed[:, 0] += 0.1
        pred_first = model._predict_single_frame(past_first_changed, sign_img)
        diff_first = (pred_orig - pred_first).abs().mean().item()
        print(f"5. 只改 past 第一帧的输出差异: {diff_first:.6f}")
        
        if diff_last > diff_first * 10:
            print("   ✓ 模型更关注最近的帧（符合预期）")
        elif diff_last < diff_first:
            print("   ⚠️ 模型更关注远处的帧（奇怪）")


def tensor_to_pose_complete(t_btjc, header, ref_pose):
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc
    
    t_np = t.detach().cpu().numpy().astype(np.float32)
    arr = t_np[:, None, :, :]
    conf = ref_pose.body.confidence[:len(t_np)].copy()
    fps = ref_pose.body.fps
    
    body = NumPyPoseBody(fps=fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)
    
    if HAS_UNSHIFT:
        try:
            unshift_hands(pose_obj)
        except:
            pass
    
    # Scale & translate
    T_pred = pose_obj.body.data.shape[0]
    ref_arr = np.asarray(ref_pose.body.data[:T_pred, 0], dtype=np.float32)
    pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)
    
    def _var(a):
        c = a.mean(axis=1, keepdims=True)
        return float(((a - c) ** 2).mean())
    
    var_ref = _var(ref_arr)
    var_pred = _var(pred_arr)
    
    if var_pred > 1e-8:
        scale = np.sqrt((var_ref + 1e-6) / (var_pred + 1e-6))
        pose_obj.body.data *= scale
        pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)
    
    ref_c = ref_arr.reshape(-1, 3).mean(axis=0)
    pred_c = pred_arr.reshape(-1, 3).mean(axis=0)
    pose_obj.body.data += (ref_c - pred_c)
    
    return pose_obj


if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    out_dir = "logs/minimal_178_v2"
    os.makedirs(out_dir, exist_ok=True)

    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"

    print("\n" + "=" * 70)
    print("诊断 + 测试脚本 v2")
    print("=" * 70)

    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )

    sample_0 = base_ds[0]

    class FixedSampleDataset(torch.utils.data.Dataset):
        def __init__(self, sample):
            self.sample = sample
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return self.sample

    train_ds = FixedSampleDataset(sample_0)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=zero_pad_collator,
    )

    # Model
    num_joints = sample_0["data"].shape[-2]
    num_dims = sample_0["data"].shape[-1]

    model = LitMinimalAutoregressive(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,
        diffusion_steps=50,
        beta_start=1e-4,
        beta_end=2e-2,
        pred_target="x0",
        train_mode="direct",
        vel_weight=0.5,
        acc_weight=0.25,
        motion_weight=0.1,
    )

    # === 训练前诊断 ===
    print("\n" + "=" * 70)
    print("训练前诊断（随机权重）")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    batch = next(iter(train_loader))
    cond = batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    
    past_norm = model.normalize(past_raw)
    diagnose_model_sensitivity(model, past_norm, sign, device)
    
    # === 训练 ===
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=50,
    )
    
    trainer.fit(model, train_loader)
    
    # === 训练后诊断 ===
    print("\n" + "=" * 70)
    print("训练后诊断")
    print("=" * 70)
    
    model = model.to(device)
    model.eval()
    
    batch = next(iter(train_loader))
    cond = batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(batch["data"][:1]).to(device)
    
    past_norm = model.normalize(past_raw)
    diagnose_model_sensitivity(model, past_norm, sign, device)
    
    # === Inference ===
    print("\n" + "=" * 70)
    print("INFERENCE")
    print("=" * 70)
    
    future_len = gt_raw.size(1)
    
    with torch.no_grad():
        # Baseline
        baseline = gt_raw.mean(dim=1, keepdim=True).repeat(1, future_len, 1, 1)
        mse_base = torch.mean((baseline - gt_raw) ** 2).item()
        disp_base = mean_frame_disp(baseline)
        print(f"Baseline MSE: {mse_base:.4f}, disp: {disp_base:.6f}")
        
        # Prediction
        pred_raw = model.predict_direct(past_raw, sign, future_len=future_len, debug=True)
        
        mse_pred = torch.mean((pred_raw - gt_raw) ** 2).item()
        disp_pred = mean_frame_disp(pred_raw)
        disp_gt = mean_frame_disp(gt_raw)
        
        print(f"\nPrediction MSE: {mse_pred:.4f}")
        print(f"Prediction displacement: {disp_pred:.6f}")
        print(f"GT displacement: {disp_gt:.6f}")
        
        if disp_pred < 1e-6:
            print("\n❌ 预测仍然是静态的！")
        elif disp_pred < disp_gt * 0.1:
            print("\n⚠️ 预测运动太小")
        else:
            print("\n✓ 预测有运动！")
        
        # DTW
        mask = torch.ones(1, future_len, device=device)
        dtw = masked_dtw(pred_raw, gt_raw, mask)
        print(f"DTW: {dtw:.4f}")
    
    # === 保存 .pose ===
    print("\n" + "=" * 70)
    print("保存文件")
    print("=" * 70)
    
    ref_path = base_ds.records[0]["pose"]
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    
    ref_pose = reduce_holistic(ref_pose)
    ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    header = ref_pose.header
    
    out_gt = os.path.join(out_dir, "gt.pose")
    with open(out_gt, "wb") as f:
        ref_pose.write(f)
    print(f"✓ GT: {out_gt}")
    
    pose_pred = tensor_to_pose_complete(pred_raw, header, ref_pose)
    out_pred = os.path.join(out_dir, "pred.pose")
    with open(out_pred, "wb") as f:
        pose_pred.write(f)
    print(f"✓ PRED: {out_pred}")
    
    print("\n✓ 完成！")