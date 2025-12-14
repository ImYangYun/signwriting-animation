# -*- coding: utf-8 -*-"
import os
import torch
import torch.nn.functional as F
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitDiffusion, sanitize_btjc
from pose_evaluation.metrics.dtw import PE_DTW



def mean_frame_disp(x_btjc):
    x = sanitize_btjc(x_btjc)
    if x.size(1) < 2:
        return 0.0
    return (x[:, 1:] - x[:, :-1]).abs().mean().item()


def compute_dtw(pred, gt):
    if not HAS_DTW:
        return F.mse_loss(pred, gt).item()
    try:
        dtw_metric = PE_DTW()
        p = pred[0].cpu().numpy().astype("float32")[:, None, :, :]
        g = gt[0].cpu().numpy().astype("float32")[:, None, :, :]
        return float(dtw_metric.get_distance(p, g))
    except:
        return F.mse_loss(pred, gt).item()


def tensor_to_pose(t_btjc, header, ref_pose, gt_btjc=None):
    """转换 tensor 到 pose"""
    t = t_btjc[0] if t_btjc.dim() == 4 else t_btjc
    t_np = t.detach().cpu().numpy().astype(np.float32)
    
    arr = t_np[:, None, :, :]
    conf = ref_pose.body.confidence[:len(t_np)].copy()
    body = NumPyPoseBody(fps=ref_pose.body.fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)
    
    # Scale and translate
    T_pred = len(t_np)
    T_ref = ref_pose.body.data.shape[0]
    future_start = max(0, T_ref - T_pred)
    ref_arr = np.asarray(ref_pose.body.data[future_start:future_start+T_pred, 0], dtype=np.float32)
    pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)
    
    if gt_btjc is not None:
        gt_np = gt_btjc[0].cpu().numpy() if gt_btjc.dim() == 4 else gt_btjc.cpu().numpy()
        var_gt = ((gt_np - gt_np.mean(axis=1, keepdims=True)) ** 2).mean()
        var_ref = ((ref_arr - ref_arr.mean(axis=1, keepdims=True)) ** 2).mean()
        if var_gt > 1e-8:
            scale = np.sqrt(var_ref / var_gt)
            pose_obj.body.data *= scale
            pred_arr = np.asarray(pose_obj.body.data[:T_pred, 0], dtype=np.float32)
    
    delta = ref_arr.reshape(-1, 3).mean(0) - pred_arr.reshape(-1, 3).mean(0)
    pose_obj.body.data += delta
    
    return pose_obj


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    pl.seed_everything(42)

    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = "logs/4sample_test"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("4-Sample Overfit Test")
    print("=" * 70)

    # 配置
    NUM_SAMPLES = 4
    MAX_EPOCHS = 500
    BATCH_SIZE = 4

    # Dataset
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )
    print(f"数据集大小: {len(base_ds)}")

    # 选 4 个样本
    class SubsetDataset(torch.utils.data.Dataset):
        def __init__(self, base, indices):
            self.samples = [base[i] for i in indices]
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    train_ds = SubsetDataset(base_ds, list(range(NUM_SAMPLES)))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=zero_pad_collator)

    # 获取维度
    sample = train_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    
    num_joints, num_dims, future_len = sample.shape[-2], sample.shape[-1], sample.shape[0]
    print(f"J={num_joints}, D={num_dims}, T_future={future_len}")

    # 创建模型（直接用你的 LitDiffusion！）
    model = LitDiffusion(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,
        diffusion_steps=8,
        vel_weight=1.0,
        t_past=40,
        t_future=future_len,
    )

    # 训练
    print("\n开始训练...")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=50,
    )
    trainer.fit(model, train_loader)

    # Inference
    print("\n" + "=" * 70)
    print("Inference")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_batch = zero_pad_collator([train_ds[0]])
    cond = test_batch["conditions"]
    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(test_batch["data"][:1]).to(device)

    with torch.no_grad():
        pred_raw = model.sample(past_raw, sign, future_len)
        
        # 评估
        mse = F.mse_loss(pred_raw, gt_raw).item()
        disp_pred = mean_frame_disp(pred_raw)
        disp_gt = mean_frame_disp(gt_raw)
        disp_ratio = disp_pred / (disp_gt + 1e-8)
        dtw = compute_dtw(pred_raw, gt_raw)
        
        pred_np = pred_raw[0].cpu().numpy()
        gt_np = gt_raw[0].cpu().numpy()
        per_joint_err = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
        mpjpe = per_joint_err.mean()
        pck_01 = (per_joint_err < 0.1).mean() * 100
        pck_02 = (per_joint_err < 0.2).mean() * 100

    print(f"""
结果:
  MSE: {mse:.6f}
  MPJPE: {mpjpe:.6f}
  PCK@0.1: {pck_01:.1f}%
  PCK@0.2: {pck_02:.1f}%
  DTW: {dtw:.6f}
  Disp GT: {disp_gt:.6f}
  Disp Pred: {disp_pred:.6f}
  Disp Ratio: {disp_ratio:.4f}
""")

    # 保存 Pose
    print("保存 Pose 文件...")
    ref_path = base_ds.records[0]["pose"]
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(data_dir, ref_path)
    
    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    ref_pose = reduce_holistic(ref_pose)
    if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
        ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])
    
    # GT
    gt_data = ref_pose.body.data[-future_len:]
    gt_conf = ref_pose.body.confidence[-future_len:]
    gt_pose = Pose(header=ref_pose.header, body=NumPyPoseBody(fps=ref_pose.body.fps, data=gt_data, confidence=gt_conf))
    with open(f"{out_dir}/gt.pose", "wb") as f:
        gt_pose.write(f)
    
    # Pred
    pred_pose = tensor_to_pose(pred_raw, ref_pose.header, ref_pose, gt_raw)
    with open(f"{out_dir}/pred.pose", "wb") as f:
        pred_pose.write(f)
    
    print(f"✓ 保存到 {out_dir}/")

    # 结论
    print("\n" + "=" * 70)
    passed = disp_ratio > 0.5 and pck_01 > 50
    if passed:
        print("🎉 4-Sample Overfit 测试通过！")
    else:
        print("⚠️ 测试未通过")
    print("=" * 70)