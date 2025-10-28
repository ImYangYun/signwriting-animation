import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from pose_format import Pose
from pose_format.pose import PoseHeader
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer


REF_POSE_PATH = "/data/yayun/pose_data/ss73703bb95c9fc3ba670662f4c4429320.pose"
NORM_GT_PATH  = "logs/groundtruth.pose"
NORM_PR_PATH  = "logs/prediction.pose"
OUT_DIR = "logs"


def load_pose(path):
    with open(path, "rb") as f:
        return Pose.read(f)


def pose_to_numpy_TJ3(p: Pose):
    """
    Pose.body.data is (T,P,J,3) masked array.
    Return dense np array [T,J,3].
    """
    arr = p.body.data.filled(np.nan)  # (T,P,J,3)
    # assume 1 person
    arr = arr[:, 0, :, :]  # (T,J,3)
    return arr


def fit_scale_shift(ref_xyz, norm_xyz):
    """
    Solve ref ≈ s * norm + t, using least squares on frame 0.
    ref_xyz: [J,3] world/original coords
    norm_xyz:[J,3] normalized coords
    returns s (scalar float), t (3,)
    """
    # We'll solve per-dim together by stacking equations.
    # Let’s write: ref_j = s * norm_j + t
    # => ref_j - t = s * norm_j
    # It's easier to solve s and t by linear regression:
    # We solve for each coord dimension independently, but force same s across x,y,z.
    # We'll estimate s as argmin sum_j || ref_j - (s*norm_j + t) ||^2

    J = ref_xyz.shape[0]
    # Build design matrix for least squares:
    # For each joint j and coord c:
    # ref_xyz[j,c] ≈ s * norm_xyz[j,c] + t_c
    # Unknowns are: s (1 scalar), t_x, t_y, t_z (3 scalars) -> total 4 unknowns.

    A_list = []
    b_list = []
    for j in range(J):
        nx, ny, nz = norm_xyz[j]
        rx, ry, rz = ref_xyz[j]

        # x coord eq: rx ≈ s*nx + tx
        A_list.append([nx, 1.0, 0.0, 0.0])
        b_list.append(rx)

        # y coord eq: ry ≈ s*ny + ty
        A_list.append([ny, 0.0, 1.0, 0.0])
        b_list.append(ry)

        # z coord eq: rz ≈ s*nz + tz
        A_list.append([nz, 0.0, 0.0, 1.0])
        b_list.append(rz)

    A = np.array(A_list, dtype=np.float64)  # (3J,4)
    b = np.array(b_list, dtype=np.float64)  # (3J,)

    # Solve least squares
    # x = [s, tx, ty, tz]
    x_hat, *_ = np.linalg.lstsq(A, b, rcond=None)
    s  = x_hat[0]
    tx = x_hat[1]
    ty = x_hat[2]
    tz = x_hat[3]
    t = np.array([tx, ty, tz], dtype=np.float32)

    return float(s), t.astype(np.float32)


def apply_scale_shift(norm_seq, s, t):
    """
    norm_seq: [T,J,3] normalized
    s: scalar
    t: [3]
    return [T,J,3] restored
    """
    return norm_seq * s + t[None, None, :]


def build_pose_from_xyz(restored_xyz, header_ref, fps):
    """
    restored_xyz: [T,J,3] world-like coords we just restored
    header_ref: PoseHeader with the right components ordering
    fps: float
    return Pose
    """
    T, J, C = restored_xyz.shape
    data_TPJC = restored_xyz[:, np.newaxis, :, :]          # (T,1,J,3)
    confidence = np.ones((T, 1, J), dtype=np.float32)      # (T,1,J)

    body = NumPyPoseBody(
        fps=fps,
        data=data_TPJC,
        confidence=confidence
    )
    return Pose(header_ref, body)


def drop_world(p: Pose):
    """
    Use pose_format's built-in API:
    remove "POSE_WORLD_LANDMARKS" component,
    so we don't get ghost person in top-left.
    """
    return p.remove_components("POSE_WORLD_LANDMARKS")


def render_pose_video(pose_obj, out_path, title_prefix="SEQ"):
    """
    Same renderer as之前，固定视角，PoseVisualizer.draw(frame).
    """
    viz = PoseVisualizer(pose_obj)
    T = pose_obj.body.data.shape[0]

    data_np = pose_obj.body.data.filled(np.nan)  # (T,1,J,3)
    xy = data_np[..., :2]                        # (T,1,J,2)
    x_min = np.nanmin(xy[...,0]); x_max = np.nanmax(xy[...,0])
    y_min = np.nanmin(xy[...,1]); y_max = np.nanmax(xy[...,1])
    pad_x = (x_max - x_min)*0.1 + 1e-5
    pad_y = (y_max - y_min)*0.1 + 1e-5
    x_min -= pad_x; x_max += pad_x
    y_min -= pad_y; y_max += pad_y

    fig, ax = plt.subplots(figsize=(5,5))

    def init():
        ax.cla()
        ax.set_aspect("equal","box")
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xticks([]); ax.set_yticks([])
        viz.draw(ax, frame_id=0)
        ax.set_title(f"{title_prefix} t=0")
        return ax,

    def update(i):
        ax.cla()
        ax.set_aspect("equal","box")
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xticks([]); ax.set_yticks([])
        viz.draw(ax, frame_id=i)
        ax.set_title(f"{title_prefix} t={i}")
        return ax,

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=T,
        interval=200,
        blit=False
    )

    # try mp4 via ffmpeg, else gif
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='pose_format'), bitrate=2400)
        anim.save(out_path, writer=writer)
        print(f"✅ Saved MP4: {out_path}")
    except Exception as e:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        anim.save(gif_path, writer='pillow', fps=5)
        print(f"⚠ ffmpeg failed ({e}), saved GIF instead: {gif_path}")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 读三个pose
    pose_ref  = load_pose(REF_POSE_PATH)   # 原始: 正常人大小
    pose_norm_gt = load_pose(NORM_GT_PATH) # 我们导出: 小点
    try:
        pose_norm_pred = load_pose(NORM_PR_PATH)
    except Exception:
        pose_norm_pred = None
        print("⚠ No prediction.pose or failed to load prediction.pose")

    # 2. 取numpy
    ref_xyz_full  = pose_to_numpy_TJ3(pose_ref)      # [T_ref,J_ref,3]
    norm_xyz_gt   = pose_to_numpy_TJ3(pose_norm_gt)  # [Tn_gt,Jn_gt,3]
    print("ref_xyz_full.shape =", ref_xyz_full.shape)
    print("norm_xyz_gt.shape  =", norm_xyz_gt.shape)

    # 3. 我们只能拟合在重叠的 joints / 帧上
    #    假设关节数量一致 (J_ref == J_norm)
    #    如果不一致，就截成 min
    Tref, Jref, _ = ref_xyz_full.shape
    Tgt,  Jgt,  _ = norm_xyz_gt.shape
    Jmin = min(Jref, Jgt)
    # 用第0帧
    ref0  = ref_xyz_full[0, :Jmin, :]    # (Jmin,3)
    norm0 = norm_xyz_gt[0, :Jmin, :]     # (Jmin,3)

    # 4. 拟合 scale+shift
    s, t = fit_scale_shift(ref0, norm0)
    print(f"[fit] scale s={s}, shift t={t}")

    # 5. 反归一化 groundtruth
    #    对齐J维度再恢复
    norm_xyz_gt_use = norm_xyz_gt[:, :Jmin, :]
    restored_gt_xyz = apply_scale_shift(norm_xyz_gt_use, s, t)  # [Tgt,Jmin,3]

    # 用 ref 的 header，因为那是原始世界坐标定义（最靠谱）
    header_ref = pose_ref.header
    fps_ref = getattr(pose_ref.body, "fps", 5.0)

    pose_restored_gt = build_pose_from_xyz(restored_gt_xyz, header_ref, fps=fps_ref)

    # 6. 同样还原 prediction（如果有）
    pose_restored_pred = None
    if pose_norm_pred is not None:
        norm_xyz_pred = pose_to_numpy_TJ3(pose_norm_pred)  # [Tp,Jp,3]
        Tp, Jp, _ = norm_xyz_pred.shape
        Jmin2 = min(Jref, Jp)
        norm_xyz_pred_use = norm_xyz_pred[:, :Jmin2, :]
        restored_pred_xyz = apply_scale_shift(norm_xyz_pred_use, s, t)
        pose_restored_pred = build_pose_from_xyz(restored_pred_xyz, header_ref, fps=fps_ref)
    else:
        print("⚠ skip pred restore (no pred pose)")

    # 7. 去掉 world component用于可视化 (ghost人)
    pose_vis_gt   = pose_restored_gt.remove_components("POSE_WORLD_LANDMARKS")
    pose_vis_pred = pose_restored_pred.remove_components("POSE_WORLD_LANDMARKS") if pose_restored_pred else None

    # 8. 导出新的 .pose (unnorm)
    out_gt_pose_path = os.path.join(OUT_DIR, "groundtruth_unnorm.pose")
    with open(out_gt_pose_path, "wb") as f:
        pose_restored_gt.write(f)
    print("✅ wrote", out_gt_pose_path)

    if pose_restored_pred is not None:
        out_pr_pose_path = os.path.join(OUT_DIR, "prediction_unnorm.pose")
        with open(out_pr_pose_path, "wb") as f:
            pose_restored_pred.write(f)
        print("✅ wrote", out_pr_pose_path)

    # 9. 导出可视化视频 (用vis版本，去world后)
    render_pose_video(pose_vis_gt, os.path.join(OUT_DIR, "groundtruth_unnorm.mp4"), title_prefix="GT (unnorm)")
    if pose_vis_pred is not None:
        render_pose_video(pose_vis_pred, os.path.join(OUT_DIR, "prediction_unnorm.mp4"), title_prefix="PRED (unnorm)")


if __name__ == "__main__":
    main()
