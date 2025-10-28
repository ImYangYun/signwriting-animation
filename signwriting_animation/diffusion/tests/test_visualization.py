import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from pose_format import Pose

OUT_DIR = "logs"
GT_PATH = os.path.join(OUT_DIR, "groundtruth.pose")
PRED_PATH = os.path.join(OUT_DIR, "prediction.pose")

def load_pose_33(path):
    """
    读取 .pose，抽取第一个 component (POSE_LANDMARKS, 33点)。
    返回 joints_TJC: (T,33,3)
    """
    with open(path, "rb") as f:
        pose_obj = Pose.read(f)

    data_np = np.array(pose_obj.body.data)  # (T,1,V,3)
    T, P, V, C = data_np.shape
    assert P == 1
    pose33 = data_np[:, 0, 0:33, :]  # (T,33,3)
    return pose33

def normalize_sequence(joints_TJC):
    """
    joints_TJC: (T,33,3).
    我们做两件事：
    1. 只用 (x,y) 投影画，y 取负方便正立
    2. 让人物身体的中点(左右髋平均)居中，并把整体缩放到合适范围
    返回 norm_TJC: (T,33,2) 只保留 (x,y) 2D
    """
    T, J, _ = joints_TJC.shape

    # 选一个躯干中心点：左右髋(23=left_hip,24=right_hip)的平均
    left_hip_idx = 23
    right_hip_idx = 24

    hips_xy = joints_TJC[:, [left_hip_idx, right_hip_idx], 0:2]  # (T,2,2)
    hips_center = hips_xy.mean(axis=1)  # (T,2)

    xy = joints_TJC[:,:,0:2]  # (T,33,2)
    # 平移，使胯中心在(0,0)
    xy_centered = xy - hips_center[:,None,:]  # (T,33,2)

    # 计算一个全局scale，基于肩宽或身高
    # 简单用全序列的最大半径
    radius = np.sqrt((xy_centered**2).sum(axis=2)).max()
    scale = 1.0 / (radius + 1e-6)
    xy_scaled = xy_centered * scale

    # y 方向翻一下（matplotlib里我们想正立，y向上）
    xy_scaled[:,:,1] *= -1.0

    return xy_scaled  # (T,33,2)

def make_video(joints_2d_TJC, out_path, title="seq"):
    """
    joints_2d_TJC: (T,33,2)
    我们用一套手写的人体骨架连接，而不是 header.limbs 那套乱接头部的线。
    """
    T, J, _ = joints_2d_TJC.shape

    # 定义我们自己的骨架拓扑 (MediaPipe Pose style)
    # 这些 index 对应 MediaPipe Pose 33点定义中：肩-肘-腕，髋-膝-踝，肩-肩，髋-髋，肩-髋，脊柱-头
    bones = [
        (11,13), (13,15),        # left shoulder->elbow->wrist
        (12,14), (14,16),        # right shoulder->elbow->wrist
        (23,25), (25,27),        # left hip->knee->ankle
        (24,26), (26,28),        # right hip->knee->ankle
        (11,12),                 # shoulders
        (23,24),                 # hips
        (11,23), (12,24),        # torso lines
        # head-ish: shoulders to nose
        (11,0), (12,0)
    ]

    # 全局bounds，画布范围固定，视觉更稳
    xs = joints_2d_TJC[:,:,0].reshape(-1)
    ys = joints_2d_TJC[:,:,1].reshape(-1)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    pad_x = (x_max - x_min)*0.2 + 1e-5
    pad_y = (y_max - y_min)*0.2 + 1e-5
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    fig, ax = plt.subplots(figsize=(4,4))
    # init lines
    line_elems = []
    for _a,_b in bones:
        line, = ax.plot([], [], linewidth=3)
        line_elems.append((line,_a,_b))
    scatter = ax.scatter([], [], s=20)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    def init():
        artists = [scatter] + [ln for (ln,_,_) in line_elems]
        return artists

    def update(fi):
        pts = joints_2d_TJC[fi]  # (33,2)

        scatter.set_offsets(pts)

        for (ln,a,b) in line_elems:
            if a < J and b < J:
                ln.set_data([pts[a,0], pts[b,0]],
                            [pts[a,1], pts[b,1]])

        ax.set_title(f"{title} t={fi}")
        artists = [scatter] + [ln for (ln,_,_) in line_elems]
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        blit=True,
        interval=200  # ms/frame ~5fps
    )

    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='me'), bitrate=2000)
        ani.save(out_path, writer=writer)
        print(f"✅ Saved MP4 to {out_path}")
    except Exception as e:
        print(f"⚠ ffmpeg failed: {e}")
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        ani.save(gif_path, writer='pillow', fps=5)
        print(f"✅ Saved GIF to {gif_path} (fallback)")

    plt.close(fig)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    gt_3d = load_pose_33(GT_PATH)     # (T,33,3)
    pred_3d = load_pose_33(PRED_PATH) # (T,33,3)

    gt_2d = normalize_sequence(gt_3d)       # (T,33,2)
    pred_2d = normalize_sequence(pred_3d)   # (T,33,2)

    make_video(gt_2d,   os.path.join(OUT_DIR, "groundtruth_clean.mp4"), title="GT")
    make_video(pred_2d, os.path.join(OUT_DIR, "prediction_clean.mp4"),  title="Pred")

if __name__ == "__main__":
    main()
