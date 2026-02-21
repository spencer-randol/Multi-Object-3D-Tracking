""" viz data promt engineerd"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

INT_MS = 100


def viz_clean_vs_dirty_3d(seed: int, N: int, box: float | None = None):
    """
    3D animated overlay: clean ground truth (circles) vs dirty detections (x's).

    Args:
        seed: dataset seed folder name
        N: dataset N
        box: if provided, sets axes to (0, box) for x/y/z. If None, auto-scales.
    """
    gt_path = f"data/{seed}_{N}/ground_truth.csv"
    det_path = f"data/{seed}_{N}/detections.csv"

    gt = pd.read_csv(gt_path)
    det = pd.read_csv(det_path)

    T = int(gt["frame"].max()) + 1

    if box is not None:
        xlim = (0.0, float(box))
        ylim = (0.0, float(box))
        zlim = (0.0, float(box))
    else:
        mins = gt[["x", "y", "z"]].min().to_numpy()
        maxs = gt[["x", "y", "z"]].max().to_numpy()
        pad = 0.05 * (maxs - mins + 1e-9)
        mins = mins - pad
        maxs = maxs + pad
        xlim, ylim, zlim = (mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set(xlim=xlim, ylim=ylim, zlim=zlim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    gt_sc = ax.scatter([], [], [], s=35, marker="o", label="GT")
    det_sc = ax.scatter([], [], [], s=25, marker="x", label="Detections")
    ax.legend(loc="upper right")

    # Pre-group for speed
    gt_by_frame = {f: g[["x", "y", "z"]].to_numpy() for f, g in gt.groupby("frame")}
    det_by_frame = {f: g[["x", "y", "z"]].to_numpy() for f, g in det.groupby("frame")}

    def update(frame: int):
        g = gt_by_frame.get(frame, np.empty((0, 3)))
        d = det_by_frame.get(frame, np.empty((0, 3)))

        gt_sc._offsets3d = (g[:, 0], g[:, 1], g[:, 2]) if len(g) else ([], [], [])
        det_sc._offsets3d = (d[:, 0], d[:, 1], d[:, 2]) if len(d) else ([], [], [])

        ax.set_title(f"Seed {seed} | Frame {frame} | GT={len(g)} Det={len(d)}")

    _ = FuncAnimation(fig, update, frames=T, interval=INT_MS, blit=False)
    plt.show()
