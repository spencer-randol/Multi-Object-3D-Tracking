"""promt engineerd plot for vizualing results"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

NAME = '123_5'
gt = pd.read_csv(f"data/{NAME}/ground_truth.csv")
res = pd.read_csv(f"data/{NAME}/results.csv")

T = int(max(gt["frame"].max(), res["frame"].max())) + 1

# bounds (use gt if you have it)
mins = gt[["x","y","z"]].min().values
maxs = gt[["x","y","z"]].max().values

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(mins[0], maxs[0])
ax.set_ylim(mins[1], maxs[1])
ax.set_zlim(mins[2], maxs[2])
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

gt_scat = ax.scatter([], [], [], s=15, alpha=0.4)
res_scat = ax.scatter([], [], [], s=30)

# optional trails: keep last K points per track id
K = 20
trail_lines = {}  # id -> line

def set_scatter(scat, df):
    xyz = df[["x","y","z"]].to_numpy()
    if len(xyz) == 0:
        scat._offsets3d = ([], [], [])
    else:
        scat._offsets3d = (xyz[:,0], xyz[:,1], xyz[:,2])

def update(frame):
    ax.set_title(f"Frame {frame}")

    gtf = gt[gt["frame"] == frame]
    rsf = res[res["frame"] == frame]

    set_scatter(gt_scat, gtf)
    set_scatter(res_scat, rsf)

    # trails for results
    for tid, rtrack in res[res["frame"].between(max(0, frame-K+1), frame)].groupby("id"):
        pts = rtrack.sort_values("frame")[["x","y","z"]].to_numpy()
        if tid not in trail_lines:
            (line,) = ax.plot([], [], [], linewidth=1)
            trail_lines[tid] = line
        trail_lines[tid].set_data(pts[:,0], pts[:,1])
        trail_lines[tid].set_3d_properties(pts[:,2])

    return gt_scat, res_scat, *trail_lines.values()

ani = FuncAnimation(fig, update, frames=T, interval=30, blit=False)
plt.show()