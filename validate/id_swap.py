"""promt engineerd plot for tracking id swapping"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

NAME = "111_5"
GATE = 1.0  # set to None to disable gating (recommended to keep some gate)

gt = pd.read_csv(f"data/{NAME}/ground_truth.csv")
res = pd.read_csv(f"data/{NAME}/results.csv")

T = int(gt["frame"].max()) + 1  # assumes GT covers whole run

gt_ids = sorted(gt["id"].unique())
gt_id_to_idx = {gid: i for i, gid in enumerate(gt_ids)}
nG = len(gt_ids)

track_ids = sorted(res["id"].unique())

# -----------------------------
# 1) YOUR VISUAL (unchanged)
# track_id -> array[T] storing nearest GT index each frame (NaN for gaps)
# -----------------------------
# 1) VISUAL (FIXED: enforce 1–1 assignment per frame)
assign = {tid: np.full(T, np.nan) for tid in track_ids}

frames = sorted(set(gt["frame"]).intersection(res["frame"]))
for f in frames:
    gtf = gt[gt["frame"] == f][["id", "x", "y", "z"]]
    rsf = res[res["frame"] == f][["id", "x", "y", "z"]]
    if len(gtf) == 0 or len(rsf) == 0:
        continue

    G = gtf[["x", "y", "z"]].to_numpy()
    R = rsf[["x", "y", "z"]].to_numpy()

    # cost: (n_tracks, n_gt)
    D = cdist(R, G)

    # Hungarian gives 1–1 matches between tracks (rows) and GT (cols)
    r_idx, g_idx = linear_sum_assignment(D)

    if GATE is not None:
        keep = D[r_idx, g_idx] <= GATE
        r_idx, g_idx = r_idx[keep], g_idx[keep]

    r_ids = rsf["id"].to_numpy()
    g_ids = gtf["id"].to_numpy()

    for r_row, g_col in zip(r_idx, g_idx):
        rid = r_ids[r_row]
        gid = g_ids[g_col]
        assign[rid][f] = gt_id_to_idx[gid]

# -----------------------------
# 2) SWAPS YOU ACTUALLY WANT (GT-centric)
# Build frame-aligned mapping: gt_object -> track_id per frame using 1-1 assignment
gt_to_track = np.full((T, nG), np.nan)

for f in frames:
    gtf = gt[gt["frame"] == f][["id", "x", "y", "z"]]
    rsf = res[res["frame"] == f][["id", "x", "y", "z"]]
    if len(gtf) == 0 or len(rsf) == 0:
        continue

    G = gtf[["x", "y", "z"]].to_numpy()
    R = rsf[["x", "y", "z"]].to_numpy()

    # cost (n_gt, n_tracks)
    C = cdist(G, R)
    gi, rj = linear_sum_assignment(C)

    if GATE is not None:
        keep = C[gi, rj] <= GATE
        gi, rj = gi[keep], rj[keep]

    g_ids = gtf["id"].to_numpy()
    r_ids = rsf["id"].to_numpy()

    for g_row, r_col in zip(gi, rj):
        g_global = gt_id_to_idx[g_ids[g_row]]
        gt_to_track[f, g_global] = r_ids[r_col]


def count_swaps_ignore_gaps(track_id_series: np.ndarray) -> int:
    """Count changes in assigned track-id, ignoring NaN gaps entirely."""
    idx = np.flatnonzero(np.isfinite(track_id_series))
    if idx.size <= 1:
        return 0
    vals = track_id_series[idx].astype(int)
    return int(np.sum(vals[1:] != vals[:-1]))


per_gt = []
total_swaps = 0
for j, gid in enumerate(gt_ids):
    swaps = count_swaps_ignore_gaps(gt_to_track[:, j])
    per_gt.append((gid, swaps))
    total_swaps += swaps

avg_swaps_per_object = total_swaps / nG if nG else 0.0

# Console output (like the last one)
print("Per-GT-object swaps (gt_id, swaps):")
for gid, swaps in per_gt:
    print(f"  GT {gid:>3}: swaps={swaps}")

print("\nOverall:")
print(f"  n_gt_objects: {nG}")
print(f"  total_swaps: {total_swaps}")
print(f"  avg_swaps_per_object: {avg_swaps_per_object:.3f}")

# -----------------------------
# 3) PLOT (same visual, just add stats to title)
plt.figure(figsize=(12, 6))
for tid, y in assign.items():
    if np.isfinite(y).any():
        plt.plot(np.arange(T), y, label=f"track {tid}", linewidth=1)

plt.yticks(range(len(gt_ids)), gt_ids)
plt.xlim(0, T - 1)
plt.xlabel("frame")
plt.ylabel("object id")

plt.title(
    f"avg swaps/object = {avg_swaps_per_object:.3f} | total swaps = {total_swaps}"
)
plt.show()
