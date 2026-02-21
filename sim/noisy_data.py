""" generates noisy data"""

import numpy as np
import pandas as pd
from pathlib import Path

SIGMA = 0.1  # measurement std-dev
P_KEPT = 0.99 # probability a true detection is kept
LAMBDA_FA = 2  # avg false alarms per frame

SEED = None
RNG = None
N = None


def gen_dirty_data(seed: int, n: int, box: int):
    """
    generates dirty data

    generates dirty data that is created from
     the ground truth clean data, must be run
     on a seed that has gt data.

    adds a gauisian noise to gt using sigam
    drops 1 - P_DET detections
    adds LAMBDA_FA false detections per frame avg
    uses bounds to make it realistic
    """
    global SEED, RNG, N
    SEED = seed
    RNG = np.random.default_rng(SEED)
    N = n

    gt_path = Path(f"data/{SEED}_{N}/ground_truth.csv")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground truth: {gt_path}")
    gt = pd.read_csv(gt_path)
    rows = []

    for frame, g in gt.groupby('frame', sort=True):
        xyz = g[["x", "y", "z"]].to_numpy()
        N = xyz.shape[0]
        # apply randomized keep mask to drop points
        kepp_mask = RNG.random(N) < P_KEPT
        xyz = xyz[kepp_mask]
        # apply guaisna noise
        if xyz.size:
            xyz = xyz + RNG.normal(0.0, SIGMA, size=xyz.shape)
        # create false alarms
        k = RNG.poisson(LAMBDA_FA)
        if k > 0:
            false_alarm = RNG.uniform(0, box, size=(k, 3))
            xyz = np.vstack([xyz, false_alarm])
        # shuffle frame data
        if xyz.shape[0] > 1:
            RNG.shuffle(xyz)

        for x, y, z in xyz:
            rows.append([int(frame), float(x), float(y), float(z)])

    df = pd.DataFrame(rows, columns=["frame", "x", "y", "z"])
    file = f"data/{SEED}_{N}/detections.csv"
    _path = Path(file)
    df.to_csv(_path, index=False)
