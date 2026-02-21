"""script file for generating clean data"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from pathlib import Path

# ------------------------ params
T = 300  # frames
MAX_VEL = 1  # soft speed limit
DAMP = 0.975  # velocity damping
ACC_CORR = 0.05  # accel corilatoin
ACC_SIG = 0.05  # accel noise
WALL_MARGIN = 1.0  # how close to wall before pull
WALL_K = 0.005  # pull stength

N = None
SEED = None
RNG = None
pos = None
vel = None
accel = None
CENTER = None
BOX = None


def create_data() -> None:
    """
    Generates smoothish random motion, to bound the problem i
    intruced a pull factor when objects get clode to the wall that
    gets srtonger the more past the wall they are, this can be
    thought of as a manoover. as well as a soft max of velocity to
    keep the problem space constrained.
    """
    global pos, vel, accel

    for t in range(1, T):
        prev = pos[t - 1]
        pull = np.zeros_like(prev)
        for d in range(3):
            x = prev[:, d]
            pull[:, d] += np.where(x < WALL_MARGIN,  WALL_K * (WALL_MARGIN - x), 0.0)
            pull[:, d] += np.where(x > BOX - WALL_MARGIN,
                                   -WALL_K * (x - (BOX - WALL_MARGIN)), 0.0)
        accel = ACC_CORR * accel + (1.0 - ACC_CORR) * RNG.normal(0.0, ACC_SIG, size=(N, 3))
        accel = accel + pull
        vel = (vel + accel) * DAMP
        speed = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12
        vel *= np.minimum(1.0, MAX_VEL / speed)

        pos[t] = prev + vel


def plot_data() -> None:
    """
    plots data

    plots the per frame data for the ground truth of motion
     sim

    Args:
        None
    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set(xlim=(0, BOX), ylim=(0, BOX), zlim=(0, BOX))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    scat = ax.scatter([], [], [], s=20)

    def update(frame) -> None:
        xyz = pos[frame]
        scat._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        ax.set_title(f"Frame {frame}")

    _ = FuncAnimation(fig, update, frames=T, interval=30, blit=False)
    plt.show()


def write_csv_gt() -> None:
    """
    writes data to csv
    """
    df = pd.DataFrame({
        "frame": np.repeat(np.arange(T), N),
        "id": np.tile(np.arange(N), T),
        "x": pos[:, :, 0].ravel(),
        "y": pos[:, :, 1].ravel(),
        "z": pos[:, :, 2].ravel(),
    })
    file = f"data/{SEED}_{N}/ground_truth.csv"
    _path = Path(file)
    _path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file, index=False)


def Clean_data_gen(seed: int, n: int, box: int, plot: bool = False) -> None:
    """ callable cealn data gen function

    Generates clean data and writes it in csv to data/seed folder
     as ground truth

    Args:
        seed: random seed to use
        n: number of objects to gen
        plot: vizulize data
    Returns:
        None
    """
    global SEED, RNG, N, BOX, CENTER, pos, vel, accel
    SEED = seed
    N = n
    RNG = np.random.default_rng(SEED)
    BOX = box
    CENTER = np.array([BOX/2, BOX/2, BOX/2])
    pos = np.zeros((T, N, 3), dtype=float)
    pos[0] = RNG.uniform(0, BOX, size=(N, 3))
    vel = RNG.normal(0, 0.15, size=(N, 3))
    accel = RNG.normal(0, 0.1, size=(N, 3))
    create_data()
    if plot:
        plot_data()
    write_csv_gt()
