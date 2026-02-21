"""file for 9d aka acceleration motion model"""
from motion_model.mm_interface import MotionModel
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import numpy as np
from numpy.typing import NDArray


class V9DModel(MotionModel):
    """9 dimension motion model using pos, vel, accel, kf model"""

    DT = 1.0

    # state transtions
    state_trans = np.eye(9)
    state_trans[0, 3] = DT
    state_trans[1, 4] = DT
    state_trans[2, 5] = DT
    state_trans[0, 6] = 0.5 * DT * DT
    state_trans[1, 7] = 0.5 * DT * DT
    state_trans[2, 8] = 0.5 * DT * DT
    state_trans[3, 6] = DT
    state_trans[4, 7] = DT
    state_trans[5, 8] = DT

    # measurement function
    measure_func = np.hstack([np.eye(3), np.zeros((3, 6))])

    # --- [covariance / uncertainty] ---
    # measurement noise
    sigma_m = 0.2
    # process noise driver (jerk std)
    sigma_j = .5

    # init values
    sigma_p0 = 2 * sigma_m
    sigma_v0 = 5 * sigma_m
    sigma_a0 = 10 * sigma_m

    # measurement noise matrix
    measure_noise = np.eye(3) * sigma_m**2

    def __init__(self, pos: NDArray[np.float64]):
        self.kf = KalmanFilter(dim_x=9, dim_z=3)
        # initial state
        self.kf.x = np.concatenate([
            pos, np.zeros(6, dtype=np.float64)]).reshape(9, 1)
        # state transition matrix
        self.kf.F = V9DModel.state_trans
        self.kf.H = V9DModel.measure_func
        # initial covariance
        self.kf.P = np.diag([
            V9DModel.sigma_p0**2, V9DModel.sigma_p0**2, V9DModel.sigma_p0**2,
            V9DModel.sigma_v0**2, V9DModel.sigma_v0**2, V9DModel.sigma_v0**2,
            V9DModel.sigma_a0**2, V9DModel.sigma_a0**2, V9DModel.sigma_a0**2,
        ])
        # measurement noise
        self.kf.R = V9DModel.measure_noise
        # process noise:
        self.kf.Q = Q_discrete_white_noise(
            dim=3,
            block_size=3,
            dt=V9DModel.DT,
            var=V9DModel.sigma_j**2,
        )

    def predict(self) -> NDArray[np.float64]:
        """predicts using kf model"""
        self.kf.predict()
        return self.kf.x[:3].copy()

    def update(self, pos: NDArray[np.float64]) -> None:
        """updates the kf model"""
        self.kf.update(pos.reshape(3, 1))

    def pos(self) -> NDArray[np.float64]:
        """returns pos of kf model"""
        return self.kf.x[:3]

    def __repr__(self) -> str:
        """
        allows for nice prints & debugs
        """
        px, py, pz, vx, vy, vz, ax, ay, az = self.kf.x.ravel()
        return (
            f"px:{px}, py:{py}, pz:{pz}, \n"
            f"\tvx:{vx}, vy:{vy}, vz:{vz}, \n"
            f"\tax:{ax}, ay:{ay}, az:{az}"
        )
