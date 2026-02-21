"""file for 6d aka velocity motion model"""
from motion_model.mm_interface import MotionModel
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import numpy as np
from numpy.typing import NDArray


class V6DModel(MotionModel):
    """6 demention motion model using pos and vel, kf model"""

    DT = 1.0
    # state transtions
    state_trans = np.eye(6)
    state_trans[0, 3] = DT
    state_trans[1, 4] = DT
    state_trans[2, 5] = DT
    # measurement function
    measure_func = np.hstack([np.eye(3), np.zeros((3, 3))])

    #  --- [covarence matrix uncertinty] ----
    # measurement noise
    sigma_m = 0.2
    # acceleration noise
    sigma_a = 1
    # init values
    sigma_p0 = 2 * sigma_m
    sigma_v0 = 25 * sigma_m
    # measurement noise matrix
    measure_noise = np.eye(3) * sigma_m**2

    def __init__(self, pos: NDArray[np.float64]):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        # initial state
        self.kf.x = np.append(pos, (0, 0, 0)).reshape(6, 1)
        # state transition matrix
        self.kf.F = V6DModel.state_trans
        self.kf.H = V6DModel.measure_func
        # initial covariance
        self.kf.P = np.diag([V6DModel.sigma_p0**2, V6DModel.sigma_p0**2,
                             V6DModel.sigma_p0**2, V6DModel.sigma_v0**2,
                             V6DModel.sigma_v0**2, V6DModel.sigma_v0**2])
        # measurement noise
        self.kf.R = V6DModel.measure_noise
        # process noise:
        self.kf.Q = Q_discrete_white_noise(dim=2,
                                           block_size=3,
                                           dt=V6DModel.DT,
                                           var=V6DModel.sigma_a**2)

    def predict(self,) -> NDArray[np.float64]:
        """predicts using kf model"""
        self.kf.predict()
        return self.kf.x[:3].copy()

    def update(self, pos: NDArray[np.float64]) -> None:
        """updates the kf model"""
        self.kf.update(pos.reshape(3, 1))

    def pos(self,) -> NDArray[np.float64]:
        """returns pos of kf model"""
        return self.kf.x[:3]

    def __repr__(self) -> str:
        """
        allows for nice prints & debugs
        """
        px, py, pz, vx, vy, vz = self.kf.x.ravel()
        return (
            f"px:{px}, py:{py}, pz:{pz}, \n"
            f"\tvx:{vx}, vy:{vy}, vz:{vz}")
