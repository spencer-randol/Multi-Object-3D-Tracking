from motion_model.mm_interface import MotionModel
from filterpy.kalman import KalmanFilter

import numpy as np
from numpy.typing import NDArray


class P3Model(MotionModel):
    """3 demention motion model using pos, kf model"""

    DT = 1.0
    # state transtions
    state_trans = np.eye(3)
    # measurement function
    measure_func = np.eye(3)

    #  --- [covarence matrix uncertinty] ----
    # measurement noise
    sigma_m = 0.2
    measure_noise = np.eye(3) * sigma_m**2
    # pos noise
    sigma_p = 0.25
    process_noise = np.eye(3) * sigma_p**2
    # init values
    sigma_p0 = 2 * sigma_m

    def __init__(self, pos: NDArray[np.float64]):
        self.kf = KalmanFilter(dim_x=3, dim_z=3)
        # initial state
        self.kf.x = pos.reshape(3, 1)
        # state transition matrix
        self.kf.F = P3Model.state_trans
        self.kf.H = P3Model.measure_func
        # initial covariance
        self.kf.P = np.eye(3) * (P3Model.sigma_p0**2)
        # measurement noise
        self.kf.R = P3Model.measure_noise
        # process noise:
        self.kf.Q = P3Model.process_noise

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
        """allows for nice prints & debugs"""
        px, py, pz = self.kf.x.ravel()
        return f"px:{px}, py:{py}, pz:{pz}"
