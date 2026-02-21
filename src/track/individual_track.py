""" file for specfic track instens"""
from motion_model.mm_interface import MotionModel

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Track:
    """track class used to track a single object"""
    id: int
    model: MotionModel
    max_time_since_update: int = 3

    age: int = 0
    hits: int = 0
    time_since_update: int = 0

    def predict(self) -> NDArray[np.float64]:
        """
        predicts movement of track

        predicticst the motion of the current track object
         by using the kalman filter, returning the gues of
         new postion as well as incrimentg the age of the
         track

        Returns:
            temp:
        """
        self.age += 1
        self.time_since_update += 1
        return self.model.predict()

    def update(self, pos: NDArray[np.float64]) -> None:
        """
        updates the kalman motion model

        Args:
            new pos assighned to track
        Returns:
            None
        """
        self.hits += 1
        self.time_since_update = 0
        self.model.update(pos)

    def pos(self) -> NDArray[np.float64]:
        """returns the pos of the track"""
        return self.model.pos()

    def is_dead(self) -> bool:
        """
        check if a track is dead

        track is dead if time_since_update is greater
         then the max allowed

        """
        return self.time_since_update >= self.max_time_since_update

    def __repr__(self) -> str:
        """
        allows for nice prints & debugs
        """
        return (
            f"Track(id={self.id}, age={self.age}, hits={self.hits}, "
            f"tsu={self.time_since_update},\n\t model=[{self.model}])")
