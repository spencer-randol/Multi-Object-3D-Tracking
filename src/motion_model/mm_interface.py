"""motion model interface"""

from __future__ import annotations
from typing import Protocol
import numpy as np
from numpy.typing import NDArray


class MotionModel(Protocol):
    """interface for motion models"""
    def predict(self,) -> NDArray[np.float64]: ...
    """predicts where the object should be"""
    def update(self, pos: NDArray[np.float64]) -> None: ...
    """updates the motion model with whre it was"""
    def pos(self,) -> NDArray[np.float64]: ...
    """returns the current postion of model"""
