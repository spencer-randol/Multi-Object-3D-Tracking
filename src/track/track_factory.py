"""track factory class"""
from track.individual_track import Track
from motion_model.mm_interface import MotionModel
from motion_model.V6d_Model import V6DModel
from motion_model.a9d_Model import V9DModel
from motion_model.p3d_Model import P3Model

from typing import Callable, Literal
import numpy as np
from numpy.typing import NDArray

ModelKind = Literal["p3", "v6", "v9"]

ModelCallable = Callable[[NDArray[np.float64]], MotionModel]


class TrackFactory:
    """Creates Track objects with consistent IDs and motion models."""
    _MODELS: dict[str, ModelCallable] = {
        "p3": P3Model,
        "v6": V6DModel,
        "a9": V9DModel,
    }

    def __init__(self, Model_Kind: ModelKind):
        self._next_id = 1
        self.Model_Kind = Model_Kind

    def create(self, pos: NDArray[np.float64]) -> Track:
        """creates indivdual track object"""
        track_id = self._next_id
        self._next_id += 1

        model = self._MODELS[self.Model_Kind](pos)
        return Track(id=track_id, model=model)
