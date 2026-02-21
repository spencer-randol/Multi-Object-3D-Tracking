"""cli for running data"""
from track.track_manager import Track_manager
from track.track_factory import TrackFactory, ModelKind

import time

import pandas as pd
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class CLI():

    def __init__(self, gate, model: ModelKind):
        self.track_m = None
        self.results = None
        self.gate = gate
        self.model = model
        self.factory = TrackFactory(model)

    def run(self, path: str) -> None:
        """
        runs full tracking pipeline

        Args:
            path: path to data file eg /data/252_1/
        """
        self.track_m = Track_manager(self.gate, self.factory)
        self.results = []
        dt_path = Path(f"{path}/detections.csv")
        results_path = Path(f"{path}/results{self.model}.csv")
        if not dt_path.exists():
            raise FileNotFoundError(f"Missing detections: {dt_path}")

        frame_list = self._load_detections(dt_path)
        # add timer for fps testing
        t_start = time.perf_counter()
        for frame in frame_list:
            tracked = self.track_m.update(frame)
            t_elapsed = time.perf_counter() - t_start
            for row in tracked:
                row.append(t_elapsed)
            self.results.extend(tracked)
        print(self.track_m)

        self._dump_results(results_path)

    def _load_detections(self, path: Path) -> list[NDArray[np.float64]]:
        """
        loads detections:

        Args:
            path to detetcions
        Returns:
            list of frames with each being nx3 detections
            note n can differ per frame
        """
        detections = pd.read_csv(path)
        frames = []
        for frame, g in detections.groupby('frame', sort=True):
            xyz = g[["x", "y", "z"]].to_numpy()
            frames.append(xyz)
        return frames

    def _dump_results(self, path: str):
        """
        dumps detections into a csv

        Args:
            path:
            tracked:
        """
        cols = ["frame", "id", "x", "y", "z", "time(s)"]
        df = pd.DataFrame(self.results, columns=cols)
        self.results = None
        self.track_m = None
        df.to_csv(path, index=False)
