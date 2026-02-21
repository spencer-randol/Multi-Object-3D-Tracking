"""file for class magning tracks"""
from track.individual_track import Track
from track.track_factory import TrackFactory

from typing import List

import numpy as np
from numpy.typing import NDArray

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class Track_manager:

    def __init__(self, gate: float, factory: TrackFactory):
        self.tracks: List[Track] = []
        self.frame_idx = -1
        self.gate = gate
        self.factory = factory

    def update(self, detections: NDArray):
        """
        reuns tracking on detetcions

        Long sum TODO

        Args:
            detections: nx3 array of detections
        """
        self.frame_idx += 1
        # predictions of current tracked as m x 3
        if len(self.tracks) == 0:
            for detetcion in detections:
                self.tracks.append(self.factory.create(detetcion))
            return self._tracks_to_output()
        else:
            pred_pos = np.asarray(
                [t.predict()[:3].ravel()for t in self.tracks])
        # if no detection stop after predictions
        if len(detections) == 0:
            return self._tracks_to_output()
        # nxm cost matrix of euclidean distance
        Cost_matrix = cdist(pred_pos, detections, metric='euclidean')
        # hungarian flow
        row_ind, col_ind = linear_sum_assignment(Cost_matrix)
        # gate
        cost_match = Cost_matrix[row_ind, col_ind]
        keep_mask = cost_match <= self.gate
        m_row, m_col = row_ind[keep_mask], col_ind[keep_mask]
        matches = np.stack([m_row, m_col], axis=1)
        # update matches
        for r, c in matches:
            self.tracks[r].update(detections[c])
        # update detections
        det_used = np.zeros(Cost_matrix.shape[1], dtype=bool)
        det_used[m_col] = True
        not_matched_dets = np.flatnonzero(~det_used)
        for i in not_matched_dets:
            new_track = self.factory.create(detections[i])
            self.tracks.append(new_track)
        # drop bad tracks
        self.tracks = [t for t in self.tracks if not t.is_dead()]
        return self._tracks_to_output()

    def _tracks_to_output(self,) -> List[list]:
        """
        Trurns tracks to output

        reuturns a list of lists:
            inner list
            [frame_idx, x, y, z, vx, vy, vz]
        """
        rows = []
        for t in self.tracks:
            # validate track before returning
            if t.age > t.max_time_since_update + 1:
                x, y, z = t.pos().ravel()
                rows.append([self.frame_idx, t.id, x, y, z])
        return rows

    def __repr__(self,):
        """for pretty prints and debugg"""
        header = (f"Track_manager(frame_idx={self.frame_idx},"
                  f'gate={self.gate},'
                  f"n_tracks={len(self.tracks)})")
        if not self.tracks:
            return header
        tracks_str = ",\n  ".join(repr(t) for t in self.tracks)
        return f"{header}\n[\n  {tracks_str}\n]"
