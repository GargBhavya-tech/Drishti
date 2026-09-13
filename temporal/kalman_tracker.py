"""
temporal/kalman_tracker.py

A real constant-velocity Kalman-filter multi-object tracker -- closes
the descope named explicitly in DRISHTI_MASTER_BIBLE.md Part E.2
("The tracker is simpler than the Bible's Kalman-filter design") and
unlocks two things flagged as "designed, not built" elsewhere in the
bible: the fovea controller's c_object term (Part C.13, which needs a
tracked entity to refine around regardless of TTC) and persistent
per-object IDs for the dashboard.

Scope, stated honestly up front: this module tracks 2D position +
velocity (constant-velocity model) per object, built and validated
against SYNTHETIC multi-frame trajectories with known ground truth (see
tests/test_kalman_tracker.py) -- it has NOT been run against a real
continuous multi-frame LiDAR sequence in this session, because none of
this project's three real datasets (RELLIS-3D, nuScenes-mini,
SemanticPOSS) has a wired frame-to-frame detection pipeline feeding
consistent per-frame object centroids yet (perception/
geometric_instance_detector.py and perception/detection_decode.py both
run per-frame, independently, with no continuity between calls). This
tracker's own logic (predict/associate/update/birth/death) is real and
tested against ground-truth trajectories; its performance on a REAL
noisy, ID-switching, sometimes-missed-detection real LiDAR sequence is
a genuinely separate, larger, un-done validation step -- named here
rather than implied by "tested".

Association: Hungarian algorithm (scipy.optimize.linear_sum_assignment)
on pairwise Euclidean distance between predicted track positions and
new-frame detection centroids, gated by MAX_ASSOCIATION_DISTANCE_M --
a real optimal assignment, not greedy nearest-neighbor (which can lock
in a wrong early match that a later, better match could have avoided).

Track lifecycle: a track is TENTATIVE until it accumulates
MIN_HITS_TO_CONFIRM consecutive real detections, then CONFIRMED; a
CONFIRMED track survives up to MAX_COAST_FRAMES consecutive missed
detections (coasting on its own prediction) before being deleted --
standard SORT-family tracker lifecycle, not a novel design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

MAX_ASSOCIATION_DISTANCE_M = 2.0  # a detection farther than this from a track's predicted position is never associated to it
MIN_HITS_TO_CONFIRM = 3  # consecutive matched detections before a track is reported as CONFIRMED, not just TENTATIVE
MAX_COAST_FRAMES = 5  # consecutive missed detections before a CONFIRMED track is deleted
PROCESS_NOISE_STD = 0.5  # m/s^2-scale process noise -- how much we distrust the constant-velocity assumption per step
MEASUREMENT_NOISE_STD = 0.3  # m -- how much we distrust a single detection's centroid


@dataclass
class Track:
    """One tracked object's full Kalman state. `state` = [x, y, vx, vy].
    `covariance` is the 4x4 state covariance matrix."""

    track_id: int
    state: np.ndarray  # (4,): x, y, vx, vy
    covariance: np.ndarray  # (4, 4)
    drishti_class: int
    n_hits: int = 1
    n_missed_consecutive: int = 0
    confirmed: bool = False

    @property
    def position(self) -> np.ndarray:
        return self.state[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[2:]


def _predict(state: np.ndarray, covariance: np.ndarray, dt: float) -> tuple:
    """Constant-velocity motion model: x' = x + vx*dt, v' = v."""
    F = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    Q = (PROCESS_NOISE_STD ** 2) * np.array(
        [
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2],
        ]
    )
    new_state = F @ state
    new_covariance = F @ covariance @ F.T + Q
    return new_state, new_covariance


def _update(state: np.ndarray, covariance: np.ndarray, measurement_xy: np.ndarray) -> tuple:
    """Standard Kalman measurement update -- measurement is (x, y) only
    (velocity is never directly observed, only inferred over time)."""
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    R = (MEASUREMENT_NOISE_STD ** 2) * np.eye(2)

    y = measurement_xy - H @ state  # innovation
    S = H @ covariance @ H.T + R
    K = covariance @ H.T @ np.linalg.inv(S)  # Kalman gain

    new_state = state + K @ y
    new_covariance = (np.eye(4) - K @ H) @ covariance
    return new_state, new_covariance


class MultiObjectTracker:
    """Maintains a set of Kalman-filter Tracks across successive calls
    to `update()`, one call per frame. See module docstring for scope
    and validation status."""

    def __init__(
        self,
        max_association_distance_m: float = MAX_ASSOCIATION_DISTANCE_M,
        min_hits_to_confirm: int = MIN_HITS_TO_CONFIRM,
        max_coast_frames: int = MAX_COAST_FRAMES,
    ):
        self.tracks: List[Track] = []
        self._next_id = 0
        self.max_association_distance_m = max_association_distance_m
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_coast_frames = max_coast_frames

    def update(self, detections_xy: np.ndarray, detection_classes: np.ndarray, dt: float) -> List[Track]:
        """`detections_xy`: (M, 2) real-valued centroids for this frame's
        detections. `detection_classes`: (M,) DrishtiClass ints, same
        order. `dt`: seconds since the last call. Returns the current
        list of tracks (both tentative and confirmed) AFTER this
        frame's predict/associate/update/birth/death cycle."""
        for track in self.tracks:
            track.state, track.covariance = _predict(track.state, track.covariance, dt)

        n_tracks = len(self.tracks)
        n_dets = detections_xy.shape[0]
        matched_track_idx = set()
        matched_det_idx = set()

        if n_tracks > 0 and n_dets > 0:
            cost = np.zeros((n_tracks, n_dets))
            for i, track in enumerate(self.tracks):
                cost[i, :] = np.linalg.norm(detections_xy - track.position[None, :], axis=1)

            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] <= self.max_association_distance_m:
                    track = self.tracks[r]
                    track.state, track.covariance = _update(track.state, track.covariance, detections_xy[c])
                    track.drishti_class = int(detection_classes[c])
                    track.n_hits += 1
                    track.n_missed_consecutive = 0
                    if track.n_hits >= self.min_hits_to_confirm:
                        track.confirmed = True
                    matched_track_idx.add(r)
                    matched_det_idx.add(c)

        for i, track in enumerate(self.tracks):
            if i not in matched_track_idx:
                track.n_missed_consecutive += 1

        self.tracks = [t for t in self.tracks if t.n_missed_consecutive <= self.max_coast_frames]

        for j in range(n_dets):
            if j not in matched_det_idx:
                init_state = np.array([detections_xy[j, 0], detections_xy[j, 1], 0.0, 0.0])
                init_covariance = np.diag([MEASUREMENT_NOISE_STD**2, MEASUREMENT_NOISE_STD**2, 4.0, 4.0])
                self.tracks.append(
                    Track(
                        track_id=self._next_id,
                        state=init_state,
                        covariance=init_covariance,
                        drishti_class=int(detection_classes[j]),
                    )
                )
                self._next_id += 1

        return self.tracks

    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.confirmed]
