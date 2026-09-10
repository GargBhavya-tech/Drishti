"""
observability/negative_obstacle.py

Ticket #36 -- negative obstacle detection.

For each (ring, azimuth) with a ground-plane fit (Ticket #35), compare
the measured range against `expected_ground_range`. Flags when the
residual is large RELATIVE TO ITS NEIGHBOURS IN THE SAME COLUMN --
ring-to-ring local inconsistency, never an absolute threshold.

Build Map "Watch out" #1: an absolute-threshold version fires
everywhere on a slope, because a slope's fit residual grows smoothly
with ring index even when the fit is good -- that smooth growth looks
identical to a real gap under an absolute threshold. The ring-to-ring
DIFFERENCE stays small on a slope (each ring's error is close to its
neighbours') and only spikes at a genuine local gap, which is what this
module actually thresholds on.

Watch out #2: order matters -- carve (#34) BEFORE testing. Occlusion
behind a truck produces the exact same missing-return signature as a
ditch; `occluded_mask` must be computed from the ALREADY-CARVED Clipmap
and passed in, and any cell it marks True is skipped outright here,
never scored.

Range shadow reference (Bible): a ditch of depth d at range r, sensor
mounted at height h, produces Delta ~= r*d/h -- used below only to size
a synthetic test case, not hardcoded into the detector itself (the
detector never assumes a specific ditch geometry).

Persistence: a single frame's anomaly only reaches SUSPECT.
`PersistenceTracker` promotes SUSPECT -> NEGATIVE_OBSTACLE after the
SAME world cell is flagged on >= 3 CONSECUTIVE frames (a miss on any
frame resets that cell's streak) -- kept as its own small state machine,
decoupled from the geometry-heavy per-frame detector, so persistence
behaviour can be tested (miss-resets-streak, non-consecutive frames
never promote) without needing a live sensor/world simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, Set

import numpy as np

DEFAULT_ANOMALY_MARGIN_M = 1.0
DEFAULT_PERSISTENCE_FRAMES = 3


def detect_anomalous_cells(
    measured_range: np.ndarray,
    expected_range: np.ndarray,
    valid_expected: np.ndarray,
    occluded_mask: np.ndarray,
    usable_range_m: float,
    anomaly_margin_m: float = DEFAULT_ANOMALY_MARGIN_M,
) -> Set[tuple]:
    """One frame's (ring, azimuth) anomaly detection.

    `measured_range`: (H, W) float, NaN where no return was received.
    `expected_range`: (H, W) float, `expected_ground_range`'s output per
    cell (undefined where `valid_expected` is False).
    `valid_expected`: (H, W) bool -- False wherever Ticket #35's fit
    could not predict this cell (too few confirmed ground points, or a
    beam parallel to the local slope) -- such cells are skipped, not
    scored, since there is nothing to compare against.
    `occluded_mask`: (H, W) bool -- True wherever the ALREADY-CARVED
    (Ticket #34) Clipmap reports OCCLUDED for this cell's world
    position. Skipped outright (Watch out #2).

    Returns the set of (ring, azimuth_col) cells flagged this frame.
    """
    H, W = measured_range.shape
    no_return = np.isnan(measured_range)
    # A beam that never returned went at least to the sensor's usable
    # range without finding ground -- the strongest possible evidence of
    # a gap, standing in for an unknown (larger) true range.
    effective_measured = np.where(no_return, usable_range_m, measured_range)
    residual = effective_measured - expected_range  # positive = beam went farther than expected ground

    flagged: Set[tuple] = set()
    for col in range(W):
        for ring in range(H):
            if not valid_expected[ring, col] or occluded_mask[ring, col]:
                continue
            if residual[ring, col] <= 0:
                continue  # closer than expected = an obstacle, not a gap

            neighbour_residuals = []
            for nb_ring in (ring - 1, ring + 1):
                if 0 <= nb_ring < H and valid_expected[nb_ring, col] and not occluded_mask[nb_ring, col]:
                    neighbour_residuals.append(residual[nb_ring, col])
            if not neighbour_residuals:
                continue  # no neighbour to compare against -- cannot judge local inconsistency

            baseline = min(neighbour_residuals)
            if (residual[ring, col] - baseline) > anomaly_margin_m:
                flagged.add((ring, col))

    return flagged


@dataclass
class PersistenceTracker:
    """Ticket #36's >= 3-CONSECUTIVE-scans promotion, keyed by whatever
    hashable world-cell identity the caller uses (e.g. (level, gi, gj)
    from Ticket #10's addressing) -- deliberately NOT keyed by
    (ring, azimuth), since those shift every frame as the vehicle moves
    and the same physical gap would never accumulate persistence under
    sensor-relative coordinates."""

    required_frames: int = DEFAULT_PERSISTENCE_FRAMES
    _streaks: Dict[Hashable, int] = field(default_factory=dict)

    def update(self, flagged_world_cells: Iterable[Hashable]) -> Set[Hashable]:
        """One frame's flagged world cells -> the set now promoted to
        NEGATIVE_OBSTACLE (streak >= required_frames). A cell flagged
        this frame but not on the previous one starts a FRESH streak at
        1 -- persistence requires consecutive frames, not a running
        total; any cell not flagged this frame has its streak reset to
        zero (dropped from the tracker) even if it was previously
        SUSPECT."""
        flagged = set(flagged_world_cells)
        promoted: Set[Hashable] = set()

        for cell in list(self._streaks.keys()):
            if cell not in flagged:
                del self._streaks[cell]  # missed a frame -- streak resets

        for cell in flagged:
            self._streaks[cell] = self._streaks.get(cell, 0) + 1
            if self._streaks[cell] >= self.required_frames:
                promoted.add(cell)

        return promoted
