"""
tests/test_negative_obstacle.py

Ticket #36's own test:
- Synthetic flat ground -> zero detections.
- Flat ground with a 0.5m x 2m ditch at 15m -> detected, with
  Delta ~= 15 * 0.5 / 1.84 = 4.08m on nuScenes.
- Constant 8 degree downslope -> zero detections ("write this test
  first; it guards the fix that stops the demo embarrassing you on a
  hill").
- Occlusion behind a synthetic wall -> not flagged.
"Done when: the slope test passes with zero false positives."

Plus PersistenceTracker's own >= 3-consecutive-scans contract.
"""

from __future__ import annotations

import numpy as np

from observability.negative_obstacle import PersistenceTracker, detect_anomalous_cells

USABLE_RANGE_M = 120.0
H, W = 8, 4  # rings, azimuth columns


def _base_expected(ring_range_m: float = 15.0) -> tuple:
    """8 rings x 4 columns, every column identical: expected_ground_range
    grows slightly with ring index (as it would moving down a real beam
    ladder toward the sensor's nadir) -- a fixed, arbitrary but valid
    per-ring baseline shared by every scenario below."""
    expected = np.tile((ring_range_m - np.arange(H) * 0.3).reshape(H, 1), (1, W))
    valid = np.ones((H, W), dtype=bool)
    return expected, valid


def test_flat_ground_zero_detections():
    expected, valid = _base_expected()
    measured = expected.copy()  # perfect ground returns, no gap anywhere
    occluded = np.zeros((H, W), dtype=bool)

    flagged = detect_anomalous_cells(measured, expected, valid, occluded, USABLE_RANGE_M)
    assert flagged == set()


def test_ditch_at_15m_is_detected_with_expected_delta_magnitude():
    expected, valid = _base_expected(ring_range_m=15.0)
    measured = expected.copy()

    # nuScenes-scale ditch: r=15m, d=0.5m, h=1.84m -> Delta ~= 4.08m.
    r, d, h = 15.0, 0.5, 1.84
    delta = r * d / h
    ditch_ring, ditch_col = 3, 1
    measured[ditch_ring, ditch_col] = expected[ditch_ring, ditch_col] + delta

    occluded = np.zeros((H, W), dtype=bool)
    flagged = detect_anomalous_cells(measured, expected, valid, occluded, USABLE_RANGE_M)

    assert (ditch_ring, ditch_col) in flagged
    # Every OTHER cell in that column stayed clean ground -- must not
    # also be flagged just because their neighbour was.
    assert flagged == {(ditch_ring, ditch_col)}


def test_constant_8_degree_downslope_zero_false_positives():
    # A slope makes the RAW residual (if compared against a fixed
    # baseline) grow smoothly and non-trivially with ring index -- an
    # absolute-threshold detector would fire here. The ring-to-ring
    # DIFFERENCE stays constant and small, so the actual (relative)
    # detector must not fire anywhere.
    expected, valid = _base_expected()
    slope_residual_per_ring_m = 0.4  # smooth, systematic, well above the anomaly margin in absolute terms
    measured = expected + (np.arange(H) * slope_residual_per_ring_m).reshape(H, 1)
    occluded = np.zeros((H, W), dtype=bool)

    flagged = detect_anomalous_cells(measured, expected, valid, occluded, USABLE_RANGE_M)
    assert flagged == set(), "absolute residual grows with ring index but the RELATIVE (neighbour) discriminator must ignore a smooth slope trend"


def test_occlusion_behind_a_wall_is_not_flagged():
    expected, valid = _base_expected()
    measured = expected.copy()

    # Same missing-return signature as the ditch case, but this cell is
    # marked OCCLUDED (Ticket #34 already carved it that way, e.g. a
    # truck in front) -- must be skipped, not flagged.
    occ_ring, occ_col = 3, 1
    measured[occ_ring, occ_col] = np.nan  # no return at all
    occluded = np.zeros((H, W), dtype=bool)
    occluded[occ_ring, occ_col] = True

    flagged = detect_anomalous_cells(measured, expected, valid, occluded, USABLE_RANGE_M)
    assert flagged == set()


def test_no_return_at_all_is_treated_as_strong_gap_evidence():
    expected, valid = _base_expected()
    measured = expected.copy()
    gap_ring, gap_col = 3, 1
    measured[gap_ring, gap_col] = np.nan  # beam never returned -- ran out to usable range
    occluded = np.zeros((H, W), dtype=bool)

    flagged = detect_anomalous_cells(measured, expected, valid, occluded, USABLE_RANGE_M)
    assert (gap_ring, gap_col) in flagged


# ---------------------------------------------------------------------------
# PersistenceTracker
# ---------------------------------------------------------------------------


def test_persistence_promotes_only_after_three_consecutive_frames():
    tracker = PersistenceTracker(required_frames=3)
    cell = ("L0", 10, 20)

    assert tracker.update([cell]) == set()          # frame 1 -- streak 1
    assert tracker.update([cell]) == set()          # frame 2 -- streak 2
    assert tracker.update([cell]) == {cell}          # frame 3 -- promoted


def test_a_missed_frame_resets_the_streak():
    tracker = PersistenceTracker(required_frames=3)
    cell = ("L0", 10, 20)

    tracker.update([cell])       # streak 1
    tracker.update([cell])       # streak 2
    tracker.update([])           # missed -- streak reset to 0
    tracker.update([cell])       # streak 1 again, NOT 3
    assert tracker.update([cell]) == set()  # streak 2, still not promoted
    assert tracker.update([cell]) == {cell}  # streak 3 -- now promoted


def test_multiple_cells_tracked_independently():
    tracker = PersistenceTracker(required_frames=2)
    a, b = ("L0", 1, 1), ("L0", 2, 2)

    assert tracker.update([a]) == set()
    assert tracker.update([a, b]) == {a}  # a promoted (2nd consecutive), b just started
    assert tracker.update([b]) == {b}     # b promoted; a not flagged this frame -> its streak drops
