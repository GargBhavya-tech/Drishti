"""
tests/test_motion.py

Ticket #45 tests, per the Build Map's own test list:
- Two identical sweeps with a synthetic ego translation -> ZERO motion
  detected anywhere (world-anchored: identical per-world-cell evidence
  produces no transitions regardless of where the ego itself is).
- One synthetic object translated 2m between frames -> motion detected
  on that object's cells only.
- Parked-car test: a static vehicle across 30 frames is never flagged.
"""

from __future__ import annotations

from grid.cell import OBS_FREE, OBS_OCCUPIED
from temporal.motion import MotionDetector


def test_identical_evidence_across_frames_detects_zero_motion():
    detector = MotionDetector()
    frame = {(10, 10): OBS_OCCUPIED, (10, 11): OBS_FREE, (20, 20): OBS_FREE}

    detector.detect(frame)  # first frame: nothing to compare against yet
    moving = detector.detect(dict(frame))  # identical second frame

    assert moving == set()


def test_an_object_translated_between_frames_is_detected_on_both_sides():
    detector = MotionDetector()
    # Frame 0: object occupies (50, 50); everywhere around it -- INCLUDING
    # the cell it is about to move into -- is observed FREE (a real
    # observability-carving frame would have swept FREE evidence across
    # any cell near enough to a return to matter; a cell with NO prior
    # evidence at all is not the same as a cell known to be FREE, so
    # the neighbour it moves into must be explicitly FREE here for the
    # FREE->OCCUPIED transition to be a genuine one).
    frame0 = {
        (50, 50): OBS_OCCUPIED,
        (51, 50): OBS_FREE,
        (52, 50): OBS_FREE,
        (100, 100): OBS_FREE,  # unrelated, far-away cell
    }
    # Frame 1: object moved to the adjacent cell (51, 50) -- vacating
    # (50, 50) and arriving at (51, 50).
    frame1 = {
        (50, 50): OBS_FREE,
        (51, 50): OBS_OCCUPIED,
        (100, 100): OBS_FREE,  # still free -- must never be flagged
    }

    detector.detect(frame0)
    moving = detector.detect(frame1)

    assert moving == {(50, 50), (51, 50)}
    assert (100, 100) not in moving


def test_unrelated_vacate_and_arrive_pair_too_far_apart_is_not_flagged():
    detector = MotionDetector()
    frame0 = {(0, 0): OBS_OCCUPIED, (10, 10): OBS_FREE}
    frame1 = {(0, 0): OBS_FREE, (10, 10): OBS_OCCUPIED}  # not 8-connected to (0,0)

    detector.detect(frame0)
    moving = detector.detect(frame1)

    assert moving == set()


def test_parked_car_across_30_frames_is_never_flagged_and_stays_occupied():
    detector = MotionDetector()
    car_cells = {(30, 30), (30, 31), (31, 30), (31, 31)}
    frame = {cell: OBS_OCCUPIED for cell in car_cells}
    frame[(60, 60)] = OBS_FREE  # unrelated background cell, also static

    all_flagged = set()
    for _ in range(30):
        all_flagged |= detector.detect(dict(frame))

    assert all_flagged == set()


def test_reset_clears_all_prior_evidence():
    detector = MotionDetector()
    detector.detect({(1, 1): OBS_OCCUPIED})
    detector.reset()
    # With no prior evidence, even a real transition-shaped input can't
    # be compared against anything -- first post-reset call must be a
    # no-op, exactly like the very first call ever.
    moving = detector.detect({(1, 1): OBS_FREE, (2, 1): OBS_OCCUPIED})
    assert moving == set()
