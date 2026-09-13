"""
tests/test_kalman_tracker.py

Validates temporal/kalman_tracker.py against SYNTHETIC multi-frame
trajectories with known ground truth -- see that module's own docstring
for why synthetic (no real continuous multi-frame detection pipeline
exists yet to validate against) and what this does and does not prove.
"""

from __future__ import annotations

import numpy as np

from perception.taxonomy import DrishtiClass
from temporal.kalman_tracker import MultiObjectTracker


def test_single_object_constant_velocity_is_tracked_with_one_consistent_id():
    """A single object moving at constant velocity, with small
    measurement noise, should be tracked under ONE consistent track_id
    for its whole trajectory -- the most basic thing a tracker must do."""
    rng = np.random.default_rng(42)
    tracker = MultiObjectTracker()
    true_velocity = np.array([2.0, 0.5])  # m/s
    true_start = np.array([0.0, 0.0])
    dt = 0.1

    seen_ids = set()
    for step in range(30):
        true_pos = true_start + true_velocity * (step * dt)
        noisy_pos = true_pos + rng.normal(scale=0.1, size=2)
        detections_xy = noisy_pos[None, :]
        detection_classes = np.array([int(DrishtiClass.VEHICLE)])
        tracks = tracker.update(detections_xy, detection_classes, dt)
        if step >= 3:  # after MIN_HITS_TO_CONFIRM
            confirmed = tracker.confirmed_tracks()
            assert len(confirmed) == 1, f"expected exactly 1 confirmed track at step {step}, got {len(confirmed)}"
            seen_ids.add(confirmed[0].track_id)

    assert len(seen_ids) == 1, f"expected ONE consistent track_id across the whole trajectory, saw {seen_ids}"


def test_velocity_is_correctly_estimated():
    """After enough frames, the filter's own estimated velocity should
    converge close to the true constant velocity -- checks the Kalman
    update math itself, not just ID consistency."""
    rng = np.random.default_rng(7)
    tracker = MultiObjectTracker()
    true_velocity = np.array([3.0, -1.0])
    dt = 0.1

    for step in range(50):
        true_pos = true_velocity * (step * dt)
        noisy_pos = true_pos + rng.normal(scale=0.1, size=2)
        tracker.update(noisy_pos[None, :], np.array([int(DrishtiClass.VEHICLE)]), dt)

    confirmed = tracker.confirmed_tracks()
    assert len(confirmed) == 1
    estimated_velocity = confirmed[0].velocity
    assert np.allclose(estimated_velocity, true_velocity, atol=0.3), (
        f"estimated velocity {estimated_velocity} too far from true {true_velocity}"
    )


def test_two_well_separated_objects_get_two_distinct_persistent_ids():
    """Two objects moving in different directions, always well-separated
    (never within MAX_ASSOCIATION_DISTANCE_M of each other), must get
    and KEEP two distinct track_ids -- checks the Hungarian association
    doesn't merge or swap identities between well-separated tracks."""
    rng = np.random.default_rng(99)
    tracker = MultiObjectTracker()
    dt = 0.1
    vel_a = np.array([1.0, 0.0])
    vel_b = np.array([-1.0, 0.0])
    start_a = np.array([0.0, 10.0])
    start_b = np.array([0.0, -10.0])

    id_a_history = []
    id_b_history = []
    for step in range(20):
        pos_a = start_a + vel_a * (step * dt) + rng.normal(scale=0.05, size=2)
        pos_b = start_b + vel_b * (step * dt) + rng.normal(scale=0.05, size=2)
        detections_xy = np.stack([pos_a, pos_b])
        detection_classes = np.array([int(DrishtiClass.PEDESTRIAN), int(DrishtiClass.PEDESTRIAN)])
        tracker.update(detections_xy, detection_classes, dt)

        if step >= 3:
            confirmed = sorted(tracker.confirmed_tracks(), key=lambda t: t.position[1])  # sort by y: b (y~-10) first, a (y~10) second
            assert len(confirmed) == 2, f"expected 2 confirmed tracks at step {step}, got {len(confirmed)}"
            id_b_history.append(confirmed[0].track_id)
            id_a_history.append(confirmed[1].track_id)

    assert len(set(id_a_history)) == 1, f"object A's track_id should stay constant, saw {set(id_a_history)}"
    assert len(set(id_b_history)) == 1, f"object B's track_id should stay constant, saw {set(id_b_history)}"
    assert id_a_history[0] != id_b_history[0], "the two objects must get DIFFERENT track_ids"


def test_track_deleted_after_max_coast_frames_of_no_detections():
    """A track with no matching detection for MAX_COAST_FRAMES+1
    consecutive frames must be deleted, not tracked forever on pure
    prediction."""
    tracker = MultiObjectTracker(max_coast_frames=3)
    dt = 0.1
    for step in range(5):  # confirm a track first
        tracker.update(np.array([[0.0, 0.0]]), np.array([int(DrishtiClass.VEHICLE)]), dt)
    assert len(tracker.confirmed_tracks()) == 1

    empty = np.zeros((0, 2))
    empty_classes = np.zeros((0,), dtype=np.int64)
    for _ in range(5):  # far more than max_coast_frames=3
        tracker.update(empty, empty_classes, dt)

    assert len(tracker.tracks) == 0, "track should have been deleted after exceeding max_coast_frames with no detections"


def test_new_track_is_tentative_not_confirmed_on_first_detection():
    """A single new detection must NOT immediately count as a confirmed
    track -- MIN_HITS_TO_CONFIRM exists specifically to avoid reporting
    one-frame noise as a real tracked object."""
    tracker = MultiObjectTracker()
    tracker.update(np.array([[5.0, 5.0]]), np.array([int(DrishtiClass.PEDESTRIAN)]), 0.1)
    assert len(tracker.confirmed_tracks()) == 0
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].confirmed is False
