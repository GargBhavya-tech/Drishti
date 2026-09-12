"""
tests/test_multi_sweep.py

perception/multi_sweep.py: the single-sweep passthrough, correct
point-count concatenation, and -- the one genuinely easy way to get
this wrong -- that a STATIC point (same world position every sweep)
motion-compensates back to the SAME sensor-frame location it already
has in the current sweep, verified by hand-picked poses.
"""

from __future__ import annotations

import numpy as np

from perception.multi_sweep import merge_sweeps_motion_compensated, transform_points
from perception.sweep import Sweep


def _sweep(xyz, T_world, timestamp=0.0, sensor_id="test") -> Sweep:
    n = xyz.shape[0]
    return Sweep(
        xyz=xyz.astype(np.float32),
        intensity=np.zeros(n, dtype=np.float32),
        ring=np.full(n, -1, dtype=np.int16),
        timestamp=timestamp,
        T_world=T_world,
        sensor_id=sensor_id,
    )


def test_single_sweep_passthrough_unchanged():
    xyz = np.array([[1.0, 2.0, 3.0]])
    sweep = _sweep(xyz, np.eye(4))
    result = merge_sweeps_motion_compensated([sweep])
    assert result is sweep


def test_merged_point_count_is_the_sum():
    a = _sweep(np.random.default_rng(0).normal(size=(50, 3)), np.eye(4), timestamp=0.0)
    b = _sweep(np.random.default_rng(1).normal(size=(30, 3)), np.eye(4), timestamp=1.0)
    merged = merge_sweeps_motion_compensated([a, b])
    assert merged.xyz.shape[0] == 80
    assert merged.intensity.shape[0] == 80
    assert merged.ring.shape[0] == 80


def test_static_world_point_lands_back_at_its_own_sensor_frame_position():
    """A point that is STATIC in the world, seen by a sensor that has
    moved 5m along x between two sweeps: transforming it from the OLD
    sweep's sensor frame into the NEW sweep's sensor frame must recover
    exactly where it would appear in the new frame's own raw data --
    checked here by placing the SAME world point directly into both
    sweeps' own raw coordinates and confirming the merge reproduces the
    second sweep's own value for it, not some incorrectly-composed
    transform's answer."""
    # Ego at world x=0 for sweep 1, world x=5 for sweep 2 (both facing
    # the same way -- pure translation, easiest case to verify by hand).
    T1 = np.eye(4)
    T1[0, 3] = 0.0
    T2 = np.eye(4)
    T2[0, 3] = 5.0

    world_point = np.array([10.0, 0.0, 0.0])
    point_in_sensor1 = world_point - T1[:3, 3]  # (10, 0, 0)
    point_in_sensor2 = world_point - T2[:3, 3]  # (5, 0, 0)

    sweep1 = _sweep(point_in_sensor1[None, :], T1, timestamp=0.0)
    sweep2 = _sweep(np.zeros((0, 3)), T2, timestamp=1.0)  # no points of its own, just receives the merge

    merged = merge_sweeps_motion_compensated([sweep1, sweep2])
    np.testing.assert_allclose(merged.xyz[0], point_in_sensor2, atol=1e-5)


def test_transform_points_applies_pure_translation():
    xyz = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    T = np.eye(4)
    T[:3, 3] = [10.0, 20.0, 30.0]
    result = transform_points(xyz, T)
    np.testing.assert_allclose(result, [[11.0, 20.0, 30.0], [10.0, 21.0, 30.0]])
