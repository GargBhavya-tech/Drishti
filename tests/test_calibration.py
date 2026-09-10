"""Ticket #7 tests, against synthetic data (no real nuScenes calibration
records loaded in this environment yet)."""

import numpy as np
import pytest

from perception.sweep import Sweep
from sensor.calibration import h_from_calibration, h_from_ground_plane_fit, cross_check_mount_height


def make_flat_ground_sweep(h_m: float, n_points: int = 2000, seed: int = 0) -> Sweep:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-20, 20, n_points)
    y = rng.uniform(-20, 20, n_points)
    z = np.full(n_points, -h_m) + rng.normal(0, 0.01, n_points)  # tiny sensor noise
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    return Sweep(
        xyz=xyz,
        intensity=np.ones(n_points, dtype=np.float32),
        ring=np.zeros(n_points, dtype=np.int16),
        timestamp=0.0,
        T_world=np.eye(4, dtype=np.float64),
        sensor_id="test",
    )


def test_h_from_calibration_is_a_sum():
    assert h_from_calibration(1.4, 0.44) == pytest.approx(1.84)


def test_h_from_ground_plane_fit_recovers_known_height():
    sweeps = [make_flat_ground_sweep(1.84, seed=i) for i in range(5)]
    h_est, residual = h_from_ground_plane_fit(sweeps)
    assert h_est == pytest.approx(1.84, abs=0.02)
    assert residual < 0.05


def test_cross_check_agrees_when_both_estimates_are_close():
    sweeps = [make_flat_ground_sweep(1.84, seed=i) for i in range(5)]
    result = cross_check_mount_height(
        sensor_to_ego_translation_z_m=1.40,
        ego_height_above_ground_m=0.44,
        sweeps=sweeps,
    )
    assert result.agrees


def test_cross_check_flags_disagreement():
    sweeps = [make_flat_ground_sweep(1.84, seed=i) for i in range(5)]
    result = cross_check_mount_height(
        sensor_to_ego_translation_z_m=1.40,
        ego_height_above_ground_m=0.10,  # deliberately wrong -> h_calib = 1.50, off by 0.34m
        sweeps=sweeps,
    )
    assert not result.agrees


def test_missing_ground_points_raises():
    empty_sweep = Sweep(
        xyz=np.zeros((0, 3), dtype=np.float32),
        intensity=np.zeros((0,), dtype=np.float32),
        ring=np.zeros((0,), dtype=np.int16),
        timestamp=0.0,
        T_world=np.eye(4, dtype=np.float64),
        sensor_id="test",
    )
    with pytest.raises(ValueError):
        h_from_ground_plane_fit([empty_sweep])
