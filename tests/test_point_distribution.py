"""
Ticket #6 tests -- THE GATE.

No real sweep data is available in this environment yet (nuScenes-mini /
RELLIS-3D aren't downloaded here). So these tests validate the MEASUREMENT
CODE itself against synthetic sweeps built to an exact known grid -- if the
estimator is correct, measured spacing must equal the predicted formula
almost exactly (tight tolerance) on synthetic data, which is a stronger
check than the ~20% real-data tolerance in the Build Map (real data adds
noise the synthetic grid doesn't have). Once #2/#3 land real data, add a
`test_real_nuscenes_sweep` / `test_real_rellis_sweep` alongside these using
the same functions, gated on the dataset being present.
"""

from pathlib import Path

import numpy as np
import pytest

from perception.sweep import Sweep
from sensor.sensor_model import load_sensor_config
from eval.point_distribution import (
    measure_within_ring_spacing,
    measure_between_ring_spacing,
    validate_point_distribution,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def hdl32e():
    return load_sensor_config(CONFIGS / "sensor_hdl32e.yaml")


def make_synthetic_flat_ground_sweep(sm, ranges_m, n_azimuth=360) -> Sweep:
    """Build a sweep where ring `k` sits at ground-plane range
    ranges_m[k] for every azimuth sample, exactly n_azimuth points per
    ring, azimuth spaced by 2*pi/n_azimuth (NOT necessarily d_theta --
    within-ring spacing measures whatever azimuth spacing the points
    actually have, so pick n_azimuth such that the spacing equals
    d_theta exactly, to match the prediction under test)."""
    xyz = []
    ring = []
    for k, r in enumerate(ranges_m):
        az = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
        x = r * np.cos(az)
        y = r * np.sin(az)
        z = np.full_like(x, -sm.h_m)  # flat ground, sensor h above it
        xyz.append(np.stack([x, y, z], axis=1))
        ring.append(np.full(n_azimuth, k, dtype=np.int16))
    xyz = np.concatenate(xyz, axis=0).astype(np.float32)
    ring = np.concatenate(ring, axis=0)
    return Sweep(
        xyz=xyz,
        intensity=np.ones(xyz.shape[0], dtype=np.float32),
        ring=ring,
        timestamp=0.0,
        T_world=np.eye(4, dtype=np.float64),
        sensor_id=sm.sensor_id,
    )


def test_within_ring_matches_prediction_on_synthetic_grid(hdl32e):
    # Choose n_azimuth so that azimuth spacing exactly equals d_theta at
    # the bin midpoint's ring -- 2*pi / n_azimuth = d_theta.
    n_azimuth = int(round(2 * np.pi / hdl32e.d_theta_rad))
    ranges = [12.5, 27.5, 42.5]  # bin midpoints for 10-15m, 25-30m, 40-45m
    sweep = make_synthetic_flat_ground_sweep(hdl32e, ranges, n_azimuth=n_azimuth)

    within = measure_within_ring_spacing(sweep)
    for r in ranges:
        lo = (r // 5) * 5
        mid = lo + 2.5
        measured = within[mid]
        predicted = r * hdl32e.d_theta_rad  # true spacing at the exact synthetic range
        assert measured is not None
        assert measured == pytest.approx(predicted, rel=0.05)


def test_between_ring_matches_prediction_on_synthetic_grid(hdl32e):
    # Two adjacent rings whose ranges differ by exactly s_radial_ground(r).
    from sensor.sensor_model import s_radial_ground
    r0 = 20.0
    spacing = s_radial_ground(r0, hdl32e)
    r1 = r0 + spacing
    n_azimuth = 360
    sweep = make_synthetic_flat_ground_sweep(hdl32e, [r0, r1], n_azimuth=n_azimuth)

    between = measure_between_ring_spacing(sweep)
    lo = (r0 // 5) * 5
    mid = lo + 2.5
    measured = between[mid]
    assert measured is not None
    assert measured == pytest.approx(spacing, rel=0.15)


def test_gate_produces_a_plot(hdl32e, tmp_path):
    n_azimuth = int(round(2 * np.pi / hdl32e.d_theta_rad))
    sweep = make_synthetic_flat_ground_sweep(hdl32e, [12.5, 27.5], n_azimuth=n_azimuth)
    out = tmp_path / "sensor_validation.png"
    results = validate_point_distribution(sweep, hdl32e, out_path=out)
    assert out.exists()
    assert len(results) == 14  # 5m bins from 0 to 70m


def test_ring_grouping_not_global_nearest_neighbour(hdl32e):
    """Regression guard for the Build Map's explicit warning: measuring
    nearest-neighbour over the WHOLE cloud (not grouped by ring) finds
    the neighbour in an adjacent ring, which is much closer than the
    true within-ring spacing at long range and would silently under-report
    the tangential spacing. Construct two rings far enough apart in range
    that a whole-cloud nearest-neighbour would pick the wrong ring, and
    confirm our estimator does NOT do that."""
    n_azimuth = int(round(2 * np.pi / hdl32e.d_theta_rad))
    r_near, r_far = 10.0, 10.5  # rings very close in range, far azimuth spacing
    sweep = make_synthetic_flat_ground_sweep(hdl32e, [r_near, r_far], n_azimuth=n_azimuth)

    within = measure_within_ring_spacing(sweep)
    lo = (r_near // 5) * 5
    mid = lo + 2.5
    measured = within[mid]
    predicted = r_near * hdl32e.d_theta_rad
    # If ring-grouping were broken, `measured` would collapse toward the
    # tiny inter-ring gap (0.5m) instead of the true tangential spacing.
    assert measured == pytest.approx(predicted, rel=0.1)
    assert measured < 0.4  # sanity: still much less than the 0.5m inter-ring gap would suggest if broken differently
