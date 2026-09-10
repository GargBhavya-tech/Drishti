"""
tests/test_ground_plane.py

Ticket #35's own test: "on an 8 degree slope the fitted plane tracks the
slope; residuals stay near zero. On flat ground the fit is level.
Done when: residuals are near zero on both flat and sloped terrain."
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from observability.ground_plane import expected_ground_range, fit_local_ground_planes
from perception.sweep import Sweep
from sensor.sensor_model import SensorConfig

RESIDUAL_TOL_M = 1e-6


def _sensor() -> SensorConfig:
    return SensorConfig(
        sensor_id="test",
        n_beams=16,
        d_theta_rad=math.radians(0.5),
        d_phi_rad=math.radians(2.0),
        phi_max_rad=math.radians(15.0),
        h_m=1.73,
        usable_range_m=100.0,
    )


def _single_column_sweep(r_values: np.ndarray, z_values: np.ndarray, azimuth_rad: float = 0.0) -> Sweep:
    """All points along ONE azimuth direction (a single column once
    binned), at increasing horizontal range, with the given heights --
    exactly what compute_ground_prior's incremental walk needs to accept
    every point as ground (slope between consecutive points stays under
    its default threshold as long as z_values follows a gentle line)."""
    n = r_values.shape[0]
    x = r_values * math.cos(azimuth_rad)
    y = r_values * math.sin(azimuth_rad)
    xyz = np.stack([x, y, z_values], axis=1).astype(np.float32)
    return Sweep(
        xyz=xyz,
        intensity=np.ones(n, dtype=np.float32),
        ring=np.full(n, -1, dtype=np.int16),
        timestamp=0.0,
        T_world=np.eye(4, dtype=np.float64),
        sensor_id="test",
    )


def test_flat_ground_fit_is_level_with_near_zero_residual():
    r = np.linspace(2.0, 10.0, 8)
    z = np.zeros_like(r)
    sweep = _single_column_sweep(r, z)

    fit = fit_local_ground_planes(sweep, _sensor(), n_azimuth_bins=360)
    col = next(iter(fit.column_fits.values()))

    assert abs(col.slope) < 1e-6
    assert abs(col.intercept) < 1e-6
    assert col.max_residual_m < RESIDUAL_TOL_M


def test_8_degree_slope_fit_tracks_the_slope_with_near_zero_residual():
    slope_rad = math.radians(8.0)
    r = np.linspace(2.0, 10.0, 8)
    z = r * math.tan(slope_rad)
    sweep = _single_column_sweep(r, z)

    fit = fit_local_ground_planes(sweep, _sensor(), n_azimuth_bins=360)
    col = next(iter(fit.column_fits.values()))

    assert col.slope == pytest.approx(math.tan(slope_rad), abs=1e-4)
    assert col.max_residual_m < RESIDUAL_TOL_M


def test_local_fit_tracks_the_recent_segment_not_the_whole_columns_average():
    # A column that starts FLAT close in, then breaks into an 8 degree
    # slope farther out -- the LOCAL fit (last few points) must track
    # the recent slope, not average flat-then-sloped into something in
    # between (Ticket #35's own "Watch out": this is what stops #36
    # firing on every hill).
    flat_r = np.linspace(1.0, 4.0, 5)
    flat_z = np.zeros_like(flat_r)
    slope_rad = math.radians(8.0)
    slope_r = np.linspace(4.5, 8.0, 5)
    # Continuous at the join (z=0 at r=4.0 -> 4.5) so compute_ground_prior's
    # incremental walk accepts every point as ground.
    slope_z = (slope_r - 4.0) * math.tan(slope_rad)

    r = np.concatenate([flat_r, slope_r])
    z = np.concatenate([flat_z, slope_z])
    sweep = _single_column_sweep(r, z)

    fit = fit_local_ground_planes(sweep, _sensor(), n_azimuth_bins=360, k_recent=4)
    col = next(iter(fit.column_fits.values()))

    # Tracks the recent (sloped) segment, not something near a
    # flat-vs-slope average.
    assert col.slope == pytest.approx(math.tan(slope_rad), abs=1e-3)


def test_expected_ground_range_returns_none_for_a_column_with_no_fit():
    sweep = _single_column_sweep(np.array([5.0]), np.array([0.0]))  # only 1 point -> no fit
    fit = fit_local_ground_planes(sweep, _sensor(), n_azimuth_bins=360)
    assert expected_ground_range(fit, ring=0, azimuth_col=0) is None


def test_expected_ground_range_recovers_the_known_flat_ground_range():
    # Flat ground at z = -h_m (sensor mounted h_m above the ground
    # plane): every beam with a non-zero downward elevation should hit
    # it at range r_full = h_m / sin(|phi|) for a beam looking down at
    # angle phi (phi negative below horizontal).
    sm = _sensor()
    h_m = 1.73
    r = np.linspace(2.0, 10.0, 8)
    z = np.full_like(r, -h_m)
    sweep = _single_column_sweep(r, z)

    fit = fit_local_ground_planes(sweep, sm, n_azimuth_bins=360)
    col_idx = next(iter(fit.column_fits.keys()))

    # Pick the bottom (most downward-looking) ring -- guaranteed to have
    # negative elevation for this sensor's phi_max/d_phi.
    bottom_ring = sm.n_beams - 1
    predicted = expected_ground_range(fit, ring=bottom_ring, azimuth_col=col_idx)
    assert predicted is not None

    # Independently recompute the expected physical range for that same
    # ring's elevation angle and compare.
    phi_min = sm.phi_max_rad - (sm.n_beams - 1) * sm.d_phi_rad
    fov = sm.phi_max_rad - phi_min
    phi = phi_min + fov * (1.0 - (bottom_ring + 0.5) / sm.n_beams)
    expected = h_m / abs(math.sin(phi))

    assert predicted == pytest.approx(expected, rel=1e-6)
