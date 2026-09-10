"""
tests/test_ground_prior.py

Ticket #26 tests: flat ground, an 8-degree slope (the test that
distinguishes the incremental walk from a plane fit), a 30 cm step, and
the label-don't-strip regression (terrain reaches the elevation map).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from grid.addressing import flat_index, global_to_storage
from grid.clipmap import Clipmap
from grid.scatter import scatter
from perception.ground_prior import compute_ground_prior
from perception.rellis_loader import load_rellis_sweep
from perception.sweep import Sweep
from sensor.sensor_model import load_sensor_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
RELLIS_SEQ_DIR = Path(__file__).resolve().parents[1] / "data" / "rellis" / "00004"


def _column_sweep(zs, az=0.0) -> Sweep:
    """Points along one azimuth column at increasing horizontal range,
    given heights `zs`."""
    n = len(zs)
    r_xy = np.arange(1, n + 1, dtype=np.float64)
    x = r_xy * math.cos(az)
    y = r_xy * math.sin(az)
    xyz = np.stack([x, y, np.array(zs, dtype=np.float64)], axis=1).astype(np.float32)
    return Sweep(
        xyz=xyz,
        intensity=np.full(n, 0.5, dtype=np.float32),
        ring=np.full(n, -1, dtype=np.int16),
        timestamp=0.0,
        T_world=np.eye(4),
        sensor_id="test",
    )


def test_flat_ground_is_all_ground():
    sweep = _column_sweep([0.0] * 10)
    result = compute_ground_prior(sweep)
    assert result.is_ground.all()


def test_constant_8deg_slope_is_all_ground():
    """The test that distinguishes the incremental walk from a plane
    fit: a plane fit over a wide sector would misclassify a consistent
    slope; the walk, comparing only consecutive points, tracks it."""
    r_xy = np.arange(1, 21, dtype=np.float64)
    zs = (r_xy * math.tan(math.radians(8.0))).tolist()
    sweep = _column_sweep(zs)
    result = compute_ground_prior(sweep)
    assert result.is_ground.all()


def test_30cm_step_points_immediately_above_are_not_ground():
    # r_xy = 1..5 at z=0 (ground), then a 30 cm step at r_xy = 5.5, 6.0 --
    # close enough to the last ground point that the slope test reliably
    # catches it (a pure two-point slope check dilutes over long enough
    # horizontal distance, an inherent limit of the algorithm as
    # specified, not tested here).
    n = 7
    r_xy = np.array([1, 2, 3, 4, 5, 5.5, 6.0], dtype=np.float64)
    zs = np.array([0, 0, 0, 0, 0, 0.30, 0.30], dtype=np.float64)
    x = r_xy
    y = np.zeros(n)
    xyz = np.stack([x, y, zs], axis=1).astype(np.float32)
    sweep = Sweep(
        xyz=xyz, intensity=np.full(n, 0.5, dtype=np.float32),
        ring=np.full(n, -1, dtype=np.int16), timestamp=0.0,
        T_world=np.eye(4), sensor_id="test",
    )
    result = compute_ground_prior(sweep)
    assert result.is_ground[:5].all()  # the ground before the step
    assert not result.is_ground[5]  # the point right at the top of the step
    assert not result.is_ground[6]


def test_never_strips_points_output_length_matches_input():
    """Bible Part 5.2's core invariant: LABEL, never STRIP -- the point
    count is unchanged, nothing is removed from the sweep."""
    sweep = _column_sweep([0.0, 0.0, 5.0, 5.0, 0.0])  # includes clearly non-ground points
    result = compute_ground_prior(sweep)
    assert result.is_ground.shape[0] == sweep.xyz.shape[0]
    assert not result.is_ground.all()  # some points genuinely aren't ground
    assert result.is_ground.any()  # but some are


@pytest.mark.skipif(not RELLIS_SEQ_DIR.exists(), reason="real RELLIS-3D sequence 00004 not present locally")
def test_regression_terrain_reaches_the_elevation_map():
    """The automated form of the label-don't-strip bug: run ground
    labelling on a real sweep, scatter the GROUND-LABELLED points (not a
    filtered subset that dropped anything) into the clipmap, and confirm
    the map genuinely contains terrain cells with valid elevation."""
    sweep = load_rellis_sweep(RELLIS_SEQ_DIR, frame_idx=1000)
    result = compute_ground_prior(sweep)
    assert result.is_ground.any(), "no ground points found at all -- can't test the regression"

    ground_xyz = sweep.xyz[result.is_ground].astype(np.float64)

    sm = load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")
    cm = Clipmap(sm, n_levels=4, N=512, c0=0.05)
    cm.scroll_to(0.0, 0.0)
    scatter(cm, ground_xyz)

    level = 0
    touched_flat = np.nonzero(cm.flags[level] != 0)[0]
    assert touched_flat.size > 0, "ground points were computed but never reached the elevation map"
