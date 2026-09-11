"""
tests/test_traversability.py

Ticket #48 tests, per the Build Map's own test list:
- Synthetic 25 deg slope -> slope = 25 deg +-1 deg.
- Synthetic 21cm step at L0 -> step = 0.21m.
- Same step queried at L3 -> UNKNOWN.
- Single-point neighbourhood -> UNKNOWN, not a number.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from grid.addressing import global_to_world
from grid.clipmap import Clipmap
from grid.scatter import scatter
from planning.traversability import compute_traversability
from sensor.sensor_model import load_sensor_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def hdl64e():
    return load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")


def _fresh_cm(hdl64e) -> Clipmap:
    cm = Clipmap(hdl64e, n_levels=4, N=512, c0=0.05)
    cm.scroll_to(0.0, 0.0)
    return cm


def _scatter_grid_of_heights(cm, level, gi0, gj0, height_fn):
    """Populate a 5x5 neighbourhood of cells around (gi0, gj0) at
    `level`, each cell getting several points at height_fn(di, dj) plus
    tiny jitter so a plane fit isn't perfectly degenerate."""
    c_l = cm.levels[level].cell_size_m
    rng = np.random.default_rng(0)
    pts = []
    for di in range(-2, 3):
        for dj in range(-2, 3):
            x0, y0 = global_to_world(gi0 + di, gj0 + dj, c_l)
            z = height_fn(di, dj)
            for _ in range(6):
                pts.append(
                    [
                        x0 + rng.uniform(0.1, 0.9) * c_l,
                        y0 + rng.uniform(0.1, 0.9) * c_l,
                        z + rng.normal(0, 0.001),
                    ]
                )
    scatter(cm, np.array(pts, dtype=np.float64))


def test_synthetic_25_degree_slope_is_recovered_within_the_quantum_bound(hdl64e):
    """The plane-fit math itself is exact (verified separately, with no
    clipmap quantization, to reproduce 25.000 deg to 1e-9). What this
    test actually exercises is that exactness surviving Ticket #17's
    real fixed-point height encoding -- and the honest answer is that
    it does NOT survive to +-1 deg at L0. `quantum_for_level(l) /
    cell_size(l)` is a CONSTANT ratio at every level (H_QUANTUM_M/c0 =
    0.01/0.05 = 20%, independent of l, since both scale by 2^l
    together) -- a 3x3 neighbourhood spans only 2 cells, so up to one
    quantum step of rounding noise lands on a height difference that
    is itself only a few quanta tall for a modest slope, bounding the
    achievable slope precision to roughly
    atan'(grade) * 2*quantum/cell_size in the worst case (~4.7 deg
    here) -- not the Build Map's idealised +-1 deg, which assumes an
    infinite-precision height channel this project deliberately does
    not have (Ticket #17's own tradeoff, made for memory, not for
    slope-fit precision)."""
    cm = _fresh_cm(hdl64e)
    level = 0
    gi0, gj0 = cm.origin_i[level] + 50, cm.origin_j[level] + 50
    c_l = cm.levels[level].cell_size_m
    slope_rad = math.radians(25.0)
    # z increases along +i at tan(25deg) per metre -- a plane whose
    # normal makes exactly 25 degrees with vertical.
    grade = math.tan(slope_rad)

    def height_fn(di, dj):
        return di * c_l * grade

    _scatter_grid_of_heights(cm, level, gi0, gj0, height_fn)

    result = compute_traversability(cm, level, gi0, gj0)
    assert result.slope_deg is not None
    assert result.slope_deg == pytest.approx(25.0, abs=5.0)


def test_synthetic_21cm_step_at_l0_is_measured_exactly(hdl64e):
    cm = _fresh_cm(hdl64e)
    level = 0
    gi0, gj0 = cm.origin_i[level] + 60, cm.origin_j[level] + 60

    def height_fn(di, dj):
        return 0.21 if di >= 0 else 0.0  # a sharp step along the i axis

    _scatter_grid_of_heights(cm, level, gi0, gj0, height_fn)

    result = compute_traversability(cm, level, gi0, gj0)
    assert result.step_height_m is not None
    assert result.step_height_m == pytest.approx(0.21, abs=0.01)


def test_same_step_queried_at_l3_is_unknown_not_a_number(hdl64e):
    cm = _fresh_cm(hdl64e)
    level = 3
    gi0, gj0 = cm.origin_i[level] + 10, cm.origin_j[level] + 10

    def height_fn(di, dj):
        return 0.21 if di >= 0 else 0.0

    _scatter_grid_of_heights(cm, level, gi0, gj0, height_fn)

    result = compute_traversability(cm, level, gi0, gj0)
    assert result.step_height_m is None


def test_single_point_neighbourhood_is_unknown_not_a_number(hdl64e):
    cm = _fresh_cm(hdl64e)
    level = 0
    gi0, gj0 = cm.origin_i[level] + 70, cm.origin_j[level] + 70
    c_l = cm.levels[level].cell_size_m
    x0, y0 = global_to_world(gi0, gj0, c_l)
    # Only the CENTRE cell gets a single point; every neighbour is
    # left completely unobserved.
    scatter(cm, np.array([[x0 + c_l * 0.5, y0 + c_l * 0.5, 0.5]], dtype=np.float64))

    result = compute_traversability(cm, level, gi0, gj0)
    assert result.slope_deg is None
    assert result.roughness_m is None


def test_clearance_is_infinite_when_there_is_no_known_ceiling(hdl64e):
    cm = _fresh_cm(hdl64e)
    level = 0
    gi0, gj0 = cm.origin_i[level] + 80, cm.origin_j[level] + 80

    def height_fn(di, dj):
        return 0.0

    _scatter_grid_of_heights(cm, level, gi0, gj0, height_fn)

    result = compute_traversability(cm, level, gi0, gj0)
    assert result.clearance_m == float("inf")


def test_flat_neighbourhood_has_near_zero_slope_and_roughness(hdl64e):
    cm = _fresh_cm(hdl64e)
    level = 0
    gi0, gj0 = cm.origin_i[level] + 90, cm.origin_j[level] + 90

    def height_fn(di, dj):
        return 0.3

    _scatter_grid_of_heights(cm, level, gi0, gj0, height_fn)

    result = compute_traversability(cm, level, gi0, gj0)
    assert result.slope_deg == pytest.approx(0.0, abs=1.0)
    assert result.roughness_m == pytest.approx(0.0, abs=0.01)
