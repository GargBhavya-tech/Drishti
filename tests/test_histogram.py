"""
tests/test_histogram.py

Ticket #20 tests: bin-width derivation from vehicle config, and the
two-surface (bridge-shaped) case from the Build Map's own spec.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grid.addressing import flat_index, global_to_storage, global_to_world
from grid.clipmap import Clipmap
from grid.histogram import histogram_spec, scatter_histogram
from sensor.sensor_model import load_sensor_config
from sensor.vehicle_config import load_vehicle_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def hdl64e():
    return load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")


@pytest.fixture
def vehicle():
    return load_vehicle_config(CONFIGS / "vehicle_ugv.yaml")


def _fresh(hdl64e) -> Clipmap:
    cm = Clipmap(hdl64e, n_levels=4, N=512, c0=0.05)
    cm.scroll_to(0.0, 0.0)
    return cm


def test_bin_width_is_derived_not_a_literal(vehicle):
    spec = histogram_spec(vehicle)
    # vehicle_ugv.yaml's min_clearance_m = 2.50 -> upper = 4.0, lower = -0.5,
    # bin_width = 4.5 / 8 = 0.5625 -- the Bible's own worked number, but
    # arrived at via the formula, not typed in directly.
    assert vehicle.min_clearance_m == pytest.approx(2.50)
    assert spec.upper_offset_m == pytest.approx(4.0)
    assert spec.lower_offset_m == pytest.approx(-0.5)
    assert spec.bin_width_m == pytest.approx(0.5625)
    assert spec.n_bins == 8


def test_different_min_clearance_changes_bin_width():
    """If bin width were a hardcoded literal, this would fail."""
    from sensor.vehicle_config import VehicleConfig

    stricter = VehicleConfig(
        max_slope_deg=25, max_step_height_m=0.20, min_clearance_m=4.0,
        ground_clearance_m=0.35, width_m=2.10, max_roughness_m=0.08,
        min_object_t_m=1.0, min_object_w_m=0.10, braking_a_ms2=4.0, t_react_s=0.30,
    )
    spec = histogram_spec(stricter)
    assert spec.bin_width_m == pytest.approx((4.0 + 1.5 - (-0.5)) / 8)
    assert spec.bin_width_m != pytest.approx(0.5625)


def test_bridge_case_bins_zero_and_seven_occupied_middle_empty(hdl64e, vehicle):
    """Ticket #20's own test: z=0.0 and z=4.2 (relative to ground=0) ->
    occupied bins at index 0 and index 7, six empty bins between."""
    cm = _fresh(hdl64e)
    level = 0
    c_l = cm.levels[level].cell_size_m
    gi, gj = cm.origin_i[level] + 8, cm.origin_j[level] + 8
    x0, y0 = global_to_world(gi, gj, c_l)

    xyz = np.array(
        [
            [x0 + c_l * 0.3, y0 + c_l * 0.5, 0.0],
            [x0 + c_l * 0.7, y0 + c_l * 0.5, 4.2],
        ],
        dtype=np.float64,
    )
    z_ground = np.array([0.0, 0.0])

    scatter_histogram(cm, xyz, z_ground, vehicle)

    si, sj = global_to_storage(gi, gj, cm.N)
    flat = flat_index(si, sj, cm.N)
    bins = cm.histogram[level, flat, :]

    assert bins[0] == 1
    assert bins[7] == 1
    assert list(bins[1:7]) == [0, 0, 0, 0, 0, 0]


def test_histogram_relative_to_local_ground_not_absolute_z(hdl64e, vehicle):
    """The same two points, but on a slope where local ground sits at
    z=10.0 rather than 0.0 -- must land in the SAME bins as the flat-
    ground case (0 and 7), not collapse into one bin the way an
    absolute-z histogram would."""
    cm = _fresh(hdl64e)
    level = 0
    c_l = cm.levels[level].cell_size_m
    gi, gj = cm.origin_i[level] + 9, cm.origin_j[level] + 9
    x0, y0 = global_to_world(gi, gj, c_l)

    xyz = np.array(
        [
            [x0 + c_l * 0.3, y0 + c_l * 0.5, 10.0],
            [x0 + c_l * 0.7, y0 + c_l * 0.5, 14.2],
        ],
        dtype=np.float64,
    )
    z_ground = np.array([10.0, 10.0])

    scatter_histogram(cm, xyz, z_ground, vehicle)

    si, sj = global_to_storage(gi, gj, cm.N)
    flat = flat_index(si, sj, cm.N)
    bins = cm.histogram[level, flat, :]

    assert bins[0] == 1
    assert bins[7] == 1


def test_bin_counts_saturate_never_wrap(hdl64e, vehicle):
    cm = _fresh(hdl64e)
    level = 0
    c_l = cm.levels[level].cell_size_m
    gi, gj = cm.origin_i[level] + 2, cm.origin_j[level] + 2
    x0, y0 = global_to_world(gi, gj, c_l)

    n_points = 300  # > uint8 max (255), all in bin 0
    xyz = np.stack(
        [
            np.full(n_points, x0 + c_l * 0.5),
            np.full(n_points, y0 + c_l * 0.5),
            np.zeros(n_points),
        ],
        axis=1,
    )
    z_ground = np.zeros(n_points)

    scatter_histogram(cm, xyz, z_ground, vehicle)

    si, sj = global_to_storage(gi, gj, cm.N)
    flat = flat_index(si, sj, cm.N)
    assert cm.histogram[level, flat, 0] == 255
