"""
tests/test_layers.py

Ticket #21 tests -- the machine-checked form of Claim 1 (Bible Part 9.1):
a single height value cannot represent "drivable road under a bridge".
Four cases from the Build Map's own test table, plus the max-height-only
ablation that proves the point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grid.addressing import flat_index, global_to_storage, global_to_world
from grid.cell import NO_CEILING_SENTINEL, decode_clearance, decode_h
from grid.clipmap import Clipmap
from grid.layers import extract_layer_bins, scatter_layers
from grid.histogram import histogram_spec
from grid.scatter import scatter
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


def _run_cell(cm, vehicle, level, gi, gj, zs, **kwargs):
    c_l = cm.levels[level].cell_size_m
    x0, y0 = global_to_world(gi, gj, c_l)
    n = len(zs)
    offsets = np.linspace(0.1, 0.9, n)
    xyz = np.array([[x0 + c_l * f, y0 + c_l * 0.5, z] for f, z in zip(offsets, zs)], dtype=np.float64)
    z_ground = np.zeros(n)
    scatter(cm, xyz)
    scatter_layers(cm, xyz, z_ground, vehicle, **kwargs)
    return xyz


def _read_clearance(cm, level, gi, gj):
    si, sj = global_to_storage(gi, gj, cm.N)
    flat = flat_index(si, sj, cm.N)
    return decode_clearance(int(cm.h_ceil_min[level, flat]), int(cm.h_max[level, flat]))


def test_road_under_bridge(hdl64e, vehicle):
    cm = _fresh(hdl64e)
    level = 0
    gi, gj = cm.origin_i[level] + 10, cm.origin_j[level] + 10
    _run_cell(cm, vehicle, level, gi, gj, [0.00, 0.05, 4.2, 4.6])

    clearance = _read_clearance(cm, level, gi, gj)
    assert clearance == pytest.approx(4.15, abs=0.02)
    assert clearance >= vehicle.min_clearance_m  # DRIVABLE-eligible


def test_low_branch_overhang(hdl64e, vehicle):
    cm = _fresh(hdl64e)
    level = 0
    gi, gj = cm.origin_i[level] + 11, cm.origin_j[level] + 11
    _run_cell(cm, vehicle, level, gi, gj, [0.00, 0.02, 1.90, 2.30])

    clearance = _read_clearance(cm, level, gi, gj)
    assert clearance == pytest.approx(1.88, abs=0.02)
    assert clearance < vehicle.min_clearance_m  # OVERHANG: non-traversable at 2.5 m required


def test_flat_ground_clearance_is_infinite(hdl64e, vehicle):
    cm = _fresh(hdl64e)
    level = 0
    gi, gj = cm.origin_i[level] + 12, cm.origin_j[level] + 12
    _run_cell(cm, vehicle, level, gi, gj, [0.00, 0.01, 0.02, 0.03])

    clearance = _read_clearance(cm, level, gi, gj)
    assert clearance == float("inf")

    si, sj = global_to_storage(gi, gj, cm.N)
    flat = flat_index(si, sj, cm.N)
    assert int(cm.h_ceil_min[level, flat]) == int(NO_CEILING_SENTINEL)


def test_sparse_canopy_is_not_a_ceiling(hdl64e, vehicle):
    """Scattered, low-density returns at 2.5-3.5 m must NOT be reported
    as a ceiling -- Bible Part 9.3's vegetation-canopy edge case."""
    cm = _fresh(hdl64e)
    level = 0
    gi, gj = cm.origin_i[level] + 13, cm.origin_j[level] + 13
    # Dense ground (4 points) + exactly 1 scattered point per bin across
    # bins 5/6/7 (2.6, 3.1, 3.6 m) -- below the density-2 threshold in
    # every individual bin.
    _run_cell(
        cm, vehicle, level, gi, gj,
        [0.00, 0.01, 0.02, 0.03, 2.6, 3.1, 3.6],
        min_bin_count_for_ceiling=2,
    )

    clearance = _read_clearance(cm, level, gi, gj)
    assert clearance == float("inf")  # not a ceiling


def test_extract_layer_bins_pure_function_four_cases(vehicle):
    """The pure-function form (Bible Part 9's 'how to test': hand-
    construct histograms directly), independent of the scatter pipeline."""
    spec = histogram_spec(vehicle)

    # Bridge: bin 0 occupied (ground), bin 7 occupied (ceiling).
    bridge = np.zeros(8, dtype=np.int64)
    bridge[0] = 2
    bridge[7] = 2
    bins = extract_layer_bins(bridge, spec)
    assert bins.ground_hi == 0
    assert bins.ceiling_bin == 7

    # Branch: bin 0 (ground), bin 4 (branch, close overhead).
    branch = np.zeros(8, dtype=np.int64)
    branch[0] = 2
    branch[4] = 2
    bins = extract_layer_bins(branch, spec)
    assert bins.ground_hi == 0
    assert bins.ceiling_bin == 4

    # Flat ground only.
    flat = np.zeros(8, dtype=np.int64)
    flat[0] = 4
    bins = extract_layer_bins(flat, spec)
    assert bins.ceiling_bin is None

    # Sparse canopy: ground + single-count bins 5,6,7 below threshold 2.
    canopy = np.zeros(8, dtype=np.int64)
    canopy[0] = 4
    canopy[5] = 1
    canopy[6] = 1
    canopy[7] = 1
    bins = extract_layer_bins(canopy, spec, min_bin_count_for_ceiling=2)
    assert bins.ceiling_bin is None


def test_max_height_only_gets_the_bridge_case_wrong(hdl64e, vehicle):
    """The ablation: a max-height-only cell (Ticket #18's raw h_max
    before layer extraction narrows it) reports the BRIDGE DECK's height
    (4.6 m) as if it were an obstacle sitting on the ground -- exactly
    the 'impassable wall' failure Claim 1 exists to fix. Layer extraction
    corrects h_max back down to the ground layer's own top (0.05 m)."""
    cm = _fresh(hdl64e)
    level = 0
    gi, gj = cm.origin_i[level] + 14, cm.origin_j[level] + 14
    c_l = cm.levels[level].cell_size_m
    x0, y0 = global_to_world(gi, gj, c_l)
    xyz = np.array(
        [
            [x0 + c_l * 0.2, y0 + c_l * 0.5, 0.00],
            [x0 + c_l * 0.4, y0 + c_l * 0.5, 0.05],
            [x0 + c_l * 0.6, y0 + c_l * 0.5, 4.2],
            [x0 + c_l * 0.8, y0 + c_l * 0.5, 4.6],
        ],
        dtype=np.float64,
    )
    z_ground = np.zeros(4)

    scatter(cm, xyz)  # Ticket #18 alone: max-height-only view
    si, sj = global_to_storage(gi, gj, cm.N)
    flat = flat_index(si, sj, cm.N)
    max_height_only_view = decode_h(int(cm.h_max[level, flat]))
    assert max_height_only_view == pytest.approx(4.6, abs=0.01)  # WRONG: reads as a 4.6 m wall

    scatter_layers(cm, xyz, z_ground, vehicle)  # Ticket #21: multi-layer correction
    ground_top_after_layers = decode_h(int(cm.h_max[level, flat]))
    assert ground_top_after_layers == pytest.approx(0.05, abs=0.01)  # CORRECT: ground is clear
    assert _read_clearance(cm, level, gi, gj) == pytest.approx(4.15, abs=0.02)
