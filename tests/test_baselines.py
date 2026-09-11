"""
tests/test_baselines.py

Ticket #56 tests, per the Build Map's own literal acceptance numbers:
"assert DRISHTI = 1,048,576 cells = 12.58 MB (or 10.22 MB with #17),
dense 2.5D = 16,777,216 cells = 201.3 MB, ratio 16.0x. Assert occupancy
of the uniform grid on one sweep is < 1% (34k points / 16.8M cells on
nuScenes ~ 0.2%)." Plus sanity checks on the dense-3D strawman and
sparse-hash baselines, which the Build Map does not pin to exact
numbers (see eval/baselines.py's own module docstring on "267x").
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.baselines import (
    compare_all,
    dense_2p5d_bytes,
    dense_3d_bytes,
    drishti_bytes,
    occupancy_fraction,
    sparse_hash_voxel_bytes,
)
from grid.clipmap import Clipmap
from sensor.sensor_model import load_sensor_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def hdl64e():
    return load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")


@pytest.fixture
def cm(hdl64e) -> Clipmap:
    return Clipmap(hdl64e, n_levels=4, N=512, c0=0.05)


def test_drishti_v1_matches_the_build_maps_own_headline_number(cm):
    result = drishti_bytes(cm, use_v2=False)
    assert result.n_cells == 1_048_576
    assert result.megabytes == pytest.approx(12.58, abs=0.01)


def test_dense_2p5d_at_matched_extent_matches_the_build_maps_own_headline_number(cm):
    extent_m = cm.N * cm.levels[-1].cell_size_m  # DRISHTI's own coarsest-level array extent
    result = dense_2p5d_bytes(extent_m)
    assert result.n_cells == 16_777_216
    assert result.megabytes == pytest.approx(201.3, abs=0.1)


def test_ratio_of_dense_2p5d_over_drishti_is_16x(cm):
    report = compare_all(cm, n_occupied_cells=34_000)
    assert report.ratio_2p5d_over_drishti == pytest.approx(16.0, abs=0.05)


def test_occupancy_of_one_sweep_on_the_uniform_grid_is_under_one_percent(cm):
    extent_m = cm.N * cm.levels[-1].cell_size_m
    dense = dense_2p5d_bytes(extent_m)
    frac = occupancy_fraction(n_occupied_cells=34_000, n_total_cells=dense.n_cells)
    assert frac < 0.01
    assert frac == pytest.approx(0.002, abs=0.001)  # ~0.2%, the Build Map's own illustrative figure


def test_drishti_v2_reports_real_measured_savings_below_v1(cm):
    v1 = drishti_bytes(cm, use_v2=False)
    v2 = drishti_bytes(cm, use_v2=True)
    assert v2.bytes_total < v1.bytes_total  # Ticket #17's real savings
    assert v2.megabytes == pytest.approx(cm.memory_bytes_v2() / 1e6)


# ---------------------------------------------------------------------------
# Dense-3D strawman and sparse-hash baselines: honest, documented
# estimates, not exact Bible-quoted figures (see module docstring).
# ---------------------------------------------------------------------------


def test_dense_3d_strawman_is_far_larger_than_dense_2p5d_at_the_same_extent(cm):
    extent_m = cm.N * cm.levels[-1].cell_size_m
    dense_3d = dense_3d_bytes(extent_m)
    dense_2p5d = dense_2p5d_bytes(extent_m)
    # 3D adds a whole vertical dimension on top of 2.5D's flattening --
    # it must be strictly larger, and by roughly the vertical voxel
    # count (not exactly, since dense_3d_bytes rounds n_z independently).
    assert dense_3d.bytes_total > dense_2p5d.bytes_total
    assert dense_3d.n_cells == dense_2p5d.n_cells * round(10.0 / 0.05)


def test_dense_3d_is_labelled_as_the_naive_strawman_never_the_lead_number():
    result = dense_3d_bytes(100.0)
    assert "naive" in result.name or "strawman" in result.name


def test_sparse_hash_voxel_scales_with_occupied_cells_not_extent():
    few = sparse_hash_voxel_bytes(n_occupied_cells=100)
    many = sparse_hash_voxel_bytes(n_occupied_cells=100_000)
    assert many.bytes_total > few.bytes_total
    assert many.bytes_total == pytest.approx(few.bytes_total * 1000, rel=1e-9)  # linear in occupied cells


def test_sparse_hash_voxel_costs_more_per_cell_than_the_raw_payload_alone():
    from eval.baselines import BYTES_PER_CELL

    result = sparse_hash_voxel_bytes(n_occupied_cells=1)
    assert result.bytes_total > BYTES_PER_CELL  # real key/pointer overhead is not free


def test_compare_all_reports_all_four_baselines_at_the_same_implied_extent(cm):
    report = compare_all(cm, n_occupied_cells=34_000)
    # dense_3d and dense_2p5d must share the same XY cell count (same extent)
    n_z = round(10.0 / 0.05)
    assert report.dense_3d.n_cells == report.dense_2p5d.n_cells * n_z
