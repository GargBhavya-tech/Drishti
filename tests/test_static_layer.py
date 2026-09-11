"""
tests/test_static_layer.py

Ticket #44 tests. Build Map's own acceptance test: "merge two groups
with known statistics; assert the combined mean and variance match a
direct computation over the union of all points, exactly to floating-
point tolerance. Assert count saturates rather than wraps at 65,535."
Plus an integration check that repeated frames on a live Clipmap
accumulate rather than overwrite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grid.addressing import global_to_world
from grid.clipmap import Clipmap
from sensor.sensor_model import load_sensor_config
from temporal.static_layer import (
    ACCUMULATION_WINDOW_S,
    GroupStats,
    StaticLayerAccumulator,
    chan_merge,
    chan_merge_batch,
    confidence_from_age,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def hdl64e():
    return load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")


def _direct_stats(values: np.ndarray) -> GroupStats:
    n = len(values)
    mean = float(np.mean(values))
    m2 = float(np.sum((values - mean) ** 2))
    return GroupStats(n=n, mean=mean, m2=m2)


# ---------------------------------------------------------------------------
# chan_merge -- exact combination, the ticket's own acceptance test
# ---------------------------------------------------------------------------


def test_chan_merge_matches_direct_computation_over_the_union():
    rng = np.random.default_rng(0)
    a_vals = rng.normal(1.0, 0.3, size=37)
    b_vals = rng.normal(1.5, 0.6, size=53)

    merged = chan_merge(_direct_stats(a_vals), _direct_stats(b_vals))
    direct = _direct_stats(np.concatenate([a_vals, b_vals]))

    assert merged.n == direct.n
    assert merged.mean == pytest.approx(direct.mean, abs=1e-9)
    assert merged.m2 == pytest.approx(direct.m2, abs=1e-6)


def test_chan_merge_with_an_empty_group_returns_the_other_group_exactly():
    a = GroupStats(n=0, mean=0.0, m2=0.0)
    b = GroupStats(n=10, mean=3.3, m2=12.4)
    assert chan_merge(a, b) == b
    assert chan_merge(b, a) == b


def test_chan_merge_three_way_is_associative_to_floating_point_tolerance():
    rng = np.random.default_rng(1)
    a_vals = rng.normal(0.0, 1.0, size=20)
    b_vals = rng.normal(0.5, 1.0, size=15)
    c_vals = rng.normal(-0.5, 1.0, size=25)

    left = chan_merge(chan_merge(_direct_stats(a_vals), _direct_stats(b_vals)), _direct_stats(c_vals))
    right = chan_merge(_direct_stats(a_vals), chan_merge(_direct_stats(b_vals), _direct_stats(c_vals)))
    direct = _direct_stats(np.concatenate([a_vals, b_vals, c_vals]))

    for result in (left, right):
        assert result.n == direct.n
        assert result.mean == pytest.approx(direct.mean, abs=1e-9)
        assert result.m2 == pytest.approx(direct.m2, abs=1e-6)


def test_chan_merge_batch_matches_scalar_chan_merge_elementwise():
    rng = np.random.default_rng(2)
    n_a = rng.integers(0, 20, size=8)
    n_b = rng.integers(0, 20, size=8)
    mean_a = rng.normal(size=8)
    mean_b = rng.normal(size=8)
    m2_a = rng.uniform(0, 5, size=8)
    m2_b = rng.uniform(0, 5, size=8)
    # Zero-count groups must carry zero mean/m2, matching how the
    # accumulator itself always zeroes an "empty" existing group.
    m2_a = np.where(n_a == 0, 0.0, m2_a)
    m2_b = np.where(n_b == 0, 0.0, m2_b)
    mean_a = np.where(n_a == 0, 0.0, mean_a)
    mean_b = np.where(n_b == 0, 0.0, mean_b)

    n, mean, m2 = chan_merge_batch(n_a, mean_a, m2_a, n_b, mean_b, m2_b)

    for i in range(8):
        scalar = chan_merge(
            GroupStats(int(n_a[i]), float(mean_a[i]), float(m2_a[i])),
            GroupStats(int(n_b[i]), float(mean_b[i]), float(m2_b[i])),
        )
        assert n[i] == pytest.approx(scalar.n)
        assert mean[i] == pytest.approx(scalar.mean, abs=1e-9)
        assert m2[i] == pytest.approx(scalar.m2, abs=1e-6)


# ---------------------------------------------------------------------------
# confidence_from_age -- Bible Part 12.3
# ---------------------------------------------------------------------------


def test_confidence_from_age_is_one_at_zero_and_decays_toward_zero():
    assert confidence_from_age(0.0) == pytest.approx(1.0)
    assert confidence_from_age(1000.0) == pytest.approx(0.0, abs=1e-9)
    assert confidence_from_age(10.0) < confidence_from_age(5.0) < confidence_from_age(0.0)


# ---------------------------------------------------------------------------
# StaticLayerAccumulator -- integration against a live Clipmap
# ---------------------------------------------------------------------------


def _fresh_cm(hdl64e) -> Clipmap:
    cm = Clipmap(hdl64e, n_levels=4, N=512, c0=0.05)
    cm.scroll_to(0.0, 0.0)
    return cm


def test_count_saturates_rather_than_wraps_at_65535(hdl64e):
    cm = _fresh_cm(hdl64e)
    acc = StaticLayerAccumulator(cm)
    level = 0
    gi, gj = cm.origin_i[level] + 10, cm.origin_j[level] + 10
    c_l = cm.levels[level].cell_size_m
    x0, y0 = global_to_world(gi, gj, c_l)
    xyz = np.array([[x0 + c_l * 0.5, y0 + c_l * 0.5, 0.1]] * 40000, dtype=np.float64)

    for t in range(3):
        acc.accumulate_frame(xyz, t_s=float(t) * 0.5)  # within the 3s window

    _, cell = cm.lookup(x0 + c_l * 0.5, y0 + c_l * 0.5)
    assert cell.count == 65535


def test_repeated_frames_on_the_same_cell_produce_the_exact_chan_merged_mean_and_variance(hdl64e):
    cm = _fresh_cm(hdl64e)
    acc = StaticLayerAccumulator(cm)
    level = 0
    gi, gj = cm.origin_i[level] + 20, cm.origin_j[level] + 20
    c_l = cm.levels[level].cell_size_m
    x0, y0 = global_to_world(gi, gj, c_l)
    cx, cy = x0 + c_l * 0.5, y0 + c_l * 0.5

    rng = np.random.default_rng(3)
    frame_a = rng.normal(1.0, 0.05, size=200)
    frame_b = rng.normal(1.02, 0.06, size=150)

    xyz_a = np.array([[cx, cy, z] for z in frame_a], dtype=np.float64)
    xyz_b = np.array([[cx, cy, z] for z in frame_b], dtype=np.float64)

    acc.accumulate_frame(xyz_a, t_s=0.0)
    acc.accumulate_frame(xyz_b, t_s=0.1)  # well within the 3s window

    _, cell = cm.lookup(cx, cy)
    direct_mean = float(np.mean(np.concatenate([frame_a, frame_b])))

    assert cell.count == 350
    assert cell.h_mean_m == pytest.approx(direct_mean, abs=0.01)  # within L0's 1cm quantum


def test_a_frame_older_than_the_accumulation_window_starts_fresh_not_merged(hdl64e):
    cm = _fresh_cm(hdl64e)
    acc = StaticLayerAccumulator(cm)
    level = 0
    gi, gj = cm.origin_i[level] + 30, cm.origin_j[level] + 30
    c_l = cm.levels[level].cell_size_m
    x0, y0 = global_to_world(gi, gj, c_l)
    cx, cy = x0 + c_l * 0.5, y0 + c_l * 0.5

    xyz_old = np.array([[cx, cy, 5.0]] * 100, dtype=np.float64)  # a wildly different height
    xyz_new = np.array([[cx, cy, 0.1]] * 100, dtype=np.float64)

    acc.accumulate_frame(xyz_old, t_s=0.0)
    acc.accumulate_frame(xyz_new, t_s=ACCUMULATION_WINDOW_S + 1.0)  # past the cap

    _, cell = cm.lookup(cx, cy)
    # A merge would pull the mean toward 5.0; a fresh start reads ~0.1.
    assert cell.h_mean_m == pytest.approx(0.1, abs=0.02)
    assert cell.count == 100


def test_30_frames_produce_a_visibly_denser_map_than_a_single_sweep(hdl64e):
    cm = _fresh_cm(hdl64e)
    acc = StaticLayerAccumulator(cm)
    level = 0
    gi, gj = cm.origin_i[level] + 40, cm.origin_j[level] + 40
    c_l = cm.levels[level].cell_size_m
    x0, y0 = global_to_world(gi, gj, c_l)
    cx, cy = x0 + c_l * 0.5, y0 + c_l * 0.5

    rng = np.random.default_rng(4)
    for frame in range(30):
        xyz = np.array([[cx, cy, z] for z in rng.normal(0.2, 0.01, size=10)], dtype=np.float64)
        acc.accumulate_frame(xyz, t_s=frame * 0.1)

    _, cell = cm.lookup(cx, cy)
    assert cell.count == 300  # 30 frames x 10 points, correctly ACCUMULATED not overwritten
