"""
tests/test_scatter.py

Ticket #18 tests: a hand-checked 3x3-cell pattern, a slow-reference-loop
cross-check on random points, and a timing sanity check that this is
genuinely vectorised (no Python loop over points).
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from grid.addressing import flat_index, global_to_storage, global_to_world, world_to_global
from grid.cell import H_QUANTUM_M, OBS_OCCUPIED, OBS_UNOBSERVED, decode_h, encode_h
from grid.clipmap import Clipmap
from grid.scatter import scatter
from sensor.sensor_model import load_sensor_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def hdl64e():
    return load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")


def _fresh(hdl64e) -> Clipmap:
    cm = Clipmap(hdl64e, n_levels=4, N=512, c0=0.05)
    cm.scroll_to(0.0, 0.0)
    return cm


def test_nine_points_three_cells_hand_checked(hdl64e):
    cm = _fresh(hdl64e)
    level = 0
    c_l = cm.levels[level].cell_size_m
    oi, oj = cm.origin_i[level], cm.origin_j[level]

    # Three distinct cells (a "3x3" pattern of cells), 3 points each.
    cell_a = (oi + 5, oj + 5)
    cell_b = (oi + 5, oj + 6)
    cell_c = (oi + 6, oj + 5)

    def pts_in(cell, zs):
        gi, gj = cell
        x0, y0 = global_to_world(gi, gj, c_l)
        # Slightly different offsets within the cell so points are
        # distinguishable, all landing in the same cell.
        return [(x0 + c_l * f, y0 + c_l * f, z) for f, z in zip((0.1, 0.5, 0.9), zs)]

    points = (
        pts_in(cell_a, [1.0, 2.0, 3.0])
        + pts_in(cell_b, [0.0, 0.0, 0.0])
        + pts_in(cell_c, [-1.0, 1.0, 5.0])
    )
    xyz = np.array(points, dtype=np.float64)
    assert xyz.shape == (9, 3)

    scatter(cm, xyz)

    def read(cell):
        si, sj = global_to_storage(cell[0], cell[1], cm.N)
        flat = flat_index(si, sj, cm.N)
        return {
            "h_min": decode_h(int(cm.h_min[level, flat])),
            "h_max": decode_h(int(cm.h_max[level, flat])),
            "h_mean": decode_h(int(cm.h_mean[level, flat])),
            "count": int(cm.count[level, flat]),
            "flags": int(cm.flags[level, flat]),
        }

    a = read(cell_a)
    assert a["h_min"] == pytest.approx(1.0, abs=0.01)
    assert a["h_max"] == pytest.approx(3.0, abs=0.01)
    assert a["h_mean"] == pytest.approx(2.0, abs=0.01)
    assert a["count"] == 3
    assert a["flags"] == OBS_OCCUPIED

    b = read(cell_b)
    assert b["h_min"] == pytest.approx(0.0, abs=0.01)
    assert b["h_max"] == pytest.approx(0.0, abs=0.01)
    assert b["h_mean"] == pytest.approx(0.0, abs=0.01)
    assert b["count"] == 3

    c = read(cell_c)
    assert c["h_min"] == pytest.approx(-1.0, abs=0.01)
    assert c["h_max"] == pytest.approx(5.0, abs=0.01)
    assert c["h_mean"] == pytest.approx(5.0 / 3.0, abs=0.01)
    assert c["count"] == 3

    # An untouched cell must remain exactly as allocated -- UNOBSERVED.
    si, sj = global_to_storage(oi + 100, oj + 100, cm.N)
    flat = flat_index(si, sj, cm.N)
    assert cm.flags[level, flat] == OBS_UNOBSERVED


def test_include_self_keeps_a_single_point_in_a_fresh_cell(hdl64e):
    """Ticket #18 'Watch out' #2 -- a single point into a previously-empty
    cell must not be discarded."""
    cm = _fresh(hdl64e)
    level = 0
    c_l = cm.levels[level].cell_size_m
    gi, gj = cm.origin_i[level] + 42, cm.origin_j[level] + 42
    x, y = global_to_world(gi, gj, c_l)
    xyz = np.array([[x + c_l * 0.5, y + c_l * 0.5, 1.5]], dtype=np.float64)

    scatter(cm, xyz)

    si, sj = global_to_storage(gi, gj, cm.N)
    flat = flat_index(si, sj, cm.N)
    assert cm.count[level, flat] == 1
    assert decode_h(int(cm.h_max[level, flat])) == pytest.approx(1.5, abs=0.01)
    assert decode_h(int(cm.h_min[level, flat])) == pytest.approx(1.5, abs=0.01)


def test_count_saturates_never_wraps(hdl64e):
    cm = _fresh(hdl64e)
    level = 0
    c_l = cm.levels[level].cell_size_m
    gi, gj = cm.origin_i[level] + 3, cm.origin_j[level] + 3
    x, y = global_to_world(gi, gj, c_l)
    n_points = 70_000  # > uint16 max (65535), all landing in one cell
    xyz = np.stack(
        [
            np.full(n_points, x + c_l * 0.5),
            np.full(n_points, y + c_l * 0.5),
            np.linspace(0.0, 1.0, n_points),
        ],
        axis=1,
    )
    scatter(cm, xyz)
    si, sj = global_to_storage(gi, gj, cm.N)
    flat = flat_index(si, sj, cm.N)
    assert cm.count[level, flat] == 65535  # saturated, not wrapped to a small number


def _reference_scatter(xyz: np.ndarray, cm: Clipmap, level: int):
    """Slow, obviously-correct oracle: a plain Python loop. Ticket #18
    asks for exactly this as the cross-check, not as production code."""
    agg = defaultdict(list)
    c_l = cm.levels[level].cell_size_m
    oi, oj = cm.origin_i[level], cm.origin_j[level]
    N = cm.N
    for x, y, z in xyz:
        gi, gj = world_to_global(x, y, c_l)
        if not (oi <= gi < oi + N and oj <= gj < oj + N):
            continue
        si, sj = global_to_storage(gi, gj, N)
        flat = flat_index(si, sj, N)
        agg[flat].append(z)
    ref_count = {flat: min(len(zs), 65535) for flat, zs in agg.items()}
    ref_max = {flat: max(zs) for flat, zs in agg.items()}
    return ref_count, ref_max


def test_matches_slow_reference_loop_on_random_points(hdl64e):
    cm = _fresh(hdl64e)
    level = 0
    rng = np.random.default_rng(42)
    n = 5000
    xyz = np.stack(
        [
            rng.uniform(-10.0, 10.0, n),
            rng.uniform(-10.0, 10.0, n),
            rng.uniform(-1.0, 3.0, n),
        ],
        axis=1,
    )

    scatter(cm, xyz)
    ref_count, ref_max = _reference_scatter(xyz, cm, level)

    assert len(ref_count) > 0  # sanity: the random points actually hit cells
    for flat, count in ref_count.items():
        assert int(cm.count[level, flat]) == count
        assert int(cm.h_max[level, flat]) == int(encode_h(ref_max[flat]))

    # And nothing outside the reference's touched set got marked observed.
    all_flats = np.arange(cm.N * cm.N)
    untouched = np.setdiff1d(all_flats, np.array(list(ref_count.keys())))
    assert not np.any(cm.flags[level, untouched] != 0)


def test_scatter_is_vectorised_not_a_python_loop(hdl64e):
    """Ticket #18: 'A Python loop over 34k-120k points is fatal ... It
    must be a single vectorised call per level.' A real point-by-point
    Python loop across 4 levels would take many seconds; this asserts a
    bound loose enough not to flake on a slow CI box but tight enough
    that a regression to a Python loop would fail it immediately."""
    cm = _fresh(hdl64e)
    rng = np.random.default_rng(7)
    n = 35_000
    xyz = np.stack(
        [
            rng.uniform(-100.0, 100.0, n),
            rng.uniform(-100.0, 100.0, n),
            rng.uniform(-2.0, 5.0, n),
        ],
        axis=1,
    )

    start = time.perf_counter()
    scatter(cm, xyz)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"scatter() took {elapsed:.3f}s for {n} points -- looks like a Python loop"
