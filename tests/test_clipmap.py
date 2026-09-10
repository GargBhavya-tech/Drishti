"""
tests/test_clipmap.py

Tickets #11 (allocation), #13 (scroll / clear-on-scroll), #14 (stamp
validation), #15 (lookup), #16 (checkpoint: 200-frame integrity drive).

Bible Part 8 calls the clipmap "the load-bearing wall of the project" and
asks for "the most thorough test file in the repo" -- this is that file
for the production module. See DRISHTI_Build_Map.md tickets #11-#16 for
the exact test matrix each function below is checked against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grid.addressing import flat_index, global_to_storage
from grid.cell import OBS_FREE, OBS_OCCUPIED, OBS_UNOBSERVED, decode_h, encode_h, expected_stamp
from grid.clipmap import Clipmap
from sensor.sensor_model import load_sensor_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def hdl64e():
    return load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")


def _fresh(hdl64e) -> Clipmap:
    cm = Clipmap(hdl64e, n_levels=4, N=512, c0=0.05)
    cm.scroll_to(0.0, 0.0)  # establish an initial window centred on the origin
    return cm


def _fill_level_with_pattern(cm: Clipmap, level: int, h_value: int = 42) -> None:
    """Fill every cell of one level with a known, internally-consistent
    pattern: h_max = h_value, flags = OCCUPIED, and a correct stamp for
    each cell's own global address -- vectorised, not a 262,144-iteration
    Python loop.

    NOTE: storage index si is `gi & (N-1)`, NOT simply `gi - origin_i`.
    Those coincide only when origin_i is itself a multiple of N, which
    is not the general case (e.g. origin_i=-256 with N=512 is not a
    multiple of 512). An earlier version of this helper assumed the
    simple-offset mapping and silently wrote each stamp into the wrong
    storage slot -- harmless for the uniform h_max/flags fields (every
    slot gets the same value regardless), but it produced internally
    inconsistent stamps that only surfaced once something actually read
    them (test_checkpoint_200_frame_integrity_drive). Always derive si
    via `& (N-1)`, never via array-index-equals-offset.
    """
    N = cm.N
    shift = N.bit_length() - 1  # matches grid.cell.expected_stamp's tag shift
    gi = cm.origin_i[level] + np.arange(N, dtype=np.int64)
    gj = cm.origin_j[level] + np.arange(N, dtype=np.int64)
    si = gi & (N - 1)  # true storage index for each gi in the current window
    sj = gj & (N - 1)
    flat_grid = (sj[:, None] * N + si[None, :])  # shape (N, N), matches flat_index(si,sj,N)
    gi_grid = np.broadcast_to(gi[None, :], (N, N))
    gj_grid = np.broadcast_to(gj[:, None], (N, N))
    stamp_grid = (((gi_grid >> shift) & 0xFF) << 8) | ((gj_grid >> shift) & 0xFF)

    flat_flat = flat_grid.ravel()
    cm.h_max[level, flat_flat] = h_value
    cm.flags[level, flat_flat] = OBS_OCCUPIED
    cm.stamp[level, flat_flat] = stamp_grid.ravel().astype(np.uint16)


# ---------------------------------------------------------------------------
# Ticket #11 -- allocation, SoA planes, fixed-point encode/decode
# ---------------------------------------------------------------------------


def test_v1_memory_allocation_matches_reference_figure(hdl64e):
    cm = Clipmap(hdl64e, n_levels=4, N=512, c0=0.05)
    assert cm.memory_bytes_v1() == 4 * 512 * 512 * 12
    assert cm.memory_bytes_v1() == 12_582_912


def test_soa_planes_are_c_contiguous(hdl64e):
    cm = Clipmap(hdl64e, n_levels=4, N=512, c0=0.05)
    for name in ("h_min", "h_max", "h_mean", "h_m2", "count", "class_conf", "flags", "stamp"):
        arr = getattr(cm, name)
        assert arr.flags["C_CONTIGUOUS"], f"{name} is not C-contiguous"


def test_unobserved_is_the_zero_value_by_construction(hdl64e):
    cm = Clipmap(hdl64e, n_levels=4, N=512, c0=0.05)
    assert not np.any(cm.flags != 0)
    assert (cm.flags & 0b11).max() == OBS_UNOBSERVED


def test_encode_decode_height_round_trip_to_the_1cm_quantum():
    # 1.234 m must round-trip to 1.23 m (the 1 cm quantum), not to 1.0 m
    # (which would happen under truncation instead of rounding).
    v = encode_h(1.234)
    assert decode_h(int(v)) == pytest.approx(1.23, abs=1e-9)
    assert decode_h(int(v)) != pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("z", [0.0, -0.005, 3.456, -12.341, 100.004])
def test_encode_decode_round_trip_within_half_quantum(z):
    v = encode_h(z)
    assert abs(decode_h(int(v)) - z) <= 0.005 + 1e-9


# ---------------------------------------------------------------------------
# Ticket #13 -- scroll and clear-on-scroll
# ---------------------------------------------------------------------------


def test_scroll_clears_incoming_columns_and_retains_the_rest(hdl64e):
    cm = _fresh(hdl64e)
    level = 0
    old_origin_i, old_origin_j = cm.origin_i[level], cm.origin_j[level]
    N = cm.N
    _fill_level_with_pattern(cm, level, h_value=42)

    shift_cells = 30
    c_l = cm.levels[level].cell_size_m
    cm.scroll_to(shift_cells * c_l, 0.0)

    new_origin_i, new_origin_j = cm.origin_i[level], cm.origin_j[level]
    assert new_origin_i - old_origin_i == shift_cells
    assert new_origin_j == old_origin_j

    # Newly revealed global columns: [old_origin_i + N, old_origin_i + N + 30).
    for gi in range(old_origin_i + N, old_origin_i + N + shift_cells):
        for gj in range(new_origin_j, new_origin_j + N, 64):  # sample every 64th row
            si, sj = global_to_storage(gi, gj, N)
            flat = flat_index(si, sj, N)
            assert cm.flags[level, flat] == 0
            assert cm.h_max[level, flat] == 0

    # A retained column, well inside the overlap of old and new windows.
    # Must be > shift_cells from the old left edge, or it scrolled out of
    # the new window entirely and "retained" would be the wrong claim.
    gi_retained = old_origin_i + 200
    for gj in range(new_origin_j, new_origin_j + N, 64):
        si, sj = global_to_storage(gi_retained, gj, N)
        flat = flat_index(si, sj, N)
        assert cm.flags[level, flat] == OBS_OCCUPIED
        assert cm.h_max[level, flat] == 42


def test_adversarial_full_wrap_clears_everything(hdl64e):
    cm = _fresh(hdl64e)
    level = 0
    _fill_level_with_pattern(cm, level, h_value=42)

    c_l = cm.levels[level].cell_size_m
    # Scroll by exactly N cells at L0 -- a full wrap.
    cm.scroll_to(cm.N * c_l, 0.0)

    assert not np.any(cm.flags[level] != 0)
    assert not np.any(cm.h_max[level] != 0)
    assert not np.any(cm.stamp[level] != 0)


def test_scroll_cost_is_bounded_not_quadratic(hdl64e):
    cm = _fresh(hdl64e)
    c_l = cm.levels[0].cell_size_m
    cm.scroll_to(30 * c_l, 0.0)
    total_cells = cm.N * cm.N * len(cm.levels)
    # A correct O(N * (|di|+|dj|)) clear touches a small fraction of the
    # map; a naive O(N^2) rebuild would touch (close to) all of it.
    assert 0 < cm.last_scroll_cells_cleared < 0.1 * total_cells


def test_no_op_scroll_clears_nothing(hdl64e):
    cm = _fresh(hdl64e)
    cm.scroll_to(0.0, 0.0)  # same position again
    assert cm.last_scroll_cells_cleared == 0


# ---------------------------------------------------------------------------
# Ticket #14 -- stamp validation (the cross-check, not a replacement for #13)
# ---------------------------------------------------------------------------


def test_stamp_cross_check_catches_a_missed_clear(hdl64e):
    cm = _fresh(hdl64e)
    level = 0
    old_origin_i, old_origin_j = cm.origin_i[level], cm.origin_j[level]

    gi_old, gj_old = old_origin_i + 10, old_origin_j + 10
    si, sj = global_to_storage(gi_old, gj_old, cm.N)
    flat = flat_index(si, sj, cm.N)
    cm.h_max[level, flat] = 500
    cm.flags[level, flat] = OBS_OCCUPIED
    cm.stamp[level, flat] = expected_stamp(gi_old, gj_old, cm.N)

    # Deliberately disable the primary defence -- Ticket #14 exists
    # precisely to catch what this leaves behind.
    cm._clear_enabled = False
    c_l = cm.levels[level].cell_size_m
    shift_cells = 100
    cm.scroll_to(shift_cells * c_l, 0.0)
    new_origin_i = cm.origin_i[level]

    # gi_old is now behind the new window entirely (100 > 10), so whatever
    # global cell now shares storage slot `si` is guaranteed different.
    gi_new = None
    for candidate in range(new_origin_i, new_origin_i + cm.N):
        csi, _ = global_to_storage(candidate, gj_old, cm.N)
        if csi == si:
            gi_new = candidate
            break
    assert gi_new is not None
    assert gi_new != gi_old

    x_query = gi_new * c_l + c_l * 0.5
    y_query = gj_old * c_l + c_l * 0.5

    mismatches_before = cm.stamp_mismatches
    level_found, cell = cm.lookup(x_query, y_query)
    assert level_found == level
    assert not cell.observed
    assert cell.observability == OBS_UNOBSERVED
    assert cm.stamp_mismatches == mismatches_before + 1


def test_stamp_mismatches_never_fire_with_clearing_enabled(hdl64e):
    cm = _fresh(hdl64e)
    _fill_level_with_pattern(cm, level=0, h_value=7)
    for step in range(1, 21):
        cm.scroll_to(step * 1.5, step * 0.7)
    assert cm.stamp_mismatches == 0


# ---------------------------------------------------------------------------
# Ticket #15 -- lookup(): finest level containing the point, UNOBSERVED past extent
# ---------------------------------------------------------------------------


def test_lookup_selects_finest_level_at_each_ring_boundary(hdl64e):
    cm = _fresh(hdl64e)
    level, _ = cm.lookup(11.8, 0.0)
    assert level == 0
    level, _ = cm.lookup(40.0, 0.0)
    assert level == 2
    level, cell = cm.lookup(150.0, 0.0)
    assert level is None
    assert not cell.observed


def test_lookup_out_of_extent_returns_unobserved_never_a_wrapped_cell(hdl64e):
    cm = _fresh(hdl64e)
    level, cell = cm.lookup(1000.0, 1000.0)
    assert level is None
    assert cell.observability == OBS_UNOBSERVED
    assert cell.h_min_m is None
    assert cell.h_max_m is None
    assert cell.h_mean_m is None


def test_lookup_returns_written_values_for_an_observed_cell(hdl64e):
    cm = _fresh(hdl64e)
    level = 0
    gi, gj = cm.origin_i[level] + 5, cm.origin_j[level] + 5
    si, sj = global_to_storage(gi, gj, cm.N)
    flat = flat_index(si, sj, cm.N)
    cm.h_max[level, flat] = encode_h(1.23)
    cm.flags[level, flat] = OBS_FREE
    cm.stamp[level, flat] = expected_stamp(gi, gj, cm.N)

    c_l = cm.levels[level].cell_size_m
    x = gi * c_l + c_l * 0.5
    y = gj * c_l + c_l * 0.5
    found_level, cell = cm.lookup(x, y)
    assert found_level == 0
    assert cell.observed
    assert cell.observability == OBS_FREE
    assert cell.h_max_m == pytest.approx(1.23)


# ---------------------------------------------------------------------------
# Ticket #16 -- checkpoint: 200-frame clipmap integrity drive
# ---------------------------------------------------------------------------


def test_checkpoint_200_frame_integrity_drive(hdl64e):
    """No new production code, per the ticket -- drive 200 consecutive
    synthetic frames through scroll + lookup with the stamp-mismatch
    counter watched throughout. Must read zero for the whole drive, and
    memory must not grow (Bible Part 8: 'the stamp-mismatch counter on
    the HUD ... must read zero throughout')."""
    cm = _fresh(hdl64e)
    initial_bytes = cm.memory_bytes_v1()

    _fill_level_with_pattern(cm, level=0, h_value=1)

    rng = np.random.default_rng(1234)
    x, y = 0.0, 0.0
    for _frame in range(200):
        # ~15 m/s at 10 Hz -> up to ~1.5 m/frame, with some heading noise.
        x += rng.uniform(0.5, 1.5)
        y += rng.uniform(-0.3, 0.3)
        cm.scroll_to(x, y)
        cm.lookup(x + 10.0, y)  # exercise the read path every frame too

    assert cm.stamp_mismatches == 0
    assert cm.memory_bytes_v1() == initial_bytes
