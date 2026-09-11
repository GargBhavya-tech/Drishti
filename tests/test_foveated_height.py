"""
tests/test_foveated_height.py

Ticket #17 tests: the foveated height quantum (1/2/4/8 cm for L0-L3) and
its per-tile base elevation. Build Map's own spec:

  "round-trip 10,000 random heights per level; max error <= quantum/2.
   Assert L3 saves 3 bytes/cell versus flat int16: 786,432 x 3 = 2.36 MB,
   taking the map from 12.58 -> 10.22 MB."

  "Watch out: a model will apply the quantum but forget the per-tile
   base, so coarse levels lose absolute elevation. Decode must be
   exact-inverse within half a quantum; test it rather than assume."

Also covers this build's own real risk (not in the Build Map, found
while implementing): a naive per-tile base that gets recomputed on every
write silently corrupts sibling cells already encoded relative to the
OLD base. The "reuse if live, else create" lifecycle in
grid/clipmap.py::Clipmap.get_or_create_tile_bases is what this file
spends most of its tests proving.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from grid.addressing import flat_index, global_to_storage, global_to_world
from grid.cell import (
    INT8_HI,
    INT8_LO,
    TILE_BASE_UNSET,
    TILE_SIZE,
    decode_h,
    encode_h,
    n_tiles_per_axis,
    quantum_for_level,
)
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


# ---------------------------------------------------------------------------
# Ticket #17's own required test: round-trip 10,000 random heights per level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_round_trip_10000_random_heights_within_half_quantum(level):
    rng = np.random.default_rng(level)  # deterministic, distinct seed per level
    quantum = quantum_for_level(level)

    if level == 0:
        heights = rng.uniform(-300.0, 300.0, 10_000)
        base = 0.0
    else:
        # Heights within the int8 range around an arbitrary, non-zero
        # base -- exactly the "coarse level, tile-relative" case.
        base = 5.37
        span = (INT8_HI - 1) * quantum  # stay inside, away from clamp edges
        heights = base + rng.uniform(-span, span, 10_000)

    max_err = 0.0
    for z in heights:
        v = encode_h(z, level=level, base_elevation_m=base)
        back = decode_h(int(v), level=level, base_elevation_m=base)
        max_err = max(max_err, abs(back - z))

    assert max_err <= quantum / 2 + 1e-9, f"level {level}: max round-trip error {max_err} exceeds quantum/2 ({quantum/2})"


def test_level_0_encode_decode_unchanged_from_pre_ticket_17_behaviour():
    """Backward-compat guarantee: every existing L0 call site
    (encode_h(z), decode_h(v), no level/base args) must see bit-identical
    output to before Ticket #17 existed."""
    for z in (0.0, -0.005, 3.456, -12.341, 100.004, 1.234):
        assert encode_h(z) == np.int16(round(z / 0.01))
        v = encode_h(z)
        assert decode_h(int(v)) == pytest.approx(float(int(v)) * 0.01)


def test_level_0_rejects_a_nonzero_base():
    with pytest.raises(ValueError):
        encode_h(1.0, level=0, base_elevation_m=0.5)


def test_offset_clamps_to_int8_range_rather_than_wrapping():
    quantum = quantum_for_level(3)
    base = 0.0
    # Deliberately far outside int8's range at this quantum.
    v = encode_h(base + 1000.0, level=3, base_elevation_m=base)
    assert int(v) == INT8_HI  # clamped to the top, not wrapped to something negative
    v2 = encode_h(base - 1000.0, level=3, base_elevation_m=base)
    assert int(v2) == INT8_LO


# ---------------------------------------------------------------------------
# Memory accounting: "Assert L3 saves 3 bytes/cell ... 12.58 -> 10.22 MB"
# ---------------------------------------------------------------------------


def test_v1_reference_figure_is_unchanged_by_ticket_17(hdl64e):
    """memory_bytes_v1() must keep reporting the ORIGINAL Ticket #11
    baseline (12.58 MB) regardless of what #17 actually allocates --
    it's a fixed comparison point, not a live measurement."""
    cm = Clipmap(hdl64e, n_levels=4, N=512, c0=0.05)
    assert cm.memory_bytes_v1() == 4 * 512 * 512 * 12
    assert cm.memory_bytes_v1() == 12_582_912


def test_v2_reports_a_real_reduction_close_to_the_build_maps_headline(hdl64e):
    """The Build Map's own '786,432 x 3 = 2.36 MB, 12.58 -> 10.22 MB'
    arithmetic doesn't account for the tile_base_h plane's own storage
    cost -- v2 does, honestly, so it lands close to but not exactly at
    10.22 MB (this build: ~10.23 MB, ~18.7% saved vs the claimed 18.8%)."""
    cm = Clipmap(hdl64e, n_levels=4, N=512, c0=0.05)
    v1 = cm.memory_bytes_v1()
    v2 = cm.memory_bytes_v2()

    assert v2 < v1  # a REAL reduction, not just a claim
    savings_fraction = (v1 - v2) / v1
    assert 0.17 < savings_fraction < 0.19  # close to the Build Map's 18.8%, honestly disclosed overhead included

    # The exact byte-for-byte figure this build's own TILE_SIZE produces,
    # so a future change to TILE_SIZE (or the plane layout) is caught.
    expected_h_fields = 512 * 512 * 2 * 3 + 3 * (512 * 512 * 1 * 3)  # L0 int16 x3 fields + (L1,L2,L3) int8 x3 fields
    expected_fixed = 4 * 512 * 512 * 6  # h_m2+count(2B each)+class_conf+flags(1B each)
    n_tiles = n_tiles_per_axis(512, TILE_SIZE) ** 2
    expected_tile_base = 4 * n_tiles * 2
    assert v2 == expected_h_fields + expected_fixed + expected_tile_base


# ---------------------------------------------------------------------------
# Tile-base lifecycle: the real risk this build identified and designed around
# ---------------------------------------------------------------------------


def test_fresh_tile_base_is_created_from_the_first_write(hdl64e):
    cm = _fresh(hdl64e)
    level = 1
    c_l = cm.levels[level].cell_size_m
    gi, gj = cm.origin_i[level] + 3, cm.origin_j[level] + 3
    x, y = global_to_world(gi, gj, c_l)

    scatter(cm, np.array([[x + c_l * 0.5, y + c_l * 0.5, 12.0]]))

    si, sj = global_to_storage(gi, gj, cm.N)
    flat = flat_index(si, sj, cm.N)
    _, view = cm.lookup(x + c_l * 0.5, y + c_l * 0.5)
    assert view.observed
    assert view.h_max_m == pytest.approx(12.0, abs=quantum_for_level(level) / 2 + 1e-6)


def test_second_write_to_the_same_tile_reuses_the_existing_base_not_a_new_one(hdl64e):
    """The core correctness property: two DIFFERENT cells in the SAME
    tile, written in TWO SEPARATE scatter() calls, must both decode
    correctly afterwards -- if the second write recomputed the tile's
    base instead of reusing it, the FIRST cell's already-stored int8
    value would now decode relative to the wrong base and silently
    read back wrong."""
    cm = _fresh(hdl64e)
    level = 1
    c_l = cm.levels[level].cell_size_m
    oi, oj = cm.origin_i[level], cm.origin_j[level]

    # Two adjacent cells, same tile (both // TILE_SIZE == 0 from origin).
    gi_a, gj_a = oi + 2, oj + 2
    gi_b, gj_b = oi + 3, oj + 2
    assert gi_a // TILE_SIZE == gi_b // TILE_SIZE  # sanity: genuinely the same tile

    xa, ya = global_to_world(gi_a, gj_a, c_l)
    xb, yb = global_to_world(gi_b, gj_b, c_l)

    # First write: cell A, height far from zero (a real terrain elevation).
    scatter(cm, np.array([[xa + c_l * 0.5, ya + c_l * 0.5, 50.0]]))
    # Second, SEPARATE write: cell B, a nearby but different height.
    scatter(cm, np.array([[xb + c_l * 0.5, yb + c_l * 0.5, 50.3]]))

    _, view_a = cm.lookup(xa + c_l * 0.5, ya + c_l * 0.5)
    _, view_b = cm.lookup(xb + c_l * 0.5, yb + c_l * 0.5)

    q = quantum_for_level(level)
    assert view_a.h_max_m == pytest.approx(50.0, abs=q / 2 + 1e-6), (
        "cell A's height changed after a LATER write to a different cell in the same tile -- "
        "the tile base was recomputed instead of reused, corrupting an already-encoded cell"
    )
    assert view_b.h_max_m == pytest.approx(50.3, abs=q / 2 + 1e-6)


def test_different_tiles_get_independent_bases(hdl64e):
    cm = _fresh(hdl64e)
    level = 2
    c_l = cm.levels[level].cell_size_m
    oi, oj = cm.origin_i[level], cm.origin_j[level]

    # Two cells far enough apart to land in different tiles.
    gi_a, gj_a = oi + 1, oj + 1
    gi_b, gj_b = oi + 1 + TILE_SIZE * 3, oj + 1
    assert gi_a // TILE_SIZE != gi_b // TILE_SIZE

    xa, ya = global_to_world(gi_a, gj_a, c_l)
    xb, yb = global_to_world(gi_b, gj_b, c_l)

    scatter(cm, np.array([[xa + c_l * 0.5, ya + c_l * 0.5, -20.0]]))
    scatter(cm, np.array([[xb + c_l * 0.5, yb + c_l * 0.5, 90.0]]))

    q = quantum_for_level(level)
    _, view_a = cm.lookup(xa + c_l * 0.5, ya + c_l * 0.5)
    _, view_b = cm.lookup(xb + c_l * 0.5, yb + c_l * 0.5)
    assert view_a.h_max_m == pytest.approx(-20.0, abs=q / 2 + 1e-6)
    assert view_b.h_max_m == pytest.approx(90.0, abs=q / 2 + 1e-6)


def test_tile_base_is_recreated_fresh_after_a_full_clear(hdl64e):
    """After scroll_to() fully clears a level (adversarial full wrap),
    a tile that gets written again must get a FRESH base derived from
    the new data, not a stale one left over from a completely different
    real-world location that used to alias the same storage slot."""
    cm = _fresh(hdl64e)
    level = 1
    c_l = cm.levels[level].cell_size_m
    gi, gj = cm.origin_i[level] + 3, cm.origin_j[level] + 3
    x, y = global_to_world(gi, gj, c_l)

    scatter(cm, np.array([[x + c_l * 0.5, y + c_l * 0.5, 200.0]]))
    _, view_before = cm.lookup(x + c_l * 0.5, y + c_l * 0.5)
    assert view_before.observed

    # Full wrap: clears everything at every level.
    cm.scroll_to(cm.N * c_l, 0.0)

    # A NEW real-world point at the SAME storage slot the old one
    # occupied (toroidal alias), with a wildly different height, as if
    # the vehicle returned to a slot that now represents different
    # terrain far away.
    si_old, sj_old = global_to_storage(gi, gj, cm.N)
    target_gi = cm.origin_i[level] + ((si_old - cm.origin_i[level]) % cm.N)
    target_gj = cm.origin_j[level] + ((sj_old - cm.origin_j[level]) % cm.N)
    tx, ty = global_to_world(target_gi, target_gj, c_l)

    scatter(cm, np.array([[tx + c_l * 0.5, ty + c_l * 0.5, -5.0]]))
    _, view_after = cm.lookup(tx + c_l * 0.5, ty + c_l * 0.5)
    q = quantum_for_level(level)
    assert view_after.observed
    assert view_after.h_max_m == pytest.approx(-5.0, abs=q / 2 + 1e-6), (
        "a stale tile base from before the clear leaked into the new write"
    )


def test_level_0_never_uses_tile_bases(hdl64e):
    cm = _fresh(hdl64e)
    with pytest.raises(ValueError):
        cm.get_or_create_tile_bases(0, np.array([0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# Property test: the round-trip holds for arbitrary values within range,
# not just the specific cases above.
# ---------------------------------------------------------------------------


@given(
    z=st.floats(min_value=-300.0, max_value=300.0, allow_nan=False, allow_infinity=False),
    level=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=2000)
def test_property_round_trip_never_exceeds_half_quantum_when_in_range(z, level):
    quantum = quantum_for_level(level)
    base = 0.0 if level == 0 else round(z / quantum) * quantum  # a base that keeps z well in-range for this test
    v = encode_h(z, level=level, base_elevation_m=base)
    back = decode_h(int(v), level=level, base_elevation_m=base)
    assert abs(back - z) <= quantum / 2 + 1e-9
