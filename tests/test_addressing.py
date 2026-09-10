"""
tests/test_addressing.py

Ticket #10 tests, per DRISHTI_Build_Map.md: the worked example from Bible
Part 8, a negative-coordinate round-trip property test, and the explicit
floor-vs-truncation assertion.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from grid.addressing import (
    flat_index,
    global_to_storage,
    global_to_world,
    is_power_of_two,
    world_to_global,
)


def test_worked_example_from_bible_part_8():
    # Vehicle at (1000.37, -240.12); query point (1012.00, -238.00) m.
    #
    # NOTE: DRISHTI_Project_Bible_v3.md Part 8 states si=16, sj=328,
    # flat=167952 for this example. That arithmetic is wrong -- verified
    # independently outside this codebase: 20240 & 511 == 272 (not 16)
    # and -4760 & 511 == 360 (not 328), since 20240 mod 512 = 272 and
    # -4760 mod 512 = 360. This test asserts the mathematically correct
    # values, not the document's. Flag this to whoever owns the Bible
    # text before it goes in front of a judge who checks the arithmetic.
    c_l = 0.05
    N = 512

    i, j = world_to_global(1012.00, -238.00, c_l)
    assert (i, j) == (20240, -4760)

    si, sj = global_to_storage(i, j, N)
    assert (si, sj) == (272, 360)

    flat = flat_index(si, sj, N)
    assert flat == 184_592


def test_negative_floor_not_truncation():
    # int(-3.2) == -3 (truncation, wrong); floor(-3.2) == -4 (correct).
    c_l = 0.05
    i, _ = world_to_global(-3.2 * c_l, 0.0, c_l)
    assert i == -4


@given(
    x=st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=10_000)
def test_round_trip_lands_inside_correct_cell(x, y):
    # Tiny-magnitude x/y (e.g. ~1e-25, hypothesis loves to probe these)
    # can make `x - x_cell` round to exactly c_l in float64 even though
    # the true mathematical difference is a hair below it -- the true
    # difference (~1e-25) is far smaller than c_l's ULP (~7e-18), so it
    # vanishes under subtraction. That is IEEE-754 rounding at the noise
    # floor, not an indexing bug, so the bound needs a tolerance many
    # orders of magnitude below any real off-by-one (which would be
    # wrong by ~c_l, not by 1e-9).
    eps = 1e-9
    c_l = 0.05
    i, j = world_to_global(x, y, c_l)
    x_cell, y_cell = global_to_world(i, j, c_l)
    # global_to_world returns the cell's lower-left corner.
    assert -eps <= x - x_cell < c_l + eps
    assert -eps <= y - y_cell < c_l + eps


@given(
    i=st.integers(min_value=-1_000_000, max_value=1_000_000),
    j=st.integers(min_value=-1_000_000, max_value=1_000_000),
)
@settings(max_examples=10_000)
def test_storage_index_always_in_range_including_negative(i, j):
    N = 512
    si, sj = global_to_storage(i, j, N)
    assert 0 <= si < N
    assert 0 <= sj < N


def test_negative_and_matches_two_complement_not_python_mod_abuse():
    # -4760 & 511 == 360 (verified independently; -4760 mod 512 = 360).
    # The Bible's Part 8 text states 328 for this example, which is an
    # arithmetic error in the document -- see
    # test_worked_example_from_bible_part_8 above.
    N = 512
    si, _ = global_to_storage(-4760, 0, N)
    assert si == 360


def test_flat_index_is_row_major_not_transposed():
    N = 512
    # sj * N + si, not si * N + sj.
    assert flat_index(si=16, sj=328, N=N) == 328 * N + 16
    assert flat_index(si=16, sj=328, N=N) != 16 * N + 328


def test_global_to_storage_requires_power_of_two_N():
    with pytest.raises(ValueError):
        global_to_storage(0, 0, 500)


def test_is_power_of_two():
    assert is_power_of_two(512)
    assert is_power_of_two(1)
    assert not is_power_of_two(0)
    assert not is_power_of_two(500)
    assert not is_power_of_two(-4)
