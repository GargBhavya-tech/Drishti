"""
tests/test_nesting_exactness.py

Ticket #12 -- THE PROOF. No new production code: this file proves the
central claim of the project (Bible Part 8) -- a level-(l+1) cell is
exactly the union of four level-l cells, at every position, for every ego
pose. If anyone ever adds a per-level origin offset "to centre the grid",
this test is what breaks and catches it.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from grid.addressing import world_to_global

C0 = 0.05


def c(level: int) -> float:
    return C0 * (2**level)


@given(
    x=st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False),
    l=st.integers(min_value=0, max_value=2),
)
@settings(max_examples=10_000)
def test_nesting_exact(x, y, l):
    i0, j0 = world_to_global(x, y, c(l))
    i1, j1 = world_to_global(x, y, c(l + 1))
    assert i1 == i0 >> 1
    assert j1 == j0 >> 1


@given(
    I=st.integers(min_value=-200_000, max_value=200_000),
    J=st.integers(min_value=-200_000, max_value=200_000),
)
@settings(max_examples=10_000)
def test_four_children_tile_parent_with_no_gap_or_overlap(I, J):
    """A level-(l+1) cell at (I, J) covers exactly (2I,2J), (2I+1,2J),
    (2I,2J+1), (2I+1,2J+1) at level l -- every point in the parent's
    world-space footprint maps to one of exactly those four children,
    with no gap and no overlap.
    """
    l = 1
    c_parent = c(l + 1)
    c_child = c(l)

    children = {(2 * I, 2 * J), (2 * I + 1, 2 * J), (2 * I, 2 * J + 1), (2 * I + 1, 2 * J + 1)}
    assert len(children) == 4  # no accidental collision

    # Sample points across the *interior* of the parent cell's footprint
    # and confirm each lands in exactly one of the four declared children
    # -- never zero (gap), never more than one (overlap, impossible by
    # construction of world_to_global but checked anyway since this is
    # the safety-critical proof). Deliberately avoid the exact 0.0 and
    # c_parent boundary offsets: reconstructing a cell edge as I*c_parent
    # via multiplication is not always bit-exact in float64, so a sample
    # placed exactly ON that reconstructed boundary can round into the
    # mathematically-adjacent cell -- a float-representability artifact
    # of this test's own construction, not a world_to_global bug (see
    # test_addressing.py::test_round_trip_lands_inside_correct_cell,
    # which stress-tests exactly this with a tolerance). Staying inside
    # the interior avoids relying on bit-exact boundary reconstruction.
    x0, y0 = I * c_parent, J * c_parent
    offsets = [c_parent * f for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    for dx in offsets:
        for dy in offsets:
            x, y = x0 + dx, y0 + dy
            ci, cj = world_to_global(x, y, c_child)
            assert (ci, cj) in children


def test_worked_example_children_of_a_known_parent():
    # Sanity-check the abstract property against one concrete cell.
    I, J = 100, -37
    children = {(2 * I, 2 * J), (2 * I + 1, 2 * J), (2 * I, 2 * J + 1), (2 * I + 1, 2 * J + 1)}
    assert children == {(200, -74), (201, -74), (200, -73), (201, -73)}
