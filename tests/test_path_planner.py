"""
tests/test_path_planner.py

A* over the cost grid: a straight shot when nothing's in the way, a
real detour around a LETHAL wall, "no path" (not a crash) when
genuinely blocked, and UNKNOWN-cost terrain being expensive but never
forbidden -- the tri-state argument this project holds everywhere
else, now also true of the planner.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from planning.path_planner import find_path


def test_straight_line_on_an_all_free_grid():
    grid = np.zeros((10, 10))
    result = find_path(grid, (0, 0), (9, 9))
    assert result.path is not None
    assert result.path[0] == (0, 0)
    assert result.path[-1] == (9, 9)
    # Diagonal-optimal route on an all-free grid: 9 diagonal steps.
    assert len(result.path) == 10


def test_lethal_wall_forces_a_real_detour():
    grid = np.zeros((10, 10))
    grid[:, 5] = math.inf  # a solid LETHAL wall down column 5
    grid[9, 5] = 0.0  # one gap at the bottom row

    result = find_path(grid, (0, 0), (0, 9))
    assert result.path is not None
    # The route must pass through the gap -- confirms it actually
    # detoured, not just found a shorter fictitious path.
    assert (9, 5) in result.path
    for (r, c) in result.path:
        assert grid[r, c] != math.inf  # never steps on a LETHAL cell


def test_completely_blocked_goal_returns_none_not_an_exception():
    grid = np.zeros((5, 5))
    grid[:, 2] = math.inf  # a solid, gapless wall

    result = find_path(grid, (0, 0), (0, 4))
    assert result.path is None
    assert result.total_cost == math.inf


def test_unknown_cost_terrain_is_expensive_but_not_forbidden():
    """A route through UNKNOWN_COST terrain must still be RETURNED
    when it is the only option -- unknown is cautious, never the same
    as blocked (Bible Part 10's tri-state argument, extended to the
    planner)."""
    grid = np.zeros((5, 5))
    grid[:, 2] = 50.0  # UNKNOWN_COST-like terrain, not LETHAL

    result = find_path(grid, (0, 0), (0, 4))
    assert result.path is not None
    assert result.total_cost < math.inf
    assert result.total_cost > 4.0  # strictly more expensive than the free-only straight line


def test_a_cheaper_detour_is_preferred_over_a_costly_direct_route():
    grid = np.zeros((5, 5))
    grid[2, :] = 100.0  # an expensive but passable band straight across the middle
    grid[2, 4] = 0.0  # one cheap gap at the edge

    result = find_path(grid, (0, 2), (4, 2))
    assert result.path is not None
    # Must route through the cheap gap rather than paying the 100-cost
    # band directly -- proves the planner is actually cost-aware, not
    # just finding *a* path.
    assert any(cell == (2, 4) for cell in result.path)


def test_start_equals_goal_returns_trivial_single_cell_path():
    grid = np.zeros((3, 3))
    result = find_path(grid, (1, 1), (1, 1))
    assert result.path == [(1, 1)]
    assert result.total_cost == 0.0


def test_start_or_goal_outside_grid_raises():
    grid = np.zeros((3, 3))
    with pytest.raises(ValueError):
        find_path(grid, (-1, 0), (1, 1))
    with pytest.raises(ValueError):
        find_path(grid, (0, 0), (5, 5))


def test_cells_expanded_is_tracked_and_positive():
    grid = np.zeros((4, 4))
    result = find_path(grid, (0, 0), (3, 3))
    assert result.cells_expanded > 0
