"""
planning/path_planner.py

A* path planning over the cost grid `planning.costmap.cost()` produces
-- closes the loop Ticket #49 leaves open: a cost map is a NUMBER per
cell, not yet a DECISION. Bible Part 14's own framing: "a coloured map
shows perception; a path re-routing around a pedestrian shows
CONSEQUENCE, and consequence is what judges remember."

Grid convention: a 2D numpy array of per-cell costs (float), the SAME
units `planning.conservatism`/`planning.costmap` already use
(FREE_COST=0, UNKNOWN_COST=50, LETHAL=inf). A* treats LETHAL cells as
literally impassable (never expanded); everything else is a real
traversal cost -- so a route through UNKNOWN terrain is expensive but
NOT forbidden, matching this project's own tri-state argument (Bible
Part 10): unknown is cautious, never the same as blocked.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

Cell = Tuple[int, int]

# 8-connected neighbourhood; a diagonal step costs sqrt(2) x the
# destination cell's own cost multiplier, matching real travelled
# distance -- an unweighted diagonal would make diagonal routes
# artificially cheap relative to axis-aligned ones.
_NEIGHBOR_OFFSETS: List[Tuple[int, int, float]] = [
    (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
    (0, -1, 1.0), (0, 1, 1.0),
    (1, -1, math.sqrt(2)), (1, 0, 1.0), (1, 1, math.sqrt(2)),
]


def _heuristic(a: Cell, b: Cell) -> float:
    """Euclidean distance -- admissible for an 8-connected grid with
    real (not underestimated) diagonal step costs, so A* remains
    optimal (never overestimates true remaining cost)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class PathResult:
    path: Optional[List[Cell]]  # None if no path exists -- goal unreachable, not an exception
    total_cost: float
    cells_expanded: int


def find_path(cost_grid: np.ndarray, start: Cell, goal: Cell) -> PathResult:
    """A* over `cost_grid` (H, W) from `start` to `goal`, both (row,
    col) grid indices. A cell whose cost is `math.inf` (LETHAL, per
    `planning.conservatism.LETHAL`) is NEVER expanded -- a hard wall,
    not merely expensive. Returns `path=None` (never raises) when no
    route exists: "no safe route" is itself a valid, expected planner
    answer this project's own conservatism culture requires the caller
    to see, not one hidden behind a crash.
    """
    H, W = cost_grid.shape
    if not (0 <= start[0] < H and 0 <= start[1] < W):
        raise ValueError(f"start {start} outside grid shape {cost_grid.shape}")
    if not (0 <= goal[0] < H and 0 <= goal[1] < W):
        raise ValueError(f"goal {goal} outside grid shape {cost_grid.shape}")

    open_heap: List[Tuple[float, Cell]] = [(0.0, start)]
    g_score: Dict[Cell, float] = {start: 0.0}
    came_from: Dict[Cell, Cell] = {}
    visited: set = set()
    cells_expanded = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        cells_expanded += 1

        if current == goal:
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return PathResult(path=path, total_cost=g_score[current], cells_expanded=cells_expanded)

        for di, dj, dist in _NEIGHBOR_OFFSETS:
            ni, nj = current[0] + di, current[1] + dj
            if not (0 <= ni < H and 0 <= nj < W):
                continue
            neighbor = (ni, nj)
            if neighbor in visited:
                continue
            neighbor_cost = float(cost_grid[ni, nj])
            if not math.isfinite(neighbor_cost):
                continue  # LETHAL -- never expanded, a hard wall, not just expensive

            # +1 so a FREE (cost=0) cell still costs real travelled
            # distance, not zero -- otherwise A* would treat an
            # arbitrarily long detour through all-FREE cells as
            # equally good as the direct route.
            step_cost = dist * (1.0 + neighbor_cost)
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + _heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))

    return PathResult(path=None, total_cost=math.inf, cells_expanded=cells_expanded)
