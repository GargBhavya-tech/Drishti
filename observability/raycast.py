"""
observability/raycast.py

Ticket #33 -- DDA ray traversal (2D, Amanatides-Woo) along each beam's
ground projection, sensor to return.

Traverses at the clipmap's COARSEST level by default; refines to the
finest level only within that level's own "fine ring" (its Ticket #5
sensor-Nyquist radius) -- Build Map "Watch out": traversing every beam
at L0 across the full extent is millions of cell visits per frame and
destroys the latency budget. Coarse-first is mandatory.

DDA visits every grid cell a line segment crosses, in order, without
skipping any (unlike sampling the ray at fixed steps, which can miss a
cell at shallow angles). Amanatides & Woo, 1987.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from grid.addressing import world_to_global
from sensor.schedule import Level


@dataclass(frozen=True)
class RayHit:
    gi: int
    gj: int
    # True only for the single cell containing the ray's actual endpoint
    # -- Ticket #33's off-by-one warning: that cell is OCCUPIED, not FREE.
    is_terminal: bool


def dda_trace(x0: float, y0: float, x1: float, y1: float, c_l: float) -> List[RayHit]:
    """Amanatides-Woo DDA from (x0, y0) to (x1, y1) at cell size c_l, in
    GLOBAL cell-index space (not yet wrapped to toroidal storage -- that
    conversion is the caller's job at write time, per Ticket #10). Visits
    the exact sequence of cells the segment crosses; cells beyond the
    endpoint are never visited (the loop stops the instant it reaches the
    endpoint's cell).
    """
    gi0, gj0 = world_to_global(x0, y0, c_l)
    gi1, gj1 = world_to_global(x1, y1, c_l)

    dx, dy = x1 - x0, y1 - y0

    gi, gj = gi0, gj0
    hits = [RayHit(gi, gj, is_terminal=(gi == gi1 and gj == gj1))]

    if dx == 0.0 and dy == 0.0:
        return hits

    step_i = 1 if dx > 0 else -1
    step_j = 1 if dy > 0 else -1

    if dx != 0.0:
        t_delta_x = c_l / abs(dx)
        next_boundary_x = (gi0 + (1 if dx > 0 else 0)) * c_l
        t_max_x = (next_boundary_x - x0) / dx
    else:
        t_delta_x = math.inf
        t_max_x = math.inf

    if dy != 0.0:
        t_delta_y = c_l / abs(dy)
        next_boundary_y = (gj0 + (1 if dy > 0 else 0)) * c_l
        t_max_y = (next_boundary_y - y0) / dy
    else:
        t_delta_y = math.inf
        t_max_y = math.inf

    while not (gi == gi1 and gj == gj1):
        if t_max_x < t_max_y:
            gi += step_i
            t_max_x += t_delta_x
        else:
            # Ties (exact diagonal crossings) step j -- an arbitrary but
            # fixed, deterministic choice; either order visits a valid
            # 8-connected cell sequence for the segment.
            gj += step_j
            t_max_y += t_delta_y
        hits.append(RayHit(gi, gj, is_terminal=(gi == gi1 and gj == gj1)))

    return hits


def trace_beam(
    ego_x: float,
    ego_y: float,
    end_x: float,
    end_y: float,
    levels: List[Level],
) -> Dict[int, List[RayHit]]:
    """Coarse-first traversal of one beam's ground projection, per Ticket
    #33: the WHOLE beam at the clipmap's coarsest level, plus -- only for
    the portion of the beam inside the finest level's own Nyquist radius
    (the "fine ring") -- a second, finer pass at that level's cell size.
    Returns {level_index: [RayHit, ...]} for just the (at most two)
    levels actually traversed, not all four.
    """
    coarsest = levels[-1]
    finest = levels[0]

    result: Dict[int, List[RayHit]] = {
        coarsest.level: dda_trace(ego_x, ego_y, end_x, end_y, coarsest.cell_size_m)
    }

    dist = math.hypot(end_x - ego_x, end_y - ego_y)
    if dist <= 0.0:
        return result

    fine_dist = min(dist, finest.nyquist_radius_m)
    if fine_dist <= 0.0:
        return result

    t = fine_dist / dist
    fx = ego_x + (end_x - ego_x) * t
    fy = ego_y + (end_y - ego_y) * t
    fine_hits = dda_trace(ego_x, ego_y, fx, fy, finest.cell_size_m)

    if t < 1.0 and fine_hits:
        # The beam's actual return is outside the fine ring -- the cell
        # at the ring's cutoff is NOT where the beam terminated, so it
        # must not be marked OCCUPIED by the fine-level pass (Ticket #34
        # reads is_terminal to decide that). The coarse-level pass above
        # already carries the real terminal cell.
        fine_hits = [RayHit(h.gi, h.gj, is_terminal=False) for h in fine_hits]

    result[finest.level] = fine_hits
    return result
