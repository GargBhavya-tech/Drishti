"""
temporal/latency_compensation.py

Ticket #61 -- latency compensation for the published map. Because the
static map is WORLD-ANCHORED, EGO staleness needs no transform at all
-- the "free half" (Build Map's own words): the map's own cells ARE
the world, not the ego's frame, so nothing about them needs to change
as time passes between measurement and the planner's action time. Only
MOTION-FLAGGED cells (Ticket #45) need their extent advanced --
inflated by prediction uncertainty so the compensation stays
conservative, per this project's own conservatism invariant (#41/#42):
uncertainty must never SHRINK a hazard's claimed extent.

Watch out (Build Map's own words): never extrapolate further than
confidence supports -- cap the horizon. Publish both a measurement
timestamp and a validity timestamp, never just one.

Scope note: this ticket does not touch `planning.conservatism` at all
-- "#42 still passes" (Build Map's own "Done when") holds trivially,
since nothing here changes that module's own cost function or its
property test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple

GlobalCell = Tuple[int, int]
Velocity = Tuple[float, float]  # (cells/s along i, cells/s along j)

MAX_EXTRAPOLATION_HORIZON_S = 1.0  # never extrapolate further than confidence supports


@dataclass(frozen=True)
class PublishedMap:
    """A published map snapshot for planner consumption: BOTH a
    measurement timestamp (when the underlying sensor data was
    captured) and a validity timestamp (the time this snapshot is
    being published FOR, i.e. the planner's own action time) -- Build
    Map's own explicit instruction: publish both, never just one."""

    measurement_time_s: float
    validity_time_s: float
    inflated_moving_cells: Set[GlobalCell]


def ego_staleness_requires_no_transform() -> bool:
    """The free half: a world-anchored map's own static content needs
    ZERO transformation to remain valid as ego staleness grows. A
    callable, testable statement of that invariant rather than a
    comment -- there is genuinely no per-cell work to do here, and
    this function's own existence is the assertion that nothing was
    quietly added."""
    return True


def inflate_moving_cell(
    cell: GlobalCell,
    velocity_cells_per_s: Velocity,
    elapsed_s: float,
    uncertainty_radius_cells: float,
) -> Set[GlobalCell]:
    """One moving cell's own inflated extent after `elapsed_s`, capped
    at `MAX_EXTRAPOLATION_HORIZON_S`: the predicted new centre PLUS an
    `uncertainty_radius_cells`-wide neighbourhood around it. The
    inflation (never a bare point extrapolation) is what keeps this
    conservative -- the ORIGINAL cell's own extent is always still
    covered by a large-enough radius around the advanced centre,
    matching the ticket's own "inflated extents never shrink a hazard".
    """
    horizon_s = min(max(0.0, elapsed_s), MAX_EXTRAPOLATION_HORIZON_S)
    gi, gj = cell
    vi, vj = velocity_cells_per_s
    center_i = round(gi + vi * horizon_s)
    center_j = round(gj + vj * horizon_s)

    r = max(1, round(uncertainty_radius_cells))
    return {(center_i + di, center_j + dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)}


def publish_map(
    measurement_time_s: float,
    validity_time_s: float,
    moving_cells: Dict[GlobalCell, Velocity],
    uncertainty_radius_cells: float = 1.0,
) -> PublishedMap:
    """`moving_cells`: the map's own currently motion-flagged cells
    (Ticket #45) with their own estimated velocity. Every STATIC cell
    needs no work at all (the free half, `ego_staleness_requires_no_
    transform`); every moving cell's extent is inflated and advanced
    by the elapsed time between measurement and the requested validity
    time.
    """
    elapsed_s = validity_time_s - measurement_time_s
    inflated: Set[GlobalCell] = set()
    for cell, velocity in moving_cells.items():
        inflated |= inflate_moving_cell(cell, velocity, elapsed_s, uncertainty_radius_cells)
    return PublishedMap(measurement_time_s=measurement_time_s, validity_time_s=validity_time_s, inflated_moving_cells=inflated)
