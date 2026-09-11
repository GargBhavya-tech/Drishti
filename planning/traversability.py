"""
planning/traversability.py

Ticket #48 -- per-cell geometric derivatives a planner needs, from the
local 3x3 neighbourhood at the QUERIED level (Bible Part 14):

    slope      = arccos(n . z_hat)          -- n: plane fitted to the 3x3's h_mean values
    roughness  = sqrt(Var(z)_3x3)           -- of the 3x3's own h_mean values
    step       = max_k |h_max^cell - h_max^(k)|   -- k over the 8-connected neighbours
    clearance  -- from the multi-layer ceiling extraction (Ticket #21)

Watch out (Build Map's own words): STEP HEIGHT IS ONLY MEANINGFUL AT
L0/L1. A 15cm kerb is invisible in a 40cm cell -- reporting a confident
step-height number there would be reporting geometry the cell is
physically too coarse to resolve, not "no step". Beyond L1 this
function returns UNKNOWN (`None`), never a number. A neighbourhood with
too few observed points similarly cannot support a meaningful plane
normal or variance -- also `None`, never a confidently-wrong number
from a degenerate fit.

Clearance note: `grid.cell.decode_clearance` assumes BOTH its raw
arguments are absolute int16 (level-0-style) encodings -- true for
`h_ceil_min` (Ticket #21's ceiling plane is plain int16 at every level,
never migrated to Ticket #17's tile-relative `LeveledPlane`), but NOT
true for `h_max` at L1-L3, which Ticket #17 made a tile-relative int8.
Calling `decode_clearance` directly at L>=1 would silently misdecode
`h_max`. This module sidesteps that instead of extending
`decode_clearance` itself: `Clipmap.lookup()` already decodes
`h_max_m` correctly (level- and tile-base-aware) for whatever level a
query resolves at, so clearance here is computed as
`decode_h(h_ceil_min_raw, level=0) - cell.h_max_m` using that
already-correct value, rather than re-deriving a second decode path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from grid.addressing import flat_index, global_to_storage, global_to_world
from grid.cell import NO_CEILING_SENTINEL, decode_h
from grid.clipmap import Clipmap

STEP_HEIGHT_MAX_LEVEL = 1  # L0/L1 only -- see module docstring
MIN_NEIGHBOURHOOD_POINTS = 3  # fewer than 3 points cannot constrain a plane (need >= 3 for a well-posed fit)


@dataclass(frozen=True)
class TraversabilityResult:
    slope_deg: Optional[float]
    roughness_m: Optional[float]
    step_height_m: Optional[float]
    clearance_m: float


def _cell_center_world(cm: Clipmap, level: int, gi: int, gj: int) -> Tuple[float, float]:
    c_l = cm.levels[level].cell_size_m
    wx, wy = global_to_world(gi, gj, c_l)
    return wx + c_l / 2.0, wy + c_l / 2.0


def _neighbourhood_samples(cm: Clipmap, level: int, gi: int, gj: int):
    """(dx, dy, h_mean_m) for every OBSERVED cell in the 3x3
    neighbourhood around (gi, gj) at `level` (world-space offsets, cell
    centres), plus the centre cell's own h_max_m and the neighbours'
    h_max_m list (for the step-height max)."""
    c_l = cm.levels[level].cell_size_m
    points: List[Tuple[float, float, float]] = []
    center_h_max: Optional[float] = None
    neighbour_h_max: List[float] = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ni, nj = gi + di, gj + dj
            wx, wy = _cell_center_world(cm, level, ni, nj)
            found_level, cell = cm.lookup(wx, wy)
            if found_level != level or not cell.observed:
                continue
            points.append((di * c_l, dj * c_l, cell.h_mean_m))
            if di == 0 and dj == 0:
                center_h_max = cell.h_max_m
            else:
                neighbour_h_max.append(cell.h_max_m)
    return points, center_h_max, neighbour_h_max


def _fit_slope_and_roughness(points: List[Tuple[float, float, float]]) -> Tuple[Optional[float], Optional[float]]:
    if len(points) < MIN_NEIGHBOURHOOD_POINTS:
        return None, None
    pts = np.array(points, dtype=np.float64)
    z = pts[:, 2]
    roughness_m = float(np.sqrt(np.var(z)))

    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b, _c = coeffs
    normal = np.array([-a, -b, 1.0])
    norm = np.linalg.norm(normal)
    if norm == 0.0:
        return None, roughness_m
    normal /= norm
    cos_slope = float(np.clip(normal[2], -1.0, 1.0))
    slope_deg = math.degrees(math.acos(cos_slope))
    return slope_deg, roughness_m


def _read_clearance_m(cm: Clipmap, level: int, gi: int, gj: int, cell) -> float:
    """+inf when there is no known ceiling (NO_CEILING_SENTINEL) --
    Ticket #21's own invariant: "must decode to +inf clearance, never
    0" -- a cell with no evidence of anything above it must never read
    as zero clearance (which would be LETHAL)."""
    oi, oj = cm.origin_i[level], cm.origin_j[level]
    if not (oi <= gi < oi + cm.N and oj <= gj < oj + cm.N):
        return float("inf")
    si, sj = global_to_storage(gi, gj, cm.N)
    flat = flat_index(si, sj, cm.N)
    h_ceil_min_raw = int(cm.h_ceil_min[level, flat])
    if h_ceil_min_raw == int(NO_CEILING_SENTINEL) or cell.h_max_m is None:
        return float("inf")
    return decode_h(h_ceil_min_raw, level=0) - cell.h_max_m


def compute_traversability(cm: Clipmap, level: int, gi: int, gj: int) -> TraversabilityResult:
    """The Ticket #48 per-cell derivatives at global (gi, gj), queried
    at `level`. Slope/roughness/step are `None` (UNKNOWN) exactly when
    the geometry genuinely cannot support them (see module docstring);
    clearance always returns a real number (+inf standing in for "no
    known ceiling", never `None`, since Ticket #21's own layer already
    has a defined answer for that case)."""
    points, center_h_max, neighbour_h_max = _neighbourhood_samples(cm, level, gi, gj)
    slope_deg, roughness_m = _fit_slope_and_roughness(points)

    if level > STEP_HEIGHT_MAX_LEVEL or center_h_max is None or not neighbour_h_max:
        step_height_m = None
    else:
        step_height_m = float(max(abs(center_h_max - h) for h in neighbour_h_max))

    wx, wy = _cell_center_world(cm, level, gi, gj)
    _, center_cell = cm.lookup(wx, wy)
    clearance_m = _read_clearance_m(cm, level, gi, gj, center_cell)

    return TraversabilityResult(
        slope_deg=slope_deg, roughness_m=roughness_m, step_height_m=step_height_m, clearance_m=clearance_m
    )
