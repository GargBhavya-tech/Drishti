"""
perception/ground_prior.py

Ticket #26 -- ground prior: column-wise incremental walk. Bible Part 5.2
calls this "the single most important correction in v2": it LABELS
points as ground, it never STRIPS them -- terrain must reach the
elevation map, because terrain IS the elevation map.

Per azimuth column, walk OUTWARD BY RANGE (not ring index -- Ticket #26
"Watch out": a range image's row order does not always match ascending
ground-return range, especially near a slope) accepting a point as
ground when the slope from the last ACCEPTED ground point in that column
is below threshold. Never fits a plane -- Bible Part 5.2: a plane fit
assumes flatness over its whole region and misclassifies an entire
sector on a slope or crest; the incremental walk only ever compares to
its immediate predecessor, so it tracks terrain instead of assuming it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from perception.sweep import Sweep

DEFAULT_SLOPE_THRESHOLD_DEG = 10.0  # Bible Part 5.2: theta_max ~= 10 degrees


@dataclass(frozen=True)
class GroundPriorResult:
    is_ground: np.ndarray  # (N,) bool, one per source point
    column_ground_height: dict  # {azimuth_column: mean z of accepted ground points in that column}


def compute_ground_prior(
    sweep: Sweep,
    n_azimuth_bins: int = 1080,
    slope_threshold_deg: float = DEFAULT_SLOPE_THRESHOLD_DEG,
) -> GroundPriorResult:
    """LABELS every point ground/not-ground; never removes a point from
    the sweep. `n_azimuth_bins` only controls how points are grouped into
    columns for the walk -- it does not filter or resample anything.
    """
    n = sweep.xyz.shape[0]
    is_ground = np.zeros(n, dtype=bool)
    if n == 0:
        return GroundPriorResult(is_ground=is_ground, column_ground_height={})

    x, y, z = sweep.xyz[:, 0].astype(np.float64), sweep.xyz[:, 1].astype(np.float64), sweep.xyz[:, 2].astype(np.float64)
    r_xy = np.sqrt(x * x + y * y)  # horizontal range -- the "outward" axis a column is walked along
    real_point = r_xy > 1e-6

    azimuth = np.arctan2(y, x)
    col = np.floor(0.5 * (1.0 - azimuth / np.pi) * n_azimuth_bins).astype(np.int64)
    col = np.clip(col, 0, n_azimuth_bins - 1)

    slope_threshold = math.tan(math.radians(slope_threshold_deg))

    column_heights: dict = {}
    valid_idx = np.nonzero(real_point)[0]
    if valid_idx.size == 0:
        return GroundPriorResult(is_ground=is_ground, column_ground_height={})

    # Sort within each column by outward (horizontal) range ascending --
    # walking "outward from the sensor" -- via a stable (col, r_xy) sort,
    # NOT ring-index order.
    order = np.lexsort((r_xy[valid_idx], col[valid_idx]))
    sorted_idx = valid_idx[order]
    sorted_col = col[sorted_idx]
    sorted_x, sorted_y, sorted_z = x[sorted_idx], y[sorted_idx], z[sorted_idx]

    group_start = np.flatnonzero(np.r_[True, sorted_col[1:] != sorted_col[:-1]])
    group_end = np.r_[group_start[1:], sorted_col.shape[0]]

    for start, end in zip(group_start, group_end):
        prev_x, prev_y, prev_z = sorted_x[start], sorted_y[start], sorted_z[start]
        is_ground[sorted_idx[start]] = True  # seed: first (closest) return in the column
        ground_zs = [prev_z]

        for k in range(start + 1, end):
            cx, cy, cz = sorted_x[k], sorted_y[k], sorted_z[k]
            dxy = math.hypot(cx - prev_x, cy - prev_y)
            if dxy < 1e-9:
                continue  # coincident points -- nothing to compare
            slope = abs(cz - prev_z) / dxy
            if slope < slope_threshold:
                is_ground[sorted_idx[k]] = True
                prev_x, prev_y, prev_z = cx, cy, cz
                ground_zs.append(cz)
            # else: rejected -- prev_* stays at the last ACCEPTED ground
            # point, not this rejected one, so one bad point can't drag
            # the reference off terrain.

        column_heights[int(sorted_col[start])] = float(np.mean(ground_zs))

    return GroundPriorResult(is_ground=is_ground, column_ground_height=column_heights)
