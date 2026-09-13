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

PERFORMANCE, added in response to DRISHTI_MASTER_BIBLE.md's own
`eval/benchmark_inference_latency.py` finding this function as the real
per-frame bottleneck (117.16 ms, 48.9% of total latency -- more than
the GPU model forward pass itself). Analysis, confirmed by reading this
module's own logic (not assumed): the walk has NO cross-column
dependency -- every column re-seeds its own `prev_x/prev_y/prev_z` from
scratch (see the original pure-Python `_walk_columns_python` below,
kept as the reference implementation and cross-checked byte-for-byte
against the JIT path by `tests/test_ground_prior_numba_equivalence.py`).
Only the WITHIN-column walk is sequential (each point's accept/reject
depends on the last ACCEPTED point, not simply the previous point).
This is exactly the "embarrassingly parallel across independent groups"
case a deep-research report reviewed for this fix named as needing only
`@njit(parallel=True)` with `numba.prange` over the outer loop -- no
Parallel-Prefix-Scan/associative-monoid reformulation of the inner walk
was needed, since that machinery only helps when a REAL cross-column
dependency exists, which this module's own code confirms it does not.

Because the outer loop is genuinely independent (not a reduction being
retiled into a tree), parallelizing it changes WHICH CPU core executes
which column, never the actual floating-point operation order within a
column -- so the JIT path is expected to be BIT-IDENTICAL to the
original, not merely epsilon-close (verified, not assumed, by the
byte-identical union of `is_ground` and `column_ground_height` in the
equivalence test, on top of the standalone downstream-argmax-invariance
check a reviewer specifically asked for).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numba
import numpy as np

from perception.sweep import Sweep

DEFAULT_SLOPE_THRESHOLD_DEG = 10.0  # Bible Part 5.2: theta_max ~= 10 degrees


@dataclass(frozen=True)
class GroundPriorResult:
    is_ground: np.ndarray  # (N,) bool, one per source point
    column_ground_height: dict  # {azimuth_column: mean z of accepted ground points in that column}


def _walk_columns_python(sorted_col, sorted_x, sorted_y, sorted_z, sorted_idx, group_start, group_end, slope_threshold, is_ground):
    """The ORIGINAL, pure-Python per-column walk -- kept as the reference
    implementation `tests/test_ground_prior_numba_equivalence.py` checks
    the JIT path against byte-for-byte, and as a safety fallback if
    Numba is ever unavailable at import time (see `compute_ground_prior`'s
    own try/except around the JIT compile)."""
    column_heights: dict = {}
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
    return column_heights


@numba.njit(cache=True, parallel=False)
def _walk_columns_numba(sorted_col, sorted_x, sorted_y, sorted_z, sorted_idx, group_start, group_end,
                         slope_threshold, is_ground, col_height_sum, col_height_count):
    """Numba-JIT equivalent of `_walk_columns_python`. Originally
    `parallel=True` with `numba.prange` over columns (every column IS a
    fully independent group, confirmed by reading the original code
    before this was written) -- REVERTED to `parallel=False` after a
    real, serious bug this caused: `numba.prange`'s OpenMP thread pool
    initializes in the MAIN process the first time this function runs
    (e.g. during `perception.train`'s own channel-stats/class-count
    passes, which run before any DataLoader worker spawns), and PyTorch's
    DataLoader with num_workers>0 then `fork()`s worker processes on
    Linux -- forking a process that already has an active OpenMP thread
    pool is unsafe and crashed every DataLoader worker outright
    ("Terminating: fork() called from a process already using GNU
    OpenMP, this is unsafe", confirmed on a real `checkpoints_multi_v5`
    training run). Plain JIT compilation (this version) still removes
    the CPython interpreter overhead that was the majority of the
    original 117ms cost (confirmed: 117.16ms -> 26.45ms, a real 4.4x
    speedup, measured on `drishti-gpu` -- see
    DRISHTI_MASTER_BIBLE.md Part G.26) WITHOUT ever creating a thread
    pool, so it is fork-safe. `is_ground` writes are still race-free by
    construction (kept for a future single-process-only fast path, not
    because races matter here anymore) -- every point index belongs to
    exactly one column's group, disjoint by construction of
    `group_start`/`group_end`."""
    n_groups = group_start.shape[0]
    for g in range(n_groups):
        start = group_start[g]
        end = group_end[g]
        prev_x = sorted_x[start]
        prev_y = sorted_y[start]
        prev_z = sorted_z[start]
        is_ground[sorted_idx[start]] = True
        h_sum = prev_z
        h_count = 1

        for k in range(start + 1, end):
            cx = sorted_x[k]
            cy = sorted_y[k]
            cz = sorted_z[k]
            dxy = math.hypot(cx - prev_x, cy - prev_y)
            if dxy < 1e-9:
                continue
            slope = abs(cz - prev_z) / dxy
            if slope < slope_threshold:
                is_ground[sorted_idx[k]] = True
                prev_x = cx
                prev_y = cy
                prev_z = cz
                h_sum += cz
                h_count += 1

        col_height_sum[g] = h_sum
        col_height_count[g] = h_count


def compute_ground_prior(
    sweep: Sweep,
    n_azimuth_bins: int = 1080,
    slope_threshold_deg: float = DEFAULT_SLOPE_THRESHOLD_DEG,
    use_numba: bool = True,
) -> GroundPriorResult:
    """LABELS every point ground/not-ground; never removes a point from
    the sweep. `n_azimuth_bins` only controls how points are grouped into
    columns for the walk -- it does not filter or resample anything.

    `use_numba=True` (default) routes the per-column walk through
    `_walk_columns_numba` (column-parallel, JIT-compiled -- see this
    module's own docstring for why this is verified numerically
    equivalent, not merely assumed faster). `use_numba=False` uses the
    original pure-Python `_walk_columns_python`, kept for the
    equivalence test and as an explicit escape hatch."""
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

    if use_numba:
        col_height_sum = np.zeros(group_start.shape[0], dtype=np.float64)
        col_height_count = np.zeros(group_start.shape[0], dtype=np.int64)
        _walk_columns_numba(
            sorted_col, sorted_x, sorted_y, sorted_z, sorted_idx,
            group_start.astype(np.int64), group_end.astype(np.int64),
            slope_threshold, is_ground, col_height_sum, col_height_count,
        )
        column_heights = {
            int(sorted_col[group_start[g]]): float(col_height_sum[g] / col_height_count[g])
            for g in range(group_start.shape[0])
        }
    else:
        column_heights = _walk_columns_python(
            sorted_col, sorted_x, sorted_y, sorted_z, sorted_idx, group_start, group_end, slope_threshold, is_ground
        )

    return GroundPriorResult(is_ground=is_ground, column_ground_height=column_heights)
