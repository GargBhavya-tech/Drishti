"""
temporal/static_layer.py

Ticket #44 -- static accumulation: the temporal counterpart to Ticket
#18's `grid.scatter.scatter()`. `scatter()` deliberately computes each
frame's count/h_mean/h_m2 FRESH and overwrites whatever a cell already
held (its own "Scope note": correctly merging many frames needs
Welford/Chan, and defers that to this ticket). h_max/h_min are the one
exception -- `scatter()` already accumulates those correctly via
`scatter_reduce_(..., include_self=True)` against each cell's existing
stored value. This module reproduces that same elementwise-max/min-
against-existing-decoded-value behaviour explicitly (rather than
calling `grid.scatter.scatter()` itself, which would overwrite
count/mean/variance before this module got a chance to merge them),
so a caller uses `StaticLayerAccumulator.accumulate_frame()` in place
of `grid.scatter.scatter()`, not in addition to it.

Bible Part 12.3: confidence decays as exp(-(t - t_cell)/tau), tau=10s
-- the SAME constant `planning.conservatism` uses for its own age-based
decay, imported from there rather than redefined so the two decay
curves cannot silently drift apart. Accumulation itself is capped at
~3s of history (Build Map's own words: "a crisp 3-second map beats a
blurry 60-second one" -- odometry drift smears a world-anchored map
over long horizons). A cell whose last update is older than the cap is
treated as having no prior evidence: this frame starts it fresh,
exactly as if the cell had never been observed before.

Watch out (Build Map's own words): naive incremental mean, or the
numerically unstable E[x^2] - E[x]^2 form, silently loses variance
across many merges. `chan_merge`/`chan_merge_batch` use Chan, Golub &
LeVeque's exact parallel-axis M2 form throughout, and reproducing a
direct recomputation over the union of all points (to floating-point
tolerance) IS this ticket's own acceptance test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch

from grid.addressing import flat_index_batch, global_to_storage_batch, world_to_global_batch
from grid.cell import OBS_OCCUPIED, OBS_UNOBSERVED, TILE_SIZE, decode_h_batch, encode_h_batch, expected_stamp_batch, tile_index_batch
from grid.clipmap import Clipmap
from grid.scatter import COUNT_MAX, _VARIANCE_QUANTUM_M2
from planning.conservatism import CONFIDENCE_DECAY_TAU_S

ACCUMULATION_WINDOW_S = 3.0  # Build Map #44: "a crisp 3-second map beats a blurry 60-second one"


@dataclass(frozen=True)
class GroupStats:
    """One group's (count, mean, M2). M2 is Chan/Welford's sum of
    squared deviations from the mean, NOT variance (variance = M2/n) --
    kept in this form because M2, not variance, is what combines
    exactly under Chan's formula."""

    n: int
    mean: float
    m2: float


def chan_merge(a: GroupStats, b: GroupStats) -> GroupStats:
    """Combine two groups' statistics into the statistics of their
    union -- Chan, Golub & LeVeque (1979)'s parallel-axis formula.
    Exact: merging reproduces a direct recomputation over the union of
    all points to floating-point tolerance, unlike the naive
    E[x^2]-E[x]^2 form (Build Map #44's explicit warning)."""
    if a.n == 0:
        return b
    if b.n == 0:
        return a
    n = a.n + b.n
    delta = b.mean - a.mean
    mean = a.mean + delta * (b.n / n)
    m2 = a.m2 + b.m2 + delta * delta * (a.n * b.n / n)
    return GroupStats(n=n, mean=mean, m2=m2)


def chan_merge_batch(
    n_a: np.ndarray,
    mean_a: np.ndarray,
    m2_a: np.ndarray,
    n_b: np.ndarray,
    mean_b: np.ndarray,
    m2_b: np.ndarray,
):
    """Vectorised `chan_merge` over many independent (per-cell) groups
    at once -- matching `grid.scatter`'s own "no Python loop" budget.
    Returns (n, mean, m2) arrays. Empty-group edge cases fall back
    EXACTLY to the non-empty side (mirroring `chan_merge`'s scalar
    early returns) rather than relying on the general formula agreeing
    with that when the empty side happens to carry mean=0/m2=0."""
    n_a = n_a.astype(np.float64)
    n_b = n_b.astype(np.float64)
    n = n_a + n_b
    safe_n = np.where(n > 0, n, 1.0)  # avoid 0/0 when both groups are empty
    delta = mean_b - mean_a
    mean = mean_a + delta * (n_b / safe_n)
    m2 = m2_a + m2_b + delta * delta * (n_a * n_b / safe_n)

    a_empty = n_a == 0
    b_empty = n_b == 0
    mean = np.where(a_empty, mean_b, np.where(b_empty, mean_a, mean))
    m2 = np.where(a_empty, m2_b, np.where(b_empty, m2_a, m2))
    return n, mean, m2


def confidence_from_age(age_s: Union[np.ndarray, float], tau_s: float = CONFIDENCE_DECAY_TAU_S):
    """Bible Part 12.3: exp(-(t - t_cell)/tau) -- 1.0 at age 0, decaying
    toward 0 as a cell goes stale. A pure function of elapsed time; the
    map need not store this, only `t_cell` (this module's own last-
    update bookkeeping) and the query time."""
    return np.exp(-np.asarray(age_s, dtype=np.float64) / tau_s)


class StaticLayerAccumulator:
    """Owns the one piece of per-cell state `grid.clipmap.Clipmap`
    itself does not track: LAST-UPDATE TIME, needed for the ~3s
    accumulation window and Part 12.3's confidence decay. Kept OUTSIDE
    Clipmap (mirroring how Ticket #17 added `tile_base_h` as its own
    plane only once a ticket actually needed it) rather than growing
    Clipmap's core schema for a feature only the temporal layer
    consumes.

    Correctness note: this accumulator's OWN staleness bookkeeping is
    NEVER trusted alone to mean "this cell's existing stats are still
    valid" -- `cm.flags` is the ground truth for whether a storage slot
    currently holds real data for THIS global cell (Ticket #13's
    clear-on-scroll may have reset this exact storage slot for a
    DIFFERENT global cell since this accumulator last touched it, and
    this accumulator has no independent way to detect that -- the same
    class of risk Ticket #14's stamp check exists to catch elsewhere).
    Every merge decision below checks `cm.flags != OBS_UNOBSERVED`
    first; the time-window check only decides whether otherwise-valid,
    same-cell evidence is still trusted or should be treated as if it
    had never been observed.
    """

    def __init__(self, cm: Clipmap):
        self._cm = cm
        self._last_update_s: List[np.ndarray] = [
            np.full(cm.N * cm.N, -np.inf, dtype=np.float64) for _ in cm.levels
        ]

    def accumulate_frame(self, xyz_world: np.ndarray, t_s: float) -> None:
        """One frame's points -> Chan-merged into the clipmap's running
        per-cell (count, h_mean, h_m2), with h_max/h_min elementwise-
        merged against their existing decoded values -- at every level
        whose current window contains them. Vectorised per level, no
        Python loop over points (Ticket #18's own performance
        requirement, carried over here)."""
        if xyz_world.shape[0] == 0:
            return
        cm = self._cm
        xyz = torch.from_numpy(np.ascontiguousarray(xyz_world, dtype=np.float64))
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        N = cm.N
        n_cells = N * N

        for level, lvl in enumerate(cm.levels):
            c_l = lvl.cell_size_m
            gi = world_to_global_batch(x, c_l)
            gj = world_to_global_batch(y, c_l)

            oi, oj = cm.origin_i[level], cm.origin_j[level]
            in_window = (gi >= oi) & (gi < oi + N) & (gj >= oj) & (gj < oj + N)
            if not torch.any(in_window):
                continue

            gi_in, gj_in, z_in = gi[in_window], gj[in_window], z[in_window]
            si = global_to_storage_batch(gi_in, N)
            sj = global_to_storage_batch(gj_in, N)
            flat = flat_index_batch(si, sj, N)

            h_max_scratch = torch.full((n_cells,), float("-inf"), dtype=torch.float64)
            h_min_scratch = torch.full((n_cells,), float("inf"), dtype=torch.float64)
            count_scratch = torch.zeros(n_cells, dtype=torch.int64)
            h_sum_scratch = torch.zeros(n_cells, dtype=torch.float64)
            h_sq_scratch = torch.zeros(n_cells, dtype=torch.float64)

            h_max_scratch.scatter_reduce_(0, flat, z_in, reduce="amax", include_self=True)
            h_min_scratch.scatter_reduce_(0, flat, z_in, reduce="amin", include_self=True)
            count_scratch.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int64))
            h_sum_scratch.scatter_add_(0, flat, z_in)
            h_sq_scratch.scatter_add_(0, flat, z_in * z_in)

            touched = torch.nonzero(count_scratch > 0, as_tuple=True)[0]
            if touched.numel() == 0:
                continue

            counts_f = count_scratch[touched].to(torch.float64)
            frame_n = count_scratch[touched].numpy()
            frame_mean = (h_sum_scratch[touched] / counts_f).numpy()
            # Single-frame batch: the naive E[x^2]-E[x]^2 form is fine
            # HERE (Ticket #18's own docstring makes the same point) --
            # it is only unsafe for accumulating many frames on top of
            # each other, which is exactly what chan_merge_batch below
            # does instead of repeating this formula across frames.
            frame_m2 = (h_sq_scratch[touched] - h_sum_scratch[touched] ** 2 / counts_f).numpy()
            frame_m2 = np.clip(frame_m2, 0.0, None)  # guard tiny fp noise
            frame_h_max = h_max_scratch[touched].numpy()
            frame_h_min = h_min_scratch[touched].numpy()

            si_t = touched % N
            sj_t = touched // N
            gi_t = (oi + torch.remainder(si_t - oi, N)).numpy()
            gj_t = (oj + torch.remainder(sj_t - oj, N)).numpy()
            flat_t = touched.numpy()

            self._merge_into_clipmap(
                level, flat_t, gi_t, gj_t, frame_n, frame_mean, frame_m2, frame_h_max, frame_h_min, t_s
            )

    def _merge_into_clipmap(
        self, level, flat_idx, gi, gj, frame_n, frame_mean, frame_m2, frame_h_max, frame_h_min, t_s
    ) -> None:
        cm = self._cm
        last_t = self._last_update_s[level][flat_idx]
        stale = (t_s - last_t) > ACCUMULATION_WINDOW_S  # also True when last_t == -inf
        has_existing = (cm.flags[level, flat_idx] != OBS_UNOBSERVED) & (~stale)

        existing_count_raw = cm.count[level, flat_idx].astype(np.int64)
        existing_count = np.where(has_existing, existing_count_raw, 0).astype(np.float64)

        existing_mean = self._decode_level_aware(level, "h_mean", flat_idx)
        existing_h_max = self._decode_level_aware(level, "h_max", flat_idx)
        existing_h_min = self._decode_level_aware(level, "h_min", flat_idx)
        existing_variance = cm.h_m2[level, flat_idx].astype(np.float64) * _VARIANCE_QUANTUM_M2
        existing_m2 = np.where(has_existing, existing_variance * existing_count_raw.astype(np.float64), 0.0)
        existing_mean = np.where(has_existing, existing_mean, 0.0)

        merged_n, merged_mean, merged_m2 = chan_merge_batch(
            existing_count, existing_mean, existing_m2,
            frame_n.astype(np.float64), frame_mean, frame_m2,
        )
        merged_variance = np.where(merged_n > 0, merged_m2 / np.maximum(merged_n, 1.0), 0.0)
        merged_count = np.minimum(merged_n, COUNT_MAX).astype(np.uint16)
        merged_h_max = np.where(has_existing, np.maximum(existing_h_max, frame_h_max), frame_h_max)
        merged_h_min = np.where(has_existing, np.minimum(existing_h_min, frame_h_min), frame_h_min)

        if level == 0:
            cm.h_mean[level, flat_idx] = encode_h_batch(merged_mean, level=0, base_elevation_m=0.0)
            cm.h_max[level, flat_idx] = encode_h_batch(merged_h_max, level=0, base_elevation_m=0.0)
            cm.h_min[level, flat_idx] = encode_h_batch(merged_h_min, level=0, base_elevation_m=0.0)
        else:
            # Resolve the tile base ONCE and reuse it for all three
            # encodes below -- calling get_or_create_tile_bases per
            # plane (as an earlier draft of this module did) is a real
            # bug: before flags are written (below), each call would
            # independently see the tile as "not yet live" and mint a
            # FRESH base from whichever plane's own values it was
            # given, leaving h_mean/h_max/h_min encoded relative to
            # THREE DIFFERENT bases with only the last one actually
            # stored -- exactly the kind of inconsistency Ticket #17's
            # own module docstring warns "looks like it works" until
            # two independently-based reads disagree.
            base_m = cm.get_or_create_tile_bases(level, flat_idx, representative_z_m=merged_mean)
            cm.h_mean[level, flat_idx] = encode_h_batch(merged_mean, level=level, base_elevation_m=base_m)
            cm.h_max[level, flat_idx] = encode_h_batch(merged_h_max, level=level, base_elevation_m=base_m)
            cm.h_min[level, flat_idx] = encode_h_batch(merged_h_min, level=level, base_elevation_m=base_m)

        cm.h_m2[level, flat_idx] = np.clip(
            np.round(merged_variance / _VARIANCE_QUANTUM_M2), 0, 65535
        ).astype(np.uint16)
        cm.count[level, flat_idx] = merged_count
        cm.flags[level, flat_idx] = OBS_OCCUPIED
        cm.stamp[level, flat_idx] = expected_stamp_batch(gi, gj, cm.N)

        self._last_update_s[level][flat_idx] = t_s

    def _decode_level_aware(self, level: int, plane_name: str, flat_idx: np.ndarray) -> np.ndarray:
        cm = self._cm
        plane = getattr(cm, plane_name)
        raw = plane[level, flat_idx]
        if level == 0:
            return decode_h_batch(raw, level=0, base_elevation_m=0.0)
        N = cm.N
        si = flat_idx % N
        sj = flat_idx // N
        tile_idx = tile_index_batch(si, sj, cm._n_tiles_axis, TILE_SIZE)
        bases = decode_h_batch(cm.tile_base_h[level, tile_idx], level=0, base_elevation_m=0.0)
        return decode_h_batch(raw, level=level, base_elevation_m=bases)
