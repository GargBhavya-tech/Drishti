"""
grid/layers.py

Ticket #21 -- multi-layer extraction: ground / gap / ceiling from a
cell's 8-bin height histogram. The machine-checked form of Claim 1 (Bible
Part 9.1-9.2): a single height value cannot represent "drivable road
under a bridge" -- max-height reports an impassable wall, min-height
drives into the branch, mean-height invents a wall at head height.

Once a ceiling is extracted, h_min/h_max (Ticket #18) are re-narrowed to
the GROUND layer's own range (for a single-layer cell they are already
exactly that, since ground is the only layer); h_ceil_min/h_ceil_max
(Ticket #21, this module) hold the ceiling layer's range, or the
NO_CEILING sentinel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from grid.addressing import flat_index_batch, global_to_storage_batch, world_to_global_batch
from grid.cell import H_QUANTUM_M, NO_CEILING_SENTINEL
from grid.clipmap import Clipmap
from grid.histogram import HistogramSpec, histogram_spec
from sensor.vehicle_config import VehicleConfig


@dataclass(frozen=True)
class LayerBins:
    ground_lo: int
    ground_hi: int  # inclusive, last bin of the contiguous occupied run starting at ground_lo
    ceiling_bin: Optional[int]  # None = NO_CEILING


def extract_layer_bins(
    histogram_row: np.ndarray,
    spec: HistogramSpec,
    min_bin_count_for_ceiling: int = 1,
) -> LayerBins:
    """Pure function over one cell's 8-bin histogram (Bible Part 9.2):

    1. Ground layer = the lowest occupied bin and its immediate
       CONTIGUOUS occupied neighbours.
    2. Ceiling = the first genuinely-occupied bin ABOVE the ground layer,
       whatever the gap in between turns out to be -- a branch 3 bins
       above ground IS the ceiling, with a correspondingly small
       clearance, not "no ceiling" just because the gap is short. This
       module reports the NUMBER; whether that number is enough for a
       given vehicle (DRIVABLE vs OVERHANG) is a comparison against
       `vehicle.min_clearance_m` made by the caller/traversability layer
       (Ticket #48), not decided here.

    `min_bin_count_for_ceiling`: a candidate ceiling bin with fewer
    points than this is treated as NOT occupied for ceiling-detection
    purposes only -- Bible Part 9.3's "ceiling extraction ignores layers
    ... whose return density is below threshold" (the sparse-canopy
    case: a few scattered returns shouldn't block an otherwise-clear
    path -- the scan simply continues past them). Ground-bin membership
    is never filtered this way -- Part 5.2: a false negative on ground
    is strictly worse than a false positive.
    """
    n_bins = spec.n_bins
    occupied = histogram_row > 0
    occupied_idx = np.nonzero(occupied)[0]
    if occupied_idx.size == 0:
        raise ValueError(
            "histogram_row has no occupied bins -- caller must handle the "
            "no-returns-at-all case separately (that's UNOBSERVED, not a layers question)"
        )

    ground_lo = int(occupied_idx[0])
    ground_hi = ground_lo
    while ground_hi + 1 < n_bins and occupied[ground_hi + 1]:
        ground_hi += 1

    ceiling_bin: Optional[int] = None
    for i in range(ground_hi + 1, n_bins):
        if histogram_row[i] >= min_bin_count_for_ceiling:
            ceiling_bin = i
            break

    return LayerBins(ground_lo=ground_lo, ground_hi=ground_hi, ceiling_bin=ceiling_bin)


def scatter_layers(
    cm: Clipmap,
    xyz_world: np.ndarray,
    z_ground: np.ndarray,
    vehicle: VehicleConfig,
    min_bin_count_for_ceiling: int = 1,
) -> None:
    """End-to-end: bin points, extract ground/ceiling bins per touched
    cell, then re-derive the PRECISE (non-bin-quantised) ground-layer max
    and ceiling-layer min from the raw points -- needed because a bin
    only says "somewhere in this 0.5625 m band", and clearance is a
    continuous number (Build Map Ticket #21's own test expects 4.15 m and
    1.88 m exactly, not a bin-rounded approximation).

    Requires `scatter()` (Ticket #18) to have already run on the same
    points, so h_min starts as the ground layer's floor. This function
    narrows h_max to the ground layer's ceiling (instead of the whole
    cell's) and writes h_ceil_min/h_ceil_max for the layer above the gap,
    if any.
    """
    if xyz_world.shape[0] == 0:
        return

    spec = histogram_spec(vehicle)
    xyz = torch.from_numpy(np.ascontiguousarray(xyz_world, dtype=np.float64))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    zg = torch.from_numpy(np.ascontiguousarray(z_ground, dtype=np.float64))
    z_rel = z - zg
    bin_idx = torch.clamp(
        torch.floor((z_rel - spec.lower_offset_m) / spec.bin_width_m).long(), 0, spec.n_bins - 1
    )

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

        gi_in = gi[in_window]
        gj_in = gj[in_window]
        z_in = z[in_window]
        bin_in = bin_idx[in_window]
        si = global_to_storage_batch(gi_in, N)
        sj = global_to_storage_batch(gj_in, N)
        flat = flat_index_batch(si, sj, N)

        # Per-cell histogram for this batch (same construction as
        # histogram.py, kept local since we also need bin_idx per point
        # below for the masked ground/ceiling reduction).
        combined = flat * spec.n_bins + bin_in
        hist_flat = torch.zeros(n_cells * spec.n_bins, dtype=torch.int64)
        hist_flat.scatter_add_(0, combined, torch.ones_like(combined))
        hist = hist_flat.view(n_cells, spec.n_bins).numpy()

        touched = torch.nonzero(torch.from_numpy(hist).sum(dim=1) > 0, as_tuple=True)[0]
        if touched.numel() == 0:
            continue
        touched_np = touched.numpy()

        # Per-touched-cell bin classification. A Python loop here is over
        # TOUCHED CELLS (typically far fewer than points, and the
        # sequential contiguous-run scan doesn't vectorise cleanly) --
        # not the per-POINT loop Ticket #18 forbids.
        ground_hi_per_cell = np.zeros(n_cells, dtype=np.int64)
        ceiling_bin_per_cell = np.full(n_cells, -1, dtype=np.int64)  # -1 = NO_CEILING
        for flat_cell in touched_np:
            bins = extract_layer_bins(hist[flat_cell], spec, min_bin_count_for_ceiling)
            ground_hi_per_cell[flat_cell] = bins.ground_hi
            ceiling_bin_per_cell[flat_cell] = bins.ceiling_bin if bins.ceiling_bin is not None else -1

        ground_hi_t = torch.from_numpy(ground_hi_per_cell)
        ceiling_bin_t = torch.from_numpy(ceiling_bin_per_cell)

        point_ground_hi = ground_hi_t[flat]
        point_ceiling_bin = ceiling_bin_t[flat]
        is_ground_point = bin_in <= point_ground_hi
        is_ceiling_point = (point_ceiling_bin >= 0) & (bin_in == point_ceiling_bin)

        ground_max_scratch = torch.full((n_cells,), float("-inf"), dtype=torch.float64)
        ceiling_min_scratch = torch.full((n_cells,), float("inf"), dtype=torch.float64)

        if torch.any(is_ground_point):
            flat_g = flat[is_ground_point]
            z_g = z_in[is_ground_point]
            ground_max_scratch.scatter_reduce_(0, flat_g, z_g, reduce="amax", include_self=True)
        if torch.any(is_ceiling_point):
            flat_c = flat[is_ceiling_point]
            z_c = z_in[is_ceiling_point]
            ceiling_min_scratch.scatter_reduce_(0, flat_c, z_c, reduce="amin", include_self=True)

        has_ground = ground_max_scratch[touched] > float("-inf")
        has_ceiling = ceiling_min_scratch[touched] < float("inf")

        ground_max_vals = ground_max_scratch[touched].numpy()
        ceiling_min_vals = ceiling_min_scratch[touched].numpy()

        int16_lo, int16_hi = -327.67, 327.67
        ground_max_c = np.clip(ground_max_vals, int16_lo, int16_hi)
        ground_max_enc = np.round(ground_max_c / H_QUANTUM_M).astype(np.int16)
        cm.h_max[level, touched_np[has_ground.numpy()]] = ground_max_enc[has_ground.numpy()]

        ceil_flat = touched_np[has_ceiling.numpy()]
        ceiling_min_c = np.clip(ceiling_min_vals[has_ceiling.numpy()], int16_lo, int16_hi)
        cm.h_ceil_min[level, ceil_flat] = np.round(ceiling_min_c / H_QUANTUM_M).astype(np.int16)

        # Cells whose ceiling scan found NO_CEILING must read that way,
        # even if a stale ceiling value from an earlier call is sitting
        # there -- explicit sentinel write, not "leave it alone".
        no_ceiling_flat = touched_np[~has_ceiling.numpy()]
        cm.h_ceil_min[level, no_ceiling_flat] = NO_CEILING_SENTINEL
        cm.h_ceil_max[level, no_ceiling_flat] = NO_CEILING_SENTINEL
