"""
grid/histogram.py

Ticket #20 -- 8-bin per-cell height histogram, feeding Ticket #21's
ground/gap/ceiling extraction (Claim 1: multi-layer cells, Bible Part 9.2).

Bin width must be DERIVED, not a literal (Ticket #20 "Watch out"). The
Bible's own worked range runs from half a metre below local ground to
`min_clearance_m` plus a headroom margin above it -- see the Build Map's
Ticket #20 text for the exact worked figures against `vehicle_ugv.yaml`.
This module computes the upper bound from `min_clearance_m` at call
time and derives bin width from THAT (span / n_bins), rather than typing
the resulting numbers in directly -- so a different vehicle config
regenerates a correctly-scaled histogram automatically. (Deliberately no
literal vehicle-parameter values appear in this docstring -- see
tests/test_vehicle_config.py's watched-literal guard, which greps for
exactly that outside configs/ and tests/.)

Relative to LOCAL ground, not z=0 (Ticket #20 "Watch out") -- on a slope,
an absolute-z histogram puts everything in one bin. `z_ground` here is
supplied BY THE CALLER per point: this repo does not yet have
`perception/ground_prior.py` (Ticket #26, Phase 3), which is what will
normally supply it. Until then, callers (tests, or Checkpoint #22's
synthetic integration) must pass their own per-point ground estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from grid.addressing import flat_index_batch, global_to_storage_batch, world_to_global_batch
from grid.clipmap import Clipmap
from sensor.vehicle_config import VehicleConfig

N_BINS = 8
_LOWER_OFFSET_M = -0.5  # fixed: how far below local ground the histogram starts
_UPPER_MARGIN_M = 1.5  # margin above min_clearance_m, so a genuine overhang clears the top bin
BIN_COUNT_MAX = 255  # uint8 max -- each bin count saturates, never wraps


@dataclass(frozen=True)
class HistogramSpec:
    lower_offset_m: float
    upper_offset_m: float
    bin_width_m: float
    n_bins: int = N_BINS


def histogram_spec(vehicle: VehicleConfig) -> HistogramSpec:
    upper = vehicle.min_clearance_m + _UPPER_MARGIN_M
    lower = _LOWER_OFFSET_M
    bin_width = (upper - lower) / N_BINS
    return HistogramSpec(lower_offset_m=lower, upper_offset_m=upper, bin_width_m=bin_width, n_bins=N_BINS)


def scatter_histogram(
    cm: Clipmap,
    xyz_world: np.ndarray,
    z_ground: np.ndarray,
    vehicle: VehicleConfig,
) -> None:
    """Bin each point's height ABOVE ITS OWN z_ground into cm.histogram.

    z_ground: (P,) array, one local-ground estimate per point (not per
    cell) -- matching how a real column-wise ground walk (Ticket #26)
    would supply it, so this doesn't need to change when that lands.
    Points whose relative height falls outside [lower_offset_m,
    upper_offset_m) clip into the nearest edge bin rather than being
    dropped, matching a histogram's usual open-ended edge-bin behaviour.
    """
    if xyz_world.shape[0] == 0:
        return

    spec = histogram_spec(vehicle)
    xyz = torch.from_numpy(np.ascontiguousarray(xyz_world, dtype=np.float64))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    zg = torch.from_numpy(np.ascontiguousarray(z_ground, dtype=np.float64))
    z_rel = z - zg

    bin_idx = torch.floor((z_rel - spec.lower_offset_m) / spec.bin_width_m).long()
    bin_idx = torch.clamp(bin_idx, 0, spec.n_bins - 1)

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

        gi_in, gj_in = gi[in_window], gj[in_window]
        bin_in = bin_idx[in_window]
        si = global_to_storage_batch(gi_in, N)
        sj = global_to_storage_batch(gj_in, N)
        flat = flat_index_batch(si, sj, N)

        combined = flat * spec.n_bins + bin_in
        hist_flat = torch.zeros(n_cells * spec.n_bins, dtype=torch.int64)
        hist_flat.scatter_add_(0, combined, torch.ones_like(combined))
        hist = hist_flat.view(n_cells, spec.n_bins)

        touched = torch.nonzero(hist.sum(dim=1) > 0, as_tuple=True)[0]
        if touched.numel() == 0:
            continue

        counts = torch.clamp(hist[touched], max=BIN_COUNT_MAX)  # saturate, never wrap
        cm.histogram[level, touched.numpy(), :] = counts.numpy().astype(np.uint8)
