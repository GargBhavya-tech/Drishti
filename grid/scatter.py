"""
grid/scatter.py

Ticket #18 -- the projection kernel: scatter one frame's points into every
level of the clipmap in a single vectorised pass. No Python loop over
points -- Build Map Ticket #18 calls a loop over 34k-120k points "fatal".

Ticket #19 -- class aggregation by mode -- lives here too (scatter_class),
since it's the same class of "per-point values, reduced per cell" kernel
as #18, just with a different reduction (histogram + argmax instead of
min/max/mean).

See DRISHTI_Project_Bible_v3.md Part 9.4 for the reference torch snippet
this follows, and Part 8 for the mipmap semantics (every point scatters
into every level whose current window contains it).
"""

from __future__ import annotations

import numpy as np
import torch

from grid.addressing import flat_index_batch, global_to_storage_batch, world_to_global_batch
from grid.cell import H_QUANTUM_M, OBS_OCCUPIED, encode_class_conf_batch, encode_h_batch, expected_stamp_batch
from grid.clipmap import Clipmap

COUNT_MAX = 65535  # uint16 max -- count must saturate, never wrap (Ticket #18 "Watch out" #3)

# h_m2 (Bible Part 9.3: "Welford M2, quantised -> variance") is stored here
# as variance in units of this quantum, clamped to uint16. Not specified
# to more precision than that by the ticket text; revisit if a later
# consumer (e.g. Ticket #48's roughness = sqrt(variance)) needs finer
# resolution than 1 cm^2 or a wider range than ~6.55 m^2.
_VARIANCE_QUANTUM_M2 = 1e-4  # 1 cm^2 per LSB


def scatter(cm: Clipmap, xyz_world: np.ndarray) -> None:
    """Project one frame's points into every level of `cm`.

    Mipmap semantics (Bible Part 8): a point scatters into every level
    whose CURRENT window contains it. Because windows are nested (a
    coarser level's world-space extent strictly contains every finer
    level's), this is automatically "its native level and all coarser
    ones" -- no separate filtering needed, it falls out of each level's
    own bounds check.

    Scope note -- deliberately NOT full temporal fusion: h_max/h_min use
    `scatter_reduce_(..., include_self=True)` against each cell's
    EXISTING stored value, so repeated calls across frames correctly
    extend the running min/max on their own. mean/variance/count, by
    contrast, are computed fresh from THIS call's own points and
    OVERWRITE whatever a cell already held -- correctly merging them
    across many frames needs Welford/Chan (Ticket #44's job, and that
    ticket's own "Watch out" specifically warns against the naive
    E[x^2]-E[x]^2 form used here, which is numerically fine for one
    frame's own batch but not for accumulating many frames on top of
    each other without a proper merge).
    """
    if xyz_world.shape[0] == 0:
        return

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

        # Sentinel init (Ticket #18 "Watch out" #4): +-inf, never 0, so a
        # genuinely negative (or positive) height cannot lose to a default
        # zero in the amax/amin reduction.
        h_max_scratch = torch.full((n_cells,), float("-inf"), dtype=torch.float64)
        h_min_scratch = torch.full((n_cells,), float("inf"), dtype=torch.float64)
        count_scratch = torch.zeros(n_cells, dtype=torch.int64)
        h_sum_scratch = torch.zeros(n_cells, dtype=torch.float64)
        h_sq_scratch = torch.zeros(n_cells, dtype=torch.float64)

        # include_self=True (Ticket #18 "Watch out" #2): without it, a
        # single point landing in a previously-empty cell is discarded
        # rather than kept.
        h_max_scratch.scatter_reduce_(0, flat, z_in, reduce="amax", include_self=True)
        h_min_scratch.scatter_reduce_(0, flat, z_in, reduce="amin", include_self=True)
        count_scratch.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int64))
        h_sum_scratch.scatter_add_(0, flat, z_in)
        h_sq_scratch.scatter_add_(0, flat, z_in * z_in)

        touched = torch.nonzero(count_scratch > 0, as_tuple=True)[0]
        if touched.numel() == 0:
            continue

        counts = count_scratch[touched]
        counts_f = counts.to(torch.float64)
        means = h_sum_scratch[touched] / counts_f
        variances = torch.clamp(h_sq_scratch[touched] / counts_f - means * means, min=0.0)
        h_max_vals = h_max_scratch[touched]
        h_min_vals = h_min_scratch[touched]

        # Recover each touched flat index's owning global (gi, gj) under
        # the CURRENT window -- si/sj alone don't determine it (storage is
        # toroidal); the unique representative in [origin, origin+N) is
        # origin + ((storage_offset - origin) mod N).
        si_t = touched % N
        sj_t = touched // N
        gi_t = oi + torch.remainder(si_t - oi, N)
        gj_t = oj + torch.remainder(sj_t - oj, N)

        _write_touched_cells(
            cm,
            level,
            flat_idx=touched.numpy(),
            gi=gi_t.numpy(),
            gj=gj_t.numpy(),
            h_max=h_max_vals.numpy(),
            h_min=h_min_vals.numpy(),
            h_mean=means.numpy(),
            variance=variances.numpy(),
            count=torch.clamp(counts, max=COUNT_MAX).numpy(),  # saturate, never wrap
        )


def _write_touched_cells(
    cm: Clipmap,
    level: int,
    flat_idx: np.ndarray,
    gi: np.ndarray,
    gj: np.ndarray,
    h_max: np.ndarray,
    h_min: np.ndarray,
    h_mean: np.ndarray,
    variance: np.ndarray,
    count: np.ndarray,
) -> None:
    if level == 0:
        # UNCHANGED from before Ticket #17: absolute int16, 1cm quantum.
        # Clip defensively to int16's representable range (Part 9.3: 1 cm
        # fixed point, +-327.67 m) before casting -- silent wraparound on
        # an out-of-range height would be exactly the kind of invisible
        # bug this project keeps guarding against elsewhere (count
        # saturation, N&(N-1)).
        int16_lo, int16_hi = -327.67, 327.67
        h_max_c = np.clip(h_max, int16_lo, int16_hi)
        h_min_c = np.clip(h_min, int16_lo, int16_hi)
        h_mean_c = np.clip(h_mean, int16_lo, int16_hi)
        cm.h_max[level, flat_idx] = np.round(h_max_c / H_QUANTUM_M).astype(np.int16)
        cm.h_min[level, flat_idx] = np.round(h_min_c / H_QUANTUM_M).astype(np.int16)
        cm.h_mean[level, flat_idx] = np.round(h_mean_c / H_QUANTUM_M).astype(np.int16)
    else:
        # Ticket #17: resolve each touched cell's TILE base BEFORE
        # setting flags below -- get_or_create_tile_bases decides
        # fresh-vs-reuse from the tile's CURRENT (pre-this-write) flags,
        # and setting flags first would make a genuinely-first write look
        # like it's reusing itself as evidence the tile was already live.
        base_m = cm.get_or_create_tile_bases(level, flat_idx, representative_z_m=h_mean)
        cm.h_max[level, flat_idx] = encode_h_batch(h_max, level, base_m)
        cm.h_min[level, flat_idx] = encode_h_batch(h_min, level, base_m)
        cm.h_mean[level, flat_idx] = encode_h_batch(h_mean, level, base_m)

    cm.h_m2[level, flat_idx] = np.clip(
        np.round(variance / _VARIANCE_QUANTUM_M2), 0, 65535
    ).astype(np.uint16)
    cm.count[level, flat_idx] = count.astype(np.uint16)
    cm.flags[level, flat_idx] = OBS_OCCUPIED
    cm.stamp[level, flat_idx] = expected_stamp_batch(gi, gj, cm.N)


def scatter_class(cm: Clipmap, xyz_world: np.ndarray, class_ids: np.ndarray) -> None:
    """Ticket #19 -- per-cell class by MODE, never by averaging class IDs
    (class 2 and class 6 would average to 4, an unrelated class -- Bible
    Part 9.4/9's repeated warning). Writes `class_conf`: the winning
    class in the high nibble, the runner-up fraction (this cell's second-
    most-common class's share of ITS points, not just of the top two) in
    the low nibble.

    Scope note: this writes `class_conf` only. It does not touch `flags`/
    `stamp` -- `scatter()` (Ticket #18) owns marking a cell observed;
    call both on the same point batch (as Checkpoint #22 does) rather
    than relying on this function alone to make a cell readable via
    `lookup()`.
    """
    if xyz_world.shape[0] == 0:
        return

    xyz = torch.from_numpy(np.ascontiguousarray(xyz_world, dtype=np.float64))
    x, y = xyz[:, 0], xyz[:, 1]
    cls = torch.from_numpy(np.ascontiguousarray(class_ids, dtype=np.int64))
    n_classes = int(cls.max().item()) + 1 if cls.numel() else 1
    n_classes = max(n_classes, 10)  # DrishtiClass has 10 members (0-9)
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

        gi_in, gj_in, cls_in = gi[in_window], gj[in_window], cls[in_window]
        si = global_to_storage_batch(gi_in, N)
        sj = global_to_storage_batch(gj_in, N)
        flat = flat_index_batch(si, sj, N)

        # (cell, class) joint histogram via one combined index -- a mode,
        # never an arithmetic mean, of class IDs.
        combined = flat * n_classes + cls_in
        hist = torch.zeros(n_cells * n_classes, dtype=torch.int64)
        hist.scatter_add_(0, combined, torch.ones_like(combined))
        hist = hist.view(n_cells, n_classes)

        cell_totals = hist.sum(dim=1)
        touched = torch.nonzero(cell_totals > 0, as_tuple=True)[0]
        if touched.numel() == 0:
            continue

        touched_hist = hist[touched]  # (T, n_classes)
        sorted_counts, _ = torch.sort(touched_hist, dim=1, descending=True)
        second = sorted_counts[:, 1] if n_classes > 1 else torch.zeros(touched.numel(), dtype=torch.int64)
        totals = cell_totals[touched].to(torch.float64)
        winning_class = torch.argmax(touched_hist, dim=1)
        runner_up_fraction = second.to(torch.float64) / totals

        cm.class_conf[level, touched.numpy()] = encode_class_conf_batch(
            winning_class.numpy(), runner_up_fraction.numpy()
        )
