"""
grid/clipmap.py

Tickets #11, #13, #14, #15 -- the foveated clipmap: SoA allocation, scroll
with clear-on-scroll, stamp validation, and lookup().

This is the load-bearing wall of the project (Bible Part 8; Build Map
Phase 1 intro: "It is also where generated code fails most often and most
silently"). Four nested power-of-two-cell-size levels, each a flat N*N
array, toroidally addressed, world-anchored (never rotates with the
vehicle). See DRISHTI_Project_Bible_v3.md Part 8 for why this beats a
quadtree, and DRISHTI_Build_Map.md tickets #10-#16 for the exact test
matrix this module is checked against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from grid.addressing import flat_index, global_to_storage, is_power_of_two, world_to_global
from grid.cell import NO_CEILING_SENTINEL, OBS_UNOBSERVED, OBSERVABILITY_MASK, decode_h, expected_stamp
from sensor.schedule import Level, generate_schedule
from sensor.sensor_model import SensorConfig

# The seven Bible Part 9.3 "v1" per-cell fields. `stamp` (Ticket #14) is
# an integrity-check plane layered on top and deliberately excluded from
# this list -- it is not part of the map's own claimed memory footprint
# (Part 19's 12.58 MB headline figure), it is a read-time cross-check.
_V1_CELL_PLANES = ("h_min", "h_max", "h_mean", "h_m2", "count", "class_conf", "flags")


@dataclass(frozen=True)
class CellView:
    """A read-only snapshot of one cell, as returned by `Clipmap.lookup()`.

    Height/count/class fields are None whenever the cell is not observed
    -- Bible Part 9 edge case: "height fields meaningless unless the
    observed bit is set", enforced here rather than left to every caller.
    """

    level: Optional[int]
    observability: int
    h_min_m: Optional[float]
    h_max_m: Optional[float]
    h_mean_m: Optional[float]
    count: int
    class_conf: int

    @property
    def observed(self) -> bool:
        return self.observability != OBS_UNOBSERVED

    @staticmethod
    def unobserved(level: Optional[int] = None) -> "CellView":
        return CellView(
            level=level,
            observability=OBS_UNOBSERVED,
            h_min_m=None,
            h_max_m=None,
            h_mean_m=None,
            count=0,
            class_conf=0,
        )


class Clipmap:
    """Four (by default) nested uniform grids, toroidally addressed,
    world-anchored. See module docstring.

    Invariant (Bible Part 8): levels are alternative views of one world
    and are never summed, averaged, or aggregated across. `lookup()` is
    the only public read path in this module, and no function here
    iterates over levels to accumulate a per-cell statistic -- that
    invariant is enforced by not writing such a function, not by a guard.
    """

    def __init__(self, sm: SensorConfig, n_levels: int = 4, N: int = 512, c0: float = 0.05):
        if not is_power_of_two(N):
            raise ValueError(f"N must be a power of two, got {N}")
        self.N = N
        self.levels: List[Level] = generate_schedule(sm, c0=c0, n_levels=n_levels)
        L = len(self.levels)

        # Structure-of-arrays: one contiguous plane per field, not a
        # struct/dataclass per cell and not one packed (N,N,7) array
        # (Ticket #11 "Watch out"). Flat N*N per level, matching
        # `grid.addressing.flat_index`'s convention.
        self.h_min = np.zeros((L, N * N), dtype=np.int16)
        self.h_max = np.zeros((L, N * N), dtype=np.int16)
        self.h_mean = np.zeros((L, N * N), dtype=np.int16)
        self.h_m2 = np.zeros((L, N * N), dtype=np.uint16)
        self.count = np.zeros((L, N * N), dtype=np.uint16)
        self.class_conf = np.zeros((L, N * N), dtype=np.uint8)
        # Zero by construction -- OBS_UNOBSERVED is the zero value (Bible
        # Part 10.1); every unwritten cell is correctly "never observed".
        self.flags = np.zeros((L, N * N), dtype=np.uint8)
        self.stamp = np.zeros((L, N * N), dtype=np.uint16)
        # Ticket #20 -- 8-bin per-cell height histogram, packed as one
        # uint8 count per bin (8 bytes/cell). Added on top of the Ticket
        # #11 v1 layout, like `stamp` -- excluded from memory_bytes_v1()
        # for the same reason (a later, separately-justified plane, not
        # part of the Part 9.3 baseline byte count).
        self.histogram = np.zeros((L, N * N, 8), dtype=np.uint8)
        # Ticket #21 -- v2 ceiling layer (Bible Part 9.3). Default =
        # NO_CEILING_SENTINEL, NOT 0 -- 0 would decode as "a ceiling
        # sitting at ground level", a bogus zero-clearance reading on
        # every untouched cell. h_min/h_max (Ticket #18) become the
        # GROUND layer's range once a ceiling is extracted; for a
        # single-layer cell they are unchanged (ground IS the only layer).
        self.h_ceil_min = np.full((L, N * N), NO_CEILING_SENTINEL, dtype=np.int16)
        self.h_ceil_max = np.full((L, N * N), NO_CEILING_SENTINEL, dtype=np.int16)

        # Current window's lower-left global cell index, per level.
        self.origin_i: List[int] = [0] * L
        self.origin_j: List[int] = [0] * L

        # Test-only escape hatch for Ticket #14's adversarial test, which
        # must disable clear-on-scroll to prove the stamp cross-check
        # catches what clearing would have caught. Never disable this
        # outside a test.
        self._clear_enabled = True

        self.stamp_mismatches = 0
        self.last_scroll_cells_cleared = 0

    # ------------------------------------------------------------------
    # Ticket #11 -- memory accounting
    # ------------------------------------------------------------------

    def memory_bytes_v1(self) -> int:
        """Total bytes across the seven Part 9.3 v1 cell fields, across
        all levels. Excludes `stamp`."""
        return sum(getattr(self, name).nbytes for name in _V1_CELL_PLANES)

    # ------------------------------------------------------------------
    # Ticket #13 -- scroll with clear-on-scroll
    # ------------------------------------------------------------------

    def scroll_to(self, ego_x: float, ego_y: float) -> None:
        """Recentre every level's window on the vehicle. Retained cells
        keep their flat index (toroidal addressing); only rows/columns
        scrolling in are cleared. Forgetting this is Bible Part 8's "most
        dangerous bug in the project" -- stale data from behind the
        vehicle would silently read as though it were ahead of it.
        """
        self.last_scroll_cells_cleared = 0
        for l, lvl in enumerate(self.levels):
            c_l = lvl.cell_size_m
            gi, gj = world_to_global(ego_x, ego_y, c_l)
            new_origin_i = gi - self.N // 2
            new_origin_j = gj - self.N // 2
            di = new_origin_i - self.origin_i[l]
            dj = new_origin_j - self.origin_j[l]
            if di == 0 and dj == 0:
                continue
            if self._clear_enabled:
                self._clear_scrolled_region(l, di, dj)
            self.origin_i[l] = new_origin_i
            self.origin_j[l] = new_origin_j

    def _clear_scrolled_region(self, level: int, di: int, dj: int) -> None:
        N = self.N
        old_origin_i = self.origin_i[level]
        old_origin_j = self.origin_j[level]

        if abs(di) >= N or abs(dj) >= N:
            # Full wrap: every cell in the window is new. Ticket #13's
            # adversarial test cares about correctness here, not the
            # O(N*(|di|+|dj|)) budget that applies to small shifts.
            self._reset_all(level)
            self.last_scroll_cells_cleared += N * N
            return

        flat_to_clear: List[np.ndarray] = []

        if di != 0:
            if di > 0:
                gi_range = np.arange(old_origin_i + N, old_origin_i + N + di, dtype=np.int64)
            else:
                gi_range = np.arange(old_origin_i + di, old_origin_i, dtype=np.int64)
            si_clear = gi_range & (N - 1)
            sj_all = np.arange(N, dtype=np.int64)
            flat_to_clear.append((sj_all[:, None] * N + si_clear[None, :]).ravel())

        if dj != 0:
            if dj > 0:
                gj_range = np.arange(old_origin_j + N, old_origin_j + N + dj, dtype=np.int64)
            else:
                gj_range = np.arange(old_origin_j + dj, old_origin_j, dtype=np.int64)
            sj_clear = gj_range & (N - 1)
            si_all = np.arange(N, dtype=np.int64)
            flat_to_clear.append((sj_clear[:, None] * N + si_all[None, :]).ravel())

        if not flat_to_clear:
            return
        idx = np.unique(np.concatenate(flat_to_clear))
        self._reset_indices(level, idx)
        self.last_scroll_cells_cleared += int(idx.size)

    def _reset_indices(self, level: int, flat_idx: np.ndarray) -> None:
        self.h_min[level, flat_idx] = 0
        self.h_max[level, flat_idx] = 0
        self.h_mean[level, flat_idx] = 0
        self.h_m2[level, flat_idx] = 0
        self.count[level, flat_idx] = 0
        self.class_conf[level, flat_idx] = 0
        self.flags[level, flat_idx] = 0
        self.stamp[level, flat_idx] = 0
        self.histogram[level, flat_idx, :] = 0
        self.h_ceil_min[level, flat_idx] = NO_CEILING_SENTINEL
        self.h_ceil_max[level, flat_idx] = NO_CEILING_SENTINEL

    def _reset_all(self, level: int) -> None:
        self.h_min[level, :] = 0
        self.h_max[level, :] = 0
        self.h_mean[level, :] = 0
        self.h_m2[level, :] = 0
        self.count[level, :] = 0
        self.class_conf[level, :] = 0
        self.flags[level, :] = 0
        self.stamp[level, :] = 0
        self.histogram[level, :, :] = 0
        self.h_ceil_min[level, :] = NO_CEILING_SENTINEL
        self.h_ceil_max[level, :] = NO_CEILING_SENTINEL

    # ------------------------------------------------------------------
    # Ticket #15 -- lookup(): the only public read path
    # ------------------------------------------------------------------

    def lookup(self, x: float, y: float) -> Tuple[Optional[int], CellView]:
        """Return the cell at the finest level whose current window
        contains (x, y). Bounds-checked against each level's window --
        toroidal wrap is never trusted to mean containment (Bible Part 8
        edge-case table). Ticket #14's stamp cross-check runs on every
        read of a cell whose flags claim it holds data.
        """
        for l, lvl in enumerate(self.levels):
            c_l = lvl.cell_size_m
            gi, gj = world_to_global(x, y, c_l)
            oi, oj = self.origin_i[l], self.origin_j[l]
            if not (oi <= gi < oi + self.N and oj <= gj < oj + self.N):
                continue
            si, sj = global_to_storage(gi, gj, self.N)
            flat = flat_index(si, sj, self.N)
            flag_byte = int(self.flags[l, flat])

            if flag_byte != 0:
                stored_stamp = int(self.stamp[l, flat])
                if stored_stamp != expected_stamp(gi, gj, self.N):
                    # This storage slot's flags claim it holds data, but
                    # that data belongs to a different global cell -- a
                    # missed clear-on-scroll. Report honestly rather than
                    # trust it (Bible Part 8: "a slightly slower demo
                    # that cannot lie is worth more than a fast one that
                    # might").
                    self.stamp_mismatches += 1
                    return l, CellView.unobserved(level=l)

            obs = flag_byte & OBSERVABILITY_MASK
            if obs == OBS_UNOBSERVED:
                return l, CellView.unobserved(level=l)

            return l, CellView(
                level=l,
                observability=obs,
                h_min_m=decode_h(int(self.h_min[l, flat])),
                h_max_m=decode_h(int(self.h_max[l, flat])),
                h_mean_m=decode_h(int(self.h_mean[l, flat])),
                count=int(self.count[l, flat]),
                class_conf=int(self.class_conf[l, flat]),
            )

        # Outside every level's window.
        return None, CellView.unobserved(level=None)
