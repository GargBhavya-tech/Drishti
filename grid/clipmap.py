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
from grid.cell import (
    NO_CEILING_SENTINEL,
    OBS_UNOBSERVED,
    OBSERVABILITY_MASK,
    TILE_BASE_UNSET,
    TILE_SIZE,
    decode_h,
    encode_h,
    expected_stamp,
    n_tiles_per_axis,
    tile_index_batch,
    tile_member_flat_indices,
)
from sensor.schedule import Level, generate_schedule
from sensor.sensor_model import SensorConfig

# The seven Bible Part 9.3 "v1" per-cell fields, at their ORIGINAL flat
# int16/uint16/uint8 sizes. `stamp` (Ticket #14) is an integrity-check
# plane layered on top and deliberately excluded from this list -- it is
# not part of the map's own claimed memory footprint (Part 19's 12.58 MB
# headline figure), it is a read-time cross-check. Ticket #17 changes
# h_min/h_max/h_mean's ACTUAL live storage (see LeveledPlane below), but
# this tuple and `memory_bytes_v1()` stay a FIXED historical reference
# figure -- Ticket #11's own "assert allocation matches 12.58 MB
# exactly" is a comparison baseline, not a live measurement.
_V1_CELL_PLANES = ("h_min", "h_max", "h_mean", "h_m2", "count", "class_conf", "flags")
_V1_BYTES_PER_CELL = 12


class LeveledPlane:
    """A per-level array whose dtype can differ by level (Ticket #17:
    int16 at L0, int8 at L1-L3 for h_min/h_max/h_mean) while still
    supporting the SAME `arr[level, idx]` / `arr[level]` indexing every
    existing consumer (grid/scatter.py, grid/layers.py, this module's
    own scroll/reset/lookup code, and every test) already uses against
    the old uniform (L, N*N) array -- so wiring this in touches zero
    existing call sites' syntax, only what values they read/write.
    """

    __slots__ = ("_arrays",)

    def __init__(self, arrays: list) -> None:
        self._arrays = arrays  # one (N*N,) array per level, each its own dtype

    def __getitem__(self, key):
        if isinstance(key, tuple):
            level, idx = key
            return self._arrays[level][idx]
        return self._arrays[key]

    def __setitem__(self, key, value) -> None:
        if isinstance(key, tuple):
            level, idx = key
            self._arrays[level][idx] = value
        else:
            self._arrays[key] = value

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self._arrays)


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
        #
        # Ticket #17: h_min/h_max/h_mean are LeveledPlane, not a plain
        # (L, N*N) array -- L0 stays int16 (absolute, 1cm quantum,
        # unchanged from Ticket #11); L1-L3 are int8 (an offset from a
        # per-tile base elevation, 2/4/8cm quantum). See LeveledPlane's
        # own docstring for why this preserves every existing
        # arr[level, idx] call site unchanged.
        self.h_min = LeveledPlane(
            [np.zeros(N * N, dtype=np.int16)] + [np.zeros(N * N, dtype=np.int8) for _ in range(L - 1)]
        )
        self.h_max = LeveledPlane(
            [np.zeros(N * N, dtype=np.int16)] + [np.zeros(N * N, dtype=np.int8) for _ in range(L - 1)]
        )
        self.h_mean = LeveledPlane(
            [np.zeros(N * N, dtype=np.int16)] + [np.zeros(N * N, dtype=np.int8) for _ in range(L - 1)]
        )
        # Ticket #17's per-tile base-elevation plane: one absolute,
        # 1cm-quantised int16 value per TILE_SIZE x TILE_SIZE tile, per
        # level (L0's slot is allocated but never used/read -- L0 doesn't
        # tile, see grid.cell's module docstring -- kept for indexing
        # uniformity; its tiny waste is disclosed in memory_bytes_v2()).
        # TILE_BASE_UNSET (not 0) marks "no live cell in this tile right
        # now", mirroring h_ceil_min's NO_CEILING_SENTINEL convention below.
        self._n_tiles_axis = n_tiles_per_axis(N, TILE_SIZE)
        n_tiles = self._n_tiles_axis * self._n_tiles_axis
        self.tile_base_h = np.full((L, n_tiles), TILE_BASE_UNSET, dtype=np.int16)
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
        """The ORIGINAL Ticket #11 baseline: all seven v1 cell fields as
        flat int16/uint16/uint8 at every level (Bible Part 9.3's 12
        bytes/cell). A FIXED reference figure for comparison -- Ticket
        #11's own "assert allocation matches 12.58 MB exactly" -- computed
        by formula, NOT by introspecting the live arrays, so it keeps
        reporting the pre-#17 baseline even though h_min/h_max/h_mean's
        REAL live storage is now smaller (memory_bytes_v2() reports
        that). Excludes `stamp` and `histogram`, same as always."""
        return len(self.levels) * self.N * self.N * _V1_BYTES_PER_CELL

    def memory_bytes_v2(self) -> int:
        """Ticket #17's ACTUAL live allocation: introspects the real
        arrays (h_min/h_max/h_mean now mixed int16/int8 via LeveledPlane,
        plus the tile_base_h plane's own overhead), so this number is
        only as good as what is really allocated -- not a restatement of
        the claim. Includes tile_base_h, which the Build Map's headline
        "786,432 x 3 = 2.36 MB" arithmetic does not itself account for
        (that arithmetic assumes the base plane is free); disclosing it
        is why this project's own measured savings (~18.7%) land a
        fraction below the Build Map's naive 18.8%, not a discrepancy to
        paper over (Bible Part 19's own rule: never understate memory by
        omitting a real plane)."""
        total = self.h_min.nbytes + self.h_max.nbytes + self.h_mean.nbytes
        for name in ("h_m2", "count", "class_conf", "flags"):
            total += getattr(self, name).nbytes
        total += self.tile_base_h.nbytes
        return total

    # ------------------------------------------------------------------
    # Ticket #17 -- per-tile base elevation lifecycle
    # ------------------------------------------------------------------

    def get_or_create_tile_bases(self, level: int, flat_idx: np.ndarray, representative_z_m: np.ndarray) -> np.ndarray:
        """For each touched cell (by flat storage index) at `level`,
        return the base elevation (metres) its tile should encode
        relative to -- REUSING the tile's existing base if any cell in
        that tile is currently observed (flags != 0), or else deriving a
        FRESH base from `representative_z_m` (typically this batch's own
        mean height per cell) and storing it.

        This is the one piece of state that must never be recomputed out
        from under already-encoded cells (see grid.cell's module
        docstring) -- callers (grid/scatter.py, grid/layers.py) must
        route every level>=1 height write through this before encoding,
        never invent their own base.

        level=0 is not tiled; calling this for level 0 is a caller bug.
        """
        if level == 0:
            raise ValueError("level 0 does not use tile bases -- it stores absolute height directly")

        N = self.N
        si = flat_idx % N
        sj = flat_idx // N
        tile_idx = tile_index_batch(si, sj, self._n_tiles_axis, TILE_SIZE)

        base_m = np.empty(flat_idx.shape[0], dtype=np.float64)
        for t in np.unique(tile_idx):
            in_tile = tile_idx == t
            existing = int(self.tile_base_h[level, t])
            if existing != int(TILE_BASE_UNSET):
                member_flat = tile_member_flat_indices(int(t), self._n_tiles_axis, N, TILE_SIZE)
                tile_is_live = bool(np.any(self.flags[level, member_flat] != 0))
            else:
                tile_is_live = False

            if existing != int(TILE_BASE_UNSET) and tile_is_live:
                base_m[in_tile] = decode_h(existing, level=0)  # base itself stored absolute, 1cm quantum, like L0
            else:
                fresh_base = float(np.mean(representative_z_m[in_tile]))
                self.tile_base_h[level, t] = encode_h(fresh_base, level=0)
                base_m[in_tile] = fresh_base

        return base_m

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
        # tile_base_h is DELIBERATELY not touched here (Ticket #17): this
        # clears individual CELLS, not whole tiles, so a tile spanning
        # this boundary could have some cells cleared and others
        # retained -- wiping its base here would orphan the retained
        # cells' already-encoded (relative-to-the-old-base) values.
        # get_or_create_tile_bases() checks each tile's LIVE flags
        # directly to decide fresh-vs-reuse, so a stale base here is
        # harmless: it is only ever read once a cell claims OBS_* via
        # flags, which this loop just zeroed for every cell it touches.

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
        # Safe to reset unconditionally here (unlike _reset_indices
        # above): a full-level wrap clears EVERY cell, so every tile at
        # this level is genuinely, wholly empty -- no partial-tile case.
        self.tile_base_h[level, :] = TILE_BASE_UNSET

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

            if l == 0:
                base = 0.0
            else:
                # Ticket #17: L1-L3 store an int8 OFFSET from this cell's
                # tile base -- decode needs that base too. The cell is
                # OBSERVED (checked above), so its tile is live and its
                # base is real, not TILE_BASE_UNSET.
                tile_idx = int(tile_index_batch(np.array([si]), np.array([sj]), self._n_tiles_axis, TILE_SIZE)[0])
                base = decode_h(int(self.tile_base_h[l, tile_idx]), level=0)

            return l, CellView(
                level=l,
                observability=obs,
                h_min_m=decode_h(int(self.h_min[l, flat]), level=l, base_elevation_m=base),
                h_max_m=decode_h(int(self.h_max[l, flat]), level=l, base_elevation_m=base),
                h_mean_m=decode_h(int(self.h_mean[l, flat]), level=l, base_elevation_m=base),
                count=int(self.count[l, flat]),
                class_conf=int(self.class_conf[l, flat]),
            )

        # Outside every level's window.
        return None, CellView.unobserved(level=None)

    # ------------------------------------------------------------------
    # Tickets #33-34 -- observability write path (separate from scatter()'s
    # height/class writes, Ticket #18-19; ray carving never touches
    # h_min/h_max/etc, only the flags byte's observability sub-field).
    # ------------------------------------------------------------------

    def mark_observability(self, level: int, gi: int, gj: int, obs_state: int) -> bool:
        """Write the observability sub-field of `flags` for one cell at
        (level, gi, gj), identified by GLOBAL index (Ticket #10). Returns
        False (no write performed) if (gi, gj) falls outside this level's
        CURRENT window -- writing there would alias, via toroidal wrap,
        onto a completely different global cell's storage slot (Bible
        Part 8's warning about stale/foreign data), so out-of-window
        writes are silently skipped rather than corrupting unrelated
        cells. Always refreshes `stamp` alongside `flags`, exactly like
        `grid.scatter._write_touched_cells` -- an update to `flags`
        without a matching `stamp` update would make `lookup()`'s
        integrity cross-check (Ticket #14) treat this cell as corrupted
        on its very next read.
        """
        oi, oj = self.origin_i[level], self.origin_j[level]
        if not (oi <= gi < oi + self.N and oj <= gj < oj + self.N):
            return False
        si, sj = global_to_storage(gi, gj, self.N)
        flat = flat_index(si, sj, self.N)
        self.flags[level, flat] = np.uint8(obs_state)
        self.stamp[level, flat] = expected_stamp(gi, gj, self.N)
        return True
