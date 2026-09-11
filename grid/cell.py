"""
grid/cell.py

Fixed-point height encode/decode and the observability sub-field of the
`flags` byte (Bible Part 9.3, Part 10.1). Small pure helpers shared by
`grid/clipmap.py` (Tickets #11, #13, #14, #15, #17).

Ticket #17 -- foveated height quantum, completed. `encode_h`/`decode_h`
now take an optional `level` (default 0, so every pre-#17 call site --
`encode_h(z)`, `decode_h(v)` -- is unchanged, bit-for-bit): L0 keeps the
original flat 1cm/int16 absolute encoding exactly; L1-L3 use a coarser
quantum (2/4/8cm, Bible Part 9.3) and store the height as an int8 OFFSET
from a per-TILE base elevation rather than an absolute int16 value --
matching the sensor's own vertical sampling growing coarser with range
(Part 3: ~74cm vertical spacing at L3's Nyquist radius; storing sub-cm
precision there is exactly the error this project criticises a uniform
XY grid for making).

Per-tile base elevation, and how its lifecycle is kept correct (Ticket
#17's own "Watch out": "a model will apply the quantum but forget the
per-tile base, so coarse levels lose absolute elevation" -- getting the
QUANTUM right is the easy half; keeping the BASE consistent across many
independent writes and clear-on-scroll cycles is the actual risk):

- Tiles are TILE_SIZE x TILE_SIZE STORAGE cells (toroidal, same indexing
  as everything else in this module) -- coarser than a cell, finer than
  a whole level, so real local terrain fits comfortably within one
  int8's relative range around a shared base.
- A tile's base is set ONCE, from whichever write FIRST makes any of its
  cells observed (flags != 0) again after being fully clear -- and then
  HELD FIXED for as long as at least one cell in that tile stays
  observed. Every later write to the SAME tile re-encodes relative to
  that SAME base, never a freshly recomputed one -- recomputing on every
  write would silently invalidate every OTHER already-encoded cell in
  the tile (they were encoded relative to the OLD base), which is
  exactly the kind of thing that "looks" like it works until two writes
  land in the same tile at different times.
- This means `grid.clipmap.Clipmap.get_or_create_tile_bases()` --  not
  this module -- decides whether to reuse or (re)create a base, since
  that decision needs to inspect the LIVE clipmap's own `flags` plane.
  This module only provides the pure encode/decode math once a base is
  known.
"""

from __future__ import annotations

from typing import Union

import numpy as np

H_QUANTUM_M = 0.01  # 1 cm -- L0's quantum, and the base for level_quantum's scaling

# Ticket #17's per-level quantum schedule (Bible Part 9.3): 1, 2, 4, 8 cm
# for L0-L3.
def quantum_for_level(level: int) -> float:
    return H_QUANTUM_M * (2 ** level)


INT8_LO, INT8_HI = -128, 127

# Per-tile base-elevation plane (Ticket #17). TILE_SIZE=16 chosen so real
# local terrain stays comfortably within one int8's relative range at
# every level's quantum (worst case, L1's 2cm quantum: int8 covers
# +-2.56m around the base, over a tile spanning 16 cells x 10cm/cell =
# 1.6m at L1 -- ample margin) while keeping the base plane's own
# overhead small (1024 tiles/level at N=512, 8KB total across 4 levels
# -- see Clipmap.memory_bytes_v2's own accounting, which discloses this
# rather than pretending it's free).
TILE_SIZE = 16

# Sentinel for "this tile has no live cells right now" -- distinct from
# NO_CEILING_SENTINEL (different plane, different meaning) even though
# it happens to reuse the same numeric value, matching this codebase's
# existing convention (int16's minimum can never be a real 1cm-quantised
# height within the +-327.67m the L0/tile-base encoding actually uses).
TILE_BASE_UNSET = np.int16(-32768)


def n_tiles_per_axis(N: int, tile_size: int = TILE_SIZE) -> int:
    if N % tile_size != 0:
        raise ValueError(f"N={N} is not a multiple of tile_size={tile_size}")
    return N // tile_size


def tile_index(si: int, sj: int, n_tiles_axis: int, tile_size: int = TILE_SIZE) -> int:
    """Storage index (si, sj) -> flat tile index. Tiles tile the SAME
    storage grid cells do, just coarser -- (si // tile_size, sj //
    tile_size), row-major, matching grid.addressing.flat_index's own
    sj*N+si convention so the two stay easy to reason about together."""
    ti, tj = si // tile_size, sj // tile_size
    return tj * n_tiles_axis + ti


def tile_index_batch(si: np.ndarray, sj: np.ndarray, n_tiles_axis: int, tile_size: int = TILE_SIZE) -> np.ndarray:
    ti = si // tile_size
    tj = sj // tile_size
    return tj * n_tiles_axis + ti


def tile_member_flat_indices(tile_idx: int, n_tiles_axis: int, N: int, tile_size: int = TILE_SIZE) -> np.ndarray:
    """All storage flat indices belonging to one tile -- used only to
    check whether a tile currently has any live (observed) cell before
    deciding whether to reuse or recreate its base (see module
    docstring); tile_size^2 indices, small and bounded."""
    ti = tile_idx % n_tiles_axis
    tj = tile_idx // n_tiles_axis
    si_range = np.arange(ti * tile_size, (ti + 1) * tile_size)
    sj_range = np.arange(tj * tile_size, (tj + 1) * tile_size)
    si_grid, sj_grid = np.meshgrid(si_range, sj_range)
    return (sj_grid * N + si_grid).ravel()

# Observability sub-field: the low two bits of the `flags` byte (Bible
# Part 10.1, Part 23 edge-case table: "Unobserved cells read as flat
# ground at z=0"). UNOBSERVED must be the zero value -- an allocated-but-
# never-written cell must read as "never looked at", never as flat ground.
# FREE/OCCUPIED/OCCLUDED are written by ray traversal, Ticket #33-34 (not
# yet built); only UNOBSERVED is reachable until then.
OBS_UNOBSERVED = 0
OBS_FREE = 1
OBS_OCCUPIED = 2
OBS_OCCLUDED = 3
OBSERVABILITY_MASK = 0b11


def encode_h(z_m: float, level: int = 0, base_elevation_m: float = 0.0) -> Union[np.int16, np.int8]:
    """Ticket #17: metres -> fixed-point int, level-scaled quantum.

    level=0 (default): UNCHANGED from the original Ticket #11 encoding --
    1cm quantum, absolute int16, round() not truncate (1.234 round-trips
    to 1.23m, not 1.0m). base_elevation_m must be 0.0 here -- L0 never
    uses a tile base, it stores absolute height directly, same as before
    #17 existed. Every pre-#17 caller (`encode_h(z)`) gets bit-identical
    output to before.

    level>=1: quantum scales per quantum_for_level (2/4/8cm for L1-L3);
    the VALUE stored is `round((z_m - base_elevation_m) / quantum)`,
    clamped to int8's range -- an OFFSET from whatever base the caller
    supplies (see grid.clipmap.Clipmap.get_or_create_tile_bases for how
    that base is chosen and kept consistent across writes)."""
    quantum = quantum_for_level(level)
    if level == 0:
        if base_elevation_m != 0.0:
            raise ValueError("base_elevation_m must be 0.0 at level 0 -- L0 stores absolute height, never tile-relative")
        return np.int16(round(z_m / quantum))
    offset = round((z_m - base_elevation_m) / quantum)
    offset = max(INT8_LO, min(INT8_HI, offset))
    return np.int8(offset)


def decode_h(v: int, level: int = 0, base_elevation_m: float = 0.0) -> float:
    """Inverse of encode_h. level=0: UNCHANGED (`float(v) * H_QUANTUM_M`,
    base_elevation_m ignored/must be 0.0 -- not enforced here since a
    read path should never fail loudly over a caller passing 0.0
    explicitly, but a nonzero base at level 0 is a caller bug)."""
    quantum = quantum_for_level(level)
    if level == 0:
        return float(v) * quantum
    return base_elevation_m + float(v) * quantum


def encode_h_batch(z_m: np.ndarray, level: int, base_elevation_m) -> np.ndarray:
    """Vectorised encode_h for Ticket #18/#21's scatter kernels. `level`
    is a single scalar (all points in one call share a level, matching
    how grid.scatter/grid.layers already loop per-level); `base_elevation_m`
    is either a scalar (level 0, must be 0.0) or a (P,) array giving each
    point's OWN tile's base (level>=1 -- different points can land in
    different tiles within the same batch)."""
    quantum = quantum_for_level(level)
    if level == 0:
        if np.any(np.asarray(base_elevation_m) != 0.0):
            raise ValueError("base_elevation_m must be 0.0 at level 0")
        return np.round(z_m / quantum).astype(np.int16)
    offset = np.round((z_m - base_elevation_m) / quantum)
    offset = np.clip(offset, INT8_LO, INT8_HI)
    return offset.astype(np.int8)


def decode_h_batch(v: np.ndarray, level: int, base_elevation_m) -> np.ndarray:
    quantum = quantum_for_level(level)
    if level == 0:
        return v.astype(np.float64) * quantum
    return np.asarray(base_elevation_m, dtype=np.float64) + v.astype(np.float64) * quantum


def expected_stamp(i: int, j: int, N: int) -> int:
    """Ticket #14 -- a per-cell integrity tag, recomputed from the
    CURRENTLY QUERIED global (i, j) on every read and compared against
    what's stored; a mismatch means this storage slot currently holds
    another cell's data -- a missed clear-on-scroll.

    DELIBERATE DEVIATION from the Build Map's literal Ticket #14 formula
    `stamp = ((i & 0xFF) << 8) | (j & 0xFF)`. For N=512 (Ticket #11's
    specified array size) that formula is a no-op: the storage index
    `si = i & (N-1)` already consumes bits 0-8 of `i`, and since 256
    (2**8) divides 512 (2**9), any two global cells that collide at the
    same storage slot are congruent mod 512 and therefore ALSO congruent
    mod 256 -- i.e. `(i_a & 0xFF) == (i_b & 0xFF)` for every colliding
    pair, always. The literal formula can never detect a real collision
    at this N; caught by
    tests/test_clipmap.py::test_stamp_cross_check_catches_a_missed_clear
    failing under it. This tags the bits ABOVE what the storage index
    already captures (`i >> log2(N)`), which is the smallest change that
    makes the check detect what Ticket #14 actually asks it to detect.
    """
    shift = N.bit_length() - 1  # log2(N) for a power-of-two N
    i_tag = (i >> shift) & 0xFF
    j_tag = (j >> shift) & 0xFF
    return (i_tag << 8) | j_tag


def expected_stamp_batch(gi: np.ndarray, gj: np.ndarray, N: int) -> np.ndarray:
    """Vectorised expected_stamp -- same formula, for Ticket #18's scatter
    kernel to tag many touched cells at once without a Python loop."""
    shift = N.bit_length() - 1
    i_tag = (gi.astype(np.int64) >> shift) & 0xFF
    j_tag = (gj.astype(np.int64) >> shift) & 0xFF
    return ((i_tag << 8) | j_tag).astype(np.uint16)


# ---------------------------------------------------------------------------
# class_conf byte (Bible Part 9.3): 4 bits class | 4 bits confidence.
#
# The Bible names the confidence nibble kappa (Part 11's sparsity-derived
# confidence) -- but kappa isn't computed until Ticket #38/#39 (Phase 5).
# Build Map Ticket #19 gives this nibble an earlier, explicit, provisional
# job instead: "store the runner-up fraction in the confidence nibble."
# That is what this module implements now; whatever writes kappa later
# (#38/#39) will need to decide whether it overwrites or composes with
# this value -- flagged here rather than left implicit.
# ---------------------------------------------------------------------------

CLASS_NIBBLE_MAX = 0xF  # 4 bits -- DrishtiClass has 10 members (0-9), fits with room to spare
CONF_NIBBLE_STEPS = 0xF  # 4 bits -> 16 levels, 0..15


def encode_class_conf(class_id: int, runner_up_fraction: float) -> np.uint8:
    """Pack a winning class id and a confidence-nibble value (Ticket #19:
    the runner-up fraction) into one byte. High nibble = class, low
    nibble = confidence, quantised to 1/15 steps."""
    class_nibble = int(class_id) & CLASS_NIBBLE_MAX
    conf_nibble = int(round(min(max(runner_up_fraction, 0.0), 1.0) * CONF_NIBBLE_STEPS)) & 0xF
    return np.uint8((class_nibble << 4) | conf_nibble)


def decode_class_conf(byte: int) -> tuple[int, float]:
    """Inverse of encode_class_conf -> (class_id, confidence_fraction)."""
    class_id = (int(byte) >> 4) & 0xF
    conf_nibble = int(byte) & 0xF
    return class_id, conf_nibble / CONF_NIBBLE_STEPS


def encode_class_conf_batch(class_id: np.ndarray, runner_up_fraction: np.ndarray) -> np.ndarray:
    """Vectorised encode_class_conf for Ticket #19's scatter_class kernel."""
    class_nibble = class_id.astype(np.uint8) & CLASS_NIBBLE_MAX
    conf_nibble = (
        np.round(np.clip(runner_up_fraction, 0.0, 1.0) * CONF_NIBBLE_STEPS).astype(np.uint8) & 0xF
    )
    return ((class_nibble << 4) | conf_nibble).astype(np.uint8)


# ---------------------------------------------------------------------------
# Ticket #21 -- v2 ceiling layer fields (Bible Part 9.3: h_ceil_min,
# h_ceil_max, "sentinel = no ceiling"). NO_CEILING must decode to +inf
# clearance, never 0 -- a cell reporting zero clearance is LETHAL, and a
# bug here would silently turn "nothing above you" into "you cannot move."
# int16's minimum (-32768) can never be a real encoded height (encode_h's
# valid range is clamped to +-327.67 m, i.e. +-32767), so it is safe to
# reserve as a sentinel with no collision against real data.
# ---------------------------------------------------------------------------

NO_CEILING_SENTINEL = np.int16(-32768)


def decode_clearance(h_ceil_min_raw: int, h_max_raw: int) -> float:
    """Ground layer's h_max and ceiling layer's h_ceil_min (both raw 1cm
    fixed-point int16) -> clearance in metres. +inf when h_ceil_min_raw
    is the NO_CEILING sentinel."""
    if int(h_ceil_min_raw) == int(NO_CEILING_SENTINEL):
        return float("inf")
    return decode_h(h_ceil_min_raw) - decode_h(h_max_raw)
