"""
grid/cell.py

Fixed-point height encode/decode and the observability sub-field of the
`flags` byte (Bible Part 9.3, Part 10.1). Small pure helpers shared by
`grid/clipmap.py` (Tickets #11, #13, #14, #15).

Ticket #11 ships flat 1 cm int16 height quantisation everywhere -- the
per-level foveated quantum (1/2/4/8 cm for L0..L3) is Ticket #17, an
explicit "cheap force-in" deferred to D11 in the Build Map. Do not read
level-varying precision into encode_h/decode_h until #17 lands.
"""

from __future__ import annotations

import numpy as np

H_QUANTUM_M = 0.01  # 1 cm, flat across all levels until Ticket #17

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


def encode_h(z_m: float) -> np.int16:
    """Metres -> 1 cm fixed-point int16. round(), not truncate, so 1.234
    round-trips to 1.23 m rather than being biased toward zero."""
    return np.int16(round(z_m / H_QUANTUM_M))


def decode_h(v: int) -> float:
    """1 cm fixed-point int16 -> metres."""
    return float(v) * H_QUANTUM_M


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
