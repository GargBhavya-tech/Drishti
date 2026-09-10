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
