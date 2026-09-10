"""
grid/addressing.py

Ticket #10 -- Addressing: world <-> index, toroidal wrap.

Pure functions, no state. Build Map Ticket #10 calls this "the highest-risk
paragraph in the document": floor division vs. truncation, negative-index
wraparound via `& (N-1)`, and row-major flat-index order are the three
places generated code silently gets this wrong. See
DRISHTI_Project_Bible_v3.md Part 8 for the worked example and the reasoning
behind toroidal addressing.
"""

from __future__ import annotations

import math
from typing import Tuple


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _assert_power_of_two(N: int) -> None:
    if not is_power_of_two(N):
        raise ValueError(f"N must be a power of two, got {N}")


def world_to_global(x: float, y: float, c_l: float) -> Tuple[int, int]:
    """World coordinates -> global integer cell index at level with cell
    size c_l. Floor division, NOT truncation -- floor(-3.2) == -4, whereas
    int(-3.2) == -3. Every point behind or left of the world origin lands
    in the wrong cell if this uses truncation instead. May return negative
    indices; that is expected and handled by `global_to_storage`.
    """
    return math.floor(x / c_l), math.floor(y / c_l)


def global_to_storage(i: int, j: int, N: int) -> Tuple[int, int]:
    """Global index -> toroidal storage index. `& (N-1)` is correct for
    negative `i`/`j` because Python integers emulate infinite two's-
    complement arithmetic for bitwise ops (e.g. -4760 & 511 == 328).
    Do NOT replace this with `%` or `abs()` -- Python's `%` happens to
    agree with `&` here, but a "simplification" to `abs(i) % N` does not,
    and silently corrupts negative-side indexing.
    """
    _assert_power_of_two(N)
    return i & (N - 1), j & (N - 1)


def flat_index(si: int, sj: int, N: int) -> int:
    """Storage index -> flat array offset. Row-major: `sj * N + si`, NOT
    `si * N + sj` -- swapping these transposes the whole map. It will
    still look like a plausible map on screen, just wrong.
    """
    return sj * N + si


def global_to_world(i: int, j: int, c_l: float) -> Tuple[float, float]:
    """Global cell index -> the cell's lower-left world corner."""
    return i * c_l, j * c_l
