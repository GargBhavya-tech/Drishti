"""
observability/observe.py

Ticket #34 -- four-state observability and carving.

Combines every beam's DDA trace (Ticket #33, `observability.raycast`)
for one frame into per-cell observability evidence -- FREE (a beam
passed through), OCCUPIED (a beam terminated there, a real return),
OCCLUDED (past a termination, inferred shadow) -- and writes it into a
Clipmap via `Clipmap.mark_observability()`. Cells no beam touches this
frame are left untouched (`UNOBSERVED` only ever applies to a cell that
has NEVER been written, by construction -- Bible Part 10.1: it is the
zero value and is never actively re-asserted).

Free-space carving: a cell's PREVIOUS state -- even OCCUPIED, from a
stale or since-moved object -- is overwritten the instant fresh FREE
evidence (a beam demonstrably passing through it) arrives this frame.
There is no separate decay counter; the overwrite in
`Clipmap.mark_observability` IS the decay.

Precedence when multiple beams disagree about the same cell within ONE
frame: OCCUPIED > FREE > OCCLUDED. OCCUPIED and FREE are both direct
evidence (a beam actually reached, or passed through, that exact
point); OCCLUDED is only a geometric inference ("probably blocked"), so
any real evidence from a different beam overrides it -- this is also
what keeps occlusion carving (below) from ever clobbering a genuine
return that a different beam saw in the same shadowed region.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

from grid.cell import OBS_FREE, OBS_OCCLUDED, OBS_OCCUPIED
from grid.clipmap import Clipmap
from observability.raycast import trace_beam

_PRECEDENCE = {OBS_OCCUPIED: 2, OBS_FREE: 1, OBS_OCCLUDED: 0}

DEFAULT_OCCLUSION_RANGE_M = 10.0


def carve_frame(
    cm: Clipmap,
    ego_x: float,
    ego_y: float,
    beams: Iterable[Tuple[float, float, bool]],
    occlusion_range_m: float = DEFAULT_OCCLUSION_RANGE_M,
) -> None:
    """One frame's beams -> observability writes into `cm`.

    `beams`: iterable of (end_x, end_y, has_return). `has_return=True`
    means (end_x, end_y) is a genuine LiDAR return -- its cell becomes
    OCCUPIED, and occlusion is cast `occlusion_range_m` further along
    the same direction (coarsest level only, matching Ticket #33's
    coarse-first budget -- casting a fine-resolution shadow behind every
    beam would reintroduce the same millions-of-cells problem #33 exists
    to avoid). `has_return=False` means (end_x, end_y) is just the
    sensor's max-range cutoff with nothing detected -- the whole beam
    contributes FREE only; nothing terminates there and no shadow is
    cast (there is no object to cast one).
    """
    evidence: Dict[Tuple[int, int, int], int] = {}

    def offer(level: int, gi: int, gj: int, state: int) -> None:
        key = (level, gi, gj)
        current = evidence.get(key)
        if current is None or _PRECEDENCE[state] > _PRECEDENCE[current]:
            evidence[key] = state

    for end_x, end_y, has_return in beams:
        traced = trace_beam(ego_x, ego_y, end_x, end_y, cm.levels)
        for level, hits in traced.items():
            for hit in hits:
                if hit.is_terminal and has_return:
                    offer(level, hit.gi, hit.gj, OBS_OCCUPIED)
                else:
                    offer(level, hit.gi, hit.gj, OBS_FREE)

        if has_return and occlusion_range_m > 0:
            dist = math.hypot(end_x - ego_x, end_y - ego_y)
            if dist > 0:
                ux, uy = (end_x - ego_x) / dist, (end_y - ego_y) / dist
                shadow_dist = dist + occlusion_range_m
                shadow_x = ego_x + ux * shadow_dist
                shadow_y = ego_y + uy * shadow_dist
                # Re-trace from the SENSOR (not the wall) out to the
                # shadow's far edge, using the same coarse/fine tiering
                # trace_beam used for the main ray -- lookup() always
                # resolves a world point through the FINEST level whose
                # WINDOW contains it, so occlusion must be written at
                # that same level or it would be invisible to lookup()
                # even though the coarser level correctly holds it.
                # Every cell in this extended trace is offered as
                # OCCLUDED; precedence (below) automatically protects
                # this beam's own OCCUPIED/FREE cells and any other
                # beam's real evidence for the same cells, regardless of
                # which offer happens to run first.
                shadow_traced = trace_beam(ego_x, ego_y, shadow_x, shadow_y, cm.levels)
                for level, hits in shadow_traced.items():
                    for hit in hits:
                        offer(level, hit.gi, hit.gj, OBS_OCCLUDED)

    for (level, gi, gj), state in evidence.items():
        cm.mark_observability(level, gi, gj, state)
