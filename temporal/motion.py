"""
temporal/motion.py

Ticket #45 -- map-level motion detection. Because the map is
world-anchored, motion needs NO ego compensation: compare frame t
against the map's last known state AT THE SAME WORLD CELL (global
(i, j), Ticket #10) -- never by storage index, which shifts meaning as
the window scrolls (Build Map's own "Watch out": comparing storage
indices compares different PLACES in the world after a scroll, and
generated code gets this wrong "roughly half the time").

A cell transitioning OCCUPIED -> FREE (carved by a ray, Ticket #34)
with a NEARBY (8-connected) cell transitioning FREE -> OCCUPIED in the
SAME frame indicates something moved from one to the other. Flagged
cells are meant to be excluded from Ticket #44's static accumulation
for that frame -- run motion detection before static accumulation so
the accumulator can skip flagged cells.
"""

from __future__ import annotations

from typing import Dict, Set, Tuple

from grid.cell import OBS_FREE, OBS_OCCUPIED

NEIGHBOR_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

GlobalCell = Tuple[int, int]


class MotionDetector:
    """Keyed on GLOBAL (i, j), never storage index -- see module
    docstring. Holds exactly one "last known observability" snapshot
    per cell; a cell with no fresh evidence this frame simply keeps its
    last known value (matching how observability carving itself only
    updates cells a beam actually touched), so comparisons are always
    "this frame's evidence against the most recent PRIOR evidence for
    that same world cell", not literally the immediately preceding
    call if a cell went untouched for a frame or two.
    """

    def __init__(self) -> None:
        self._last_known: Dict[GlobalCell, int] = {}

    def detect(self, current: Dict[GlobalCell, int]) -> Set[GlobalCell]:
        """`current`: {(gi, gj): observability} for every cell THIS
        frame produced fresh evidence for (Ticket #34's per-frame
        evidence, keyed by global index, at whichever level the caller
        tracks motion at -- typically the finest ring, since that is
        where a moving object's own footprint is legible).

        Returns the set of (gi, gj) flagged MOVING this frame: every
        cell that went OCCUPIED -> FREE with an 8-connected neighbour
        that went FREE -> OCCUPIED in this SAME frame. Both members of
        such a pair are flagged (the vacated cell and the arrived
        cell) -- Build Map's own "mark moving cells", plural, on both
        sides of the transition.
        """
        previous = self._last_known
        vacated: Set[GlobalCell] = set()
        arrived: Set[GlobalCell] = set()
        for key, obs in current.items():
            prev_obs = previous.get(key)
            if prev_obs is None:
                continue
            if prev_obs == OBS_OCCUPIED and obs == OBS_FREE:
                vacated.add(key)
            elif prev_obs == OBS_FREE and obs == OBS_OCCUPIED:
                arrived.add(key)

        moving: Set[GlobalCell] = set()
        for gi, gj in vacated:
            for di, dj in NEIGHBOR_OFFSETS:
                neighbor = (gi + di, gj + dj)
                if neighbor in arrived:
                    moving.add((gi, gj))
                    moving.add(neighbor)

        previous.update(current)
        return moving

    def reset(self) -> None:
        """Forget all prior evidence -- e.g. after a map re-init or a
        full-wrap scroll clear, where every storage slot's contents
        changed and any previously-remembered global cell may no
        longer correspond to anything real."""
        self._last_known.clear()
