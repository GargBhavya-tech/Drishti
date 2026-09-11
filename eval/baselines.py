"""
eval/baselines.py

Ticket #56 -- four memory baselines, all at IDENTICAL extent and
IDENTICAL bytes-per-cell payload, so the comparison is honest (Build
Map's own explicit requirement) rather than four numbers computed
under four different assumptions:

1. Dense uniform 3D voxel grid (the naive "strawman" -- explicitly
   labelled as such, never the LEAD number on a slide).
2. Dense uniform 2.5D grid at 5cm (a single flat resolution, no
   foveation) -- THIS is the honest baseline DRISHTI's own headline
   ratio is measured against.
3. A sparse hash-voxel / Octomap-style structure (only occupied cells
   stored, plus a real per-entry key/pointer overhead) -- measured
   honestly even where the answer is unflattering to DRISHTI (Build
   Map's own instruction): a sparse structure can win OR lose on
   memory depending on occupancy, and losing on memory while winning
   decisively on query latency (O(1) toroidal lookup vs. a hash probe
   or octree descent) is a real, reportable result either way.
4. DRISHTI's own foveated clipmap (Ticket #11's `memory_bytes_v1()`,
   the historical reference figure this project's own headline ratio
   is quoted against; `memory_bytes_v2()` is also reported, per Ticket
   #17's real measured savings).

Watch out (Build Map's own words): "quote 16x, not 267x." The dense-3D
figure compares against something nobody would actually build (a
uniform-resolution 3D voxel grid with no foveation and no 2.5D
flattening); report it explicitly labelled as the naive baseline and
LEAD with the honest 2.5D-vs-DRISHTI comparison instead. The 267x
figure is nowhere derived exactly in the Bible/Build Map text itself
(a targeted search for "267" in the Bible turns up nothing) -- it is
cited only as an illustrative magnitude for "how bad the naive
strawman can look", not a number this module is asked to reproduce
exactly. `dense_3d_bytes()` here uses a documented, clearly-labelled
vertical extent/resolution rather than silently assuming one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from grid.clipmap import Clipmap

BYTES_PER_CELL = 12  # Bible Part 9.3's v1 per-cell payload -- the SAME for every baseline below

# Dense-3D strawman's own vertical extent/resolution, DOCUMENTED rather
# than left implicit -- 10m total height (a generous bound above and
# below a UGV's own operating envelope) at the SAME 5cm voxel size used
# horizontally, so a voxel is a cube, not an arbitrarily-chosen slab.
DENSE_3D_HEIGHT_RANGE_M = 10.0
DENSE_3D_VOXEL_SIZE_M = 0.05

# Sparse hash-voxel baseline's per-entry overhead: a real hash-map
# entry costs more than its raw payload -- a 2D integer key (8 bytes,
# packed i/j) plus a bucket/pointer overhead. This is deliberately a
# ROUGH, DOCUMENTED estimate (Octomap's own octree node overhead varies
# by implementation) rather than a fabricated "sparse always wins"
# number -- see module docstring's "measure it even if unflattering".
SPARSE_KEY_BYTES = 8
SPARSE_OVERHEAD_BYTES = 16  # bucket/pointer bookkeeping, typical for a hash-map entry


@dataclass(frozen=True)
class BaselineResult:
    name: str
    n_cells: int
    bytes_total: int

    @property
    def megabytes(self) -> float:
        return self.bytes_total / 1e6


def dense_2p5d_bytes(extent_m: float, cell_size_m: float = 0.05, bytes_per_cell: int = BYTES_PER_CELL) -> BaselineResult:
    """A single flat-resolution 2.5D grid (no foveation) covering a
    SQUARE extent_m x extent_m area at cell_size_m -- the HONEST
    baseline DRISHTI's own headline ratio (16.0x) is measured against."""
    n_side = round(extent_m / cell_size_m)
    n_cells = n_side * n_side
    return BaselineResult(name="dense_2.5d_5cm", n_cells=n_cells, bytes_total=n_cells * bytes_per_cell)


def dense_3d_bytes(
    extent_m: float,
    voxel_size_m: float = DENSE_3D_VOXEL_SIZE_M,
    height_range_m: float = DENSE_3D_HEIGHT_RANGE_M,
    bytes_per_cell: int = BYTES_PER_CELL,
) -> BaselineResult:
    """The naive strawman: a uniform 3D voxel grid over the SAME
    square extent, with NO 2.5D flattening -- every vertical slice
    stored independently. Never the lead number (module docstring)."""
    n_side = round(extent_m / voxel_size_m)
    n_z = round(height_range_m / voxel_size_m)
    n_cells = n_side * n_side * n_z
    return BaselineResult(name="dense_3d_voxel (naive strawman)", n_cells=n_cells, bytes_total=n_cells * bytes_per_cell)


def sparse_hash_voxel_bytes(
    n_occupied_cells: int,
    bytes_per_cell: int = BYTES_PER_CELL,
    key_bytes: int = SPARSE_KEY_BYTES,
    overhead_bytes: int = SPARSE_OVERHEAD_BYTES,
) -> BaselineResult:
    """A sparse hash-voxel/Octomap-style structure: only OCCUPIED cells
    are stored, each paying its raw payload PLUS a real per-entry
    key/pointer overhead (see module docstring for why this is an
    honest, documented estimate rather than a number engineered to
    make either side look good)."""
    per_entry = bytes_per_cell + key_bytes + overhead_bytes
    return BaselineResult(
        name="sparse_hash_voxel (Octomap-style)", n_cells=n_occupied_cells, bytes_total=n_occupied_cells * per_entry
    )


def drishti_bytes(cm: Clipmap, use_v2: bool = False) -> BaselineResult:
    """DRISHTI's own foveated clipmap. `use_v2=False` (default) reports
    Ticket #11's historical `memory_bytes_v1()` reference figure (the
    number this project's own headline ratios are quoted against);
    `use_v2=True` reports Ticket #17's REAL live allocation."""
    n_cells = len(cm.levels) * cm.N * cm.N
    bytes_total = cm.memory_bytes_v2() if use_v2 else cm.memory_bytes_v1()
    name = "drishti_foveated (v2, real allocation)" if use_v2 else "drishti_foveated (v1, reference)"
    return BaselineResult(name=name, n_cells=n_cells, bytes_total=bytes_total)


def occupancy_fraction(n_occupied_cells: int, n_total_cells: int) -> float:
    """Fraction of a uniform grid's cells actually occupied by one
    sweep -- Build Map's own "< 1%" acceptance number, the honest
    argument for why a uniform grid's memory is mostly spent on empty
    space DRISHTI never allocates at fine resolution in the first
    place."""
    return n_occupied_cells / n_total_cells


@dataclass(frozen=True)
class MemoryComparisonReport:
    dense_3d: BaselineResult
    dense_2p5d: BaselineResult
    sparse: BaselineResult
    drishti: BaselineResult

    @property
    def ratio_2p5d_over_drishti(self) -> float:
        return self.dense_2p5d.bytes_total / self.drishti.bytes_total

    @property
    def ratio_3d_over_drishti(self) -> float:
        return self.dense_3d.bytes_total / self.drishti.bytes_total


def compare_all(cm: Clipmap, n_occupied_cells: int, extent_m: Optional[float] = None) -> MemoryComparisonReport:
    """All four baselines at MATCHED extent -- the extent defaults to
    DRISHTI's own coarsest level's real array extent (N * c_L_max), so
    a caller need not separately compute or duplicate what "identical
    extent" means for this specific clipmap.
    """
    if extent_m is None:
        extent_m = cm.N * cm.levels[-1].cell_size_m
    return MemoryComparisonReport(
        dense_3d=dense_3d_bytes(extent_m),
        dense_2p5d=dense_2p5d_bytes(extent_m),
        sparse=sparse_hash_voxel_bytes(n_occupied_cells),
        drishti=drishti_bytes(cm, use_v2=False),
    )
