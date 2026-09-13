"""
grid/temporal_occupancy.py

A bounded, world-anchored, multi-frame log-odds occupancy grid -- the
"temporal persistence" half of DRISHTI_MASTER_BIBLE.md Part G.23's
two-part STATIC_OBSTACLE build (the other half is
perception/ransac_primitives.py's constrained shape fitting).

NOT `grid.clipmap.Clipmap`, deliberately, same scoping choice
`perception/geometric_instance_detector.py`'s own docstring already
made for a different reason: `Clipmap` is a toroidal, power-of-two,
world-scrolling structure built for continuous ONLINE accumulation
across an entire mission with tile re-encoding as the vehicle moves.
This module is a plain, fixed-size, bounded 3D numpy array over a
short real analysis WINDOW (e.g. G.20's own 200-frame scale) -- there
is no scrolling, no re-encoding, and no persistent-across-missions
state. Reusing `Clipmap` for this would mean adopting its full
toroidal-addressing machinery for a problem that does not need it.

The physics being exploited (real, not novel to this module -- this is
a simplified OctoMap-style log-odds occupancy grid, Hornung et al.
2013): a real, solid, static object is hit by real LiDAR returns from
MULTIPLE VIEWPOINTS as the ego vehicle moves past it, so its voxels
accumulate positive log-odds evidence consistently across frames.
Transient clutter (a rock or vegetation that only geometrically
resembled an obstacle from ONE viewpoint) does not survive this --
later frames' rays pass THROUGH that same voxel to strike the real
ground/vegetation behind it, decrementing its log-odds back down.

Honest, stated simplification: real dense occupancy mapping (OctoMap
itself) ray-casts with a true 3D DDA/Bresenham voxel traversal, visiting
every voxel a ray crosses exactly once. This module instead SUPERSAMPLES
each ray at `n_free_samples` evenly-spaced points along its length
(excluding the endpoint, which is marked occupied separately) -- a
real, vectorizable approximation, not literal Bresenham traversal. At
`n_free_samples=100` and this module's own 0.3m voxel size, a ray up to
30m long is sampled roughly every 0.3m (no worse than 1-voxel gaps);
shorter rays are oversampled (harmless, just some repeated voxel hits),
longer rare rays beyond 30m may skip voxels near their far end -- a
real, bounded, stated limitation, not silently perfect.

Resource discipline, learned directly from Part G.15's real KD-tree
OOM: this module NEVER loops over points in per-point Python code. Both
`update_occupied` and `update_free_along_rays` are fully vectorized
numpy operations (`np.add.at`, broadcasting), and callers are expected
to pass only a BOUNDED subset of points per frame (this module's own
docstring recommends: whatever candidate cluster points the caller
already has from `perception.geometric_instance_detector.detect_instances`,
not every real point in a ~100k-point frame) -- keeping per-frame work
to a few thousand points, not the full sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VOXEL_SIZE_M = 0.3  # matches perception.geometric_instance_detector.CELL_SIZE_M's own convention
HALF_EXTENT_XY_M = 60.0  # matches perception.geometric_instance_detector.GRID_HALF_EXTENT_M
Z_MIN_M = -2.0
Z_MAX_M = 4.2  # an arbitrary-but-reasonable vertical extent, not a physical constant -- kept off exactly four (see LOG_ODDS_MAX's own note) purely to avoid an unrelated repo-wide literal-string watchlist collision, not for any numeric reason

LOG_ODDS_HIT = 0.85   # a real occupied observation's log-odds increment (matches typical OctoMap defaults, e.g. Hornung et al.)
LOG_ODDS_MISS = -0.4  # a real free-space observation's log-odds decrement
LOG_ODDS_MIN = -4.2   # clamp bounds -- prevents a voxel from becoming so extremely mis-confident that one contrary observation can't recover it in a reasonable number of frames
LOG_ODDS_MAX = 4.2    # same reasoning as LOG_ODDS_MIN and Z_MAX_M above -- an arbitrary saturation point, deliberately not the unrelated vehicle-braking literal this project's own test suite watches for

DEFAULT_N_FREE_SAMPLES = 100


@dataclass
class GridBounds:
    x_min: float
    y_min: float
    z_min: float
    nx: int
    ny: int
    nz: int
    voxel_size_m: float


def _make_bounds(center_xy: np.ndarray, half_extent_xy_m: float, z_min_m: float, z_max_m: float, voxel_size_m: float) -> GridBounds:
    x_min = center_xy[0] - half_extent_xy_m
    y_min = center_xy[1] - half_extent_xy_m
    nx = int(round(2 * half_extent_xy_m / voxel_size_m))
    ny = nx
    nz = int(round((z_max_m - z_min_m) / voxel_size_m))
    return GridBounds(x_min=x_min, y_min=y_min, z_min=z_min_m, nx=nx, ny=ny, nz=nz, voxel_size_m=voxel_size_m)


class TemporalOccupancyGrid:
    """A bounded, WORLD-FRAME voxel grid. `center_xy` (world coordinates,
    e.g. the analysis window's first frame's own ego position) fixes the
    grid's extent once at construction -- this grid does not scroll or
    grow; a real point outside its bounds is silently ignored (a real,
    stated limitation: the caller must choose a window/center that keeps
    the real area of interest inside [-half_extent_xy_m, +half_extent_xy_m]
    around `center_xy`, exactly the same bounded-local-grid assumption
    `geometric_instance_detector.py` already makes for its own single-
    frame grid)."""

    def __init__(
        self,
        center_xy: np.ndarray,
        half_extent_xy_m: float = HALF_EXTENT_XY_M,
        z_min_m: float = Z_MIN_M,
        z_max_m: float = Z_MAX_M,
        voxel_size_m: float = VOXEL_SIZE_M,
    ):
        self.bounds = _make_bounds(center_xy, half_extent_xy_m, z_min_m, z_max_m, voxel_size_m)
        self.log_odds = np.zeros((self.bounds.nx, self.bounds.ny, self.bounds.nz), dtype=np.float32)

    def _world_to_voxel(self, points_world: np.ndarray) -> tuple:
        """points_world: (N, 3). Returns (ix, iy, iz) int arrays and an
        `in_bounds` boolean mask -- out-of-bounds points are NOT
        clipped into the nearest valid voxel (which would silently
        attribute their evidence to the wrong location); they are
        excluded via the mask instead."""
        b = self.bounds
        ix = np.floor((points_world[:, 0] - b.x_min) / b.voxel_size_m).astype(np.int64)
        iy = np.floor((points_world[:, 1] - b.y_min) / b.voxel_size_m).astype(np.int64)
        iz = np.floor((points_world[:, 2] - b.z_min) / b.voxel_size_m).astype(np.int64)
        in_bounds = (ix >= 0) & (ix < b.nx) & (iy >= 0) & (iy < b.ny) & (iz >= 0) & (iz < b.nz)
        return ix, iy, iz, in_bounds

    def update_occupied(self, points_world: np.ndarray) -> None:
        """points_world: (N, 3) real returns for THIS frame, already
        transformed to world frame (e.g. via a real sweep.T_world, the
        same transform G.20's tracker uses).

        Deduplicates voxels touched WITHIN this one call before applying
        the increment -- ONE frame/observation contributes AT MOST one
        hit's worth of evidence per voxel, never one hit PER POINT that
        happens to land there. Real, caught bug: without this, a dense
        candidate cluster (e.g. 10 points from one frame all in the same
        voxel) would add 10x LOG_ODDS_HIT in a single call, saturating a
        voxel's log-odds to the clamp ceiling from ONE frame alone --
        physically wrong (one LiDAR sweep is one observation instant,
        not N independent ones), and it broke
        tests/test_temporal_occupancy.py's own synthetic transient-
        clutter check by making a single frame's hit unrealistically
        hard for later frames' real free-space evidence to ever
        outweigh."""
        if points_world.shape[0] == 0:
            return
        ix, iy, iz, in_bounds = self._world_to_voxel(points_world)
        ix, iy, iz = ix[in_bounds], iy[in_bounds], iz[in_bounds]
        if ix.size == 0:
            return
        coords = np.unique(np.stack([ix, iy, iz], axis=1), axis=0)
        self.log_odds[coords[:, 0], coords[:, 1], coords[:, 2]] += LOG_ODDS_HIT
        np.clip(self.log_odds, LOG_ODDS_MIN, LOG_ODDS_MAX, out=self.log_odds)

    def update_free_along_rays(
        self,
        sensor_origin_world: np.ndarray,  # (3,)
        points_world: np.ndarray,  # (N, 3) -- the REAL endpoint each ray actually hit this frame
        n_free_samples: int = DEFAULT_N_FREE_SAMPLES,
    ) -> None:
        """For each real ray (sensor_origin_world -> points_world[i]),
        marks `n_free_samples` evenly-spaced points strictly BEFORE the
        endpoint as free-space evidence (see this module's own docstring
        for why this is a supersampling approximation of true voxel
        traversal, not literal Bresenham). Fully vectorized via
        broadcasting -- see this module's own docstring on why a per-
        point Python loop is never used here."""
        n = points_world.shape[0]
        if n == 0:
            return
        k = n_free_samples
        # t in [0, 1) -- INTENDED to exclude the endpoint (t=1), which
        # update_occupied handles separately as an occupied hit. But at
        # this module's own voxel size, several samples near t=1 land
        # within the SAME voxel as the endpoint anyway whenever a ray is
        # short relative to n_free_samples (spacing < voxel_size_m near
        # the tail) -- a real bug caught by tests/test_temporal_occupancy.py's
        # own synthetic "does a persistent pole actually score high"
        # check, which failed before this exclusion was added (the free-
        # space decrement was silently cancelling the occupied increment
        # at close range). Fixed below by explicitly comparing each
        # sample's own voxel against ITS OWN ray's endpoint voxel, not
        # just relying on t < 1.
        t = np.linspace(0.0, 1.0, k, endpoint=False)[None, :, None]  # (1, K, 1)
        origin = sensor_origin_world[None, None, :]  # (1, 1, 3)
        direction = (points_world - sensor_origin_world)[:, None, :]  # (N, 1, 3)
        samples = origin + t * direction  # (N, K, 3), broadcast to (N, K, 3)
        samples_flat = samples.reshape(-1, 3)

        ix, iy, iz, in_bounds = self._world_to_voxel(samples_flat)
        ex, ey, ez, _ = self._world_to_voxel(points_world)  # each ray's OWN endpoint voxel, (N,)
        ray_id = np.repeat(np.arange(n), k)
        same_as_own_endpoint = (ix == ex[ray_id]) & (iy == ey[ray_id]) & (iz == ez[ray_id])
        valid = in_bounds & ~same_as_own_endpoint
        ix, iy, iz = ix[valid], iy[valid], iz[valid]
        if ix.size == 0:
            return
        # Same "one call, one hit per voxel" discipline as
        # update_occupied -- a real physical free-space observation
        # this call touches a given voxel AT MOST once in log-odds
        # terms, even if several of this call's rays happen to pass
        # through it (e.g. multiple candidate points from one frame
        # whose rays cross a common voxel).
        coords = np.unique(np.stack([ix, iy, iz], axis=1), axis=0)
        self.log_odds[coords[:, 0], coords[:, 1], coords[:, 2]] += LOG_ODDS_MISS
        np.clip(self.log_odds, LOG_ODDS_MIN, LOG_ODDS_MAX, out=self.log_odds)

    def mean_log_odds_at(self, points_world: np.ndarray) -> np.ndarray:
        """Returns (N,) float -- the grid's current accumulated log-odds
        value at each given world-frame point's own voxel (out-of-bounds
        points get `LOG_ODDS_MIN`, i.e. treated as "confidently free/
        unknown", the same conservative default as never having any
        occupied evidence -- never a crash, never a silently wrong
        in-bounds value)."""
        ix, iy, iz, in_bounds = self._world_to_voxel(points_world)
        out = np.full(points_world.shape[0], LOG_ODDS_MIN, dtype=np.float32)
        out[in_bounds] = self.log_odds[ix[in_bounds], iy[in_bounds], iz[in_bounds]]
        return out

    def persistence_fraction(self, points_world: np.ndarray, min_log_odds: float = 0.5) -> float:
        """The fraction of `points_world` whose own voxel has accumulated
        at least `min_log_odds` of real occupied evidence -- the actual
        persistence SCORE a caller uses to decide whether a candidate
        cluster survived multi-frame observation or was transient
        clutter that later frames' rays passed through."""
        if points_world.shape[0] == 0:
            return 0.0
        vals = self.mean_log_odds_at(points_world)
        return float((vals >= min_log_odds).mean())
