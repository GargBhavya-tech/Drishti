"""
perception/ransac_primitives.py

Constrained RANSAC shape fitting for STATIC_OBSTACLE candidates (walls,
poles) -- the "geometric primitive" half of DRISHTI_MASTER_BIBLE.md
Part G.23's two-part build (the other half is grid/temporal_occupancy.py's
multi-frame persistence filter). Built as a genuinely different mechanism
from every prior STATIC_OBSTACLE attempt this project made (G.6's learned
segmentation, G.13's single curvature-scalar threshold, G.15's two
eigenvalue-feature variants) -- none of those fit an explicit PARAMETRIC
shape model (a plane or a cylinder) to a candidate's points; they all
scored individual points or small windows against a scalar feature.

Why VERTICALLY CONSTRAINED, not generic RANSAC: off-road walls and poles
share one real physical prior generic plane/cylinder RANSAC does not
exploit -- they stand upright. Constraining plane normals to be near-
HORIZONTAL (i.e. the plane itself stands vertically) and cylinder axes
to be aligned with the world Z axis cuts the hypothesis search space
drastically and rejects the single largest source of false positives a
constrained fit would otherwise accept: sloped/rocky TERRAIN, which is
exactly what G.13's curvature threshold and G.15's eigenvalue features
could not reliably separate from real obstacles (both are single-point
or small-window features blind to whether the LARGER shape a point
belongs to stands upright or lies flat).

Both fitters take real, unlabelled candidate points (this module does
not care where the candidate came from -- the caller decides, e.g. the
existing geometric_instance_detector's over-permissive connected-
components clusters) and return a fitted model only if enough of the
points are real inliers to a vertically-constrained shape; otherwise
None, so a caller can safely do `if fit_vertical_plane(pts): ...`.

Honest, stated scope: this is a real RANSAC implementation (random
minimal-set sampling, inlier counting, best-of-N-iterations selection),
not a wrapper around a third-party library (no `pyransac3d` or `open3d`
dependency exists in this project's requirements) -- written directly
against this module's own vertical constraint rather than adapted from
a generic unconstrained fitter, which is the whole point of the
constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Vertical-plane constraint: reject any 3-point hypothesis whose normal's
# Z component exceeds this -- e.g. sin(15 deg) ~= 0.259 means the plane's
# normal must be within 15 degrees of perfectly horizontal (i.e. the
# plane itself within 15 degrees of perfectly vertical). A real wall is
# much closer to vertical than this; sloped terrain (the real source of
# false positives named in this module's own docstring) is not.
MAX_PLANE_NORMAL_Z = 0.259

# Vertical-cylinder constraint: real off-road poles/posts/trunks fall in
# this radius range -- narrower than this is more likely noise/a single
# stray point cluster, wider is more likely a tree trunk cluster already
# well-handled by VEGETATION or a real small structure, not a "pole".
MIN_CYLINDER_RADIUS_M = 0.02
MAX_CYLINDER_RADIUS_M = 0.5
# A real pole is tall relative to its radius -- a squat, wide "cylinder"
# fit (e.g. a rock that coincidentally fits a circle in XY) is rejected
# by requiring the inliers' real Z span to exceed this multiple of the
# fitted radius, not just requiring SOME minimum absolute height (which
# would still accept a wide, squat false positive at a large radius).
MIN_HEIGHT_TO_RADIUS_RATIO = 3.0

DISTANCE_THRESHOLD_M = 0.05  # a point within this distance of the fitted plane/cylinder surface counts as an inlier
MIN_INLIER_FRACTION = 0.6    # the best hypothesis must explain at least this fraction of the candidate's own points
MAX_ITERATIONS = 200


@dataclass
class PlaneModel:
    point_on_plane: np.ndarray  # (3,)
    normal: np.ndarray          # (3,), unit vector, |normal_z| <= MAX_PLANE_NORMAL_Z
    n_inliers: int
    inlier_fraction: float


@dataclass
class CylinderModel:
    center_xy: np.ndarray  # (2,) -- axis is a vertical line through this (x, y), parallel to world Z
    radius_m: float
    z_min: float
    z_max: float
    n_inliers: int
    inlier_fraction: float


def _fit_plane_from_3_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[np.ndarray]:
    """Returns a unit normal, or None if the 3 points are (near-)collinear
    (cross product near zero -- no well-defined plane)."""
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return None
    return normal / norm


def fit_vertical_plane(
    points: np.ndarray,  # (N, 3)
    max_normal_z: float = MAX_PLANE_NORMAL_Z,
    distance_threshold_m: float = DISTANCE_THRESHOLD_M,
    min_inlier_fraction: float = MIN_INLIER_FRACTION,
    max_iterations: int = MAX_ITERATIONS,
    rng: Optional[np.random.Generator] = None,
) -> Optional[PlaneModel]:
    """Real RANSAC: repeatedly samples 3 random points, fits a plane,
    REJECTS the hypothesis outright if its normal is not near-horizontal
    (the vertical-wall constraint -- checked BEFORE counting inliers, so
    a well-fit sloped-terrain plane never even gets scored), otherwise
    counts inliers and keeps the best-scoring valid hypothesis across
    `max_iterations` tries. Returns None if no hypothesis ever reaches
    `min_inlier_fraction` of the input's own points, or if there are
    fewer than 3 points to sample from."""
    n = points.shape[0]
    if n < 3:
        return None
    rng = rng or np.random.default_rng()

    best: Optional[PlaneModel] = None
    for _ in range(max_iterations):
        idx = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]
        normal = _fit_plane_from_3_points(p1, p2, p3)
        if normal is None:
            continue
        if abs(normal[2]) > max_normal_z:
            continue  # not vertical enough -- reject before scoring, per this module's own docstring

        dist = np.abs((points - p1) @ normal)
        inliers = dist <= distance_threshold_m
        n_inliers = int(inliers.sum())
        if best is None or n_inliers > best.n_inliers:
            best = PlaneModel(
                point_on_plane=p1, normal=normal, n_inliers=n_inliers, inlier_fraction=n_inliers / n
            )

    if best is None or best.inlier_fraction < min_inlier_fraction:
        return None
    return best


def _fit_circle_from_3_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[tuple]:
    """2D circle through 3 points (algebraic method). Returns (center_xy,
    radius) or None if the points are (near-)collinear (no finite
    circle)."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    center = np.array([ux, uy])
    radius = float(np.linalg.norm(p1 - center))
    return center, radius


def fit_vertical_cylinder(
    points: np.ndarray,  # (N, 3)
    min_radius_m: float = MIN_CYLINDER_RADIUS_M,
    max_radius_m: float = MAX_CYLINDER_RADIUS_M,
    min_height_to_radius_ratio: float = MIN_HEIGHT_TO_RADIUS_RATIO,
    distance_threshold_m: float = DISTANCE_THRESHOLD_M,
    min_inlier_fraction: float = MIN_INLIER_FRACTION,
    max_iterations: int = MAX_ITERATIONS,
    rng: Optional[np.random.Generator] = None,
) -> Optional[CylinderModel]:
    """Real RANSAC, constrained to a Z-aligned axis (a real, physically-
    motivated simplification for off-road poles/posts/trunks, which
    stand upright rather than at an arbitrary tilt -- reduces the fit to
    a 2D circle problem in XY, not a general 3D cylinder fit). Rejects
    a hypothesis whose radius falls outside [min_radius_m, max_radius_m]
    BEFORE scoring (same "reject before counting inliers" discipline as
    fit_vertical_plane), and rejects a squat/wide false positive by
    requiring the real Z-span of its OWN inliers to exceed
    min_height_to_radius_ratio times the fitted radius."""
    n = points.shape[0]
    if n < 3:
        return None
    rng = rng or np.random.default_rng()
    xy = points[:, :2]

    best: Optional[CylinderModel] = None
    for _ in range(max_iterations):
        idx = rng.choice(n, size=3, replace=False)
        fit = _fit_circle_from_3_points(xy[idx[0]], xy[idx[1]], xy[idx[2]])
        if fit is None:
            continue
        center, radius = fit
        if not (min_radius_m <= radius <= max_radius_m):
            continue

        dist_to_surface = np.abs(np.linalg.norm(xy - center[None, :], axis=1) - radius)
        inliers = dist_to_surface <= distance_threshold_m
        n_inliers = int(inliers.sum())
        if n_inliers < 3:
            continue

        z_inliers = points[inliers, 2]
        z_span = float(z_inliers.max() - z_inliers.min())
        if z_span < min_height_to_radius_ratio * radius:
            continue  # too squat/wide relative to its own radius -- rejected before scoring, per this module's own docstring

        if best is None or n_inliers > best.n_inliers:
            best = CylinderModel(
                center_xy=center, radius_m=radius, z_min=float(z_inliers.min()), z_max=float(z_inliers.max()),
                n_inliers=n_inliers, inlier_fraction=n_inliers / n,
            )

    if best is None or best.inlier_fraction < min_inlier_fraction:
        return None
    return best
