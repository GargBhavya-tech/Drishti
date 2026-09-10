"""
observability/ground_plane.py

Ticket #35 -- local ground plane fit.

Per azimuth column, fit a LINE in the r-z projection (r = horizontal
range, matching Ticket #26's own "outward" axis) to the last few
CONFIRMED ground returns from that ticket's incremental walk
(`perception.ground_prior.compute_ground_prior`). Exposes
`expected_ground_range(ring, azimuth)` for Ticket #36 to compare
measured range against.

Deliberately LOCAL, not global (Ticket #35 "Watch out"): using the last
K ground points closest to the edge of confirmed ground, rather than
every ground point in the column, is what lets the fit track a slope
instead of averaging flat and sloped terrain together -- this is the
machinery that stops Ticket #36 firing on every hill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from perception.ground_prior import compute_ground_prior
from perception.sweep import Sweep
from sensor.sensor_model import SensorConfig

DEFAULT_K_RECENT = 5


@dataclass(frozen=True)
class ColumnFit:
    slope: float      # a, in z = a*r_xy + b
    intercept: float  # b
    max_residual_m: float  # |actual z - fitted z| over the points the fit was built from


@dataclass(frozen=True)
class LocalGroundFit:
    column_fits: Dict[int, ColumnFit]
    n_azimuth_bins: int
    sm: SensorConfig


def _fit_line(r_xy: np.ndarray, z: np.ndarray) -> ColumnFit:
    """Least-squares z = a*r + b. With exactly 2 points this is the exact
    line through them (zero residual by construction); degenerate
    all-same-r columns fall back to a flat (a=0) fit through the mean."""
    if np.ptp(r_xy) < 1e-9:
        b = float(np.mean(z))
        return ColumnFit(slope=0.0, intercept=b, max_residual_m=float(np.max(np.abs(z - b))))
    a, b = np.polyfit(r_xy, z, deg=1)
    fitted = a * r_xy + b
    return ColumnFit(slope=float(a), intercept=float(b), max_residual_m=float(np.max(np.abs(z - fitted))))


def fit_local_ground_planes(
    sweep: Sweep,
    sm: SensorConfig,
    n_azimuth_bins: int = 1080,
    k_recent: int = DEFAULT_K_RECENT,
) -> LocalGroundFit:
    """One local line fit per azimuth column that has at least 2
    confirmed-ground points, built from that column's last `k_recent`
    ground returns in ascending horizontal-range order (i.e. the ones
    nearest the current edge of confirmed ground -- Ticket #26's own
    walk order, never re-sorted by ring)."""
    result = compute_ground_prior(sweep, n_azimuth_bins=n_azimuth_bins)

    x = sweep.xyz[:, 0].astype(np.float64)
    y = sweep.xyz[:, 1].astype(np.float64)
    z = sweep.xyz[:, 2].astype(np.float64)
    r_xy = np.sqrt(x * x + y * y)

    ground_idx = np.nonzero(result.is_ground)[0]
    if ground_idx.size == 0:
        return LocalGroundFit(column_fits={}, n_azimuth_bins=n_azimuth_bins, sm=sm)

    azimuth = np.arctan2(y[ground_idx], x[ground_idx])
    col = np.floor(0.5 * (1.0 - azimuth / np.pi) * n_azimuth_bins).astype(np.int64)
    col = np.clip(col, 0, n_azimuth_bins - 1)

    order = np.lexsort((r_xy[ground_idx], col))
    sorted_idx = ground_idx[order]
    sorted_col = col[order]

    group_start = np.flatnonzero(np.r_[True, sorted_col[1:] != sorted_col[:-1]])
    group_end = np.r_[group_start[1:], sorted_col.shape[0]]

    column_fits: Dict[int, ColumnFit] = {}
    for start, end in zip(group_start, group_end):
        members = sorted_idx[start:end]
        recent = members[-k_recent:]  # last K by range ascending -> nearest the walk's current edge
        if recent.shape[0] < 2:
            continue
        column_fits[int(sorted_col[start])] = _fit_line(r_xy[recent], z[recent])

    return LocalGroundFit(column_fits=column_fits, n_azimuth_bins=n_azimuth_bins, sm=sm)


def _ring_elevation_rad(ring: int, sm: SensorConfig) -> float:
    """Inverse of perception.range_image's v_fallback row assignment
    (same uniform-beam-spacing model), evaluated at the row's CENTER
    (ring + 0.5) rather than its edge, for the least-biased single-angle
    estimate per ring."""
    phi_min = sm.phi_max_rad - (sm.n_beams - 1) * sm.d_phi_rad
    fov = sm.phi_max_rad - phi_min
    return phi_min + fov * (1.0 - (ring + 0.5) / sm.n_beams)


def expected_ground_range(fit: LocalGroundFit, ring: int, azimuth_col: int) -> Optional[float]:
    """Predicted FULL 3D range (matching perception.range_image.RangeImage
    .range's own convention) at which THIS ring's beam would intersect
    this column's local ground line, or None if the column has no fit
    (too few confirmed ground returns) or the beam's elevation angle is
    parallel to the local ground slope (no intersection)."""
    column = fit.column_fits.get(azimuth_col)
    if column is None:
        return None

    phi = _ring_elevation_rad(ring, fit.sm)
    tan_phi = math.tan(phi)
    denom = tan_phi - column.slope
    if abs(denom) < 1e-9:
        return None

    r_xy = column.intercept / denom
    if r_xy <= 0:
        return None

    cos_phi = math.cos(phi)
    if abs(cos_phi) < 1e-9:
        return None
    return r_xy / cos_phi
