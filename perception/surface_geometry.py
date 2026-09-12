"""
perception/surface_geometry.py

Per-pixel surface normal and curvature/roughness channels, computed
directly from a RangeImage's own organized (H, W) x/y/z arrays --
finite differences between neighbouring pixels in the SPHERICAL
PROJECTION's own image-space grid, not a point-cloud kNN/PCA normal
estimator (which would be far more expensive and is unnecessary here
since the range image is already a dense, organized grid).

Two correctness details a naive image-space finite-difference approach
gets wrong, both handled explicitly here:

1. AZIMUTH WRAPAROUND. Column W-1's "next column" is column 0 (the
   sensor scans a full 360 degrees) -- exactly the same wraparound
   `perception.circular_pad.CircularConv2d` already handles inside the
   network. A naive `np.diff` or `np.roll`-free neighbour difference
   would treat the seam as a real depth discontinuity and produce a
   garbage normal for every pixel in the rightmost/leftmost column.
   `np.roll` along the column axis is used instead, which wraps
   correctly by construction.

2. OCCLUSION-BOUNDARY MASKING. At a real depth discontinuity (the edge
   of a bush against the sky, a cliff edge), the two neighbouring
   pixels do NOT lie on the same physical surface, so a finite-
   difference "tangent vector" between them is meaningless -- exactly
   the pixels where a normal channel would otherwise inject noise at
   the boundaries that matter most for classification. Neighbour pairs
   whose range differs by more than `MAX_NEIGHBOR_RANGE_JUMP_M` are
   excluded from the normal/curvature computation for that pixel
   (falls back to a zero vector / zero curvature, not a fabricated one).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.range_image import RangeImage

MAX_NEIGHBOR_RANGE_JUMP_M = 1.0  # beyond this, neighbours are treated as different surfaces


@dataclass(frozen=True)
class SurfaceGeometry:
    normal_x: np.ndarray  # (H, W) float64, sensor-frame components
    normal_y: np.ndarray
    normal_z: np.ndarray
    curvature: np.ndarray  # (H, W) float64, >= 0, a discrete-Laplacian roughness proxy
    geometry_valid: np.ndarray  # (H, W) bool -- True where a real normal/curvature was computed


def _wrapped_col_diff(arr: np.ndarray) -> np.ndarray:
    """arr[:, j+1] - arr[:, j], with column W-1's neighbour wrapping to
    column 0 (real azimuth wraparound, not a discontinuity)."""
    return np.roll(arr, -1, axis=1) - arr


def _row_diff_zero_padded(arr: np.ndarray) -> np.ndarray:
    """arr[i+1, :] - arr[i, :] for i < H-1; the last row has no next
    beam, so it's given a zero difference (elevation is NOT circular --
    there is no beam beyond the top/bottom of the sensor's FOV, unlike
    azimuth)."""
    out = np.zeros_like(arr)
    out[:-1, :] = arr[1:, :] - arr[:-1, :]
    return out


def compute_surface_geometry(img: RangeImage) -> SurfaceGeometry:
    """Normal = cross(horizontal_tangent, vertical_tangent), normalised;
    curvature = discrete Laplacian of range in both image directions
    (horizontal wrapped, vertical zero-padded), a standard cheap
    roughness proxy -- flat ground gives ~0, a rock/pole/kerb edge
    gives a large value. Both are masked to `geometry_valid=False`
    (zero) wherever either input pixel is invalid or a neighbour pair
    crosses an occlusion boundary (see module docstring)."""
    valid = img.valid_mask

    dx_h = _wrapped_col_diff(img.x)
    dy_h = _wrapped_col_diff(img.y)
    dz_h = _wrapped_col_diff(img.z)

    dx_v = _row_diff_zero_padded(img.x)
    dy_v = _row_diff_zero_padded(img.y)
    dz_v = _row_diff_zero_padded(img.z)

    # Cross product of the horizontal and vertical tangent vectors.
    nx = dy_h * dz_v - dz_h * dy_v
    ny = dz_h * dx_v - dx_h * dz_v
    nz = dx_h * dy_v - dy_h * dx_v
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)

    range_h_next = np.roll(img.range, -1, axis=1)
    range_v_next = np.zeros_like(img.range)
    range_v_next[:-1, :] = img.range[1:, :]

    valid_h_next = np.roll(valid, -1, axis=1)
    valid_v_next = np.zeros_like(valid)
    valid_v_next[:-1, :] = valid[1:, :]

    no_occlusion_h = np.abs(range_h_next - img.range) <= MAX_NEIGHBOR_RANGE_JUMP_M
    no_occlusion_v = np.abs(range_v_next - img.range) <= MAX_NEIGHBOR_RANGE_JUMP_M

    geometry_valid = valid & valid_h_next & valid_v_next & no_occlusion_h & no_occlusion_v & (norm > 1e-9)

    safe_norm = np.where(norm > 1e-9, norm, 1.0)
    normal_x = np.where(geometry_valid, nx / safe_norm, 0.0)
    normal_y = np.where(geometry_valid, ny / safe_norm, 0.0)
    normal_z = np.where(geometry_valid, nz / safe_norm, 0.0)

    # Discrete Laplacian of range: |f(x+1) - 2f(x) + f(x-1)| in both
    # directions, summed. Uses the SAME wrapped/zero-padded neighbour
    # conventions as the normal computation above for consistency.
    range_h_prev = np.roll(img.range, 1, axis=1)
    lap_h = np.abs(range_h_next - 2.0 * img.range + range_h_prev)

    range_v_prev = np.zeros_like(img.range)
    range_v_prev[1:, :] = img.range[:-1, :]
    lap_v = np.abs(range_v_next - 2.0 * img.range + range_v_prev)

    curvature = np.where(geometry_valid, lap_h + lap_v, 0.0)

    return SurfaceGeometry(
        normal_x=normal_x, normal_y=normal_y, normal_z=normal_z,
        curvature=curvature, geometry_valid=geometry_valid,
    )
