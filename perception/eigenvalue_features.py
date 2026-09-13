"""
perception/eigenvalue_features.py

Real 3D structure-tensor eigenvalue features (linearity, planarity,
scattering -- the standard normalized-eigenvalue point-cloud features,
e.g. Weinmann et al.), computed from a LOCAL WINDOW of real 3D points
around each range-image pixel. Built as a genuinely different feature
from perception/surface_geometry.py's own `curvature` (a 2-neighbor
discrete-Laplacian-of-range approximation), which was tested against
real RELLIS-3D ground truth and FAILED as a STATIC_OBSTACLE predictor
(precision never exceeded 0.09% at any threshold -- see
DRISHTI_MASTER_BIBLE.md Part G.13 and
eval/validate_geometric_static_obstacle.py) despite a real, validated
class-mean separation (Part G.2) -- the per-pixel distributions
overlapped too much for a single scalar to separate.

Eigenvalue features are a genuinely different, higher-dimensional
signal: for each pixel's local window of real 3D points, compute the
3x3 covariance matrix and its eigenvalues (lambda1 >= lambda2 >= lambda3):

  linearity  = (lambda1 - lambda2) / lambda1   -- high for poles/trunks (1D structure)
  planarity  = (lambda2 - lambda3) / lambda1   -- high for walls/fences (2D structure)
  scattering = lambda3 / lambda1                -- high for rough/scattered terrain (3D, no structure)

This is NOT tested/validated as a working detector by this module
itself -- see eval/validate_eigenvalue_static_obstacle.py for the real
precision/recall sweep against real ground truth, following the exact
same "measure before shipping" discipline the curvature attempt used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.range_image import RangeImage
from perception.surface_geometry import MAX_NEIGHBOR_RANGE_JUMP_M

WINDOW_RADIUS = 1  # 3x3 window (row +-1, col +-1, with circular wrap on col) -- kept small deliberately: a larger window costs more compute per pixel and starts averaging across real object boundaries, defeating the purpose of a LOCAL structure estimate


@dataclass
class EigenvalueFeatures:
    linearity: np.ndarray  # (H, W)
    planarity: np.ndarray  # (H, W)
    scattering: np.ndarray  # (H, W)
    valid: np.ndarray  # (H, W) bool -- True where a real, occlusion-consistent local window existed


def compute_eigenvalue_features(img: RangeImage) -> EigenvalueFeatures:
    """Computes real per-pixel eigenvalue features from a 3x3 local
    window of real 3D points (img.x/y/z), circular on the azimuth (W)
    axis (same 360-degree wraparound convention as every other conv/
    neighbor operation in this project), zero-padded on the elevation
    (H) axis (a real FOV boundary, not a wraparound seam)."""
    H, W = img.valid_mask.shape
    xyz = np.stack([img.x, img.y, img.z], axis=-1)  # (H, W, 3)

    offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    stacked_valid = np.zeros((H, W, len(offsets)), dtype=bool)
    stacked_xyz = np.zeros((H, W, len(offsets), 3), dtype=np.float64)

    for k, (dr, dc) in enumerate(offsets):
        shifted_xyz = np.roll(xyz, shift=(-dr, -dc), axis=(0, 1))
        shifted_valid = np.roll(img.valid_mask, shift=(-dr, -dc), axis=(0, 1))
        shifted_range = np.roll(img.range, shift=(-dr, -dc), axis=(0, 1))

        row_in_bounds = np.ones((H, W), dtype=bool)
        if dr == -1:
            row_in_bounds[0, :] = False  # top row has no row-1 neighbor -- real FOV boundary, not circular
        elif dr == 1:
            row_in_bounds[-1, :] = False

        no_occlusion_jump = np.abs(shifted_range - img.range) <= MAX_NEIGHBOR_RANGE_JUMP_M

        stacked_valid[:, :, k] = shifted_valid & img.valid_mask & row_in_bounds & no_occlusion_jump
        stacked_xyz[:, :, k, :] = shifted_xyz

    n_valid_neighbors = stacked_valid.sum(axis=-1)
    has_enough_points = n_valid_neighbors >= 3  # need at least 3 real points for a meaningful 3x3 covariance

    safe_mask = stacked_valid[..., None].astype(np.float64)
    masked_xyz = stacked_xyz * safe_mask
    mean_xyz = masked_xyz.sum(axis=2) / np.maximum(n_valid_neighbors[..., None], 1)

    centered = (stacked_xyz - mean_xyz[:, :, None, :]) * safe_mask
    cov = np.einsum("hwki,hwkj->hwij", centered, centered) / np.maximum(n_valid_neighbors[..., None, None], 1)

    eigvals = np.linalg.eigvalsh(cov)  # ascending order: lambda3, lambda2, lambda1
    lambda3, lambda2, lambda1 = eigvals[..., 0], eigvals[..., 1], eigvals[..., 2]
    lambda1_safe = np.maximum(lambda1, 1e-12)

    linearity = (lambda1 - lambda2) / lambda1_safe
    planarity = (lambda2 - lambda3) / lambda1_safe
    scattering = lambda3 / lambda1_safe

    valid = has_enough_points & img.valid_mask

    return EigenvalueFeatures(linearity=linearity, planarity=planarity, scattering=scattering, valid=valid)
