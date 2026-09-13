"""
perception/knn_crf.py

Post-inference kNN CRF-style smoothing (DRISHTI_MASTER_BIBLE.md Part
G.19, item 3 of the report reviewed there). Operates on the REAL 3D
point cloud (not the (H,W) range-image grid) using a kD-tree spatial
neighborhood -- the report's own rationale: the ASPP module's large
dilated receptive fields blur fine boundaries in the compressed 64x2048
range image, but the void space between a vehicle panel and an adjacent
leaf is a sharp geometric discontinuity in real 3D Cartesian space, so
refining there (not in image space) is where that discontinuity is
still sharp.

Stated simplification, same honesty standard as this project's other
approximations: this is a MEAN-FIELD LABEL-SMOOTHING approximation of a
real dense CRF (Krähenbühl & Koltun-style), not a full CRF with learned
pairwise compatibility weights -- there is no training data or labeled
pairwise-potential model in this project to learn real compatibility
weights from, so a fixed Gaussian spatial kernel stands in for them.
Zero retraining required either way -- this only ever touches already-
computed per-point class probabilities.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_K = 10
DEFAULT_N_ITERS = 3
DEFAULT_SPATIAL_SIGMA_M = 0.5
DEFAULT_MIX = 0.5  # how much of the neighbor-smoothed distribution replaces the original per iteration


def knn_crf_refine(
    xyz: np.ndarray,
    probs: np.ndarray,
    k: int = DEFAULT_K,
    n_iters: int = DEFAULT_N_ITERS,
    spatial_sigma_m: float = DEFAULT_SPATIAL_SIGMA_M,
    mix: float = DEFAULT_MIX,
) -> np.ndarray:
    """`xyz`: (N, 3) real point coordinates. `probs`: (N, C) per-point
    class probabilities (e.g. softmax of the segmentation logits at
    each point's own projected pixel). Returns a NEW (N, C) refined
    probability array; `probs` itself is never mutated.

    Each iteration: for every point, take a Gaussian-spatial-distance-
    weighted average of its k nearest neighbors' CURRENT probability
    vectors, then blend `mix` of that neighbor average into the point's
    own distribution (`(1-mix)*own + mix*neighbor_avg`), renormalising.
    Iterating lets a correction propagate a few hops rather than only
    ever seeing 1-hop neighbors, at the cost of `n_iters` kD-tree passes.

    N must be > k for this to do anything meaningful; if N <= k, k is
    clamped to N-1 (no crash on tiny point sets, but no real smoothing
    either -- a real, expected degenerate case, not silently wrong).
    """
    n = xyz.shape[0]
    if n <= 1:
        return probs.copy()
    k_eff = min(k, n - 1)
    tree = cKDTree(xyz)
    # query_ball / knn: k_eff+1 because the nearest neighbor of a point
    # is itself (distance 0) -- dropped below.
    dists, idx = tree.query(xyz, k=k_eff + 1)

    current = probs.astype(np.float64).copy()
    for _ in range(n_iters):
        neighbor_dists = dists[:, 1:]
        neighbor_idx = idx[:, 1:]
        weights = np.exp(-(neighbor_dists**2) / (2.0 * spatial_sigma_m**2))
        weight_sums = weights.sum(axis=1, keepdims=True)
        weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)
        weights_norm = weights / weight_sums

        neighbor_probs = current[neighbor_idx]  # (N, k_eff, C)
        neighbor_avg = np.einsum("nk,nkc->nc", weights_norm, neighbor_probs)

        blended = (1.0 - mix) * current + mix * neighbor_avg
        row_sums = blended.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        current = blended / row_sums

    return current
