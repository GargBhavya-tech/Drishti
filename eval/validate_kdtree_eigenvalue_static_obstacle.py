"""
eval/validate_kdtree_eigenvalue_static_obstacle.py

Corrected retest of the eigenvalue linearity/planarity idea
(perception/eigenvalue_features.py), after the first attempt's real
diagnosed failure: a 3x3 RANGE-IMAGE window, once converted to 3D,
samples a wedge-shaped neighborhood (tiny elevation spacing, growing
azimuth spacing with range) that is inherently near-linear regardless
of the real surface -- dominated by the sensor's own sampling pattern,
not the object's true shape (mean linearity 0.9343 STATIC_OBSTACLE vs.
0.9347 non-STATIC -- statistically indistinguishable). This script uses
a REAL isotropic 3D radius search (scipy.spatial.cKDTree) instead --
the neighborhood the eigenvalue-feature literature (Weinmann et al.)
actually assumes.

Real cost caveat, stated up front: a per-point KD-tree radius query
over ~100k points/frame is real compute, not free -- this is deliberately
run on FEWER frames than the range-image-window attempt to keep this
diagnostic affordable.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from perception.rellis_loader import load_rellis_labels, load_rellis_sweep
from perception.taxonomy import DrishtiClass, rellis_label_ids_to_drishti
from perception.train import build_multi_sequence_splits

RADIUS_M = 0.3  # matches perception/geometric_instance_detector.py's own CELL_SIZE_M, for consistency with the rest of this session's geometric work
MIN_NEIGHBORS = 4
MAX_POINTS_PER_FRAME = 8000  # real, measured bottleneck: a per-point Python loop over ~100k-150k raw points timed out past 120s for even ONE frame -- subsampling trades total-pixel coverage for the ability to run this at all, same "measure honestly on a representative sample" standard already used throughout this session


def _prf1(flagged: np.ndarray, is_static: np.ndarray, weight: np.ndarray) -> tuple:
    """Weighted precision/recall/F1. `weight` implements inverse-
    probability correction for the stratified subsample (see main()'s
    own comment): every real STATIC_OBSTACLE point has weight 1.0 (ALL
    are kept, no reweighting needed), but a sampled "other" point
    stands in for `1/sampling_fraction` real other points in the true
    population -- without this, keeping 100% of the rare class while
    subsampling the common class would artificially inflate precision
    relative to the TRUE class balance, which is exactly the kind of
    silently-misleading number this project's own culture argues
    against."""
    tp = float(np.sum(weight[flagged & is_static]))
    fp = float(np.sum(weight[flagged & ~is_static]))
    fn = float(np.sum(weight[~flagged & is_static]))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, int(flagged.sum())


def compute_real_3d_eigenvalue_features(
    xyz: np.ndarray, query_indices: np.ndarray, radius: float = RADIUS_M, min_neighbors: int = MIN_NEIGHBORS
):
    """Real isotropic 3D radius-search eigenvalue features, computed
    only for `query_indices` (a subsample -- see MAX_POINTS_PER_FRAME's
    own comment for why: a per-point Python loop over ALL ~100k-150k
    raw points timed out/was killed past 120s for even one frame). The
    KD-tree itself is built on the FULL point set, so a query point's
    real local neighborhood still includes real nearby points that
    weren't themselves selected as query points -- only the expensive
    per-point covariance/eigenvalue computation is subsampled, not the
    neighborhood context each computation draws from."""
    tree = cKDTree(xyz)
    query_points = xyz[query_indices]
    neighbor_lists = tree.query_ball_point(query_points, r=radius)
    del tree  # a cKDTree over ~130k points is real memory; free it as soon as the query is done, not at end-of-function scope

    n = query_indices.shape[0]
    linearity = np.zeros(n, dtype=np.float64)
    planarity = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    for i in range(n):
        neighbors = neighbor_lists[i]
        neighbor_lists[i] = None  # drop each list's own reference as soon as it's consumed, rather than holding all ~8000 lists live for the whole loop
        if len(neighbors) < min_neighbors:
            continue
        pts = xyz[neighbors]
        cov = np.cov(pts.T)
        eigvals = np.linalg.eigvalsh(cov)
        lambda3, lambda2, lambda1 = eigvals[0], eigvals[1], eigvals[2]
        lambda1_safe = max(lambda1, 1e-12)
        linearity[i] = (lambda1 - lambda2) / lambda1_safe
        planarity[i] = (lambda2 - lambda3) / lambda1_safe
        valid[i] = True

    del neighbor_lists
    gc.collect()
    return linearity, planarity, valid


def main():
    parser = argparse.ArgumentParser(description="Corrected KD-tree-based eigenvalue feature validation against real RELLIS-3D labels")
    parser.add_argument("--sequence-dir", nargs="+", default=[f"data/rellis/{i:05d}" for i in range(5)])
    parser.add_argument("--n-frames", type=int, default=15)
    args = parser.parse_args()

    dirs = [Path(p) for p in args.sequence_dir]
    train_items, _val_items, _ = build_multi_sequence_splits(dirs)
    step = max(1, len(train_items) // args.n_frames)
    sample_items = train_items[::step][: args.n_frames]

    static_class = int(DrishtiClass.STATIC_OBSTACLE)

    all_linearity, all_planarity, all_is_static, all_weight = [], [], [], []

    rng = np.random.default_rng(0)
    for sequence_dir, frame_idx in sample_items:
        sweep = load_rellis_sweep(sequence_dir, frame_idx)
        raw_labels = load_rellis_labels(sequence_dir, frame_idx)
        drishti_labels = rellis_label_ids_to_drishti(raw_labels)
        xyz = sweep.xyz.astype(np.float64)

        # Stratified subsample: keep ALL real STATIC_OBSTACLE points
        # (they're only ~0.05% of a frame -- uniform random subsampling
        # to MAX_POINTS_PER_FRAME would likely include zero or very few
        # of them, making precision/recall statistically meaningless)
        # plus a random sample of everything else, bounded to the same
        # per-frame compute budget the timeout/kill already established
        # as the real ceiling. Each sampled "other" point is given an
        # inverse-probability weight (see _prf1's own docstring) so the
        # final precision/recall reflect the TRUE class balance, not
        # the artificially-balanced sample.
        static_idx = np.nonzero(drishti_labels == static_class)[0]
        other_idx = np.nonzero(drishti_labels != static_class)[0]
        n_other = max(0, MAX_POINTS_PER_FRAME - static_idx.size)
        other_sample_size = min(n_other, other_idx.size)
        other_sample = rng.choice(other_idx, size=other_sample_size, replace=False)
        other_weight = (other_idx.size / other_sample_size) if other_sample_size > 0 else 0.0
        query_indices = np.concatenate([static_idx, other_sample])
        query_weight = np.concatenate([np.ones(static_idx.size), np.full(other_sample.size, other_weight)])

        linearity, planarity, valid = compute_real_3d_eigenvalue_features(xyz, query_indices)
        query_labels = drishti_labels[query_indices]

        all_linearity.append(linearity[valid])
        all_planarity.append(planarity[valid])
        all_is_static.append(query_labels[valid] == static_class)
        all_weight.append(query_weight[valid])
        print(f"  processed {sequence_dir}/{frame_idx}: {xyz.shape[0]} total points, "
              f"{query_indices.size} queried ({static_idx.size} real static + {other_sample.size} sampled other, "
              f"other_weight={other_weight:.2f}), {valid.sum()} valid neighborhoods")

    linearity = np.concatenate(all_linearity)
    planarity = np.concatenate(all_planarity)
    is_static = np.concatenate(all_is_static)
    weight = np.concatenate(all_weight)

    n_static = int(is_static.sum())
    n_total_weighted = float(weight.sum())
    print(f"\nSampled {len(sample_items)} real RELLIS-3D frames, {is_static.size} queried valid points "
          f"({n_total_weighted:.0f} true-population-weighted), {n_static} real STATIC_OBSTACLE points")

    print(f"\nReal class means (KD-tree, radius={RADIUS_M}m): "
          f"STATIC_OBSTACLE linearity={linearity[is_static].mean():.4f} planarity={planarity[is_static].mean():.4f}  |  "
          f"non-STATIC linearity={linearity[~is_static].mean():.4f} planarity={planarity[~is_static].mean():.4f}")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for feature_name, feature_arr in [("linearity", linearity), ("planarity", planarity)]:
        print(f"\n--- {feature_name} alone ---")
        print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'n_flagged':>10}")
        for t in thresholds:
            p, r, f1, n = _prf1(feature_arr > t, is_static, weight)
            print(f"{t:>10.2f} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {n:>10d}")

    print(f"\n--- linearity OR planarity ---")
    print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'n_flagged':>10}")
    for t in thresholds:
        flagged = (linearity > t) | (planarity > t)
        p, r, f1, n = _prf1(flagged, is_static, weight)
        print(f"{t:>10.2f} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {n:>10d}")


if __name__ == "__main__":
    main()
