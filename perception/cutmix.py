"""
perception/cutmix.py

Minority-class CutMix for a range-image-projection pipeline (NOT a
point-based network) -- real point clusters of a rare class are
extracted from real training frames, then pasted at the POINT level
into ANOTHER frame's raw sweep BEFORE re-projection, so the merged
point set goes through the exact same
`perception.range_image.project_to_range_image` occlusion/z-buffering
(nearest-return-per-pixel) every other frame does. This is materially
different from -- and more correct than -- pasting a patch directly
into an already-projected (H, W) image, which would place pasted
pixels at whatever range/x/y/z values happened to already be there
rather than the pasted object's own real geometry.

Built specifically for class 4 (STATIC_OBSTACLE): real-data validation
(`eval/validate_feature_hypotheses.py`, check C) found it is 0.051% of
all training pixels, and it stayed at EXACTLY 0.0 IoU across all 20
epochs of a fine-tune that used class weighting alone
(`checkpoints_multi_v2/training_log.jsonl`) -- reweighting an
already-vanishingly-rare class's loss contribution cannot manufacture
examples that were never there to begin with; this module manufactures
more of them directly.

Scoping decision, stated plainly: each SOURCE FRAME's entire set of
class-4 points is extracted and treated as ONE paste-able cluster
(rather than running a real point-cloud clustering algorithm like
DBSCAN to separate individual objects within a frame) -- class 4 is
rare enough (see check C) that most frames contributing a cluster
contain only one or a few nearby poles/logs anyway, and a real
clustering pass would be meaningfully more engineering for a marginal
precision gain in "cluster" boundaries that doesn't change what the
network is being shown (still real class-4 geometry, just possibly
grouped slightly differently than a human would group it).

LOCAL GROUND ESTIMATION at paste time: the target location's ground
height is estimated as the median z of ORIGINAL sweep points within
`GROUND_SEARCH_RADIUS_M` of the target (x, y) -- a real, if approximate,
local estimate, not a single global constant. Falls back to the mean z
of ALL original points only when no real points exist that close (rare,
but must not crash or silently mis-place the cluster in the air).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from perception.sweep import Sweep

GROUND_SEARCH_RADIUS_M = 1.5
MIN_PLACEMENT_RANGE_M = 3.0
MAX_PLACEMENT_RANGE_M = 15.0


@dataclass(frozen=True)
class RareClusterSet:
    """A ragged list of harvested clusters, each an (M_k, 4) array of
    [x, y, z, intensity] stored RELATIVE to the cluster's own (centroid_x,
    centroid_y, min_z) anchor -- so pasting is just "add the new anchor",
    with min_z as the anchor (not mean/centroid z) since the lowest point
    of a pole/log/rock is the one that should sit AT the target ground
    height, not float above or sink below it."""

    clusters: List[np.ndarray]  # each (M_k, 4): x, y, z, intensity (anchor-relative)


def extract_rare_class_clusters(
    frames_with_labels: List[Tuple[np.ndarray, np.ndarray]], min_points: int = 5
) -> RareClusterSet:
    """`frames_with_labels`: list of (sweep_xyz_intensity (N,4), per_point_is_target_class (N,) bool)
    pairs, ALREADY filtered to whichever frames contain the target
    class -- this function itself is dataset-agnostic (no RELLIS-specific
    loading here), a thin caller in a script wires real data in. Frames
    contributing fewer than `min_points` target-class points are
    skipped (too few points to represent real object geometry, more
    likely a stray mislabeled point than a real STATIC_OBSTACLE)."""
    clusters: List[np.ndarray] = []
    for xyzi, is_target in frames_with_labels:
        pts = xyzi[is_target]
        if pts.shape[0] < min_points:
            continue
        centroid_x = float(pts[:, 0].mean())
        centroid_y = float(pts[:, 1].mean())
        min_z = float(pts[:, 2].min())
        anchored = pts.copy()
        anchored[:, 0] -= centroid_x
        anchored[:, 1] -= centroid_y
        anchored[:, 2] -= min_z
        clusters.append(anchored.astype(np.float32))
    return RareClusterSet(clusters=clusters)


def save_clusters(clusters: RareClusterSet, path: str) -> None:
    """Ragged arrays saved as a concatenated (M_total, 4) block plus
    (n_clusters+1,) offsets -- avoids numpy's slow/deprecated
    object-array pickling for a list of variable-length arrays."""
    if not clusters.clusters:
        np.savez(path, points=np.zeros((0, 4), dtype=np.float32), offsets=np.array([0], dtype=np.int64))
        return
    offsets = np.zeros(len(clusters.clusters) + 1, dtype=np.int64)
    for i, c in enumerate(clusters.clusters):
        offsets[i + 1] = offsets[i] + c.shape[0]
    points = np.concatenate(clusters.clusters, axis=0)
    np.savez(path, points=points, offsets=offsets)


def load_clusters(path: str) -> RareClusterSet:
    data = np.load(path)
    points, offsets = data["points"], data["offsets"]
    clusters = [points[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
    return RareClusterSet(clusters=[c for c in clusters if c.shape[0] > 0])


def _estimate_local_ground_z(original_xyz: np.ndarray, target_x: float, target_y: float) -> float:
    dist_sq = (original_xyz[:, 0] - target_x) ** 2 + (original_xyz[:, 1] - target_y) ** 2
    nearby = dist_sq <= GROUND_SEARCH_RADIUS_M**2
    if not np.any(nearby):
        return float(original_xyz[:, 2].mean()) if original_xyz.shape[0] > 0 else 0.0
    return float(np.median(original_xyz[nearby, 2]))


def paste_rare_cluster(
    sweep: Sweep,
    original_labels: np.ndarray,
    clusters: RareClusterSet,
    target_class: int,
    rng: np.random.Generator,
) -> Tuple[Sweep, np.ndarray]:
    """Picks one random cluster, one random (range, azimuth) placement
    within [MIN_PLACEMENT_RANGE_M, MAX_PLACEMENT_RANGE_M), estimates the
    real local ground height there from `sweep`'s OWN points, and
    returns a NEW Sweep with the cluster's points appended (translated
    to that location) plus a matching (N,) per-point label array (the
    original real per-point labels, extended with `target_class` for
    every pasted point).

    Returns the ORIGINAL sweep/labels unchanged if `clusters` is empty
    -- "nothing to paste" is a valid, expected state (e.g. before
    `extract_rare_class_clusters` has ever been run), not an error.
    """
    if not clusters.clusters:
        return sweep, original_labels

    cluster = clusters.clusters[rng.integers(0, len(clusters.clusters))]
    placement_range = rng.uniform(MIN_PLACEMENT_RANGE_M, MAX_PLACEMENT_RANGE_M)
    azimuth = rng.uniform(0.0, 2.0 * np.pi)
    target_x = placement_range * np.cos(azimuth)
    target_y = placement_range * np.sin(azimuth)
    ground_z = _estimate_local_ground_z(sweep.xyz, target_x, target_y)

    pasted_xyz = cluster[:, :3].copy()
    pasted_xyz[:, 0] += target_x
    pasted_xyz[:, 1] += target_y
    pasted_xyz[:, 2] += ground_z
    pasted_intensity = cluster[:, 3].copy()
    pasted_ring = np.full(cluster.shape[0], -1, dtype=np.int16)

    merged_xyz = np.concatenate([sweep.xyz, pasted_xyz], axis=0).astype(np.float32)
    merged_intensity = np.concatenate([sweep.intensity, pasted_intensity], axis=0).astype(np.float32)
    merged_ring = np.concatenate([sweep.ring, pasted_ring], axis=0).astype(np.int16)
    merged_labels = np.concatenate(
        [original_labels, np.full(cluster.shape[0], target_class, dtype=original_labels.dtype)], axis=0
    )

    merged_sweep = Sweep(
        xyz=merged_xyz, intensity=merged_intensity, ring=merged_ring,
        timestamp=sweep.timestamp, T_world=sweep.T_world, sensor_id=sweep.sensor_id,
    )
    return merged_sweep, merged_labels
