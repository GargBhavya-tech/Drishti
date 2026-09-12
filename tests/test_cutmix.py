"""
tests/test_cutmix.py

perception/cutmix.py: cluster extraction's min-points filter and
anchor-relative storage, the save/load round trip for the ragged .npz
format, and paste_rare_cluster's placement-range bound + local-ground
estimation + label-array extension.
"""

from __future__ import annotations

import numpy as np

from perception.cutmix import (
    MAX_PLACEMENT_RANGE_M,
    MIN_PLACEMENT_RANGE_M,
    RareClusterSet,
    extract_rare_class_clusters,
    load_clusters,
    paste_rare_cluster,
    save_clusters,
)
from perception.sweep import Sweep


def test_extract_skips_frames_below_min_points():
    xyzi = np.zeros((10, 4), dtype=np.float32)
    is_target = np.zeros(10, dtype=bool)
    is_target[:3] = True  # only 3 target points -- below the default min_points=5
    result = extract_rare_class_clusters([(xyzi, is_target)])
    assert result.clusters == []


def test_extract_anchors_cluster_to_its_own_min_z_and_xy_centroid():
    xyzi = np.array(
        [[0.0, 0.0, 5.0, 0.1], [2.0, 0.0, 6.0, 0.2], [1.0, 2.0, 7.0, 0.3], [1.0, -2.0, 4.0, 0.4], [1.0, 0.0, 8.0, 0.5]],
        dtype=np.float32,
    )
    is_target = np.ones(5, dtype=bool)
    result = extract_rare_class_clusters([(xyzi, is_target)])
    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    # Centroid x/y should now be ~0 (anchored), and the lowest z (was 4.0) should now be exactly 0.
    assert abs(cluster[:, 0].mean()) < 1e-4
    assert abs(cluster[:, 1].mean()) < 1e-4
    assert cluster[:, 2].min() == 0.0


def test_save_and_load_round_trip_preserves_clusters():
    clusters = RareClusterSet(
        clusters=[
            np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.5]], dtype=np.float32),
            np.array([[2.0, 2.0, 2.0, 0.3]], dtype=np.float32),
        ]
    )
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "clusters.npz")
        save_clusters(clusters, path)
        loaded = load_clusters(path)
        assert len(loaded.clusters) == 2
        np.testing.assert_allclose(loaded.clusters[0], clusters.clusters[0])
        np.testing.assert_allclose(loaded.clusters[1], clusters.clusters[1])


def test_save_and_load_empty_cluster_set():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "empty.npz")
        save_clusters(RareClusterSet(clusters=[]), path)
        loaded = load_clusters(path)
        assert loaded.clusters == []


def _flat_ground_sweep(n=200, seed=0) -> Sweep:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-20, 20, n)
    y = rng.uniform(-20, 20, n)
    z = np.full(n, -1.0)  # flat ground at z=-1
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    return Sweep(
        xyz=xyz, intensity=rng.uniform(0, 1, n).astype(np.float32),
        ring=np.full(n, -1, dtype=np.int16), timestamp=0.0, T_world=np.eye(4), sensor_id="test",
    )


def test_paste_with_empty_clusters_is_a_no_op():
    sweep = _flat_ground_sweep()
    labels = np.zeros(sweep.xyz.shape[0], dtype=np.int64)
    result_sweep, result_labels = paste_rare_cluster(sweep, labels, RareClusterSet(clusters=[]), target_class=4, rng=np.random.default_rng(0))
    assert result_sweep is sweep
    np.testing.assert_array_equal(result_labels, labels)


def test_paste_places_cluster_within_configured_range_and_extends_labels():
    sweep = _flat_ground_sweep()
    labels = np.zeros(sweep.xyz.shape[0], dtype=np.int64)
    # A small pole-like cluster, anchored (min z = 0, centroid xy = 0).
    cluster = np.array([[0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 2.0, 0.5]], dtype=np.float32)
    clusters = RareClusterSet(clusters=[cluster])

    merged_sweep, merged_labels = paste_rare_cluster(sweep, labels, clusters, target_class=4, rng=np.random.default_rng(42))

    assert merged_sweep.xyz.shape[0] == sweep.xyz.shape[0] + 3
    assert merged_labels.shape[0] == labels.shape[0] + 3
    assert np.all(merged_labels[-3:] == 4)

    pasted_xy_range = np.hypot(merged_sweep.xyz[-3:, 0], merged_sweep.xyz[-3:, 1])
    assert np.all(pasted_xy_range >= MIN_PLACEMENT_RANGE_M - 1e-6)
    assert np.all(pasted_xy_range <= MAX_PLACEMENT_RANGE_M + 1e-6)

    # The pasted cluster's lowest point (anchor z=0) should land near the
    # real local ground height (~-1.0 in this synthetic flat-ground sweep).
    assert abs(merged_sweep.xyz[-3, 2] - (-1.0)) < 0.5
