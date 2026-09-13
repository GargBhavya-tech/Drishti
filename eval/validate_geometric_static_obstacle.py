"""
eval/validate_geometric_static_obstacle.py

Validates a curvature-threshold STATIC_OBSTACLE flag against REAL
RELLIS-3D ground truth labels, BEFORE picking a threshold -- same
"measure, don't guess" discipline as every other real number in this
project. Builds on Part G.2's already-validated finding (real, measured
curvature means: DRIVABLE 0.20, VEGETATION 1.22, NON_TRAVERSABLE 1.51,
STATIC_OBSTACLE 2.83) by sweeping real candidate thresholds and
reporting REAL precision/recall/F1 against real per-point ground truth
labels, aggregated across many real frames -- not a single anecdote.

Directly motivated by this session's geometric-detector RELLIS result:
STATIC_OBSTACLE (class 4) was massively over-detected by the LEARNED
segmentation head feeding the clustering pipeline (152 decoded vs. 3
real, in one 40-frame sample) -- the same class that has held at
exactly 0.0 IoU across three separate training runs (Part G.6). This
script asks a genuinely different question: does the RAW GEOMETRY
(curvature alone, no learned classifier at all) separate real
STATIC_OBSTACLE points from everything else well enough to serve as a
geometry-only fallback -- the same "bypass the network, derive from
geometry" pattern negative-obstacle detection already uses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from perception.ground_prior import compute_ground_prior
from perception.rellis_loader import load_rellis_labels, load_rellis_sweep
from perception.range_image import project_to_range_image
from perception.surface_geometry import compute_surface_geometry
from perception.taxonomy import DrishtiClass, rellis_label_ids_to_drishti
from perception.train import build_multi_sequence_splits
from sensor.sensor_model import load_sensor_config


def main():
    parser = argparse.ArgumentParser(description="Validate a curvature-threshold STATIC_OBSTACLE flag against real RELLIS-3D labels")
    parser.add_argument("--sequence-dir", nargs="+", default=[f"data/rellis/{i:05d}" for i in range(5)])
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--n-frames", type=int, default=60)
    args = parser.parse_args()

    dirs = [Path(p) for p in args.sequence_dir]
    train_items, _val_items, _ = build_multi_sequence_splits(dirs)
    step = max(1, len(train_items) // args.n_frames)
    sample_items = train_items[::step][: args.n_frames]

    sm = load_sensor_config(args.sensor_config)
    static_class = int(DrishtiClass.STATIC_OBSTACLE)

    all_curvature = []
    all_is_static = []
    all_valid = []

    for sequence_dir, frame_idx in sample_items:
        sweep = load_rellis_sweep(sequence_dir, frame_idx)
        raw_labels = load_rellis_labels(sequence_dir, frame_idx)
        drishti_labels = rellis_label_ids_to_drishti(raw_labels)

        img = project_to_range_image(sweep, sm)
        geom = compute_surface_geometry(img)

        touched = img.point_index >= 0
        pixel_gt = np.full(img.valid_mask.shape, -1, dtype=np.int64)
        pixel_gt[touched] = drishti_labels[img.point_index[touched]]

        all_curvature.append(geom.curvature[touched])
        all_is_static.append((pixel_gt[touched] == static_class))
        all_valid.append(geom.geometry_valid[touched])

    curvature = np.concatenate(all_curvature)
    is_static = np.concatenate(all_is_static)
    geom_valid = np.concatenate(all_valid)

    # Only evaluate where surface geometry was actually computable
    # (geometry_valid -- see perception/surface_geometry.py's own
    # occlusion-boundary masking) -- an invalid/masked curvature value
    # is meaningless, not a real negative.
    curvature = curvature[geom_valid]
    is_static = is_static[geom_valid]

    n_static = int(is_static.sum())
    n_total = is_static.size
    print(f"Sampled {len(sample_items)} real RELLIS-3D frames, {n_total} valid-geometry pixels, {n_static} real STATIC_OBSTACLE pixels ({n_static/n_total:.4%})")

    thresholds = [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5, 4.0]
    print(f"\n{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'n_flagged':>10}")
    best_f1 = -1.0
    best_threshold = None
    for t in thresholds:
        flagged = curvature > t
        tp = int(np.sum(flagged & is_static))
        fp = int(np.sum(flagged & ~is_static))
        fn = int(np.sum(~flagged & is_static))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"{t:>10.2f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {int(flagged.sum()):>10d}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"\nBest threshold by F1 (real data): {best_threshold} (F1={best_f1:.4f})")


if __name__ == "__main__":
    main()
