"""
eval/validate_eigenvalue_static_obstacle.py

Validates perception/eigenvalue_features.py's linearity/planarity
features against REAL RELLIS-3D ground truth, using the IDENTICAL
methodology eval/validate_geometric_static_obstacle.py already used for
curvature (which failed: precision never exceeded 0.09% at any
threshold -- Part G.13). This tests whether a genuinely different,
higher-dimensional feature (3D structure-tensor eigenvalues, not a
2-neighbor curvature scalar) does better -- measured, not assumed.

Reports precision/recall/F1 for THREE candidate rules:
1. linearity alone (pole/trunk-like structure)
2. planarity alone (wall/fence-like structure)
3. linearity OR planarity (either structured-and-vertical shape)
-- swept over real threshold candidates, same as the curvature test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from perception.eigenvalue_features import compute_eigenvalue_features
from perception.rellis_loader import load_rellis_labels, load_rellis_sweep
from perception.range_image import project_to_range_image
from perception.taxonomy import DrishtiClass, rellis_label_ids_to_drishti
from perception.train import build_multi_sequence_splits
from sensor.sensor_model import load_sensor_config


def _prf1(flagged: np.ndarray, is_static: np.ndarray) -> tuple:
    tp = int(np.sum(flagged & is_static))
    fp = int(np.sum(flagged & ~is_static))
    fn = int(np.sum(~flagged & is_static))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, int(flagged.sum())


def main():
    parser = argparse.ArgumentParser(description="Validate eigenvalue linearity/planarity as a STATIC_OBSTACLE flag against real RELLIS-3D labels")
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

    all_linearity, all_planarity, all_is_static, all_valid = [], [], [], []

    for sequence_dir, frame_idx in sample_items:
        sweep = load_rellis_sweep(sequence_dir, frame_idx)
        raw_labels = load_rellis_labels(sequence_dir, frame_idx)
        drishti_labels = rellis_label_ids_to_drishti(raw_labels)

        img = project_to_range_image(sweep, sm)
        feats = compute_eigenvalue_features(img)

        touched = img.point_index >= 0
        pixel_gt = np.full(img.valid_mask.shape, -1, dtype=np.int64)
        pixel_gt[touched] = drishti_labels[img.point_index[touched]]

        all_linearity.append(feats.linearity[touched])
        all_planarity.append(feats.planarity[touched])
        all_is_static.append(pixel_gt[touched] == static_class)
        all_valid.append(feats.valid[touched])

    linearity = np.concatenate(all_linearity)
    planarity = np.concatenate(all_planarity)
    is_static = np.concatenate(all_is_static)
    valid = np.concatenate(all_valid)

    linearity, planarity, is_static = linearity[valid], planarity[valid], is_static[valid]

    n_static = int(is_static.sum())
    n_total = is_static.size
    print(f"Sampled {len(sample_items)} real RELLIS-3D frames, {n_total} valid pixels, {n_static} real STATIC_OBSTACLE pixels ({n_static/n_total:.4%})")

    print(f"\nReal class means (sanity check against ground truth): "
          f"STATIC_OBSTACLE linearity={linearity[is_static].mean():.4f} planarity={planarity[is_static].mean():.4f}  |  "
          f"non-STATIC linearity={linearity[~is_static].mean():.4f} planarity={planarity[~is_static].mean():.4f}")

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    for feature_name, feature_arr in [("linearity", linearity), ("planarity", planarity)]:
        print(f"\n--- {feature_name} alone ---")
        print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'n_flagged':>10}")
        for t in thresholds:
            p, r, f1, n = _prf1(feature_arr > t, is_static)
            print(f"{t:>10.2f} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {n:>10d}")

    print(f"\n--- linearity OR planarity ---")
    print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'n_flagged':>10}")
    for t in thresholds:
        flagged = (linearity > t) | (planarity > t)
        p, r, f1, n = _prf1(flagged, is_static)
        print(f"{t:>10.2f} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {n:>10d}")


if __name__ == "__main__":
    main()
