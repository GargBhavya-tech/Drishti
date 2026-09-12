"""
eval/extract_rare_clusters.py

Offline harvesting pass for `perception/cutmix.py`: scans every REAL
RELLIS-3D TRAINING frame (the same split `perception.train`'s own
`build_multi_sequence_splits` computes -- no leakage into val) for a
target class's points and saves them as paste-able clusters. Run ONCE
before training with `--cutmix-clusters`, not per-epoch -- a full scan
across all 5 sequences' training frames takes real minutes, not
seconds, since it re-reads every frame's real `.bin`/`.label` files.

Default target is class 4 (STATIC_OBSTACLE): real-data validation
(`eval/validate_feature_hypotheses.py`, check C) found it at 0.051% of
all training pixels, present in only a minority of frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from perception.cutmix import extract_rare_class_clusters, save_clusters
from perception.rellis_loader import load_rellis_labels, load_rellis_sweep
from perception.taxonomy import DrishtiClass, rellis_label_ids_to_drishti
from perception.train import build_multi_sequence_splits

SEQUENCE_DIRS = [Path(f"data/rellis/{i:05d}") for i in range(5)]


def main(target_class: int, out_path: str) -> None:
    train_items, _val_items, _counts = build_multi_sequence_splits(SEQUENCE_DIRS)
    class_name = DrishtiClass(target_class).name
    print(f"Scanning {len(train_items)} real TRAINING frames for class {target_class} ({class_name}) points...")

    frames_with_labels = []
    for n, (seq_dir, frame_idx) in enumerate(train_items):
        sweep = load_rellis_sweep(seq_dir, frame_idx)
        raw_labels = load_rellis_labels(seq_dir, frame_idx)
        drishti_labels = rellis_label_ids_to_drishti(raw_labels)
        is_target = drishti_labels == target_class
        if np.any(is_target):
            xyzi = np.concatenate([sweep.xyz, sweep.intensity[:, None]], axis=1)
            frames_with_labels.append((xyzi, is_target))
        if (n + 1) % 500 == 0:
            print(f"  ...scanned {n + 1}/{len(train_items)} frames, {len(frames_with_labels)} with class {target_class} so far")

    print(f"Found target-class points in {len(frames_with_labels)} of {len(train_items)} frames")
    clusters = extract_rare_class_clusters(frames_with_labels)
    print(f"Extracted {len(clusters.clusters)} clusters (after the min-points-per-frame filter)")

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    save_clusters(clusters, str(out_path_p))
    print(f"Saved to {out_path_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvest real rare-class point clusters for CutMix (perception/cutmix.py)")
    parser.add_argument("--target-class", type=int, default=int(DrishtiClass.STATIC_OBSTACLE))
    parser.add_argument("--out", default="perception/rare_clusters_class4.npz")
    args = parser.parse_args()
    main(target_class=args.target_class, out_path=args.out)
