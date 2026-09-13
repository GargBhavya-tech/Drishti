"""
eval/accuracy_by_distance_all_classes.py

Generalizes eval/diagnose_vehicle_recall_by_range.py's per-range-bucket
methodology (built ad hoc to chase the VEHICLE bug in Part G.17) to a
clean, standalone, all-class deliverable -- the PS's own "accuracy...
across varying distances" line asks for exactly this as evidence, not
as a one-off debugging tool for a single class.

Per class, per real range bucket: real per-point RECALL (of real points
belonging to that class, what fraction the model correctly predicts) --
the same metric G.17 used, for direct comparability, computed from the
SAME per-point prediction/ground-truth arrays every other eval script
here builds, on a real sample of the val set (not the full 2,034 frames
by default, to keep this affordable alongside whatever else is running
on the same GPU -- see --n-frames).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from perception.ground_prior import compute_ground_prior
from perception.input_tensor import assemble_input_tensor, load_stats
from perception.rellis_loader import load_rellis_labels, load_rellis_sweep
from perception.range_image import project_to_range_image
from perception.segnet import FusionSegNet, N_CLASSES_DEFAULT
from perception.taxonomy import DrishtiClass, rellis_label_ids_to_drishti
from perception.train import build_multi_sequence_splits
from sensor.sensor_model import load_sensor_config

RANGE_BUCKETS_M = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 999)]


def main():
    parser = argparse.ArgumentParser(description="Real per-class, per-range-bucket recall table -- a standalone deliverable, not a debugging tool for one class")
    parser.add_argument("--sequence-dir", nargs="+", default=[f"data/rellis/{i:05d}" for i in range(5)])
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--seg-checkpoint", default="checkpoints_multi_v4/best.pt")
    parser.add_argument("--stats-path", default="checkpoints_multi_v4/channel_stats.json")
    parser.add_argument("--n-frames", type=int, default=150)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="eval/out")
    args = parser.parse_args()

    dirs = [Path(p) for p in args.sequence_dir]
    _train_items, val_items, _ = build_multi_sequence_splits(dirs)
    step = max(1, len(val_items) // args.n_frames)
    sample_items = val_items[::step][: args.n_frames]

    sm = load_sensor_config(args.sensor_config)
    stats = load_stats(args.stats_path)
    device = torch.device(args.device)

    model = FusionSegNet(n_classes=N_CLASSES_DEFAULT).to(device)
    ckpt = torch.load(args.seg_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    class_names = [c.name for c in DrishtiClass]
    n_classes = len(class_names)

    # [class][bucket] -> [true_positive_count, real_point_count]
    stats_grid = {c: {b: [0, 0] for b in RANGE_BUCKETS_M} for c in range(n_classes)}
    total_points_per_class = {c: 0 for c in range(n_classes)}

    with torch.no_grad():
        for n, (sequence_dir, frame_idx) in enumerate(sample_items):
            sweep = load_rellis_sweep(sequence_dir, frame_idx)
            raw_labels = load_rellis_labels(sequence_dir, frame_idx)
            gt_labels = rellis_label_ids_to_drishti(raw_labels)

            img = project_to_range_image(sweep, sm)
            ground = compute_ground_prior(sweep, n_azimuth_bins=img.W)
            tensor_np = assemble_input_tensor(img, ground, stats)
            x = torch.from_numpy(tensor_np).float().unsqueeze(0).to(device)

            pred_grid = model(x).argmax(dim=1)[0].cpu().numpy()
            touched = img.point_index >= 0
            n_points = sweep.xyz.shape[0]
            pred_per_point = np.full(n_points, int(DrishtiClass.UNKNOWN), dtype=np.int64)
            rows, cols = np.nonzero(touched)
            pred_per_point[img.point_index[touched]] = pred_grid[rows, cols]

            ranges = np.linalg.norm(sweep.xyz[:, :2], axis=1)

            for c in range(n_classes):
                is_c = gt_labels == c
                if not np.any(is_c):
                    continue
                total_points_per_class[c] += int(is_c.sum())
                c_ranges = ranges[is_c]
                c_recalled = pred_per_point[is_c] == c
                for lo, hi in RANGE_BUCKETS_M:
                    in_bucket = (c_ranges >= lo) & (c_ranges < hi)
                    stats_grid[c][(lo, hi)][0] += int(c_recalled[in_bucket].sum())
                    stats_grid[c][(lo, hi)][1] += int(in_bucket.sum())

            if (n + 1) % 25 == 0:
                print(f"  ...{n + 1}/{len(sample_items)} frames")

    bucket_labels = [f"[{lo},{hi})" if hi < 999 else f"[{lo},inf)" for lo, hi in RANGE_BUCKETS_M]
    header = f"{'class':>18} {'total_pts':>10} " + " ".join(f"{b:>14}" for b in bucket_labels)
    print(f"\n{header}")

    table = {}
    for c in range(n_classes):
        name = class_names[c]
        if total_points_per_class[c] == 0:
            continue
        row_cells = []
        row_data = {}
        for (lo, hi), label in zip(RANGE_BUCKETS_M, bucket_labels):
            tp, n_pts = stats_grid[c][(lo, hi)]
            recall = tp / n_pts if n_pts > 0 else float("nan")
            row_cells.append(f"{recall:>13.2%}" if n_pts > 0 else f"{'n/a':>13}")
            row_data[label] = {"recall": None if n_pts == 0 else recall, "n_points": n_pts}
        print(f"{name:>18} {total_points_per_class[c]:>10} " + " ".join(row_cells))
        table[name] = {"total_points": total_points_per_class[c], "by_range": row_data}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "accuracy_by_distance_all_classes.json"
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": args.seg_checkpoint,
            "n_frames_sampled": len(sample_items),
            "range_buckets_m": RANGE_BUCKETS_M,
            "per_class": table,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
