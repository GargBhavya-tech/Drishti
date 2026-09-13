"""
eval/diagnose_vehicle_recall_by_range.py

Diagnoses WHY VEHICLE recall was weak (22.5%) in the geometric detector
benchmark (DRISHTI_MASTER_BIBLE.md Part G.13) -- and box-splitting
didn't fix it (Part G.16), pointing at an upstream classification
recall problem rather than a clustering/over-merging one. This script
measures REAL per-point VEHICLE recall from checkpoints_multi_v3
directly, broken out by range bucket, to see WHERE the recall loss
actually happens (close range vs. far range) rather than just knowing
THAT it's low.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from perception.ground_prior import compute_ground_prior
from perception.input_tensor import assemble_input_tensor, load_stats
from perception.knn_crf import knn_crf_refine
from perception.rellis_loader import load_rellis_labels, load_rellis_sweep
from perception.range_image import project_to_range_image
from perception.segnet import FusionSegNet, N_CLASSES_DEFAULT
from perception.taxonomy import DrishtiClass, rellis_label_ids_to_drishti
from perception.train import build_multi_sequence_splits
from sensor.sensor_model import load_sensor_config

RANGE_BUCKETS_M = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 999)]


def main():
    parser = argparse.ArgumentParser(description="Diagnose real per-point VEHICLE recall by range bucket on RELLIS-3D")
    parser.add_argument("--sequence-dir", nargs="+", default=[f"data/rellis/{i:05d}" for i in range(5)])
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--seg-checkpoint", default="checkpoints_multi_v3/best.pt")
    parser.add_argument("--stats-path", default="checkpoints_multi_v3/channel_stats.json")
    parser.add_argument("--n-frames", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--use-crf",
        action="store_true",
        help="Apply perception.knn_crf.knn_crf_refine to per-point softmax probabilities before argmax "
             "(Part G.19, item 3) -- tests the kNN CRF post-processing idea in isolation, no retrain.",
    )
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

    vehicle_class = int(DrishtiClass.VEHICLE)

    # Per-bucket: [true_positive_count, real_vehicle_point_count]
    bucket_stats = {b: [0, 0] for b in RANGE_BUCKETS_M}
    # Confusion: what VEHICLE points actually get predicted as, aggregated
    misclassified_as = {}
    total_real_vehicle_points = 0
    total_recalled = 0

    with torch.no_grad():
        for sequence_dir, frame_idx in sample_items:
            sweep = load_rellis_sweep(sequence_dir, frame_idx)
            raw_labels = load_rellis_labels(sequence_dir, frame_idx)
            gt_labels = rellis_label_ids_to_drishti(raw_labels)

            img = project_to_range_image(sweep, sm)
            ground = compute_ground_prior(sweep, n_azimuth_bins=img.W)
            tensor_np = assemble_input_tensor(img, ground, stats)
            x = torch.from_numpy(tensor_np).float().unsqueeze(0).to(device)

            seg_out = model(x)
            n_points = sweep.xyz.shape[0]
            touched = img.point_index >= 0
            rows, cols = np.nonzero(touched)

            if args.use_crf:
                # kNN CRF refinement (Part G.19, item 3) needs real
                # per-point class PROBABILITIES, not just the argmax --
                # gather softmax(seg_out) at each point's own projected
                # pixel, refine in real 3D space, THEN argmax. Untouched
                # points (never projected to a pixel) get a one-hot
                # UNKNOWN probability, matching this script's original
                # UNKNOWN-default behaviour for those points even after
                # CRF smoothing (a point with no real return has no
                # class evidence to smooth in the first place).
                probs_grid = torch.softmax(seg_out, dim=1)[0].cpu().numpy()  # (C, H, W)
                n_classes = probs_grid.shape[0]
                point_probs = np.zeros((n_points, n_classes), dtype=np.float64)
                point_probs[img.point_index[touched]] = probs_grid[:, rows, cols].T
                untouched = np.ones(n_points, dtype=bool)
                untouched[img.point_index[touched]] = False
                point_probs[untouched, int(DrishtiClass.UNKNOWN)] = 1.0

                point_probs = knn_crf_refine(sweep.xyz.astype(np.float64), point_probs)
                pred_per_point = point_probs.argmax(axis=1)
            else:
                pred_grid = seg_out.argmax(dim=1)[0].cpu().numpy()
                pred_per_point = np.full(n_points, int(DrishtiClass.UNKNOWN), dtype=np.int64)
                pred_per_point[img.point_index[touched]] = pred_grid[rows, cols]

            is_vehicle = gt_labels == vehicle_class
            if not np.any(is_vehicle):
                continue

            ranges = np.linalg.norm(sweep.xyz[:, :2], axis=1)
            vehicle_ranges = ranges[is_vehicle]
            vehicle_preds = pred_per_point[is_vehicle]

            recalled = vehicle_preds == vehicle_class
            total_real_vehicle_points += int(is_vehicle.sum())
            total_recalled += int(recalled.sum())

            for pred_c in vehicle_preds[~recalled]:
                misclassified_as[pred_c] = misclassified_as.get(pred_c, 0) + 1

            for lo, hi in RANGE_BUCKETS_M:
                in_bucket = (vehicle_ranges >= lo) & (vehicle_ranges < hi)
                bucket_stats[(lo, hi)][0] += int(recalled[in_bucket].sum())
                bucket_stats[(lo, hi)][1] += int(in_bucket.sum())

    print(f"Sampled {len(sample_items)} real RELLIS-3D val frames")
    print(f"Total real VEHICLE points: {total_real_vehicle_points}")
    print(f"Overall recall: {total_recalled}/{total_real_vehicle_points} = {total_recalled/max(total_real_vehicle_points,1):.2%}")
    print()
    print(f"{'range bucket (m)':>20} {'recall':>10} {'n_real_points':>15}")
    for (lo, hi), (tp, n) in bucket_stats.items():
        recall = tp / n if n > 0 else float("nan")
        label = f"[{lo},{hi})" if hi < 999 else f"[{lo},inf)"
        print(f"{label:>20} {recall:>10.2%} {n:>15d}")
    print()
    print("Real VEHICLE points misclassified as (class_id: count):")
    class_names = {int(c): c.name for c in DrishtiClass}
    for cls, count in sorted(misclassified_as.items(), key=lambda kv: -kv[1]):
        print(f"  {class_names.get(cls, cls)}: {count} ({count/max(total_real_vehicle_points,1):.2%} of all real VEHICLE points)")


if __name__ == "__main__":
    main()
