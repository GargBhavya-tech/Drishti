"""
eval/checkpoint_geometric_detection_rellis.py

Re-tests perception/geometric_instance_detector.py (Part G.12) against
RELLIS-3D instead of nuScenes -- the direct follow-up G.12 itself named
as unresolved: the geometric detector's real failure on nuScenes was
diagnosed as inherited segmentation weakness (PEDESTRIAN/VEHICLE mIoU
0.21-0.33 there), not a flaw in the clustering/confidence mechanism
itself. RELLIS-3D's own best checkpoint (checkpoints_multi_v3) has much
stronger numbers for exactly those classes (PEDESTRIAN 0.771, VEHICLE
0.537) -- this script is the first time the detector is run against
that stronger segmentation domain.

RELLIS-3D ships NO 3D box annotations (unlike nuScenes) -- there is no
"real_object_count" to borrow the way nuScenes' sample_annotation
records provided one. The proxy used here, exactly as specified: run
the SAME connected-components clustering the detector itself uses, but
over REAL ground-truth per-point labels (rellis_label_ids_to_drishti),
not model predictions -- giving an approximate real object count. The
detector itself still runs on the TRAINED model's own predictions, so
the comparison (proxy ground truth vs. model-driven detection) mirrors
the nuScenes setup's real box count vs. decoded count structure, and
still isolates whether RELLIS's stronger segmentation translates into
usable detection recall -- the exact question G.12 left open.

Honest caveat, stated once here rather than re-derived per reader: this
proxy ground truth is NOT independent of the clustering method being
evaluated (both the proxy and the detector use the identical elevate-
and-cluster logic) -- a real limitation of not having true instance-
level annotations for RELLIS-3D. It answers "does clustering real
labels versus predicted labels produce similar object counts" more
than "how many real physical objects were in the scene" -- weaker than
nuScenes' box-based ground truth, exactly as flagged when this was
proposed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from perception.geometric_instance_detector import detect_instances
from perception.ground_prior import compute_ground_prior
from perception.input_tensor import assemble_input_tensor, load_stats
from perception.rellis_loader import load_rellis_labels, load_rellis_sweep
from perception.range_image import project_to_range_image
from perception.segnet import FusionSegNet, N_CLASSES_DEFAULT
from perception.taxonomy import DrishtiClass, rellis_label_ids_to_drishti
from perception.train import build_multi_sequence_splits
from sensor.sensor_model import load_sensor_config


def _per_point_ground_z(sweep, ground, img) -> np.ndarray:
    x, y = sweep.xyz[:, 0].astype(np.float64), sweep.xyz[:, 1].astype(np.float64)
    theta = np.arctan2(y, x)
    col = np.floor((0.5 * (1.0 - theta / np.pi)) * img.W).astype(np.int64)
    col = np.clip(col, 0, img.W - 1)
    if ground.column_ground_height:
        fallback = float(np.median(list(ground.column_ground_height.values())))
    else:
        fallback = 0.0
    z_ground = np.full(sweep.xyz.shape[0], fallback, dtype=np.float64)
    for c, h in ground.column_ground_height.items():
        z_ground[col == c] = h
    return z_ground


def main():
    parser = argparse.ArgumentParser(description="Benchmark the geometric detector on RELLIS-3D, proxy ground truth vs real model predictions")
    parser.add_argument("--sequence-dir", nargs="+", default=[f"data/rellis/{i:05d}" for i in range(5)])
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--seg-checkpoint", default="checkpoints_multi_v3/best.pt")
    parser.add_argument("--stats-path", default="checkpoints_multi_v3/channel_stats.json")
    parser.add_argument("--cell-size-m", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="eval/out")
    args = parser.parse_args()

    dirs = [Path(p) for p in args.sequence_dir]
    _train_items, val_items, _ = build_multi_sequence_splits(dirs)
    if args.limit:
        step = max(1, len(val_items) // args.limit)
        val_items = val_items[::step][: args.limit]

    sm = load_sensor_config(args.sensor_config)
    stats = load_stats(args.stats_path)

    device = torch.device(args.device)
    model = FusionSegNet(n_classes=N_CLASSES_DEFAULT).to(device)
    ckpt = torch.load(args.seg_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    class_names = {int(c): c.name for c in DrishtiClass}

    per_frame_results = []
    with torch.no_grad():
        for sequence_dir, frame_idx in val_items:
            sweep = load_rellis_sweep(sequence_dir, frame_idx)
            raw_labels = load_rellis_labels(sequence_dir, frame_idx)
            gt_class_per_point = rellis_label_ids_to_drishti(raw_labels)

            img = project_to_range_image(sweep, sm)
            ground = compute_ground_prior(sweep, n_azimuth_bins=img.W)
            tensor_np = assemble_input_tensor(img, ground, stats)
            x = torch.from_numpy(tensor_np).float().unsqueeze(0).to(device)

            seg_out = model(x)
            pred_grid = seg_out.argmax(dim=1)[0].cpu().numpy()

            touched = img.point_index >= 0
            n_points = sweep.xyz.shape[0]
            pred_class_per_point = np.full(n_points, int(DrishtiClass.UNKNOWN), dtype=np.int64)
            rows, cols = np.nonzero(touched)
            pred_class_per_point[img.point_index[touched]] = pred_grid[rows, cols]

            z_ground = _per_point_ground_z(sweep, ground, img)
            xyz64 = sweep.xyz.astype(np.float64)

            # Proxy ground-truth count: SAME clustering, over REAL labels.
            gt_detections = detect_instances(xyz64, gt_class_per_point, z_ground, sm, cell_size_m=args.cell_size_m)
            real_count = len(gt_detections)

            # Actual detector output: SAME clustering, over the TRAINED model's predictions.
            pred_detections = detect_instances(xyz64, pred_class_per_point, z_ground, sm, cell_size_m=args.cell_size_m)

            pred_by_class = {}
            for d in pred_detections:
                name = class_names.get(d.drishti_class, f"class_{d.drishti_class}")
                pred_by_class[name] = pred_by_class.get(name, 0) + 1
            gt_by_class = {}
            for d in gt_detections:
                name = class_names.get(d.drishti_class, f"class_{d.drishti_class}")
                gt_by_class[name] = gt_by_class.get(name, 0) + 1

            per_frame_results.append(
                {
                    "sequence_dir": str(sequence_dir),
                    "frame_idx": frame_idx,
                    "proxy_real_object_count": real_count,
                    "proxy_real_by_class": gt_by_class,
                    "decoded_object_count": len(pred_detections),
                    "decoded_by_class": pred_by_class,
                }
            )

    real_counts = np.array([r["proxy_real_object_count"] for r in per_frame_results])
    decoded_counts = np.array([r["decoded_object_count"] for r in per_frame_results])
    mae = float(np.mean(np.abs(real_counts - decoded_counts)))
    mean_real = float(np.mean(real_counts))
    mean_decoded = float(np.mean(decoded_counts))

    # Per-class recall for PEDESTRIAN/VEHICLE specifically -- the exact
    # classes G.12 diagnosed as the nuScenes failure point.
    for cls_name in ["PEDESTRIAN", "VEHICLE", "STATIC_OBSTACLE"]:
        gt_sum = sum(r["proxy_real_by_class"].get(cls_name, 0) for r in per_frame_results)
        pred_sum = sum(r["decoded_by_class"].get(cls_name, 0) for r in per_frame_results)
        print(f"{cls_name}: proxy real={gt_sum}, decoded={pred_sum}")

    print(f"\nEvaluated {len(per_frame_results)} real RELLIS-3D val frames (geometric detector, cell_size={args.cell_size_m}m)")
    print(f"Mean PROXY real object count per frame: {mean_real:.2f}")
    print(f"Mean DECODED object count per frame:    {mean_decoded:.2f}")
    print(f"Mean Absolute Error (per-frame count):  {mae:.2f}")
    print()
    print("First 10 frames, proxy-real vs. decoded:")
    for r in per_frame_results[:10]:
        print(f"  real={r['proxy_real_object_count']:3d} ({r['proxy_real_by_class']})  decoded={r['decoded_object_count']:3d} ({r['decoded_by_class']})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "geometric_detection_rellis_eval.json", "w") as f:
        json.dump(
            {
                "n_frames": len(per_frame_results),
                "cell_size_m": args.cell_size_m,
                "mean_proxy_real_object_count": mean_real,
                "mean_decoded_object_count": mean_decoded,
                "mean_absolute_error": mae,
                "per_frame": per_frame_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {out_dir / 'geometric_detection_rellis_eval.json'}")


if __name__ == "__main__":
    main()
