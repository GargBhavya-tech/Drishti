"""
eval/checkpoint_geometric_detection.py

Head-to-head benchmark: perception/geometric_instance_detector.py
(training-free, physics-gated, metric-grid) vs. the learned detection
head (v1/v2, eval/checkpoint_detection_decode.py) on the IDENTICAL 80
real held-out nuScenes-mini val frames, using the IDENTICAL real
ground-truth object count (perception.nuscenes_boxes.build_box_targets's
n_boxes_with_any_pixel) -- an apples-to-apples comparison, not two
different benchmarks presented side by side.

Per-point class predictions come from the ALREADY-TRAINED segmentation
model (checkpoints_joint/best.pt by default) -- this detector reuses the
segmentation network's own classification (trained on ~14,000 frames
across three datasets), it does not train anything of its own.
Per-point ground z comes from perception.ground_prior's own
column_ground_height (the same column-wise walk the rest of this
project already uses), not a new ground estimate.
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
from perception.nuscenes_boxes import build_box_targets
from perception.nuscenes_loader import load_nuscenes_sweep
from perception.nuscenes_seg_dataset import build_nuscenes_scene_splits
from perception.range_image import project_to_range_image
from perception.segnet import FusionSegNet, N_CLASSES_DEFAULT
from perception.taxonomy import DrishtiClass
from sensor.sensor_model import load_sensor_config


def _per_point_ground_z(sweep, ground, img) -> np.ndarray:
    """Map each point's azimuth column to perception.ground_prior's own
    per-column ground height estimate; points in a column with no
    accepted ground return fall back to the global median of columns
    that DO have one (a real frame's own data, never a hardcoded
    constant)."""
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
    parser = argparse.ArgumentParser(description="Benchmark the training-free geometric instance detector on real nuScenes val frames")
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--sensor-config", default="configs/sensor_hdl32e.yaml")
    parser.add_argument("--seg-checkpoint", default="checkpoints_joint/best.pt")
    parser.add_argument("--stats-path", default="checkpoints_joint/channel_stats.json")
    parser.add_argument("--cell-size-m", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="eval/out")
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    _train_items, val_items, _ = build_nuscenes_scene_splits(nusc)
    if args.limit:
        val_items = val_items[: args.limit]

    sm = load_sensor_config(args.sensor_config)
    stats = load_stats(args.stats_path)

    device = torch.device(args.device)
    model = FusionSegNet(n_classes=N_CLASSES_DEFAULT).to(device)
    ckpt = torch.load(args.seg_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt["backbone_state"])
    model.eval()

    class_names = {int(c): c.name for c in DrishtiClass}

    per_frame_results = []
    with torch.no_grad():
        for sample_token in val_items:
            sweep = load_nuscenes_sweep(nusc, sample_token)
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

            detections = detect_instances(
                sweep.xyz.astype(np.float64), pred_class_per_point, z_ground, sm, cell_size_m=args.cell_size_m
            )

            box_targets = build_box_targets(nusc, sample_token, sweep.xyz, img)
            real_count = box_targets.n_boxes_with_any_pixel

            pred_by_class = {}
            for d in detections:
                name = class_names.get(d.drishti_class, f"class_{d.drishti_class}")
                pred_by_class[name] = pred_by_class.get(name, 0) + 1

            per_frame_results.append(
                {
                    "sample_token": sample_token,
                    "real_object_count": real_count,
                    "decoded_object_count": len(detections),
                    "decoded_by_class": pred_by_class,
                    "n_multi_instance_flagged": sum(1 for d in detections if d.possible_multi_instance),
                    "mean_kappa": float(np.mean([d.kappa for d in detections])) if detections else None,
                }
            )

    real_counts = np.array([r["real_object_count"] for r in per_frame_results])
    decoded_counts = np.array([r["decoded_object_count"] for r in per_frame_results])
    mae = float(np.mean(np.abs(real_counts - decoded_counts)))
    mean_real = float(np.mean(real_counts))
    mean_decoded = float(np.mean(decoded_counts))

    print(f"Evaluated {len(per_frame_results)} real held-out val frames (geometric detector, cell_size={args.cell_size_m}m)")
    print(f"Mean REAL object count per frame:    {mean_real:.2f}")
    print(f"Mean DECODED object count per frame: {mean_decoded:.2f}")
    print(f"Mean Absolute Error (per-frame count): {mae:.2f}")
    print()
    print("First 10 frames, real vs. decoded:")
    for r in per_frame_results[:10]:
        print(f"  real={r['real_object_count']:3d}  decoded={r['decoded_object_count']:3d}  by_class={r['decoded_by_class']}  multi_flagged={r['n_multi_instance_flagged']}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "geometric_detection_eval.json", "w") as f:
        json.dump(
            {
                "n_frames": len(per_frame_results),
                "cell_size_m": args.cell_size_m,
                "mean_real_object_count": mean_real,
                "mean_decoded_object_count": mean_decoded,
                "mean_absolute_error": mae,
                "per_frame": per_frame_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {out_dir / 'geometric_detection_eval.json'}")


if __name__ == "__main__":
    main()
