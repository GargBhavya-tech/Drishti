"""
eval/checkpoint_detection_decode.py

Runs perception/detection_decode.py's peak-extraction against the
TRAINED checkpoint (checkpoints_detection/best.pt) on real held-out
nuScenes-mini val frames, and compares the DECODED discrete object
count against the REAL ground-truth object count in that frame (from
perception.nuscenes_boxes.build_box_targets's own n_boxes_with_any_pixel
-- the count of real boxes that actually won at least one pixel, the
same quantity the detection head was trained to recover).

This is the real test perception/train_detection.py's own per-pixel
loss curve cannot answer on its own (stated explicitly in that script's
own epoch printout): does decoding turn per-pixel predictions into
sensible discrete instances with roughly the right COUNT, not just a
lower loss number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from perception.detection_decode import decode_detections
from perception.detection_head import DetectionHead
from perception.ground_prior import compute_ground_prior
from perception.input_tensor import assemble_input_tensor, load_stats
from perception.nuscenes_boxes import build_box_targets
from perception.nuscenes_loader import load_nuscenes_sweep
from perception.nuscenes_seg_dataset import build_nuscenes_scene_splits
from perception.range_image import project_to_range_image
from perception.segnet import FusionSegNet, N_CLASSES_DEFAULT
from perception.taxonomy import DrishtiClass
from sensor.sensor_model import load_sensor_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection-head decode against real ground-truth object counts")
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--sensor-config", default="configs/sensor_hdl32e.yaml")
    parser.add_argument("--detection-checkpoint", default="checkpoints_detection/best.pt")
    parser.add_argument("--stats-path", default="checkpoints_joint/channel_stats.json")
    parser.add_argument("--objectness-threshold", type=float, default=0.5)
    parser.add_argument("--nms-pool-size", type=int, default=5, help="perception.detection_decode.decode_detections's nms_pool_size")
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
    head = DetectionHead().to(device)
    ckpt = torch.load(args.detection_checkpoint, map_location=device)
    model.load_state_dict(ckpt["backbone_state"])
    head.load_state_dict(ckpt["head_state"])
    model.eval()
    head.eval()

    class_names = {int(c): c.name for c in DrishtiClass}

    per_frame_results = []
    with torch.no_grad():
        for sample_token in val_items:
            sweep = load_nuscenes_sweep(nusc, sample_token)
            img = project_to_range_image(sweep, sm)
            ground = compute_ground_prior(sweep, n_azimuth_bins=img.W)
            tensor_np = assemble_input_tensor(img, ground, stats)
            x = torch.from_numpy(tensor_np).float().unsqueeze(0).to(device)

            seg_out, decoder_features = model(x, return_features=True)
            pred = head(decoder_features)
            seg_classes = seg_out.argmax(dim=1)[0].cpu()

            point_xyz_at_pixel = np.full((img.H, img.W, 3), np.nan, dtype=np.float32)
            touched = img.point_index >= 0
            point_xyz_at_pixel[touched] = sweep.xyz[img.point_index[touched]]

            detections = decode_detections(
                pred[0].cpu(), seg_classes, point_xyz_at_pixel,
                objectness_threshold=args.objectness_threshold, nms_pool_size=args.nms_pool_size,
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
                }
            )

    real_counts = np.array([r["real_object_count"] for r in per_frame_results])
    decoded_counts = np.array([r["decoded_object_count"] for r in per_frame_results])
    mae = float(np.mean(np.abs(real_counts - decoded_counts)))
    mean_real = float(np.mean(real_counts))
    mean_decoded = float(np.mean(decoded_counts))

    print(f"Evaluated {len(per_frame_results)} real held-out val frames")
    print(f"Mean REAL object count per frame:    {mean_real:.2f}")
    print(f"Mean DECODED object count per frame: {mean_decoded:.2f}")
    print(f"Mean Absolute Error (per-frame count): {mae:.2f}")
    print()
    print("First 10 frames, real vs. decoded:")
    for r in per_frame_results[:10]:
        print(f"  real={r['real_object_count']:3d}  decoded={r['decoded_object_count']:3d}  by_class={r['decoded_by_class']}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "detection_decode_eval.json", "w") as f:
        json.dump(
            {
                "n_frames": len(per_frame_results),
                "objectness_threshold": args.objectness_threshold,
                "mean_real_object_count": mean_real,
                "mean_decoded_object_count": mean_decoded,
                "mean_absolute_error": mae,
                "per_frame": per_frame_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {out_dir / 'detection_decode_eval.json'}")


if __name__ == "__main__":
    main()
