"""
eval/evaluate_tta.py

Real accuracy check for perception/tta.py (Bible Part G.30): does TTA
(circular roll + geometrically-corrected flip, averaged softmax) actually
improve mIoU on real RELLIS-3D val frames, and by how much -- not assumed
from the deep-research report's own literature-transfer estimate
(+0.005 to +0.015), measured directly against `checkpoints_multi_v5/best.pt`.
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
from perception.train import build_multi_sequence_splits, confusion_matrix_update, per_class_iou
from perception.tta import predict_with_tta
from sensor.sensor_model import load_sensor_config


def main():
    parser = argparse.ArgumentParser(description="Real TTA vs baseline mIoU comparison on real RELLIS-3D val frames")
    parser.add_argument("--sequence-dir", nargs="+", default=[f"data/rellis/{i:05d}" for i in range(5)])
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--seg-checkpoint", default="checkpoints_multi_v5/best.pt")
    parser.add_argument("--stats-path", default="checkpoints_multi_v5/channel_stats.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="eval/out")
    args = parser.parse_args()

    dirs = [Path(p) for p in args.sequence_dir]
    _train_items, val_items, _ = build_multi_sequence_splits(dirs)
    if args.limit:
        step = max(1, len(val_items) // args.limit)
        val_items = val_items[::step][: args.limit]
    print(f"Evaluating {len(val_items)} real val frames, device={args.device}")

    sm = load_sensor_config(args.sensor_config)
    stats = load_stats(args.stats_path)
    device = torch.device(args.device)

    model = FusionSegNet(n_classes=N_CLASSES_DEFAULT).to(device)
    ckpt = torch.load(args.seg_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cm_baseline = np.zeros((N_CLASSES_DEFAULT, N_CLASSES_DEFAULT), dtype=np.int64)
    cm_tta = np.zeros((N_CLASSES_DEFAULT, N_CLASSES_DEFAULT), dtype=np.int64)

    with torch.no_grad():
        for n, (sequence_dir, frame_idx) in enumerate(val_items):
            sweep = load_rellis_sweep(sequence_dir, frame_idx)
            raw_labels = load_rellis_labels(sequence_dir, frame_idx)
            gt_labels = rellis_label_ids_to_drishti(raw_labels)

            img = project_to_range_image(sweep, sm)
            ground = compute_ground_prior(sweep, n_azimuth_bins=img.W)
            tensor_np = assemble_input_tensor(img, ground, stats)
            x = torch.from_numpy(tensor_np).float().unsqueeze(0).to(device)

            touched = img.point_index >= 0
            target_grid = np.zeros((img.H, img.W), dtype=np.int64)
            target_grid[touched] = gt_labels[img.point_index[touched]]

            pred_baseline = model(x).argmax(dim=1)[0].cpu().numpy()
            confusion_matrix_update(cm_baseline, pred_baseline, target_grid, img.valid_mask, N_CLASSES_DEFAULT)

            pred_tta = predict_with_tta(model, x, stats, use_flip=True)[0].cpu().numpy()
            confusion_matrix_update(cm_tta, pred_tta, target_grid, img.valid_mask, N_CLASSES_DEFAULT)

            if (n + 1) % 20 == 0:
                print(f"  ...{n + 1}/{len(val_items)} frames")

    ious_baseline = per_class_iou(cm_baseline)
    ious_tta = per_class_iou(cm_tta)
    miou_baseline = float(np.nanmean(ious_baseline))
    miou_tta = float(np.nanmean(ious_tta))
    names = [c.name for c in DrishtiClass]

    print(f"\n{'class':>18} {'baseline IoU':>14} {'TTA IoU':>14} {'delta':>10}")
    results = []
    for i, name in enumerate(names):
        b = float(ious_baseline[i]) if not np.isnan(ious_baseline[i]) else None
        t = float(ious_tta[i]) if not np.isnan(ious_tta[i]) else None
        delta = (t - b) if (b is not None and t is not None) else None
        print(f"{name:>18} {b if b is not None else 'n/a':>14} {t if t is not None else 'n/a':>14} "
              f"{f'{delta:+.4f}' if delta is not None else 'n/a':>10}")
        results.append({"class": name, "baseline_iou": b, "tta_iou": t, "delta": delta})

    print(f"\nOverall mIoU: baseline={miou_baseline:.4f}, TTA={miou_tta:.4f}, delta={miou_tta - miou_baseline:+.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tta_evaluation.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_frames": len(val_items), "miou_baseline": miou_baseline, "miou_tta": miou_tta,
            "delta": miou_tta - miou_baseline, "per_class": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
