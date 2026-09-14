"""
eval/reconcile_all_class_drift.py

Generalizes the one-off VEHICLE-only reconciliation that found real
pipeline drift (DRISHTI_MASTER_BIBLE.md Part G.21) to EVERY class, on
the FULL 2,034-frame val set -- G.21 only checked VEHICLE, and only on
a 40-frame sample; this closes that gap directly, per this project's
own reviewer flagging it as the single most urgent open item ("every
headline accuracy number in the document that predates this discovery
is currently of unknown reliability").

Method: load `checkpoints_multi_v3/best.pt` (the checkpoint whose
epoch-16 per-class IoU was originally LOGGED live during training --
see `checkpoints_multi_v3/training_log.jsonl`), re-run inference against
the CURRENT pipeline on the full val set using `perception.train`'s own
`confusion_matrix_update`/`per_class_iou` (reused verbatim, not
reimplemented -- exactly matches how train.py's own validation loop
computed the original numbers), and report BOTH the originally-logged
number and the re-measured current one, per class, side by side.

Deliberately checkpoints_multi_v3 (not v4/v5): v3 is the checkpoint
with a real LOGGED number to compare against (from its own original
training run); v4/v5 never had a "pre-drift" baseline to drift away
from in the first place, so this specific comparison is only meaningful
for v3.
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
from sensor.sensor_model import load_sensor_config

SEQUENCE_DIRS = [Path(f"data/rellis/{i:05d}") for i in range(5)]


def _logged_best_epoch_iou(training_log_path: Path) -> tuple:
    """Returns (best_epoch, per_class_iou_list) from the checkpoint's own
    real training_log.jsonl -- the exact numbers LOGGED LIVE at training
    time, which this script's whole point is to re-measure and compare
    against, not re-derive some other way."""
    best_epoch = None
    best_miou = -1.0
    best_ious = None
    with open(training_log_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["val_miou"] > best_miou:
                best_miou = entry["val_miou"]
                best_epoch = entry["epoch"]
                best_ious = entry["per_class_iou"]
    return best_epoch, best_ious


def main():
    parser = argparse.ArgumentParser(description="Re-measure ALL classes' real IoU on the FULL val set, compare against the originally-logged training-time numbers")
    parser.add_argument("--seg-checkpoint", default="checkpoints_multi_v3/best.pt")
    parser.add_argument("--stats-path", default="checkpoints_multi_v3/channel_stats.json")
    parser.add_argument("--training-log", default="checkpoints_multi_v3/training_log.jsonl")
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="eval/out")
    args = parser.parse_args()

    best_epoch, logged_ious = _logged_best_epoch_iou(Path(args.training_log))
    print(f"Originally logged best epoch: {best_epoch}")

    sm = load_sensor_config(args.sensor_config)
    stats = load_stats(args.stats_path)
    device = torch.device(args.device)

    model = FusionSegNet(n_classes=N_CLASSES_DEFAULT).to(device)
    ckpt = torch.load(args.seg_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch')} (should match {best_epoch} above -- "
          f"confirmed separately in Part G.21 via state_dict hash that best.pt IS the logged best epoch's weights)")

    _train_items, val_items, _ = build_multi_sequence_splits(SEQUENCE_DIRS)
    print(f"Re-measuring on the FULL val set: {len(val_items)} frames (not a sample)")

    cm = np.zeros((N_CLASSES_DEFAULT, N_CLASSES_DEFAULT), dtype=np.int64)
    with torch.no_grad():
        for n, (sequence_dir, frame_idx) in enumerate(val_items):
            sweep = load_rellis_sweep(sequence_dir, frame_idx)
            raw_labels = load_rellis_labels(sequence_dir, frame_idx)
            gt_labels = rellis_label_ids_to_drishti(raw_labels)

            img = project_to_range_image(sweep, sm)
            ground = compute_ground_prior(sweep, n_azimuth_bins=img.W)
            tensor_np = assemble_input_tensor(img, ground, stats)
            x = torch.from_numpy(tensor_np).float().unsqueeze(0).to(device)

            pred_grid = model(x).argmax(dim=1)[0].cpu().numpy()
            touched = img.point_index >= 0
            target_grid = np.zeros((img.H, img.W), dtype=np.int64)
            target_grid[touched] = gt_labels[img.point_index[touched]]
            confusion_matrix_update(cm, pred_grid, target_grid, img.valid_mask, N_CLASSES_DEFAULT)

            if (n + 1) % 200 == 0:
                print(f"  ...{n + 1}/{len(val_items)} frames")

    ious = per_class_iou(cm)
    names = [c.name for c in DrishtiClass]

    print(f"\n{'class':>18} {'logged (train-time)':>22} {'re-measured (now, full val)':>30} {'delta':>10} {'ratio':>8}")
    results = []
    for i, name in enumerate(names):
        logged = logged_ious[i] if i < len(logged_ious) else None
        now = float(ious[i]) if not np.isnan(ious[i]) else None
        delta = (now - logged) if (logged is not None and now is not None) else None
        ratio = (now / logged) if (logged and now is not None and logged > 0) else None
        logged_s = f"{logged:.4f}" if logged is not None else "n/a"
        now_s = f"{now:.4f}" if now is not None else "n/a"
        delta_s = f"{delta:+.4f}" if delta is not None else "n/a"
        ratio_s = f"{ratio:.2f}x" if ratio is not None else "n/a"
        print(f"{name:>18} {logged_s:>22} {now_s:>30} {delta_s:>10} {ratio_s:>8}")
        results.append({"class": name, "logged_iou": logged, "remeasured_iou_full_val": now, "delta": delta, "ratio": ratio})

    n_dropped = sum(1 for r in results if r["delta"] is not None and r["delta"] < -0.01)
    n_improved = sum(1 for r in results if r["delta"] is not None and r["delta"] > 0.01)
    print(f"\n{n_dropped} classes dropped by >0.01 IoU, {n_improved} improved by >0.01 IoU, "
          f"{len(results) - n_dropped - n_improved} roughly unchanged (of {sum(1 for r in results if r['delta'] is not None)} comparable classes)")

    # Where do a class's real points actually land when the network rarely
    # predicts that class? Reuses the SAME confusion matrix already built
    # above (cm[true_class, pred_class]) -- no second expensive pass over
    # the val set -- to answer a real safety question the per-class IoU
    # table alone can't: a class that collapses to 0 IoU could be leaking
    # into a SAFE class (a real regression) or into another HAZARD class
    # (harmless to the Conservatism Invariant, which costs hazard classes
    # equally-or-worse). Printed for every class, not just NON_TRAVERSABLE,
    # since this is cheap once the matrix exists and any class could hide
    # the same failure mode.
    print(f"\nPer-class prediction breakdown (where do a class's REAL points actually get predicted?):")
    for i, name in enumerate(names):
        row_total = int(cm[i].sum())
        if row_total == 0:
            continue
        top = np.argsort(cm[i])[::-1][:4]
        breakdown = ", ".join(f"{names[j]}={cm[i][j]/row_total*100:.1f}%" for j in top if cm[i][j] > 0)
        print(f"  {name:>18} ({row_total} real points): {breakdown}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_class_drift.json"
    with open(out_path, "w") as f:
        json.dump({
            "best_epoch": best_epoch, "n_val_frames": len(val_items), "per_class": results,
            "confusion_matrix": cm.tolist(), "class_names": names,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
