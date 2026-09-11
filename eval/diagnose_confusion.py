"""
eval/diagnose_confusion.py

Diagnostic (not a Build Map ticket): WHAT is class 4 (STATIC_OBSTACLE)
actually being confused with, and why is class 2 (CAUTION) still weak
-- TRAINING_RESULTS.md's own "Next Steps" flags both as needing this
exact investigation before spending more GPU hours guessing.

Runs a trained checkpoint over the SAME validation split
`perception.train.split_frames` computes (last 15% of each sequence,
per sequence -- no leakage), accumulates a full confusion matrix using
the SAME per-pixel methodology `perception.train`'s own val loop uses
(`confusion_matrix_update`, `per_class_iou`) -- not a re-derivation --
so the mIoU this script reports should land in the same ballpark as
TRAINING_RESULTS.md's own numbers, which is itself a check that this
script measures the same thing training already measures.

For each class, reports the TOP confusions on both sides:
- recall-side: of all TRUE pixels of this class, what they got
  predicted AS (a false negative breakdown -- "class 4 pixels mostly
  become class X").
- precision-side: of all PIXELS PREDICTED as this class, what they
  actually WERE (a false positive breakdown -- "class X pixels get
  mistaken for class 4").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from eval.cache_inference import load_trained_model
from perception.input_tensor import assemble_input_tensor, load_stats
from perception.taxonomy import DrishtiClass
from perception.train import _load_frame, build_multi_sequence_splits, confusion_matrix_update, per_class_iou
from sensor.sensor_model import load_sensor_config

CLASS_NAMES = [c.name for c in DrishtiClass]


def sample_val_items(val_items: List[Tuple[Path, int]], max_frames: int, seed: int = 0) -> List[Tuple[Path, int]]:
    """A deterministic, evenly-spread sample of the val split -- running
    EVERY val frame (2,034 in Run #2) on CPU is unnecessary for a
    diagnostic; a spread sample across all 5 sequences still gives a
    stable confusion matrix without an hours-long CPU pass."""
    if len(val_items) <= max_frames:
        return val_items
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(val_items), size=max_frames, replace=False)
    return [val_items[i] for i in sorted(idx)]


def build_confusion_matrix(
    checkpoint_path: str,
    sequence_dirs: List[str],
    sensor_config_path: str,
    n_classes: int,
    max_frames: int,
    device: str = "cpu",
) -> Tuple[np.ndarray, int]:
    model, trained_epoch = load_trained_model(checkpoint_path, device=device)
    stats_path = Path(checkpoint_path).parent / "channel_stats.json"
    stats = load_stats(stats_path)
    sm = load_sensor_config(sensor_config_path)

    _train_items, val_items, _counts = build_multi_sequence_splits([Path(d) for d in sequence_dirs])
    val_sample = sample_val_items(val_items, max_frames)
    print(f"Diagnosing on {len(val_sample)} sampled val frames (of {len(val_items)} total val frames)")

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for seq_dir, frame_idx in val_sample:
            img, ground, target = _load_frame(seq_dir, frame_idx, sm)
            tensor = assemble_input_tensor(img, ground, stats)
            x = torch.from_numpy(tensor).float().unsqueeze(0).to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()[0]
            confusion_matrix_update(cm, pred, target, img.valid_mask, n_classes)

    return cm, trained_epoch


def report_top_confusions(cm: np.ndarray, class_id: int, top_k: int = 3) -> None:
    name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
    row = cm[class_id].astype(np.float64)  # true=class_id, by predicted
    col = cm[:, class_id].astype(np.float64)  # predicted=class_id, by true
    row_total = row.sum()
    col_total = col.sum()

    print(f"\n--- class {class_id} ({name}) ---")
    print(f"  true pixels: {int(row_total)}   predicted pixels: {int(col_total)}")
    if row_total == 0:
        print("  no true pixels in this sample -- cannot report recall-side confusion")
    else:
        recall_frac = row / row_total
        top = np.argsort(-recall_frac)
        print("  recall-side (true class {} pixels end up predicted as):".format(class_id))
        shown = 0
        for c in top:
            if c == class_id or recall_frac[c] <= 0:
                continue
            other = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
            print(f"    -> class {c} ({other}): {recall_frac[c]:.1%}")
            shown += 1
            if shown >= top_k:
                break
        print(f"    (correct, class {class_id}: {recall_frac[class_id]:.1%})")

    if col_total == 0:
        print("  no predicted pixels of this class in this sample -- cannot report precision-side confusion")
    else:
        precision_frac = col / col_total
        top = np.argsort(-precision_frac)
        print("  precision-side (pixels predicted as class {} actually were):".format(class_id))
        shown = 0
        for c in top:
            if c == class_id or precision_frac[c] <= 0:
                continue
            other = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
            print(f"    <- class {c} ({other}): {precision_frac[c]:.1%}")
            shown += 1
            if shown >= top_k:
                break
        print(f"    (correct, class {class_id}: {precision_frac[class_id]:.1%})")


def run_diagnosis(
    checkpoint_path: str,
    sequence_dirs: List[str],
    sensor_config_path: str,
    n_classes: int = 10,
    max_frames: int = 300,
    out_path: str = "eval/out/confusion_matrix.json",
    device: str = "cpu",
) -> dict:
    cm, trained_epoch = build_confusion_matrix(checkpoint_path, sequence_dirs, sensor_config_path, n_classes, max_frames, device)

    ious = per_class_iou(cm)
    miou = float(np.nanmean(ious))
    print(f"\nCheckpoint epoch {trained_epoch} -- mIoU on this sample: {miou:.4f}")
    for c, iou in enumerate(ious):
        label = "n/a (no pixels)" if np.isnan(iou) else f"{iou:.4f}"
        print(f"  class {c} ({CLASS_NAMES[c]}): {label}")

    for class_id in (2, 4):
        report_top_confusions(cm, class_id)

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    out_path_p.write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_path),
                "trained_epoch": trained_epoch,
                "n_frames_sampled": max_frames,
                "confusion_matrix": cm.tolist(),
                "per_class_iou": [None if np.isnan(v) else float(v) for v in ious],
                "miou": miou,
            },
            indent=2,
        )
    )
    print(f"\nFull confusion matrix + metrics saved to {out_path_p}")
    return {"confusion_matrix": cm, "per_class_iou": ious, "miou": miou, "trained_epoch": trained_epoch}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose per-class confusion for a trained FusionSegNet checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints_multi_remote/checkpoint_epoch19.pt")
    parser.add_argument("--sequence-dirs", nargs="+", default=[f"data/rellis/{i:05d}" for i in range(5)])
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--out", default="eval/out/confusion_matrix.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_diagnosis(
        checkpoint_path=args.checkpoint,
        sequence_dirs=args.sequence_dirs,
        sensor_config_path=args.sensor_config,
        max_frames=args.max_frames,
        out_path=args.out,
        device=args.device,
    )
