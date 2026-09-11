"""
eval/checkpoint_accuracy_by_distance.py

Ticket #57's own missing evidence: runs a REAL trained checkpoint over
REAL RELLIS-3D validation frames and reports mIoU BINNED BY DISTANCE
BAND (`eval.metrics.compute_miou_by_distance_band`), directly producing
the PS's own "high accuracy in object classification across varying
distances" requirement as a real artifact -- `eval/metrics.py`'s own
docstring names this exact requirement, but the function had never been
run against a real checkpoint before this script existed.

Bands are the clipmap's OWN level boundaries (N=512, c0=0.05m, 4 levels
-> [12.8, 25.6, 51.2, 102.4] m for this project's default schedule,
computed here from a REAL `Clipmap`, not hand-copied) -- the map's own
native resolution levels, not an arbitrary round-number grid
(`eval.metrics`'s own stated rationale).

Follows the SAME real-data checkpoint pattern as
`eval/checkpoint_attention_overlay.py` and `eval/diagnose_confusion.py`:
a real checkpoint, real RELLIS-3D val-split frames
(`perception.train`'s own split, no leakage), real per-pixel
predictions -- never synthetic data. Per this project's own established
finding (`HANDOFF.md`'s CPU/GPU numerical-discrepancy write-up), run
with `--device cuda` on the remote GPU for trustworthy numbers; CPU
inference on this dev machine is known to be systematically degraded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from eval.cache_inference import load_trained_model
from eval.diagnose_confusion import sample_val_items
from eval.metrics import band_boundaries_m, compute_miou_by_distance_band
from grid.clipmap import Clipmap
from perception.input_tensor import assemble_input_tensor, load_stats
from perception.taxonomy import DrishtiClass
from perception.train import _load_frame, build_multi_sequence_splits
from sensor.sensor_model import load_sensor_config

CLASS_NAMES = [c.name for c in DrishtiClass]
DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "accuracy_by_distance.json"


def _band_label(index: int, boundaries: List[float]) -> str:
    lo = 0.0 if index == 0 else boundaries[index - 1]
    if index < len(boundaries) - 1:
        return f"[{lo:.1f}, {boundaries[index]:.1f})m"
    return f"[{lo:.1f}, inf)m"


def run_accuracy_by_distance(
    checkpoint_path: str,
    sequence_dirs: List[str],
    sensor_config_path: str,
    n_classes: int = 10,
    max_frames: int = 300,
    out_path: Path = DEFAULT_OUT,
    device: str = "cpu",
) -> dict:
    model, trained_epoch = load_trained_model(checkpoint_path, device=device)
    stats = load_stats(Path(checkpoint_path).parent / "channel_stats.json")
    sm = load_sensor_config(sensor_config_path)
    boundaries = band_boundaries_m(Clipmap(sm))

    _train_items, val_items, _counts = build_multi_sequence_splits([Path(d) for d in sequence_dirs])
    val_sample = sample_val_items(val_items, max_frames)
    print(
        f"Evaluating accuracy-by-distance on {len(val_sample)} sampled val frames "
        f"(of {len(val_items)} total), band boundaries {boundaries} m"
    )

    all_ranges: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    with torch.no_grad():
        for seq_dir, frame_idx in val_sample:
            img, ground, target = _load_frame(seq_dir, frame_idx, sm)
            tensor = assemble_input_tensor(img, ground, stats)
            x = torch.from_numpy(tensor).float().unsqueeze(0).to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()[0]

            valid = np.asarray(img.valid_mask).astype(bool)
            all_ranges.append(np.asarray(img.range)[valid].ravel())
            all_true.append(np.asarray(target)[valid].ravel())
            all_pred.append(pred[valid].ravel())

    ranges_m = np.concatenate(all_ranges)
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    report = compute_miou_by_distance_band(ranges_m, y_true, y_pred, n_classes, boundaries)
    miou_by_band = report.miou_by_band()
    band_idx = np.clip(np.searchsorted(boundaries, ranges_m, side="right"), 0, len(boundaries) - 1)

    print(
        f"\nCheckpoint epoch {trained_epoch} -- mIoU by distance band "
        f"({len(val_sample)} frames, {len(ranges_m):,} valid points):"
    )
    band_labels = [_band_label(i, boundaries) for i in range(len(boundaries))]
    n_points_by_band = {}
    for i, label in enumerate(band_labels):
        val = miou_by_band.get(i, float("nan"))
        val_str = "n/a" if val is None or np.isnan(val) else f"{val:.4f}"
        n_points = int((band_idx == i).sum())
        n_points_by_band[str(i)] = n_points
        print(f"  band {i} {label:>16}: mIoU={val_str}  (n={n_points:,} points)")

    result = {
        "checkpoint_path": str(checkpoint_path),
        "trained_epoch": trained_epoch,
        "device": device,
        "n_frames_sampled": len(val_sample),
        "n_points_total": int(len(ranges_m)),
        "boundaries_m": boundaries,
        "band_labels": band_labels,
        "n_points_by_band": n_points_by_band,
        "miou_by_band": {str(k): (None if v is None or np.isnan(v) else float(v)) for k, v in miou_by_band.items()},
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nFull report saved to {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PS 'accuracy across varying distances' evidence: real mIoU by distance band"
    )
    parser.add_argument("--checkpoint", default="checkpoints_multi_remote/checkpoint_epoch19.pt")
    parser.add_argument("--sequence-dirs", nargs="+", default=[f"data/rellis/{i:05d}" for i in range(5)])
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_accuracy_by_distance(
        checkpoint_path=args.checkpoint,
        sequence_dirs=args.sequence_dirs,
        sensor_config_path=args.sensor_config,
        max_frames=args.max_frames,
        out_path=Path(args.out),
        device=args.device,
    )
