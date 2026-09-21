"""
eval/export_story_labels.py

Exports REAL FusionSegNet per-point class predictions for the same frame
eval/export_story_frames.py writes to story/static/data/baseline_frame.bin,
so the story app can recolour the real point cloud by predicted class.

Output: story/static/data/<name>_labels.bin -- one uint8 DrishtiClass id per
point slot, in the exact order of baseline_frame.bin (load_rellis_sweep order).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval.cache_inference import cache_inference_for_frame, load_trained_model
from eval.export_story_frames import STORY_DATA_DIR
from perception.train import load_stats
from sensor.sensor_model import load_sensor_config


def main():
    parser = argparse.ArgumentParser(description="Export real per-point predictions for the story frame")
    parser.add_argument("--sequence-dir", default="data/rellis/00000")
    parser.add_argument("--frame", type=int, default=50)
    parser.add_argument("--name", default="baseline_frame")
    parser.add_argument("--checkpoint", default="checkpoints_multi_v5/best.pt")
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    stats = load_stats(ckpt.parent / "channel_stats.json")
    sm = load_sensor_config(args.sensor_config)
    model, epoch = load_trained_model(ckpt)

    result = cache_inference_for_frame(model, Path(args.sequence_dir), args.frame, sm, stats)
    labels = result["point_labels"].astype(np.uint8)

    STORY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = STORY_DATA_DIR / f"{args.name}_labels.bin"
    labels.tofile(out)
    counts = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    print(f"Wrote {out} ({labels.shape[0]} labels, checkpoint epoch {epoch}); class counts: {counts}")


if __name__ == "__main__":
    main()
