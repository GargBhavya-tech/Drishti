"""
eval/export_story_frames.py

Minimal, story-specific data export for the new Svelte/Threlte scrollytelling
frontend (story/). Deliberately NOT a reuse of eval/export_frames.py's richer
per-frame format (points + predictions + cells) built for the old dashboard --
this is just real raw points for the "Baseline" beat (build order step 2).
No model inference here yet; that comes in a later beat's export.

Output, per frame:
  story/static/data/<name>.bin   -- raw interleaved float32: x, y, z, intensity
                                     (no header; real RELLIS-3D values, no
                                     quantization yet -- one frame is small
                                     enough that this is not a bottleneck)
  story/static/data/<name>.json  -- {"count": <n>, "sequence": <str>, "frame": <int>}
                                     so the frontend knows the real provenance
                                     and how many points to read back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from perception.rellis_loader import load_rellis_sweep

STORY_DATA_DIR = Path(__file__).resolve().parent.parent / "story" / "static" / "data"


def export_frame(sequence_dir: Path, frame_idx: int, name: str) -> None:
    sweep = load_rellis_sweep(sequence_dir, frame_idx)
    n = sweep.xyz.shape[0]

    interleaved = np.empty((n, 4), dtype=np.float32)
    interleaved[:, 0:3] = sweep.xyz
    interleaved[:, 3] = sweep.intensity

    STORY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = STORY_DATA_DIR / f"{name}.bin"
    json_path = STORY_DATA_DIR / f"{name}.json"

    interleaved.tofile(bin_path)
    with open(json_path, "w") as f:
        json.dump({"count": n, "sequence": sequence_dir.name, "frame": frame_idx}, f)

    print(f"Wrote {bin_path} ({n} real points, {bin_path.stat().st_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Export real RELLIS-3D frames for the story frontend")
    parser.add_argument("--sequence-dir", default="data/rellis/00000")
    parser.add_argument("--frame", type=int, default=50)
    parser.add_argument("--name", default="baseline_frame")
    args = parser.parse_args()

    export_frame(Path(args.sequence_dir), args.frame, args.name)


if __name__ == "__main__":
    main()
