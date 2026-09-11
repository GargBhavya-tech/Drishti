"""
eval/checkpoint_latency.py

Ticket #55's own missing evidence: profiles the REAL trained
checkpoint's forward pass (plus the real preprocessing and post-
processing around it) over REAL RELLIS-3D frames using
`eval.latency.LatencyProfiler`, producing the PS's own "evidence of low
latency" as a real measured artifact -- `LatencyProfiler` existed but
had never been run against the real pipeline before this script.

Deliberately reports LATENCY (P50/P95), never FPS -- `eval/latency.py`'s
own module docstring states why: the sensor's own frame rate (10-20 Hz)
already caps any FPS number at something meaningless above it, so a raw
"FPS" figure derived purely from inference time would overstate what
matters, which is whether inference fits inside the SENSOR's own frame
period with headroom to spare. This script reports exactly that
comparison instead: measured P50/P95 against the sensor's own 10 Hz
(100ms) and 20 Hz (50ms) frame budgets -- the honest form of the PS's
"high FPS" ask, consistent with this project's own already-made design
decision rather than silently overriding it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from eval.cache_inference import load_trained_model
from eval.diagnose_confusion import sample_val_items
from eval.latency import LatencyProfiler
from perception.input_tensor import assemble_input_tensor, load_stats
from perception.train import _load_frame, build_multi_sequence_splits
from sensor.sensor_model import load_sensor_config

DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "checkpoint_latency.json"

# The sensor's own real frame rates (10-20 Hz is a typical spinning
# mechanical LiDAR range, matching this project's own stated assumption
# in eval/latency.py) -- inference latency is only meaningful relative
# to these, not in isolation.
SENSOR_FRAME_BUDGETS_S = {"10Hz": 0.100, "20Hz": 0.050}


def run_latency_checkpoint(
    checkpoint_path: str,
    sequence_dirs: List[str],
    sensor_config_path: str,
    n_frames: int = 200,
    out_path: Path = DEFAULT_OUT,
    device: str = "cpu",
) -> dict:
    model, trained_epoch = load_trained_model(checkpoint_path, device=device)
    stats = load_stats(Path(checkpoint_path).parent / "channel_stats.json")
    sm = load_sensor_config(sensor_config_path)

    _train_items, val_items, _counts = build_multi_sequence_splits([Path(d) for d in sequence_dirs])
    val_sample = sample_val_items(val_items, n_frames)
    print(f"Profiling latency over {len(val_sample)} real frames on device={device}")

    profiler = LatencyProfiler()
    forward_stage = profiler.cuda_stage if device == "cuda" else profiler.stage

    with torch.no_grad():
        for seq_dir, frame_idx in val_sample:
            with profiler.frame():
                with profiler.stage("load_and_assemble"):
                    img, ground, _target = _load_frame(seq_dir, frame_idx, sm)
                    tensor = assemble_input_tensor(img, ground, stats)
                    x = torch.from_numpy(tensor).float().unsqueeze(0).to(device)
                with forward_stage("model_forward"):
                    logits = model(x)
                with profiler.stage("argmax_to_cpu"):
                    _pred = logits.argmax(dim=1).cpu().numpy()[0]

    report = profiler.report()
    unaccounted_s = profiler.unaccounted_time_s()

    print(f"\nCheckpoint epoch {trained_epoch} -- latency over {report.n_frames} real frames (device={device}):")
    print(f"  end-to-end  P50={report.end_to_end_p50_s * 1000:.2f}ms  P95={report.end_to_end_p95_s * 1000:.2f}ms")
    for name in report.per_stage_p50_s:
        print(
            f"    {name}: P50={report.per_stage_p50_s[name] * 1000:.2f}ms  "
            f"P95={report.per_stage_p95_s[name] * 1000:.2f}ms"
        )
    print(f"  unaccounted time: {unaccounted_s * 1000:.2f}ms total across all frames")

    budget_comparison = {}
    for label, budget_s in SENSOR_FRAME_BUDGETS_S.items():
        headroom_x = budget_s / report.end_to_end_p95_s if report.end_to_end_p95_s > 0 else float("inf")
        fits = report.end_to_end_p95_s <= budget_s
        budget_comparison[label] = {"budget_ms": budget_s * 1000, "headroom_x": headroom_x, "fits_p95": fits}
        print(
            f"  vs {label} sensor frame budget ({budget_s * 1000:.0f}ms): "
            f"{'FITS' if fits else 'DOES NOT FIT'}, {headroom_x:.1f}x headroom at P95"
        )

    result = {
        "checkpoint_path": str(checkpoint_path),
        "trained_epoch": trained_epoch,
        "device": device,
        "n_frames": report.n_frames,
        "end_to_end_p50_ms": report.end_to_end_p50_s * 1000,
        "end_to_end_p95_ms": report.end_to_end_p95_s * 1000,
        "per_stage_p50_ms": {k: v * 1000 for k, v in report.per_stage_p50_s.items()},
        "per_stage_p95_ms": {k: v * 1000 for k, v in report.per_stage_p95_s.items()},
        "unaccounted_time_ms_total": unaccounted_s * 1000,
        "sensor_frame_budget_comparison": budget_comparison,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nFull report saved to {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PS 'low latency' evidence: real end-to-end inference latency")
    parser.add_argument("--checkpoint", default="checkpoints_multi_remote/checkpoint_epoch19.pt")
    parser.add_argument("--sequence-dirs", nargs="+", default=[f"data/rellis/{i:05d}" for i in range(5)])
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--n-frames", type=int, default=200)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_latency_checkpoint(
        checkpoint_path=args.checkpoint,
        sequence_dirs=args.sequence_dirs,
        sensor_config_path=args.sensor_config,
        n_frames=args.n_frames,
        out_path=Path(args.out),
        device=args.device,
    )
