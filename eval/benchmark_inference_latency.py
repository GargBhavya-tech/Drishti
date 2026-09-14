"""
eval/benchmark_inference_latency.py

The real "how fast does this run live" number the PS's own Performance
Metrics bullet asks for ("evidence of low latency (high FPS)") -- a
gap a reviewer correctly flagged, but not quite the gap they named.
`FrameCache` (Part G.11/G.18/G.19) measures and speeds up MULTI-EPOCH
TRAINING throughput (the same frames reprocessed many times); it has
never been wired into the actual demo/inference path (`eval/export_frames.py`),
and this script does NOT change that, for a real, stated reason: a
genuinely LIVE, single-pass sensor feed sees every frame exactly ONCE,
so a cache that only pays off on a SECOND pass over the same frame
would never help a true real-time deployment at all -- wiring it in
would not produce a more meaningful FPS number, only a misleading one
if the benchmark frame set happened to repeat.

What actually answers the PS's question is a clean, honest, single-pass
timing of the REAL inference pipeline this project actually runs per
frame (projection -> ground-prior -> model forward pass -> argmax) --
exactly what `eval/export_frames.py` already does per frame, timed
here in isolation as a standalone deliverable rather than a debugging
side-effect. Never previously reported as a standalone number in this
project despite the pipeline existing the whole session -- G.11 measured
per-EPOCH training time, not per-FRAME inference latency.

GPU warmup: the first few forward passes on a fresh CUDA context are
real but not representative (kernel compilation, cuDNN autotuning) --
excluded from the reported statistics via `--warmup-frames`, timed and
reported separately for transparency, not silently dropped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from perception.ground_prior import compute_ground_prior
from perception.input_tensor import assemble_input_tensor, load_stats
from perception.rellis_loader import load_rellis_sweep
from perception.range_image import project_to_range_image
from perception.segnet import FusionSegNet, N_CLASSES_DEFAULT
from perception.train import build_multi_sequence_splits
from sensor.sensor_model import load_sensor_config


def main():
    parser = argparse.ArgumentParser(description="Real single-pass inference latency/FPS benchmark -- the PS's own 'low latency (high FPS)' metric")
    parser.add_argument("--sequence-dir", nargs="+", default=[f"data/rellis/{i:05d}" for i in range(5)])
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--seg-checkpoint", default="checkpoints_multi_v4/best.pt")
    parser.add_argument("--stats-path", default="checkpoints_multi_v4/channel_stats.json")
    parser.add_argument("--n-frames", type=int, default=200)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="eval/out")
    parser.add_argument("--fp16", action="store_true",
                         help="Wrap the forward pass in torch.autocast(dtype=torch.float16) "
                              "-- Bible Part G.28's Phase 1 FPS fix, item 2. Tier-A/B verified "
                              "separately (scratchpad/check_fp16_invariance.py, this session: "
                              "0.098%% argmax disagreement on 15 real val frames, well inside "
                              "the >99.5%% IoU-agreement bar) before this flag was added.")
    parser.add_argument("--compile", action="store_true",
                         help="Wrap the model in torch.compile(mode='reduce-overhead') -- "
                              "Bible Part G.28's Phase 1 FPS fix, item 3. Requires a STATIC "
                              "input shape (1,13,H,W) every frame -- true here since W is "
                              "fixed by --sensor-config, not per-frame point-count-dependent. "
                              "First-call compilation is real but excluded via --warmup-frames "
                              "same as CUDA kernel warmup; set TORCHINDUCTOR_CACHE_DIR for a "
                              "persistent cache across process restarts (a real deployment "
                              "would not want a multi-minute cold start every boot).")
    args = parser.parse_args()

    dirs = [Path(p) for p in args.sequence_dir]
    _train_items, val_items, _ = build_multi_sequence_splits(dirs)
    n_total = args.n_frames + args.warmup_frames
    step = max(1, len(val_items) // n_total)
    sample_items = val_items[::step][:n_total]
    warmup_items = sample_items[: args.warmup_frames]
    timed_items = sample_items[args.warmup_frames :]

    sm = load_sensor_config(args.sensor_config)
    stats = load_stats(args.stats_path)
    device = torch.device(args.device)

    model = FusionSegNet(n_classes=N_CLASSES_DEFAULT).to(device)
    ckpt = torch.load(args.seg_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    def _one_frame(sequence_dir, frame_idx):
        """Real per-frame inference pipeline, timed PER STAGE, not as one
        undifferentiated number -- a reviewer correctly asked for the
        actual bottleneck to be named, not just a total. Returns a dict
        of real wall-clock seconds per stage plus the total. Excludes
        only the one-time model/checkpoint load (done once above, not
        per frame, matching a real deployment where the model stays
        resident)."""
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        sweep = load_rellis_sweep(sequence_dir, frame_idx)
        t_load = time.perf_counter() - t0

        t0 = time.perf_counter()
        img = project_to_range_image(sweep, sm)
        t_project = time.perf_counter() - t0

        t0 = time.perf_counter()
        ground = compute_ground_prior(sweep, n_azimuth_bins=img.W)
        t_ground = time.perf_counter() - t0

        t0 = time.perf_counter()
        tensor_np = assemble_input_tensor(img, ground, stats)
        t_assemble = time.perf_counter() - t0

        t0 = time.perf_counter()
        x = torch.from_numpy(tensor_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            if args.fp16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
            else:
                logits = model(x)
            _pred = logits.argmax(dim=1).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()  # real GPU completion, not just kernel-launch return
        t_forward = time.perf_counter() - t0

        total = time.perf_counter() - t_start
        return {
            "load": t_load, "project": t_project, "ground_prior": t_ground,
            "assemble_tensor": t_assemble, "forward_pass": t_forward, "total": total,
        }

    print(f"Warming up on {len(warmup_items)} frames (CUDA kernel compilation/autotuning, excluded from reported stats)...")
    warmup_records = [_one_frame(s, i) for s, i in warmup_items]
    print(f"  warmup totals (s): {[round(r['total'], 3) for r in warmup_records]}")

    print(f"Timing {len(timed_items)} real frames...")
    records = [_one_frame(s, i) for s, i in timed_items]

    stage_names = ["load", "project", "ground_prior", "assemble_tensor", "forward_pass", "total"]
    stage_stats = {}
    for stage in stage_names:
        vals_ms = np.array([r[stage] for r in records]) * 1000.0
        stage_stats[stage] = {
            "mean_ms": float(np.mean(vals_ms)),
            "p50_ms": float(np.percentile(vals_ms, 50)),
            "p95_ms": float(np.percentile(vals_ms, 95)),
            "p99_ms": float(np.percentile(vals_ms, 99)),
            "max_ms": float(np.max(vals_ms)),
        }

    total_mean_ms = stage_stats["total"]["mean_ms"]
    fps_mean = 1000.0 / total_mean_ms
    bottleneck = max((s for s in stage_names if s != "total"), key=lambda s: stage_stats[s]["mean_ms"])

    print(f"\nReal single-pass inference latency, PER STAGE, {len(timed_items)} frames, device={args.device}:")
    print(f"{'stage':>18} {'mean_ms':>10} {'p50_ms':>10} {'p95_ms':>10} {'p99_ms':>10} {'max_ms':>10}")
    for stage in stage_names:
        s = stage_stats[stage]
        print(f"{stage:>18} {s['mean_ms']:>10.2f} {s['p50_ms']:>10.2f} {s['p95_ms']:>10.2f} {s['p99_ms']:>10.2f} {s['max_ms']:>10.2f}")
    print(f"\nTOTAL mean: {total_mean_ms:.2f} ms ({fps_mean:.1f} FPS)")
    print(f"BOTTLENECK STAGE (by mean): {bottleneck} ({stage_stats[bottleneck]['mean_ms']:.2f} ms, "
          f"{100 * stage_stats[bottleneck]['mean_ms'] / total_mean_ms:.1f}% of total)")
    print(f"\nHonest scope: NOT sped up by FrameCache, which only pays off on a SECOND pass over the same frame "
          f"(training's many epochs), not a genuinely live single-pass feed where every frame is seen exactly once.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inference_latency.json"
    with open(out_path, "w") as f:
        json.dump({
            "device": args.device,
            "n_frames_timed": len(timed_items),
            "n_warmup_frames": len(warmup_items),
            "stage_stats": stage_stats,
            "bottleneck_stage": bottleneck,
            "fps_mean": fps_mean,
            "warmup_totals_s": [r["total"] for r in warmup_records],
            "all_records_s": records,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
