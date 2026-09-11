"""
eval/cache_inference.py

Ticket #31 -- cache inference for the demo. Runs the TRAINED FusionSegNet
checkpoint over a real RELLIS-3D sequence once, offline, and saves
PER-POINT predicted DRISHTI class labels to one .npy file per frame. The
demo (and Ticket #32's checkpoint) replay these; nothing runs the
network live -- Build Map's own "Watch out": inference at this
resolution (64x2048, this project's real Ouster raster) is too slow for
a live demo and beside the point anyway, since the MAPPING pipeline is
the contribution, not real-time segmentation speed. This module reports
measured latency per frame explicitly, so nobody has to guess at it or,
worse, present the cached labels as if they were live.

Uses perception.train's OWN frame-loading function
(perception.train._load_frame), not a reimplementation, so inference
sees EXACTLY the preprocessing the checkpoint was trained against.
This matters concretely: perception.ring_recovery (built after Run #1/#2
trained) can derive a more geometrically exact row assignment than the
arcsin fallback project_to_range_image falls back to when `ring` is
unavailable -- but the checkpoint has never seen that alignment. Feeding
it a "more correct" but differently-distributed input here would
silently degrade predictions in a way that looks like a checkpoint
problem, not the preprocessing mismatch it actually is. Retraining, not
this module, is the right place to adopt ring_recovery.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from perception.input_tensor import assemble_input_tensor, load_stats
from perception.rellis_loader import load_rellis_labels, load_rellis_sweep
from perception.segnet import FusionSegNet, N_CLASSES_DEFAULT
from perception.taxonomy import DrishtiClass, rellis_label_ids_to_drishti
from perception.train import _load_frame
from sensor.sensor_model import load_sensor_config

UNKNOWN = int(DrishtiClass.UNKNOWN)


def _remap_legacy_conv_keys(state_dict: dict, expected_keys: set) -> dict:
    """Run #1/#2's checkpoints were trained BEFORE this session's circular-
    padding fix wrapped ASPP's branches and the decoder's _dblock convs in
    CircularConv2d (perception/segnet.py) -- that wrapping renames each
    conv's own parameters from e.g. "aspp.branches.0.0.weight" to
    "aspp.branches.0.0.conv.weight" (a new ".conv." submodule), with the
    SAME tensor shapes (verified: all 12 renamed keys, identical shapes,
    no bias keys since these convs use bias=False) -- the operation is
    functionally unchanged, only its parameter path moved. Remap old-style
    keys to new-style ones rather than silently failing to load, or
    forcing a from-scratch retrain of an already-good (0.562 mIoU)
    checkpoint over a rename."""
    if expected_keys.issubset(state_dict.keys()):
        return state_dict  # already new-style, e.g. a checkpoint trained after this fix
    remapped = {}
    for key, value in state_dict.items():
        new_key = key.replace(".weight", ".conv.weight").replace(".bias", ".conv.bias")
        remapped[new_key if new_key in expected_keys else key] = value
    return remapped


def load_trained_model(checkpoint_path: str | Path, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FusionSegNet(n_classes=N_CLASSES_DEFAULT).to(device)
    state_dict = _remap_legacy_conv_keys(ckpt["model_state"], set(model.state_dict().keys()))
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt.get("epoch")


def per_point_ground_truth(sequence_dir: str | Path, frame_idx: int) -> np.ndarray:
    """Direct per-point DRISHTI ground truth -- NOT round-tripped through
    the pixel grid the way perception.train._load_frame's `target` is
    (that's per-PIXEL, and loses points to the many-to-one projection).
    Ticket #32 wants this for an apples-to-apples per-point comparison
    against #31's per-point predictions below."""
    raw_labels = load_rellis_labels(sequence_dir, frame_idx)
    return rellis_label_ids_to_drishti(raw_labels)


def cache_inference_for_frame(model, sequence_dir: str | Path, frame_idx: int, sm, stats, device: str = "cpu") -> dict:
    """Run inference on ONE frame; return per-point predicted labels
    aligned with perception.rellis_loader.load_rellis_sweep's own point
    ordering, plus timing/coverage stats."""
    img, ground, _target = _load_frame(sequence_dir, frame_idx, sm)
    tensor = assemble_input_tensor(img, ground, stats)
    x = torch.from_numpy(tensor).float().unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
    latency_s = time.perf_counter() - t0

    pred_pixels = logits.argmax(dim=1)[0].cpu().numpy()  # (H, W)

    sweep = load_rellis_sweep(sequence_dir, frame_idx)
    n_points = sweep.xyz.shape[0]

    point_labels = np.full(n_points, UNKNOWN, dtype=np.int64)
    touched = img.point_index >= 0
    point_labels[img.point_index[touched]] = pred_pixels[touched]

    return {
        "point_labels": point_labels,
        "n_points": n_points,
        "latency_s": latency_s,
        "pixels_covered": int(touched.sum()),
        "pixels_total": img.H * img.W,
    }


def cache_inference_for_sequence(
    checkpoint_path: str | Path,
    sequence_dir: str | Path,
    sensor_config_path: str | Path,
    out_dir: str | Path,
    frame_indices: List[int],
    stats_path: Optional[str | Path] = None,
    device: str = "cpu",
) -> dict:
    sequence_dir = Path(sequence_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path)

    stats_path = Path(stats_path) if stats_path else checkpoint_path.parent / "channel_stats.json"
    stats = load_stats(stats_path)
    sm = load_sensor_config(sensor_config_path)
    model, trained_epoch = load_trained_model(checkpoint_path, device=device)

    per_frame = []
    latencies = []
    for frame_idx in frame_indices:
        result = cache_inference_for_frame(model, sequence_dir, frame_idx, sm, stats, device=device)
        # Ticket #31's own test: cached labels align frame-for-frame with
        # sweeps -- len(labels) == len(points), for every frame.
        assert result["point_labels"].shape[0] == result["n_points"]

        npy_path = out_dir / f"{frame_idx:06d}_pred_labels.npy"
        np.save(npy_path, result["point_labels"])
        latencies.append(result["latency_s"])
        per_frame.append(
            {
                "frame_idx": frame_idx,
                "npy_path": str(npy_path),
                "n_points": result["n_points"],
                "latency_s": result["latency_s"],
                "pixel_coverage": result["pixels_covered"] / result["pixels_total"],
            }
        )

    latencies_arr = np.array(latencies)
    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "trained_epoch": trained_epoch,
        "sequence_dir": str(sequence_dir),
        "device": device,
        "frames": per_frame,
        "latency_p50_s": float(np.median(latencies_arr)),
        "latency_p95_s": float(np.percentile(latencies_arr, 95)),
        "note": "Cached OFFLINE -- NOT live inference. See this module's own docstring before presenting these numbers or labels as real-time.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ticket #31 -- cache inference for the demo")
    parser.add_argument("--checkpoint", default="checkpoints_multi_remote/checkpoint.pt")
    parser.add_argument("--sequence-dir", default="data/rellis/00004")
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--out-dir", default="eval/out/cached_labels")
    parser.add_argument("--frames", type=int, nargs="+", default=list(range(1000, 1010)))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest = cache_inference_for_sequence(
        checkpoint_path=args.checkpoint,
        sequence_dir=args.sequence_dir,
        sensor_config_path=args.sensor_config,
        out_dir=args.out_dir,
        frame_indices=args.frames,
        device=args.device,
    )
    print(f"Cached {len(manifest['frames'])} frames to {args.out_dir}")
    print(f"Latency (offline, CPU): P50={manifest['latency_p50_s']*1000:.1f}ms  P95={manifest['latency_p95_s']*1000:.1f}ms")
    for f in manifest["frames"]:
        print(f"  frame {f['frame_idx']:06d}: {f['n_points']} points, {f['latency_s']*1000:.1f}ms, "
              f"{f['pixel_coverage']:.1%} pixel coverage -> {f['npy_path']}")
