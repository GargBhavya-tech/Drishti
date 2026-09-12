"""
eval/eval_nuscenes.py

Zero-shot cross-dataset evaluation: evaluates a RELLIS-3D-trained FusionSegNet
on nuScenes-mini LiDAR point clouds and lidarseg semantic annotations.

Maps nuScenes 32 classes into the 10-class DRISHTI taxonomy (perception/taxonomy.py)
and evaluates both:
1. Direct mode: feed (32, 1080) range images directly to FusionSegNet.
2. Resample mode: interpolate (32, 1080) to (64, 2048) to match the vertical
   receptive field the network was trained on, then evaluate predictions.

Reports point-level mIoU, per-class IoU, precision, recall, and confusion matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from eval.metrics import ConfusionMatrix, accumulate_confusion
from perception.ground_prior import compute_ground_prior
from perception.input_tensor import (
    ChannelStats,
    assemble_input_tensor,
    load_stats,
)
from perception.nuscenes_loader import load_nuscenes_sweep
from perception.range_image import RangeImage, project_to_range_image
from perception.segnet import FusionSegNet, N_CLASSES_DEFAULT
from perception.sweep import Sweep
from perception.taxonomy import (
    NUSCENES_LIDARSEG_TO_DRISHTI,
    DrishtiClass,
)
from sensor.sensor_model import SensorConfig, load_sensor_config


CLASS_NAMES = [c.name for c in DrishtiClass]


def build_lidarseg_lut(nusc) -> np.ndarray:
    """Build a lookup table mapping raw uint8 lidarseg category index (0..31)
    to DrishtiClass integer value."""
    lut = np.full(256, int(DrishtiClass.UNKNOWN), dtype=np.int64)
    for idx, name in nusc.lidarseg_idx2name_mapping.items():
        drishti_class = NUSCENES_LIDARSEG_TO_DRISHTI.get(name, DrishtiClass.UNKNOWN)
        lut[int(idx)] = int(drishti_class)
    return lut


def load_point_predictions_and_truth(
    nusc,
    sample_token: str,
    model: torch.nn.Module,
    sm: SensorConfig,
    stats: ChannelStats,
    lut: np.ndarray,
    device: torch.device,
    mode: str = "direct",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run inference for one sample and return:
    (point_truth, point_pred, range_truth, range_pred)
    """
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"]["LIDAR_TOP"]

    # Load canonical Sweep
    sweep = load_nuscenes_sweep(nusc, sample_token)

    # Load raw point lidarseg labels
    lidarseg_record = nusc.get("lidarseg", sd_token)
    lidarseg_path = os.path.join(nusc.dataroot, lidarseg_record["filename"])
    raw_labels = np.fromfile(lidarseg_path, dtype=np.uint8)

    # Map to DrishtiClass
    point_truth = lut[raw_labels]

    # Project to range image using HDL-32E sensor model
    img = project_to_range_image(sweep, sm)
    ground = compute_ground_prior(sweep, n_azimuth_bins=img.W)

    # Assemble 9-channel input tensor
    tensor_np = assemble_input_tensor(img, ground, stats)
    tensor = torch.from_numpy(tensor_np).float().unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        if mode == "resample":
            # Upsample to 64x2048 to match RELLIS training resolution
            orig_h, orig_w = tensor.shape[2:]
            tensor_resampled = F.interpolate(
                tensor, size=(64, 2048), mode="bilinear", align_corners=False
            )
            logits = model(tensor_resampled)
            # Downsample logits back to (32, 1080)
            logits = F.interpolate(
                logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )
        else:
            # Direct inference at (32, 1080)
            logits = model(tensor)

        pred_grid = torch.argmax(logits[0], dim=0).cpu().numpy().astype(np.int64)

    # Map grid predictions back to each 3D point in the sweep
    xyz = sweep.xyz.astype(np.float64)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x * x + y * y + z * z)

    H, W = img.H, img.W
    u = np.floor(0.5 * (1.0 - np.arctan2(y, x) / np.pi) * W).astype(np.int64)
    u = np.clip(u, 0, W - 1)

    ring = sweep.ring.astype(np.int64)
    has_ring = ring >= 0
    v = np.where(has_ring, ring, 0)
    if not np.all(has_ring):
        phi_min = sm.phi_max_rad - (sm.n_beams - 1) * sm.d_phi_rad
        fov = sm.phi_max_rad - phi_min
        with np.errstate(invalid="ignore", divide="ignore"):
            elevation = np.arcsin(np.clip(z / np.maximum(r, 1e-9), -1.0, 1.0))
        v_fallback = np.floor((1.0 - (elevation - phi_min) / fov) * H).astype(np.int64)
        v = np.where(has_ring, v, v_fallback)
    v = np.clip(v, 0, H - 1)

    point_pred = pred_grid[v, u]

    # Range-level truth (for valid pixels only)
    touched = img.point_index >= 0
    range_truth = point_truth[img.point_index[touched]]
    range_pred = pred_grid[touched]

    return point_truth, point_pred, range_truth, range_pred


def compute_metrics_from_matrix(matrix: np.ndarray) -> Dict[str, Any]:
    """Compute per-class IoU, precision, recall, and overall mIoU."""
    n_classes = matrix.shape[0]
    tp = np.diag(matrix).astype(np.float64)
    fp = matrix.sum(axis=0).astype(np.float64) - tp
    fn = matrix.sum(axis=1).astype(np.float64) - tp
    total_true = matrix.sum(axis=1).astype(np.float64)

    denom = tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
        precision = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
        recall = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)

    valid_iou = iou[~np.isnan(iou)]
    miou = float(np.mean(valid_iou)) if valid_iou.size > 0 else float("nan")

    per_class = []
    for c in range(n_classes):
        per_class.append({
            "class_id": c,
            "name": CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}",
            "iou": None if np.isnan(iou[c]) else float(iou[c]),
            "precision": None if np.isnan(precision[c]) else float(precision[c]),
            "recall": None if np.isnan(recall[c]) else float(recall[c]),
            "point_count": int(total_true[c]),
        })

    return {
        "miou": miou,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "total_points": int(matrix.sum()),
    }


def evaluate_mode(
    nusc,
    sample_tokens: List[str],
    model: torch.nn.Module,
    sm: SensorConfig,
    stats: ChannelStats,
    lut: np.ndarray,
    device: torch.device,
    mode: str,
    progress_interval: int = 20,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate all samples under a specific resolution mode."""
    print(f"\n=======================================================")
    print(f" Running Evaluation: Mode = '{mode.upper()}' ({len(sample_tokens)} frames)")
    print(f"=======================================================")

    point_matrix = np.zeros((N_CLASSES_DEFAULT, N_CLASSES_DEFAULT), dtype=np.int64)
    range_matrix = np.zeros((N_CLASSES_DEFAULT, N_CLASSES_DEFAULT), dtype=np.int64)

    t0 = time.time()
    for i, token in enumerate(sample_tokens):
        t_frame_start = time.time()
        pt_truth, pt_pred, rng_truth, rng_pred = load_point_predictions_and_truth(
            nusc=nusc,
            sample_token=token,
            model=model,
            sm=sm,
            stats=stats,
            lut=lut,
            device=device,
            mode=mode,
        )

        point_matrix = accumulate_confusion(pt_truth, pt_pred, N_CLASSES_DEFAULT, point_matrix)
        range_matrix = accumulate_confusion(rng_truth, rng_pred, N_CLASSES_DEFAULT, range_matrix)

        if (i + 1) % progress_interval == 0 or (i + 1) == len(sample_tokens):
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            # Current running mIoU
            curr_tp = np.diag(point_matrix).astype(np.float64)
            curr_denom = (
                point_matrix.sum(axis=0) + point_matrix.sum(axis=1) - curr_tp
            ).astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                curr_iou = np.where(curr_denom > 0, curr_tp / curr_denom, np.nan)
            valid = curr_iou[~np.isnan(curr_iou)]
            curr_miou = float(np.mean(valid)) if valid.size > 0 else 0.0

            print(
                f"[{i+1:3d}/{len(sample_tokens):3d}] "
                f"FPS: {fps:4.1f} | "
                f"Running Point mIoU: {curr_miou * 100:5.2f}% | "
                f"Elapsed: {elapsed:5.1f}s",
                flush=True,
            )

    print()
    total_time = time.time() - t0
    print(f"Completed {len(sample_tokens)} frames in {total_time:.1f}s ({len(sample_tokens)/total_time:.1f} FPS)")

    point_metrics = compute_metrics_from_matrix(point_matrix)
    range_metrics = compute_metrics_from_matrix(range_matrix)
    point_metrics["total_time_s"] = total_time
    point_metrics["fps"] = len(sample_tokens) / total_time

    return point_metrics, range_metrics


def print_results_table(metrics: Dict[str, Any], title: str = "RESULTS") -> None:
    """Print a clean Markdown-like table of per-class results."""
    print(f"\n--- {title} ---")
    print(f"Overall mIoU: {metrics['miou'] * 100:.2f}% (Total Points: {metrics['total_points']:,})")
    print(f"Speed: {metrics.get('fps', 0):.1f} FPS\n")
    print(f"{'Class':<20} | {'IoU':<8} | {'Precision':<10} | {'Recall':<10} | {'Points':<10}")
    print("-" * 68)
    for c in metrics["per_class"]:
        iou_str = f"{c['iou']*100:6.2f}%" if c["iou"] is not None else "   N/A "
        prec_str = f"{c['precision']*100:6.2f}%" if c["precision"] is not None else "   N/A "
        rec_str = f"{c['recall']*100:6.2f}%" if c["recall"] is not None else "   N/A "
        print(f"{c['name']:<20} | {iou_str} | {prec_str} | {rec_str} | {c['point_count']:<10,}")
    print("-" * 68)


def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation of FusionSegNet on nuScenes-mini")
    parser.add_argument("--dataroot", default="data/nuscenes", help="Path to nuScenes dataset")
    parser.add_argument("--version", default="v1.0-mini", help="nuScenes version (v1.0-mini)")
    parser.add_argument("--checkpoint", default="checkpoints_multi_v2/best.pt", help="Model checkpoint")
    parser.add_argument("--sensor-config", default="configs/sensor_hdl32e.yaml", help="Sensor config YAML")
    parser.add_argument("--stats-path", default="checkpoints_multi_v2/channel_stats.json", help="Channel normalization stats")
    parser.add_argument("--mode", choices=["direct", "resample", "both"], default="both", help="Inference resolution mode")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of keyframes evaluated")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--out-dir", default="eval/out", help="Output directory for results")
    args = parser.parse_args()

    print("==================================================================")
    print("  DRISHTI: Zero-Shot Domain Generalization (RELLIS-3D -> nuScenes)")
    print("==================================================================")
    print(f"Device:         {args.device}")
    print(f"Dataroot:       {args.dataroot} ({args.version})")
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Sensor Config:  {args.sensor_config}")
    print(f"Channel Stats:  {args.stats_path}")
    print(f"Mode:           {args.mode}")

    # 1. Initialize nuScenes
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    lut = build_lidarseg_lut(nusc)

    # 2. Collect all sample tokens that have lidarseg data
    sample_tokens = []
    for scene in nusc.scene:
        cur = scene["first_sample_token"]
        while cur:
            sd_token = nusc.get("sample", cur)["data"]["LIDAR_TOP"]
            # Verify lidarseg record exists for this sample
            try:
                nusc.get("lidarseg", sd_token)
                sample_tokens.append(cur)
            except KeyError:
                pass
            cur = nusc.get("sample", cur)["next"]

    if args.limit:
        sample_tokens = sample_tokens[: args.limit]

    print(f"Annotated keyframes found: {len(sample_tokens)} across {len(nusc.scene)} scenes")
    if not sample_tokens:
        print("ERROR: No lidarseg samples found in dataroot.")
        sys.exit(1)

    # 3. Load SensorConfig and ChannelStats
    sm = load_sensor_config(args.sensor_config)
    stats = load_stats(args.stats_path)

    # 4. Load Model
    device = torch.device(args.device)
    model = FusionSegNet(n_classes=N_CLASSES_DEFAULT)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    epoch_val = ckpt.get("epoch", "unknown") if isinstance(ckpt, dict) else "unknown"
    print(f"Model loaded successfully from epoch {epoch_val}")

    # 5. Execute Evaluation
    modes_to_run = ["direct", "resample"] if args.mode == "both" else [args.mode]
    results = {}

    for m in modes_to_run:
        pt_metrics, rng_metrics = evaluate_mode(
            nusc=nusc,
            sample_tokens=sample_tokens,
            model=model,
            sm=sm,
            stats=stats,
            lut=lut,
            device=device,
            mode=m,
        )
        results[m] = {
            "point_level": pt_metrics,
            "range_level": rng_metrics,
        }
        print_results_table(pt_metrics, title=f"POINT-LEVEL EVALUATION (MODE: {m.upper()})")

    # 6. Comparative Summary if both were run
    if len(modes_to_run) > 1:
        print("\n==================================================================")
        print("  COMPARISON: Direct (32x1080) vs. Resampled (64x2048)")
        print("==================================================================")
        d_miou = results["direct"]["point_level"]["miou"] * 100
        r_miou = results["resample"]["point_level"]["miou"] * 100
        d_fps = results["direct"]["point_level"]["fps"]
        r_fps = results["resample"]["point_level"]["fps"]
        print(f"Direct (32x1080):    mIoU = {d_miou:5.2f}% | Speed = {d_fps:4.1f} FPS")
        print(f"Resampled (64x2048): mIoU = {r_miou:5.2f}% | Speed = {r_fps:4.1f} FPS")
        diff = r_miou - d_miou
        print(f"Delta:               {'+' if diff >= 0 else ''}{diff:.2f}% mIoU with resolution matching")
        print("==================================================================")

    # 7. Save output
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nuscenes_zero_shot.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint": str(args.checkpoint),
                "dataset": "nuScenes-mini",
                "num_frames": len(sample_tokens),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nDetailed metrics saved to: {out_path}")


if __name__ == "__main__":
    main()
