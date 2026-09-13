"""
eval/track_rellis_sequence.py

The real, end-to-end integration `temporal/kalman_tracker.py`'s own
module docstring named as NOT done: feeding `MultiObjectTracker` a
continuous, real, frame-to-frame stream of detections from an actual
RELLIS-3D sequence, rather than only the synthetic ground-truth
trajectories `tests/test_kalman_tracker.py` uses.

Pipeline, all real, per frame IN TEMPORAL ORDER (NOT the shuffled
train/val split `perception.train.build_multi_sequence_splits` makes --
tracking needs a genuinely CONTIGUOUS sequence, frame i immediately
followed by frame i+1, so this script iterates one sequence's frames
0..N-1 directly):

1. Load the real sweep + run the trained segmentation checkpoint ->
   per-point class predictions (identical to every other eval script
   here).
2. `perception.geometric_instance_detector.detect_instances` -> real
   measured-geometry detections for this frame.
3. WORLD-FRAME TRANSFORM (the one genuinely new piece this script adds,
   not reused from any existing eval script): `detect_instances`
   returns centroids in the SENSOR's own local frame, which moves with
   the ego vehicle. Feeding raw sensor-frame centroids straight into a
   constant-velocity Kalman tracker would be physically wrong -- a
   perfectly static tree would appear to "fly backward" at the ego
   vehicle's own speed, and the tracker's constant-velocity model would
   dutifully (and wrongly) try to track that ego-induced motion. Each
   detection's centroid is transformed by that frame's own real
   `sweep.T_world` (sensor -> world, loaded from RELLIS-3D's real
   `poses.txt` -- confirmed present for all 5 sequences before writing
   this script, not assumed) into a common WORLD frame before being
   handed to `MultiObjectTracker`, which then tracks real object motion
   only, ego motion cancelled out by construction.
4. `dt` between consecutive frames: RELLIS-3D does not ship real
   per-frame timestamps (`perception.rellis_loader`'s own stated
   simplification -- a fixed FRAME_PERIOD_S synthesises one). Using
   that same fixed period here keeps `dt` consistent with how
   `sweep.timestamp` was itself constructed, rather than inventing a
   second, different assumption.

Honest scope, stated up front: this is real code run against a real
sequence, but it inherits whatever the segmentation checkpoint's own
real accuracy is (VEHICLE/PEDESTRIAN recall, see
DRISHTI_MASTER_BIBLE.md Part G.17) -- this script tests whether the
TRACKER behaves sensibly given real (imperfect) detections, not a new
claim about detection accuracy itself. It also does not attempt any
track-to-ground-truth-object identity matching (RELLIS-3D ships no
instance IDs) -- reported metrics are properties of the tracker's own
output (track count, confirmed-track lifespan, velocity sanity), not
a real MOTA/MOTP score against ground truth, which would need instance-
level annotations this dataset does not provide.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from perception.geometric_instance_detector import detect_instances
from perception.ground_prior import compute_ground_prior
from perception.input_tensor import assemble_input_tensor, load_stats
from perception.rellis_loader import load_rellis_labels, load_rellis_sweep
from perception.range_image import project_to_range_image
from perception.segnet import FusionSegNet, N_CLASSES_DEFAULT
from perception.taxonomy import DrishtiClass
from sensor.sensor_model import load_sensor_config
from temporal.kalman_tracker import MultiObjectTracker

FRAME_PERIOD_S = 0.1  # matches perception.rellis_loader.load_rellis_sweep's own default


def _per_point_ground_z(sweep, ground, img) -> np.ndarray:
    x, y = sweep.xyz[:, 0].astype(np.float64), sweep.xyz[:, 1].astype(np.float64)
    theta = np.arctan2(y, x)
    col = np.floor((0.5 * (1.0 - theta / np.pi)) * img.W).astype(np.int64)
    col = np.clip(col, 0, img.W - 1)
    if ground.column_ground_height:
        fallback = float(np.median(list(ground.column_ground_height.values())))
    else:
        fallback = 0.0
    z_ground = np.full(sweep.xyz.shape[0], fallback, dtype=np.float64)
    for c, h in ground.column_ground_height.items():
        z_ground[col == c] = h
    return z_ground


def _transform_to_world(centroid_xyz: np.ndarray, T_world: np.ndarray) -> np.ndarray:
    """centroid_xyz: (3,) sensor-frame. T_world: (4,4) sensor->world.
    Returns (3,) world-frame."""
    homog = np.array([centroid_xyz[0], centroid_xyz[1], centroid_xyz[2], 1.0])
    world = T_world @ homog
    return world[:3]


def main():
    parser = argparse.ArgumentParser(description="Real end-to-end Kalman tracker test on a real, temporally contiguous RELLIS-3D sequence")
    parser.add_argument("--sequence-dir", default="data/rellis/00000")
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--seg-checkpoint", default="checkpoints_multi_v3/best.pt")
    parser.add_argument("--stats-path", default="checkpoints_multi_v3/channel_stats.json")
    parser.add_argument("--n-frames", type=int, default=200, help="how many consecutive frames of the sequence to run (from frame 0)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="eval/out")
    args = parser.parse_args()

    sequence_dir = Path(args.sequence_dir)
    n_available = len(list((sequence_dir / "os1_cloud_node_kitti_bin").glob("*.bin")))
    n_frames = min(args.n_frames, n_available)
    print(f"Running real contiguous tracking over {sequence_dir} frames [0, {n_frames}) of {n_available} available")

    sm = load_sensor_config(args.sensor_config)
    stats = load_stats(args.stats_path)
    device = torch.device(args.device)

    model = FusionSegNet(n_classes=N_CLASSES_DEFAULT).to(device)
    ckpt = torch.load(args.seg_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tracker = MultiObjectTracker()
    class_names = {int(c): c.name for c in DrishtiClass}

    per_frame_log = []
    max_confirmed_this_run = 0
    track_id_first_seen = {}
    track_id_last_seen = {}

    with torch.no_grad():
        for frame_idx in range(n_frames):
            sweep = load_rellis_sweep(sequence_dir, frame_idx)
            img = project_to_range_image(sweep, sm)
            ground = compute_ground_prior(sweep, n_azimuth_bins=img.W)
            tensor_np = assemble_input_tensor(img, ground, stats)
            x = torch.from_numpy(tensor_np).float().unsqueeze(0).to(device)

            seg_out = model(x)
            pred_grid = seg_out.argmax(dim=1)[0].cpu().numpy()
            touched = img.point_index >= 0
            n_points = sweep.xyz.shape[0]
            pred_class_per_point = np.full(n_points, int(DrishtiClass.UNKNOWN), dtype=np.int64)
            rows, cols = np.nonzero(touched)
            pred_class_per_point[img.point_index[touched]] = pred_grid[rows, cols]

            z_ground = _per_point_ground_z(sweep, ground, img)
            xyz64 = sweep.xyz.astype(np.float64)
            detections = detect_instances(xyz64, pred_class_per_point, z_ground, sm)

            if detections:
                world_xy = np.array([_transform_to_world(d.centroid_xyz, sweep.T_world)[:2] for d in detections])
                det_classes = np.array([d.drishti_class for d in detections], dtype=np.int64)
            else:
                world_xy = np.zeros((0, 2))
                det_classes = np.zeros((0,), dtype=np.int64)

            tracks = tracker.update(world_xy, det_classes, dt=FRAME_PERIOD_S)
            confirmed = tracker.confirmed_tracks()
            max_confirmed_this_run = max(max_confirmed_this_run, len(confirmed))

            for t in tracks:
                if t.track_id not in track_id_first_seen:
                    track_id_first_seen[t.track_id] = frame_idx
                track_id_last_seen[t.track_id] = frame_idx

            per_frame_log.append({
                "frame_idx": frame_idx,
                "n_detections": len(detections),
                "n_tracks_total": len(tracks),
                "n_tracks_confirmed": len(confirmed),
                "confirmed_speeds_mps": [float(np.linalg.norm(t.velocity)) for t in confirmed],
            })

            if frame_idx % 25 == 0:
                print(f"  frame {frame_idx}/{n_frames}: {len(detections)} detections -> "
                      f"{len(tracks)} tracks ({len(confirmed)} confirmed)")

    n_total_tracks_created = len(track_id_first_seen)
    lifespans = [track_id_last_seen[tid] - track_id_first_seen[tid] + 1 for tid in track_id_first_seen]
    all_confirmed_speeds = [s for f in per_frame_log for s in f["confirmed_speeds_mps"]]

    print(f"\nReal tracking run over {n_frames} contiguous frames of {sequence_dir}:")
    print(f"  Total distinct track IDs ever created: {n_total_tracks_created}")
    print(f"  Max simultaneously-confirmed tracks in one frame: {max_confirmed_this_run}")
    print(f"  Mean track lifespan (frames, tentative+confirmed): {np.mean(lifespans) if lifespans else float('nan'):.1f}")
    print(f"  Median track lifespan (frames): {np.median(lifespans) if lifespans else float('nan'):.1f}")
    if all_confirmed_speeds:
        print(f"  Confirmed-track speed (m/s): mean={np.mean(all_confirmed_speeds):.2f}, "
              f"max={np.max(all_confirmed_speeds):.2f} -- a real off-road UGV/pedestrian/vehicle scene; "
              f"a max far above ~20 m/s here would indicate the world-frame transform or association "
              f"is doing something wrong, not a real object.")
    else:
        print("  No track ever reached CONFIRMED status in this run.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "track_rellis_sequence.json"
    with open(out_path, "w") as f:
        json.dump({
            "sequence_dir": str(sequence_dir),
            "n_frames": n_frames,
            "n_total_tracks_created": n_total_tracks_created,
            "max_simultaneously_confirmed": max_confirmed_this_run,
            "mean_track_lifespan_frames": float(np.mean(lifespans)) if lifespans else None,
            "median_track_lifespan_frames": float(np.median(lifespans)) if lifespans else None,
            "mean_confirmed_speed_mps": float(np.mean(all_confirmed_speeds)) if all_confirmed_speeds else None,
            "max_confirmed_speed_mps": float(np.max(all_confirmed_speeds)) if all_confirmed_speeds else None,
            "per_frame": per_frame_log,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
