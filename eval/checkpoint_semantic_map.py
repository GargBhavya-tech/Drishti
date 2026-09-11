"""
eval/checkpoint_semantic_map.py

Ticket #32 -- checkpoint: the 2.5D map coloured by the TRAINED model's
PREDICTED class looks right, compared side by side against the SAME map
coloured by real ground truth. Blocked by #31 (cached predictions,
eval/cache_inference.py) and #22 (the checkpoint pattern this reuses:
run_checkpoint(out_dir) -> dict, matplotlib Agg, one PNG per view).

DEVIATION FROM THE LITERAL TICKET, matching #22's own precedent: #22
used a SYNTHETIC scene because no real sweep data existed in that
environment yet, and #32's own spec text says to compare "against the
ground-truth-label version from #22" -- i.e. that synthetic scene. Real
RELLIS-3D data AND a real trained checkpoint (Run #2, val mIoU 0.562,
see TRAINING_RESULTS.md) both exist locally now, so this checkpoint uses
a REAL frame instead of replaying #22's synthetic one -- a stronger
fulfilment of the ticket's own stated intent ("PS requirements for
segmentation and the 2.5D grid are now both demonstrably met") than a
literal #22 replay would be, for the same reason eval/measure_ouster_config.py
preferred real data over a synthetic stand-in once real data existed.

One real RELLIS-3D frame, transformed into WORLD coordinates via its own
recorded pose (sweep.T_world -- a real trajectory from poses.txt, not an
assumed origin), scattered into TWO separate Clipmaps with the SAME
points: one classed by real ground truth, one by #31's cached
predictions. Same colour palette as eval/checkpoint_first_map.py, for
visual consistency across this project's checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.cache_inference import cache_inference_for_sequence, per_point_ground_truth
from eval.checkpoint_first_map import CLASS_COLORS
from grid.cell import decode_class_conf
from grid.clipmap import Clipmap
from grid.scatter import scatter, scatter_class
from perception.rellis_loader import load_rellis_sweep
from sensor.sensor_model import load_sensor_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints_multi_remote" / "checkpoint.pt"

CROP_HALF_EXTENT_M = 20.0  # requested crop; clipped to L0's actual window, same pattern as checkpoint_first_map.py


def _real_frame_world_points(sequence_dir, frame_idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real sweep -> (world_xyz, ego_xy, valid_mask). Filters out
    zero-range placeholder points BEFORE transforming to world -- the
    raw RELLIS raster is ~45% invalid (0,0,0) points (see
    perception/ring_recovery.py's own docstring for how this was
    discovered), and transforming an untouched zero vector by T_world
    would place every one of them exactly at the ego position, flooding
    the map with a fake cluster at the vehicle."""
    sweep = load_rellis_sweep(sequence_dir, frame_idx)
    r = np.linalg.norm(sweep.xyz, axis=1)
    valid = r > 1e-6

    xyz_h = np.hstack([sweep.xyz[valid], np.ones((valid.sum(), 1), dtype=np.float64)])
    world = (sweep.T_world @ xyz_h.T).T[:, :3]
    ego_xy = sweep.T_world[:2, 3]
    return world, ego_xy, valid


def _build_class_map(cm: Clipmap, xyz_world: np.ndarray, class_ids: np.ndarray, ego_xy: np.ndarray, level: int = 0):
    cm.scroll_to(float(ego_xy[0]), float(ego_xy[1]))
    scatter(cm, xyz_world)
    scatter_class(cm, xyz_world, class_ids)

    c_l = cm.levels[level].cell_size_m
    half_cells = min(cm.N // 2, int(CROP_HALF_EXTENT_M / c_l) + 2)
    center = cm.N // 2  # world_to_global(ego) sits at the window centre by construction (scroll_to's own convention)
    lo = max(0, center - half_cells)
    hi = min(cm.N, center + half_cells)

    flags = cm.flags[level].reshape(cm.N, cm.N)[lo:hi, lo:hi]
    class_conf = cm.class_conf[level].reshape(cm.N, cm.N)[lo:hi, lo:hi]
    return flags, class_conf


def _render_class_map(flags: np.ndarray, class_conf: np.ndarray, title: str, out_path: Path) -> None:
    observed = flags != 0
    rgb = np.ones(flags.shape + (3,))  # white background for unobserved
    class_ids = np.zeros(flags.shape, dtype=np.int64)
    class_ids[observed] = [decode_class_conf(int(b))[0] for b in class_conf[observed]]

    for cid, color in CLASS_COLORS.items():
        mask = observed & (class_ids == cid)
        rgb[mask] = color

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("storage column (si)")
    ax.set_ylabel("storage row (sj)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_checkpoint(
    out_dir: Path = DEFAULT_OUT_DIR,
    sequence_dir: Path = REPO_ROOT / "data" / "rellis" / "00004",
    frame_idx: int = 1000,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    sensor_config_path: Path = CONFIGS / "sensor_ouster_os1_64.yaml",
    cached_labels_dir: Path | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sequence_dir = Path(sequence_dir)

    sm = load_sensor_config(sensor_config_path)

    # Ticket #31's cache, run fresh for this one frame if the caller
    # didn't already supply a cache directory -- keeps this checkpoint
    # runnable standalone (`python -m eval.checkpoint_semantic_map`)
    # without a separate manual step, while still going through the
    # SAME cached-inference path the real pipeline uses (never runs the
    # network "live" inside a checkpoint script either).
    if cached_labels_dir is None:
        cache_manifest = cache_inference_for_sequence(
            checkpoint_path=checkpoint_path,
            sequence_dir=sequence_dir,
            sensor_config_path=sensor_config_path,
            out_dir=out_dir / "cached_labels",
            frame_indices=[frame_idx],
            device="cpu",
        )
        pred_npy_path = Path(cache_manifest["frames"][0]["npy_path"])
    else:
        pred_npy_path = Path(cached_labels_dir) / f"{frame_idx:06d}_pred_labels.npy"

    world_xyz, ego_xy, valid = _real_frame_world_points(sequence_dir, frame_idx)

    gt_labels_all = per_point_ground_truth(sequence_dir, frame_idx)
    pred_labels_all = np.load(pred_npy_path)
    gt_labels = gt_labels_all[valid]
    pred_labels = pred_labels_all[valid]

    cm_gt = Clipmap(sm, n_levels=4, N=512, c0=0.05)
    cm_pred = Clipmap(sm, n_levels=4, N=512, c0=0.05)

    flags_gt, conf_gt = _build_class_map(cm_gt, world_xyz, gt_labels, ego_xy)
    flags_pred, conf_pred = _build_class_map(cm_pred, world_xyz, pred_labels, ego_xy)

    gt_path = out_dir / "checkpoint_semantic_map_ground_truth.png"
    pred_path = out_dir / "checkpoint_semantic_map_predicted.png"
    _render_class_map(flags_gt, conf_gt, f"Ground truth -- {sequence_dir.name} frame {frame_idx}\n(REAL RELLIS-3D data)", gt_path)
    _render_class_map(flags_pred, conf_pred, f"Predicted (checkpoint epoch 19, val mIoU 0.562)\n{sequence_dir.name} frame {frame_idx}", pred_path)

    agreement = float(np.mean(gt_labels == pred_labels)) if gt_labels.size else 0.0

    return {
        "sequence_dir": str(sequence_dir),
        "frame_idx": frame_idx,
        "n_points_used": int(valid.sum()),
        "per_point_agreement": agreement,
        "ground_truth_png": gt_path,
        "predicted_png": pred_path,
        "touched_cells_gt": int(np.count_nonzero(flags_gt)),
        "touched_cells_pred": int(np.count_nonzero(flags_pred)),
    }


if __name__ == "__main__":
    result = run_checkpoint()
    print(f"Frame: {result['sequence_dir']} #{result['frame_idx']} ({result['n_points_used']} valid points)")
    print(f"Per-point label agreement (pred vs ground truth): {result['per_point_agreement']:.1%}")
    print(f"Ground truth map:  {result['ground_truth_png']}")
    print(f"Predicted map:     {result['predicted_png']}")
