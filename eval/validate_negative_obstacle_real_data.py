"""
eval/validate_negative_obstacle_real_data.py

Phase 4's real-data validation pass -- Bible Part 28's own required
checklist, step 8: "Negative obstacle. Approach a ditch; then drive a
hillside and confirm zero false detections" against REAL data. Every
existing test for observability/ground_plane.py and
observability/negative_obstacle.py (Tickets #35, #36) uses hand-
constructed SYNTHETIC geometry -- a perfect flat plane, a perfect
constant 8-degree slope. That proves the discriminator's MATH is right;
it cannot prove the discriminator survives genuinely irregular off-road
terrain, which synthetic geometry cannot manufacture by construction.
This module is that proof, run against real local RELLIS-3D sequences
(a real, if short, "drive a hillside" in the Bible's own words).

**Finding (2026-09-11, 20 real consecutive frames, sequence 00004
frames 1000-1019):** run exactly as Ticket #36 specifies (no-return
cells substituted with usable_range_m, no carving), ~18% of the whole
(ring, azimuth) grid gets flagged EVERY frame, and thousands of world
cells get promoted to NEGATIVE_OBSTACLE. This is NOT evidence the ring-
to-ring discriminator itself is broken -- isolating genuinely-measured
returns only (skipping the no-return substitution entirely, as a proxy
for what carving would have marked OCCLUDED) drops the flagged fraction
from ~50% of scoreable cells to ~7.5%, and the ground-plane fit's own
residual stays small (mean ~0.11m, max ~1.4m across the sequence) --
the LOCAL LINE FIT is behaving correctly. The dominant driver is exactly
the failure mode Ticket #36's own "Watch out" #2 already names:
"Occlusion behind a large vehicle read as a ditch ... Order matters --
carve first, then test." Real off-road terrain is dense with vegetation
that absorbs a beam well short of any assumed max range, and this
validation deliberately does not run the carve step first (see Scope
below) -- so it is reproducing exactly the failure the ticket warned
about, using real data instead of the synthetic wall-occlusion test
that already existed. The remaining ~7.5% "measured-only" rate reflects
genuine real-terrain roughness a 5-point local line fit doesn't fully
smooth out -- smaller, and a separate, more inherent limitation.

**What this means for the build:** #33/#34 (raycast + carving) are
already built and correct; what's not yet built is running them BEFORE
the negative-obstacle detector on a real sequence -- full raycasting at
real point-cloud density (131,072 beams/frame) with this codebase's
current per-beam-Python-call DDA implementation would be slow enough to
warrant its own optimisation pass, so it is called out here as the
concrete next step rather than attempted inline.

Scope, stated explicitly rather than silently assumed: this does NOT
run the full raycast/carve pipeline (Tickets #33-34) first, so
`occluded_mask` is all-False throughout. That means a detection here
could be a genuine ring-to-ring geometric anomaly (what #35/#36 exist to
catch) OR unmodeled occlusion -- a bush or tree trunk blocking the
ground return produces almost the same "ground return missing/pushed
back" signature a real negative obstacle does (Bible Part 10.3's own
named failure mode: "Occlusion behind a large vehicle read as a ditch
... Carve first, then test"). Off-road terrain is thick with exactly
this kind of occluder, so every promoted detection is cross-referenced
against the REAL ground-truth class of the point that produced it, to
separate a genuine terrain false positive (the thing this validation is
actually checking for) from an occlusion the carving step would have
filtered, had it run first (a known, already-documented limitation --
not a new bug this module discovers).

Ring assignment uses perception.ring_recovery's exact position-based
decode (Ticket #6's own finding); azimuth binning uses the SAME formula
observability.ground_plane / perception.ground_prior already use
internally, so a queried (ring, col) here means the identical physical
direction in the measured grid and in the ground-plane fit's own
binning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from grid.addressing import world_to_global
from observability.ground_plane import expected_ground_range, fit_local_ground_planes
from observability.negative_obstacle import PersistenceTracker, detect_anomalous_cells
from perception.ring_recovery import recover_ring_from_scan_order
from perception.rellis_loader import load_rellis_labels, load_rellis_sweep
from perception.taxonomy import DrishtiClass, rellis_label_ids_to_drishti
from sensor.sensor_model import SensorConfig, load_sensor_config

REPO_ROOT = Path(__file__).resolve().parents[1]
N_BEAMS = 64
WORLD_CELL_SIZE_M = 0.5
N_PERSISTENCE_FRAMES = 3


def build_measured_range_grid(sweep, n_beams: int, n_azimuth_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Real frame -> (measured_range, point_index), both (n_beams,
    n_azimuth_bins). measured_range is NaN where no return; point_index
    is -1 there, else the index into `sweep.xyz` of the point that
    produced that grid cell -- used afterwards to look up that point's
    real ground-truth class for a flagged cell."""
    xyz = sweep.xyz.astype(np.float64)
    r = np.linalg.norm(xyz, axis=1)
    valid = r > 1e-6

    ring = recover_ring_from_scan_order(xyz.shape[0], n_beams=n_beams)
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    col = np.floor(0.5 * (1.0 - azimuth / np.pi) * n_azimuth_bins).astype(np.int64)
    col = np.clip(col, 0, n_azimuth_bins - 1)

    range_grid = np.full((n_beams, n_azimuth_bins), np.nan)
    point_index_grid = np.full((n_beams, n_azimuth_bins), -1, dtype=np.int64)

    valid_idx = np.nonzero(valid)[0]
    if valid_idx.size == 0:
        return range_grid, point_index_grid

    pix = ring[valid_idx] * n_azimuth_bins + col[valid_idx]
    r_valid = r[valid_idx]
    # Keep the NEAREST return per (ring, col) on a collision -- same
    # convention perception/range_image.py's own many-to-one projection
    # uses.
    order = np.lexsort((r_valid, pix))
    sorted_pix = pix[order]
    sorted_r = r_valid[order]
    sorted_src = valid_idx[order]
    is_first = np.ones(order.shape[0], dtype=bool)
    is_first[1:] = sorted_pix[1:] != sorted_pix[:-1]

    range_grid.flat[sorted_pix[is_first]] = sorted_r[is_first]
    point_index_grid.flat[sorted_pix[is_first]] = sorted_src[is_first]
    return range_grid, point_index_grid


def _azimuth_of_column(col: np.ndarray, n_azimuth_bins: int) -> np.ndarray:
    return np.pi * (1.0 - 2.0 * (col.astype(np.float64) + 0.5) / n_azimuth_bins)


def flagged_cells_to_world(
    flagged: Set[Tuple[int, int]], expected_range: np.ndarray, n_azimuth_bins: int, T_world: np.ndarray
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """(ring, col) flagged cells -> WORLD grid cells, via the fit's own
    predicted range and the REAL recorded ego pose. Unlike
    eval/checkpoint_trench.py's synthetic case (ego fixed at the world
    origin, so sensor frame IS world frame), a real multi-frame sequence
    needs the actual T_world to key persistence by genuine world
    position -- otherwise a stationary hazard would appear to drift
    frame to frame as the vehicle itself moves (Ticket #36's own design
    note: PersistenceTracker is keyed by world cell for exactly this
    reason). Uses expected_range directly as horizontal range, same
    simplification eval/checkpoint_trench.py's own _flagged_cell_to_world
    already makes (off by cos(elevation) at these shallow beam angles --
    acceptable for a tracking key, not used as a hazard-geometry number)."""
    mapping = {}
    for ring, col in flagged:
        r_exp = expected_range[ring, col]
        az = float(_azimuth_of_column(np.array([col]), n_azimuth_bins)[0])
        x_sensor = r_exp * np.cos(az)
        y_sensor = r_exp * np.sin(az)
        world = T_world @ np.array([x_sensor, y_sensor, 0.0, 1.0])
        mapping[(ring, col)] = world_to_global(world[0], world[1], WORLD_CELL_SIZE_M)
    return mapping


def _detect_anomalous_measured_only(
    measured_range: np.ndarray, expected_range: np.ndarray, valid_expected: np.ndarray, anomaly_margin_m: float = 1.0
) -> Set[Tuple[int, int]]:
    """DIAGNOSTIC ABLATION, not a replacement for
    observability.negative_obstacle.detect_anomalous_cells -- that
    function's "no return -> treat as measured at usable_range_m" branch
    is the CORRECT documented behaviour for its actual use case (Ticket
    #36's own spec), which assumes carving (#33/34) already ran and
    marked genuinely-occluded cells before this ever sees them. This
    variant instead SKIPS every no-return cell from scoring entirely,
    to isolate how the ring-to-ring RELATIVE discriminator alone behaves
    on real, genuinely-measured returns -- separating "the discriminator
    itself is noisy on rough real terrain" from "no-return substitution
    without carve-first floods the detector with occlusion look-alikes",
    which turned out to be the dominant effect (see module docstring)."""
    H, W = measured_range.shape
    residual = measured_range - expected_range
    flagged: Set[Tuple[int, int]] = set()
    for col in range(W):
        for ring in range(H):
            if not valid_expected[ring, col] or np.isnan(measured_range[ring, col]):
                continue
            if residual[ring, col] <= 0:
                continue
            neighbours = [
                residual[nb, col]
                for nb in (ring - 1, ring + 1)
                if 0 <= nb < H and valid_expected[nb, col] and not np.isnan(measured_range[nb, col])
            ]
            if not neighbours:
                continue
            if (residual[ring, col] - min(neighbours)) > anomaly_margin_m:
                flagged.add((ring, col))
    return flagged


def _promotions_for_mode(
    sequence_dir: Path,
    frame_indices: List[int],
    sm: SensorConfig,
    n_azimuth_bins: int,
    use_measured_only: bool,
) -> Tuple[List[int], Dict[Tuple[int, int], dict]]:
    tracker = PersistenceTracker(required_frames=N_PERSISTENCE_FRAMES)
    promoted_ever: Dict[Tuple[int, int], dict] = {}
    raw_flagged_counts = []

    for frame_idx in frame_indices:
        sweep = load_rellis_sweep(sequence_dir, frame_idx)
        measured, point_index = build_measured_range_grid(sweep, N_BEAMS, n_azimuth_bins)
        fit = fit_local_ground_planes(sweep, sm, n_azimuth_bins=n_azimuth_bins)

        expected = np.full((N_BEAMS, n_azimuth_bins), np.nan)
        valid_expected = np.zeros((N_BEAMS, n_azimuth_bins), dtype=bool)
        for col in fit.column_fits.keys():
            for ring in range(N_BEAMS):
                v = expected_ground_range(fit, ring, col)
                if v is not None:
                    expected[ring, col] = v
                    valid_expected[ring, col] = True

        if use_measured_only:
            flagged = _detect_anomalous_measured_only(measured, expected, valid_expected)
        else:
            occluded_mask = np.zeros((N_BEAMS, n_azimuth_bins), dtype=bool)  # scope note: see module docstring
            flagged = detect_anomalous_cells(measured, expected, valid_expected, occluded_mask, sm.usable_range_m or 60.0)
        raw_flagged_counts.append(len(flagged))

        world_map = flagged_cells_to_world(flagged, expected, n_azimuth_bins, sweep.T_world)
        promoted = tracker.update(set(world_map.values()))

        for (ring, col), world_cell in world_map.items():
            if world_cell in promoted and world_cell not in promoted_ever:
                gt_class = None
                src = int(point_index[ring, col])
                if src >= 0:
                    raw_label = load_rellis_labels(sequence_dir, frame_idx)[src]
                    gt_class = int(rellis_label_ids_to_drishti(np.array([raw_label]))[0])
                promoted_ever[world_cell] = {
                    "world_cell": world_cell,
                    "first_promoted_frame": frame_idx,
                    "ring": ring,
                    "col": col,
                    "ground_truth_class_at_source_point": gt_class,
                    "ground_truth_class_name": DrishtiClass(gt_class).name if gt_class is not None else None,
                }

    return raw_flagged_counts, promoted_ever


def run_validation(
    sequence_dir: str | Path,
    frame_indices: List[int],
    sensor_config_path: str | Path,
    n_azimuth_bins: int = 2048,
) -> dict:
    sequence_dir = Path(sequence_dir)
    sm = load_sensor_config(sensor_config_path)

    # Two modes, reported side by side rather than picking one number:
    # "full_spec" is Ticket #36's actual documented behaviour (no-return
    # substituted with usable_range_m) -- the honest answer to "what
    # happens if you run this on real data exactly as specified, without
    # first carving (#33/34)". "measured_only" is the diagnostic
    # ablation above, isolating the ring-to-ring discriminator's own
    # noise on genuinely-measured real returns.
    full_spec_counts, full_spec_promotions = _promotions_for_mode(
        sequence_dir, frame_indices, sm, n_azimuth_bins, use_measured_only=False
    )
    measured_only_counts, measured_only_promotions = _promotions_for_mode(
        sequence_dir, frame_indices, sm, n_azimuth_bins, use_measured_only=True
    )

    ground_fit_residuals = []
    for frame_idx in frame_indices:
        sweep = load_rellis_sweep(sequence_dir, frame_idx)
        fit = fit_local_ground_planes(sweep, sm, n_azimuth_bins=n_azimuth_bins)
        ground_fit_residuals.extend(cf.max_residual_m for cf in fit.column_fits.values())

    raw_flagged_counts = full_spec_counts
    promoted_ever = full_spec_promotions

    vegetation_like = {int(DrishtiClass.VEGETATION), int(DrishtiClass.STATIC_OBSTACLE)}
    promotions = list(promoted_ever.values())
    likely_occlusion = [p for p in promotions if p["ground_truth_class_at_source_point"] in vegetation_like]
    unexplained = [p for p in promotions if p["ground_truth_class_at_source_point"] not in vegetation_like]
    measured_only_promotion_list = list(measured_only_promotions.values())

    return {
        "sequence_dir": str(sequence_dir),
        "n_frames": len(frame_indices),
        "frame_indices": frame_indices,
        "full_spec": {
            "description": "Ticket #36 exactly as specified: no-return cells substituted with usable_range_m. "
                            "This is the honest answer to 'what happens on real off-road data without carving first' -- "
                            "see module docstring for why that's expected to (and does) fire a lot.",
            "raw_flagged_per_frame": raw_flagged_counts,
            "raw_flagged_mean": float(np.mean(raw_flagged_counts)) if raw_flagged_counts else 0.0,
            "n_promoted_total": len(promotions),
            "n_promoted_vegetation_or_obstacle_source": len(likely_occlusion),
            "n_promoted_unexplained": len(unexplained),
            "promotions": promotions,
        },
        "measured_only_diagnostic": {
            "description": "DIAGNOSTIC ABLATION (not Ticket #36's real behaviour): no-return cells are SKIPPED "
                            "entirely rather than substituted, isolating the ring-to-ring discriminator's own "
                            "noise on genuinely-measured real returns, as a proxy for 'what carving would likely "
                            "leave behind'.",
            "raw_flagged_per_frame": measured_only_counts,
            "raw_flagged_mean": float(np.mean(measured_only_counts)) if measured_only_counts else 0.0,
            "n_promoted_total": len(measured_only_promotion_list),
            "promotions": measured_only_promotion_list,
        },
        "ground_fit_residual_mean_m": float(np.mean(ground_fit_residuals)) if ground_fit_residuals else None,
        "ground_fit_residual_max_m": float(np.max(ground_fit_residuals)) if ground_fit_residuals else None,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4 real-data validation: negative-obstacle false positives")
    parser.add_argument("--sequence-dir", default=str(REPO_ROOT / "data" / "rellis" / "00004"))
    parser.add_argument("--sensor-config", default=str(REPO_ROOT / "configs" / "sensor_ouster_os1_64.yaml"))
    parser.add_argument("--start-frame", type=int, default=1000)
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--out", default=str(REPO_ROOT / "eval" / "out" / "negative_obstacle_real_data_validation.json"))
    args = parser.parse_args()

    frames = list(range(args.start_frame, args.start_frame + args.n_frames))
    result = run_validation(args.sequence_dir, frames, args.sensor_config)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    fs = result["full_spec"]
    mo = result["measured_only_diagnostic"]
    print(f"Sequence: {result['sequence_dir']}, frames {frames[0]}-{frames[-1]} ({result['n_frames']} frames)")
    print(f"Ground-plane fit residual: mean={result['ground_fit_residual_mean_m']:.3f}m, max={result['ground_fit_residual_max_m']:.3f}m "
          f"(the local line fit itself is behaving well)")
    print()
    print("FULL SPEC (Ticket #36 exactly as documented, no carve-first):")
    print(f"  raw flagged/frame: mean={fs['raw_flagged_mean']:.0f}  per-frame={fs['raw_flagged_per_frame']}")
    print(f"  promoted to NEGATIVE_OBSTACLE (>= {N_PERSISTENCE_FRAMES} consecutive frames): {fs['n_promoted_total']}")
    print(f"    of which source point was VEGETATION/STATIC_OBSTACLE (occlusion-shaped, not a genuine terrain false positive): "
          f"{fs['n_promoted_vegetation_or_obstacle_source']}")
    print(f"    unexplained by that (no traceable source point -- almost all are 'no return' cells, see below): "
          f"{fs['n_promoted_unexplained']}")
    print()
    print("MEASURED-ONLY DIAGNOSTIC (no-return cells skipped, isolating the discriminator's real-terrain noise):")
    print(f"  raw flagged/frame: mean={mo['raw_flagged_mean']:.0f}  per-frame={mo['raw_flagged_per_frame']}")
    print(f"  promoted to NEGATIVE_OBSTACLE: {mo['n_promoted_total']}")
    print()
    print(f"Full report: {out_path}")
