"""
eval/validate_conservatism_real_degradation.py

The REAL, end-to-end version of the Conservatism Invariant test that
tests/test_conservatism.py does NOT provide. That existing test proves
`planning.conservatism.cost()` is monotone over an ABSTRACT
`CellState` flag lattice (10,000 Hypothesis-generated cases) -- a real,
valuable proof that the COST FUNCTION ITSELF cannot be tricked. It does
NOT prove that DEGRADING A REAL SENSOR INPUT actually produces the
correctly-degraded CellState the cost function is fed. This script
closes exactly that gap: inject realistic degradation into REAL RELLIS-3D
points, recompute the REAL sparsity classification
(observability.sparsity.classify_sparsity) before and after, and assert
the resulting cost() never decreases.

Two real degradation modes, per the specific request this was built for:

1. Sector dropout (hardware/occlusion failure): zero out every point in
   a chosen azimuth range entirely -- simulating a stuck mirror, a dead
   sensor sector, or a large occluding object.
2. Distance-proportional attenuation (dust/rain, Beer-Lambert-style):
   each point survives with probability exp(-2*alpha*r) -- attenuation
   growing with range, same functional form the report this was built
   in response to specifically named. `alpha` is a free parameter (not
   independently derived from a real dust/rain physical measurement in
   this build -- flagged as a real, stated simplification, same
   honesty standard as this project's other physical approximations).

Scope, stated honestly: this operates on azimuth SECTORS of the raw
point cloud, computing observability/sparsity directly from real points
-- it does NOT run the full clipmap + raycast pipeline (grid.clipmap +
observability.raycast), which would require world-frame ego tracking
and multi-frame accumulation this single-frame test doesn't set up.
`observability` here is derived directly from the REAL sparsity verdict
semantics (FREE verdict -> OBS_FREE, a real return present -> OBS_OCCUPIED,
UNKNOWN verdict -> OBS_UNOBSERVED) rather than from real raycasting --
this isolates and proves the SPARSITY-driven half of the invariant
against real data; the RAYCAST/OCCLUDED half (Part 10) is a genuinely
separate, larger test this script does not attempt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from grid.cell import OBS_FREE, OBS_OCCUPIED, OBS_UNOBSERVED
from observability.sparsity import SparsityVerdict, classify_sparsity
from planning.conservatism import CellState, cost, merge_with_prior
from perception.rellis_loader import load_rellis_labels, load_rellis_sweep
from perception.taxonomy import rellis_label_ids_to_drishti
from perception.train import build_multi_sequence_splits
from sensor.sensor_model import load_sensor_config
from sensor.vehicle_config import load_vehicle_config

N_AZIMUTH_SECTORS = 72  # 5-degree sectors


def _sector_index(xyz: np.ndarray, n_sectors: int) -> np.ndarray:
    theta = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.floor((theta + np.pi) / (2 * np.pi) * n_sectors).astype(np.int64) % n_sectors


def _build_cell_state(sector_xyz: np.ndarray, sector_class: np.ndarray, vehicle, sm) -> CellState:
    n_obs = sector_xyz.shape[0]
    if n_obs == 0:
        r = 30.0  # representative range for an empty sector -- real system would use the LAST known range; using a fixed mid-range value here is a stated simplification for this standalone test
    else:
        r = float(np.median(np.linalg.norm(sector_xyz[:, :2], axis=1)))
        # A real, observed edge case: a handful of near-origin returns
        # (self-occlusion / very-close vehicle-body returns) can put the
        # median at ~0m, which sensor_model.n_expected() divides by r^2
        # -- a genuine physical floor (the sensor's own minimum range),
        # not an arbitrary guard.
        r = max(r, 0.5)

    result = classify_sparsity(n_obs=n_obs, r=r, vehicle=vehicle, sm=sm)

    if result.verdict == SparsityVerdict.FREE:
        observability = OBS_FREE
    elif n_obs > 0:
        observability = OBS_OCCUPIED
    else:
        observability = OBS_UNOBSERVED

    class_id = int(np.bincount(sector_class).argmax()) if n_obs > 0 else None

    return CellState(
        observability=observability,
        class_id=class_id,
        class_confidence=1.0,  # real ground-truth labels used for class -- isolates the sparsity/observability half of the invariant from segmentation-quality confounds
        sparsity_verdict=result.verdict,
        count=n_obs,
    )


def apply_sector_dropout(xyz: np.ndarray, sector_idx: np.ndarray, target_sectors: set) -> np.ndarray:
    keep = ~np.isin(sector_idx, list(target_sectors))
    return keep


def apply_beer_lambert_attenuation(xyz: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    r = np.linalg.norm(xyz[:, :2], axis=1)
    survival_prob = np.exp(-2.0 * alpha * r)
    return rng.random(xyz.shape[0]) < survival_prob


def main():
    parser = argparse.ArgumentParser(description="Real end-to-end Conservatism Invariant test under real sensor degradation")
    parser.add_argument("--sequence-dir", nargs="+", default=[f"data/rellis/{i:05d}" for i in range(5)])
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--vehicle-config", default="configs/vehicle_ugv.yaml")
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05, help="Beer-Lambert extinction coefficient (1/m) -- stated simplification, not independently measured")
    parser.add_argument("--dropout-sector-width", type=int, default=3, help="how many of the 72 5-degree sectors to zero out per frame")
    parser.add_argument("--use-merge", action="store_true", help="apply planning.conservatism.merge_with_prior before cost() -- the fix for the gap this script's own first run found (Part G.14)")
    args = parser.parse_args()

    dirs = [Path(p) for p in args.sequence_dir]
    train_items, _val_items, _ = build_multi_sequence_splits(dirs)
    step = max(1, len(train_items) // args.n_frames)
    sample_items = train_items[::step][: args.n_frames]

    sm = load_sensor_config(args.sensor_config)
    vehicle = load_vehicle_config(args.vehicle_config)
    rng = np.random.default_rng(0)

    total_cells_checked = 0
    total_violations = 0
    violation_examples = []

    for scenario_name, degrade_fn in [
        ("sector_dropout", None),  # handled specially below (needs sector_idx)
        ("beer_lambert_attenuation", lambda xyz: apply_beer_lambert_attenuation(xyz, args.alpha, rng)),
    ]:
        for sequence_dir, frame_idx in sample_items:
            sweep = load_rellis_sweep(sequence_dir, frame_idx)
            raw_labels = load_rellis_labels(sequence_dir, frame_idx)
            drishti_labels = rellis_label_ids_to_drishti(raw_labels)
            xyz = sweep.xyz.astype(np.float64)

            sector_idx = _sector_index(xyz, N_AZIMUTH_SECTORS)

            if scenario_name == "sector_dropout":
                target_start = rng.integers(0, N_AZIMUTH_SECTORS)
                target_sectors = {(target_start + k) % N_AZIMUTH_SECTORS for k in range(args.dropout_sector_width)}
                keep_mask = apply_sector_dropout(xyz, sector_idx, target_sectors)
                affected_sectors = target_sectors
            else:
                keep_mask = degrade_fn(xyz)
                affected_sectors = set(range(N_AZIMUTH_SECTORS))  # attenuation can affect any sector

            xyz_degraded = xyz[keep_mask]
            class_degraded = drishti_labels[keep_mask]
            sector_idx_degraded = sector_idx[keep_mask]

            for s in affected_sectors:
                before_mask = sector_idx == s
                after_mask = sector_idx_degraded == s

                before_state = _build_cell_state(xyz[before_mask], drishti_labels[before_mask], vehicle, sm)
                fresh_after_state = _build_cell_state(xyz_degraded[after_mask], class_degraded[after_mask], vehicle, sm)

                cost_before = cost(before_state, vehicle)

                if args.use_merge:
                    # THE FIX: merge the fresh reclassification with the
                    # prior state before ever calling cost() on it --
                    # closes the exact gap this script's own earlier run
                    # found (DRISHTI_MASTER_BIBLE.md Part G.14).
                    after_state = merge_with_prior(before_state, fresh_after_state, dt_s=0.1, vehicle=vehicle)
                else:
                    after_state = fresh_after_state
                cost_after = cost(after_state, vehicle)

                total_cells_checked += 1
                if cost_after < cost_before - 1e-9:
                    total_violations += 1
                    if len(violation_examples) < 10:
                        violation_examples.append(
                            {
                                "scenario": scenario_name,
                                "sequence_dir": str(sequence_dir),
                                "frame_idx": frame_idx,
                                "sector": s,
                                "cost_before": cost_before,
                                "cost_after": cost_after,
                                "state_before": before_state,
                                "state_after": after_state,
                            }
                        )

    print(f"Checked {total_cells_checked} real (frame, sector) cells across sector-dropout and Beer-Lambert-attenuation scenarios")
    print(f"Conservatism violations (cost DECREASED under degradation): {total_violations}")
    if total_violations > 0:
        print("\nFirst violations:")
        for v in violation_examples:
            print(f"  {v}")
    else:
        print("\nNO VIOLATIONS: the real end-to-end pipeline (real points -> real sparsity classification -> real cost()) "
              "never let degrading a real sensor input LOWER the reported cost, across both real degradation scenarios.")


if __name__ == "__main__":
    main()
