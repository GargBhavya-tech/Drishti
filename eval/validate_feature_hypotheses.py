"""
eval/validate_feature_hypotheses.py

Pre-training validation: checks, against REAL RELLIS-3D data, whether
each of four proposed input-feature additions actually carries the
class-discriminative or density-boosting signal its own justification
claims -- BEFORE spending GPU hours retraining on top of them. Per the
user's own explicit instruction: "if there is way to check without
training that our model will become better from it do that first."

Four checks, each answering one falsifiable question:

  A. RANGE-CORRECTED REFLECTIVITY: does range-corrected intensity
     separate DRIVABLE from VEGETATION (the "Ground Paradox" axis --
     BASELINE_COMPARISON.md's own zero-shot finding) better than raw
     intensity does? Also reports raw intensity's own basic statistics,
     since `perception/rellis_loader.py`'s own comment left whether
     RELLIS-3D's intensity field is trustworthy at all as an OPEN,
     never-verified question.
  B. SURFACE NORMALS/CURVATURE: do STATIC_OBSTACLE and NON_TRAVERSABLE
     pixels actually show lower normal_z (verticality) / higher
     curvature (roughness) than DRIVABLE/VEGETATION, as the physical
     hypothesis claims?
  C. CLASS 4 (STATIC_OBSTACLE) PREVALENCE: what fraction of real
     training pixels are class 4 -- quantifies whether its persistent
     0.0 IoU (see checkpoints_multi_v2's full 20-epoch training_log.jsonl)
     is a "too rare to learn from reweighting alone" problem, which is
     exactly what CutMix is for.
  D. MULTI-SWEEP DENSITY GAIN: how many more valid points fall beyond
     40m when 3 motion-compensated sweeps are merged vs a single sweep,
     on real sequences.

Every number below is computed directly from real downloaded RELLIS-3D
data -- no synthetic frames, no assumptions asserted without a
measurement backing them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from perception.multi_sweep import merge_sweeps_motion_compensated
from perception.range_image import project_to_range_image
from perception.reflectivity import range_corrected_intensity
from perception.rellis_loader import load_rellis_labels, load_rellis_sweep
from perception.surface_geometry import compute_surface_geometry
from perception.taxonomy import DrishtiClass, rellis_label_ids_to_drishti
from perception.train import build_multi_sequence_splits
from sensor.sensor_model import load_sensor_config

SEQUENCE_DIRS = [Path(f"data/rellis/{i:05d}") for i in range(5)]
DEFAULT_SENSOR_CONFIG = "configs/sensor_ouster_os1_64.yaml"
MAX_FRAMES = 60


def _drishti_labels_for_frame(sequence_dir: Path, frame_idx: int, img):
    raw_labels = load_rellis_labels(sequence_dir, frame_idx)
    drishti_labels = rellis_label_ids_to_drishti(raw_labels)
    target = np.zeros((img.H, img.W), dtype=np.int64)
    touched = img.point_index >= 0
    src = img.point_index[touched]
    target[touched] = drishti_labels[src]
    return target


def _fisher_separability(mean_a: float, mean_b: float, std_a: float, std_b: float) -> float:
    """(mean_a - mean_b)^2 / (std_a^2 + std_b^2) -- a standard, simple
    class-separability score: large when two classes' distributions are
    far apart relative to their own spread, ~0 when they overlap."""
    denom = std_a**2 + std_b**2
    return float((mean_a - mean_b) ** 2 / denom) if denom > 1e-9 else 0.0


def check_a_reflectivity(sm, frames: List[Tuple[Path, int]]) -> dict:
    raw_by_class: Dict[int, List[float]] = {}
    corrected_by_class: Dict[int, List[float]] = {}

    for seq_dir, frame_idx in frames:
        sweep = load_rellis_sweep(seq_dir, frame_idx)
        img = project_to_range_image(sweep, sm)
        target = _drishti_labels_for_frame(seq_dir, frame_idx, img)
        corrected = range_corrected_intensity(img.intensity, img.range)

        valid = img.valid_mask
        for c in (DrishtiClass.DRIVABLE, DrishtiClass.VEGETATION, DrishtiClass.CAUTION):
            mask = valid & (target == int(c))
            raw_by_class.setdefault(int(c), []).append(img.intensity[mask])
            corrected_by_class.setdefault(int(c), []).append(corrected[mask])

    def stats(by_class: Dict[int, List[float]], c: int) -> Tuple[float, float, int]:
        pooled = np.concatenate(by_class[c]) if by_class.get(c) else np.array([])
        if pooled.size == 0:
            return float("nan"), float("nan"), 0
        return float(pooled.mean()), float(pooled.std()), int(pooled.size)

    drivable_raw = stats(raw_by_class, int(DrishtiClass.DRIVABLE))
    vegetation_raw = stats(raw_by_class, int(DrishtiClass.VEGETATION))
    drivable_corr = stats(corrected_by_class, int(DrishtiClass.DRIVABLE))
    vegetation_corr = stats(corrected_by_class, int(DrishtiClass.VEGETATION))

    sep_raw = _fisher_separability(drivable_raw[0], vegetation_raw[0], drivable_raw[1], vegetation_raw[1])
    sep_corrected = _fisher_separability(drivable_corr[0], vegetation_corr[0], drivable_corr[1], vegetation_corr[1])

    return {
        "drivable_raw_mean_std_n": drivable_raw,
        "vegetation_raw_mean_std_n": vegetation_raw,
        "drivable_corrected_mean_std_n": drivable_corr,
        "vegetation_corrected_mean_std_n": vegetation_corr,
        "drivable_vs_vegetation_separability_raw": sep_raw,
        "drivable_vs_vegetation_separability_corrected": sep_corrected,
        "correction_improves_separability": sep_corrected > sep_raw,
    }


def check_b_surface_geometry(sm, frames: List[Tuple[Path, int]]) -> dict:
    normal_z_by_class: Dict[int, List[float]] = {}
    curvature_by_class: Dict[int, List[float]] = {}

    classes_of_interest = [
        DrishtiClass.DRIVABLE, DrishtiClass.VEGETATION,
        DrishtiClass.STATIC_OBSTACLE, DrishtiClass.NON_TRAVERSABLE,
    ]

    for seq_dir, frame_idx in frames:
        sweep = load_rellis_sweep(seq_dir, frame_idx)
        img = project_to_range_image(sweep, sm)
        target = _drishti_labels_for_frame(seq_dir, frame_idx, img)
        geom = compute_surface_geometry(img)

        for c in classes_of_interest:
            mask = geom.geometry_valid & (target == int(c))
            normal_z_by_class.setdefault(int(c), []).append(np.abs(geom.normal_z[mask]))
            curvature_by_class.setdefault(int(c), []).append(geom.curvature[mask])

    def stats(by_class: Dict[int, List[float]], c: int) -> Tuple[float, float, int]:
        pooled = np.concatenate(by_class[c]) if by_class.get(c) else np.array([])
        if pooled.size == 0:
            return float("nan"), float("nan"), 0
        return float(pooled.mean()), float(pooled.std()), int(pooled.size)

    result = {}
    for c in classes_of_interest:
        result[c.name] = {
            "abs_normal_z_mean_std_n": stats(normal_z_by_class, int(c)),
            "curvature_mean_std_n": stats(curvature_by_class, int(c)),
        }
    return result


def check_c_class4_prevalence(frames: List[Tuple[Path, int]], sm) -> dict:
    total_pixels = 0
    class_counts: Dict[int, int] = {}
    frames_with_class4 = 0

    for seq_dir, frame_idx in frames:
        sweep = load_rellis_sweep(seq_dir, frame_idx)
        img = project_to_range_image(sweep, sm)
        target = _drishti_labels_for_frame(seq_dir, frame_idx, img)
        valid = img.valid_mask
        total_pixels += int(valid.sum())
        has_class4 = False
        for c in np.unique(target[valid]):
            count = int(((target == c) & valid).sum())
            class_counts[int(c)] = class_counts.get(int(c), 0) + count
            if int(c) == int(DrishtiClass.STATIC_OBSTACLE) and count > 0:
                has_class4 = True
        if has_class4:
            frames_with_class4 += 1

    class4_count = class_counts.get(int(DrishtiClass.STATIC_OBSTACLE), 0)
    return {
        "total_valid_pixels_sampled": total_pixels,
        "class4_pixel_count": class4_count,
        "class4_fraction_of_total": class4_count / total_pixels if total_pixels else 0.0,
        "frames_with_any_class4_pixels": frames_with_class4,
        "frames_sampled": len(frames),
    }


def check_d_multi_sweep_density(sm, frames: List[Tuple[Path, int]], n_sweeps: int = 3) -> dict:
    results = []
    for seq_dir, frame_idx in frames:
        if frame_idx < n_sweeps - 1:
            continue
        sweeps = [load_rellis_sweep(seq_dir, frame_idx - k) for k in range(n_sweeps - 1, -1, -1)]
        single_img = project_to_range_image(sweeps[-1], sm)
        merged = merge_sweeps_motion_compensated(sweeps)
        merged_img = project_to_range_image(merged, sm)

        single_range = single_img.range[single_img.valid_mask]
        merged_range = merged_img.range[merged_img.valid_mask]

        results.append({
            "frame_idx": frame_idx,
            "single_sweep_valid_points": int(single_img.valid_mask.sum()),
            "merged_valid_points": int(merged_img.valid_mask.sum()),
            "single_sweep_points_beyond_40m": int((single_range > 40.0).sum()),
            "merged_points_beyond_40m": int((merged_range > 40.0).sum()),
        })
        if len(results) >= 5:
            break

    return {"per_frame": results}


def run_all_checks(
    sensor_config_path: str = DEFAULT_SENSOR_CONFIG,
    max_frames: int = MAX_FRAMES,
    out_path: str = "eval/out/feature_hypothesis_validation.json",
) -> dict:
    sm = load_sensor_config(sensor_config_path)
    train_items, _val_items, _counts = build_multi_sequence_splits(SEQUENCE_DIRS)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(train_items), size=min(max_frames, len(train_items)), replace=False)
    frames = [train_items[i] for i in sorted(idx)]
    print(f"Validating against {len(frames)} real, randomly-sampled TRAINING frames across all 5 sequences")

    result = {
        "n_frames_sampled": len(frames),
        "check_a_reflectivity": check_a_reflectivity(sm, frames),
        "check_b_surface_geometry": check_b_surface_geometry(sm, frames),
        "check_c_class4_prevalence": check_c_class4_prevalence(frames, sm),
        "check_d_multi_sweep_density": check_d_multi_sweep_density(sm, frames),
    }

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    out_path_p.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nFull results saved to {out_path_p}")
    return result


if __name__ == "__main__":
    result = run_all_checks()
    print(json.dumps(result, indent=2, default=str))
