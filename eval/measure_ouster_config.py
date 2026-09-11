"""
eval/measure_ouster_config.py

Ticket #6, completed for the Ouster OS1-64 / RELLIS-3D half of the gate
(the nuScenes/HDL-32E half needs nuScenes-mini data, not available in
this environment -- see configs/sensor_hdl32e.yaml's own remaining TODO
on h_m).

`configs/sensor_ouster_os1_64.yaml` shipped with PLACEHOLDER
`d_theta_rad`/`d_phi_rad`/`phi_max_deg` and an unset `h_m`, explicitly
marked "do not trust any number computed from this config ... until
Ticket #6 ... has been re-run against real RELLIS-3D scans." This script
IS that re-run, now that real RELLIS-3D data exists locally
(`data/rellis/00000`..`00004`) and `perception.ring_recovery` supplies
the ring field the gate needs (which RELLIS's raw `.bin` dumps don't
ship -- see that module's docstring for the finding that makes this
possible: array position modulo 64 is an EXACT ring index, confirmed
against real data to ~1e-6 degree agreement across sequences).

What this measures, and how each number is produced:

- `d_theta_rad`: 2*pi / W, where W = n_points_per_frame / 64. Every real
  frame checked (all 5 sequences) has exactly 131,072 = 2048*64 points,
  so W = 2048, not the 1024 the placeholder assumed -- a firmware
  setting, confirmed by direct point-count division, not statistically
  estimated.
- `phi_max_rad`, `d_phi_rad`: from `aggregate_beam_elevation_table`
  across many real frames spanning all 5 sequences (fills in the top
  beams a single frame under-samples). `d_phi_rad` is the MEAN adjacent-
  beam spacing -- Ouster's real spacing is non-uniform (confirmed:
  observed gaps range ~0.43-0.62 degrees, a real physical property of
  this sensor, not a measurement artifact), so a single scalar is an
  approximation, same convention this project's other sensor configs
  already use for a single scalar d_phi.
- `h_m`: median height of `perception.ground_prior`-labelled ground
  points (Ticket #26's real classifier, not the old crude proxy) across
  many real frames spanning all 5 sequences. Reported with its spread
  across sequences, since RELLIS-3D's off-road terrain is genuinely
  undulating -- this is an honest single-scalar summary of real
  variation, not a precise fixed quantity the way nuScenes' flat urban
  ground would give.

Run: `python -m eval.measure_ouster_config`
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from eval.point_distribution import validate_point_distribution
from perception.ground_prior import compute_ground_prior
from perception.rellis_loader import load_rellis_sweep
from perception.ring_recovery import aggregate_beam_elevation_table, recover_ring_from_scan_order
from perception.sweep import Sweep
from sensor.sensor_model import SensorConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
RELLIS_ROOT = REPO_ROOT / "data" / "rellis"
SEQUENCES = ["00000", "00001", "00002", "00003", "00004"]
N_BEAMS = 64


def _sampled_frames(n_per_sequence: int = 30):
    """(sequence, frame_idx) pairs, strided evenly across each available
    sequence's actual frame count -- not the first N frames, which would
    only sample one stretch of one route."""
    pairs = []
    for seq in SEQUENCES:
        bin_dir = RELLIS_ROOT / seq / "os1_cloud_node_kitti_bin"
        if not bin_dir.exists():
            continue
        n_frames = len(list(bin_dir.glob("*.bin")))
        if n_frames == 0:
            continue
        stride = max(1, n_frames // n_per_sequence)
        for f in range(0, n_frames, stride):
            pairs.append((seq, f))
    return pairs


def measure_d_theta_rad(sample_pairs) -> float:
    seq, f = sample_pairs[0]
    sweep = load_rellis_sweep(RELLIS_ROOT / seq, frame_idx=f)
    n = sweep.xyz.shape[0]
    if n % N_BEAMS != 0:
        raise ValueError(f"{seq}/{f}: n_points={n} not a multiple of {N_BEAMS}")
    W = n // N_BEAMS
    for seq2, f2 in sample_pairs[1:]:
        sweep2 = load_rellis_sweep(RELLIS_ROOT / seq2, frame_idx=f2)
        n2 = sweep2.xyz.shape[0]
        if n2 // N_BEAMS != W:
            raise ValueError(
                f"inconsistent azimuth resolution: {seq}/{f} implies W={W}, "
                f"but {seq2}/{f2} implies W={n2 // N_BEAMS} -- do not trust a "
                f"single d_theta_rad if this fires."
            )
    return 2 * np.pi / W, W


def measure_beam_table(sample_pairs) -> np.ndarray:
    def frames():
        for seq, f in sample_pairs:
            yield load_rellis_sweep(RELLIS_ROOT / seq, frame_idx=f).xyz

    return aggregate_beam_elevation_table(frames(), n_beams=N_BEAMS, min_valid_per_beam=5)


def measure_h_m(sample_pairs) -> dict:
    all_ground_z = []
    per_seq_median = {}
    by_seq: dict = {}
    for seq, f in sample_pairs:
        by_seq.setdefault(seq, []).append(f)

    for seq, frames in by_seq.items():
        seq_ground_z = []
        for f in frames:
            sweep = load_rellis_sweep(RELLIS_ROOT / seq, frame_idx=f)
            r = np.linalg.norm(sweep.xyz, axis=1)
            valid = r > 1e-6
            valid_sweep = Sweep(
                xyz=sweep.xyz[valid], intensity=sweep.intensity[valid],
                ring=sweep.ring[valid], timestamp=sweep.timestamp,
                T_world=sweep.T_world, sensor_id=sweep.sensor_id,
            )
            ground = compute_ground_prior(valid_sweep, n_azimuth_bins=720)
            ground_z = valid_sweep.xyz[ground.is_ground, 2].astype(np.float64)
            seq_ground_z.append(ground_z)
        seq_ground_z = np.concatenate(seq_ground_z)
        per_seq_median[seq] = float(-np.median(seq_ground_z))
        all_ground_z.append(seq_ground_z)

    all_ground_z = np.concatenate(all_ground_z)
    return {
        "h_m_median": float(-np.median(all_ground_z)),
        "h_m_mean": float(-all_ground_z.mean()),
        "per_sequence_median": per_seq_median,
        "n_ground_points": int(all_ground_z.size),
    }


def run(n_per_sequence: int = 30, gate_frame: tuple = ("00004", 1000)) -> dict:
    sample_pairs = _sampled_frames(n_per_sequence=n_per_sequence)
    if not sample_pairs:
        raise RuntimeError(f"no RELLIS-3D sequences found under {RELLIS_ROOT}")

    d_theta_rad, W = measure_d_theta_rad(sample_pairs)
    beam_table = measure_beam_table(sample_pairs)
    present = np.nonzero(~np.isnan(beam_table))[0]
    phi_max_rad = float(beam_table[present[0]])
    phi_min_rad = float(beam_table[present[-1]])
    d_phi_rad = (phi_max_rad - phi_min_rad) / (len(present) - 1)
    gaps_deg = np.degrees(-np.diff(beam_table[present]))

    h_result = measure_h_m(sample_pairs)

    measured_sm = SensorConfig(
        sensor_id="ouster_os1_64",
        n_beams=N_BEAMS,
        d_theta_rad=d_theta_rad,
        d_phi_rad=d_phi_rad,
        phi_max_rad=phi_max_rad,
        h_m=h_result["h_m_median"],
        usable_range_m=None,
    )

    gate_seq, gate_idx = gate_frame
    gate_sweep = load_rellis_sweep(RELLIS_ROOT / gate_seq, frame_idx=gate_idx)
    r = np.linalg.norm(gate_sweep.xyz, axis=1)
    valid = r > 1e-6
    ring_full = recover_ring_from_scan_order(gate_sweep.xyz.shape[0], n_beams=N_BEAMS)
    gate_sweep_valid = Sweep(
        xyz=gate_sweep.xyz[valid], intensity=gate_sweep.intensity[valid],
        ring=ring_full[valid], timestamp=gate_sweep.timestamp,
        T_world=gate_sweep.T_world, sensor_id=gate_sweep.sensor_id,
    )
    out_path = REPO_ROOT / "eval" / "out" / "sensor_validation_ouster.png"
    bin_results = validate_point_distribution(gate_sweep_valid, measured_sm, out_path=out_path)

    return {
        "d_theta_rad": d_theta_rad,
        "d_theta_deg": np.degrees(d_theta_rad),
        "W_columns_per_rev": W,
        "phi_max_rad": phi_max_rad,
        "phi_max_deg": np.degrees(phi_max_rad),
        "phi_min_deg": np.degrees(phi_min_rad),
        "d_phi_rad": d_phi_rad,
        "d_phi_deg": np.degrees(d_phi_rad),
        "d_phi_gap_min_deg": float(gaps_deg.min()),
        "d_phi_gap_max_deg": float(gaps_deg.max()),
        "n_beams_with_data": int(len(present)),
        **h_result,
        "gate_plot": out_path,
        "gate_bin_results": bin_results,
        "n_frames_sampled": len(sample_pairs),
    }


if __name__ == "__main__":
    result = run()
    print(f"Sampled {result['n_frames_sampled']} real frames across all available sequences.\n")
    print(f"d_theta: {result['d_theta_deg']:.6f} deg  ({result['W_columns_per_rev']} columns/rev)")
    print(f"phi_max: {result['phi_max_deg']:.4f} deg   phi_min: {result['phi_min_deg']:.4f} deg")
    print(f"d_phi (mean spacing): {result['d_phi_deg']:.4f} deg  "
          f"(observed gap range {result['d_phi_gap_min_deg']:.4f}-{result['d_phi_gap_max_deg']:.4f} deg -- non-uniform, as expected)")
    print(f"beams with sufficient real data: {result['n_beams_with_data']}/64")
    print(f"h_m (median over {result['n_ground_points']} ground points): {result['h_m_median']:.4f} m "
          f"(mean: {result['h_m_mean']:.4f} m)")
    print("per-sequence h_m median:")
    for seq, h in result["per_sequence_median"].items():
        print(f"  {seq}: {h:.4f} m")
    print(f"\nGate plot written to: {result['gate_plot']}")
    print("\nper-bin gate results (range_mid, within-ring measured vs predicted, between-ring measured vs predicted):")
    for b in result["gate_bin_results"]:
        wm = f"{b.within_ring_measured_m:.4f}" if b.within_ring_measured_m is not None else "n/a"
        bm = f"{b.between_ring_measured_m:.4f}" if b.between_ring_measured_m is not None else "n/a"
        bp = f"{b.between_ring_predicted_m:.4f}" if b.between_ring_predicted_m is not None else "n/a"
        print(f"  {b.range_mid:5.1f}m  n={b.n_points:6d}  within: meas={wm:>8} pred={b.within_ring_predicted_m:.4f}  "
              f"between: meas={bm:>8} pred={bp:>8}")
