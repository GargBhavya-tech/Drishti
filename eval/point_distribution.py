"""
eval/point_distribution.py

Ticket #6 — THE GATE. The single most important file in the build: it
measures whether the sensor constants in configs/sensor_*.yaml actually
describe the point cloud the sensor produces. Every downstream number in
the Bible, the Build Map, and the slides depends on this passing.

Two measurements, per 5m range bin out to 70m:

  - within-ring spacing: nearest-neighbour distance between points on the
    SAME ring, compared against s_tangential(r) = r * d_theta.
  - between-ring spacing: nearest-neighbour distance between adjacent
    rings on near-flat ground, compared against
    s_radial_ground(r) = r^2 * d_phi / h.

Watch out (restated from the Build Map so it's next to the code that has
to get it right): grouping must happen BY RING FIRST. Measuring nearest-
neighbour over the whole cloud finds the neighbour in the adjacent ring,
not the adjacent azimuth sample -- that produces a plausible-looking plot
of the wrong quantity. Also: the between-ring measurement is only valid
on near-flat ground; on a slope or near an object, s_radial_ground's
derivation doesn't hold.

Ground-flatness proxy note: Ticket #26 (the real ground-prior classifier)
is not built yet, and this ticket is blocked only by #2/#4, not #26. So
`_flat_ground_mask` below uses a crude z-band heuristic as a stand-in --
good enough to validate the estimator's logic against synthetic data now,
but SHOULD be swapped for the real ground prior once #26 lands, since a
z-band heuristic will include non-ground points on any real, non-trivial
scene and bias the between-ring measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from perception.sweep import Sweep
from sensor.sensor_model import SensorConfig, s_tangential, s_radial_ground

RANGE_BIN_EDGES_M = np.arange(0, 75, 5)  # 5m bins to 70m


@dataclass
class BinResult:
    range_lo: float
    range_hi: float
    range_mid: float
    within_ring_measured_m: Optional[float]
    within_ring_predicted_m: float
    between_ring_measured_m: Optional[float]
    between_ring_predicted_m: Optional[float]  # None if h_m unset
    n_points: int


def _range_of(xyz: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xyz[:, :2], axis=1)  # ground-plane range, matches Bible's r


def _flat_ground_mask(sweep: Sweep, z_band_m: float = 0.15) -> np.ndarray:
    """Crude stand-in for Ticket #26's real ground classifier -- see
    module docstring. Keeps points whose z sits within z_band_m of the
    5th percentile z (a cheap proxy for 'near the lowest surface, i.e.
    probably ground')."""
    if sweep.xyz.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    z = sweep.xyz[:, 2]
    floor = np.percentile(z, 5)
    return np.abs(z - floor) <= z_band_m


def measure_within_ring_spacing(sweep: Sweep) -> dict:
    """For each range bin, median nearest-neighbour distance between
    points on the same ring. Returns {bin_mid: median_distance_m}."""
    xyz = sweep.xyz
    ring = sweep.ring
    r = _range_of(xyz)

    out = {}
    for lo, hi in zip(RANGE_BIN_EDGES_M[:-1], RANGE_BIN_EDGES_M[1:]):
        mid = (lo + hi) / 2
        in_bin = (r >= lo) & (r < hi)
        dists = []
        for ring_id in np.unique(ring[in_bin]):
            if ring_id < 0:
                continue  # sentinel for "ring unavailable" (Sweep spec)
            sel = in_bin & (ring == ring_id)
            pts = xyz[sel]
            if pts.shape[0] < 2:
                continue
            az = np.arctan2(pts[:, 1], pts[:, 0])
            order = np.argsort(az)
            pts_sorted = pts[order]
            # nearest neighbour = adjacent in azimuth order, wrapping
            nxt = np.roll(pts_sorted, -1, axis=0)
            d = np.linalg.norm(pts_sorted - nxt, axis=1)
            dists.append(d)
        if dists:
            out[mid] = float(np.median(np.concatenate(dists)))
        else:
            out[mid] = None
    return out


def measure_between_ring_spacing(sweep: Sweep) -> dict:
    """For each range bin, median nearest-neighbour distance between
    adjacent rings, restricted to the flat-ground proxy mask.

    Deliberately does NOT require both rings' points to fall in the same
    range bin before pairing them: s_radial_ground(r) can exceed the 5m
    bin width in the near/mid field on a coarse sensor (e.g. it's ~5.06m
    at r=20m on the HDL-32E, i.e. already wider than one bin), so an
    early per-bin filter would silently drop exactly the pairs the ticket
    needs to measure. Instead, pair adjacent rings globally on the flat-
    ground-proxy subset, then bucket each measured distance by the range
    of its SOURCE point (the point in the lower ring of the pair)."""
    xyz = sweep.xyz
    ring = sweep.ring
    flat = _flat_ground_mask(sweep)

    valid_rings = sorted(set(int(x) for x in np.unique(ring[flat]) if x >= 0))
    per_source_range = []  # (range_of_source_point, distance)
    for a, b in zip(valid_rings[:-1], valid_rings[1:]):
        pa = xyz[flat & (ring == a)]
        pb = xyz[flat & (ring == b)]
        if pa.shape[0] == 0 or pb.shape[0] == 0:
            continue
        for p in pa:
            d = np.linalg.norm(pb[:, :2] - p[:2], axis=1)
            per_source_range.append((float(np.linalg.norm(p[:2])), float(d.min())))

    out = {}
    if not per_source_range:
        for lo, hi in zip(RANGE_BIN_EDGES_M[:-1], RANGE_BIN_EDGES_M[1:]):
            out[(lo + hi) / 2] = None
        return out

    source_r = np.array([x[0] for x in per_source_range])
    source_d = np.array([x[1] for x in per_source_range])

    for lo, hi in zip(RANGE_BIN_EDGES_M[:-1], RANGE_BIN_EDGES_M[1:]):
        mid = (lo + hi) / 2
        sel = (source_r >= lo) & (source_r < hi)
        out[mid] = float(np.median(source_d[sel])) if sel.any() else None
    return out


def validate_point_distribution(
    sweep: Sweep,
    sm: SensorConfig,
    out_path: Optional[str | Path] = None,
) -> list[BinResult]:
    within = measure_within_ring_spacing(sweep)
    between = measure_between_ring_spacing(sweep)
    r = _range_of(sweep.xyz)

    results = []
    for lo, hi in zip(RANGE_BIN_EDGES_M[:-1], RANGE_BIN_EDGES_M[1:]):
        mid = (lo + hi) / 2
        n_pts = int(((r >= lo) & (r < hi)).sum())
        predicted_within = s_tangential(mid, sm)
        predicted_between = s_radial_ground(mid, sm) if sm.h_m is not None else None
        results.append(
            BinResult(
                range_lo=lo,
                range_hi=hi,
                range_mid=mid,
                within_ring_measured_m=within.get(mid),
                within_ring_predicted_m=predicted_within,
                between_ring_measured_m=between.get(mid),
                between_ring_predicted_m=predicted_between,
                n_points=n_pts,
            )
        )

    if out_path is not None:
        _plot(results, sm, out_path)

    return results


def _plot(results: list[BinResult], sm: SensorConfig, out_path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    mids = [r.range_mid for r in results]
    pred_w = [r.within_ring_predicted_m for r in results]
    meas_w = [r.within_ring_measured_m for r in results]
    axes[0].plot(mids, pred_w, label="predicted s_tangential(r)")
    axes[0].scatter(mids, meas_w, label="measured", color="tab:red")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("range (m)")
    axes[0].set_ylabel("within-ring spacing (m)")
    axes[0].set_title(f"Within-ring, {sm.sensor_id}")
    axes[0].legend()

    pred_b = [r.between_ring_predicted_m for r in results]
    meas_b = [r.between_ring_measured_m for r in results]
    axes[1].plot(mids, pred_b, label="predicted s_radial_ground(r)")
    axes[1].scatter(mids, meas_b, label="measured", color="tab:red")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("range (m)")
    axes[1].set_ylabel("between-ring spacing (m)")
    axes[1].set_title(f"Between-ring (flat-ground proxy), {sm.sensor_id}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
