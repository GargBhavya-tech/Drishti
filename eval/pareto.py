"""
eval/pareto.py

Ticket #58 -- the Pareto curve via SELF-CONSISTENCY (Build Map's own
framing: this does NOT need ground truth, despite the Bible describing
it that way -- comparing a coarsened map against the FINEST map
buildable from the SAME data answers "what did coarsening cost me
relative to the best map I could have built", which runs on plain data
with no simulator, and is arguably the more relevant question anyway).

Sweep gamma from 0 (uniform, finest) upward:

    c(r, gamma) = c0 * (1 + r)^gamma, floored at c0

At gamma=0, c(r, 0) = c0 for EVERY range r -- a uniform map at the
finest resolution, matching the ticket's own words ("gamma=0 (uniform,
finest)") exactly. This is deliberately a pure RANGE-based power law,
not Ticket #47's velocity/TTC-dependent c_ttc: the Build Map's own #58
text never mentions ego velocity or TTC, only Bible Part 13 happens to
reuse the symbol "gamma" for an unrelated purpose (the fovea's own
foveation exponent). Using the simpler range-only law is the more
direct reading of "sweep gamma from 0 (uniform, finest) upward" and is
what this module's own tests below are written against.

For each gamma this module records:
- total IMPLIED memory, at the same bytes-per-cell as every other
  baseline (`eval.baselines.BYTES_PER_CELL`, Ticket #56).
- elevation DEVIATION per distance band (`eval.metrics`'s own band
  boundaries, Ticket #57) between this gamma's coarsened aggregate
  height and a uniform-c0 map's own value at the SAME location, from
  the SAME underlying height field -- never an independent ground
  truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from eval.baselines import BYTES_PER_CELL
from eval.metrics import band_index_for_range

DEFAULT_C0_M = 0.05
DEFAULT_EXTENT_M = 102.4  # matches this project's own coarsest band radius (Ticket #57)
DEFAULT_BOUNDARIES_M = [12.8, 25.6, 51.2, 102.4]

HeightField = Callable[[np.ndarray, np.ndarray], np.ndarray]


def cell_size_for_gamma(r_m: np.ndarray, gamma: float, c0: float = DEFAULT_C0_M) -> np.ndarray:
    """c(r, gamma) = c0 * (1+r)^gamma, floored at c0 -- the schedule's
    own absolute finest quantum (Ticket #5), never a physically
    meaningless sub-c0 cell. gamma=0 collapses this to c0 EVERYWHERE,
    unconditionally -- "uniform, finest", per the Build Map's own
    words."""
    raw = c0 * np.power(1.0 + np.asarray(r_m, dtype=np.float64), gamma)
    return np.maximum(raw, c0)


def default_height_field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A smooth synthetic terrain with real structure at multiple
    scales (a gentle grade, broad rolling terrain, and a finer ripple)
    -- deliberately deterministic (no RNG) so the same sweep is exactly
    reproducible run to run. A perfectly flat field would make every
    gamma's deviation exactly 0, trivially "passing" the monotonicity
    test below without actually exercising it."""
    return 0.001 * x + 0.3 * np.sin(x / 8.0) * np.cos(y / 8.0) + 0.05 * np.sin(x / 2.0)


@dataclass(frozen=True)
class ParetoPoint:
    gamma: float
    total_memory_bytes: float
    deviation_rms_m: float
    deviation_by_band_m: Dict[int, float]


def _sample_grid(extent_m: float, n_per_axis: int) -> Tuple[np.ndarray, np.ndarray, float]:
    coords = np.linspace(-extent_m, extent_m, n_per_axis)
    xs, ys = np.meshgrid(coords, coords)
    xs, ys = xs.ravel(), ys.ravel()
    sample_area = (2 * extent_m) ** 2 / (n_per_axis**2)
    return xs, ys, sample_area


def _windowed_mean(height_fn: HeightField, cx: np.ndarray, cy: np.ndarray, window_m: np.ndarray, n_sub: int = 5) -> np.ndarray:
    """Approximate the mean of `height_fn` over an axis-aligned
    `window_m` x `window_m` square centred at each (cx, cy), via an
    n_sub x n_sub sub-sample grid per point -- vectorised across every
    point per sub-offset (n_sub^2 whole-array evaluations, not a Python
    loop per point)."""
    offsets = np.linspace(-0.5, 0.5, n_sub)
    total = np.zeros_like(cx, dtype=np.float64)
    for ox in offsets:
        for oy in offsets:
            total = total + height_fn(cx + ox * window_m, cy + oy * window_m)
    return total / (n_sub * n_sub)


def sweep_gamma(
    gammas: List[float],
    height_fn: HeightField = default_height_field,
    extent_m: float = DEFAULT_EXTENT_M,
    c0: float = DEFAULT_C0_M,
    boundaries_m: Optional[List[float]] = None,
    n_per_axis: int = 41,
    bytes_per_cell: int = BYTES_PER_CELL,
) -> List[ParetoPoint]:
    """The sweep itself: for each gamma, one `ParetoPoint` recording
    implied total memory and RMS elevation deviation (overall and per
    distance band), all measured against the SAME sample grid and the
    SAME height field -- self-consistency, per the module docstring.
    """
    boundaries = boundaries_m or DEFAULT_BOUNDARIES_M
    xs, ys, sample_area = _sample_grid(extent_m, n_per_axis)
    r = np.hypot(xs, ys)
    bands = np.array([band_index_for_range(float(ri), boundaries) for ri in r])

    fine_h = _windowed_mean(height_fn, xs, ys, np.full_like(xs, c0))

    points: List[ParetoPoint] = []
    for gamma in gammas:
        c = cell_size_for_gamma(r, gamma, c0=c0)
        coarse_h = _windowed_mean(height_fn, xs, ys, c)

        total_cells = float(np.sum(sample_area / (c**2)))
        total_memory_bytes = total_cells * bytes_per_cell

        sq_err = (coarse_h - fine_h) ** 2
        deviation_rms_m = float(np.sqrt(np.mean(sq_err)))
        deviation_by_band = {
            band: (float(np.sqrt(np.mean(sq_err[bands == band]))) if np.any(bands == band) else float("nan"))
            for band in range(len(boundaries))
        }
        points.append(
            ParetoPoint(
                gamma=gamma,
                total_memory_bytes=total_memory_bytes,
                deviation_rms_m=deviation_rms_m,
                deviation_by_band_m=deviation_by_band,
            )
        )

    return points


def find_knee_index(points: List[ParetoPoint]) -> int:
    """The elbow of the memory-vs-deviation curve: the point of maximum
    perpendicular distance from the straight line connecting the FIRST
    and LAST points, after normalising both axes to [0, 1] (so
    memory's and deviation's very different units/scales don't bias
    the distance calculation) -- the core idea behind the "Kneedle"
    heuristic, without its extra smoothing machinery, which this small,
    monotonic sweep does not need.
    """
    mem = np.array([p.total_memory_bytes for p in points], dtype=np.float64)
    dev = np.array([p.deviation_rms_m for p in points], dtype=np.float64)

    def normalise(v: np.ndarray) -> np.ndarray:
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)

    mem_n, dev_n = normalise(mem), normalise(dev)
    p0 = np.array([mem_n[0], dev_n[0]])
    p1 = np.array([mem_n[-1], dev_n[-1]])
    line_vec = p1 - p0
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return 0
    line_unit = line_vec / line_len

    distances = []
    for i in range(len(points)):
        pi = np.array([mem_n[i], dev_n[i]]) - p0
        proj_len = np.dot(pi, line_unit)
        perp = pi - proj_len * line_unit
        distances.append(float(np.linalg.norm(perp)))

    return int(np.argmax(distances))
