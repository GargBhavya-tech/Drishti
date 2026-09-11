"""
eval/inject_hazards.py

Ticket #50 -- synthetic hazard injection into a (ring, azimuth) range
grid, parameterised so the SAME tool drives both a demo trench/kerb/
pole scene and the detection-vs-range sweep in Ticket #60.

Data-availability note (this project's own standing constraint, set
early in this build): nuScenes has not been downloaded into this
environment -- the user's own words: "leave the nuscence one i will
download it and give you to test." Ticket #50's literal spec asks for
injection "into a real nuScenes sweep"; that step is DEFERRED here,
the same way every other nuScenes-gated check in this repository is
deferred (see e.g. `tests/test_taxonomy.py`'s real-data cross-checks).
This module instead injects into a SYNTHETIC flat-ground (ring,
azimuth) baseline built from a REAL `SensorConfig`'s own real
d_phi/h_m/n_beams constants -- everything about the SENSOR geometry is
real; only the underlying terrain is synthetic. That is an honest,
disclosed limitation, never presented as equivalent to a real sweep.

Injection formula (Bible Part 11's own reference, already cited in
`observability.negative_obstacle`'s own docstring): a trench/ditch of
depth d at range r, sensor mounted at height h, produces a range
shadow Delta ~= r*d/h.

Watch out (Build Map's own words, carried over verbatim): injecting
with the SAME Delta=r*d/h geometry the detector uses tests that this
implementation inverts its own forward model, not that the physics is
correct. An independent renderer (CARLA) would test the physics; this
project does not have that, and says so rather than implying otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from sensor.sensor_model import SensorConfig


@dataclass(frozen=True)
class RangeGrid:
    measured_range: np.ndarray  # (H, W) float, NaN where no return
    expected_range: np.ndarray  # (H, W) float, the flat-ground baseline (always defined here)
    valid_expected: np.ndarray  # (H, W) bool -- True wherever expected_range is meaningful
    occluded_mask: np.ndarray  # (H, W) bool -- always False for this synthetic baseline (no separate occluders)
    ring_phi_rad: np.ndarray  # (H,) each ring's depression angle below horizontal (positive = downward)


def _ring_depression_angles(sm: SensorConfig) -> np.ndarray:
    """Evenly-spaced synthetic ring elevation angles spanning downward
    from `phi_max_rad`, one `d_phi_rad` apart -- this project's own
    schedule convention (`sensor.schedule`), not any one real dataset's
    specific beam firing order (irrelevant here: only the ANGLE matters
    for a flat-ground range baseline). Returned as DEPRESSION angle
    (positive = pointed below horizontal), the sign convention this
    module's own formulas use throughout."""
    idx = np.arange(sm.n_beams, dtype=np.float64)
    elevation = sm.phi_max_rad - idx * sm.d_phi_rad  # top ring most elevated, decreasing
    return -elevation  # depression = -elevation; positive once elevation goes below horizontal


def flat_ground_baseline(sm: SensorConfig, n_azimuth: int = 360) -> RangeGrid:
    """A synthetic flat-ground (ring, azimuth) baseline at sensor
    height `sm.h_m`: `expected_range[ring] = h_m / sin(phi_ring)` for
    every downward-canted ring (phi_ring > 0); a ring aimed at or above
    the horizon never strikes flat ground and is marked invalid."""
    phi = _ring_depression_angles(sm)
    downward = phi > 0
    ranges = np.full(sm.n_beams, np.inf, dtype=np.float64)
    ranges[downward] = sm.h_m / np.sin(phi[downward])

    expected = np.tile(ranges[:, None], (1, n_azimuth))
    valid = np.tile(downward[:, None], (1, n_azimuth))
    measured = np.where(valid, expected, np.nan)
    occluded = np.zeros_like(measured, dtype=bool)
    return RangeGrid(measured_range=measured, expected_range=expected, valid_expected=valid, occluded_mask=occluded, ring_phi_rad=phi)


def inject_trench(
    grid: RangeGrid,
    sm: SensorConfig,
    azimuth_col: int,
    width_m: float,
    depth_m: float,
    near_range_m: float,
) -> RangeGrid:
    """A trench of `width_m`, starting at `near_range_m`, at ONE
    azimuth column: every ring whose flat-ground return would have
    landed inside [near_range_m, near_range_m + width_m] is displaced
    OUTWARD by Delta = r*d/h (Bible Part 11), or REMOVED entirely if
    the displaced range would fall beyond the trench's own far edge --
    physically, that means the far wall's own geometry intercepts the
    beam before it could reach that displaced point, and (per this
    module's own simplification, same as the ticket's) a beam
    absorbed/scattered by a near-vertical far wall at grazing incidence
    produces no return at all, rather than a wall-face return.
    """
    measured = grid.measured_range.copy()
    far_edge = near_range_m + width_m
    for ring in range(measured.shape[0]):
        if not grid.valid_expected[ring, azimuth_col]:
            continue
        r = grid.expected_range[ring, azimuth_col]
        if not (near_range_m <= r <= far_edge):
            continue
        delta = r * depth_m / sm.h_m
        displaced = r + delta
        measured[ring, azimuth_col] = np.nan if displaced > far_edge else displaced
    return replace(grid, measured_range=measured)


def inject_pole(
    grid: RangeGrid,
    sm: SensorConfig,
    azimuth_cols,
    diameter_m: float,
    height_m: float,
    range_m: float,
) -> RangeGrid:
    """A vertical cylinder of returns at `range_m`, spanning
    `azimuth_cols` (the pole's own angular width at that range) and
    every ring whose beam height at `range_m` falls within [0,
    height_m] -- those rings now return the POLE's surface (a CLOSER
    range than the flat-ground beyond it), all others are unaffected.
    """
    measured = grid.measured_range.copy()
    for ring in range(measured.shape[0]):
        phi = grid.ring_phi_rad[ring]
        if phi <= 0:
            continue  # never strikes the ground at all, so never strikes a pole standing on it either
        beam_height_at_range = sm.h_m - range_m * math.tan(phi)
        if not (0.0 <= beam_height_at_range <= height_m):
            continue
        for col in azimuth_cols:
            measured[ring, col] = range_m
    return replace(grid, measured_range=measured)


def kerb_face_ring_window_rad(height_m: float, range_m: float, sm: SensorConfig) -> float:
    """The ANGULAR WIDTH a kerb of `height_m` at `range_m` subtends
    from the sensor's own mount height -- `atan(h_m/r) -
    atan((h_m-height_m)/r)`, the EXACT form of Bible Part 3's
    `t/Delta_phi` height-resolution formula (which is this window's
    own small-angle approximation, verified to agree closely at the
    ranges this project actually operates over)."""
    top = math.atan(sm.h_m / range_m)
    bottom = math.atan((sm.h_m - height_m) / range_m)
    return top - bottom


def kerb_resolved(height_m: float, range_m: float, sm: SensorConfig, min_rings: int = 1) -> bool:
    """Whether a kerb of `height_m` at `range_m` is height-RESOLVED --
    at least `min_rings` beam ring(s) fall within the kerb's own
    angular face window, per `kerb_face_ring_window_rad`. This is a
    distinct question from Ticket #36's negative-obstacle ANOMALY
    detector (`observability.negative_obstacle.detect_anomalous_cells`,
    which only ever fires on cells FARTHER than expected -- a kerb, a
    POSITIVE obstacle, is a closer return and is explicitly skipped by
    that detector's own residual<=0 check). "Presence is not height"
    (Bible Part 11): this function measures the height-resolution
    question, matching Ticket #60's own kerb curve.
    """
    window_rad = kerb_face_ring_window_rad(height_m, range_m, sm)
    return window_rad >= min_rings * sm.d_phi_rad
