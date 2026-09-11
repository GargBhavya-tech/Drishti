"""
planning/path_smoothing.py

Kinodynamic path smoothing. `planning.path_planner.find_path`'s A* runs
on an 8-connected grid, so every corner in its output path is a
multiple of 45 degrees -- a heading discontinuity no real skid-steer or
Ackermann UGV can execute at speed without slipping or rolling (a step
change in heading implies infinite instantaneous curvature, hence
infinite required lateral force). This module fits a smooth curve
through the A* waypoints and derives a physically real cornering-speed
limit from that curve's curvature, reusing `planning.friction`'s own
CLASS_TO_MU table -- so "how fast can I take this turn" and "how fast
can I brake on this surface" share one friction assumption, not two
coincidentally similar numbers.

    a_lat_max = mu * g                    (friction-limited lateral accel)
    v_max(R)  = sqrt(a_lat_max * R)        (standard skid-limited cornering-
                                             speed relation, any vehicle-
                                             dynamics text)

Curve fit: a uniform Catmull-Rom spline through the waypoints, NOT a
true Dubins curve. A Dubins curve enforces a MINIMUM turning radius
exactly and is the physically correct model for Ackermann steering, but
its construction has six case-by-case word types (LSL/RSR/LSR/RSL/RLR/
LRL) -- a substantial extra implementation surface for a demo-time
addition. Catmull-Rom gives a visually and numerically smooth (C1-
continuous) path through the SAME waypoints at a fraction of the
complexity. Documented explicitly here so nobody mistakes this for a
minimum-turning-radius guarantee -- it is not one; it removes the grid's
artificial 45-degree kinks, it does not enforce a vehicle's real
steering-lock limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

from perception.taxonomy import DrishtiClass
from planning.friction import mu_for_class
from planning.path_planner import Cell

G_MS2 = 9.81
DEFAULT_SAMPLES_PER_SEGMENT = 8


def _catmull_rom_axis(a0: float, a1: float, a2: float, a3: float, t: float) -> float:
    """Standard uniform Catmull-Rom basis, evaluated for one coordinate
    axis at parameter t in [0, 1] between control points a1 and a2
    (a0/a3 are the neighbours either side, used only to shape the
    tangents)."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * a1)
        + (-a0 + a2) * t
        + (2 * a0 - 5 * a1 + 4 * a2 - a3) * t2
        + (-a0 + 3 * a1 - 3 * a2 + a3) * t3
    )


def _catmull_rom_point(
    p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], t: float
) -> Tuple[float, float]:
    return (
        _catmull_rom_axis(p0[0], p1[0], p2[0], p3[0], t),
        _catmull_rom_axis(p0[1], p1[1], p2[1], p3[1], t),
    )


def smooth_path(
    waypoints: Sequence[Cell], samples_per_segment: int = DEFAULT_SAMPLES_PER_SEGMENT
) -> List[Tuple[float, float]]:
    """Catmull-Rom-interpolated points through `waypoints` (the (row,
    col) integer cells from `find_path`'s own `PathResult.path`).
    Endpoint waypoints are duplicated (the standard Catmull-Rom
    convention for an open, non-looping curve) so the curve actually
    reaches the true start/goal instead of falling short by one
    control point's worth of curve. Fewer than 3 waypoints is returned
    unchanged: a spline through 0-2 points is either nothing or a
    straight line, and resampling a straight line would waste cycles
    for zero visual or physical difference.
    """
    pts = [(float(r), float(c)) for r, c in waypoints]
    if len(pts) < 3:
        return pts

    padded = [pts[0]] + pts + [pts[-1]]
    out: List[Tuple[float, float]] = []
    for i in range(len(padded) - 3):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        for s in range(samples_per_segment):
            t = s / samples_per_segment
            out.append(_catmull_rom_point(p0, p1, p2, p3, t))
    out.append(pts[-1])
    return out


def curvature_radius(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Circumradius of the triangle p0-p1-p2: R = (a*b*c) / (4*Area),
    the standard three-point discrete curvature estimate (R -> infinity
    as the three points approach collinear, matching a real straight
    stretch's true zero curvature). Returns `math.inf` for
    (near-)collinear or coincident points rather than dividing by
    (near-)zero.
    """
    a = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    b = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    c = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
    if a == 0.0 or b == 0.0 or c == 0.0:
        return math.inf
    area = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])) / 2.0
    if area < 1e-9:
        return math.inf
    return (a * b * c) / (4.0 * area)


def lateral_speed_limit_ms(radius_m: float, mu: float, g_ms2: float = G_MS2) -> float:
    """v_max(R) = sqrt(mu * g * R), the friction-limited cornering
    speed. `radius_m = math.inf` (a straight stretch) returns
    `math.inf` -- no curvature-imposed limit at all there, matching this
    project's own "advisory, only ever adds a constraint, never removes
    one that already exists" discipline (see `speed_envelope`'s own
    module docstring).
    """
    if mu <= 0:
        raise ValueError(f"mu must be positive, got {mu}")
    if not math.isfinite(radius_m):
        return math.inf
    return math.sqrt(mu * g_ms2 * radius_m)


@dataclass(frozen=True)
class CurvatureSpeedSample:
    point: Tuple[float, float]   # smoothed-path point, in the SAME grid units as the input waypoints
    radius_m: float
    v_max_ms: float
    drishti_class: DrishtiClass
    mu: float


def curvature_speed_profile(
    smoothed_points: Sequence[Tuple[float, float]],
    cell_size_m: float,
    class_at_point: Callable[[Tuple[float, float]], DrishtiClass],
) -> List[CurvatureSpeedSample]:
    """For every interior point of `smoothed_points`, estimate local
    curvature radius (converted to METRES via `cell_size_m`) from its
    two neighbours, look up the DRISHTI class under that point
    (`class_at_point`, caller-supplied so this module stays grid-
    representation-agnostic), and derive the friction-limited cornering
    speed using `planning.friction`'s own `mu_for_class` -- the SAME mu
    table the braking governor uses, so a tight turn through mud is
    doubly penalised (low braking a_max AND low cornering speed) for
    one shared, honest reason (low friction), not two independently
    tuned numbers that happen to agree.

    Fewer than 3 points has no interior point to evaluate and returns
    an empty list, not an error -- a 1-2 point "path" has no curvature
    to speak of.
    """
    if len(smoothed_points) < 3:
        return []
    out: List[CurvatureSpeedSample] = []
    for i in range(1, len(smoothed_points) - 1):
        p_prev = smoothed_points[i - 1]
        p_curr = smoothed_points[i]
        p_next = smoothed_points[i + 1]
        r_grid = curvature_radius(p_prev, p_curr, p_next)
        r_m = r_grid * cell_size_m if math.isfinite(r_grid) else math.inf
        cls = class_at_point(p_curr)
        mu = mu_for_class(cls)
        v_max = lateral_speed_limit_ms(r_m, mu)
        out.append(CurvatureSpeedSample(point=p_curr, radius_m=r_m, v_max_ms=v_max, drishti_class=cls, mu=mu))
    return out
