"""
attention/fovea_controller.py

Ticket #47 -- TTC (time-to-contact) fovea controller, Bible Part 13.
Where resolution is spent is a function of how SOON the vehicle could
reach a point, not how far away it is: a stationary point 40m dead
ahead at 60 km/h is 2.4s away and safety-critical; the same point 40m
to the side, on a straight path, will never be reached.

    v_close(p) = v . p_hat
    TTC(p)     = ||p|| / max(v_close(p), v_min)
    c_ttc(p)   = c0 * (TTC(p) / tau0) ** gamma

The composition rule IS the safety property (Build Map's own framing:
"the min() is the safety property... every term can only refine, never
coarsen"):

    c(p) = min(c_range(r), c_ttc(p), c_boundary(p))

`c_range` (Ticket #5's own resolution schedule, `sensor.schedule`) is a
HARD FLOOR that is never relaxed -- Profile A (default, spec-compliant)
is c_range alone; Profile B adds the refinement terms. No bug in this
module and no bad velocity estimate can ever make the map coarser than
c_range: the worst Profile B can do is degrade to Profile A.

`c_object` (any tracked entity gets a fine patch regardless of TTC) is
DESIGNED, NOT BUILT -- it needs a Kalman tracker (Build Map's own cut
list), which this project does not build. Its absence means a distant
laterally-moving pedestrian is not specially refined; Bible Part 13
names this precisely ("this is what makes lateral coarsening safe" --
an argument FOR c_object, not against noting its absence honestly).
Profile B here is therefore min(c_range, c_ttc, c_boundary), a proper
(if incomplete) subset of Bible Part 13's four-term composition --
every term it DOES include still only ever refines, never coarsens,
c_range's floor, which is the one property that must hold regardless
of how many of the four terms are actually implemented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from grid.clipmap import Clipmap
from sensor.schedule import generate_schedule
from sensor.sensor_model import SensorConfig

TAU0_S = 1.0
V_MIN_MS = 2.0
DEFAULT_GAMMA = 1.0
DEFAULT_C0_M = 0.05

Point2D = Tuple[float, float]


def v_close(v: Point2D, p: Point2D) -> float:
    """v . p_hat -- the ego velocity's component TOWARD point p. Zero
    at the origin (r=0), where "toward" is undefined; TTC's own
    `max(v_close, v_min)` is what actually guards against a
    negative/undefined closing speed for a receding or lateral point,
    not this function."""
    r = math.hypot(p[0], p[1])
    if r == 0.0:
        return 0.0
    return (v[0] * p[0] + v[1] * p[1]) / r


def ttc_s(v: Point2D, p: Point2D, v_min: float = V_MIN_MS) -> float:
    """||p|| / max(v_close(p), v_min). `v_min` floors the closing speed
    so a lateral or receding point (v_close <= 0) never produces an
    infinite or negative TTC -- Bible Part 13's standstill defence."""
    r = math.hypot(p[0], p[1])
    closing = max(v_close(v, p), v_min)
    return r / closing


def c_ttc(
    v: Point2D,
    p: Point2D,
    gamma: float = DEFAULT_GAMMA,
    tau0: float = TAU0_S,
    c0: float = DEFAULT_C0_M,
    v_min: float = V_MIN_MS,
) -> float:
    t = ttc_s(v, p, v_min=v_min)
    return c0 * (t / tau0) ** gamma


def c_range(r_m: float, sm: SensorConfig, c0: float = DEFAULT_C0_M, n_levels: int = 4) -> float:
    """Ticket #5's own resolution schedule (`sensor.schedule`) as a
    HARD FLOOR: the finest level whose sensor-Nyquist radius still
    covers `r_m`, or the coarsest level if `r_m` exceeds every level's
    radius. This is the SAME schedule the clipmap itself is built from
    -- not a parallel re-derivation -- so this floor can never silently
    disagree with what the map's own levels actually are."""
    levels = generate_schedule(sm, c0=c0, n_levels=n_levels)
    for lvl in levels:
        if r_m <= lvl.nyquist_radius_m:
            return lvl.cell_size_m
    return levels[-1].cell_size_m


def c_boundary(p: Point2D, cm: Optional[Clipmap], level: int) -> float:
    """Refine to the queried level's OWN cell size wherever the class
    label differs between a cell and one of its 4-connected neighbours
    -- a class change IS the kerb, the verge, the drop-off (Bible Part
    13); a flat interior needs none of this, an edge needs all of it.
    No live map (`cm=None`, e.g. a pure-geometry caller) or an
    unobserved query cell means no boundary evidence exists yet --
    returns +inf so `min()` simply ignores this term rather than
    fabricating an edge that hasn't actually been observed."""
    if cm is None:
        return math.inf
    lvl = cm.levels[level]
    c_l = lvl.cell_size_m
    _, center = cm.lookup(p[0], p[1])
    if not center.observed:
        return math.inf
    center_class = (center.class_conf >> 4) & 0xF
    for dx, dy in ((c_l, 0.0), (-c_l, 0.0), (0.0, c_l), (0.0, -c_l)):
        _, nbr = cm.lookup(p[0] + dx, p[1] + dy)
        if nbr.observed and ((nbr.class_conf >> 4) & 0xF) != center_class:
            return c_l
    return math.inf


@dataclass(frozen=True)
class FoveaParams:
    gamma: float = DEFAULT_GAMMA
    c0: float = DEFAULT_C0_M
    tau0_s: float = TAU0_S
    v_min_ms: float = V_MIN_MS
    n_levels: int = 4


def fovea_cell_size(
    p: Point2D,
    v: Point2D,
    sm: SensorConfig,
    profile: str = "A",
    cm: Optional[Clipmap] = None,
    boundary_level: int = 0,
    params: FoveaParams = FoveaParams(),
) -> float:
    """The public entry point. Profile A (default, spec-compliant) is
    `c_range(r)` alone -- "demoed first", per the Bible, and what every
    number elsewhere in this project is reported for. Profile B
    composes `min(c_range, c_ttc, c_boundary)` -- see module docstring
    for why `c_object` is absent. The `min()` here is not an
    implementation detail; it is THE safety property (Build Map's own
    words): every additional term can only shrink the returned cell
    size relative to Profile A's floor, never grow it.
    """
    r = math.hypot(p[0], p[1])
    floor = c_range(r, sm, c0=params.c0, n_levels=params.n_levels)
    if profile == "A":
        return floor
    if profile != "B":
        raise ValueError(f"unknown fovea profile {profile!r}, expected 'A' or 'B'")
    ttc_term = c_ttc(v, p, gamma=params.gamma, tau0=params.tau0_s, c0=params.c0, v_min=params.v_min_ms)
    boundary_term = c_boundary(p, cm, boundary_level)
    return min(floor, ttc_term, boundary_term)


@dataclass(frozen=True)
class GazeTarget:
    point: Point2D
    ttc_s: float
    c_ttc_m: float


def find_gaze_target(
    candidates: Sequence[Point2D],
    v: Point2D,
    gamma: float = DEFAULT_GAMMA,
    tau0: float = TAU0_S,
    c0: float = DEFAULT_C0_M,
    v_min: float = V_MIN_MS,
) -> Optional[GazeTarget]:
    """Saccadic gaze steering. `c_ttc` above already answers "how fine
    must resolution be HERE," per point, on demand -- a saccade needs
    the complementary question: "which ONE point, among several
    plausible hazards, is most urgent RIGHT NOW." Urgency here is the
    SAME `ttc_s` the resolution schedule already computes (a lower TTC
    is a sooner, more urgent point), so this is not a new metric, just
    a new REDUCTION (argmin) over an existing one -- the fovea's
    required-resolution term and the gaze beam's chosen target answer
    two different questions about the same underlying "how soon"
    quantity, they do not duplicate it.

    `candidates` is caller-supplied (e.g. the range-shadow "SUSPECT"
    cells `observability.negative_obstacle` flags, or any tracked
    entity's position) -- this function has no opinion on WHERE
    candidates come from, only on which one is currently most urgent.
    Returns `None` for an empty candidate list: no candidate hazards
    means no saccade target, never a fabricated one at the origin.
    """
    if not candidates:
        return None
    best: Optional[GazeTarget] = None
    for p in candidates:
        t = ttc_s(v, p, v_min=v_min)
        if best is None or t < best.ttc_s:
            c = c_ttc(v, p, gamma=gamma, tau0=tau0, c0=c0, v_min=v_min)
            best = GazeTarget(point=p, ttc_s=t, c_ttc_m=c)
    return best
