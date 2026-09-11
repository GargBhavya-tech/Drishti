"""
planning/costmap.py

Ticket #49 -- the real cost function a planner consumes (Bible Part
14), built ON TOP of `planning.conservatism`'s evidence-quality floor
(#41/#42) rather than duplicating it: this module adds exactly the
piece `conservatism.py`'s own docstring says is NOT yet included there
-- slope/step/clearance-derived LETHAL conditions and a real
traversability-driven cost, from Ticket #48's geometric derivatives --
and nothing else. Reusing `conservatism.deficit_floor()` (the same
function `conservatism.cost()` itself calls) rather than writing a
second, parallel evidence-quality check is what keeps Ticket #42's
monotonicity property test still meaningful once real geometry is
wired in (Build Map's own "Done when": "#42's property test still
passes with the real cost function wired in" -- it does, unchanged,
because this module does not touch `conservatism.py`'s own cost path
at all, only reuses one already-tested piece of it).

    cost = LETHAL         if class == NEGATIVE_OBSTACLE     (conservatism's own check)
    cost = LETHAL         if slope > max_slope
    cost = LETHAL         if step > max_step_height
    cost = LETHAL         if clearance < min_clearance
    cost = UNKNOWN_COST   if UNOBSERVED, SPARSE_STRUCTURED, or evidence otherwise deficient
                              (conservatism.deficit_floor()'s own table)
    cost = w1*slope/max_slope + w2*roughness/max_roughness + w3*class_penalty   otherwise

All thresholds come from `vehicle_ugv.yaml` (`VehicleConfig`) -- never
hardcoded (Bible Part 14: "declared once, in config").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from planning.conservatism import CONFIDENCE_DECAY_TAU_S, LETHAL, UNKNOWN_COST, CellState, deficit_floor
from planning.traversability import TraversabilityResult
from perception.taxonomy import DrishtiClass
from sensor.vehicle_config import VehicleConfig

# Weighted-blend coefficients for the "otherwise" branch (Bible Part
# 14's w1/w2/w3). Not specified numerically by the Bible/Build Map
# beyond "weighted blend"; chosen so slope and roughness contribute
# comparably once each crosses its OWN threshold (w1=w2=1, since each
# term is already normalised by its own vehicle limit) and class
# penalty is a smaller tie-breaking term (w3=0.5) -- all deliberately
# well below UNKNOWN_COST so this branch alone can never accidentally
# reach the tri-state ceiling. Exposed as module constants specifically
# so they can be retuned in one place if a real Pareto sweep (Ticket
# #58) later argues for different weights.
W_SLOPE = 1.0
W_ROUGHNESS = 1.0
W_CLASS = 0.5

# Bible Part 14: "kappa < tau_kappa" triggers UNKNOWN_COST. Ticket
# #39's own NORMAL/SPARSE_STRUCTURED boundary is kappa=1.0 (classify_sparsity:
# kappa >= 1.0 is NORMAL, kappa < 1.0 with n_obs>0 is SPARSE_STRUCTURED
# or NOISE_SUPPRESSED) -- reusing that SAME boundary here rather than a
# second, independently-tuned threshold. In practice this module
# receives kappa's CONSEQUENCE (a `CellState.sparsity_verdict` already
# classified by Ticket #39) via `deficit_floor()`, not a raw kappa
# value -- this constant is kept for callers that want to duplicate the
# threshold check themselves against a raw kappa.
TAU_KAPPA = 1.0


@dataclass(frozen=True)
class GeometryResult:
    """A thin alias of `planning.traversability.TraversabilityResult`
    plus a class penalty -- kept as its own type (rather than importing
    `TraversabilityResult` directly into every costmap call site) so a
    caller who does not have a live `Clipmap` handy (e.g. a unit test)
    can construct one directly."""

    slope_deg: Optional[float]
    roughness_m: Optional[float]
    step_height_m: Optional[float]
    clearance_m: float
    class_penalty: float = 0.0

    @staticmethod
    def from_traversability(t: TraversabilityResult, class_penalty: float = 0.0) -> "GeometryResult":
        return GeometryResult(
            slope_deg=t.slope_deg,
            roughness_m=t.roughness_m,
            step_height_m=t.step_height_m,
            clearance_m=t.clearance_m,
            class_penalty=class_penalty,
        )


def geometric_lethal(geo: GeometryResult, vehicle: VehicleConfig) -> bool:
    """Any of the three geometric LETHAL conditions independently
    firing. `None` (UNKNOWN, per Ticket #48) never triggers LETHAL on
    its own here -- an un-resolvable slope/step is an evidence deficit,
    handled by `deficit_floor()`'s own UNKNOWN_COST path below, not a
    confirmed hazard."""
    if geo.slope_deg is not None and geo.slope_deg > vehicle.max_slope_deg:
        return True
    if geo.step_height_m is not None and geo.step_height_m > vehicle.max_step_height_m:
        return True
    if geo.clearance_m < vehicle.min_clearance_m:
        return True
    return False


def cost(cell: CellState, geo: GeometryResult, vehicle: VehicleConfig) -> float:
    """The full Ticket #49 cost. `cell.class_id ==
    DrishtiClass.NEGATIVE_OBSTACLE` and the three geometric conditions
    are all independent, unconditional LETHAL triggers (Bible Part 14
    lists all four side by side, not as an if/elif chain) -- any one
    firing is enough, regardless of what the others say.

    Otherwise: `conservatism.deficit_floor(cell)` -- the SAME
    evidence-quality floor `conservatism.cost()` itself uses -- is
    combined with the real geometric blend via `max()`, exactly
    mirroring `conservatism.cost()`'s own `max(class_based,
    deficit_floor(cell))` structure, just substituting the REAL
    traversability blend for that module's placeholder class-based
    table (its own docstring names this substitution as #49's job).
    An un-observed or otherwise evidence-deficient cell is therefore
    never made to look safe merely because it happens to measure flat
    -- the deficit floor still applies even when the geometric blend
    alone would have computed something lower.
    """
    if cell.class_id == int(DrishtiClass.NEGATIVE_OBSTACLE):
        return LETHAL
    if geometric_lethal(geo, vehicle):
        return LETHAL

    floor = deficit_floor(cell)

    if geo.slope_deg is None or geo.roughness_m is None:
        # Missing geometry (Ticket #48's own UNKNOWN case) is itself a
        # deficit -- it must never silently fall through to a "free"
        # blend just because there was nothing to blend.
        blend = UNKNOWN_COST
    else:
        blend = (
            W_SLOPE * (geo.slope_deg / vehicle.max_slope_deg)
            + W_ROUGHNESS * (geo.roughness_m / vehicle.max_roughness_m)
            + W_CLASS * geo.class_penalty
        )

    base = max(blend, floor)
    if base < UNKNOWN_COST:
        decay_factor = 1.0 - math.exp(-cell.age_s / CONFIDENCE_DECAY_TAU_S)
        base = base + (UNKNOWN_COST - base) * decay_factor
    return base
