"""
planning/conservatism.py

Ticket #41 -- the caution order, and Bible Part 16's conservatism
invariant made checkable: cost is monotone non-decreasing under
information loss. Ticket #42's property test (tests/test_conservatism.py)
is what actually proves that; this module defines the order and the
public `cost()` surface the property is stated over.

The order: FREE <= known-rough <= UNKNOWN_COST < LETHAL, with
UNKNOWN_COST strictly below LETHAL and finite. Build Map's own trap:
setting UNKNOWN_COST = LETHAL "to be safe" technically satisfies
monotonicity and paralyses the vehicle -- monotonicity is necessary,
not sufficient. UNKNOWN_COST below is a stated, tunable POLICY
constant (not a physical vehicle parameter, so it does not live in
`vehicle_ugv.yaml` alongside braking_a_ms2/min_object_t_m/etc -- those
are declared platform facts; this is a caution-vs-mobility trade-off
the Build Map itself expects to be tuned, e.g. once Ticket #55's
latency numbers and a real Pareto sweep exist).

Scope note: `cost()` here covers the OBSERVABILITY/SPARSITY axis of
Bible Part 16's table (Tickets #34, #39, both already built) plus the
generic evidence-quality modifiers (provisional, inferred, single-point,
low class confidence, staleness) -- it does NOT yet incorporate real
slope/step/clearance-derived LETHAL conditions, since that needs
Ticket #48's traversability derivatives (not yet built) and is
explicitly Ticket #49's (`planning/costmap.py`) own job to add on top of
this module's order. The one LETHAL trigger this module DOES encode
(class == NEGATIVE_OBSTACLE) is included because it's what makes
"UNKNOWN_COST < LETHAL" a testable, non-vacuous statement before #49
exists -- see Part 14/16's own framing of negative obstacles as a
certain, geometry-derived hazard rather than a slope/clearance
violation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Optional

from grid.cell import OBS_FREE, OBS_OCCLUDED, OBS_OCCUPIED, OBS_UNOBSERVED
from observability.sparsity import SparsityVerdict
from perception.taxonomy import DrishtiClass
from sensor.vehicle_config import VehicleConfig

# ---------------------------------------------------------------------------
# The cost order. FREE_COST <= known-rough costs <= UNKNOWN_COST < LETHAL.
# ---------------------------------------------------------------------------

FREE_COST = 0.0
UNKNOWN_COST = 50.0  # high, finite, tunable -- see module docstring
LETHAL = math.inf

# "Known-rough" placeholder class costs -- a simplified stand-in for
# Ticket #49's real traversability-driven costmap (slope/roughness),
# which doesn't exist yet. Ordered so DRIVABLE=FREE, mild terrain sits
# BELOW UNKNOWN_COST (we're confident it's merely rough, cheaper than
# not knowing), and a CONFIRMED hazard class sits ABOVE UNKNOWN_COST
# (we're confident it's bad, which should cost more than merely not
# knowing) but still strictly below LETHAL.
_VEGETATION_COST = 2.0
_CAUTION_COST = 5.0
_KNOWN_HAZARD_COST = 200.0

_CLASS_BASE_COST = {
    int(DrishtiClass.DRIVABLE): FREE_COST,
    int(DrishtiClass.VEGETATION): _VEGETATION_COST,
    int(DrishtiClass.CAUTION): _CAUTION_COST,
    int(DrishtiClass.NON_TRAVERSABLE): _KNOWN_HAZARD_COST,
    int(DrishtiClass.STATIC_OBSTACLE): _KNOWN_HAZARD_COST,
    int(DrishtiClass.VEHICLE): _KNOWN_HAZARD_COST,
    int(DrishtiClass.PEDESTRIAN): _KNOWN_HAZARD_COST,
    int(DrishtiClass.OVERHANG): _KNOWN_HAZARD_COST,
    int(DrishtiClass.UNKNOWN): UNKNOWN_COST,
    # NEGATIVE_OBSTACLE deliberately absent -- handled before this table
    # is ever consulted (it is LETHAL, not a "known-rough" class cost).
}

_PROVISIONAL_INFERRED_FLOOR = UNKNOWN_COST * 0.6
_SINGLE_POINT_FLOOR = UNKNOWN_COST * 0.5
_LOW_CONFIDENCE_FLOOR = UNKNOWN_COST * 0.5
_LOW_CONFIDENCE_THRESHOLD = 0.3
CONFIDENCE_DECAY_TAU_S = 10.0  # Bible Part 12.3


@dataclass(frozen=True)
class CellState:
    """The evidence-quality state `cost()` reasons over. Deliberately a
    plain, independently-constructible dataclass (not read from a live
    Clipmap) so Ticket #42's property test can generate arbitrary
    combinations, including ones a real pipeline might never currently
    produce -- the property is meant to hold over the whole reachable
    space, not just today's code paths."""

    observability: int = OBS_FREE
    class_id: Optional[int] = int(DrishtiClass.DRIVABLE)
    class_confidence: float = 1.0  # 0..1
    sparsity_verdict: Optional[SparsityVerdict] = SparsityVerdict.NORMAL
    count: int = 10
    provisional: bool = False
    inferred: bool = False
    age_s: float = 0.0
    step_height_known: bool = True
    incidence_usable: bool = True
    has_positive_water_signature: bool = False


class InformationDeficitEntry(NamedTuple):
    name: str
    bible_part: str
    output: str
    direction: str  # always "cautious" in this table -- Part 16's own claim


# Bible Part 16's own enumeration, in code -- Ticket #41's explicit
# "Watch out" ask. `direction` is "cautious" for every row by
# construction: that uniformity IS the conservatism invariant, visible
# here as a single object instead of fourteen scattered decisions.
INFORMATION_DEFICIT_TABLE = (
    InformationDeficitEntry("never_observed", "Part 10", "UNOBSERVED -> UNKNOWN_COST", "cautious"),
    InformationDeficitEntry("observed_but_occluded", "Part 10", "OCCLUDED, never FREE", "cautious"),
    InformationDeficitEntry("sparse_structured", "Part 11", "SPARSE_STRUCTURED -> UNKNOWN_COST", "cautious"),
    InformationDeficitEntry("zero_returns_past_r_blind", "Part 11", "UNKNOWN, never FREE", "cautious"),
    InformationDeficitEntry("inherited_from_coarser_level", "Part 12.4", "PROVISIONAL, barred from fine decisions", "cautious"),
    InformationDeficitEntry("geometrically_completed_gap", "Part 12.5", "INFERRED, forbidden from reducing cost", "cautious"),
    InformationDeficitEntry("single_point_cell", "Part 9", "low confidence, never upgrades to DRIVABLE", "cautious"),
    InformationDeficitEntry("low_class_confidence_on_terrain", "Part 5", "degrades toward CAUTION, never up to DRIVABLE", "cautious"),
    InformationDeficitEntry("grazing_incidence_albedo_unusable", "Part 9.5", "radiometric channel unusable, no false clear", "cautious"),
    InformationDeficitEntry("missing_returns_ambiguous_cause", "Part 10 / 9.5", "UNKNOWN unless a positive signature discriminates", "cautious"),
    InformationDeficitEntry("step_height_beyond_l0_l1", "Part 14", "UNKNOWN, not \"no step\"", "cautious"),
    InformationDeficitEntry("stale_cell_confidence_decayed", "Part 12.3", "cost rises with age, saturating at UNKNOWN_COST", "cautious"),
    InformationDeficitEntry("bad_ego_velocity", "Part 13", "fovea degrades to the spec floor (Profile A)", "cautious"),
    InformationDeficitEntry("high_residual_variance_on_track", "Part 12.6", "track confidence lowered", "cautious"),
    InformationDeficitEntry("sensor_degraded_dust_rain_dropout", "Part 15", "speed envelope contracts", "cautious"),
)


def cost(cell: CellState, vehicle: VehicleConfig) -> float:
    """The public decision surface Part 16's invariant is stated over.
    Any fast path bypassing this function is, by definition, unsupported
    (Ticket #42's own framing) -- there must be no other way to ask
    "how costly is this cell" than through here.
    """
    # A certain, geometry-derived hazard: LETHAL regardless of any
    # OTHER information deficit (this is what makes "UNKNOWN_COST <
    # LETHAL" a real, testable ceiling rather than a vacuous one).
    if cell.class_id == int(DrishtiClass.NEGATIVE_OBSTACLE):
        return LETHAL

    # Class-based cost first, UNCHANGED by any information-deficit flag
    # below -- deficits are a FLOOR (max), never a substitute for
    # already-known evidence. This is the fix for a real bug this
    # module's own test suite caught while it was being written:
    # returning UNKNOWN_COST outright for e.g. "observed but occluded"
    # would silently LOWER a cell's cost if its class was already known
    # to be worse than UNKNOWN_COST (e.g. a confirmed STATIC_OBSTACLE) --
    # forgetting a wall must never make routing through it look safer.
    # Rows 1-4, 9-11 of the table above become FLOOR contributions
    # (deficit_floor), never a direct return.
    class_id = cell.class_id if cell.class_id is not None else int(DrishtiClass.UNKNOWN)
    class_based = _CLASS_BASE_COST.get(class_id, UNKNOWN_COST)

    deficit_floor = FREE_COST
    if cell.observability == OBS_UNOBSERVED:
        deficit_floor = max(deficit_floor, UNKNOWN_COST)
    if cell.observability == OBS_OCCLUDED:
        deficit_floor = max(deficit_floor, UNKNOWN_COST)
    if cell.sparsity_verdict in (SparsityVerdict.SPARSE_STRUCTURED, SparsityVerdict.UNKNOWN):
        deficit_floor = max(deficit_floor, UNKNOWN_COST)
    if not cell.step_height_known:
        deficit_floor = max(deficit_floor, UNKNOWN_COST)
    if not cell.incidence_usable and not cell.has_positive_water_signature:
        deficit_floor = max(deficit_floor, UNKNOWN_COST)
    if cell.provisional or cell.inferred:
        deficit_floor = max(deficit_floor, _PROVISIONAL_INFERRED_FLOOR)
    if cell.count <= 1:
        deficit_floor = max(deficit_floor, _SINGLE_POINT_FLOOR)
    if cell.class_confidence < _LOW_CONFIDENCE_THRESHOLD:
        deficit_floor = max(deficit_floor, _LOW_CONFIDENCE_FLOOR)

    base = max(class_based, deficit_floor)

    # Row 12: confidence decay with age -- cost rises toward
    # UNKNOWN_COST as a cell goes stale, saturating there (Part 16's own
    # edge case: "Decay saturates at UNKNOWN_COST; it never crosses into
    # LETHAL"). A no-op at age_s=0. Only ever able to INCREASE base
    # (skipped entirely once base already reaches/exceeds UNKNOWN_COST),
    # so it cannot violate monotonicity either.
    if base < UNKNOWN_COST:
        decay_factor = 1.0 - math.exp(-cell.age_s / CONFIDENCE_DECAY_TAU_S)
        base = base + (UNKNOWN_COST - base) * decay_factor

    return base
