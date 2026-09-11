"""
planning/friction.py

The Semantic Friction Governor. `planning.speed_envelope.speed_envelope`
takes ONE scalar braking deceleration (`vehicle.braking_a_ms2`, from
`configs/vehicle_ugv.yaml`) for the whole vehicle -- an implicit "dry,
firm ground" assumption baked into that single declared platform
parameter. This module makes that assumption explicit and PER-CLASS,
using the terrain classifier DRISHTI already trains and already emits
(`perception.taxonomy.DrishtiClass`) -- so a real, already-existing
neural-net output directly changes how fast the vehicle is allowed to
brake, not a new sensor or a new model.

IMPORTANT taxonomy-honesty note, read before citing this to anyone:
DRISHTI's 10-class taxonomy (Ticket #8, `perception/taxonomy.py`)
already MERGES finer real-world materials for hazard-detection
purposes -- grass/tree/bush all become VEGETATION, mud/puddle both
become CAUTION, dirt/asphalt/concrete all become DRIVABLE (see
`RELLIS_TO_DRISHTI` there). This module can therefore only modulate
friction at DRISHTI-class granularity (three real drivable tiers), NOT
distinguish "mud" from "wet grass" from "sand" the way a dedicated
from-scratch material classifier could -- that finer distinction was
deliberately traded away at Ticket #8 in favour of hazard-detection
accuracy. The honest claim is still real and still novel (a trained
semantic segmentation network's own output changing a physically
derived braking limit); just don't oversell the granularity.

CLASS_TO_MU values below are DECLARED ENGINEERING ESTIMATES (a
dry-ground/wet-vegetation/mud-or-puddle spread consistent with
published friction-coefficient ranges for those surface types), NOT
measured on this specific vehicle -- the same "declared, not measured"
status `vehicle_ugv.yaml` itself gives `t_react_s` until a real test
exists. Treat MU_DRY_REFERENCE as the terrain this project's own
`braking_a_ms2 = 4.0` already assumes; every other class's derated
`a_max` is expressed as a fraction of that one number, so a future
correction to `vehicle_ugv.yaml`'s own figure rescales every tier with
it automatically rather than needing a second edit somewhere else.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, Optional, Tuple

from perception.taxonomy import DrishtiClass
from planning.speed_envelope import SpeedEnvelopeResult, speed_envelope
from sensor.vehicle_config import VehicleConfig

# The terrain assumption implicit in vehicle_ugv.yaml's braking_a_ms2 --
# dry, firm ground. See module docstring: every other class's derated
# a_max is `braking_a_ms2 * (mu_class / MU_DRY_REFERENCE)`.
MU_DRY_REFERENCE = 0.80

CLASS_TO_MU: Dict[DrishtiClass, float] = {
    DrishtiClass.DRIVABLE: 0.80,       # dry dirt/asphalt/concrete -- the config's own baseline
    DrishtiClass.VEGETATION: 0.45,     # grass/bush/tree litter -- meaningfully less grip than bare ground
    DrishtiClass.CAUTION: 0.35,        # mud/puddle -- the worst realistic DRIVABLE-adjacent surface
    DrishtiClass.UNKNOWN: 0.50,        # unobserved -- never assume best-case; a conservative middle value
    # The remaining classes are hazards/obstacles the cost map already
    # keeps the planner off of (LETHAL or high-cost, see
    # planning.costmap) -- given a low value only so that IF one is
    # ever queried anyway (e.g. a path that grazes a hazard cell's
    # neighbour), binding_mu()'s min() stays conservative rather than
    # silently assuming full dry-ground grip on ground the vehicle
    # should not be on in the first place.
    DrishtiClass.NON_TRAVERSABLE: 0.30,
    DrishtiClass.STATIC_OBSTACLE: 0.30,
    DrishtiClass.VEHICLE: 0.30,
    DrishtiClass.PEDESTRIAN: 0.30,
    DrishtiClass.NEGATIVE_OBSTACLE: 0.30,
    DrishtiClass.OVERHANG: 0.30,
}


def mu_for_class(drishti_class: DrishtiClass) -> float:
    """CLASS_TO_MU lookup with a conservative fallback. The fallback is
    UNKNOWN's own mu, never MU_DRY_REFERENCE -- an unrecognised class
    must never silently receive the BEST-case friction assumption."""
    return CLASS_TO_MU.get(drishti_class, CLASS_TO_MU[DrishtiClass.UNKNOWN])


def binding_mu(classes_along_path: Iterable[DrishtiClass]) -> Tuple[float, Optional[DrishtiClass]]:
    """The worst (lowest) friction coefficient among classes observed
    along the vehicle's LOOKAHEAD path -- the whole point of a
    perception-limited governor is anticipation (slow down BEFORE the
    slick patch), not reaction (skid, then slow down). Returns
    `(MU_DRY_REFERENCE, None)` for an empty iterable: no terrain
    evidence ahead is not a reason to derate anything, matching
    `speed_envelope`'s own "advisory, only ever adds a constraint, does
    not gate" discipline.
    """
    worst_mu = MU_DRY_REFERENCE
    worst_class: Optional[DrishtiClass] = None
    for c in classes_along_path:
        mu = mu_for_class(c)
        if mu < worst_mu:
            worst_mu = mu
            worst_class = c
    return worst_mu, worst_class


def friction_adjusted_vehicle(vehicle: VehicleConfig, mu: float) -> VehicleConfig:
    """A NEW `VehicleConfig` (frozen dataclass -- `dataclasses.replace`,
    the original `vehicle` is never mutated) whose `braking_a_ms2` is
    scaled by `mu / MU_DRY_REFERENCE`. Physically: under the simplified
    friction-limited-braking model, maximum deceleration scales
    linearly with the available friction coefficient
    (`a_max = mu * g`) -- so a `mu` half the dry reference means
    `a_max` halves, and because `stopping_distance_m`'s own reaction-
    distance term is linear in `v` while its braking term is
    `v^2 / (2a)`, halving `a` more than doubles stopping distance at
    any real speed, not merely doubles it.
    """
    if mu <= 0:
        raise ValueError(f"mu must be positive, got {mu}")
    scale = mu / MU_DRY_REFERENCE
    return replace(vehicle, braking_a_ms2=vehicle.braking_a_ms2 * scale)


def friction_adjusted_speed_envelope(
    hazard_ranges_m: Dict[str, float],
    vehicle: VehicleConfig,
    classes_along_path: Iterable[DrishtiClass],
) -> Tuple[SpeedEnvelopeResult, float, Optional[DrishtiClass]]:
    """`speed_envelope`'s own sensor-range-limited v_max, computed
    against a vehicle whose braking capability has been derated for the
    worst terrain class observed along the lookahead path.
    `planning.speed_envelope.speed_envelope` itself is untouched by this
    module -- this is a pure wrapper, so every existing caller/test of
    it is unaffected. Returns `(result, mu_used, binding_class)` so a
    caller can report WHY the speed limit is what it is, the same
    discipline `speed_envelope` already applies to its own hazard-range
    argument.
    """
    mu, binding_class = binding_mu(classes_along_path)
    derated_vehicle = friction_adjusted_vehicle(vehicle, mu)
    result = speed_envelope(hazard_ranges_m, derated_vehicle)
    return result, mu, binding_class
