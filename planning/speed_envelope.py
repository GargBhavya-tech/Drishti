"""
planning/speed_envelope.py

Ticket #40 -- the perception-limited speed envelope (Claim 4). Bible
Part 15: stopping distance set equal to detection range gives the
fastest speed at which the vehicle can still stop for what it can see.
The sensor, not the drivetrain, is the speed limit.

    d_stop(v) = v*t_react + v^2/(2a)

Setting d_stop(v) = R and solving the quadratic for the largest v:

    v_max(R) = -a*t_react + sqrt(a^2*t_react^2 + 2*a*R)

`a` (braking_a_ms2) and `t_react` (t_react_s) come from
`vehicle_ugv.yaml` -- declared platform parameters, never literals here
(same discipline `tests/test_vehicle_config.py` already enforces for
the config's other watched values).

This is advisory output only (Build Map's own explicit warning) -- it
reports what the sensor justifies, and does not gate or control
anything. What a planner does with it is the planner's contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from sensor.vehicle_config import VehicleConfig


def stopping_distance_m(v_ms: float, vehicle: VehicleConfig) -> float:
    """d_stop(v) = v*t_react + v^2/(2a)."""
    return v_ms * vehicle.t_react_s + (v_ms ** 2) / (2.0 * vehicle.braking_a_ms2)


def v_max_for_range(detection_range_m: float, vehicle: VehicleConfig) -> float:
    """The largest speed at which stopping distance still fits inside
    `detection_range_m`. Solving d_stop(v) = R for v via the quadratic
    formula (v^2/(2a) + v*t_react - R = 0), taking the positive root --
    the negative root is not a physical speed."""
    a = vehicle.braking_a_ms2
    t_r = vehicle.t_react_s
    return -a * t_r + math.sqrt((a * t_r) ** 2 + 2.0 * a * detection_range_m)


@dataclass(frozen=True)
class SpeedEnvelopeResult:
    v_max_ms: float
    v_max_kmh: float
    binding_hazard: Optional[str]
    binding_range_m: Optional[float]
    per_hazard_v_max_ms: Dict[str, float]


def speed_envelope(hazard_ranges_m: Dict[str, float], vehicle: VehicleConfig) -> SpeedEnvelopeResult:
    """The operational speed limit given a SET of hazards the terrain
    could plausibly contain, each with its own detection range (Bible
    Part 15: "the vehicle's safe speed is set by the SHORTEST detection
    range among the hazard classes the terrain can plausibly contain").
    Reports which hazard is binding, not just the final number -- an
    operator needs to know WHY the limit is what it is, not just what
    it is (Build Map Ticket #40's own "Done when").

    `hazard_ranges_m`: e.g. {"2m_ditch": 21.6, "5cm_cable": 6.7} --
    detection ranges already derived elsewhere (sensor_model.r_max_ditch,
    r_max_height, observability.sparsity.r_blind_for_min_object, ...);
    this function's only job is turning a range into a speed and picking
    the binding (minimum) one.
    """
    if not hazard_ranges_m:
        raise ValueError("hazard_ranges_m must contain at least one hazard -- there is no envelope over an empty set")

    per_hazard = {name: v_max_for_range(r, vehicle) for name, r in hazard_ranges_m.items()}
    binding_hazard = min(per_hazard, key=per_hazard.get)
    v_max_ms = per_hazard[binding_hazard]

    return SpeedEnvelopeResult(
        v_max_ms=v_max_ms,
        v_max_kmh=v_max_ms * 3.6,
        binding_hazard=binding_hazard,
        binding_range_m=hazard_ranges_m[binding_hazard],
        per_hazard_v_max_ms=per_hazard,
    )
