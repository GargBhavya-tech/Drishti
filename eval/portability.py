"""
eval/portability.py

Ticket #59 -- the portability ablation: the SAME five formulas run over
three real sensor configs (HDL-64E/SemanticKITTI reference, HDL-32E/
nuScenes, Ouster OS1-64/RELLIS-3D) with no per-sensor code path at all
-- only a config swap, which this module's own tests assert directly.

The five quantities and their formulas (Bible Part 3 / Part 11,
verified here to reproduce the Bible's own HDL-64E and HDL-32E
headline numbers EXACTLY before trusting the formulas for the Ouster
column):

    5cm level reaches   : c0 / d_theta_rad                     (Ticket #5's own schedule)
    15cm kerb resolved  : 0.15 / d_phi_rad                     (height-resolution range, Bible Part 3)
    2m ditch range      : sqrt(2.0 * h_m / d_phi_rad)          (Bible Part 3's worked example)
    pedestrian r_blind  : sqrt(1.7 * 0.5 / (d_phi_rad * d_theta_rad))  (Bible Part 11 -- an
                          explicit 1.7m x 0.5m PEDESTRIAN silhouette, NOT
                          vehicle_ugv.yaml's own min_object_t_m/w_m, which
                          is a separate, thinner hazard class -- Ticket
                          #43's own "thin_fence_post_presence" scenario
                          uses THAT one instead. Conflating the two would
                          silently answer a different question.)
    safe speed (ditch)  : v_max_for_range(ditch_range, vehicle) * 3.6

Watch out (Build Map's own words): "the point is correct scaling, not
just different numbers... if both [level boundaries and detection
ranges] moved together, something is conflated." Level boundaries
(row 1) are a pure function of d_theta ALONE; every other row is a
function of d_phi (and, for the ditch/pedestrian rows, d_theta too, via
the sqrt(.../ (d_phi*d_theta)) form) -- this module keeps those two
families of formula visibly separate rather than deriving one from the
other.

Watch out #2: "do not present the table as-is if [the Ouster
measurement] hasn't run, since it would silently be reporting a sensor
you didn't test." `configs/sensor_ouster_os1_64.yaml` holds REAL
measured constants (Ticket #6, `eval/measure_ouster_config.py`, cross-
checked against real RELLIS-3D point clouds) -- not the Build Map's own
placeholder table, which this module's tests explicitly do NOT assert
against (see `tests/test_portability.py`'s own module docstring).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from planning.speed_envelope import v_max_for_range
from sensor.sensor_model import SensorConfig, load_sensor_config
from sensor.vehicle_config import VehicleConfig, load_vehicle_config

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

# The pedestrian silhouette used for the r_blind row -- a real person,
# not vehicle_ugv.yaml's own min_object_t_m/w_m (a THINNER hazard
# class this vehicle must also not hit, but a different question).
PEDESTRIAN_HEIGHT_M = 1.7
PEDESTRIAN_WIDTH_M = 0.5

DITCH_WIDTH_M = 2.0
KERB_HEIGHT_M = 0.15
FOVEA_C0_M = 0.05


@dataclass(frozen=True)
class SensorRow:
    sensor_id: str
    level0_reach_m: float
    kerb_range_m: float
    ditch_range_m: float
    pedestrian_r_blind_m: float
    safe_speed_ditch_kmh: float


def compute_row(sm: SensorConfig, vehicle: VehicleConfig) -> SensorRow:
    """The five Ticket #59 quantities for ONE sensor config -- the same
    function, unchanged, for every sensor this project has a real
    config for. Adding a fourth sensor is a `load_sensor_config(new_yaml)`
    call plus one more `compute_row()`, never a new code path here."""
    level0_reach_m = FOVEA_C0_M / sm.d_theta_rad
    kerb_range_m = KERB_HEIGHT_M / sm.d_phi_rad
    ditch_range_m = math.sqrt(DITCH_WIDTH_M * sm.h_m / sm.d_phi_rad)
    pedestrian_r_blind_m = math.sqrt(
        (PEDESTRIAN_HEIGHT_M * PEDESTRIAN_WIDTH_M) / (sm.d_phi_rad * sm.d_theta_rad)
    )
    safe_speed_ditch_kmh = v_max_for_range(ditch_range_m, vehicle) * 3.6

    return SensorRow(
        sensor_id=sm.sensor_id,
        level0_reach_m=level0_reach_m,
        kerb_range_m=kerb_range_m,
        ditch_range_m=ditch_range_m,
        pedestrian_r_blind_m=pedestrian_r_blind_m,
        safe_speed_ditch_kmh=safe_speed_ditch_kmh,
    )


def build_portability_table(vehicle: VehicleConfig, configs_dir: Path = CONFIGS_DIR) -> Dict[str, SensorRow]:
    """All three real configs this project actually has, run through
    the SAME `compute_row()` -- HDL-64E (reference), Ouster OS1-64
    (RELLIS-3D, real measured constants), HDL-32E (nuScenes)."""
    names = {
        "hdl64e": "sensor_hdl64e.yaml",
        "ouster_os1_64": "sensor_ouster_os1_64.yaml",
        "hdl32e": "sensor_hdl32e.yaml",
    }
    return {key: compute_row(load_sensor_config(configs_dir / fname), vehicle) for key, fname in names.items()}


def format_table(rows: Dict[str, SensorRow]) -> str:
    """A plain-text rendering matching the Build Map's own table
    shape -- for a slide or a log, not a UI (no frontend work here)."""
    order = ["hdl64e", "ouster_os1_64", "hdl32e"]
    labels = {
        "hdl64e": "HDL-64E (reference)",
        "ouster_os1_64": "Ouster OS1-64 (RELLIS-3D, measured)",
        "hdl32e": "HDL-32E (nuScenes)",
    }
    lines: List[str] = []
    header = f"{'Quantity':<22}" + "".join(f"{labels[k]:>36}" for k in order if k in rows)
    lines.append(header)
    quantities = [
        ("5cm level reaches", "level0_reach_m", "m"),
        ("15cm kerb", "kerb_range_m", "m"),
        ("2m ditch", "ditch_range_m", "m"),
        ("Pedestrian r_blind", "pedestrian_r_blind_m", "m"),
        ("Safe speed (ditch)", "safe_speed_ditch_kmh", "km/h"),
    ]
    for label, attr, unit in quantities:
        cells = "".join(f"{getattr(rows[k], attr):>32.1f} {unit:<3}" for k in order if k in rows)
        lines.append(f"{label:<22}{cells}")
    return "\n".join(lines)
