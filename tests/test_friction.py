"""
tests/test_friction.py

The Semantic Friction Governor (planning/friction.py). Covers: the
CLASS_TO_MU table's own internal consistency (DRIVABLE is the best
tier, nothing exceeds MU_DRY_REFERENCE), binding_mu's "worst class
wins" reduction and its empty-input no-op, friction_adjusted_vehicle's
linear scaling and immutability of the input, and
friction_adjusted_speed_envelope's end-to-end wrapper behaviour against
the REAL vehicle_ugv.yaml config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from perception.taxonomy import DrishtiClass
from planning.friction import (
    CLASS_TO_MU,
    MU_DRY_REFERENCE,
    binding_mu,
    friction_adjusted_speed_envelope,
    friction_adjusted_vehicle,
    mu_for_class,
)
from sensor.vehicle_config import VehicleConfig, load_vehicle_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def vehicle() -> VehicleConfig:
    return load_vehicle_config(CONFIGS / "vehicle_ugv.yaml")


def test_class_to_mu_never_exceeds_dry_reference():
    # DRIVABLE is declared to equal MU_DRY_REFERENCE exactly (it IS the
    # dry-ground assumption); nothing else may be assigned a friction
    # coefficient better than that, or "worst class along the path"
    # would stop being conservative.
    for cls, mu in CLASS_TO_MU.items():
        assert mu <= MU_DRY_REFERENCE, f"{cls.name} has mu={mu} > MU_DRY_REFERENCE"


def test_mu_for_class_unknown_fallback_is_not_best_case():
    fallback = mu_for_class(DrishtiClass.UNKNOWN)
    # A class this table has never heard of (can't construct an invalid
    # DrishtiClass, so this documents the CONTRACT: the fallback used
    # inside mu_for_class for anything missing from CLASS_TO_MU is
    # UNKNOWN's own value, never the dry-ground best case).
    assert fallback == CLASS_TO_MU[DrishtiClass.UNKNOWN]
    assert fallback < MU_DRY_REFERENCE


def test_binding_mu_empty_path_is_a_no_op():
    mu, binding_class = binding_mu([])
    assert mu == MU_DRY_REFERENCE
    assert binding_class is None


def test_binding_mu_picks_the_worst_class_present():
    classes = [DrishtiClass.DRIVABLE, DrishtiClass.DRIVABLE, DrishtiClass.CAUTION, DrishtiClass.VEGETATION]
    mu, binding_class = binding_mu(classes)
    assert binding_class == DrishtiClass.CAUTION  # lowest mu among these four
    assert mu == CLASS_TO_MU[DrishtiClass.CAUTION]


def test_binding_mu_all_drivable_keeps_dry_reference():
    mu, binding_class = binding_mu([DrishtiClass.DRIVABLE, DrishtiClass.DRIVABLE])
    assert mu == MU_DRY_REFERENCE
    assert binding_class is None  # nothing was ever WORSE than the reference


def test_friction_adjusted_vehicle_scales_linearly_and_does_not_mutate_input(vehicle):
    half_mu = MU_DRY_REFERENCE / 2.0
    derated = friction_adjusted_vehicle(vehicle, half_mu)
    assert derated.braking_a_ms2 == pytest.approx(vehicle.braking_a_ms2 * 0.5)
    # Every other field is untouched.
    assert derated.t_react_s == vehicle.t_react_s
    assert derated.width_m == vehicle.width_m
    # The original is a frozen dataclass and dataclasses.replace() never
    # mutates it -- assert the ORIGINAL still reports its own value.
    assert vehicle.braking_a_ms2 == 4.0


def test_friction_adjusted_vehicle_at_dry_reference_is_unchanged(vehicle):
    derated = friction_adjusted_vehicle(vehicle, MU_DRY_REFERENCE)
    assert derated.braking_a_ms2 == pytest.approx(vehicle.braking_a_ms2)


def test_friction_adjusted_vehicle_rejects_nonpositive_mu(vehicle):
    with pytest.raises(ValueError):
        friction_adjusted_vehicle(vehicle, 0.0)
    with pytest.raises(ValueError):
        friction_adjusted_vehicle(vehicle, -0.1)


def test_friction_adjusted_speed_envelope_matches_plain_envelope_on_dry_path(vehicle):
    from planning.speed_envelope import speed_envelope

    hazards = {"2m_ditch": 21.6}
    plain = speed_envelope(hazards, vehicle)
    derated_result, mu, binding_class = friction_adjusted_speed_envelope(hazards, vehicle, [DrishtiClass.DRIVABLE])
    assert mu == MU_DRY_REFERENCE
    assert binding_class is None
    assert derated_result.v_max_ms == pytest.approx(plain.v_max_ms)


def test_friction_adjusted_speed_envelope_lowers_v_max_on_mud(vehicle):
    hazards = {"2m_ditch": 21.6}
    dry_result, dry_mu, _ = friction_adjusted_speed_envelope(hazards, vehicle, [DrishtiClass.DRIVABLE])
    muddy_result, muddy_mu, binding_class = friction_adjusted_speed_envelope(
        hazards, vehicle, [DrishtiClass.DRIVABLE, DrishtiClass.CAUTION]
    )
    assert binding_class == DrishtiClass.CAUTION
    assert muddy_mu < dry_mu
    # Lower braking a_max -> stopping distance grows for the same range,
    # so the range-limited v_max the sensor can still justify must fall.
    assert muddy_result.v_max_ms < dry_result.v_max_ms


def test_friction_adjusted_speed_envelope_never_exceeds_dry_baseline(vehicle):
    # "Advisory, only ever adds a constraint" -- friction deratation can
    # only ever LOWER v_max relative to the dry-reference baseline,
    # never raise it above what speed_envelope() alone would allow.
    hazards = {"2m_ditch": 21.6, "5cm_cable": 6.7}
    from planning.speed_envelope import speed_envelope

    baseline = speed_envelope(hazards, vehicle)
    for cls in DrishtiClass:
        result, _, _ = friction_adjusted_speed_envelope(hazards, vehicle, [cls])
        assert result.v_max_ms <= baseline.v_max_ms + 1e-9
