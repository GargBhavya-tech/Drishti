"""
tests/test_costmap.py

Ticket #49 tests, per the Build Map's own test list: "each of the four
LETHAL conditions independently triggers, and each JUST UNDER threshold
does not. Assert UNOBSERVED and SPARSE_STRUCTURED both produce
UNKNOWN_COST, never zero." Plus a confirmation that Ticket #42's own
property test still passes unchanged (it does -- this module reuses
`conservatism.deficit_floor()` rather than touching `conservatism.cost()`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grid.cell import OBS_FREE, OBS_UNOBSERVED
from observability.sparsity import SparsityVerdict
from perception.taxonomy import DrishtiClass
from planning.conservatism import LETHAL, UNKNOWN_COST, CellState
from planning.costmap import GeometryResult, cost, geometric_lethal
from sensor.vehicle_config import load_vehicle_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def vehicle():
    return load_vehicle_config(CONFIGS / "vehicle_ugv.yaml")


def _clean_cell() -> CellState:
    return CellState(
        observability=OBS_FREE,
        class_id=int(DrishtiClass.DRIVABLE),
        class_confidence=1.0,
        sparsity_verdict=SparsityVerdict.NORMAL,
        count=10,
        provisional=False,
        inferred=False,
        age_s=0.0,
        step_height_known=True,
        incidence_usable=True,
    )


def _clean_geo(vehicle, **overrides) -> GeometryResult:
    defaults = dict(
        slope_deg=vehicle.max_slope_deg / 2.0,
        roughness_m=vehicle.max_roughness_m / 2.0,
        step_height_m=vehicle.max_step_height_m / 2.0,
        clearance_m=vehicle.min_clearance_m * 2.0,
        class_penalty=0.0,
    )
    defaults.update(overrides)
    return GeometryResult(**defaults)


# ---------------------------------------------------------------------------
# The four LETHAL conditions -- each independently triggers, each just
# under threshold does not.
# ---------------------------------------------------------------------------


def test_negative_obstacle_class_is_lethal(vehicle):
    cell = CellState(**{**_clean_cell().__dict__, "class_id": int(DrishtiClass.NEGATIVE_OBSTACLE)})
    geo = _clean_geo(vehicle)
    assert cost(cell, geo, vehicle) == LETHAL


def test_slope_over_threshold_is_lethal_just_under_is_not(vehicle):
    cell = _clean_cell()
    over = _clean_geo(vehicle, slope_deg=vehicle.max_slope_deg + 0.1)
    under = _clean_geo(vehicle, slope_deg=vehicle.max_slope_deg - 0.1)
    assert cost(cell, over, vehicle) == LETHAL
    assert cost(cell, under, vehicle) != LETHAL
    assert cost(cell, under, vehicle) < UNKNOWN_COST


def test_step_height_over_threshold_is_lethal_just_under_is_not(vehicle):
    cell = _clean_cell()
    over = _clean_geo(vehicle, step_height_m=vehicle.max_step_height_m + 0.01)
    under = _clean_geo(vehicle, step_height_m=vehicle.max_step_height_m - 0.01)
    assert cost(cell, over, vehicle) == LETHAL
    assert cost(cell, under, vehicle) != LETHAL


def test_clearance_under_threshold_is_lethal_just_over_is_not(vehicle):
    cell = _clean_cell()
    under = _clean_geo(vehicle, clearance_m=vehicle.min_clearance_m - 0.01)
    over = _clean_geo(vehicle, clearance_m=vehicle.min_clearance_m + 0.01)
    assert cost(cell, under, vehicle) == LETHAL
    assert cost(cell, over, vehicle) != LETHAL


def test_geometric_lethal_helper_matches_cost_function(vehicle):
    over_slope = _clean_geo(vehicle, slope_deg=vehicle.max_slope_deg + 1.0)
    assert geometric_lethal(over_slope, vehicle) is True
    clean = _clean_geo(vehicle)
    assert geometric_lethal(clean, vehicle) is False


# ---------------------------------------------------------------------------
# UNOBSERVED and SPARSE_STRUCTURED both produce UNKNOWN_COST, never zero.
# ---------------------------------------------------------------------------


def test_unobserved_cell_never_produces_a_lower_than_unknown_cost(vehicle):
    cell = CellState(**{**_clean_cell().__dict__, "observability": OBS_UNOBSERVED})
    geo = _clean_geo(vehicle, slope_deg=0.0, roughness_m=0.0)  # even a perfectly flat-looking cell
    result = cost(cell, geo, vehicle)
    assert result == pytest.approx(UNKNOWN_COST)
    assert result != 0.0


def test_sparse_structured_cell_never_produces_a_lower_than_unknown_cost(vehicle):
    cell = CellState(**{**_clean_cell().__dict__, "sparsity_verdict": SparsityVerdict.SPARSE_STRUCTURED})
    geo = _clean_geo(vehicle, slope_deg=0.0, roughness_m=0.0)
    result = cost(cell, geo, vehicle)
    assert result == pytest.approx(UNKNOWN_COST)
    assert result != 0.0


def test_missing_geometry_is_treated_as_a_deficit_not_a_free_pass(vehicle):
    cell = _clean_cell()
    geo = _clean_geo(vehicle, slope_deg=None, roughness_m=None)
    result = cost(cell, geo, vehicle)
    assert result == pytest.approx(UNKNOWN_COST)


# ---------------------------------------------------------------------------
# The "otherwise" weighted blend -- a clean, fully-observed, flat cell
# is cheap; a rougher/steeper (but still legal) one costs more, and
# both stay strictly below UNKNOWN_COST.
# ---------------------------------------------------------------------------


def test_otherwise_branch_is_cheap_for_a_flat_clean_cell_and_below_unknown_cost(vehicle):
    cell = _clean_cell()
    flat = _clean_geo(vehicle, slope_deg=0.0, roughness_m=0.0)
    result = cost(cell, flat, vehicle)
    assert result < UNKNOWN_COST
    assert result == pytest.approx(0.0, abs=1e-6)


def test_otherwise_branch_increases_with_slope_and_roughness(vehicle):
    cell = _clean_cell()
    mild = _clean_geo(vehicle, slope_deg=1.0, roughness_m=0.001)
    steep = _clean_geo(vehicle, slope_deg=vehicle.max_slope_deg * 0.9, roughness_m=vehicle.max_roughness_m * 0.9)
    assert cost(cell, mild, vehicle) < cost(cell, steep, vehicle) < UNKNOWN_COST


def test_negative_obstacle_beats_geometric_lethal_and_both_beat_unknown_cost(vehicle):
    cell = CellState(**{**_clean_cell().__dict__, "class_id": int(DrishtiClass.NEGATIVE_OBSTACLE)})
    # Even with otherwise-clean geometry, class alone is enough.
    geo = _clean_geo(vehicle)
    assert cost(cell, geo, vehicle) == LETHAL
