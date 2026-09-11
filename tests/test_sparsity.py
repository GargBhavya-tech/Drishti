"""
tests/test_sparsity.py

Tickets #38-39 tests -- the Sparsity Trap (Claim 3).

Ticket #38: assert against the Bible's own table -- HDL-64E pedestrian
r_blind = 194.8m, pole = 163.7m, fence post = 66.8m; HDL-32E pedestrian
= 79.2m, fence post = 27.2m. Assert N_exp falls as 1/r^2, not 1/r.

Ticket #39: **the machine-checked form of Claim 3** --
test_unknown_past_r_blind: a cell at 90m on nuScenes (past the 27.2m
fence-post r_blind) with zero returns asserts UNKNOWN, not FREE. A cell
at 15m with zero returns asserts FREE.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from observability.sparsity import (
    SparsityVerdict,
    classify_sparsity,
    is_structured,
    n_exp_for_min_object,
    r_blind_for_min_object,
)
from sensor.sensor_model import load_sensor_config, n_expected, r_blind
from sensor.vehicle_config import load_vehicle_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


@pytest.fixture
def hdl64e():
    return load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")


@pytest.fixture
def hdl32e():
    return load_sensor_config(CONFIGS / "sensor_hdl32e.yaml")


@pytest.fixture
def vehicle():
    return load_vehicle_config(CONFIGS / "vehicle_ugv.yaml")


# ---------------------------------------------------------------------------
# Ticket #38 -- the Bible's own r_blind table (via sensor_model.r_blind
# directly, since these use LITERAL (t, w) pairs from Part 11's table,
# not the vehicle's own min-object spec -- test_sensor_model.py doesn't
# cover the pedestrian/pole rows, only fence-post-shaped numbers, so
# they're reasserted here against this ticket's own acceptance table).
# ---------------------------------------------------------------------------


def test_hdl64e_r_blind_table(hdl64e):
    assert r_blind(1.7, 0.5, hdl64e) == pytest.approx(194.8, abs=0.1)  # pedestrian
    assert r_blind(3.0, 0.2, hdl64e) == pytest.approx(163.7, abs=0.1)  # utility pole
    assert r_blind(1.0, 0.1, hdl64e) == pytest.approx(66.8, abs=0.1)  # thin fence post


def test_hdl32e_r_blind_table(hdl32e):
    assert r_blind(1.7, 0.5, hdl32e) == pytest.approx(79.2, abs=0.1)  # pedestrian
    assert r_blind(1.0, 0.1, hdl32e) == pytest.approx(27.2, abs=0.1)  # thin fence post


def test_n_exp_falls_as_inverse_square_not_inverse_range(hdl64e, vehicle):
    n_10 = n_exp_for_min_object(10.0, vehicle, hdl64e)
    n_20 = n_exp_for_min_object(20.0, vehicle, hdl64e)
    ratio = n_10 / n_20
    assert ratio == pytest.approx(4.0, rel=0.01)  # (20/10)^2 = 4, not 2


# ---------------------------------------------------------------------------
# The vehicle-config-driven r_blind -- "the smallest object THIS VEHICLE
# must not hit". vehicle_ugv.yaml's min_object_t_m=1.00/min_object_w_m=0.10
# happen to equal the Bible's own "thin fence post" case, so these must
# match the fence-post rows above exactly -- and via the CONFIG, never a
# literal (t, w) pair.
# ---------------------------------------------------------------------------


def test_r_blind_for_min_object_matches_fence_post_case(hdl64e, hdl32e, vehicle):
    assert r_blind_for_min_object(vehicle, hdl64e) == pytest.approx(66.8, abs=0.1)
    assert r_blind_for_min_object(vehicle, hdl32e) == pytest.approx(27.2, abs=0.1)


def test_sparsity_module_itself_has_no_hardcoded_min_object_literals():
    """observability/sparsity.py must read min_object_t_m/w_m from the
    VehicleConfig it's given, never a literal 1.00/0.10 of its own.
    Scoped to this one module rather than a repo-wide grep: "1.00" and
    "0.10" are common bare floats (unlike the more distinctive "0.20"/
    "4.0" tests/test_vehicle_config.py's own guard watches for) and a
    repo-wide scan false-positives on unrelated constants elsewhere
    (e.g. sensor/calibration.py's own agree_within_m=0.10 default,
    nothing to do with the vehicle's minimum-object spec)."""
    import inspect

    import observability.sparsity as sparsity_module

    src = inspect.getsource(sparsity_module)
    # Strip the module docstring/comments' own prose mentions (e.g. this
    # docstring itself says "1.00"/"0.10" in prose) by checking only
    # actual code lines feeding a computation -- simplest robust check:
    # the module must reference vehicle.min_object_t_m / min_object_w_m
    # and never write a bare 1.00 or 0.10 as a function DEFAULT or
    # computation operand.
    assert "vehicle.min_object_t_m" in src
    assert "vehicle.min_object_w_m" in src
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or '"""' in line:
            continue
        assert not re.search(r"(?<![\d.\w])1\.00(?![\d])", line), f"literal 1.00 in: {line}"
        assert not re.search(r"(?<![\d.\w])0\.10(?![\d])", line), f"literal 0.10 in: {line}"


# ---------------------------------------------------------------------------
# Ticket #39 -- the decision table, and its single most important row.
# ---------------------------------------------------------------------------


def test_unknown_past_r_blind(hdl32e, vehicle):
    """THE machine-checked form of Claim 3."""
    result = classify_sparsity(n_obs=0, r=90.0, vehicle=vehicle, sm=hdl32e)
    assert result.verdict == SparsityVerdict.UNKNOWN
    assert result.verdict != SparsityVerdict.FREE


def test_free_before_r_blind_with_zero_returns(hdl32e, vehicle):
    result = classify_sparsity(n_obs=0, r=15.0, vehicle=vehicle, sm=hdl32e)
    assert result.verdict == SparsityVerdict.FREE


def test_normal_confidence_when_kappa_at_least_one(hdl64e, vehicle):
    # AT r_blind, N_exp == 1 exactly (that's r_blind's own definition) --
    # 2 observed returns there gives kappa=2 >= 1 unambiguously.
    r_at_blind = r_blind_for_min_object(vehicle, hdl64e)
    n_exp = n_exp_for_min_object(r_at_blind, vehicle, hdl64e)
    assert n_exp == pytest.approx(1.0, rel=0.01)
    result = classify_sparsity(n_obs=2, r=r_at_blind, vehicle=vehicle, sm=hdl64e)
    assert result.verdict == SparsityVerdict.NORMAL
    assert result.kappa >= 1.0


def test_single_return_always_sparse_structured_never_proven_unstructured(hdl64e, vehicle):
    """N_obs == 1 must default to SPARSE_STRUCTURED regardless of what
    (nonsensical, for a single point) ranges/ring_indices say."""
    # At r = 0.5 * r_blind, N_exp = (r_blind/r)^2 = 4.0 -- kappa = 1/4 < 1.
    half_blind_r = r_blind_for_min_object(vehicle, hdl64e) * 0.5
    result = classify_sparsity(n_obs=1, r=half_blind_r, vehicle=vehicle, sm=hdl64e)
    assert 0 < result.kappa < 1
    assert result.verdict == SparsityVerdict.SPARSE_STRUCTURED


def test_structured_returns_flagged_sparse_structured_not_free(hdl64e, vehicle):
    # N_exp = 4.0 here (see above); n_obs=3 -> kappa = 3/4 = 0.75 < 1.
    r = r_blind_for_min_object(vehicle, hdl64e) * 0.5
    ranges = np.array([r, r + 0.01, r - 0.01])  # tightly clustered in range
    rings = np.array([10, 11, 12])  # contiguous
    result = classify_sparsity(n_obs=3, r=r, vehicle=vehicle, sm=hdl64e, ranges=ranges, ring_indices=rings)
    assert 0 < result.kappa < 1
    assert result.verdict == SparsityVerdict.SPARSE_STRUCTURED


def test_unstructured_returns_suppressed_as_noise(hdl64e, vehicle):
    r = r_blind_for_min_object(vehicle, hdl64e) * 0.5  # N_exp = 4.0, n_obs=3 -> kappa = 0.75 < 1
    ranges = np.array([r, r + 5.0, r - 6.0])  # widely scattered in range
    rings = np.array([5, 30, 60])  # widely scattered, not contiguous
    result = classify_sparsity(n_obs=3, r=r, vehicle=vehicle, sm=hdl64e, ranges=ranges, ring_indices=rings)
    assert 0 < result.kappa < 1
    assert result.verdict == SparsityVerdict.NOISE_SUPPRESSED


def test_missing_ranges_defaults_to_not_structured_the_cautious_direction(hdl64e, vehicle):
    """Omitting ranges/ring_indices in the ambiguous (0<kappa<1, n_obs>1)
    regime must NOT default to the more permissive SPARSE_STRUCTURED
    verdict by accident -- failing to prove structure is the cautious
    direction (Bible Principle 6), so it defaults to NOISE_SUPPRESSED."""
    r = r_blind_for_min_object(vehicle, hdl64e) * 0.5  # N_exp = 4.0, n_obs=3 -> kappa = 0.75 < 1
    result = classify_sparsity(n_obs=3, r=r, vehicle=vehicle, sm=hdl64e)
    assert 0 < result.kappa < 1
    assert result.verdict == SparsityVerdict.NOISE_SUPPRESSED


# ---------------------------------------------------------------------------
# is_structured, as a pure function
# ---------------------------------------------------------------------------


def test_is_structured_true_for_tight_contiguous_returns(vehicle):
    ranges = np.array([20.0, 20.02, 19.98])
    rings = np.array([7, 8, 9])
    assert is_structured(ranges, rings, vehicle) is True


def test_is_structured_false_for_wide_range_spread(vehicle):
    ranges = np.array([20.0, 25.0, 15.0])
    rings = np.array([7, 8, 9])
    assert is_structured(ranges, rings, vehicle) is False


def test_is_structured_false_for_noncontiguous_rings(vehicle):
    ranges = np.array([20.0, 20.01, 19.99])
    rings = np.array([7, 20, 40])
    assert is_structured(ranges, rings, vehicle) is False


def test_is_structured_false_for_empty_returns(vehicle):
    assert is_structured(np.array([]), np.array([]), vehicle) is False
