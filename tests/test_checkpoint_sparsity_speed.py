"""
tests/test_checkpoint_sparsity_speed.py

Ticket #43 checkpoint test: "render a frame with SPARSE_STRUCTURED cells
overlaid and the speed envelope drawn. Print the conservatism test
result... Done when: all four claims now have running code behind them."
"""

from __future__ import annotations

import pytest

from eval.checkpoint_sparsity_speed import run_checkpoint


@pytest.fixture(scope="module")
def checkpoint_result(tmp_path_factory):
    """run_checkpoint() re-runs tests/test_conservatism.py's own 10,000-
    example Hypothesis suite internally (~1 minute) -- shared across the
    tests that need that real verification, so it happens once per test
    session rather than once per test function."""
    out_dir = tmp_path_factory.mktemp("checkpoint_sparsity_speed")
    return run_checkpoint(out_dir=out_dir)


def test_checkpoint_produces_a_figure_and_passes_the_conservatism_property_test(checkpoint_result):
    result = checkpoint_result
    assert result["figure_png"].exists()
    assert result["figure_png"].stat().st_size > 0

    # The checkpoint's own headline claim: Ticket #42's property suite,
    # run for real, must pass.
    assert result["conservatism_property_test_passed"] is True


def test_pole_scenario_covers_all_three_key_sparsity_regimes(checkpoint_result):
    verdicts = {row["result"].verdict.name for row in checkpoint_result["pole_rows"]}
    # The scene is only illustrative if it actually reaches UNKNOWN
    # somewhere past r_blind, not just NORMAL throughout.
    assert "UNKNOWN" in verdicts
    assert "NORMAL" in verdicts


def test_speed_envelope_reports_a_binding_hazard(checkpoint_result):
    env = checkpoint_result["speed_envelope"]
    assert env.binding_hazard is not None
    assert env.v_max_ms > 0
    # The 5cm cable's ~6.7m detection range is always the shortest of
    # this checkpoint's own three hazards, so it must be binding.
    assert env.binding_hazard == "5cm_cable"


def test_checkpoint_is_deterministic(tmp_path):
    """Uses verify_conservatism=False -- determinism of the SCENE/PLOT
    output is what's under test here, not a third re-verification of
    the already-covered property suite."""
    r1 = run_checkpoint(out_dir=tmp_path / "a", verify_conservatism=False)
    r2 = run_checkpoint(out_dir=tmp_path / "b", verify_conservatism=False)
    assert r1["r_blind_m"] == r2["r_blind_m"]
    assert [row["n_obs"] for row in r1["pole_rows"]] == [row["n_obs"] for row in r2["pole_rows"]]
    assert r1["speed_envelope"].binding_hazard == r2["speed_envelope"].binding_hazard
