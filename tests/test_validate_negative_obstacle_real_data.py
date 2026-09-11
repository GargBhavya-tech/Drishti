"""
tests/test_validate_negative_obstacle_real_data.py

Regression tests for eval/validate_negative_obstacle_real_data.py --
Phase 4's real-data validation pass. Skipped if the local RELLIS-3D
dataset isn't present.

Deliberately does NOT assert "zero promotions" in the full-spec mode --
this validation's own finding (see that module's docstring) is that
zero promotions is NOT what real data produces without carving (Tickets
#33/34) running first, and asserting otherwise would just be a test
that lies about what was actually measured. What IS asserted: the
ground-plane fit itself stays well-behaved on real terrain, and the
measured-only diagnostic's flagged rate stays within the bounds this
build actually observed -- a regression guard, not a false "it's
perfect" claim.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.validate_negative_obstacle_real_data import run_validation

REPO_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_DIR = REPO_ROOT / "data" / "rellis" / "00004"
SENSOR_CONFIG = REPO_ROOT / "configs" / "sensor_ouster_os1_64.yaml"

pytestmark = pytest.mark.skipif(not SEQUENCE_DIR.exists(), reason="local RELLIS-3D sequence not present")


def test_runs_end_to_end_on_real_consecutive_frames():
    result = run_validation(SEQUENCE_DIR, list(range(1000, 1005)), SENSOR_CONFIG, n_azimuth_bins=512)
    assert result["n_frames"] == 5
    assert len(result["full_spec"]["raw_flagged_per_frame"]) == 5
    assert len(result["measured_only_diagnostic"]["raw_flagged_per_frame"]) == 5


def test_ground_plane_fit_stays_well_behaved_on_real_terrain():
    """The local line fit's own quality, independent of the detector
    built on top of it -- this is the piece that must NOT be the
    explanation for a high flagged rate (and this build confirmed it
    isn't: real residuals stay small even where the detector fires a
    lot)."""
    result = run_validation(SEQUENCE_DIR, list(range(1000, 1010)), SENSOR_CONFIG, n_azimuth_bins=512)
    assert result["ground_fit_residual_mean_m"] is not None
    assert result["ground_fit_residual_mean_m"] < 0.5
    assert result["ground_fit_residual_max_m"] < 5.0


def test_measured_only_diagnostic_flags_far_fewer_cells_than_full_spec():
    """The core finding, as a regression guard: excluding the no-return
    substitution (a proxy for what carving would filter) must flag
    substantially fewer cells than the full-spec mode -- if this ever
    stops being true, either the no-return branch stopped dominating
    (worth knowing) or something else changed underneath both modes."""
    result = run_validation(SEQUENCE_DIR, list(range(1000, 1005)), SENSOR_CONFIG, n_azimuth_bins=512)
    fs_mean = result["full_spec"]["raw_flagged_mean"]
    mo_mean = result["measured_only_diagnostic"]["raw_flagged_mean"]
    assert mo_mean < fs_mean
    assert mo_mean < 0.5 * fs_mean  # not just lower -- substantially lower, matching this build's ~15x gap


def test_full_spec_promotions_with_a_traceable_source_point_are_rare():
    """Almost every full-spec promotion has NO traceable source point
    (it came from a no-return cell) -- this build measured 170/13000.
    A regression guard that this stays a small minority, not the
    majority (which would suggest the no-return/measured-only split
    itself is broken)."""
    result = run_validation(SEQUENCE_DIR, list(range(1000, 1005)), SENSOR_CONFIG, n_azimuth_bins=512)
    fs = result["full_spec"]
    if fs["n_promoted_total"] > 0:
        traceable_fraction = fs["n_promoted_vegetation_or_obstacle_source"] / fs["n_promoted_total"]
        assert traceable_fraction < 0.5
