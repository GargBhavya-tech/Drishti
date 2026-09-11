"""
tests/test_measure_ouster_config.py

Regression test for eval/measure_ouster_config.py -- Ticket #6, completed
for RELLIS-3D's Ouster OS1-64 sensor against real local data. Skipped if
the dataset isn't present locally (matching every other real-data test's
pattern in this repo, e.g. tests/test_rellis_loader.py).

Uses a light sample (few frames/sequence) to stay fast in CI -- this is
a regression guard that the measurement pipeline still runs and lands in
sane bounds, not a re-derivation of the exact numbers written into
configs/sensor_ouster_os1_64.yaml (that comparison is the next test).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from eval.measure_ouster_config import RELLIS_ROOT, run
from sensor.sensor_model import load_sensor_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SEQUENCES_AVAILABLE = [s for s in ("00000", "00001", "00002", "00003", "00004") if (RELLIS_ROOT / s).exists()]


@pytest.mark.skipif(not SEQUENCES_AVAILABLE, reason="no local RELLIS-3D sequences present")
def test_measurement_pipeline_produces_sane_values():
    result = run(n_per_sequence=3)

    # d_theta is an exact firmware fact (point-count / 64), not a fit --
    # must reproduce to high precision regardless of which frames sampled.
    assert result["d_theta_deg"] == pytest.approx(360.0 / 2048.0, rel=1e-6)
    assert result["W_columns_per_rev"] == 2048

    # phi_max/phi_min/d_phi: real physical beam angles, should land close
    # to this build's own measured table regardless of exactly which
    # frames were sampled (small differences in top-beam averaging are
    # expected since those beams return real data least often).
    assert 15.0 < result["phi_max_deg"] < 19.0
    assert 0.3 < result["d_phi_deg"] < 0.8
    assert result["n_beams_with_data"] >= 55  # most beams should have enough data even on a light sample

    # h_m: a plausible ground-vehicle mount height, consistent across
    # whichever sequences got sampled.
    assert 0.5 < result["h_m_median"] < 2.0
    for seq, h in result["per_sequence_median"].items():
        assert 0.5 < h < 2.0, f"sequence {seq}'s h_m median {h} is not a plausible mount height"

    assert result["gate_plot"].exists()
    assert result["gate_plot"].stat().st_size > 0
    assert len(result["gate_bin_results"]) == 14  # 5m bins, 0-70m


@pytest.mark.skipif(not SEQUENCES_AVAILABLE, reason="no local RELLIS-3D sequences present")
def test_committed_config_matches_the_measurement_method():
    """The values actually written into configs/sensor_ouster_os1_64.yaml
    must be reproducible from this measurement pipeline, within the
    natural spread of which frames get sampled -- if this ever drifts
    far apart, either the config was hand-edited inconsistently with the
    method that's supposed to produce it, or the measurement code
    regressed. Uses a bigger sample than the sanity test above since this
    is specifically checking numeric agreement, not just plausibility."""
    sm = load_sensor_config(CONFIGS / "sensor_ouster_os1_64.yaml")
    result = run(n_per_sequence=10)

    assert sm.d_theta_rad == pytest.approx(result["d_theta_rad"], rel=1e-6)
    assert sm.phi_max_rad == pytest.approx(result["phi_max_rad"], abs=math.radians(0.5))
    assert sm.d_phi_rad == pytest.approx(result["d_phi_rad"], abs=math.radians(0.05))
    # h_m has real cross-sequence/cross-frame spread (see the yaml's own
    # comment) -- a looser absolute tolerance than the angular constants.
    assert sm.h_m == pytest.approx(result["h_m_median"], abs=0.1)
