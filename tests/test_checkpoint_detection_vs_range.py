"""
tests/test_checkpoint_detection_vs_range.py

Ticket #60 checkpoint test: the figure exists and both curves' measured
ranges track their predictions within the same ~20% tolerance the
Build Map asks for.
"""

from __future__ import annotations

import pytest

from eval.checkpoint_detection_vs_range import run_checkpoint


def test_checkpoint_produces_a_figure(tmp_path):
    result = run_checkpoint(out_dir=tmp_path)
    assert result["figure_png"].exists()
    assert result["figure_png"].stat().st_size > 0


def test_checkpoint_kerb_and_ditch_both_track_prediction(tmp_path):
    result = run_checkpoint(out_dir=tmp_path)
    for curve in (result["kerb_curve"], result["ditch_curve"]):
        assert curve.measured_range_m == pytest.approx(curve.predicted_range_m, rel=0.2)
