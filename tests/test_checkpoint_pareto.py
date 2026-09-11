"""
tests/test_checkpoint_pareto.py

Ticket #58 checkpoint test: the figure exists and the knee/operating
point are real, self-consistent values (not placeholders).
"""

from __future__ import annotations

from eval.checkpoint_pareto import run_checkpoint


def test_checkpoint_produces_a_figure(tmp_path):
    result = run_checkpoint(out_dir=tmp_path)
    assert result["figure_png"].exists()
    assert result["figure_png"].stat().st_size > 0


def test_checkpoint_knee_is_not_an_endpoint_gamma(tmp_path):
    result = run_checkpoint(out_dir=tmp_path)
    gammas = [p.gamma for p in result["points"]]
    assert result["knee_gamma"] != gammas[0]
    assert result["knee_gamma"] != gammas[-1]


def test_checkpoint_is_deterministic(tmp_path):
    r1 = run_checkpoint(out_dir=tmp_path / "a")
    r2 = run_checkpoint(out_dir=tmp_path / "b")
    assert r1["knee_gamma"] == r2["knee_gamma"]
    assert [p.deviation_rms_m for p in r1["points"]] == [p.deviation_rms_m for p in r2["points"]]
