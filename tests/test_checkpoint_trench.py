"""
tests/test_checkpoint_trench.py

Ticket #37 checkpoint test: "render a scene with a synthetic ditch; the
map marks it, a plain 2D occupancy grid built from the same sweep does
not." Mirrors tests/test_checkpoint_first_map.py's pattern (Ticket #22).
"""

from __future__ import annotations

from eval.checkpoint_trench import run_checkpoint


def test_checkpoint_detects_the_ditch_and_produces_a_figure(tmp_path):
    result = run_checkpoint(out_dir=tmp_path)

    assert result["n_points"] > 0
    assert result["n_ditch_cells_omitted"] > 0  # the scene actually has a gap to find
    assert result["drishti_detected_the_ditch"] is True
    assert result["negative_obstacle_cells_promoted"] > 0

    assert result["figure_png"].exists()
    assert result["figure_png"].stat().st_size > 0


def test_checkpoint_is_deterministic(tmp_path):
    r1 = run_checkpoint(out_dir=tmp_path / "a")
    r2 = run_checkpoint(out_dir=tmp_path / "b")
    assert r1["n_points"] == r2["n_points"]
    assert r1["negative_obstacle_cells_promoted"] == r2["negative_obstacle_cells_promoted"]
