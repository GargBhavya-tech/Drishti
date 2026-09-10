"""
tests/test_checkpoint_first_map.py

Ticket #22 checkpoint test: the pipeline runs end to end and produces a
picture. See eval/checkpoint_first_map.py's docstring for the synthetic-
scene deviation from the literal "one nuScenes sweep" ticket text.
"""

from __future__ import annotations

from eval.checkpoint_first_map import run_checkpoint


def test_checkpoint_runs_end_to_end_and_produces_pngs(tmp_path):
    result = run_checkpoint(out_dir=tmp_path, seed=0)

    assert result["n_points"] > 0
    assert result["touched_cells_l0"] > 0

    assert result["height_png"].exists()
    assert result["height_png"].stat().st_size > 0
    assert result["class_png"].exists()
    assert result["class_png"].stat().st_size > 0


def test_checkpoint_is_deterministic_given_a_seed(tmp_path):
    r1 = run_checkpoint(out_dir=tmp_path / "a", seed=0)
    r2 = run_checkpoint(out_dir=tmp_path / "b", seed=0)
    assert r1["n_points"] == r2["n_points"]
    assert r1["touched_cells_l0"] == r2["touched_cells_l0"]
