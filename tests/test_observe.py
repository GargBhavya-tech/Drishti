"""
tests/test_observe.py

Ticket #34's own test: "a cell behind a wall reads OCCLUDED, not FREE.
A cell outside the sensor FOV reads UNOBSERVED. Assert the height
fields of an UNOBSERVED cell are never read by any consumer (guard in
CellView)." Plus: "all four states are reachable and distinguishable in
a real sweep" and the free-space-carving/precedence behaviour the
module's own docstring commits to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grid.cell import OBS_FREE, OBS_OCCLUDED, OBS_OCCUPIED, OBS_UNOBSERVED
from grid.clipmap import Clipmap
from observability.observe import carve_frame
from sensor.sensor_model import load_sensor_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def hdl64e():
    return load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")


def _fresh(hdl64e) -> Clipmap:
    cm = Clipmap(hdl64e, n_levels=4, N=512, c0=0.05)
    cm.scroll_to(0.0, 0.0)
    return cm


def _obs(cm: Clipmap, x: float, y: float) -> int:
    _, view = cm.lookup(x, y)
    return view.observability


def test_beam_marks_pass_through_free_and_terminal_occupied(hdl64e):
    cm = _fresh(hdl64e)
    # A single beam straight along +x, returning at (5.0, 0.0).
    carve_frame(cm, 0.0, 0.0, [(5.0, 0.0, True)], occlusion_range_m=0.0)

    assert _obs(cm, 2.5, 0.0) == OBS_FREE  # passed through, well before the return
    assert _obs(cm, 5.0, 0.0) == OBS_OCCUPIED  # the return itself


def test_cell_behind_a_termination_reads_occluded_not_free(hdl64e):
    cm = _fresh(hdl64e)
    # A "wall" at (5.0, 0.0); occlusion should extend past it.
    carve_frame(cm, 0.0, 0.0, [(5.0, 0.0, True)], occlusion_range_m=3.0)

    assert _obs(cm, 6.5, 0.0) == OBS_OCCLUDED  # in the wall's shadow
    assert _obs(cm, 6.5, 0.0) != OBS_FREE


def test_cell_outside_fov_reads_unobserved_and_height_fields_are_none(hdl64e):
    cm = _fresh(hdl64e)
    carve_frame(cm, 0.0, 0.0, [(5.0, 0.0, True)], occlusion_range_m=3.0)

    level, view = cm.lookup(0.0, 5.0)  # off to the side -- no beam went there
    assert view.observability == OBS_UNOBSERVED
    # The CellView guard (Ticket #34's own requirement): an unobserved
    # cell's height fields must never carry a numeric value a consumer
    # could mistake for real data.
    assert view.h_min_m is None
    assert view.h_max_m is None
    assert view.h_mean_m is None


def test_no_return_beam_contributes_free_only_no_occupied_or_shadow(hdl64e):
    cm = _fresh(hdl64e)
    # Max-range cutoff, nothing detected -- has_return=False.
    carve_frame(cm, 0.0, 0.0, [(5.0, 0.0, False)], occlusion_range_m=3.0)

    assert _obs(cm, 5.0, 0.0) == OBS_FREE  # no object there, just ran out of range
    assert _obs(cm, 6.5, 0.0) == OBS_UNOBSERVED  # nothing to cast a shadow from


def test_free_space_carving_decays_stale_occupancy(hdl64e):
    cm = _fresh(hdl64e)
    # Frame 1: an object sits at (5.0, 0.0).
    carve_frame(cm, 0.0, 0.0, [(5.0, 0.0, True)], occlusion_range_m=0.0)
    assert _obs(cm, 5.0, 0.0) == OBS_OCCUPIED

    # Frame 2: the object is gone -- a beam now passes straight through
    # the same cell to a farther return.
    carve_frame(cm, 0.0, 0.0, [(9.0, 0.0, True)], occlusion_range_m=0.0)
    assert _obs(cm, 5.0, 0.0) == OBS_FREE  # stale OCCUPIED decayed away
    assert _obs(cm, 9.0, 0.0) == OBS_OCCUPIED


def test_occupied_beats_free_when_beams_disagree_within_one_frame(hdl64e):
    cm = _fresh(hdl64e)
    # Beam A terminates at (5.0, 0.0) -- OCCUPIED there.
    # Beam B passes straight through the same point on its way to a
    # farther return -- FREE evidence for the exact same cell, same
    # frame. OCCUPIED must win (real return beats pass-through).
    carve_frame(
        cm, 0.0, 0.0,
        [(5.0, 0.0, True), (10.0, 0.0, True)],
        occlusion_range_m=0.0,
    )
    assert _obs(cm, 5.0, 0.0) == OBS_OCCUPIED


def test_all_four_states_reachable_and_distinguishable(hdl64e):
    cm = _fresh(hdl64e)
    carve_frame(cm, 0.0, 0.0, [(5.0, 0.0, True)], occlusion_range_m=3.0)

    states = {
        _obs(cm, 2.5, 0.0),   # FREE (pass-through)
        _obs(cm, 5.0, 0.0),   # OCCUPIED (the return)
        _obs(cm, 6.5, 0.0),   # OCCLUDED (behind the return)
        _obs(cm, 0.0, 5.0),   # UNOBSERVED (no beam went there)
    }
    assert states == {OBS_FREE, OBS_OCCUPIED, OBS_OCCLUDED, OBS_UNOBSERVED}
