"""
tests/test_no_smear.py

Ticket #46 -- no-smear verification. Build Map's own spec: "No new
code." This test wires together modules that already exist
(`observability.observe.carve_frame`, Ticket #34) to check the
INTEGRATION property Bible's demo beat 6 depends on: replay a sequence
containing a walking pedestrian, then query every cell along the path
they walked -- all must be FREE or UNOBSERVED, none OCCUPIED.

This works by construction of Ticket #34's own free-space carving (a
cell's previous state, even OCCUPIED, is overwritten the instant fresh
FREE evidence arrives) -- this test is the "put it in CI" artifact the
Build Map explicitly asks for, not new production logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grid.cell import OBS_OCCUPIED
from grid.clipmap import Clipmap
from observability.observe import carve_frame
from sensor.sensor_model import load_sensor_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def hdl64e():
    return load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")


def test_pedestrian_walking_path_leaves_no_smear_after_they_pass(hdl64e):
    cm = Clipmap(hdl64e, n_levels=4, N=512, c0=0.05)
    cm.scroll_to(0.0, 0.0)

    ego = (0.0, 0.0)
    # A pedestrian walking in a straight line, one cell (world position)
    # occupied per frame; every OTHER beam that frame passes cleanly
    # through the space the pedestrian occupied in EARLIER frames.
    path = [(10.0 + 0.5 * step, 5.0) for step in range(20)]

    for step, ped_pos in enumerate(path):
        beams = []
        # This frame's beam that actually hits the pedestrian.
        beams.append((ped_pos[0], ped_pos[1], True))
        # Beams re-sampling every PREVIOUS position the pedestrian has
        # already vacated -- these come back with no return (the
        # pedestrian is no longer there), carving fresh FREE evidence
        # exactly as a real re-scan of empty space would.
        for prior_pos in path[:step]:
            beams.append((prior_pos[0], prior_pos[1], False))
        carve_frame(cm, ego[0], ego[1], beams)

    # After the full sequence, every position the pedestrian EVER
    # occupied except their FINAL one must read FREE (re-observed empty
    # on a later frame) or UNOBSERVED -- never OCCUPIED (that would be
    # smear: a phantom obstacle left behind at a place nothing is
    # anymore).
    for pos in path[:-1]:
        _, cell = cm.lookup(pos[0], pos[1])
        assert cell.observability != OBS_OCCUPIED, f"smeared obstacle left behind at {pos}"
