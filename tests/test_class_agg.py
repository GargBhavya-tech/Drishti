"""
tests/test_class_agg.py

Ticket #19 tests: mode (not mean) aggregation of per-point class labels,
and the class_conf byte's encode/decode round-trip.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from grid.addressing import flat_index, global_to_storage, global_to_world
from grid.cell import decode_class_conf, encode_class_conf
from grid.clipmap import Clipmap
from grid.scatter import scatter_class
from sensor.sensor_model import load_sensor_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def hdl64e():
    return load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")


def _fresh(hdl64e) -> Clipmap:
    cm = Clipmap(hdl64e, n_levels=4, N=512, c0=0.05)
    cm.scroll_to(0.0, 0.0)
    return cm


def test_encode_decode_class_conf_round_trip():
    for class_id in (0, 1, 4, 9):
        for frac in (0.0, 0.3, 0.5, 1.0):
            byte = encode_class_conf(class_id, frac)
            decoded_class, decoded_frac = decode_class_conf(int(byte))
            assert decoded_class == class_id
            assert decoded_frac == pytest.approx(frac, abs=1.0 / 15)


def test_seven_class1_three_class4_reports_class1_with_point_three_runner_up(hdl64e):
    cm = _fresh(hdl64e)
    level = 0
    c_l = cm.levels[level].cell_size_m
    gi, gj = cm.origin_i[level] + 20, cm.origin_j[level] + 20
    x0, y0 = global_to_world(gi, gj, c_l)

    n_total = 10
    xs = x0 + c_l * np.linspace(0.05, 0.95, n_total)
    ys = np.full(n_total, y0 + c_l * 0.5)
    xyz = np.stack([xs, ys, np.zeros(n_total)], axis=1)
    class_ids = np.array([1] * 7 + [4] * 3, dtype=np.int64)

    scatter_class(cm, xyz, class_ids)

    si, sj = global_to_storage(gi, gj, cm.N)
    flat = flat_index(si, sj, cm.N)
    winning_class, runner_up_fraction = decode_class_conf(int(cm.class_conf[level, flat]))

    assert winning_class == 1
    assert runner_up_fraction == pytest.approx(0.3, abs=1.0 / 15)


def test_untouched_cell_stays_zero(hdl64e):
    cm = _fresh(hdl64e)
    level = 0
    gi, gj = cm.origin_i[level] + 300, cm.origin_j[level] + 300
    si, sj = global_to_storage(gi, gj, cm.N)
    flat = flat_index(si, sj, cm.N)
    assert cm.class_conf[level, flat] == 0


def test_no_mean_or_average_anywhere_in_the_class_aggregation_path():
    """Ticket #19's own required assertion: class IDs must never be
    averaged (class 2 and class 6 would average to the unrelated class
    4). A structural guard on the source's CODE (not its docstrings,
    which legitimately explain this rule in prose using the word
    "average")."""
    import ast

    import grid.cell as cell_module
    import grid.scatter as scatter_module

    for fn in (scatter_module.scatter_class, cell_module.encode_class_conf_batch, cell_module.encode_class_conf):
        src = inspect.getsource(fn)
        tree = ast.parse(src)
        func_def = tree.body[0]
        # Drop the docstring by its LINE RANGE, not by string-matching its
        # text -- ast.get_docstring() returns a cleaned/dedented string
        # that won't match the raw indented source verbatim. lineno/
        # end_lineno are available since Python 3.8 (this server-side
        # test needs to run there too; ast.unparse does not).
        lines = src.splitlines()
        if ast.get_docstring(func_def) is not None:
            first_stmt = func_def.body[0]
            lines = lines[: first_stmt.lineno - 1] + lines[first_stmt.end_lineno :]
        code_only = "\n".join(lines)
        assert ".mean(" not in code_only
        assert "np.mean" not in code_only
        assert "torch.mean" not in code_only
        assert "average" not in code_only.lower()
