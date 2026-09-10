"""Ticket #9 tests."""

import re
from pathlib import Path

import pytest

from sensor.vehicle_config import load_vehicle_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "vehicle_ugv.yaml"

# The literals the Build Map's Ticket #9 names explicitly.
WATCHED_LITERALS = ["0.20", "2.50", "4.0"]


def test_config_loads():
    cfg = load_vehicle_config(CONFIG_PATH)
    assert cfg.max_step_height_m == pytest.approx(0.20)
    assert cfg.min_clearance_m == pytest.approx(2.50)
    assert cfg.braking_a_ms2 == pytest.approx(4.0)


def test_no_watched_literals_outside_config_and_tests():
    """Grep the repo for the watched literals outside vehicle_ugv.yaml
    and the tests/ directory. Any hit means a vehicle parameter got
    hardcoded somewhere instead of read from this file -- the whole
    'the map is vehicle-agnostic, the cost map is not' claim depends on
    there being exactly one source for these numbers."""
    skip_dirs = {".git", "__pycache__", ".pytest_cache"}
    hits = []
    for path in ROOT.rglob("*.py"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if "tests" in path.relative_to(ROOT).parts:
            continue
        text = path.read_text(errors="ignore")
        for literal in WATCHED_LITERALS:
            # word-boundary-ish match so e.g. "0.20" doesn't false-positive
            # inside an unrelated longer number like "10.20"
            pattern = r"(?<![\d.])" + re.escape(literal) + r"(?![\d])"
            if re.search(pattern, text):
                hits.append((str(path.relative_to(ROOT)), literal))
    assert not hits, f"Hardcoded vehicle-parameter literals found outside config/tests: {hits}"
