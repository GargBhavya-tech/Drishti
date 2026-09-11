"""
tests/test_latency.py

Ticket #55 tests, per the Build Map's own test list: "run 500 frames;
assert the per-stage totals sum to the measured end-to-end time. If
they do not, there is unaccounted time." Plus P50/P95 recorded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.latency import LatencyProfiler, vehicle_with_measured_t_react
from sensor.vehicle_config import load_vehicle_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def vehicle():
    return load_vehicle_config(CONFIGS / "vehicle_ugv.yaml")


def _busy_wait(seconds: float) -> None:
    """A deterministic, CPU-bound delay -- time.sleep() is not
    guaranteed to be precise enough at sub-millisecond scale for this
    reconciliation test to be tight."""
    import time

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        pass


def test_500_frames_stage_sum_reconciles_with_end_to_end_time(vehicle):
    profiler = LatencyProfiler()
    for _ in range(500):
        with profiler.frame():
            with profiler.stage("perception"):
                _busy_wait(0.0002)
            with profiler.stage("scatter"):
                _busy_wait(0.0001)
            with profiler.stage("costmap"):
                _busy_wait(0.0001)

    unaccounted = profiler.unaccounted_time_s()
    # Every bit of frame() time was inside a stage() -- any gap here is
    # profiler/Python bookkeeping overhead, not pipeline work, and
    # should be a tiny fraction of the ~0.0004s x 500 = 0.2s of real
    # work measured.
    assert abs(unaccounted) < 0.05


def test_p50_and_p95_are_recorded_and_p95_is_never_below_p50(vehicle):
    profiler = LatencyProfiler()
    for i in range(20):
        with profiler.frame():
            with profiler.stage("variable"):
                _busy_wait(0.0001 * (1 + i % 5))  # a real spread of durations

    report = profiler.report()
    assert report.n_frames == 20
    assert "variable" in report.per_stage_p50_s
    assert "variable" in report.per_stage_p95_s
    assert report.per_stage_p95_s["variable"] >= report.per_stage_p50_s["variable"]
    assert report.end_to_end_p95_s >= report.end_to_end_p50_s


def test_report_raises_with_no_frames_recorded():
    profiler = LatencyProfiler()
    with pytest.raises(ValueError):
        profiler.report()


def test_a_gap_outside_any_stage_shows_up_as_unaccounted_time():
    profiler = LatencyProfiler()
    with profiler.frame():
        with profiler.stage("only_stage"):
            _busy_wait(0.001)
        _busy_wait(0.02)  # deliberately OUTSIDE any stage() call

    assert profiler.unaccounted_time_s() > 0.01  # the gap must be visible, not silently absorbed


def test_vehicle_with_measured_t_react_replaces_only_that_field(vehicle):
    updated = vehicle_with_measured_t_react(vehicle, measured_p95_s=0.123)
    assert updated.t_react_s == pytest.approx(0.123)
    assert updated.braking_a_ms2 == vehicle.braking_a_ms2  # every other field untouched
    assert updated.max_slope_deg == vehicle.max_slope_deg
    assert vehicle.t_react_s != 0.123  # the ORIGINAL config is not mutated (frozen dataclass)
