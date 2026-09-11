"""
eval/latency.py

Ticket #55 -- per-stage latency instrumentation across the full
pipeline, reported as P50/P95, end to end from scan-complete to
map-ready. "Latency, not FPS" (Build Map's own words): the sensor's own
frame rate (10-20 Hz) already caps any FPS number at something
meaningless above it, so this module never computes or reports one.

Watch out (Build Map's own words): time GPU stages with CUDA events,
not wall-clock around an async kernel launch -- a wall-clock stopwatch
around a non-blocking CUDA call measures "how long it took to QUEUE the
kernel", not how long it ran, and everything downstream looks fast
right up until a real bottleneck appears. `LatencyProfiler.stage()`
uses wall-clock (`time.perf_counter`) for CPU-only stages;
`cuda_stage()` uses `torch.cuda.Event` timing for GPU stages when CUDA
is actually available, falling back to wall-clock otherwise (this
project's own dev/CI environment has no GPU -- verified directly,
`torch.cuda.is_available()` is False here -- so `cuda_stage` must
degrade gracefully rather than raise).

The reconciliation test this ticket asks for ("run 500 frames; assert
the per-stage totals sum to the measured end-to-end time... if they do
not, there is unaccounted time") is checked here against the SAME
recorded durations the P50/P95 report is built from, not a second,
independently-computed estimate that could quietly drift from what was
actually measured.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Dict, List

import numpy as np

from sensor.vehicle_config import VehicleConfig

try:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:  # pragma: no cover -- torch is a hard dependency elsewhere in this project
    torch = None  # type: ignore[assignment]
    _CUDA_AVAILABLE = False


@dataclass(frozen=True)
class LatencyReport:
    per_stage_p50_s: Dict[str, float]
    per_stage_p95_s: Dict[str, float]
    end_to_end_p50_s: float
    end_to_end_p95_s: float
    n_frames: int


class LatencyProfiler:
    """Records per-stage durations across many frames: one call to the
    `frame()` context manager brackets each frame, `stage(name)` (or
    `cuda_stage(name)`) brackets each named stage inside it.
    `report()` computes P50/P95 per stage AND end-to-end from those
    same recordings.
    """

    def __init__(self) -> None:
        self._stage_durations: Dict[str, List[float]] = defaultdict(list)
        self._frame_totals: List[float] = []
        self._current_frame_stage_total: float = 0.0

    @contextmanager
    def frame(self):
        self._current_frame_stage_total = 0.0
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            self._frame_totals.append(time.perf_counter() - t0)

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._stage_durations[name].append(dt)
            self._current_frame_stage_total += dt

    @contextmanager
    def cuda_stage(self, name: str):
        """CUDA-event-timed stage -- see module docstring. Falls back
        to `stage()`'s wall-clock timing when no CUDA device is
        available, rather than raising, so GPU-aware pipeline code can
        still be exercised on a CPU-only machine."""
        if not _CUDA_AVAILABLE:
            with self.stage(name):
                yield
            return
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        try:
            yield
        finally:
            end_evt.record()
            torch.cuda.synchronize()
            dt = start_evt.elapsed_time(end_evt) / 1000.0  # ms -> s
            self._stage_durations[name].append(dt)
            self._current_frame_stage_total += dt

    def report(self) -> LatencyReport:
        if not self._frame_totals:
            raise ValueError("no frames recorded -- call profiler.frame() at least once")
        per_stage_p50 = {name: float(np.percentile(d, 50)) for name, d in self._stage_durations.items()}
        per_stage_p95 = {name: float(np.percentile(d, 95)) for name, d in self._stage_durations.items()}
        totals = np.asarray(self._frame_totals)
        return LatencyReport(
            per_stage_p50_s=per_stage_p50,
            per_stage_p95_s=per_stage_p95,
            end_to_end_p50_s=float(np.percentile(totals, 50)),
            end_to_end_p95_s=float(np.percentile(totals, 95)),
            n_frames=len(self._frame_totals),
        )

    def total_measured_stage_time_s(self) -> float:
        """The SUM of every recorded stage duration, across every
        stage and every frame -- the "per-stage totals" the Build Map's
        own test reconciles against the end-to-end total."""
        return sum(sum(d) for d in self._stage_durations.values())

    def total_measured_end_to_end_s(self) -> float:
        return sum(self._frame_totals)

    def unaccounted_time_s(self) -> float:
        """Ticket #55's own diagnostic: end-to-end minus the sum of
        every stage. A real gap here means work happened inside
        `frame()` but outside any `stage()`/`cuda_stage()` call --
        Build Map's own words: "usually a synchronisation stall, and
        finding it is worth more than optimising any stage." """
        return self.total_measured_end_to_end_s() - self.total_measured_stage_time_s()


def vehicle_with_measured_t_react(vehicle: VehicleConfig, measured_p95_s: float) -> VehicleConfig:
    """Ticket #55's own explicit instruction: "feed measured P95 into
    t_react for #40". Returns a NEW VehicleConfig (frozen dataclass,
    `dataclasses.replace`) with `t_react_s` set to the measured
    end-to-end P95 latency, rather than mutating the caller's config or
    hand-editing `vehicle_ugv.yaml` -- so `planning.speed_envelope`'s
    reaction-time budget can reflect what the pipeline actually
    measured once real latency numbers exist, without this module
    reaching into config-loading at all.
    """
    return replace(vehicle, t_react_s=measured_p95_s)
