"""
tests/test_latency_compensation.py

Ticket #61 tests, per the Build Map's own test list: "assert ego
staleness requires zero map transformation. Assert inflated extents
never shrink a hazard."
"""

from __future__ import annotations

from temporal.latency_compensation import (
    MAX_EXTRAPOLATION_HORIZON_S,
    ego_staleness_requires_no_transform,
    inflate_moving_cell,
    publish_map,
)


def test_ego_staleness_requires_zero_map_transformation():
    assert ego_staleness_requires_no_transform() is True


def test_inflated_extent_always_contains_the_advanced_center_cell():
    cell = (10, 10)
    inflated = inflate_moving_cell(cell, velocity_cells_per_s=(2.0, 0.0), elapsed_s=0.5, uncertainty_radius_cells=1.0)
    assert (11, 10) in inflated  # advanced by v*t = 2*0.5 = 1 cell along i


def test_larger_uncertainty_radius_never_shrinks_the_inflated_set():
    cell = (0, 0)
    small = inflate_moving_cell(cell, velocity_cells_per_s=(0.0, 0.0), elapsed_s=0.2, uncertainty_radius_cells=1.0)
    large = inflate_moving_cell(cell, velocity_cells_per_s=(0.0, 0.0), elapsed_s=0.2, uncertainty_radius_cells=3.0)
    assert small.issubset(large)
    assert len(large) > len(small)


def test_longer_elapsed_time_never_shrinks_the_inflated_set_extent():
    """Longer elapsed time moves the CENTER further but must never
    reduce the number of cells covered (same radius) -- inflation is
    a property of uncertainty (the radius), not of elapsed time alone,
    so the SIZE of the inflated set at a fixed radius stays constant
    regardless of elapsed time; what matters for "never shrink a
    hazard" is that the covered AREA never becomes smaller than a
    single static point would have been."""
    cell = (0, 0)
    short = inflate_moving_cell(cell, velocity_cells_per_s=(5.0, 0.0), elapsed_s=0.1, uncertainty_radius_cells=1.0)
    long = inflate_moving_cell(cell, velocity_cells_per_s=(5.0, 0.0), elapsed_s=0.8, uncertainty_radius_cells=1.0)
    assert len(short) == len(long) == 9  # a 3x3 neighbourhood either way -- inflation size unaffected by elapsed time


def test_extrapolation_horizon_is_capped():
    cell = (0, 0)
    at_cap = inflate_moving_cell(cell, velocity_cells_per_s=(10.0, 0.0), elapsed_s=MAX_EXTRAPOLATION_HORIZON_S, uncertainty_radius_cells=1.0)
    way_beyond_cap = inflate_moving_cell(cell, velocity_cells_per_s=(10.0, 0.0), elapsed_s=MAX_EXTRAPOLATION_HORIZON_S * 100, uncertainty_radius_cells=1.0)
    assert at_cap == way_beyond_cap  # extrapolation never goes further than the capped horizon


def test_zero_velocity_cell_inflated_extent_still_covers_the_original_cell():
    cell = (5, 5)
    inflated = inflate_moving_cell(cell, velocity_cells_per_s=(0.0, 0.0), elapsed_s=0.5, uncertainty_radius_cells=1.0)
    assert cell in inflated


def test_publish_map_carries_both_measurement_and_validity_timestamps():
    published = publish_map(measurement_time_s=10.0, validity_time_s=10.3, moving_cells={(0, 0): (1.0, 0.0)})
    assert published.measurement_time_s == 10.0
    assert published.validity_time_s == 10.3
    assert published.measurement_time_s != published.validity_time_s


def test_publish_map_with_no_moving_cells_needs_zero_work():
    published = publish_map(measurement_time_s=10.0, validity_time_s=12.0, moving_cells={})
    assert published.inflated_moving_cells == set()


def test_publish_map_inflates_every_moving_cell():
    moving = {(0, 0): (1.0, 0.0), (100, 100): (0.0, -1.0)}
    published = publish_map(measurement_time_s=0.0, validity_time_s=0.5, moving_cells=moving)
    assert len(published.inflated_moving_cells) > 0
    # Both original regions must be represented -- one moving cell's
    # inflation must not silently overwrite another's.
    near_first = any(abs(gi) < 5 and abs(gj) < 5 for gi, gj in published.inflated_moving_cells)
    near_second = any(abs(gi - 100) < 5 and abs(gj - 100) < 5 for gi, gj in published.inflated_moving_cells)
    assert near_first and near_second
