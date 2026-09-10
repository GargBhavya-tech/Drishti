"""
tests/test_raycast.py

Ticket #33's own test: "a single ray across a known grid visits exactly
the expected cell sequence (hand-computed for a 45 degree ray). The
terminal cell is OCCUPIED [is_terminal=True]. Cells beyond the return
are untouched, not FREE."
"""

from __future__ import annotations

from observability.raycast import RayHit, dda_trace, trace_beam
from sensor.schedule import Level


def test_45_degree_ray_visits_the_hand_computed_cell_sequence():
    # c_l = 1.0 m, from the centre of cell (0,0) to the centre of cell
    # (3,3), i.e. dx == dy exactly -- ties at every step. Hand-traced
    # (Amanatides-Woo, x0=y0=0.5, boundaries at 1.0, t_max_x==t_max_y at
    # every tie so the implementation's fixed j-first tie-break applies):
    # (0,0) -> (0,1) -> (1,1) -> (1,2) -> (2,2) -> (2,3) -> (3,3).
    hits = dda_trace(0.5, 0.5, 3.5, 3.5, c_l=1.0)
    cells = [(h.gi, h.gj) for h in hits]
    assert cells == [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]


def test_terminal_cell_is_occupied_not_free():
    hits = dda_trace(0.5, 0.5, 3.5, 3.5, c_l=1.0)
    # Exactly one hit is terminal, and it is the LAST one -- the cell
    # containing the actual return, per Ticket #33's off-by-one warning.
    terminal_flags = [h.is_terminal for h in hits]
    assert terminal_flags == [False, False, False, False, False, False, True]
    assert (hits[-1].gi, hits[-1].gj) == (3, 3)


def test_cells_beyond_the_return_are_never_visited():
    hits = dda_trace(0.5, 0.5, 3.5, 3.5, c_l=1.0)
    # The ray keeps going conceptually past (3.5, 3.5) if extended, but
    # dda_trace must stop the instant it reaches the endpoint's cell --
    # no cell at gi/gj > 3 appears anywhere in the sequence.
    assert all(h.gi <= 3 and h.gj <= 3 for h in hits)
    assert len(hits) == len(set((h.gi, h.gj) for h in hits))  # no duplicate cell visits


def test_single_cell_ray_degenerate_case():
    """Start and end in the same cell -- the loop guard must not spin."""
    hits = dda_trace(0.2, 0.2, 0.3, 0.4, c_l=1.0)
    assert hits == [RayHit(0, 0, is_terminal=True)]


def test_axis_aligned_ray_horizontal():
    hits = dda_trace(0.5, 0.5, 3.5, 0.5, c_l=1.0)
    cells = [(h.gi, h.gj) for h in hits]
    assert cells == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert hits[-1].is_terminal


def test_negative_direction_ray_uses_floor_not_truncation():
    # From cell (3,3) back to cell (0,0) -- exercises the negative-index
    # world_to_global path (Ticket #10's floor-vs-truncation trap).
    hits = dda_trace(3.5, 3.5, 0.5, 0.5, c_l=1.0)
    cells = [(h.gi, h.gj) for h in hits]
    assert cells[0] == (3, 3)
    assert cells[-1] == (0, 0)
    assert hits[-1].is_terminal


def test_trace_beam_coarse_pass_covers_the_whole_ray():
    levels = [
        Level(level=0, cell_size_m=1.0, nyquist_radius_m=5.0),
        Level(level=1, cell_size_m=4.0, nyquist_radius_m=40.0),
    ]
    # Return well beyond the fine ring (5 m) -- only the coarse pass
    # should reach it.
    result = trace_beam(0.0, 0.0, 20.0, 0.0, levels)
    assert set(result.keys()) == {0, 1}
    coarse_hits = result[1]
    assert coarse_hits[-1].is_terminal
    coarse_last_gi = coarse_hits[-1].gi
    assert coarse_last_gi == 20 // 4  # (20.0, 0.0) at cell size 4.0

    fine_hits = result[0]
    # The fine pass only reaches the ring's edge (5 m), not the real
    # return -- none of its cells may claim is_terminal, since the beam
    # didn't actually stop there.
    assert all(not h.is_terminal for h in fine_hits)
    assert max(h.gi for h in fine_hits) <= 5  # never travels past 5 m at c_l=1.0


def test_trace_beam_return_inside_fine_ring_is_terminal_there_too():
    levels = [
        Level(level=0, cell_size_m=1.0, nyquist_radius_m=5.0),
        Level(level=1, cell_size_m=4.0, nyquist_radius_m=40.0),
    ]
    # Return well INSIDE the fine ring -- the fine pass's own terminal
    # cell should correctly be flagged, since it covers the full beam.
    result = trace_beam(0.0, 0.0, 2.0, 0.0, levels)
    fine_hits = result[0]
    assert fine_hits[-1].is_terminal
    assert fine_hits[-1].gi == 2
