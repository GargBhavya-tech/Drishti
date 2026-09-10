"""
tests/test_circular_pad.py

Ticket #25 tests: continuity across the seam, vertical axis untouched.
"""

from __future__ import annotations

import torch

from perception.circular_pad import circular_pad_horizontal


def test_horizontal_wrap_is_continuous_across_the_seam():
    # A feature map with a distinct value at each edge column -- after
    # circular padding, the padded left column must equal the original
    # rightmost column, and vice versa (the seam is genuinely adjacent).
    x = torch.arange(1, 11, dtype=torch.float32).view(1, 1, 1, 10)
    padded = circular_pad_horizontal(x, pad=1)
    assert padded.shape == (1, 1, 1, 12)
    assert padded[0, 0, 0, 0].item() == x[0, 0, 0, -1].item()  # left pad = original right edge
    assert padded[0, 0, 0, -1].item() == x[0, 0, 0, 0].item()  # right pad = original left edge


def test_vertical_axis_is_untouched_not_circular():
    x = torch.arange(1, 13, dtype=torch.float32).view(1, 1, 3, 4)
    padded = circular_pad_horizontal(x, pad=1)
    # Height dimension unchanged in size -- only width grew.
    assert padded.shape == (1, 1, 3, 6)


def test_zero_pad_is_a_no_op():
    x = torch.rand(2, 3, 4, 5)
    out = circular_pad_horizontal(x, pad=0)
    assert torch.equal(out, x)


def test_object_straddling_seam_has_continuous_response():
    """A synthetic 'object' (a spike) placed right at the seam: after
    circular padding, a 3-wide window centred on the seam sees the spike
    intact, not cut off by a zero-padded edge."""
    W = 20
    x = torch.zeros(1, 1, 1, W)
    x[0, 0, 0, 0] = 5.0
    x[0, 0, 0, -1] = 5.0  # the same object, straddling index W-1 / 0

    padded = circular_pad_horizontal(x, pad=1)
    # Centred 3-wide window around the seam (original indices -1,0,1 ->
    # padded indices 0,1,2) should read [5, 5, 0] -- object intact,
    # continuous -- not [0, 5, 0] as zero-padding would give.
    window = padded[0, 0, 0, 0:3]
    assert window[0].item() == 5.0
    assert window[1].item() == 5.0
