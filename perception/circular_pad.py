"""
perception/circular_pad.py

Ticket #25 -- circular padding at the 360-degree seam. The range image's
left and right edges are not real boundaries (359deg is physically
adjacent to 0deg); zero-padding there destroys context for anything
straddling the seam. Vertical padding stays zero -- the top and bottom
ARE real boundaries of the sensor's field of view.

Ticket #25 "Watch out": applying circular padding in both dimensions
(one flag) wraps the ground onto the sky. This function only ever pads
horizontally.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def circular_pad_horizontal(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Pad only the last (width/azimuth) dimension circularly; the
    second-to-last (height/elevation) dimension is left untouched.

    x: (..., H, W) tensor, any leading batch/channel dims.
    pad: columns added on each side (so a 3x3 conv wants pad=1).
    """
    if pad == 0:
        return x
    # PyTorch's non-constant F.pad requires the pad tuple's length to
    # match how many trailing dims it covers, and for circular mode that
    # must be an even count covering whole dim-pairs (e.g. 4 for a 4D
    # tensor -> pads the last TWO dims, not just width). The vertical
    # (top, bottom) amounts are 0 -- a 0-amount pad changes nothing
    # regardless of mode, which is what keeps this "horizontal only" in
    # practice even though circular mode is technically applied to both
    # of the trailing spatial dims at the API level.
    ndim = x.dim()
    if ndim >= 4:
        pad_arg = (pad, pad, 0, 0)
    elif ndim == 3:
        pad_arg = (pad, pad)  # 3D input: only the last dim is padded, per torch's own rule
    else:
        raise ValueError(f"circular_pad_horizontal expects a 3D or 4D+ tensor, got {ndim}D")
    return F.pad(x, pad_arg, mode="circular")
