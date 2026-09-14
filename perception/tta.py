"""
perception/tta.py

Test-Time Augmentation for FusionSegNet inference -- Bible Part G.30,
following a deep-research report's own recommendation: average softmax
predictions across the base view, a circular azimuth roll, and a
horizontal flip, to reduce the artificial "seam" bias `project_to_range_image`
introduces by cutting the sensor's continuous 360-degree cylindrical scan
at an arbitrary azimuth=0.

CIRCULAR ROLL is exact and physically lossless: `perception/range_image.py`'s
own azimuth formula (`u = floor(0.5*(1 - atan2(y,x)/pi) * W)`) maps column
index to a specific azimuth angle -- rolling the (H,W) tensor by W/2 columns
is EXACTLY equivalent to a real sensor mounted 180deg differently. The x/y/z
coordinate VALUES at each pixel don't need to change; only which column they
sit in does.

HORIZONTAL FLIP is NOT a free lunch the way it is for a natural photo: it is
NOT physically equivalent to any real sensor transform. Flipping the array
left-right corresponds to azimuth theta -> -theta (verified directly from
the same `atan2(y,x)` formula: cos(-theta)=cos(theta) so x is unchanged,
sin(-theta)=-sin(theta) so y negates). Feeding the network a spatially-
flipped image WITHOUT negating channel 1 (y) and channel 10 (normal_y)
would present physically-inconsistent geometry (pixel positions say
"mirrored" but the y/normal_y VALUES still describe the original, non-
mirrored scan) -- likely degrading, not improving, that view's prediction.

The exact correction (not the naive "just negate the normalized value"
approximation, which would carry a small but real, systematic bias since
y's real per-dataset mean is 0.24, not exactly 0 -- checked directly against
`channel_stats.json`, not assumed): given normalized y_norm=(y_raw-mean)/std,
the normalized value of the NEGATED raw quantity is
    -(y_raw) normalized = (-y_raw - mean) / std = -y_norm - 2*mean/std
Occlusion_count/occlusion_spread, range, intensity, curvature, x, z,
normal_x, normal_z, valid_mask, and ground_prior are all either scalar
(rotation-invariant) quantities or components along axes flip doesn't
touch -- no correction needed for them, spatial re-arrangement is enough.
"""

from __future__ import annotations

from typing import Optional

import torch

from perception.input_tensor import ChannelStats

Y_CHANNEL = 1
NORMAL_Y_CHANNEL = 10


def _flip_input(x: torch.Tensor, stats: ChannelStats) -> torch.Tensor:
    """x: (B, 13, H, W), already channel-normalised. Returns a new tensor,
    spatially flipped along the azimuth (last) axis, with channels 1 (y)
    and 10 (normal_y) exactly corrected for the theta -> -theta reflection
    (see module docstring) -- not merely sign-flipped in place."""
    flipped = torch.flip(x, dims=[-1]).clone()
    mean_y, std_y = stats.mean[Y_CHANNEL], stats.std[Y_CHANNEL]
    mean_ny, std_ny = stats.mean[NORMAL_Y_CHANNEL], stats.std[NORMAL_Y_CHANNEL]
    flipped[:, Y_CHANNEL] = -flipped[:, Y_CHANNEL] - 2 * mean_y / std_y
    flipped[:, NORMAL_Y_CHANNEL] = -flipped[:, NORMAL_Y_CHANNEL] - 2 * mean_ny / std_ny
    return flipped


def predict_with_tta(
    model,
    x: torch.Tensor,
    stats: ChannelStats,
    use_flip: bool = True,
    roll_fraction: float = 0.5,
) -> torch.Tensor:
    """Runs the base view, a circular azimuth roll, and (optionally) the
    geometrically-corrected horizontal flip through `model`, averages the
    resulting softmax probabilities (each view's output un-transformed
    back to the ORIGINAL spatial layout before averaging -- transforming
    probabilities, not logits, per standard TTA practice), and returns
    the final (B, H, W) argmax prediction. `model` must be in eval mode
    and this must be called under `torch.no_grad()` by the caller (not
    wrapped here, matching every other inference call site's own
    convention in this project)."""
    W = x.shape[-1]
    shift = int(round(W * roll_fraction))

    views = [(x, lambda p: p)]
    x_roll = torch.roll(x, shifts=shift, dims=-1)
    views.append((x_roll, lambda p, s=shift: torch.roll(p, shifts=-s, dims=-1)))
    if use_flip:
        x_flip = _flip_input(x, stats)
        views.append((x_flip, lambda p: torch.flip(p, dims=[-1])))

    probs_sum: Optional[torch.Tensor] = None
    for view_x, undo in views:
        logits = model(view_x)
        probs = torch.softmax(logits, dim=1)
        probs = undo(probs)
        probs_sum = probs if probs_sum is None else probs_sum + probs

    probs_avg = probs_sum / len(views)
    return probs_avg.argmax(dim=1)
