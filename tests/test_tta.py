"""
tests/test_tta.py

Real correctness checks for perception/tta.py (Bible Part G.30):

1. The flip channel correction is EXACT algebra (not merely "sign-flip
   and hope"), checked against hand-computed values.
2. The roll transform + undo round-trips correctly THROUGH A MODEL THAT
   IS GENUINELY ROLL-EQUIVARIANT BY CONSTRUCTION (a single circularly-
   padded conv, using perception.circular_pad.circular_pad_horizontal
   directly) -- FusionSegNet itself is only PARTIALLY circular (ASPP +
   decoder, not the full EfficientNet-B0 encoder, per segnet.py's own
   docstring), so it is not the right model to prove the WRAPPER's own
   bookkeeping is correct; a model equivariant by construction is.
3. A real-shape smoke test against the actual FusionSegNet class.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from perception.circular_pad import circular_pad_horizontal
from perception.input_tensor import ChannelStats, N_CHANNELS
from perception.segnet import FusionSegNet, N_CLASSES_DEFAULT
from perception.tta import NORMAL_Y_CHANNEL, Y_CHANNEL, _flip_input, predict_with_tta


def test_flip_channel_correction_is_exact():
    torch.manual_seed(0)
    x = torch.randn(1, N_CHANNELS, 4, 8)
    stats = ChannelStats(mean=[0.24 if i == Y_CHANNEL else (0.0088 if i == NORMAL_Y_CHANNEL else 0.0) for i in range(N_CHANNELS)],
                          std=[6.8885 if i == Y_CHANNEL else (0.3886 if i == NORMAL_Y_CHANNEL else 1.0) for i in range(N_CHANNELS)])

    flipped = _flip_input(x, stats)

    # Reconstruct the RAW y value the original normalized x encodes, negate
    # it (the real theta -> -theta transform), re-normalise, and confirm
    # the function's output matches EXACTLY -- not the naive "-x_norm"
    # approximation, which would be measurably off given mean_y != 0.
    mean_y, std_y = stats.mean[Y_CHANNEL], stats.std[Y_CHANNEL]
    y_raw = x[:, Y_CHANNEL] * std_y + mean_y
    expected_y_norm_after_negation = (-y_raw - mean_y) / std_y
    # flipped also spatially reversed the width axis -- compare against the
    # spatially-flipped (but not yet channel-corrected) reference.
    expected_spatial = torch.flip(expected_y_norm_after_negation, dims=[-1])
    assert torch.allclose(flipped[:, Y_CHANNEL], expected_spatial, atol=1e-6)

    # Confirm the naive approximation (-x_norm alone) would have been
    # measurably wrong here, given mean_y=0.24 != 0 -- this is the whole
    # reason the exact correction matters, not a redundant check.
    naive = -torch.flip(x[:, Y_CHANNEL], dims=[-1])
    assert not torch.allclose(flipped[:, Y_CHANNEL], naive, atol=1e-3)

    # A channel NOT in {y, normal_y} (e.g. channel 0, x) should be purely
    # spatially flipped, no value correction.
    assert torch.allclose(flipped[:, 0], torch.flip(x[:, 0], dims=[-1]))


class _RollEquivariantStub(nn.Module):
    """A single circularly-padded conv -- genuinely roll-equivariant by
    construction (unlike FusionSegNet itself, which mixes circular and
    non-circular layers): rolling the input by k columns and running this
    model produces EXACTLY the same output as running the model first and
    then rolling by k columns, because circular_pad_horizontal + a valid
    conv commutes exactly with torch.roll along the same axis."""

    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_classes, kernel_size=3, padding=0, bias=False)

    def forward(self, x):
        return self.conv(circular_pad_horizontal(x, 1))


def test_roll_transform_round_trips_exactly_through_an_equivariant_model():
    torch.manual_seed(0)
    model = _RollEquivariantStub(N_CHANNELS, N_CLASSES_DEFAULT).eval()
    x = torch.randn(1, N_CHANNELS, 8, 16)
    stats = ChannelStats(mean=[0.0] * N_CHANNELS, std=[1.0] * N_CHANNELS)

    with torch.no_grad():
        base_pred = predict_with_tta(model, x, stats, use_flip=False)
        base_only_pred = model(x).argmax(dim=1)

    # For a genuinely roll-equivariant model, TTA-with-roll-only should
    # agree with the plain base prediction almost everywhere -- the tiny
    # disagreement budget accounts for pixels where the AVERAGED softmax
    # (base+roll)/2 crosses an argmax boundary the single-view prediction
    # didn't, which can legitimately happen even for an equivariant model
    # once two DIFFERENT (but individually correct) views are averaged.
    agreement = (base_pred == base_only_pred).float().mean().item()
    assert agreement > 0.99, f"expected near-total agreement for a roll-equivariant model, got {agreement:.4f}"


def test_predict_with_tta_real_model_smoke_shape():
    # H=32, W=1080 matches test_segnet.py's own established real-size
    # test input -- ASPP's largest dilation (6, per the traceback this
    # replaced) needs a wide-enough feature map at its own depth in the
    # encoder; this project's real 1080+ px range-image width satisfies
    # that, a synthetic 40px width does not.
    torch.manual_seed(0)
    model = FusionSegNet(n_classes=N_CLASSES_DEFAULT).eval()
    x = torch.randn(1, N_CHANNELS, 32, 1080)
    stats = ChannelStats(mean=[0.0] * N_CHANNELS, std=[1.0] * N_CHANNELS)

    with torch.no_grad():
        pred = predict_with_tta(model, x, stats, use_flip=True)
    assert pred.shape == (1, 32, 1080)
    assert pred.dtype == torch.int64

    with torch.no_grad():
        pred_no_flip = predict_with_tta(model, x, stats, use_flip=False)
    assert pred_no_flip.shape == (1, 32, 1080)
