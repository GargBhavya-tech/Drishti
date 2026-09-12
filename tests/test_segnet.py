"""
tests/test_segnet.py

Ticket #28 tests: forward pass shape at the project's own input size,
deep-supervision aux head shape, parameter count sanity, and a
structural guard that nothing camera-specific leaked into the extraction.

Also covers the Ticket #25 integration fix: circular_pad.py's utility was
previously only tested in isolation and never actually wired into this
network's own convolutions -- see CircularConv2d's tests below and
segnet.py's module docstring, deviation 5, for the encoder-scope note.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

import perception.segnet as segnet_module
from perception.segnet import (
    ASPP,
    STEM_CONV_STATE_DICT_KEY,
    CircularConv2d,
    FusionSegNet,
    N_CLASSES_DEFAULT,
    N_INPUT_CHANNELS,
    _dblock,
    expand_stem_conv_for_checkpoint,
)


def test_eval_forward_pass_matches_ticket_28_shape():
    """Ticket #28's own test: a (1, 9, 32, 1080) input -> main logits
    (1, 10, 32, 1080). Eval mode -- see module docstring for why train
    mode needs batch_size > 1 (BatchNorm cannot compute statistics from
    a single sample once ASPP's global-pool branch collapses to 1x1,
    a fundamental PyTorch constraint unrelated to this extraction)."""
    net = FusionSegNet(n_classes=10)
    net.eval()
    x = torch.randn(1, N_INPUT_CHANNELS, 32, 1080)
    with torch.no_grad():
        out = net(x)
    assert out.shape == (1, 10, 32, 1080)


def test_train_forward_pass_returns_main_and_aux():
    net = FusionSegNet(n_classes=10)
    net.train()
    x = torch.randn(2, N_INPUT_CHANNELS, 32, 1080)
    out, aux = net(x)
    assert out.shape == (2, 10, 32, 1080)
    # Aux head is at 1/8 decoder resolution (Bible Part 5.3): H=32/8=4
    # exactly; W=1080's 1/8-scale value floors through the encoder's
    # stride-2 convs to 135 (not a clean 135.0 -- verified empirically).
    assert aux.shape[0] == 2
    assert aux.shape[1] == 10
    assert aux.shape[2] == 4


def test_default_n_classes_is_ten_matching_drishti_taxonomy():
    assert N_CLASSES_DEFAULT == 10


def test_first_conv_accepts_nine_channels_not_three():
    net = FusionSegNet()
    stem_conv = net.e1[0][0]
    assert isinstance(stem_conv, torch.nn.Conv2d)
    assert stem_conv.in_channels == N_INPUT_CHANNELS
    assert stem_conv.in_channels != 3


def test_parameter_count_in_the_competitive_band():
    """Bible Part 5.1: the whole FusionSegNet backbone should sit "in the
    same competitive band as SalsaNext (6.7M), CENet (6.8M) and FIDNet
    (6M)" -- roughly 4-9M. NOT the narrower "4-4.5M" in Ticket #28's own
    test text, which is Part 5.3's ENCODER-ONLY figure (see module
    docstring for the full reasoning)."""
    net = FusionSegNet()
    total = sum(p.numel() for p in net.parameters())
    assert 4_000_000 <= total <= 9_000_000, f"{total:,} params outside the competitive band"


def test_no_camera_specific_imports_leaked_into_the_extraction():
    """Ticket #28 'Done when': 'No import errors from removed camera
    code.' A structural guard: none of the notebook's camera/data-
    pipeline dependencies (albumentations, pyquaternion, shapely,
    google.colab, cv2) should appear anywhere in this module's source."""
    src = inspect.getsource(segnet_module)
    forbidden = ("albumentations", "pyquaternion", "shapely", "google.colab", "cv2", "sklearn")
    for name in forbidden:
        assert name not in src, f"camera-domain dependency '{name}' leaked into segnet.py"


def test_weights_are_from_scratch_not_imagenet_pretrained():
    """Deviation #1 in the module docstring: the notebook's actual code
    loads ImageNet weights despite its own docstring claiming otherwise.
    This extraction must use weights=None -- verified structurally
    (the source calls efficientnet_b0(weights=None), not DEFAULT).

    Reads the file directly via ast, rather than
    inspect.getsource(segnet_module.FusionSegNet.__init__): the latter
    is flaky when this test runs as part of the FULL suite on this
    project's dev environment (Python 3.14.3) -- it intermittently
    returns a near-empty string (observed: '        )\\n') for a bound
    method's source depending on what ran earlier in the same session,
    even though the module's own file on disk is unchanged and the same
    call succeeds every time in isolation. That looks like a
    linecache/inspect interaction specific to a very new CPython
    version under pytest's collection machinery, not a bug in this
    file's actual content -- reading the source text straight from disk
    sidesteps it entirely."""
    import ast

    file_path = inspect.getsourcefile(segnet_module)
    tree = ast.parse(Path(file_path).read_text())

    init_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FusionSegNet":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    init_node = item
                    break
    assert init_node is not None, "FusionSegNet.__init__ not found in segnet.py"

    lines = Path(file_path).read_text().splitlines()
    src = "\n".join(lines[init_node.lineno - 1 : init_node.end_lineno])

    assert "weights=None" in src
    assert "EfficientNet_B0_Weights" not in src


# ---------------------------------------------------------------------------
# Ticket #25 integration fix: circular padding actually wired into the
# network, not just built and tested in isolation (perception/circular_pad.py).
# ---------------------------------------------------------------------------


def test_circular_conv2d_preserves_shape():
    """CircularConv2d must produce the same (H, W) as an equivalent
    ordinary same-padded Conv2d -- the manual horizontal pre-pad plus
    zero-vertical-padding-only conv must net out to "same" shape, for
    every dilation rate ASPP actually uses.

    W=50 here, not a tiny width: torch's circular padding mode cannot
    pad more than the input's own size along an axis ("Padding value
    causes wrapping around more than once"), so the width must exceed
    the largest dilation (18) actually used at rate=18 -- exactly the
    constraint ASPP's real deployment already satisfies (it only ever
    runs on the encoder's deepest, 1/32-scale feature map, e.g. ~34 px
    wide from this project's own 1080 px range image, still > 18)."""
    x = torch.randn(2, 4, 5, 50)
    for dilation in (1, 6, 12, 18):
        conv = CircularConv2d(4, 8, kernel_size=3, dilation=dilation)
        out = conv(x)
        assert out.shape == (2, 8, 5, 50), f"shape changed at dilation={dilation}: {out.shape}"


def test_circular_conv2d_is_continuous_across_the_seam():
    """The actual property Ticket #25 exists for: a feature straddling
    the 359/0-degree seam must produce a continuous response, not a
    discontinuous one the way zero-padding would. Uses a single fixed
    (non-random) kernel so the output is exactly predictable, matching
    tests/test_circular_pad.py's own style of proof."""
    conv = CircularConv2d(1, 1, kernel_size=3, dilation=1, bias=False)
    with torch.no_grad():
        conv.conv.weight.fill_(1.0 / 9.0)  # a 3x3 box filter

    W = 10
    x = torch.zeros(1, 1, 3, W)
    x[0, 0, 1, 0] = 9.0  # a spike exactly at the seam's right side (column 0)
    x[0, 0, 1, -1] = 9.0  # the SAME object, straddling into column W-1

    out = conv(x)
    # A pixel just left of the seam (column W-1) should "see" the spike at
    # column 0 through the circular wrap, exactly as it would see a
    # spike at column W-2 if the object were fully interior -- i.e. the
    # response at the seam must be indistinguishable from an interior
    # spike's response, not suppressed the way zero-padding would
    # suppress it.
    interior = torch.zeros(1, 1, 3, W)
    interior[0, 0, 1, 4] = 9.0
    interior[0, 0, 1, 5] = 9.0  # two adjacent columns, interior, same total energy
    interior_out = conv(interior)

    seam_peak = out[0, 0, 1, 0].item()
    interior_peak = interior_out[0, 0, 1, 4].item()
    assert seam_peak == pytest.approx(interior_peak, abs=1e-6), (
        "circular wrap not applied -- the seam pixel's response should match "
        "an equivalent interior pixel's response exactly"
    )


def test_aspp_branches_use_circular_conv2d():
    aspp = ASPP(in_ch=8, out_ch=4, rates=(1, 6))
    for branch in aspp.branches:
        assert isinstance(branch[0], CircularConv2d), "ASPP branch's conv is not circular-padded"


def test_dblock_convs_use_circular_conv2d():
    block = _dblock(cin=8, cout=4)
    conv_layers = [m for m in block if isinstance(m, (CircularConv2d, torch.nn.Conv2d))]
    assert len(conv_layers) == 2
    assert all(isinstance(m, CircularConv2d) for m in conv_layers), (
        "decoder block still uses plain (zero-padded) Conv2d instead of CircularConv2d"
    )


def test_full_network_still_forward_passes_with_circular_padding_wired_in():
    """Regression guard: wiring CircularConv2d into ASPP and the decoder
    must not change FusionSegNet's output shape at the project's own
    input size (Ticket #28's own shape test)."""
    net = FusionSegNet(n_classes=10)
    net.eval()
    x = torch.randn(1, N_INPUT_CHANNELS, 32, 1080)
    with torch.no_grad():
        out = net(x)
    assert out.shape == (1, 10, 32, 1080)


def test_return_attention_default_false_is_bit_for_bit_unchanged():
    """The explainability flag must be fully opt-in -- every existing
    caller (perception/train.py, eval/cache_inference.py) calls
    model(x) with no return_attention argument at all, and must keep
    getting EXACTLY the original return value. Uses the same (32, 1080)
    input size as this file's own existing shape test -- ASPP's largest
    dilation rate (18) needs a wide-enough bottleneck feature map, per
    CircularConv2d's own documented "cannot wrap more than once" limit."""
    net = FusionSegNet(n_classes=10)
    net.eval()
    x = torch.randn(1, N_INPUT_CHANNELS, 32, 1080)
    with torch.no_grad():
        torch.manual_seed(0)
        out_default = net(x)
        torch.manual_seed(0)
        out_explicit_false = net(x, return_attention=False)
    assert torch.equal(out_default, out_explicit_false)
    assert isinstance(out_default, torch.Tensor)  # not a tuple -- eval mode, no attention requested


def test_return_attention_true_yields_four_valid_gate_maps():
    net = FusionSegNet(n_classes=10)
    net.eval()
    x = torch.randn(1, N_INPUT_CHANNELS, 32, 1080)
    with torch.no_grad():
        out, attention_maps = net(x, return_attention=True)

    assert out.shape == (1, 10, 32, 1080)
    assert set(attention_maps.keys()) == {"ag1", "ag2", "ag3", "ag4"}
    for name, attn in attention_maps.items():
        assert attn.shape[0] == 1 and attn.shape[1] == 1, f"{name} attention map must be single-channel"
        # Sigmoid output -- a real per-pixel attention weight, not a
        # placeholder or unbounded activation.
        assert torch.all(attn >= 0.0) and torch.all(attn <= 1.0), f"{name} attention values outside [0,1]"


def test_return_attention_true_during_training_still_returns_aux_head():
    net = FusionSegNet(n_classes=10)
    net.train()
    x = torch.randn(2, N_INPUT_CHANNELS, 32, 1080)
    out, aux, attention_maps = net(x, return_attention=True)
    assert out.shape[0] == 2
    assert aux is not None
    assert len(attention_maps) == 4


def test_expand_stem_conv_preserves_old_channels_and_zero_inits_new_ones():
    old_weight = torch.randn(32, 9, 3, 3)
    state_dict = {STEM_CONV_STATE_DICT_KEY: old_weight, "other.key": torch.randn(5)}

    expanded = expand_stem_conv_for_checkpoint(state_dict, new_in_channels=13)

    new_weight = expanded[STEM_CONV_STATE_DICT_KEY]
    assert new_weight.shape == (32, 13, 3, 3)
    torch.testing.assert_close(new_weight[:, :9, :, :], old_weight)
    assert torch.all(new_weight[:, 9:, :, :] == 0.0)
    # Every other key must be untouched (same tensor, not a copy that
    # happens to have the same values).
    assert expanded["other.key"] is state_dict["other.key"]
    # The input dict itself must not be mutated.
    assert state_dict[STEM_CONV_STATE_DICT_KEY] is old_weight


def test_expand_stem_conv_same_channel_count_is_a_no_op():
    old_weight = torch.randn(32, 9, 3, 3)
    state_dict = {STEM_CONV_STATE_DICT_KEY: old_weight}
    expanded = expand_stem_conv_for_checkpoint(state_dict, new_in_channels=9)
    torch.testing.assert_close(expanded[STEM_CONV_STATE_DICT_KEY], old_weight)


def test_expand_stem_conv_rejects_shrinking():
    state_dict = {STEM_CONV_STATE_DICT_KEY: torch.randn(32, 13, 3, 3)}
    with pytest.raises(ValueError):
        expand_stem_conv_for_checkpoint(state_dict, new_in_channels=9)


def test_expanded_model_produces_bit_identical_output_when_new_channels_are_zero():
    """The whole POINT of zero-initialising new channels: a model built
    with the expanded channel count, loading a weight-surgeried old
    checkpoint, must produce EXACTLY the same output as the original
    model did, as long as the new input channels are fed ZEROS (they
    contribute nothing through a zero weight, regardless of their
    actual input values -- but zero input makes the equivalence
    trivially checkable without depending on that fact separately)."""
    torch.manual_seed(0)
    old_model = FusionSegNet(n_classes=10, in_channels=9)
    old_model.eval()
    x_old = torch.randn(1, 9, 32, 1080)
    with torch.no_grad():
        old_out = old_model(x_old)

    expanded_state = expand_stem_conv_for_checkpoint(old_model.state_dict(), new_in_channels=13)
    new_model = FusionSegNet(n_classes=10, in_channels=13)
    new_model.load_state_dict(expanded_state)
    new_model.eval()

    x_new = torch.zeros(1, 13, 32, 1080)
    x_new[:, :9, :, :] = x_old
    with torch.no_grad():
        new_out = new_model(x_new)

    torch.testing.assert_close(new_out, old_out)
