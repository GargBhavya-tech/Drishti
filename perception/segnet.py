"""
perception/segnet.py

Ticket #28 -- FusionSegNet architecture, lifted from the team's existing
camera-domain notebook (`FusionSegNet_v5 (1).ipynb`, cell "## 7.
FusionSegNet Architecture") and adapted per the Build Map's own spec:
"Change the first conv from 3 to 9 input channels. Nothing else."

ONLY the architecture transfers -- Bible Part 5.1 / Build Map Ticket #28:
"the repo is a camera model trained on nuScenes images -- there are no
reusable weights and no LiDAR data pipeline." Nothing from the notebook's
data pipeline, pseudo-labelling, copy-paste augmentation, temporal-
consistency loss, or camera-specific training schedule is carried over --
only the four building blocks (SEBlock, AttentionGate, ASPP,
FusionSegNet) and the network's own forward pass.

DEVIATIONS from the notebook, each deliberate:

1. `weights=None` (train from scratch), not the notebook's
   `EfficientNet_B0_Weights.DEFAULT`. The notebook's OWN docstring
   claims "weights=None -- fully from scratch" while its actual code
   loads ImageNet-pretrained weights -- an internal contradiction in the
   source notebook, caught while extracting it. Build Map Ticket #28 is
   explicit that this adapted network has no reusable weights (a
   9-channel LiDAR range image has no correspondence to ImageNet's
   3-channel statistics anyway, and the first conv is being resized
   regardless), so this extraction follows the documented design intent,
   not the notebook's inconsistent code.
2. The stem/stage freeze-for-10-epochs schedule is dropped -- that is a
   training-loop decision (protecting pretrained features), not an
   architecture-definition concern, and doesn't apply the same way
   starting from scratch. Ticket #30 (training, not yet built) owns any
   freeze schedule if one turns out to be wanted.
3. `n_classes` defaults to 10, matching `perception.taxonomy.DrishtiClass`
   (the notebook's camera model used 5 classes for its own dataset).
4. `_match_size()` resizes each decoder upsampling stage to its skip
   connection's exact spatial size before concatenation. The unmodified
   notebook code crashes on this project's own range-image width (1080):
   5 successive stride-2 encoder convs floor an odd intermediate
   dimension, so ConvTranspose2d's exact doubling on the way back up
   lands one pixel narrower than the matching skip tensor (verified:
   `torch.cat` fails with "Expected size 136 but got size 135" at the
   d4 stage for a (1,9,32,1080) input). The camera inputs the notebook
   was written against were evidently always chosen to divide cleanly by
   32; LiDAR range-image widths generally aren't. `AttentionGate` already
   uses this exact technique internally for its own gating signal; this
   applies the same fix to the decoder's main path.

Measured parameter count: 5.82M. Note Ticket #28's own test text says
"Parameter count ~= 4-4.5M", but that figure is Bible Part 5.3's
ENCODER-ONLY count; Bible Part 5.1 separately states the WHOLE
FusionSegNet backbone should be "~8M params ... same competitive band as
SalsaNext (6.7M), CENet (6.8M), FIDNet (6M)" -- 5.82M sits squarely in
that band. The Build Map appears to have conflated the two figures in
Ticket #28's test text; this module is tested against the whole-network
framing (Part 5.1), not the encoder-only number, since cutting real
architecture (ASPP, decoder, heads) to hit 4-4.5M would mean shipping a
different network than the one specified.

5. Ticket #25's circular horizontal padding is now actually wired into
   this network's OWN convolutions (ASPP's dilated branches and the
   decoder's `_dblock` convs), not just built and tested in isolation in
   `perception/circular_pad.py` -- see `CircularConv2d` below. Bible
   Part 4.5's rule ("replace ALL horizontal padding in the network")
   is applied to every conv authored in this file. It is deliberately
   NOT applied inside the EfficientNet-B0 encoder (`self.e1`..`self.e5`):
   those are torchvision's own `MBConv`/`Conv2dNormActivation` internals,
   several dozen depthwise/pointwise convs across 5 stages, and rewriting
   each one's padding scheme is a materially larger and riskier change
   than swapping the convs this file itself defines -- the same kind of
   scope boundary this codebase already draws around the Meta-Kernel
   (Bible Part 5.5, ablation-gated, never built). Flagged here rather
   than silently claimed as network-wide, per this project's own
   discipline (Principle 4: "state limitations, do not hide them").
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0

from perception.circular_pad import circular_pad_horizontal
from perception.input_tensor import N_CHANNELS as N_INPUT_CHANNELS

# N_INPUT_CHANNELS now imported from perception.input_tensor (the single
# source of truth for the assembled tensor's channel count) rather than
# duplicated as a literal here -- the two constants had already drifted
# apart once (this file said 9 while input_tensor.py grew to 13) before
# this fix, exactly the class of bug this project's own culture guards
# against elsewhere (e.g. vehicle_ugv.yaml's "declared once" discipline).
N_CLASSES_DEFAULT = 10  # perception.taxonomy.DrishtiClass


class CircularConv2d(nn.Module):
    """A 3x3 (optionally dilated) conv with circular padding on the
    horizontal (azimuth) axis and ordinary zero padding on the vertical
    (elevation) axis -- Ticket #25's rule, applied here instead of only
    living as a standalone utility. The range image's left/right edges
    are the sensor's 360-degree seam (not a real boundary); top/bottom
    ARE real FOV boundaries (Bible Part 4.5), so only width wraps.

    Implemented as manual horizontal pad (`circular_pad_horizontal`,
    already unit-tested in isolation) + a Conv2d whose OWN padding is
    zero on width and the usual symmetric amount on height -- not
    `nn.Conv2d(padding_mode='circular')`, which would wrap both axes
    uniformly and is exactly the "wraps the ground onto the sky" bug
    Ticket #25's own docstring warns against.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, bias: bool = False):
        super().__init__()
        # "Same"-shape padding for stride 1: pad = dilation * (k-1) // 2.
        pad = dilation * (kernel_size - 1) // 2
        self._pad_w = pad
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size,
            padding=(pad, 0), dilation=dilation, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        width = x.shape[-1]
        if self._pad_w >= width:
            # torch's circular padding cannot wrap more than once around
            # an axis ("Padding value causes wrapping around more than
            # once"); at ASPP's largest dilation (18) this only matters
            # if a feature map narrower than 19px reaches here, which
            # doesn't happen at this project's own input size (the
            # deepest ASPP-feeding stage is ~1/32 of a >=1080px range
            # image) -- but a future change to input width/dilation
            # rates should fail with a clear message here, not a
            # confusing error three frames down inside torch internals.
            raise ValueError(
                f"CircularConv2d: padding {self._pad_w} >= input width {width} -- "
                f"circular padding cannot wrap more than once around an axis. "
                f"Either the input is too narrow for this dilation rate, or the "
                f"dilation rate is too large for this input's width."
            )
        x = circular_pad_horizontal(x, self._pad_w)
        return self.conv(x)


class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, max(1, ch // r)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, ch // r), ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).view(x.shape[0], -1, 1, 1)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x, return_attention: bool = False):
        g_up = F.interpolate(self.W_g(g), size=x.shape[2:], mode="bilinear", align_corners=False)
        psi = F.relu(g_up + self.W_x(x), inplace=True)
        attn = self.psi(psi)
        gated = x * attn
        # `return_attention=False` (the default) preserves the exact
        # original return value/signature -- every existing caller
        # (perception/train.py, eval/cache_inference.py) is unaffected.
        # `attn` itself (not `gated`) is what an explainability overlay
        # wants: it IS the per-pixel "how much did this skip connection's
        # signal matter here" map, in [0, 1] by construction (Sigmoid).
        if return_attention:
            return gated, attn
        return gated


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=128, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            self.branches.append(
                nn.Sequential(
                    CircularConv2d(in_ch, out_ch, kernel_size=3, dilation=r, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [b(x) for b in self.branches]
        gap = F.interpolate(self.gap(x), size=(h, w), mode="bilinear", align_corners=False)
        feats.append(gap)
        return self.proj(torch.cat(feats, dim=1))


def _dblock(cin, cout):
    return nn.Sequential(
        CircularConv2d(cin, cout, kernel_size=3, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        CircularConv2d(cout, cout, kernel_size=3, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        SEBlock(cout),  # SE attention in every decoder block
    )


def _match_size(x: torch.Tensor, target) -> torch.Tensor:
    """Resize x's spatial dims to match `target` (a tensor, whose
    shape[2:] is used, or an explicit (H, W) tuple) -- a no-op when they
    already agree. Needed because ConvTranspose2d(kernel=2, stride=2)
    doubles exactly, but the ENCODER's stride-2 convs floor odd
    dimensions on the way down, so an input whose width isn't cleanly
    divisible by 32 (e.g. 1080, this project's own range-image width --
    the camera inputs the source notebook was written against evidently
    always were) comes back one pixel narrower than its matching skip
    connection. `AttentionGate` already does exactly this for its own
    gating signal internally; this applies the same fix to the decoder's
    main upsampling path before each concatenation.
    """
    target_hw = target.shape[2:] if torch.is_tensor(target) else tuple(target)
    if tuple(x.shape[2:]) == tuple(target_hw):
        return x
    return F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)


STEM_CONV_STATE_DICT_KEY = "e1.0.0.weight"  # verified against a real checkpoint: shape (32, 9, 3, 3)


def remap_legacy_conv_keys(state_dict: dict, expected_keys: set) -> dict:
    """Run #1/#2's checkpoints were trained BEFORE ASPP's branches and the
    decoder's _dblock convs were wrapped in CircularConv2d -- that
    wrapping renames each conv's own parameters from e.g.
    "aspp.branches.0.0.weight" to "aspp.branches.0.0.conv.weight" (a new
    ".conv." submodule), with the SAME tensor shapes (all renamed keys,
    identical shapes, no bias keys since these convs use bias=False) --
    the operation is functionally unchanged, only its parameter path
    moved. Remap old-style keys to new-style ones rather than silently
    failing to load, or forcing a from-scratch retrain of an
    already-good checkpoint over a rename.

    Moved here from eval/cache_inference.py (where it originated) so
    perception.train's own `--init-from-checkpoint` path can share the
    SAME logic instead of re-implementing it -- that path never had this
    fix, which is exactly why `checkpoint_epoch19.pt` (itself pre-dating
    this rename) failed to warm-start a fresh run until this move.
    """
    if expected_keys.issubset(state_dict.keys()):
        return state_dict  # already new-style, e.g. a checkpoint trained after this fix
    remapped = {}
    for key, value in state_dict.items():
        new_key = key.replace(".weight", ".conv.weight").replace(".bias", ".conv.bias")
        remapped[new_key if new_key in expected_keys else key] = value
    return remapped


def expand_stem_conv_for_checkpoint(
    state_dict: dict, new_in_channels: int, stem_key: str = STEM_CONV_STATE_DICT_KEY
) -> dict:
    """Grows an existing checkpoint's stem conv weight from its old
    in_channels to `new_in_channels`, PRESERVING the old channels'
    trained weights and zero-initialising the new ones -- so a model
    built with `FusionSegNet(in_channels=new_in_channels)` can load this
    checkpoint and, for the channels that already existed, start
    numerically IDENTICAL to the old checkpoint (a zero-weighted new
    channel contributes nothing to the conv's output, whatever value it
    holds) rather than the fresh model's random reinitialisation of the
    entire stem, which would discard everything the old channels
    already learned. Only the stem conv key is touched; every other key
    in the returned dict is the input state_dict's own value, unchanged
    (shapes are identical everywhere else since only the FIRST layer's
    input width depends on the channel count).

    Returns a NEW dict (the input is not mutated). Raises if
    `new_in_channels` is smaller than the checkpoint's own channel count
    -- shrinking would silently discard trained channels, which is never
    the intended use of this function.
    """
    old_weight = state_dict[stem_key]
    old_in_channels = old_weight.shape[1]
    if new_in_channels < old_in_channels:
        raise ValueError(
            f"new_in_channels ({new_in_channels}) < checkpoint's own {old_in_channels} -- "
            f"this function only grows channel count, it does not shrink it."
        )
    if new_in_channels == old_in_channels:
        return dict(state_dict)

    out_channels, _, kh, kw = old_weight.shape
    new_weight = torch.zeros((out_channels, new_in_channels, kh, kw), dtype=old_weight.dtype)
    new_weight[:, :old_in_channels, :, :] = old_weight

    expanded = dict(state_dict)
    expanded[stem_key] = new_weight
    return expanded


class FusionSegNet(nn.Module):
    """EfficientNet-B0 encoder (from scratch) + ASPP bottleneck + U-Net
    decoder with Attention Gates + SE blocks on every skip + deep
    supervision auxiliary head at 1/8 scale. See module docstring for
    what changed vs. the source notebook."""

    def __init__(self, n_classes: int = N_CLASSES_DEFAULT, aspp_ch: int = 128, in_channels: int = N_INPUT_CHANNELS):
        super().__init__()
        eff = efficientnet_b0(weights=None)
        feats = eff.features

        # Ticket #28: "Change the first conv from 3 to 9 input channels.
        # Nothing else." feats[0] is the stem Conv2dNormActivation; its
        # first submodule is the Conv2d being resized -- every other
        # parameter (out_channels, kernel_size, stride, padding) is
        # copied from the original, unchanged.
        stem_conv = feats[0][0]
        assert isinstance(stem_conv, nn.Conv2d)
        feats[0][0] = nn.Conv2d(
            in_channels,
            stem_conv.out_channels,
            kernel_size=stem_conv.kernel_size,
            stride=stem_conv.stride,
            padding=stem_conv.padding,
            bias=stem_conv.bias is not None,
        )

        # EfficientNet-B0 feature stages (unchanged from the notebook):
        # 0: stem, 1: MBConv1 (16ch,H/2), 2: MBConv6 (24ch,H/4),
        # 3: MBConv6 (40ch,H/8), 4: MBConv6 (80ch,H/16),
        # 5-7: MBConv6 (320ch,H/32; feats[8] excluded)
        self.e1 = nn.Sequential(feats[0], feats[1])
        self.e2 = feats[2]
        self.e3 = feats[3]
        self.e4 = feats[4]
        self.e5 = nn.Sequential(*feats[5:8])

        self.aspp = ASPP(320, aspp_ch)
        self.aux_head = nn.Conv2d(40, n_classes, 1)

        self.ag4 = AttentionGate(aspp_ch, 80, 40)
        self.u5 = nn.ConvTranspose2d(aspp_ch, aspp_ch, 2, 2)
        self.d5 = _dblock(aspp_ch + 80, 96)

        self.ag3 = AttentionGate(96, 40, 32)
        self.u4 = nn.ConvTranspose2d(96, 96, 2, 2)
        self.d4 = _dblock(96 + 40, 64)

        self.ag2 = AttentionGate(64, 24, 24)
        self.u3 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.d3 = _dblock(64 + 24, 48)

        self.ag1 = AttentionGate(48, 16, 16)
        self.u2 = nn.ConvTranspose2d(48, 48, 2, 2)
        self.d2 = _dblock(48 + 16, 32)

        self.upf = nn.ConvTranspose2d(32, 32, 2, 2)
        self.clf = nn.Conv2d(32, n_classes, 1)

    def forward(self, x, return_attention: bool = False):
        """`return_attention=False` (the default) is BIT-FOR-BIT the
        original method -- every existing caller is unaffected.
        `return_attention=True` additionally returns a dict of the four
        decoder stages' own attention maps ("ag1".."ag4", finest to
        coarsest skip connection), each already a real [0, 1] per-pixel
        map the trained network produced -- not a synthetic/approximated
        saliency method bolted on afterward. Explainability tooling
        (eval/checkpoint_attention_overlay.py) is the only intended
        caller of this flag; training/inference call sites never pass it.
        """
        input_hw = x.shape[2:]
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        b = self.aspp(e5)
        aux = self.aux_head(e3)

        attention_maps = {}

        x = _match_size(self.u5(b), e4)
        if return_attention:
            e4a, attention_maps["ag4"] = self.ag4(b, e4, return_attention=True)
        else:
            e4a = self.ag4(b, e4)
        x = self.d5(torch.cat([x, e4a], 1))

        x = _match_size(self.u4(x), e3)
        if return_attention:
            e3a, attention_maps["ag3"] = self.ag3(x, e3, return_attention=True)
        else:
            e3a = self.ag3(x, e3)
        x = self.d4(torch.cat([x, e3a], 1))

        x = _match_size(self.u3(x), e2)
        if return_attention:
            e2a, attention_maps["ag2"] = self.ag2(x, e2, return_attention=True)
        else:
            e2a = self.ag2(x, e2)
        x = self.d3(torch.cat([x, e2a], 1))

        x = _match_size(self.u2(x), e1)
        if return_attention:
            e1a, attention_maps["ag1"] = self.ag1(x, e1, return_attention=True)
        else:
            e1a = self.ag1(x, e1)
        x = self.d2(torch.cat([x, e1a], 1))

        x = _match_size(self.upf(x), input_hw)  # final decode targets the ORIGINAL input resolution
        out = self.clf(x)

        if return_attention:
            if self.training:
                return out, aux, attention_maps
            return out, attention_maps
        if self.training:
            return out, aux  # deep supervision: both heads during training
        return out
