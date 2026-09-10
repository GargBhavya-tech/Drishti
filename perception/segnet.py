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
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0

N_INPUT_CHANNELS = 9  # Ticket #27's assembled tensor channel count
N_CLASSES_DEFAULT = 10  # perception.taxonomy.DrishtiClass


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

    def forward(self, g, x):
        g_up = F.interpolate(self.W_g(g), size=x.shape[2:], mode="bilinear", align_corners=False)
        psi = F.relu(g_up + self.W_x(x), inplace=True)
        return x * self.psi(psi)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=128, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
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
        nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
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

    def forward(self, x):
        input_hw = x.shape[2:]
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        b = self.aspp(e5)
        aux = self.aux_head(e3)

        x = _match_size(self.u5(b), e4)
        e4a = self.ag4(b, e4)
        x = self.d5(torch.cat([x, e4a], 1))

        x = _match_size(self.u4(x), e3)
        e3a = self.ag3(x, e3)
        x = self.d4(torch.cat([x, e3a], 1))

        x = _match_size(self.u3(x), e2)
        e2a = self.ag2(x, e2)
        x = self.d3(torch.cat([x, e2a], 1))

        x = _match_size(self.u2(x), e1)
        e1a = self.ag1(x, e1)
        x = self.d2(torch.cat([x, e1a], 1))

        x = _match_size(self.upf(x), input_hw)  # final decode targets the ORIGINAL input resolution
        out = self.clf(x)

        if self.training:
            return out, aux  # deep supervision: both heads during training
        return out
