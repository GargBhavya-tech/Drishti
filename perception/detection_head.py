"""
perception/detection_head.py

Lightweight detection head sharing FusionSegNet's existing decoder
features (via FusionSegNet.forward(..., return_features=True) --
perception/segnet.py) -- deliberately NOT a separate 3D detector model.
Built following the deep-research report's two strongest, most
actionable findings for this project's exact situation:

1. Wilson et al. 2024 ("What Matters in Range View 3D Object
   Detection"): a shared backbone with enough feature capacity + a
   SIMPLE loss (plain classification + smooth-L1 regression) matches or
   beats architecturally-complex heads with multi-resolution pyramids,
   IoU losses, or ensembling. This head is deliberately small: two
   convs, one output layer -- complexity was NOT added "to be safe".

2. SVM (2024)'s View Adaptive Regression (VAR) insight -- objectness/
   classification and geometric regression are DIFFERENT learning
   problems and benefit from being decoupled, but for a first version
   at this project's scale a single shared trunk with a wide-enough
   final layer is the pragmatic middle ground; VAR's full split (cross-
   range offsets processed differently from depth) is named here as a
   real, un-taken option for a v2, not silently assumed unnecessary.

Output channels, per pixel: 1 (objectness logit) + 8 (regression: dx,
dy, dz, w, l, h, sin_yaw, cos_yaw -- see perception/nuscenes_boxes.py's
own docstring for the exact target convention these regress against).
Classification of WHICH DrishtiClass a positive pixel belongs to is
NOT duplicated here -- it's read directly from FusionSegNet's own
existing segmentation output at that pixel (Principle 3: "reuse what is
already built and tested" -- a second classification head predicting
the same taxonomy the segmentation head already predicts would be
redundant and could disagree with it in a way nothing resolves).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from perception.nuscenes_boxes import N_REGRESSION_CHANNELS
from perception.segnet import CircularConv2d

DETECTION_HEAD_IN_CHANNELS = 32  # FusionSegNet's final decoder feature width -- see segnet.py's `decoder_features`
DETECTION_HEAD_OUT_CHANNELS = 1 + N_REGRESSION_CHANNELS  # objectness + 8 regression targets


class DetectionHead(nn.Module):
    """Two CircularConv2d layers (same horizontal-wraparound convention
    as every other conv this project defines directly, per Bible Part
    4.5) + a final 1x1 projection. Small on purpose -- see module
    docstring's Wilson et al. citation."""

    def __init__(self, in_channels: int = DETECTION_HEAD_IN_CHANNELS, hidden_channels: int = 32):
        super().__init__()
        self.conv1 = CircularConv2d(in_channels, hidden_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = CircularConv2d(hidden_channels, hidden_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.out = nn.Conv2d(hidden_channels, DETECTION_HEAD_OUT_CHANNELS, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, decoder_features: torch.Tensor) -> torch.Tensor:
        """decoder_features: (B, 32, H, W) from FusionSegNet's
        return_features=True. Returns (B, 9, H, W): channel 0 =
        objectness logit, channels 1-8 = regression (dx,dy,dz,w,l,h,
        sin_yaw,cos_yaw), all raw (un-activated) -- sigmoid/interpretation
        happens in perception.detection_loss and at inference/decode
        time, not inside this module."""
        x = self.relu(self.bn1(self.conv1(decoder_features)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.out(x)
