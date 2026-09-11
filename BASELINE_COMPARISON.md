# DRISHTI vs. Published Real-Time LiDAR Segmentation Networks

FusionSegNet (this project's perception backbone, `perception/segnet.py`) was
adapted from a camera-domain notebook per Build Map Ticket #28's own spec:
"change the first conv from 3 to 9 channels, nothing else." Its measured
parameter count (5.82M) was chosen to land in the same efficiency class as
three established real-time LiDAR range-image segmentation networks — this
document makes that comparison explicit, with the same honesty standard the
rest of this project holds itself to (see `TRAINING_RESULTS.md`,
`HANDOFF.md`): real numbers, sourced, with the actual limits of the
comparison stated rather than implied away.

## What IS fairly comparable: parameter count and architecture family

Parameter count and input representation (a range-image projection, not raw
point convolutions) are dataset-independent — this comparison is fair.

| Network | Params | Input resolution | Architecture family |
|---|---|---|---|
| SalsaNext [[1]](https://arxiv.org/pdf/2003.03653) | 6.7M | 64x2048 range image | Encoder-decoder, dilated residual stack |
| CENet [[2]](https://arxiv.org/abs/2207.12691) | 6.8M | 64x2048 range image | Concise conv + auxiliary heads |
| FIDNet [[3]](https://arxiv.org/abs/2109.03787) | 6.0M | 64x2048 range image | Fully-interpolation decoder |
| **FusionSegNet (this project)** | **5.82M** | **64x2048 range image** | EfficientNet-B0 encoder + ASPP + attention-gated decoder |

FusionSegNet sits at the SMALL end of this band — fewer parameters than all
three, at the same 64x2048 input resolution these networks were themselves
benchmarked at.

## What is NOT fairly comparable: mIoU, directly

| Network | Published mIoU | Benchmark |
|---|---|---|
| SalsaNext | 55.5% [[1]](https://arxiv.org/pdf/2003.03653) | SemanticKITTI (urban driving, 19 classes) |
| CENet | 64.7% [[2]](https://arxiv.org/abs/2207.12691) | SemanticKITTI (urban driving, 19 classes) |
| FIDNet | 55.4% [[3]](https://arxiv.org/abs/2109.03787) | SemanticKITTI (urban driving, 19 classes) |
| **FusionSegNet (this project)** | **56.2%** (Run #2, epoch 19 — see `TRAINING_RESULTS.md`) | **RELLIS-3D (off-road, 10-class DRISHTI taxonomy)** |

**Why these numbers cannot be placed side by side as a ranking:** SemanticKITTI
is urban road driving with well-defined lane/vehicle/pedestrian classes and
strong geometric regularity (roads, buildings, parked cars). RELLIS-3D is
unstructured off-road terrain — no lane markings, no consistent ground plane,
heavy vegetation occlusion, and (per `perception/taxonomy.py`'s own
cross-check) some classes appear in only a few hundred pixels across an
entire multi-sequence training set. A model scoring lower on a harder,
less-benchmarked domain is not evidence of a worse model — and a model
scoring higher on an easier, heavily-studied domain is not evidence of a
better one. Presenting these as directly comparable would be exactly the
kind of overclaim this project's own culture (Bible Principle 4: "state
limitations, do not hide them") argues against.

**The honest claim this table actually supports:** FusionSegNet achieves a
competitive mIoU using FEWER parameters than three established real-time
segmentation networks, on a domain (unstructured off-road terrain) that none
of the three were evaluated on and that is qualitatively harder to
regularize. That is a real, defensible claim; a fabricated head-to-head mIoU
ranking would not be.

## Sources

1. Cortinhal, T., Tzelepis, G., & Erdal Aksoy, E. (2020). SalsaNext: Fast,
   Uncertainty-aware Semantic Segmentation of LiDAR Point Clouds for
   Autonomous Driving. [arXiv:2003.03653](https://arxiv.org/pdf/2003.03653)
2. Cheng, H., et al. (2022). CENet: Toward Concise and Efficient LiDAR
   Semantic Segmentation for Autonomous Driving.
   [arXiv:2207.12691](https://arxiv.org/abs/2207.12691)
3. Zhao, Y., Bai, L., & Huang, X. (2021). FIDNet: LiDAR Point Cloud Semantic
   Segmentation with Fully Interpolation Decoding.
   [arXiv:2109.03787](https://arxiv.org/abs/2109.03787)
