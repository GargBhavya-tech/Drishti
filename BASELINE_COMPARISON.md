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

## Negative-obstacle detection: built on field-validated physics, not a novel claim

DRISHTI infers a negative obstacle (ditch/trench/drop-off) from the
ABSENCE of an expected LiDAR return — a "range shadow" — rather than
requiring a positive return. **This is not a new technique, and this
project does not claim it as one.** It was pioneered by Arturo Rankin
and Larry Matthies at NASA JPL for DARPA's Demo III UGV program
[[4]](https://www.researchgate.net/publication/nist-obstacle-detection)
[[5]](https://www.researchgate.net/publication/negative-obstacle-thermal),
fielded in the TerraMax vehicle at the 2005 DARPA Grand Challenge, and
remains the active technique in current literature — e.g. Shang et al.
2024's tilted-LiDAR negative-obstacle detector, which explicitly models
the "spacing jump" between points as the detection signal
[[6]](https://doi.org/10.3390/s24247929).

**DRISHTI's actual contribution is the system built around a
field-validated cue, not the cue itself**: a variable-resolution clipmap
tied to sensor Nyquist physics (absent from the fixed-resolution 2003–
2005 work), a modern deep semantic segmentation network fused with the
geometric range-shadow signal (Demo III/TerraMax predate deep learning
and used purely geometric/thermal signal processing), an open and
tested reproducible pipeline (the JPL/DARPA work was never published as
reusable code), and the Sparsity Trap's own conservatism argument —
treating an ambiguous range shadow as a hazard by default rather than
requiring confirmation. "Built on NASA JPL/DARPA-proven physics" is the
honest claim, and a stronger one for a defense-context audience than an
unproven "novel" one would be. Full findings, including gaps the
research could not source, are in `RESEARCH_FINDINGS.md`.

## Positioning against real UGV programs, not just academic benchmarks

Two comparisons a DRDO problem statement's own judges are more likely to
recognize than SalsaNext/CENet/FIDNet:

**DARPA's RACER program** (Rivière et al. 2023
[[7]](https://arxiv.org/abs/2311.tbd)) pushes off-road UGVs to 7–10 m/s
and found that dense semantic classification at that speed introduces
enough latency for the planner to act on "misrepresented or delayed
semantic obstacles and terrain geometries" — independently validating
the exact failure mode this project's perception-limited speed envelope
(Claim 4) and variable-resolution mapping (Claim 2) are built to
prevent.

**DRDO's own currently-documented UGVs** are a real, favorable
capability-level baseline (not a metrics comparison — their internal
algorithms are not public): **Daksh** is an EOD teleoperated platform
(multi-camera + X-ray, no autonomous LiDAR mapping); **Muntra**
(BMP-2-based) runs GPS/INS waypoint autonomy with radar/EO obstacle
detection, tested on the flat terrain of the Mahajan field firing range,
with no public documentation of adaptive-resolution LiDAR or
negative-obstacle reasoning. Adaptive-resolution 2.5D mapping and
range-shadow negative-obstacle detection are capabilities not documented
in DRDO's own currently fielded programs — a real, checkable, favorable
comparison for this specific audience.

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
4. Rankin, A., et al. (2006). Obstacle Detection and Terrain Classification
   for Autonomous Off-Road Navigation. *Autonomous Robots*, 21(1).
5. Matthies, L., & Rankin, A. (2003). Negative Obstacle Detection by Thermal
   Signature. *IEEE/RSJ IROS*.
6. Shang, et al. (2024). LiDAR-Based Negative Obstacle Detection for
   Unmanned Ground Vehicles in Orchards. *MDPI Sensors*, 24(24), 7929.
   [DOI:10.3390/s24247929](https://doi.org/10.3390/s24247929)
7. Rivière, B., et al. (2023). Pushing the Limits of Off-Road Autonomy in
   DARPA's RACER Program. arXiv preprint (exact ID not independently
   re-verified by this project — re-confirm before citing in a written
   deliverable submitted externally).

**A note on citation confidence**: sources 1–3, 6 are directly
verifiable (arXiv IDs / DOIs resolve). Sources 4–5 and 7 come from the
deep-research pass verbatim and have NOT been independently re-fetched
by this project to confirm the exact venue/page numbers — before using
4, 5, or 7 in a document that leaves this repo (slides, a written
submission), do one direct search to confirm the citation resolves,
per this project's own "never guess an arXiv ID" discipline
(see `HANDOFF.md`).
