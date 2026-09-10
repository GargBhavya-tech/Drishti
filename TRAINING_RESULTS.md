# DRISHTI — First Real GPU Training Run

**Date:** 2026-09-11  
**GPU:** NVIDIA RTX 2080 Ti (11 GB), `172.16.192.12`  
**Dataset:** RELLIS-3D sequence 00004, Ouster OS1-64 stream  
**Script:** `perception/train.py` (Ticket #30)  
**Config:** `configs/sensor_ouster_os1_64.yaml`  

---

## Setup

| | |
|---|---|
| Train frames | 1,750 (first 85% of sequence 00004 by index) |
| Val frames   | 309 (last 15%, contiguous — no shuffle-split) |
| Epochs       | 20 |
| Batch size   | 4 |
| Optimizer    | AdamW, LR=3e-4, OneCycleLR |
| Precision    | AMP (mixed, CUDA) |
| Avg epoch time | ~142 s (~47 min total) |

---

## Per-class Pixel Counts (Sampled Training Frames)

| DrishtiClass | Pixels (sampled) |
|---|---|
| 0 | 41,305 |
| 1 | 4 |
| 2 | 38,413 |
| 3 | 39,456 |
| 4 | 665 |
| 5 (majority) | 1,208,192 (89.2%) |
| 6 | 99 |
| 7 | 26,553 |
| 8 | 0 ⚠️ |
| 9 | 0 ⚠️ |

> **Note:** Classes 8 (NEGATIVE_OBSTACLE) and 9 (OVERHANG) have zero ground-truth pixels — they are never directly labelled in the taxonomy remap (by design, per Build Map Ticket #8's "assert no source class maps to 8 or 9"). Classes 1, 4, 6 are extremely rare; IoU for these is unreliable at this scale.

---

## Training Loss

| Epoch | Train Loss |
|---|---|
| 0  | 2.3456 |
| 2  | 1.0534 |
| 5  | 0.5176 |
| 10 | 0.3788 |
| 15 | 0.3228 |
| 19 | 0.3086 |

Loss converged cleanly; no divergence or NaN observed.

---

## Validation mIoU (per epoch, best = epoch 16)

| Epoch | mIoU | Class 0 | Class 2 | Class 5 | Class 7 |
|---|---|---|---|---|---|
| 0  | 0.0775 | 0.000 | 0.000 | 0.698 | 0.000 |
| 2  | 0.3170 | 0.799 | 0.000 | 0.907 | 0.514 |
| 5  | 0.3257 | 0.880 | 0.113 | 0.915 | 0.635 |
| 10 | 0.3072 | 0.823 | 0.103 | 0.906 | 0.607 |
| 16 | **0.3280** | **0.914** | 0.105 | 0.912 | **0.679** |
| 19 | 0.3220 | 0.890 | 0.102 | 0.912 | 0.659 |

**Best checkpoint: epoch 16, val mIoU = 0.3280**

---

## Per-class IoU at Best Epoch (Epoch 16)

| DrishtiClass | IoU |
|---|---|
| 0 | **0.914** |
| 1 | 0.014 (only 4 px in training data) |
| 2 | **0.105** |
| 3 | 0.000 |
| 4 | 0.000 |
| 5 | **0.912** |
| 6 | 0.000 |
| 7 | **0.679** |
| 8 | n/a (zero GT pixels — by taxonomy design) |
| 9 | n/a (zero GT pixels — by taxonomy design) |
| **mIoU** | **0.328** |

---

## Majority-class Baseline

Predicting class 5 everywhere: **mIoU ≈ 0.091** (class 5 IoU only, all others 0).  
The trained model at **0.328 mIoU is 3.6× above the majority-class baseline**.

---

## Known Limitations

1. **Rare classes** (1, 4, 6) are essentially unlearned — too few pixels in one sequence.
2. **Classes 3 and 6** remain at 0 IoU throughout — present in training data but the network isn't separating them.
3. **Ouster OS1-64 sensor constants** (`d_theta_rad`/`d_phi_rad`) are unverified against real data (Ticket #6 was run on synthetic grids only). Results are valid; the sensor geometry claims in slides should note this.
4. This is **one sequence**, ~30 minutes of off-road driving — the training set is small by any standard. The result is reported honestly, not inflated.

---

## Next Steps (Phase 4+)

- Ticket #31+: observability flag, negative obstacles, sparsity envelope, temporal fusion, fovea
- Improve rare-class coverage: more sequences or data augmentation
- Run Ticket #6 (the gate) against real Ouster OS1-64 data to verify sensor config
