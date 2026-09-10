# DRISHTI — The Complete Technical Bible (v3)

**D**istance-**R**esolved **I**nstantaneous **S**emantic **H**eight & **T**raversability **I**maging

*Smart India Hackathon 2026 · Problem Statement 26053 · DRDO / Department of Defence Production (IDEX) · Category: Software · Theme: Smart Vehicles*

*Written so someone with zero prior context can read this end to end and understand exactly what the system does, why every design decision was made, the exact math behind every module, worked numerical examples, every edge case and how it's handled, and how to independently test each piece.*

> **What changed from v1.** v2 merged the polar range-image front end (residual motion channels, circular padding, the FusionSegNet backbone reuse, the Lovász loss stack, the Sparsity Trap) with v1's world-anchored Cartesian clipmap, sensor-derived resolution schedule, multi-layer cells, and observability model. Three corrections were made in that merge and are argued in place: the ground filter **labels** rather than **strips** (Part 5), the continuous-polar grid is a *front end* rather than *the map* (Part 7), and the Sparsity Trap's threshold is **derived from beam geometry** rather than tuned (Part 11).
>
> **What changed in v3.** Sixteen additions, of which three are structural and get their own parts: the **perception-limited speed envelope** (Part 15) turns the derived detection ranges into an operational speed limit; the **conservatism invariant** (Part 16) makes "uncertainty always resolves toward caution" a machine-checked property rather than a stated principle; and **degraded-comms operation** (Part 21) shows the mipmap giving progressive level-of-detail transmission for free. The rest are folded into existing parts: extrinsic self-calibration (3.6), occlusion-depth channels and an IMU-gated residual threshold (4.3–4.4), the Meta-Kernel stem (5.5), multi-echo vegetation depth (6.1), foveated height quantisation and incidence-corrected albedo (9.3, 9.5), `INFERRED` geometric completion and per-track residual consistency (12.5–12.6), latency compensation (20), and a 32-beam portability ablation (18.1).
>
> **Four things were considered in v3 and deliberately rejected** — generative terrain inpainting, runtime weight adaptation, general monocular-depth redundancy, and radar fusion. Part 22 gives the reasons, and two of them are reasons worth saying out loud to a judge.

---

## Part 0 — What Problem Is This Solving, In Plain English

A LiDAR on a vehicle fires laser beams in all directions and measures how long each takes to return. Do that a hundred thousand times per rotation, ten times a second, and you get a **point cloud**: roughly a million 3D dots per second describing the world around the vehicle.

There are two obvious things to do with that, and both are wrong:

1. **Keep it all in 3D.** Perfectly accurate, completely unusable. You cannot process a million points a second on a vehicle computer, and you cannot store a fine 3D grid of a 200-metre neighbourhood without gigabytes.
2. **Flatten it to a 2D occupancy grid** — "is something here: yes/no". Cheap, fast, and it throws away height. A 12 cm kerb, a 40 cm ditch, and a branch hanging at 1.9 m all vanish. For a wheeled ground vehicle those three things *are* the problem.

The middle path the statement asks for is **2.5D**: a flat grid of cells, each remembering height as well as occupancy, at **variable resolution** — fine near the vehicle where safety depends on it, coarser further out where it does not.

DRISHTI answers six questions, in order, before handing anything to a planner:

1. **What is physically knowable here?** — what does the sensor's beam geometry permit at this range? (Part 3)
2. **What is each point, and is it moving?** — terrain, static obstacle, dynamic object; parked or driving? (Parts 5–6)
3. **Where should detail live?** — how big should a cell be here, and why that size? (Parts 8, 12)
4. **What does a cell actually contain?** — not "how tall", but the vertical story: ground, gap, ceiling, confidence. (Part 9)
5. **What did we fail to see?** — because a hole in the ground is defined by returns that never came back. (Part 10)
6. **When does "no returns" stop meaning "nothing there"?** — because past a computable range, silence is not evidence of absence. (Part 11)
7. **How fast may we go, given all of the above?** — because a vehicle that cannot stop within its detection range is driving blind regardless of how good the map is. (Part 15)

Questions 5 and 6 are the ones almost nobody asks, and between them they cover the two hazards most likely to kill a ground vehicle: the ditch you drove into, and the pole you never saw. Question 7 is what makes the answers operational instead of academic.

### The four claims this project stands on

**Claim 1 — The naive reading of the problem statement is unimplementable.** It asks for *overhang detection* and a *2.5D elevation map* in one breath. A single height per cell **cannot** represent "drivable road under a bridge": max-height reports an impassable wall, min-height drives you into the branch, mean-height invents a wall at head height. Part 9 fixes it with multi-layer cells. Naming this contradiction and resolving it is worth more than any accuracy number.

**Claim 2 — The resolution numbers in the statement are not arbitrary; they are the sensor's own sampling limit.** Part 3 derives them from four lines of beam geometry. The schedule becomes *defensible* rather than *chosen*, regenerates for any sensor from a config file, and strictly exceeds what was specified. As a corollary it also shows the statement's own checkpoints (5 cm → 50 cm across 10 m → 100 m) are **linear in range**, which is exactly what constant angular resolution produces — the statement is quoting physics without saying so.

**Claim 3 — "No returns" is not "no obstacle", and the boundary is computable.** Thin objects at range — poles, cyclists, fence posts — fall below the sensor's sampling density and are silently smoothed away by every system that treats sparsity as noise. Part 11 computes the expected return count for the smallest object we care about, per cell, per range, and reports `UNKNOWN` rather than `FREE` past the point where absence stops being informative. This is the piece with no standard approach in the literature, and the sensor model is what turns it from a tuned heuristic into a derived quantity.

**Claim 4 — The sensor, not the vehicle, is the speed limit — and we can compute it.** Detection range and stopping distance together define the fastest speed at which a vehicle can still stop for what it can see. Part 15 derives it: with this sensor at this mount height, a UGV cannot safely exceed **43 km/h** in terrain where a 2 m ditch is possible, because it cannot see one far enough ahead to stop. That single number is the whole perception analysis expressed in terms the sponsor operates in, and it converts the sensor recommendation from an opinion into a costed trade-off: doubling beam count buys 21% more speed, raising the mount 77 cm buys 11%.

Claims 1–3 make the map honest. Claim 4 makes the honesty *actionable*, and Part 16 makes it *checkable*: the conservatism invariant is a property test asserting no code path in the system can turn `UNKNOWN` into `FREE`.

---

## Part 1 — Design Philosophy: Six Principles Behind Every Decision

**Principle 1 — Solve problems structurally, not with careful code.** Any design that relies on "handle the boundary case carefully" eventually leaks an edge case under time pressure. Prefer representations where the failure mode *cannot occur*. The clipmap's power-of-two nesting (Part 8) does not handle alignment error; it removes the conditions for it. The fovea controller's `min()` composition (Part 12) does not avoid violating the spec; it makes violation impossible.

**Principle 2 — Let the sensor's native geometry do the work.** A rotating LiDAR already samples the world at constant angular resolution, so the ground footprint of one return grows with range automatically. Do not fight that with a uniform Cartesian grid and bolt foveation back on afterwards. Work in the sensor's coordinate system for perception (Part 4), and derive the map's resolution schedule from the same geometry (Part 3).

**Principle 3 — Reuse what is already built and tested.** Where the existing FusionSegNet codebase (EfficientNet-B0 + ASPP + Attention U-Net with SE blocks, confidence-weighted loss, deep supervision, ground-plane extraction) adapts, adapt it rather than build new architecture families under time pressure. Every deviation from reuse below is justified by a specific documented failure mode, not by preference.

**Principle 4 — State limitations; do not hide them.** Every real system in this space has a named, published failure mode. Identify ours explicitly (Parts 21–22) and either mitigate or measure it. A panel that finds a hidden weakness stops believing everything else; a panel that is *shown* the weakness believes the rest.

**Principle 5 — Every number must be derived or measured, never fitted to look right.** A constant that reproduces the two checkpoints you were given proves nothing — reproducing them is what the fit was for. Prefer one parameter with a physical meaning over two tuned to endpoints. Where a threshold must be empirical, calibrate it against ground truth and say which ground truth.

**Principle 6 — Never invent information, and make that checkable.** Everything the map asserts is measured or derived; nothing is generated. Where information is missing the map says so, and where it is inherited rather than measured it is flagged (`PROVISIONAL`, `INFERRED`) and forbidden from reducing cost. This is not caution for its own sake — it is what lets Part 16 state a *monotonicity property* over the whole system and test it with ten thousand generated cases. It is also why v3 rejects generative terrain completion and runtime weight adaptation (Part 22): both are capable techniques that would trade this property away, and the property is what makes the rest credible to a defence sponsor.

---

## Part 2 — The Full Pipeline, At a Glance

```
LiDAR sweep arrives (~120k points, 10 Hz)  +  K=4 previous sweeps  +  ego pose
        │
        ▼
LAYER 0 — Sensor model (Part 3)
   Beam geometry → resolution schedule, hazard spec sheet,
   expected-return curves. Everything numeric starts here.
        │
        ▼
LAYER 1 — Ingest & polar front end (Part 4)
   Deskew → ego-compensate K past sweeps → spherical projection
   → 64×2048×14 tensor:  x,y,z,range,intensity,valid_mask,
     ground_prior, 4×motion residual, range_gradient,
     occlusion_count, occlusion_spread
   Circular padding at the 360° seam. Foveated column stride.
   Residual threshold gated by |ω| × ‖∇r‖ (drift suppression).
        │
        ▼
LAYER 2 — Segmentation (Part 5)         LAYER 2b — Ground prior (Part 5.2)
   FusionSegNet adapted:                   Column-wise incremental walk.
   Meta-Kernel stem (relative-coord         LABELS points, never strips them.
   dynamic weights) →                       Feeds Layer 2 as a channel AND
   EfficientNet-B0 → ASPP →                 Layer 10 as the ground reference.
   Attention U-Net + SE                     Multi-echo: first/last return
   Loss: Lovász + conf-CE + deep sup.       → vegetation depth, bare earth.
        │                                         │
        └────────────────┬────────────────────────┘
                         ▼
        LAYER 3 — Taxonomy (Part 6): dataset classes → vehicle classes.
        Class proposes; residuals + tracker dispose.
                         ▼
        LAYER 4 — The Cartesian clipmap (Part 8)
        Nested power-of-two levels, toroidal, world-anchored.
        Alignment error IMPOSSIBLE by construction.
                         ▼
        LAYER 5 — Cell aggregation (Part 9): GPU scatter, multi-layer
        cells (ground / gap / ceiling), height distribution, clearance,
        foveated height quantisation, incidence-corrected albedo
        (→ the flat-but-lethal case: standing water, wet mud)
                         ▼
        LAYER 6 — Observability (Part 10): ray-casting,
        UNKNOWN / FREE / OCCUPIED / OCCLUDED, negative obstacles
                         ▼
        LAYER 7 — Sparsity Trap & derived confidence (Part 11)
        Expected return count vs. observed. Past r_blind,
        silence is reported as UNKNOWN, never FREE.
                         ▼
        LAYER 8 — Temporal fusion (Part 12): static layer accumulates;
        dynamic objects tracked as entities (no smear); resolution
        promotion/demotion under ego-motion
                         ▼
        LAYER 9 — Adaptive fovea (Part 13): allocate by TIME-TO-CONTACT.
        Can only refine, never coarsen below Layer 0's floor.
                         ▼
        LAYER 10 — Traversability & planner interface (Part 14)
                         ▼
        LAYER 11 — Perception-limited speed envelope (Part 15)
        Detection range + stopping distance → the fastest speed
        at which the vehicle can still stop for what it can see
                         ▼
   LAYER 12 — Dashboard (Part 17)    LAYER 13 — Eval harness (Part 18)

        ┌──────────────────────────────────────────────────┐
        │ THE CONSERVATISM INVARIANT (Part 16) applies      │
        │ across every layer above: degrading information   │
        │ may never decrease reported cost. Property-tested.│
        └──────────────────────────────────────────────────┘
```

### The four architectural boundaries, enforced deliberately

These are the invariants. Blurring any of them is how this design fails.

1. **The network never invents a hazard.** `NEGATIVE_OBSTACLE` and `OVERHANG` come from geometry (Parts 9–10), never from a predicted class. A network can hallucinate a label; it cannot hallucinate a missing laser return.
2. **The adaptive fovea can only refine, never coarsen.** Layer 0's schedule is a hard floor (Part 13). No bug in the clever feature can break the specified requirement.
3. **The ground filter labels; it never strips.** Terrain points must reach the elevation map — they *are* the elevation map (Part 5.2).
4. **Levels are alternative views, never summed.** One query returns one cell from one level (Part 8).
5. **Nothing is generated.** Every value is measured, or derived from measurements by a stated formula, or inherited from a coarser measurement and flagged as such. There is no code path that invents geometry, and Part 16 tests that claim rather than asserting it.

---

## Part 3 — Layer 0: The Sensor Model (where every number in this project comes from)

**Files:** `sensor/sensor_model.py`, `sensor/schedule.py`
**One-line goal:** describe the beam geometry once, and derive from it — not from taste — the resolution schedule, the level boundaries, the hazard detection ranges, and the expected-return curves that Part 11 needs.

### Why this needs to exist at all

Every other team will hardcode "5 cm inside 10 m, 50 cm out to 100 m" because the statement said so. The first question a DRDO evaluator asks is *"why those numbers?"*, and "because the statement said so" is not an engineering answer. Worse, the design then breaks silently the moment the sensor changes — and DRDO will not be fielding the exact sensor SemanticKITTI was recorded with.

Layer 0 makes the schedule a **derived quantity**. Change the sensor description, and level boundaries, memory budget, hazard spec sheet, and confidence thresholds all move consistently.

### The sensor description

Four numbers, all on any datasheet:

| Symbol | Meaning | HDL-64E (SemanticKITTI) | Ouster OS1-128 |
|---|---|---|---|
| $\Delta\theta$ | azimuth angular step | $0.1728° = 3.016\times10^{-3}$ rad | $0.35°$ or $0.176°$ |
| $\Delta\phi$ | elevation (beam-to-beam) step | $26.8°/63 = 0.4254° = 7.424\times10^{-3}$ rad | $45°/127 = 0.354°$ |
| $\phi_{\max}$ | highest beam elevation | $+2°$ | $+22.5°$ |
| $h$ | sensor height above ground | $1.73$ m | vehicle-specific |

**Verify these against your own sensor before quoting any number below.** Every figure here derives from the HDL-64E column; they are the right *shape* for any sensor and the wrong *values* for a different one.

### The five formulas — the intellectual spine of the whole project

**1. Tangential (across-track) spacing** between adjacent points in one beam ring:

$$s_t(r) = r \cdot \Delta\theta$$

**2. Radial (along-track) ground spacing** between consecutive rings on flat ground. The one everybody forgets. A beam at depression $\alpha$ strikes ground at $r = h/\tan\alpha$; differentiating and substituting $\tan\alpha \approx h/r$ for distant rings:

$$s_r(r) = \frac{h\,\Delta\phi}{\sin^2\alpha} \approx \frac{r^2\,\Delta\phi}{h}$$

**3. Vertical spacing on a vertical surface** (wall, pole, person):

$$s_v(r) = r \cdot \Delta\phi$$

**4. Range at which a feature of vertical extent $t$ still receives a beam ring** — the constraint that governs *resolving height* (a kerb, a step, an overhang's underside):

$$r_{\max}(t) = \frac{t}{\Delta\phi}$$

**5. Range at which a negative obstacle of width $w$ is still straddled by a ring** — you need at least one ring to fall *inside* the gap:

$$s_r(r) \le w \quad\Longrightarrow\quad r_{\max}(w) = \sqrt{\frac{w\,h}{\Delta\phi}}$$

A sixth formula — the expected return count $N_{\exp}(r)$, which governs *detecting presence* rather than *resolving height* — follows from formulas 1 and 3 and is developed in Part 11 where it is used.

### What these numbers actually say (HDL-64E, $h = 1.73$ m)

| Range | Tangential $s_t$ | Radial ground $s_r$ | Vertical $s_v$ |
|---|---|---|---|
| 10 m | 3.0 cm | **0.43 m** | 7.4 cm |
| 20 m | 6.0 cm | **1.72 m** | 14.8 cm |
| 50 m | 15.1 cm | **10.7 m** | 37.1 cm |
| 100 m | 30.2 cm | **42.9 m** | 74.2 cm |

Read the middle column again. **At 100 m, consecutive beam rings land 43 metres apart on flat ground.** Terrain beyond roughly 50 m is not sampled in any meaningful sense by a 64-beam sensor at this height.

Three consequences that shape everything downstream:

- **A uniform 5 cm grid at 100 m is not merely wasteful — it is unfillable.** Cells there can never receive a point. That is a proof, not an argument about taste.
- **The far field is an object problem, not a terrain problem.** Vertical structures are sampled at $r\Delta\phi$ (74 cm at 100 m — a wall or a person still returns several rings) while the ground is sampled at $r^2\Delta\phi/h$ (43 m — nothing). So coarse rings should be *scored on objects* and fine rings *on terrain*. Evaluate them that way (Part 18) instead of pretending 100 m terrain accuracy means anything.
- **Sampling is wildly anisotropic** — 30 cm tangentially, 43 m radially, at the same point. Square cells are an approximation, and an honest document says so. (Anisotropic cells are Future Work; square cells keep the indexing exact, which is worth more.)

### Deriving the resolution schedule — and confirming the statement's own numbers

A cell should match the sensor's in-plane footprint: small enough not to blur real detail, large enough not to be mostly empty. Using tangential spacing (the finer of the two in-plane spacings, so we never under-resolve):

$$c(r) = r \cdot \Delta\theta$$

**This is linear in range.** Now look at what the problem statement asked for: 5 cm at 10 m, 50 cm at 100 m — a 10× size change across a 10× range change. **The statement's own checkpoints are linear in range**, which is precisely what constant angular resolution produces. The statement is quoting beam geometry without saying so.

That matters for a reason beyond elegance. A schedule of the form $c(\rho) = a + k\rho^2$, fitted to those two checkpoints, has two constants and neither has a physical meaning — and reproducing the checkpoints proves nothing, since reproducing them is what the fit was *for*. It also costs real resolution:

| Range | Fitted $0.05 + 4.5\!\times\!10^{-5}\rho^2$ | Sensor limit $r\Delta\theta$ | |
|---|---|---|---|
| 10 m | 5.45 cm | 3.02 cm | **1.8× coarser than the sensor delivers** |
| 25 m | 7.81 cm | 7.54 cm | ≈ matches |
| 50 m | 16.3 cm | 15.1 cm | ≈ matches |
| 100 m | 50 cm | 30.2 cm | 1.66× coarser |

It throws away resolution in the safety-critical near field, and again at range, and matches only in the middle where nobody asked. **One parameter with a physical meaning beats two fitted to endpoints** (Principle 5).

Quantising the linear law onto power-of-two levels (Part 8 explains why powers of two):

$$c(r) = c_0 \cdot 2^{\lceil \log_2(r\Delta\theta / c_0)\rceil}, \qquad c_0 = 5\ \text{cm} \qquad\Longrightarrow\qquad r_\ell = \frac{c_\ell}{\Delta\theta}$$

| Level | Cell size | Sensor-derived outer radius |
|---|---|---|
| L0 | 5 cm | 16.6 m |
| L1 | 10 cm | 33.2 m |
| L2 | 20 cm | 66.3 m |
| L3 | 40 cm | 132.6 m |

The statement asked for 5 cm within 10 m; the derivation gives 5 cm within **16.6 m**. It asked for 50 cm out to 100 m; the derivation gives 40 cm out to **132 m**. It strictly dominates in both directions, and was not tuned to — it fell out of $\Delta\theta$.

That is the slide: *"We did not pick these numbers. We derived them from beam geometry; they reproduce and exceed the requirement; and they regenerate for any sensor from a config file."*

### Worked example — the hazard spec sheet

The table a DRDO evaluator actually reads, straight from formulas 4 and 5:

| Hazard | Governing rule | Max detection range |
|---|---|---|
| 15 cm kerb (vertical face) | $t/\Delta\phi = 0.15/0.007424$ | **20.2 m** |
| 30 cm kerb / low wall | $0.30/0.007424$ | **40.4 m** |
| 1 m wide trench | $\sqrt{wh/\Delta\phi}$ | **15.3 m** |
| 2 m wide ditch | $\sqrt{2\times1.73/0.007424}$ | **21.6 m** |
| 4 m wide crater | $\sqrt{4\times1.73/0.007424}$ | **30.5 m** |
| 40 cm thick branch (overhang) | $0.40/0.007424$ | **53.9 m** |
| 5 cm cable (overhang) | $0.05/0.007424$ | **6.7 m** |

Two of these should be said out loud rather than buried:

- **A hanging cable is effectively undetectable beyond ~7 m** with this sensor. Physics, not a bug. Stating it is what makes the other numbers believable.
- **Overhang range also depends on $\phi_{\max}$.** An HDL-64E looks only 2° above horizontal. If reliable overhang detection matters to the platform, the sensor recommendation changes — which is a *procurement finding produced by your analysis*, worth more than a tenth of a point of mIoU.

These are **derived predictions**. Part 18 measures them in CARLA against known-geometry hazards; predicted-vs-measured is one of the strongest results available in this project.

### 3.6 Extrinsic self-calibration — the silent killer

Everything above assumes the sensor's mounting pose is known. It usually isn't, quite. And a small error here is catastrophic in a way that is completely invisible on screen.

A mount **pitch or roll error** of $\epsilon$ radians tilts the entire perceived ground plane, producing a height error that grows linearly with range:

$$\Delta z(r) = r \cdot \epsilon$$

| Mount tilt error | Height error @20 m | @50 m |
|---|---|---|
| 0.2° | 7.0 cm | 17.5 cm |
| 0.5° | 17.5 cm | **43.6 cm** |
| 1.0° | 34.9 cm | 87.3 cm |

**Read that against the hazard spec sheet.** We are trying to resolve a 15 cm kerb and a 20 cm step. A half-degree mounting error — well inside what you get from a hand-measured bracket — produces a 44 cm phantom slope at 50 m, larger than every hazard we claim to detect. It will not look like an error; it will look like gently sloping terrain, and the traversability layer will confidently cost it.

**The fix is nearly free, and it uses machinery we already have.** Over many frames, the ground surface a vehicle drives on is level *on average* — not flat, but unbiased. So:

1. Take the ground points the column-wise prior (Part 5.2) already confirms, across a rolling window of a few hundred frames.
2. Fit a plane to the aggregate in the vehicle frame.
3. Its normal's deviation from $\hat z$ is the accumulated **mount pitch and roll error**, and its offset is the true sensor height $h$.
4. Feed both back into `SensorModel` — which then regenerates the schedule, the ground formulas, and the spec sheet consistently, because they all take $h$ as an argument (Principle 5).

Two disciplines make this safe. **Update slowly** — an EMA over minutes, not frames, because a genuine long uphill would otherwise be absorbed as calibration error. And **display it**: put the estimated pitch, roll and $h$ on the HUD. A number that drifts is either a loosening bracket or a bug in the estimator, and either way you want to see it before a judge does.

This also quietly strengthens Claim 2: a schedule derived from beam geometry is only as good as the geometry you believe you have, so the system measures its own extrinsics rather than trusting a config file written by hand.

### Edge cases and how they're fixed

| Edge case | What would go wrong | The fix |
|---|---|---|
| Non-uniform beam spacing (most real sensors are denser near the horizon) | A single $\Delta\phi$ misrepresents the sensor; the spec sheet is wrong | `SensorModel` stores the **per-beam elevation table** from the datasheet; $\Delta\phi$ is looked up per ring, the constant used only for headline figures and labelled as such |
| Different mount height | Every ground formula changes as $h$ or $\sqrt{h}$ | $h$ is a config field, never a literal |
| Vehicle pitches (braking, slope) | $h$ and depression angles both change, invalidating expected-range math | Part 10 fits a local ground plane from recent confirmed returns; $h$ is used only for the static spec sheet |
| Solid-state / non-repetitive LiDAR (Livox) | No fixed $\Delta\theta$/$\Delta\phi$; the formulas do not apply | `empirical_spacing(r)` path measures nearest-neighbour spacing vs. range from real data; the schedule generator consumes either source |
| Quoting the spec sheet as measured performance | Overclaiming — these are geometric upper bounds | Every entry carries a `source` field (`derived` / `measured`); the dashboard renders them differently and never mixes them in one column |

### How to test this module

`pytest tests/test_sensor_model.py tests/test_schedule.py`. Pure functions over a config: feed the HDL-64E description, assert level boundaries at 16.6 / 33.2 / 66.3 / 132.6 m and the 2 m ditch range at 21.6 m.

The **real** validation is empirical and comes first, before anything else is built: take one real scan, bin points by range, measure actual nearest-neighbour spacing per bin, and plot against the $r\Delta\theta$ and $r^2\Delta\phi/h$ curves. **If measured spacing does not sit on those curves, your sensor constants are wrong and every number downstream is wrong.** `eval/point_distribution.py` does this in about forty lines and it is the highest-value hour in the whole build.

---
## Part 4 — Layer 1: Ingest and the Polar Front End

**Files:** `sensor/ego_motion.py`, `perception/range_image.py`, `perception/residuals.py`, `grid/foveated_voxelize.py`
**One-line goal:** turn a raw, motion-smeared sweep into a clean 2D tensor in the sensor's own coordinate system, carrying every signal the network needs — including motion — as ordinary input channels.

### 4.1 Why a range image, and why it is the *geometrically* correct choice

| Paradigm | Typical throughput | Why not chosen |
|---|---|---|
| Point-based (PointNet++, KPConv) | ~11–15 FPS | $O(N\log N)$ neighbour search dominates; degrades badly at long range where points are sparse — exactly where our low-confidence far cells need help, not more noise |
| Voxel / sparse conv (SPVNAS, Cylinder3D) | ~6–10 FPS | Dynamic hash-table construction every frame; on embedded hardware kernel-launch and allocation overhead alone threatens the budget. Also a well-known source of CUDA toolchain conflicts — **do not spend two days of a build window on a build system** |
| **Range image (ours)** | ~25–130 FPS | Dense 2D convolutions run near peak on any GPU or Jetson Tensor Core |

*(Throughputs are literature ballparks across varying hardware — measure your own, and see Part 20 on why FPS above the sensor rate is not the metric that matters.)*

The speed argument is real but secondary. The **geometric** argument is why this is correct rather than merely fast: a rotating LiDAR samples at constant angular resolution, so the ground footprint of one range-image pixel grows with range automatically. That is *the same foveation* Part 3 derived for the map. The front end and the resolution schedule are therefore expressions of one physical fact, not two systems that must be reconciled.

The price is a known weakness on thin structures and boundary bleed at depth discontinuities — declared in Part 24, and partly what Part 11 exists to catch.

### 4.2 Deskew — do this first or the fine ring is worthless

A spinning LiDAR takes 100 ms per rotation. At 15 m/s the vehicle moves **1.5 m** during one "single scan", so its first and last points were measured from positions 1.5 m apart. Treating them as simultaneous smears every wall in the scene.

Each point carries a sweep-relative timestamp $t_i$. Given poses at sweep start and end:

$$T_i = \text{slerp/lerp}\big(T_{\text{start}}, T_{\text{end}}, u_i\big), \qquad u_i = \frac{t_i - t_{\text{start}}}{t_{\text{end}} - t_{\text{start}}}$$

Transform every point into the sweep-end frame — SLERP for rotation, linear for translation.

**Why this is in the bible and not treated as boilerplate:** the entire value proposition is 5 cm cells. A 1.5 m smear is thirty cells wide. Skipping deskew makes the fine ring *worse than useless* — it costs the memory of 5 cm resolution while delivering 1.5 m accuracy. If you build one thing from this part, build this.

### 4.3 The projection and the fourteen channels

Spherical projection to a $64 \times 2048$ tensor:

$$u = \left\lfloor \tfrac{1}{2}\Big[1 - \tfrac{\arctan2(y,x)}{\pi}\Big] W \right\rfloor, \qquad v = \left\lfloor \Big[1 - \tfrac{\arcsin(z/r) - \phi_{\min}}{\Delta\phi_{\text{fov}}}\Big] H \right\rfloor$$

| # | Channel | Why it is needed |
|---|---|---|
| 1–3 | `x, y, z` | Standard convolutions assume translation invariance — that adjacent pixels have a constant physical relationship. LiDAR's non-uniform vertical beam spacing violates that. Explicit 3D coordinates let the network learn spatially-varying behaviour instead of assuming uniform geometry it does not have |
| 4 | `range` | The core depth signal |
| 5 | `intensity` | Material cue — lane paint, metal poles, wet surfaces |
| 6 | `valid_mask` | 1 if a real return, 0 if empty (beyond max range, absorbed, specular loss). **Without it the network reads an empty zero pixel as an object at the sensor origin** — a common structural hallucination in naive implementations, and the input-level twin of Part 10's `UNOBSERVED ≠ FREE` |
| 7 | `ground_prior` | Layer 2b's geometric ground label (§5.2) as a hint, *not* as a filter |
| 8–11 | `residual` × 4 | Motion evidence from the previous $K=4$ sweeps (§4.4) |
| 12 | `range_gradient` | Local $\|\nabla r\|$ across the pixel neighbourhood. Cheap, and it is what §4.4's drift mitigation and Part 11's structure test both consume |
| 13 | `occlusion_count` | How many returns were **discarded** at this pixel by the many-to-one projection (§4.3.1) |
| 14 | `occlusion_spread` | Deepest discarded return minus the kept return — the actual depth of what is hidden behind this pixel |

#### 4.3.1 Occlusion depth — turning a projection artifact into a feature

Spherical projection has a many-to-one problem: when a foreground pedestrian and a background building fall in the same angular bin, the standard implementation keeps the nearest return and throws the rest away. That discard is normally treated as unavoidable loss.

It is free information. You are already iterating those points to find the nearest; counting the rest costs a `scatter_add`.

A pixel with `occlusion_count = 0` sits on a solid surface with nothing behind it. A pixel with `occlusion_count = 5` has five returns hidden directly behind it — which is, almost by definition, an **object silhouette edge**. That is an explicit geometric boundary cue handed to the Attention U-Net for nothing.

**Count alone is ambiguous, which is why there are two channels.** Five discarded returns 20 cm behind the kept one is a thick or steeply-angled surface. Five discarded returns 40 m behind it is a true silhouette against a distant building. Identical count, completely different meaning. `occlusion_spread` is what separates them, and it is the channel that actually localises boundaries.

**This strengthens a rejection we already made.** Part 22 rejects KNN boundary refinement because it costs up to ~46% of inference time. Occlusion depth gives the network a boundary cue for a rounding error of that cost. The rejection stops reading as a compromise forced by the budget and starts reading as a substitution — we did not skip boundary handling, we did it in the projection instead of in post-processing.

Note it is complementary to `range_gradient`, not redundant: the gradient measures depth change *across* neighbouring pixels; occlusion depth measures what is hidden *within* one. A thin pole against a distant background has a large occlusion spread at its edges and a large gradient; a pixel on a smooth angled wall has a gradient but no occlusion. Both signals, different failure modes.

### 4.4 Motion residual channels — temporal reasoning without a temporal module

**The problem:** class alone does not tell you motion state. A parked car and a moving car carry the same label and completely different implications for a planner.

**The method:**

1. Take the previous $K = 4$ sweeps.
2. Ego-motion-compensate each into the current frame: $p' = T_t^{-1} T_{t-k}\, p$.
3. Project each into the current range-image view with the same projection function.
4. Compute the per-pixel **relative** range residual:

$$R_k(u,v) = \frac{\big|\, r_{t-k}(u,v) - r_t(u,v) \,\big|}{r_t(u,v)}$$

5. Append as input channels.

**Why it works:** if a physical point is static, compensating a past sweep onto the current frame lands it at almost exactly the same range — near-zero residual. If it is moving, past and present diverge, leaving a visible shadow trail of high residual. This converts a temporal-reasoning problem into an ordinary spatial pattern a 2D CNN already recognises — no tracker, no scene-flow network, no recurrent module in the inference path.

**Why relative and not absolute.** A 0.5 m displacement at 5 m is enormous; at 80 m it is inside the noise. An absolute residual therefore needs a range-dependent threshold, which is one more thing to tune and defend. Normalising by $r_t$ makes the threshold approximately range-invariant, which is both simpler and more honest.

**The drift failure mode, and a mitigation neither source document had.** Odometry error makes static structure appear to "vibrate" with non-zero residual. Quantify it: an angular pose error $\sigma_\theta$ displaces a point at range $r$ laterally by $r\sigma_\theta$, which in pixels is

$$\delta_{\text{px}} = \frac{r\,\sigma_\theta}{r\,\Delta\theta} = \frac{\sigma_\theta}{\Delta\theta} \qquad \textbf{— independent of range.}$$

The resulting *range* residual is that pixel error times the local depth gradient:

$$R_{\text{false}} \approx \|\nabla r\| \cdot \frac{\sigma_\theta}{\Delta\theta}$$

**Worked example.** $\sigma_\theta = 0.1° = 1.745\times10^{-3}$ rad, $\Delta\theta = 3.016\times10^{-3}$ rad → $\delta_{\text{px}} = 0.58$ pixels. On flat ground with a depth gradient of 0.5 m/pixel, the false residual is 0.29 m. At the silhouette edge of a pole against a background 20 m behind it, the gradient is ~20 m/pixel and the false residual is **11.6 m**.

So drift-induced false motion is **not uniform — it concentrates almost entirely at depth discontinuities.** That is why the classic failure looks like "edges vibrating" rather than "everything vibrating", and it hands us half the mitigation directly: **down-weight residual response where `range_gradient` is high** (channel 12).

**The other half is temporal, and the two multiply.** $\sigma_\theta$ is not a constant — pose error accumulates faster during rotation, so it grows with the vehicle's instantaneous angular velocity $|\omega|$, available from the IMU or from differencing the odometry stream. The two effects are factors in the same expression, so the threshold is their product:

$$\tau(u,v) = \alpha + \beta \cdot \|\nabla r(u,v)\| \cdot |\omega|$$

**Why the product and not either term alone.** A threshold gated on $|\omega|$ only would rise *everywhere* during a turn — suppressing genuine moving objects at precisely the moment you are turning into a junction and a pedestrian matters most. A threshold gated on $\|\nabla r\|$ only would stay needlessly conservative at every silhouette edge while driving straight, where the pose is good and edges are trustworthy. The product is high only where **both** conditions hold — a steep depth gradient *during* a sharp turn — and everywhere else the detector keeps full sensitivity.

$\omega$ has a second use worth noting: deskew quality (§4.2) also degrades under rotation, so the same signal can gate cell confidence more broadly. One hardware channel, two corrections.

Together this converts a vague known limitation into a targeted, explainable correction, and it answers the "what about odometry drift in a sharp turn?" question with a mechanism instead of a shrug.

**Threshold calibration.** Do not pick $\alpha$ or $\beta$ by eye. Measure the residual distribution over known-static structures (buildings, ground) across a full sequence and set the threshold at a high percentile of that distribution — empirical, defensible, and reproducible. Calibrate against SemanticKITTI-MOS moving-object ground truth where available.

**Residuals decide the label; the tracker decides the entity.** This is the division of labour that makes the whole motion story coherent, and it is why Part 12 keeps a tracker despite having residuals. Residuals give *per-point motion evidence* — cheap, dense, and exactly what stops a parked car from ever entering the dynamic candidate set. The tracker gives *entity-level state* — identity, velocity vector, extent — which the fovea controller (Part 13) and the query-time composite (Part 12) both need and which no per-pixel signal can supply. Two independent mechanisms that can disagree is a feature; one mechanism silently wrong is not.

### 4.5 Circular padding at the 360° seam

The sensor covers a full circle, so the image's left and right edges are not real boundaries — 359° is physically adjacent to 0°. Standard zero-padding treats them as hard edges and destroys context for anything straddling the seam: a building, a long guardrail, a wall the vehicle is driving alongside.

Replace **all horizontal padding in the network** with circular padding (`F.pad(..., mode='circular')`). It costs essentially nothing and fixes a documented, named failure mode of naive range-image networks. Vertical padding stays zero — the top and bottom of the image *are* real boundaries of the sensor's field of view.

### 4.6 Foveated input reduction — one idea, used twice

The map will store points at 80 m in 40 cm cells. So why segment them at 5 cm resolution?

Two equivalent implementations; pick one and measure both:

- **Range-image column stride.** Full azimuth resolution near, decimated columns far. Natural for this front end.
- **Foveated voxel quantisation.** $\hat p_i = c(r_i)\lfloor p_i / c(r_i)\rfloor$ with $c(\cdot)$ from Part 3, then deduplicate.

The same foveation now governs **both compute and storage**. That is not a coincidence to point out in the pitch — it is why the architecture is coherent rather than assembled.

**Two things must not be foveated.** Dynamic-candidate points bypass reduction entirely (a pedestrian at 40 m cut to 6 points may fall below the tracker's cluster threshold and vanish), and the reduction must be applied **during training as an augmentation**, not only at inference — otherwise you have a train/test distribution mismatch and far-field accuracy collapses silently.

**Measure the reduction, do not estimate it.** `eval/point_distribution.py` prints the real per-band histogram; the *measured* reduction is a headline number in Part 18. An estimate presented as a result is exactly what a judge catches.

### Edge cases and how they're fixed

| Edge case | What would go wrong | The fix |
|---|---|---|
| No per-point timestamps | Deskew impossible as specified | Reconstruct $u_i$ from azimuth: $u_i = (\text{atan2}(y,x) - \theta_{\text{start}})/2\pi$. Exact for a constant-rate spinner |
| No ego-pose source | Deskew and residuals both break | LiDAR odometry (KISS-ICP is small and dependency-light). Never accumulate or compute residuals without a pose source |
| Foveated reduction applied before deskew | Points quantised at wrong positions, then moved — quantisation error becomes position error | Strict order: deskew → transform → reduce. Enforced by type signature (`foveated_voxelize()` takes a `DeskewedCloud`) |
| Zero-padding at the azimuth seam | Objects straddling 359°/0° lose all context | Circular horizontal padding throughout the network |
| Empty pixels read as objects at the origin | Structural hallucination near the sensor | `valid_mask` channel, and loss masked to valid pixels only |
| Absolute rather than relative residual | Needs a range-dependent threshold; far-field motion missed or near-field over-triggered | Normalise by $r_t$ |
| Odometry drift making static edges "vibrate" | False dynamic classification at every silhouette | `range_gradient` channel; residual response suppressed where the gradient is high (§4.4) |
| Many-to-one projection collisions (two points, one pixel) | The farther point is silently discarded | Keep the nearest return per pixel and record a collision count; collisions are an input to Part 11's structure test, not just an artifact |
| Mixed pixels (beam clipping a foreground edge + background) | Phantom returns floating between two surfaces | Accepted and declared (Part 24). Attention gates + Lovász absorb most of it; KNN cleanup is deliberately rejected (Part 22) |

### How to test this module

`pytest tests/test_deskew.py tests/test_range_image.py tests/test_residuals.py tests/test_foveated_voxelize.py`.

**Deskew, manual:** record while moving fast past a long straight wall, project to bird's-eye view, look at the wall. Undeskewed it bends or doubles; deskewed it is straight. Binary, visual, unmistakable — put the before/after in the slides.

**Residuals, automated:** feed two identical sweeps with a known synthetic ego transform and assert every residual is ≈ 0. Then translate one object by a known amount and assert the residual appears only on that object. Then inject a known pose error $\sigma_\theta$ and assert the false-residual magnitude matches $\|\nabla r\|\sigma_\theta/\Delta\theta$ to within tolerance — **that test is the machine-checked form of §4.4's drift analysis.**

**Circular padding:** place a synthetic object straddling the seam, run a forward pass with and without circular padding, and confirm the feature response is continuous across the wrap in one case and discontinuous in the other.

---

## Part 5 — Layer 2: Segmentation Network, Ground Prior, and Loss

**Files:** `perception/ground_prior.py`, `perception/segnet.py`, `perception/losses.py`
**One-line goal:** label every point terrain / static obstacle / dynamic object — reusing an architecture that already works, and spending the saved time where the actual contribution is.

### 5.1 The deliberate stance: segmentation is not the contribution

Say this plainly, because it drives decisions that look lazy in isolation. Mature architectures solve LiDAR semantic segmentation well. A team that spends its time squeezing two points of mIoU out of a backbone and bolts on a naive grid **has skipped the contribution**, and an experienced judge sees it immediately.

The FusionSegNet backbone (~8M params) sits in the same competitive band as SalsaNext (6.7M), CENet (6.8M) and FIDNet (6M) on SemanticKITTI. Under a build window with no large pretraining budget, writing a sparse-convolution or frustum-fusion network from scratch is a high-risk bet on custom CUDA work. **The existing 2D backbone has the right capacity; what it needs is the right geometric grounding, not a rebuild** (Principle 3).

### 5.2 The ground prior — label, never strip

**This is the single most important correction in v2.** A geometric ground pass is cheap and correct and should absolutely run first. But its output must be a **label**, not a filter.

If ground points are removed before projection, they never reach the elevation map — and the ground *is* the elevation map. You would have filtered away the thing you were asked to build. The saving is also smaller than it looks: ground is commonly quoted at 60–80% of returns, but on SemanticKITTI-style urban data expect closer to 40–55%. **Measure it before the number goes on a slide** (Principle 5), because the whole compute argument scales with it.

So: run the prior, keep every point, feed the result as input channel 7, and let the network refine it. You keep terrain, the network gets a strong free hint, and the compute saving comes instead from §4.6's foveated reduction, which costs you nothing you needed.

**The method — column-wise incremental walk, not RANSAC.** Per azimuth column, walk outward from the sensor and accept a point as ground if the slope from the previous accepted ground point is below threshold:

$$\text{slope} = \frac{z_i - z_{i-1}}{\sqrt{(x_i-x_{i-1})^2 + (y_i-y_{i-1})^2}}, \qquad \text{ground if } |\text{slope}| < \tan\theta_{\max}, \ \ \theta_{\max}\approx10°$$

**Why not per-sector plane fitting:** a plane — even a local one — assumes flatness over the sector. On a slope, a crest, or a camber, that is wrong across the whole sector, and it will confidently call the far side of a rise an obstacle. The incremental walk compares only to its immediate predecessor, so it *tracks* terrain instead of assuming it. This is $O(n)$, runs in ~3 ms, and it also produces the local ground reference Part 10 needs for expected-return reasoning — so it earns its place twice.

**Risk owned here:** an over-aggressive threshold can absorb a genuine low obstacle (kerb lip, pothole rim) into "terrain". Keep it conservative: a false negative here (missing a real obstacle) is strictly worse than a false positive (passing a slightly-too-large point set to a network that can still correct it).

### 5.3 The network, layer by layer

**Encoder — EfficientNet-B0** (~4–4.5M params, trained from scratch). Depthwise separable convolutions and inverted residual blocks; near-peak Tensor Core utilisation on dense 2D tensors, which carries over from camera images to range images unchanged. **Only modification:** the first conv expands from 3 input channels to 14 (§4.3) — or is preceded by the Meta-Kernel stem (§5.5). A shape change, nothing else.

**Bottleneck — ASPP.** Parallel atrous convolutions at several dilation rates. This matters *more* for LiDAR than for cameras: in a spherical projection an object's pixel footprint varies enormously with range — a car at 5 m and the same car at 50 m are drastically different sizes on the image. ASPP's multi-scale receptive fields are exactly the mechanism for recognising one class across that scale range without separate detectors per distance band. **It is the network-side counterpart of the map-side foveation argument**, and worth presenting as such.

**Decoder — Attention U-Net with SE blocks.** Attention gates on the skip connections suppress irrelevant background and amplify sparse high-frequency foreground. Distant returns are inherently sparse; a plain skip connection lets low-information background dilute the decoder, while gates let it focus where signal actually is. SE blocks recalibrate channel importance, which matters because our fourteen channels mix very different physical quantities — geometry, range, reflectance, motion residual, gradient, occlusion depth — and uniform treatment wastes capacity.

**Deep supervision head** at 1/8 decoder resolution, supervised alongside the main output. Speeds convergence when training from scratch by giving early layers a direct gradient, and with Lovász applied there too it enforces coarse structural correctness before the full-resolution decoder refines detail.

### 5.4 Loss stack

| Component | Role | Why |
|---|---|---|
| **Lovász-Softmax** | primary | A convex surrogate for the Jaccard/IoU metric — optimises directly for what you are graded on (mIoU), and is specifically effective under the extreme class imbalance of LiDAR data, where ground points outnumber pedestrians by orders of magnitude. Dice loss is insufficient at this degree of imbalance |
| **Confidence-weighted cross-entropy** | secondary | Reused from FusionSegNet's pseudo-label weighting, repurposed to weight by **point-density confidence** — a label from a sparse region contributes less than one from a dense region. Note this is the same quantity Part 11 formalises as $\kappa$, so the loss and the map's confidence field share one definition rather than two |
| **Deep-supervision auxiliary term** | small | Applied at 1/8 resolution (§5.3) |

**Deliberately dropped:** the temporal-consistency loss from the original stack. It assumes smooth pixel correspondence across frames, which video has and a rotating, discretely-sampled LiDAR does not — transplanted directly it becomes unstable. Its job (teaching the network what moves) is done *structurally* by the residual channels instead, which is more stable and needs no specialised loss term.

**All losses are masked to valid pixels** (channel 6). Training on empty pixels teaches the network to predict classes for returns that do not exist.

### 5.5 The Meta-Kernel stem — a real fix for the translation-invariance problem

§4.3 explains why we hand the network explicit $x,y,z$ channels: a standard convolution assumes adjacent pixels have a constant physical relationship, and LiDAR's non-uniform vertical beam spacing violates that. Explicit coordinates are a *hint* — the network may learn to use them. A **Meta-Kernel** is a *mechanism* that forces the geometry into the convolution itself.

**The formulation** (from RangeDet — cite it, this is not ours; verify the reference before it goes on a slide):

1. For each pixel, gather its $3\times3$ neighbourhood with `unfold`.
2. Compute the **relative** 3D offset from the centre pixel to each of the 9 neighbours: $\delta_k = p_k - p_{\text{centre}}$.
3. Pass those offsets through a small shared MLP to produce a weight vector per neighbour.
4. Aggregate the neighbours' features using those dynamically generated weights instead of fixed learned kernel weights.

**Relative, not absolute — this is the whole point.** What is broken is not that the network does not know where a pixel is; it is that the *physical distance between adjacent pixels* varies enormously across the image (Part 3: 7.4 cm vertical spacing at 10 m, 74 cm at 100 m). Only relative offsets express that. Feeding absolute $x,y,z$ through a $1\times1$ convolution does not fix it, because a $1\times1$ convolution never looks at a neighbour.

**Scope it deliberately.** Apply the Meta-Kernel **at the stem only**, on the raw input tensor, and leave the rest of the backbone untouched. Geometry is rawest there and the tensor is cheapest; running dynamic kernels through the decoder buys little and costs a great deal. Budget honestly: this is roughly 30–50 lines with an `unfold`, not ten, and it costs real runtime because it is a per-pixel MLP over a neighbourhood.

**Gate it on evidence.** There is a genuine tension with Principle 3 — this adds a novel module to a backbone chosen partly to avoid new architecture under time pressure. So it ships as an **ablation against the explicit-coordinate-channel baseline** (Part 18.1), and if it does not win by a margin worth its latency, it is cut. That is the honest way to include it, and being able to say *"we measured it and kept it"* is worth more than including it because it sounds advanced.

### Edge cases and how they're fixed

| Edge case | What would go wrong | The fix |
|---|---|---|
| Ground stripped before projection | Terrain never reaches the elevation map — the deliverable is missing | **Label, never strip.** Ground prior is input channel 7 |
| Ground prior threshold too aggressive | Kerb lips and pothole rims absorbed into "terrain" | Conservative threshold; false negatives strictly worse than false positives here |
| Per-sector plane fit on a slope | Whole sector misclassified on exactly the terrain DRDO cares about | Column-wise incremental walk comparing only to the previous accepted neighbour |
| Loss computed over empty pixels | Network learns to hallucinate classes in empty space | Mask every loss term by `valid_mask` |
| Class imbalance with plain CE or Dice | Rare classes (pedestrian, pole) collapse | Lovász-Softmax as the primary term |
| Range-image label bleed at depth discontinuities | A pole's label smears onto the ground 20 m behind it | k-NN re-projection from pixels back to points using 3D distance, not pixel adjacency |
| Confident prediction on a class never seen in training (off-road rubble) | Silent, confident misclassification | Per-class confidence written into the cell; low-confidence terrain degrades to `TRAVERSABLE_CAUTION` and never upgrades to `DRIVABLE` — **uncertainty always resolves toward caution** |

### How to test this module

`pytest tests/test_ground_prior.py tests/test_segnet_io.py tests/test_losses.py`.

**Ground prior, manual:** colour ground points on a scan containing a rise and a dip. A plane-fit implementation visibly fails on both; the incremental walk should track them.

**The strip-vs-label regression test:** run the full pipeline and assert the output map contains terrain cells with valid elevation. It sounds trivial. It is the automated form of the bug this part exists to fix, and it belongs in CI.

**Loss masking:** assert the loss is bit-identical when invalid pixels are filled with garbage versus zeros. If it is not, the mask is not applied everywhere.

---

## Part 6 — Layer 3: Taxonomy

**File:** `perception/taxonomy.py`
**One-line goal:** speak the vehicle's language, not the dataset's.

Public datasets label for urban driving research: `road`, `sidewalk`, `car`, `traffic-sign`. A defence UGV does not care whether a surface is a road; it cares whether it can cross it, at what cost, and what happens if it is wrong.

| ID | Class | Meaning to the planner |
|---|---|---|
| 0 | `UNKNOWN` | Never observed, **or observed past the point where absence is informative** (Part 11). Not the same as empty |
| 1 | `DRIVABLE` | Smooth, load-bearing, full speed |
| 2 | `TRAVERSABLE_CAUTION` | Passable with a speed/cost penalty — grass, gravel, verge |
| 3 | `NON_TRAVERSABLE_TERRAIN` | Slope, rubble, water. **Geometry says no — or, for standing water and wet mud, radiometry does (§9.5)** |
| 4 | `STATIC_OBSTACLE` | Wall, pole, building, fence |
| 5 | `VEGETATION_PERMEABLE` | Looks solid to the LiDAR, can be driven through |
| 6 | `DYNAMIC_VEHICLE` | Tracked entity with a velocity |
| 7 | `DYNAMIC_VRU` | Pedestrian / cyclist — highest protection priority |
| 8 | `NEGATIVE_OBSTACLE` | **Geometry only** (Part 10). Never predicted by the network |
| 9 | `OVERHANG` | **Geometry only** (Part 9). Never predicted by the network |

Classes 8 and 9 exist solely as outputs of geometric reasoning — architectural boundary 1 from Part 2.

**SemanticKITTI → DRISHTI mapping**, declared explicitly and version-controlled:

| SemanticKITTI | DRISHTI |
|---|---|
| road, parking | `DRIVABLE` |
| sidewalk, other-ground, terrain | `TRAVERSABLE_CAUTION` |
| building, fence, pole, traffic-sign, trunk, other-structure | `STATIC_OBSTACLE` |
| vegetation | `VEGETATION_PERMEABLE` **or** `STATIC_OBSTACLE` — gotcha 1 |
| car, truck, other-vehicle, bicycle, motorcycle | `DYNAMIC_VEHICLE` **candidate** — gotcha 2 |
| person, bicyclist, motorcyclist | `DYNAMIC_VRU` **candidate** |

**Gotcha 1 — `vegetation` is two different things.** It covers knee-high grass (drive through it) and tree canopy (do not drive into the trunk, and the canopy is an *overhang*, not ground). Resolve by height above local ground: below ~0.5 m → `VEGETATION_PERMEABLE`; above → part of the ceiling layer. A single label cannot carry this; the cell structure in Part 9 can.

**Gotcha 2 — class proposes, motion disposes.** A `car` label makes a point a dynamic *candidate*, nothing more. Promotion to `DYNAMIC_*` requires motion evidence: residual channels (Part 4.4) at the point level, confirmed by the tracker (Part 12) at the entity level. Writing every `car` into the dynamic layer means parked cars never enter the static map and the planner routes straight through them.

State the remap as a deliberate decision in the writeup. It shows you understood whose problem this is — a defence navigation scheme, not a civilian street-scene taxonomy — rather than inheriting labels by accident.

**A note on where each class comes from.** Classes 1, 2, 4, 6 and 7 come from the network. Classes 8 and 9 come from geometry alone. Class 5 comes from multi-echo return structure (§6.1). Class 3 comes from geometry (slope, roughness) *and* from radiometry for the cases geometry cannot see (§9.5). Class 0 comes from observability and the sparsity model. **Only five of the ten classes are learned**, which is the concrete form of the domain-gap argument in Part 24: the hazard classes a vehicle's survival depends on are the ones that do not need to generalise, because they were never trained.

### 6.1 Multi-echo returns — vegetation depth and bare earth

A laser pulse striking tall grass does not stop at the first blade. Part of the beam penetrates and returns from the ground beneath, so a single pulse produces **multiple returns** — first return from the canopy top, last return from the true ground. Most sensors report at least strongest-and-last; Ouster-class sensors report two.

This is standard practice in a field nobody in this competition will have read: **airborne LiDAR bare-earth extraction**, where separating canopy from terrain to build a digital terrain model is the entire problem. Borrowing it here is the same cross-domain move as taking the clipmap from real-time graphics, and it converts `VEGETATION_PERMEABLE` from a height heuristic into a measurement.

Per cell:

$$\text{vegetation depth} = \overline{z_{\text{first}}} - \overline{z_{\text{last}}}, \qquad z_{\text{ground}} = \overline{z_{\text{last}}}$$

Three signatures, cleanly separable:

| Return structure | Interpretation |
|---|---|
| First ≈ last, dense, coherent surface | **Solid.** A wall, a vehicle, a rock. `STATIC_OBSTACLE` |
| First − last ≈ 0.2–0.8 m, many multi-return points, last returns form a coherent surface | **Penetrable vegetation.** `VEGETATION_PERMEABLE`, and the last-return surface is the real ground the wheels will meet |
| First − last large and inconsistent, few multi-returns | Canopy over a gap, or a genuine overhang — resolve in the multi-layer cell (Part 9) |

This directly fixes Gotcha 1 above. `vegetation` at knee height with a coherent last-return surface 40 cm below it is grass you drive through, and the elevation you store is the *last* return, not the first. Store the first-return height as the ceiling layer's lower bound so the clearance calculation still sees the canopy.

**Availability caveat:** this needs a sensor that reports multiple returns per pulse, and it needs the dataset to preserve them. SemanticKITTI does not ship dual returns, so this is validated in CARLA and on a live sensor if one is available, and declared as sensor-dependent rather than assumed. Where dual returns are unavailable the height heuristic from Gotcha 1 remains the fallback, at lower confidence.

### How to test

`pytest tests/test_taxonomy.py`. Assert every dataset class maps to exactly one DRISHTI class, and that **no dataset class maps to 8 or 9**. That single assertion is the machine-checked form of architectural boundary 1, and it should fail loudly if anyone ever wires the network to a geometry-only hazard class.

---
## Part 7 — The Coordinate System Decision: Polar Front End, Cartesian Map

**One-line goal:** settle, with reasons, the single biggest architectural question in this project — and explain why the answer is "both", not "either".

This part exists because two defensible designs pull in opposite directions, and a bible that does not resolve the tension out loud will produce an implementation that quietly does half of each.

### The case for polar, which is correct — for perception

Part 4 already made it: a rotating LiDAR samples at constant angular resolution, the range image *is* the sensor's native layout, and a continuously-growing footprint with range is the same foveation the map wants. Projecting a labelled range-image pixel into a polar grid is a direct coordinate transform with no rasterisation of unstructured 3D points. All of that is true and it is why the front end is polar.

### Why polar is the wrong choice for the map itself

**1. The "no boundaries" claim is not available.** A continuously-varying cell size is not a partition of space. To store anything you must bin $\rho$, and the bin edges *are* discrete boundaries — you trade a few large discontinuities for many small ones, which may well be better but is not "provably no boundary anywhere". What *is* provable, and worth claiming instead, is that **a monotonic polar binning tiles the plane exactly, so every point is assigned to exactly one cell and nothing is double-counted.** That is a real guarantee. Claim it; do not claim the stronger thing, because it will not survive a judge who thinks about it for thirty seconds.

**2. Temporal accumulation is the hidden cost.** An ego-anchored polar grid must re-bin its entire accumulated contents every frame as the vehicle **translates and rotates**. That is precisely the re-sampling cost the polar argument was meant to avoid, relocated from the spatial axis to the temporal one. A world-anchored Cartesian clipmap makes translation a modular index shift and makes rotation *free*, because the grid does not rotate with the vehicle at all.

**3. The origin degenerates.** Tangential cell width is $\rho\Delta\theta$, so it goes to zero as $\rho \to 0$. At $\rho = 5$ m with a 5 cm radial bin the cell is 1.5 cm × 5 cm (3.3:1); at $\rho = 1$ m it is 0.3 cm × 5 cm (16:1). Those slivers sit **directly under and around the vehicle** — the region where a UGV most needs a clean local map.

**4. Exact demotion requires power-of-two nesting.** Part 12's coarse-cell reduction is *exact* — a coarse cell's statistics equal the statistics over the union of its children's points, with no error term. That property depends on four children tiling one parent perfectly (Part 8). Polar bins with a continuous width schedule have no such relationship, so every level transition becomes an approximation with no error bound.

**5. Planners consume Cartesian.** `nav_msgs/OccupancyGrid` and `grid_map` are Cartesian. Emitting a polar map means every downstream consumer resamples it — you have not removed the transform, you have exported it to someone else's latency budget.

### The objection to Cartesian, and why it dissolves

The stated cost was "re-sampling between two mismatched coordinate systems". That cost is real **grid-to-grid** — resampling a polar *grid* into a Cartesian *grid* genuinely does interpolate and lose information.

But we never do that. We go **point → Cartesian**, directly:

$$x = \rho\cos\theta, \qquad y = \rho\sin\theta, \qquad i = \lfloor x/c_\ell \rfloor, \quad j = \lfloor y/c_\ell \rfloor$$

One sine, one cosine, two floor-divides per point, fully vectorised on GPU, inside the same kernel that does the scatter (Part 9.4). There is no intermediate grid and therefore no resampling. The objection applies to an architecture we are not building.

### The decision

> **Polar for perception, Cartesian for the map, one point-wise transform between them.**

The range image stays the sensor-native front end and keeps every advantage Part 4 claimed for it. The clipmap stays the world-anchored store and keeps exact nesting, free rotation, cheap accumulation, and a native planner interface. Neither gives anything up, because they are solving different problems: **the front end's job is to be shaped like the sensor; the map's job is to be shaped like the world.**

---

## Part 8 — Layer 4: The Foveated Clipmap

**Files:** `grid/clipmap.py`, `grid/addressing.py`
**One-line goal:** hold a variable-resolution 2.5D map indexable in constant time, scrollable with the vehicle for free, in which **cross-resolution alignment error is impossible rather than merely handled.**

### Why a quadtree is the wrong answer

Every team will say "quadtree" — it is the textbook answer to "variable resolution spatial structure". Four concrete reasons it is a poor fit:

1. **Pointer chasing.** Every lookup walks a tree through scattered memory, millions of times per second. Cache misses dominate.
2. **Rebuild cost.** The fovea is ego-centric, so tree structure changes every frame. This — not the network — is where naive implementations lose their frame rate.
3. **The 2:1 balancing ripple.** To stay seamless a tree must keep adjacent cells within one refinement level. Enforcing that triggers a cascade of re-subdivision every time a single high-resolution cell updates near the vehicle, which is the bottleneck research identifies as unsolved at the 10–20 Hz rates real-time perception needs.
4. **It solves a problem we do not have.** Trees shine when the resolution pattern is arbitrary and data-dependent. Ours is a known function of range. Paying for unused generality is the definition of over-engineering.

**Note carefully what this argument does and does not cover.** Reason 3 is fatal to *trees*. The clipmap below is **not a tree** — it has no parent pointers, no subdivision, and therefore no balancing constraint to ripple. The anti-tree argument does not transfer to it, and saying so pre-empts the obvious "isn't this just a quadtree?" question.

### The structure

A stack of $L$ uniform grids, borrowed from foveated rendering — mipmaps and cascaded shadow maps solved exactly this problem (detail near, coarse far, moving viewpoint) decades ago.

- Level $\ell$ has cell size $c_\ell = c_0 \cdot 2^\ell$
- Each level is a **flat $N \times N$ array**, $N$ a power of two
- Every level is centred on the vehicle and covers extent $N c_\ell$

**Reference configuration** ($c_0 = 5$ cm, $N = 512$, $L = 4$):

| Level | Cell size | Coverage | Sensor-Nyquist limit (Part 3) | Cells |
|---|---|---|---|---|
| L0 | 5 cm | ±12.8 m | 16.6 m ✓ | 262,144 |
| L1 | 10 cm | ±25.6 m | 33.2 m ✓ | 262,144 |
| L2 | 20 cm | ±51.2 m | 66.3 m ✓ | 262,144 |
| L3 | 40 cm | ±102.4 m | 132.6 m ✓ | 262,144 |
| | | | **Total** | **1,048,576** |

Every level's coverage is *tighter* than its sensor-Nyquist limit, so no cell is ever asked to resolve detail the sensor cannot deliver. And it satisfies the statement outright: 5 cm well past 10 m, 40 cm past 100 m.

### Constant-time addressing

World point → global integer index at level $\ell$: $i = \lfloor x/c_\ell\rfloor$, $j = \lfloor y/c_\ell\rfloor$. Storage index by **toroidal addressing**, one bitwise AND because $N$ is a power of two, correct for negative indices in two's complement:

$$s_i = i \;\&\; (N-1), \qquad s_j = j \;\&\; (N-1), \qquad \text{flat} = s_j N + s_i$$

No pointers, no tree walk, no branching. One multiply and two ANDs.

### Why alignment error is structurally impossible

All levels index from the **same world origin** with cell sizes in exact powers of two. A level-$(\ell{+}1)$ cell at $(I,J)$ therefore covers exactly four level-$\ell$ cells:

$$(2I, 2J), \quad (2I{+}1, 2J), \quad (2I, 2J{+}1), \quad (2I{+}1, 2J{+}1)$$

with no fractional overlap, no partial cells, no straddling — **ever**, at any position, for any ego pose. The statement names alignment error and data loss at resolution boundaries as the hard part. This design does not solve that problem; it removes the conditions under which it can exist.

**This is why the schedule is 5 → 10 → 20 → 40 cm and not 5 → 50 cm.** $50/5 = 10$ is not a power of two, so a 5→50 hierarchy cannot nest exactly, and any non-binary subdivision reintroduces straddling. Choosing 40 cm keeps the hierarchy exact **and** exceeds the requirement. When a judge asks why you deviated from 50 cm, that is the answer — and it is a better answer than compliance would have been.

### Scrolling: ego-motion is an index shift

When the level-$\ell$ window shifts by $(\Delta i, \Delta j)$ cells, toroidal addressing means retained cells keep their flat index. Only rows and columns scrolling *in* need clearing:

- Cost: $O(N(|\Delta i| + |\Delta j|))$ per level per frame — a few hundred cells at 10 Hz, versus $O(N^2)$ for a rebuild
- At 15 m/s, L0 shifts 30 cells per frame; L3 shifts 3
- Rotation costs **nothing** — the grid is world-anchored and does not rotate with the vehicle

**The most dangerous bug in this project lives here.** Forget to clear incoming rows and wrap-around silently returns data from 100 m *behind* the vehicle as though it were 100 m *ahead*. It looks entirely plausible on screen. Two independent defences, ship both:

1. **Clear on scroll** (primary): zero the incoming rows/columns every frame.
2. **Stamp validation** (cross-check): each cell stores the low bits of its global $(i,j)$; on read, a mismatch returns `UNOBSERVED`. Cheap, and it turns a silent wrong answer into an honest "I don't know".

Leave the stamp check enabled in the demo build. A slightly slower demo that cannot lie is worth more than a fast one that might.

### The invariant that prevents double-counting

Levels form a **mipmap pyramid**: every point contributes to its native level and all coarser ones, so each level is independently a complete map of its extent. The rule that makes this safe:

> **Levels are alternative views of one world. They are never summed, averaged together, or aggregated across.** A query returns exactly one cell from exactly one level.

`clipmap.lookup(x, y)` selects the **finest level whose window contains the point** and returns that cell. Total memory is the one legitimate sum across levels, because those are physically distinct allocations. Every other statistic is per-level.

**Allocation note.** Full squares at every level mean each coarse level redundantly covers what finer levels already hold: $3 \times 256^2 = 196{,}608$ cells, **18.75%**. Ring allocation removes it at the cost of messier indexing. Build full squares first — the 18.75% is a known, quantified, honestly-reported cost, and ring allocation is a one-afternoon optimisation if the number bothers anyone.

### Worked example

Vehicle at $(1000.37, -240.12)$ m; query a point at $(1012.00, -238.00)$ m — 11.6 m ahead, 2.1 m left, 11.8 m away.

- 11.8 m < 12.8 m → **L0 is the finest valid level**
- $i = \lfloor 1012.00/0.05\rfloor = 20240$, $j = \lfloor -238.00/0.05\rfloor = -4760$
- $s_i = 20240 \,\&\, 511 = 16$, $s_j = -4760 \,\&\, 511 = 328$
- flat $= 328 \times 512 + 16 = 167{,}952$

The same point also exists in L1 ($i=10120$), L2 ($i=5060$), L3 ($i=2530$). `lookup()` returns only L0. Nothing is summed.

### Edge cases and how they're fixed

| Edge case | What would go wrong | The fix |
|---|---|---|
| Incoming scrolled rows not cleared | Stale data from behind the vehicle appears ahead of it, plausibly | Clear-on-scroll every frame **plus** per-cell stamp validation |
| Query beyond L3's extent | Toroidal wrap silently returns a cell from the opposite side | `lookup()` bounds-checks the window; wrap is never trusted to mean containment |
| Summing statistics across levels | Every point counted 4× | Mipmap invariant enforced in code: no API aggregates across levels |
| 5 → 50 cm as literally specified | Not a power-of-two ratio; exact nesting impossible | 5 → 10 → 20 → 40 cm, stated openly as a deliberate deviation that exceeds the requirement |
| Large world coordinates (UTM ~10⁶ m) | float32 ULP exceeds 5 cm; cells alias | All map math in a local ego-anchored frame, rebased periodically; global anchor kept in float64 |
| Vehicle stationary | Nothing scrolls; stale cells never refresh | Per-cell timestamps; confidence decays with age (Part 12) independently of scrolling |
| $N$ not a power of two | `& (N-1)` silently produces wrong indices | `Clipmap.__init__` asserts `N & (N-1) == 0` |

### How to test this module

`pytest tests/test_clipmap.py tests/test_addressing.py`. This is the load-bearing wall of the project and deserves the most thorough test file in the repo:

1. **Nesting exactness** — for many random world points and ego poses, assert the L$\ell$ index is exactly the L$(\ell{+}1)$ index doubled (±1). This is the machine-checked proof of the central claim.
2. **Scroll correctness** — write a known pattern, scroll far, assert every incoming cell reads `UNOBSERVED` and every retained cell keeps its value.
3. **Round-trip** — `world → index → world` lands inside the original cell, including negative coordinates (where the two's-complement path breaks).
4. **Adversarial scroll** — scroll by exactly $N$ cells (a full wrap) and confirm nothing stale survives. This is the test that catches the silent-wrap bug.

Manual: drive a long sequence with the stamp-mismatch counter on the HUD. **It must read zero throughout.** Nonzero means clear-on-scroll has a hole.

---

## Part 9 — Layer 5: Cell Aggregation (where 2.5D stops being a lie)

**Files:** `grid/scatter.py`, `grid/cell.py`, `grid/layers.py`
**One-line goal:** collapse labelled points into cells **without losing the vertical information that makes the map worth building.**

### 9.1 The contradiction at the heart of the problem statement

A road passing under a bridge. Points at $z \approx 0.0$ (road) and $z \approx 4.2$–$4.6$ m (bridge underside). One height value per cell must choose:

| Storage choice | Reported height | Planner concludes | Reality |
|---|---|---|---|
| Max height | 4.6 m | Impassable wall | Drivable |
| Min height | 0.0 m | Fully clear | Drivable — but the same rule drives you into a 1.9 m branch elsewhere |
| Mean height | ~2.3 m | Impassable wall | Drivable |

**Every single-value representation gets this wrong.** Not tuned wrong — structurally wrong. No threshold fixes it, because the information needed to answer was destroyed at write time.

Say this to judges in exactly these terms. You are not reporting a bug; you are reporting that the naive reading of the requirement is unimplementable, and handing them the fix.

### 9.2 Multi-layer cells

A cell stores **vertical intervals**, not a scalar:

- **Ground layer** $[z^{g}_{\min}, z^{g}_{\max}]$ — the lowest coherent surface
- **Gap** — the free vertical span above it
- **Ceiling layer** $[z^{c}_{\min}, z^{c}_{\max}]$ — the lowest surface above the gap, if any

$$\text{clearance} = z^{c}_{\min} - z^{g}_{\max}, \qquad = +\infty \text{ when no ceiling exists}$$

**Extraction** from the cell's 8-bin height histogram over $[z_{\text{ground}} - 0.5,\ z_{\text{ground}} + 4.0]$ m (bin width 0.5625 m):

1. Ground layer = lowest occupied bin and its immediate occupied neighbours.
2. Scan upward for the first run of **empty** bins at least $\lceil H_{\text{req}}/w_{\text{bin}}\rceil$ long.
3. Ceiling = first occupied bin above that run, if any.

Bin width is deliberately coarse — clearance is a metre-scale question, and 8 bins pack into 8 bytes.

**Worked example A — road under a bridge.** Ground $[0.00, 0.05]$; bins empty from 0.05 to 4.2 (a 4.15 m run, well past a 2.5 m requirement); ceiling 4.2. Cell reports `DRIVABLE`, clearance 4.15 m. Correct, where all three single-value schemes failed.

**Worked example B — low branch over a clear path.** Ground $[0.00, 0.02]$, branch at $[1.90, 2.30]$. Gap 1.88 m < 2.5 m → clearance violation. Cell reports `OVERHANG`, non-traversable *for this vehicle*. Note the ground beneath is perfectly flat: a 2D occupancy grid, a min-height map, and a naive ground-segmentation pipeline all say "clear" and drive the vehicle's roof into the branch.

**Worked example C — the same branch, a different vehicle.** A platform needing 1.5 m clearance passes underneath. Because the cell stores *clearance* rather than a traversability boolean, one map serves both vehicles. **The map does not bake in the vehicle** — a property worth naming, because it is what makes this a component rather than a demo.

### 9.3 Cell contents and byte layout

Twelve bytes single-layer (v1); sixteen with the ceiling layer (v2).

| Field | Type | Bytes | Notes |
|---|---|---|---|
| `h_min` | int16 | 2 | 1 cm fixed point, ±327.67 m |
| `h_max` | int16 | 2 | |
| `h_mean` | int16 | 2 | |
| `h_m2` | uint16 | 2 | Welford $M_2$, quantised → variance |
| `count` | uint16 | 2 | saturating |
| `class_conf` | uint8 | 1 | 4 bits class, 4 bits confidence ($\kappa$, Part 11) |
| `flags` | uint8 | 1 | observed / free / occupied / occluded / dynamic / negative-suspect / sparse-structured / provisional |
| *(v2)* `h_ceil_min` | int16 | 2 | sentinel = no ceiling |
| *(v2)* `h_ceil_max` | int16 | 2 | |

**Structure-of-arrays, not array-of-structures.** Separate parallel planes per field: SIMD/GPU kernels vectorise over a plane, and it enables a **hot/cold split** — the planner touches only `(traversability_cost: uint8, h_max: int16)` = 3 bytes per cell every control cycle. The L0 hot plane is $512^2 \times 3 = 786$ KB, small enough to stay cache-resident. Part 19 turns this into the memory argument that actually matters.

**Do not store heights as float32.** 1 cm precision over ±300 m fits in int16 with room to spare; float32 doubles the largest field for precision no LiDAR delivers.

**Foveate the height axis too (optional optimisation).** Everything in this design foveates in XY while $z$ stays at a uniform 1 cm everywhere — which is inconsistent, because at L3 the sensor's *vertical* sampling is 74 cm (Part 3). Storing millimetre-scale height precision there is exactly the error we criticise a uniform grid for making in XY.

Matching the height quantum to the level — 1, 2, 4, 8 cm for L0…L3 — lets the three height fields drop to int8 relative to a per-tile base elevation at L1–L3, saving 3 bytes per cell across 786,432 cells:

$$786{,}432 \times 3\ \text{B} = 2.36\ \text{MB}, \qquad 12.58 \to 10.22\ \text{MB} \quad (\textbf{18.8\%})$$

Be honest about the trade: it costs a per-tile base elevation and a decode step on every read, for a saving comparable to ring allocation. **Take it for the consistency argument, not the bytes** — "we foveate in all three axes, and here is the vertical sampling curve that says why" is a better sentence than 2.36 MB is a number. Build it after the flat int16 version works, and only if the Pareto sweep says the complexity earns its place.

### 9.4 The projection kernel: no Python loops, ever

This is where naive implementations lose their frame rate — not in the network. A Python loop over 120,000 points at 10 Hz is hopeless. The whole scatter is a flat index plus `scatter_reduce`, GPU-resident, including the polar→Cartesian transform from Part 7:

```python
# per level; all tensors on GPU, no host round-trip
x = rho * torch.cos(theta)                            # Part 7 transform,
y = rho * torch.sin(theta)                            # fused into this kernel
i = torch.div(x, c_l, rounding_mode='floor').long()
j = torch.div(y, c_l, rounding_mode='floor').long()
flat = (j & (N - 1)) * N + (i & (N - 1))              # (P,)

h_max.scatter_reduce_(0, flat, z, reduce='amax', include_self=True)
h_min.scatter_reduce_(0, flat, z, reduce='amin', include_self=True)
count.scatter_add_(0, flat, torch.ones_like(flat, dtype=count.dtype))
h_sum.scatter_add_(0, flat, z)
```

About twenty lines for the entire projection — no custom CUDA, no C++ build step. Mean follows from `h_sum/count`; variance from a second `scatter_add_` of $z^2$. For true integer atomics, encode height monotonically: $\tilde z = \text{round}(100z) + 32768$, which preserves ordering so integer `amax`/`amin` behave identically.

**Class aggregation is a mode, not a mean** — averaging class IDs is meaningless. Scatter a per-class count into a small `(cells, 10)` histogram and take the argmax, and expose the runner-up in the confidence field so a cell can say *"mostly ground, but 30% of my points said obstacle"* — exactly the signal a cautious planner wants.

### 9.5 Incidence-corrected albedo, and the flat-but-lethal case

`NON_TRAVERSABLE_TERRAIN` (class 3) covers slope, rubble **and water**. Slope and rubble have geometric signatures. Standing water and wet mud do not — they are, geometrically, perfectly flat drivable ground. A pure-geometry map says *drive here*, and a UGV that does is immobilised. This is the single most common way a ground vehicle gets stuck, and until now this design had no mechanism for it.

**Radiometry has one, and the 2.5D map is what unlocks it.**

For a Lambertian surface, received power depends on three things — the material's albedo $\rho$, the angle the beam strikes it at, and range:

$$P_r \propto \frac{\rho \cos\theta_{\text{inc}}}{r^2} \quad\Longrightarrow\quad \hat\rho = \frac{I \cdot r^2}{\cos\theta_{\text{inc}}}, \qquad \cos\theta_{\text{inc}} = |\hat n \cdot \hat d|$$

Everyone uses **raw intensity**, which conflates material with geometry and range — the same asphalt returns wildly different intensities at 10 m head-on and 60 m at a grazing angle, so raw intensity is nearly useless as a material cue. Correcting it requires the **surface normal** $\hat n$, which almost nobody has at the point of measurement.

**We have it.** The normal comes from the local $3\times3$ elevation neighbourhood, which Part 14 already computes for slope. So the elevation map *enables* a material estimate that a raw point cloud cannot produce. That is a genuine synergy between two layers rather than a bolted-on feature, and it is worth naming as one.

**Ordering.** Aggregate heights first, derive normals, then correct intensity and aggregate $\hat\rho$ — a second pass within the same frame, or use the previous frame's normals since the map persists across frames anyway. Two-pass is cleaner and costs almost nothing.

**Guard the grazing case.** As $\cos\theta_{\text{inc}} \to 0$ the correction blows up. Compute $\hat\rho$ only where $\cos\theta_{\text{inc}} > 0.15$ (incidence under ~81°) and mark it unusable otherwise. Near-grazing geometry on distant ground is exactly where this fails, and pretending otherwise would manufacture confident nonsense at range.

**The water signature is mostly a hole, not a low number.** At grazing incidence water reflects specularly *away* from the sensor, so the dominant effect is **dropout** — no return at all — rather than a weak return. So the cell-level signature is: flat local geometry, anomalously low $\hat\rho$ where returns exist, and a high proportion of `valid_mask = 0` pixels over the cell.

**And that disambiguates something Part 10 had to leave as `UNKNOWN`.** A ditch and a puddle both produce missing ground returns. But a ditch **displaces** returns to greater range — the range shadow $\Delta = rd/h$ — while water **absorbs** them and nothing comes back at all. So:

| Signature | Interpretation |
|---|---|
| Missing returns **plus** displaced returns beyond $r_{\text{exp}}$ | Negative obstacle (Part 10.2) |
| Missing returns, **no** displaced return, flat surrounding geometry, low $\hat\rho$ | Specular surface — standing water or wet mud |
| Missing returns, no displaced return, no radiometric support | Genuinely `UNKNOWN` |

Two hazards that previously collapsed into one cautious answer now separate, and the third case stays honest.

**State the confidence honestly.** This is the most research-flavoured element in the document. The Lambertian model is an approximation; water response differs substantially between 905 nm and 1550 nm sensors; and wet asphalt, fresh tarmac and standing water are not trivially separable. So $\hat\rho$ feeds a **confidence channel that degrades cells toward `TRAVERSABLE_CAUTION`** — it does not act as a hard water detector, and no detection rate is claimed until it is measured in CARLA (Part 18.3). Degrading toward caution on a weak signal is safe; asserting water on one is not.

### Edge cases and how they're fixed

| Edge case | What would go wrong | The fix |
|---|---|---|
| Single-value elevation with an overhang | Structurally impossible to represent | Multi-layer cells with an explicit clearance field |
| Cell receives one point | `h_max = h_min`, variance zero → maximally confident where it knows least | Confidence is a function of `count` and $\kappa$ (Part 11); a 1-point cell never upgrades to `DRIVABLE` |
| Empty cell vs. cell with no returns | An unobserved hole looks like flat ground at $z=0$ | Explicit `flags` tri-state; height fields meaningless unless the observed bit is set |
| Averaging class IDs | Class 2 and 6 average to unrelated class 4 | Mode/histogram aggregation, never arithmetic |
| `count` overflow in dense near cells | uint16 wraps; confidence collapses on the *best*-observed cells | Saturating add |
| Vegetation canopy read as a ceiling | Canopy at 3 m becomes a false overhang | Ceiling extraction ignores layers classed `VEGETATION_PERMEABLE` whose return density is below threshold — canopy scatters sparse returns, a bridge deck returns a dense flat surface |
| Bin width too fine | 8 bins no longer span the vehicle-relevant range | Bin width derives from `H_req` in the vehicle config, not a literal |

### How to test this module

`pytest tests/test_cell.py tests/test_layers.py tests/test_scatter.py`.

Layer extraction is a pure function over a histogram: hand-construct the bridge case, the low-branch case, flat ground, and the vegetation canopy, and assert extracted clearance. **These four unit tests are the machine-checked form of Claim 1.**

Scatter: build a synthetic cloud whose answer you know analytically, assert every reduced field, then compare the GPU path against a slow obviously-correct NumPy reference on random clouds — ten lines, and it catches index bugs the fast path hides.

Performance: time the scatter alone on a real 120k-point scan. Above ~5 ms means something fell back to the CPU — look for a stray `.cpu()` or `.item()`.

---
## Part 10 — Layer 6: Observability and Negative Obstacles

**Files:** `observability/raycast.py`, `observability/negative_obstacle.py`
**One-line goal:** track what the sensor **failed** to see, and turn that absence into the hazard class that matters most to a ground vehicle.

### Why absence is the signal

Every layer so far reasons about returns that came back. A ditch, trench, culvert or crater — the signature defence-relevant hazard — is defined by returns that **did not**. A 2D occupancy grid renders a 3 m trench as perfectly clear ground because nothing occupied it. The statement gestures at this with "potholes"; DRDO lives with it.

This is also where `UNKNOWN` starts meaning something. A cell never observed and a cell observed and found empty are radically different facts, and collapsing them is how vehicles drive confidently into things they never looked at.

### 10.1 The four-state model

| State | Meaning | Established by |
|---|---|---|
| `UNOBSERVED` | No beam ever interacted with this cell | Default |
| `FREE` | Beams passed through and terminated beyond | Ray traversal |
| `OCCUPIED` | Beams terminated here | Point returns |
| `OCCLUDED` | A beam would have passed but was blocked closer in | Ray traversal behind a termination |

`OCCLUDED` is what keeps the map honest — the area behind a truck is not free space, and a planner that treats it as free will route through it.

Traversal is a 2D DDA along each beam's ground projection, run at the coarsest level for speed and refined to L0 only inside the fine ring. Full 3D voxel traversal is unnecessary in a 2.5D map.

**Free-space carving earns its keep twice:** it also removes stale dynamic objects. A car that has driven off leaves a stale `OCCUPIED` cell if it was ever mistakenly written to the static layer; beams now passing through carve it back to `FREE`. Without carving, the static map slowly fills with ghosts.

### 10.2 The expected-return test

The reasoning: *"geometry says ring $k$ should have returned from the ground at $r_{\text{exp}}$. It came back at $r_{\text{meas}} \gg r_{\text{exp}}$, or not at all. So the ground dropped away."*

For a beam at depression $\alpha$ from height $h$ over locally flat ground, $r_{\text{exp}} = h/\tan\alpha$. If the ground drops by $d$:

$$r' = \frac{h+d}{\tan\alpha} = r_{\text{exp}}\left(1 + \frac{d}{h}\right) \quad\Longrightarrow\quad \boxed{\ \Delta = r' - r_{\text{exp}} = \frac{r_{\text{exp}}\, d}{h}\ }$$

**The range shadow is amplified by $r/h$** — the effect *grows* with range even as sampling gets sparser, which is why the test stays sensitive despite the ground being poorly sampled far away.

**Worked example.** $h = 1.73$ m, a $d = 0.5$ m ditch at $r = 20$ m: $\Delta = 20 \times 0.5/1.73 = 5.78$ m. The beam that should have landed at 20 m lands at 25.8 m — a **5.8 metre** hole in the ground return pattern from a half-metre ditch. Not a subtle statistical signal; unmistakable.

**So why is detection range only ~21 m?** Because detection is limited not by $\Delta$ but by whether a ring lands in the ditch at all. Radial ring spacing at 20 m is 1.72 m and grows as $r^2$; a 2 m ditch is straddled out to $r_{\max} = \sqrt{wh/\Delta\phi} = 21.6$ m, beyond which rings step clean over it.

**Detection is sampling-limited, not signal-limited.** That sentence deserves its own slide, because it tells the sponsor exactly what to change — mount height, beam count, or a downward-canted auxiliary sensor — to extend the range. You are not only detecting hazards; you are telling them how to detect them further out.

### 10.3 The false positive that will embarrass you if you skip it

The model above assumes **locally flat ground at height $h$**. On a crest, a downslope, or a banked turn that is violated and the test fires everywhere. A demo screaming "ditch!" while cresting a hill is worse than no detector.

Compute $r_{\text{exp}}$ from a **locally fitted ground plane** built from the last few *confirmed* ground returns in the same azimuth column — which Part 5.2's column-wise walk already produced. A consistent downslope shifts all rings together and the residual stays near zero; a ditch shifts one or two rings against their neighbours. **The discriminator is local inconsistency, not absolute deviation.**

Then require temporal confirmation: a candidate must persist across $\ge 3$ consecutive scans before promotion from `SUSPECT` to `NEGATIVE_OBSTACLE`. A single missing return is far more likely to be a specular surface, a puddle, or a dropout than a hole in the world.

### Edge cases and how they're fixed

| Edge case | What would go wrong | The fix |
|---|---|---|
| Slope or crest | Fires everywhere on exactly the terrain DRDO cares about | Locally fitted plane; test measures ring-to-ring inconsistency |
| Water, wet asphalt, polished metal | Specular return of nothing — identical to a hole | Multi-echo/intensity check, 3-scan confirmation; where unresolved report `UNKNOWN`, never `FREE` |
| Rain, dust, exhaust | Spurious near returns and dropouts | Multi-echo filtering (keep last return for dust), low-intensity dropout rejection. Partially mitigated, declared in Part 24 |
| Occlusion behind a large vehicle read as a ditch | Both produce missing ground returns | Carve first, then test; the detector never runs on `OCCLUDED` cells. **Ordering matters** |
| Ditch beyond sampling range | Silently reports clear ground where a hazard exists | Spec sheet states the range; beyond it cells are `UNKNOWN` (Part 11), not `FREE` |
| Ray traversal at L0 across 102 m | Millions of cell visits per beam; blows the budget | Traverse coarse, refine to L0 only inside the fine ring |

### How to test this module

`pytest tests/test_raycast.py tests/test_negative_obstacle.py`.

Pure geometry, so build synthetic beam patterns with analytically known answers: flat ground (expect zero detections); flat ground with one 0.5 m × 2 m ditch at 15 m (expect detection, $\Delta \approx 4.3$ m); **a constant 8° downslope (expect zero detections — write this test first, it guards the fix that stops the demo embarrassing you).**

CARLA validation turns this from a claim into a result: dig trenches of known width and depth at known ranges, sweep past them, plot detection rate vs. range, and overlay the predicted $\sqrt{wh/\Delta\phi}$ curve. **If measurement tracks prediction, you have demonstrated the sensor model is correct** — one plot validating Parts 3, 10 and 11 simultaneously.

---

## Part 11 — Layer 7: The Sparsity Trap and Derived Confidence

**File:** `observability/sparsity.py`
**One-line goal:** stop reporting "clear" where the sensor merely lacked the resolving power to see — and compute, rather than tune, the range at which that happens.

### The trap

Low-density, low-resolution far cells routinely cause thin real objects — poles, fence posts, cyclists, mast antennas — to be statistically indistinguishable from noise, and to be silently smoothed away. The system then reports **"no obstacle"**, which is a much stronger claim than it can support. It should be reporting **"I cannot tell."**

This is a named gap with no standard treatment in the literature, and it is the most citable original contribution in the pipeline. But an original contribution defended by a hand-tuned threshold is a liability under questioning. Part 3 lets us derive it instead.

### The expected-return count

An object of vertical extent $t$ and horizontal extent $w$ at range $r$ subtends $t/(r\Delta\phi)$ beam rings vertically and $w/(r\Delta\theta)$ samples horizontally, so:

$$\boxed{\ N_{\exp}(r; t, w) = \frac{t}{r\Delta\phi}\cdot\frac{w}{r\Delta\theta} = \frac{t\,w}{r^2\,\Delta\phi\,\Delta\theta}\ }$$

Returns fall off as $1/r^2$ — not as $1/r$, which is the intuition most people carry, and it is why long-range thin-object detection collapses so much faster than expected.

The **blind range**, where a minimum object of interest drops below one expected return:

$$r_{\text{blind}}(t,w) = \sqrt{\frac{t\,w}{\Delta\phi\,\Delta\theta}}$$

With $\Delta\phi\Delta\theta = 2.239\times10^{-5}$ for the HDL-64E:

| Object | $t \times w$ | $N_{\exp}$ @50 m | $N_{\exp}$ @100 m | $r_{\text{blind}}$ |
|---|---|---|---|---|
| Pedestrian | 1.7 × 0.5 m | 15.2 | 3.8 | **195 m** |
| Utility pole | 3.0 × 0.2 m | 10.7 | 2.7 | **164 m** |
| Thin fence post | 1.0 × 0.1 m | 1.8 | **0.45** | **67 m** |
| 1 m of 15 cm kerb | 0.15 × 1.0 m | 2.7 | 0.67 | **82 m** |

**Presence is not height.** Note the last row against Part 3's spec sheet: a kerb still *returns points* out to 82 m, but its 15 cm face cannot be *resolved* beyond $t/\Delta\phi = 20.2$ m. Two different questions with two different formulas — $N_{\exp}$ answers "is anything there?", $t/\Delta\phi$ answers "how tall is it?". Conflating them is how a system reports a kerb it cannot measure. Both are reported, separately, in the spec sheet.

### The decision rule

Per cell, with $N_{\text{obs}}$ observed returns and $\kappa = N_{\text{obs}} / N_{\exp}(r; t_{\min}, w_{\min})$ for the smallest object the vehicle must not hit:

| Condition | Verdict | Rationale |
|---|---|---|
| $\kappa \ge 1$, $N_{\text{obs}} > 0$ | Normal confidence | Resolvable and observed |
| $0 < \kappa < 1$, returns **structured** | `SPARSE_STRUCTURED` — flagged, low confidence, **not free** | Too few returns for a minimum object, but they are not noise-shaped |
| $0 < \kappa < 1$, returns unstructured | Noise; suppressed | Scattered in range and ring index |
| $N_{\text{obs}} = 0$, $N_{\exp} \ge 1$ | `FREE` | The sensor *would* have seen it. Absence is informative |
| $N_{\text{obs}} = 0$, $N_{\exp} < 1$ | **`UNKNOWN`, never `FREE`** | Past $r_{\text{blind}}$. The sensor **could not** have seen it. Absence proves nothing |

The last row is the whole contribution in one line: **there is a range beyond which "no returns" cannot mean "no obstacle", and it is computable from the datasheet.**

**The structure test.** Real thin objects return points that are *co-located*: tightly clustered in range and contiguous in ring index. Noise returns are scattered in both. Concretely, flag as structured when the range spread of a cell's returns is below a small fraction of the expected object depth **and** the returns occupy adjacent ring indices. Both inputs are already computed — Part 4's collision count and `range_gradient` channel.

### Why this is a derived threshold, not a tuned one

$N_{\exp}$ contains no free parameters. It takes $\Delta\phi$ and $\Delta\theta$ from the datasheet and $(t_{\min}, w_{\min})$ from the vehicle's own safety requirement — *"the smallest thing we refuse to hit"* — which is a specification the sponsor supplies, not a number you fit.

So the answer to *"is your confidence layer genuinely calibrated or just plausible-looking?"* is: **it is not calibrated at all, it is derived.** The claim changes from *"low density but structured, so probably real"* to:

> *"This cell returned 2 points where beam geometry predicts 2.7 for a 20 cm pole at 100 m. That is consistent with a pole and inconsistent with noise. And at 130 m the same pole would predict 1.6 returns, so beyond that we report UNKNOWN rather than FREE."*

The same $\kappa$ also weights the confidence-weighted cross-entropy in Part 5.4, so training loss and map confidence share one definition rather than two that can drift apart.

### The metric this enables

**Sparsity Trap recovery rate** — the percentage of small/thin ground-truth objects at long range correctly flagged (as `SPARSE_STRUCTURED` or `UNKNOWN`) rather than silently reported free. No standard benchmark defines this; we define it, say so explicitly, and report it in its own column. An admitted gap turned into a measured contribution.

Because $N_{\exp}$ is a curve, the metric has a **predicted shape to be measured against** — recovery rate should hold near 1 up to $r_{\text{blind}}$ and degrade past it. Plotting measured against predicted is the same predicted-vs-measured structure as the hazard spec sheet, and it is far more convincing than a single percentage.

### Edge cases and how they're fixed

| Edge case | What would go wrong | The fix |
|---|---|---|
| Everything past $r_{\text{blind}}$ marked `UNKNOWN` | Half the map is unknown; planner paralysed | `UNKNOWN` carries a *finite* cost (Part 14), and $r_{\text{blind}}$ is computed for $t_{\min}$ — a *small* object. Large obstacles remain confidently `FREE`/`OCCUPIED` far beyond it |
| $t_{\min}$ chosen unrealistically small | $r_{\text{blind}}$ collapses toward the vehicle; everything is unknown | $t_{\min}, w_{\min}$ come from the vehicle config as a stated safety requirement, and the resulting $r_{\text{blind}}$ is reported so the choice is visible and arguable |
| Occlusion mistaken for sparsity | An object hidden behind a truck returns nothing and gets flagged sparse | Part 10 runs first; `OCCLUDED` cells are excluded from the sparsity test |
| Structure test on a single return | One point is never "structured" or "unstructured" | $N_{\text{obs}} = 1$ is `SPARSE_STRUCTURED` by default — the cautious verdict — and confidence reflects the single sample |
| Sparse cells flooding the display | Dashboard unreadable, judges see noise | Rendered as a distinct low-salience overlay, toggleable; the demo beat shows it turning *on* to reveal what the baseline silently dropped |
| $N_{\exp}$ computed with the wrong sensor constants | The entire layer is quietly wrong | Shares one `SensorModel` with Part 3; the point-distribution validation in Part 3 gates it |

### How to test this module

`pytest tests/test_sparsity.py`.

Pure functions: assert $N_{\exp}$ matches hand-computed values from the table above; assert $r_{\text{blind}}$ for a pedestrian is ~195 m and for a thin fence post ~67 m; assert that $N_{\text{obs}} = 0$ past $r_{\text{blind}}$ yields `UNKNOWN` and before it yields `FREE`. **That last assertion is the machine-checked form of Claim 3.**

CARLA validation: place poles of known diameter at a ladder of ranges, sweep past, and plot observed return count against the $N_{\exp}$ curve. If observations track prediction, the confidence layer is grounded in measured physics rather than assertion — and that plot is what you show when a judge pushes on it.

---

## Part 12 — Layer 8: Temporal Fusion

**Files:** `temporal/static_layer.py`, `temporal/tracker.py`, `temporal/promote_demote.py`
**One-line goal:** make the map a *map* rather than a per-frame snapshot — accumulating static structure while keeping moving things out of it, and handling regions migrating between resolution levels as the vehicle drives.

### 12.1 The static/dynamic split, and the smear that gives everyone else away

Accumulate every point into a persistent grid and a walking pedestrian leaves a **trail of occupied cells** — a smear the planner treats as a wall. Every team that treats accumulation as "just add the new points" will have this in their demo, and it is visible from across the room.

Two representations:

- **Static layer** — the clipmap of Part 8, accumulated in a world-anchored frame. Structure, terrain, buildings.
- **Dynamic set** — tracked objects as *parametric entities* `(id, position, velocity, extent, class, confidence)`, never written into the grid.

**Composited at query time, not merged at write time.** `lookup()` returns the static cell plus any dynamic entity overlapping it. Because the pedestrian is an entity rather than written cells, there is nothing to leave behind.

### 12.2 Two motion mechanisms, two jobs

This is the division of labour that makes the motion story coherent, and it is why we keep a tracker despite having residual channels:

| Mechanism | Grain | Answers | Feeds |
|---|---|---|---|
| **Residual channels** (Part 4.4) | Per point, per frame | *Is this point moving?* | The class decision — stops a parked car ever entering the dynamic candidate set |
| **Tracker** | Per entity, across frames | *What is this thing, where is it going, how big is it?* | Query-time composite; fovea refinement (Part 13); track termination |

Residuals are cheap and dense but stateless — they cannot give an identity or a velocity vector, and the fovea controller needs both. The tracker is stateful but needs candidates, and residuals supply better candidates than a class label ever could. **Two independent mechanisms that can disagree is a feature; one mechanism silently wrong is not** — a residual spike with no corresponding track is exactly the signature of odometry drift (Part 4.4), and having both lets you notice.

**The tracker itself is deliberately simple** — not the contribution, do not gold-plate it. Cluster dynamic-candidate points (DBSCAN or connected components on the fine grid) → associate to existing tracks by nearest centroid within a gate → constant-velocity Kalman filter → confirm after $M$ of $N$ associations, drop after $K$ misses.

### 12.3 Accumulation and forgetting

Static cells fuse per field: $h_{\max} \leftarrow \max(h_{\max}, z)$, $\text{count} \leftarrow \min(\text{count}+n,\ 65535)$, and mean/variance combine exactly via the Chan parallel-axis formula, which merges two independently-accumulated groups without revisiting a point:

$$n = n_A + n_B, \quad \delta = \mu_B - \mu_A, \quad \mu = \mu_A + \delta\frac{n_B}{n}, \quad M_2 = M_{2,A} + M_{2,B} + \delta^2\frac{n_A n_B}{n}$$

**Confidence decays with age** so a cell observed 40 seconds ago does not present itself as freshly measured:

$$\text{conf} \leftarrow \text{conf}\cdot e^{-(t - t_{\text{cell}})/\tau}, \qquad \tau \approx 10\ \text{s}$$

**Bound the accumulation window.** Odometry drift smears the static layer over long horizons. Cap history at a few seconds of travel unless the pose source is genuinely good. A crisp 3-second map beats a blurry 60-second one, and for *local* perception 3 seconds is what a planner needs. This is a local perception map, not a SLAM system, and it should never be described as one.

### 12.4 Resolution promotion and demotion — the problem nobody will anticipate

The fovea is ego-centric, so a patch at 60 m (L2, 20 cm) becomes 20 m (L1, 10 cm) then 8 m (L0, 5 cm). Cells migrate both ways.

**Demotion (fine → coarse) is exact.** Four L$\ell$ cells reduce to one L$(\ell{+}1)$ cell with no error: $h_{\max}$ by max, $h_{\min}$ by min, counts summed, mean/variance by Chan, class by weighted mode. **This is exact only because of power-of-two nesting** (Part 8) — with a tree over non-aligned cells, or a continuously-varying polar bin width, the four children do not tile the parent and the reduction becomes an approximation with no error bound. The data structure choice pays off here, concretely.

**Promotion (coarse → fine) cannot create information.** Three options, one honest:

| Option | Behaviour | Verdict |
|---|---|---|
| Leave fine cells `UNOBSERVED` | Safe, but discards real prior knowledge; the fine ring flickers empty as you approach | Wasteful |
| Copy the coarse value into all four children | Fabricates detail and presents it as measured | **Never** |
| Copy, mark `PROVISIONAL`, reduce confidence | Retains prior knowledge, tells the truth about its quality, overwritten by the first real measurement | **Correct** |

A planner may use provisional cells for coarse routing and must not use them for a 12 cm kerb decision. Any cell receiving a real measurement clears the flag immediately.

**Why this matters in the demo:** teams that ignore it get a flickering seam at the ring boundary, roughly 12 m ahead of the vehicle, in the safety-critical zone, every frame. If your seam is clean and theirs flickers, a judge sees the difference without any explanation. Worth a side-by-side.

### 12.5 `INFERRED` completion — filling occlusion gaps without inventing terrain

Occlusion leaves holes: the blind spot behind a parked car, the far side of a crest. Those holes are `UNKNOWN`, the planner costs them highly, and in a cluttered scene that can be most of the map.

The tempting fix is **generative inpainting** — train a model to hallucinate plausible terrain in occluded regions, and emit an uncertainty layer alongside. It is an active research direction and it produces impressive pictures. **We reject it, and Part 22 records why:** it breaks Principle 6 and with it the conservatism invariant. The blind spot behind a parked car is precisely where a ground vehicle needs the map to be *honest* rather than *plausible*, and a system whose primary output includes generated geometry is much harder to defend to a sponsor who has to certify it.

There is a version that keeps the invariant intact, and `PROVISIONAL` is already its skeleton. **Geometric completion, not generation:**

1. Only across **small** gaps — a few cells, bounded by a configured maximum span.
2. Only by **continuing the local ground plane** already fitted from confirmed returns on both sides (Part 5.2's output, reused a third time).
3. Only where the surrounding cells **agree** — if the plane fits on either side of the gap disagree, the gap stays `UNKNOWN`.
4. Result is flagged **`INFERRED`**, carries reduced confidence, and is **forbidden from ever reducing cost** (Part 16).

The difference from generative inpainting is not a matter of degree. A plane continuation is a deterministic, explainable, single-formula extension of a measurement, with a stated span limit and an agreement precondition. It can be wrong, but it can be *audited* — you can point at the two fits it interpolated between. A generated surface cannot.

**And the honest limit:** this does not recover the car-shaped occlusion shadow, which is exactly the case where a bounded plane continuation refuses to act. That case stays `UNKNOWN` and gets costed as such, which is the correct answer even though it is the less impressive one.

### 12.6 Per-track residual consistency — using the two mechanisms against each other

§12.2 says two independent motion mechanisms that can disagree is a feature. This makes it operational.

A rigid object moves rigidly: every point on a real vehicle shares one velocity. The tracker enforces that **structurally** — a Kalman-filtered track has one velocity by construction, so there is no per-point "tearing" of the kind that afflicts scene-flow networks predicting velocity independently per point. (Some architectures add an *instance consistency loss* to obtain this; we get it from the representation instead, which is cheaper and cannot be violated.)

That structural guarantee is also a **diagnostic**. Compute the variance of the motion residual across the pixels belonging to each confirmed track:

$$\text{Var}\big(R_k(u,v)\big)_{(u,v)\in\text{track}}$$

For a genuine rigid moving object this is low — all its points disagree with the past sweep by a consistent amount. High variance means one of three things, all worth knowing:

| High per-track residual variance means | Response |
|---|---|
| Partial occlusion — half the object is hidden | Lower track confidence; do not trust the extent estimate |
| Track ID swap between two crossing objects | Flag for re-initialisation |
| Odometry drift being read as motion | Cross-check against $\|\nabla r\| \cdot \|\omega\|$ (§4.4) — if the variance concentrates at the track's silhouette edges during a turn, it is drift, not motion |

This costs one reduction per track per frame and turns "the mechanisms can disagree" from a rhetorical strength into a measured track-quality signal that feeds the fovea controller's confidence in what it is refining around.

### Worked example

A kerb patch is first observed at 45 m (L2). The kerb is 15 cm tall; at 45 m the vertical sampling is 33 cm, and Part 3 says the face cannot be resolved beyond 20.2 m — so the cell records flat ground, **correctly, because that is all the sensor can know from there.** Part 11 marks it low-$\kappa$, so it is not confidently `DRIVABLE`.

At 18 m the patch migrates to L1, promoted `PROVISIONAL` from its L2 parent. Still marginal — at most one ring on the face.

At 10 m it migrates to L0. Vertical sampling is 7.4 cm; two rings land on the face. `h_max − h_min` jumps to 0.15 m, `PROVISIONAL` clears, and the cell reports a 15 cm step — at exactly the range the physics predicted.

**Notice what the map never did:** it never claimed the kerb was absent. It said *low confidence, provisional, flat as far as I can tell from here*, and upgraded when the geometry permitted. Knowing the boundary of its own knowledge is what separates this from a system that reports flat ground with full confidence and then finds the kerb with a wheel.

### Edge cases and how they're fixed

| Edge case | What would go wrong | The fix |
|---|---|---|
| Dynamic objects accumulated into the static grid | Pedestrian smear trails; planner sees ghost walls | Static/dynamic split, composited at query time |
| Parked car labelled `car` → dynamic | Never enters the static map; planner routes through it | Motion evidence from residuals, confirmed by tracker velocity — not the class label |
| Residual spike with no corresponding track | Odometry drift misread as motion | The two mechanisms disagreeing is the detector; suppress where `range_gradient` is high (Part 4.4) |
| Promotion fabricating fine detail | Invented 5 cm structure presented as measured | `PROVISIONAL` + reduced confidence, cleared on first measurement |
| Odometry drift over long accumulation | Static layer smears into thick blurred walls | Bounded window; LiDAR odometry if no pose source |
| Track not dropped when an object leaves | Ghost entity persists in the composite | $K$-miss termination, plus Part 10's free-space carving as an independent second defence |
| Track ID swap between crossing pedestrians | Velocity estimates jump; fovea chases the wrong target | Accepted at this scope and declared; refinement covers a *region* around both, so a swap degrades allocation but never safety |
| Cell never re-observed but still confident | Stale data drives decisions | Exponential confidence decay, $\tau \approx 10$ s |

### How to test this module

`pytest tests/test_static_layer.py tests/test_tracker.py tests/test_promote_demote.py`.

**Demotion exactness** is the important one: fill four children with known values, demote, and assert the parent's `h_max`, `h_min`, `count` and combined variance match a direct computation over the union of all points. Exact to floating-point tolerance, not approximately right — approximate means the nesting is broken.

**The smear test:** replay a sequence with a walking pedestrian, then query every cell along their path. All must be `FREE` or `UNOBSERVED`; none `OCCUPIED`. Automated form of the visual check; put it in CI.

**Promotion, manual:** drive toward a known kerb with `PROVISIONAL` rendered distinctly. Provisional cells should appear ahead and clear as you close — a visible, explicable wave. If they never clear, promotion is not being overwritten by measurement.

---
## Part 13 — Layer 9: The Adaptive Fovea Controller (time-to-contact, not distance)

**Files:** `attention/ttc.py`, `attention/fovea_controller.py`
**One-line goal:** decide *where* resolution is worth spending — by how soon the vehicle could reach something, not by how far away it is.

### Why distance is the wrong variable

The statement's title says **adaptive**, and every team will make resolution a function of Euclidean range. But a fovea does not sit fixed at the optical axis; it *moves*, following where attention is needed.

At 60 km/h (16.7 m/s), an obstacle 40 m directly ahead is **2.4 seconds** away and is the most safety-critical object in the scene. The same obstacle 40 m to the left, travelling straight, will never be reached. Spending identical resolution on both is not neutral — it is spending the safety budget on the object that cannot hurt you.

### The math

For a static point $p$ with ego velocity $\mathbf v$:

$$v_{\text{close}}(p) = \mathbf v \cdot \hat p, \qquad \text{TTC}(p) = \frac{\|p\|}{\max\big(v_{\text{close}}(p),\ v_{\min}\big)}, \qquad c_{\text{ttc}}(p) = c_0\left(\frac{\text{TTC}(p)}{\tau_0}\right)^{\gamma}$$

$\tau_0$ is the fine horizon (~1 s), $\gamma$ the foveation exponent (the demo slider), $v_{\min}$ (~2 m/s) prevents blow-up at standstill. The fine region becomes an **elongated teardrop aligned with the velocity vector**, stretching with speed and rotating with steering — visually striking *and* physically principled, which is a rare combination.

### The composition rule that keeps this safe

The most important paragraph in this part, and what stops a clever idea becoming a liability:

$$c(p) = \min\Big(c_{\text{range}}(r),\ c_{\text{ttc}}(p),\ c_{\text{object}}(p),\ c_{\text{boundary}}(p)\Big)$$

Taking the **minimum** means every extra term can only make a cell *finer*. Part 3's schedule $c_{\text{range}}$ is a **hard floor that is never relaxed**, so:

- The requirement is satisfied unconditionally, in every direction, at every speed, in every mode.
- No bug in the fovea controller and no bad velocity estimate can make the map coarser than specified.
- The clever feature can only ever *add* safety margin.

When a judge asks *"what if your TTC estimate is wrong?"*, the answer is **"the map degrades to exactly the specified behaviour"** — a stronger answer than any accuracy figure.

The refinement terms:

- **$c_{\text{object}}$** — any tracked entity gets a fine patch regardless of range or TTC. A pedestrian 45 m to the side has TTC ≈ ∞ and would otherwise coarsen; they are refined because they are *a thing that can move*, and the point of tracking them is that their future TTC is not their present one. **This is what makes lateral coarsening safe.**
- **$c_{\text{boundary}}$** — refine where the class label changes between neighbours. The `DRIVABLE` → `NON_TRAVERSABLE` transition *is* the kerb, the verge, the drop-off. A flat road's interior needs almost no resolution; its edge needs all of it. One comparison per neighbour pair, and it puts detail exactly where geometry is changing.

### The two profiles

| Profile | Definition | Use |
|---|---|---|
| **A — Spec** (default) | $c = c_{\text{range}}(r)$ only | Demoed first; what every number in Part 18 is reported for |
| **B — Adaptive** | Full composition, relaxed lateral baseline | The differentiator, demoed second, with its own metric |

**Demo Profile A first and say the numbers out loud.** SIH judges can be conservative, and a team that appears to have ignored the stated requirement in favour of something cleverer loses on compliance before it is judged on innovation. Show you meet the spec exactly. *Then* show you can do better.

### The metric that makes Profile B rigorous

"Uses less memory" is not a claim of merit — you can always use less memory by being worse. Hold *safety* constant and compare *cost*:

> **Safety-equivalent resolution at fixed time-to-contact.** For every bearing around the vehicle, find the point where TTC = 2 s and record the cell size available there. Profile B beats Profile A only if it provides cell sizes at least as fine at **every** TTC-2s point while using less total memory.

"We reduced memory 30%" is not defensible. "We reduced memory 30% while providing equal or finer resolution at every point the vehicle could reach within two seconds" is. Report it as a polar plot, both profiles overlaid.

### Worked example

15 m/s, $\tau_0 = 1.0$ s, $c_0 = 5$ cm, $\gamma = 1$, $v_{\min} = 2$ m/s:

| Point | $v_{\text{close}}$ | TTC | $c_{\text{ttc}}$ | $c_{\text{range}}$ | **Final** |
|---|---|---|---|---|---|
| 15 m ahead | 15 m/s | 1.0 s | 5 cm | 5 cm | **5 cm** |
| 60 m ahead | 15 m/s | 4.0 s | 20 cm | 20 cm | **20 cm** |
| 100 m ahead | 15 m/s | 6.7 s | 33 cm | 40 cm | **33 cm** — TTC refines beyond spec |
| 15 m left | 0 → $v_{\min}$ | 7.5 s | 38 cm | 5 cm | **5 cm** — floor holds |
| 45 m left, tracked pedestrian | 0 | ∞ | 40 cm (clamp) | 20 cm | **5 cm** — object override |

The last two rows are where a naive TTC implementation becomes dangerous, and the composition rule handles both without a special case.

**Standstill.** As $\mathbf v \to 0$ every TTC → ∞ and a naive implementation coarsens the whole map exactly when a pedestrian might walk up to the vehicle. Two defences: $v_{\min}$ floors the closing speed, and the closing speed of *tracked objects toward the ego* is added to $v_{\text{close}}$ — so at standstill, detail follows the things that are moving, which is correct.

### Edge cases and how they're fixed

| Edge case | What would go wrong | The fix |
|---|---|---|
| Adaptive fovea coarsens below spec | Fails the stated requirement — disqualifying | $\min()$ with $c_{\text{range}}$ as a hard floor; structurally impossible |
| Vehicle stationary | Coarsens exactly when a pedestrian approaches | $v_{\min}$ floor plus object closing speed |
| Bad velocity estimate | Fovea points the wrong way | Degrades to Profile A by the floor rule; velocity low-pass filtered and rate-limited |
| Sharp steering | Fovea lags the actual path | TTC uses the predicted path from current yaw rate, not the instantaneous velocity vector |
| Object override thrashing | Resolution oscillates as tracks appear/disappear | Hysteresis: refine immediately, coarsen only after $N$ track-free frames |
| Claiming Profile B is better on memory alone | Trivially true and meaningless | Safety-equivalent-resolution-at-fixed-TTC, as a polar plot |

### How to test this module

`pytest tests/test_ttc.py tests/test_fovea_controller.py`.

**The floor test is the first test in the file:** generate thousands of random ego velocities, yaw rates and query points, and assert `c(p) <= c_range(r)` for every one. That property test is the machine-checked form of "we cannot violate the spec".

Then: standstill produces cell sizes no coarser than Profile A; a tracked object always yields $c \le 2c_0$ inside its radius; the fovea's principal axis rotates with the velocity vector.

Manual: put $\gamma$ on a live slider and drive. **A live slider is worth more than any static figure** — it proves the system is parameterised rather than a hardcoded animation, and judges have seen enough hardcoded animations to be checking.

---

## Part 14 — Layer 10: Traversability and the Planner Interface

**Files:** `planning/traversability.py`, `planning/costmap.py`, `planning/ros_bridge.py`
**One-line goal:** turn the map into something a planner consumes, parameterised by the *vehicle* rather than by the dataset.

A semantic map is not a decision. `TRAVERSABLE_CAUTION` is not an instruction. The gap between "this cell is grass" and "this vehicle may cross at 4 m/s" is where a perception project becomes a system. It is also what closes the loop in the demo: a coloured map shows perception; a path re-routing around a pedestrian shows **consequence**, and consequence is what judges remember.

### Per-cell geometric derivatives

From the local $3\times3$ neighbourhood at the queried level:

$$\text{slope} = \arccos(\mathbf n \cdot \hat z), \qquad \text{roughness} = \sqrt{\text{Var}(z)_{3\times3}}, \qquad \text{step} = \max_{k\in\mathcal N}\big|h^{\text{cell}}_{\max} - h^{(k)}_{\max}\big|$$

plus **clearance** from the multi-layer cell (Part 9).

### The vehicle model

Declared once, in config, never hardcoded:

```yaml
vehicle:
  max_slope_deg:       25
  max_step_height_m:   0.20
  min_clearance_m:     2.50
  ground_clearance_m:  0.35
  width_m:             2.10
  max_roughness_m:     0.08
  min_object_t_m:      1.00   # feeds r_blind in Part 11
  min_object_w_m:      0.10
```

$$\text{cost} = \begin{cases}
\text{LETHAL} & \text{slope} > \text{max\_slope} \\
\text{LETHAL} & \text{step} > \text{max\_step\_height} \\
\text{LETHAL} & \text{clearance} < \text{min\_clearance} \\
\text{LETHAL} & \text{class} = \texttt{NEGATIVE\_OBSTACLE} \\
\text{UNKNOWN\_COST} & \texttt{UNOBSERVED},\ \texttt{SPARSE\_STRUCTURED},\ \text{or } \kappa < \tau_\kappa \\
w_1\frac{\text{slope}}{\text{max\_slope}} + w_2\frac{\text{rough}}{\text{max\_rough}} + w_3\,\text{class\_penalty} & \text{otherwise}
\end{cases}$$

Three properties worth defending out loud:

- **The map is vehicle-agnostic; the cost map is not.** Swap the config and the same stored map yields a different cost map. A jeep and a tracked platform read one map differently and neither needs re-mapping. That is what makes this a component, not a demo. Note the vehicle config also supplies $(t_{\min}, w_{\min})$ to Part 11 — *the sponsor's safety requirement propagates all the way into what the map is willing to call free.*
- **`UNKNOWN_COST` is not zero and not lethal.** A tunable high-but-finite cost, so the planner prefers known-good routes but can traverse unknown ground when it must. Zero drives into unmapped holes; lethal paralyses the vehicle in any partially-observed scene. Both are common failure modes and both come from collapsing the tri-state.
- **Low confidence degrades toward caution, never optimism.** Every uncertainty path in this document resolves in one direction, by construction.

### Output format

Emit standard **ROS `grid_map`** and `nav_msgs/OccupancyGrid`, with the multi-layer fields as named layers (`elevation`, `clearance`, `traversability`, `confidence`, `class`, `observability`). An afternoon of work that buys a great deal: the system drops into an existing autonomy stack without a rewrite, which is exactly the question a defence-production evaluator asks after "does it work". A module nobody can integrate is a research artifact.

### Edge cases and how they're fixed

| Edge case | What would go wrong | The fix |
|---|---|---|
| Unknown treated as free | Plans confidently through unobserved ground into a ditch | Explicit finite `UNKNOWN_COST`; the tri-state propagates from Part 10 all the way here |
| Unknown treated as lethal | Vehicle refuses to move in any partially-observed scene | Same fix — finite, tunable, not binary |
| Vehicle parameters hardcoded | Map useless on the sponsor's actual platform | Vehicle config; the cost function takes it as an argument |
| Step height computed at a coarse level | A 15 cm kerb is invisible in a 40 cm cell | Computed only at L0/L1; returns `UNKNOWN` beyond, with the spec-sheet range stated |
| Slope from a 3×3 fit on single-point cells | Meaningless normal from noise | Minimum point count across the neighbourhood, else `UNKNOWN_COST` |
| Cost map recomputed over the whole extent every frame | Wastes budget on cells nobody queries | Lazy evaluation on the hot plane, within the planner horizon only |

### How to test this module

`pytest tests/test_traversability.py tests/test_costmap.py tests/test_ros_bridge.py`.

Pure functions over a synthetic neighbourhood: a 25° slope, a 21 cm step, a 2.4 m clearance each independently produce `LETHAL` for the config above, and each *just* under threshold does not. **Assert `UNOBSERVED` and `SPARSE_STRUCTURED` both produce `UNKNOWN_COST` and never zero** — that assertion encodes the whole tri-state argument, so make it prominent.

Integration: replay a pedestrian crossing, run A* over the cost map each frame, confirm the path re-routes and recovers. **That clip is your demo.**

---

## Part 15 — Layer 11: The Perception-Limited Speed Envelope

**File:** `planning/speed_envelope.py`
**One-line goal:** answer the question the whole document has been building toward — *given what this sensor can actually see, how fast may this vehicle safely go?*

### Why this part exists

Every layer so far produces a better map. This one produces an **operational limit**, and it is the layer that translates the entire perception analysis into a unit the sponsor works in.

The argument is short. A vehicle can only stop within some distance. The sensor can only detect a given hazard out to some range. If stopping distance exceeds detection range, **the vehicle is driving faster than it can see**, and no amount of mapping quality fixes that — it is a kinematics problem wearing a perception costume.

Both quantities are already derived. Part 3 gives detection range per hazard from beam geometry; the vehicle config gives braking capability. Putting them together takes one line of algebra and produces the most actionable number in the project.

### The math

$$d_{\text{stop}}(v) = v\,t_{\text{react}} + \frac{v^2}{2a}$$

Setting $d_{\text{stop}} = R$ and solving the quadratic for the largest safe speed given detection range $R$:

$$\boxed{\ v_{\max}(R) = -a\,t_{\text{react}} + \sqrt{a^2 t_{\text{react}}^2 + 2aR}\ }$$

With $a = 4$ m/s² (a conservative figure for a wheeled UGV on loose surface) and $t_{\text{react}} = 0.3$ s covering pipeline latency plus actuation:

| Hazard the terrain may contain | Detection range (Part 3 / Part 11) | **Max safe speed** |
|---|---|---|
| 5 cm cable at 2.1 m | 6.7 m | **6.24 m/s — 22.5 km/h** |
| 15 cm kerb (height resolved) | 20.2 m | **11.57 m/s — 41.7 km/h** |
| 2 m ditch | 21.6 m | **12.00 m/s — 43.2 km/h** |
| 4 m crater | 30.5 m | **14.47 m/s — 52.1 km/h** |
| Thin fence post (presence, $r_{\text{blind}}$) | 66.8 m | **21.95 m/s — 79.0 km/h** |

Read the table as a *binding constraint*: the vehicle's safe speed is set by the **shortest** detection range among the hazard classes the terrain can plausibly contain. On a maintained track where the worst case is a kerb, that is 42 km/h. In open terrain where ditches are possible, 43 km/h. Anywhere overhead cables exist, **22.5 km/h**.

### The headline

> **The sensor, not the vehicle, is the speed limit.**

A UGV with this LiDAR at this mount height cannot safely exceed roughly **43 km/h** in terrain where a 2 m ditch is possible — not because the drivetrain cannot go faster, and not because the perception software is slow, but because a 64-beam sensor at 1.73 m physically cannot see that ditch from far enough away to stop.

That is one sentence, it follows from four lines of beam geometry, and it is the kind of statement a Department of Defence Production evaluator can act on immediately.

### It turns the sensor recommendation into a costed trade-off

Part 3 already observed that detection is *sampling-limited, not signal-limited*, and told the sponsor what to change. This puts a price on each option, for the 2 m ditch case:

| Configuration | Ditch detection range | Safe speed | Gain |
|---|---|---|---|
| 64-beam @ 1.73 m (baseline) | 21.6 m | 43.2 km/h | — |
| Raise mount to 2.5 m | 26.0 m | 47.7 km/h | **+11%** |
| 128-beam sensor | 30.5 m | 52.1 km/h | **+21%** |
| Both | 36.7 m | 57.5 km/h | **+33%** |

*"Doubling your beam count buys 21% more operational speed; raising the mount 77 cm buys 11% and costs nothing"* is a procurement finding that fell out of a perception analysis. Very few competing teams will produce a sentence in that register at all.

Note the shape of the curve, because it is worth explaining: ditch detection range scales as $\sqrt{wh/\Delta\phi}$, so both interventions give **square-root** returns while safe speed goes roughly as $\sqrt{R}$ again — meaning speed scales as the *fourth root* of beam count. There are steeply diminishing returns to buying beams, and the honest version of this slide says so rather than implying a 256-beam sensor doubles your speed.

### The live envelope

Static tables are for the report. The demo version is **per-bearing and live**.

For each bearing around the vehicle, the current effective detection range depends on more than the datasheet: occlusion (Part 10), dust or rain reducing returns, and $r_{\text{blind}}$ for whichever minimum object the terrain class implies (Part 11). Compute $R(\theta)$ per bearing, apply $v_{\max}$, and you get a **speed envelope polygon** that breathes:

- It **shrinks** when a truck occludes the road ahead
- It **shrinks** in dust or rain as returns drop out
- It **stretches forward** and narrows laterally as speed rises, mirroring the TTC fovea (Part 13)
- It **collapses** toward the vehicle when the sensor is degraded

Render it as a speedometer whose red zone *moves*, with the current ego speed on it. A judge watching the safe-speed limit fall as the vehicle enters a dust cloud understands the entire contribution of this project in about four seconds, without a single equation.

### The metric nobody reports

**"Outdriving fraction"** — the percentage of a recorded sequence during which actual ego speed exceeded the perception-limited speed for the hazards that terrain could contain.

This is computable directly from any dataset with ego poses, including SemanticKITTI, in roughly fifty lines. It answers a question nobody in autonomous driving benchmarking asks: *was this vehicle, in this recording, driving faster than its own sensors justified?* Report it per sequence and per hazard class.

Expect the answer to be uncomfortable — urban driving datasets contain plenty of driving at speeds that would not survive a ditch-detection constraint, because urban roads do not contain ditches. That nuance is the point, and it is why the metric is reported **per hazard class the terrain can contain** rather than as a single number.

### Edge cases and how they're fixed

| Edge case | What would go wrong | The fix |
|---|---|---|
| $a$ and $t_{\text{react}}$ guessed | The whole envelope is arbitrary | Both come from the vehicle config as declared platform parameters, like every other vehicle-specific number (Part 14). The sponsor supplies them; we do not invent them |
| Braking capability assumed constant | 4 m/s² on tarmac is not 4 m/s² on wet grass | $a$ is scaled by the terrain class beneath the vehicle — the map already knows it. Loose or wet surfaces lower $a$, which lowers the safe speed, which is the correct direction |
| Envelope treated as a controller | A perception module silently governing the throttle is a serious integration decision, not ours to make | The envelope is **published as an advisory limit**, alongside the cost map. What the planner does with it is the planner's contract |
| Reaction time ignores pipeline latency | The limit is optimistic by exactly our own processing time | $t_{\text{react}}$ explicitly includes the measured P95 end-to-end latency (Part 20), so improving latency measurably raises the safe speed — a rare case where an engineering optimisation has a directly quotable operational payoff |
| Hazard class assumed | Quoting the cable-limited 22.5 km/h everywhere is uselessly conservative | The binding hazard set comes from the terrain classification and the mission profile; the envelope reports *which* hazard is binding at any moment, so the operator can see why |
| Detection ranges are derived, not measured | The envelope inherits any error in the sensor model | Once Part 18.3 measures detection ranges in CARLA, the envelope is recomputed from **measured** ranges and both versions are reported |

### How to test this module

`pytest tests/test_speed_envelope.py`.

Pure algebra, so assert against the table above: $v_{\max}(21.6) = 12.00$ m/s, $v_{\max}(6.7) = 6.24$ m/s. Assert monotonicity — a shorter detection range must never produce a higher safe speed. Assert that lowering $a$ lowers $v_{\max}$, and that the binding hazard is correctly identified as the minimum over the active hazard set.

Integration: replay a SemanticKITTI sequence, compute the envelope per frame, and plot it against actual ego speed. **That plot is a slide** — and if the recorded vehicle spends time above the line, say so rather than quietly rescaling the axis.

---

## Part 16 — The Conservatism Invariant

**File:** `planning/conservatism.py`, `tests/test_conservatism.py`
**One-line goal:** turn "uncertainty always resolves toward caution" from a principle stated in nine places into a **property that is machine-checked over the whole system**.

### Why a stated principle is not enough

Principle 6 says nothing is invented, and a dozen individual mechanisms in this document each resolve uncertainty toward caution. But scattered good intentions are not a guarantee. Someone adds a fast path, an optimisation drops a flag, a refactor makes `UNKNOWN` default to zero cost — and nothing catches it, because there is no single place where the property lives.

There should be. And it is testable, which turns a design philosophy into a verified property — the difference between a claim and a safety argument.

### The property

Order outputs by caution — how much the system is willing to let the vehicle do:

$$\text{cost}(\texttt{FREE}) \ \le\ \text{cost}(\text{known rough}) \ \le\ \text{cost}(\texttt{UNKNOWN}) \ \le\ \texttt{LETHAL}$$

Then state the invariant:

> **Cost is monotone non-decreasing under information loss.** For any cell, if evidence $E'$ is a degraded version of evidence $E$ — fewer points, lower confidence, occluded rather than observed, inherited rather than measured, beyond $r_{\text{blind}}$ rather than within it — then
> $$E' \sqsubseteq E \quad\Longrightarrow\quad \text{cost}(E') \ \ge\ \text{cost}(E)$$

In plain language: **losing information can never make the system more permissive.** There is no input, no sensor degradation, no missing measurement that causes DRISHTI to report a cell as *more* drivable than it would have with better data.

### Every path where information is missing, and where it resolves

The invariant is only meaningful if it is exhaustive. This table is the enumeration, and each row is a place the design already made the cautious choice — now collected in one place so the set can be audited and tested.

| Information deficit | Where | Output | Direction |
|---|---|---|---|
| Never observed | Part 10 | `UNOBSERVED` → `UNKNOWN_COST` | ↑ cautious |
| Observed but occluded | Part 10 | `OCCLUDED`, never `FREE` | ↑ |
| Too few returns for a minimum object, structured | Part 11 | `SPARSE_STRUCTURED` → `UNKNOWN_COST` | ↑ |
| Zero returns past $r_{\text{blind}}$ | Part 11 | `UNKNOWN`, never `FREE` | ↑ |
| Inherited from a coarser level | Part 12.4 | `PROVISIONAL`, barred from fine decisions | ↑ |
| Geometrically completed across a small gap | Part 12.5 | `INFERRED`, barred from reducing cost | ↑ |
| Single-point cell | Part 9 | Low confidence, never upgrades to `DRIVABLE` | ↑ |
| Low class confidence on terrain | Part 5 | Degrades to `TRAVERSABLE_CAUTION`, never up to `DRIVABLE` | ↑ |
| Grazing incidence, albedo unusable | Part 9.5 | Radiometric channel marked unusable, no water claim either way | ↑ (no false clear) |
| Missing returns, ambiguous cause | Part 10 / 9.5 | `UNKNOWN` unless a positive signature discriminates | ↑ |
| Step height requested beyond L0/L1 | Part 14 | `UNKNOWN`, not "no step" | ↑ |
| Stale cell, confidence decayed | Part 12.3 | Cost rises with age | ↑ |
| Bad ego velocity | Part 13 | Fovea degrades to Profile A — the specified floor | ↑ (never coarser than spec) |
| High residual variance on a track | Part 12.6 | Track confidence lowered | ↑ |
| Sensor degraded (dust, rain, dropout) | Part 15 | Speed envelope contracts | ↑ |

Every arrow points the same way. That is the property, and it is now visible as a single object rather than fourteen scattered decisions.

### The property test

Ten thousand generated cases, one afternoon of work, using Hypothesis or any property-testing library:

```python
@given(cell=cell_states(), degradation=degradations())
def test_cost_is_monotone_under_information_loss(cell, degradation):
    degraded = degradation.apply(cell)      # drop points, lower kappa, mark
                                            # occluded/provisional/inferred,
                                            # push past r_blind, age the cell
    assert cost(degraded, VEHICLE) >= cost(cell, VEHICLE)
```

Plus the strongest single assertion in the repo, worth stating separately because it is the one a judge will remember:

```python
def test_unknown_never_becomes_free():
    # exhaustive over the flag lattice, every vehicle config, every level
    for state in all_reachable_cell_states():
        if state.observability is UNOBSERVED:
            assert cost(state, VEHICLE) > cost(FREE_REFERENCE, VEHICLE)
```

### Why this matters more than it looks

For a hackathon panel it is a strong line: *"no code path in this system can turn `UNKNOWN` into `FREE`, and here is the property test that proves it rather than the paragraph that claims it."*

For a **defence** sponsor it is a different category of thing. A monotonicity property over the full output space is the shape of a **safety case** — the argument form used to qualify systems for fielding. It does not make DRISHTI certifiable, and claiming otherwise would be exactly the overreach this document keeps warning against. But it demonstrates that the team understands what such an argument looks like and has structured the system so one could be built, which is a meaningfully different signal from "our mIoU is 68%".

It is also what makes the Part 22 rejections coherent rather than merely cautious. Generative inpainting and runtime weight adaptation are both rejected *because they break this property* — not by taste, but by a stated criterion that is tested in CI.

### Edge cases and how they're fixed

| Edge case | What would go wrong | The fix |
|---|---|---|
| A performance optimisation bypasses the flag checks | Silent violation, no test catches it because the fast path isn't exercised | The property test generates cell states directly and calls the *public* cost function; any path that does not go through it is not a supported path |
| `UNKNOWN_COST` set above `LETHAL` "to be safe" | Vehicle paralysed; the invariant is satisfied and the system is useless | The order is a **partial order with a finite ceiling**: `UNKNOWN_COST < LETHAL` strictly. Monotonicity is necessary, not sufficient — Part 14 owns usability |
| Confidence decay makes an old `FREE` cell exceed a fresh obstacle's cost | Invariant satisfied, behaviour absurd | Decay saturates at `UNKNOWN_COST`; it never crosses into `LETHAL` |
| The invariant applied to the raw network output | The network is not monotone and cannot be made so | The property is stated over **`cost()`**, the system's decision surface, not over intermediate learned values. Everything upstream may be non-monotone; the boundary is where it is enforced |
| Someone adds a new flag and forgets the ordering | Incomplete lattice, untested combination | `all_reachable_cell_states()` is generated from the flag enum, so a new flag automatically expands the exhaustive test and fails until its ordering is declared |

### How to test this module

`pytest tests/test_conservatism.py` — and it should be the test file that never gets skipped. Run the property test with a high example count in CI, not the default.

The most valuable failure this suite can produce is on a **future** change: someone adds a fast path, an optimisation, or a new inference mechanism, and this test tells them immediately that they have made the system more permissive. That is the entire point of writing it down as a property instead of a paragraph.

---

## Part 17 — Layer 12: Visualisation and Instrumentation

**Files:** `viz/dashboard.py`, `viz/colormap.py`
**One-line goal:** show the map, and — equally important — show the *evidence* that its claims are true, live, while it runs.

Recommended stack: **rerun.io** for the live 3D/2.5D view (built for this, native time-series scrubbing, far less code than a custom Open3D viewer), plus a small panel for the HUD.

### What the dashboard must show

1. **The 2.5D map**, colour-coded by class, height as elevation, cell borders visible so the foveation is *legible* — coarsening outward must be seen, not described.
2. **Split-screen: uniform 5 cm vs. foveated**, same scene, same frame, synchronised, with memory and latency counters under each.
3. **Live HUD:** cells per level, MB in use, end-to-end latency P50/P95, FPS, tracked object count, **stamp-mismatch counter** (Part 8 — must read zero), and the **estimated extrinsics** (mount pitch, roll, $h$ — Part 3.6), which should sit still.
4. **The foveation slider** ($\gamma$), live. Non-negotiable.
5. **Overlay toggles:** `PROVISIONAL`, `INFERRED`, `UNOBSERVED` vs `FREE`, `OCCLUDED` shadows, negative-obstacle candidates, **`SPARSE_STRUCTURED` cells**, clearance heat-map, residual-channel motion field, corrected-albedo channel, occlusion-depth channel.
6. **The planner panel** — cost map plus a live path re-routing around dynamic objects.
7. **The speed envelope** (Part 15) — a speedometer whose red zone moves, with the current binding hazard named beside it, and the per-bearing envelope polygon drawn on the map.

### Colour discipline

Class colour is categorical; height is continuous; confidence is a separate channel (alpha or saturation). Do not encode three variables in one hue ramp — impressive in a screenshot, unreadable on a projector at 4 m. Reserve one high-salience colour (red) exclusively for `LETHAL`/`NEGATIVE_OBSTACLE`. Use a colourblind-safe categorical palette; a meaningful fraction of any panel will not reliably distinguish red from green.

### The seven demo beats, in order

1. **The trench.** Drive at a ditch. Split screen: a standard 2D occupancy grid says `CLEAR`, DRISHTI says `LETHAL`. Five seconds, no narration. **The whole pitch in one shot** — open with it.
2. **The bridge and the branch.** Two overhangs, one you pass under and one you must not. Show max-height calling the bridge a wall and min-height driving into the branch, then the multi-layer cell getting both right.
3. **The Sparsity Trap — make them watch the baseline fail.** This is the most important staging decision in the demo, so build it deliberately: find a sequence with a **pedestrian approaching from far away**, and run split-screen — naive uniform Cartesian voxel grid on the left, DRISHTI on the right.

   The judges watch the pedestrian **vanish into the noise on the left** while remaining flagged on the right. Motion is what makes the failure legible; a static side-by-side does not land the same way, because the eye has nothing to track.

   **Then add the one overlay that turns a UI win into a validation:** mark the predicted $r_{\text{blind}}$ on the timeline. The target should become visible on your side *at the range the beam geometry predicted*, not at some arbitrary point. The beat then says not merely "ours is better" but **"ours is better at exactly the range we derived from the datasheet"** — which is a claim about the physics, not about the rendering.
4. **Uniform vs. foveated**, memory counters running, identical frame rate.
5. **The slider.** Sweep $\gamma$ live; the teardrop stretches, the memory counter moves.
6. **The pedestrian and the planner.** Someone crosses; the residual field lights up; the fine patch follows them; the path re-routes and recovers; **no smear trail behind them.**
7. **The speed envelope collapsing.** Drive into a dust cloud or behind an occluding truck and watch the safe-speed limit fall in real time, with the binding hazard named. Four seconds, and it communicates the entire contribution without an equation: *the map knows how far it can see, and therefore how fast you may go.*

Then the failure slide (Part 24). Teams that show their limits are trusted on everything else, and a DRDO panel is more attuned to that than most.

### How to test this module

Visual, but not untestable. Snapshot-test the colour mapper. **Assert the HUD's memory figure equals `eval/memory.py`'s independently-computed total** — a dashboard reporting a different number from the harness is the single most damaging thing that can happen live, so make that an automated equality test, not a habit of checking.

Rehearse twice end-to-end on the demo machine and record a backup video of a clean run.

---

## Part 18 — Layer 13: The Evaluation Harness

**Files:** `eval/metrics.py`, `eval/pareto.py`, `eval/spec_sheet.py`, `eval/baselines.py`, `eval/latency.py`, `eval/point_distribution.py`
**One-line goal:** produce numbers that survive contact with a sceptical judge.

### The metrics, and why each exists

| # | Metric | Why |
|---|---|---|
| 1 | **mIoU per distance band** (0–12.8 / 12.8–25.6 / 25.6–51.2 / 51.2–102.4 m) | The statement asks for accuracy across distance. Bands are the clipmap levels, so accuracy is reported *per resolution level*, not against arbitrary bins |
| 2 | **Elevation RMSE per band** vs. ground truth | The real quality of a 2.5D map. Needs ground truth — which is why CARLA matters (§18.3) |
| 3 | **Memory**: cells and MB against four baselines | Part 19. Lead with the honest one |
| 4 | **End-to-end latency** P50/P95, per stage | Part 20. Scan-complete to map-ready, not inference time |
| 5 | **Hazard spec sheet**: predicted vs. measured range per hazard | Validates the sensor model itself |
| 6 | **Sparsity Trap recovery rate** vs. the $N_{\exp}$ curve | Part 11. Self-defined, declared as such, with a predicted shape to measure against |
| 7 | **Safety-equivalent resolution at TTC = 2 s**, polar, A vs B | Part 13. The only honest way to claim the adaptive fovea is better |
| 8 | **Memory vs. elevation-RMSE Pareto curve**, sweeping $\gamma$ | The most convincing single artifact (§18.4) |
| 9 | **Moving-object F1** (residual channels) vs. SemanticKITTI-MOS | Part 4.4. Validates the motion mechanism independently of segmentation |
| 10 | **Overhang and negative-obstacle rates — separate column** | Geometry-derived, not network-predicted; blending them into mIoU hides exactly what a judge probes first |
| 11 | **Outdriving fraction** — % of a sequence where ego speed exceeded the perception-limited speed, per hazard class | Part 15. Nobody reports this; it is ~50 lines against any dataset with ego poses |
| 12 | **Conservatism property-test result** — examples run, violations found | Part 16. Reported as a pass/fail with the example count, like a test suite, not as a percentage |
| 13 | **Sensor-portability ablation** — schedule, spec sheet and accuracy regenerated for 32 beams | §18.2. Proof that the config-driven claim is real |
| 14 | **Water/mud flag rate and false-positive rate** (CARLA only, declared preliminary) | Part 9.5. Reported separately and cautiously; no headline claim until measured |

Metrics 6 and 10 are a discipline point. **Report weak numbers in their own column rather than averaging them into a strong one.** A panel that finds a hidden weakness stops believing the rest; a panel *shown* the weakness believes everything else.

### 18.1 Ablations

Each should visibly break one specific demo case — that is what makes the table an argument rather than a formality.

| Ablation | Expected failure | Claim it defends |
|---|---|---|
| Uniform 5 cm everywhere | Memory and latency blow up; far cells 99%+ empty | Foveation is necessary, not decorative |
| Max-height cells | Bridge becomes an impassable wall | Claim 1 |
| Min-height cells | Vehicle drives into the 1.9 m branch | Claim 1 |
| No residual channels | Parked and moving cars indistinguishable | Part 4.4 |
| No `range_gradient` suppression | Static edges flagged as moving under drift | Part 4.4 |
| Ground *stripped* rather than labelled | **No terrain in the elevation map at all** | Part 5.2 |
| No static/dynamic split | Pedestrian smear trails | Part 12 |
| No ray-casting | Ditch invisible; occluded space read as free | Part 10 |
| Flat-ground negative-obstacle model | False ditches all over a hillside | Part 10.3 |
| No sparsity layer | Thin pole at 90 m silently reported free | Claim 3 |
| No `PROVISIONAL` flag | Flickering seam at ~12.8 m, or fabricated detail | Part 12.4 |
| Quadtree instead of clipmap | Boundary artifacts; rebuild cost dominates | Part 8 |
| No Meta-Kernel stem (explicit coord channels only) | Accuracy drop on thin structures and at range — **or no drop, in which case cut the module** | Part 5.5, and this is the ablation that decides whether it ships |
| No occlusion-depth channels | Boundary quality falls toward what KNN cleanup was for | Part 4.3.1, and it justifies the KNN rejection |
| Constant residual threshold (no $\|\omega\|\cdot\|\nabla r\|$ gating) | Static edges flagged as moving during turns | Part 4.4 |
| No albedo correction (raw intensity) | Standing water reads as drivable flat ground | Part 9.5 |
| No multi-echo (first return only) | Grass reads as a solid obstacle; bare earth lost under canopy | Part 6.1 |
| No `INFERRED` completion | More `UNKNOWN` cells, more conservative routes — **this ablation should look *worse*, and that is correct** | Part 12.5 |

### 18.2 The 32-beam ablation — proving portability costs almost nothing

Part 3 claims the whole design regenerates for any sensor from a config file. That is a claim, and claims should be demonstrated rather than asserted.

**Demonstrate it in an afternoon:** drop every second beam ring from SemanticKITTI to synthesise a 32-beam sensor, write `sensor_hdl32.yaml`, and re-run the pipeline unchanged.

Everything should move — and, critically, **should move by predictable amounts**, because $\Delta\phi$ has doubled while $\Delta\theta$ has not:

| Quantity | Scaling | 64-beam | Predicted 32-beam |
|---|---|---|---|
| Kerb height-resolution range | $t/\Delta\phi \to \times\tfrac12$ | 20.2 m | **10.1 m** |
| 2 m ditch range | $\sqrt{wh/\Delta\phi} \to \times\tfrac{1}{\sqrt2}$ | 21.6 m | **15.3 m** |
| Thin fence post $r_{\text{blind}}$ | $\sqrt{tw/\Delta\phi\Delta\theta} \to \times\tfrac{1}{\sqrt2}$ | 66.8 m | **47.3 m** |
| Safe speed, ditch-limited (Part 15) | $v_{\max}(R)$ | 43.2 km/h | **35.7 km/h** |
| Level boundaries | unchanged — depends on $\Delta\theta$ only | 16.6 / 33.2 / 66.3 m | **identical** |

That last row is the detail worth pointing at. The *resolution schedule* depends only on azimuth resolution, so halving the beam count changes every **detection range** while leaving the **cell sizes** untouched. A design that had conflated the two would move both, and this ablation would expose it.

**If measured tracks predicted across that table, the sensor model is validated a second, independent way** — once by absolute agreement against known-geometry hazards (§18.3), and once by *correct scaling* under a controlled change. Agreement in levels and agreement in derivatives are different kinds of evidence, and having both is what makes the model credible rather than fitted.

The speed-envelope row also makes it concrete for a non-specialist: *"with half the beams, this vehicle must drive 8 km/h slower."*

Cost: one YAML file, one decimation function, one re-run. Value: the portability claim stops being a promise.

### 18.3 The CARLA advantage — the thing no other team has

Spend it on the two things public datasets **cannot** provide.

**(a) Ground truth for the otherwise unmeasurable.** On SemanticKITTI you cannot measure "how much did coarsening cost me" — there is no true elevation map. In CARLA the exact mesh is available, which converts the central trade-off from an assertion into a **measured curve**.

CARLA's semantic LiDAR also emits per-point ground-truth labels, so **the grid engine can be built and benchmarked in parallel with the segmentation work rather than waiting on it.** For a solo build that is the difference between finishing and not.

**(b) Manufactured hazards with known geometry.** Public datasets contain no labelled trenches and no bridges at a measured height. In CARLA you place them: a trench of exactly 2.0 m × 0.5 m, a cable at exactly 2.1 m, a 15 cm kerb, a 4 m crater, a 20 cm pole at a ladder of ranges. Then sweep and record detection rate vs. range.

| Hazard | Predicted (Part 3 / Part 11) | Measured (CARLA) |
|---|---|---|
| 15 cm kerb, height resolved | 20.2 m | *measure* |
| 2 m × 0.5 m ditch | 21.6 m | *measure* |
| 4 m crater | 30.5 m | *measure* |
| 40 cm branch @ 2.1 m | 53.9 m | *measure* |
| 5 cm cable @ 2.1 m | 6.7 m | *measure* |
| 20 cm pole, presence ($r_{\text{blind}}$) | 164 m | *measure* |
| Thin fence post, presence | 67 m | *measure* |

**If measured tracks predicted, one table validates the sensor model, the resolution schedule, the hazard claims and the confidence layer simultaneously.** No other team will have a spec sheet at all, let alone a validated one. Start these sweeps early — they take wall-clock time.

**(c) Failure injection.** CARLA's weather gives rain and fog; sensor dropout and pose noise can be injected — and pose noise in particular lets you validate the drift analysis in Part 4.4 with a *known* $\sigma_\theta$.

**State the sim-to-real gap before anyone asks.** CARLA for *characterisation* (Pareto, spec sheet, ablations — anything needing ground truth); SemanticKITTI/nuScenes for *validation* (segmentation accuracy, latency, memory on real scans); RELLIS-3D or GOOSE for off-road realism. Volunteering the limitation is what makes the CARLA numbers credible.

### 18.4 The Pareto curve

Sweep $\gamma$ from 0 (uniform, finest everywhere) to aggressive coarsening. For each, record total memory and elevation RMSE per band. Plot memory against RMSE and **mark the knee**.

The claim becomes: *"here is the full trade-off space, here is where the curve turns, here is why we operate there, and here is what operating anywhere else costs."* A judge cannot argue with a measured Pareto frontier — only ask where you sit on it, and you will have the answer with the reason. This is what most separates a research-grade submission from a demo.

### 18.5 Held-out discipline

Fix an evaluation split **before** any tuning and never touch it. Do not tune the foveation exponent, the residual threshold, the negative-obstacle thresholds, or the segmentation checkpoint on data you later report from. Cross-check file lists programmatically, in CI. The least interesting paragraph here and the one most likely to save the submission.

### How to test this module

`pytest tests/test_metrics.py tests/test_baselines.py tests/test_spec_sheet.py`. Test the metrics against hand-computed cases: a 3×3 confusion matrix whose mIoU you compute on paper, an elevation field whose RMSE is obvious by construction. **A metrics harness with a bug produces confidently wrong numbers, and those go straight onto slides.** Spot-check five frames' verdicts against your own manual read; disagreement means the harness is wrong until proven otherwise.

---

## Part 19 — The Memory Argument, Done Honestly

**File:** `eval/baselines.py`

The metric the statement asks for by name, and the easiest in the project to win dishonestly. Do not.

Reference configuration from Part 8: 4 levels × 512², $c_0 = 5$ cm, ±102.4 m, 12 bytes/cell.

| Representation | Cells / voxels | Memory | Ratio |
|---|---|---|---|
| Dense uniform 3D voxel grid, 5 cm, 10 m vertical | 3.36 × 10⁹ | **3.36 GB** | 267× — **strawman, label it** |
| Dense uniform 2.5D grid, 5 cm | 16,777,216 | **201.3 MB** | **16.0×** — the honest headline |
| Sparse hash-voxel / Octomap / VDB | *measure it* | *measure it* | What real systems actually use |
| **DRISHTI foveated clipmap** | 1,048,576 | **12.58 MB** | — |
| DRISHTI, ring-allocated | 851,968 | **10.22 MB** | 19.7× vs. dense 2.5D |

**Quote 16×, not 267×.** The dense-3D figure compares against something no competent engineer would build; a sharp judge will say so, and once they have caught one inflated number they will not trust the others. A team that says *"the number everyone quotes is 267×, but that baseline is a strawman, so here is ours against a real one: 16×"* is instantly more credible than one quoting 267× with a straight face.

**Measure the sparse baseline.** Beating a dense array proves nothing; beating Octomap — or losing to it on memory while winning decisively on *query latency*, which is the likely and still-excellent outcome — is a real result either way. Do not skip it because the answer might not flatter you; the honest comparison *is* the finding.

### Why the uniform grid is so wasteful — quantified

One scan is ~120,000 points; a uniform 5 cm grid over ±102.4 m has 16,777,216 cells. Even if every point landed in a distinct cell:

$$\text{occupancy} \le \frac{120{,}000}{16{,}777{,}216} = 0.72\%$$

**Over 99.3% of a uniform fine grid is empty on any given scan** — and Part 3 says *why*, precisely: at 100 m the footprint is 30 cm tangentially and 43 m radially, so 5 cm cells out there cannot be filled even in principle. Geometric certainty, not a tuning artifact. Being able to say that with a derivation behind it is the difference between an observation and an argument.

### Reframe: memory is a latency argument

The point most teams will miss. **Memory does not matter because RAM is scarce on a Jetson — it matters because cache misses kill real-time performance.** A map that fits in cache makes every downstream consumer faster, the planner included.

| Plane | Contents | L0 only | All levels |
|---|---|---|---|
| **Hot** (planner, every cycle) | `traversability_cost` (uint8) + `h_max` (int16) = 3 B | **786 KB** | 3.1 MB |
| **Cold** (update only) | histogram, class confidence, timestamps, ceiling layer | ~2.4 MB | ~9.5 MB |

*"The planner-facing hot slice of the safety-critical ring is under a megabyte and stays cache-resident; the full attributed map is under 13 MB and leaves the rest of the board for everything else."* A systems-engineering statement rather than a bar chart, and the version an experienced evaluator finds interesting.

### Edge cases and how they're fixed

| Edge case | What would go wrong | The fix |
|---|---|---|
| Comparing against a dense 3D strawman | Credibility gone once someone notices | Four baselines; strawman labelled; lead with the honest one |
| Counting cells across levels as distinct storage | Correct for memory, wrong for occupancy or coverage | Memory sums (distinct allocations); every other statistic is per-level |
| Ignoring the 18.75% nested-square redundancy | Understating your own memory | Reported explicitly, ring allocation stated as the optimisation |
| Dashboard and harness disagreeing | Catastrophic live | Automated equality test |
| Comparing at different extents | Meaningless ratio | All baselines at identical ±102.4 m and identical bytes/cell |

---

## Part 20 — The Latency Budget

**File:** `eval/latency.py`

### Measure end to end, or do not claim it

The number that matters is **scan-complete to map-ready**, including projection, ray-casting and cost-map update. A team quoting only network inference time is quoting the fast half; the grid update is very often the slow half, and that is the question a knowledgeable judge asks second.

Instrument per stage with CUDA events — not wall-clock around async kernels, where a hidden `.cpu()` makes everything look fast until it doesn't. Report **P50 and P95**, not the mean: a planner cares about the frame that was late, and a mean hides it.

### Budget at 10 Hz (100 ms per scan)

Targets to design against, then measure and replace — not claims:

| Stage | Budget |
|---|---|
| Deskew + ego-motion | 2 ms |
| Ego-compensate K=4 past sweeps + residuals | 4 ms |
| Range-image projection (12 channels) | 3 ms |
| Geometric ground prior | 3 ms |
| Segmentation network (EfficientNet-B0 / ASPP / Attention U-Net, 64×2048) | 14 ms |
| Foveated reduction + label scatter | 3 ms |
| Clipmap scatter, 4 levels (incl. polar→Cartesian) | 5 ms |
| Layer extraction + cell reduce | 6 ms |
| Ray-casting + observability | 5 ms |
| Sparsity / $\kappa$ evaluation | 2 ms |
| Tracker update | 3 ms |
| Traversability + cost map (lazy, planner horizon) | 4 ms |
| **Total** | **≈ 54 ms** |

Roughly 1.9× headroom against a 10 Hz sensor.

### The framing that will separate you

> **Frames per second above the sensor rate is not a virtue.** The sensor produces 10 scans per second. Running at 60 FPS means running five times on data that has not changed. What a planner feels is **latency** — the age of the map when a decision is made.

Most teams will put a large FPS number on a slide because the statement says "high FPS". Reporting latency instead, and explaining *why* it is the right metric for a 10 Hz sensor, demonstrates you understand the system rather than the rubric. Report both; lead with latency.

### Where naive implementations actually lose, ranked

1. **A Python loop over points** in the projection step. Fatal. Vectorised `scatter_reduce` only (Part 9.4).
2. **A host round-trip** — a stray `.cpu()`, `.item()` or `.numpy()` in the per-frame path. Synchronises the GPU and destroys pipelining.
3. **Rebuilding the spatial structure every frame** instead of scrolling (Part 8).
4. **Ray-casting at L0 across the full 102 m.** Traverse coarse, refine near.
5. **Recomputing the cost map over the whole extent** rather than the planner horizon.
6. **Projecting K=4 past sweeps at full resolution** for residuals. Compute residuals on a decimated image; motion is a low-frequency signal.

### Compensating for latency instead of only measuring it

Measuring latency is table stakes. The better move is to **cancel most of it**, and one of the two components cancels for free because of a decision made in Part 7.

The map is ready at $t_{\text{scan}} + L$, but the planner acts at some $t_{\text{act}}$ later still. So the planner is always reasoning about a slightly old world. At 15 m/s a 54 ms pipeline means:

| Ego speed | Distance travelled in 54 ms | Cells at L0 |
|---|---|---|
| 5 m/s | 0.27 m | 5 |
| 10 m/s | 0.54 m | 11 |
| 15 m/s | 0.81 m | **16** |
| 20 m/s | 1.08 m | 22 |

Sixteen 5 cm cells of staleness at moderate speed is not negligible in a map whose entire premise is 5 cm resolution.

**Ego staleness costs nothing to fix, because the static map is world-anchored.** A cell's contents are tied to world coordinates, not to the vehicle, so the map does not "move" as the vehicle does — the *window* moves, and the planner simply queries at the vehicle's current pose. An ego-centric map (polar or otherwise) would need to transform its entire contents forward; ours needs no transform at all. **This is a benefit of the Part 7 decision that is easy to miss, and worth claiming explicitly.**

**Dynamic staleness needs one step.** Tracked entities *were* somewhere at $t_{\text{scan}}$ and are somewhere else now. Advance each track to $t_{\text{act}}$ using its Kalman velocity before publishing the composite, and inflate its extent by the prediction uncertainty over that interval — so the compensation widens the object rather than merely displacing it, which keeps it conservative under Part 16.

Two disciplines: never extrapolate further than the track's confidence supports (cap the horizon), and publish both the measurement timestamp and the validity timestamp so a downstream consumer knows what it has.

The payoff is a sentence worth having: **"we don't just measure latency, we compensate for it — and the map's world-anchored design means the ego half of the problem was already solved."** It also gives Part 15 a real coupling: $t_{\text{react}}$ in the speed envelope includes measured P95 latency, so reducing latency *raises the safe speed*. An engineering optimisation with a directly quotable operational payoff is rare; say so when you have one.

### How to test

`pytest tests/test_latency_instrumentation.py`. For the measurement: 500 consecutive frames, per-stage histogram, and **check the stage totals sum to the measured end-to-end time.** If they do not there is unaccounted time — usually a synchronisation stall — and finding it is worth more than optimising any single stage.

---
## Part 21 — Degraded-Comms Operation: Progressive Level of Detail

**File:** `planning/lod_stream.py`
**One-line goal:** get useful map information to a remote operator or a following vehicle over a link that may be slow, intermittent, or contested — and get it for free from a structure we already built.

### Why this belongs in a perception bible

A fielded UGV is rarely alone. An operator watches from a distance, a following vehicle wants what the lead vehicle learned, and the link between them is nothing like a datacentre network — it is bandwidth-constrained, high-latency, and in a defence context actively degraded. A perception system with no answer for that is a laboratory result.

Most map representations handle this badly: they are a single resolution, so you either send everything or nothing, and when bandwidth drops the map simply stops updating.

### The mipmap already solved this

The clipmap is a pyramid where **each level is independently a complete map of its extent** (Part 8's invariant). That is exactly the property progressive transmission needs, and we get it for nothing:

1. **Send L3 first** — 40 cm cells covering the full ±102.4 m. Coarse, complete, and immediately useful for situational awareness and long-range routing.
2. **Refine with L2, then L1, then L0** as bandwidth allows.
3. **If the link degrades mid-stream, stop.** The receiver holds a coarser but entirely coherent map — not a half-updated one with a torn seam.

Graceful degradation is structural rather than engineered. A receiver never holds an inconsistent map, only a coarser one, and it always knows which level it has.

### Send deltas, not frames

A full hot-plane level is $512^2 \times 3\ \text{B} = 786$ KB, which is far too much to push at 10 Hz over a constrained link. But almost nothing changes between frames:

- At 15 m/s, L3 scrolls **3 cells per frame** — roughly $3 \times 512 = 1536$ newly-revealed cells, about 4.6 KB before compression
- The overwhelming majority of cells in any level are `UNOBSERVED`, so run-length encoding over rows is very effective
- Static cells that have not been re-observed do not need retransmitting at all

So the stream is: an occasional coarse keyframe, then small per-frame deltas, prioritised coarse-level-first. **Measure the actual rates rather than quoting these** — they depend entirely on scene occupancy, and Part 19's discipline about honest baselines applies here too.

### What to send first when bandwidth is scarce

Priority order is a **safety** decision, not a bandwidth one, so declare it:

1. `LETHAL` and `NEGATIVE_OBSTACLE` cells, at whatever level they were found — a ditch matters more than terrain texture
2. The speed envelope (Part 15) — a few hundred bytes, and it is the single most decision-relevant object in the system
3. Tracked dynamic entities — parametric, so tens of bytes each rather than thousands of cells
4. Coarse static structure (L3), then refinements

Note how well the architecture already serves this: dynamic objects are *entities*, not cells (Part 12.1), so transmitting them is nearly free. A design that had baked moving objects into the grid would have to send the cells.

### The demo

A **bandwidth slider**. Drag it down and watch the remote view degrade gracefully — L0 detail disappears first, then L1, then L2, while the ditch, the speed envelope and the tracked pedestrian keep arriving. Drag it back up and detail refills.

It takes minutes to build on top of what exists, and it answers a question a defence evaluator will certainly have and almost no competing team will have considered.

### Edge cases and how they're fixed

| Edge case | What would go wrong | The fix |
|---|---|---|
| Link drops mid-delta | Receiver holds a torn, inconsistent map | Deltas are applied atomically per level; a partial delta is discarded and the level stays at its last coherent state |
| Receiver has never had a keyframe | Deltas are meaningless | Periodic coarse keyframes; a receiver joining mid-stream requests one |
| Coarse-first means the ditch arrives late | The most important cell waits behind terrain texture | Explicit priority order above — hazards jump the queue regardless of level |
| Receiver treats a coarse map as a fine one | Fine-grained decisions made on 40 cm cells | Every transmitted cell carries its level; the receiver's `lookup()` reports the level it answered from, exactly as the local one does (Part 8) |
| Compression cost eats the latency budget | Transmission stalls the pipeline | The stream runs asynchronously off the hot plane; the perception loop never blocks on it |

### How to test

`pytest tests/test_lod_stream.py`. Assert a receiver applying keyframe + N deltas holds a map bit-identical to the sender's at that level. Assert that truncating the stream at any point leaves a coherent map at some level, never a torn one — that property test is the whole value of the design. Then measure real bandwidth on a recorded sequence at each level and report the numbers rather than the estimates above.

---

## Part 22 — Design Decisions We Rejected, and Why

Naming what you rejected *and why* is stronger than silently omitting it. Every row here is a question a judge might ask, pre-answered.

| Rejected | Why it looks right | Why we didn't |
|---|---|---|
| **Quadtree / octree map** | Textbook answer for variable resolution | Pointer chasing, per-frame rebuild, and the 2:1 balancing ripple that is unsolved at 10–20 Hz. We need a *known* resolution function, not arbitrary subdivision (Part 8) |
| **Continuous-resolution polar grid as the map** | Genuinely elegant; matches the sensor; no large discrete tiers | "No boundary anywhere" is not provable once you bin $\rho$; ego-anchored accumulation must re-bin every frame under translation *and* rotation; origin degenerates into slivers under the vehicle; exact demotion is lost; planners want Cartesian (Part 7). **Kept as the front end, rejected as the map** |
| **$c(\rho) = a + k\rho^2$ schedule** | Hits the statement's two checkpoints | Two constants with no physical meaning fitted to two points; reproducing them proves nothing; 1.8× coarser than the sensor delivers at 10 m and 1.66× at 100 m. Linear $r\Delta\theta$ has one parameter and it *is* the physics (Part 3) |
| **5 → 50 cm hierarchy, as literally specified** | It is what the statement says | $50/5 = 10$ is not a power of two, so exact nesting is impossible and boundary straddling returns. 5 → 10 → 20 → 40 cm exceeds the requirement *and* stays exact (Part 8) |
| **Sparse convolution backbone (SPVNAS, Cylinder3D)** | Highest accuracy on paper | Per-frame hash-table construction; kernel-launch overhead on embedded hardware; a well-known source of CUDA toolchain conflicts. Outputs a uniform grid needing re-sampling into ours (Part 4.1) |
| **Point-based backbone (PointNet++, KPConv)** | Operates on the native representation | $O(N\log N)$ neighbour search; degrades badly exactly where our far cells already need help (Part 4.1) |
| **KNN boundary refinement** | Cleans many-to-one projection artifacts, real mIoU gain | Up to ~46% of inference time. Attention gates + Lovász absorb most boundary noise at a fraction of the cost (Part 5) |
| **Frustum-point fusion (FRNet-style)** | Highest accuracy ceiling in the literature | Requires a custom hierarchical back-projection module — too much new architecture for the window, and unnecessary once boundary noise is absorbed cheaply |
| **Temporal-consistency loss** | Worked in the original camera pipeline | Assumes smooth pixel correspondence across frames; a rotating, discretely-sampled LiDAR has no stable pixel-to-pixel correspondence, so it is mathematically unstable transplanted. Its job is done structurally by residual channels (Part 5.4) |
| **Global / per-sector RANSAC ground plane** | Standard, fast, well understood | Assumes flatness over the fit region; on a slope or crest it misclassifies the whole sector — exactly the terrain DRDO cares about. Column-wise incremental walk instead (Part 5.2) |
| **Stripping ground points before the network** | Big apparent compute saving | Terrain never reaches the elevation map, which *is* the deliverable. Label, never strip (Part 5.2) |
| **Absolute range residual** | Simpler | Needs a range-dependent threshold; relative residual is approximately range-invariant (Part 4.4) |
| **Tuned confidence threshold for thin objects** | Works well enough on a demo | Indefensible under questioning. $N_{\exp}$ derives it from the datasheet and the vehicle's stated minimum object (Part 11) |
| **Dice loss** | Common default for imbalance | Insufficient at LiDAR's degree of imbalance; Lovász-Softmax optimises the metric directly (Part 5.4) |
| **float32 elevation** | Obvious default | 1 cm precision over ±300 m fits int16; float32 doubles the largest field for precision no LiDAR delivers (Part 9.3) |
| **Building custom odometry** | Removes a dependency | SemanticKITTI ships ground-truth poses; KISS-ICP covers the rest. Not where the contribution is |
| **Generative Bayesian terrain inpainting** | Fills occlusion holes; emits an uncertainty layer; produces impressive pictures | Breaks Principle 6 and the conservatism invariant (Part 16). See below — this is the most important rejection in the document |
| **Unsupervised test-time optimisation (Floxels-style)** | Genuinely addresses our biggest risk, the urban→off-road domain gap | Weights that change at runtime mean the system that was validated is not the system that is running. See below |
| **General monocular-depth redundancy from a vision foundation model** | Fail-operational backup when LiDAR degrades | Roughly doubles scope on a LiDAR problem statement, and zero-shot monocular depth needs LiDAR to anchor its scale — so it degrades exactly when the LiDAR does. That is fail-*correlated*, not fail-operational. The narrow version survives (Part 24, future work item 9) |
| **Radar Doppler fusion** | Native per-return radial velocity would make most of the K=4 residual machinery unnecessary, instantly | No radar in SemanticKITTI, and it is a different problem statement. But see the extensibility note below — the architecture already accommodates it |
| **Instance consistency loss for motion** | Enforces rigid-body motion, prevents per-point "tearing" | We get the property *structurally* — a Kalman-filtered track has one velocity by construction, so there is nothing to tear. Adopted instead as a diagnostic (Part 12.6) |

### The two rejections worth saying out loud

**Generative inpainting.** The blind spot behind a parked car is precisely where a ground vehicle needs its map to be *honest* rather than *plausible*. A model that hallucinates terrain there — however good, however well-calibrated its uncertainty layer — makes the system's primary output partly generated, and the safety argument in Part 16 collapses: there is no longer a monotonicity property to test, because the fabricated value is not a function of the evidence in any auditable way. We took the conservative half instead (`INFERRED`, Part 12.5): deterministic plane continuation, bounded span, agreement precondition, forbidden from reducing cost. It fills less, and everything it fills can be pointed at and explained.

*"We considered generative terrain completion and rejected it because it violates our conservatism invariant"* is a stronger sentence to a defence panel than any inpainting demo would have been.

**Runtime weight adaptation.** The domain gap is real and it is our biggest risk. But for a fielded defence system, weights that adapt during operation are a **qualification problem**: the system that passed acceptance testing is not the system driving the vehicle, and no amount of measured accuracy resolves that. This is a fielding objection, not a schedule objection, and it is worth giving as such.

The better answer to the same problem is already in the architecture and is worth stating positively: **the layers that matter most have no domain gap at all.** The resolution schedule, the clipmap, multi-layer cells, negative-obstacle detection and the Sparsity Trap are beam geometry, not learned functions — they transfer from Karlsruhe to Rajasthan unchanged because they were never fitted to either. Only five of the ten output classes come from the network (Part 6). *The hazard detection a vehicle's survival depends on is precisely the part that does not need to generalise.*

**And an extensibility note on radar.** Motion evidence in DRISHTI is a **channel, not a module** (Part 4.4). A radar Doppler channel drops into the same tensor slot with no architectural change — no new module, no retraining of anything but the first layer. If a platform has radar, the design absorbs it. That single paragraph converts "we did not do sensor fusion" into "the design already accommodates it", and it costs nothing to be true.

*(Cheap validation option: CARLA has a radar sensor. One experiment comparing radar Doppler against the residual channels on the same scene would validate the residual mechanism against independent motion ground truth. That is a validation, not a feature — roughly half a day if the schedule allows.)*

---

## Part 23 — Full Edge-Case Reference Table

| Failure condition | Why it's hard | How DRISHTI handles it |
|---|---|---|
| Resolution schedule chosen by taste | "Why 5 cm?" has no engineering answer; breaks on a different sensor | Derived from beam geometry (Part 3); regenerates from config; exceeds the specified numbers |
| Schedule fitted to two checkpoints | Reproducing them proves nothing | Linear $r\Delta\theta$ — one parameter, physical meaning (Part 3) |
| Ground beyond ~50 m barely sampled | Terrain claims at 100 m are meaningless | $s_r = r^2\Delta\phi/h$ derived and stated; far field scored on objects, near field on terrain |
| 1.5 m intra-scan smear at speed | Thirty 5 cm cells wide — makes the fine ring worthless | Per-point deskew before anything else (Part 4.2) |
| Empty pixels read as objects at the origin | Structural hallucination near the sensor | `valid_mask` channel; all losses masked (Parts 4.3, 5.4) |
| Objects straddling the 359°/0° seam | Zero-padding destroys their context | Circular horizontal padding throughout (Part 4.5) |
| Class alone cannot distinguish parked from moving | Planner treats both identically | Residual channels give per-point motion evidence (Part 4.4) |
| Odometry drift makes static edges "vibrate" | Classic scene-flow failure mode | Quantified as $\|\nabla r\|\sigma_\theta/\Delta\theta$ — concentrated at depth discontinuities; suppressed via the `range_gradient` channel (Part 4.4) |
| Ground stripped before projection | Terrain never reaches the elevation map | **Label, never strip** (Part 5.2) |
| Plane-fit ground prior on a slope | Whole sector misclassified | Column-wise incremental walk (Part 5.2) |
| Dataset taxonomy adopted silently | Urban vocabulary presented to a defence sponsor | Explicit version-controlled remap (Part 6) |
| `vegetation` covers grass and canopy | Drive into trunks, or refuse to cross grass | Split by height above local ground, resolved in the multi-layer cell (Part 6) |
| Network run at full resolution on distant points | Compute spent on detail the map discards | Foveated reduction — one foveation for compute and storage (Part 4.6) |
| Foveated at inference but not in training | Train/test mismatch; far-field accuracy collapses silently | Applied as a training augmentation too (Part 4.6) |
| Quadtree rebuild every frame | Pointer chasing plus $O(N^2)$ rebuild — the real FPS bottleneck | Clipmap: flat arrays, $O(1)$ indexing, $O(N)$ scroll (Part 8) |
| Alignment error at resolution boundaries | The statement names this as the hard part | Power-of-two nesting from a shared origin makes it structurally impossible (Part 8) |
| Ego-polar map under rotation | Whole accumulated map must be re-binned every frame | World-anchored Cartesian: translation is an index shift, rotation is free (Part 7) |
| Toroidal wrap returning stale data | Data from 100 m behind appears 100 m ahead, plausibly | Clear-on-scroll **plus** per-cell stamp validation; mismatch counter on the HUD (Part 8) |
| Summing statistics across mipmap levels | Every point counted 4× | Levels are alternative views, never summed (Part 8) |
| Single height value with an overhang | Structurally unrepresentable — every choice is wrong | Multi-layer cells: ground / gap / ceiling + clearance (Part 9) |
| Coarse cell storing only max height | All vertical information destroyed at write time | Lossy in XY, statistically complete in Z: min/max/mean/variance/histogram |
| Class IDs averaged | Class 2 and 6 average to unrelated class 4 | Mode/histogram aggregation, never arithmetic |
| `count` overflow in dense near cells | Confidence collapses on the *best*-observed cells | Saturating add |
| Unobserved cells read as flat ground at $z=0$ | Vehicle drives into a hole it never looked at | Four-state observability (Part 10) |
| Ditch invisible to 2D occupancy | The signature UGV hazard reads as clear ground | Expected-return test; $\Delta = rd/h$ (Part 10.2) |
| Negative-obstacle test on a slope | Fires everywhere on exactly the terrain that matters | Locally fitted plane; ring-to-ring inconsistency, not absolute deviation (Part 10.3) |
| Occlusion behind a truck read as a ditch | Both produce missing ground returns | Carve first, then test; detector never runs on `OCCLUDED` cells |
| Thin object at range smoothed away as noise | System reports "no obstacle", a claim it cannot support | $N_{\exp}$, $\kappa$, `SPARSE_STRUCTURED`, and `UNKNOWN` past $r_{\text{blind}}$ (Part 11) |
| "No returns" equated with "no obstacle" | Silence treated as evidence | $r_{\text{blind}} = \sqrt{tw/\Delta\phi\Delta\theta}$ — computed, per cell, from the datasheet (Part 11) |
| Presence confused with height resolution | Reporting a kerb you cannot measure | Two formulas, two columns: $N_{\exp}$ for presence, $t/\Delta\phi$ for height (Part 11) |
| Dynamic objects accumulated into the grid | Pedestrian smear trails — visible across the room | Static/dynamic split, composited at query time (Part 12) |
| Coarse → fine promotion | Cannot create information; copying fabricates detail | `PROVISIONAL` + reduced confidence, cleared on first measurement (Part 12.4) |
| Odometry drift over long accumulation | Static layer smears into thick blurred walls | Bounded accumulation window |
| Adaptive fovea coarsening below spec | Fails the stated requirement — disqualifying | $\min()$ with the sensor schedule as a hard floor (Part 13) |
| Stationary vehicle, TTC → ∞ | Coarsens exactly when a pedestrian walks up | $v_{\min}$ floor plus object closing speed |
| Lateral coarsening hides a pedestrian | Something that can move is not where it will be | Tracked-object refinement override regardless of TTC |
| Unknown space costed as free | Plans through unmapped ground | `UNKNOWN_COST`: high, finite, tunable (Part 14) |
| Unknown space costed as lethal | Vehicle paralysed in any partially-observed scene | Same fix — finite, not binary |
| Step height computed in a 40 cm cell | A 15 cm kerb is invisible | Computed only at L0/L1; `UNKNOWN` beyond, with the range stated |
| Vehicle parameters hardcoded | Map useless on the sponsor's platform | Vehicle config; map stores clearance, not a traversability boolean |
| Memory quoted against a dense 3D strawman | Credibility lost once a judge notices | Four baselines; strawman labelled; honest 16× led with (Part 19) |
| FPS quoted instead of latency | Frame rate above 10 Hz on a 10 Hz sensor is meaningless | End-to-end P50/P95 per stage (Part 20) |
| Network inference timed instead of the pipeline | The grid update is often the slower half | Per-stage instrumentation; stage sum must equal measured end-to-end |
| Weak sub-results averaged into headline numbers | Hides exactly what a judge probes first | Sparsity recovery, overhang and negative-obstacle rates in their own columns (Part 18) |
| Tuning on the evaluation set | Every reported number invalid | Split fixed before tuning; overlap check in CI |
| Mount pitch/roll error of half a degree | 44 cm phantom slope at 50 m — larger than every hazard we detect, and it looks like terrain | Extrinsic self-calibration from aggregate ground fits, slowly updated, displayed on the HUD (Part 3.6) |
| Discarded returns in many-to-one projection | Free boundary information thrown away; KNN cleanup then needed to recover it | `occlusion_count` + `occlusion_spread` channels turn the artifact into a feature (Part 4.3.1) |
| Residual threshold constant across manoeuvres | Static edges flagged as moving in turns, or real motion suppressed throughout every turn | Product gating $\tau = \alpha + \beta\|\nabla r\|\,\|\omega\|$ — conservative only where both conditions hold (Part 4.4) |
| Explicit coordinate channels as the only geometry fix | A hint the network may ignore, not a mechanism | Meta-Kernel stem on *relative* offsets — ablation-gated so it ships only if it earns its latency (Part 5.5) |
| Grass read as a solid obstacle | Vehicle refuses traversable terrain; or drives into a trunk | Multi-echo first/last return separation; last-return surface is the true ground (Part 6.1) |
| Uniform 1 cm height quantum at every level | Millimetre precision stored where vertical sampling is 74 cm | Foveated height quantum, int8 relative to a per-tile base at coarse levels (Part 9.3) |
| Standing water read as flat drivable ground | The most common way a UGV is immobilised, invisible to pure geometry | Incidence-corrected albedo using the map's own normals (Part 9.5) |
| Ditch and puddle both produce missing returns | Two very different hazards collapse to one cautious answer | A ditch **displaces** returns beyond $r_{\text{exp}}$; water **absorbs** them. Presence or absence of a displaced return discriminates (Part 9.5) |
| Occlusion holes left entirely `UNKNOWN` | Most of a cluttered map is uncosted; planner over-conservative | Bounded, agreement-gated plane continuation flagged `INFERRED`, forbidden from reducing cost (Part 12.5) |
| Generative inpainting used to fill those holes | Fabricated geometry in the primary output; safety property destroyed | Rejected (Part 22); the deterministic bounded version is what ships |
| Rigid object's points assigned inconsistent motion | Per-point "tearing" in the motion estimate | Structural — one Kalman velocity per track. Residual variance across a track becomes a quality diagnostic instead (Part 12.6) |
| Map arrives stale by one pipeline latency | 16 cells of error at L0 at 15 m/s | World-anchoring makes ego staleness free; dynamic entities advanced to action time with inflated extent (Part 20) |
| Vehicle driven faster than its sensors justify | No map quality fixes a kinematics problem | Perception-limited speed envelope, published as an advisory limit (Part 15) |
| A future optimisation quietly makes the system permissive | Nothing catches it; the safety story silently becomes false | Conservatism invariant as a property test over the public `cost()` surface, in CI (Part 16) |
| Constrained or contested comms link | Map updates stop entirely, or arrive torn | Coarse-first progressive LOD from the mipmap, atomic per-level deltas, hazards prioritised (Part 21) |
| Portability claimed but never demonstrated | "It regenerates for any sensor" is an untested promise | 32-beam ablation; detection ranges must scale by the predicted factors while level boundaries stay fixed (Part 18.2) |

---

## Part 24 — Honest Limitations and Future Work

Not weaknesses to hide — the slide that makes the rest believable. Every item should be answerable without hesitation.

### Hard physical limits (no software fixes these)

1. **Thin overhangs are nearly invisible.** A 5 cm cable is detectable to ~6.7 m ($t/\Delta\phi$). Power lines, guy wires and thin branches are a blind spot for this sensor class. The mitigation is a sensor with finer vertical resolution or a camera, not better code.
2. **Negative obstacles are sampling-limited to ~20–30 m.** 21.6 m for a 2 m ditch, 30.5 m for a 4 m crater. Extending it needs a higher mount, more beams, or a downward-canted auxiliary sensor — a *procurement* answer, and worth giving.
3. **Terrain beyond ~50 m is barely sampled.** Ring spacing on the ground reaches 43 m at 100 m. The far field is object detection; treating it as terrain mapping is a claim the physics does not support.
4. **Upward field of view bounds overhang range.** An HDL-64E sees 2° above horizontal.
5. **Thin objects have a computable blind range.** A 10 cm × 1 m fence post drops below one expected return at 67 m. We report `UNKNOWN` there rather than `FREE` — which is honest, and still means we cannot see it.
6. **2.5D cannot represent genuinely multi-storey structure.** A spiral car-park ramp, a two-level flyover with traffic on both decks — one ground and one ceiling layer per cell is an approximation, and some scenes make it the wrong one.

### Engineering limitations at this scope

7. **Square cells for anisotropic sampling** (30 cm tangential vs. 43 m radial at 100 m). Chosen because they keep power-of-two indexing exact, which buys more than anisotropy would.
8. **Mixed pixels.** A beam clipping a foreground edge and the background produces a phantom return between them. Attention gates and Lovász absorb most of it; KNN cleanup is deliberately rejected (Part 22). Accepted and declared.
9. **Residual channels degrade under odometry drift**, concentrated at depth discontinuities. Quantified and partly suppressed (Part 4.4), not eliminated. Very low-velocity motion remains hard to separate from drift.
10. **Track ID swaps** between crossing pedestrians degrade fovea allocation. Safety unaffected (refinement covers both); velocity estimates jump.
11. **Bounded accumulation window.** This is a *local* perception map, not SLAM, and should never be described as one.
12. **Range-image weaknesses**: thin structures and boundary bleed at depth discontinuities. Mitigated by k-NN re-projection; not eliminated.
13. **Rain, dust and exhaust** produce spurious returns. Multi-echo filtering helps and does not solve it.
14. **CARLA-derived numbers carry a sim-to-real gap.** Characterisation in simulation, validation on real scans (Part 18.3).
15. **The albedo channel is the least mature element here** (Part 9.5). The Lambertian model is an approximation, water response differs substantially between 905 nm and 1550 nm sensors, and wet asphalt versus standing water is not trivially separable. It degrades cells toward caution; it is not a water detector, and we claim no detection rate for it until measured.
16. **Multi-echo vegetation depth is sensor-dependent** (Part 6.1). It needs dual returns, which SemanticKITTI does not ship. Where unavailable the height heuristic is the fallback, at lower confidence.
17. **`INFERRED` completion cannot recover a car-shaped occlusion shadow** (Part 12.5) — the bounded plane continuation refuses to act across a gap that large, so the shadow stays `UNKNOWN` and gets costed as such. That is the correct answer and it is also the less impressive one.
18. **The speed envelope is advisory, not a controller** (Part 15). Whether a planner respects it is the planner's contract, and $a$ and $t_{\text{react}}$ are platform parameters we take from config rather than measure.

### The domain-gap answer, stated positively

This deserves to be said as a strength rather than buried as a caveat, because it is the honest structure of the system:

**Only five of ten output classes are learned.** The resolution schedule, the clipmap, multi-layer cells and clearance, negative-obstacle detection, the Sparsity Trap, the speed envelope — all beam geometry. They transfer from an urban German dataset to Indian off-road terrain **unchanged**, because they were never fitted to either. The learned component sits on top as a refinement, and where it is uncertain the geometry beneath it still holds and everything degrades toward caution (Part 16).

So when the domain gap is raised — and it will be, because we train on SemanticKITTI for a sponsor who operates off-road — the answer is not a promise about generalisation. It is: *the hazard detection a vehicle's survival depends on is the part that does not need to generalise.*

### Future work (ranked by leverage per hour)

1. **Boundary-artifact stress test** — construct a scene with an object straddling a level transition and show it is handled cleanly. Turns an abstract design claim into a concrete visual proof. Cheap, high value.
2. **Cylindrical vs. Cartesian ablation** — measure the front-end/map split against a pure-polar variant. Pre-empts "why not just use Cylinder3D's grid?" with data instead of argument.
3. **Occupancy-flow-style per-cell motion vectors** instead of a binary dynamic flag — actual velocity per cell, same forward pass, no extra tracker. The natural v2 of Part 4.4.
4. **Jetson deployment benchmarking** — quantisation + TensorRT on real embedded hardware rather than desktop GPU. A legitimate concern for a fielded system, and almost no competing team will measure it.
5. **Evidential / MC-dropout uncertainty** alongside the geometric $\kappa$ — a second, learned confidence channel to cross-check the derived one.
6. **Anisotropic or polar-aligned cells** matching true sampling geometry. Better fidelity, harder indexing, gives up exact nesting — measure before committing.
7. **$n$-layer cells** for genuine multi-storey structure.
8. **Sparsity Trap recovery as a standalone proposed benchmark** — written up cleanly enough to stand alone, since no standard metric exists.
9. **Cross-modal fusion with a camera, targeted at one derived blind spot.** Not general redundancy — the spec sheet says a 5 cm cable is invisible past **6.7 m**, and a camera sees a cable against sky trivially. This is the strongest form the argument can take: not *"add a camera for robustness"* but *"add a camera for the one failure mode we computed, quantified, and cannot fix in software."* The spec sheet earning its keep a third time.
10. **Multi-vehicle map sharing** — the clipmap's world-anchored, level-aligned structure makes merging two vehicles' maps unusually clean, since cells at the same level already tile identically.
11. **Learned foveation** — predicting where resolution is worth spending from scene content. Interesting, needs a reward signal, and the Part 13 floor rule would have to survive it.
12. **Hardware-in-the-loop on the sponsor's platform** — different sensor, mount height and geometry, all of which the config-driven design exists to absorb.

---

## Part 25 — Risk Register (build-time, not run-time)

Part 24 is what the finished system cannot do. This is what could go wrong while building it.

| Risk | Severity | Mitigation |
|---|---|---|
| Sensor constants wrong → every derived number wrong | **Critical** | `eval/point_distribution.py` validation on real data **before anything else is built** (Part 3) |
| Silent toroidal wrap bug | **Critical** | Clear-on-scroll + stamp validation + adversarial-scroll test, all three (Part 8) |
| Ground stripped instead of labelled → no terrain in the map | **Critical** | Regression test asserting terrain cells exist with valid elevation, in CI (Part 5.2) |
| Sparse-conv / CUDA toolchain rabbit hole | High | Range-image backbone chosen partly to avoid it. If sparse conv is ever added, timebox it hard |
| Residual channels need accurate odometry | High | Use dataset ground-truth poses; KISS-ICP as fallback. Never build custom odometry under time pressure |
| Confidence thresholds look arbitrary under questioning | High | $\kappa$ is *derived*, not tuned (Part 11); residual threshold calibrated against MOS ground truth, not by eye |
| $\gamma$ / foveation exponent hand-tuned live | Medium | Sweep it offline into the Pareto curve early (Part 18.4); pick the knee, do not tune under demo pressure |
| Ground-prior threshold absorbs a real low obstacle | Medium | Conservative threshold; false negatives strictly worse than false positives here |
| Metrics harness bug → confidently wrong slide numbers | Medium | Hand-computed test cases; five-frame manual spot check (Part 18) |
| Dashboard and harness report different memory | Medium | Automated equality test (Part 17) |
| Demo machine differs from dev machine | Medium | Rehearse twice on the demo machine; record a backup video (Part 17) |
| Tuning leaks into the evaluation split | Medium | Split fixed before tuning; overlap check in CI (Part 18.5) |
| Scope creep into segmentation accuracy chasing | Medium | Principle 3. Segmentation is explicitly not the contribution (Part 5.1) |
| Mixed-pixel artifacts at boundaries | Low | Accepted trade-off, declared; attention + Lovász absorb most without KNN's runtime cost |
| **Extrinsic mount error unnoticed** | **Critical** | A 0.5° tilt is a 44 cm phantom slope at 50 m — larger than every hazard we claim to detect. Self-calibration (Part 3.6) plus the estimated extrinsics on the HUD, where drift is visible |
| Meta-Kernel becomes a rabbit hole | High | Timebox it. It ships **only** if the Part 18.1 ablation says it earns its latency; the explicit-coordinate baseline is the fallback and it already works |
| Albedo channel overclaimed | High | It degrades toward caution and feeds confidence; no water-detection rate is quoted until CARLA measures one. If it does not validate, it stays in as a confidence input and out of the pitch |
| Conservatism property test written but not run in CI | Medium | It is the one suite that must never be skipped, at high example count. A property test that does not run is a comment |
| Multi-echo assumed available | Medium | Sensor-dependent; SemanticKITTI has no dual returns. Height heuristic is the declared fallback |
| Speed envelope mistaken for a controller | Medium | Published as advisory alongside the cost map; the integration contract is stated explicitly (Part 15) |

---

## Part 26 — Open Questions To Grill

Questions worth being asked hard about, with where the answer currently stands. **Answered** means the design has a mechanism; **open** means it does not yet, and saying so is the point.

| Question | Status |
|---|---|
| How does an object straddling a resolution transition behave — smooth or discontinuous? | **Answered.** Power-of-two nesting makes the transition exact; the nesting-exactness test is the proof. Note this question is *only* open in a continuous-schedule design (Part 7) |
| Does the ground prior generalise across terrain slopes, or is it tuned to flat urban scenes? | **Answered in design, open in measurement.** The column-wise walk compares only to its predecessor rather than fitting a plane. Must still be measured on RELLIS/GOOSE, not just SemanticKITTI |
| Is the confidence layer genuinely calibrated, or plausible-looking? | **Answered.** It is neither — it is *derived*. $N_{\exp}$ has no free parameters (Part 11) |
| What happens to residual-based motion detection under odometry drift in a sharp turn? | **Partly answered.** Quantified as $\|\nabla r\|\sigma_\theta/\Delta\theta$ and suppressed via `range_gradient`. Very low-velocity motion remains hard to separate — open |
| Does ground filtering belong before or inside segmentation? | **Answered.** Inside, as an input channel. Before-and-stripping loses the terrain that *is* the deliverable (Part 5.2) |
| Does skipping KNN cleanup cost more mIoU than the budget requires? | **Open — measure it.** On a 10 Hz sensor latency headroom is the currency. If P95 sits near 54 ms there is room to buy accuracy back. Decide from the measurement, not the argument |
| Is $t_{\min}$ for $r_{\text{blind}}$ the right choice, and who owns it? | **Open by design.** It is a sponsor-supplied safety requirement, not our parameter. Report the resulting $r_{\text{blind}}$ so the choice is visible and arguable |
| Does the fovea's TTC formulation hold under heavy braking or reverse? | **Open.** Braking changes closing speed faster than the low-pass filter tracks; the floor rule bounds the damage but the behaviour is unmeasured |
| How does the map behave when odometry fails entirely mid-sequence? | **Open.** Accumulation and residuals both degrade. The intended behaviour is to fall back to single-frame mapping with everything marked low-confidence — designed, not yet implemented |
| Are the level count and $N$ optimal, or just convenient powers of two? | **Open.** 4 × 512² satisfies the spec with margin; the Pareto sweep should include $N$ and $L$, not only $\gamma$ |
| Does the Meta-Kernel actually earn its latency? | **Open by construction.** It ships only if the ablation says so. Deciding in advance would be the mistake |
| Is $a = 4$ m/s² defensible across the terrain the sponsor cares about? | **Open.** It is scaled by terrain class, but the scaling factors are assumed rather than measured. This is the weakest input to the speed envelope and should be flagged when presenting it |
| Can the albedo channel actually separate wet asphalt from standing water? | **Open, and possibly no.** Measured in CARLA before any claim. If it cannot, it remains a caution-degrading confidence input rather than a discriminator |
| Does the conservatism invariant hold once someone adds a performance fast path? | **Answered by construction, open in practice.** The property test generates states and calls the public `cost()`; a fast path that bypasses it is unsupported. Whether anyone adds one anyway is a process question |
| Is `INFERRED` completion's span limit principled or arbitrary? | **Currently arbitrary.** It should derive from the local plane fit's residual — the span over which the fit stays trustworthy — rather than from a configured cell count. Not yet done |
| How much of the far-field map is `UNKNOWN` in a typical scene, and is that usable? | **Open — measure it.** If Part 11 marks most of the far field unknown, the planner's `UNKNOWN_COST` tuning carries more weight than any perception improvement |

---

## Part 27 — What To Say To Judges

### The one-sentence pitch

> *Foveation is not a compression trick — it is a statement about where information is worth having, and time-to-contact, not distance, is what decides that.*

### The thirty-second version

> A LiDAR gives you a million points a second and you can't process them. Flatten to 2D and you lose the kerb, the ditch and the low branch — which for a ground vehicle is the entire problem. We build a variable-resolution 2.5D map where the resolution schedule is *derived* from the sensor's beam geometry rather than picked, where cross-resolution alignment error is structurally impossible rather than handled, where the map knows what it *failed* to see — that's how you detect a trench, because a trench is returns that never came back — and where, past a range we compute from the datasheet, it says "unknown" instead of "clear". And because we know how far it can see, we can tell you how fast this vehicle may safely go: about 43 km/h in ditch country with this sensor. The sensor, not the drivetrain, is the speed limit.

### The ten questions you will be asked

**Q: "Why 5 cm and 40 cm, and why deviate from the 50 cm in the statement?"**
The schedule is derived from angular resolution: a cell should match the beam footprint, $c(r) = r\Delta\theta$, so a 5 cm cell at 80 m is one that can never be filled. That gives 5 cm to 16.6 m and 40 cm to 132 m — exceeding the requirement in both directions, and regenerating for any sensor from a config file. 40 rather than 50 because $50/5 = 10$ is not a power of two, so a 5→50 hierarchy cannot nest exactly, and exact nesting is what makes alignment error impossible. Incidentally, your own checkpoints — 5 cm at 10 m, 50 cm at 100 m — are linear in range, which is exactly what constant angular resolution produces.

**Q: "How do you handle alignment error between resolutions?"**
We don't handle it, we remove the conditions for it. All levels index from a shared world origin with power-of-two cell sizes, so a coarse cell is *exactly* the union of four fine cells at every position and every ego pose. There is no straddling case to handle. [Show the nesting-exactness unit test.]

**Q: "Isn't the segmentation network off-the-shelf?"**
Yes, deliberately — it's our existing backbone, adapted. Segmentation is solved well enough by mature architectures, and a team that spends its time there has skipped the contribution. What's ours is that the same foveation governs the network's input as governs storage, that motion is an input channel rather than a separate module, and that the hazard classes you care about — negative obstacles, overhangs — come from geometry the network cannot fabricate.

**Q: "How do you tell a parked car from a moving one?"**
Not from the class label — that only makes it a candidate. We ego-motion-compensate the previous four sweeps into the current frame and feed the per-pixel range residual as an input channel. Static structure lands at near-zero residual; moving objects leave a shadow trail. The tracker then confirms at the entity level. Two independent mechanisms, and when they disagree that's usually odometry drift, which we can identify because drift concentrates at depth discontinuities.

**Q: "Your memory number seems too good."**
The 267× everyone quotes is against a dense 3D voxel grid nobody would build. Against a dense uniform 2.5D grid at the same extent and bytes per cell it's 16×, and here's the comparison against a sparse hash-voxel baseline, which is what real systems use. The number matters less than what it buys: the planner-facing slice of the fine ring is under a megabyte and stays cache-resident.

**Q: "What about a thin pole at 90 metres?"**
That's the one we're proudest of. Beam geometry says a 20 cm pole returns about 3 points at 100 m, falling as $1/r^2$. If we see two, that's consistent with a pole and inconsistent with noise, so we flag it rather than smooth it away. And past the range where a minimum-size object drops below one expected return — 67 m for a thin fence post — we report `UNKNOWN`, not `FREE`. Most systems report "clear" there, which is a much stronger claim than the sensor can support.

**Q: "So how fast can it actually drive?"**
About 43 km/h in terrain where a 2 m ditch is possible — and that number is not a guess, it's stopping distance set equal to detection range. With this sensor at 1.73 m we see that ditch at 21.6 m, and at 4 m/s² braking plus 0.3 s reaction that's 12 m/s. It drops to 22 km/h anywhere overhead cables exist, because a 5 cm cable is only visible at 6.7 m. If you want more speed, doubling the beam count buys 21% and raising the mount 77 cm buys 11% — and both give square-root returns, so a 256-beam sensor won't double it. The interesting part is that this is a *perception* limit, not a vehicle limit: no amount of better software moves it, only sensor geometry does.

**Q: "How do I know your map won't tell my vehicle something is safe when it isn't?"**
We made that a property rather than a promise. Every path where information is missing — unobserved, occluded, too sparse, inherited from a coarser level, geometrically inferred, stale, beyond the blind range — resolves in the same direction, and we test it: ten thousand generated cases asserting that degrading a cell's evidence can never lower its reported cost. Plus an exhaustive check that no reachable state turns `UNKNOWN` into `FREE`. It doesn't make the system certifiable, and I wouldn't claim that, but it's the shape a safety argument takes and the system is built so one could be made. It's also why we rejected generative terrain inpainting — it would have filled our occlusion holes with plausible geometry and destroyed the property.

**Q: "You trained on German urban data. Our vehicles operate off-road."**
Real risk, and here's the honest structure: only five of our ten output classes are learned. The resolution schedule, the clipmap, the multi-layer cells, negative-obstacle detection, the sparsity model, the speed envelope — that's all beam geometry, and it transfers unchanged because it was never fitted to any dataset. The learned part sits on top as a refinement, and where it's uncertain the geometry underneath still holds and everything degrades toward caution. The hazard detection you'd actually care about is the part that doesn't need to generalise. We also evaluate on RELLIS-3D and GOOSE rather than only SemanticKITTI.

**Q: "What doesn't this do?"**
[Part 24, without hesitation.] A 5 cm cable is invisible beyond about 7 m — beam geometry, not a bug. Negative obstacles are sampling-limited to roughly 20 m for a 2 m ditch, and extending that is a mounting-height and beam-count decision, not a software one. 2.5D can't represent genuinely multi-storey structure. Residual-based motion detection degrades under odometry drift, concentrated at depth edges — we quantify and suppress it, we don't eliminate it. And our CARLA characterisation carries a sim-to-real gap, which is why accuracy, latency and memory are all reported on real scans.

### Demo order

Trench → bridge and branch → the sparsity split-screen with an approaching pedestrian and the predicted $r_{\text{blind}}$ marked → uniform vs. foveated → the slider → the pedestrian and the re-planned path → the speed envelope collapsing in dust → the limitations slide. Rehearse twice on the demo machine. Record a backup. If any of it can run on a Jetson in the room rather than a laptop, do that — hardware present beats hardware described.

---

## Part 28 — Repo Layout and How To Test Each Module

```
drishti/
  sensor/
    sensor_model.py        # beam geometry: the five formulas + N_exp
    schedule.py            # resolution schedule, level boundaries
    ego_motion.py          # pose interpolation, deskew, K-sweep compensation
    extrinsics.py          # self-calibration of mount pitch/roll/height
  perception/
    range_image.py         # spherical projection, 14 channels, circular padding,
                           #   occlusion count + spread
    residuals.py           # K-sweep motion residuals, |omega| x |grad r| gating
    ground_prior.py        # column-wise incremental walk — LABELS, never strips
    multi_echo.py          # first/last return → vegetation depth, bare earth
    meta_kernel.py         # relative-coordinate dynamic stem (ablation-gated)
    segnet.py              # EfficientNet-B0 + ASPP + Attention U-Net + SE
    losses.py              # Lovász-Softmax + confidence-weighted CE + deep sup.
    taxonomy.py            # dataset classes → DRISHTI classes
  grid/
    addressing.py          # world ↔ index, toroidal wrap, stamp validation
    clipmap.py             # nested level stack, scroll, lookup
    cell.py                # SoA planes, hot/cold split, foveated height quantum
    scatter.py             # GPU projection (incl. polar→Cartesian)
    layers.py              # ground / gap / ceiling extraction
    albedo.py              # incidence-corrected reflectance, water/mud signature
    foveated_voxelize.py   # range-dependent input reduction
  observability/
    raycast.py             # DDA traversal, four-state model
    negative_obstacle.py   # expected-return test, local plane fit
    sparsity.py            # N_exp, kappa, r_blind, structure test
  temporal/
    static_layer.py        # accumulation, decay, Chan merge
    tracker.py             # cluster → associate → Kalman, per-track residual var
    promote_demote.py      # level migration, PROVISIONAL flag
    completion.py          # bounded plane continuation → INFERRED flag
  attention/
    ttc.py                 # time-to-contact
    fovea_controller.py    # min() composition, floor enforcement
  planning/
    traversability.py      # slope, roughness, step, clearance
    costmap.py             # vehicle model → cost
    conservatism.py        # the caution order + the monotonicity property
    speed_envelope.py      # detection range + stopping distance → safe speed
    lod_stream.py          # coarse-first progressive transmission
    ros_bridge.py          # grid_map / OccupancyGrid output
  viz/
    dashboard.py  colormap.py
  eval/
    point_distribution.py  # measured spacing vs. the Part 3 curves — RUN FIRST
    metrics.py  baselines.py  spec_sheet.py  pareto.py  latency.py
  configs/
    sensor_hdl64.yaml  sensor_hdl32.yaml  sensor_os1_128.yaml  vehicle_ugv.yaml
  tests/
```

### The testing pattern that makes this tractable

Every function depending on ego state, elapsed time, or sensor configuration takes it as an **explicit argument** rather than reading a global. That single discipline lets you test "does the map behave correctly after 200 m of travel" in milliseconds by advancing a synthetic pose, without running a simulator inside a unit test.

### The thirteen tests that matter most

If time runs short, these protect the load-bearing claims:

1. `test_clipmap.py::test_nesting_exactness` — machine-checked proof that alignment error is impossible.
2. `test_clipmap.py::test_adversarial_scroll` — catches the silent wrap-around bug.
3. `test_layers.py::test_bridge_and_branch` — machine-checked form of Claim 1.
4. `test_sparsity.py::test_unknown_past_r_blind` — machine-checked form of Claim 3.
5. `test_fovea_controller.py::test_floor_never_violated` — property test proving the spec cannot be broken.
6. `test_negative_obstacle.py::test_no_false_positives_on_slope` — guards the fix that stops the demo embarrassing you on a hill.
7. `test_promote_demote.py::test_demotion_is_exact` — proves the power-of-two payoff is real.
8. `test_static_layer.py::test_no_smear_trail` — automated form of the visual pedestrian check.
9. `test_ground_prior.py::test_terrain_reaches_the_map` — regression test for the label-don't-strip bug.
10. `test_residuals.py::test_drift_matches_predicted_magnitude` — machine-checked form of the Part 4.4 drift analysis.
11. `test_conservatism.py::test_cost_is_monotone_under_information_loss` — the property over the whole decision surface. **Never skip this one.**
12. `test_conservatism.py::test_unknown_never_becomes_free` — exhaustive over the flag lattice; the single assertion a judge will remember.
13. `test_speed_envelope.py::test_shorter_range_never_raises_speed` — monotonicity of the operational limit.

```bash
pytest drishti/tests/ -v
pytest drishti/tests/test_conservatism.py --hypothesis-profile=ci   # high example count
```

### Real-data validation pass, in order

1. **Sensor model first.** Measured nearest-neighbour spacing must lie on the $r\Delta\theta$ and $r^2\Delta\phi/h$ curves. **If it doesn't, everything downstream is wrong** — fix before building anything else.
2. **Deskew.** Bird's-eye view of a long straight wall while moving fast: straight, not bent.
3. **Circular padding.** Object straddling the seam; feature response continuous across the wrap.
4. **Residuals.** Known static scene → near-zero everywhere. Known moving object → residual only on it. Injected pose error → false residual concentrated at depth edges, magnitude matching prediction.
5. **Terrain present.** Output map contains terrain cells with valid elevation (the label-don't-strip check).
6. **Clipmap integrity.** Long drive, stamp-mismatch counter reads zero throughout.
7. **Layer extraction.** Under a bridge, then under a low branch. Clearance correct in both.
8. **Negative obstacle.** Approach a ditch; then drive a hillside and confirm **zero** false detections.
9. **Sparsity.** Thin pole at a ladder of ranges; observed count tracks $N_{\exp}$; `UNKNOWN` past $r_{\text{blind}}$.
10. **Static/dynamic.** Pedestrian crosses; every cell on their path afterwards is `FREE` or `UNOBSERVED`.
11. **Promotion.** Approach a known kerb with `PROVISIONAL` rendered distinctly; the flag clears as you close.
12. **Fovea.** Sweep $\gamma$ at speed and at standstill; the floor holds in both.
13. **Extrinsics.** Estimated mount pitch, roll and $h$ on the HUD over a long drive: they should converge and then sit still. Drift means a loosening bracket or a bug in the estimator.
14. **Multi-echo.** Drive through tall grass; the last-return surface should be the ground the wheels meet, not the canopy.
15. **Albedo.** A wet patch and a dry patch of the same surface; corrected $\hat\rho$ should separate them where raw intensity does not. If it does not, say so and keep it as a confidence input only.
16. **Speed envelope.** Drive into occlusion or simulated dust; the limit must fall, and the binding hazard must be correctly named.
17. **Degraded comms.** Truncate the LOD stream at random points; the receiver must always hold a coherent map at *some* level, never a torn one.
18. **End to end.** 500 frames; per-stage latency histogram; stage sum equals measured total.

---

## Part 29 — Glossary

- **2.5D map** — a 2D grid of cells where each stores height information. Cheaper than 3D, richer than occupancy, and (Part 9) inadequate in its single-value form.
- **Foveation** — allocating detail non-uniformly, finest where it matters most, by analogy with the eye's fovea.
- **Range image** — a LiDAR scan laid out as a 2D image indexed by (beam ring, azimuth): the sensor's native sampling layout.
- **Circular padding** — padding a convolution's horizontal edges by wrapping, because 359° is physically adjacent to 0°.
- **Motion residual** — the per-pixel range difference between an ego-motion-compensated past sweep and the current one. Near zero for static structure, large for moving objects.
- **Clipmap** — a stack of nested uniform grids at power-of-two resolutions centred on the viewer. From real-time graphics, where detail-near/coarse-far with a moving viewpoint was solved decades ago.
- **Toroidal addressing** — indexing a fixed array with wrap-around (`i & (N-1)`), so a moving window costs an index shift rather than a copy.
- **Mipmap** — a pyramid of progressively coarser versions of the same data. Here: levels are alternative views, never summed.
- **Deskew** — correcting for vehicle motion *during* one LiDAR rotation, so points measured 100 ms apart are placed correctly relative to each other.
- **ASPP** — Atrous Spatial Pyramid Pooling; parallel dilated convolutions capturing multiple scales at once. Matters here because an object's pixel footprint varies enormously with range in a spherical projection.
- **Lovász-Softmax** — a convex surrogate for the Jaccard/IoU metric; optimises directly for mIoU and handles extreme class imbalance better than Dice.
- **Negative obstacle** — a hazard *below* the ground plane: ditch, trench, culvert, crater. Invisible to 2D occupancy grids and the signature hazard for ground vehicles.
- **Range shadow ($\Delta$)** — the extra distance a beam travels when the ground drops away, $\Delta = rd/h$. The signal that reveals a negative obstacle.
- **Sparsity Trap** — the failure where a thin object at range returns too few points to distinguish from noise and is silently reported as free space.
- **$N_{\exp}$ / $\kappa$ / $r_{\text{blind}}$** — expected return count for an object of given size at a given range; the ratio of observed to expected; and the range beyond which that count drops below one, past which absence of returns proves nothing.
- **Time-to-contact (TTC)** — how long until the vehicle reaches a point at current closing speed. The variable resolution should track, in place of distance.
- **Multi-level surface map (MLS)** — an elevation map storing several vertical intervals per cell rather than one height; what makes overhangs representable.
- **Clearance** — vertical free space between the ground surface and the lowest thing above it.
- **Traversability** — whether *this specific vehicle* can cross a cell, from slope, roughness, step height and clearance against a declared vehicle model.
- **`PROVISIONAL`** — a cell inherited from a coarser level rather than measured at this resolution. Usable for coarse routing, never for a fine safety decision, cleared on first real measurement.
- **Welford / Chan formulas** — numerically stable incremental variance, and exact merging of two independently-accumulated groups without revisiting the points.
- **DDA traversal** — walking a ray through a grid cell by cell; used for free-space carving and occlusion reasoning.
- **Structure-of-arrays (SoA)** — each field in its own contiguous plane rather than packed records. Vectorises better and enables the hot/cold split.
- **P50 / P95 latency** — median and 95th percentile. A planner cares about the frame that was late, which a mean hides.
- **mIoU** — mean intersection-over-union. Reported here per distance band, never as a single number.
- **Pareto frontier** — configurations where one objective (memory) cannot improve without worsening another (elevation accuracy). Marking your operating point on a measured frontier turns a design choice into a defended one.
- **Meta-Kernel** — a convolution whose weights are generated per-neighbour from the *relative* 3D offsets to those neighbours, so the operator adapts to the varying physical spacing between pixels in a spherical projection. From RangeDet.
- **Occlusion depth** — how many returns, and how far behind, were discarded at a pixel by the many-to-one projection. A silhouette-edge cue obtained for free from an artifact.
- **Incidence-corrected albedo ($\hat\rho$)** — intensity corrected for range and for the angle the beam struck the surface, $\hat\rho = Ir^2/\cos\theta_{\text{inc}}$. Requires surface normals, which the elevation map supplies, and estimates material rather than geometry.
- **`INFERRED`** — a cell filled by bounded, agreement-gated continuation of a fitted ground plane across a small occlusion gap. Deterministic and auditable, unlike generative inpainting; forbidden from reducing cost.
- **Conservatism invariant** — the property that degrading a cell's evidence can never lower its reported cost. Enumerated in Part 16 and enforced by a property test rather than by convention.
- **Perception-limited speed / see-and-stop speed** — the fastest speed at which stopping distance still fits inside detection range, $v_{\max}(R) = -at_r + \sqrt{a^2t_r^2 + 2aR}$. The point where the perception analysis becomes an operational limit.
- **Outdriving fraction** — the share of a drive spent above the perception-limited speed. A metric we define ourselves, because none exists.
- **Progressive level of detail (LOD)** — transmitting the coarsest map level first and refining as bandwidth allows, so a degraded link yields a coarser map rather than a torn or absent one.
- **Multi-echo / first-and-last return** — a single pulse returning from both a canopy top and the ground beneath it. Their difference is vegetation depth; the last return is bare earth. Standard practice in airborne LiDAR terrain modelling.

---

*This bible explains WHY. It is the source of truth on intent: when an implementation detail and this document disagree, this document defines what the system is trying to be. The companion build map defines WHAT to build, in what order, how to test each piece, and what is genuinely future work.*

*Four claims carry this project. **First:** the naive reading of the problem statement is unimplementable, because a single-value elevation map cannot represent an overhang — multi-layer cells fix it. **Second:** the resolution numbers in the statement are not arbitrary choices but the sensor's own sampling limit — deriving them makes the design defensible, portable, and better than specified. **Third:** "no returns" is not "no obstacle", and the range where that stops being true is computable from the datasheet — so past it we say `UNKNOWN` instead of `FREE`. **Fourth:** detection range and stopping distance together set an operational speed limit — the sensor, not the vehicle, is what caps how fast this thing may safely go, and we can hand the sponsor that number.*

*Everything else in these thirty parts is machinery in service of those four — and Part 16 is the argument that the machinery is safe: every path where information is missing resolves the same way, and that is a property we test rather than a principle we assert.*

*The through-line, if you read nothing else: **beam geometry → detection range → what the map may claim → how fast you may drive.** Each arrow is one derivation, none of them are fitted, and the last one is the only number the sponsor has to act on.*





