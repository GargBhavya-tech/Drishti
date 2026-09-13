# DRISHTI — The Master Bible

*Distance-Resolved Instantaneous Semantic Height & Traversability Imaging*
*Smart India Hackathon 2026 · DRDO Problem Statement 26053 · Team Phir Hera Pheri*

**This is the single consolidated reference for the whole project** — theory, architecture, real training data and results, the accuracy-improvement work, the frontend/demo, prior-art positioning, honest limitations, and how to run everything. It merges `DRISHTI_Project_Bible_v3.md` (why), `DRISHTI_Build_Map.md` (what/when, referenced not reproduced), `HANDOFF.md` (session-by-session build log), `TRAINING_RESULTS.md`, `BASELINE_COMPARISON.md`, `RESEARCH_FINDINGS.md`, and `README.md` into one document, organized to read start-to-end with zero prior context.

**A note on staying in sync**: the original `DRISHTI_Project_Bible_v3.md` remains the canonical *intent* document — where an implementation detail and it disagree, it wins. `DRISHTI_Build_Map.md` remains the canonical ticket-by-ticket build order and acceptance criteria. This file is a *consolidation*, not a replacement; it will drift from the source files over time unless updated alongside them. If a number here ever looks stale, check `HANDOFF.md`'s own session log first — that is where day-to-day status actually gets updated.

---

# PART A — THE PROBLEM AND THE FOUR CLAIMS

## A.1 What problem is this solving, in plain English

A LiDAR fires laser beams in all directions and measures how long each takes to return — roughly a million 3D points per second. Two obvious things to do with that are both wrong:

1. **Keep it all in 3D.** Accurate, unusable — you cannot process a million points/second on a vehicle computer, nor store a fine 3D grid of a 200 m neighbourhood in reasonable memory.
2. **Flatten to a 2D occupancy grid.** Cheap, fast, throws away height. A 12 cm kerb, a 40 cm ditch, and a branch at 1.9 m all vanish — which for a wheeled ground vehicle *are* the problem.

The middle path is **2.5D**: a flat grid of cells, each remembering height as well as occupancy, at **variable resolution** — fine near the vehicle where safety depends on it, coarser further out where it does not.

DRISHTI answers six questions, in order, before handing anything to a planner:

1. What is physically knowable here, given the sensor's beam geometry at this range?
2. What is each point, and is it moving?
3. Where should detail live — how big should a cell be, and why?
4. What does a cell actually contain — not "how tall" but the vertical story (ground / gap / ceiling / confidence)?
5. What did we fail to see — because a hole in the ground is defined by returns that never came back?
6. When does "no returns" stop meaning "nothing there"?
7. How fast may we go, given all of the above?

Questions 5 and 6 are the ones almost nobody asks, and between them cover the two hazards most likely to kill a ground vehicle: the ditch you drove into, and the pole you never saw. Question 7 makes the answers operational instead of academic.

## A.2 The four claims this project stands on

**Claim 1 — The naive reading of the problem statement is unimplementable.** It asks for *overhang detection* and a *2.5D elevation map* in one breath. A single height per cell **cannot** represent "drivable road under a bridge": max-height reports an impassable wall, min-height drives you into the branch, mean-height invents a wall at head height. Multi-layer cells (Part C.9) fix it.

**Claim 2 — The resolution numbers in the statement are not arbitrary; they are the sensor's own sampling limit.** Derived from four lines of beam geometry (Part C.3). The schedule becomes *defensible* rather than *chosen*, regenerates for any sensor from a config file, and strictly exceeds what was specified. Corollary: the statement's own checkpoints (5 cm → 50 cm across 10 m → 100 m) are **linear in range**, exactly what constant angular resolution produces — the statement is quoting physics without saying so.

**Claim 3 — "No returns" is not "no obstacle", and the boundary is computable.** Thin objects at range — poles, cyclists, fence posts — fall below the sensor's sampling density and are silently smoothed away by systems that treat sparsity as noise. The Sparsity Trap (Part C.11) computes the expected return count for the smallest object of interest, per cell, per range, and reports `UNKNOWN` rather than `FREE` past the point where absence stops being informative.

**Claim 4 — The sensor, not the vehicle, is the speed limit — and it's computable.** Detection range and stopping distance together define the fastest speed at which a vehicle can still stop for what it can see (Part C.15). With this sensor at this mount height, a UGV cannot safely exceed **43 km/h** in terrain where a 2 m ditch is possible. That number converts the sensor recommendation from an opinion into a costed trade-off: doubling beam count buys 21% more speed, raising the mount 77 cm buys 11%.

Claims 1–3 make the map honest. Claim 4 makes the honesty *actionable*, and the Conservatism Invariant (Part C.16) makes it *checkable*: a property test asserting no code path can turn `UNKNOWN` into `FREE`.

## A.3 Six design principles behind every decision

1. **Solve problems structurally, not with careful code.** Prefer representations where the failure mode *cannot occur* (the clipmap's power-of-two nesting; the fovea's `min()` composition).
2. **Let the sensor's native geometry do the work.** Work in the sensor's coordinate system for perception; derive the map's resolution schedule from the same geometry.
3. **Reuse what is already built and tested.** The FusionSegNet backbone adapts rather than gets rebuilt under time pressure; every deviation is justified by a specific documented failure mode.
4. **State limitations; do not hide them.** A panel that finds a hidden weakness stops believing everything else; a panel that is *shown* the weakness believes the rest.
5. **Every number must be derived or measured, never fitted to look right.** One parameter with a physical meaning beats two tuned to endpoints.
6. **Never invent information, and make that checkable.** Missing information is said so; inherited information is flagged (`PROVISIONAL`, `INFERRED`) and forbidden from reducing cost. This is what lets the Conservatism Invariant be a *tested property*, not an assertion — and why generative terrain completion and runtime weight adaptation are rejected (Part C.22): both would trade this property away.

---

# PART B — THE FULL PIPELINE, AT A GLANCE

```
LiDAR sweep (~120k points, 10 Hz) + K past sweeps + ego pose
    │
LAYER 0 — Sensor model: beam geometry → resolution schedule, hazard spec sheet, N_exp curves
    │
LAYER 1 — Ingest & polar front end: deskew → ego-compensate → spherical projection →
           13-channel tensor (this build's actual channel set, see Part E) → circular padding
    │
LAYER 2 — Segmentation (FusionSegNet: EfficientNet-B0 → ASPP → Attention U-Net + SE)
LAYER 2b — Ground prior (column-wise incremental walk; LABELS, never strips)
    │
LAYER 3 — Taxonomy: dataset classes → 10-class DRISHTI vehicle taxonomy
    │
LAYER 4 — The Cartesian clipmap: nested power-of-two levels, toroidal, world-anchored
    │
LAYER 5 — Cell aggregation: GPU scatter, multi-layer cells, height distribution
    │
LAYER 6 — Observability: ray-casting, UNKNOWN/FREE/OCCUPIED/OCCLUDED, negative obstacles
    │
LAYER 7 — Sparsity Trap & derived confidence
    │
LAYER 8 — Temporal fusion: static accumulation, dynamic tracking (no smear)
    │
LAYER 9 — Adaptive fovea: allocate by TIME-TO-CONTACT, never coarsen below Layer 0's floor
    │
LAYER 10 — Traversability & planner interface
    │
LAYER 11 — Perception-limited speed envelope
    │
LAYER 12 — Dashboard        LAYER 13 — Eval harness

┌────────────────────────────────────────────────────┐
│ THE CONSERVATISM INVARIANT applies across every     │
│ layer: degrading information may never decrease     │
│ reported cost. Property-tested, not asserted.       │
└────────────────────────────────────────────────────┘
```

**Four architectural boundaries, enforced deliberately:**
1. The network never invents a hazard — `NEGATIVE_OBSTACLE`/`OVERHANG` come from geometry, never a predicted class.
2. The adaptive fovea can only refine, never coarsen — Layer 0's schedule is a hard floor.
3. The ground filter labels; it never strips.
4. Levels are alternative views, never summed — one query returns one cell from one level.
5. Nothing is generated — every value is measured, derived by a stated formula, or inherited and flagged as such.

---

# PART C — THE ARCHITECTURE, LAYER BY LAYER

## C.3 Layer 0 — The Sensor Model

**Files:** `sensor/sensor_model.py`, `sensor/schedule.py`. Every number in this project starts here: the schedule becomes a **derived quantity**, not a hardcoded literal. Change the sensor description and level boundaries, memory budget, hazard spec sheet, and confidence thresholds all move consistently.

**The five formulas** (HDL-64E: $\Delta\theta=0.1728°$, $\Delta\phi=0.4254°$, $\phi_{max}=+2°$, $h=1.73$m; this build's real sensor is Ouster OS1-64, see Part E.1):

$$s_t(r) = r\Delta\theta \quad \text{(tangential spacing)}$$
$$s_r(r) = \frac{h\Delta\phi}{\sin^2\alpha} \approx \frac{r^2\Delta\phi}{h} \quad \text{(radial ground spacing — the one everybody forgets)}$$
$$s_v(r) = r\Delta\phi \quad \text{(vertical spacing on a wall/pole/person)}$$
$$r_{\max}(t) = \frac{t}{\Delta\phi} \quad \text{(range at which a feature of height } t \text{ still resolves)}$$
$$r_{\max}(w) = \sqrt{\frac{wh}{\Delta\phi}} \quad \text{(range at which a negative obstacle of width } w \text{ is still straddled)}$$

At 100 m, consecutive beam rings land **43 metres apart** on flat ground (HDL-64E) — terrain beyond ~50 m is not meaningfully sampled by a 64-beam sensor at this height. Consequence: **a uniform 5 cm grid at 100 m is unfillable, not merely wasteful.**

**Resolution schedule**, quantized onto power-of-two levels ($c_0=5$cm):
$$c(r) = c_0 \cdot 2^{\lceil\log_2(r\Delta\theta/c_0)\rceil}, \qquad r_\ell = c_\ell/\Delta\theta$$

| Level | Cell size | Sensor-derived outer radius (HDL-64E) |
|---|---|---|
| L0 | 5 cm | 16.6 m |
| L1 | 10 cm | 33.2 m |
| L2 | 20 cm | 66.3 m |
| L3 | 40 cm | 132.6 m |

The statement asked for 5 cm within 10 m and 50 cm out to 100 m; the derivation gives 5 cm within 16.6 m and 40 cm out to 132 m — **strictly exceeds the requirement in both directions, not tuned to it.**

**Hazard spec sheet** (worked example, HDL-64E): 15 cm kerb → 20.2 m; 30 cm kerb → 40.4 m; 1 m trench → 15.3 m; 2 m ditch → 21.6 m; 4 m crater → 30.5 m; 40 cm branch (overhang) → 53.9 m; **5 cm cable → 6.7 m** (a hanging cable is effectively undetectable beyond ~7 m — physics, not a bug, and stating it is what makes the other numbers believable).

**Extrinsic self-calibration (§3.6):** a mount pitch/roll error of $\epsilon$ radians produces height error $\Delta z(r)=r\epsilon$ that grows linearly with range. A 0.5° error → 44 cm phantom slope at 50 m, **larger than every hazard we claim to detect** — it looks like gently-sloping terrain, not an error. Fix: fit a plane to confirmed ground points over a rolling window; the normal's deviation from $\hat z$ is the accumulated mount error, fed back into `SensorModel`, updated slowly (EMA over minutes), displayed on the HUD.

## C.4 Layer 1 — Ingest and the Polar Front End

**Why a range image**: a rotating LiDAR samples at constant angular resolution, so the ground footprint of one pixel grows with range automatically — the same foveation the map wants. Point-based (PointNet++/KPConv) and voxel/sparse-conv (SPVNAS/Cylinder3D) backbones were considered and rejected (Part C.22) for $O(N\log N)$ neighbour search cost and per-frame hash-table construction / CUDA toolchain risk respectively.

**Deskew** (do this first): a spinning LiDAR takes 100ms/rotation; at 15 m/s the vehicle moves 1.5m during one "scan" — thirty 5cm cells wide. Transform every point via SLERP/lerp between start/end poses using its own timestamp.

**Channel tensor** — the Bible's original 14-channel design (x,y,z,range,intensity,valid_mask,ground_prior,4×motion residual,range_gradient,occlusion_count,occlusion_spread) was descoped for this build to a **13-channel tensor**, evolved across two sessions — see Part E.1 for the exact, current, real channel list and why it differs from the Bible's original spec (motion residuals were cut per Build Map Ticket #27; range-corrected reflectivity and surface-geometry channels were added later, validated against real data first).

**Circular padding**: the sensor covers 360°, so image left/right edges are not real boundaries. All horizontal padding uses `mode='circular'`; vertical stays zero (a real FOV boundary).

**Occlusion depth** (`occlusion_count`, `occlusion_spread`): free information from the many-to-one projection collision that's normally discarded — a pixel with 5 discarded returns close behind it is a thick surface; 5 discarded returns 40m behind it is a silhouette edge against something distant. Two channels because count alone is ambiguous.

## C.5 Layer 2 — Segmentation, Ground Prior, Loss

**Files:** `perception/ground_prior.py`, `perception/segnet.py`, `perception/losses.py`. Deliberate stance: **segmentation is not the contribution.** The FusionSegNet backbone (~5.82M params, see Part E for the real measured count) sits in the same competitive band as SalsaNext/CENet/FIDNet (Part F). Adapt an existing, working architecture rather than build a new one under time pressure.

**Ground prior — label, never strip** (the single most important correction across all bible revisions): if ground points are removed before projection, they never reach the elevation map — and the ground *is* the elevation map. Column-wise incremental walk (not per-sector plane fit, which fails on slopes/crests): compare each point's slope only to the previous *accepted* ground point in the same azimuth column.

**Network**: EfficientNet-B0 encoder (from scratch, first conv expanded from 3→13 channels) → ASPP bottleneck (multi-scale receptive fields matter more for LiDAR than cameras, since an object's pixel footprint varies enormously with range in spherical projection) → Attention U-Net decoder with SE blocks on every skip → deep-supervision aux head at 1/8 resolution.

**Loss stack**: Lovász-Softmax (primary, optimises IoU directly, handles extreme imbalance) + confidence-weighted cross-entropy (secondary, class-weighted — see Part G) + deep-supervision aux term + **optional focal loss** (added this session, see Part G.5). All losses masked to valid pixels only.

## C.6 Layer 3 — Taxonomy

**File:** `perception/taxonomy.py`. Ten DRISHTI classes, a defence-UGV vocabulary, not a dataset's urban vocabulary:

| ID | Class | Meaning to the planner |
|---|---|---|
| 0 | `UNKNOWN` | Never observed, or observed past the point where absence is informative |
| 1 | `DRIVABLE` | Smooth, load-bearing, full speed |
| 2 | `CAUTION` | Passable with a speed/cost penalty |
| 3 | `NON_TRAVERSABLE` | Slope, rubble, water |
| 4 | `STATIC_OBSTACLE` | Wall, pole, building, fence, log |
| 5 | `VEGETATION` | Looks solid to the LiDAR, often driveable through |
| 6 | `VEHICLE` | Tracked entity with a velocity |
| 7 | `PEDESTRIAN` | Highest protection priority |
| 8 | `NEGATIVE_OBSTACLE` | **Geometry only** — never predicted by the network |
| 9 | `OVERHANG` | **Geometry only** — never predicted by the network |

**Machine-checked boundary**: `assert_taxonomy_valid()` runs at import time and raises if any dataset class maps to 8 or 9 — the "network never invents a hazard" architectural boundary, enforced, not just stated.

**Real RELLIS-3D mapping** (this build's actual dataset): `pole`→STATIC_OBSTACLE, `log`→STATIC_OBSTACLE, `building`→STATIC_OBSTACLE, `object`→STATIC_OBSTACLE (catch-all), `grass`/`tree`/`bush`→VEGETATION, `mud`/`puddle`→CAUTION, `dirt`/`asphalt`/`concrete`→DRIVABLE, `water`/`fence`/`barrier`/`rubble`→NON_TRAVERSABLE, `person`→PEDESTRIAN, `vehicle`→VEHICLE.

## C.7 The Coordinate System Decision

**Decision: polar for perception, Cartesian for the map, one point-wise transform between them.** The front end stays a range image (sensor-native, foveation-for-free). The map stays a world-anchored Cartesian clipmap (exact nesting, free rotation, cheap accumulation, native planner interface). A continuous polar grid was considered *as the map itself* and rejected: bin edges are still discrete boundaries (the "no boundary anywhere" claim isn't available), ego-anchored accumulation must re-bin every frame under translation *and* rotation, the origin degenerates into slivers, and exact demotion requires power-of-two nesting which a continuous polar schedule doesn't have.

## C.8 Layer 4 — The Foveated Clipmap

**Files:** `grid/clipmap.py`, `grid/addressing.py`. A quadtree was considered and rejected: pointer chasing, per-frame rebuild cost (the fovea is ego-centric), and the 2:1 balancing ripple that cascades re-subdivision every update. The clipmap is **not a tree** — no parent pointers, no subdivision, no balancing constraint.

**Structure**: $L=4$ flat $N{\times}N$ arrays ($N=512$, power of two), level $\ell$ cell size $c_\ell = c_0 \cdot 2^\ell$, each centred on the vehicle. Reference config: 262,144 cells/level × 4 = 1,048,576 total cells, 12.58 MB (v1 byte layout).

**Addressing**: world → global index $i=\lfloor x/c_\ell\rfloor$; toroidal storage index via bitwise AND ($N$ a power of two): $s_i = i \& (N{-}1)$. One multiply, two ANDs — no pointer chase.

**Why alignment error is structurally impossible**: all levels share one world origin with power-of-two cell sizes, so a level-$(\ell{+}1)$ cell at $(I,J)$ covers *exactly* four level-$\ell$ cells $(2I,2J),(2I{+}1,2J),(2I,2J{+}1),(2I{+}1,2J{+}1)$ — no fractional overlap, ever, at any position. This is why the schedule is 5→10→20→40cm and not 5→50cm: $50/5=10$ is not a power of two.

**Scrolling**: ego motion is an index shift, $O(N(|\Delta i|+|\Delta j|))$ per level, not $O(N^2)$ rebuild. Rotation costs nothing (world-anchored). **The most dangerous bug in the project**: forgetting to clear incoming scrolled rows silently returns stale data from 100m behind as though it were ahead. Two defences: clear-on-scroll (primary) + per-cell stamp validation (cross-check, cheap, turns a silent wrong answer into an honest `UNOBSERVED`).

**Invariant**: levels are alternative views, never summed. `lookup()` returns exactly one cell from exactly one level.

## C.9 Layer 5 — Cell Aggregation

**Files:** `grid/scatter.py`, `grid/cell.py`, `grid/layers.py`. **The contradiction Claim 1 resolves**: a road under a bridge has points at z≈0 and z≈4.2-4.6m. Max-height reports an impassable wall; min-height drives you into a branch elsewhere; mean-height invents a wall at head height. **Every single-value representation is structurally wrong**, not tuned wrong.

**Multi-layer cells**: ground layer $[z^g_{min}, z^g_{max}]$, gap, ceiling layer $[z^c_{min}, z^c_{max}]$ if present. $\text{clearance} = z^c_{min} - z^g_{max}$ ($+\infty$ if no ceiling). Extracted from an 8-bin height histogram: ground = lowest occupied bin + neighbours; scan upward for the first sufficiently-long empty run; ceiling = first occupied bin above it.

**Byte layout**: 12 bytes single-layer / 16 with ceiling — `h_min/h_max/h_mean` (int16, 1cm fixed-point), `h_m2` (Welford variance), `count`, `class_conf` (4-bit class + 4-bit confidence κ), `flags`. Structure-of-arrays (not array-of-structs) enables a hot/cold split: the planner touches only `(cost:uint8, h_max:int16)` = 3 bytes/cell every control cycle.

**The scatter kernel** — no Python loops, ever. The whole projection is a flat index plus `scatter_reduce`, ~20 lines, GPU-resident, including the polar→Cartesian transform. Class aggregation is a **mode**, not a mean (averaging class IDs is meaningless — scatter a per-class histogram, argmax).

**Incidence-corrected albedo (§9.5)**: `NON_TRAVERSABLE` includes standing water/wet mud, which are geometrically *flat and driveable-looking*. Raw intensity conflates material with range/angle; correcting requires the surface normal (which the elevation map's own 3×3 neighbourhood already computes for slope): $\hat\rho = Ir^2/\cos\theta_{inc}$. Guard the grazing case ($\cos\theta_{inc}>0.15$). A ditch **displaces** returns beyond expected range; water **absorbs** them (dropout, not a weak return) — this disambiguates two hazards that would otherwise both collapse to "missing returns."

## C.10 Layer 6 — Observability and Negative Obstacles

**Files:** `observability/raycast.py`, `observability/negative_obstacle.py`. Four-state model: `UNOBSERVED` (default) / `FREE` (beam passed through) / `OCCUPIED` (beam terminated here) / `OCCLUDED` (beam would have passed but was blocked closer in — keeps the map honest; the area behind a truck is not free space).

**The expected-return test**: for a beam at depression $\alpha$ from height $h$, if the ground drops by $d$: $\Delta = r' - r_{exp} = \frac{r_{exp}\,d}{h}$ — the range shadow. Worked example: $h=1.73$m, $d=0.5$m ditch at $r=20$m → $\Delta = 5.78$m, a huge, unmistakable signal. **But detection range is only ~21.6m** because it's sampling-limited (does a ring land in the ditch at all), not signal-limited — a real, quotable finding: *"detection is sampling-limited, not signal-limited,"* which tells a sponsor exactly what to change (mount height, beam count) to extend range.

**The false-positive fix**: the naive model assumes flat ground; on a crest/downslope it fires everywhere. Fix: compute expected range from a *locally fitted* ground plane (reusing the column-wise walk's own confirmed returns) — the discriminator is ring-to-ring **inconsistency**, not absolute deviation — plus 3-consecutive-scan temporal confirmation before promoting `SUSPECT`→`NEGATIVE_OBSTACLE`.

## C.11 Layer 7 — The Sparsity Trap

**File:** `observability/sparsity.py`. Claim 3's engine. An object of extent $t{\times}w$ at range $r$ gives:
$$N_{exp}(r;t,w) = \frac{tw}{r^2\Delta\phi\Delta\theta}$$
Returns fall off as $1/r^2$, not $1/r$. **Blind range** where a minimum object drops below one expected return: $r_{blind} = \sqrt{tw/(\Delta\phi\Delta\theta)}$. Worked (HDL-64E): pedestrian → 195m; utility pole → 164m; thin fence post → **67m**; 1m kerb strip → 82m.

**Decision rule**: $\kappa = N_{obs}/N_{exp}$. $\kappa\ge1$ → normal confidence. $0<\kappa<1$, structured returns → `SPARSE_STRUCTURED` (low confidence, not free). $N_{obs}=0, N_{exp}\ge1$ → `FREE` (sensor would have seen it). **$N_{obs}=0, N_{exp}<1$ → `UNKNOWN`, never `FREE`** — the sensor *could not* have seen it, so absence proves nothing. This is the whole contribution in one line, and it has zero free parameters — $t_{min}/w_{min}$ come from the vehicle's own stated safety requirement, not a fit.

## C.12 Layer 8 — Temporal Fusion

**Files:** `temporal/static_layer.py`, `temporal/motion.py` (this build's tracker is simpler than the Bible's full Kalman-filter design — see Part E). Static/dynamic split: static layer accumulates in the clipmap; dynamic objects are tracked **entities**, never written into the grid — composited at query time, so there's nothing to leave behind (no pedestrian smear trail, the failure every naive accumulation implementation exhibits).

**Chan-merge accumulation** (exact, no revisiting points): $n=n_A+n_B$, $\delta=\mu_B-\mu_A$, $\mu=\mu_A+\delta\frac{n_B}{n}$, $M_2=M_{2,A}+M_{2,B}+\delta^2\frac{n_An_B}{n}$. Confidence decays with age: $\text{conf}\leftarrow\text{conf}\cdot e^{-(t-t_{cell})/\tau}$, $\tau\approx10$s.

**Promotion/demotion**: demotion (fine→coarse) is **exact** — provably so, only because of power-of-two nesting. Promotion (coarse→fine) cannot create information: copy the coarse value, flag `PROVISIONAL`, reduce confidence, clear on first real measurement. Never fabricate fine detail and present it as measured.

## C.13 Layer 9 — The Adaptive Fovea Controller

**File:** `attention/fovea_controller.py`. **Why distance is the wrong variable**: at 60 km/h an obstacle 40m directly ahead is 2.4s away (critical); the same obstacle 40m to the side, travelling straight, is never reached. Resolution should follow **time-to-contact**, not range.

$$v_{close}(p)=\mathbf v\cdot\hat p, \quad \text{TTC}(p)=\frac{\|p\|}{\max(v_{close}(p),v_{min})}, \quad c_{ttc}(p)=c_0\left(\frac{\text{TTC}(p)}{\tau_0}\right)^\gamma$$

**The composition rule that keeps this safe**: $c(p)=\min(c_{range}(r), c_{ttc}(p), c_{object}(p), c_{boundary}(p))$. Taking the **minimum** means every extra term can only make a cell *finer* — the sensor-derived schedule is a hard floor that is never relaxed. No bug in the fovea logic and no bad velocity estimate can ever make the map coarser than spec.

**This build's Profile B** = $\min(c_{range}, c_{ttc}, c_{boundary})$ — $c_{object}$ (any tracked entity gets a fine patch regardless of TTC) is **designed, not built**; it needs a Kalman tracker this project doesn't build. Its absence is named explicitly, not hidden — a distant laterally-moving pedestrian is not specially refined here, which is exactly the case $c_{object}$ exists for.

**Worked table** (15 m/s, $\tau_0=1$s, $c_0=5$cm, $\gamma=1$, $v_{min}=2$m/s), verified against the real HDL-64E config: 15m ahead → 5cm; 100m ahead → 33cm (TTC refines beyond the 40cm floor); 15m to the side → 5cm (floor holds, standstill-safe).

**New this session — saccadic gaze steering**: `find_gaze_target()`, an argmin-TTC reduction over candidate hazard points, picks the single most urgent point to visually lock onto — not a new metric, a new *reduction* over the existing TTC quantity. Frontend renders this as a beam + pulsing ring onto the nearest urgent hazard (Part H.2).

## C.14 Layer 10 — Traversability and the Planner Interface

**Files:** `planning/traversability.py`, `planning/costmap.py`. Vehicle model declared once in `configs/vehicle_ugv.yaml`, never hardcoded (enforced by `tests/test_vehicle_config.py`, which greps the repo for the config's own literals appearing anywhere else):

```yaml
max_slope_deg: 25          max_step_height_m: 0.20
min_clearance_m: 2.50      ground_clearance_m: 0.35
width_m: 2.10              max_roughness_m: 0.08
min_object_t_m: 1.00       min_object_w_m: 0.10
braking_a_ms2: 4.0         t_react_s: 0.30
```

$$\text{cost} = \begin{cases}\text{LETHAL} & \text{slope/step/clearance violated, or NEGATIVE\_OBSTACLE}\\ \text{UNKNOWN\_COST} & \text{UNOBSERVED, SPARSE\_STRUCTURED, or }\kappa<\tau_\kappa\\ w_1\frac{\text{slope}}{\text{max}}+w_2\frac{\text{rough}}{\text{max}}+w_3\cdot\text{class\_penalty} & \text{otherwise}\end{cases}$$

`UNKNOWN_COST` is neither zero nor lethal — a tunable, high, *finite* cost, so the planner prefers known-good routes but can traverse unknown ground when it must. **The map is vehicle-agnostic; the cost map is not** — swap the config, same stored map yields a different cost map, which is what makes this a component rather than a demo.

**This session's A\* planner** (`planning/path_planner.py`): 8-connected, Euclidean admissible heuristic, LETHAL cells never expanded (hard walls), UNKNOWN cells expensive-but-passable (never forbidden — the tri-state argument extended to the planner). **Kinodynamic smoothing** (`planning/path_smoothing.py`): a Catmull-Rom spline through A\*'s waypoints (not a true Dubins curve — see Part H.2 for why that trade was made deliberately) + circumradius curvature estimate + friction-limited cornering speed, sharing the same μ table as the friction governor below.

## C.15 Layer 11 — The Perception-Limited Speed Envelope

**File:** `planning/speed_envelope.py`. $d_{stop}(v)=vt_{react}+\frac{v^2}{2a}$. Setting $d_{stop}=R$ and solving:
$$v_{max}(R) = -at_{react} + \sqrt{a^2t_{react}^2 + 2aR}$$

With $a=4$m/s², $t_{react}=0.3$s: 5cm cable (6.7m) → **22.5 km/h**; 15cm kerb (20.2m) → **41.7 km/h**; 2m ditch (21.6m) → **43.2 km/h**; 4m crater (30.5m) → **52.1 km/h**; thin fence post presence (66.8m) → **79.0 km/h**. **The vehicle's safe speed is set by the shortest detection range among hazards the terrain can plausibly contain.**

**Costed trade-off**: raising mount 1.73m→2.5m: ditch range 21.6→26.0m, speed **+11%**. Doubling to 128 beams: 21.6→30.5m, speed **+21%**. Both give square-root returns (speed scales as the *fourth root* of beam count) — diminishing returns, stated honestly.

**Advisory only** — published alongside the cost map; what a planner does with it is the planner's contract, never a controller itself.

**This session's Semantic Friction Governor** (`planning/friction.py`) extends this: `braking_a_ms2` is derated per-class using real terramechanics-cited friction coefficients (Part H.1), so the network's own terrain classification changes the braking-limited speed, not just the sensor-range-limited one.

## C.16 The Conservatism Invariant

**File:** `planning/conservatism.py`. Turns "uncertainty resolves toward caution" from a principle stated in a dozen places into a **property machine-checked over the whole system**:
$$\text{cost(FREE)} \le \text{cost(known rough)} \le \text{cost(UNKNOWN)} \le \text{LETHAL}, \qquad E'\sqsubseteq E \implies \text{cost}(E') \ge \text{cost}(E)$$

**Losing information can never make the system more permissive.** Fourteen distinct information-deficit paths (never observed, occluded, sparse, provisional, inferred, single-point, low class confidence, grazing incidence, ambiguous absence, step beyond L0/L1, stale, bad velocity, high track residual variance, sensor degraded) all resolve the same direction — enumerated once, tested exhaustively (10,000 Hypothesis-generated cases in this build, `tests/test_conservatism.py`). The strongest single assertion: **no reachable state turns `UNKNOWN` into `FREE`.** Not certifiable, but it is the *shape* a safety case takes — a meaningfully different signal than a bare accuracy number for a defence sponsor.

## C.17–C.21 Visualisation, Eval Harness, Memory, Latency, Degraded Comms

Condensed (full detail in the original Bible; the practical, built forms are in Parts E–H below):

- **Dashboard** (Part H): live 2.5D map, split-screen comparisons, HUD (cells/level, MB, P50/P95 latency, stamp-mismatch counter — must read zero), the foveation slider, overlay toggles. **Seven demo beats**: the trench (split-screen 2D-occupancy-says-clear vs. DRISHTI-says-lethal); the bridge and branch; the Sparsity Trap split-screen with a vanishing-then-recovered pedestrian; uniform vs. foveated; the γ slider; the pedestrian + re-planned path; the speed envelope collapsing in dust.
- **Eval harness**: mIoU per distance band (clipmap-level-aligned, not arbitrary bins — real result in Part F.3), memory/latency/hazard-spec-sheet metrics, ablations, a 32-beam portability ablation, held-out-split discipline (fixed before any tuning, checked in CI).
- **Memory, honestly**: quote **16×** against a dense uniform 2.5D grid, not the 267× against a dense 3D strawman nobody would build. Real measured: 12.58 MB (v1) for the full clipmap; hot plane 786KB (L0) / 3.1MB (all levels).
- **Latency**: measure scan-complete-to-map-ready, not just inference. Report P50/P95, not mean. "FPS above the sensor rate is not a virtue" — a 10Hz sensor makes 60fps mean running 5× on unchanged data; report latency and explain why.
- **Degraded comms**: the clipmap's own mipmap invariant (each level is independently a complete map) gives progressive coarse-first transmission for free — designed, not yet built in this codebase.

## C.22 Design decisions rejected, and why

Quadtree/octree (pointer-chasing, rebuild cost, 2:1 balancing ripple). Continuous polar map (no boundary-free claim; re-bins under ego rotation; origin degenerates). $c(\rho)=a+k\rho^2$ fitted schedule (two constants with no physical meaning). 5→50cm hierarchy as literally specified (not power-of-two, exact nesting impossible). Sparse-conv/point-based backbones (CUDA toolchain risk / $O(N\log N)$ cost). KNN boundary refinement (~46% of inference time for a gain attention gates + Lovász already absorb). Global/per-sector RANSAC ground plane (fails on slopes). Dice loss (insufficient at LiDAR's imbalance). float32 elevation (int16 sufficient, doubles memory for precision the sensor doesn't deliver). **Generative Bayesian terrain inpainting** — breaks the conservatism invariant, the most important rejection in the document; the honest alternative (`INFERRED`, bounded deterministic plane continuation) fills less but is auditable. **Runtime weight adaptation** — a fielding/qualification problem for a defence system: the system that passed acceptance testing is not the system driving the vehicle.

---

# PART D — REAL DATA, TRAINING RUNS, AND MEASURED RESULTS

## D.1 Dataset and sensor, as actually used

**RELLIS-3D** (Texas A&M, off-road proving ground), all 5 sequences (00000–00004), **Ouster OS1-64** stream — chosen over SemanticKITTI/HDL-64E as the background dataset because it matches an off-road UGV use case. Sensor config **measured from real data**, not placeholder: `d_theta_deg=0.17578125` (2048 columns/rev — the placeholder assumed 1024), vertical FOV `+17.0°/-16.4°` (33.5° span — the placeholder assumed ±22.5°/45°), `h_m=1.086` (median of real ground-classified points, ~0.15m real spread across routes). RELLIS-3D's real numeric label-ID set cross-checked against 783 frames across all 5 sequences: 17 of 20 mapped IDs confirmed, zero unmapped surprises.

**Real class-4 (STATIC_OBSTACLE) prevalence**: 4,341 of 11,522 real training frames (**37.7%**) contain at least one class-4 pixel — but only 42 ground-truth points were found in a 600-frame validation-split sample (~48M points). This large discrepancy is real and diagnosed: the chronological (last-15%-per-sequence) train/val split likely under-samples whatever trail stretches the fence posts/logs sit along. Treat any single run's val-split class-4 number as high-variance.

## D.2 Training Run #1 — single sequence

RELLIS-3D sequence 00004 only, 20 epochs, AdamW/OneCycleLR, AMP, RTX 2080 Ti. **Best mIoU 0.328** (epoch 16). Classes 1, 4, 6 essentially unlearned (too few pixels in one sequence — class 1 had only 4 training pixels). Majority-class (VEGETATION) baseline: 0.091 mIoU — the trained model was 3.6× above it.

## D.3 Training Run #2 — all 5 sequences (`checkpoints_multi`)

Same recipe, 11,522 train / 2,034 val frames across all 5 sequences. **mIoU 0.328 → 0.562, a 71% relative improvement from data alone**, same architecture/hyperparameters/epoch count. Classes 1, 3, 6 went from unlearned (0.00–0.01 IoU) to genuinely useful (0.46–0.84). **Class 4 stayed at exactly 0.0 in both runs** — the specific problem the accuracy-improvement work (Part G) targets. Classes 8/9 (`NEGATIVE_OBSTACLE`/`OVERHANG`) show zero ground-truth pixels **by permanent design**, not a data gap — no dataset label is permitted to map onto them (Part C.6).

## D.4 Fine-tune `checkpoints_multi_v2` — class weighting alone

Warm-started from `checkpoints_multi/checkpoint_epoch19.pt`, 20 epochs, inverse-sqrt-frequency class weighting added (secondary CE term only). **Final: best epoch 15, mIoU 0.5736.** Class 4 IoU: **still exactly 0.0 across all 20 epochs** — hard evidence that class weighting alone cannot fix a class this rare (0.05%–0.051% of pixels by two independent samples); the problem is data scarcity, not loss-weighting, which directly motivated CutMix (Part G.5).

## D.5 Zero-shot cross-domain benchmark — nuScenes-mini

`checkpoints_multi_v2/best.pt` (RELLIS-3D-trained, Ouster 64-beam) evaluated **zero-shot, zero fine-tuning** against nuScenes-mini (Velodyne HDL-32E, 32-beam, urban Singapore/Boston) — an extreme dual domain shift (sensor + environment).

| Mode | Overall mIoU | Speed | Vegetation recall | Drivable precision |
|---|---|---|---|---|
| Direct (32×1080, native) | **3.43%** | 10.7 FPS | 80.77% | 52.26% |
| Resampled (64×2048) | 3.13% | 8.8 FPS | 79.96% | 45.77% |

**In-domain vs. zero-shot, side by side** (the number that actually matters): RELLIS-3D in-domain 57.4% mIoU is *competitive with published SemanticKITTI SOTA* (SalsaNext 55.5%, FIDNet 55.4%) at fewer parameters. Zero-shot transfer's 3.43% sits inside the 2–8% literature norm for this class of domain shift (published cross-sensor-only shifts already drop 60%→12-18%; this is off-road→urban *and* 64→32 beam simultaneously).

**Diagnosis — "the Ground Paradox"**: RELLIS-3D's ground is 85% grass/soil, so the network learned *flat ground = VEGETATION*. On nuScenes' flat asphalt, it predicts vegetation (explaining 80.8% vegetation recall alongside 0.33% drivable recall, despite 52.3% drivable *precision*). Geometry alone cannot fix this — it's the direct motivation for the range-corrected-reflectivity work (Part G.1), which showed a real, measured +38% separability improvement on exactly this DRIVABLE-vs-VEGETATION axis.

**A real, checked positive**: unlike fixed-tensor networks (SalsaNext/FIDNet, which crash on a 32-beam input with a dimension mismatch), FusionSegNet's decoder handles the resolution change gracefully via its own `_match_size()` logic — 10.7 FPS, no crash. This means the model's beam-count adaptation is *already working*; beam-dropout augmentation (Part G.3) targets a secondary robustness margin, not the primary diagnosed failure.

## D.6 The evidence gallery — every figure in `eval/out/`, with the real numbers behind it

Ten checkpoint artifacts live in `eval/out/`: 9 PNGs + their source JSON. Read together, they are the project's actual proof set — what follows is each figure plus the exact numbers that produced it, so nobody has to re-derive them from the pictures.

**[sensor_validation_ouster.png](eval/out/sensor_validation_ouster.png) — Layer 0's own formulas, checked against real Ouster OS1-64 returns.** Two log-log plots: within-ring spacing $s_t(r)$ and between-ring (flat-ground-proxy) spacing $s_r(r)$, predicted line vs. measured points from real RELLIS-3D sweeps. **Honest reading, not a clean match**: the predicted curves undershoot the measured points at every range on both plots — real ring spacing runs roughly 3–10× the idealized flat-infinite-ground formula, worse at short range. This is expected and explainable, not a bug: real terrain isn't a perfect flat plane (the formula's core assumption), so real point-to-point spacing on rutted, sloped, vegetated ground is always going to exceed the flat-plane prediction — the formula gives a physically-grounded *lower bound*, and the real data sits above it, consistently, in the same log-log slope family. This is the single most important honesty check in the whole project: it would have been easy to only show a case where the fit looked clean.

**[checkpoint_detection_vs_range.png](eval/out/checkpoint_detection_vs_range.png) — Ticket #60, the sensor's own hazard spec sheet, verified empirically.** Two panels, 15cm kerb and 2m×0.5m ditch, detection rate vs. range, real measured curve against the Part C.3 formula's predicted 50%-detection range. **15cm kerb**: predicted 20.2m, measured 18.4m — detection is a clean step function (100%→0% in one bin), and the measured range is *slightly more conservative* than predicted (detection fails a bit closer than the formula promised, not further — the safe direction to be wrong in). **2m ditch**: predicted 21.6m, measured 20.6m — same direction, and here detection degrades gradually rather than stepping (100% at 9m down to 17% at 43m), which is itself informative: a large hazard doesn't have a hard cliff, it has a fade, exactly matching the Sparsity Trap's own $\kappa$ framing (Part C.11) rather than a binary detect/no-detect model.

**[checkpoint_trench.png](eval/out/checkpoint_trench.png) — Ticket #37, Claims 3 & 4's headline demo, side by side.** Left: a plain 2D occupancy grid built from the same synthetic sweep — a radial fan of return points with a blank disc in the middle where the ditch swallows returns; **the disc reads as unmapped/unknown space, indistinguishable from "sensor hasn't looked there yet."** Right: DRISHTI's own negative-obstacle pipeline on the identical data, flagging a small tight cluster of `NEGATIVE_OBSTACLE` cells exactly at the ditch's near edge — the concrete, visual form of "a plain 2D grid cannot see a ditch; DRISHTI can" that every version of the pitch deck leads with.

**[checkpoint_sparsity_speed.png](eval/out/checkpoint_sparsity_speed.png) — Ticket #43, Claims 3 & 4 and the conservatism property, together, on one plot.** Ego at the origin, $r_{blind}=66.8$m ring (dashed) for the binding hazard (5cm cable, R=6.7m → giving $v_{max}=22.4$km/h — note this is a *different* worked value than Part C.15's table because the scene's own synthetic object mix sets a different binding hazard than the cable-alone case), scattered points labeled `NORMAL`/`SPARSE_STRUCTURED`/`UNKNOWN` by observed-vs-expected return count at their real range. **The title states the result directly: `Conservatism property test (tests/test_conservatism.py): PASS`** — this is the one figure that visually ties the abstract invariant (Part C.16) to a concrete passing test, not just an assertion in prose.

**[checkpoint_pareto.png](eval/out/checkpoint_pareto.png) — Ticket #58, a self-consistency Pareto curve that needs no ground truth at all.** Sweeps the fovea's γ parameter (Part C.13) from 0 to 2.0, plotting implied total memory (MB) against RMS elevation deviation from the finest self-consistent map. **The curve is a near-vertical cliff**: γ=0 costs ~200MB for near-zero deviation; γ≥0.5 collapses memory to single digits of MB while deviation stays under ~2cm — the marked "knee" at γ=0.75 and chosen "operating point" at γ=1.0 both sit essentially on top of the y-axis, meaning almost all of the fovea's real memory saving is captured immediately past γ=0, with diminishing (and eventually reversing, past γ~1.5, deviation climbing past 13cm) returns beyond that. **This validates the memory claim (Part C.19's 16×) using the map's own internal agreement across resolution levels, not a labeled dataset** — genuinely a different, complementary kind of evidence from the mIoU numbers elsewhere.

**[checkpoint_semantic_map_ground_truth.png](eval/out/checkpoint_semantic_map_ground_truth.png) / [checkpoint_semantic_map_predicted.png](eval/out/checkpoint_semantic_map_predicted.png) — a real frame (RELLIS-3D sequence 00004, frame 1000), ground truth vs. `checkpoint_epoch19.pt`'s actual prediction, same bird's-eye layout.** Side by side, the two images are visually close to indistinguishable at this scale: the same two ego-circle blind spots, the same trail corridor traced in orange/yellow flanked by dense green vegetation on both sides, the same sparse red cluster near the bottom ego position. The predicted panel is explicitly titled with its real number, **"val mIoU 0.562"** (Run #2's own best figure, Part D.3) — this is what that single scalar actually looks like on one frame, not just a number in a table.

**[checkpoint_class.png](eval/out/checkpoint_class.png) / [checkpoint_height.png](eval/out/checkpoint_height.png) — Tickets covering clipmap bird's-eye rendering, both explicitly titled `SYNTHETIC scene, not real data` in the figure itself.** Class-colour and height-colour renderings of the same synthetic three-cluster scene (a dense square block plus two smaller clusters against sparse background noise), used to sanity-check that per-cell class-mode aggregation and per-cell height-range aggregation (Part C.9) produce sensible, spatially coherent output before ever being pointed at real data. **Kept as synthetic deliberately** — the module docstring says so directly — because it isolates the aggregation math from any real-data labeling noise while these two channels were first being validated.

**Numbers behind two JSON-only artifacts** (no PNG, but load-bearing):
- `accuracy_by_distance.json` (8 real frames, CPU): mIoU by distance band — [0,12.8)m: **0.339**, [12.8,25.6)m: **0.363**, [25.6,51.2)m: **0.298**, [51.2m,∞): **0.159**. This is the artifact behind the "accuracy across varying distances" PS-compliance claim in Part H.3 — note this run (8 frames, CPU) gives noticeably lower absolute numbers than the 300-frame run quoted there (0.478/0.506/0.363/0.156), consistent with both a much smaller sample and the documented CPU-vs-GPU discrepancy (Part I.1); the *shape* — accuracy holding up through the mid-bands, dropping hardest past 51m — replicates across both runs and is the trustworthy part of the finding.
- `confusion_matrix.json` (100 real frames, CPU, `checkpoint_epoch19.pt`): overall mIoU **0.354**, per-class IoU DRIVABLE 0.712, VEGETATION 0.936, NON_TRAVERSABLE 0.274, CAUTION 0.365, STATIC_OBSTACLE **0.0** (confirms Part D.3/D.4's finding independently, on a fresh 100-frame CPU sample), VEHICLE 0.396, PEDESTRIAN 0.0095. Again lower than the GPU-measured 0.562 headline — same CPU/GPU discrepancy, not a contradiction.
- `feature_hypothesis_validation.json` — the exact numbers behind Part G.1/G.2's real-data validation claims, confirmed present verbatim in this artifact: reflectivity separability 0.206→0.284, and the surface-geometry table's source numbers (DRIVABLE \|normal_z\|=0.986/curvature=0.202; VEGETATION 0.626/1.219; NON_TRAVERSABLE 0.434/1.514; STATIC_OBSTACLE 0.515/2.829 — all four figures in Part G.2's table trace to this file).
- `checkpoint_latency.json` (10 real frames, CPU, `checkpoint_epoch19.pt`): **P50=845.6ms, P95=1042.5ms end-to-end** — `load_and_assemble` (P50 447ms) and `model_forward` (P50 385ms) split roughly evenly, `argmax_to_cpu` negligible (28ms). Neither the 10Hz (100ms) nor 20Hz (50ms) sensor budget is met on CPU (headroom 0.048–0.096×, both `fits_p95: false`) — this is the artifact underlying the latency honesty note in Part H.3; **the specific P50/P95 numbers quoted there (270ms/360ms) come from a different, larger GPU-side run and are not reproduced in this repo's own `eval/out/` — treat the CPU numbers in this JSON as the locally-reproducible reference, and the GPU numbers as needing a fresh export from `drishti-gpu` before being quoted again.**

**What this gallery adds up to, honestly**: the sensor-validation and hazard-detection plots show the physics model tracks reality in the *correct direction* even where it doesn't fit tightly; the trench and sparsity-speed plots turn Claims 3/4 into pictures a judge can read in five seconds; the Pareto curve proves the memory saving without needing labels at all; and the ground-truth-vs-predicted pair is the honest, unedited look of what 0.562 mIoU actually produces on a real trail scene — including the parts (thin trail edge under vegetation) where prediction and truth start to diverge if you look closely.

---

# PART E — WHAT ACTUALLY DIFFERS FROM THE ORIGINAL BIBLE

The Bible (Part C above) describes the *intended* v3 architecture. Two build-time descopes and one major later expansion are worth stating explicitly, since the Bible itself is not automatically updated when the code diverges.

## E.1 The real input channel set (9, then 13 — not the Bible's original 14)

Ticket #27 shipped a **9-channel** tensor, deliberately smaller than the Bible's 14: `x, y, z, range, intensity, valid_mask, ground_prior, occlusion_count, occlusion_spread`. Motion residual channels (4 of the Bible's 14) were cut — "additive later, not a rewrite" per the ticket's own note; they were never added, since the temporal/motion story ended up built at the map level (`temporal/motion.py`) rather than the input-channel level.

**This session extended it to 13 channels**, validated against real data *before* implementation (Part G):
- Channel 4 (`intensity`) now holds **range-corrected** intensity, `I·(r/r_ref)²`, not raw — same slot, no channel-count change for this one.
- Channels 9–12 (new): `normal_x, normal_y, normal_z, curvature` — per-pixel surface geometry from `perception/surface_geometry.py`, azimuth-wraparound-aware, occlusion-boundary-masked.

`FusionSegNet`'s stem conv grows from 9→13 input channels via `expand_stem_conv_for_checkpoint()` — old channels' trained weights preserved exactly, new channels zero-initialised (so an old checkpoint, fed zeros on the new channels, reproduces its original output bit-for-bit; verified by a dedicated test). `perception.input_tensor.N_CHANNELS` is now the single source of truth — `perception/segnet.py` previously duplicated it as a hardcoded literal that had silently drifted; fixed to import it directly.

## E.2 The tracker is simpler than the Bible's Kalman-filter design

Built (`temporal/motion.py`, `temporal/static_layer.py`): world-anchored motion detection, Chan-merge static accumulation. Not built: the full cluster→associate→Kalman-filter entity tracker with per-track residual-variance diagnostics (Part C.12's §12.2/§12.6). The fovea controller's `c_object` term (any tracked entity gets a fine patch regardless of TTC) is consequently **designed, not built** — named explicitly in Part C.13, not hidden.

## E.3 Real, measured FusionSegNet parameter count

**5.82M parameters** (measured), not the Bible's ~8M estimate — this became the number quoted against SalsaNext (6.7M)/CENet (6.8M)/FIDNet (6M) in Part F.

---

# PART F — POSITIONING: BASELINES, PRIOR ART, AND WHAT'S REALLY NOVEL

*(Full detail and every citation in `BASELINE_COMPARISON.md` and `RESEARCH_FINDINGS.md`; this is the load-bearing summary.)*

## F.1 Fair comparison: parameter count and architecture family

| Network | Params | Input | Family |
|---|---|---|---|
| SalsaNext [arXiv:2003.03653] | 6.7M | 64×2048 range image | Encoder-decoder, dilated residual |
| CENet [arXiv:2207.12691] | 6.8M | 64×2048 range image | Concise conv + aux heads |
| FIDNet [arXiv:2109.03787] | 6.0M | 64×2048 range image | Fully-interpolation decoder |
| **FusionSegNet (this project)** | **5.82M** | **64×2048 range image** | EfficientNet-B0 + ASPP + attention-gated decoder |

FusionSegNet is the smallest of the four, same input resolution these networks were benchmarked at.

## F.2 What is NOT fairly comparable: mIoU, directly

FusionSegNet's 56.2% (Run #2) is on RELLIS-3D (unstructured off-road); the published 55.4–64.7% figures above are on SemanticKITTI (structured urban). **Presenting these side by side as a ranking would be exactly the overclaim this project's own culture argues against.** The honest claim: competitive mIoU at fewer parameters, on a domain none of the three competitors were evaluated on and which is qualitatively harder to regularize (heavy vegetation occlusion, some classes at a few hundred pixels across the entire multi-sequence set).

## F.3 Negative-obstacle detection: built on field-validated physics, not a novel claim

**Important correction, load-bearing for how this project is pitched.** DRISHTI's range-shadow negative-obstacle detection (Part C.10) is **not a new technique** — it was pioneered by Rankin & Matthies at NASA JPL for DARPA's Demo III program (2003–2006), fielded in TerraMax at the 2005 DARPA Grand Challenge, and remains the active technique in 2024 literature (Shang et al., tilted-LiDAR "spacing jump" detection, MDPI Sensors 24(24):7929). **Do not claim this as invented here** — a technically literate DRDO judge may already know Demo III.

The honest, still-strong framing: DRISHTI's real contribution is the *system* around a field-validated physical cue — a variable-resolution clipmap tied to sensor Nyquist physics (absent from the 2003–2005 fixed-resolution work), a modern deep segmentation network fused with the geometric cue (that work predates deep learning), an open tested reproducible pipeline (the JPL/DARPA work was never published as reusable code), and the Sparsity Trap's own conservatism default. **"Built on NASA JPL/DARPA-validated physics" is a stronger defence-context claim than an unproven "novel" one.**

## F.4 Comparable real UGV programs (not just academic benchmarks)

**DARPA's RACER program** (Rivière et al. 2023) independently validates this project's central architectural bet: pushing off-road UGVs to 7–10 m/s, RACER found dense semantic classification introduces enough latency for the planner to act on "misrepresented or delayed semantic obstacles and terrain geometries" — exactly the failure mode the perception-limited speed envelope (Claim 4) and variable-resolution mapping (Claim 2) exist to prevent.

**DRDO's own currently-documented UGVs** — a real, checkable, favorable capability-level comparison for this specific audience: **Daksh** (EOD teleoperated, multi-camera + X-ray, no autonomous LiDAR mapping) and **Muntra** (BMP-2-based, GPS/INS waypoint autonomy + radar/EO, tested on the flat Mahajan range terrain, no public documentation of adaptive-resolution LiDAR or negative-obstacle reasoning). Adaptive-resolution 2.5D mapping and range-shadow negative-obstacle detection are capabilities **not documented in DRDO's own currently fielded programs** — not a metrics comparison (their internal algorithms aren't public), a capability-level one.

## F.5 Citation confidence note

Sources 1–3 above (SalsaNext/CENet/FIDNet) and the Shang et al. 2024 MDPI citation are directly verifiable (arXiv IDs/DOIs resolve). The Rankin/Matthies exact venue and the Rivière et al. RACER arXiv ID came from a deep-research pass and were **not independently re-fetched** — confirm they resolve before using them in anything that leaves this repo.

---

# PART G — THIS SESSION'S ACCURACY-IMPROVEMENT WORK

Five real-LiDAR-physics improvements were proposed, then **validated against real RELLIS-3D data before any implementation** (per explicit instruction — "if there is a way to check without training that the model will become better, do that first"), then implemented, then used to launch a retrain.

## G.1 Range-corrected reflectivity

**Hypothesis**: raw intensity conflates material reflectivity with range (received power ∝ 1/r²); correcting for range should help distinguish materials that are geometrically identical (the Ground Paradox axis, Part D.5).

**Real-data validation** (60 sampled training frames): raw intensity is real, meaningfully-varying data (resolves an open question `perception/rellis_loader.py` had flagged and never checked). Range correction improved DRIVABLE-vs-VEGETATION Fisher separability **0.206 → 0.284 (+38%)** — real, positive, but modest in absolute terms; a signal booster, not a silver bullet alone.

**Implementation**: `perception/reflectivity.py`, `range_corrected_intensity(I, r) = I·(r/r_ref)²`, `r_ref=10m`. Wired into the existing intensity channel slot (no channel-count change).

## G.2 Surface normals and curvature — the strongest-validated of the five

**Hypothesis**: STATIC_OBSTACLE and NON_TRAVERSABLE pixels should show erratic normals (low |normal_z|) and high curvature vs. flat DRIVABLE/VEGETATION.

**Real-data validation**: a clean, monotonic ordering, exactly matching the hypothesis —

| Class | mean \|normal_z\| | mean curvature |
|---|---|---|
| DRIVABLE | 0.986 | 0.20 |
| VEGETATION | 0.626 | 1.22 |
| NON_TRAVERSABLE | 0.434 | 1.51 |
| STATIC_OBSTACLE | 0.515 | **2.83** (14× DRIVABLE) |

**Implementation**: `perception/surface_geometry.py`, `compute_surface_geometry()` — cross product of horizontal (azimuth-wrapped) and vertical tangent vectors for the normal; discrete Laplacian of range (wrapped horizontally, zero-padded vertically) for curvature. Masked to zero wherever a neighbour pair crosses an occlusion boundary (range jump > 1.0m) — real depth discontinuities are exactly where a naive normal computation would inject noise at the pixels that matter most. Added as 4 new input channels (9→13).

## G.3 Beam-dropout augmentation

**Real-data context**: the zero-shot nuScenes result (Part D.5) already showed FusionSegNet handles a 32-beam input gracefully via its own `_match_size()` decoder logic — the failure diagnosed there is primarily semantic (the Ground Paradox), not geometric beam-count breakage. Beam-dropout targets a secondary robustness margin, not the primary diagnosed problem — implemented anyway since it's a free training-time augmentation with zero architecture cost.

**Implementation**: `perception/train.py`'s `_apply_beam_dropout()` — **evenly-spaced row decimation** (stride 2/3/4, simulating ~32/21/16 effective beams), not random pixel dropout, because that's what a real coarser-beam sensor actually looks like.

## G.4 Multi-sweep temporal accumulation — built, deliberately not wired into training

**Real-data validation**: motion-compensated merging of 3 sweeps (real poses, `perception/multi_sweep.py`) gives only ~3% overall point-count growth (not the "3×" originally pitched) — but **+55% to +90% specifically beyond 40m**, concentrated exactly where sparse-distant-return density matters most.

**Honest limitation documented in the module itself**: only ego motion is compensated. A genuinely moving object's points from older sweeps land at its *past* world positions — it fragments into ghost copies rather than densifying, an emergent, usable motion cue (static objects densify; moving ones don't) but must be described as exactly that, not as uniform 3× density.

**Decision**: tripling per-sample data-load time for a measured ~3% overall gain was judged not worth the training-speed cost for this round without being asked to make that trade explicitly — built, tested, available via direct import (`merge_sweeps_motion_compensated`), not enabled by default.

## G.5 Minority-class CutMix + Focal Loss

**Real-data validation, confirming the priority**: class 4 (STATIC_OBSTACLE) is **0.051%** of all training pixels by one sample, **0.0000875%** in a validation-split sample (Part D.1's discrepancy) — genuinely extreme scarcity, and hard evidence (Part D.4: exactly 0.0 IoU across 20 epochs of class-weighting-only fine-tuning) that reweighting alone cannot fix it.

**Implementation — real point-level CutMix**, not naive image-space patch pasting (which would be geometrically broken for a range-image pipeline): `perception/cutmix.py` pastes a harvested real point cluster into the raw sweep **before** re-projection, so it goes through the same occlusion-aware z-buffering every real point does. Clusters extracted once from every real training frame (`eval/extract_rare_clusters.py`, one-time ~44-minute scan of 11,522 frames): **3,408 real clusters found in 4,341 of 11,522 frames** (37.7% — the same discrepancy noted in D.1). Placement: random azimuth/range within [3m, 15m), local ground height estimated from the median z of real nearby original points (not a global constant).

**Focal loss** (`perception/losses.py`, `focal_loss()`): $\text{FL} = (1-p_t)^\gamma \cdot \text{CE}$, opt-in via `DrishtiSegLoss(focal_gamma=...)`, default `None` = byte-identical to prior behaviour. Complements class weighting rather than replacing it — the two address different things (class_weight scales overall gradient magnitude per class; gamma reshapes per-pixel weighting toward hard examples within any class).

## G.6 The retrain — `checkpoints_multi_v3` — complete, and an honest negative result on the one number that mattered most

Warm-started from `checkpoints_multi/checkpoint_epoch19.pt` via the channel-expansion path (Part E.1), 20 epochs, `--focal-gamma 2.0 --cutmix-clusters rare_clusters_class4.npz`, on the same recipe as Run #2/`_v2` otherwise. **The one number that mattered most going in: class 4 (STATIC_OBSTACLE) IoU**, which had never moved off exactly 0.0 across two prior training attempts (Run #2, `_v2`).

**Two real bugs found and fixed while wiring this up**, both now permanent fixes:
1. `eval/cache_inference.py`'s `load_trained_model()` didn't have the channel-expansion fix — broke every `eval/checkpoint_*.py` script (they all load old 9-channel checkpoints through it). Fixed identically to `train.py`'s path.
2. `perception/train.py`'s `--init-from-checkpoint` path was missing a **pre-existing** legacy state-dict key remap (`e1.0.0.weight`-style keys renamed to `...conv.weight` when ASPP/decoder convs were wrapped in `CircularConv2d`, sometime between `checkpoints_multi_v2`'s original training and now) — `eval/cache_inference.py` already had this fix (`_remap_legacy_conv_keys`), `train.py` never did. Moved to `perception/segnet.py` as `remap_legacy_conv_keys()`, shared by both call sites now (single source of truth, not two copies that can drift).

**One known, expected, non-blocking test failure documented, not silently fixed**: `tests/test_checkpoint_sparsity_speed.py`'s conservatism-property check fails when run against the *old* `checkpoint_epoch19.pt`, because that checkpoint was trained on raw intensity and is now being fed range-corrected intensity — a genuinely different value on a channel it has real (non-zero) trained weights for, unlike the 4 brand-new zero-weighted channels which *are* verified bit-identical-safe. Resolves once `checkpoints_multi_v3` exists to test against (it now does).

### Final result, all 20 epochs complete: **class 4 IoU is still exactly 0.0**

**Best epoch: 16, val mIoU 0.5709.** Per-class IoU at the best epoch: UNKNOWN 0.777, DRIVABLE 0.869, CAUTION 0.169, NON_TRAVERSABLE 0.484, **STATIC_OBSTACLE 0.0**, VEGETATION 0.959, VEHICLE 0.537, PEDESTRIAN 0.771.

| Run | Best epoch | Best val mIoU | Class 4 IoU |
|---|---|---|---|
| Run #2 (5 sequences, no class weighting) | 19 | 0.562 | 0.0 |
| `_v2` (class weighting only) | 15 | 0.574 | 0.0 |
| **`_v3` (class weighting + CutMix + focal loss)** | **16** | **0.571** | **0.0** |

**The honest headline: three different training strategies, three different accuracy-improvement techniques stacked on top of each other, and class 4's IoU has never once moved off exactly 0.0.** Overall mIoU held roughly flat across all three runs (0.562→0.574→0.571 — `_v3` is not even the best of the three by that measure) — the CutMix+focal-loss changes neither helped nor meaningfully hurt the other classes, but did not fix the one class they specifically targeted.

**A real, transient, non-repeating signal worth naming rather than hiding**: at epoch 15 (not the eventual best epoch), class 4 IoU briefly showed **0.000124** — the first ever nonzero value across all three runs, a genuine break from the exact-0.0 pattern. It did **not** hold: by epoch 16 (the actual best epoch by overall mIoU) and every epoch after, class 4 was back to exactly 0.0. This reads as noise from one lucky CutMix-pasted cluster landing in a validation-adjacent position for one epoch, not a real learning signal — reported here specifically so it isn't later mistaken for evidence the fix "sort of worked."

**The training run itself also flagged its own overfitting** in its last few epochs (`WARNING: val loss has risen for 3 consecutive epochs while train loss fell`) — consistent with the model continuing to memorize training-specific detail past its useful point, unrelated to the class-4 finding but worth noting alongside it.

**What this actually means, stated plainly**: the diagnosis in Part D.4 was correct (class 4 is data-scarcity-limited, not loss-function-limited) but the specific fix (3,408 real CutMix-pasted clusters, covering 37.7% of training frames) was **not sufficient** to teach the network this class. Plausible reasons, none yet tested: the pasted clusters may not be visually/geometrically distinct enough at the placement ranges used (3-15m) for the network to learn a discriminative signal; `AUG_CUTMIX_PROB=0.3` may be too low relative to the class's extreme rarity; or the val-split methodology gap already flagged in Part D.1 (class 4 measured at 0.051% in one train sample vs. 0.0000875% in a val sample) means the *val* set itself may contain so few class-4 pixels that even a model that partially learned the class would show ~0 IoU from evaluation noise alone, not from the model's own failure — a real, unresolved methodological confound that a future session should check directly (count actual class-4 pixels in this run's held-out val split) before concluding CutMix "didn't work," rather than assuming the training fix failed when the measurement itself might be the limiting factor.

## G.7 The nuScenes fine-tune — `checkpoints_nuscenes_ft`, a direct real-data test of the Ground Paradox diagnosis

Built and run the same session as G.6, alongside `_v3` on the same GPU (confirmed safe beforehand: `_v3` runs at 6% GPU utilization, CPU/disk-bound — see Part D.6's latency numbers — so a second small job shares the card without contention). Motivation: Part D.5 diagnosed the Ground Paradox (RELLIS-3D-trained model predicts VEGETATION for nuScenes' flat asphalt, since RELLIS' own ground truth is ~85% grass/soil) purely from a zero-shot eval; this fine-tune is the first real test of whether that diagnosis is *actionable* — does a small amount of real labeled urban asphalt actually fix it — rather than a second measurement of the same gap.

**What was built** (net-new, all real code, none of it touching `perception/train.py`'s own RELLIS path): a shared nuScenes-lidarseg→DRISHTI lookup table moved into `perception/taxonomy.py` (`build_nuscenes_lidarseg_lut`, now used by both the zero-shot eval script and this new training path — one source of truth instead of two copies that could drift); `perception/nuscenes_seg_dataset.py`, a nuScenes-mini training dataset with the identical `(input_tensor, target, valid_mask)` contract as `RellisSegDataset`, splitting train/val by **whole scene** (not shuffled frames — same "don't leak near-duplicate adjacent frames" reasoning as RELLIS' own per-sequence split); and `perception/train_nuscenes.py`, a deliberately separate training script/CLI that imports and reuses `train.py`'s own checkpoint/loss/optimizer machinery rather than editing it.

**Dataset reality, stated up front**: nuScenes-mini's lidarseg-annotated set is 404 keyframes across 10 scenes — split scene-wise into 324 train / 80 val (2 held-out scenes) — two orders of magnitude smaller than RELLIS-3D's 11,522 training frames. This is a fine-tune, not from-scratch training: warm-started from `checkpoints_multi_v2/best.pt` via the same stem-conv channel-expansion path used everywhere else (9→13 channels, old weights preserved).

**Real result, 8 epochs, ~34s/epoch (finished in under 5 minutes)**:

| Epoch | val mIoU | train loss | val loss |
|---|---|---|---|
| 0 | 0.151 | 3.264 | 2.006 |
| 1 | 0.254 | 2.090 | 1.106 |
| 2 | 0.309 | 1.387 | 0.925 |
| 3 | 0.326 | 1.194 | 0.870 |
| 4 | 0.329 | 1.118 | 0.855 |
| 5 | 0.332 | 1.072 | 0.841 |
| **6 (best)** | **0.334** | 1.052 | 0.843 |
| 7 | 0.328 ↓ | 1.050 | 0.851 ↑ |

Best checkpoint saved as `checkpoints_nuscenes_ft/best.pt` (epoch 6) — epoch 7 was already ticking slightly worse on val loss while train loss kept falling, exactly the overfitting signature `perception/train.py`'s own automatic best-checkpoint tracking exists to catch; the last epoch is not blindly the one to use.

**Best-epoch (6) per-class IoU:**

| Class | IoU |
|---|---|
| UNKNOWN | 0.826 |
| **DRIVABLE** | **0.739** |
| CAUTION | 0.182 |
| NON_TRAVERSABLE | 0.0 — no learnable signal, too few pixels in nuScenes-mini's train scenes (the class-4-in-RELLIS scarcity problem, recurring here on a different class in a different dataset) |
| STATIC_OBSTACLE | 0.383 |
| VEGETATION | 0.226 |
| VEHICLE | 0.283 |
| PEDESTRIAN | 0.035 — rare class, small dataset |
| NEGATIVE_OBSTACLE / OVERHANG | n/a — zero ground-truth pixels, by permanent taxonomy design (Part C.6) |

**The headline finding**: DRIVABLE went from **0.33% recall** in the zero-shot benchmark (Part D.5, `checkpoints_multi_v2/best.pt`, no fine-tuning at all) to **0.739 IoU** after this fine-tune — a real, measured confirmation that the Ground Paradox is fixable with a small amount of real labeled target-domain data, not just a correctly-diagnosed but unaddressed failure mode.

**Honest caveat, not yet closed**: this 0.334 is **not a clean apples-to-apples number against the zero-shot benchmark's 3.43%** — the zero-shot figure was point-level mIoU over all 404 frames with no train/val split (a pure zero-shot measurement), while this fine-tune's 0.334 is pixel/range-level mIoU on 80 held-out frames from 2 scenes never trained on. A fair before/after claim needs the *pre-fine-tune* checkpoint re-evaluated on those exact same 80 held-out frames — that comparison has not been run yet. Until it is, the defensible claim is the per-class DRIVABLE number above (0.33% recall → 0.739 IoU), not a single-number "3.43%→33.4%" headline, which would overstate what was actually measured.

## G.8 A third real dataset — SemanticPOSS (Hesai Pandar40P, Peking University campus)

Built and run the same session as G.7, this time from a genuinely cold start: the dataset was not on the server, had never been downloaded, and had no loader, taxonomy mapping, or sensor config anywhere in this project before this task began. Motivation: RELLIS-3D (off-road trail) and nuScenes (highway/street driving) are both vehicle-centric domains; SemanticPOSS is a close-range, pedestrian-and-cyclist-**dense** walking-campus environment — a third, genuinely different generalization test, not a repeat of either prior one.

### What SemanticPOSS actually is

Published by Peking University's POSS Lab ([arXiv:2002.09147](https://arxiv.org/abs/2002.09147), *"SemanticPOSS: A Point Cloud Dataset with Large Quantity of Dynamic Instances"*): 2,988 real LiDAR scans across 6 sequences, collected on PKU's own campus with a Hesai Pandar40P (40-beam, 360° rotating LiDAR) mounted on a vehicle driven through walkways, roads, and building courtyards. Its explicit reason for existing, stated in its own title: most driving datasets (SemanticKITTI included) are dominated by static scenes, with genuinely moving instances rare; a university campus has a much higher density of walking pedestrians and cyclists, which SemanticPOSS was built specifically to capture. Labels follow SemanticKITTI's own file format (uint32 per point, low 16 bits = class id) across 17 real classes.

### The download itself was not trivial — a real, worked-through problem

The official download page (`www.poss.pku.edu.cn/download.html`) was **unreachable from this session's own tool network** (`ECONNREFUSED`) but reachable from `drishti-gpu` directly — confirmed by testing both, not assumed. The real direct-download link (`.../OpenDataResource/SemanticPOSS/SemanticPOSS_dataset.zip`, 2,410,727,418 bytes) needed no account or license click-through, but **a plain full-file HTTP GET to it hangs indefinitely** — verified with 90+ second waits, zero bytes received, with and without a Referer header. Diagnosed by testing bounded HTTP Range requests instead: a 1KB range returned in 0.8s, a 50MB range in ~25s (~2MB/s) — the server (or a proxy in front of it) apparently cannot or will not stream an open-ended response, but serves bounded ranges reliably. Worked around with a purpose-written chunked downloader (50MB sequential Range requests, retried up to 3× each, looped until the full byte count from the server's own `Content-Range` header was reached) — completed cleanly, exact byte count matched, zero retries needed across the whole download.

### Real structural discoveries made before writing any loader code

- **Directory layout**: `sequences/XX/{velodyne,labels,tag,calib.txt,poses.txt,instances.txt}` — SemanticKITTI-compatible, as the paper claims, confirmed directly rather than assumed.
- **A real quirk RELLIS-3D does not have**: sequence 00's frame files start at `000000`, but sequences 01-05 all start at `000001` — confirmed by listing actual files, not assumed contiguous. This broke the obvious approach of reusing `perception.train.build_multi_sequence_splits` (which assumes a `range(n_frames)` starting at 0) — `perception/semanticposs_seg_dataset.py`'s own split function instead works from each sequence's real on-disk frame-id strings.
- **The `tag/*.tag` files are a red herring for this project**: the dataset's own `read_data.py` (shipped in the zip) uses them to scatter its variable-length per-frame point/label arrays into a fixed 40×1800 grid of its own construction. This project's `project_to_range_image` computes its own azimuth/elevation projection directly from each point's real xyz against DRISHTI's own resolution schedule — independent of any dataset-provided grid, the same stance already taken for RELLIS-3D and nuScenes. `tag` files are read by nothing in this codebase.
- **Points and labels are 1:1 aligned per frame** (confirmed directly: `points.shape[0] == labels.shape[0]` for every checked frame) — a real per-frame valid-return count (66,000-70,000-ish observed), not a fixed grid size.

### The sensor config — measured from real data, not the datasheet

`configs/sensor_pandar40p.yaml` deliberately does **not** use Hesai's published spec-sheet vertical FOV (-25°/+15°, 40° span). Instead, elevation angles were computed directly from real downloaded points (0.1st/99.9th percentile, sampled across sequences 00/01/02): the real observed span is **-15.5°/+7.3°, ~22.8°** — narrower than the datasheet's maximum mechanical range, because that real span is what actual ground-and-object returns populate, not the sensor's theoretical limit. `h_m` (mount height, 1.988m) was similarly computed from the median z of real ground-classified points (label id 22, "ground", per the dataset's own `read_data.py`) — the identical method already used to derive `sensor_ouster_os1_64.yaml`'s own `h_m`. `d_theta_deg` (0.2°, horizontal resolution) is the one figure taken from Hesai's spec sheet rather than independently re-measured, and is flagged as such in the config file's own comments — the same honesty convention `sensor_hdl32e.yaml` already uses for its own unverified `h_m` placeholder. `d_phi_rad` is a **uniform approximation** of the real measured span (39 gaps across 40 beams); the Pandar40P's true per-channel spacing is non-uniform (denser mid-FOV), documented as a simplification rather than presented as a measured per-beam table.

### Taxonomy mapping — one real gap found and left honest, not guessed at

`perception/taxonomy.py`'s `SEMANTICPOSS_ID_TO_NAME` is transcribed directly from the dataset's own `read_data.py` `LABEL_DICT`. One real discrepancy: **label id 20 was observed in real downloaded frame data but does not appear anywhere in `read_data.py`'s own dictionary** (which jumps 17→21). Left unmapped rather than guessed — an unrecognised id falls through to `UNKNOWN`, the same stance `rellis_label_ids_to_drishti` already takes. Two mapping decisions worth stating explicitly (matching Bible Part A.3's "state limitations" principle): "rider" (a person on a bike) was mapped to PEDESTRIAN rather than VEHICLE, since the entity to protect is a person, not because a rider behaves like a stationary pedestrian; "traffic sign 2/3" (hanging/high-hanging signs) are physically overhead obstacles — exactly OVERHANG's real-world referent — but no dataset label may ever map to classes 8/9 (`assert_taxonomy_valid` enforces this at import time, now checked against this mapping too), so they were mapped to STATIC_OBSTACLE, honestly under-representing their overhead nature rather than breaking that architectural boundary.

### Real training result — 10 epochs, complete, the strongest single-dataset mIoU this session

Fine-tuned from `checkpoints_multi_v2/best.pt` (same channel-expansion path as every other fine-tune this session), 2,540 train / 448 val frames (per-sequence 15%-tail split, same convention as RELLIS), run **alongside** both `_v3` and (briefly) the nuScenes fine-tune with zero GPU contention.

| Epoch | val mIoU | avg epoch time |
|---|---|---|
| 0 | 0.258 | 326.8s |
| 1 | 0.414 | 295.7s |
| 2 | 0.475 | 300.2s |
| 3 | 0.519 | 294.9s |
| 4 | 0.548 | 289.5s |
| 5 | 0.563 | 292.6s |
| 6 | 0.568 | 291.7s |
| 7 | 0.576 | 285.5s |
| 8 | 0.576 | 284.3s |
| **9 (best)** | **0.579** | 287.6s |

Real per-epoch time held at ~285-300s throughout (no slowdown from disk cache pressure or memory growth); total wall-clock ~48-49 minutes for all 10 epochs, matching the very first epoch's own ETA projection almost exactly. No overfitting signal — val mIoU rose monotonically epoch over epoch all the way to the final epoch, unlike the nuScenes fine-tune's mild epoch-7 dip.

**Final per-class IoU (epoch 9, best):**

| Class | IoU |
|---|---|
| UNKNOWN | 0.380 |
| DRIVABLE | 0.811 |
| CAUTION | n/a — zero ground-truth pixels (SemanticPOSS's campus paths have no mud/puddle-equivalent label) |
| NON_TRAVERSABLE | 0.327 |
| STATIC_OBSTACLE | 0.652 |
| VEGETATION | 0.677 |
| VEHICLE | 0.628 |
| PEDESTRIAN | **0.576** |
| NEGATIVE_OBSTACLE / OVERHANG | n/a — zero pixels, by permanent taxonomy design |

**Cross-dataset comparison, real numbers, same architecture and fine-tune recipe family:**

| Dataset | Best val mIoU |
|---|---|
| RELLIS-3D `_v2` (5 sequences, class-weighting only) | 0.574 |
| nuScenes-mini fine-tune | 0.334 |
| **SemanticPOSS fine-tune** | **0.579** |

**The one number worth calling out specifically**: PEDESTRIAN IoU (0.576) is the best of any dataset tested this session — consistent with SemanticPOSS being the one dataset purpose-built around dense pedestrian/cyclist instances, and a real, direct piece of evidence that domain match (not just raw frame count — SemanticPOSS's 2,540 training frames sit between nuScenes-mini's 324 and RELLIS's 11,522) drives per-class performance on the classes that domain actually contains.

### Files added this task

`configs/sensor_pandar40p.yaml` (new sensor config, real-measured values flagged above); `perception/taxonomy.py` additions (`SEMANTICPOSS_TO_DRISHTI`, `SEMANTICPOSS_ID_TO_NAME`, `semanticposs_label_ids_to_drishti`, and `build_nuscenes_lidarseg_lut` moved here from `eval/eval_nuscenes.py` as a shared function during the same taxonomy pass); `perception/semanticposs_loader.py` (Sweep loading, real frame-id-string-based, not index-based); `perception/semanticposs_seg_dataset.py` (dataset + per-sequence real-frame-id split); `perception/train_semanticposs.py` (standalone training script, same "don't touch `perception/train.py` while `_v3` is running" discipline as `train_nuscenes.py`).

## G.9 Joint multi-dataset training — `checkpoints_joint`, a real negative result

Built in response to a direct methodology question: instead of three separate single-dataset fine-tunes (G.7, G.8, and RELLIS `_v3`), does training on RELLIS-3D + nuScenes-mini + SemanticPOSS **together, in the same batches**, produce a model that's better across all three than any of the three dedicated fine-tunes was on its own domain?

**The real engineering problem this required solving**: the three datasets project through three different sensors at three different resolutions (RELLIS 64×2048, nuScenes 32×1080, SemanticPOSS 40×1800) — PyTorch's default batch collation cannot stack tensors of different shapes. Built `perception/joint_seg_dataset.py`: a `ConcatDataset` wrapper that resizes every domain's tensor to RELLIS's own native 64×2048 (the largest, richest resolution — upsampling the smaller sensors rather than downsampling RELLIS's real 64-beam detail away), **bilinear for the continuous input channels, nearest-neighbor for integer class labels and the boolean valid mask** (bilinear-interpolating a class ID would invent meaningless fractional intermediate "classes" — never done here). Validation deliberately stayed **three separate per-domain passes** at each domain's own native resolution, not one blended metric — a single averaged mIoU across three different class distributions and resolutions would hide exactly the kind of per-domain regression this experiment needed to be able to see.

**Real result, 10 epochs, ~1872-2630s/epoch (grew over the run as detection-head training started sharing the same CPU resources)**:

| Epoch | RELLIS | nuScenes | SemanticPOSS | Mean |
|---|---|---|---|---|
| 0 | 0.448 | 0.200 | 0.190 | 0.279 |
| 1 | 0.503 | 0.204 | 0.254 | 0.320 |
| **6 (best)** | — | — | — | **0.351** |
| 9 (final) | 0.558 | 0.213 | 0.257 | 0.342 |

**Honest headline: joint training underperformed every single domain's own dedicated fine-tune, on every domain.**

| Domain | Single-dataset fine-tune (best) | Joint training (best epoch) |
|---|---|---|
| RELLIS-3D | 0.571 (`_v3`, Part G.6) | 0.558 |
| nuScenes-mini | 0.334 (Part G.7) | ~0.213 |
| SemanticPOSS | 0.579 (Part G.8) | ~0.257 |

**Why, most likely**: the script's own docstring flagged this risk before training even started — RELLIS-3D contributes 80.1% of the combined 14,386-frame training set (nuScenes only 2.3%, SemanticPOSS 17.7%), and **no domain-balancing (e.g. oversampling the smaller domains) was applied**. The much smaller nuScenes/SemanticPOSS gradients get diluted by RELLIS's dominant sample count every single batch, on average. RELLIS itself came out only mildly worse (0.571→0.558, a real but small regression, consistent with it dominating the training signal and mostly getting to keep its own performance), while the two minority domains lost roughly a third to a half of their solo-fine-tune mIoU.

**What this genuinely establishes, stated plainly**: for this specific setup (three real datasets of very different sizes, no domain balancing, one shared BatchNorm), **naive joint training is not a free win** — it is a real, measured negative result against the alternative of separate fine-tunes, not a hypothetical concern. A future attempt at joint training would need real domain-balancing (oversampling nuScenes/SemanticPOSS, or a weighted sampler) before it could be expected to do better than training on each domain alone.

## G.10 The detection head — closing PS gap #2 ("identify and classify," not just dense segmentation)

Built in direct response to a named gap: the PS asks the system to "identify and classify" objects including "walls, poles" (static obstacles); dense per-pixel segmentation alone cannot answer "how many pedestrians were in that frame" — a judge's natural question.

**Real data used**: nuScenes-mini's own 3D box annotations — **18,538 real `sample_annotation` boxes across all 404 samples**, confirmed present and unused on the server before this session (only per-point lidarseg labels had been used until now). `perception/nuscenes_boxes.py` projects each box (global frame → sensor frame, via the exact same pose composition `perception/nuscenes_loader.py` already uses) into per-pixel objectness + regression targets (offset-to-center, box dims, sin/cos yaw), using the devkit's own tested `points_in_box()` rather than re-deriving projection math.

**Architecture**: `perception/detection_head.py`, a small 2-conv head sharing FusionSegNet's own decoder features (`FusionSegNet.forward(..., return_features=True)` — one additive, backward-compatible flag added to `perception/segnet.py`, verified bit-identical for all 14 existing callers). Loss deliberately simple (`perception/detection_loss.py`): weighted BCE for objectness (real measured pos_weight, not guessed) + masked smooth-L1 for regression — matching a deep-research report's own finding (run this session on lightweight range-view 3D detection literature) that architectural complexity (IoU losses, multi-resolution pyramids) isn't necessary for range-view detection when the shared backbone has enough capacity.

**Decode**: `perception/detection_decode.py` — deliberately **not** classical clustering on the semantic mask (the report's own analysis names why that fails: the "touching object" dilemma, where objects at different depths but the same azimuth merge into one blob). Instead, peak-extraction on the **learned** objectness map via max-pool NMS, the same family of approach CenterPoint uses for its own final decode step.

### Run 1 (`checkpoints_detection`, 15 epochs, flat LR — a real gap caught before v2)

Warm-started from `checkpoints_joint/best.pt`, backbone fine-tuned at 0.1× the head's LR. Val loss dropped cleanly (1.24 → 0.78, best at epoch 13). **A real bug found and fixed during first evaluation**: `perception/detection_decode.py`'s first version reported the segmentation head's raw class prediction at any objectness peak — including VEGETATION and DRIVABLE — as a "detected object" (one frame showed 58 "vegetation objects"). Fixed by filtering decode output to only `DETECTABLE_DRISHTI_CLASSES` (PEDESTRIAN, VEHICLE, STATIC_OBSTACLE) — nuScenes' own boxes never cover DRIVABLE/VEGETATION/UNKNOWN, so a peak landing there is always a false positive, never a real detection.

**Real decode evaluation, 80 held-out val frames, real object counts (`eval/checkpoint_detection_decode.py`)**: mean real object count 30.07/frame; at the default 0.5 objectness threshold, decoded **6.00/frame (~20% recall)**. Sweeping the threshold down to 0.1 (very permissive) only reached **10.40/frame (~35% recall)** — ruling out "just needs a lower confidence threshold" as the fix; this was genuine under-training, not a calibration issue.

**One real, separate gap found while diagnosing this**: `perception/train_detection.py`'s first version had **no learning-rate scheduler at all** — a flat LR the whole run, unlike every other training script in this project (`train.py`, `train_nuscenes.py`, `train_semanticposs.py`, `train_joint.py` all use `OneCycleLR`). Fixed by adding a per-param-group `OneCycleLR` (backbone and head keep their intended relative LRs throughout the schedule, not just at step 0).

### Run 2 (`checkpoints_detection_v2`, 40 epochs, OneCycleLR)

**Best val loss: 0.6925 at epoch 39** (vs. v1's 0.7785) — a real, clean improvement by the training metric, still improving at the final epoch, no plateau.

**But decode evaluation showed the opposite** — v2 decodes to *fewer* real objects than v1 at every threshold tested:

| Threshold | v1 decoded (worse val loss) | v2 decoded (better val loss) |
|---|---|---|
| 0.5 | 6.00 | **1.88** |
| 0.4 | 6.31 | 1.94 |
| 0.3 | 6.66 | 2.04 |
| 0.2 | 7.47 | 2.29 |
| 0.1 | 10.40 | 3.52 |

**The real, important, somewhat counter-intuitive finding: a lower per-pixel loss does not imply better decoded-instance recall.** Diagnosed rather than left as a mystery: v2's longer, fully-annealed OneCycleLR schedule produced a more *confident but more spatially concentrated* objectness map — fewer, sharper peaks the network is more certain about, which is exactly what BCE loss rewards, but it means fewer real objects ever produce a local maximum that survives the decode step's max-pool NMS. v1's less-converged, noisier map happened to scatter more local maxima across the image, incidentally overlapping more real objects even with worse overall calibration.

**Confirmed by a follow-up ablation, cheaply, with zero retraining**: reducing the decode step's NMS pool size from 5×5 to 3×3 roughly **doubled** v2's decoded count at every threshold (0.5: 1.88→4.17, 0.3: 2.04→4.66, 0.1: 3.52→8.04) — real, direct evidence that the 5×5 window was over-suppressing genuinely distinct nearby peaks in v2's sharper objectness map. Still short of v1's numbers at the same settings, but a meaningful, free recovery.

**Honest state of this component, right now**: neither checkpoint is a finished detection system. v1 decodes more real objects but from a less-confident, less-converged model; v2 trains "better" by the loss but decodes worse under the current NMS settings. The single highest-value next step, not yet done: re-run v1 through the same NMS pool-size sweep for a fully matched comparison, and/or add a loss term that directly penalizes a real box producing zero surviving peaks (rather than only rewarding per-pixel calibration) — named as the concrete next fix, not assumed unnecessary.

### Files added this session for the detection head

`perception/nuscenes_boxes.py`, `perception/detection_head.py`, `perception/detection_loss.py`, `perception/detection_decode.py`, `perception/nuscenes_detection_dataset.py`, `perception/train_detection.py`, `eval/checkpoint_detection_decode.py`; one additive flag (`return_features`) on `perception/segnet.py`'s `FusionSegNet.forward()`.

## G.11 Why the GPU sat idle — real diagnosis, and the frame-caching infrastructure built to fix it

While the above ran, a direct question was asked and answered with real measurements rather than assumption: **why is GPU utilization so low during training?** `nvidia-smi` showed **14% GPU utilization, 2.3GB/11.26GB memory used**, while `top` showed a **12.2 load average on this 10-core server** with two real training jobs running — one process alone pegged at 788% CPU.

**Real cause, confirmed, not guessed**: every dataset's per-frame preprocessing — spherical projection (`project_to_range_image`), the ground-prior column-wise walk, surface-geometry (normals/curvature), and (for the detection dataset) real 3D box-target projection — is pure NumPy/Python CPU work, recomputed from scratch on **every** `__getitem__` call, **every** epoch, even though none of it depends on which epoch is training. The GPU finishes its forward/backward pass on one batch quickly and then sits idle waiting for CPU workers to finish preparing the next one. This is the same root cause already named in Part H.3's latency breakdown (`load_and_assemble` ~447ms vs. `model_forward` ~385ms, even running one job alone) — worse here because two real jobs were sharing the same 10 CPU cores.

**Fix built**: `perception/frame_cache.py`, a generic `FrameCache` — caches the RAW (pre-normalization) channel stack + target/label arrays to disk as `.npz`, keyed by frame identity. Channel normalization stays *outside* the cache (a cheap per-call subtract/divide using whichever `ChannelStats` a given run supplies) — caching *after* normalization would silently bake one run's stats into the cache file, corrupting any future run with different stats.

**Correctness verified directly against real data, not assumed**: cached output confirmed bit-identical to uncached output, on both a cache miss (first write) and a cache hit (subsequent read), for both the segmentation and detection datasets, using real nuScenes frames. **Radial jitter augmentation required real care**: jitter must apply *after* the cache read (as a direct scale on the raw x/y/z/range channels via the new `apply_radial_jitter_to_raw()`), never baked into the cached file — otherwise the "random" jitter would repeat identically every epoch, silently defeating the augmentation. Ported to match the original `_apply_radial_jitter`'s exact math, verified against it.

**Disk-safety, not an afterthought**: this project's server sat at **31GB free / 92% used** when this was built, on a machine other work also runs on. `FrameCache` tracks bytes written this run and **raises `RuntimeError` rather than silently overrunning** once writes would exceed a configurable fraction (default 50%) of the free space measured when the cache was created — verified directly by forcing an artificially tiny budget and confirming it actually refuses.

**A real, stated, deliberate limitation**: RELLIS-3D is **never cached**, even when `--cache-dir` is passed to `perception/train_joint.py` — its full raw stack is an estimated **~78GB uncompressed** (11,522 frames × ~6.8MB/frame at 64×2048×13 channels), against 31GB actually free. Caching it would not be an optimization, it would be a disk-filling mistake on a shared server. Only `perception/train_nuscenes.py`, `perception/train_semanticposs.py`, and `perception/train_detection.py` (plus `train_joint.py`'s nuScenes/SemanticPOSS sub-datasets) got `--cache-dir` wiring; RELLIS's own `RellisSegDataset` in `perception/train.py` was deliberately left untouched.

**Real measured per-frame cache cost** (nuScenes, segmentation dataset): **~3.9MB/frame** — for the full 404-frame nuScenes set, ~1.6GB total, trivial. SemanticPOSS's full 2,988-frame set is estimated at ~24GB — large enough that the disk-safety check may legitimately refuse partway through on this server; expected behavior if hit, not a bug.

**Now measured, on a real 4-epoch nuScenes fine-tune** (`--cache-dir`, `checkpoints_multi_v2/best.pt` warm start): epoch 0 (cache-populating, pays full compute + write cost) took 76.9s; epochs 1-3 (cache hits) took 15.6s, 16.3s, 15.5s — a real **~4.8-5x speedup** for cached epochs versus the cache-miss epoch, measured within one controlled run rather than compared against a different run's numbers (system load and OS page-cache state differ across separate sessions, so a cross-run comparison would conflate caching with those confounds — the within-run before/after comparison is the clean one). This confirms `FrameCache`'s own designed break-even point (epoch 2) and expected effect size on real data, not just in principle.

### Files added this session for caching

`perception/frame_cache.py` (new); `perception/input_tensor.py` (additive `normalize_raw_channels()`, extracted from `assemble_input_tensor`'s existing tail, behavior-preserving refactor); `--cache-dir` wiring added to `perception/nuscenes_seg_dataset.py`, `perception/semanticposs_seg_dataset.py`, `perception/nuscenes_detection_dataset.py`, and the four `train_*.py` scripts.

## G.12 A training-free geometric detector — proposed with confidence, tested, and genuinely worse

After G.10's learned detection head showed a real, hard-to-resolve tension (v2 trained "better" by loss but decoded fewer real objects than v1), a deliberately different alternative was proposed: detect objects as **measured geometry in a fixed-size metric grid** rather than learned peaks in a range image — no training, confidence from the sensor's own physics (`sensor_model.n_expected`, the same formula the Sparsity Trap already uses for Claim 3) instead of a learned score. The pitch, made with real confidence beforehand: a metric grid structurally avoids the range image's "touching object" problem (two objects at different depths but the same azimuth are adjacent in a range image but not adjacent in real space), and it would let the project's own four claims do real work in detection, not just segmentation.

**Built**: `perception/geometric_instance_detector.py` — bins points into a fixed 0.3m local grid (not the persistent, toroidal `grid.clipmap.Clipmap` object, which is built for continuous multi-frame accumulation this single-frame use case doesn't need; stated explicitly in the module's own docstring so it is never mistaken for reusing that structure directly), takes a cell as a detection candidate only if it holds points BOTH classified into an instance-like class (PEDESTRIAN/VEHICLE/STATIC_OBSTACLE) AND elevated above `perception/ground_prior.py`'s own per-column ground estimate, connected-components (`scipy.ndimage.label`) over the candidate mask, and computes confidence as κ = observed points / `n_expected(range, height, width, sensor_config)` — a real, explainable, physics-derived number for every detection, with no learned score anywhere in the pipeline. `eval/checkpoint_geometric_detection.py` benchmarks it against the IDENTICAL 80 held-out nuScenes val frames and the identical real ground-truth object count already used for the learned head (Part G.10), for a genuine apples-to-apples comparison.

### Real result: substantially worse than the learned head, for two distinct, diagnosed reasons

| Approach | MAE (all classes, per-frame count) |
|---|---|
| Learned head v1 (threshold 0.5) | 28.20 |
| Learned head v2 (threshold 0.5) | 28.20 |
| **Geometric detector (`checkpoints_joint` backbone)** | **47.65** |
| **Geometric detector (`checkpoints_nuscenes_ft` backbone)** | **274.21** (worse, not better) |

**Failure 1 — the ground-truth comparison itself was mismatched for STATIC_OBSTACLE.** nuScenes' box-annotation protocol only puts 3D boxes around discrete movable/human/vehicle categories — it never annotates buildings, walls, or fences as countable objects. The geometric detector correctly finds real elevated static structures in the scene; nuScenes' own ground truth simply never counted them as "objects" to begin with. This was a flaw in the evaluation's own framing, caught only after running it, not anticipated beforehand.

**Failure 2 — restricted to the classes nuScenes actually boxes (PEDESTRIAN + VEHICLE), the detector finds almost nothing real.** Mean decoded: **0.55 objects/frame against 30.07 real (≈1.8% recall)**. This is not a threshold or grid-size problem — it means the elevated-and-classified candidate mask is failing for people and vehicles specifically, almost completely.

**Most likely root cause, tying back to an already-known weakness**: both the geometric detector and the learned head are downstream of the SAME segmentation network's per-point PEDESTRIAN/VEHICLE classification on nuScenes, and Part G.9 already measured that domain's mIoU as genuinely weak (0.213 joint-trained, 0.334 dedicated fine-tune — the best case among a set of not-great numbers). Switching to the stronger dedicated nuScenes checkpoint made results **worse, not better** (274.21 MAE), because that checkpoint also predicts STATIC_OBSTACLE more liberally across the same scene. The geometric approach has no learned regression to partially compensate for weak upstream classification the way the trained detection head's own end-to-end training does — it inherits the segmentation network's real weakness directly and without any mitigation.

**What the original pitch got wrong, stated plainly**: the metric-grid argument against the range-image's touching-object failure is still geometrically true, but the pitch didn't account for how leaky "elevated + classified as instance-like" is as a candidate filter in a real, cluttered urban scene — buildings, fences, and parked structures are genuinely elevated non-ground obstacles, and nothing in the original design excluded them from being counted as instances the way nuScenes' own benchmark protocol does. This gap was found only by running the real benchmark, not reasoned out in advance.

**Honest recommendation, not pursued further this session given cumulative cost**: this is not a replacement for the learned detection head in its current form. If revisited, the concrete next steps are (1) restrict candidate classes to PEDESTRIAN/VEHICLE only and diagnose the near-zero recall directly (ground-clearance threshold vs. genuine per-class segmentation recall on nuScenes), and (2) if static-structure detection is wanted at all, evaluate it against a ground truth that actually contains static-structure instances, not nuScenes' own box set.

### Files added this session for the geometric detector

`perception/geometric_instance_detector.py`, `eval/checkpoint_geometric_detection.py`.

## G.13 Following up on G.12 — RELLIS-3D retest, a failed geometric static-obstacle fallback, and a real Kalman tracker

Three further, directly targeted follow-ups to G.12's open questions, run in immediate succession.

### The RELLIS-3D retest — the core hypothesis holds

G.12 diagnosed the geometric detector's nuScenes failure as inherited segmentation weakness, not a flaw in the clustering/confidence mechanism — but never tested that claim against a domain with genuinely strong PEDESTRIAN/VEHICLE segmentation. `eval/checkpoint_geometric_detection_rellis.py` does exactly that: RELLIS-3D has no 3D box annotations, so the "real" reference count is a stated, honest proxy — the SAME clustering method applied to real ground-truth per-point labels rather than model predictions (weaker evidence than nuScenes' real boxes, flagged as such in the script's own docstring).

| Metric | nuScenes (G.12) | **RELLIS-3D** |
|---|---|---|
| MAE (all classes) | 47.65–274.21 | **4.03** |
| PEDESTRIAN | 0.55/30.07 (≈1.8%) | **154/133 (≈116%)** |
| VEHICLE | (included above) | **9/40 (≈22.5%)** |
| STATIC_OBSTACLE | dominant false-positive source | **152/3 (≈50× over-detection)** |

**The core hypothesis holds**: with strong segmentation (RELLIS `checkpoints_multi_v3`: PEDESTRIAN 0.771, VEHICLE 0.537), the clustering/confidence mechanism itself works — PEDESTRIAN recall lands in a genuinely usable range, MAE improves by roughly an order of magnitude. Two real, specific gaps remain, not smoothed over: **VEHICLE under-detection** (22.5% recall — plausibly the ground-clearance/footprint filter mishandling larger, partially-elevated vehicle bodies, not yet diagnosed further), and **STATIC_OBSTACLE massively over-firing even here** (152 vs. 3) — connecting directly to this project's single longest-standing unsolved problem.

### The geometric static-obstacle fallback — proposed with real prior evidence, tested, and it fails cleanly

STATIC_OBSTACLE's repeated over-firing (nuScenes AND RELLIS, learned head AND geometric detector) motivated a direct question: since Part G.2 already validated a real, measured curvature separation for this exact class (STATIC_OBSTACLE mean curvature 2.83 vs. DRIVABLE 0.20 — a real, 14× difference), could a simple geometry-only threshold — no learned classifier at all, the same "bypass the network, derive from geometry" pattern negative-obstacle detection already uses — serve as a usable fallback?

`eval/validate_geometric_static_obstacle.py` swept real candidate thresholds against real RELLIS-3D ground truth (60 sampled frames, 3,021,507 valid-geometry pixels, 1,818 real STATIC_OBSTACLE pixels — 0.0602% of the total):

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 1.5 | 0.0007 | 0.190 | 0.0014 |
| 2.5 | 0.0008 | 0.129 | 0.0015 |
| 4.0 (best F1) | 0.0009 | 0.094 | 0.0018 |

**Precision never exceeds 0.09% at any threshold tested — essentially random.** The reason the class-mean signal didn't translate into a usable rule: STATIC_OBSTACLE's real measured curvature has **std=5.59, nearly double its own mean of 2.83** — the per-pixel distributions overlap enormously even though the class means are real and well-separated, and the class is so rare (0.06% of pixels) that any threshold with meaningful recall floods with false positives from the other 99.94%. This is a real separability limit, not a tuning problem — no threshold value fixes it. **Not shipped, not pursued further**: this is the third confidently-proposed idea this session to fail real testing (the learned head's decode tension, the geometric instance detector on nuScenes, now this), reported with the same honesty as the other two. STATIC_OBSTACLE remains this project's single hardest unsolved problem across every approach tried — learned segmentation (0.0 IoU, three separate training runs), geometric clustering (order-of-magnitude over-detection on two datasets), and now geometry-only thresholding (near-random precision).

### The Kalman-filter tracker — closes Part E.2's descope, and this one actually works

Built `temporal/kalman_tracker.py`: a real constant-velocity Kalman filter (`[x, y, vx, vy]` state, standard predict/update equations) with Hungarian-algorithm (`scipy.optimize.linear_sum_assignment`) association between predicted track positions and new detections, and a standard SORT-family lifecycle (tentative → confirmed after 3 consecutive hits, deleted after 5 consecutive missed frames of coasting). This directly closes the gap Part E.2 names explicitly ("the tracker is simpler than the Bible's Kalman-filter design") and unlocks two things named elsewhere as "designed, not built": the fovea controller's `c_object` term (Part C.13) and persistent per-object IDs.

**Validated against 5 synthetic trajectory tests, all passing**: a single constant-velocity object keeps one consistent ID across 30 frames; the filter's own estimated velocity converges to within 0.3 m/s of the true value; two well-separated objects moving in different directions get and keep two distinct IDs (no identity swap) across 20 frames; a track is correctly deleted after exceeding its max-coast-frame budget with no detections; a single new detection is correctly held as tentative, not immediately reported as a confirmed track.

**Honest scope, stated in the module's own docstring rather than implied by "tested"**: this validates the tracker's own predict/associate/update/lifecycle logic against known-ground-truth synthetic trajectories — it has **not** been run against a real continuous multi-frame LiDAR sequence, because none of the three real datasets currently has a wired frame-to-frame detection pipeline feeding it consistent per-frame centroids (both `geometric_instance_detector.py` and `detection_decode.py` run per-frame, independently, with no continuity between calls yet). The tracker's own math is real and tested; real-sequence validation (noisy detections, missed frames, real ID-switch risk) is a genuinely separate, larger, not-yet-done step.

### Files added this round

`eval/checkpoint_geometric_detection_rellis.py`, `eval/validate_geometric_static_obstacle.py`, `temporal/kalman_tracker.py`, `tests/test_kalman_tracker.py`.

## G.14 Real fault-injection testing of the Conservatism Invariant — a genuine precondition violation found, not just a test result

`tests/test_conservatism.py` proves `planning.conservatism.cost()` is monotone under information loss across 10,000 Hypothesis-generated abstract `CellState` transitions (Part C.16) — a real, valuable proof that the cost function itself cannot be tricked by any state pair its own `_DEGRADATIONS` library can construct. What it cannot prove is that a REAL degraded sensor input actually produces the kind of state pair that library assumes. `eval/validate_conservatism_real_degradation.py` closes exactly that gap: real RELLIS-3D points, degraded with two real scenarios (azimuth sector dropout — simulating a hardware/occlusion failure — and Beer-Lambert-style range-proportional attenuation, simulating dust/rain), re-classified via the real `observability.sparsity.classify_sparsity`, fed through the real `cost()`.

**Result: 2 real violations found in 1,500 checked (frame, sector) cells across 20 real frames.**

1. **A real PEDESTRIAN cell (1,040 points, cost 200) had its sector fully dropped (0 points remaining) → reclassified as `SparsityVerdict.FREE`, `OBS_FREE`, cost 50.** `classify_sparsity`'s FREE verdict is correct on its own terms — it means "the sensor would have detected even the *smallest* object of concern here and detected nothing" — but it has no way to know a *large*, already-confirmed hazard occupied that exact cell a moment before. The verdict is calibrated for "was there ever a small hidden object," not "did something large that was just here disappear."
2. **A real pedestrian cell, thinned by Beer-Lambert attenuation (1,390 → 783 points), had its majority-vote class flip from PEDESTRIAN to UNKNOWN → cost dropped from 200 to 50**, despite `observability` staying `OCCUPIED` and `sparsity_verdict` staying `NORMAL` (there is unambiguously still something there).

**Checked whether the existing abstract property test could have caught this — it structurally cannot.** Its `_DEGRADATIONS` list (mark occluded, mark provisional, mark inferred, lower confidence, push past r_blind, age it, lose step-height/incidence knowledge) contains **no transform for "class_id: known hazard → None"** and **no transform for "observability: OCCUPIED → FREE via a fresh reclassification."** The abstract test's entire design implicitly assumes every `degraded` state it checks is a genuine monotonic degradation of the `cell` it started from — which is true for every hand-written function in that list, and false for what a real memory-less pipeline actually does each frame: throw away the prior state and reclassify from scratch.

**The real, precise finding**: the Conservatism Invariant is provably correct **given** a precondition — that the state handed to `cost()` is a true degradation of memory, never an independent fresh reclassification. **Nothing in this codebase currently enforces that precondition outside the abstract test's own hand-written degradation functions.** The architecture already has the right answer designed (`PROVISIONAL` / stale-confidence carry-forward, Bible Part 12, Layer 8 temporal fusion) — merging a fresh per-frame classification with the cell's own prior state via something like the existing Chan-merge machinery, so a real object's disappearance registers as `UNKNOWN`/`PROVISIONAL` rather than a confident `FREE`. What's missing is the **wiring**: nothing currently sits between "fresh single-frame classification" and "call `cost()`" to enforce merge-not-overwrite.

**Why this is not merely academic**: this project's own frontend/eval pipeline is currently single-sweep and memory-less (Part I's own stated limitation — temporal accumulation exists in the backend, Layers 4-8, but is not wired into the demo path). This means the exact failure mode found here is not a hypothetical edge case invented for a test — it is a real, currently-unmitigated gap in the pipeline as it is actually runnable today, not just a theoretical precondition violation.

**Fix built and confirmed**: `planning.conservatism.merge_with_prior(prior, fresh, dt_s, vehicle) -> CellState`, called on every real classification before it ever reaches `cost()`. Policy: if `cost(fresh) >= cost(prior)`, trust `fresh` outright (a genuinely new, worse hazard must take effect immediately); otherwise carry `prior` forward, aged by `dt_s` and flagged `provisional=True` — reusing the EXISTING age-based confidence-decay mechanism (`cost()`'s own one-directional age term, Part 12.3) rather than inventing a new one, so a hazard that's genuinely gone for good still correctly decays to `UNKNOWN_COST` over real elapsed time, while a one-frame dropout is correctly held over rather than instantly reported as clear.

**Verified two ways, not just asserted**: (1) a new 10,000-case Hypothesis property test (`test_merge_with_prior_never_lowers_cost_below_prior`) proves the general guarantee — `cost(merge_with_prior(prior, fresh, dt_s, vehicle)) >= cost(prior, vehicle)` for any prior/fresh pair — plus a direct regression test reproducing the exact real PEDESTRIAN-dropout scenario found above and confirming it's fixed. (2) `eval/validate_conservatism_real_degradation.py` now takes a `--use-merge` flag; re-run on the identical real data that found the original 3 violations, **without** the flag it reproduces the same 3 violations deterministically, **with** the flag it reports **0/1500** across both degradation scenarios. Full existing `tests/test_conservatism.py` suite (14 tests) passes with zero regressions (one pre-existing Hypothesis timing flake found and fixed along the way — a `deadline=None` addition, not a logic change).

### Files added/modified this round

`eval/validate_conservatism_real_degradation.py` (new, plus a `--use-merge` flag added after the fix); `planning/conservatism.py` (additive `merge_with_prior()`); `tests/test_conservatism.py` (3 new tests for `merge_with_prior`, plus a `deadline=None` fix to a pre-existing flaky test).

## G.17 VEHICLE's real recall failure, diagnosed — the Ground Paradox, again, within RELLIS-3D's own primary domain

Part G.13/G.16 established that VEHICLE under-detection (22.5% recall in the geometric detector) is a classification-recall problem upstream, not a clustering/under-segmentation one that box-splitting could fix. `eval/diagnose_vehicle_recall_by_range.py` measured real per-point VEHICLE recall from `checkpoints_multi_v3` directly, broken out by range, to find out *why*.

**Real result, 40 RELLIS-3D val frames, 1,769 real VEHICLE points**: overall recall 15.60% (lower than the geometric detector's own 22.5%, since this measures raw per-point classification before any clustering/aggregation smooths it). By range: [10,20)m 2.75%, [20,30)m 10.73%, [50,∞)m 61.54% (small sample, 221 points).

**The real, precise cause — the exact same failure mode already diagnosed for cross-domain nuScenes transfer (Part D.5), but happening *within* RELLIS-3D's own primary training domain**: **81.80% of all real VEHICLE points are misclassified as VEGETATION.** RELLIS-3D is heavily vegetated off-road terrain; VEHICLE is a comparatively rare class next to VEGETATION's dominance, and the network's strong learned prior toward VEGETATION apparently wins in ambiguous cases (a vehicle partially occluded by or adjacent to vegetation) rather than the network genuinely learning VEHICLE's own distinguishing geometry. This reframes the Ground Paradox from a purely cross-domain generalization problem into a **within-domain class-confusion problem driven by class imbalance**, present even on the dataset the network was trained on.

**Not yet attempted**: the same fixes that helped the cross-domain case (range-corrected reflectivity, Part G.1; real fine-tuning, Part G.7) were never specifically targeted at RELLIS's own internal VEHICLE-vs-VEGETATION confusion — a real, concrete next experiment this finding points to directly, not attempted in this round.

## G.18 Real `FrameCache` speedup — measured, not just designed

Part G.11 built and correctness-verified `FrameCache` but left its real speedup unmeasured. A real 4-epoch nuScenes fine-tune with `--cache-dir` closes that gap:

| Epoch | Time | Cache state |
|---|---|---|
| 0 | 76.9s | cache-populating (compute + write) |
| 1 | 15.6s | cache hit |
| 2 | 16.3s | cache hit |
| 3 | 15.5s | cache hit |

**Real measured speedup: ~4.8-5× for cache-hit epochs versus the cache-populating epoch** — measured within one controlled run (comparing against a different session's older numbers would conflate caching with differing system load/OS page-cache state, so the within-run comparison is the honest one). Confirms `FrameCache`'s own designed break-even point (epoch 2) and expected effect size hold on real data, not just in principle.

## G.19 Acting on G.17's diagnosis — an external report reviewed, then its two most defensible ideas built and retrained

Part G.17 found the real mechanism behind VEHICLE's weak recall: 81.80% of misclassified VEHICLE points land specifically on VEGETATION, worst at 10-30m (2.75%-10.73% recall) and recovering past 50m (61.54%). A user-supplied external "Technical Directive" report analyzed this same finding and proposed five interventions. It was reviewed critically before building anything (not accepted at face value): its two most speculative claims — that beam-divergence mixed-pixel boundary blending is the "conclusive" mechanism, and that 865nm reflectivity is categorically a dead end because mud/dust saturates it — were flagged as plausible but unproven (the network could simply be failing to exploit an existing channel under class-imbalance gradient starvation, not proof the signal is absent). Its most defensible, cheapest-to-falsify idea (a confusion-matrix-directed loss term) and its highest-value-but-costlier idea (targeted geometric copy-paste augmentation) were prioritized; its architectural-overhaul suggestions (E-CRF, SphereFormer) were discarded as disproportionate to where this project stands.

**Built, all three, real code (no simulation):**

1. **Composite Confusion-Aware Loss (CCAL)** — `perception/losses.py`'s `confusion_aware_penalty()` + `DrishtiSegLoss`'s new `use_confusion_aware` path. Maintains an EMA row-normalized confusion matrix (`update_confusion_ema`, called once per training step from `perception/train.py`'s training loop, AFTER the optimizer step so it reflects current weights) and up-weights a pixel's CE loss by `1 + kappa * confusion_ema[true_class, predicted_class]` — but ONLY for pixels currently predicted wrong (a correct prediction is never penalised extra just because its class has a high-confusion row elsewhere). Starts as a real no-op (confusion_ema initialised to all zeros — weight=1 for every pixel until real confusion data accrues), not an arbitrary cold-start bias. Off by default (`use_confusion_aware=False`); every existing caller of `DrishtiSegLoss` is byte-identical.

2. **VEHICLE copy-paste augmentation** — reuses the EXISTING class-4 CutMix mechanism (`perception/cutmix.py`'s `paste_rare_cluster`, already pastes real harvested point clusters into the raw sweep BEFORE `project_to_range_image`'s projection) rather than building a new PolarMix pipeline from scratch. Two real facts made this far cheaper than the report's own 12-18h estimate: (a) `eval/extract_rare_clusters.py` was already dataset/class-agnostic (`--target-class` was already a CLI flag) — harvesting VEHICLE clusters needed zero new code, just a different flag value (real harvest run: 2,151 clusters extracted from 3,688/11,522 real training frames containing class 6 points); (b) the report's own flagged hard part — ray-cast occlusion culling, so a pasted vehicle doesn't co-exist with the vegetation it should occlude — is **already a property of pasting pre-projection**, not new logic: `project_to_range_image`'s nearest-return-per-pixel z-buffering already drops any real point that a pasted point occludes along the same ray, for every existing CutMix paste, and inherits identically for VEHICLE. The only real new code was parameterising `paste_rare_cluster`'s placement range (previously hardcoded to class-4's 3-15m band) so the VEHICLE variant targets **[10, 30)m directly** — the exact band Part G.17 measured as the failure zone — via `perception/train.py`'s new `--vehicle-copypaste-clusters` flag, `AUG_VEHICLE_COPYPASTE_PROB=0.3`, independent of and composable with class-4 CutMix.

3. **kNN CRF post-processing** — new `perception/knn_crf.py`, `knn_crf_refine()`: mean-field Gaussian-spatial-kernel label smoothing over a real kD-tree neighborhood in Cartesian point space (not the range-image grid), stated honestly as an approximation of a real dense CRF (no learned pairwise compatibility weights exist in this project to fit one). Wired into `eval/diagnose_vehicle_recall_by_range.py` as a `--use-crf` flag so it can be tested standalone against the existing `checkpoints_multi_v3` checkpoint with zero retraining.

**Real retrain launched**: `checkpoints_multi_v4` on `drishti-gpu`, `perception/train.py` with `--use-ccal --vehicle-copypaste-clusters perception/rare_clusters_vehicle.npz`, VEHICLE clusters harvested fresh via `eval/extract_rare_clusters.py --target-class 6`. [Results pending — see below / next update.]

**Local verification before shipping**: all 21 pre-existing `tests/test_cutmix.py` + `tests/test_losses.py` tests still pass unmodified (no regression from the `paste_rare_cluster` signature extension or the new CCAL constructor kwargs). A standalone synthetic smoke test confirmed `DrishtiSegLoss(use_confusion_aware=True)` forward+backward+`update_confusion_ema` all run without error and populate a non-zero confusion matrix, and `knn_crf_refine` on synthetic Dirichlet-distributed probabilities preserves row-sum-to-1 and actually changes the input (not a silent no-op).

## G.15 Eigenvalue-based STATIC_OBSTACLE features — one failed, one inconclusive

Following G.13's curvature-threshold failure (precision never exceeded 0.09%), a genuinely different, higher-dimensional feature was tried: 3D structure-tensor eigenvalues (linearity, planarity — the standard normalized-eigenvalue point-cloud features), which theoretically should separate pole-like/wall-like STATIC_OBSTACLE structure from rough terrain far better than a single curvature scalar.

**First attempt (range-image 3x3 window) — failed cleanly, and the failure is itself informative.** `perception/eigenvalue_features.py` computed eigenvalues from a 3x3 range-image-adjacent window converted to 3D. Real result: STATIC_OBSTACLE mean linearity (0.9343) was statistically indistinguishable from non-STATIC (0.9347) — near-total collapse, worse separation than curvature had. **Diagnosed cause**: a range-image window, once converted to 3D, samples a wedge-shaped neighborhood (tiny spacing in elevation, spacing that grows with range in azimuth) — that shape is inherently near-linear for almost any real surface, dominated by the sensor's own angular sampling pattern rather than the true local surface shape. This is a real, structural mismatch between range-image adjacency and what eigenvalue features are designed to measure (an isotropic 3D neighborhood).

**Second attempt (real KD-tree 3D radius search) — inconclusive due to a genuine resource constraint, not disproven.** `eval/validate_kdtree_eigenvalue_static_obstacle.py` used `scipy.spatial.cKDTree` for a real isotropic 3D radius search (0.3m, matching the geometric detector's own cell size) — the neighborhood the eigenvalue literature actually assumes. Correctly used **inverse-probability-weighted precision/recall**: since STATIC_OBSTACLE is only ~0.05% of points, a stratified sample keeping all real static points while subsampling everything else would otherwise silently inflate precision relative to the true class balance — caught and fixed before running, not after. **Two consecutive runs (15 frames, then 10 frames with added explicit memory cleanup) both died silently partway through** (9/15, then 6/10 frames processed) with no error message — consistent with an OOM kill on the server's ~20GB RAM. This was judged a real, reproducible resource constraint on the per-point Python loop's memory footprint, not a cheap-to-fix bug, and not pursued further given the cumulative cost of debugging it. **This result is genuinely inconclusive, not negative** — unlike the two completed and cleanly-failed attempts (curvature, range-image-window eigenvalues), no final aggregated precision/recall numbers exist for the KD-tree version. A real fix (vectorized batch KD-tree queries, or a machine with more RAM) is named as the concrete next step, not attempted here.

**Net state of the STATIC_OBSTACLE problem after three geometry-only attempts this session**: curvature alone fails (near-random precision), range-image-window eigenvalues fail (worse than curvature, diagnosed cause), and real 3D eigenvalues remain untested to completion. Combined with the learned segmentation head's own 0.0 IoU across three training runs (Part G.6) and the geometric clustering approach's 50× over-detection on this exact class (Part G.13), **STATIC_OBSTACLE remains this project's single hardest unsolved problem, now across five separate attempted approaches** (three fully tested and failed, one genuinely worked around via a different mechanism entirely — Part G.12's clustering — and still failed there too, one inconclusive).

### Files added this round

`perception/eigenvalue_features.py`, `eval/validate_eigenvalue_static_obstacle.py`, `eval/validate_kdtree_eigenvalue_static_obstacle.py`.

## G.16 ALPINE-style recursive box-splitting — implemented correctly, negligible real-world benefit

The last item of a prioritized list built from an external technical report's recommendations: augment the geometric detector's connected-components clustering with ALPINE-style (Sautier et al.) recursive bounding-box splitting — when a cluster's principal-axis extent exceeds a real class-specific size prior, bisect along that axis at the median and recurse. Implemented in `perception/geometric_instance_detector.py` as `_recursive_split()`, using real PCA on each cluster's own points (never a learned split), with real physical size priors (PEDESTRIAN 1.0m, VEHICLE 5.5m) — **STATIC_OBSTACLE deliberately excluded**: its real problem this session (Part G.12/G.13) is wrong candidate generation entirely, not under-segmentation, and giving it a size prior would only fragment its already-dominant false positives further.

**Verified correct on synthetic data before testing on real data**: a dense, uniformly-sampled 9m-long blob (two touching 4.5m vehicles) correctly bisected into two ~4.5m sub-clusters centered at the right positions. (A first synthetic test using sparse Gaussian-scattered points produced a misleading result — many small disconnected islands from point sparsity, not from splitting — corrected before drawing any conclusion from it.)

**Real result on the same RELLIS-3D benchmark as Part G.13**:

| Metric | Before (connected components only) | After (+ box-splitting) |
|---|---|---|
| VEHICLE | 9/40 (22.5%) | 10/42 (≈23.8%) |
| PEDESTRIAN | 154/133 (≈116%) | 173/155 (≈112%) |
| STATIC_OBSTACLE | 152/3 (~50×) | 152/3 (unchanged, as designed) |

**Honest conclusion: negligible real-world benefit.** VEHICLE recall moved by ~1 percentage point — within noise. This is itself a real, informative finding: it means VEHICLE under-detection is **not primarily an under-segmentation problem** (multiple real vehicles merging into one oversized cluster that needs splitting) as the original report's reasoning assumed — it is more likely a **candidate-generation/classification recall problem further upstream**: the segmentation head simply isn't producing enough points classified as "elevated + VEHICLE" to form candidate clusters in the first place. Splitting a cluster only helps when a large-enough cluster already exists to split; if the real bottleneck is recall at the classification stage, no amount of downstream geometric post-processing can recover it.

### Summary of this whole external-report-driven round (G.13-G.16)

Of the four items pursued (RELLIS retest, geometric static-obstacle fallback, real fault-injection testing, box-splitting): **one succeeded outright** (the RELLIS retest, confirming the core clustering mechanism works given strong segmentation), **one found something more valuable than what was asked for** (the fault-injection test, which surfaced a genuine architectural precondition gap in the Conservatism Invariant rather than merely confirming it), and **two produced honest negative-or-negligible results** (the geometric static-obstacle fallback across three variants, and box-splitting) that nonetheless narrowed down *where* the real remaining problems are: STATIC_OBSTACLE's failure is in candidate generation, not feature separability; VEHICLE's failure is in upstream classification recall, not clustering.

### Files modified this round

No new files — box-splitting (`_recursive_split`, `_principal_axis_extent_m`, `MAX_EXTENT_M`) added directly to the existing `perception/geometric_instance_detector.py`.

---

# PART H — THE FRONTEND / DEMO DASHBOARD

Not part of the original Build Map — built because a live interactive 3D dashboard communicates the project far better than static plots for a hackathon pitch. Vite + React 19 + TypeScript + `@react-three/fiber` + `motion` (Framer Motion's successor) + Zustand + Tailwind v4.

## H.1 Wow-factor differentiators (backend Python + frontend TypeScript ports, each pair kept numerically identical by construction)

- **Semantic Friction Governor** (`planning/friction.py` / `frictionMath.ts`): derates the vehicle's declared `braking_a_ms2` per DRISHTI class using **cited terramechanics literature** (Wong 2001; Samuelraj et al. 2018; Salimi et al. 2015), not guessed numbers — DRIVABLE μ=0.40, VEGETATION μ=0.15, CAUTION (mud/puddle) μ=0.07. Cross-checks cleanly against `vehicle_ugv.yaml`'s own `braking_a_ms2=4.0`, which implies μ≈0.408 — right at the top of the cited dry-dirt/gravel range, independently. **Honesty caveat baked into the module**: the 10-class taxonomy already merges finer materials (mud/grass/sand all collapse into 3 tiers) — this cannot claim finer granularity than that.
- **A\* + kinodynamic smoothing** (Part C.14): real-time path replanning around a moving hazard, rendered as a curve colour-coded by the combined friction+curvature speed limit.
- **Explainability / attention overlay**: `perception/segnet.py`'s `AttentionGate` gained `return_attention=True`, exposing the network's own real Sigmoid attention activations — not a synthetic saliency method bolted on. Backward-compatible (default path verified bit-identical). Grounded in real policy, not just asserted: NATO's AI Strategy (2021) names "Explainability and Traceability" as a Principle of Responsible Use; the US DoD's 2026 ML System Safety Engineering Guidebook names "lack of explainability" as a certification hazard. **No equivalent public Indian/DRDO-specific mandate was found** — say "consistent with NATO/DoD principles," not "DRDO requires this."
- **ROS 2 bridge skeleton** (`ros2_bridge/drishti_bridge.py`): **explicitly labeled untested** in its own module docstring — no ROS 2 install exists in this dev environment, the point-cloud handler raises `NotImplementedError` with a documented sketch of the real pipeline it would call, rather than silently no-op'ing. Never present this as a tested integration.
- **Comparison wipe**: interactive drag-reveal between a naive flattened-2D-occupancy view (negative obstacles silently reclassified as drivable) and the real DRISHTI view — the concrete, demoable form of Bible Part C.17's "trench" demo beat.

## H.2 The real-data rendering pipeline (built this session, replacing the original all-synthetic mock demo)

`eval/export_frames.py` exports **real** RELLIS-3D points + real FusionSegNet predictions, binned into the real Nyquist resolution schedule (Part C.3), as zero-parse-overhead binary files. **Honest scope note in the script's own docstring**: each exported frame is a self-contained single-sweep snapshot, not a live temporally-accumulated clipmap replay — real geometry and real predictions, no ego-motion accumulation across frames.

Three new rendering primitives: `RealPointCloud` (custom shader-based point sprites, pre-allocated buffer, no per-frame GC churn), `RealTerrain` (one InstancedMesh **per real resolution level** — the first time all 4 adaptive-resolution tiers are visible simultaneously, from real data, directly answering the PS's own headline "variable resolution" ask), `VariableResGrid` (a single fullscreen-shader plane drawing distance-based grid lines using the real schedule — GPU-only, zero CPU cost per frame). Assembled in `RealScene.tsx` with `@react-three/postprocessing` bloom + ACES tone mapping, toggled in via a dedicated control-strip button, independent from the original mock-data timeline (different frame count, not yet unified).

## H.3 PS-literal compliance, checked directly against the actual problem statement text

- **"5cm within 10m, 50cm by 100m"**: the real, measured Ouster OS1-64 schedule gives **5cm out to ~16.3m** and **40cm by 100m** (not yet 130m, the schedule's coarsest tier) — exceeds the PS's own numeric ask in both directions, using the real sensor config, not a placeholder.
- **"Accuracy across varying distances"**: `eval/metrics.py`'s `compute_miou_by_distance_band()` (its own docstring literally cites this PS requirement) was implemented but never run against a real checkpoint until this session's `eval/checkpoint_accuracy_by_distance.py` closed that loop — real result (300 frames, real GPU): 0.0–12.8m band mIoU 0.478, 12.8–25.6m 0.506, 25.6–51.2m 0.363, 51.2m+ 0.156.
- **"Low latency (high FPS)"**: real, measured end-to-end pipeline latency on the actual GPU: P50=270ms, P95=360ms (200 real frames) — does *not* fit the 10Hz/20Hz sensor budget on this dev GPU. Reported honestly rather than a flattering cherry-pick: `model_forward` itself is only ~39ms of that 270ms; `load_and_assemble` (mostly disk I/O in this evaluation harness, not the model) dominates — a real production deployment would stream/cache frames rather than reload from disk per call, and that's the fix, not the model.
- **"Significant memory reduction vs. uniform 3D"**: already backed by a real artifact (`eval/pareto.py`'s checkpoint figure) — 12.58MB (v1) foveated clipmap vs. 201.3MB dense uniform 2.5D grid at the same extent, an honest **16×**, not the 267× dense-3D strawman ratio (Part C.19).

---

# PART I — HONEST LIMITATIONS (this build, on top of the Bible's own Part C.24)

**Hard physical limits, unchanged from the Bible**: a 5cm cable is undetectable beyond ~6.7m; negative obstacles are sampling-limited to ~20–30m; terrain beyond ~50m is barely sampled; 2.5D cannot represent genuinely multi-storey structure.

**This build's specific, additional gaps**:
1. **CPU vs. GPU numerical discrepancy, confirmed empirically**: identical checkpoint, identical data (sha256-verified), CPU inference (torch 2.11.0, local) gives systematically *worse* per-class results than the same run on the remote GPU (torch 2.0.1) — e.g. class 6 IoU 0.012 (CPU) vs. 0.528 (GPU) on the same checkpoint. **Always run trustworthy diagnostics on the remote GPU, never locally on CPU.**
2. **The train/val split may under-represent rare classes non-uniformly** — the class-4 prevalence discrepancy (0.051% train-sample vs. 0.0000875% val-sample, Part D.1) is real and not fully explained; a single run's val-split class-4 IoU should be read with real skepticism about sample size.
3. **The frontend runs on real exported data, but as a fixed, non-temporally-accumulated single-sweep snapshot per frame** — not a live, world-anchored, ego-motion-integrated clipmap replay. The genuine Layer 4–8 temporal machinery (Part C.8, C.12) exists in the backend but is not what the frontend currently visualizes.
4. **Multi-sweep accumulation is real, tested, and NOT used** in the current retrain or the current frontend export, by a stated, deliberate cost/benefit call (Part G.4) — not an oversight, but also not "done."
5. **Some citations in Part F (Rankin/Matthies exact venue, the RACER arXiv ID) were not independently re-verified** — confirm before external use (Part F.5).
6. **The gamma-slider frontend control computes locally in TypeScript** (a direct port of the real Python formulas), not by calling a live Python backend — a real FastAPI bridge was deliberately deferred multiple times ("build frontend first, don't touch backend").
7. **Stretch tickets #66/#67** (sparsity recovery rate vs. predicted $N_{exp}$ curve; outdriving fraction, Part C.18/C.15) are proposed in the Bible but not built.

---

# PART J — HOW TO RUN EVERYTHING

There is no live backend server — the "backend" is a Python pipeline run as scripts (training, eval, data export), not a long-running process.

```bash
# Backend setup (once)
pip install -r requirements.txt
pytest -q                                  # full test suite

# Regenerate the frontend's real data
python -m eval.export_frames --checkpoint checkpoints_multi_remote/checkpoint_epoch19.pt \
    --n-frames 24 --stride 4 --device cpu

# Frontend
cd frontend && npm install && npm run dev   # http://localhost:5173

# Remote GPU (training server)
ssh drishti-gpu "ps aux | grep perception.train | grep -v grep"   # check status
.\scripts\sync_training_results.ps1 -RemoteDir "checkpoints_multi_v3"   # live progress
.\scripts\export_from_gpu.ps1 -RemoteDir "checkpoints_multi_v3"         # pull everything down
```

The training server (`drishti-gpu` SSH alias) runs detached (`nohup`/`disown`) — closing your laptop does not stop it; only the server itself needs to stay powered and networked.

---

# PART K — GLOSSARY

**2.5D map** — a 2D grid where each cell stores height information; cheaper than 3D, richer than occupancy, inadequate in single-value form (Claim 1). **Foveation** — allocating detail non-uniformly, finest where it matters, by analogy with the eye. **Range image** — a LiDAR scan laid out as a 2D image indexed by (beam ring, azimuth): the sensor's native layout. **Clipmap** — a stack of nested power-of-two-resolution uniform grids centred on the viewer, from real-time graphics. **Toroidal addressing** — indexing a fixed array with wrap-around, so a moving window costs an index shift, not a copy. **Deskew** — correcting for vehicle motion *during* one LiDAR rotation. **Lovász-Softmax** — a convex surrogate for the Jaccard/IoU metric; optimises mIoU directly. **Negative obstacle** — a hazard below the ground plane (ditch/trench/crater), invisible to 2D occupancy grids. **Range shadow ($\Delta$)** — the extra distance a beam travels when the ground drops away, $\Delta=rd/h$. **Sparsity Trap** — a thin object at range returning too few points to distinguish from noise, silently reported as free space. **$N_{exp}/\kappa/r_{blind}$** — expected return count / observed-to-expected ratio / the range past which that count drops below one. **Time-to-contact (TTC)** — how long until the vehicle reaches a point at current closing speed; what resolution should track, in place of raw distance. **`PROVISIONAL`** — a cell inherited from a coarser level, not measured at this resolution; usable for coarse routing only, cleared on first real measurement. **`INFERRED`** — a cell filled by bounded, agreement-gated ground-plane continuation across a small gap; deterministic and auditable, forbidden from reducing cost. **Conservatism invariant** — the property that degrading a cell's evidence can never lower its reported cost. **Perception-limited speed** — the fastest speed at which stopping distance still fits inside detection range. **Ground Paradox** — a model trained where flat ground is almost always one material (e.g. grass) predicts that material for *any* flat ground, including a different one (e.g. asphalt) it has never seen flat before.

---

*This document explains the whole of DRISHTI as it actually stands: the theory it was built from, the real data it has been trained and measured against, the accuracy work validated and shipped this session, and the honest gaps still open. Four claims carry it: a single-value elevation map cannot represent an overhang, so multi-layer cells fix it; the resolution numbers in the statement are the sensor's own sampling limit, derived rather than chosen; "no returns" is not "no obstacle," and the boundary is computable from the datasheet; and detection range plus stopping distance together set an operational speed limit — the sensor, not the vehicle, is what caps how fast this thing may safely go.*
