# DRISHTI — Build Map (12-Day Solo Sprint)

**Companion to `DRISHTI_Project_Bible_v3.md`.** The Bible explains WHY. This tells you WHAT to build, in what order, what generated code will get wrong, and how to prove each piece works.

**Build model:** solo, AI-assisted. You specify and verify; a model writes most of the code. That inversion is why these tickets are shaped the way they are — see below.

**Target:** SIH 2026 internal round. Deliverable is a **PPT plus a short recorded demo video**, not a live interactive demo.

**Constraints as of writing:** Colab only, no local GPU. No CARLA unless lab access appears. nuScenes-mini available immediately; **one RELLIS-3D sequence** (off-road, 64-ch Ouster OS1 + 32-ch Velodyne Ultra Puck) downloading in background, replacing the earlier SemanticKITTI plan — see the dataset note under Ticket #3 for why.

---

## How To Read This Map

Tickets are in strict build order — `Blocked by` only ever points to a lower number, so working top to bottom never leaves you stuck.

| Tag | Meaning |
|---|---|
| 🟥 **CORE** | The spine, or a headline claim. If you build nothing else, build these. |
| 🟨 **SUPPORTING** | Real and demoed, but one panel or one number. Build after the spine works end to end. |
| 🟦 **STRETCH** | Only if you are ahead. Cutting these costs a metric, never a capability. |
| ⬜ **DETACHABLE** | Runs in parallel, blocks nothing, can fail without hurting the demo. |

### Why these tickets look different from a normal build map

You are not writing most of this code. That changes what a ticket is for.

A model will produce plausible, well-structured, confidently-wrong code for exactly the things this project depends on: **index arithmetic, coordinate conventions, sign errors, negative-number edge cases, and off-by-one in level nesting.** It will not tell you it got them wrong. Your tests are the only thing standing between that and a demo that lies.

So each ticket is:

- **Spec** — precise enough to paste as a prompt. Formulas, types, signatures, invariants. If a spec is vague, the generated code will be confidently vague too.
- **Watch out** — the specific way generated code fails *this* ticket. Not general advice.
- **Test** — concrete assertions with expected numbers. Where a number is known, it is written down so you can check without re-deriving.
- **Done when** — the pass condition.

**Golden rule:** do not start a ticket until its blockers pass their own Test. "Mostly works" silently poisons everything downstream, and with generated code you will not notice until the demo.

---

## The Two (Now Three) Sensor Configs

Everything is config-driven, and you run every dataset you actually have. This is not extra work — it is Ticket #59, the portability proof, obtained for free.

**Note on this table:** the HDL-64E column below is derived from Velodyne's published datasheet constants and stays here as the reference 64-beam-class case. It applies exactly **only if SemanticKITTI eventually lands** (it is no longer on the plan, but the loader interface still accepts it — see #3). The dataset actually feeding the background download is **RELLIS-3D**, sensed by a **64-channel Ouster OS1** and a **32-channel Velodyne Ultra Puck** — a different 64-beam sensor with its own $\Delta\theta/\Delta\phi$, not yet confirmed against a datasheet. Do not assume the HDL-64E numbers below transfer to it. Ticket #7 measures `h` from calibration; the Ouster equivalent of that ticket must also measure $\Delta\theta/\Delta\phi$ from the point cloud itself (via #6, the gate) rather than from an assumed spec, and a new row for the Ouster OS1 config gets added to this table once that's done.

| | HDL-64E (reference only, needs KITTI) | HDL-32E (nuScenes) |
|---|---|---|
| $\Delta\theta$ | 0.1728° = 3.016e-3 rad | 0.3333° = 5.818e-3 rad |
| $\Delta\phi$ | 0.4254° = 7.424e-3 rad | 1.3335° = 2.3275e-2 rad |
| $h$ | 1.73 m | **~1.84 m — verify from calibration (#7)** |
| 5 cm level reaches | 16.6 m | 8.6 m |
| 40 cm level reaches | 132.6 m | 68.8 m |
| 15 cm kerb | 20.2 m | 6.4 m |
| 2 m ditch | 21.6 m | 12.6 m |
| Pedestrian $r_\text{blind}$ | 194.8 m | 79.2 m |
| Safe speed (ditch) | 43.2 km/h | 32.0 km/h |
| Usable sensor range | ~120 m | **~70 m** |

**Two findings fall out of this table and both belong in the slides:**

1. **The PS's 5 cm at 10 m implies a 64-beam-class sensor.** On the HDL-32E, tangential spacing at 10 m is **5.82 cm** — larger than a 5 cm cell — so those cells physically cannot all be filled. Claim 2 inverts on this sensor, and saying so with a measured occupancy curve is stronger than the original claim.
2. **The sensor-derived 40 cm level reaches 68.8 m on the HDL-32E, and the sensor's usable range is ~70 m.** The schedule stops exactly where the sensor stops. Nobody designed that; it fell out of $\Delta\theta$.

**A third finding, once the Ouster OS1 config is measured, is the more important one for the pitch:** on RELLIS-3D the range ceiling is set by off-road terrain (vegetation, elevation change swallowing returns), not by a hardware floor the way nuScenes's 70 m is. That is the honest way to validate the far-range claims real data can support — say this explicitly rather than letting a judge assume RELLIS-3D "proves" 100 m the way a flat urban scene would.

---

## Day Allocation

Twelve working days. Slippage is expected — the buffer is real, do not spend it early.

| Day | Focus | Tickets |
|---|---|---|
| D1 | Foundations, sensor model, **the gate** | #1–#9 |
| D2 | Clipmap: addressing, scroll, lookup | #10–#16 |
| D3 | Cells, multi-layer, clearance | #17–#22 |
| D4 | Range image, channels, ground prior | #23–#27 |
| D5 | Network: extract, adapt, train | #28–#32 |
| D6 | Observability, negative obstacles | #33–#37 |
| D7 | Sparsity, speed envelope, conservatism | #38–#43 |
| D8 | Motion, fovea, traversability | #44–#49 |
| D9 | Hazard injection, dashboard | #50–#54 |
| D10 | Evidence: metrics, Pareto, portability | #55–#60 |
| D11 | Record, edit, cheap force-ins | #61–#64 |
| D12 | Slides, buffer, stretch | #65–#68 |

**Checkpoint discipline:** at the end of D3, D6 and D9 you should have something you could show. If you do not, cut a 🟨 rather than push the deadline.

---

## Track C / Track S

Hazard validation branches on whether you get CARLA:

- **Track S (default)** — synthetic hazard injection into real scans, tickets #50, #59.
- **Track C (if lab access)** — CARLA sweeps replace #59 and upgrade #50. Tickets marked `[Track C]`.

**Track S carries a caveat you must state:** injecting a trench using the same $\Delta = rd/h$ geometry the detector uses tests that your implementation inverts your own forward model — not that the physics is right. CARLA, being a different renderer, would test the physics. Say it before a judge finds it.

---

# Phase 0 — Foundations and The Gate (D1)

### #1: Repo skeleton and Colab environment
**Tier:** 🟥 CORE · **Day:** D1 · **Blocked by:** —

**Spec:** Package layout per Bible Part 28: `sensor/ perception/ grid/ observability/ temporal/ attention/ planning/ viz/ eval/ configs/ tests/`. Python 3.10+, `numpy scipy torch torchvision rerun-sdk pytest hypothesis matplotlib nuscenes-devkit pyyaml tqdm`. Pin versions in `requirements.txt` now. Add a `colab_setup.ipynb` that mounts Drive, installs, and runs `pytest -q`.

**Watch out:** Colab resets every session. Anything not in Drive or the repo is gone. Put the repo on GitHub on day one and clone it in Colab rather than editing notebooks in place — you will otherwise lose work to a disconnect.

**Test:** `python -c "import numpy, torch, rerun, nuscenes"` clean, and `pytest -q` collects zero tests without error.

**Done when:** a fresh Colab session reaches a green `pytest` in under three minutes from a single cell.

---

### #2: nuScenes-mini loader → canonical cloud
**Tier:** 🟥 CORE · **Day:** D1 · **Blocked by:** #1

**Spec:** One function returning a canonical struct for any dataset:

```python
@dataclass
class Sweep:
    xyz:        np.ndarray  # (N,3) float32, SENSOR frame
    intensity:  np.ndarray  # (N,)  float32, 0..1 normalised
    ring:       np.ndarray  # (N,)  int16, beam index; -1 if unavailable
    timestamp:  float       # seconds
    T_world:    np.ndarray  # (4,4) float64, sensor→world
    sensor_id:  str         # 'hdl32e' | 'hdl64e'
```

`load_nuscenes_sweep(nusc, sample_token) -> Sweep`. Read the ego pose and calibrated sensor from the devkit and compose `T_world = T_ego_world @ T_sensor_ego`.

**Watch out:** nuScenes stores LiDAR as `(x, y, z, intensity, ring_index)` float32 — five fields, not four. Generated code very often assumes four and silently misparses every point. Also: nuScenes intensity is 0–255, not 0–1. And the pose composition order is easy to invert — a sweep transformed by the inverse will look plausible but sit in the wrong place.

**Test:**
- Load 3 sweeps; assert `xyz.shape[1] == 3` and `N` is roughly 34,000 for nuScenes.
- Assert `ring` values span 0–31 exactly (32 beams).
- Transform a sweep to world and back with `inv(T_world)`; assert max residual < 1e-6 m.
- Plot one sweep in bird's-eye view and look at it. Ground should be a disc, not a cone or a smear.

**Done when:** three sweeps load, round-trip through world coordinates exactly, and look correct by eye.

---

### #3: RELLIS-3D download — background
**Tier:** ⬜ DETACHABLE · **Day:** D1, runs for hours · **Blocked by:** #1

**Spec:** Pull **one RELLIS-3D sequence** — `00004` — into Google Drive from the [official repo](https://github.com/unmannedlab/RELLIS-3D). Write `load_rellis_sweep()` returning the same `Sweep` struct as #2, with `sensor_id='ouster_os1_64'`.

**Correction, confirmed against the repo's actual download links:** the KITTI-format point clouds this loader needs are **not** published per-sequence — they ship as one combined Google Drive archive across all 5 sequences (14GB for Ouster OS1-64, 5.58GB for Velodyne Ultra Puck). Download that archive and extract only `00004/`, then discard the rest — `scripts/download_rellis.sh` in the repo does this. There is a separate, genuinely per-sequence "synced" **ROS bag** download for `00004` alone (~7GB) — that is a different format (rosbag, not `.bin`) and is not what this ticket downloads; do not substitute it without adding bag-extraction tooling first.

**Why RELLIS-3D and not SemanticKITTI:** three reasons, in order of weight. (1) It's **off-road** — Texas A&M campus terrain, not urban — which matches the DRDO domain framing directly instead of aspirationally; a judge who asks "did you test this off-road" gets a real answer. (2) It fixes what nuScenes structurally can't: nuScenes' HDL-32E tops out at ~70 m usable range, so it can never back a 100 m claim regardless of how the pipeline performs. RELLIS-3D's terrain also limits long-range returns, but for the honest reason (vegetation, elevation) rather than a hardware ceiling — see the caveat in the sensor-config table above. (3) Its class ontology has separate **puddle** and **deep water** labels, called out specifically for presenting "different traversability scenarios" — that's real ground truth for the incidence-corrected albedo work (Bible §9.5), which neither KITTI nor nuScenes offers at all.

**Sizing reality check — do not oversell this as "the small option":** RELLIS-3D is not small. The ROS-bag distribution across all 5 sequences runs 38–137 GB depending on format, comparable to or larger than SemanticKITTI. The KITTI-format distribution you actually need is smaller but still not per-sequence: 14GB (Ouster) or 5.58GB (Velodyne) combined across all 5 sequences, of which you keep only the `00004/` slice locally after extracting. Budget for the full 14GB download even though the kept result is a fraction of that — Colab/Drive has to hold the whole archive transiently to unzip it.

**Watch out:** Colab disconnects mid-transfer. Use a resumable transfer (`aria2c -c` or `wget -c`) and write directly to Drive, not to session storage. Do not block on this — everything geometric runs on nuScenes. Also: RELLIS-3D's Ouster OS1-64 is a **different 64-beam sensor** from SemanticKITTI's Velodyne HDL-64E — do not reuse the HDL-64E config constants for it. Its $\Delta\theta/\Delta\phi$ are not yet confirmed from a datasheet in this build map; they get measured directly from the point cloud via Ticket #6 (the gate), the same way #7 measures `h` from calibration rather than assuming it.

**Test:** `load_rellis_sweep()` returns a `Sweep` with `ring` spanning 0–63 for the Ouster OS1 frames (0–31 for the Velodyne Ultra Puck frames, if you also pull that sensor's stream — optional, not required). Labels align: `len(labels) == len(xyz)` for the same frame. Point cloud passes Ticket #6's gate against a provisional Ouster config before being trusted for anything else.

**Done when:** sequence `00004` is on Drive with labels, the loader returns the same struct type as #2, and Ticket #6 has been re-run against it to confirm the sensor constants. **If this has not landed by end of D5, the network trains on nuScenes-mini and you report the overfitting honestly.**

---

### #4: SensorModel — the five formulas
**Tier:** 🟥 CORE · **Day:** D1 · **Blocked by:** #1

**Spec:** `sensor/sensor_model.py`, config-driven from `configs/sensor_*.yaml` holding `d_theta_rad, d_phi_rad, phi_max_rad, h_m, n_beams, usable_range_m`, plus an optional `beam_elevations` table.

```python
def s_tangential(r, sm)   -> r * sm.d_theta
def s_radial_ground(r, sm)-> r**2 * sm.d_phi / sm.h
def s_vertical(r, sm)     -> r * sm.d_phi
def r_max_height(t, sm)   -> t / sm.d_phi                       # resolve height
def r_max_ditch(w, sm)    -> sqrt(w * sm.h / sm.d_phi)          # straddle a gap
def n_expected(r, t, w, sm) -> t*w / (r**2 * sm.d_phi * sm.d_theta)
def r_blind(t, w, sm)     -> sqrt(t*w / (sm.d_phi * sm.d_theta))
```

Pure functions of a config object. No globals, no module-level constants.

**Watch out:** degrees vs radians. A model will happily mix them and every number will be wrong by 57×. Store radians in the config, name the fields `*_rad`, and never accept a bare float. Also `r_max_height` and `n_expected` answer *different questions* (resolve height vs detect presence) — do not let them get collapsed into one helper.

**Test:** assert against the table at the top of this document, tolerance 0.1 m:

| Call (HDL-64E) | Expected |
|---|---|
| `s_tangential(10)` | 0.0302 m |
| `s_radial_ground(100)` | 42.9 m |
| `r_max_height(0.15)` | 20.2 m |
| `r_max_ditch(2.0)` | 21.6 m |
| `r_blind(1.7, 0.5)` | 194.8 m |
| `n_expected(100, 3.0, 0.2)` | 2.68 |

| Call (HDL-32E) | Expected |
|---|---|
| `r_max_height(0.15)` | 6.4 m |
| `r_max_ditch(2.0)` | 12.6 m |
| `r_blind(1.7, 0.5)` | 79.2 m |

**Done when:** both configs reproduce every number in the table above.

---

### #5: Resolution schedule generator
**Tier:** 🟥 CORE · **Day:** D1 · **Blocked by:** #4

**Spec:** `sensor/schedule.py`. Given a `SensorModel` and `c0 = 0.05`, emit levels where level $\ell$ has $c_\ell = c_0 2^\ell$ and sensor-Nyquist outer radius $r_\ell = c_\ell / \Delta\theta$. Return `[(level, cell_size_m, nyquist_radius_m)]`. Also emit `n_levels` needed to cover a requested extent.

**Watch out:** the Nyquist radius and the *array* extent are different things and generated code will conflate them. The array extent is `N * c_l` (Ticket #11); the Nyquist radius is what the sensor can fill. On KITTI the array is tighter; on nuScenes the array overshoots. Both are correct and the difference is a reportable result — do not "fix" it.

**Test:** HDL-64E → `[(0, 0.05, 16.58), (1, 0.10, 33.17), (2, 0.20, 66.33), (3, 0.40, 132.7)]` ±0.1. HDL-32E → `[(0, 0.05, 8.59), (1, 0.10, 17.19), (2, 0.20, 34.38), (3, 0.40, 68.75)]` ±0.1.

**Done when:** both schedules generate from config with no hardcoded radii anywhere in the file.

---

### #6: 🚨 THE GATE — point-distribution validation
**Tier:** 🟥 CORE · **Day:** D1 · **Blocked by:** #2, #4

**This is the most important ticket in the build. Nothing downstream is trustworthy until it passes.**

**Spec:** `eval/point_distribution.py`. Take one real sweep. Bin points by range in 5 m bins to 70 m. In each bin measure:
- median nearest-neighbour distance **within the same ring** → compare against `s_tangential(r)`
- median distance **between adjacent rings** on near-flat ground → compare against `s_radial_ground(r)`

Plot measured vs predicted curves on log axes, save to `eval/out/sensor_validation.png`.

**Watch out:** "nearest neighbour" over the whole cloud measures the wrong thing — it will find the neighbour in the adjacent ring, not the adjacent azimuth. **Group by `ring` first, then measure within-ring spacing.** This is the single most likely place for generated code to produce a beautiful plot of the wrong quantity. Also restrict the radial measurement to near-flat ground or terrain slope will dominate.

**Test:** measured within-ring spacing must track `r * d_theta` within ~20% across bins from 5–50 m. On nuScenes at 10 m expect **≈5.8 cm**; on KITTI **≈3.0 cm**.

**Done when:** the plot shows measured points sitting on the predicted curves. **If they do not, your sensor constants are wrong and every number in the Bible, the slides and the spec sheet is wrong with them. Stop and fix this before writing another line.**

**Re-run required for RELLIS-3D once #3 lands:** this is also how `configs/sensor_ouster_os1_64.yaml` gets its real $\Delta\theta/\Delta\phi$ — there is no verified datasheet value for those in this document, so this ticket doubles as the measurement, not just the check, for that config.

---

### #7: Verify mount height from calibration
**Tier:** 🟥 CORE · **Day:** D1 · **Blocked by:** #2

**Spec:** Read the actual sensor-to-ego translation from the nuScenes `calibrated_sensor` table; take the z component plus the ego height above ground. Cross-check by fitting a plane to ground points across ~50 sweeps and taking the offset. Write the measured value into `configs/sensor_hdl32e.yaml`.

**Watch out:** `h` in every ground formula is height **above the ground plane**, not above the ego origin or the rear axle. A model will grab whatever z it finds. Bible Part 3.6: a 0.5° tilt is a 44 cm phantom slope at 50 m — the same class of error applies to a wrong `h`, since `r_max_ditch ∝ √h`.

**Test:** the two estimates (calibration table, plane fit) agree within 10 cm. Ground points across many sweeps have mean z ≈ −h in sensor frame.

**Done when:** `h` in the config is a measured number with a comment saying how it was obtained, not a number from a datasheet.

---

### #8: Taxonomy remap
**Tier:** 🟥 CORE · **Day:** D1 · **Blocked by:** #2

**Spec:** `perception/taxonomy.py`. Explicit dict mapping nuScenes-lidarseg's 32 classes **and** RELLIS-3D's 20 (19 + void) into the 10 DRISHTI classes of Bible Part 6. Classes 8 (`NEGATIVE_OBSTACLE`) and 9 (`OVERHANG`) must have **no** dataset class mapping to them. (Leave the SemanticKITTI branch of the dict stubbed — same shape, filled in only if #3's KITTI fallback ever gets revived.)

**Watch out:** nuScenes lidarseg classes are `noise, animal, human.pedestrian.*, vehicle.*, flat.driveable_surface, flat.sidewalk, flat.terrain, static.manmade, static.vegetation, ...` — a much finer taxonomy than RELLIS-3D's. Write it out fully; do not let a model guess at prefix matching. RELLIS-3D's ontology derives from RUGD plus off-road-specific classes (`puddle`, `deep water`, `mud`, `rubble`, `log`, `pole`, `barrier`) — map `puddle`/`deep water` to a DRISHTI class you can actually validate against (`CAUTION` or `NON_TRAVERSABLE_TERRAIN`, not the unbuilt albedo mechanism), and do not let `deep water` accidentally map to `NEGATIVE_OBSTACLE` just because it's low-lying — it's a flat surface, not a range shadow, and conflating the two breaks the class-8 assertion below by target rather than by construction.

**Test:** every source class maps to exactly one DRISHTI class. **Assert no source class maps to 8 or 9** — this is the machine-checked form of the "network never invents a hazard" boundary. Assert both datasets' maps produce the same DRISHTI class set.

**Done when:** both mappings are complete, version-controlled, and the class-8/9 assertion passes.

---

### #9: Vehicle config
**Tier:** 🟥 CORE · **Day:** D1 · **Blocked by:** #1

**Spec:** `configs/vehicle_ugv.yaml` per Bible Part 14, plus the fields the later tickets need:

```yaml
max_slope_deg: 25
max_step_height_m: 0.20
min_clearance_m: 2.50
ground_clearance_m: 0.35
width_m: 2.10
max_roughness_m: 0.08
min_object_t_m: 1.00      # feeds r_blind
min_object_w_m: 0.10
braking_a_ms2: 4.0        # feeds speed envelope
t_react_s: 0.30           # includes measured P95 latency
```

**Watch out:** these are *declared platform parameters*, not tuned constants. Any code that hardcodes one of them is a bug — the whole "the map is vehicle-agnostic, the cost map is not" argument depends on this file being the only source.

**Test:** grep the repo for the literals `0.20`, `2.50`, `4.0` outside this file and the tests. There should be none.

**Done when:** the config loads and no vehicle parameter appears as a literal anywhere else.

---
# Phase 1 — The Clipmap (D2)

*This phase is the load-bearing wall. It is also where generated code fails most often and most silently. Budget the whole day and do not rush the tests.*

### #10: Addressing — world ↔ index, toroidal wrap
**Tier:** 🟥 CORE · **Day:** D2 · **Blocked by:** #5

**Spec:** `grid/addressing.py`. Pure functions, no state.

```python
def world_to_global(x, y, c_l)   -> (i, j)      # floor division, may be negative
def global_to_storage(i, j, N)   -> (si, sj)    # si = i & (N-1)
def flat_index(si, sj, N)        -> sj * N + si
def global_to_world(i, j, c_l)   -> (x, y)      # cell's lower-left corner
```

Assert `N & (N-1) == 0` at construction.

**Watch out — read this twice, it is the highest-risk paragraph in the document:**
1. **Use floor division, not truncation.** `int(-3.2)` is `-3`; `floor(-3.2)` is `-4`. Every point behind or left of the origin lands in the wrong cell if a model writes `int()` or C-style integer division. In NumPy use `np.floor_divide`; in Torch use `torch.div(..., rounding_mode='floor')`.
2. **The `& (N-1)` trick works for negative indices in two's complement** — `-4760 & 511 == 328` — and a model may "helpfully" replace it with `% N` or add a `if i < 0` branch. Python's `%` happens to agree here, but a rewrite to `abs()` or a C-style `%` does not. Keep the bitwise form and test negatives explicitly.
3. **Row-major order.** `flat = sj * N + si`, not `si * N + sj`. Getting this backwards produces a map that is transposed — which looks like a plausible map, just wrong, and you will not see it until something is mirrored in the demo.

**Test:**
- Exact worked example from Bible Part 8: `c_l=0.05, N=512, x=1012.00, y=-238.00` → `i=20240, j=-4760` → `si=16, sj=328` → `flat=167952`. Assert all four.
- Property test over 10,000 random `(x, y)` including negatives: `global_to_world(world_to_global(x,y))` lands inside the correct cell, i.e. `0 <= x - x_cell < c_l`.
- Assert `world_to_global(-3.2*c_l, 0, c_l)[0] == -4`, not `-3`.

**Done when:** all four assertions in the worked example pass, and the negative round-trip property test passes 10,000 cases.

---

### #11: Clipmap allocation, SoA planes
**Tier:** 🟥 CORE · **Day:** D2 · **Blocked by:** #10

**Spec:** `grid/clipmap.py`. Four levels, `N=512`, cell sizes from #5. **Structure-of-arrays** — one contiguous plane per field, not an array of records:

```python
h_min, h_max, h_mean : int16[L][N*N]   # 1cm fixed point
h_m2                 : uint16[L][N*N]
count                : uint16[L][N*N]
class_conf           : uint8[L][N*N]   # 4 bits class | 4 bits confidence
flags                : uint8[L][N*N]
stamp                : uint16[L][N*N]  # low bits of global (i,j), Ticket #14
```

Plus `origin_i[L], origin_j[L]` tracking the current window's lower-left global cell.

**Watch out:** a model will default to a struct/dataclass per cell or a single `(N,N,7)` array. Both break the hot/cold split in #55 and both are slower. Insist on separate named planes. Also: `int16` at 1 cm means values are `round(z*100)` — a model may store metres in an int16 and silently truncate everything to whole metres.

**Test:** assert total allocation is `4 * 512 * 512 * 12` bytes = **12,582,912 B (12.58 MB)** for the v1 layout. Assert each plane is C-contiguous. Assert encode/decode round-trips a height of 1.234 m to 1.23 m (1 cm quantum), not to 1.0 m.

**Done when:** allocation matches 12.58 MB exactly and the fixed-point round-trip preserves centimetres.

---

### #12: Nesting exactness — the proof
**Tier:** 🟥 CORE · **Day:** D2 · **Blocked by:** #10

**Spec:** No new production code. This ticket is a **test file** that proves the central claim of the project: a level-$(\ell{+}1)$ cell is exactly the union of four level-$\ell$ cells, at every position, for every ego pose.

**Watch out:** the property only holds because all levels index from the **same world origin** with power-of-two cell sizes. If anyone later adds a per-level origin offset "to centre the grid", this breaks and the whole alignment argument collapses. That is what this test exists to catch.

**Test:**
```python
@given(x=floats(-500, 500), y=floats(-500, 500), l=integers(0, 2))
def test_nesting_exact(x, y, l):
    i0, j0 = world_to_global(x, y, c(l))
    i1, j1 = world_to_global(x, y, c(l+1))
    assert i1 == i0 >> 1 and j1 == j0 >> 1
```
Run 10,000 cases including negative coordinates. Also assert the four children of `(I,J)` at level $\ell{+}1$ are exactly `(2I,2J), (2I+1,2J), (2I,2J+1), (2I+1,2J+1)` and that their union covers the parent with no gap or overlap.

**Done when:** 10,000 cases pass including negatives. **This test is your answer when a judge asks how you handle alignment error — show it running.**

---

### #13: Scroll and clear-on-scroll
**Tier:** 🟥 CORE · **Day:** D2 · **Blocked by:** #11

**Spec:** `Clipmap.scroll_to(ego_x, ego_y)`. Compute the new window origin per level; for each level, zero **only** the rows and columns scrolling in. Cost must be `O(N * (|di| + |dj|))`, never `O(N²)`.

**Watch out:** this is the single most dangerous bug in the project. Forget to clear and toroidal wrap silently returns data from 100 m *behind* the vehicle as though it were 100 m *ahead* — and it looks entirely plausible on screen. A model will often write a scroll that updates the origin and clears nothing, because clearing is not obviously necessary from the code's shape. Also: when `|di| >= N` the whole level must be cleared, and a naive modulo will clear the wrong strip.

**Test:**
- Fill L0 with a known pattern. Scroll by `(+30, 0)` cells. Assert every cell whose global index is outside the new window reads as cleared, and every retained cell still holds its original value.
- **Adversarial:** scroll by exactly `N` cells (a full wrap). Assert *nothing* stale survives — every cell must be cleared.
- Assert clear cost: instrument the number of cells written and assert it is `≤ N * (|di| + |dj|) + N*|di|*0` for small shifts, not `N²`.

**Done when:** the full-wrap test passes. That is the one that catches the silent bug.

---

### #14: Stamp validation
**Tier:** 🟥 CORE · **Day:** D2 · **Blocked by:** #13

**Spec:** Each cell stores `stamp = ((i & 0xFF) << 8) | (j & 0xFF)`. On read, recompute the expected stamp from the queried global `(i,j)`; on mismatch return `UNOBSERVED` rather than the stored contents. Expose a counter `stamp_mismatches` for the HUD.

**Watch out:** this is a *cross-check*, not a replacement for #13 — ship both. A model may propose replacing clear-on-scroll with stamps "for efficiency"; refuse, because stamps only catch reads, while stale cells also corrupt writes and accumulation.

**Test:** deliberately disable clear-on-scroll, scroll far, and assert `stamp_mismatches > 0` and that the reads return `UNOBSERVED`. Re-enable clearing and assert the counter stays at exactly **0** across a 200-frame drive.

**Done when:** the counter is zero for a full sequence with clearing on, and non-zero with it off.

---

### #15: `lookup()` — one query, one level
**Tier:** 🟥 CORE · **Day:** D2 · **Blocked by:** #14

**Spec:** `lookup(x, y) -> (level, CellView)`. Select the **finest level whose current window contains the point**. Bounds-check against the window; outside all levels return `UNOBSERVED`. This is the only public read path.

**Watch out:** the mipmap invariant is that levels are alternative views and are **never summed**. A model may write a helper that aggregates across levels for "completeness" — every point would then be counted four times, inflating occupancy and memory statistics. There must be no API that aggregates across levels. Total memory is the one legitimate sum, and it lives in `eval/baselines.py`, not here.

**Test:** for a point 11.8 m from the ego, assert `lookup` returns level 0. For 40 m, level 2. For 150 m, `UNOBSERVED`. Assert no function in `grid/` iterates over levels and accumulates a per-cell statistic.

**Done when:** level selection is correct at each ring boundary and the out-of-extent case returns `UNOBSERVED` rather than a wrapped cell.

---

### #16: Checkpoint — clipmap integrity drive
**Tier:** 🟥 CORE · **Day:** D2 · **Blocked by:** #15

**Spec:** No new code. Drive 200 consecutive nuScenes sweeps through scroll + lookup with the stamp counter displayed.

**Test:** counter reads 0 throughout. Memory does not grow. Scroll cost per frame stays bounded.

**Done when:** 200 frames, zero mismatches. **You now have a correct variable-resolution spatial structure — the hardest part of the project is behind you.**

---

# Phase 2 — Cells and the Overhang Claim (D3)

### #17: Cell encode/decode and foveated height quantum
**Tier:** 🟥 CORE · **Day:** D3 · **Blocked by:** #11

**Spec:** `grid/cell.py`. Fixed-point encode `z → int16` at 1 cm for L0. Per Bible Part 9.3, the height quantum scales with level: 1, 2, 4, 8 cm for L0–L3, stored relative to a per-tile base elevation at coarse levels. Provide `encode_h(z, level)` / `decode_h(v, level)`.

*(This is one of the two cheap force-ins. If D3 runs long, ship flat 1 cm int16 everywhere and do this on D11 — it is a byte-layout change, not an architectural one.)*

**Watch out:** a model will apply the quantum but forget the per-tile base, so coarse levels lose absolute elevation. Decode must be exact-inverse within half a quantum; test it rather than assume.

**Test:** round-trip 10,000 random heights per level; max error `≤ quantum/2`. Assert L3 saves 3 bytes/cell versus flat int16: `786,432 × 3 = 2.36 MB`, taking the map from **12.58 → 10.22 MB**.

**Done when:** round-trip is exact to the quantum and the memory saving matches 18.8%.

---

### #18: Scatter — the projection kernel
**Tier:** 🟥 CORE · **Day:** D3 · **Blocked by:** #10, #11

**Spec:** `grid/scatter.py`. Vectorised, no Python loop over points. Per level: compute `flat` per point (Ticket #10), then

```python
h_max.scatter_reduce_(0, flat, z, reduce='amax', include_self=True)
h_min.scatter_reduce_(0, flat, z, reduce='amin', include_self=True)
count.scatter_add_(0, flat, ones)
h_sum.scatter_add_(0, flat, z)
h_sq.scatter_add_(0, flat, z*z)
```

Mean from `h_sum/count`; variance from `h_sq`. Mipmap semantics: **every point scatters into its native level and all coarser levels.**

**Watch out:**
1. **A Python loop over 34k–120k points is fatal** and a model will write one if the spec is vague. It must be a single vectorised call per level.
2. `include_self=True` matters — with `False`, the first point into an empty cell is discarded.
3. `count` is `uint16` and must **saturate**, not wrap. A dense near cell can exceed 65,535 across accumulated frames; wrapping makes your best-observed cells report near-zero confidence.
4. `scatter_reduce_` on an empty-initialised `h_max` needs sentinel init (`-inf` equivalent in fixed point), otherwise zeros win the max.

**Test:**
- Nine points in a known 3×3 cell pattern; assert every reduced field by hand.
- Compare the vectorised path against a 10-line NumPy reference loop on 5,000 random points; assert bit-identical `count` and `h_max`.
- Time it on one real sweep: **< 20 ms on CPU** for 34k points. If it is seconds, there is a loop.

**Done when:** matches the reference implementation exactly and runs in tens of milliseconds.

---

### #19: Class aggregation by mode
**Tier:** 🟥 CORE · **Day:** D3 · **Blocked by:** #18

**Spec:** Scatter a per-class count into `(n_cells, 10)`, take argmax as the cell class, and store the runner-up fraction in the confidence nibble.

**Watch out:** **never average class IDs.** Class 2 and class 6 average to 4, an unrelated class. A model will reach for `mean` because every other field uses it. Also the histogram is `(n_cells, 10)` int16 — at L0 that is 512×512×10×2 = 5.2 MB of scratch; allocate once and reuse, do not build it per frame.

**Test:** a cell with 7 points of class 1 and 3 of class 4 reports class 1 with runner-up fraction 0.3. Assert no `mean`/`average` appears anywhere in the class path.

**Done when:** mode aggregation is correct and the runner-up fraction is exposed.

---

### #20: Height histogram
**Tier:** 🟥 CORE · **Day:** D3 · **Blocked by:** #18

**Spec:** 8 bins per cell over `[z_ground − 0.5, z_ground + 4.0]` m, bin width 0.5625 m, packed into 8 bytes (one uint8 count per bin, saturating).

**Watch out:** bin width must derive from `min_clearance_m` in the vehicle config (#9), not be a literal. The histogram is relative to *local* ground, not to z=0 — on a slope an absolute-z histogram puts everything in one bin.

**Test:** a cell with points at z = 0.0 and z = 4.2 produces occupied bins at index 0 and index 7 with the six between empty.

**Done when:** the two-surface case produces the expected bin occupancy on sloped ground as well as flat.

---

### #21: Multi-layer extraction — Claim 1
**Tier:** 🟥 CORE · **Day:** D3 · **Blocked by:** #20

**Spec:** `grid/layers.py`. From a cell's histogram: ground layer = lowest occupied bin and contiguous occupied neighbours; scan up for the first empty run of length `≥ ceil(min_clearance / bin_width)`; ceiling = first occupied bin above that run, else `NO_CEILING`. Emit `clearance = z_ceil_min − z_ground_max`, `+inf` when no ceiling.

**Watch out:** the empty-run search must require *contiguous* empty bins; a model may accept any gap. Also `NO_CEILING` must be a sentinel that propagates as `+inf` clearance — if it decodes as 0 the cell reports zero clearance and everything becomes `LETHAL`.

**Test — these four cases are the machine-checked form of Claim 1:**

| Case | Points | Expected |
|---|---|---|
| Road under bridge | z ∈ [0.00,0.05] and [4.2,4.6] | `DRIVABLE`, clearance **4.15 m** |
| Low branch | z ∈ [0.00,0.02] and [1.90,2.30] | `OVERHANG`, clearance **1.88 m**, non-traversable at 2.5 m required |
| Flat ground | z ∈ [0.00,0.03] | clearance `+inf` |
| Sparse canopy | scattered z ∈ [2.5,3.5], low density | not a ceiling — vegetation, per Bible Part 9 |

**Done when:** all four pass. Also assert that a max-height-only implementation gets case 1 wrong — **that assertion is your ablation and your slide.**

---

### #22: Checkpoint — first real map
**Tier:** 🟥 CORE · **Day:** D3 · **Blocked by:** #21

**Spec:** Run one nuScenes sweep end to end: load → scatter → cells → layers. Dump a bird's-eye PNG coloured by height and a second coloured by class (from ground-truth lidarseg labels for now).

**Test:** the map looks like the scene. Ground is continuous, buildings are tall, the coarsening at ring boundaries is visible. Print per-level occupancy — **on nuScenes, L0 occupancy should visibly collapse beyond ~8.6 m, which is the measured evidence for the "5 cm at 10 m implies a 64-beam sensor" finding.**

**Done when:** you have a picture of a real 2.5D foveated map. Save it — this is the first slide.

---
# Phase 3 — Perception (D4–D5)

### #23: Spherical range-image projection
**Tier:** 🟥 CORE · **Day:** D4 · **Blocked by:** #2

**Spec:** `perception/range_image.py`. Project a `Sweep` to `(H, W)` — `32×1080` for nuScenes, `64×2048` for KITTI, both from config.

$$u = \left\lfloor \tfrac{1}{2}\Big[1 - \tfrac{\text{atan2}(y,x)}{\pi}\Big] W \right\rfloor, \qquad v = \left\lfloor \Big[1 - \tfrac{\arcsin(z/r) - \phi_{\min}}{\Delta\phi_{\text{fov}}}\Big] H \right\rfloor$$

Keep the **nearest** return per pixel. Emit base channels: `x, y, z, range, intensity, valid_mask`.

**Watch out:**
1. When `ring` is available (both datasets have it) **use it directly for `v`** instead of computing from `arcsin`. The arcsin form assumes uniform beam spacing, which is false for both sensors, and produces a subtly warped image where a few rings collide and others are empty.
2. `valid_mask` must be 0 for pixels with no return, and **every downstream consumer including the loss must respect it.** Without it the network learns to predict classes for returns that do not exist.
3. `atan2` argument order is `(y, x)`. Reversed, the image is mirrored — plausible-looking and wrong.

**Test:** project and unproject; assert ≥99% of points land back within one cell of their origin. Assert `valid_mask.sum()` is close to `N` (few collisions at 32 beams). Assert `v` computed from `ring` matches the ring index exactly.

**Done when:** round-trip is lossless for the kept returns, and the image displays as a recognisable panorama.

---

### #24: Occlusion depth channels
**Tier:** 🟥 CORE · **Day:** D4 · **Blocked by:** #23

**Spec:** During projection, count discarded returns per pixel (`occlusion_count`) and record `occlusion_spread = max(discarded_range) − kept_range`. Two extra channels.

**Watch out:** this is free information you are already computing — do not let it become a second pass over the cloud. Count during the same scatter that selects the nearest return. Note that at 32 beams collisions are rarer than at 64; if counts are near-zero everywhere on nuScenes, that is correct, not a bug — it will matter more on KITTI.

**Test:** construct a synthetic sweep with two returns in one angular bin 20 m apart; assert `count == 1` and `spread ≈ 20`. Assert the channel is non-zero at object silhouettes in a real sweep and near-zero on open ground.

**Done when:** both channels populate and the spread channel visibly highlights silhouette edges.

---

### #25: Circular padding utility
**Tier:** 🟨 SUPPORTING · **Day:** D4 · **Blocked by:** #23

**Spec:** Replace **all horizontal** padding in the network with `F.pad(..., mode='circular')`. Vertical padding stays zero — the top and bottom of the image are real FOV boundaries.

**Watch out:** a model will apply circular padding in both dimensions because it is one flag. That wraps the ground onto the sky. Horizontal only.

**Test:** place a synthetic object straddling the 359°/0° seam; run one forward pass with and without; assert the feature response is continuous across the wrap in one case and discontinuous in the other.

**Done when:** the seam response is continuous and vertical padding is confirmed zero.

---

### #26: Ground prior — column-wise walk
**Tier:** 🟥 CORE · **Day:** D4 · **Blocked by:** #23

**Spec:** `perception/ground_prior.py`. Per azimuth column, walk outward from the sensor; accept a point as ground when the slope from the previous accepted ground point satisfies `|Δz / Δxy| < tan(10°)`. Output a per-point boolean **and** a per-column local ground height, which #34 and #36 both consume.

**Watch out:** **this LABELS, it does not STRIP.** Bible Part 5.2 — if ground points are removed before projection, terrain never reaches the elevation map and the deliverable is missing. A model given "ground filter" will write a filter. Say "classifier" in the prompt. Also: walk outward in **range order within a column**, not in ring-index order, and never fit a plane — a plane assumes flatness over the sector and fails on exactly the terrain that matters.

**Test:**
- Synthetic flat ground → 100% ground.
- Synthetic constant 8° slope → still ~100% ground (this is the test that distinguishes the walk from a plane fit).
- Synthetic 30 cm step → points above the step are not ground.
- **Regression:** run the full pipeline and assert the output map contains terrain cells with valid elevation. This is the automated form of the label-don't-strip bug and belongs in CI.

**Done when:** the 8° slope case passes and the terrain-reaches-the-map regression test is green.

---

### #27: Assemble the input tensor
**Tier:** 🟥 CORE · **Day:** D4 · **Blocked by:** #24, #26

**Spec:** Stack 9 channels: `x, y, z, range, intensity, valid_mask, ground_prior, occlusion_count, occlusion_spread`. Normalise each to roughly zero-mean/unit-variance using statistics computed over the training split and **saved to disk** — not recomputed per batch.

*(Motion residual channels are cut from this build; the tensor is sized 9, not 14. Adding them later is additive.)*

**Watch out:** per-batch normalisation makes training and inference statistics differ, which degrades silently. Save the stats. Also `valid_mask` must not be normalised — it is a mask, keep it 0/1.

**Test:** assert normalised channels have mean ≈0 std ≈1 on a held-out batch, except `valid_mask` which stays binary. Assert the saved stats file is loaded at inference, not recomputed.

**Done when:** the tensor assembles with saved normalisation and shape `(9, H, W)`.

---

### #28: Extract the FusionSegNet architecture
**Tier:** 🟥 CORE · **Day:** D5 · **Blocked by:** #1

**Spec:** Lift the model-definition cells out of `FusionSegNet_v5.ipynb` into `perception/segnet.py` as importable modules: EfficientNet-B0 encoder → ASPP bottleneck → attention-gated U-Net decoder with SE blocks → main head + 1/8 auxiliary head. Change the first conv from 3 to **9** input channels. Nothing else.

**Watch out:** the repo is a **camera** model trained on nuScenes images — there are no reusable weights and no LiDAR data pipeline. Only the architecture transfers. Do not import anything touching cameras, HD maps, IPM, or pseudo-labels. And when you present this, the honest line is *"we adapted the architecture from our earlier camera work"*, not *"we reused our model"* — a judge who opens the repo will see the difference.

**Test:** forward pass on a `(1, 9, 32, 1080)` tensor returns main logits `(1, 10, 32, 1080)` and aux logits at 1/8 resolution. Parameter count ≈4–4.5 M. No import errors from removed camera code.

**Done when:** the architecture runs standalone on a range-image tensor with the right output shapes.

---

### #29: Loss stack
**Tier:** 🟥 CORE · **Day:** D5 · **Blocked by:** #28

**Spec:** `perception/losses.py`. Lovász-Softmax (primary) + confidence-weighted cross-entropy (secondary, weighted by point-density confidence) + deep-supervision aux term at 1/8. **Every term masked by `valid_mask`.** Drop the temporal-consistency loss from the original stack — it assumes stable pixel correspondence that a rotating LiDAR does not provide.

**Watch out:** a model may reach for Dice — insufficient at LiDAR's class imbalance, which is why the Bible specifies Lovász. And masking must be applied *inside* each loss, not by zeroing the input, or the empty pixels still contribute gradient through the softmax denominator.

**Test:** assert the loss is bit-identical when invalid pixels are filled with garbage versus zeros. If it is not, the mask is not applied everywhere. Assert Lovász on a perfect prediction is ≈0.

**Done when:** the garbage-invariance test passes.

---

### #30: Train
**Tier:** 🟥 CORE · **Day:** D5 (GPU hours) · **Blocked by:** #27, #29, and ideally #3

**Spec:** Colab T4. Train on **RELLIS-3D sequence `00004`** if #3 landed; otherwise nuScenes-mini. AdamW, OneCycleLR, mixed precision. Checkpoint to Drive every epoch. Validate on a held-out split of the same sequence (RELLIS-3D ships one continuous route per sequence — carve off the last ~15% by frame index for val, do not shuffle-split a continuous drive or you leak near-duplicate adjacent frames into val).

**Watch out:**
- **Colab disconnects.** Checkpoint every epoch to Drive and make training resumable from the last checkpoint. Do not run a 6-hour job that loses everything at hour 5.
- **If you are on nuScenes-mini you have ~404 annotated keyframes.** It will overfit. Report train and val mIoU both, say the number of samples, and do not pretend otherwise — an honestly-labelled weak result costs you far less than a strong-looking one a judge probes.
- Class imbalance: check that rare classes are not collapsing to zero IoU before burning hours.

**Test:** val mIoU meaningfully above a majority-class baseline. Per-class IoU printed, not just the mean. Training resumes correctly from a killed session.

**Done when:** a checkpoint exists on Drive with recorded per-class val IoU and a stated training-set size.

---

### #31: Cache inference for the demo
**Tier:** 🟥 CORE · **Day:** D5 · **Blocked by:** #30

**Spec:** Run the trained model over the demo sequence on Colab GPU, save per-point labels to `.npy` per frame. The demo replays these; nothing runs the network live.

**Watch out:** CPU inference at `32×1080` is ~100–300 ms/frame — too slow for the video and irrelevant anyway, since the *mapping pipeline* is the contribution. But **be explicit in the video and slides about which labels the live view is using**, and report measured network latency separately as a number. Presenting cached labels as live inference is exactly the overclaim that loses credibility.

**Test:** cached labels align frame-for-frame with sweeps; `len(labels) == len(points)` for every frame.

**Done when:** the demo sequence has cached labels and a README line stating they are precomputed.

---

### #32: Checkpoint — semantically coloured map
**Tier:** 🟥 CORE · **Day:** D5 · **Blocked by:** #31, #22

**Test:** the 2.5D map coloured by predicted class looks right — road is road, buildings are obstacles. Compare side by side against the ground-truth-label version from #22. Differences should be plausible errors, not structural nonsense.

**Done when:** predicted-label and ground-truth-label maps are visibly comparable. **PS requirements for segmentation and the 2.5D grid are now both demonstrably met.**

---

# Phase 4 — Observability and Negative Obstacles (D6)

### #33: DDA ray traversal
**Tier:** 🟥 CORE · **Day:** D6 · **Blocked by:** #15

**Spec:** `observability/raycast.py`. 2D DDA (Amanatides–Woo) along each beam's ground projection, from sensor to return. Traverse at the **coarsest** level; refine to L0 only inside the fine ring.

**Watch out:** traversing every beam at L0 across the full extent is millions of cell visits per frame and will destroy the latency budget — a model will write the simple version. Coarse-first is mandatory. Also DDA implementations commonly have an off-by-one at the terminating cell: the cell containing the return is `OCCUPIED`, not `FREE`.

**Test:** a single ray across a known grid visits exactly the expected cell sequence (hand-computed for a 45° ray). The terminal cell is `OCCUPIED`. Cells beyond the return are untouched, not `FREE`.

**Done when:** the hand-computed ray matches and timing is within budget for a full sweep.

---

### #34: Four-state observability and carving
**Tier:** 🟥 CORE · **Day:** D6 · **Blocked by:** #33

**Spec:** Flags `UNOBSERVED | FREE | OCCUPIED | OCCLUDED`. Beams passing through → `FREE`; terminating → `OCCUPIED`; behind a termination → `OCCLUDED`; never touched → `UNOBSERVED`. Free-space carving decays occupancy where beams now pass through.

**Watch out:** **`UNOBSERVED` and `FREE` must never collapse.** A model will default the flag byte to 0 and treat 0 as free — which means a hole in the ground you never looked at reports as flat clear ground at z=0. Make `UNOBSERVED` the zero value and make `FREE` require positive evidence.

**Test:** a cell behind a wall reads `OCCLUDED`, not `FREE`. A cell outside the sensor FOV reads `UNOBSERVED`. Assert the height fields of an `UNOBSERVED` cell are never read by any consumer (guard in `CellView`).

**Done when:** all four states are reachable and distinguishable in a real sweep, rendered as four distinct colours.

---

### #35: Local ground plane fit
**Tier:** 🟥 CORE · **Day:** D6 · **Blocked by:** #26

**Spec:** Per azimuth column, fit a plane (or line in the r–z projection) to the last few *confirmed* ground returns from #26. Expose `expected_ground_range(ring, azimuth)`.

**Watch out:** use the *local* recent returns, not a global fit. This is the machinery that stops #36 firing on every hill, so getting it wrong makes the whole negative-obstacle feature unusable on the terrain DRDO cares about.

**Test:** on an 8° slope the fitted plane tracks the slope; residuals stay near zero. On flat ground the fit is level.

**Done when:** residuals are near zero on both flat and sloped terrain.

---

### #36: Negative obstacle detection
**Tier:** 🟥 CORE · **Day:** D6 · **Blocked by:** #34, #35

**Spec:** `observability/negative_obstacle.py`. For each (ring, azimuth), compare measured range against `expected_ground_range` from #35. Flag when the residual is large **relative to its neighbours in the same column** — local inconsistency, not absolute deviation. Range shadow: $\Delta = r\,d/h$. Require persistence across **≥3 consecutive scans** before promoting `SUSPECT → NEGATIVE_OBSTACLE`. Never run on `OCCLUDED` cells.

**Watch out:**
1. **Absolute-threshold versions fire everywhere on a slope.** The discriminator is ring-to-ring inconsistency. A model given "detect missing ground returns" will write the absolute version.
2. **Order matters** — carve (#34) first, then test. Occlusion behind a truck produces the same missing-return signature as a ditch.

**Test:**
- Synthetic flat ground → **zero** detections.
- Flat ground with one 0.5 m × 2 m ditch at 15 m → detected, with $\Delta ≈ 15 \times 0.5 / 1.84 = 4.08$ m on nuScenes (4.34 m on KITTI).
- **Constant 8° downslope → zero detections. Write this test first**; it guards the fix that stops the demo embarrassing you on a hill.
- Occlusion behind a synthetic wall → not flagged.

**Done when:** the slope test passes with zero false positives.

---

### #37: Checkpoint — the trench works
**Tier:** 🟥 CORE · **Day:** D6 · **Blocked by:** #36

**Test:** render a scene with a synthetic ditch; the map marks it, a plain 2D occupancy grid built from the same sweep does not.

**Done when:** you have the side-by-side still image. **This is demo beat 1 and the most persuasive single frame in the deck.**

---

# Phase 5 — Claims 3 and 4 (D7)

### #38: Expected returns, kappa, r_blind
**Tier:** 🟥 CORE · **Day:** D7 · **Blocked by:** #4, #18

**Spec:** `observability/sparsity.py`. Per cell: $N_{\exp}(r) = t_{\min} w_{\min} / (r^2 \Delta\phi\Delta\theta)$ with $(t_{\min}, w_{\min})$ from the vehicle config; $\kappa = N_{\text{obs}}/N_{\exp}$; $r_{\text{blind}} = \sqrt{t_{\min}w_{\min}/(\Delta\phi\Delta\theta)}$.

**Watch out:** $(t_{\min}, w_{\min})$ come from `vehicle_ugv.yaml` — they are the sponsor's stated safety requirement, not tuned constants. That is the entire "derived, not calibrated" argument. If they appear as literals here, the claim collapses.

**Test:** assert against the Bible's table — HDL-64E pedestrian $r_\text{blind}$ = 194.8 m, pole = 163.7 m, fence post = 66.8 m; HDL-32E pedestrian = 79.2 m, fence post = 27.2 m. Assert $N_{\exp}$ falls as $1/r^2$, not $1/r$.

**Done when:** both sensor configs reproduce the table.

---

### #39: Structure test and sparsity flags
**Tier:** 🟥 CORE · **Day:** D7 · **Blocked by:** #38

**Spec:** Decision rule from Bible Part 11:

| Condition | Verdict |
|---|---|
| $\kappa \ge 1$, $N_\text{obs} > 0$ | normal confidence |
| $0 < \kappa < 1$, returns structured | `SPARSE_STRUCTURED` — flagged, **not free** |
| $0 < \kappa < 1$, unstructured | noise, suppressed |
| $N_\text{obs} = 0$, $N_\exp \ge 1$ | `FREE` |
| $N_\text{obs} = 0$, $N_\exp < 1$ | **`UNKNOWN`, never `FREE`** |

Structured = returns tightly clustered in range and contiguous in ring index.

**Watch out:** the last row is Claim 3 and a model will "simplify" it to `no returns → free` because that is what every other system does. It is the whole contribution. Also `N_obs == 1` should default to `SPARSE_STRUCTURED` — a single point is never provably unstructured, so take the cautious verdict.

**Test:** `test_unknown_past_r_blind` — a cell at 90 m on nuScenes (past the 27.2 m fence-post $r_\text{blind}$) with zero returns asserts `UNKNOWN`, not `FREE`. A cell at 15 m with zero returns asserts `FREE`. **This is the machine-checked form of Claim 3.**

**Done when:** both assertions pass and `SPARSE_STRUCTURED` cells appear at range in a real sweep.

---

### #40: Speed envelope — Claim 4
**Tier:** 🟥 CORE · **Day:** D7 · **Blocked by:** #4, #9

**Spec:** `planning/speed_envelope.py`.

$$v_{\max}(R) = -a t_r + \sqrt{a^2 t_r^2 + 2aR}$$

with `a`, `t_react` from vehicle config. Per-bearing: compute effective detection range $R(\theta)$ from the binding hazard set, apply $v_{\max}$, emit a polygon. Report **which hazard is binding**.

**Watch out:** `t_react` must include measured P95 pipeline latency (#55), so it cannot be a fixed literal — wire it from the latency harness once that exists, and until then use 0.30 s with a TODO. And this is **advisory output**, not a controller; do not let it gate anything.

**Test:** assert `v_max(21.6) = 12.00 m/s = 43.2 km/h` and `v_max(6.7) = 6.24 m/s = 22.5 km/h` for KITTI; `v_max(12.6) = 8.99 m/s = 32.0 km/h` for nuScenes. Assert monotonicity: a shorter range never yields a higher speed. Assert lowering `a` lowers `v_max`.

**Done when:** the numbers match and the binding hazard is correctly identified as the minimum over the active set.

---

### #41: Conservatism — the caution order
**Tier:** 🟥 CORE · **Day:** D7 · **Blocked by:** #34, #39

**Spec:** `planning/conservatism.py`. Define the cost order `FREE ≤ known-rough ≤ UNKNOWN_COST < LETHAL`, with `UNKNOWN_COST` strictly below `LETHAL` and finite. Enumerate the fourteen information-deficit paths from Bible Part 16 as a table in code, each mapping a flag combination to its cost floor.

**Watch out:** setting `UNKNOWN_COST = LETHAL` "to be safe" satisfies monotonicity and paralyses the vehicle. Monotonicity is necessary, not sufficient.

**Test:** assert the ordering holds for every flag combination in the lattice. Assert `UNKNOWN_COST < LETHAL` strictly.

**Done when:** the order is total over reachable states.

---

### #42: The monotonicity property test
**Tier:** 🟥 CORE · **Day:** D7 · **Blocked by:** #41

**Spec:** `tests/test_conservatism.py`, Hypothesis:

```python
@given(cell=cell_states(), degradation=degradations())
def test_cost_monotone_under_information_loss(cell, degradation):
    degraded = degradation.apply(cell)   # drop points, lower kappa, mark occluded,
                                         # mark provisional, push past r_blind, age it
    assert cost(degraded, VEHICLE) >= cost(cell, VEHICLE)
```

Plus the exhaustive `test_unknown_never_becomes_free()` over the flag lattice.

**Watch out:** the property must call the **public** `cost()`, so any fast path bypassing it is by definition unsupported. And `degradations()` must actually degrade — a model may generate transformations that add information, which makes the test vacuous. Assert the degradation is a true information loss.

**Test:** 10,000 examples, zero violations. Deliberately introduce a bug (make `UNOBSERVED` cost zero) and confirm the test **fails** — a property test you have never seen fail is not evidence.

**Done when:** 10,000 examples pass and the deliberate-bug check fails as expected.

---

### #43: Checkpoint — claims 3 and 4 demonstrable
**Tier:** 🟥 CORE · **Day:** D7 · **Blocked by:** #40, #42

**Test:** render a frame with `SPARSE_STRUCTURED` cells overlaid and the speed envelope drawn. Print the conservatism test result.

**Done when:** all four claims now have running code behind them.

---
# Phase 6 — Motion, Fovea, Planner Interface (D8)

### #44: Static accumulation
**Tier:** 🟥 CORE · **Day:** D8 · **Blocked by:** #18, #13

**Spec:** `temporal/static_layer.py`. Accumulate across frames in the world-anchored map: `h_max ← max`, `h_min ← min`, `count ← min(count+n, 65535)`, mean/variance merged with the Chan parallel-axis formula:

$$n = n_A + n_B, \quad \delta = \mu_B - \mu_A, \quad \mu = \mu_A + \delta\tfrac{n_B}{n}, \quad M_2 = M_{2,A} + M_{2,B} + \delta^2\tfrac{n_A n_B}{n}$$

Confidence decays as $e^{-(t-t_\text{cell})/\tau}$, $\tau = 10$ s. Cap accumulation at ~3 s of history.

**Watch out:** a model will write naive incremental mean and quietly lose the variance, or use the numerically unstable `E[x²] − E[x]²` form. Use Chan explicitly. And the accumulation window matters — odometry drift smears the map over long horizons, and a crisp 3-second map beats a blurry 60-second one.

**Test:** merge two groups with known statistics; assert the combined mean and variance match a direct computation over the union of all points, exactly to floating-point tolerance. Assert `count` saturates rather than wraps at 65,535.

**Done when:** Chan merge is exact and accumulation over 30 frames produces a visibly denser map than a single sweep.

---

### #45: Map-level motion detection
**Tier:** 🟥 CORE · **Day:** D8 · **Blocked by:** #34, #44

**Spec:** `temporal/motion.py`. Because the map is **world-anchored**, motion needs no ego compensation: compare frame $t$ against frame $t{-}1$ **at the same world cell**. A cell transitioning `OCCUPIED → FREE` (carved by a ray) with a nearby cell going `FREE → OCCUPIED` indicates motion. Mark moving cells and exclude them from the static layer.

**Watch out:** this only works if you compare by **global cell index**, not by storage index — storage indices shift as the window scrolls, so comparing them compares different places in the world. This is the subtle failure mode and generated code will get it wrong roughly half the time. Keep a previous-frame snapshot keyed on global `(i, j)`.

**Test:**
- Two identical sweeps with a synthetic ego translation → **zero** motion detected anywhere.
- One synthetic object translated by 2 m between frames → motion detected on that object only.
- **Parked-car test:** a static vehicle across 30 frames is never flagged and stays in the static map. This is the capability that class-only exclusion would have lost.

**Done when:** the static-scene test yields zero false motion and the parked car is retained.

---

### #46: No-smear verification
**Tier:** 🟥 CORE · **Day:** D8 · **Blocked by:** #45

**Spec:** No new code. Automated test plus demo footage.

**Test:** replay a sequence containing a walking pedestrian. Afterwards, query every cell along the path they walked. **All must be `FREE` or `UNOBSERVED`; none `OCCUPIED`.** Put this in CI.

**Done when:** the assertion passes and you have the side-by-side clip (with and without the split) for demo beat 6.

---

### #47: TTC fovea controller
**Tier:** 🟥 CORE · **Day:** D8 · **Blocked by:** #5

**Spec:** `attention/fovea_controller.py`.

$$v_\text{close}(p) = \mathbf{v}\cdot\hat p, \quad \text{TTC} = \frac{\|p\|}{\max(v_\text{close}, v_\min)}, \quad c_\text{ttc} = c_0\Big(\tfrac{\text{TTC}}{\tau_0}\Big)^{\gamma}$$

$$c(p) = \min\big(c_\text{range}(r),\ c_\text{ttc}(p),\ c_\text{boundary}(p)\big)$$

$\tau_0 = 1.0$ s, $v_\min = 2$ m/s, $\gamma$ exposed as a parameter. Profile A = `c_range` only (default, spec-compliant); Profile B = full composition.

*(The `c_object` term needs a tracker, which is cut. Note it as designed-not-built.)*

**Watch out:** the **`min()` is the safety property** — every term can only refine, never coarsen. A model may write `c_ttc` alone, or use `max`, either of which lets the adaptive feature break the PS requirement. Also `v_min` must floor the closing speed or a stationary vehicle coarsens the entire map.

**Test — the floor test is the first test in the file:**
```python
@given(v=vectors(), yaw=floats(), p=points())
def test_floor_never_violated(v, yaw, p):
    assert c(p, v, yaw) <= c_range(norm(p))
```
10,000 cases. Plus: standstill produces cell sizes no coarser than Profile A; the fovea's principal axis rotates with the velocity vector. Assert the worked table from Bible Part 13 — 15 m ahead at 15 m/s → 5 cm; 15 m left → 5 cm (floor holds); 100 m ahead → 33 cm.

**Done when:** the floor property passes 10,000 cases. **That test is your answer to "what if your TTC estimate is wrong?"**

---

### #48: Traversability derivatives
**Tier:** 🟥 CORE · **Day:** D8 · **Blocked by:** #21

**Spec:** `planning/traversability.py`. Per cell from the local 3×3 at the queried level: slope from the fitted normal, roughness from stored height variance, step height as the max single-cell `h_max` discontinuity, clearance from #21.

**Watch out:** **step height is only meaningful at L0/L1.** A 15 cm kerb is invisible in a 40 cm cell, so beyond the fine ring the function must return `UNKNOWN`, not "no step". A model will happily compute it at every level and report confident nonsense at range. Also require a minimum point count across the neighbourhood or a single-point cell yields a meaningless normal.

**Test:** synthetic 25° slope → slope = 25° ±1°. Synthetic 21 cm step at L0 → step = 0.21 m. Same step queried at L3 → `UNKNOWN`. Single-point neighbourhood → `UNKNOWN`, not a number.

**Done when:** all four cases pass, including both `UNKNOWN` returns.

---

### #49: Cost map
**Tier:** 🟥 CORE · **Day:** D8 · **Blocked by:** #48, #41

**Spec:** `planning/costmap.py`, per Bible Part 14. `LETHAL` for slope/step/clearance/negative-obstacle violations; `UNKNOWN_COST` for `UNOBSERVED`, `SPARSE_STRUCTURED`, or low $\kappa$; otherwise a weighted blend of slope, roughness and class penalty. All thresholds from `vehicle_ugv.yaml`.

**Watch out:** `UNKNOWN_COST` is high, **finite**, and tunable. Zero drives into unmapped holes; lethal paralyses the vehicle in any partially observed scene. Both are common and both come from collapsing the tri-state.

**Test:** each of the four `LETHAL` conditions independently triggers, and each *just* under threshold does not. **Assert `UNOBSERVED` and `SPARSE_STRUCTURED` both produce `UNKNOWN_COST`, never zero** — that assertion encodes the whole tri-state argument.

**Done when:** the threshold matrix passes and #42's property test still passes with the real cost function wired in.

---

# Phase 7 — Demo Assets (D9)

### #50: Synthetic hazard injection [Track S]
**Tier:** 🟥 CORE · **Day:** D9 · **Blocked by:** #36

**Spec:** `eval/inject_hazards.py`. Modify a **real** sweep to contain a hazard of known geometry:
- **Trench** (width $w$, depth $d$, at range $r$): for beams that would strike inside the trench, displace the return outward by $\Delta = r\,d/h$; where the far wall occludes, remove the return entirely.
- **Kerb** (height $t$): add returns on a vertical face at the specified range.
- **Pole** (diameter, height): add a vertical cylinder of returns at a specified range.

Parameterised so the same tool generates the demo footage **and** the detection-vs-range sweep in #60.

**Watch out — state this caveat in the slides, do not let a judge find it:** injecting with the same $\Delta = rd/h$ geometry the detector uses tests that your implementation inverts your own forward model, not that the physics is correct. CARLA, being an independent renderer, would test the physics. `[Track C]` replaces this ticket if lab access appears.

**Test:** inject a 2 m × 0.5 m trench at 15 m into a real nuScenes sweep; assert #36 detects it. Inject at 30 m (past the 12.6 m nuScenes ditch range) and assert it is **not** detected — the failure case matters as much as the success.

**Done when:** injection produces geometrically consistent sweeps and both the positive and negative range cases behave as predicted.

---

### #51: rerun logging
**Tier:** 🟥 CORE · **Day:** D9 · **Blocked by:** #22

**Spec:** `viz/dashboard.py`. Log per frame to rerun: the 2.5D map as a coloured point/box cloud with height as elevation, cell borders visible so foveation is legible, plus the raw sweep for reference. One timeline, scrubbable.

**Watch out:** rerun can drown in geometry. Log the *cells*, not the raw points, for the map view — 1 M cells per frame will stall it. Log only occupied cells, and decimate coarse levels for display.

**Test:** a 100-frame sequence loads in the rerun viewer and scrubs smoothly.

**Done when:** the map plays back as a video-able timeline.

---

### #52: Overlays and HUD
**Tier:** 🟥 CORE · **Day:** D9 · **Blocked by:** #51

**Spec:** Toggleable layers: class colour, `UNOBSERVED` vs `FREE`, `OCCLUDED`, negative-obstacle candidates, `SPARSE_STRUCTURED`, clearance heat-map, motion cells. HUD text: cells per level, MB in use, latency P50/P95, stamp-mismatch counter, estimated extrinsics.

**Watch out:** colour discipline — class is categorical, height is continuous, confidence is a separate channel (alpha). Do not encode three variables in one hue ramp; it looks fine on your screen and is unreadable in a compressed video. Reserve red exclusively for `LETHAL`/`NEGATIVE_OBSTACLE`, and use a colourblind-safe categorical palette.

**Test:** each overlay toggles independently. **Assert the HUD's memory figure equals `eval/baselines.py`'s independently computed total** — a dashboard disagreeing with the harness is the most damaging thing that can happen on a slide.

**Done when:** overlays work and the memory equality test is automated.

---

### #53: Split-screen comparison
**Tier:** 🟥 CORE · **Day:** D9 · **Blocked by:** #52

**Spec:** Two panes, same frame, synchronised: **left** a uniform 5 cm Cartesian grid (the naive baseline), **right** the foveated clipmap. Memory and latency counters under each. Second mode: 2D occupancy grid vs DRISHTI for the trench beat.

**Watch out:** the baseline must be genuinely built and genuinely measured, not a mock-up with a fake number. If it is too slow or too large to run live, run it offline and show real recorded numbers — but never a placeholder.

**Test:** both panes render the same frame; the memory counters differ by the ratio from #56.

**Done when:** the split-screen runs on the demo sequence with real numbers on both sides.

---

### #54: Speed envelope display
**Tier:** 🟨 SUPPORTING · **Day:** D9 · **Blocked by:** #40, #52

**Spec:** A speedometer-style gauge with a red zone driven by $v_\max$, the binding hazard named beside it, and the per-bearing envelope polygon drawn on the map.

**Test:** drive into an occluded region; the limit visibly falls and the binding hazard label changes.

**Done when:** the gauge responds to occlusion in recorded footage. **This is demo beat 7.**

---

# Phase 8 — Evidence (D10)

### #55: Latency instrumentation
**Tier:** 🟥 CORE · **Day:** D10 · **Blocked by:** #49

**Spec:** `eval/latency.py`. Per-stage timing across the full pipeline, reported as **P50 and P95**, end to end from scan-complete to map-ready. Feed measured P95 into `t_react` for #40.

**Watch out:** time with CUDA events on GPU stages, not wall-clock around async kernels — a hidden `.cpu()` makes everything look fast until it doesn't. And report **latency, not FPS**: the sensor produces 20 Hz (nuScenes) or 10 Hz (KITTI), so frame rate above that is meaningless. A model asked for "FPS" will give you FPS; ask for latency.

**Test:** run 500 frames; assert the per-stage totals sum to the measured end-to-end time. **If they do not, there is unaccounted time** — usually a synchronisation stall — and finding it is worth more than optimising any stage.

**Done when:** the stage sum reconciles with the total and P50/P95 are recorded.

---

### #56: Memory baselines
**Tier:** 🟥 CORE · **Day:** D10 · **Blocked by:** #11

**Spec:** `eval/baselines.py`. Four baselines at **identical extent and identical bytes per cell**: dense uniform 3D voxel (label as strawman), dense uniform 2.5D at 5 cm, a sparse hash-voxel/Octomap-style structure, and DRISHTI.

**Watch out:** **quote 16×, not 267×.** The dense-3D figure compares against something nobody would build; report it explicitly labelled as the naive baseline and lead with the honest one. Measure the sparse baseline even if the answer is unflattering — losing on memory while winning decisively on query latency is a real result either way.

**Test:** assert DRISHTI = **1,048,576 cells = 12.58 MB** (or 10.22 MB with #17), dense 2.5D = **16,777,216 cells = 201.3 MB**, ratio **16.0×**. Assert occupancy of the uniform grid on one sweep is **< 1%** (34k points / 16.8M cells on nuScenes ≈ 0.2%).

**Done when:** all four baselines report at matched extent and the ratios match.

---

### #57: mIoU per distance band
**Tier:** 🟥 CORE · **Day:** D10 · **Blocked by:** #31

**Spec:** `eval/metrics.py`. Confusion matrix and per-class IoU, **binned by the clipmap level boundaries** (0–12.8 / 12.8–25.6 / 25.6–51.2 / 51.2–102.4 m) rather than arbitrary bins, so accuracy is reported per resolution level. Directly satisfies the PS's "accuracy across varying distances".

**Watch out:** hand-verify the metric on a small confusion matrix you compute on paper. A buggy metrics harness produces confidently wrong numbers and those go straight onto slides.

**Test:** a 3×3 confusion matrix whose mIoU you computed by hand. Spot-check five frames' verdicts against your own read.

**Done when:** the hand-computed case matches and per-band numbers are produced for the demo sequence.

---

### #58: Pareto curve via self-consistency
**Tier:** 🟥 CORE · **Day:** D10 · **Blocked by:** #47, #56

**Spec:** `eval/pareto.py`. Sweep $\gamma$ from 0 (uniform, finest) upward. For each setting record total memory and **elevation deviation against a uniform 5 cm map built from the same sweep** — self-consistency, not ground truth. Plot memory vs deviation per band and **mark the knee**.

**Watch out:** the Bible frames this as needing ground truth; it does not. The self-consistency proxy answers "what did coarsening cost me relative to the finest map I could have built", which is arguably the more relevant question and runs on plain data with no simulator.

**Test:** at $\gamma = 0$ deviation is ~0 and memory is maximal. Deviation increases monotonically with $\gamma$. The knee is identifiable.

**Done when:** the curve is plotted with the knee marked and your operating point on it. **The Bible calls this the most convincing single artifact in the project.**

---

### #59: Portability ablation — 32-beam vs 64-beam
**Tier:** 🟥 CORE · **Day:** D10 · **Blocked by:** #5, #38, #40

**Spec:** Run the identical pipeline over both sensor configs you actually have — **HDL-32E (nuScenes)** and **Ouster OS1-64 (RELLIS-3D)** — and tabulate. Because both are real datasets on real sensors, this is stronger than the Bible's original proposed version, and stronger again than decimating one dataset to fake a second sensor: it's two structurally different 64→32 channel sensors from two different manufacturers, not one sensor downsampled.

**Watch out:** the point is *correct scaling*, not just different numbers. Detection ranges scale as $1/\Delta\phi$ or $1/\sqrt{\Delta\phi}$; level boundaries scale with $\Delta\theta$ only. If both moved together, something is conflated. The specific numbers below are **placeholders carried from the HDL-64E reference config** — replace the HDL-64E column with the Ouster OS1-64 numbers once Ticket #6 has measured its real $\Delta\theta/\Delta\phi$ against RELLIS-3D point clouds; do not present the table below as-is if that measurement hasn't run, since it would silently be reporting a sensor you didn't actually test.

**Test:** once the Ouster OS1-64 config is measured, assert a table of this shape (HDL-64E numbers shown are the *reference* case, not a substitute for the measured Ouster row):

| Quantity | HDL-64E (reference) | Ouster OS1-64 (RELLIS-3D, measured) | HDL-32E (nuScenes) |
|---|---|---|---|
| 5 cm level reaches | 16.6 m | *fill from #6* | 8.6 m |
| 15 cm kerb | 20.2 m | *fill from #6* | 6.4 m |
| 2 m ditch | 21.6 m | *fill from #6* | 12.6 m |
| Pedestrian $r_\text{blind}$ | 194.8 m | *fill from #6* | 79.2 m |
| Safe speed (ditch) | 43.2 km/h | *fill from #6* | 32.0 km/h |

Assert no code change was required to add the third config — only a config swap.

**Done when:** the nuScenes and RELLIS-3D configs both run end to end with only a YAML change, and the table reproduces with real measured numbers in every cell (no reference-only placeholders left in what goes on a slide).

---

### #60: Hazard detection vs range [Track S / Track C]
**Tier:** 🟨 SUPPORTING · **Day:** D10 · **Blocked by:** #50

**Spec:** Sweep injected hazards across a ladder of ranges; plot detection rate against range and overlay the predicted curve ($t/\Delta\phi$ for kerbs, $\sqrt{wh/\Delta\phi}$ for ditches). Produces the spec-sheet table with a measured column.

`[Track C]` — if CARLA lands, replace injection with real rendered hazards. That upgrades the caveat in #50 from "tests my implementation" to "tests the physics", and it is worth the trip to the lab.

**Test:** measured detection range tracks prediction within ~20%. Where it does not, investigate before reporting — a systematic offset usually means the sensor constants or `h` are off.

**Done when:** predicted vs measured is plotted for at least kerb and ditch.

---

# Phase 9 — Ship (D11–D12)

### #61: Latency compensation
**Tier:** 🟨 SUPPORTING · **Day:** D11 · **Blocked by:** #55

**Spec:** Publish the map with a validity timestamp at the planner's action time. Because the static map is **world-anchored, ego staleness needs no transform at all** — that is the free half. Only motion-flagged cells need advancing, and their extent is inflated by prediction uncertainty so the compensation stays conservative.

*(One of the two cheap force-ins. Cut it if D11 is tight.)*

**Watch out:** never extrapolate further than confidence supports; cap the horizon. Publish both measurement and validity timestamps.

**Test:** assert ego staleness requires zero map transformation. Assert inflated extents never shrink a hazard.

**Done when:** the map publishes with both timestamps and #42 still passes.

---

### #62: Record the seven beats
**Tier:** 🟥 CORE · **Day:** D11 · **Blocked by:** #54, #59

**Spec:** Record each beat as a separate clip:
1. **The trench** — 2D occupancy says `CLEAR`, DRISHTI says `LETHAL`
2. **Bridge and branch** — max-height and min-height both fail, multi-layer gets both right
3. **Sparsity split-screen** — approaching pedestrian vanishes on the left, persists on the right, **with the predicted $r_\text{blind}$ marked on the timeline**
4. **Uniform vs foveated** — memory counters running
5. **The $\gamma$ sweep** — teardrop stretches, memory moves
6. **The pedestrian** — motion field, no smear trail, parked cars retained
7. **The speed envelope** — limit falls under occlusion, binding hazard named

**Watch out:** render offline at whatever speed you like, but **report measured latency separately as a number and say the video is not real-time if it isn't.** Presenting a sped-up render as live performance is precisely the overclaim that costs credibility, and it is unnecessary — your latency numbers are good.

**Test:** each clip is intelligible without narration. Beat 1 should need no explanation at all.

**Done when:** seven clips exist and beat 1 works on a muted playback.

---

### #63: Edit the video
**Tier:** 🟥 CORE · **Day:** D11 · **Blocked by:** #62

**Spec:** Assemble in the beat order above. Open on the trench. Caption each beat with one line. End on the limitations slide.

**Watch out:** every number spoken or captioned must trace to #55–#60, not to memory. Cross-check each caption against the harness output before export.

**Done when:** the video exports and every number in it is traceable to a harness run.

---

### #64: Failure-mode documentation
**Tier:** 🟥 CORE · **Day:** D11 · **Blocked by:** —

**Spec:** Write out what the system does not do, from Bible Part 24 — thin cables invisible past 2.2 m on this sensor, negative obstacles sampling-limited to 12.6 m, terrain beyond ~35 m barely sampled, 2.5D cannot do multi-storey, nuScenes reaches 70 m against a 100 m requirement (a hardware ceiling), RELLIS-3D's own range is separately limited by off-road terrain — vegetation and elevation swallowing returns, not the sensor's floor — so it confirms real-world off-road behavior without independently validating the 100 m number either, training-set size if you trained on mini.

**Watch out:** this is not a weakness slide, it is the credibility slide. A panel that is *shown* the limits believes everything else. Do not soften the numbers.

**Done when:** you can state each limitation and its cause without hesitation.

---

### #65: Slides
**Tier:** 🟥 CORE · **Day:** D12 · **Blocked by:** #63, #64

**Spec:** Structure around the four claims and the through-line — **beam geometry → detection range → what the map may claim → how fast you may drive.** Include: the derivation slide (5 cm/40 cm from $\Delta\theta$), the nesting-exactness test running, the bridge/branch contradiction, the trench, $r_\text{blind}$, the speed envelope with the procurement table, the Pareto curve, the portability table, the memory table with the strawman labelled, and the limitations.

**Watch out:** the PS asks for specific metrics — FPS, accuracy by distance, memory vs uniform 3D. Make sure each is visibly answered even though you are also arguing for better metrics. Compliance first, then the improvement.

**Test:** dry-run against someone who has not seen the project. Every number spoken must appear in the slides and trace to a harness run.

**Done when:** the dry run is clean and no number is from memory.

---

### #66–#68: Stretch, only if ahead
**Tier:** 🟦 STRETCH · **Day:** D12

- **#66 Sparsity recovery rate** vs the $N_\exp$ curve — Claim 3's *measured* evidence. Without it Claim 3 stays derived. ~3 h.
- **#67 Outdriving fraction** — % of the sequence where ego speed exceeded $v_\max$, per hazard class. Claim 4's measured evidence, and no benchmark anywhere reports it. ~2 h.
- **#68 Residual motion channels** — the Bible's per-point motion mechanism. Only if everything else is done and rehearsed. ~1.5 days.

---

# What Is Deliberately Not Built

Name these as scoped-out, not forgotten. Each has a reason.

| Cut | Why | Cost of cutting |
|---|---|---|
| Residual input channels | 1.5–2 days; competes with the eval harness | Motion is per-cell not per-point; weaker at long range |
| Kalman tracker | Its main consumer is fovea object-refinement | `c_object` term unavailable; a distant pedestrian is not specially refined |
| Promotion/demotion + `PROVISIONAL` | ~1 day | Visible seam flicker at the ring boundary — choose demo footage where it is less obvious, and declare it |
| Meta-Kernel | Ablation-gated by design; no time to run the ablation | Explicit coordinate channels remain the fallback, and they work |
| Albedo / water detection | Least mature element; unvalidated | `NON_TRAVERSABLE_TERRAIN` has no radiometric mechanism |
| Multi-echo vegetation | **Data does not support it** — neither dataset ships dual returns | Height heuristic remains, at lower confidence |
| `INFERRED` completion | ~1 day | More `UNKNOWN` cells; more conservative routes. Correct, just less impressive |
| LOD streaming | ~1.5 days | No degraded-comms story to demo |

---

*This build map is a companion to `DRISHTI_Project_Bible_v3.md`. The Bible explains WHY; this tells you WHAT, in what order, what generated code gets wrong, and how to prove each piece works. When they disagree, the Bible defines intent and this map defines execution.*

*Three tickets decide whether the rest is worth anything: **#6** (the sensor model is right), **#13** (the map does not silently lie), and **#42** (the system cannot become permissive). If you are short on time, those three still get their full tests.*



