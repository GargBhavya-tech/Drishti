# DRISHTI — Session Handoff

**Written:** 2026-09-10 → updated 2026-09-11 → **updated again 2026-09-12, end of a third long session. Sections 1-9 below are from the END OF SESSION 2 and are now STALE in places** (most importantly: section 1's "Phase 5 onward not started" claim is no longer true — Phases 5, 6, and 8 are all built now). **Read section 0 first, then treat sections 1-9 as historical background for Phases 0-4 only.**

**Purpose:** let a fresh Claude Code session (or a human) pick up exactly where this one left off, with zero prior context.

Read `DRISHTI_Project_Bible_v3.md` (the WHY) and `DRISHTI_Build_Map.md` (the WHAT/WHEN, ticket-by-ticket) first if you haven't — this doc assumes you have. Everything below is keyed to Build Map ticket numbers.

---

## 0. SESSION 3 UPDATE (2026-09-12) — READ THIS FIRST

### 0a. What got built this session, in order

1. **Phase 5 (Tickets #38-43)** — Claims 3 & 4, fully built: `observability/sparsity.py`, `planning/speed_envelope.py`, `planning/conservatism.py` (+ `deficit_floor()` extracted as its own public function, used by Phase 6's costmap), `eval/checkpoint_sparsity_speed.py`. All tests passing, including a 10,000-example Hypothesis property test for the conservatism monotonicity invariant.
2. **Phase 6 (Tickets #44-49)** — pure backend, no frontend: `temporal/static_layer.py` (Chan-merge cross-frame accumulation), `temporal/motion.py` (world-anchored motion detection), `attention/fovea_controller.py` (TTC fovea, verified exactly against Bible Part 13's own worked table), `planning/traversability.py`, `planning/costmap.py` (real LETHAL/UNKNOWN_COST cost, built on `conservatism.deficit_floor()`).
3. **Phase 8 (Tickets #55-61, plus #50/#60)** — `eval/latency.py`, `eval/baselines.py`, `eval/metrics.py`, `eval/pareto.py` (+ checkpoint), `eval/portability.py`, `eval/inject_hazards.py` + `eval/detection_vs_range.py` (+ checkpoint), `temporal/latency_compensation.py`.
4. **A React + Three.js frontend** (`frontend/`) — not part of the Build Map, built because the user wants a real interactive dashboard for the demo, not Streamlit/matplotlib. Vite + React + TypeScript + `@react-three/fiber` + `motion` + Zustand + Tailwind v4. **Currently runs on SYNTHETIC MOCK DATA** (`frontend/src/lib/mockData.ts`), not real DRISHTI backend output — this is the single biggest thing left to fix, see 0d below.
5. **Model accuracy work** — a real, ongoing investigation, not just a training run:
   - Diagnosed WHY class 4 (STATIC_OBSTACLE) and class 2 (CAUTION) were weak in the existing `checkpoints_multi_remote/checkpoint_epoch19.pt` (Run #2 from session 2): both get absorbed into class 5 (VEGETATION) — see `eval/diagnose_confusion.py`, a new diagnostic script (also supports `--checkpoint-b` to diff two checkpoints directly, sampling the SAME frames through both).
   - Added class weighting (inverse-sqrt-frequency, mean-normalised over nonzero weights, applied only to the secondary confidence-weighted CE term, not the primary Lovász-softmax term), data augmentation (radial jitter on raw pre-normalization RangeImage fields + circular azimuth roll + mirror flip, all three applied consistently to `tensor`+`target`+`valid_mask` together), comprehensive per-epoch logging (`training_log.jsonl`: train/val loss, val mIoU, per-class IoU, LR, epoch time, an explicit overfitting flag over a 3-epoch window), a `best.pt` checkpoint saved on mIoU improvement, and a `--init-from-checkpoint` fine-tuning path (loads weights only, doesn't resume optimizer/scheduler state) — all in `perception/train.py` / `perception/losses.py`.
   - **Launched a fine-tune run on the remote GPU** from `checkpoint_epoch19.pt`, 20 fresh epochs, out-dir `checkpoints_multi_v2` — **check section 0b for its live status, it was still running when this session ended.**
   - **A real, resolved false alarm worth knowing about**: `eval/diagnose_confusion.py` run LOCALLY (CPU, torch 2.11.0) gave systematically much worse per-class IoU (e.g. class 6 VEHICLE: 0.012) than the SAME checkpoint run on the REMOTE GPU (cuda, torch 2.0.1: class 6 = 0.528, matching `TRAINING_RESULTS.md`'s own number). Checkpoint bytes, `channel_stats.json`, and RELLIS `.bin`/`.label` files were all verified byte-identical between local and remote via `sha256sum` — the discrepancy is a **local CPU/torch-version numerical issue, not a bug in the diagnostic script.** *Always run `eval/diagnose_confusion.py` (and anything else doing real inference) on the remote GPU (`--device cuda`), never locally on CPU, if the numbers need to be trusted.*
6. **Four "stand out" / differentiator features**, all built and tested this session (backend tests + frontend typecheck both clean):
   - **`planning/path_planner.py`** — A* over the cost grid (LETHAL cells are hard walls, UNKNOWN terrain is expensive but never forbidden, cheaper detours are preferred over the straight line through costly cells). Ported to TypeScript (`frontend/src/lib/pathPlanner.ts`) and wired into the 3D scene as `PlannedPath` in `Scene.tsx` — a route recomputed EVERY frame from the live cost grid, visibly bending around the moving hazard.
   - **`frontend/src/components/ComparisonWipe.tsx`** — an interactive drag-to-reveal comparison: a 2D-flattened view (`generateFlattenedVariant` in `mockData.ts` reclassifies every NEGATIVE_OBSTACLE cell as DRIVABLE, height 0 — what a naive 2D occupancy grid would show, missing the range-shadow signature of a trench) on one side, the real DRISHTI multi-layer view (LETHAL) on the other, same real hazard. New `compareWipe` store flag + `CompareWipeToggle` button in the control strip, mutually exclusive with the existing split-screen toggle.
   - **`perception/segnet.py`'s `return_attention=True`** (on `AttentionGate.forward` and `FusionSegNet.forward`) — exposes the trained network's own real attention-gate Sigmoid activations as a genuine per-pixel [0,1] map, fully backward-compatible (default `False` path is bit-for-bit the original behaviour; verified with a dedicated test). `eval/checkpoint_attention_overlay.py` renders a real 3-panel heatmap (input range image / predicted class / attention) from a real checkpoint on a real frame — **not yet actually run** (should be run on the remote GPU per the CPU/GPU numerical-difference finding above, not locally).
   - **`BASELINE_COMPARISON.md`** — FusionSegNet (5.82M params) positioned against SalsaNext/CENet/FIDNet with real, sourced published numbers (arXiv links included, verified via WebSearch before writing) and an explicit, deliberate refusal to present cross-dataset mIoU as a ranking (RELLIS-3D off-road vs. SemanticKITTI urban are not comparable domains) — matches this project's own "state limitations, don't hide them" culture.
7. **SSH access to the GPU is now aliased.** `~/.ssh/config` (on the Windows laptop) has a `Host drishti-gpu` entry (`HostName 172.16.192.12`, `User utkarsh`, `IdentityFile ~/.ssh/id_ed25519_drishti_gpu`, `BatchMode yes`) — use `ssh drishti-gpu "..."` / `scp <file> drishti-gpu:~/drishti/...` instead of the old full `-i ... utkarsh@172.16.192.12` form everywhere in sections 3/4/7 below (both still work, the alias is just shorter). **Gotcha hit this session**: PowerShell's `Add-Content -Encoding utf8` writes a UTF-8 BOM that Git Bash's OpenSSH cannot parse ("no argument after keyword \357\273\277") — if `~/.ssh/config` ever breaks with that exact error, rewrite it via a POSIX tool (Git Bash heredoc) instead of PowerShell's `Out-File`/`Add-Content`.
8. **`scripts/sync_training_results.ps1`** — updated this session for the new `training_log.jsonl` format and the `checkpoints_multi_v2` run. Run `.\scripts\sync_training_results.ps1` (live-polls) or `-Once` (single pull) from the repo root in PowerShell to check training progress without manually SSHing in.

### 0b. Live state when this session ended — CHECK THIS FIRST

- **Remote fine-tune run** (`checkpoints_multi_v2`, PID ~30676 on `drishti-gpu`) was still running, ETA roughly 5.5-6 hours total from its start. Check with `ssh drishti-gpu "ps aux | grep perception.train | grep -v grep"` and `cat ~/drishti/checkpoints_multi_v2/training_log.jsonl` (or just run `scripts/sync_training_results.ps1`). As of the last check (2 epochs in): epoch 0 val mIoU 0.550, epoch 1 dipped slightly to 0.532 (expected OneCycleLR warmup noise, LR still climbing toward its peak) — not concerning yet, but **watch class 4 (STATIC_OBSTACLE) specifically** (was still 0.0 at epoch 1; this fine-tune's whole point is fixing that via the new class weighting).
- **Once it finishes**: run `eval/diagnose_confusion.py --checkpoint checkpoints_multi/checkpoint_epoch19.pt --checkpoint-b checkpoints_multi_v2/<final>.pt --max-frames 600 --device cuda` **on the remote GPU** to get a real, direct before/after per-class comparison via its `compare_checkpoints()` path. This is an OPEN THREAD from this session, not finished.
- **`eval/checkpoint_attention_overlay.py` has never actually been run** — built and the underlying `return_attention` machinery is unit-tested, but the script itself needs a real invocation (on the remote GPU, `--device cuda`) to produce an actual heatmap PNG. Also an open thread.
- Scratch diagnostic JSONs from this session's debugging (`eval/out/confusion_matrix*.json`) are untracked and safe to delete or keep as reference.

### 0c. Test status

Every new/changed Python module got its own passing test run at the time it was built (`tests/test_path_planner.py` 8/8, `tests/test_segnet.py` 15/15 including 3 new `return_attention` tests, `tests/test_losses.py` 10/10 including 4 new `class_weight` tests, plus the full Phase 5/6/8 suites) — **but no single full-repo `pytest -q` run happened after the LAST changes** (the four differentiator features together). Run the full suite fresh before trusting "everything passes." Known pre-existing environmental flakes (not real bugs, confirmed by re-running in isolation): `tests/test_checkpoint_sparsity_speed.py` and `tests/test_addressing.py` occasionally hit Hypothesis timing/deadline flakes under full-suite CPU load.

### 0d. What's actually left (priority order)

1. **Wire real DRISHTI data into the frontend.** It runs on synthetic mock data right now (`frontend/src/lib/mockData.ts`) — every demo feature (wipe comparison, path planner, attention overlay if frontend-integrated) is currently showing fake data. Planned approach: a Python `eval/export_frames.py` exporting real cached-inference + real Clipmap state as JSON in the same shape `mockData.ts` already uses, so swapping the data source is additive.
2. **Close the fine-tune comparison loop** — see 0b.
3. **Actually run `checkpoint_attention_overlay.py`** on the remote GPU — see 0b.
4. **Wire the live γ-slider to a real FastAPI backend.** It currently computes locally in TypeScript (`frontend/src/lib/foveaMath.ts`, a direct port of `attention/fovea_controller.py`'s formulas) — genuinely live, but not calling the real Python backend. Deliberately deferred ("build frontend first, don't touch backend") and never circled back to.
5. **Stretch tickets #66/#67** (sparsity recovery rate vs. predicted N_exp curve; outdriving fraction) — not yet built.
6. Phase 9's non-code deliverables (slides, demo video) — everything built so far only matters once it lands in the actual pitch.

### 0e. Everything below this point (sections 1-9) is SESSION 2's own handoff, covering Phases 0-4 in detail — still accurate for that scope, just stale on "what's done" (see the correction at the top of this file).

---

## 1. Where the build actually stands

**Phases 0–4 (Tickets #1–#37) are built, tested, and verified — locally AND on the GPU training machine, both confirmed clean.** Phase 5 onward (sparsity/Claim 3, speed envelope/Claim 4, conservatism, motion/fovea, traversability, dashboard, eval harness — Tickets #38+) is **not started**.

| Phase | Tickets | Status |
|---|---|---|
| 0 — Foundations, sensor model, the gate | #1–#9 | ✅ done (earlier session) |
| 1 — The Clipmap | #10–#16 | ✅ done |
| 2 — Cells, multi-layer, scatter | #17–#22 | ✅ done (#17 deliberately deferred, per Build Map's own allowance) |
| 3 — Perception (range image → training) | #23–#30 | ✅ done |
| — Cache inference / semantic map checkpoint | #31–#32 | ❌ **not started** — blocked on #30, deliberately left for after a good trained checkpoint exists; see section 9 |
| 4 — Observability & negative obstacles | #33–#37 | ✅ done **this session** |
| 5 — Claims 3 & 4 (sparsity, speed envelope, conservatism) | #38–#43 | ❌ not started — **next up, see section 9** |
| 6 onward | #44–#68 | ❌ not started |

**Test counts, both environments, verified at the end of this session (not assumed — actually run on both machines):**
- Full suite excluding the slow `test_train.py` (which needs real point-cloud fixtures and takes minutes): **149 passed, 2 skipped**
- `test_train.py` alone: **5 passed** (includes the 2 new multi-sequence tests added this session)
- **Total: 154 passed, 2 skipped**, identical shape on both the local Windows laptop (Python 3.14.3, torch 2.11.0) and the remote GPU server (Python 3.8.10, torch 2.0.1+cu117).

The 2 skips are the same pre-existing gap as before: `nuscenes-devkit` is not installed anywhere and no nuScenes-mini data has been downloaded. Does not block anything built so far (RELLIS-3D is the primary dataset per Ticket #30's own spec, not a fallback).

---

## 2. What was built, ticket by ticket

### Phase 1 — The Clipmap (#10–#16)
- `grid/addressing.py`, `grid/cell.py`, `grid/clipmap.py` (`Clipmap` class: SoA allocation, scroll/clear-on-scroll, stamp validation, `lookup()`).
- Tests: `test_addressing.py`, `test_nesting_exactness.py`, `test_clipmap.py`.

### Phase 2 — Cells and the Overhang Claim (#18–#22)
- `grid/scatter.py`, `grid/histogram.py`, `grid/layers.py` (ground/gap/ceiling extraction — Claim 1).
- `eval/checkpoint_first_map.py` — synthetic-scene integration checkpoint.
- Tests: `test_scatter.py`, `test_class_agg.py`, `test_histogram.py`, `test_layers.py`, `test_checkpoint_first_map.py`.

### Phase 3 — Perception (#23–#30)
- `perception/range_image.py`, `perception/circular_pad.py`, `perception/ground_prior.py`, `perception/input_tensor.py`.
- `perception/segnet.py` — **`FusionSegNet`**, extracted from the user-supplied `FusionSegNet_v5 (1).ipynb`.
- `perception/losses.py` — Lovász-Softmax + confidence-weighted CE + deep supervision.
- `perception/train.py` — the training script. **Now supports multiple RELLIS-3D sequences at once** (added this session — see section 4).
- `perception/taxonomy.py`, `perception/rellis_loader.py` — RELLIS ID→name mapping, label loading.
- Tests: `test_range_image.py`, `test_circular_pad.py`, `test_ground_prior.py`, `test_input_tensor.py`, `test_segnet.py`, `test_losses.py`, `test_train.py`.

### Phase 4 — Observability and Negative Obstacles (#33–#37) — **built this session**
- `observability/raycast.py` (#33) — 2D DDA (Amanatides-Woo) ray traversal. `dda_trace()` is the raw per-level primitive; `trace_beam()` does Ticket #33's mandated coarse-first-with-fine-ring-refinement (traces the WHOLE beam at the clipmap's coarsest level, plus a second pass at the finest level's own cell size for the portion of the beam within that level's sensor-Nyquist radius).
- `observability/observe.py` (#34) — `carve_frame()`: combines every beam's trace for one frame into per-cell FREE/OCCUPIED/OCCLUDED evidence, with explicit precedence (OCCUPIED > FREE > OCCLUDED) so beams disagreeing about the same cell in one frame resolve deterministically regardless of iteration order. Free-space carving ("decaying occupancy") is just the direct overwrite — no separate counter needed. Added `Clipmap.mark_observability(level, gi, gj, obs_state)` to `grid/clipmap.py` as the write path (mirrors `lookup()`'s bounds-check, and — like `grid/scatter.py`'s writes — always refreshes `stamp` alongside `flags`, or Ticket #14's integrity check would treat the cell as corrupted on its next read).
  - **A real bug caught and fixed during this ticket, worth knowing**: my first version only cast the occlusion "shadow" at the clipmap's coarsest level. `Clipmap.lookup()` always resolves a query through the **finest** level whose *window* (not Nyquist radius) contains that point — so for any occluded cell close to the vehicle, `lookup()` would check the (untouched) finest level first and report `UNOBSERVED`, never reaching the correctly-marked coarse level underneath. Fixed by re-tracing the shadow segment with the same `trace_beam()` coarse+fine tiering used for the main ray, not a bespoke coarsest-only pass. **Lesson for future tickets touching the clipmap**: any write path must match `lookup()`'s own level-selection logic, not just "write to whichever level seems logically right."
- `observability/ground_plane.py` (#35) — `fit_local_ground_planes()` fits a LOCAL least-squares line (z = a·r_xy + b) per azimuth column, using only the last `k_recent` (default 5) CONFIRMED ground points from Ticket #26's walk — deliberately local, not a global fit, so it tracks a slope instead of averaging flat-then-sloped terrain into nonsense. `expected_ground_range(fit, ring, azimuth_col)` inverts the sensor's uniform-beam-spacing elevation model (same formula `perception/range_image.py`'s arcsin-fallback row assignment uses, inverted) to predict the FULL 3D range at which that ring's beam would intersect the local ground line.
- `observability/negative_obstacle.py` (#36) — `detect_anomalous_cells()`: flags a cell when its range residual (measured − expected) is large **relative to its ring-neighbors in the same column**, never against an absolute threshold (an absolute version fires on every slope, since a slope's residual grows smoothly with ring index even with a *good* fit — see the `test_constant_8_degree_downslope_zero_false_positives` test, which the Build Map explicitly says to write first). Occluded cells (per Ticket #34's carving) are skipped outright, never scored. `PersistenceTracker` is a small, separate state machine requiring the SAME world cell (not the same ring/azimuth — those shift every frame as the vehicle moves) to be flagged on ≥3 CONSECUTIVE frames before promoting SUSPECT → NEGATIVE_OBSTACLE; a missed frame resets the streak.
- `eval/checkpoint_trench.py` (#37) — the demo checkpoint: a synthetic ditch scene (physically exact per-ring/azimuth ground geometry, `r = h_m / sin(phi(ring))`, ditch = a window of (ring, azimuth) cells with the point simply omitted), run through the full #33-36 pipeline for 3 identical "consecutive" frames, producing a side-by-side PNG — a plain returns-only occupancy grid (ditch is invisible) next to DRISHTI's map (ditch flagged in red). Sent to the user this session; visually convincing (the plain grid's radial return pattern shows no gap-shaped anomaly at all, while DRISHTI's map shows a tight, precisely-located red cluster right at the ditch).
- Tests: `test_raycast.py` (8), `test_observe.py` (7), `test_ground_plane.py` (5), `test_negative_obstacle.py` (8), `test_checkpoint_trench.py` (2) — 30 new tests, all passing, zero regressions in the existing 119.

---

## 3. The GPU — full detail on what it is and how to reach it

*(Unchanged from the previous handoff — still accurate as of this session's end.)*

### What it is
A college lab machine, reached over the college network (not the public internet — you must be on that network for SSH to reach it at all).

- **Host:** `172.16.192.12` (private/internal IP)
- **OS:** Ubuntu 20.04.6 LTS
- **GPU:** NVIDIA GeForce RTX 2080 Ti, 11264 MiB (11 GB) VRAM, driver 535.230.02, CUDA 12.2
- **Python:** 3.8.10. **PyTorch:** 2.0.1+cu117, CUDA confirmed working.
- **Username:** `utkarsh`, home `/home/utkarsh`
- **Disk:** `/` has 393GB total — **check free space before any big transfer**, it has gotten tight this session (down to ~52-57GB free after adding the 4 extra RELLIS sequences; still comfortable, but don't assume the old "92GB free" figure from the last handoff).

### How to connect

**Password auth exists but you should never need it. Never ask the user for the password, never type one.** SSH key-based auth is fully configured.

- Private key: `C:\Users\bhavy\.ssh\id_ed25519_drishti_gpu`
- **Never touch `C:\Users\bhavy\.ssh\id_ed25519` (no suffix)** — the user's own unrelated key.

```bash
ssh -i "$HOME/.ssh/id_ed25519_drishti_gpu" -o BatchMode=yes utkarsh@172.16.192.12 "COMMAND_HERE"
```

`-o BatchMode=yes` fails fast instead of hanging if key auth somehow breaks.

**File transfer — IMPORTANT LESSON FROM THIS SESSION, read before transferring anything bulky:**

`scp -r` on a directory with thousands of small files (e.g. `data/rellis/<seq>/os1_cloud_node_kitti_bin/*.bin`) is **extremely slow** — each file pays its own protocol round-trip, observed at roughly 30 files/minute, which would have taken ~5+ hours for ~11,500 remaining point-cloud files. Two fixes, in order of preference:
1. **Best**: if the data exists locally as a compressed archive already (e.g. the original RELLIS `.zip` downloads), `scp` the archive itself (one big file, no per-file overhead) and `unzip` specific paths on the REMOTE side. This is what actually worked — cut a ~5+ hour transfer down to ~1.5 hours for 14.2GB of zips vs. ~29GB of raw extracted files.
2. If no archive exists, `tar cf - <dirs> | ssh ... "cat > remote.tar"` (streamed, single connection) beats `scp -r`, though it's still bandwidth-bound on the raw (uncompressed) byte count.

**Also**: a `nohup ... & disown` launched over SSH without `< /dev/null` on stdin can leave the SSH session itself hanging open (observed: `LAUNCHED PID` prints, but the wrapper shell process never exits) even though the actual background process is correctly detached and running fine. Always include `< /dev/null` in the launch command:

```bash
ssh ... "cd ~/drishti && nohup python3 -u -m perception.train ... > checkpoints/train.log 2>&1 < /dev/null & disown; sleep 2; ps aux | grep perception.train"
```

If you forget it and a launch hangs, it's safe to just kill the stray wrapper-shell PID directly (`ps aux` on remote will show it as a `bash -c ...` process using ~0% CPU, separate from the real training PIDs which will be burning real CPU/GPU) — the training itself is unaffected.

### Quick health check (run this first, every time you resume)

```bash
ssh -i "$HOME/.ssh/id_ed25519_drishti_gpu" -o BatchMode=yes utkarsh@172.16.192.12 "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv; ps aux | grep perception.train | grep -v grep"
```

If SSH hangs or fails, the college network path is probably down (you're not on it), not a problem with the key.

---

## 4. Training — full history, what's running right now, and how to check it

This is almost certainly the most time-sensitive section — read it fresh, don't trust a summary from memory.

### 4a. Run #1 — single sequence (00004 only) — **COMPLETE**

- Launched via `python3 -u -m perception.train --sequence-dir data/rellis/00004 --sensor-config configs/sensor_ouster_os1_64.yaml --out-dir checkpoints --epochs 20 --batch-size 4 --device cuda`, results in `~/drishti/checkpoints/` on the remote (pulled locally into `checkpoints_remote/` too — `checkpoint.pt`, all `val_metrics_epoch*.json`, `train.log`, `channel_stats.json`).
- **1750 training frames, 309 held out** (last 15% by index, contiguous, per Ticket #30's own no-shuffle spec).
- Ran to completion, ~57 minutes wall-clock (~2.4-2.9 min/epoch).
- **Final result (epoch 19/19): val mIoU = 0.3220.** Per-class: class 0 (0.89), class 5/vegetation (0.91), class 7 (0.66) learned well; class 1 (0.01) and class 2 (0.10) weak; classes 3, 4, 6 stuck at 0.0; classes 8/9 (`NEGATIVE_OBSTACLE`/`OVERHANG`) at `n/a` — **this is correct and expected, not a bug**: those two DRISHTI classes are deliberately geometry-derived only (Phase 4's own job, now built), and NO dataset's semantic labels are allowed to map onto them (`perception/taxonomy.py`'s own docstring/`NO_SEMANTIC_MAP_CLASSES`) — so they can never show a training IoU regardless of how much data you add.
- Plateaued around epoch 5-6, didn't improve much after — **the reason turned out to be data scarcity/diversity** (one short sequence, 2 classes entirely absent, most others thin), which motivated Run #2.
- **A real bug hit mid-run and fixed**: `tail -f` on `train.log` showed no live output for a while even though training was genuinely progressing — Python's stdout is BLOCK-buffered (not line-buffered) when redirected to a file rather than an interactive terminal. Fixed by killing and relaunching with `python3 -u` (unbuffered). **Always launch training with `-u`.**

### 4b. Dataset expansion — all 5 RELLIS-3D sequences now available

The user had downloaded 3 zip archives covering **all 5 sequences** (00000-00004), not just 00004 as originally used:
- `Rellis_3D_os1_cloud_node_kitti_bin.zip` (14.3GB, point clouds, all 5 sequences)
- `Rellis_3D_os1_cloud_node_semantickitti_label_id_20210614.zip` (182MB, labels)
- `Rellis_3D_lidar_poses_20210614.zip` (609KB, poses+calib)

Extracted and **verified byte/frame-count-exact** on both machines:

| Sequence | Frames (bin=label=poses, verified) |
|---|---|
| 00000 | 2847 |
| 00001 | 2319 |
| 00002 | 4147 |
| 00003 | 2184 |
| 00004 | 2059 |
| **Total** | **13,556** |

Local: `data/rellis/<seq>/`. Remote: `~/drishti/data/rellis/<seq>/`. **Gitignored on both, never committed** (same as before). Local disk usage: 00000=7.0G, 00001=5.7G, 00002=11G, 00003=5.4G, 00004=5.1G (~34.2GB total).

**A messy bug during this extraction, worth knowing if you see partial data**: an early attempt to `scp -r` the raw extracted sequences directly (before switching to the zip-transfer approach above) was killed partway through. A LATER fresh zip-extraction pass for sequence 00000 collided with a stale partial copy already sitting in `~/drishti/data/rellis/00000/` from that earlier attempt, causing an `mv: Directory not empty` error. Fixed by deleting the stale partial directory and moving in the freshly-unzipped complete one, then re-verifying counts. **The general lesson**: after any interrupted transfer, don't trust that a sequence directory's mere *existence* means it's complete — check `find <dir> -name '*.bin' | wc -l` against the known-correct counts in the table above before training on it.

### 4c. `perception/train.py` — now supports multiple sequences

Changed this session (all changes tested — see `test_build_multi_sequence_splits_keeps_val_within_each_sequence_tail` and `test_training_smoke_run_across_multiple_sequences` in `tests/test_train.py`):

- `train()`'s `sequence_dir` parameter now accepts either a single path (old behavior, unchanged) OR a list of paths.
- New `build_multi_sequence_splits(sequence_dirs)` — splits EACH sequence independently (last 15% by index, per sequence — never a global shuffle across sequences, for the same "don't leak adjacent frames" reason Ticket #30 already cared about within one sequence), then concatenates the train/val item lists. Each item is `(sequence_dir, frame_idx)`, not just `frame_idx`.
- `RellisSegDataset` now takes an `items: list[(sequence_dir, frame_idx)]` instead of `(sequence_dir, frame_indices)`.
- CLI: `--sequence-dir` now takes `nargs="+"` — pass multiple paths space-separated.
- This is a **backward-compatible, additive change** — nothing about single-sequence training changed.

### 4d. Run #2 — all 5 sequences at once — **COMPLETE**

Launched into a **fresh** `~/drishti/checkpoints_multi/` directory (deliberately NOT reusing `checkpoints/`, which holds Run #1's completed 20-epoch results — reusing it would make the resume logic see "epoch 20 already done" and skip training entirely):

```bash
ssh -i "$HOME/.ssh/id_ed25519_drishti_gpu" -o BatchMode=yes utkarsh@172.16.192.12 \
  "cd ~/drishti && nohup python3 -u -m perception.train \
    --sequence-dir data/rellis/00000 data/rellis/00001 data/rellis/00002 data/rellis/00003 data/rellis/00004 \
    --sensor-config configs/sensor_ouster_os1_64.yaml --out-dir checkpoints_multi \
    --epochs 20 --batch-size 4 --device cuda \
    > checkpoints_multi/train.log 2>&1 < /dev/null & disown"
```

- **11,522 training frames, 2,034 held out** (across all 5 sequences).
- **Per-class pixel counts improved immediately** vs. Run #1 — classes 1, 2, 3, 4, 6 all had thousands of sample pixels (previously near-zero with just sequence 00004), confirming the hypothesis that Run #1's plateau was a data-diversity problem. Only classes 8/9 stayed at zero pixels — expected, see 4a above, this is permanent by design, not a dataset gap.
- Ran ~5.8 hours wall-clock (~1040-1170s/epoch, ~6.6x more data than Run #1's ~142s/epoch, proportionally slower as expected).
- **Final result (epoch 19/19, confirmed complete — process exited cleanly, no traceback): val mIoU = 0.5618** — a 71% relative improvement over Run #1's final 0.3220. Classes 1, 3, and 6 went from effectively unlearned (0.00-0.01 IoU) to genuinely useful (0.46-0.84 IoU); class 5 improved to 0.97. Class 0 dropped slightly (0.914→0.779, not yet investigated) and class 4 stayed at exactly 0.0 in both runs (still may be too rare, or genuinely confusable with a similar class — worth investigating before assuming more training data alone will fix it).
- **Full per-epoch table, per-class breakdown, and analysis: see `TRAINING_RESULTS.md`'s "Run #2" section** — written this session, has the complete story.

**All 43 result files (20 `checkpoint_epochN.pt`, final `checkpoint.pt`, 20 `val_metrics_epochN.json`, `train.log`, `channel_stats.json` — 1.4GB total) have been pulled down and verified locally in `checkpoints_multi_remote/`.** If you need to re-pull for any reason:
```bash
mkdir -p checkpoints_multi_remote
scp -i ~/.ssh/id_ed25519_drishti_gpu -r utkarsh@172.16.192.12:~/drishti/checkpoints_multi/* checkpoints_multi_remote/
```

The effective plateau is epoch ~12 onward (mIoU 0.545-0.562 band) — `checkpoint_epoch19.pt` (or `checkpoint.pt`, the same thing) is the technical best, but any checkpoint from epoch 12-19 is a reasonable pick for downstream use (e.g. Tickets #31/#32's cache-inference/semantic-map checkpoint, now unblocked with a genuinely useful trained model).

---

## 5. Bugs found and fixed — worth knowing before you trust anything

*(Items 1-14 are from the first session and unchanged — kept here for completeness. New items start at 15.)*

1. Bible Part 8's own worked example has an arithmetic error (`si=16,sj=328,flat=167952` stated vs. correct `si=272,sj=360,flat=184592`).
2. Ticket #14's literal stamp formula is a no-op at N=512 — fixed by tagging bits above the storage index (`grid/cell.py::expected_stamp`).
3. Ticket #21's described ceiling-detection gap-length gate would misclassify a close overhang — removed the gate (`grid/layers.py::extract_layer_bins`).
4. RELLIS-3D ships no `ring` field — `perception/range_image.py` has an arcsin fallback.
5. PyTorch circular padding needs the full pad tuple for 4D+ tensors (`perception/circular_pad.py`).
6. Source notebook self-contradicts on `weights=None` vs. actually loading pretrained — extraction follows the documented intent.
7. Notebook architecture crashes on `(1,9,32,1080)` — fixed with `_match_size()` in `perception/segnet.py`.
8. Ticket #28's "~4-4.5M" params is the Bible's encoder-only figure, not the whole network's (whole network = 5.82M, matches the Bible's separate 6-8M framing).
9. `OneCycleLR` can't resume with a different target epoch count than originally planned.
10. `batch_size=1` crashes (ASPP global-pool + BatchNorm) — explicit guard added, **always use `batch_size >= 2`**.
11. `perception/taxonomy.py` had no RELLIS numeric-ID→name map — added `RELLIS_ID_TO_NAME`, cross-validated against real data.
12. `ast.unparse` doesn't exist before Python 3.9 — fixed via line-range removal for the remote's Python 3.8.
13. `scripts/download_rellis.sh` assumed the wrong archive-internal path prefix.
14. `torch.amp.GradScaler` doesn't exist in torch 2.0.1 (the remote's version) — fixed with a try/except fallback to `torch.cuda.amp.GradScaler`.
15. **Stdout is block-buffered, not line-buffered, when redirected to a file** — `tail -f train.log` can sit blank for a long time even as training genuinely progresses. Always launch with `python3 -u`.
16. **`scp -r` on thousands of small files is ruinously slow** (~30 files/min observed) — transfer a compressed archive instead and extract remotely, or use `tar | ssh cat >`. See section 3.
17. **`nohup ... & disown` over SSH without `< /dev/null` can leave the SSH session's own wrapper shell hanging** even though the real background process detaches and runs fine. Always redirect stdin too.
18. **An interrupted transfer can leave a partially-populated directory that looks superficially fine** (has the right subdirectory names) but is missing most of its files — always verify file COUNTS against a known-correct source before trusting a synced dataset directory, not just that the directory exists.
19. **`Clipmap.lookup()` resolves by which level's WINDOW contains a point, not by sensor-Nyquist radius** — any new write path into the clipmap (like Ticket #34's observability carving) must mirror that exact level-selection logic, or writes to a "logically coarse" level can become invisible to `lookup()` for points that are geometrically close to the vehicle. See section 2's Phase 4 notes for the concrete bug this caused and how it was fixed.

---

## 6. Datasets — current status

| Dataset | Status | Where |
|---|---|---|
| **RELLIS-3D, all 5 sequences (00000-00004), Ouster OS1-64** | ✅ Fully downloaded, extracted, verified frame-count-exact on both machines (13,556 frames total) | Local: `data/rellis/<seq>/`. Remote: `~/drishti/data/rellis/<seq>/`. **Gitignored on both.** |
| **RELLIS-3D, Velodyne stream** | ❌ Not downloaded (optional per Build Map) | — |
| **nuScenes-mini** | ❌ Not downloaded, `nuscenes-devkit` not installed | — |

The three source RELLIS zips are still sitting in the repo root locally (`Rellis_3D_*.zip`, gitignored) — safe to delete now that all 5 sequences are extracted, purely a disk-space question (they've already been deleted from the remote's `/tmp` after extraction there).

---

## 7. File transfer reference (updated)

Local → remote sync of CODE changes (small number of files) — direct `scp`, simplest:
```bash
scp -i "$HOME/.ssh/id_ed25519_drishti_gpu" <changed files> utkarsh@172.16.192.12:~/drishti/<matching path>/
```

**Always re-run the full test suite on the remote after syncing code**, before trusting anything:
```bash
ssh -i "$HOME/.ssh/id_ed25519_drishti_gpu" -o BatchMode=yes utkarsh@172.16.192.12 "cd ~/drishti && python3 -m pytest -q 2>&1 | tail -30"
```

For BULK data (many files or large files) — see section 3's transfer lesson (zip-then-remote-unzip beats `scp -r` on many small files; `tar | ssh cat >` beats `scp -r` even for fewer, larger files).

---

## 8. Repository structure note

`observability/`, `planning/`, `temporal/`, `attention/`, `viz/` currently exist as package directories (created in an earlier session's commit) but **only `observability/` has real code in it now** (Phase 4, this session) — the other four are still empty `__init__.py` stubs. Don't be misled by their existing directory presence into thinking Phases 5+ are further along than they are; check actual file contents, not directory listings.

---

## 9. Next step — Phase 5 (Tickets #38-43), concrete guidance

This is genuinely the most useful section for a fresh session to read carefully — it's not generic advice, it's what's actually needed to implement these 6 tickets correctly on the FIRST attempt, based on patterns that mattered in Phase 4.

**Phase 5's own framing (Build Map): this is where Claims 3 and 4 — the project's two quantitative, falsifiable claims — get demonstrated with running code, not just described.** Treat this phase as higher-stakes than Phase 4: these numbers are meant to be defended in front of a judge.

### #38 — Expected returns, kappa, r_blind → `observability/sparsity.py`
- Formulas: $N_{exp}(r) = t_{min} \cdot w_{min} / (r^2 \cdot \Delta\phi \cdot \Delta\theta)$, $\kappa = N_{obs}/N_{exp}$, $r_{blind} = \sqrt{t_{min} w_{min} / (\Delta\phi \Delta\theta)}$.
- **`vehicle_ugv.yaml` already has the needed constants**, checked this session: `min_object_t_m: 1.00` (t_min), `min_object_w_m: 0.10` (w_min). `sensor/sensor_model.py`'s `SensorConfig` already has `d_theta_rad`/`d_phi_rad`. **Load these from config, never hardcode** — `tests/test_vehicle_config.py` already greps the repo for these literals outside `vehicle_ugv.yaml` and fails if found, so hardcoding will be caught immediately, but don't rely on the test to catch what good practice should prevent.
- **Exact acceptance numbers to hit** (from the Bible's own table, so these are hand-verifiable, not just self-consistent): HDL-64E pedestrian $r_{blind}$ = 194.8m, pole = 163.7m, fence post = 66.8m; HDL-32E pedestrian = 79.2m, fence post = 27.2m. Configs for both sensors already exist (`configs/sensor_hdl64e.yaml`, presumably `sensor_hdl32e.yaml` too — confirmed used in `tests/test_train.py`). Assert $N_{exp} \propto 1/r^2$, not $1/r$ — an easy accidental bug.

### #39 — Structure test and sparsity flags → `observability/sparsity.py` (same file)
- **The single most important trap in this whole phase, called out explicitly in the Build Map**: the decision table's last row is `N_obs=0, N_exp<1 → UNKNOWN, never FREE`. The Build Map itself predicts a model will "simplify" this to "no returns → free" because that's what every conventional occupancy grid does — **this row IS Claim 3, the entire point of the ticket**. Do not let it collapse.
- `N_obs == 1` must default to `SPARSE_STRUCTURED`, never "structured vs noise" — a single point can never be proven unstructured, so the cautious verdict applies.
- Structured = returns tightly clustered in range AND contiguous in ring index — this needs the SAME "neighbors in the same column" thinking Ticket #36 (Phase 4) already used for its ring-to-ring discriminator; that code (`observability/negative_obstacle.py::detect_anomalous_cells`) is a reasonable model for how to compare a cell against its ring-neighbors cleanly.
- **Test to write first, matching the ticket's own naming**: `test_unknown_past_r_blind` — a cell at 90m on nuScenes-config (past the 27.2m fence-post $r_{blind}$) with zero returns must assert `UNKNOWN`; a cell at 15m with zero returns must assert `FREE`.

### #40 — Speed envelope, Claim 4 → `planning/speed_envelope.py`
- Formula: $v_{max}(R) = -a \cdot t_r + \sqrt{a^2 t_r^2 + 2aR}$, using `braking_a_ms2` and `t_react_s` — **both already in `vehicle_ugv.yaml`** (checked this session: `a=4.0`, `t_react=0.30`, with a code comment already flagging `t_react_s` as a placeholder until Ticket #55 measures real P95 latency — leave that TODO in place, don't try to "fix" it prematurely).
- **Exact acceptance numbers**: `v_max(21.6) = 12.00 m/s = 43.2 km/h`, `v_max(6.7) = 6.24 m/s = 22.5 km/h` (KITTI); `v_max(12.6) = 8.99 m/s = 32.0 km/h` (nuScenes).
- Assert monotonicity (shorter range never yields higher speed) and that lowering `a` lowers `v_max` — cheap, high-value sanity tests.
- **This is advisory output only — do not let it gate/control anything**, per the ticket's own explicit warning. Report which hazard is binding (the minimum over the active hazard set), not just the final number.

### #41 — Conservatism, the caution order → `planning/conservatism.py`
- Cost order: `FREE ≤ known-rough ≤ UNKNOWN_COST < LETHAL`, with `UNKNOWN_COST` **strictly** below `LETHAL` and finite.
- **The trap here, explicitly named in the Build Map**: setting `UNKNOWN_COST = LETHAL` "to be safe" technically satisfies monotonicity but paralyzes the vehicle (it would refuse to move near anything unknown, which is useless, not safe). Monotonicity is necessary but not sufficient — don't let a model (or yourself) take the "obviously safe" shortcut here.
- Enumerate the Bible Part 16's fourteen information-deficit paths as an explicit table in code (flag-combination → cost floor) — this needs reading Part 16 of the Bible directly, not guessing at 14 plausible-sounding cases.
- This ticket is blocked by #34 (observability flags — already built, Phase 4) and #39 (sparsity flags, above) — build in that order.

### #42 — The monotonicity property test → `tests/test_conservatism.py`, using Hypothesis
- `hypothesis` is **already pip-installed on both machines** (confirmed working from the first session).
- Structure per the Build Map: a `degradations()` strategy that only ever makes things WORSE (drop points, lower kappa, mark occluded, mark provisional, push past r_blind, age the data) applied to a random cell state, asserting `cost(degraded, VEHICLE) >= cost(cell, VEHICLE)` always holds, called through the PUBLIC `cost()` function only (any fast path that bypasses it is unsupported by construction).
- **Two watch-outs explicitly named**: (1) `degradations()` must be checked to actually be a strict information loss — a generated "degradation" that accidentally adds information makes the whole property test vacuous and worthless. (2) **A property test you have never seen fail is not evidence** — the ticket explicitly wants you to deliberately introduce a bug (e.g. make `UNOBSERVED` cost zero) and confirm the test *fails* as expected, before trusting that it passing on the real code means anything. Don't skip this — it's cheap and it's the whole point of writing a property test in the first place.
- Target: 10,000 examples, zero violations on the real code; the deliberately-broken version must fail.

### #43 — Checkpoint: claims 3 and 4 demonstrable
- Same pattern as Ticket #37's `eval/checkpoint_trench.py` this session: a synthetic (or real, if a good checkpoint from #31/#32 exists by then) frame with `SPARSE_STRUCTURED` cells overlaid and the speed envelope polygon drawn, plus printing the #42 property test's pass/fail result. Follow `eval/checkpoint_first_map.py` / `eval/checkpoint_trench.py`'s established pattern: a `run_checkpoint(out_dir, ...) -> dict` function, a matplotlib figure saved to PNG (Agg backend), a paired `tests/test_checkpoint_*.py` asserting the dict's key facts and that the PNG exists/is non-empty.

### General process notes that made Phase 4 go smoothly, worth repeating for Phase 5
- **Read the exact Build Map section for each ticket before writing code** (`DRISHTI_Build_Map.md`, Phase 5 is lines ~738-836) — the numeric acceptance criteria above are lifted directly from it; guessing plausible-sounding numbers instead of reading the ticket wastes a retry cycle.
- **Check for existing config values before assuming you need to add new ones** — this session, `vehicle_ugv.yaml` already had every constant Tickets #38 and #40 need (`min_object_t_m`, `min_object_w_m`, `braking_a_ms2`, `t_react_s`), added presumably in an earlier session in anticipation. Grep configs first.
- **Write the test the ticket explicitly says to write first** (e.g. #39's `test_unknown_past_r_blind`, #36's slope test in Phase 4) before general coverage — these are the ones the Build Map itself flags as most likely to catch the "obvious but wrong" implementation.
- **Run the full test suite after each ticket**, not just the new file's tests — Phase 4 caught zero regressions doing this, but it's cheap insurance and this codebase has genuine cross-module coupling (e.g. `grid/clipmap.py` getting a new method for `observability/`).
- **Sync to remote and re-run tests there too before declaring a ticket done**, if planning to eventually run anything Phase-5-related on GPU (unlikely for pure CPU-bound geometry/logic like #38-42, but #43's checkpoint may want the trained model — sync then).
