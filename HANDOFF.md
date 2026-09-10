# DRISHTI — Session Handoff

**Written:** 2026-09-10, end of a single long build session.
**Purpose:** let a fresh Claude Code session (or a human) pick up exactly where this one left off, with zero prior context, and get training running.

Read `DRISHTI_Project_Bible_v3.md` (the WHY) and `DRISHTI_Build_Map.md` (the WHAT/WHEN, ticket-by-ticket) first if you haven't — this doc assumes you have. Everything below is keyed to Build Map ticket numbers.

---

## 1. Where the build actually stands

**Phases 0–3 (Tickets #1–#30) are built, tested, and verified — locally AND on the actual GPU training machine, both confirmed clean.** Phase 4 onward (observability, negative obstacles, sparsity, temporal fusion, fovea, traversability, dashboard, eval harness — Tickets #33+) is **not started**.

| Phase | Tickets | Status |
|---|---|---|
| 0 — Foundations, sensor model, the gate | #1–#9 | ✅ done (built in an earlier session, verified this session) |
| 1 — The Clipmap | #10–#16 | ✅ done this session |
| 2 — Cells, multi-layer, scatter | #17–#22 | ✅ done this session (#17 deliberately deferred, per Build Map's own allowance) |
| 3 — Perception (range image → training) | #23–#30 | ✅ done this session |
| 4 onward | #31–#68 | ❌ not started |

**Test counts — fully verified in both environments, at the end of this session, after fixing every cross-environment bug found (section 4):**
- Local (this Windows laptop, Python 3.14.3, torch 2.11.0): **122 passed, 2 skipped**
- Remote (the GPU server, Python 3.8.10, torch 2.0.1+cu117): **122 passed, 2 skipped** — identical, confirmed by actually running it there, not assumed.

The 2 skips in both places are the same pre-existing gap: `nuscenes-devkit` is not installed anywhere (not locally, not remotely) and no nuScenes-mini data has been downloaded. This does not block anything built so far — see section 5.

---

## 2. What was actually built this session, ticket by ticket

### Phase 1 — The Clipmap (#10–#16)
- `grid/addressing.py` — world↔index, toroidal wrap, scalar + vectorised (torch) variants.
- `grid/cell.py` — fixed-point height encode/decode, observability flags, stamp tag, class_conf byte, NO_CEILING sentinel.
- `grid/clipmap.py` — the `Clipmap` class: SoA allocation, scroll/clear-on-scroll, stamp validation, `lookup()`.
- Tests: `tests/test_addressing.py`, `tests/test_nesting_exactness.py`, `tests/test_clipmap.py`.

### Phase 2 — Cells and the Overhang Claim (#18–#22)
- `grid/scatter.py` — vectorised height scatter (`scatter()`) + class-mode scatter (`scatter_class()`).
- `grid/histogram.py` — 8-bin per-cell height histogram, bin width derived from `vehicle_ugv.yaml`'s `min_clearance_m`.
- `grid/layers.py` — ground/gap/ceiling extraction (`extract_layer_bins()`, `scatter_layers()`) — the machine-checked proof of **Claim 1**.
- `eval/checkpoint_first_map.py` — Ticket #22's integration checkpoint (synthetic scene, since no real nuScenes data — see section 5).
- Tests: `tests/test_scatter.py`, `tests/test_class_agg.py`, `tests/test_histogram.py`, `tests/test_layers.py`, `tests/test_checkpoint_first_map.py`.

### Phase 3 — Perception (#23–#30)
- `perception/range_image.py` — spherical projection + occlusion-depth channels (Tickets #23+#24, built together per the Build Map's own "same pass" requirement).
- `perception/circular_pad.py` — horizontal-only circular padding utility (#25).
- `perception/ground_prior.py` — column-wise incremental ground walk (#26) — labels, never strips.
- `perception/input_tensor.py` — assembles the 9-channel network input tensor, saved normalisation stats (#27).
- `perception/segnet.py` — **`FusionSegNet`**, extracted from the user-supplied `FusionSegNet_v5 (1).ipynb` and adapted (#28).
- `perception/losses.py` — Lovász-Softmax + confidence-weighted CE + deep-supervision aux (#29).
- `perception/train.py` — **the actual training script** (#30): `RellisSegDataset`, resumable checkpointing, AdamW + OneCycleLR + AMP, per-class IoU validation.
- `perception/taxonomy.py` — added `RELLIS_ID_TO_NAME` (was missing — see section 4) and `rellis_label_ids_to_drishti()`.
- `perception/rellis_loader.py` — added `load_rellis_labels()` (was missing — nothing previously read `.label` files at all).
- Tests: `tests/test_range_image.py`, `tests/test_circular_pad.py`, `tests/test_ground_prior.py`, `tests/test_input_tensor.py`, `tests/test_segnet.py`, `tests/test_losses.py`, `tests/test_train.py`.

### Also this session
- `scripts/download_rellis.sh` — fixed a real extraction-path bug (see section 4).
- `colab_download_rellis_os1.ipynb`, `colab_download_rellis_vel.ipynb` — standalone Colab notebooks for the two RELLIS streams (14GB Ouster, 5.58GB Velodyne). Not used in the end — the user downloaded the zips directly and we extracted locally instead.
- `data/rellis/00004/` — the real, extracted RELLIS-3D sequence (see section 5). **Gitignored, not in git.**

---

## 3. The GPU — full detail on what it is and how to reach it

### What it is
A college lab machine, reached over the college network (not the public internet — you must be on that network for SSH to reach it at all).

- **Host:** `172.16.192.12` (private/internal IP)
- **OS:** Ubuntu 20.04.6 LTS, hostname `ubuntu-Standard-PC-Q35-ICH9-2009`
- **GPU:** NVIDIA GeForce RTX 2080 Ti, **11264 MiB (11 GB) VRAM**, driver 535.230.02, CUDA 12.2
- **Was idle** the whole session (0% util, ~6 MiB used) — nobody else appeared to be using it, but it's shared lab hardware, so re-check before assuming it's free (command below).
- **Python:** 3.8.10 (system python3) — noticeably older than this laptop's 3.14.3. Caused two real bugs this session (section 4, items 12 and 14) — treat "works locally" as unproven for anything torch/ast-version-sensitive until it's actually run there.
- **PyTorch:** 2.0.1+cu117, **CUDA confirmed working** (`torch.cuda.is_available()` → `True`).
- **torchvision:** 0.15.2+cu117 — already present, did not need installing.
- **Disk:** `/` has 393GB total, ~92GB free at last check.
- **Username:** `utkarsh`
- **Home dir:** `/home/utkarsh`

### How to actually connect — the important part

**Password auth is set up but you should never need it again.** SSH key-based auth is already configured and working. Do not ask the user for the password — it isn't needed, and this project's own policy is that Claude must never type a password into anything anyway. If key auth somehow stops working, that's a problem to raise with the user, not solve by asking for the password again.

**The key:**
- Private key: `C:\Users\bhavy\.ssh\id_ed25519_drishti_gpu` (on this Windows laptop)
- Public key: already appended to `~/.ssh/authorized_keys` on the remote server.
- **Never touch `C:\Users\bhavy\.ssh\id_ed25519` (no suffix)** — that's the user's own pre-existing key for something else (likely GitHub), unrelated to this project. The `_drishti_gpu` suffix is what makes it ours.

**The exact connect command** (works from Bash/Git Bash; same idea in PowerShell with backslash paths):

```bash
ssh -i "$HOME/.ssh/id_ed25519_drishti_gpu" -o BatchMode=yes utkarsh@172.16.192.12 "COMMAND_HERE"
```

`-o BatchMode=yes` makes it fail fast instead of hanging if key auth somehow breaks, rather than silently sitting at a password prompt forever.

**File transfer** (same key, standard `scp`):

```bash
scp -i "$HOME/.ssh/id_ed25519_drishti_gpu" <local_path> utkarsh@172.16.192.12:<remote_path>
```

**No `rsync` is available** on this laptop's Git Bash — use `scp` or `tar` + `scp` for bulk transfers (see section 6).

### What's on the remote server right now

- **`~/drishti/`** — the full project repo, as a **plain file copy** (NOT a git clone — `git status` there fails with "not a git repository"). Transferred via `tar` + `scp`, not `git clone`. **Confirmed fully in sync with local as of the end of this session** (122/122 passed, identical) — but nothing auto-syncs. If you make more local changes, `scp` them over manually and re-run the remote test suite before trusting anything.
- **`~/drishti/data/rellis/00004/`** — the real RELLIS-3D sequence, 2059 frames, points + labels + poses, identical to the local copy (see section 5).
- **`~/drishti_transfer.tar.gz`** — the original transfer archive. Safe to delete once you've confirmed `~/drishti/` is current; just disk space otherwise.
- **No `~/drishti/checkpoints/` yet** — training has not actually been run for real. Only tiny CPU smoke tests (a handful of fake frames, 1-2 epochs, both locally and on the remote) have run, to verify the training loop's mechanics work. **The real multi-epoch GPU training run has not happened yet** — that's the very next thing to do.
- pip packages installed with `--user` on top of whatever was already there: `pytest`, `hypothesis`, `pyyaml`, `matplotlib`.

### Quick health check (run this first, every time you resume)

```bash
ssh -i "$HOME/.ssh/id_ed25519_drishti_gpu" -o BatchMode=yes utkarsh@172.16.192.12 "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv"
```

If that hangs or fails, the college network path is probably down (you're not on it), not a problem with the key.

---

## 4. Bugs found and fixed this session — worth knowing before you trust anything

These were all caught by actually testing against real data/real environments, not assumed. A fresh session should know the *reasoning*, not just that a fix exists — several of these reveal real gaps in the Build Map / Bible / source notebook themselves.

1. **Bible Part 8's own worked example has an arithmetic error.** States `si=16, sj=328, flat=167952` for a specific query point; independently verified the correct values are `si=272, sj=360, flat=184592`. My code matches the correct math; the document doesn't. Worth fixing in the Bible text before it's in front of a judge.
2. **Ticket #14's literal stamp formula is a no-op at N=512.** `(i&0xFF)<<8 | (j&0xFF)` uses bits already fully determined by the storage index at this array size (256 divides 512), so it can never detect a missed clear. Fixed by tagging bits *above* the storage index instead. See `grid/cell.py::expected_stamp`'s docstring.
3. **Ticket #21's described ceiling-detection algorithm (gate on a sufficient gap length) would misclassify a close overhang as "no ceiling" (infinite clearance) instead of reporting its real, insufficient clearance** — verified against all four of the Build Map's own test cases before fixing. See `grid/layers.py::extract_layer_bins`'s docstring.
4. **RELLIS-3D ships no `ring` field** (unlike what Ticket #23 assumed "both datasets have it") — `perception/range_image.py` has an arcsin-based elevation fallback for when `ring < 0`.
5. **PyTorch's circular padding needs the full pad tuple matching tensor rank**, not just `(left, right)`, for 4D+ input. Fixed in `perception/circular_pad.py`.
6. **The source notebook (`FusionSegNet_v5 (1).ipynb`) contradicts itself**: docstring says `weights=None` (from scratch), code actually loads `EfficientNet_B0_Weights.DEFAULT` (ImageNet pretrained). Extraction follows the documented intent (from scratch), not the notebook's actual code — see `perception/segnet.py`'s docstring.
7. **The unmodified notebook architecture crashes on this project's own input shape** `(1,9,32,1080)` — 5 stride-2 downsamples floor an odd intermediate width, so the decoder's upsampling lands one pixel narrower than its skip connection. Fixed with `_match_size()` in `perception/segnet.py`.
8. **Ticket #28's stated parameter count ("~4-4.5M") is actually Bible Part 5.3's *encoder-only* figure**, not the whole network's. The whole network measures 5.82M, correctly matching Bible Part 5.1's separate "~8M, competitive band 6-7M" framing. Don't be alarmed if a future check sees ~5.8M and worries it's wrong against Ticket #28's text — it isn't.
9. **`OneCycleLR` cannot be resumed with a *different* target epoch count than originally planned** — its schedule is tied to a fixed total-step count baked in at construction, and `load_state_dict` restores that total. `perception/train.py`'s resumability is designed around "same target, resume after a crash," not "extend the plan mid-run."
10. **Training with `batch_size=1` crashes** — ASPP's global-average-pool branch collapses spatial size to 1x1, and BatchNorm2d can't compute training-mode statistics from a single value per channel. `perception/train.py` now raises a clear error instead of letting this surface as a cryptic PyTorch exception. **Always use `batch_size >= 2` for training** (eval/inference with batch_size=1 is fine).
11. **`perception/taxonomy.py` had a RELLIS name→DrishtiClass map but NO numeric-ID→name map at all** — meaning nothing could actually decode a real `.label` file before this session. Added `RELLIS_ID_TO_NAME`, cross-checked against real downloaded data (the exact ID set observed in 20 real frames — `{0,3,4,8,17,19,27,33,34}` — matches the published ontology precisely).
12. **`ast.unparse` doesn't exist before Python 3.9** — broke a structural test on the GPU server's Python 3.8. Fixed to use line-range removal via `lineno`/`end_lineno` instead (available since 3.8).
13. **`scripts/download_rellis.sh` assumed archive-internal paths start with `00004/`** — the real archives are all prefixed `Rellis-3D/00004/...`. Confirmed against the actual downloaded zips and fixed.
14. **`torch.amp.GradScaler` (the modern, non-deprecated device-agnostic API) doesn't exist in torch 2.0.1** — the version actually installed on the GPU training machine. Fixing the deprecation warning locally (newer torch) broke real training on the remote (older torch) entirely — it would have crashed on the very first call. Fixed with a try/except fallback to `torch.cuda.amp.GradScaler` in `perception/train.py`. **This is exactly the kind of bug that only shows up by actually running on the target machine** — worth remembering before "fixing" any other deprecation warning without re-testing remotely.

---

## 5. Datasets — what you actually have

| Dataset | Status | Where |
|---|---|---|
| **RELLIS-3D, sequence 00004, Ouster OS1-64 stream** | ✅ Fully downloaded, extracted, verified against real data (points/labels/poses counts all match: 2059/2059/2059) | Local: `data/rellis/00004/`. Remote: `~/drishti/data/rellis/00004/`. **Gitignored on both — never committed.** |
| **RELLIS-3D, Velodyne stream** | ❌ Not downloaded (optional per Build Map) | — |
| **nuScenes-mini** | ❌ Not downloaded. `nuscenes-devkit` not installed anywhere. | — |

**Important framing from earlier in this session, worth restating:** RELLIS-3D is actually *preferred* for training per Ticket #30's own spec ("Train on RELLIS-3D sequence 00004 if #3 landed; otherwise nuScenes-mini") — nuScenes is the fallback, not the primary. Skipping nuScenes costs you the Ticket #59 portability ablation's strongest form (two genuinely different real sensors) and one specific demonstrated finding about nuScenes' 32-beam resolution — but it does not block training or anything built so far.

The three source RELLIS zip files (14GB Ouster cloud, 174MB labels, 174MB poses) are sitting in the repo root locally (`Rellis_3D_*.zip`) — gitignored, safe to delete now that `data/rellis/00004/` has what's needed, purely a disk-space question.

---

## 6. How the GPU transfer was actually done (for reference, if you need to re-sync)

No `rsync` available locally, so: `tar -czf` (excluding `.git`, caches, the source zips, `eval/out/`) → `scp` the ~2GB archive → `ssh ... tar -xzf` on the remote. This is a one-way, manual push — nothing watches for changes.

**A real gotcha from this session, worth internalising:** the big archive transfer was kicked off early, then MORE files kept getting built locally afterward (Tickets #27-30 and several bugfixes) without re-syncing — so the remote silently fell behind for a while, and a "run the tests on the remote" check would have been misleadingly reassuring (it ran, and passed, but only because it was running a smaller, stale set of tests that didn't include the new files at all). **Always check file counts/timestamps match, not just "pytest passed," when trusting a remote copy is current.** For a handful of changed files, direct `scp` is simpler than rebuilding the whole archive:

```bash
scp -i "$HOME/.ssh/id_ed25519_drishti_gpu" <changed files> utkarsh@172.16.192.12:~/drishti/<matching path>/
```

**Always re-run the full test suite on the remote after syncing**, before trusting anything:

```bash
ssh -i "$HOME/.ssh/id_ed25519_drishti_gpu" -o BatchMode=yes utkarsh@172.16.192.12 "cd ~/drishti && python3 -m pytest -q 2>&1 | tail -30"
```

---

## 7. The actual next step

**Nothing has been trained yet.** Everything above is infrastructure and verified-correct code, confirmed working end-to-end on the actual training machine. The next action is:

1. Run a **short 1-epoch validation run on real GPU hardware** first (not yet done) — to measure real per-epoch timing (unknown — CPU smoke tests don't tell you this) and sanity-check loss/IoU output before committing to a longer run.
2. Based on that timing, decide a realistic epoch count (Build Map suggests ~15-20 epochs as a reasonable target, per Ticket #30, but this should follow from measured timing, not be assumed).
3. Launch the real run, e.g.:

```bash
ssh -i "$HOME/.ssh/id_ed25519_drishti_gpu" -o BatchMode=yes utkarsh@172.16.192.12 \
  "cd ~/drishti && python3 -m perception.train --sequence-dir data/rellis/00004 --sensor-config configs/sensor_ouster_os1_64.yaml --out-dir checkpoints --epochs 1 --batch-size 4 --device cuda"
```

(Run this as a background task and monitor — it's a genuinely long-running job. `perception/train.py` has no `if __name__` guard issue running as `-m`, confirmed by the smoke tests calling `train()` directly, but the actual `python3 -m perception.train ...` CLI invocation itself has not been separately smoke-tested — worth a quick dry check, e.g. `--epochs 1` on a tiny manually-copied subset first, if being extra cautious.)

4. **Before burning real GPU hours**, note `perception/train.py` prints a per-class pixel-count sanity check up front, with an explicit warning if any class has zero training pixels. In the tiny fake-data smoke tests this session, most classes showed zero pixels (expected — the fake data only used 3 label IDs). On the REAL 2059-frame sequence this has not been checked yet — do this first, on the real data, before assuming a full run is worthwhile as configured.
5. Report results with **per-class IoU, not just mean** (Ticket #30's own requirement), and state the training-set size explicitly — both already built into `perception/train.py`'s output and its saved `val_metrics_epoch*.json` files.

One open config question worth deciding, not yet resolved: `configs/sensor_ouster_os1_64.yaml`'s `d_theta_rad`/`d_phi_rad` are flagged elsewhere in this codebase as **not yet measured against real data** (Ticket #6's "the gate" was only ever run against synthetic grids and against the HDL-32E/HDL-64E configs, never against a real Ouster OS1-64 sweep). Training will run fine regardless (the network doesn't care what these values are), but any downstream claim about this specific sensor's derived hazard ranges would rest on an unverified config until that gate is actually run against `data/rellis/00004`.
