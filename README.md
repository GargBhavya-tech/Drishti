# DRISHTI

Adaptive variable-resolution 2.5D LiDAR mapping for dynamic environment perception.
SIH 2026 — DRDO Problem Statement 26053. Team Phir Hera Pheri.

This repo is a companion to `DRISHTI_Project_Bible_v3.md` (why) and
`DRISHTI_Build_Map.md` (what, in ticket order). Read those first — this
README only orients you inside the code.

## Status

**Phase 0 (Tickets #1–#9) is complete.** See `DRISHTI_Build_Map.md` for what each ticket covers.

| Ticket | What | Where |
|---|---|---|
| #1 | Repo skeleton, Colab environment | this layout, `colab_setup.ipynb` |
| #2 | nuScenes-mini loader → canonical `Sweep` | `perception/nuscenes_loader.py` |
| #3 | RELLIS-3D loader + background download script | `perception/rellis_loader.py`, `scripts/download_rellis.sh` |
| #4 | `SensorModel` — the seven formulas | `sensor/sensor_model.py`, `configs/sensor_*.yaml` |
| #5 | Resolution schedule generator | `sensor/schedule.py` |
| #6 | 🚨 THE GATE — point-distribution validation | `eval/point_distribution.py` |
| #7 | Verify mount height from calibration | `sensor/calibration.py` |
| #8 | Taxonomy remap | `perception/taxonomy.py` |
| #9 | Vehicle config | `configs/vehicle_ugv.yaml`, `sensor/vehicle_config.py` |

**#6/#7 status, updated 2026-09-11:** RELLIS-3D (Ouster OS1-64) is now downloaded locally, and the gate (`eval/point_distribution.py`) has been run against it for real — see `eval/measure_ouster_config.py`, which measures `d_theta`/`d_phi`/`phi_max`/`h_m` from real data (not synthetic) and writes the result into `configs/sensor_ouster_os1_64.yaml`. Two real findings came out of that: (1) the sensor constants were significantly wrong as placeholders — real azimuth resolution is 2048 columns/rev, not the assumed 1024, and the real vertical FOV is +17.0/-16.4° (33.5° span), not ±22.5° (45°); (2) the gate's own within/between-ring measured-vs-predicted plot does *not* cleanly validate on this off-road dataset the way it does on flat urban scenes — diagnosed and documented in that script's module docstring (off-road terrain decorrelates a ring's return range from azimuth, which the gate's range-then-ring binning assumes doesn't happen; this is a methodology caveat for cluttered natural terrain, not evidence the measured constants are wrong — they're independently cross-validated to ~1e-6° agreement across all 5 sequences). **The nuScenes/HDL-32E half of #6/#7 remains open** — nuScenes-mini still hasn't been downloaded in this environment, so `sensor_hdl32e.yaml`'s `h_m: 1.84` is still an un-measured placeholder. Once nuScenes-mini is available, call `validate_point_distribution()` and `cross_check_mount_height()` against real sweeps and look at `eval/out/sensor_validation.png` yourself — do not treat the synthetic-data tests as having already cleared that half of the gate.

**#8 status, updated 2026-09-11:** the RELLIS-3D numeric ID cross-check has been substantially expanded — from 20 frames in one sequence to 783 frames sampled across all 5 local sequences, confirming 17 of 20 mapped IDs against real data (up from 9), with zero surprise/unmapped IDs found (see `perception/taxonomy.py`'s module docstring for the full list). Three IDs remain unconfirmed — `dirt` (1), `sky` (7), `building` (12) — the last two plausibly because these 5 sequences' specific routes never pass a building, and LiDAR may never produce a `sky` return at all (no physical surface to reflect off), which would make that one a structural non-gap rather than a data gap. An authoritative `ontology.yaml` still hasn't been found; `perception/taxonomy.py`'s RELLIS dict stays keyed by name for the same reason as before.

**Status, updated 2026-09-11 — this line is well past Phase 0 now:** Phases 1–4 (the clipmap, cells, perception/training, observability/negative obstacles — Tickets #10–#37) are built and tested, plus Ticket #17 (foveated height quantum) and Tickets #31/#32 below. See `HANDOFF.md` for the full ticket-by-ticket build log; Phase 5 onward (Tickets #38+, the sparsity trap and speed envelope) is not yet started.

**#31 (cache inference), done 2026-09-11:** `eval/cache_inference.py` runs the trained checkpoint (`checkpoints_multi_remote/checkpoint.pt`, epoch 19, val mIoU 0.562 — see `TRAINING_RESULTS.md`) over real RELLIS-3D frames **offline** and saves per-point predicted labels to `.npy`, one file per frame, plus a `manifest.json` recording measured latency (P50/P95, ~500ms/frame on CPU at this project's real 64×2048 resolution). **These cached labels are precomputed, not live** — nothing in this repo runs the segmentation network in real time, and any demo or checkpoint using them should say so explicitly, per the Build Map's own warning against presenting cached labels as live inference.

**#32 (semantic map checkpoint), done 2026-09-11:** `eval/checkpoint_semantic_map.py` renders one real RELLIS-3D frame's 2.5D map twice — once coloured by ground truth, once by #31's cached predictions — side by side, using the frame's own recorded pose (world-anchored, not an assumed origin). Deviates from the literal ticket text (which compares against Ticket #22's synthetic scene) by using real data throughout instead, now that real data and a real trained checkpoint both exist locally — see that module's own docstring for the full reasoning.

## Layout

```
sensor/         SensorModel, resolution schedule (#4, #5)
perception/     dataset loaders → canonical Sweep struct (#2, #3)
grid/           clipmap, addressing, cells                  [not yet built]
observability/  ray traversal, four-state grid, negative obstacles [not yet built]
temporal/       static accumulation, motion detection        [not yet built]
attention/      TTC fovea controller                         [not yet built]
planning/       speed envelope, conservatism, cost map        [not yet built]
viz/            rerun dashboard                               [not yet built]
eval/           metrics, baselines, latency                   [not yet built]
configs/        sensor_*.yaml, vehicle_ugv.yaml
tests/          pytest suite, one file per module
scripts/        one-off / download scripts, not imported by the package
```

## Setup

Colab:

```
!git clone <this-repo-url> drishti
%cd drishti
!pip install -q -r requirements.txt
!pytest -q
```

Local:

```
pip install -r requirements.txt
pytest -q
```

## Running the tests

```
pytest -q                          # everything built so far
pytest -q tests/test_sensor_model.py
pytest -q tests/test_schedule.py
pytest -q tests/test_nuscenes_loader.py   # needs nuScenes-mini on disk, see below
```

## nuScenes-mini

`tests/test_nuscenes_loader.py` needs the nuScenes-mini devkit data.
Download it from https://www.nuscenes.org/download (the "Mini" split, ~4 GB)
and point `NUSCENES_DATAROOT` at the extracted folder:

```
export NUSCENES_DATAROOT=/path/to/v1.0-mini
pytest -q tests/test_nuscenes_loader.py
```

Without that variable set, the nuScenes tests are skipped (not failed) —
`sensor_model` and `schedule` tests do not need any dataset and always run.

## RELLIS-3D

`scripts/download_rellis.sh` pulls sequence `00004`'s KITTI-format point
clouds from the official RELLIS-3D Google Drive distribution.

**Correction as of 2026-09-10:** the KITTI-format data is not published
per-sequence — it's one combined archive across all 5 sequences (14GB for
the Ouster OS1-64 stream, 5.58GB for Velodyne). The script downloads that
combined archive and extracts only the `00004/` subfolder into
`<dest>/rellis/00004/`, then tells you to delete the rest. (There is a
separate, genuinely per-sequence "synced" ROS bag download for `00004`
alone at ~7GB — but that's rosbag format, not usable by this loader
without bag-extraction tooling this repo doesn't have, so don't use it
here.)

This is a background/detachable step (Ticket #3) — nothing else in the
build depends on it landing on time. See the ticket in the Build Map for
why `00004` specifically and why RELLIS-3D replaced SemanticKITTI as the
background dataset.

`configs/sensor_ouster_os1_64.yaml` ships with **placeholder** `d_theta_rad`
/ `d_phi_rad` values marked `# TODO: measure via Ticket #6`. Do not trust
numbers computed from this config until Ticket #6 (point-distribution gate)
has been run against real RELLIS-3D scans — there is no verified datasheet
value for the Ouster OS1-64's angular resolution anywhere in this repo.

## A note for whoever is prompting a model to write the next ticket

Every function in `sensor/` is a pure function of a config object — no
module-level constants, no hardcoded sensor numbers. If you ask a model to
extend this and it produces a literal like `0.1728` outside a `configs/*.yaml`
file, that is a bug per the Build Map's own rule for Ticket #4 and #9.
