# DRISHTI

Adaptive variable-resolution 2.5D LiDAR mapping for dynamic environment perception.
SIH 2026 — DRDO Problem Statement 26053. Team Phir Hera Pheri.

This repo is a companion to `DRISHTI_Project_Bible_v3.md` (why) and
`DRISHTI_Build_Map.md` (what, in ticket order). Read those first — this
README only orients you inside the code.

## Status

Tickets **#1–#5** are built (see `DRISHTI_Build_Map.md` for what each covers):

| Ticket | What | Where |
|---|---|---|
| #1 | Repo skeleton, Colab environment | this layout, `colab_setup.ipynb` |
| #2 | nuScenes-mini loader → canonical `Sweep` | `perception/nuscenes_loader.py` |
| #3 | RELLIS-3D loader + background download script | `perception/rellis_loader.py`, `scripts/download_rellis.sh` |
| #4 | `SensorModel` — the seven formulas | `sensor/sensor_model.py`, `configs/sensor_*.yaml` |
| #5 | Resolution schedule generator | `sensor/schedule.py` |

Everything downstream (#6 onward) is not yet built.

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
