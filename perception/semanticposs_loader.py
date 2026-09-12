"""
perception/semanticposs_loader.py

SemanticPOSS loader -> canonical Sweep. SemanticPOSS ships in
SemanticKITTI-compatible format (confirmed against the real downloaded
`SemanticPOSS_dataset.zip`, not assumed from the paper's claim):

    <seq>/
      velodyne/000000.bin, 000001.bin, ...   (x, y, z, intensity) float32
      labels/000000.label, ...                uint32, low 16 bits = semantic id
      tag/000000.tag                          NOT used by this loader -- see below
      calib.txt, poses.txt, instances.txt

Watch out -- a real, confirmed quirk different from RELLIS-3D:
sequence 00 starts at frame 000000, but sequences 01-05 start at
000001 (verified against the actual downloaded files, not assumed).
Frame indices are NOT a reliable 0..n-1 range across sequences the way
`perception.train.build_multi_sequence_splits` assumes for RELLIS-3D --
this loader and perception.semanticposs_seg_dataset therefore work with
the ACTUAL frame-id strings present on disk (via
`collect_semanticposs_frame_ids`), never a synthesised `range(n)`.

Watch out #2 -- why `tag/*.tag` is NOT used here: the official
`read_data.py` (shipped inside the dataset) uses `tag` to scatter the
variable-length (points, labels) arrays into a FIXED 40x1800 range-image
grid of the dataset's own construction. This project's own
`perception.range_image.project_to_range_image` computes its own
azimuth/elevation-derived range image directly from each point's real
xyz and this project's OWN resolution schedule (Bible Part C.3) --
independent of any dataset-provided grid -- exactly the same way
perception.rellis_loader and perception.nuscenes_loader never use
their source datasets' own pixel layouts either. The `tag` files are
therefore read by nothing in this project; only `velodyne/*.bin` and
`labels/*.label` matter here.

Confirmed against real downloaded data (not assumed from the paper):
points and labels are 1:1 aligned for a given frame id, with a
PER-FRAME point count that varies (66,000-70,000-ish observed) -- this
is the dataset's own valid-return count, not a fixed grid size.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from perception.sweep import Sweep

SEMANTICPOSS_POINT_FIELDS = 4  # x, y, z, intensity -- same KITTI-style layout as RELLIS-3D


def collect_semanticposs_frame_ids(sequence_dir: str | Path) -> List[str]:
    """Sorted list of frame-id strings (e.g. "000001") actually present
    in a sequence's velodyne/ directory -- see module docstring on why
    this must NOT be synthesised as range(n_frames)."""
    sequence_dir = Path(sequence_dir)
    return sorted(p.stem for p in (sequence_dir / "velodyne").glob("*.bin"))


def load_semanticposs_sweep(
    sequence_dir: str | Path,
    frame_id: str,
    frame_period_s: float = 0.1,
) -> Sweep:
    """Load one frame from a SemanticPOSS sequence directory. `frame_id`
    is the exact on-disk stem (e.g. "000001"), not an assumed index --
    see module docstring."""
    sequence_dir = Path(sequence_dir)
    bin_path = sequence_dir / "velodyne" / f"{frame_id}.bin"
    raw = np.fromfile(str(bin_path), dtype=np.float32)
    if raw.size % SEMANTICPOSS_POINT_FIELDS != 0:
        raise ValueError(
            f"{bin_path}: point buffer size {raw.size} is not a multiple of "
            f"{SEMANTICPOSS_POINT_FIELDS} -- this file is not (x,y,z,intensity) "
            f"float32 as SemanticPOSS's real downloaded format uses."
        )
    points = raw.reshape(-1, SEMANTICPOSS_POINT_FIELDS)

    xyz = points[:, 0:3].astype(np.float32)
    # Real-data check (this session): SemanticPOSS intensity values are
    # already in a small float range in the downloaded .bin files, same
    # convention as RELLIS-3D's KITTI-format intensity -- NOT the
    # nuScenes 0..255 convention. No /255 normalisation applied here.
    intensity = points[:, 3].astype(np.float32)

    ring = np.full(xyz.shape[0], -1, dtype=np.int16)  # not shipped, same as RELLIS-3D KITTI-format

    return Sweep(
        xyz=xyz,
        intensity=intensity,
        ring=ring,
        timestamp=0.0,  # real per-frame timestamps not confirmed against poses.txt/calib.txt -- not needed for single-frame training
        T_world=np.eye(4, dtype=np.float64),  # not needed for single-frame training (no multi-sweep use here)
        sensor_id="pandar40p",
    )


def load_semanticposs_labels(sequence_dir: str | Path, frame_id: str) -> np.ndarray:
    """Raw per-point label IDs for one frame (uint32, class id in the
    low 16 bits -- confirmed against real downloaded data this session:
    points.shape[0] == labels.shape[0] for every checked frame). NOT run
    through the DRISHTI taxonomy remap here -- that's
    perception.taxonomy.semanticposs_label_ids_to_drishti's job."""
    sequence_dir = Path(sequence_dir)
    label_path = sequence_dir / "labels" / f"{frame_id}.label"
    raw = np.fromfile(str(label_path), dtype=np.uint32)
    return raw & 0xFFFF
