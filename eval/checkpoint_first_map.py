"""
eval/checkpoint_first_map.py

Ticket #22 -- checkpoint: run one sweep end to end (load -> scatter ->
cells -> layers) and dump two bird's-eye PNGs, coloured by height and by
class. No new production code, per the ticket -- this is an integration
exercise over grid/scatter.py, grid/histogram.py, grid/layers.py, which
are already built and individually tested.

DEVIATION FROM THE LITERAL TICKET: it asks for "one nuScenes sweep."
`nuscenes-devkit` and the actual nuScenes-mini dataset are not available
in this environment (same gap tests/test_nuscenes_loader.py is skipped
for). This uses a SYNTHETIC scene instead -- flat ground, a building, a
bridge deck over a clear road (Claim 1's own scene, Bible Part 9.1), and
a vegetation patch -- clearly documented as a stand-in, never presented
as real data. Swap in perception.nuscenes_loader.load_nuscenes_sweep()
and re-run once nuscenes-devkit is installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grid.cell import H_QUANTUM_M, decode_class_conf
from grid.clipmap import Clipmap
from grid.histogram import scatter_histogram
from grid.layers import scatter_layers
from grid.scatter import scatter, scatter_class
from perception.taxonomy import DrishtiClass
from sensor.sensor_model import load_sensor_config
from sensor.vehicle_config import load_vehicle_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"

CLASS_COLORS = {
    int(DrishtiClass.UNKNOWN): (0.8, 0.8, 0.8),
    int(DrishtiClass.DRIVABLE): (0.29, 0.49, 0.35),
    int(DrishtiClass.CAUTION): (0.71, 0.65, 0.26),
    int(DrishtiClass.NON_TRAVERSABLE): (0.48, 0.29, 0.29),
    int(DrishtiClass.STATIC_OBSTACLE): (0.33, 0.33, 0.33),
    int(DrishtiClass.VEGETATION): (0.18, 0.42, 0.18),
    int(DrishtiClass.VEHICLE): (0.12, 0.37, 0.66),
    int(DrishtiClass.PEDESTRIAN): (0.75, 0.22, 0.17),
    int(DrishtiClass.NEGATIVE_OBSTACLE): (0.9, 0.1, 0.1),
    int(DrishtiClass.OVERHANG): (0.9, 0.5, 0.1),
}

SCENE_HALF_EXTENT_M = 25.0  # matches build_synthetic_scene's world extent


def build_synthetic_scene(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """A plausible stand-in for one real sweep. Returns (xyz (P,3),
    class_ids (P,)), world-anchored, ground genuinely at z=0 everywhere
    (a flat synthetic world, so z_ground=0 is exact, not an
    approximation)."""
    points, classes = [], []

    n_ground = 4000
    gx = rng.uniform(-SCENE_HALF_EXTENT_M, SCENE_HALF_EXTENT_M, n_ground)
    gy = rng.uniform(-SCENE_HALF_EXTENT_M, SCENE_HALF_EXTENT_M, n_ground)
    gz = rng.normal(0.0, 0.01, n_ground)
    points.append(np.stack([gx, gy, gz], axis=1))
    classes.append(np.full(n_ground, int(DrishtiClass.DRIVABLE), dtype=np.int64))

    n_building = 1500
    bx = rng.uniform(6, 9, n_building)
    by = rng.uniform(-15, -10, n_building)
    bz = rng.uniform(0.0, 5.0, n_building)
    points.append(np.stack([bx, by, bz], axis=1))
    classes.append(np.full(n_building, int(DrishtiClass.STATIC_OBSTACLE), dtype=np.int64))

    # Claim 1's own scene: a clear road with a bridge deck above it.
    n_road = 400
    rx = rng.uniform(-5, -2, n_road)
    ry = rng.uniform(5, 8, n_road)
    rz = rng.normal(0.0, 0.01, n_road)
    points.append(np.stack([rx, ry, rz], axis=1))
    classes.append(np.full(n_road, int(DrishtiClass.DRIVABLE), dtype=np.int64))

    n_deck = 400
    dx = rng.uniform(-5, -2, n_deck)
    dy = rng.uniform(5, 8, n_deck)
    dz = rng.uniform(4.2, 4.6, n_deck)
    points.append(np.stack([dx, dy, dz], axis=1))
    classes.append(np.full(n_deck, int(DrishtiClass.STATIC_OBSTACLE), dtype=np.int64))

    n_veg = 800
    vx = rng.uniform(-15, -10, n_veg)
    vy = rng.uniform(2, 8, n_veg)
    vz = rng.uniform(0.0, 1.5, n_veg)
    points.append(np.stack([vx, vy, vz], axis=1))
    classes.append(np.full(n_veg, int(DrishtiClass.VEGETATION), dtype=np.int64))

    xyz = np.concatenate(points, axis=0)
    class_ids = np.concatenate(classes, axis=0)
    return xyz, class_ids


def run_checkpoint(out_dir: Path = DEFAULT_OUT_DIR, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sm = load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")
    vehicle = load_vehicle_config(CONFIGS / "vehicle_ugv.yaml")
    cm = Clipmap(sm, n_levels=4, N=512, c0=0.05)
    cm.scroll_to(0.0, 0.0)

    rng = np.random.default_rng(seed)
    xyz, class_ids = build_synthetic_scene(rng)
    z_ground = np.zeros(xyz.shape[0])

    scatter(cm, xyz)
    scatter_class(cm, xyz, class_ids)
    scatter_histogram(cm, xyz, z_ground, vehicle)
    scatter_layers(cm, xyz, z_ground, vehicle)

    level = 0
    c_l = cm.levels[level].cell_size_m
    half_cells = int(SCENE_HALF_EXTENT_M / c_l) + 2
    center_si = cm.N // 2  # ego is at (0,0), which by construction sits at the window centre
    lo = max(0, center_si - half_cells)
    hi = min(cm.N, center_si + half_cells)

    flags = cm.flags[level].reshape(cm.N, cm.N)[lo:hi, lo:hi]
    h_max = cm.h_max[level].reshape(cm.N, cm.N)[lo:hi, lo:hi]
    class_conf = cm.class_conf[level].reshape(cm.N, cm.N)[lo:hi, lo:hi]

    height_path = out_dir / "checkpoint_height.png"
    class_path = out_dir / "checkpoint_class.png"
    _plot_height(flags, h_max, height_path)
    _plot_class(flags, class_conf, class_path)

    return {
        "touched_cells_l0": int(np.count_nonzero(cm.flags[level])),
        "height_png": height_path,
        "class_png": class_path,
        "n_points": xyz.shape[0],
    }


def _plot_height(flags: np.ndarray, h_max: np.ndarray, out_path: Path) -> None:
    observed = flags != 0
    heights = np.full(flags.shape, np.nan)
    # decode_h is scalar-only; this is the same 1cm fixed-point formula
    # applied vectorised across the whole cropped plane.
    heights[observed] = h_max[observed].astype(np.float64) * H_QUANTUM_M

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(heights, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, label="height (m)")
    ax.set_title("DRISHTI checkpoint -- bird's-eye, coloured by height\n(SYNTHETIC scene, not real data -- see module docstring)")
    ax.set_xlabel("storage column (si)")
    ax.set_ylabel("storage row (sj)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_class(flags: np.ndarray, class_conf: np.ndarray, out_path: Path) -> None:
    observed = flags != 0
    rgb = np.ones(flags.shape + (3,))  # white background for unobserved
    class_ids = np.zeros(flags.shape, dtype=np.int64)
    class_ids[observed] = [decode_class_conf(int(b))[0] for b in class_conf[observed]]

    for cid, color in CLASS_COLORS.items():
        mask = observed & (class_ids == cid)
        rgb[mask] = color

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb, origin="lower")
    ax.set_title("DRISHTI checkpoint -- bird's-eye, coloured by class\n(SYNTHETIC scene, not real data -- see module docstring)")
    ax.set_xlabel("storage column (si)")
    ax.set_ylabel("storage row (sj)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    result = run_checkpoint()
    print(f"Touched L0 cells: {result['touched_cells_l0']}")
    print(f"Points scattered: {result['n_points']}")
    print(f"Height map: {result['height_png']}")
    print(f"Class map:  {result['class_png']}")
