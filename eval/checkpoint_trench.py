"""
eval/checkpoint_trench.py

Ticket #37 -- checkpoint: "the trench works." Renders a synthetic scene
with a ditch and shows DRISHTI's negative-obstacle pipeline (Tickets
#33-36) flagging it, side by side with a plain 2D occupancy grid built
from the SAME points that does not -- a plain grid only marks cells a
real return landed in, and the ditch floor has no return at all, so it
just looks like unremarkable empty space to it. DRISHTI catches it
because it reasons about what the LOCAL GROUND PLANE predicted a beam
should have hit, not just what it did hit.

No new production code -- an integration exercise over
observability.ground_plane, observability.negative_obstacle, and
grid.addressing, which are already built and individually tested. Same
synthetic-scene pattern as eval/checkpoint_first_map.py (Ticket #22):
flat ground, physically exact per-ring geometry (h_m / sin(phi(ring))),
not real sensor data.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grid.addressing import world_to_global
from observability.ground_plane import expected_ground_range, fit_local_ground_planes
from observability.negative_obstacle import PersistenceTracker, detect_anomalous_cells
from perception.sweep import Sweep
from sensor.sensor_model import SensorConfig, load_sensor_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"

N_AZIMUTH_COLS = 36  # 10 deg per column -- plenty for a still-image checkpoint
DITCH_AZIMUTH_COLS = (0, 1)  # a narrow window of columns near azimuth 0
DITCH_RANGE_LO_M = 13.0
DITCH_RANGE_HI_M = 17.0
N_PERSISTENCE_FRAMES = 3  # Ticket #36's own >= 3-consecutive-scans requirement
WORLD_CELL_SIZE_M = 0.5


def _ring_elevation_rad(ring: int, sm: SensorConfig) -> float:
    phi_min = sm.phi_max_rad - (sm.n_beams - 1) * sm.d_phi_rad
    fov = sm.phi_max_rad - phi_min
    return phi_min + fov * (1.0 - (ring + 0.5) / sm.n_beams)


def _azimuth_of_column(col: int, W: int) -> float:
    """Inverse of perception.range_image's forward azimuth-binning
    formula (u = floor(0.5*(1 - azimuth/pi)*W)), evaluated at the
    column's centre."""
    return math.pi * (1.0 - 2.0 * (col + 0.5) / W)


def build_synthetic_ditch_scene(sm: SensorConfig) -> Tuple[Sweep, np.ndarray, np.ndarray, np.ndarray]:
    """Flat ground at z = -h_m, physically exact per-ring/azimuth
    returns, EXCEPT the ditch window's rings within the ditch's radial
    span -- no point generated there (simulating no return: the beam
    either reaches the far wall well beyond DITCH_RANGE_HI_M, or exceeds
    usable range; either way, nothing to compare here except "missing").

    Returns (sweep_of_ground_returns, measured_range (H,W), valid_ring
    (H,) bool -- rings that ever look downward enough to hit flat
    ground, ditch_mask (H,W) bool -- True at the omitted cells).
    """
    H = sm.n_beams
    W = N_AZIMUTH_COLS
    measured = np.full((H, W), np.nan)
    ditch_mask = np.zeros((H, W), dtype=bool)
    valid_ring = np.zeros(H, dtype=bool)

    xs, ys, zs = [], [], []
    for ring in range(H):
        phi = _ring_elevation_rad(ring, sm)
        if phi >= 0:
            continue  # looks level or upward -- never hits flat ground below
        r = sm.h_m / abs(math.sin(phi))
        if r > sm.usable_range_m:
            continue
        valid_ring[ring] = True

        for col in range(W):
            azimuth = _azimuth_of_column(col, W)
            is_ditch = (
                col in DITCH_AZIMUTH_COLS and DITCH_RANGE_LO_M <= r <= DITCH_RANGE_HI_M
            )
            if is_ditch:
                ditch_mask[ring, col] = True
                continue  # no return -- the whole point of the scene
            measured[ring, col] = r
            xs.append(r * math.cos(azimuth))
            ys.append(r * math.sin(azimuth))
            zs.append(-sm.h_m)

    xyz = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    sweep = Sweep(
        xyz=xyz,
        intensity=np.ones(xyz.shape[0], dtype=np.float32),
        ring=np.full(xyz.shape[0], -1, dtype=np.int16),
        timestamp=0.0,
        T_world=np.eye(4, dtype=np.float64),
        sensor_id=sm.sensor_id,
    )
    return sweep, measured, valid_ring, ditch_mask


def run_checkpoint(out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sm = load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")
    sweep, measured, valid_ring, ditch_mask = build_synthetic_ditch_scene(sm)

    fit = fit_local_ground_planes(sweep, sm, n_azimuth_bins=N_AZIMUTH_COLS)

    H, W = measured.shape
    expected = np.full((H, W), np.nan)
    valid_expected = np.zeros((H, W), dtype=bool)
    for ring in range(H):
        if not valid_ring[ring]:
            continue
        for col in range(W):
            v = expected_ground_range(fit, ring, col)
            if v is not None:
                expected[ring, col] = v
                valid_expected[ring, col] = True

    occluded_mask = np.zeros((H, W), dtype=bool)  # no occluders in this scene

    tracker = PersistenceTracker(required_frames=N_PERSISTENCE_FRAMES)
    promoted_world_cells: set = set()
    for _ in range(N_PERSISTENCE_FRAMES):
        # A static ditch looks identical on every consecutive scan --
        # simulated here by re-running detection on the same frame.
        flagged = detect_anomalous_cells(measured, expected, valid_expected, occluded_mask, sm.usable_range_m)
        # PersistenceTracker is keyed by WORLD cell (Ticket #36's own
        # design note: (ring, azimuth) shifts every frame as the vehicle
        # moves, so persistence must key off world position instead) --
        # convert each flagged (ring, col) to the world cell its
        # EXPECTED ground point (the anomaly is "ground should be here
        # and isn't") falls into.
        world_cells = {
            _flagged_cell_to_world(ring, col, expected[ring, col], W)
            for ring, col in flagged
        }
        promoted_world_cells = tracker.update(world_cells)

    drishti_map = _rasterize_drishti_map(promoted_world_cells)
    plain_grid = _rasterize_plain_occupancy(sweep)

    fig_path = out_dir / "checkpoint_trench.png"
    _plot_side_by_side(drishti_map, plain_grid, fig_path)

    return {
        "n_points": sweep.xyz.shape[0],
        "n_ditch_cells_omitted": int(ditch_mask.sum()),
        "negative_obstacle_cells_promoted": len(promoted_world_cells),
        "drishti_detected_the_ditch": len(promoted_world_cells) > 0,
        "figure_png": fig_path,
    }


def _flagged_cell_to_world(ring: int, col: int, expected_range_m: float, W: int) -> Tuple[int, int]:
    azimuth = _azimuth_of_column(col, W)
    x = expected_range_m * math.cos(azimuth)
    y = expected_range_m * math.sin(azimuth)
    return world_to_global(x, y, WORLD_CELL_SIZE_M)


def _rasterize_drishti_map(promoted_world_cells: set, half_extent_m: float = 22.0) -> np.ndarray:
    """A simple top-down grid at WORLD_CELL_SIZE_M, marked with the
    promoted negative-obstacle world cells highlighted -- enough to make
    the checkpoint's point visually obvious without depending on the
    full Clipmap's fixed N=512 window matching this scene's scale."""
    n = int(2 * half_extent_m / WORLD_CELL_SIZE_M)
    grid = np.zeros((n, n))  # 0 = free/unknown
    for gi, gj in promoted_world_cells:
        si = gi + n // 2
        sj = gj + n // 2
        if 0 <= si < n and 0 <= sj < n:
            grid[sj, si] = 1.0  # 1 = NEGATIVE_OBSTACLE
    return grid


def _rasterize_plain_occupancy(sweep: Sweep, half_extent_m: float = 22.0) -> np.ndarray:
    """The naive baseline: a cell is occupied only if a real return
    landed there. No ground-plane reasoning, so the ditch (which has NO
    returns) is indistinguishable from ordinary empty space."""
    n = int(2 * half_extent_m / WORLD_CELL_SIZE_M)
    grid = np.zeros((n, n))
    gi = np.floor(sweep.xyz[:, 0] / WORLD_CELL_SIZE_M).astype(np.int64) + n // 2
    gj = np.floor(sweep.xyz[:, 1] / WORLD_CELL_SIZE_M).astype(np.int64) + n // 2
    in_bounds = (gi >= 0) & (gi < n) & (gj >= 0) & (gj < n)
    grid[gj[in_bounds], gi[in_bounds]] = 1.0
    return grid


def _plot_side_by_side(drishti_map: np.ndarray, plain_grid: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    axes[0].imshow(plain_grid, origin="lower", cmap="Greys")
    axes[0].set_title("Plain 2D occupancy grid\n(returns only -- the ditch is invisible)")

    drishti_rgb = np.ones(drishti_map.shape + (3,)) * 0.85
    drishti_rgb[drishti_map == 1.0] = (0.9, 0.1, 0.1)  # NEGATIVE_OBSTACLE red, matches checkpoint_first_map's palette
    axes[1].imshow(drishti_rgb, origin="lower")
    axes[1].set_title("DRISHTI map\n(negative-obstacle pipeline flags the ditch)")

    for ax in axes:
        ax.set_xlabel("x (grid cells)")
        ax.set_ylabel("y (grid cells)")

    fig.suptitle("Ticket #37 checkpoint -- SYNTHETIC scene, not real data")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    result = run_checkpoint()
    print(f"Points in scene: {result['n_points']}")
    print(f"Ditch cells with no return: {result['n_ditch_cells_omitted']}")
    print(f"Negative-obstacle cells promoted: {result['negative_obstacle_cells_promoted']}")
    print(f"Ditch detected: {result['drishti_detected_the_ditch']}")
    print(f"Figure: {result['figure_png']}")
