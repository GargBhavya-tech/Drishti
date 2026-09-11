"""
eval/checkpoint_sparsity_speed.py

Ticket #43 -- checkpoint: Claims 3 and 4 demonstrable. Same pattern as
Tickets #22/#37's checkpoints: a synthetic scene (a thin pole scattered
with sparse, range-limited returns, matching Bible Part 11's own worked
example) with SPARSE_STRUCTURED/UNKNOWN cells overlaid, the speed
envelope polygon drawn per bearing, and the conservatism property
test's own pass/fail result printed alongside -- run for real via
pytest, not asserted inline, so this checkpoint can't silently drift
from what tests/test_conservatism.py actually verifies.
"""

from __future__ import annotations

import io
import math
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from observability.sparsity import SparsityVerdict, classify_sparsity, r_blind_for_min_object
from planning.speed_envelope import speed_envelope
from sensor.sensor_model import load_sensor_config
from sensor.vehicle_config import load_vehicle_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"


def _pole_scenario(sm, vehicle, azimuth_deg: float, ranges_m: List[float]) -> List[dict]:
    """A thin pole at several ranges, one entry per range, with a
    plausible number of returns derived from N_exp itself (rounded,
    capped by a small integer draw) -- illustrating the decision table
    moving through NORMAL -> SPARSE_STRUCTURED -> UNKNOWN as range grows,
    exactly Bible Part 11's own worked table."""
    rng = np.random.default_rng(0)
    az = math.radians(azimuth_deg)
    rows = []
    for r in ranges_m:
        n_exp = classify_sparsity(n_obs=0, r=r, vehicle=vehicle, sm=sm).n_exp
        # A real pole returns close to n_exp points when kappa~1, and
        # increasingly fewer (down to 0) past r_blind -- sampled here,
        # not asserted, to make the plotted scene look like real data.
        n_obs = int(np.clip(round(n_exp + rng.normal(0, 0.3)), 0, None))
        ring_base = rng.integers(10, 40)
        ranges = r + rng.normal(0, 0.02, size=max(n_obs, 1))[:n_obs] if n_obs > 0 else np.array([])
        rings = ring_base + np.arange(n_obs) if n_obs > 0 else np.array([])
        result = classify_sparsity(n_obs=n_obs, r=r, vehicle=vehicle, sm=sm, ranges=ranges, ring_indices=rings)
        x, y = r * math.cos(az), r * math.sin(az)
        rows.append({"r": r, "x": x, "y": y, "n_obs": n_obs, "n_exp": n_exp, "result": result})
    return rows


def _run_conservatism_property_test() -> Tuple[bool, str]:
    """Runs tests/test_conservatism.py for real via pytest, rather than
    re-asserting a copy of its logic here -- this checkpoint reports
    whatever that suite ACTUALLY says, so it can never silently drift
    from it."""
    import pytest

    buf = io.StringIO()
    test_path = str(REPO_ROOT / "tests" / "test_conservatism.py")
    with redirect_stdout(buf):
        exit_code = pytest.main(["-q", test_path])
    passed = exit_code == 0
    return passed, buf.getvalue()


def run_checkpoint(out_dir: Path = DEFAULT_OUT_DIR, verify_conservatism: bool = True) -> dict:
    """`verify_conservatism=False` skips the (real, but ~1 minute for
    10,000 Hypothesis examples) pytest sub-invocation -- for callers
    that already know the property holds and only need this
    checkpoint's OWN scene/plot output (e.g. a test comparing two runs'
    determinism), not a second re-verification of a suite that's
    already run elsewhere. The real CLI entry point below always
    verifies it; only test code should ever pass False."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sm = load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")
    vehicle = load_vehicle_config(CONFIGS / "vehicle_ugv.yaml")

    r_blind = r_blind_for_min_object(vehicle, sm)
    ranges = [r_blind * f for f in (0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0)]
    pole_rows = _pole_scenario(sm, vehicle, azimuth_deg=25.0, ranges_m=ranges)

    envelope = speed_envelope(
        {
            "2m_ditch": 21.6,
            "5cm_cable": 6.7,
            "thin_fence_post_presence": r_blind,
        },
        vehicle,
    )

    if verify_conservatism:
        property_test_passed, property_test_output = _run_conservatism_property_test()
    else:
        property_test_passed, property_test_output = None, "(skipped -- verify_conservatism=False)"

    # --- plot: bird's-eye scene with sparsity verdicts + speed envelope ---
    fig, ax = plt.subplots(figsize=(7, 7))

    verdict_colors = {
        SparsityVerdict.NORMAL: (0.2, 0.6, 0.2),
        SparsityVerdict.SPARSE_STRUCTURED: (0.9, 0.6, 0.1),
        SparsityVerdict.NOISE_SUPPRESSED: (0.6, 0.6, 0.6),
        SparsityVerdict.FREE: (0.8, 0.8, 0.8),
        SparsityVerdict.UNKNOWN: (0.85, 0.1, 0.1),
    }

    for row in pole_rows:
        color = verdict_colors[row["result"].verdict]
        ax.scatter(row["x"], row["y"], s=80, color=color, edgecolors="black", zorder=3)
        ax.annotate(
            f"{row['result'].verdict.name}\nr={row['r']:.0f}m n_obs={row['n_obs']}",
            (row["x"], row["y"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=7,
        )

    # r_blind boundary circle -- the geometric line the whole scene
    # illustrates: everything past it with zero returns is UNKNOWN.
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r_blind * np.cos(theta), r_blind * np.sin(theta), "--", color="red", alpha=0.5, label=f"r_blind = {r_blind:.1f}m")

    # Speed envelope: draw each hazard's own v_max as a labelled ring,
    # with the BINDING one highlighted -- "which hazard is binding" per
    # Ticket #40's own "Done when".
    for name, r in {"2m_ditch": 21.6, "5cm_cable": 6.7, "thin_fence_post_presence": r_blind}.items():
        is_binding = name == envelope.binding_hazard
        ax.plot(
            r * np.cos(theta), r * np.sin(theta),
            color="blue" if is_binding else "lightblue",
            linewidth=2.5 if is_binding else 1.0,
            alpha=0.9 if is_binding else 0.4,
        )

    ax.scatter([0], [0], marker="^", s=150, color="black", zorder=5, label="ego")
    ax.set_aspect("equal")
    ax.set_xlim(-r_blind * 2.2, r_blind * 2.2)
    ax.set_ylim(-r_blind * 2.2, r_blind * 2.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        f"Ticket #43 -- Claims 3 & 4\n"
        f"Binding hazard: {envelope.binding_hazard} (R={envelope.binding_range_m:.1f}m) -> "
        f"v_max={envelope.v_max_kmh:.1f} km/h\n"
        f"Conservatism property test (tests/test_conservatism.py): "
        f"{'PASS' if property_test_passed else ('SKIPPED' if property_test_passed is None else 'FAIL')}"
    )
    fig.tight_layout()
    out_path = out_dir / "checkpoint_sparsity_speed.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return {
        "r_blind_m": r_blind,
        "pole_rows": pole_rows,
        "speed_envelope": envelope,
        "conservatism_property_test_passed": property_test_passed,
        "conservatism_property_test_output": property_test_output,
        "figure_png": out_path,
    }


if __name__ == "__main__":
    result = run_checkpoint()
    print(f"r_blind (thin fence post, this vehicle's min object spec): {result['r_blind_m']:.2f} m")
    print("Pole scenario, by range:")
    for row in result["pole_rows"]:
        r = row["result"]
        print(f"  r={row['r']:6.1f}m  n_obs={row['n_obs']:2d}  n_exp={r.n_exp:6.2f}  kappa={r.kappa:6.2f}  verdict={r.verdict.name}")
    env = result["speed_envelope"]
    print(f"\nSpeed envelope: binding hazard = {env.binding_hazard} (R={env.binding_range_m:.1f}m) "
          f"-> v_max = {env.v_max_ms:.2f} m/s = {env.v_max_kmh:.1f} km/h")
    for name, v in env.per_hazard_v_max_ms.items():
        print(f"  {name}: v_max = {v:.2f} m/s = {v*3.6:.1f} km/h")
    print(f"\nConservatism property test (tests/test_conservatism.py): "
          f"{'PASS' if result['conservatism_property_test_passed'] else 'FAIL'}")
    print(f"Figure: {result['figure_png']}")
