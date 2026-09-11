"""
eval/checkpoint_detection_vs_range.py

Ticket #60 checkpoint: "Done when: predicted vs measured is plotted
for at least kerb and ditch."
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.detection_vs_range import predicted_ditch_range_m, predicted_kerb_range_m, sweep_ditch_detection, sweep_kerb_detection
from sensor.sensor_model import load_sensor_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"

KERB_HEIGHT_M = 0.15
DITCH_WIDTH_M = 2.0
DITCH_DEPTH_M = 0.5


def run_checkpoint(out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sm = load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")

    kerb_predicted = predicted_kerb_range_m(KERB_HEIGHT_M, sm)
    kerb_ranges = [kerb_predicted * f for f in (0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0)]
    kerb_curve = sweep_kerb_detection(KERB_HEIGHT_M, sm, kerb_ranges)

    ditch_predicted = predicted_ditch_range_m(DITCH_WIDTH_M, sm)
    ditch_ranges = [ditch_predicted * f for f in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)]
    ditch_curve = sweep_ditch_detection(DITCH_WIDTH_M, DITCH_DEPTH_M, sm, ditch_ranges)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, curve, title in ((axes[0], kerb_curve, "15cm kerb"), (axes[1], ditch_curve, "2m x 0.5m ditch")):
        ax.plot(curve.ranges_m, curve.detection_rate, "o-", color="steelblue", label="measured")
        ax.axvline(curve.predicted_range_m, color="green", linestyle="--", label=f"predicted ({curve.predicted_range_m:.1f}m)")
        ax.axvline(curve.measured_range_m, color="red", linestyle=":", label=f"measured ({curve.measured_range_m:.1f}m)")
        ax.set_xlabel("range (m)")
        ax.set_ylabel("detection rate")
        ax.set_title(f"{title} -- HDL-64E")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Ticket #60 -- hazard detection vs range (predicted vs measured)")
    fig.tight_layout()
    out_path = out_dir / "checkpoint_detection_vs_range.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return {
        "kerb_curve": kerb_curve,
        "ditch_curve": ditch_curve,
        "figure_png": out_path,
    }


if __name__ == "__main__":
    result = run_checkpoint()
    for name, curve in (("kerb", result["kerb_curve"]), ("ditch", result["ditch_curve"])):
        error_pct = 100.0 * abs(curve.measured_range_m - curve.predicted_range_m) / curve.predicted_range_m
        print(f"{name}: predicted={curve.predicted_range_m:.2f}m measured={curve.measured_range_m:.2f}m error={error_pct:.1f}%")
    print(f"Figure: {result['figure_png']}")
