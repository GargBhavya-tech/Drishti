"""
eval/checkpoint_attention_overlay.py

Explainability checkpoint: runs a REAL trained checkpoint over a REAL
RELLIS-3D frame with `return_attention=True` (perception/segnet.py),
and renders the decoder's finest attention gate (ag1) as a heatmap
overlay -- "why does the network think this is X", using the network's
own real activations, not a synthetic/approximated saliency method
bolted on afterward. Same checkpoint pattern as Tickets #43/#58's own
real-data checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from eval.cache_inference import load_trained_model
from perception.input_tensor import assemble_input_tensor, load_stats
from perception.train import _load_frame
from sensor.sensor_model import load_sensor_config

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"


def run_checkpoint(
    checkpoint_path: str = "checkpoints_multi_remote/checkpoint_epoch19.pt",
    sequence_dir: str = "data/rellis/00004",
    frame_idx: int = 1500,
    sensor_config_path: str = "configs/sensor_ouster_os1_64.yaml",
    out_dir: Path = DEFAULT_OUT_DIR,
    device: str = "cpu",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, trained_epoch = load_trained_model(checkpoint_path, device=device)
    stats = load_stats(Path(checkpoint_path).parent / "channel_stats.json")
    sm = load_sensor_config(sensor_config_path)

    img, ground, _target = _load_frame(sequence_dir, frame_idx, sm)
    tensor = assemble_input_tensor(img, ground, stats)
    x = torch.from_numpy(tensor).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits, attention_maps = model(x, return_attention=True)
    pred = logits.argmax(dim=1)[0].cpu().numpy()

    # ag1 is the FINEST-resolution gate (closest to the input) -- the
    # most directly interpretable "where is the network looking" map
    # for a human reading it next to the raw scene. Upsampled to the
    # prediction's own resolution for a pixel-aligned overlay (ag1
    # itself runs at e1's resolution, half the input's, since e1 is
    # EfficientNet-B0's own stride-2 stem output).
    with torch.no_grad():
        attn_upsampled = F.interpolate(
            attention_maps["ag1"], size=pred.shape, mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    axes[0].imshow(img.range, cmap="viridis", aspect="auto")
    axes[0].set_title(f"Input range image (frame {frame_idx})")
    axes[1].imshow(pred, cmap="tab10", aspect="auto", vmin=0, vmax=9)
    axes[1].set_title(f"Predicted class (checkpoint epoch {trained_epoch})")
    im = axes[2].imshow(attn_upsampled, cmap="inferno", aspect="auto", vmin=0, vmax=1)
    axes[2].set_title("Attention gate ag1 -- real activations from the trained network (not a synthetic saliency method)")
    fig.colorbar(im, ax=axes[2], orientation="horizontal", fraction=0.05, pad=0.3)
    fig.tight_layout()

    out_path = out_dir / "checkpoint_attention_overlay.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    attention_stats = {
        name: {"mean": float(a.mean().item()), "std": float(a.std().item())}
        for name, a in attention_maps.items()
    }
    return {"figure_png": out_path, "attention_stats": attention_stats, "trained_epoch": trained_epoch}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explainability checkpoint: real attention-gate overlay")
    parser.add_argument("--checkpoint", default="checkpoints_multi_remote/checkpoint_epoch19.pt")
    parser.add_argument("--sequence-dir", default="data/rellis/00004")
    parser.add_argument("--frame-idx", type=int, default=1500)
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = run_checkpoint(
        checkpoint_path=args.checkpoint,
        sequence_dir=args.sequence_dir,
        frame_idx=args.frame_idx,
        sensor_config_path=args.sensor_config,
        out_dir=Path(args.out_dir),
        device=args.device,
    )
    print(f"Figure: {result['figure_png']}")
    for name, s in result["attention_stats"].items():
        print(f"  {name}: mean={s['mean']:.3f} std={s['std']:.3f}")
