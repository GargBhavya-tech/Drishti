"""
perception/joint_seg_dataset.py

Joint multi-dataset training support: wraps the three already-existing,
already-separately-trained-against datasets (RellisSegDataset,
NuscenesSegDataset, SemanticPossSegDataset -- see perception/train.py,
perception/nuscenes_seg_dataset.py, perception/semanticposs_seg_dataset.py)
into ONE torch.utils.data.ConcatDataset a single DataLoader can draw
mixed-domain batches from.

Why this needs a resize step, not just concatenation: the three
datasets each project through a DIFFERENT sensor config (Ouster OS1-64
-> 64x2048, HDL-32E -> 32x1080, Pandar40P -> 40x1800 -- see each
dataset's own sensor yaml). PyTorch's default collate_fn stacks tensors
and requires identical shapes within a batch; three different spatial
resolutions in the same batch would crash on the very first mixed
batch. Every prior use of these three datasets this session trained
each on its OWN sensor's native resolution separately -- joint training
is the first case that needs one common shape.

Resize target: 64x2048 (RELLIS-3D's own native resolution, and the
resolution the original FusionSegNet checkpoints were trained at) --
chosen to UPSAMPLE the two smaller sensors rather than downsample
RELLIS-3D's real 64-beam detail away. Same "upsample to the network's
native training resolution" precedent eval/eval_nuscenes.py's own
"resample" mode already established for zero-shot eval; the difference
here is this also resizes the TARGET labels and valid mask, which that
eval-only script never needed to (it only ever resized the INPUT and
downsampled LOGITS back to native resolution for comparison against
ground truth at native resolution).

Resize method matters and is deliberately different per tensor:
- Input tensor (13 real-valued channels): bilinear -- continuous
  physical quantities (range, intensity, normals, ...), interpolating
  between them is physically meaningful.
- Target (integer DrishtiClass ids) and valid_mask (bool): NEAREST
  neighbor only -- bilinear-interpolating a class id would invent
  fractional, meaningless intermediate "classes" between e.g. DRIVABLE
  (1) and STATIC_OBSTACLE (4). Never do that to a label map.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset

CANONICAL_HW: Tuple[int, int] = (64, 2048)  # RELLIS-3D's native resolution -- see module docstring


class _ResizedDatasetWrapper(Dataset):
    """Wraps one of the three underlying per-dataset Dataset objects,
    resizing every (tensor, target, valid) item to CANONICAL_HW. A
    dataset already at CANONICAL_HW (RELLIS-3D itself) is passed
    through with a cheap shape check and no actual interpolation call."""

    def __init__(self, underlying: Dataset, name: str):
        self.underlying = underlying
        self.name = name  # for error messages / dataset provenance logging only

    def __len__(self):
        return len(self.underlying)

    def __getitem__(self, i: int):
        tensor, target, valid = self.underlying[i]
        h, w = tensor.shape[-2], tensor.shape[-1]
        if (h, w) == CANONICAL_HW:
            return tensor, target, valid

        tensor_r = F.interpolate(
            tensor.unsqueeze(0), size=CANONICAL_HW, mode="bilinear", align_corners=False
        ).squeeze(0)

        # nearest-neighbor only for integer/boolean data -- see module docstring
        target_r = F.interpolate(
            target.float().unsqueeze(0).unsqueeze(0), size=CANONICAL_HW, mode="nearest"
        ).squeeze(0).squeeze(0).long()
        valid_r = F.interpolate(
            valid.float().unsqueeze(0).unsqueeze(0), size=CANONICAL_HW, mode="nearest"
        ).squeeze(0).squeeze(0).bool()

        return tensor_r, target_r, valid_r


def build_joint_train_dataset(rellis_ds: Dataset, nuscenes_ds: Dataset, semanticposs_ds: Dataset) -> ConcatDataset:
    """One ConcatDataset spanning all three domains' TRAIN items, each
    resized to CANONICAL_HW. Shuffling the resulting DataLoader
    naturally interleaves domains batch-to-batch; no domain-balancing
    logic is applied beyond that -- RELLIS-3D's 11,522 frames will
    dominate raw sample counts (nuScenes: 324, SemanticPOSS: 2,540),
    stated here explicitly rather than silently assumed balanced."""
    return ConcatDataset(
        [
            _ResizedDatasetWrapper(rellis_ds, "rellis"),
            _ResizedDatasetWrapper(nuscenes_ds, "nuscenes"),
            _ResizedDatasetWrapper(semanticposs_ds, "semanticposs"),
        ]
    )
