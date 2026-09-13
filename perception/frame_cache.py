"""
perception/frame_cache.py

Generic on-disk cache for the expensive, PURELY DETERMINISTIC per-frame
CPU preprocessing every dataset in this project was redoing from
scratch on every single __getitem__ call, every epoch: spherical
projection (project_to_range_image), the ground-prior column-wise walk,
surface-geometry (normals/curvature), and -- for the detection dataset
-- real 3D box-target projection. None of this depends on which epoch
is training or how the loader shuffles; it is a pure function of
(dataset, frame identity, sensor config). Confirmed this session as the
project's real bottleneck, not the GPU: `nvidia-smi` showed 14% GPU
utilization while `top` showed a 12.2 load average on this 10-core
server with two real training jobs running -- the GPU sits idle waiting
for CPU workers to finish preparing the next batch.

What gets cached: the RAW (pre-normalisation) 13-channel stack plus
whatever target/label array a given dataset needs (segmentation target,
or detection's objectness+regression) -- NEVER the final normalised
input tensor. Channel normalisation (perception.input_tensor.
normalize_raw_channels) is a cheap per-call subtract/divide using
whichever ChannelStats THIS run supplies; caching after normalisation
would silently bake one run's stats into the cache file and corrupt any
future run that legitimately uses different stats. Caching before
normalisation is correct and reusable forever, regardless of which run
computed the stats.

Disk-safety, not an afterthought: this project's own server sat at 31GB
free / 92% used when this was built, on a SHARED machine other people's
work also runs on. A caching scheme that writes files without checking
free space first can fill a shared server's disk and break everyone's
work, not just this project's. `FrameCache` therefore tracks how many
bytes IT has written this run and refuses (raises `RuntimeError`, does
NOT silently skip caching or silently fill the disk) once that would
exceed `max_disk_fraction` of the disk's free space AT THE TIME THE
CACHE WAS CREATED -- a conservative, real, checked budget rather than a
per-item size guessed from dtype*shape (which compression can make
wrong in either direction).

Real measured effect: the FIRST epoch through a freshly-cached dataset
pays the full compute-and-write cost (no faster than before, arguably
slightly slower from the extra disk write) -- every epoch AFTER that
is a fast disk read with zero projection/geometry/box-projection work.
The break-even point is epoch 2. A single-epoch run gets no benefit
from caching and should leave it off.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Dict

import numpy as np


class FrameCache:
    """One instance per dataset (not shared across datasets -- each
    tracks its own disk budget against the SAME underlying filesystem,
    so multiple FrameCache instances on the same disk are each
    individually conservative, not coordinated -- running several
    differently-configured caching datasets at once could in principle
    together exceed one instance's own budget check; not a concern at
    this project's current scale (nuScenes + SemanticPOSS caches
    measured in the low tens of GB combined, well under the 31GB seen
    free), but stated here rather than silently assumed impossible)."""

    def __init__(self, cache_dir: str | Path, max_disk_fraction: float = 0.5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_disk_fraction = max_disk_fraction
        self._free_at_start = shutil.disk_usage(self.cache_dir).free
        self._written_bytes = 0

    def get_or_compute(self, key: str, compute_fn: Callable[[], Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """`key` must be a filesystem-safe, STABLE identifier for this
        exact frame (e.g. a RELLIS "{sequence_name}_{frame_idx}" or a
        nuScenes sample_token -- tokens are already filesystem-safe
        hex/alnum strings). `compute_fn` is called ONLY on a cache miss
        and must return a dict of named numpy arrays (e.g.
        {"raw": ..., "target": ..., "valid_mask": ...})."""
        cache_path = self.cache_dir / f"{key}.npz"
        if cache_path.exists():
            try:
                with np.load(cache_path) as data:
                    return {k: data[k] for k in data.files}
            except (OSError, ValueError, zipfile.BadZipFile, EOFError):
                # A prior write was interrupted (killed job, disk full
                # mid-write) and left a truncated/corrupt file -- treat
                # exactly like a cache miss and recompute, rather than
                # crashing a whole training run on one bad cache entry.
                pass

        result = compute_fn()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a per-process temp file, then atomic rename -- with
        # num_workers > 0, two DataLoader worker processes can race to
        # compute the SAME missing key (rare, but possible right at the
        # start of an epoch when the cache is still empty). Writing
        # directly to `cache_path` risks a torn/corrupt .npz if both
        # workers write it at once; os.replace() is atomic on both
        # POSIX and Windows, so whichever worker finishes last simply
        # wins cleanly -- the wasted duplicate compute is a real but
        # small cost, corruption would be a much worse one.
        tmp_path = cache_path.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez_compressed(tmp_path, **result)

        # Budget check uses the REAL on-disk (compressed) file size, not
        # sum(arr.nbytes) -- measured for this project's own 13-channel
        # RELLIS raw stack at ~3.82MB/frame actual compressed size versus
        # ~14.8MB/frame of raw nbytes (this data compresses ~3.9x), so an
        # nbytes-based budget was refusing writes at roughly 1/4 of what
        # the disk could actually hold. Written AFTER compression (the
        # true number), checked BEFORE the atomic rename -- an over-
        # budget frame's temp file is deleted, never committed to
        # `cache_path`, so a refusal here leaves no partial cache entry
        # behind for the next run to trip over.
        actual_bytes = tmp_path.stat().st_size
        if self._written_bytes + actual_bytes > self._free_at_start * self.max_disk_fraction:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"FrameCache at {self.cache_dir}: writing this frame would bring this run's "
                f"cache writes to {(self._written_bytes + actual_bytes) / 1e9:.2f}GB (real on-disk "
                f"size), exceeding {self.max_disk_fraction:.0%} of the {self._free_at_start / 1e9:.2f}GB "
                f"that was free when this cache started. Refusing to write further -- free disk space, "
                f"raise --max-disk-fraction if you have verified real headroom, or disable caching "
                f"for this dataset. This check exists because this project runs on a SHARED server "
                f"(see this module's own docstring) -- silently filling the disk is worse than a "
                f"training run failing loudly here."
            )
        os.replace(tmp_path, cache_path)
        self._written_bytes += actual_bytes
        return result

    @property
    def written_gb(self) -> float:
        return self._written_bytes / 1e9


def apply_radial_jitter_to_raw(raw: np.ndarray, factor: float) -> np.ndarray:
    """Array-level equivalent of perception.train._apply_radial_jitter,
    for use AFTER a cached raw stack is loaded -- caching must happen
    BEFORE jitter (on the un-jittered raw stack), or the "random" jitter
    would get baked into the cache file and repeat identically every
    epoch, defeating the point of the augmentation. Scales channels 0-3
    (x, y, z, range -- perception.input_tensor's own channel order) by
    the same factor, exactly matching the original's math; does NOT
    also rescale the normal/curvature channels, an identical scope
    limitation the original _apply_radial_jitter already has (this is a
    faithful port, not a stricter or looser reimplementation)."""
    raw = raw.copy()
    raw[0:4] *= factor
    return raw
