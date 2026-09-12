"""
perception/nuscenes_boxes.py

Real 3D bounding-box targets for the new detection head (Part I of this
session's work -- see the deep-research report on lightweight range-view
3D detection this was built in response to). nuScenes-mini ships REAL
3D box annotations for every one of its 404 samples (confirmed this
session: 18,538 total `sample_annotation` records) -- a genuinely
different, richer supervision signal than the per-point semantic labels
`perception/nuscenes_seg_dataset.py` already uses. This module is the
bridge between that box data and per-pixel regression targets in the
SAME range-image space `perception.range_image.project_to_range_image`
already produces, so the detection head can share FusionSegNet's
existing backbone rather than needing a second projection pipeline.

Design, and why it's built this way rather than re-deriving projection
math from scratch: nuScenes stores box pose in the GLOBAL frame.
Rather than hand-rolling the spherical-projection formulas the research
report describes (theta=atan2(y,x), phi=arcsin(z/r)) a second time,
this module:
  1. Transforms each box from global -> ego -> SENSOR frame using the
     exact same pose composition perception.nuscenes_loader already
     uses (so a box's frame convention matches sweep.xyz's frame
     convention exactly -- both are sensor-frame after this step).
  2. Uses nuscenes-devkit's OWN `points_in_box()` (a tested, canonical
     utility, not reimplemented) to find which of the sweep's raw 3D
     points fall inside each box.
  3. Uses the RangeImage's own `point_index` array (built by
     project_to_range_image) to look up which pixel each of those
     points ALREADY won during projection -- a point that lost the
     many-to-one occlusion competition for its pixel (see
     project_to_range_image's own docstring) contributes NO training
     signal here, which is correct: it isn't the value that pixel
     actually holds.

Regression target convention (per pixel, only defined where a box is
present): offset = box_center_sensor_frame - point_xyz (point-to-center
vector, matching the report's "regress center-ness/offset" pattern from
SVM/CenterPoint-style heads), dims = box wlh (constant across every
pixel of that box), yaw_sincos = (sin(yaw), cos(yaw)) rather than raw
yaw radians -- sin/cos avoids the 0/2pi wraparound discontinuity a
network would otherwise have to learn around, standard practice for
any angle regression target.

Known simplification, stated rather than hidden: yaw is extracted via
the sensor-frame quaternion's own yaw_pitch_roll decomposition, which
assumes the LiDAR mount has negligible pitch/roll relative to the
ground plane (true for essentially every real vehicle mount, including
nuScenes' own calibration). A mount with real pitch/roll would need the
full 3D orientation, not a single yaw angle -- not needed here.

Which classes get box targets: nuScenes' 3D boxes exist only for
discrete, instance-like objects (people, vehicles, animals, movable
objects) -- there are no boxes for DRIVABLE/VEGETATION/UNKNOWN, so
DETECTABLE_DRISHTI_CLASSES below is exactly the object-like subset of
the taxonomy (perception/taxonomy.py), reusing the SAME
NUSCENES_LIDARSEG_TO_DRISHTI mapping already validated for the
segmentation path -- category names in nuScenes' `category.json` are
shared between lidarseg and box annotations, not two different tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from perception.range_image import RangeImage
from perception.taxonomy import NUSCENES_LIDARSEG_TO_DRISHTI, DrishtiClass

# Only these DRISHTI classes are ever instance-like enough for nuScenes
# to ship a 3D box for -- PEDESTRIAN and VEHICLE cover the vast
# majority of real annotations; STATIC_OBSTACLE picks up the rarer
# "animal" / "static_object.bicycle_rack" box categories.
DETECTABLE_DRISHTI_CLASSES = {
    int(DrishtiClass.PEDESTRIAN),
    int(DrishtiClass.VEHICLE),
    int(DrishtiClass.STATIC_OBSTACLE),
}

N_REGRESSION_CHANNELS = 8  # offset(3) + dims(3) + yaw_sincos(2)


@dataclass
class BoxTargets:
    """Per-pixel detection targets, same (H, W) as the RangeImage they
    were built against. `objectness` is 1.0 exactly at pixels covered by
    a real box, 0.0 elsewhere. `class_id` is -1 where objectness is 0
    (never a valid DrishtiClass there -- checked by the loss, which
    masks on objectness, not on class_id, to avoid ever training on a
    -1 "class")."""

    objectness: np.ndarray  # (H, W) float32, {0.0, 1.0}
    class_id: np.ndarray  # (H, W) int64, -1 where objectness == 0
    regression: np.ndarray  # (8, H, W) float32: dx,dy,dz,w,l,h,sin_yaw,cos_yaw -- meaningless where objectness == 0
    n_boxes_with_any_pixel: int  # diagnostic: how many of this frame's real boxes actually won at least one pixel


def _box_to_sensor_frame(nusc, box, sd_token: str):
    """Global-frame Box -> SENSOR-frame Box, via the exact global->ego->
    sensor composition perception.nuscenes_loader.load_nuscenes_sweep
    already uses (inverted) -- see module docstring."""
    sd = nusc.get("sample_data", sd_token)

    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
    box = box.copy()
    box.translate(-np.array(ego_pose["translation"]))
    box.rotate(_quat_inverse(ego_pose["rotation"]))

    calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    box.translate(-np.array(calib["translation"]))
    box.rotate(_quat_inverse(calib["rotation"]))

    return box


def _quat_inverse(rotation_quat_wxyz):
    from pyquaternion import Quaternion

    return Quaternion(rotation_quat_wxyz).inverse


def build_box_targets(nusc, sample_token: str, sweep_xyz: np.ndarray, img: RangeImage) -> BoxTargets:
    """Build per-pixel detection targets for one nuScenes sample.
    `sweep_xyz` must be the SAME raw sensor-frame points array
    `img` (a RangeImage) was projected from -- `img.point_index` indexes
    into it."""
    from nuscenes.utils.geometry_utils import points_in_box

    H, W = img.H, img.W
    objectness = np.zeros((H, W), dtype=np.float32)
    class_id = np.full((H, W), -1, dtype=np.int64)
    regression = np.zeros((N_REGRESSION_CHANNELS, H, W), dtype=np.float32)

    # Inverse of img.point_index: point_idx -> flat pixel index, -1 if
    # that point never won a pixel during projection (occluded / lost
    # the many-to-one competition). See module docstring.
    n_points = sweep_xyz.shape[0]
    point_to_flat_pixel = np.full(n_points, -1, dtype=np.int64)
    touched = img.point_index >= 0
    flat_idx_grid = np.arange(H * W, dtype=np.int64).reshape(H, W)
    point_to_flat_pixel[img.point_index[touched]] = flat_idx_grid[touched]

    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"]["LIDAR_TOP"]

    n_boxes_with_any_pixel = 0
    points_T = sweep_xyz.T.astype(np.float64)  # (3, n) -- points_in_box's own expected shape

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        category_name = ann["category_name"]
        drishti_class = NUSCENES_LIDARSEG_TO_DRISHTI.get(category_name)
        if drishti_class is None or int(drishti_class) not in DETECTABLE_DRISHTI_CLASSES:
            continue  # not an instance-like class this detection head targets (see module docstring)

        box = nusc.get_box(ann_token)
        box = _box_to_sensor_frame(nusc, box, sd_token)

        inside = points_in_box(box, points_T)
        if not np.any(inside):
            continue

        flat_pixels = point_to_flat_pixel[inside]
        flat_pixels = flat_pixels[flat_pixels >= 0]  # drop points that never won a pixel
        # Skip pixels ALREADY claimed by an earlier (closer-processed) box --
        # deterministic "first box wins" on the rare occlusion overlap,
        # rather than silently overwriting -- see module docstring.
        flat_pixels = flat_pixels[objectness.flat[flat_pixels] == 0.0]
        if flat_pixels.size == 0:
            continue
        n_boxes_with_any_pixel += 1

        rows, cols = np.unravel_index(flat_pixels, (H, W))
        box_points = sweep_xyz[inside]
        # Re-derive which of `box_points` correspond to `flat_pixels` --
        # `inside` and `point_to_flat_pixel[inside]` are aligned by
        # construction, so re-filter box_points the same way.
        inside_point_indices = np.nonzero(inside)[0]
        kept_mask = point_to_flat_pixel[inside_point_indices] >= 0
        kept_mask &= objectness.flat[point_to_flat_pixel[inside_point_indices].clip(min=0)] == 0.0
        box_points_kept = sweep_xyz[inside_point_indices[kept_mask]]

        offset = box.center[None, :] - box_points_kept  # (n_kept, 3): point -> box-center vector
        w, l, h = box.wlh
        sin_yaw = np.sin(box.orientation.yaw_pitch_roll[0])
        cos_yaw = np.cos(box.orientation.yaw_pitch_roll[0])

        objectness.flat[flat_pixels] = 1.0
        class_id.flat[flat_pixels] = int(drishti_class)
        for i, (r, c) in enumerate(zip(rows, cols)):
            regression[0, r, c] = offset[i, 0]
            regression[1, r, c] = offset[i, 1]
            regression[2, r, c] = offset[i, 2]
            regression[3, r, c] = w
            regression[4, r, c] = l
            regression[5, r, c] = h
            regression[6, r, c] = sin_yaw
            regression[7, r, c] = cos_yaw

    return BoxTargets(
        objectness=objectness,
        class_id=class_id,
        regression=regression,
        n_boxes_with_any_pixel=n_boxes_with_any_pixel,
    )
