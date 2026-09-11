"""
perception/taxonomy.py

Ticket #8 — Taxonomy remap.

Maps each dataset's native class list into the 10-class DRISHTI taxonomy
(Bible Part 6). Classes 8 (NEGATIVE_OBSTACLE) and 9 (OVERHANG) are
structural properties this system DERIVES from geometry (range shadows,
multi-layer cells) -- no dataset's semantic labels may map to them. A
source class landing on 8 or 9 would mean the network can "invent" a
hazard class from a label instead of the geometry earning it, which
breaks the whole "network never invents a hazard" boundary the Build Map
tests for.

Confirmation status of the two source lists:

- NUSCENES_LIDARSEG_TO_DRISHTI: keyed by nuScenes-lidarseg's 32 published
  class names, which are stable and well-documented -- reasonably
  confident these are correct as of nuscenes-devkit's current release.

- RELLIS_TO_DRISHTI: keyed by class NAME. RELLIS_ID_TO_NAME (below) maps
  the numeric IDs `.label` files actually store to those names.
  CROSS-CHECKED against real downloaded data, expanded 2026-09-11 from
  an earlier 20-frame/1-sequence check to 783 frames sampled across ALL
  5 local sequences (00000-00004): the observed numeric-ID set is
  {0,3,4,5,6,8,9,10,15,17,18,19,23,27,31,33,34} -- 17 of this table's 20
  entries, up from 9 -- and every one of them matches an entry already
  in RELLIS_ID_TO_NAME below (no surprise/unmapped ID ever fell through
  to the UNKNOWN fallback in `rellis_label_ids_to_drishti`). Newly
  confirmed since the first check: 5=pole, 6=water, 9=object, 10=asphalt,
  15=log, 18=fence, 23=concrete, 31=puddle.

  Still NOT observed anywhere across all 5 sequences: 1 (dirt), 7 (sky),
  12 (building). `sky` (7) may be permanently unobservable by a LiDAR
  regardless of how much more data is sampled -- there is no physical
  surface for a beam aimed at open sky to reflect off, so RELLIS-3D's
  camera-derived `sky` label plausibly never appears in the LiDAR
  `.label` files at all, which would make this a structural non-gap
  rather than a data-coverage gap. `dirt` (1) and `building` (12) remain
  genuine open items -- these 5 sequences' specific routes may simply
  not pass over bare dirt or past a building; if training produces odd
  per-class behaviour on either, check this table against the dataset's
  own `ontology.yaml` (ships with the annotations download) before
  assuming the network is at fault.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict


class DrishtiClass(IntEnum):
    UNKNOWN = 0
    DRIVABLE = 1
    CAUTION = 2
    NON_TRAVERSABLE = 3
    STATIC_OBSTACLE = 4
    VEGETATION = 5
    VEHICLE = 6
    PEDESTRIAN = 7
    NEGATIVE_OBSTACLE = 8   # geometry-derived only, see module docstring
    OVERHANG = 9            # geometry-derived only, see module docstring


NO_SEMANTIC_MAP_CLASSES = {DrishtiClass.NEGATIVE_OBSTACLE, DrishtiClass.OVERHANG}


# ---------------------------------------------------------------------------
# nuScenes-lidarseg (32 classes) -> DRISHTI
# ---------------------------------------------------------------------------

NUSCENES_LIDARSEG_TO_DRISHTI: Dict[str, DrishtiClass] = {
    "noise": DrishtiClass.UNKNOWN,

    "human.pedestrian.adult": DrishtiClass.PEDESTRIAN,
    "human.pedestrian.child": DrishtiClass.PEDESTRIAN,
    "human.pedestrian.wheelchair": DrishtiClass.PEDESTRIAN,
    "human.pedestrian.stroller": DrishtiClass.PEDESTRIAN,
    "human.pedestrian.personal_mobility": DrishtiClass.PEDESTRIAN,
    "human.pedestrian.police_officer": DrishtiClass.PEDESTRIAN,
    "human.pedestrian.construction_worker": DrishtiClass.PEDESTRIAN,

    "animal": DrishtiClass.STATIC_OBSTACLE,  # treated as an obstacle, not vegetation or a vehicle

    "vehicle.car": DrishtiClass.VEHICLE,
    "vehicle.motorcycle": DrishtiClass.VEHICLE,
    "vehicle.bicycle": DrishtiClass.VEHICLE,
    "vehicle.bus.bendy": DrishtiClass.VEHICLE,
    "vehicle.bus.rigid": DrishtiClass.VEHICLE,
    "vehicle.truck": DrishtiClass.VEHICLE,
    "vehicle.construction": DrishtiClass.VEHICLE,
    "vehicle.emergency.ambulance": DrishtiClass.VEHICLE,
    "vehicle.emergency.police": DrishtiClass.VEHICLE,
    "vehicle.trailer": DrishtiClass.VEHICLE,
    "vehicle.ego": DrishtiClass.UNKNOWN,  # points on the ego vehicle itself; should be filtered upstream

    "movable_object.barrier": DrishtiClass.NON_TRAVERSABLE,
    "movable_object.trafficcone": DrishtiClass.CAUTION,
    "movable_object.pushable_pullable": DrishtiClass.CAUTION,
    "movable_object.debris": DrishtiClass.NON_TRAVERSABLE,
    "static_object.bicycle_rack": DrishtiClass.STATIC_OBSTACLE,

    "flat.driveable_surface": DrishtiClass.DRIVABLE,
    "flat.sidewalk": DrishtiClass.CAUTION,   # not the intended path for a UGV, but not lethal
    "flat.terrain": DrishtiClass.DRIVABLE,
    "flat.other": DrishtiClass.UNKNOWN,

    "static.manmade": DrishtiClass.STATIC_OBSTACLE,
    "static.vegetation": DrishtiClass.VEGETATION,
    "static.other": DrishtiClass.STATIC_OBSTACLE,
}


# ---------------------------------------------------------------------------
# RELLIS-3D (19 classes + void) -> DRISHTI. Keyed by NAME -- see docstring
# for why the numeric-ID mapping is not yet trustworthy.
# ---------------------------------------------------------------------------

RELLIS_TO_DRISHTI: Dict[str, DrishtiClass] = {
    "void": DrishtiClass.UNKNOWN,
    "dirt": DrishtiClass.DRIVABLE,
    "grass": DrishtiClass.VEGETATION,
    "tree": DrishtiClass.VEGETATION,
    "pole": DrishtiClass.STATIC_OBSTACLE,
    "water": DrishtiClass.NON_TRAVERSABLE,   # "deep water" -- more severe than puddle, see Bible 9.5
    "sky": DrishtiClass.UNKNOWN,
    "vehicle": DrishtiClass.VEHICLE,
    "object": DrishtiClass.STATIC_OBSTACLE,   # catch-all
    "asphalt": DrishtiClass.DRIVABLE,
    "building": DrishtiClass.STATIC_OBSTACLE,
    "log": DrishtiClass.STATIC_OBSTACLE,
    "person": DrishtiClass.PEDESTRIAN,
    "fence": DrishtiClass.NON_TRAVERSABLE,
    "bush": DrishtiClass.VEGETATION,
    "concrete": DrishtiClass.DRIVABLE,
    "barrier": DrishtiClass.NON_TRAVERSABLE,
    "puddle": DrishtiClass.CAUTION,           # flat but possibly shallow -- caution, not lethal
    "mud": DrishtiClass.CAUTION,
    "rubble": DrishtiClass.NON_TRAVERSABLE,
}


# Numeric ID -> name, as RELLIS-3D's `.label` files actually store them
# (SemanticKITTI convention: uint32 per point, class id in the low 16
# bits). See module docstring for cross-check status: IDs
# {0,3,4,8,17,19,27,33,34} confirmed against real downloaded data; the
# rest taken from RELLIS-3D's published ontology, not independently
# re-verified in this build.
RELLIS_ID_TO_NAME: Dict[int, str] = {
    0: "void",
    1: "dirt",
    3: "grass",
    4: "tree",
    5: "pole",
    6: "water",
    7: "sky",
    8: "vehicle",
    9: "object",
    10: "asphalt",
    12: "building",
    15: "log",
    17: "person",
    18: "fence",
    19: "bush",
    23: "concrete",
    27: "barrier",
    31: "puddle",
    33: "mud",
    34: "rubble",
}


def rellis_label_ids_to_drishti(label_ids) -> "object":
    """Vectorised: raw RELLIS `.label` numeric IDs (any array-like of
    int) -> DrishtiClass IDs (numpy int64 array), via
    RELLIS_ID_TO_NAME -> RELLIS_TO_DRISHTI. An ID not present in
    RELLIS_ID_TO_NAME maps to DrishtiClass.UNKNOWN rather than raising --
    an unrecognised label is exactly what UNKNOWN means, not a crash.
    """
    import numpy as np

    label_ids = np.asarray(label_ids)
    out = np.full(label_ids.shape, int(DrishtiClass.UNKNOWN), dtype=np.int64)
    for raw_id, name in RELLIS_ID_TO_NAME.items():
        drishti_class = RELLIS_TO_DRISHTI.get(name)
        if drishti_class is None:
            continue
        out[label_ids == raw_id] = int(drishti_class)
    return out


def assert_taxonomy_valid() -> None:
    """Machine-checked form of the class-8/9 boundary. Run at import time
    (below) so a bad edit fails immediately, not just when the Ticket #8
    test happens to run."""
    for name, mapping in (
        ("nuScenes-lidarseg", NUSCENES_LIDARSEG_TO_DRISHTI),
        ("RELLIS-3D", RELLIS_TO_DRISHTI),
    ):
        for source_class, drishti_class in mapping.items():
            if drishti_class in NO_SEMANTIC_MAP_CLASSES:
                raise AssertionError(
                    f"{name} class '{source_class}' maps to {drishti_class.name}, "
                    f"which must be geometry-derived only. This breaks the "
                    f"'network never invents a hazard' boundary -- fix the mapping."
                )


assert_taxonomy_valid()
