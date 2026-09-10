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

- RELLIS_TO_DRISHTI: keyed by class NAME, not numeric ID, and this is
  deliberate. The 19 (+ void) class NAMES are confirmed against the
  dataset's own documentation. The numeric IDs RELLIS-3D's
  `.label` files actually use were NOT confirmed against an authoritative
  ontology.yaml in this build -- I could not find one online. Before this
  mapping is used against real `.label` files, pull RELLIS-3D's own
  ontology config (it should ship inside the annotations download, e.g.
  something like `ontology.yaml` or embedded in the point_labeler tool)
  and confirm name<->ID, then swap `RELLIS_ID_TO_NAME` below for the real
  one. Shipping the wrong ID<->name mapping silently mislabels every
  point, so treat this file as blocked on that check, not done.
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
