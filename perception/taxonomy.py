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
    # fence/barrier moved from NON_TRAVERSABLE to STATIC_OBSTACLE
    # (DRISHTI_MASTER_BIBLE.md Part G.24) -- real point-count check
    # against all 5 local sequences found fence=0.098% and barrier=0.268%
    # of all points, versus STATIC_OBSTACLE's OWN existing members
    # (pole+object+building+log) totalling only 0.054% combined -- i.e.
    # fence+barrier together are ~8x the real prevalence of everything
    # already in STATIC_OBSTACLE, and physically they are the same kind
    # of thing the PS names as the class's own canonical example ("walls,
    # poles, and fixed vertical structures") that this project's five
    # prior STATIC_OBSTACLE attempts all failed to learn from ~0.06%
    # scarcity. Confirmed BEFORE this change that it is planning-safety-
    # neutral: `planning/conservatism.py` already maps BOTH
    # NON_TRAVERSABLE and STATIC_OBSTACLE to the identical
    # _KNOWN_HAZARD_COST -- moving fence/barrier between them changes
    # which bucket the SEGMENTATION NETWORK must learn, not how the
    # PLANNER treats a real fence/barrier point once classified, in
    # either bucket.
    "fence": DrishtiClass.STATIC_OBSTACLE,
    "bush": DrishtiClass.VEGETATION,
    "concrete": DrishtiClass.DRIVABLE,
    "barrier": DrishtiClass.STATIC_OBSTACLE,
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


# ---------------------------------------------------------------------------
# SemanticPOSS (Peking University campus dataset) -> DRISHTI. Keyed by
# NAME, same two-level ID->name->DrishtiClass pattern as RELLIS-3D above.
#
# Confirmation status: SEMANTICPOSS_ID_TO_NAME is transcribed directly
# from the dataset's OWN published `read_data.py` LABEL_DICT (shipped
# inside the downloaded SemanticPOSS_dataset.zip), not a third-party
# summary -- high confidence. One real discrepancy found and left
# UNRESOLVED rather than guessed at: id 20 was observed in real
# downloaded frame data (sequence 00, frame 0) but does NOT appear in
# read_data.py's own LABEL_DICT at all (which jumps 17 -> 21). Left
# out of SEMANTICPOSS_ID_TO_NAME below -- an ID this loader has never
# seen documented falls through to UNKNOWN via the same "unrecognised
# label IS what UNKNOWN means" stance as rellis_label_ids_to_drishti,
# rather than guessing what id 20 represents.
#
# Mapping decisions worth stating explicitly (Bible Part A.3's "state
# limitations" principle):
# - "rider" (a person on a bike/vehicle) has no dedicated DRISHTI class.
#   Mapped to PEDESTRIAN (protect-the-person priority) rather than
#   VEHICLE -- the entity to avoid hitting is a person-shaped moving
#   thing, matching this project's stated PEDESTRIAN = "highest
#   protection priority" framing (Bible Part C.6), not a judgment that
#   a rider behaves like a stationary pedestrian.
# - "traffic sign 2/3" (hanging/high-hanging signs) are physically
#   overhead obstacles -- exactly what OVERHANG exists to represent --
#   but NO dataset semantic label may EVER map to classes 8/9
#   (assert_taxonomy_valid enforces this at import time, Bible Part
#   C.6's "network never invents a hazard" boundary). Mapped to
#   STATIC_OBSTACLE instead, honestly under-representing their real
#   overhead nature rather than breaking that boundary.
# - "cone/stone" merges two real-world objects of very different
#   character (a soft traffic cone vs. a solid rock) into one dataset
#   label -- mapped to STATIC_OBSTACLE (the conservative choice: a
#   stone that gets treated as CAUTION would be the actually dangerous
#   direction to be wrong in, not the reverse).
SEMANTICPOSS_TO_DRISHTI: Dict[str, DrishtiClass] = {
    "unlabeled": DrishtiClass.UNKNOWN,
    "1 person": DrishtiClass.PEDESTRIAN,
    "2+ person": DrishtiClass.PEDESTRIAN,
    "rider": DrishtiClass.PEDESTRIAN,          # see docstring
    "car": DrishtiClass.VEHICLE,
    "trunk": DrishtiClass.STATIC_OBSTACLE,     # tree trunk -- solid, pole-like
    "plants": DrishtiClass.VEGETATION,
    "traffic sign 1": DrishtiClass.STATIC_OBSTACLE,   # standing sign
    "traffic sign 2": DrishtiClass.STATIC_OBSTACLE,   # hanging sign -- see docstring re: OVERHANG
    "traffic sign 3": DrishtiClass.STATIC_OBSTACLE,   # high/big hanging sign -- see docstring
    "pole": DrishtiClass.STATIC_OBSTACLE,
    "trashcan": DrishtiClass.STATIC_OBSTACLE,
    "building": DrishtiClass.STATIC_OBSTACLE,
    "cone/stone": DrishtiClass.STATIC_OBSTACLE,       # see docstring
    "fence": DrishtiClass.NON_TRAVERSABLE,
    "bike": DrishtiClass.VEHICLE,
    "ground": DrishtiClass.DRIVABLE,
}


# Numeric ID -> name, transcribed directly from SemanticPOSS's own
# read_data.py LABEL_DICT (see docstring above for the id-20 gap this
# deliberately leaves unmapped).
SEMANTICPOSS_ID_TO_NAME: Dict[int, str] = {
    0: "unlabeled",
    4: "1 person",
    5: "2+ person",
    6: "rider",
    7: "car",
    8: "trunk",
    9: "plants",
    10: "traffic sign 1",
    11: "traffic sign 2",
    12: "traffic sign 3",
    13: "pole",
    14: "trashcan",
    15: "building",
    16: "cone/stone",
    17: "fence",
    21: "bike",
    22: "ground",
}


def semanticposs_label_ids_to_drishti(label_ids) -> "object":
    """Vectorised: raw SemanticPOSS `.label` numeric IDs (low 16 bits
    already masked by the caller, same convention as RELLIS) -> DrishtiClass
    IDs (numpy int64 array). Mirrors rellis_label_ids_to_drishti exactly:
    an ID not present in SEMANTICPOSS_ID_TO_NAME (e.g. the real-but-
    undocumented id 20 -- see docstring above) maps to DrishtiClass.UNKNOWN
    rather than raising."""
    import numpy as np

    label_ids = np.asarray(label_ids)
    out = np.full(label_ids.shape, int(DrishtiClass.UNKNOWN), dtype=np.int64)
    for raw_id, name in SEMANTICPOSS_ID_TO_NAME.items():
        drishti_class = SEMANTICPOSS_TO_DRISHTI.get(name)
        if drishti_class is None:
            continue
        out[label_ids == raw_id] = int(drishti_class)
    return out


def build_nuscenes_lidarseg_lut(nusc) -> "object":
    """Raw uint8 lidarseg category index (0..31, as nuScenes' own
    `.bin` label files store per point) -> DrishtiClass int, as a 256-
    entry numpy lookup table. Shared by eval/eval_nuscenes.py (zero-shot
    eval) and perception/nuscenes_seg_dataset.py (fine-tune training) --
    moved here (single source of truth) rather than left duplicated in
    eval/eval_nuscenes.py, which is where this first lived (Ticket #2's
    zero-shot benchmark) before a training path needed the identical
    lookup. An index absent from `nusc.lidarseg_idx2name_mapping` (should
    not happen for a real nuScenes install) falls back to UNKNOWN rather
    than raising, matching `rellis_label_ids_to_drishti`'s own stance
    that an unrecognised label IS what UNKNOWN means."""
    import numpy as np

    lut = np.full(256, int(DrishtiClass.UNKNOWN), dtype=np.int64)
    for idx, name in nusc.lidarseg_idx2name_mapping.items():
        drishti_class = NUSCENES_LIDARSEG_TO_DRISHTI.get(name, DrishtiClass.UNKNOWN)
        lut[int(idx)] = int(drishti_class)
    return lut


def assert_taxonomy_valid() -> None:
    """Machine-checked form of the class-8/9 boundary. Run at import time
    (below) so a bad edit fails immediately, not just when the Ticket #8
    test happens to run."""
    for name, mapping in (
        ("nuScenes-lidarseg", NUSCENES_LIDARSEG_TO_DRISHTI),
        ("SemanticPOSS", SEMANTICPOSS_TO_DRISHTI),
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
