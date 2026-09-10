"""Ticket #8 tests."""

from perception.taxonomy import (
    DrishtiClass,
    NUSCENES_LIDARSEG_TO_DRISHTI,
    RELLIS_TO_DRISHTI,
    NO_SEMANTIC_MAP_CLASSES,
)


def test_no_source_class_maps_to_negative_obstacle_or_overhang():
    """Machine-checked form of the 'network never invents a hazard'
    boundary. Both NEGATIVE_OBSTACLE and OVERHANG must be reachable only
    through geometry (Tickets #21, #36), never a semantic label."""
    for name, mapping in (
        ("nuScenes-lidarseg", NUSCENES_LIDARSEG_TO_DRISHTI),
        ("RELLIS-3D", RELLIS_TO_DRISHTI),
    ):
        for source_class, drishti_class in mapping.items():
            assert drishti_class not in NO_SEMANTIC_MAP_CLASSES, (
                f"{name}:{source_class} must not map to {drishti_class.name}"
            )


def test_every_mapped_value_is_a_valid_drishti_class():
    for mapping in (NUSCENES_LIDARSEG_TO_DRISHTI, RELLIS_TO_DRISHTI):
        for drishti_class in mapping.values():
            assert isinstance(drishti_class, DrishtiClass)


def test_nuscenes_class_count():
    # 32 published nuScenes-lidarseg classes.
    assert len(NUSCENES_LIDARSEG_TO_DRISHTI) == 32


def test_rellis_class_count():
    # 19 documented classes + void = 20.
    assert len(RELLIS_TO_DRISHTI) == 20


def test_both_datasets_produce_same_reachable_drishti_class_set():
    """Not every DRISHTI class needs a source in every dataset, but the
    two datasets shouldn't produce disjoint semantic worlds either --
    this is a soft sanity check that flags an obviously broken mapping
    (e.g. one dataset never producing DRIVABLE) rather than a strict
    equality."""
    nuscenes_targets = set(NUSCENES_LIDARSEG_TO_DRISHTI.values())
    rellis_targets = set(RELLIS_TO_DRISHTI.values())
    core = {DrishtiClass.DRIVABLE, DrishtiClass.VEHICLE, DrishtiClass.PEDESTRIAN, DrishtiClass.VEGETATION}
    assert core.issubset(nuscenes_targets)
    assert core.issubset(rellis_targets)
