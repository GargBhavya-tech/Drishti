"""Ticket #8 tests."""

from pathlib import Path

import pytest

from perception.taxonomy import (
    DrishtiClass,
    NUSCENES_LIDARSEG_TO_DRISHTI,
    RELLIS_TO_DRISHTI,
    RELLIS_ID_TO_NAME,
    NO_SEMANTIC_MAP_CLASSES,
    rellis_label_ids_to_drishti,
)
from perception.rellis_loader import load_rellis_labels

RELLIS_ROOT = Path(__file__).resolve().parents[1] / "data" / "rellis"
SEQUENCES_AVAILABLE = [s for s in ("00000", "00001", "00002", "00003", "00004") if (RELLIS_ROOT / s).exists()]


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


@pytest.mark.skipif(not SEQUENCES_AVAILABLE, reason="no local RELLIS-3D sequences present")
def test_real_rellis_label_ids_never_produce_a_surprise_unmapped_id():
    """Regression form of the module docstring's expanded cross-check
    (783 real frames across all 5 local sequences, 2026-09-11): every
    numeric ID actually observed in real RELLIS-3D `.label` files must
    be a key in RELLIS_ID_TO_NAME -- an ID that ISN'T would silently
    fall through to UNKNOWN in `rellis_label_ids_to_drishti` without
    anyone noticing, which is exactly the gap this test closes. Uses a
    light sample (not all 783 frames) to stay fast; the point is
    catching a NEW unmapped ID in future data, not re-deriving the
    module docstring's own numbers every run."""
    observed_ids = set()
    for seq in SEQUENCES_AVAILABLE:
        bin_dir = RELLIS_ROOT / seq / "os1_cloud_node_kitti_bin"
        n_frames = len(list(bin_dir.glob("*.bin")))
        stride = max(1, n_frames // 15)
        for f in range(0, n_frames, stride):
            labels = load_rellis_labels(RELLIS_ROOT / seq, f)
            observed_ids.update(int(i) for i in set(labels.tolist()))

    unmapped = observed_ids - set(RELLIS_ID_TO_NAME.keys())
    assert not unmapped, (
        f"real RELLIS-3D data contains label ID(s) {unmapped} not present in "
        f"RELLIS_ID_TO_NAME -- these are silently mapping to UNKNOWN via "
        f"rellis_label_ids_to_drishti's fallback; add them to the table."
    )

    # And the machine-checked form of the docstring's own finding: at
    # least the previously-confirmed set should still be observed.
    previously_confirmed = {0, 3, 4, 8, 17, 19, 27, 33, 34}
    assert previously_confirmed.issubset(observed_ids)


@pytest.mark.skipif(not SEQUENCES_AVAILABLE, reason="no local RELLIS-3D sequences present")
def test_real_rellis_labels_remap_without_crashing():
    seq = SEQUENCES_AVAILABLE[0]
    labels = load_rellis_labels(RELLIS_ROOT / seq, frame_idx=0)
    drishti_ids = rellis_label_ids_to_drishti(labels)
    assert drishti_ids.shape == labels.shape
    assert set(drishti_ids.tolist()).issubset(set(int(c) for c in DrishtiClass))
