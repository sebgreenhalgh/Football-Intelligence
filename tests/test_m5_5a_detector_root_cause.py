from __future__ import annotations

import pytest

from football_intelligence.replay.occlusion_detector_recovery_diagnostic import (
    BBox,
    aggregate_trajectory_regions,
    assert_allowed_classification,
    canonical_match_metrics,
    classify_case,
    crop_to_panorama_bbox,
    decode_spatial_annotation,
    detector_configurations,
    select_control_frames,
)


def test_localization_note_decoding_and_invalid_bbox_fail_closed() -> None:
    decoded = decode_spatial_annotation({"bbox": {"x1": 1, "y1": 2, "x2": 5, "y2": 8}, "candidate_number": 3})
    invalid = decode_spatial_annotation({"bbox": {"x1": 5, "y1": 2, "x2": 1, "y2": 8}})

    assert decoded["status"] == "visible_localized"
    assert invalid["status"] == "invalid_bbox"


def test_canonical_matching_by_iou_and_normalized_footpoint() -> None:
    loc = BBox(10, 10, 30, 70)
    iou_match = canonical_match_metrics(
        localization_bbox=loc,
        candidate_bbox=BBox(12, 12, 31, 71),
        candidate_id="candidate",
        confidence=0.8,
        original_radius_center=loc.center,
    )
    footpoint_match = canonical_match_metrics(
        localization_bbox=loc,
        candidate_bbox=BBox(15, 20, 35, 80),
        candidate_id="candidate2",
        confidence=0.7,
        original_radius_center=(500, 500),
    )

    assert iou_match["diagnostic_compatible_match"] is True
    assert footpoint_match["diagnostic_compatible_match"] is True
    assert footpoint_match["inside_original_140px_radius"] is False


def test_detector_config_hashes_are_deterministic_and_post_nms_limited() -> None:
    first = detector_configurations()
    second = detector_configurations()

    assert first == second
    assert all(row["pre_nms_evidence_status"] == "PRE_NMS_EVIDENCE_UNAVAILABLE" for row in first)


def test_detector_classification_blocks_proved_claims() -> None:
    result = classify_case(
        localization_status="visible_localized",
        canonical_matches=[],
        recovery_mechanisms=["relaxed_nms_post_nms_only"],
    )

    assert result == "VISIBLE_TARGET_NO_CANONICAL_MATCH_RECOVERED_BY_RELAXED_NMS_POST_NMS_ONLY"
    with pytest.raises(ValueError):
        assert_allowed_classification("PROVED_NMS_SUPPRESSION")


def test_controls_crop_mapping_and_004_016_region_aggregation() -> None:
    mapped = crop_to_panorama_bbox(BBox(1, 2, 3, 4), (10, 20))
    controls = select_control_frames(target_frame=145, all_frames=list(range(100, 200)), excluded_frames={145})
    aggregate = aggregate_trajectory_regions(
        [
            {"case_id": "case_004", "case_short_id": "004", "trajectory_safe_group_id": "shared"},
            {"case_id": "case_016", "case_short_id": "016", "trajectory_safe_group_id": "shared"},
        ]
    )

    assert mapped.to_dict() == {"x1": 11, "y1": 22, "x2": 13, "y2": 24}
    assert len(controls) == 2
    assert aggregate["case_004_016_share_region"] is True
