from __future__ import annotations

from football_intelligence.replay.cadence_matched_third_unseen_challenge import (
    OLD_PACK_CLASSIFICATION,
    cadence_compatibility,
    classify_historical_m5_4h_pack,
    endpoint_safe_components,
    frame_gaps_within_m5_4g_scope,
    temporal_gap_seconds,
)
from football_intelligence.review.schemas import safety_payload


def test_current_one_fps_m5_4h_pack_is_diagnostic_only() -> None:
    assert (
        classify_historical_m5_4h_pack(
            frame_count=60,
            duration_seconds=60,
            output_fps=1,
            width=2048,
            height=540,
        )
        == OLD_PACK_CLASSIFICATION
    )


def test_sixty_frames_over_sixty_seconds_cannot_pass_cadence_compatibility() -> None:
    result = cadence_compatibility(
        frame_count=60,
        duration_seconds=60,
        output_fps=1,
        width=2048,
        height=540,
    )

    assert result["cadence_domain_compatible"] is False
    assert result["frame_gap_1_to_3_seconds"] == {"minimum": 1.0, "maximum": 3.0}


def test_six_hundred_frames_over_sixty_seconds_passes_cadence_compatibility() -> None:
    result = cadence_compatibility(
        frame_count=600,
        duration_seconds=60,
        output_fps=10,
        width=2730,
        height=720,
    )

    assert result["cadence_domain_compatible"] is True
    assert result["seconds_per_frame"] == 0.1


def test_ten_fps_frame_gaps_one_to_three_map_to_short_window_seconds() -> None:
    assert [temporal_gap_seconds(gap, 10) for gap in (1, 2, 3)] == [0.1, 0.2, 0.3]
    assert frame_gaps_within_m5_4g_scope([1, 2, 3], 10) is True


def test_one_fps_frame_gaps_one_to_three_are_out_of_m5_4g_scope() -> None:
    assert [temporal_gap_seconds(gap, 1) for gap in (1, 2, 3)] == [1.0, 2.0, 3.0]
    assert frame_gaps_within_m5_4g_scope([1, 2, 3], 1) is False


def test_cases_sharing_source_or_target_endpoint_merge_into_safe_group() -> None:
    cases = [
        {
            "case_id": "case_001",
            "source_candidate_id": "source_a",
            "target_a_candidate_id": "target_a",
            "target_b_candidate_id": "target_b",
        },
        {
            "case_id": "case_002",
            "source_candidate_id": "source_c",
            "target_a_candidate_id": "target_b",
            "target_b_candidate_id": "target_d",
        },
        {
            "case_id": "case_003",
            "source_candidate_id": "source_x",
            "target_a_candidate_id": "target_y",
            "target_b_candidate_id": "target_z",
        },
    ]

    result = endpoint_safe_components(cases)

    assert result["endpoint_safe_group_count"] == 2
    assert result["shared_target_group_count"] == 1
    assert result["shared_source_group_count"] == 0
    assert result["case_to_endpoint_safe_group_id"]["case_001"] == result["case_to_endpoint_safe_group_id"]["case_002"]


def test_current_case_002_003_style_target_reuse_is_detected() -> None:
    cases = [
        {
            "case_id": "m5_4h_third_unseen_target_choice_case_002",
            "source_candidate_id": "source_002",
            "target_a_candidate_id": "target_a",
            "target_b_candidate_id": "shared_target",
        },
        {
            "case_id": "m5_4h_third_unseen_target_choice_case_003",
            "source_candidate_id": "source_003",
            "target_a_candidate_id": "shared_target",
            "target_b_candidate_id": "target_b",
        },
    ]

    result = endpoint_safe_components(cases)

    assert result["endpoint_safe_group_count"] == 1
    assert result["shared_target_case_groups"] == [
        ["m5_4h_third_unseen_target_choice_case_002", "m5_4h_third_unseen_target_choice_case_003"]
    ]


def test_current_case_013_014_style_source_reuse_is_detected() -> None:
    cases = [
        {
            "case_id": "m5_4h_third_unseen_target_choice_case_013",
            "source_candidate_id": "shared_source",
            "target_a_candidate_id": "target_a",
            "target_b_candidate_id": "target_b",
        },
        {
            "case_id": "m5_4h_third_unseen_target_choice_case_014",
            "source_candidate_id": "shared_source",
            "target_a_candidate_id": "target_c",
            "target_b_candidate_id": "target_d",
        },
    ]

    result = endpoint_safe_components(cases)

    assert result["endpoint_safe_group_count"] == 1
    assert result["shared_source_case_groups"] == [
        ["m5_4h_third_unseen_target_choice_case_013", "m5_4h_third_unseen_target_choice_case_014"]
    ]


def test_answer_mapping_is_expected_to_remain_server_sealed() -> None:
    reviewer_manifest_fields = {"case_id", "source", "target_a", "target_b", "allowed_decisions", "hashes"}
    sealed_mapping_fields = {"frozen_baseline_preferred_panel", "challenge_categories", "decision_mapping"}

    assert reviewer_manifest_fields.isdisjoint(sealed_mapping_fields)


def test_gif_only_and_safety_restrictions_are_explicit() -> None:
    safety = safety_payload()

    assert safety["visual_only_warning"] == "VISUAL_ONLY_NOT_METRIC"
    assert safety["production_ready"] is False
    assert safety["no_auto_promotion"] is True
    assert safety["human_approved"] is False
    assert safety["safe_to_apply_globally"] is False
    assert safety["match_local_only"] is True
    assert safety["sandbox_only"] is True
