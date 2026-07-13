from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_intelligence.replay.geometry_matched_counterfactual_review import (
    F3_TRIVIAL_CLASSIFICATION,
    UNRESOLVED_CONTEXT,
    _audit_overlap,
    _candidate_quality_gate,
    _geometry_classifier_audit,
    _meaningful_role_compatible,
    audit_f3_counterfactual_pack,
    confirm_m5_4f4_smoke,
    direct_wrong_target_features,
    mine_local_counterfactual_candidates,
    mine_trajectory_swap_candidates,
)
from football_intelligence.review.server import _parse_byte_range


def _bbox(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _positive(case_id: str, frame: int, bbox: dict[str, float], component: str = "component_a") -> dict[str, object]:
    return {
        "review_case_id": case_id,
        "source_visible_person_base_id": f"{case_id}_source",
        "target_visible_person_base_id": f"{case_id}_target",
        "source_candidate_id": f"{case_id}_source_candidate",
        "target_candidate_id": f"{case_id}_target_candidate",
        "source_frame_sequence": frame,
        "target_frame_sequence": frame + 1,
        "frame_gap": 1,
        "team_partition": "team_1",
        "effective_role_context": "team_1_outfield_visual_context",
        "reviewed_or_reconciled_role_context": "team_1_outfield_visual_context",
        "accepted_local_visual_trajectory_component_id": component,
        "source_bbox": bbox,
        "target_bbox": _bbox(bbox["x1"] + 4, bbox["y1"], bbox["x2"] + 4, bbox["y2"]),
        "raw_features": {
            "bbox_iou": 0.72,
            "center_delta_px": 4.0,
            "footpoint_delta_px": 4.0,
            "continuity_score": 0.64,
            "appearance_similarity": 0.72,
        },
        "temporal_quartile": "q1_000_149",
        "thirty_frame_window": "f000_029",
        "source_spatial_bucket": "x0:y0",
    }


def _node(visible: str, frame: int, bbox: dict[str, float]) -> dict[str, object]:
    return {
        "visible_person_base_id": visible,
        "candidate_id": f"candidate_{visible}",
        "frame_sequence": frame,
        "continuity_eligible": True,
        "entity_validity_state": "valid_on_pitch_person",
        "bbox": bbox,
    }


def test_f3_trivial_shortcut_and_unresolved_role_equality_are_detected() -> None:
    candidate_rows = [
        {
            "source_to_alternative_center_delta_px": 80.0,
            "source_to_accepted_center_delta_px": 4.0,
            "accepted_target_iou_with_alternative": 0.0,
            "source_role_context": UNRESOLVED_CONTEXT,
            "alternative_role_context": UNRESOLVED_CONTEXT,
            "same_role_context": True,
        },
        {
            "source_to_alternative_center_delta_px": 120.0,
            "source_to_accepted_center_delta_px": 5.0,
            "accepted_target_iou_with_alternative": 0.0,
            "source_role_context": "team_1_outfield_visual_context",
            "alternative_role_context": "team_1_outfield_visual_context",
            "same_role_context": True,
        },
    ]
    manifest = {
        "review_cases": [
            {
                "review_case_id": "neg",
                "control_status": "not_control",
                "selection_metadata": {
                    "blind_hidden_model_info": {
                        "raw_features": {
                            "source_to_alternative_center_delta_px": 80.0,
                            "source_to_accepted_center_delta_px": 4.0,
                            "accepted_target_to_alternative_center_delta_px": 75.0,
                            "accepted_target_iou_with_alternative": 0.0,
                        }
                    },
                    "blind_context": {"team_partition": "team_1"},
                },
            },
            {
                "review_case_id": "control",
                "control_status": "positive_control",
                "selection_metadata": {
                    "blind_hidden_model_info": {
                        "raw_features": {
                            "source_to_alternative_center_delta_px": 4.0,
                            "source_to_accepted_center_delta_px": 4.0,
                            "accepted_target_to_alternative_center_delta_px": 0.0,
                            "accepted_target_iou_with_alternative": 1.0,
                        }
                    },
                    "blind_context": {"team_partition": "team_1"},
                },
            },
        ]
    }
    difficulty, shortcut = audit_f3_counterfactual_pack(candidate_rows=candidate_rows, manifest=manifest)

    assert difficulty["current_pack_classification"] == F3_TRIVIAL_CLASSIFICATION
    assert difficulty["unresolved_unresolved_role_equality_detected"] is True
    assert difficulty["one_dimensional_shortcut_detected"] is True
    assert shortcut["review_pack_unlock_allowed"] is False
    assert _meaningful_role_compatible(UNRESOLVED_CONTEXT, UNRESOLVED_CONTEXT) is False


def test_direct_wrong_target_features_are_explicit_and_role_compatibility_is_meaningful() -> None:
    features = direct_wrong_target_features(
        source_bbox=_bbox(100, 100, 140, 180),
        accepted_bbox=_bbox(104, 100, 144, 180),
        alternative_bbox=_bbox(138, 100, 178, 180),
        accepted_score=0.7,
        alternative_rank=2,
        local_candidate_density=3,
        source_role="team_1_outfield_visual_context",
        alternative_role="team_1_outfield_visual_context",
    )

    assert features["source_to_alternative_center_delta_px"] == 38.0
    assert features["source_to_alternative_normalised_center_delta"] < 0.5
    assert features["accepted_target_to_alternative_target_iou"] > 0.0
    assert features["alternative_candidate_rank"] == 2
    assert features["meaningful_role_compatibility"] is True


def test_local_counterfactual_mining_excludes_remote_and_selects_second_ranked_local_candidate() -> None:
    anchor = _positive("anchor", 10, _bbox(100, 100, 140, 180))
    remote = _node("remote", 11, _bbox(500, 100, 540, 180))
    good_alt = _node("good_alt", 11, _bbox(138, 100, 178, 180))
    nodes = [
        _node("anchor_target", 11, _bbox(104, 100, 144, 180)),
        good_alt,
        remote,
    ]
    roles = {
        "anchor_source": "team_1_outfield_visual_context",
        "anchor_target": "team_1_outfield_visual_context",
        "good_alt": "team_1_outfield_visual_context",
        "remote": "team_1_outfield_visual_context",
    }
    candidates, rejections = mine_local_counterfactual_candidates(
        positive_examples=[anchor],
        node_rows=nodes,
        role_by_visible=roles,
    )

    assert len(candidates) == 1
    assert candidates[0]["alternative_target_visible_person_base_id"] == "good_alt"
    assert candidates[0]["alternative_candidate_rank"] == 2
    assert any(row["reason"] == "remote_candidate_exceeds_three_bbox_heights" for row in rejections)


def test_trajectory_swap_candidates_require_same_team_and_meaningful_role() -> None:
    left = _positive("left", 20, _bbox(100, 100, 140, 180), component="component_left")
    right = _positive("right", 21, _bbox(130, 100, 170, 180), component="component_right")
    different_team = {**_positive("other_team", 21, _bbox(132, 100, 172, 180)), "team_partition": "team_2"}
    unresolved = {
        **_positive("unresolved", 21, _bbox(134, 100, 174, 180)),
        "reviewed_or_reconciled_role_context": UNRESOLVED_CONTEXT,
        "effective_role_context": UNRESOLVED_CONTEXT,
    }

    swaps = mine_trajectory_swap_candidates([left, right, different_team, unresolved])

    assert len(swaps) == 2
    assert {row["candidate_type"] for row in swaps} == {"trajectory_swap"}
    assert all(row["alternative_candidate_rank"] == 2 for row in swaps)


def test_geometry_only_shortcut_blocks_candidate_quality_gate() -> None:
    negatives = []
    controls = []
    for index in range(12):
        negative = {
            **direct_wrong_target_features(
                source_bbox=_bbox(100, 100, 140, 180),
                accepted_bbox=_bbox(104, 100, 144, 180),
                alternative_bbox=_bbox(114 + index, 100, 154 + index, 180),
                accepted_score=0.7,
                alternative_rank=2,
                local_candidate_density=2,
                source_role="team_1_outfield_visual_context",
                alternative_role="team_1_outfield_visual_context",
            ),
            "proposed_class": "counterfactual_negative",
        }
        control = {
            **negative,
            "proposed_class": "positive_control",
            "source_to_alternative_bbox_iou": 1.0,
            "source_to_alternative_center_delta_px": 2.0,
            "source_to_alternative_footpoint_delta_px": 2.0,
            "source_to_alternative_normalised_center_delta": 0.02,
            "source_to_alternative_normalised_footpoint_delta": 0.02,
            "alternative_candidate_rank": 1,
            "local_candidate_density": 1,
        }
        negatives.append(negative)
        controls.append(control)
    overlap = _audit_overlap(negatives, controls)
    classifier = _geometry_classifier_audit(negatives, controls)
    passed, blocker = _candidate_quality_gate(
        negatives=negatives,
        controls=controls,
        overlap_audit=overlap,
        classifier_audit=classifier,
    )

    assert overlap["passes_raw_feature_overlap_gates"] is False
    assert classifier["geometry_only_grouped_balanced_accuracy"] == 1.0
    assert passed is False
    assert blocker in {"RAW_FEATURE_OVERLAP_GATE_FAILED", "GEOMETRY_ONLY_SHORTCUT_GATE_FAILED"}


def test_manual_smoke_confirmation_is_atomic_and_requires_exactly_one_state(tmp_path: Path) -> None:
    payload = confirm_m5_4f4_smoke(stage_root=tmp_path, passed=False, failed=True, reason="mp4 duration 0:00")
    path = tmp_path / "continuity_v4" / "audit" / "manual_smoke_confirmation.json"

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["reason"] == "mp4 duration 0:00"
    assert payload["manual_smoke_confirmation_failed"] is True
    assert not path.with_suffix(".tmp").exists()
    with pytest.raises(ValueError, match="exactly one"):
        confirm_m5_4f4_smoke(stage_root=tmp_path, passed=True, failed=True, reason=None)


def test_review_server_byte_range_parser_supports_browser_video_ranges() -> None:
    assert _parse_byte_range("bytes=0-99", 1000) == (0, 99)
    assert _parse_byte_range("bytes=900-", 1000) == (900, 999)
    assert _parse_byte_range("bytes=-50", 1000) == (950, 999)
    assert _parse_byte_range("bytes=1000-1001", 1000) is None
    assert _parse_byte_range("bytes=a-b", 1000) is None
    assert _parse_byte_range("items=0-99", 1000) is None
