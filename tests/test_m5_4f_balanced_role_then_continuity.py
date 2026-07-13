from __future__ import annotations

import json

from football_intelligence.replay.balanced_role_then_continuity import (
    ROLE_TARGETS,
    audit_continuity_review_selection,
    audit_role_review_selection,
    class_level_cluster_id_detected,
    continuity_equivalence_cluster_id,
    role_equivalence_cluster_id,
    select_balanced_role_cases,
)
from football_intelligence.review.schemas import CONTINUITY_DECISIONS, CONTINUITY_NOT_APPLICABLE_DECISION


def _role_case(index: int, prediction: str, frame: int, cluster: str | None = None) -> dict[str, object]:
    return {
        "review_case_id": f"role_{index}",
        "candidate_artifact_id": f"candidate_{index}",
        "model_prediction": prediction,
        "source_frame_sequence": frame,
        "equivalence_cluster_id": cluster or f"cluster_{index}",
        "evidence_manifest": {"source_bbox": {"x1": 20.0 + index, "y1": 40.0, "x2": 45.0 + index, "y2": 110.0}},
        "selection_metadata": {},
    }


def _continuity_case(index: int, bucket: str, gap: int, team: str, cluster: str) -> dict[str, object]:
    role = f"{team}_outfield_visual_context"
    return {
        "review_case_id": f"continuity_{index}",
        "candidate_artifact_id": f"edge_{index}",
        "category": bucket,
        "source_frame_sequence": 30 + index,
        "target_frame_sequence": 30 + index + gap,
        "equivalence_cluster_id": cluster,
        "uncertainty_reasons": [
            f"source_role={role}",
            f"target_role={role}",
        ],
        "selection_metadata": {
            "gate_features": {
                "frame_gap": gap,
                "bbox_iou": 0.4,
                "center_delta_px": 20.0,
                "footpoint_delta_px": 20.0,
                "bbox_area_ratio": 1.1,
                "aspect_ratio_change": 0.01,
            }
        },
        "evidence_manifest": {
            "frame_gap": gap,
            "source_bbox": {"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 100.0},
            "target_bbox": {"x1": 12.0, "y1": 22.0, "x2": 42.0, "y2": 102.0},
        },
    }


def test_m5_4e_role_pack_imbalance_missing_roles_and_frame_concentration_detected() -> None:
    cases = [
        *[_role_case(i, "unknown_visible_person_visual_context", i) for i in range(17)],
        *[_role_case(20 + i, "team_1_outfield_visual_context", 20 + i) for i in range(2)],
        *[_role_case(30 + i, "team_2_outfield_visual_context", 30 + i) for i in range(2)],
        *[_role_case(40 + i, "central_referee_visual_context", 40 + i) for i in range(2)],
        *[_role_case(50 + i, "assistant_referee_far_camera_context", 50 + i) for i in range(2)],
    ]
    audit = audit_role_review_selection({"review_cases": cases})

    assert audit["balanced_role_review"] is False
    assert "team_1_goalkeeper" in audit["missing_requested_classes"]
    assert "team_2_goalkeeper" in audit["missing_requested_classes"]
    assert "assistant_referee_near_camera" in audit["missing_requested_classes"]
    assert "frame_range_concentration" in audit["issues"]
    assert "category_concentration_exceeds_limit" in audit["issues"]
    assert audit["cases_per_proposed_class"]["unknown_or_disagreement_control"] == 17


def test_class_level_bucket_ids_are_rejected_as_real_equivalence_ids() -> None:
    assert class_level_cluster_id_detected("m5_4e_continuity_likely_positive_continuity", "likely_positive")
    assert class_level_cluster_id_detected("m5_4e_continuity_difficult_or_likely_negative_continuity")
    assert not class_level_cluster_id_detected("m5_4f_continuity_cluster_abcd1234", "likely_positive")
    role_cluster = role_equivalence_cluster_id(
        {"candidate_id": "c1", "frame_sequence": 101, "bbox": {"x1": 1, "y1": 2, "x2": 21, "y2": 62}},
        {"spatial_context": "playing_area_roi_candidate"},
    )
    assert "team_1" not in role_cluster


def test_continuity_audit_detects_frame_gap_shortcut_team_imbalance_and_cluster_leakage() -> None:
    cases = [
        *[
            _continuity_case(
                i,
                "likely_positive_continuity",
                1,
                "team_2",
                "m5_4e_continuity_likely_positive_continuity",
            )
            for i in range(10)
        ],
        *[
            _continuity_case(
                20 + i,
                "difficult_or_likely_negative_continuity",
                3,
                "team_1" if i < 3 else "team_2",
                "m5_4e_continuity_difficult_or_likely_negative_continuity",
            )
            for i in range(10)
        ],
    ]
    audit = audit_continuity_review_selection({"review_cases": cases})

    assert audit["potential_shortcut_features"]["frame_gap_perfectly_predicts_review_bucket"] is True
    assert audit["potential_shortcut_features"]["team_partition_imbalance"] is True
    assert audit["potential_shortcut_features"]["class_level_bucket_ids_used_as_equivalence_ids"] is True
    assert audit["equivalence_clusters_per_class"] == {"likely_negative": 1, "likely_positive": 1}
    assert audit["leakage_and_confounding_risk"] == "high"


def _candidate(candidate_id: str, bucket: str, frame: int, spatial: str = "region") -> dict[str, object]:
    quartile_index = frame // 150
    quartile_start = quartile_index * 150
    quartile_end = min(599, (quartile_index + 1) * 150 - 1)
    frame_quartile = f"q{quartile_index + 1}_{quartile_start:03d}_{quartile_end:03d}"
    return {
        "candidate_id": candidate_id,
        "target_review_bucket": bucket,
        "selection_score": 0.9,
        "frame_sequence": frame,
        "frame_quartile": frame_quartile,
        "thirty_frame_window": f"f{(frame // 30) * 30:03d}_{(frame // 30) * 30 + 29:03d}",
        "spatial_region_bucket": f"{spatial}:{candidate_id[-2:]}",
        "bbox_size_bucket": "medium_bbox",
        "role_equivalence_cluster_id": f"m5_4f_role_cluster_{candidate_id}",
        "visual_role_context_state": "unknown_visible_person_visual_context",
        "visual_role_context_confidence": 0.5,
        "belief_scores": {},
        "bbox": {"x1": 1.0, "y1": 2.0, "x2": 22.0, "y2": 82.0},
    }


def test_balanced_role_selection_bounds_category_spatial_window_and_spans_quartiles() -> None:
    records = {}
    for bucket_index, (bucket, target) in enumerate(ROLE_TARGETS.items()):
        rows = []
        for i in range(target + 6):
            frame = (i % 4) * 150 + ((i // 4) * 31 + bucket_index * 11) % 140
            rows.append(_candidate(f"{bucket}_{i:02d}", bucket, frame, spatial=bucket))
        records[bucket] = rows

    selected = select_balanced_role_cases(records, limit=40)
    category_counts = {}
    quartile_counts = {}
    for row in selected:
        category_counts[row["target_review_bucket"]] = category_counts.get(row["target_review_bucket"], 0) + 1
        quartile_counts[row["frame_quartile"]] = quartile_counts.get(row["frame_quartile"], 0) + 1

    assert len(selected) == 40
    assert max(category_counts.values()) <= 8
    assert min(quartile_counts.values()) >= 8


def test_continuity_cluster_ids_are_candidate_specific_not_bucket_level() -> None:
    a = continuity_equivalence_cluster_id(
        {
            "source_visible_person_base_id": "s1",
            "target_visible_person_base_id": "t1",
            "source_frame_sequence": 1,
            "target_frame_sequence": 2,
            "source_visual_role_context": "team_1_outfield_visual_context",
            "target_visual_role_context": "team_1_outfield_visual_context",
            "frame_gap": 1,
            "gate_features": {"center_delta_px": 10.0, "footpoint_delta_px": 10.0, "bbox_area_ratio": 1.0},
        }
    )
    b = continuity_equivalence_cluster_id(
        {
            "source_visible_person_base_id": "s2",
            "target_visible_person_base_id": "t2",
            "source_frame_sequence": 1,
            "target_frame_sequence": 2,
            "source_visual_role_context": "team_1_outfield_visual_context",
            "target_visual_role_context": "team_1_outfield_visual_context",
            "frame_gap": 1,
            "gate_features": {"center_delta_px": 10.0, "footpoint_delta_px": 10.0, "bbox_area_ratio": 1.0},
        }
    )
    assert a != b
    assert not class_level_cluster_id_detected(a, "likely_positive")


def test_n_continuity_decision_is_available_but_not_binary_negative() -> None:
    assert CONTINUITY_NOT_APPLICABLE_DECISION in CONTINUITY_DECISIONS
    training_labels = [
        "accept_continuity",
        "reject_continuity",
        CONTINUITY_NOT_APPLICABLE_DECISION,
        "unresolved",
    ]
    binary_usable = [label for label in training_labels if label in {"accept_continuity", "reject_continuity"}]

    assert binary_usable == ["accept_continuity", "reject_continuity"]
    assert CONTINUITY_NOT_APPLICABLE_DECISION not in binary_usable


def test_continuity_balance_requirements_overlap_gaps_and_bound_endpoints() -> None:
    cases = [
        _continuity_case(1, "likely_positive_continuity", 1, "team_1", "cluster_a1"),
        _continuity_case(2, "likely_positive_continuity", 2, "team_2", "cluster_a2"),
        _continuity_case(3, "likely_positive_continuity", 3, "team_1", "cluster_a3"),
        _continuity_case(4, "likely_negative_continuity", 1, "team_1", "cluster_b1"),
        _continuity_case(5, "likely_negative_continuity", 2, "team_2", "cluster_b2"),
        _continuity_case(6, "likely_negative_continuity", 3, "team_2", "cluster_b3"),
        _continuity_case(7, "likely_positive_continuity", 1, "team_2", "cluster_a4"),
        _continuity_case(8, "likely_positive_continuity", 2, "team_1", "cluster_a5"),
        _continuity_case(9, "likely_negative_continuity", 1, "team_2", "cluster_b4"),
        _continuity_case(10, "likely_negative_continuity", 2, "team_1", "cluster_b5"),
    ]
    audit = audit_continuity_review_selection({"review_cases": cases})
    positive_gaps = set(audit["frame_gap_distribution_by_proposed_class"]["likely_positive"])
    negative_gaps = set(audit["frame_gap_distribution_by_proposed_class"]["likely_negative"])

    assert positive_gaps & negative_gaps == {"1", "2", "3"}
    assert audit["potential_shortcut_features"]["frame_gap_perfectly_predicts_review_bucket"] is False
    assert audit["equivalence_clusters_per_class"] == {"likely_negative": 5, "likely_positive": 5}
    assert audit["endpoint_reuse_max"] <= 2


def test_audit_payloads_are_json_serializable() -> None:
    payload = audit_role_review_selection({"review_cases": [_role_case(1, "team_1_outfield_visual_context", 1)]})
    json.dumps(payload, sort_keys=True)
