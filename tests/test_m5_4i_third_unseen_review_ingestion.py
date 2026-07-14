from __future__ import annotations

from pathlib import Path

from football_intelligence.replay.third_unseen_review_ingestion import (
    FROZEN_PRIMARY_THRESHOLDS,
    FROZEN_SECONDARY_THRESHOLD,
    _decode_decisions,
    _endpoint_revalidation_rows,
    _evaluate_rule,
    _label_novelty,
    _model_gate,
    _raw_edge_labels,
    _replay_review_events,
    _trajectory_safe_grouping,
    _validate_sealed_mapping,
)
from football_intelligence.replay.rebuilt_human_calibrated_pipeline import write_json
from football_intelligence.review_chassis.hashing import stable_hash


def _bbox(offset: float = 0.0) -> dict[str, float]:
    return {"x1": 10.0 + offset, "y1": 10.0, "x2": 30.0 + offset, "y2": 50.0}


def _event(sequence: int, case_id: str, decision: str, prior: str | None = None) -> dict[str, object]:
    return {
        "event_id": f"event_{sequence:03d}",
        "event_sequence": sequence,
        "event_type": "decision",
        "case_id": case_id,
        "new_decision": decision,
        "prior_decision": prior,
        "keyboard_or_click_input_source": "keyboard",
        "reviewer_session_id": "local-a4c73332",
    }


def _case_id(index: int) -> str:
    return f"m5_4h1_cadence_matched_target_choice_case_{index:03d}"


def _challenge(index: int, *, random_control: bool = False) -> dict[str, object]:
    source_frame = index
    target_frame = index + 1
    return {
        "challenge_candidate_id": f"challenge_{index:03d}",
        "source_candidate_id": f"source_{index:03d}",
        "source_visible_person_base_id": f"vpb_f{source_frame:06d}_source",
        "source_frame_sequence": source_frame,
        "target_frame_sequence": target_frame,
        "frame_gap": 1,
        "temporal_gap_seconds": 0.1,
        "source_bbox": _bbox(),
        "random_unseen_control": random_control,
        "target_options": [
            {
                "target_candidate_id": f"target_a_{index:03d}",
                "target_visible_person_base_id": f"vpb_f{target_frame:06d}_a",
                "target_frame_sequence": target_frame,
                "target_bbox": _bbox(1.0),
                "role_status": "UNKNOWN_NOT_CONTRADICTED",
                "team_status": "UNKNOWN_NOT_CONTRADICTED",
            },
            {
                "target_candidate_id": f"target_b_{index:03d}",
                "target_visible_person_base_id": f"vpb_f{target_frame:06d}_b",
                "target_frame_sequence": target_frame,
                "target_bbox": _bbox(40.0),
                "role_status": "UNKNOWN_NOT_CONTRADICTED",
                "team_status": "UNKNOWN_NOT_CONTRADICTED",
            },
        ],
    }


def _mapping(index: int) -> dict[str, object]:
    return {
        "case_id": _case_id(index),
        "challenge_candidate_id": f"challenge_{index:03d}",
        "source_candidate_id": f"source_{index:03d}",
        "source_visible_person_base_id": f"vpb_f{index:06d}_source",
        "target_a_candidate_id": f"target_a_{index:03d}",
        "target_a_visible_person_base_id": f"vpb_f{index + 1:06d}_a",
        "target_b_candidate_id": f"target_b_{index:03d}",
        "target_b_visible_person_base_id": f"vpb_f{index + 1:06d}_b",
        "endpoint_safe_group_id": f"endpoint_group_{index:03d}",
        "local_assignment_neighbourhood_id": f"neighbourhood_{index:03d}",
        "frozen_baseline_preferred_panel": "target_a",
        "challenge_categories": ["NEAR_IOU_THRESHOLD"],
        "registered_frozen_rule_outputs": {
            "target_a": {
                "bbox_iou": 0.4,
                "normalised_center_displacement": 0.2,
                "normalised_footpoint_displacement": 0.2,
                "primary_rule_accept": True,
                "secondary_threshold_accept": True,
                "appearance_similarity": 0.8,
            },
            "target_b": {
                "bbox_iou": 0.1,
                "normalised_center_displacement": 1.0,
                "normalised_footpoint_displacement": 1.0,
                "primary_rule_accept": False,
                "secondary_threshold_accept": False,
                "appearance_similarity": 0.6,
            },
        },
        "server_side_only": True,
        "browser_served_before_decision": False,
    }


def test_all_23_events_are_explained_and_final_20_decisions_reconstructed() -> None:
    case_ids = [_case_id(index) for index in range(1, 21)]
    events = [_event(index, case_id, "target_a_continues_source") for index, case_id in enumerate(case_ids, start=1)]
    events.append(_event(21, case_ids[-1], "target_b_continues_source", prior="target_a_continues_source"))
    events.append(_event(22, case_ids[-1], "target_b_continues_source", prior="target_b_continues_source"))
    events.append(
        {
            "event_id": "event_023",
            "event_sequence": 23,
            "event_type": "complete",
            "case_id": None,
            "new_decision": None,
            "reviewer_session_id": "local-a4c73332",
        }
    )

    replay = _replay_review_events(events, case_ids)

    assert replay["event_count"] == 23
    assert replay["all_events_explained"] is True
    assert replay["decision_count"] == 20
    assert replay["overwritten_decision_event_count"] == 2
    assert replay["completion_after_all_expected_decided"] is True
    assert replay["no_reveal_occurred"] is True


def test_decode_derives_16_ab_4_n_0_u_without_hard_coding() -> None:
    decisions = ["target_a_continues_source"] * 10 + ["target_b_continues_source"] * 6
    decisions += ["neither_target_is_valid_or_compatible"] * 4
    case_ids = [_case_id(index) for index in range(1, 21)]
    events = [_event(index, case_id, decisions[index - 1]) for index, case_id in enumerate(case_ids, start=1)]
    replay = _replay_review_events(events, case_ids)
    mapping_by_case = {_case_id(index): _mapping(index) for index in range(1, 21)}
    challenge_by_id = {f"challenge_{index:03d}": _challenge(index) for index in range(1, 21)}

    rows, summary = _decode_decisions(
        replay=replay,
        mapping_by_case=mapping_by_case,
        challenge_by_id=challenge_by_id,
    )

    assert len(rows) == 20
    assert summary["decisive_ab_count"] == 16
    assert summary["neither_count"] == 4
    assert summary["unresolved_count"] == 0


def test_sealed_mapping_validation_checks_hash_binding_and_server_only_policy(tmp_path: Path) -> None:
    stage_root = tmp_path
    review_root = stage_root / "continuity_v11" / "review"
    mapping = _mapping(1)
    sealed_payload = {
        "artifact": "test_sealed_mapping",
        "server_side_only": True,
        "browser_served_before_decision": False,
        "mappings": [mapping],
    }
    sealed_hash = stable_hash(sealed_payload)
    write_json(
        review_root / "sealed" / "target_choice_server_sealed_mapping.json",
        {**sealed_payload, "sealed_mapping_hash": sealed_hash},
    )
    write_json(review_root / "target_choice_server_sealed_reference.json", {"sealed_mapping_hash": sealed_hash})
    manifest_payload = {
        "cases": [
            {
                "case_id": _case_id(1),
                "source_frame_sequence": 1,
                "target_frame_sequence": 2,
            }
        ]
    }
    challenge_by_id = {"challenge_001": _challenge(1, random_control=True)}

    validation, mapping_by_case = _validate_sealed_mapping(
        stage_root=stage_root,
        expected_case_ids=[_case_id(1)],
        manifest_payload=manifest_payload,
        challenge_by_id=challenge_by_id,
    )

    assert validation["passed"] is True
    assert validation["mapping_was_not_browser_routable_before_completion"] is True
    assert mapping_by_case[_case_id(1)]["source_candidate_id"] == "source_001"
    assert validation["binding_rows"][0]["random_control_status"] is True


def test_n_creates_no_binary_label_and_ab_creates_at_most_positive_and_negative() -> None:
    case_ids = [_case_id(1), _case_id(2)]
    events = [
        _event(1, case_ids[0], "target_a_continues_source"),
        _event(2, case_ids[1], "neither_target_is_valid_or_compatible"),
    ]
    replay = _replay_review_events(events, case_ids)
    mapping_by_case = {_case_id(index): _mapping(index) for index in (1, 2)}
    challenge_by_id = {f"challenge_{index:03d}": _challenge(index) for index in (1, 2)}
    decoded, _summary = _decode_decisions(
        replay=replay,
        mapping_by_case=mapping_by_case,
        challenge_by_id=challenge_by_id,
    )
    manifest_payload = {
        "cases": [
            {"case_id": _case_id(1), "source_frame_sequence": 1, "target_frame_sequence": 2},
            {"case_id": _case_id(2), "source_frame_sequence": 2, "target_frame_sequence": 3},
        ]
    }
    _endpoint_rows, _audit, eligibility = _endpoint_revalidation_rows(
        decoded,
        mapping_by_case=mapping_by_case,
        challenge_by_id=challenge_by_id,
        manifest_payload=manifest_payload,
    )

    labels, non_binary = _raw_edge_labels(decoded, eligibility)

    assert len(labels) == 2
    assert {row["label"] for row in labels} == {"accept_continuity", "reject_continuity"}
    assert len(non_binary) == 1
    assert non_binary[0]["binary_labels_created"] == 0


def test_invalid_unchosen_endpoint_cannot_become_negative() -> None:
    case_id = _case_id(1)
    replay = _replay_review_events([_event(1, case_id, "target_a_continues_source")], [case_id])
    mapping_by_case = {case_id: _mapping(1)}
    challenge = _challenge(1)
    challenge["target_options"][1]["target_visible_person_base_id"] = "vpb_f999999_wrong_frame"
    challenge_by_id = {"challenge_001": challenge}
    decoded, _summary = _decode_decisions(
        replay=replay,
        mapping_by_case=mapping_by_case,
        challenge_by_id=challenge_by_id,
    )
    manifest_payload = {"cases": [{"case_id": case_id, "source_frame_sequence": 1, "target_frame_sequence": 2}]}
    _endpoint_rows, audit, eligibility = _endpoint_revalidation_rows(
        decoded,
        mapping_by_case=mapping_by_case,
        challenge_by_id=challenge_by_id,
        manifest_payload=manifest_payload,
    )
    labels, _non_binary = _raw_edge_labels(decoded, eligibility)

    assert audit["endpoint_invalid_case_count"] == 1
    assert labels == []


def test_canonical_deduplication_detects_exact_contradictions() -> None:
    positive = {
        "canonical_edge_key": "edge_1",
        "label": "accept_continuity",
        "source_candidate_id": "source",
        "target_candidate_id": "target",
        "source_frame_sequence": 1,
        "target_frame_sequence": 2,
    }
    raw_negative = {**positive, "label": "reject_continuity", "case_id": "case_001"}

    novelty, _new_pos, _new_neg, contradiction = _label_novelty(
        [raw_negative],
        historical_positive_rows=[positive],
        historical_negative_rows=[],
    )

    assert novelty["exact_edge_label_contradiction_count"] == 1
    assert contradiction["exact_edge_contradiction_count"] == 1


def test_trajectory_safe_groups_may_merge_beyond_endpoint_safe_groups() -> None:
    left = {
        "case_id": "case_001",
        "source_candidate_id": "source_a",
        "source_visible_person_base_id": "vpb_f000001_a",
        "source_frame_sequence": 10,
        "target_frame_sequence": 11,
        "source_bbox": _bbox(),
        "endpoint_safe_group_id": "endpoint_a",
        "selected_canonical_target": {"candidate_id": "target_a", "visible_person_base_id": "vpb_f000011_a"},
        "unselected_canonical_target": {"candidate_id": "target_b", "visible_person_base_id": "vpb_f000011_b"},
        "random_control_status": False,
    }
    right = {
        **left,
        "case_id": "case_002",
        "source_candidate_id": "source_c",
        "source_visible_person_base_id": "vpb_f000010_c",
        "endpoint_safe_group_id": "endpoint_b",
        "selected_canonical_target": {"candidate_id": "target_c", "visible_person_base_id": "vpb_f000011_c"},
        "unselected_canonical_target": {"candidate_id": "target_d", "visible_person_base_id": "vpb_f000011_d"},
    }

    audit, assignments = _trajectory_safe_grouping([left, right])

    assert audit["exact_endpoint_safe_group_count"] == 2
    assert audit["trajectory_safe_group_count"] == 1
    assert assignments["case_001"] == assignments["case_002"]


def test_frozen_thresholds_remain_unchanged_and_no_threshold_is_selected_from_review_labels() -> None:
    assert FROZEN_PRIMARY_THRESHOLDS == {
        "bbox_iou": 0.35,
        "normalised_center_displacement": 0.60,
        "normalised_footpoint_displacement": 0.80,
    }
    assert FROZEN_SECONDARY_THRESHOLD == 0.303375


def test_challenge_and_random_controls_are_evaluated_separately_and_n_is_candidate_failure() -> None:
    decoded_rows = [
        {
            "case_id": "case_001",
            "human_decision": "target_a_continues_source",
            "human_outcome": "TARGET_A_SELECTED",
            "frozen_primary_preferred_panel": "target_a",
            "frame_gap": 1,
            "random_control_status": False,
        },
        {
            "case_id": "case_002",
            "human_decision": "neither_target_is_valid_or_compatible",
            "human_outcome": "NEITHER_TARGET_VALID_OR_COMPATIBLE",
            "frozen_primary_preferred_panel": "target_b",
            "frame_gap": 1,
            "random_control_status": True,
        },
    ]

    result = _evaluate_rule(decoded_rows, "frozen_primary_preferred_panel", {"case_001": "g1", "case_002": "g2"})

    assert result["end_to_end_candidate_choice_success"]["correct_target_selected"] == 1
    assert result["end_to_end_candidate_choice_success"]["candidate_set_invalid"] == 1


def test_model_gate_never_fits_or_updates_learned_rows() -> None:
    gate = _model_gate(
        decoded_summary={"decisive_ab_count": 16, "neither_count": 4},
        primary_results={"agreement_count": 8},
        appearance={"appearance_corrections": 3, "appearance_regressions": 4},
        failure_taxonomy={"failure_counts": {"HUMAN_NEITHER": 4}},
    )

    assert gate["model_fit_performed"] is False
    assert gate["learned_continuity_rows_updated"] == 0
    assert gate["gate_does_not_authorize_model_application"] is True
