from __future__ import annotations

from football_intelligence.replay.server_sealed_target_choice_ingestion import (
    _baseline_results,
    _canonical_edge_key,
    _decode_decisions,
    _endpoint_leakage_audit,
    _exact_edge_contradiction_audit,
    _negative_novelty_audit,
    _positive_confirmation_deduplication,
    _replay_review_events,
    _training_readiness,
)


def _bbox(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _event(sequence: int, case_id: str, decision: str) -> dict[str, object]:
    return {
        "event_id": f"event_{sequence}",
        "event_sequence": sequence,
        "event_type": "decision",
        "case_id": case_id,
        "prior_decision": None,
        "new_decision": decision,
        "keyboard_or_click_input_source": "click",
        "reviewer_session_id": "session",
        "evidence_hash": f"evidence_{case_id}",
    }


def _row() -> dict[str, object]:
    return {
        "candidate_id": "anchor_001",
        "source_candidate_id": "source_candidate",
        "source_visible_person_base_id": "source_vpb",
        "source_frame_sequence": 10,
        "target_frame_sequence": 12,
        "frame_gap": 2,
        "source_bbox": _bbox(10, 10, 30, 50),
        "accepted_target_candidate_id": "accepted_candidate",
        "accepted_target_visible_person_base_id": "accepted_vpb",
        "accepted_target_bbox": _bbox(12, 11, 32, 51),
        "alternative_target_candidate_id": "alternative_candidate",
        "alternative_target_visible_person_base_id": "alternative_vpb",
        "alternative_target_bbox": _bbox(120, 110, 140, 150),
        "local_assignment_neighbourhood_id": "neighbourhood_001",
        "candidate_type": "review_only_local_same_frame_wrong_target",
        "compatibility_status": "review_only",
        "compatibility_uncertainty": "generic_visible_person",
    }


def _mapping(accepted_panel: str = "target_a") -> dict[str, object]:
    if accepted_panel == "target_a":
        target_a = ("accepted_candidate", "accepted_vpb")
        target_b = ("alternative_candidate", "alternative_vpb")
    else:
        target_a = ("alternative_candidate", "alternative_vpb")
        target_b = ("accepted_candidate", "accepted_vpb")
    return {
        "case_id": "case_001",
        "source_anchor_candidate_id": "anchor_001",
        "accepted_target_panel": accepted_panel,
        "target_a_candidate_id": target_a[0],
        "target_b_candidate_id": target_b[0],
        "target_a_visible_person_base_id": target_a[1],
        "target_b_visible_person_base_id": target_b[1],
        "local_assignment_neighbourhood_id": "neighbourhood_001",
        "candidate_construction_type": "review_only_local_same_frame_wrong_target",
        "decision_mapping": {
            "target_a_continues_source": {
                "chosen_panel": "target_a",
                "chosen_visible_person_base_id": target_a[1],
                "creates_binary_labels_when_decisive": True,
                "conflict_if_chosen_panel_is_not_prior_accept": accepted_panel != "target_a",
            },
            "target_b_continues_source": {
                "chosen_panel": "target_b",
                "chosen_visible_person_base_id": target_b[1],
                "creates_binary_labels_when_decisive": True,
                "conflict_if_chosen_panel_is_not_prior_accept": accepted_panel != "target_b",
            },
            "neither_target_is_valid_or_compatible": {"creates_binary_labels_when_decisive": False},
            "unresolved": {"creates_binary_labels_when_decisive": False},
        },
    }


def _label(label: str, source: str, target: str, group: str, iou: float, displacement: float) -> dict[str, object]:
    return {
        "canonical_edge_key": _canonical_edge_key(
            source_candidate_id=source,
            target_candidate_id=target,
            source_frame_sequence=10,
            target_frame_sequence=12,
        ),
        "label": label,
        "source_candidate_id": source,
        "target_candidate_id": target,
        "source_visible_person_base_id": f"{source}_vpb",
        "target_visible_person_base_id": f"{target}_vpb",
        "evaluation_group_id": group,
        "local_assignment_neighbourhood_id": group,
        "features": {
            "bbox_iou": iou,
            "normalised_center_displacement": displacement,
            "normalised_footpoint_displacement": displacement,
            "appearance_similarity": 0.9 if label == "accept_continuity" else 0.3,
        },
    }


def test_final_decisions_are_replayed_from_events_and_no_reveal_occurred() -> None:
    expected = [f"case_{index:03d}" for index in range(1, 7)]
    events = [_event(index, case_id, "target_a_continues_source") for index, case_id in enumerate(expected, start=1)]
    events.append(
        {
            "event_id": "event_7",
            "event_sequence": 7,
            "event_type": "complete",
            "case_id": None,
            "new_decision": None,
            "reviewer_session_id": "session",
        }
    )

    replay = _replay_review_events(events, expected)

    assert replay["decision_count"] == 6
    assert replay["missing_cases"] == []
    assert replay["no_reveal_occurred"] is True
    assert replay["completion_after_all_expected_decided"] is True
    assert replay["event_sequence_monotonic"] is True


def test_reveal_before_decision_is_detected() -> None:
    events = [
        {
            "event_id": "event_1",
            "event_sequence": 1,
            "event_type": "reveal",
            "case_id": "case_001",
            "reviewer_session_id": "session",
        },
        _event(2, "case_001", "target_a_continues_source"),
    ]

    replay = _replay_review_events(events, ["case_001"])

    assert replay["no_reveal_before_persisted_decision"] is False
    assert replay["no_reveal_occurred"] is False


def test_sealed_mapping_decodes_chosen_and_unchosen_panels() -> None:
    replay = _replay_review_events([_event(1, "case_001", "target_b_continues_source")], ["case_001"])
    decoded, summary = _decode_decisions(
        replay=replay,
        mapping_by_case={"case_001": _mapping(accepted_panel="target_b")},
        row_by_anchor={"anchor_001": _row()},
    )

    assert summary["agreement_count"] == 1
    assert summary["conflict_count"] == 0
    assert decoded[0]["chosen_displayed_panel"] == "target_b"
    assert decoded[0]["unchosen_displayed_panel"] == "target_a"
    assert decoded[0]["chosen_canonical_candidate_id"] == "accepted_candidate"
    assert decoded[0]["unchosen_canonical_candidate_id"] == "alternative_candidate"


def test_conflict_and_n_u_decisions_create_no_positive_agreement() -> None:
    conflict_replay = _replay_review_events([_event(1, "case_001", "target_b_continues_source")], ["case_001"])
    decoded, summary = _decode_decisions(
        replay=conflict_replay,
        mapping_by_case={"case_001": _mapping(accepted_panel="target_a")},
        row_by_anchor={"anchor_001": _row()},
    )
    assert decoded[0]["decoded_outcome"] == "REVIEW_CONFLICT_WITH_PRIOR_ACCEPTED_TARGET"
    assert summary["conflict_count"] == 1

    unresolved = _replay_review_events([_event(1, "case_001", "unresolved")], ["case_001"])
    _decoded, unresolved_summary = _decode_decisions(
        replay=unresolved,
        mapping_by_case={"case_001": _mapping()},
        row_by_anchor={"anchor_001": _row()},
    )
    assert unresolved_summary["unresolved_count"] == 1
    assert unresolved_summary["agreement_count"] == 0


def test_positive_confirmations_do_not_create_new_independent_positives() -> None:
    f2_positive = _label("accept_continuity", "source_candidate", "accepted_candidate", "component_001", 0.8, 0.1)
    raw_confirmation = {
        **f2_positive,
        "case_id": "case_001",
        "label_source": "m5_4f6_2_server_sealed_target_choice_review",
    }

    report, canonical = _positive_confirmation_deduplication(
        raw_labels=[raw_confirmation],
        f2_positive_rows=[f2_positive],
        positive_component_count=1,
    )

    assert report["exact_positive_confirmation_count"] == 1
    assert report["new_distinct_positive_count"] == 0
    assert report["canonical_unique_positive_count"] == 1
    assert canonical[0]["m5_4g_positive_confirmation_count"] == 1


def test_negative_novelty_distinguishes_endpoint_reuse_from_exact_contradiction() -> None:
    positive = _label("accept_continuity", "source_candidate", "accepted_candidate", "component_001", 0.8, 0.1)
    negative = _label("reject_continuity", "source_candidate", "alternative_candidate", "neighbourhood_001", 0.0, 3.0)
    negative.update({"case_id": "case_001", "local_assignment_neighbourhood_id": "neighbourhood_001"})

    report, canonical = _negative_novelty_audit(
        raw_labels=[negative],
        positive_keys={positive["canonical_edge_key"]},
        historical_keys={positive["canonical_edge_key"]},
    )
    contradiction = _exact_edge_contradiction_audit([positive], canonical)

    assert report["canonical_unique_negative_count"] == 1
    assert report["exact_edge_label_contradiction_count"] == 0
    assert contradiction["exact_edge_contradiction_count"] == 0

    contradictory_report, _canonical = _negative_novelty_audit(
        raw_labels=[{**positive, "label": "reject_continuity", "case_id": "case_002"}],
        positive_keys={positive["canonical_edge_key"]},
        historical_keys={positive["canonical_edge_key"]},
    )
    assert contradictory_report["exact_edge_label_contradiction_count"] == 1


def test_endpoint_leakage_is_reported_but_does_not_fit_model() -> None:
    rows = [
        _label("accept_continuity", "shared_source", "target_a", "group_a", 0.8, 0.1),
        _label("reject_continuity", "shared_source", "target_b", "group_b", 0.0, 3.0),
    ]

    audit = _endpoint_leakage_audit(rows)

    assert audit["shared_endpoint_across_groups_count"] == 2
    assert audit["shared_endpoint_across_safe_groups_count"] == 0
    assert "shared_source" in audit["shared_endpoint_groups"]
    assert audit["endpoint_crosses_train_validation_folds"] is False
    assert audit["model_application_allowed"] is False


def test_perfect_geometry_baseline_blocks_unjustified_model_fit() -> None:
    rows = []
    for index in range(6):
        rows.append(_label("accept_continuity", f"source_{index}", f"pos_{index}", f"group_{index}", 0.7, 0.1))
        rows.append(_label("reject_continuity", f"source_{index}", f"neg_{index}", f"group_{index}", 0.0, 3.0))
    baseline, _diagnostic, geometry = _baseline_results(paired_rows=rows, full_rows=rows)
    endpoint_leakage = _endpoint_leakage_audit(rows)
    training = _training_readiness(
        inventory={
            "canonical_unique_positive_count": 6,
            "canonical_unique_negative_count": 6,
            "independent_positive_trajectory_component_count": 6,
            "independent_negative_assignment_neighbourhood_count": 6,
            "conflict_count": 0,
        },
        exact_contradiction={"exact_edge_contradiction_count": 0},
        endpoint_audit={"endpoint_invalid_count": 0, "role_incompatible_count": 0},
        endpoint_leakage=endpoint_leakage,
        geometry_audit=geometry,
    )

    assert baseline["best_geometry_baseline"]["balanced_accuracy"] == 1.0
    assert geometry["paired_target_choice_subset"]["geometry_only_grouped_validation_is_perfect"] is True
    assert training["readiness_state"] == "READY_FOR_GROUPED_DIAGNOSTIC_EVALUATION"
    assert training["ready_for_model_application"] is False
    assert training["model_fit_performed"] is False
    assert training["learned_continuity_rows_updated"] == 0
    assert "candidate_construction_type" in training["excluded_feature_names"]
