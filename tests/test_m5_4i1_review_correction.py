from __future__ import annotations

from football_intelligence.replay.third_unseen_geometry_challenge import _bbox_hash
from football_intelligence.replay.third_unseen_review_correction import (
    N_DECISION,
    _endpoint_status_from_candidate,
    _safety_guardrail_audit,
    canonical_trajectory_safe_grouping,
    corrected_challenge_control_split,
    corrected_failure_taxonomy,
    corrected_rule_results,
    label_binding_status,
    review_event_semantics,
    trajectory_merge_reason,
)


def _row(index: int, *, decision: str, rule_panel: str | None, random_control: bool = False) -> dict:
    selected = {"target_a_continues_source": "target_a", "target_b_continues_source": "target_b"}.get(decision)
    return {
        "case_id": f"case_{index:03d}",
        "human_decision": decision,
        "frozen_primary_preferred_panel": rule_panel,
        "primary_rule_multiple_accepts": False,
        "primary_rule_rejected_both": rule_panel is None,
        "frozen_secondary_preferred_panel": rule_panel,
        "secondary_rule_multiple_accepts": False,
        "secondary_rule_rejected_both": rule_panel is None,
        "random_control_status": random_control,
        "frame_gap": 1,
        "endpoint_safe_group_id": f"endpoint_{index:03d}",
        "trajectory_safe_group_id": f"trajectory_{index:03d}",
        "selected_displayed_panel": selected,
        "selected_canonical_target": {"candidate_id": f"target_{index}", "visible_person_base_id": f"vpb_t_{index}"},
        "unselected_canonical_target": {
            "candidate_id": f"other_{index}",
            "visible_person_base_id": f"vpb_o_{index}",
        },
        "source_candidate_id": f"source_{index}",
        "source_visible_person_base_id": f"vpb_s_{index}",
        "source_frame_sequence": index,
        "target_frame_sequence": index + 1,
        "source_bbox": {"x1": float(index), "y1": 0.0, "x2": float(index + 10), "y2": 20.0},
    }


def test_n_abstention_is_not_binary_agreement_and_primary_challenge_is_zero_of_twelve() -> None:
    rows = []
    for index in range(1, 13):
        rows.append(_row(index, decision="target_a_continues_source", rule_panel=None))
    rows.extend(
        [
            _row(13, decision=N_DECISION, rule_panel=None),
            _row(14, decision=N_DECISION, rule_panel=None),
            _row(15, decision=N_DECISION, rule_panel="target_a"),
            _row(16, decision=N_DECISION, rule_panel="target_b"),
        ]
    )
    rows.extend(
        [
            _row(17, decision="target_a_continues_source", rule_panel="target_a", random_control=True),
            _row(18, decision="target_b_continues_source", rule_panel="target_b", random_control=True),
            _row(19, decision="target_a_continues_source", rule_panel="target_a", random_control=True),
            _row(20, decision="target_b_continues_source", rule_panel=None, random_control=True),
        ]
    )

    split = corrected_challenge_control_split(rows, ("primary",))

    challenge = split["primary"]["challenge"]
    assert challenge["decisive_case_count"] == 12
    assert challenge["correct_decisive_target_choices"] == 0
    assert challenge["decisive_abstentions"] == 12
    assert challenge["human_n_cases"] == 4
    assert challenge["rule_abstained_on_n"] == 2
    assert challenge["rule_selected_panel_on_n"] == 2
    assert challenge["n_rows_excluded_from_binary_agreement"] is True

    control = split["primary"]["random_control"]
    assert control["decisive_case_count"] == 4
    assert control["correct_decisive_target_choices"] == 3
    assert control["wrong_decisive_target_choices"] == 0
    assert control["decisive_abstentions"] == 1


def test_same_value_reconfirmations_are_not_changed_overwrites() -> None:
    events = [
        {"event_sequence": 1, "event_type": "decision", "prior_decision": None, "new_decision": "A"},
        {"event_sequence": 9, "event_type": "decision", "prior_decision": "A", "new_decision": "A"},
        {"event_sequence": 22, "event_type": "decision", "prior_decision": "B", "new_decision": "B"},
        {"event_sequence": 23, "event_type": "complete", "prior_decision": None, "new_decision": None},
    ]

    summary = review_event_semantics(events)

    assert summary["initial_decisions"] == 1
    assert summary["same_value_reconfirmations"] == 2
    assert summary["changed_value_overwrites"] == 0
    assert summary["completion_events"] == 1


def test_endpoint_existence_requires_canonical_lookup_and_source_base_binding() -> None:
    bbox = {"x1": 1.0, "y1": 2.0, "x2": 11.0, "y2": 22.0}
    candidate = {
        "candidate_id": "m5_4h1_pc_f000001_001",
        "visible_person_base_id": "m5_4h1_vpb_f000001_abc",
        "frame_sequence": 1,
        "bbox": bbox,
        "bbox_hash": _bbox_hash(bbox),
        "entity_validity": "unknown_not_false",
        "role_status": "UNKNOWN_NOT_CONTRADICTED",
        "team_status": "UNKNOWN_NOT_CONTRADICTED",
        "visual_role_context": "unknown_visible_person_visual_context",
    }
    frame_by_sequence = {1: {"frame_sequence": 1}}

    passed = _endpoint_status_from_candidate(
        case_id="case_001",
        endpoint_kind="source",
        candidate_id=candidate["candidate_id"],
        visible_person_base_id=candidate["visible_person_base_id"],
        bbox=bbox,
        declared_frame_sequence=1,
        candidate_by_id={candidate["candidate_id"]: candidate},
        base_ids={candidate["visible_person_base_id"]},
        frame_by_sequence=frame_by_sequence,
    )
    missing = _endpoint_status_from_candidate(
        case_id="case_001",
        endpoint_kind="source",
        candidate_id="m5_4h1_pc_f000001_missing",
        visible_person_base_id=candidate["visible_person_base_id"],
        bbox=bbox,
        declared_frame_sequence=1,
        candidate_by_id={candidate["candidate_id"]: candidate},
        base_ids={candidate["visible_person_base_id"]},
        frame_by_sequence=frame_by_sequence,
    )
    mismatch = _endpoint_status_from_candidate(
        case_id="case_001",
        endpoint_kind="source",
        candidate_id=candidate["candidate_id"],
        visible_person_base_id="m5_4h1_vpb_f000001_other",
        bbox=bbox,
        declared_frame_sequence=1,
        candidate_by_id={candidate["candidate_id"]: candidate},
        base_ids={"m5_4h1_vpb_f000001_other"},
        frame_by_sequence=frame_by_sequence,
    )

    assert passed["endpoint_binding_passed"] is True
    assert missing["endpoint_binding_passed"] is False
    assert missing["endpoint_binding_status"] == "CANONICAL_EVIDENCE_UNAVAILABLE"
    assert mismatch["candidate_to_base_relationship_matches"] is False
    assert mismatch["endpoint_binding_passed"] is False


def test_contradiction_fields_are_status_values_not_constant_false() -> None:
    bbox = {"x1": 1.0, "y1": 2.0, "x2": 11.0, "y2": 22.0}
    candidate = {
        "candidate_id": "m5_4h1_pc_f000001_001",
        "visible_person_base_id": "m5_4h1_vpb_f000001_abc",
        "frame_sequence": 1,
        "bbox": bbox,
        "bbox_hash": _bbox_hash(bbox),
        "entity_validity": "non_person_false_positive",
        "role_status": "UNKNOWN_NOT_CONTRADICTED",
        "team_status": "UNKNOWN_NOT_CONTRADICTED",
        "visual_role_context": "unknown_visible_person_visual_context",
    }

    row = _endpoint_status_from_candidate(
        case_id="case_001",
        endpoint_kind="target_a",
        candidate_id=candidate["candidate_id"],
        visible_person_base_id=candidate["visible_person_base_id"],
        bbox=bbox,
        declared_frame_sequence=1,
        candidate_by_id={candidate["candidate_id"]: candidate},
        base_ids={candidate["visible_person_base_id"]},
        frame_by_sequence={1: {"frame_sequence": 1}},
    )

    assert row["endpoint_is_known_false_positive_status"] == "CONFIRMED_INCOMPATIBLE"
    assert row["endpoint_is_duplicate_detector_row_status"] == "UNKNOWN_NOT_CONTRADICTED"
    assert row["known_off_pitch_on_pitch_contradiction_status"] == "UNKNOWN_NOT_CONTRADICTED"
    assert row["endpoint_is_duplicate_detector_row_status"] is not False


def test_challenge_control_status_cannot_prevent_trajectory_grouping() -> None:
    left = _row(1, decision="target_a_continues_source", rule_panel="target_a")
    right = _row(2, decision="target_b_continues_source", rule_panel="target_b", random_control=True)
    right["source_candidate_id"] = left["selected_canonical_target"]["candidate_id"]
    right["source_visible_person_base_id"] = left["selected_canonical_target"]["visible_person_base_id"]

    audit, group_by_case = canonical_trajectory_safe_grouping([left, right])

    assert group_by_case[left["case_id"]] == group_by_case[right["case_id"]]
    assert audit["cross_subset_trajectory_group_count"] == 1
    assert audit["challenge_control_status_prevented_grouping"] is False


def test_bbox_overlap_alone_cannot_prove_trajectory_merge() -> None:
    left = _row(1, decision="target_a_continues_source", rule_panel="target_a")
    right = _row(2, decision="target_a_continues_source", rule_panel="target_a")
    right["source_bbox"] = dict(left["source_bbox"])
    right["source_frame_sequence"] = 100
    right["target_frame_sequence"] = 101

    assert trajectory_merge_reason(left, right) is None


def test_failure_taxonomy_case_lists_are_unique() -> None:
    rows = [
        _row(1, decision="target_a_continues_source", rule_panel=None),
        _row(2, decision="target_a_continues_source", rule_panel="target_b"),
        _row(3, decision=N_DECISION, rule_panel=None),
    ]
    groups = {row["case_id"]: f"group_{index}" for index, row in enumerate(rows)}
    primary = corrected_rule_results(rows, "primary", groups)
    secondary = corrected_rule_results(
        rows,
        "secondary",
        groups,
    )

    taxonomy = corrected_failure_taxonomy(
        primary,
        secondary,
        groups,
    )

    for section in ["primary_rule", "secondary_rule", "candidate_set_failures", "human_non_binary_outcomes"]:
        for bucket in taxonomy[section].values():
            assert bucket["case_ids"] == sorted(set(bucket["case_ids"]))


def test_n_creates_no_binary_labels_and_training_freeze_is_explicit() -> None:
    status_rows, promotable_pos, promotable_neg, combined = label_binding_status(
        [
            {
                "case_id": "case_001",
                "canonical_edge_key": "edge_1",
                "label": "accept_continuity",
                "source_candidate_id": "source",
                "target_candidate_id": "target",
            }
        ],
        {"case_001": {"all_displayed_endpoints_bind": True, "endpoint_failures": []}},
        {
            "historical_row_count": 46,
            "canonical_unique_edge_counts": {"accept_continuity": 56, "reject_continuity": 22},
            "new_positive_count": 16,
            "new_negative_count": 16,
        },
    )
    safety = _safety_guardrail_audit()

    assert all(row["binary_label_created_from_n_or_u"] is False for row in status_rows)
    assert len(promotable_pos) == 1
    assert len(promotable_neg) == 0
    assert combined["model_fit_performed"] is False
    assert combined["learned_continuity_rows_updated"] == 0
    assert safety["model_fit_performed"] is False
    assert safety["learned_continuity_rows_updated"] == 0
    assert safety["production_ready"] is False
