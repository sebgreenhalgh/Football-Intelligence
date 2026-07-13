from __future__ import annotations

from pathlib import Path

from football_intelligence.replay.positive_only_counterfactual_continuity import (
    F2_DIAGNOSTIC_CLASSIFICATION,
    GENERIC_POSITIVE_LABEL,
    NEAR_ASSISTANT_CONTEXT,
    TRAINING_BLOCKED_SINGLE_CLASS,
    _accepted_examples_with_geometry,
    _build_counterfactual_candidates,
    _hard_negative_failure_audit,
    _training_readiness,
    _write_repaired_workbench,
    accepted_local_trajectory_components,
    role_context_reconciliation,
    validate_completed_f2_review,
)
from football_intelligence.review.schemas import stable_hash


def _bbox(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _manifest_case(index: int, bucket: str = "likely_positive") -> dict[str, object]:
    case_id = f"case_{index:03d}"
    candidate_hash = f"candidate_hash_{index:03d}"
    evidence_hash = f"evidence_hash_{index:03d}"
    frame = index * 3
    return {
        "review_case_id": case_id,
        "candidate_artifact_id": f"candidate_{index:03d}",
        "candidate_hash": candidate_hash,
        "evidence_hash": evidence_hash,
        "source_frame_sequence": frame,
        "target_frame_sequence": frame + 1,
        "equivalence_cluster_id": f"old_unique_cluster_{index:03d}",
        "evidence_manifest": {
            "evidence_hash": evidence_hash,
            "source_bbox": _bbox(100 + index, 200, 130 + index, 260),
            "target_bbox": _bbox(104 + index, 201, 134 + index, 261),
            "frame_gap": 1,
        },
        "selection_metadata": {
            "blind_context": {
                "source_visible_person_base_id": f"s_{index:03d}",
                "target_visible_person_base_id": f"t_{index:03d}",
                "source_candidate_id": f"src_candidate_{index:03d}",
                "target_candidate_id": f"target_candidate_{index:03d}",
                "team_partition": "team_1" if index % 2 == 0 else "team_2",
                "effective_role_context": "team_1_outfield_visual_context",
                "competing_candidates": {"count": 0},
            },
            "blind_hidden_model_info": {
                "proposed_bucket": bucket,
                "raw_features": {
                    "bbox_iou": 0.7,
                    "center_delta_px": 4.0,
                    "footpoint_delta_px": 4.5,
                    "continuity_score": 0.6,
                },
            },
        },
    }


def _completed_fixture() -> tuple[dict[str, object], dict[str, object], dict[str, object], list[dict[str, object]]]:
    cases = [_manifest_case(index, "likely_negative" if index >= 20 else "likely_positive") for index in range(40)]
    manifest = {
        "review_cases": cases,
        "candidate_manifest_hash": stable_hash([case["candidate_hash"] for case in cases]),
        "evidence_manifest_hash": stable_hash([case["evidence_hash"] for case in cases]),
    }
    decisions = {case["review_case_id"]: "accept_continuity" for case in cases}
    state = {"decisions": decisions, "completed": True}
    summary = {
        "total_cases": 40,
        "accepted": 40,
        "rejected": 0,
        "unresolved": 0,
        "decision_state_hash": stable_hash(state),
    }
    completed = {"state": state}
    events = [
        {
            "event_type": "decision",
            "review_case_id": case["review_case_id"],
            "decision": "accept_continuity",
        }
        for case in cases
    ]
    return manifest, completed, summary, events


def test_f2_all_accept_review_is_preserved_positive_only_and_training_blocked() -> None:
    manifest, completed, summary, events = _completed_fixture()
    validation = validate_completed_f2_review(
        manifest=manifest,
        completed_review=completed,
        completed_summary=summary,
        completed_events=events,
    )
    examples = _accepted_examples_with_geometry(manifest, completed["state"]["decisions"])
    readiness = _training_readiness(examples)
    failure = _hard_negative_failure_audit(examples)

    assert validation["valid"] is True
    assert validation["f2_review_classification"] == F2_DIAGNOSTIC_CLASSIFICATION
    assert all(row["semantic_training_label"] == GENERIC_POSITIVE_LABEL for row in examples)
    assert failure["proposed_negative_accepted_count"] == 20
    assert failure["proposed_negative_rejected_count"] == 0
    assert readiness["status"] == TRAINING_BLOCKED_SINGLE_CLASS
    assert readiness["continuity_model_fit_performed"] is False
    assert readiness["continuity_rows_updated"] == 0


def _accepted_example(case_id: str, frame: int, bbox: dict[str, float], team: str = "team_1") -> dict[str, object]:
    return {
        "review_case_id": case_id,
        "source_visible_person_base_id": f"{case_id}_source",
        "target_visible_person_base_id": f"{case_id}_target",
        "source_candidate_id": f"{case_id}_src_candidate",
        "target_candidate_id": f"{case_id}_target_candidate",
        "candidate_artifact_id": f"{case_id}_candidate",
        "source_frame_sequence": frame,
        "target_frame_sequence": frame + 1,
        "frame_gap": 1,
        "team_partition": team,
        "effective_role_context": f"{team}_outfield_visual_context",
        "source_bbox": bbox,
        "target_bbox": _bbox(bbox["x1"] + 4, bbox["y1"], bbox["x2"] + 4, bbox["y2"]),
        "near_camera_assistant_band": bbox["x1"] > 2100,
        "raw_features": {"continuity_score": 0.6, "center_delta_px": 4.0},
    }


def test_semantic_components_collapse_nearby_edges_not_unique_rows() -> None:
    examples = [
        _accepted_example("a", 100, _bbox(100, 200, 130, 260)),
        _accepted_example("b", 104, _bbox(112, 202, 142, 262)),
        _accepted_example("c", 400, _bbox(1900, 200, 1930, 260)),
    ]
    components = accepted_local_trajectory_components(examples)

    assert len(components["components"]) == 2
    assert max(component["case_count"] for component in components["components"]) == 2


def test_role_context_reconciliation_preserves_original_and_exact_role_takes_precedence() -> None:
    examples = [
        {
            **_accepted_example("exact", 10, _bbox(2200, 380, 2230, 450)),
            "accepted_local_visual_trajectory_component_id": "component_exact",
        },
        {
            **_accepted_example("near", 20, _bbox(2220, 390, 2250, 460)),
            "accepted_local_visual_trajectory_component_id": "component_near",
        },
    ]
    rows, audit = role_context_reconciliation(
        examples,
        exact_role_by_candidate={"exact_src_candidate": "team_2_outfield_visual_context"},
    )

    exact = next(row for row in rows if row["review_case_id"] == "exact")
    near = next(row for row in rows if row["review_case_id"] == "near")
    assert exact["original_visual_role_context"] == "team_1_outfield_visual_context"
    assert exact["reviewed_or_reconciled_role_context"] == "team_2_outfield_visual_context"
    assert exact["role_context_source"] == "exact_previous_human_role_label"
    assert near["reviewed_or_reconciled_role_context"] == NEAR_ASSISTANT_CONTEXT
    assert audit["assistant_referee_examples_not_counted_as_outfield_training_examples"] is True


def _node(visible: str, frame: int, bbox: dict[str, float]) -> dict[str, object]:
    return {
        "visible_person_base_id": visible,
        "candidate_id": f"candidate_{visible}",
        "frame_sequence": frame,
        "continuity_eligible": True,
        "entity_validity_state": "valid_on_pitch_person",
        "bbox": bbox,
    }


def test_counterfactual_alternatives_exclude_accepted_duplicate_and_same_component() -> None:
    anchor = {
        **_accepted_example("anchor", 10, _bbox(100, 100, 140, 180)),
        "source_visible_person_base_id": "source",
        "target_visible_person_base_id": "accepted",
        "accepted_local_visual_trajectory_component_id": "component_a",
    }
    same_component_positive = {
        **_accepted_example("same", 11, _bbox(200, 100, 240, 180)),
        "source_visible_person_base_id": "same_source",
        "target_visible_person_base_id": "same_component",
        "accepted_local_visual_trajectory_component_id": "component_a",
    }
    nodes = [
        _node("source", 10, _bbox(100, 100, 140, 180)),
        _node("accepted", 11, _bbox(110, 100, 150, 180)),
        _node("duplicate", 11, _bbox(111, 100, 151, 180)),
        _node("same_component", 11, _bbox(180, 100, 220, 180)),
        _node("good_alt", 11, _bbox(260, 105, 300, 185)),
    ]
    candidates, rejections = _build_counterfactual_candidates(
        positive_examples=[anchor, same_component_positive],
        node_rows=nodes,
        role_rows_by_visible={},
    )

    assert candidates[0]["alternative_target_visible_person_base_id"] == "good_alt"
    assert all(row["alternative_target_visible_person_base_id"] != "accepted" for row in candidates)
    assert any(row["reason"] == "duplicate_box_for_accepted_target_excluded" for row in rejections)
    assert any(row["reason"] == "same_accepted_local_trajectory_component_excluded" for row in rejections)


def test_repaired_workbench_hides_buckets_has_shortcuts_and_blocks_note_focus(tmp_path: Path) -> None:
    _write_repaired_workbench(tmp_path)
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    script = (tmp_path / "app.js").read_text(encoding="utf-8")

    assert "Reveal reference and model information" in index
    assert '<section id="modelPanel" class="hidden"></section>' in index
    assert 'a: "accept_continuity"' in script
    assert 'r: "reject_continuity"' in script
    assert 'n: "not_applicable_invalid_or_incompatible_endpoint"' in script
    assert 'u: "unresolved"' in script
    assert "typingTarget(ev.target)" in script
    assert "Decision saved:" in script
    assert "Completion blocked until all cases have decisions" in script
    assert "video.currentTime" in script
    assert "video.playbackRate" in script
