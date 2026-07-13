from __future__ import annotations

import copy
import json
from pathlib import Path

from football_intelligence.learning.mixed_review_ingestion import ingest_mixed_review
from football_intelligence.learning.review_label_validation import class_sufficiency_readiness
from football_intelligence.review.persistence import ReviewPersistence
from football_intelligence.review.schemas import (
    CONTINUITY_DECISIONS,
    CONTINUITY_QUESTION,
    ENTITY_VALIDITY_DECISIONS,
    ENTITY_VALIDITY_QUESTION,
    VISUAL_TEAM_ROLE_DECISIONS,
    VISUAL_TEAM_ROLE_QUESTION,
    EvidenceManifest,
    ReviewCase,
    ReviewManifest,
    safety_payload,
    stable_hash,
)
from football_intelligence.step1_visual_reconstruction.visual_role_context import (
    ASSISTANT_FAR,
    ASSISTANT_NEAR,
    CENTRAL_REFEREE,
    NON_PERSON,
    TEAM_1_GOALKEEPER,
    TEAM_1_OUTFIELD,
    TEAM_2_OUTFIELD,
    UNKNOWN_PERSON,
    classify_visual_role_context,
)
from football_intelligence.step2_visual_continuity.positive_selector import select_continuity_review_candidates
from football_intelligence.step2_visual_continuity.role_partitioning import apply_role_partitioning


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _evidence(source: int = 1, target: int | None = None) -> EvidenceManifest:
    return EvidenceManifest(
        evidence_id=f"evidence_{source}_{target}",
        evidence_assets=[],
        source_frame_hashes=[],
        source_frame_sequence=source,
        target_frame_sequence=target,
        source_bbox={"x1": 1.0, "y1": 1.0, "x2": 11.0, "y2": 41.0},
        target_bbox={"x1": 2.0, "y1": 2.0, "x2": 12.0, "y2": 42.0} if target is not None else None,
        frame_gap=None if target is None else target - source,
        temporal_evidence_available=True,
        evidence_hash="hash",
    )


def _case(case_id: str, task_type: str, decision_family: str) -> ReviewCase:
    if decision_family == "entity":
        question = ENTITY_VALIDITY_QUESTION
        decisions = ENTITY_VALIDITY_DECISIONS
        target = None
    elif decision_family == "continuity":
        question = CONTINUITY_QUESTION
        decisions = CONTINUITY_DECISIONS
        target = 2
    else:
        question = VISUAL_TEAM_ROLE_QUESTION
        decisions = VISUAL_TEAM_ROLE_DECISIONS
        target = None
    evidence = _evidence(1, target)
    return ReviewCase(
        review_case_id=case_id,
        task_type=task_type,  # type: ignore[arg-type]
        concise_question=question,
        allowed_decisions=decisions,
        candidate_artifact_id=f"candidate_{case_id}",
        source_artifact_references=[],
        source_frame_sequence=1,
        target_frame_sequence=target,
        evidence_manifest=evidence,
        uncertainty_reasons=[],
        category=task_type,
        priority=1,
        control_status="not_control",
        candidate_hash=stable_hash({"case": case_id}),
        evidence_hash="hash",
        safety_payload=safety_payload(),
        review_round=1,
        equivalence_cluster_id=f"cluster_{case_id}",
    )


def test_mixed_review_ingestion_counts_entity_and_continuity_labels_from_decisions(tmp_path: Path) -> None:
    manifest = {
        "review_cases": [
            _case("entity_1", "entity_validity", "entity").model_dump(mode="json"),
            _case("entity_2", "entity_validity", "entity").model_dump(mode="json"),
            _case("continuity_1", "visual_continuity_edge_review", "continuity").model_dump(mode="json"),
        ]
    }
    completed = {
        "state": {
            "decisions": {
                "entity_1": "valid_on_pitch_person",
                "entity_2": "valid_official",
                "continuity_1": "reject_continuity",
            },
            "notes": {},
            "completed_at": "2026-07-13T00:00:00+00:00",
            "reviewer_session_id": "session",
        }
    }
    manifest_path = _write_json(tmp_path / "review_manifest_round_1.json", manifest)
    completed_path = _write_json(tmp_path / "completed_review.json", completed)
    event_path = tmp_path / "completed_review_events.jsonl"
    event_path.write_text("", encoding="utf-8")

    result = ingest_mixed_review(
        completed_review_path=completed_path,
        review_manifest_path=manifest_path,
        event_log_path=event_path,
        requested_manifest_path=tmp_path / "round_1_review_manifest.json",
    )

    assert result["binding_validation"]["passed"] is True
    assert result["binding_validation"]["summary_counters_used_for_label_inventory"] is False
    assert result["distribution"]["entity_label_distribution"]["valid_on_pitch_person"] == 1
    assert result["distribution"]["entity_label_distribution"]["valid_official"] == 1
    assert result["distribution"]["continuity_label_distribution"]["reject_continuity"] == 1


def test_mixed_task_summary_exports_do_not_hide_entity_labels(tmp_path: Path) -> None:
    manifest = ReviewManifest(
        title="Mixed review",
        review_task_family="mixed",
        review_cases=[
            _case("entity", "entity_validity", "entity"),
            _case("continuity", "visual_continuity_edge_review", "continuity"),
        ],
        candidate_manifest_hash="candidate_hash",
        evidence_manifest_hash="evidence_hash",
        source_manifest_hash="source_hash",
        source_artifact_references=[],
    )
    persistence = ReviewPersistence(manifest=manifest, decision_root=tmp_path / "decisions", reviewer_session_id="t")
    persistence.save_decision(review_case_id="entity", decision="non_person_false_positive")
    persistence.save_decision(review_case_id="continuity", decision="reject_continuity")
    persistence.complete()
    summary = json.loads((tmp_path / "decisions" / "completed_review_summary.json").read_text(encoding="utf-8"))

    assert summary["rejected"] == 1
    assert summary["entity_label_distribution"]["non_person_false_positive"] == 1
    assert summary["continuity_label_distribution"]["reject_continuity"] == 1
    assert summary["decision_counts_by_task_type"]["entity_validity"] == 1
    assert summary["summary_counters_used_for_label_inventory"] is False


def test_all_negative_continuity_review_blocks_training_and_no_reject_everything_model() -> None:
    examples = [
        {
            "review_case_id": f"c{i}",
            "task_type": "visual_continuity_edge_review",
            "normalized_training_label": "reject_continuity",
            "label_usable_for_training": True,
            "equivalence_cluster_id": f"cluster_{i}",
        }
        for i in range(6)
    ]
    readiness = class_sufficiency_readiness(
        examples,
        task_type="visual_continuity_edge_review",
        required_labels={"accept_continuity", "reject_continuity"},
    )

    assert readiness["status"] == "BLOCKED_SINGLE_CLASS_REVIEW_LABELS"
    assert readiness["required_labels_present"] is False
    assert readiness["examples_per_class"] == {"reject_continuity": 6}


def test_visual_role_taxonomy_keeps_officials_camera_relative_and_uncertain_unforced() -> None:
    base = {
        "entity_validity_state": "valid_on_pitch_person",
        "reviewed_entity_label": None,
        "team_1_belief": 0.5,
        "team_2_belief": 0.48,
        "goalkeeper_belief": 0.2,
        "central_referee_belief": 0.1,
        "near_camera_assistant_belief": 0.1,
        "far_camera_assistant_belief": 0.1,
    }
    assert classify_visual_role_context({**base, "team_1_belief": 0.74})[0] == TEAM_1_OUTFIELD
    central_feature = {**base, "reviewed_entity_label": "valid_official", "central_referee_belief": 0.7}
    assert classify_visual_role_context(central_feature)[0] == CENTRAL_REFEREE
    assert (
        classify_visual_role_context(
            {**base, "reviewed_entity_label": "valid_official", "near_camera_assistant_belief": 0.72}
        )[0]
        == ASSISTANT_NEAR
    )
    assert (
        classify_visual_role_context(
            {**base, "reviewed_entity_label": "valid_official", "far_camera_assistant_belief": 0.72}
        )[0]
        == ASSISTANT_FAR
    )
    assert classify_visual_role_context({**base, "team_1_belief": 0.54, "team_2_belief": 0.5})[0] != TEAM_1_OUTFIELD
    unknown_feature = {"entity_validity_state": "ambiguous_entity_requires_review"}
    assert classify_visual_role_context(unknown_feature)[0] == UNKNOWN_PERSON


def _edge(source: str, target: str, score: float = 0.9) -> dict[str, object]:
    return {
        "continuity_candidate_id": f"edge_{source}_{target}",
        "source_visible_person_base_id": source,
        "target_visible_person_base_id": target,
        "source_frame_sequence": 1,
        "target_frame_sequence": 2,
        "frame_gap": 1,
        "continuity_score": score,
        "gate_features": {"center_delta_px": 10.0, "bbox_area_ratio": 1.0, "bbox_iou": 0.4},
        "visual_continuity_is_real_identity": False,
        "visual_continuity_is_player_slot": False,
    }


def test_role_partitioning_reduces_pool_bounds_degree_and_rejects_incompatible_roles() -> None:
    role_by_visible_id = {
        "t1a": {"visual_role_context_state": TEAM_1_OUTFIELD},
        "t1b": {"visual_role_context_state": TEAM_1_OUTFIELD},
        "t2": {"visual_role_context_state": TEAM_2_OUTFIELD},
        "gk": {"visual_role_context_state": TEAM_1_GOALKEEPER},
        "ref": {"visual_role_context_state": CENTRAL_REFEREE},
        "near": {"visual_role_context_state": ASSISTANT_NEAR},
        "non": {"visual_role_context_state": NON_PERSON},
    }
    source_rows = [
        _edge("t1a", "t1b", 0.99),
        _edge("t1a", "t2", 0.98),
        _edge("gk", "t1a", 0.97),
        _edge("ref", "near", 0.96),
        _edge("non", "t1a", 0.95),
        *[_edge("t1a", f"target_{i}", 0.9 - i * 0.01) for i in range(5)],
    ]
    for i in range(5):
        role_by_visible_id[f"target_{i}"] = {"visual_role_context_state": TEAM_1_OUTFIELD}
    before = copy.deepcopy(source_rows)
    result = apply_role_partitioning(candidate_rows=source_rows, role_by_visible_id=role_by_visible_id, max_degree=3)

    assert source_rows == before
    assert result["candidate_pool_after_role_partitioning"] < result["candidate_pool_before_role_partitioning"]
    assert result["max_source_candidate_degree"] <= 3
    assert any(row["rejection_reason"] == "role_partition_incompatible" for row in result["rejected_rows"])
    assert any(row["rejection_reason"] == "non_person_continuity_not_applicable" for row in result["rejected_rows"])
    assert all(row["visual_continuity_is_real_identity"] is False for row in result["rows"])
    assert all(row["visual_continuity_is_player_slot"] is False for row in result["rows"])
    assert all("role_partitioned_continuity_candidate_id" in row for row in result["rows"])


def test_positive_selector_requires_intermediate_support_before_positive_review() -> None:
    no_support = {
        **_edge("a", "b", 0.95),
        "source_visual_role_context": TEAM_1_OUTFIELD,
        "target_visual_role_context": TEAM_1_OUTFIELD,
        "gate_features": {"center_delta_px": 10.0, "bbox_area_ratio": 1.0, "bbox_iou": 0.0},
        "intermediate_frame_support": False,
    }
    with_support = {
        **_edge("c", "d", 0.94),
        "source_visual_role_context": TEAM_1_OUTFIELD,
        "target_visual_role_context": TEAM_1_OUTFIELD,
        "gate_features": {"center_delta_px": 10.0, "bbox_area_ratio": 1.0, "bbox_iou": 0.0},
        "intermediate_frame_support": True,
    }
    selection = select_continuity_review_candidates([no_support, with_support], positive_limit=10, negative_limit=10)

    assert selection["likely_positive_count"] == 1
    assert selection["likely_positive"][0]["source_visible_person_base_id"] == "c"
    assert selection["likely_positive"][0]["requires_intermediate_support"] is True
    assert selection["likely_negative_count"] == 1


def test_visual_team_role_review_schema_has_no_prefilled_decision() -> None:
    case = _case("role", "visual_team_role_context", "role")

    assert case.concise_question == VISUAL_TEAM_ROLE_QUESTION
    assert case.allowed_decisions == VISUAL_TEAM_ROLE_DECISIONS
    assert case.model_prediction is None
