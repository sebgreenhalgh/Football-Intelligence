from __future__ import annotations

import json
from pathlib import Path

from football_intelligence.replay.followup_candidate_supply_diagnostic import (
    DETECTOR_DIAGNOSTIC_CONFIGS,
    FOLLOWUP_DECISION,
    build_m5_4j_review_pack,
    decoded_followup_rows,
    detector_diagnostic_placeholders,
    inventory_candidate_coverage,
    root_cause_and_research_gate,
    validate_followup_events,
)
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.review_chassis.manifest import manifest_hash
from football_intelligence.review_chassis.models import (
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _case(case_id: str) -> GenericReviewCase:
    return GenericReviewCase(
        case_id=case_id,
        task_type="visual_continuity_edge_review",
        candidate_id=f"candidate_{case_id}",
        candidate_hash=stable_hash(["candidate", case_id]),
        evidence_hash=stable_hash(["evidence", case_id]),
        allowed_decisions=[FOLLOWUP_DECISION],
        concise_question="Was the correct target shown?",
        evidence_assets=[],
        source_frame_sequence=1,
        target_frame_sequence=2,
        safety_payload=safety_payload(),
    )


def _write_synthetic_followup_review(tmp_path: Path) -> tuple[Path, Path, Path, list[str]]:
    case_ids = [f"case_{index:03d}" for index in range(1, 5)]
    manifest = GenericReviewManifest(
        review_id="m5_4i1_neither_case_candidate_coverage_review",
        stage_id="m5_4i1",
        task_type="visual_continuity_edge_review",
        title="Synthetic follow-up",
        cases=[_case(case_id) for case_id in case_ids],
        evidence_manifest_hash=stable_hash(["synthetic_evidence"]),
        source_manifest_hash=stable_hash(["synthetic_source"]),
    )
    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["manifest_hash"] = manifest_hash(manifest)
    manifest_path = tmp_path / "reviewer_manifest.json"
    _write_json(manifest_path, manifest_payload)
    ui_config = ReviewUIConfig(
        page_title="Synthetic",
        review_title="Synthetic",
        task_instructions="Synthetic task.",
        decisions=[{"key": "N", "value": FOLLOWUP_DECISION, "label": "Correct target not detected"}],
    )
    ui_config_path = tmp_path / "ui_config.json"
    _write_json(ui_config_path, ui_config.model_dump(mode="json"))
    decisions_root = tmp_path / "decisions"
    persistence = GenericReviewPersistence(
        manifest=manifest,
        ui_config=ui_config,
        decisions_root=decisions_root,
        reviewer_session_id="local-c6341d77",
    )
    for case_id in case_ids:
        persistence.save_decision(case_id=case_id, decision=FOLLOWUP_DECISION, input_source="keyboard")
    persistence.complete(elapsed_active_seconds=33)
    state_path = decisions_root / "review_decisions.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["created_at"] = "not_started"
    state["reviewer_session_id"] = "local-reviewer"
    _write_json(state_path, state)
    return manifest_path, ui_config_path, decisions_root, case_ids


def test_followup_event_log_replays_five_events_and_documents_sentinel(tmp_path: Path) -> None:
    manifest_path, ui_config_path, decisions_root, case_ids = _write_synthetic_followup_review(tmp_path)

    validation, sequence_audit, session_audit = validate_followup_events(
        manifest_path=manifest_path,
        ui_config_path=ui_config_path,
        decisions_root=decisions_root,
        expected_case_ids=case_ids,
    )

    assert validation["passed"] is True
    assert validation["event_count"] == 5
    assert validation["initial_decision_events"] == 4
    assert validation["same_value_reconfirmations"] == 0
    assert validation["changed_value_overwrites"] == 0
    assert validation["completion_events"] == 1
    assert validation["no_reveal_event"] is True
    assert validation["no_answer_key_payload_delivered"] is True
    assert validation["created_at_not_started_is_expected_deterministic_sentinel"] is True
    assert validation["final_decision_counts"] == {FOLLOWUP_DECISION: 4}
    assert sequence_audit["monotonic_sequence"] is True
    assert sequence_audit["snapshot_count"] == 5
    assert session_audit["session_result"] == "NORMALIZED_ALIAS_OR_DEFAULT_SESSION_LABEL_MISMATCH"


def test_followup_decoding_creates_no_binary_labels_and_counts_trajectory_regions() -> None:
    final_decisions = {f"followup_{index:03d}": FOLLOWUP_DECISION for index in range(1, 5)}
    mapping_by_case = {
        "followup_001": {"source_case_id": "004"},
        "followup_002": {"source_case_id": "009"},
        "followup_003": {"source_case_id": "011"},
        "followup_004": {"source_case_id": "016"},
    }
    case_index_rows = [
        {
            "case_id": f"followup_{index:03d}",
            "source_frame_sequence": str(100 + index),
            "target_frame_sequence": str(103 + index),
            "candidate_count": "8",
            "intermediate_candidate_count": "2",
        }
        for index in range(1, 5)
    ]
    groups = {"004": "trajectory_group_a", "016": "trajectory_group_a", "009": "trajectory_group_b", "011": "c"}

    rows, summary = decoded_followup_rows(
        final_decisions=final_decisions,
        mapping_by_case=mapping_by_case,
        case_index_rows=case_index_rows,
        trajectory_group_by_source_case=groups,
    )

    assert len(rows) == 4
    assert summary["case_level_candidate_supply_failure_count"] == 4
    assert summary["trajectory_safe_candidate_supply_failure_region_count"] == 3
    assert rows[0]["binary_label_created"] is False
    assert summary["binary_labels_created_from_followup"] == 0
    assert summary["detector_miss_claimed_before_spatial_localization"] is False


def test_full_frame_candidate_audit_is_not_limited_to_radius_and_does_not_claim_detector_miss() -> None:
    decoded_rows = [
        {"case_id": "followup_001", "source_case_id": "004", "target_frame_sequence": 345},
    ]
    candidate_rows_by_frame = {
        345: [{"candidate_id": f"candidate_{index}", "frame_sequence": 345} for index in range(1, 11)]
    }
    followup_mapping_by_case = {
        "followup_001": {
            "anonymous_displayed_candidates": [{"candidate_id": f"candidate_{index}"} for index in range(1, 9)]
        }
    }

    coverage, radius = inventory_candidate_coverage(
        decoded_rows=decoded_rows,
        candidate_rows_by_frame=candidate_rows_by_frame,
        followup_mapping_by_case=followup_mapping_by_case,
    )

    assert coverage["all_target_frame_candidates_audited_not_only_140px"] is True
    assert coverage["detector_miss_claimed_before_spatial_localization"] is False
    assert coverage["rows"][0]["full_frame_candidate_count"] == 10
    assert coverage["rows"][0]["displayed_inside_140px_count"] == 8
    assert coverage["rows"][0]["not_displayed_candidate_count"] == 2
    assert radius["radius_failure_not_confirmed_without_spatial_localization"] is True


def test_detector_diagnostics_are_gated_and_do_not_change_project_defaults(tmp_path: Path) -> None:
    decoded_rows = [{"case_id": "followup_001"}, {"case_id": "followup_002"}]

    config_manifest, recovery_summary, control = detector_diagnostic_placeholders(tmp_path, decoded_rows)

    assert config_manifest["detector_diagnostics_run"] is False
    assert config_manifest["blocked_until_spatial_localization_is_sealed"] is True
    assert config_manifest["project_defaults_changed"] is False
    assert config_manifest["planned_configurations"] == DETECTOR_DIAGNOSTIC_CONFIGS
    assert recovery_summary["detector_configurations_run"] == 0
    assert recovery_summary["detector_recovery_by_configuration"] == []
    assert control["control_frames_evaluated"] == 0
    assert (tmp_path / "detector_diagnostic" / "recovery_rows.jsonl").exists()


def test_root_cause_is_unresolved_before_localization_and_model_updates_remain_zero() -> None:
    decoded_rows = [
        {"case_id": "followup_001", "source_case_id": "004", "trajectory_safe_group_id": "group_a"},
        {"case_id": "followup_004", "source_case_id": "016", "trajectory_safe_group_id": "group_a"},
        {"case_id": "followup_002", "source_case_id": "009", "trajectory_safe_group_id": "group_b"},
        {"case_id": "followup_003", "source_case_id": "011", "trajectory_safe_group_id": "group_c"},
    ]

    root_cause, research_gate = root_cause_and_research_gate(decoded_rows)

    assert root_cause["final_root_cause_counts"] == {"UNRESOLVED_ROOT_CAUSE": 4}
    assert root_cause["trajectory_safe_candidate_supply_failure_region_count"] == 3
    assert root_cause["detector_miss_claimed_before_spatial_localization"] is False
    assert research_gate["gate_status"] == "PENDING_SPATIAL_LOCALIZATION"
    assert research_gate["model_fit_performed"] is False
    assert research_gate["learned_continuity_rows_updated"] == 0


def test_spatial_annotation_fields_are_generic_chassis_configuration() -> None:
    config = ReviewUIConfig(
        page_title="Localization",
        review_title="Localization",
        task_instructions="Localize target.",
        decisions=[{"key": "U", "value": "UNRESOLVED", "label": "Unresolved"}],
        spatial_annotation_enabled=True,
        spatial_annotation_mode="point_plus_numeric_bbox",
        spatial_annotation_schema={"bbox_drawing_supported": False},
    )
    script = Path("src/football_intelligence/review_chassis/static/app.js").read_text(encoding="utf-8")

    assert config.spatial_annotation_enabled is True
    assert config.spatial_annotation_mode == "point_plus_numeric_bbox"
    assert "renderSpatialAnnotation" in script
    assert "data-annotation-field" in script
    assert "m5_4j" not in script.lower()


def test_review_pack_is_capped_at_twenty_files_and_excludes_sealed_mapping(tmp_path: Path) -> None:
    stage_root = tmp_path / "stage"
    v14 = stage_root / "continuity_v14"
    sources = [
        stage_root / "validation" / "m5_4j_validation_summary.json",
        v14 / "ingestion" / "followup_event_validation.json",
        v14 / "audit" / "followup_event_sequence_audit.json",
        v14 / "audit" / "followup_session_audit.json",
        v14 / "ingestion" / "followup_sealed_mapping_validation.json",
        v14 / "ingestion" / "decoded_followup_summary.json",
        v14 / "ingestion" / "decoded_followup_rows.jsonl",
        v14 / "registration" / "reconciled_continuity_inventory_registration.json",
        v14 / "localization" / "reviewer_manifest.json",
        v14 / "localization" / "ui_config.json",
        v14 / "localization" / "case_index.csv",
        v14 / "localization" / "sealed_reference.json",
        v14 / "audit" / "full_frame_candidate_coverage_audit.json",
        v14 / "audit" / "local_radius_failure_audit.json",
        v14 / "audit" / "affected_frame_detector_provenance.json",
        v14 / "audit" / "postprocess_loss_audit.json",
        v14 / "detector_diagnostic" / "recovery_summary.json",
        v14 / "audit" / "candidate_supply_root_cause.json",
        v14 / "research" / "continuity_research_gate.json",
    ]
    for index, source in enumerate(sources):
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f'{{"index": {index}}}\n', encoding="utf-8")

    result = build_m5_4j_review_pack(stage_root=stage_root)
    pack_files = sorted((v14 / "review_pack").iterdir())
    explanation = (v14 / "review_pack" / "00_REVIEW_PACK_EXPLANATION.txt").read_text(encoding="utf-8")

    assert result["review_pack_file_count"] == 20
    assert result["within_max_file_count"] is True
    assert len(pack_files) == 20
    assert not any("mapping.json" == path.name for path in pack_files)
    assert "sealed mapping excluded" in explanation.lower()
