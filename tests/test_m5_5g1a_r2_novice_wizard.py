from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

import pytest

from football_intelligence.detection_gold.models import validate_case_annotation
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.validation import validate_review_chassis_package

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1]
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G1A_R2_Novice_Guided_Wizard_Codex_Prompt_Pack"
R1 = PART3 / "M5_5G1A_R1_ANNOTATION_UI_CORRECTNESS_AND_PILOT_LAUNCH_REPAIR_v1"
R1_PACKAGE = R1 / "05_CORRECTED_DETECTION_GOLD_PILOT_ANNOTATION_PACKAGE"
STAGE = PART3 / "M5_5G1A_R2_NOVICE_GUIDED_ANNOTATION_WIZARD_AND_USABILITY_OVERHAUL_v1"
PACKAGE = STAGE / "05_NOVICE_GUIDED_DETECTION_GOLD_PILOT_PACKAGE"
REVIEW_ID = "m5_5g1a_detection_gold_pilot_v1_r2"
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer_r2"
CASE_HASH = "986604e34e6f831825dfb76601f854ece083e3ba9001f97bad26d6d596e6a401"
EVIDENCE_HASH = "58c2de8da4e6e8d2160a29550e8030e7c5225845fc58be44eae7651c0b4a1ab4"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def valid_annotation(case: object, *, unresolved_strip: bool = False) -> dict:
    metadata = case.visible_metadata
    binding = copy.deepcopy(metadata["source_binding"])
    if case.task_type == "detection_gold_player_static":
        return {
            "schema_version": "m5_5g1a_detection_gold_v1",
            "source_binding": binding,
            "visible_person_count": 0,
            "player_instances": [],
            "candidate_relations": [
                {"candidate_uuid": value, "relation": "BACKGROUND", "annotation_uuids": []}
                for value in metadata["candidate_uuids"]
            ],
            "earliest_failure_stage": "UNRESOLVED",
            "note": "",
        }
    if case.task_type == "detection_gold_dense_region":
        return {
            "schema_version": "m5_5g1a_detection_gold_v1",
            "source_binding": binding,
            "dense_region_uuid": f"dense-{case.case_id}",
            "trigger_reason": metadata["pilot_stratum"],
            "human_visible_person_count": 0,
            "visible_masks": [],
            "candidate_relations": [
                {"candidate_uuid": value, "relation": "BACKGROUND", "annotation_uuids": []}
                for value in metadata["candidate_uuids"]
            ],
            "uncertain_or_ignore": True,
            "note": "",
        }
    if case.task_type == "detection_gold_temporal_player":
        return {
            "schema_version": "m5_5g1a_detection_gold_v1",
            "source_binding": binding,
            "frames": [
                {
                    "frame_sequence": row["frame_sequence"],
                    "source_frame_sha256": row["source_frame_sha256"],
                    "state": "UNRESOLVED" if unresolved_strip else "NOT_VISIBLE",
                    "current_frame_pixel_support": False,
                    "candidate_uuids": [],
                }
                for row in metadata["frame_records"]
            ],
            "contact_strip_reviewed": True,
            "stable_run_accepted": False,
            "note": "",
        }
    if case.task_type == "detection_gold_pitch_boundary":
        return {
            "schema_version": "m5_5g1a_detection_gold_v1",
            "source_binding": binding,
            "footpoint": copy.deepcopy(metadata["machine_footpoint"]),
            "footpoint_uncertainty_pixels": 5,
            "pitch_state": "BOUNDARY_UNCERTAIN",
            "coarse_role": "UNKNOWN",
            "primary_on_pitch_supply_eligible": False,
            "note": "",
        }
    return {
        "schema_version": "m5_5g1a_detection_gold_v1",
        "source_binding": binding,
        "frames": [
            {
                "frame_sequence": row["frame_sequence"],
                "source_frame_sha256": row["source_frame_sha256"],
                "state": "UNRESOLVED" if unresolved_strip else "NOT_VISIBLE",
            }
            for row in metadata["frame_records"]
        ],
        "full_contact_strip_reviewed": True,
        "note": "",
    }


def wizard_state(case_id: str) -> dict:
    return {
        "schema_version": "football_intelligence.m5_5g1a_r2.wizard_state.v1",
        "case_id": case_id,
        "step": 4,
        "current_object_uuid": None,
        "question_index": 0,
        "candidate_index": 0,
    }


def save_payload(persistence: DetectionGoldPilotPersistence, case: object, annotation: dict) -> dict:
    event_id = str(uuid.uuid4())
    return {
        "event_type": "DETECTION_CASE_SAVED",
        "review_id": persistence.manifest.review_id,
        "reviewer_session_id": persistence.reviewer_session_id,
        "case_id": case.case_id,
        "annotation": annotation,
        "wizard_state": wizard_state(case.case_id),
        "client_event_id": event_id,
        "idempotency_key": event_id,
        "expected_server_state_hash": persistence.state()["server_state_hash"],
        "elapsed_active_seconds": 2,
    }


def test_r2_package_preserves_all_cases_evidence_and_frozen_schemas() -> None:
    manifest = read_json(PACKAGE / "reviewer_manifest.json")
    validation = read_json(PACKAGE / "review_package_validation.json")
    frozen = read_json(STAGE / "01_PRIOR_STAGE_AND_STATE_AUDIT" / "frozen_hash_preservation.json")
    assert len(manifest["cases"]) == 88
    assert stable_hash(manifest["cases"]) == CASE_HASH
    assert validation["evidence_copy"]["file_count"] == 1512
    assert validation["evidence_copy"]["tree_hash"] == EVIDENCE_HASH
    assert validation["evidence_bytes_identical"] is True
    assert frozen["passed"] is True
    assert frozen["schema_migration_performed"] is False


def test_r2_plain_language_mapping_matches_the_controlling_contract() -> None:
    expected = read_json(PROMPT / "04_PLAIN_LANGUAGE_SCHEMA_MAPPING.json")
    actual = read_json(STAGE / "02_NOVICE_WIZARD_PRODUCT_DESIGN" / "plain_language_schema_mapping.json")
    assert actual == expected
    assert actual["candidate_relation"]["Not a person"] == "BACKGROUND"
    assert actual["temporal_state"]["I can't tell"] == "UNRESOLVED"
    assert actual["football_state"]["I can't tell"] == "UNRESOLVED"


def test_novice_app_owns_candidate_queue_and_hides_technical_controls() -> None:
    app = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_app.js").read_text()
    wizard = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_wizard.js").read_text()
    styles = (REPO / "src/football_intelligence/review_chassis/static/styles.css").read_text()
    assert "candidate.diagnostic_uuid !== novicePolicy.candidateUuid" in app
    assert "if (noviceCandidateRendered) continue" in app
    assert "runtime.wizard.syncCandidate()" in app
    assert "humanInteractive === false" in app
    assert "dgFootpointUncertainty" in app
    assert "What does this machine box represent?" in wizard
    assert "The highlighted box is selected for you" in wizard
    assert "Machine Box ${state.candidate_index + 1}" in wizard
    assert "data-nw-edit-candidate" in wizard
    assert "goToFrame(index)" in wizard
    assert "nwConfirmCopiedGeometry" in wizard
    assert "nwRejectCopiedGeometry" in wizard
    assert "It is not an observation until you confirm it here" in wizard
    assert "confirmGeometryDraft" in app and "rejectGeometryDraft" in app
    assert ".dgNoviceCandidate" in styles
    assert "pointer-events: none" in styles
    assert "technical_fields_hidden_by_default" in read_json(PACKAGE / "ui_config.json")["question_contract"]


def test_novice_flow_uses_one_question_and_optional_advanced_details() -> None:
    index = (REPO / "src/football_intelligence/review_chassis/static/index.html").read_text()
    app = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_app.js").read_text()
    wizard = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_wizard.js").read_text()
    assert '<script src="/detection_gold_wizard.js"></script>' in index
    assert '<details class="nwAdvancedDetails">' in app
    assert "question_index" in wizard
    assert "ONE SHORT QUESTION" in wizard
    assert "How this works" in wizard
    assert "nwTourStart" in index and "nwTourSkip" in index


def test_every_case_can_produce_a_frozen_schema_payload() -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    for case in manifest.cases:
        validated = validate_case_annotation(case.task_type, valid_annotation(case))
        assert validated["schema_version"] == "m5_5g1a_detection_gold_v1"


def test_reviewed_uncertainty_is_allowed_only_by_the_r2_contract(tmp_path: Path) -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    case = next(row for row in manifest.cases if row.task_type == "detection_gold_temporal_player")
    fixture = manifest.model_copy(update={"review_id": "r2-unresolved-fixture", "cases": [case]})
    r2 = DetectionGoldPilotPersistence(
        manifest=fixture,
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=tmp_path / "r2",
        reviewer_session_id=REVIEWER,
    )
    saved = r2.save_detection_event(save_payload(r2, case, valid_annotation(case, unresolved_strip=True)))
    assert saved["annotations"][case.case_id]["frames"][0]["state"] == "UNRESOLVED"

    r1 = DetectionGoldPilotPersistence(
        manifest=fixture.model_copy(update={"review_id": "r1-unresolved-fixture"}),
        ui_config=load_ui_config(R1_PACKAGE / "ui_config.json"),
        decisions_root=tmp_path / "r1",
        reviewer_session_id="m5_5g1a_detection_gold_pilot_reviewer_r1",
    )
    payload = save_payload(r1, case, valid_annotation(case, unresolved_strip=True))
    payload.pop("wizard_state")
    with pytest.raises(ValueError, match="must be resolved"):
        r1.save_detection_event(payload)


def test_server_materializes_and_replays_wizard_state(tmp_path: Path) -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    case = next(row for row in manifest.cases if row.task_type == "detection_gold_pitch_boundary")
    fixture = manifest.model_copy(update={"review_id": "wizard-state-fixture", "cases": [case]})
    persistence = DetectionGoldPilotPersistence(
        manifest=fixture,
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=tmp_path / "decisions",
        reviewer_session_id=REVIEWER,
    )
    result = persistence.save_detection_event(save_payload(persistence, case, valid_annotation(case)))
    assert result["wizard_states"][case.case_id]["step"] == 4
    recovery = persistence.recover_authoritative_state(write_sidecar=False)
    assert recovery["ledger_audit"]["event_replay_matches_authoritative_state"] is True
    assert recovery["materialized_state"]["wizard_states"][case.case_id] == result["wizard_states"][case.case_id]


def test_r2_populated_historical_root_does_not_contaminate_fresh_fixture(tmp_path: Path) -> None:
    historical_state_path = PACKAGE / "decisions" / "review_decisions.json"
    historical_events_path = PACKAGE / "decisions" / "review_decision_events.jsonl"
    before = {
        "state": sha256_file(historical_state_path),
        "events": sha256_file(historical_events_path),
    }
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    persistence = DetectionGoldPilotPersistence(
        manifest=manifest,
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=tmp_path / "fresh-decisions",
        reviewer_session_id=REVIEWER,
    )
    state = persistence.ensure_state()
    events = persistence.events_path
    app = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_app.js").read_text()
    assert not state["annotations"] and not state["decisions"] and not state["wizard_states"]
    assert state["event_sequence"] == 0
    assert events.stat().st_size == 0
    assert not list(persistence.decisions_root.glob("completed_review*"))
    assert "fi_detection_gold_${runtime.manifest.review_id}" in app
    assert read_json(PACKAGE / "reviewer_manifest.json")["review_id"] == REVIEW_ID
    assert read_json(historical_state_path)["event_sequence"] == 6
    assert sha256_file(historical_state_path) == before["state"]
    assert sha256_file(historical_events_path) == before["events"]


def test_timing_counts_only_modules_with_machine_candidate_queues() -> None:
    timing = read_json(STAGE / "04_BROWSER_PERSISTENCE_AND_USABILITY" / "truthful_timing_estimate.json")
    assert timing["action_counts"]["machine_candidate_questions"] == 581
    assert timing["human_measured_active_minutes"] is None
    assert timing["scripted_browser_time_claimed_as_human_time"] is False


def test_package_and_launcher_are_valid() -> None:
    result = validate_review_chassis_package(
        manifest_path=PACKAGE / "reviewer_manifest.json",
        ui_config_path=PACKAGE / "ui_config.json",
        evidence_root=PACKAGE / "evidence",
        decisions_root=PACKAGE / "decisions",
    )
    launcher = (PACKAGE / "launch_novice_guided_review.ps1").read_text()
    assert result["passed"] is True
    assert "--port 8807" in launcher
    assert REVIEWER in launcher
    assert "will not move ports" in launcher


def test_prior_workspaces_remain_byte_identical() -> None:
    result = read_json(STAGE / "01_PRIOR_STAGE_AND_STATE_AUDIT" / "prior_state_and_empty_root_validation.json")
    assert result["passed"] is True
    assert result["original_byte_identical"] is True
    assert result["r1_byte_identical"] is True
    assert result["r1_saved_cases"] == 0
    assert result["r1_event_sequence"] == 0


def test_real_browser_acceptance_covers_every_required_r2_scenario() -> None:
    result = read_json(STAGE / "04_BROWSER_PERSISTENCE_AND_USABILITY" / "browser_acceptance_results.json")
    expected = {
        "draw_three_people_with_machine_layers_hidden",
        "complete_one_question_at_a_time_for_each_person",
        "edit_second_person_without_long_sidebar",
        "review_candidate_completely_hidden_under_human_box",
        "review_candidates_without_clicking_candidate_overlays",
        "bind_clean_candidate_to_one_numbered_person",
        "bind_duplicate_candidate_to_non_latest_person",
        "bind_merged_candidate_to_two_of_three_people",
        "mark_candidate_background_with_zero_targets",
        "undo_and_edit_previous_candidate_answer",
        "resume_mid_person_question_after_browser_restart",
        "resume_mid_candidate_queue_after_server_restart",
        "complete_static_case_without_opening_advanced_details",
        "complete_dense_case_with_plain_coverage_control",
        "complete_temporal_manual_observation_with_empty_candidate_uuid_list",
        "complete_pitch_case_with_plain_boundary_question",
        "complete_football_burst_with_full_frame_visibility_questions",
        "verify_no_semantic_truth_prefilled",
        "verify_not_sure_paths_map_to_frozen_uncertainty_values",
        "verify_all_88_cases_remain_completable",
        "verify_completion_requires_empty_outbox_and_valid_server_state",
    }
    scenarios = result["required_browser_scenarios"]
    assert set(scenarios) == expected
    assert all(scenarios.values())
    assert result["all_case_render_audit"]["passed"] is True
    assert all(row["saved"] for row in result["completed_module_flows"].values())
    assert result["passed"] is True
