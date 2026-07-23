from __future__ import annotations

import copy
import json
import shutil
import uuid
from pathlib import Path

import pytest

from football_intelligence.detection_gold.incremental import (
    R3_R1_CLIENT_BUILD_ID,
    R3_WIZARD_SCHEMA,
    authoritative_candidate_binding_hash,
    authoritative_candidate_uuids,
    authoritative_frame_record,
    validate_revision_aware_wizard_state,
)
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1]
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
R3 = PART3 / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
R3_PACKAGE = R3 / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
R3_DECISIONS = R3_PACKAGE / "decisions"
STAGE = PART3 / "M5_5G1A_R3_R1_WIZARD_STATE_INVALIDATION_AND_SAFE_CASE_RESTART_v1"
PACKAGE = STAGE / "05_REPAIRED_INCREMENTAL_ANNOTATION_PACKAGE"
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer_r3"
CASE_HASH = "986604e34e6f831825dfb76601f854ece083e3ba9001f97bad26d6d596e6a401"
EVIDENCE_HASH = "58c2de8da4e6e8d2160a29550e8030e7c5225845fc58be44eae7651c0b4a1ab4"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def case_7(package: Path = PACKAGE) -> object:
    manifest = load_manifest(package / "reviewer_manifest.json")
    return next(case for case in manifest.cases if case.case_id == "m5_5g1a_case_016")


def empty_static_annotation(case: object) -> dict:
    return {
        "schema_version": "m5_5g1a_detection_gold_v1",
        "source_binding": copy.deepcopy(case.visible_metadata["source_binding"]),
        "visible_person_count": 0,
        "player_instances": [],
        "candidate_relations": [
            {"candidate_uuid": candidate_uuid, "relation": "BACKGROUND", "annotation_uuids": []}
            for candidate_uuid in authoritative_candidate_uuids(case)
        ],
        "earliest_failure_stage": "UNRESOLVED",
        "note": "",
    }


def revision_wizard(case: object) -> dict:
    record = authoritative_frame_record(case)
    candidates = authoritative_candidate_uuids(case)
    records = {}
    for revision, candidate_uuid in enumerate(candidates, start=1):
        records[candidate_uuid] = {
            "candidate_uuid": candidate_uuid,
            "relation": "BACKGROUND",
            "annotation_uuids": [],
            "answered_against_human_truth_revision": 0,
            "answered_person_question_revision": 0,
            "candidate_answer_revision": revision,
            "validity": "VALID",
            "invalidation_reason": None,
            "answered_at": "2026-07-22T00:00:00Z",
            "revalidated_at": None,
            "revalidation_event": "INITIAL_REVIEW",
        }
    return {
        "schema_version": R3_WIZARD_SCHEMA,
        "case_id": case.case_id,
        "step": 4,
        "drawing_complete": True,
        "current_object_uuid": None,
        "question_index": 0,
        "completed_object_uuids": [],
        "footpoint_placed_uuids": [],
        "footpoint_reviews": {},
        "pending_footpoint_decision": None,
        "candidate_index": max(0, len(candidates) - 1),
        "candidate_phase": "relation",
        "candidate_relation": None,
        "candidate_targets": [],
        "candidate_answered_uuids": candidates,
        "candidate_answer_records": records,
        "human_truth_revision": 0,
        "person_question_revision": 0,
        "candidate_answer_revision": len(candidates),
        "summary_revision": 1,
        "person_question_completion_revisions": {},
        "summary_validity": "VALID",
        "summary_human_truth_revision": 0,
        "invalidation_notice": None,
        "frame_answered_sequences": [],
        "frame_phase": "visibility",
        "desired_frame_state": None,
        "pitch_footpoint_set": False,
        "pitch_question_index": 0,
        "pitch_answers": [],
        "football_candidate_answers": {},
        "failure_reviewed": True,
        "help_opened": False,
        "active_tranche_id": "B_REMAINING_STATIC",
        "authoritative_frame_sequence": int(record["frame_sequence"]),
        "authoritative_source_frame_sha256": str(record["source_frame_sha256"]),
        "primary_canvas_frame_sequence": int(record["frame_sequence"]),
        "primary_canvas_source_frame_sha256": str(record["source_frame_sha256"]),
        "candidate_queue_binding_hash": authoritative_candidate_binding_hash(case),
    }


def persistence(tmp_path: Path, package: Path = PACKAGE) -> DetectionGoldPilotPersistence:
    return DetectionGoldPilotPersistence(
        manifest=load_manifest(package / "reviewer_manifest.json"),
        ui_config=load_ui_config(package / "ui_config.json"),
        decisions_root=tmp_path / "decisions",
        reviewer_session_id=REVIEWER,
    )


def save_payload(store: DetectionGoldPilotPersistence, case: object, annotation: dict, wizard: dict) -> dict:
    event_id = str(uuid.uuid4())
    return {
        "event_type": "DETECTION_CASE_SAVED",
        "review_id": store.manifest.review_id,
        "reviewer_session_id": REVIEWER,
        "case_id": case.case_id,
        "annotation": annotation,
        "wizard_state": wizard,
        "client_event_id": event_id,
        "idempotency_key": event_id,
        "expected_server_state_hash": store.state()["server_state_hash"],
        "elapsed_active_seconds": 1,
    }


def test_repaired_package_preserves_frozen_payload_and_uses_external_decisions_root() -> None:
    manifest = read_json(PACKAGE / "reviewer_manifest.json")
    config = read_json(PACKAGE / "ui_config.json")["question_contract"]
    validation = read_json(PACKAGE / "review_package_validation.json")
    assert stable_hash(manifest["cases"]) == CASE_HASH
    assert validation["evidence_copy"]["file_count"] == 1512
    assert validation["evidence_copy"]["tree_hash"] == EVIDENCE_HASH
    assert config["client_build_id"] == R3_R1_CLIENT_BUILD_ID
    assert config["revision_aware_wizard_state"] is True
    assert config["prior_indexeddb_namespace_import_forbidden"] is True
    assert config["compatible_predecessor_ui_config_hashes"] == [
        read_json(R3_DECISIONS / "review_decisions.json")["ui_config_hash"]
    ]
    assert not (PACKAGE / "decisions").exists()


def test_live_gate_identifies_exact_six_saved_b_cases_and_unsaved_case_7() -> None:
    gate = read_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "live_state_precondition.json")
    assert gate["passed"] is True
    assert gate["tranche_a_completed"] is True
    assert gate["tranche_a_saved_case_count"] == 18
    assert gate["tranche_b_saved_case_count"] == 6
    assert gate["tranche_b_current_case_id"] == "m5_5g1a_case_016"
    assert gate["tranche_b_current_case_saved"] is False
    assert gate["tranche_b_completed"] is False
    assert gate["pending_outbox_events"] == 0
    assert gate["event_replay_matches_authoritative_state"] is True


def test_original_human_decision_files_remain_byte_identical_to_preservation_manifest() -> None:
    gate = read_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "live_state_precondition.json")
    before = gate["decisions_tree_before"]
    after = gate["decisions_tree_after"]
    assert before == after
    for row in before["files"]:
        path = R3_DECISIONS / row["path"]
        assert path.stat().st_size == row["size_bytes"]
        assert sha256_file(path) == row["sha256"]


def test_revision_validator_accepts_current_state_and_rejects_stale_or_dangling_answers() -> None:
    case = case_7()
    annotation = empty_static_annotation(case)
    wizard = revision_wizard(case)
    validate_revision_aware_wizard_state(case, annotation, wizard)

    stale = copy.deepcopy(wizard)
    first = authoritative_candidate_uuids(case)[0]
    stale["candidate_answer_records"][first]["validity"] = "NEEDS_REVIEW"
    stale["candidate_answer_records"][first]["invalidation_reason"] = "person removed"
    stale["candidate_answered_uuids"].remove(first)
    with pytest.raises(ValueError, match="still needs review"):
        validate_revision_aware_wizard_state(case, annotation, stale)

    dangling = copy.deepcopy(wizard)
    dangling["candidate_answer_records"][first]["annotation_uuids"] = ["deleted-person"]
    with pytest.raises(ValueError, match="target mismatch"):
        validate_revision_aware_wizard_state(case, annotation, dangling)

    stale_summary = copy.deepcopy(wizard)
    stale_summary["human_truth_revision"] += 1
    with pytest.raises(ValueError, match="summary is stale"):
        validate_revision_aware_wizard_state(case, annotation, stale_summary)


def test_server_rejects_stale_state_and_persists_a_fully_revalidated_case(tmp_path: Path) -> None:
    store = persistence(tmp_path)
    case = store.case_map()["m5_5g1a_case_016"]
    annotation = empty_static_annotation(case)
    stale = revision_wizard(case)
    candidate_uuid = authoritative_candidate_uuids(case)[0]
    stale["candidate_answer_records"][candidate_uuid]["validity"] = "NEEDS_REVIEW"
    stale["candidate_answer_records"][candidate_uuid]["invalidation_reason"] = "geometry changed"
    stale["candidate_answered_uuids"].remove(candidate_uuid)
    with pytest.raises(ValueError, match="still needs review"):
        store.save_detection_event(save_payload(store, case, annotation, stale))
    assert store.state()["event_sequence"] == 0

    saved = store.save_detection_event(save_payload(store, case, annotation, revision_wizard(case)))
    assert saved["event_sequence"] == 1
    assert saved["annotations"][case.case_id] == annotation
    assert saved["wizard_states"][case.case_id]["summary_validity"] == "VALID"


def test_predecessor_config_read_is_nonmutating_and_rebind_occurs_only_on_new_save(
    tmp_path: Path,
) -> None:
    decisions = tmp_path / "decisions"
    decisions.mkdir()
    for filename in ("review_decisions.json", "review_decision_events.jsonl"):
        shutil.copy2(R3_DECISIONS / filename, decisions / filename)
    before = (decisions / "review_decisions.json").read_bytes()
    store = DetectionGoldPilotPersistence(
        manifest=load_manifest(PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=decisions,
        reviewer_session_id=REVIEWER,
    )
    state = store.state()
    assert state["event_sequence"] == 26
    assert (decisions / "review_decisions.json").read_bytes() == before

    old_hashes = copy.deepcopy(state["annotation_hashes"])
    case = store.case_map()["m5_5g1a_case_016"]
    saved = store.save_detection_event(save_payload(store, case, empty_static_annotation(case), revision_wizard(case)))
    assert saved["event_sequence"] == 27
    assert all(saved["annotation_hashes"][case_id] == value for case_id, value in old_hashes.items())
    assert saved["ui_config_hash"] == store.ui_config_hash_value


def test_repair_server_refuses_to_resave_an_existing_human_case(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions"
    decisions.mkdir()
    for filename in ("review_decisions.json", "review_decision_events.jsonl"):
        shutil.copy2(R3_DECISIONS / filename, decisions / filename)
    store = DetectionGoldPilotPersistence(
        manifest=load_manifest(PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=decisions,
        reviewer_session_id=REVIEWER,
    )
    case = store.case_map()["m5_5g1a_case_003"]
    state = store.state()
    with pytest.raises(ValueError, match="saved R3 cases are immutable"):
        store.save_detection_event(
            save_payload(
                store,
                case,
                copy.deepcopy(state["annotations"][case.case_id]),
                copy.deepcopy(state["wizard_states"][case.case_id]),
            )
        )


def test_old_r3_client_build_keeps_legacy_wizard_validation_path(tmp_path: Path) -> None:
    store = persistence(tmp_path, R3_PACKAGE)
    assert store.ui_config.question_contract.get("client_build_id") != R3_R1_CLIENT_BUILD_ID
    case = store.case_map()["m5_5g1a_case_016"]
    wizard = revision_wizard(case)
    for field in (
        "candidate_answer_records",
        "human_truth_revision",
        "person_question_revision",
        "candidate_answer_revision",
        "summary_revision",
        "person_question_completion_revisions",
        "summary_validity",
        "summary_human_truth_revision",
        "invalidation_notice",
    ):
        wizard.pop(field)
    saved = store.save_detection_event(save_payload(store, case, empty_static_annotation(case), wizard))
    assert saved["event_sequence"] == 1


def test_read_only_recovery_does_not_write_a_materialization_sidecar(tmp_path: Path) -> None:
    store = persistence(tmp_path)
    store.ensure_state()
    sidecar = store.decisions_root / "detection_gold_recovery_materialization.json"
    assert not sidecar.exists()
    recovery = store.recover_authoritative_state(write_sidecar=False)
    assert recovery["server_event_sequence"] == 0
    assert not sidecar.exists()


def test_client_source_contains_revision_invalidation_and_safe_restart_controls() -> None:
    app = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_app.js").read_text()
    wizard = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_wizard.js").read_text()
    assert "candidate_answer_records" in wizard
    assert "answered_against_human_truth_revision" in wizard
    assert "Some machine-box answers need checking again because a person was removed." in wizard
    assert "Start again by drawing the visible people. Earlier machine-box answers need review." in wizard
    assert 'Keep the previous "not a person" answers?' in wizard
    assert "Review answers that need checking" in wizard
    assert "Return to drawing people" in wizard
    assert "Restart this case" in wizard
    assert "this.host.selectObject(annotationUuid)" in wizard
    assert "selectObject," in app
    assert "restartCurrentCase" in app
    assert "runtime.wizard?.reset(caseData.case_id)" in app
    assert "write_sidecar: !runtime.revisionAwareR3R1" in app
    assert "stale_prior_namespace_imported: false" in app
    assert "Six saved Tranche B cases were restored from the server." in app


def test_browser_and_review_pack_results_when_finalized() -> None:
    browser = read_json(STAGE / "04_BROWSER_PERSISTENCE_AND_REGRESSION" / "browser_persistence_results.json")
    if browser["status"] == "PENDING_REAL_BROWSER_ACCEPTANCE":
        pytest.skip("real-browser acceptance has not run yet")
    assert browser["passed"] is True
    assert all(browser["required_scenarios"].values())
    assert len(browser["visual_regression"]) == 6

    summary = read_json(STAGE / "06_COMMANDS_AND_TESTS" / "build_summary.json")
    if summary.get("review_pack_pending"):
        pytest.skip("review pack has not been finalized yet")
    files = [path for path in (STAGE / "07_REVIEW_PACK_FOR_CHATGPT").iterdir() if path.is_file()]
    assert len(files) <= 20
    assert (STAGE / "07_REVIEW_PACK_FOR_CHATGPT" / "04_SOURCE_DIFF.patch").stat().st_size > 0
