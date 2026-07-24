from __future__ import annotations

import copy
import json
import shutil
import uuid
from pathlib import Path

import pytest

from football_intelligence.detection_gold.incremental import (
    R3_R2_CLIENT_BUILD_ID,
    authoritative_candidate_binding_hash,
    authoritative_candidate_uuids,
    authoritative_frame_record,
    validate_revision_aware_wizard_state,
)
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1]
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
R3 = PART3 / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
LIVE_DECISIONS = R3 / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE" / "decisions"
R3_R1 = PART3 / "M5_5G1A_R3_R1_WIZARD_STATE_INVALIDATION_AND_SAFE_CASE_RESTART_v1"
R3_R1_PACKAGE = R3_R1 / "05_REPAIRED_INCREMENTAL_ANNOTATION_PACKAGE"
STAGE = PART3 / "M5_5G1A_R3_R2_DENSE_FIRST_TRANCHE_SPLIT_AND_ATOMIC_COMPLETION_v1"
PACKAGE = STAGE / "05_DENSE_FIRST_INCREMENTAL_ANNOTATION_PACKAGE"
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer_r3"
CASE_HASH = "986604e34e6f831825dfb76601f854ece083e3ba9001f97bad26d6d596e6a401"
EVIDENCE_HASH = "58c2de8da4e6e8d2160a29550e8030e7c5225845fc58be44eae7651c0b4a1ab4"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def wizard_state(case: object, annotation: dict) -> dict:
    record = authoritative_frame_record(case)
    candidates = authoritative_candidate_uuids(case)
    records = {}
    for index, candidate_uuid in enumerate(candidates, start=1):
        relation = next(row for row in annotation["candidate_relations"] if row["candidate_uuid"] == candidate_uuid)
        answer = {
            "candidate_uuid": candidate_uuid,
            "relation": relation["relation"],
            "annotation_uuids": relation["annotation_uuids"],
            "answered_against_human_truth_revision": 0,
            "answered_person_question_revision": 0,
            "candidate_answer_revision": index,
            "validity": "VALID",
            "invalidation_reason": None,
            "answered_at": "2026-07-23T00:00:00Z",
            "revalidated_at": None,
            "revalidation_event": "INITIAL_REVIEW",
        }
        if "candidate_visible_mask_coverage" in relation:
            answer["candidate_visible_mask_coverage"] = relation["candidate_visible_mask_coverage"]
        records[candidate_uuid] = answer
    return {
        "schema_version": "football_intelligence.m5_5g1a_r3.wizard_state.v1",
        "case_id": case.case_id,
        "step": 4,
        "drawing_complete": True,
        "current_object_uuid": None,
        "question_index": 0,
        "completed_object_uuids": [row["annotation_uuid"] for row in annotation["visible_masks"]],
        "footpoint_placed_uuids": [],
        "footpoint_reviews": {},
        "pending_footpoint_decision": None,
        "candidate_index": max(0, len(candidates) - 1),
        "candidate_phase": "relation",
        "candidate_relation": None,
        "candidate_targets": [],
        "candidate_answered_uuids": candidates,
        "candidate_answer_records": records,
        "mask_front_answers": {},
        "human_truth_revision": 0,
        "person_question_revision": 0,
        "candidate_answer_revision": len(candidates),
        "summary_revision": 1,
        "person_question_completion_revisions": {row["annotation_uuid"]: 0 for row in annotation["visible_masks"]},
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
        "active_tranche_id": "C1_DENSE_OVERLAP",
        "authoritative_frame_sequence": int(record["frame_sequence"]),
        "authoritative_source_frame_sha256": record["source_frame_sha256"],
        "primary_canvas_frame_sequence": int(record["frame_sequence"]),
        "primary_canvas_source_frame_sha256": record["source_frame_sha256"],
        "candidate_queue_binding_hash": authoritative_candidate_binding_hash(case),
    }


def empty_dense_annotation(case: object) -> dict:
    return {
        "schema_version": "m5_5g1a_detection_gold_v1",
        "source_binding": copy.deepcopy(case.visible_metadata["source_binding"]),
        "dense_region_uuid": f"dense-{case.case_id}",
        "trigger_reason": case.visible_metadata["pilot_stratum"],
        "human_visible_person_count": 0,
        "visible_masks": [],
        "candidate_relations": [
            {"candidate_uuid": value, "relation": "BACKGROUND", "annotation_uuids": []}
            for value in authoritative_candidate_uuids(case)
        ],
        "uncertain_or_ignore": True,
        "reviewer_agreement": "NOT_REVIEWED",
        "adjudication_state": "NOT_REQUIRED",
        "note": "",
    }


def save_case(store: DetectionGoldPilotPersistence, case: object, annotation: dict) -> dict:
    event_id = str(uuid.uuid4())
    return store.save_detection_event(
        {
            "event_type": "DETECTION_CASE_SAVED",
            "review_id": store.manifest.review_id,
            "reviewer_session_id": REVIEWER,
            "case_id": case.case_id,
            "annotation": annotation,
            "wizard_state": wizard_state(case, annotation),
            "client_event_id": event_id,
            "idempotency_key": event_id,
            "expected_server_state_hash": store.state()["server_state_hash"],
            "elapsed_active_seconds": 1,
        }
    )


def test_live_gate_requires_exact_completed_static_state() -> None:
    gate = read_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "live_state_precondition.json")
    assert gate["passed"] is True
    assert gate["all_32_static_cases_saved"] is True
    assert gate["tranche_a_completed"] is True and gate["tranche_b_completed"] is True
    assert gate["later_saved_case_count"] == 0
    assert gate["pending_outbox_events"] == 0
    assert gate["live_decisions_byte_identical_after_build"] is True


def test_versioned_manifest_splits_old_c_without_case_or_evidence_drift() -> None:
    versioned = read_json(STAGE / "02_TRANCHE_MANIFEST_SPLIT" / "gold_tranche_manifest_v2.json")
    split = read_json(STAGE / "02_TRANCHE_MANIFEST_SPLIT" / "tranche_split_validation.json")
    manifest = read_json(PACKAGE / "reviewer_manifest.json")
    assert versioned["tranche_order"] == [
        "A_CORE_STATIC",
        "B_REMAINING_STATIC",
        "C1_DENSE_OVERLAP",
        "C2_PITCH_BOUNDARY",
        "D_TEMPORAL_PLAYER",
        "E_FOOTBALL",
    ]
    assert [len(versioned["tranches"][key]["case_ids"]) for key in versioned["tranche_order"]] == [
        18,
        14,
        8,
        12,
        12,
        24,
    ]
    assert versioned["total_case_count"] == 88
    assert split["passed"] is True
    assert stable_hash(manifest["cases"]) == CASE_HASH
    assert sha256_file(PACKAGE / "reviewer_manifest.json") == sha256_file(R3_R1_PACKAGE / "reviewer_manifest.json")
    assert read_json(PACKAGE / "review_package_validation.json")["evidence_copy"]["tree_hash"] == EVIDENCE_HASH


def test_r3_r2_client_policy_and_dense_language_are_enabled() -> None:
    ui = read_json(PACKAGE / "ui_config.json")
    app = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_app.js").read_text(encoding="utf-8")
    wizard = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_wizard.js").read_text(
        encoding="utf-8"
    )
    assert ui["question_contract"]["client_build_id"] == R3_R2_CLIENT_BUILD_ID
    assert ui["question_contract"]["default_tranche_id"] == "C1_DENSE_OVERLAP"
    assert "first_load_forced_tranche_id" in ui["question_contract"]
    assert "refreshTrancheOptions" in app
    assert "!firstRepairLoad && runtime.revisionAwareR3R1" in app
    for text in (
        "Trace each visible person",
        "Answer short overlap questions",
        "Trace only the part of the person you can actually see.",
        "Is another person in front of this person?",
        "Which person is in front?",
        "How clear is this outline?",
        "How much of the visible person is covered by this machine box?",
    ):
        assert text in wizard


def test_dense_candidate_revision_record_binds_visible_mask_coverage() -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    case = next(row for row in manifest.cases if row.case_id == "m5_5g1a_case_033")
    record = authoritative_frame_record(case)
    candidate_uuid = authoritative_candidate_uuids(case)[0]
    box = next(row["bbox_original_pixels"] for row in record["candidates"] if row["diagnostic_uuid"] == candidate_uuid)
    mask_uuid = "mask-fixture"
    annotation = empty_dense_annotation(case)
    annotation["visible_masks"] = [
        {
            "annotation_uuid": mask_uuid,
            "polygon_original_pixels": [
                {"x": box["x1"], "y": box["y1"]},
                {"x": box["x2"], "y": box["y1"]},
                {"x": box["x2"], "y": box["y2"]},
                {"x": box["x1"], "y": box["y2"]},
            ],
            "mask_quality": "PRECISE",
            "visible_body_box": box,
            "occlusion_order": 0,
            "pairwise_overlap_annotation_uuids": [],
            "truncation_flags": [],
            "current_frame_pixel_support": True,
        }
    ]
    annotation["human_visible_person_count"] = 1
    annotation["candidate_relations"][0] = {
        "candidate_uuid": candidate_uuid,
        "relation": "CLEAN_SINGLE_INSTANCE",
        "annotation_uuids": [mask_uuid],
        "candidate_visible_mask_coverage": 0.75,
    }
    wizard = wizard_state(case, annotation)
    wizard["candidate_answer_records"][candidate_uuid]["candidate_visible_mask_coverage"] = 0.5
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_revision_aware_wizard_state(case, annotation, wizard)
    wizard["candidate_answer_records"][candidate_uuid]["candidate_visible_mask_coverage"] = 0.75
    validate_revision_aware_wizard_state(case, annotation, wizard)


def test_c1_completes_atomically_without_c2_or_full_pilot(tmp_path: Path) -> None:
    copied = tmp_path / "decisions"
    shutil.copytree(LIVE_DECISIONS, copied)
    store = DetectionGoldPilotPersistence(
        manifest=load_manifest(PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=copied,
        reviewer_session_id=REVIEWER,
    )
    initial = store.state()
    assert set(initial["tranche_completions"]) == {"A_CORE_STATIC", "B_REMAINING_STATIC"}
    c1_ids = store.ui_config.question_contract["gold_tranches"]["C1_DENSE_OVERLAP"]["case_ids"]
    for case_id in c1_ids:
        case = store.case_map()[case_id]
        save_case(store, case, empty_dense_annotation(case))
    before = store.state()
    event_id = str(uuid.uuid4())
    completed = store.complete_tranche(
        {
            "tranche_id": "C1_DENSE_OVERLAP",
            "client_event_id": event_id,
            "idempotency_key": event_id,
            "expected_server_state_hash": before["server_state_hash"],
            "pending_outbox_events": 0,
            "evidence_blocker_count": 0,
            "unresolved_draft_count": 0,
            "unresolved_divergence": False,
        }
    )
    assert set(completed["tranche_completions"]) == {"A_CORE_STATIC", "B_REMAINING_STATIC", "C1_DENSE_OVERLAP"}
    assert completed["completed"] is False
    assert "C2_PITCH_BOUNDARY" not in completed["tranche_completions"]
    assert validate_completion_bundle(copied / "completed_tranches" / "C1_DENSE_OVERLAP")["passed"] is True
    with pytest.raises(ValueError, match="(?:saved R3 cases|completed tranches) are immutable"):
        case = store.case_map()[c1_ids[0]]
        save_case(store, case, empty_dense_annotation(case))
    with pytest.raises(ValueError, match="completion is blocked"):
        store.complete_detection({"expected_server_state_hash": completed["server_state_hash"]})


def test_browser_acceptance_and_review_pack_when_finalized() -> None:
    browser = read_json(STAGE / "04_BROWSER_PERSISTENCE_AND_COMPLETION" / "browser_persistence_results.json")
    summary = read_json(STAGE / "07_COMMANDS_AND_TESTS" / "build_summary.json")
    if browser.get("status") == "PENDING_REAL_BROWSER_ACCEPTANCE" or summary.get("tests_pending") is True:
        pytest.skip("R3-R2 final acceptance has not run yet")
    assert browser["passed"] is True
    assert len(browser["required_scenarios"]) == 20
    assert all(browser["required_scenarios"].values())
    assert summary["classification"] == "PASS_DENSE_FIRST_TRANCHE_C1_READY"
    files = [path for path in (STAGE / "08_REVIEW_PACK_FOR_CHATGPT").iterdir() if path.is_file()]
    assert len(files) <= 20
    assert (STAGE / "08_REVIEW_PACK_FOR_CHATGPT" / "04_SOURCE_DIFF.patch").stat().st_size > 0
