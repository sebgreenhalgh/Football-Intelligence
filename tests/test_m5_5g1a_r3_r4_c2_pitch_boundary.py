from __future__ import annotations

import copy
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import pytest

from football_intelligence.detection_gold.incremental import (
    R3_R4_C2_CLIENT_BUILD_ID,
    authoritative_candidate_binding_hash,
    authoritative_candidate_uuids,
    authoritative_frame_record,
    validate_revision_aware_wizard_state,
)
from football_intelligence.detection_gold.models import validate_c2_pitch_boundary_annotation
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.hashing import sha256_file
from football_intelligence.review_chassis.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1]
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
R3 = PART3 / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
LIVE_DECISIONS = R3 / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE" / "decisions"
STAGE = PART3 / "M5_5G1A_R3_R4_C2_PITCH_BOUNDARY_GOLD_AND_OFF_PITCH_SUPPLY_ANNOTATION_v1"
PACKAGE = STAGE / "05_C2_PITCH_BOUNDARY_REVIEW_PACKAGE"
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer_r3"
C2 = "C2_PITCH_BOUNDARY"
C2_CASE_IDS = [f"m5_5g1a_case_{number:03d}" for number in range(53, 65)]
CASE_HASH = "986604e34e6f831825dfb76601f854ece083e3ba9001f97bad26d6d596e6a401"
EVIDENCE_HASH = "58c2de8da4e6e8d2160a29550e8030e7c5225845fc58be44eae7651c0b4a1ab4"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def copy_pre_c2_decisions_fixture(destination: Path) -> None:
    """Materialize the immutable event-44 state without touching the completed live root."""

    snapshot_path = LIVE_DECISIONS / "snapshots" / "review_state_000044.json"
    snapshot = read_json(snapshot_path)
    state = snapshot["state"]
    assert state["event_sequence"] == 44
    assert set(state["tranche_completions"]) == {
        "A_CORE_STATIC",
        "B_REMAINING_STATIC",
        "C1_DENSE_OVERLAP",
    }
    assert not set(C2_CASE_IDS) & set(state["annotations"])

    destination.mkdir(parents=True)
    (destination / "review_decisions.json").write_text(
        json.dumps(state, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    events = (LIVE_DECISIONS / "review_decision_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) >= 57
    prefix = [json.loads(line) for line in events[:44]]
    assert [event["event_sequence"] for event in prefix] == list(range(1, 45))
    assert prefix[-1]["event_type"] == "DETECTION_TRANCHE_COMPLETED"
    (destination / "review_decision_events.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n" for event in prefix),
        encoding="utf-8",
    )
    for tranche_id in state["tranche_completions"]:
        shutil.copytree(
            LIVE_DECISIONS / "completed_tranches" / tranche_id,
            destination / "completed_tranches" / tranche_id,
        )
    snapshots = destination / "snapshots"
    snapshots.mkdir()
    shutil.copy2(snapshot_path, snapshots / snapshot_path.name)
    shutil.copy2(snapshot_path.with_suffix(".json.sha256"), snapshots / f"{snapshot_path.name}.sha256")


def synthetic_annotation(case: Any, *, hidden_feet: bool = True) -> dict[str, Any]:
    record = authoritative_frame_record(case)
    candidate_uuids = authoritative_candidate_uuids(case)
    first_uuid = candidate_uuids[0] if candidate_uuids else None
    if first_uuid:
        candidate = next(row for row in record["candidates"] if row["diagnostic_uuid"] == first_uuid)
        box = copy.deepcopy(candidate["bbox_original_pixels"])
    else:
        focal = record["focal_bounds"]
        centre_x = (focal["x1"] + focal["x2"]) / 2
        centre_y = (focal["y1"] + focal["y2"]) / 2
        box = {"x1": centre_x - 8, "y1": centre_y - 20, "x2": centre_x + 8, "y2": centre_y + 20}
    person_uuid = f"person-{case.case_id}"
    person = {
        "annotation_uuid": person_uuid,
        "visible_body_box": box,
        "footpoint": None if hidden_feet else {"x": (box["x1"] + box["x2"]) / 2, "y": box["y2"]},
        "footpoint_status": "FEET_NOT_VISIBLE" if hidden_feet else "OBSERVED_CLEAR",
        "footpoint_uncertainty_pixels": 20 if hidden_feet else 3,
        "pitch_state": "OFF_PITCH",
        "pitch_state_certainty": "CLEAR",
        "coarse_role": "PLAYER",
        "minimum_visible_dimensions": {
            "width_pixels": box["x2"] - box["x1"],
            "height_pixels": box["y2"] - box["y1"],
        },
    }
    return {
        "schema_version": "m5_5g1a_c2_pitch_boundary_v1",
        "source_binding": copy.deepcopy(case.visible_metadata["source_binding"]),
        "visible_person_count": 1,
        "player_instances": [person],
        "candidate_relations": [
            {
                "candidate_uuid": candidate_uuid,
                "relation": "CLEAN_SINGLE_INSTANCE" if first_uuid and candidate_uuid == first_uuid else "BACKGROUND",
                "annotation_uuids": [person_uuid] if first_uuid and candidate_uuid == first_uuid else [],
            }
            for candidate_uuid in candidate_uuids
        ],
        "note": "",
    }


def synthetic_wizard(case: Any, annotation: dict[str, Any]) -> dict[str, Any]:
    record = authoritative_frame_record(case)
    candidates = authoritative_candidate_uuids(case)
    relations = {row["candidate_uuid"]: row for row in annotation["candidate_relations"]}
    people = annotation["player_instances"]
    answer_records = {}
    for index, candidate_uuid in enumerate(candidates, start=1):
        relation = relations[candidate_uuid]
        answer_records[candidate_uuid] = {
            "candidate_uuid": candidate_uuid,
            "relation": relation["relation"],
            "annotation_uuids": relation["annotation_uuids"],
            "answered_against_human_truth_revision": 0,
            "answered_person_question_revision": 0,
            "candidate_answer_revision": index,
            "validity": "VALID",
            "invalidation_reason": None,
            "answered_at": "2026-07-25T00:00:00Z",
            "revalidated_at": None,
            "revalidation_event": "INITIAL_REVIEW",
        }
    reviews = {
        person["annotation_uuid"]: {
            "status": person["footpoint_status"],
            "confirmed": True,
            "coarse_role": person["coarse_role"],
            "pitch_state": person["pitch_state"],
            "pitch_state_certainty": person["pitch_state_certainty"],
        }
        for person in people
    }
    return {
        "schema_version": "football_intelligence.m5_5g1a_r3.wizard_state.v1",
        "case_id": case.case_id,
        "step": 4,
        "drawing_complete": True,
        "current_object_uuid": None,
        "question_index": 0,
        "completed_object_uuids": [row["annotation_uuid"] for row in people],
        "footpoint_placed_uuids": [row["annotation_uuid"] for row in people if row["footpoint"]],
        "footpoint_reviews": reviews,
        "pending_footpoint_decision": None,
        "candidate_index": max(0, len(candidates) - 1),
        "candidate_phase": "relation",
        "candidate_relation": None,
        "candidate_targets": [],
        "candidate_answered_uuids": candidates,
        "candidate_answer_records": answer_records,
        "mask_front_answers": {},
        "human_truth_revision": 0,
        "person_question_revision": 0,
        "candidate_answer_revision": len(candidates),
        "summary_revision": 1,
        "person_question_completion_revisions": {row["annotation_uuid"]: 0 for row in people},
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
        "active_tranche_id": C2,
        "authoritative_frame_sequence": int(record["frame_sequence"]),
        "authoritative_source_frame_sha256": record["source_frame_sha256"],
        "primary_canvas_frame_sequence": int(record["frame_sequence"]),
        "primary_canvas_source_frame_sha256": record["source_frame_sha256"],
        "candidate_queue_binding_hash": authoritative_candidate_binding_hash(case),
    }


def save_case(store: DetectionGoldPilotPersistence, case: Any, annotation: dict[str, Any]) -> dict[str, Any]:
    event_id = str(uuid.uuid4())
    return store.save_detection_event(
        {
            "event_type": "DETECTION_CASE_SAVED",
            "review_id": store.manifest.review_id,
            "reviewer_session_id": REVIEWER,
            "case_id": case.case_id,
            "annotation": annotation,
            "wizard_state": synthetic_wizard(case, annotation),
            "client_event_id": event_id,
            "idempotency_key": event_id,
            "expected_server_state_hash": store.state()["server_state_hash"],
            "elapsed_active_seconds": 1,
        }
    )


def test_c2_schema_preserves_hidden_feet_and_off_pitch_people() -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    case = next(row for row in manifest.cases if row.case_id == C2_CASE_IDS[0])
    annotation = synthetic_annotation(case)
    validated = validate_c2_pitch_boundary_annotation(annotation)
    person = validated["player_instances"][0]
    assert person["coarse_role"] == "PLAYER"
    assert person["pitch_state"] == "OFF_PITCH"
    assert person["footpoint_status"] == "FEET_NOT_VISIBLE"
    assert person.get("footpoint") is None

    invalid = copy.deepcopy(annotation)
    invalid["player_instances"][0]["footpoint"] = {"x": 100, "y": 100}
    with pytest.raises(ValueError, match="cannot carry an observed point"):
        validate_c2_pitch_boundary_annotation(invalid)


def test_visible_feet_require_an_explicit_confirmed_point() -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    case = next(row for row in manifest.cases if row.case_id == C2_CASE_IDS[1])
    annotation = synthetic_annotation(case, hidden_feet=False)
    validate_c2_pitch_boundary_annotation(annotation)
    annotation["player_instances"][0]["footpoint"] = None
    with pytest.raises(ValueError, match="require a point"):
        validate_c2_pitch_boundary_annotation(annotation)


def test_c2_package_membership_and_frozen_bytes() -> None:
    membership = read_json(STAGE / "02_C2_CASE_AND_PITCH_POLYGON_VALIDATION" / "c2_case_membership_validation.json")
    polygon = read_json(
        STAGE / "02_C2_CASE_AND_PITCH_POLYGON_VALIDATION" / "pitch_polygon_and_transform_validation.json"
    )
    package = read_json(PACKAGE / "review_package_validation.json")
    ui = read_json(PACKAGE / "ui_config.json")
    assert membership["passed"] is True
    assert membership["actual_case_ids"] == C2_CASE_IDS
    assert membership["case_payload_hash"] == CASE_HASH
    assert membership["evidence_tree_hash"] == EVIDENCE_HASH
    assert polygon["passed"] is True
    assert package["passed"] is True
    assert ui["question_contract"]["client_build_id"] == R3_R4_C2_CLIENT_BUILD_ID
    assert ui["question_contract"]["default_tranche_id"] == C2
    assert ui["question_contract"]["first_load_forced_tranche_id"] == C2
    assert ui["question_contract"]["indexeddb_namespace"].endswith("r3_r4_c2_pitch_boundary_v1")
    assert len(ui["question_contract"]["static_authoritative_bindings"]) == 52


def test_c2_candidate_truth_is_independent_of_role_and_pitch_state() -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    case = next(row for row in manifest.cases if row.case_id == C2_CASE_IDS[2])
    annotation = synthetic_annotation(case)
    wizard = synthetic_wizard(case, annotation)
    original_records = copy.deepcopy(wizard["candidate_answer_records"])
    annotation["player_instances"][0]["coarse_role"] = "STAFF_OR_SPECTATOR"
    annotation["player_instances"][0]["pitch_state"] = "BOUNDARY_UNCERTAIN"
    annotation["player_instances"][0]["pitch_state_certainty"] = "UNCERTAIN"
    review = wizard["footpoint_reviews"][annotation["player_instances"][0]["annotation_uuid"]]
    review.update(
        {
            "coarse_role": "STAFF_OR_SPECTATOR",
            "pitch_state": "BOUNDARY_UNCERTAIN",
            "pitch_state_certainty": "UNCERTAIN",
        }
    )
    validate_revision_aware_wizard_state(case, annotation, wizard)
    assert wizard["candidate_answer_records"] == original_records


def test_c2_atomic_completion_uses_temporary_decisions_only(tmp_path: Path) -> None:
    source_hashes = {
        path.relative_to(LIVE_DECISIONS).as_posix(): sha256_file(path)
        for path in LIVE_DECISIONS.rglob("*")
        if path.is_file()
    }
    copied = tmp_path / "decisions"
    copy_pre_c2_decisions_fixture(copied)
    store = DetectionGoldPilotPersistence(
        manifest=load_manifest(PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=copied,
        reviewer_session_id=REVIEWER,
    )
    initial = store.state()
    assert read_json(LIVE_DECISIONS / "review_decisions.json")["event_sequence"] == 57
    assert initial["event_sequence"] == 44
    assert set(initial["tranche_completions"]) == {"A_CORE_STATIC", "B_REMAINING_STATIC", "C1_DENSE_OVERLAP"}
    prior_hashes = {case_id: initial["annotation_hashes"][case_id] for case_id in initial["annotations"]}
    for case_id in C2_CASE_IDS:
        case = store.case_map()[case_id]
        save_case(store, case, synthetic_annotation(case))
    before_completion = store.state()
    completion_id = str(uuid.uuid4())
    payload = {
        "review_id": store.manifest.review_id,
        "reviewer_session_id": REVIEWER,
        "tranche_id": C2,
        "client_event_id": completion_id,
        "idempotency_key": completion_id,
        "expected_server_state_hash": before_completion["server_state_hash"],
        "pending_outbox_events": 0,
        "evidence_blocker_count": 0,
        "unresolved_draft_count": 0,
        "unresolved_divergence": False,
    }
    completed = store.complete_tranche(payload)
    assert completed["event_sequence"] == 57
    assert completed["completed"] is False
    assert set(completed["tranche_completions"]) == {
        "A_CORE_STATIC",
        "B_REMAINING_STATIC",
        "C1_DENSE_OVERLAP",
        C2,
    }
    assert "D_TEMPORAL_PLAYER" not in completed["tranche_completions"]
    assert "E_FOOTBALL" not in completed["tranche_completions"]
    assert validate_completion_bundle(copied / "completed_tranches" / C2)["passed"] is True
    retried = store.complete_tranche(payload)
    assert retried["event_sequence"] == 57
    assert retried["completion_ack"]["idempotent_retry"] is True
    assert {case_id: retried["annotation_hashes"][case_id] for case_id in prior_hashes} == prior_hashes
    after_hashes = {
        path.relative_to(LIVE_DECISIONS).as_posix(): sha256_file(path)
        for path in LIVE_DECISIONS.rglob("*")
        if path.is_file()
    }
    assert after_hashes == source_hashes


def test_c2_ui_copy_and_invalidation_contract() -> None:
    app = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_app.js").read_text(encoding="utf-8")
    wizard = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_wizard.js").read_text(
        encoding="utf-8"
    )
    styles = (REPO / "src/football_intelligence/review_chassis/static/styles.css").read_text(encoding="utf-8")
    for text in (
        "A substitute or warming-up footballer is still a Player.",
        "Focus person + nearest pitch boundary",
        "Role and inside/outside pitch state must not change this answer.",
        "Where are this person's feet relative to the approved pitch boundary?",
    ):
        assert text in wizard
    assert "dgPitchToleranceBand" in app
    assert "pointer-events: none" in styles
    assert "objectSemanticChanged" in wizard
    assert "candidateRelevant" in wizard


def test_browser_acceptance_and_review_pack_when_finalized() -> None:
    browser = read_json(STAGE / "04_BROWSER_PERSISTENCE_AND_COMPLETION" / "browser_persistence_results.json")
    summary = read_json(STAGE / "07_COMMANDS_AND_TESTS" / "build_summary.json")
    if browser.get("status") == "PENDING_REAL_BROWSER_ACCEPTANCE" or summary.get("tests_pending") is True:
        pytest.skip("R3-R4 final browser and pack acceptance has not run yet")
    assert browser["passed"] is True
    assert len(browser["required_scenarios"]) == 24
    assert all(browser["required_scenarios"].values())
    assert summary["classification"] == "PASS_C2_PITCH_BOUNDARY_GOLD_READY_FOR_HUMAN_ANNOTATION"
    files = [path for path in (STAGE / "08_REVIEW_PACK_FOR_CHATGPT").iterdir() if path.is_file()]
    assert len(files) <= 20
    assert (STAGE / "08_REVIEW_PACK_FOR_CHATGPT" / "04_SOURCE_DIFF.patch").stat().st_size > 0
