from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

import pytest

from football_intelligence.detection_gold.incremental import (
    R3_WIZARD_SCHEMA,
    authoritative_candidate_binding_hash,
    authoritative_candidate_uuids,
    authoritative_frame_record,
    cross_frame_candidate_exclusions,
    tranche_for_case,
)
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.validation import validate_review_chassis_package

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1]
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
PACKAGE = STAGE / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
R2 = PART3 / "M5_5G1A_R2_NOVICE_GUIDED_ANNOTATION_WIZARD_AND_USABILITY_OVERHAUL_v1"
R2_PACKAGE = R2 / "05_NOVICE_GUIDED_DETECTION_GOLD_PILOT_PACKAGE"
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer_r3"
CASE_HASH = "986604e34e6f831825dfb76601f854ece083e3ba9001f97bad26d6d596e6a401"
EVIDENCE_HASH = "58c2de8da4e6e8d2160a29550e8030e7c5225845fc58be44eae7651c0b4a1ab4"
R2_HASHES = {
    "detection_gold_recovery_materialization.json": "482e2e44ae63003f35209c4c8dc52e47972570bb8b8bc500c6de70ee67b95022",
    "review_decisions.json": "10e5a87847e42ec96f3fe7ba40927395cad34779dda48b1a9acadd90e0b2a266",
    "review_decision_events.jsonl": "e77911340f624b4e0d7cf0d8fbb7f3ff271b9cd1a1d2a9305dfd210c262ab101",
}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def wizard_state(case: object, *, footpoint_reviews: dict | None = None) -> dict:
    record = (
        authoritative_frame_record(case)
        if case.task_type
        in {
            "detection_gold_player_static",
            "detection_gold_dense_region",
        }
        else None
    )
    candidate_uuids = authoritative_candidate_uuids(case) if record else []
    return {
        "schema_version": R3_WIZARD_SCHEMA,
        "case_id": case.case_id,
        "step": 4,
        "drawing_complete": True,
        "current_object_uuid": None,
        "question_index": 0,
        "completed_object_uuids": list(footpoint_reviews or {}),
        "footpoint_placed_uuids": list(footpoint_reviews or {}),
        "footpoint_reviews": footpoint_reviews or {},
        "pending_footpoint_decision": None,
        "candidate_index": max(0, len(candidate_uuids) - 1),
        "candidate_phase": "relation",
        "candidate_relation": None,
        "candidate_targets": [],
        "candidate_answered_uuids": candidate_uuids,
        "frame_answered_sequences": [],
        "frame_phase": "visibility",
        "desired_frame_state": None,
        "pitch_footpoint_set": False,
        "pitch_question_index": 0,
        "pitch_answers": [],
        "football_candidate_answers": {},
        "failure_reviewed": True,
        "help_opened": False,
        "active_tranche_id": tranche_for_case(
            load_ui_config(PACKAGE / "ui_config.json").question_contract, case.case_id
        ),
        "authoritative_frame_sequence": int(record["frame_sequence"]) if record else None,
        "authoritative_source_frame_sha256": str(record["source_frame_sha256"]) if record else None,
        "primary_canvas_frame_sequence": int(record["frame_sequence"]) if record else None,
        "primary_canvas_source_frame_sha256": str(record["source_frame_sha256"]) if record else None,
        "candidate_queue_binding_hash": authoritative_candidate_binding_hash(case) if record else None,
    }


def static_annotation(case: object, *, person: dict | None = None) -> dict:
    people = [copy.deepcopy(person)] if person else []
    return {
        "schema_version": "m5_5g1a_detection_gold_v1",
        "source_binding": copy.deepcopy(case.visible_metadata["source_binding"]),
        "visible_person_count": len(people),
        "player_instances": people,
        "candidate_relations": [
            {"candidate_uuid": candidate_uuid, "relation": "BACKGROUND", "annotation_uuids": []}
            for candidate_uuid in authoritative_candidate_uuids(case)
        ],
        "earliest_failure_stage": "UNRESOLVED",
        "note": "",
    }


def save_payload(persistence: DetectionGoldPilotPersistence, case: object, annotation: dict, wizard: dict) -> dict:
    event_id = str(uuid.uuid4())
    return {
        "event_type": "DETECTION_CASE_SAVED",
        "review_id": persistence.manifest.review_id,
        "reviewer_session_id": REVIEWER,
        "case_id": case.case_id,
        "annotation": annotation,
        "wizard_state": wizard,
        "client_event_id": event_id,
        "idempotency_key": event_id,
        "expected_server_state_hash": persistence.state()["server_state_hash"],
        "elapsed_active_seconds": 2,
    }


def persistence(tmp_path: Path) -> DetectionGoldPilotPersistence:
    return DetectionGoldPilotPersistence(
        manifest=load_manifest(PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=tmp_path / "decisions",
        reviewer_session_id=REVIEWER,
    )


def test_r3_package_preserves_frozen_payload_and_evidence() -> None:
    manifest = read_json(PACKAGE / "reviewer_manifest.json")
    validation = read_json(PACKAGE / "review_package_validation.json")
    assert len(manifest["cases"]) == 88
    assert stable_hash(manifest["cases"]) == CASE_HASH
    assert validation["evidence_copy"]["file_count"] == 1512
    assert validation["evidence_copy"]["tree_hash"] == EVIDENCE_HASH
    assert validation["evidence_bytes_identical"] is True
    assert validation["passed"] is True
    assert (
        validate_review_chassis_package(
            manifest_path=PACKAGE / "reviewer_manifest.json",
            ui_config_path=PACKAGE / "ui_config.json",
            evidence_root=PACKAGE / "evidence",
            decisions_root=PACKAGE / "decisions",
        )["passed"]
        is True
    )


def test_r2_human_work_remains_byte_identical() -> None:
    for name, expected in R2_HASHES.items():
        assert sha256_file(R2_PACKAGE / "decisions" / name) == expected
    state = read_json(R2_PACKAGE / "decisions" / "review_decisions.json")
    assert len(state["annotations"]) == 6
    assert state["event_sequence"] == 6


def test_cases_006_and_007_resolve_only_the_authoritative_middle_frame() -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    for ordinal, expected_excluded in ((6, 10), (7, 9)):
        case = next(
            row
            for row in manifest.cases
            if row.task_type == "detection_gold_player_static"
            and int(row.visible_metadata["module_case_number"]) == ordinal
        )
        record = authoritative_frame_record(case)
        authoritative = set(authoritative_candidate_uuids(case))
        excluded = {row["candidate_uuid"] for row in cross_frame_candidate_exclusions(case)}
        assert int(record["frame_sequence"]) == int(case.source_frame_sequence)
        assert record["source_frame_sha256"] == case.visible_metadata["source_binding"]["source_frame_sha256"]
        assert authoritative.isdisjoint(excluded)
        assert len(excluded) == expected_excluded


def test_server_rejects_cross_frame_candidates_and_canvas_drift(tmp_path: Path) -> None:
    store = persistence(tmp_path)
    case = next(
        row
        for row in store.manifest.cases
        if row.task_type == "detection_gold_player_static" and int(row.visible_metadata["module_case_number"]) == 6
    )
    annotation = static_annotation(case)
    excluded = cross_frame_candidate_exclusions(case)[0]["candidate_uuid"]
    annotation["candidate_relations"][0]["candidate_uuid"] = excluded
    with pytest.raises(ValueError, match="candidate relation coverage mismatch"):
        store.save_detection_event(save_payload(store, case, annotation, wizard_state(case)))

    annotation = static_annotation(case)
    wizard = wizard_state(case)
    wizard["primary_canvas_frame_sequence"] += 1
    with pytest.raises(ValueError, match="non-authoritative static canvas"):
        store.save_detection_event(save_payload(store, case, annotation, wizard))


def test_hidden_feet_estimate_cannot_reuse_upper_body_box_bottom(tmp_path: Path) -> None:
    store = persistence(tmp_path)
    case = next(
        row
        for row in store.manifest.cases
        if row.task_type == "detection_gold_player_static"
        and row.visible_metadata["pilot_stratum"] == "partial_or_occluded"
    )
    record = authoritative_frame_record(case)
    candidate = next(row for row in record["candidates"] if row["class_name"] == "person")
    box = copy.deepcopy(candidate["bbox_original_pixels"])
    person_id = "person-partial-fixture"
    person = {
        "annotation_uuid": person_id,
        "visible_body_box": box,
        "footpoint": {"x": (box["x1"] + box["x2"]) / 2, "y": box["y2"]},
        "footpoint_uncertainty_pixels": 20,
        "visibility_state": "HEAVILY_OCCLUDED",
        "occlusion_fraction": 0.75,
        "occlusion_type": "PERSON",
        "truncation_flags": [],
        "minimum_visible_dimensions": {"width_pixels": box["x2"] - box["x1"], "height_pixels": box["y2"] - box["y1"]},
        "ambiguity_ignore": False,
        "pitch_state": "BOUNDARY_UNCERTAIN",
        "coarse_role": "UNKNOWN",
    }
    review = {person_id: {"decision": "FEET_NOT_VISIBLE", "estimated": True, "adjusted": False}}
    with pytest.raises(ValueError, match="upper-body visible-box bottom"):
        store.save_detection_event(
            save_payload(
                store, case, static_annotation(case, person=person), wizard_state(case, footpoint_reviews=review)
            )
        )

    person["footpoint"]["y"] = min(
        case.visible_metadata["source_binding"]["image_height"],
        box["y2"] + max(4, (box["y2"] - box["y1"]) * 0.35),
    )
    if abs(person["footpoint"]["y"] - box["y2"]) < 0.5:
        person["footpoint"]["y"] = box["y2"] - 4
    saved = store.save_detection_event(
        save_payload(store, case, static_annotation(case, person=person), wizard_state(case, footpoint_reviews=review))
    )
    assert saved["annotations"][case.case_id]["player_instances"][0]["footpoint_uncertainty_pixels"] >= 20


def test_tranche_a_is_exact_and_completes_atomically_without_completing_pilot(tmp_path: Path) -> None:
    store = persistence(tmp_path)
    tranches = store.ui_config.question_contract["gold_tranches"]
    tranche_a = tranches["A_CORE_STATIC"]
    assert len(tranche_a["case_ids"]) == 18
    strata = [store.case_map()[case_id].visible_metadata["pilot_stratum"] for case_id in tranche_a["case_ids"]]
    assert {name: strata.count(name) for name in set(strata)} == {
        "clean_control": 3,
        "duplicate": 3,
        "merged": 3,
        "missed": 3,
        "partial_or_occluded": 3,
        "small_far_side": 3,
    }
    for case_id in tranche_a["case_ids"]:
        case = store.case_map()[case_id]
        store.save_detection_event(save_payload(store, case, static_annotation(case), wizard_state(case)))
    before = store.state()
    event_id = str(uuid.uuid4())
    completed = store.complete_tranche(
        {
            "tranche_id": "A_CORE_STATIC",
            "client_event_id": event_id,
            "idempotency_key": event_id,
            "expected_server_state_hash": before["server_state_hash"],
            "pending_outbox_events": 0,
            "evidence_blocker_count": 0,
            "unresolved_draft_count": 0,
            "unresolved_divergence": False,
        }
    )
    assert completed["tranche_completions"]["A_CORE_STATIC"]
    assert completed["completed"] is False
    bundle = store.decisions_root / "completed_tranches" / "A_CORE_STATIC"
    assert validate_completion_bundle(bundle)["passed"] is True
    assert len(list(bundle.glob("completed_review*"))) == 4
    with pytest.raises(ValueError, match="completed tranches are immutable"):
        case = store.case_map()[tranche_a["case_ids"][0]]
        store.save_detection_event(save_payload(store, case, static_annotation(case), wizard_state(case)))
    with pytest.raises(ValueError, match="completion is blocked"):
        store.complete_detection({"expected_server_state_hash": completed["server_state_hash"]})


def test_r3_client_contains_lock_footpoint_and_incremental_persistence_controls() -> None:
    app = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_app.js").read_text()
    wizard = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_wizard.js").read_text()
    html = (REPO / "src/football_intelligence/review_chassis/static/index.html").read_text()
    assert "authoritativeFrameIndex" in app
    assert "Reference frames are view-only; the middle frame remains locked." in app
    assert 'database.createObjectStore("session"' in app
    assert "/api/review/detection-gold-tranche-complete" in app
    assert "Is this roughly where their feet touch the ground?" in wizard
    assert "FEET_NOT_VISIBLE" in wizard and "CANNOT_TELL" in wizard
    assert "Estimated because the feet are not visible" in wizard
    assert "Box only the part you can actually see. Do not guess the hidden body." in wizard
    assert "Label the middle frame only. Nearby frames are reference images." in html
    assert 'id="dgCompleteTranche"' in html


def test_real_browser_acceptance_is_recorded_when_finalized() -> None:
    result = read_json(STAGE / "05_BROWSER_PERSISTENCE_AND_REGRESSION" / "browser_persistence_results.json")
    if result["status"] == "PENDING_REAL_BROWSER_ACCEPTANCE":
        pytest.skip("real-browser acceptance has not run yet")
    assert result["passed"] is True
    assert all(result["required_browser_scenarios"].values())
    assert len(result["visual_regression"]) == 6


def test_final_classification_and_review_pack_when_finalized() -> None:
    summary = read_json(STAGE / "08_COMMANDS_AND_TESTS" / "build_summary.json")
    if summary.get("tests_pending") is True:
        pytest.skip("final stage validation has not run yet")
    assert summary["classification"] == "PASS_INCREMENTAL_DETECTION_GOLD_TRANCHE_A_READY"
    assert summary["detector_or_tracker_evaluated"] is False
    assert summary["detector_or_tracker_promoted"] is False
    pack = STAGE / "09_REVIEW_PACK_FOR_CHATGPT"
    files = [path for path in pack.iterdir() if path.is_file()]
    assert len(files) <= 20
    assert not [path for path in pack.rglob("*") if path.is_file() and path.parent != pack]
    assert (pack / "04_SOURCE_DIFF.patch").stat().st_size > 0
