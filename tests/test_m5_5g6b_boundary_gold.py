from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from football_intelligence.detection_gold.incremental import (
    G6B_BOUNDARY_FOCUSED_CLIENT_BUILD_ID,
    authoritative_candidate_binding_hash,
    authoritative_candidate_uuids,
    authoritative_frame_record,
)
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.hashing import sha256_file
from football_intelligence.review_chassis.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1]
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G6B_BOUNDARY_FOCUSED_GOLD_AND_FROZEN_PROPOSAL_SUPPLY_ATTRIBUTION_v1"
PACKAGE = STAGE / "05_PERSON_CENTRIC_BOUNDARY_REVIEW_PACKAGE"
TRANCHE = "B1_BOUNDARY_FOCUSED_PERSON_GOLD"
REVIEWER = "m5_5g6b_boundary_focused_gold_reviewer"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def synthetic_annotation(case: Any, *, index: int = 0) -> dict[str, Any]:
    record = authoritative_frame_record(case)
    candidate_uuid = authoritative_candidate_uuids(case)[0]
    candidate = next(row for row in record["candidates"] if row["diagnostic_uuid"] == candidate_uuid)
    box = copy.deepcopy(candidate["bbox_original_pixels"])
    person_uuid = str(case.visible_metadata["target_annotation_uuid"])
    states = ("ON_PITCH", "OFF_PITCH", "BOUNDARY_UNCERTAIN")
    pitch_state = states[index % len(states)]
    hidden = pitch_state == "BOUNDARY_UNCERTAIN"
    person = {
        "annotation_uuid": person_uuid,
        "visible_body_box": box,
        "footpoint": None if hidden else {"x": (box["x1"] + box["x2"]) / 2, "y": box["y2"]},
        "footpoint_status": "FEET_NOT_VISIBLE" if hidden else "OBSERVED_CLEAR",
        "footpoint_uncertainty_pixels": 24 if hidden else 3,
        "pitch_state": pitch_state,
        "pitch_state_certainty": "UNCERTAIN" if hidden else "CLEAR",
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
                "relation": "CLEAN_SINGLE_INSTANCE",
                "annotation_uuids": [person_uuid],
            }
        ],
        "note": "",
    }


def synthetic_wizard(case: Any, annotation: dict[str, Any]) -> dict[str, Any]:
    record = authoritative_frame_record(case)
    candidate_uuid = authoritative_candidate_uuids(case)[0]
    person = annotation["player_instances"][0]
    person_uuid = person["annotation_uuid"]
    return {
        "schema_version": "football_intelligence.m5_5g1a_r3.wizard_state.v1",
        "case_id": case.case_id,
        "step": 4,
        "drawing_complete": True,
        "current_object_uuid": None,
        "question_index": 0,
        "completed_object_uuids": [person_uuid],
        "footpoint_placed_uuids": [] if person.get("footpoint") is None else [person_uuid],
        "footpoint_reviews": {
            person_uuid: {
                "status": person["footpoint_status"],
                "confirmed": True,
                "coarse_role": person["coarse_role"],
                "pitch_state": person["pitch_state"],
                "pitch_state_certainty": person["pitch_state_certainty"],
            }
        },
        "pending_footpoint_decision": None,
        "candidate_index": 0,
        "candidate_phase": "relation",
        "candidate_relation": None,
        "candidate_targets": [],
        "candidate_answered_uuids": [candidate_uuid],
        "candidate_answer_records": {
            candidate_uuid: {
                "candidate_uuid": candidate_uuid,
                "relation": "CLEAN_SINGLE_INSTANCE",
                "annotation_uuids": [person_uuid],
                "answered_against_human_truth_revision": 0,
                "answered_person_question_revision": 0,
                "candidate_answer_revision": 1,
                "validity": "VALID",
                "invalidation_reason": None,
                "answered_at": "2026-07-26T00:00:00Z",
                "revalidated_at": None,
                "revalidation_event": "INITIAL_REVIEW",
            }
        },
        "mask_front_answers": {},
        "human_truth_revision": 0,
        "person_question_revision": 0,
        "candidate_answer_revision": 1,
        "summary_revision": 1,
        "person_question_completion_revisions": {person_uuid: 0},
        "summary_validity": "VALID",
        "summary_human_truth_revision": 0,
        "invalidation_notice": None,
        "frame_answered_sequences": [],
        "frame_phase": "visibility",
        "desired_frame_state": None,
        "pitch_footpoint_set": False,
        "pitch_question_index": 0,
        "boundary_pitch_question_index": 2,
        "pitch_answers": [],
        "football_candidate_answers": {},
        "failure_reviewed": True,
        "help_opened": False,
        "active_tranche_id": TRANCHE,
        "authoritative_frame_sequence": int(record["frame_sequence"]),
        "authoritative_source_frame_sha256": record["source_frame_sha256"],
        "primary_canvas_frame_sequence": int(record["frame_sequence"]),
        "primary_canvas_source_frame_sha256": record["source_frame_sha256"],
        "candidate_queue_binding_hash": authoritative_candidate_binding_hash(case),
    }


def save_case(store: DetectionGoldPilotPersistence, case: Any, annotation: dict[str, Any]) -> None:
    event_id = str(uuid.uuid4())
    store.save_detection_event(
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
            "elapsed_active_seconds": 4,
        }
    )


def test_selection_is_frozen_balanced_and_source_distinct() -> None:
    spec = read_json(STAGE / "03_BOUNDARY_CANDIDATE_MINING" / "boundary_selection_specification.json")
    selection = read_json(STAGE / "04_BOUNDARY_CASE_MANIFEST" / "boundary_case_selection_validation.json")
    package = read_json(PACKAGE / "review_package_validation.json")
    assert spec["frozen_before_candidate_scoring"] is True
    assert spec["gold_pitch_labels_used_for_selection"] is False
    assert selection["passed"] is True
    assert selection["selected_case_count"] == 18
    assert selection["selected_source_group_count"] == 18
    assert selection["selected_stratum_counts"] == {
        "disagreement_hidden_feet_or_straddling": 6,
        "estimated_inside_near_boundary": 6,
        "estimated_outside_near_boundary": 6,
    }
    assert package["passed"] is True
    assert package["browser_manifest_has_selection_strata"] is False


def test_nine_missing_people_have_frozen_stage_attribution() -> None:
    missing = read_json(STAGE / "02_C2_MISSING_SUPPLY_ATTRIBUTION" / "c2_missing_on_pitch_people.json")
    gap = read_json(STAGE / "02_C2_MISSING_SUPPLY_ATTRIBUTION" / "proposal_supply_gap_summary.json")
    rows = [
        json.loads(line)
        for line in (STAGE / "02_C2_MISSING_SUPPLY_ATTRIBUTION" / "frozen_proposal_supply_attribution.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert missing["count"] == 9
    assert gap["passed"] is True
    assert len(rows) == 9
    assert sum(gap["origin_counts"].values()) == 9
    assert all(row["source_stage_candidate_counts"] for row in rows)
    assert all(row["new_inference_performed"] is False for row in rows)
    assert all(row["weak_overlap_upgraded_to_clean_supply"] is False for row in rows)


def test_browser_package_is_target_only_and_fresh() -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    ui = load_ui_config(PACKAGE / "ui_config.json")
    state = read_json(PACKAGE / "decisions" / "review_decisions.json")
    assert len(manifest.cases) == 18
    assert ui.question_contract["client_build_id"] == G6B_BOUNDARY_FOCUSED_CLIENT_BUILD_ID
    assert ui.question_contract["boundary_focused_person_gold"] is True
    assert ui.question_contract["revision_aware_wizard_state"] is True
    assert "No prior decisions or browser drafts were imported" in ui.question_contract["first_load_notice"]
    assert ui.question_contract["gold_tranches"][TRANCHE]["case_ids"] == [case.case_id for case in manifest.cases]
    assert all(len(authoritative_candidate_uuids(case)) == 1 for case in manifest.cases)
    assert all(case.visible_metadata["target_only_review"] is True for case in manifest.cases)
    assert "selection_stratum" not in json.dumps(read_json(PACKAGE / "reviewer_manifest.json"))
    assert state["annotations"] == {}
    assert state["event_sequence"] == 0
    assert not (PACKAGE / "decisions" / "completed_tranches" / TRANCHE).exists()

    raw_manifest = read_json(PACKAGE / "reviewer_manifest.json")
    for case in raw_manifest["cases"]:
        assets = {row["asset_id"]: row for row in case["evidence_assets"]}
        for frame in case["visible_metadata"]["frame_records"]:
            for kind in ("panorama", "focal", "contact"):
                asset = assets[frame[f"{kind}_asset_id"]]
                assert frame[f"{kind}_asset_path"] == asset["relative_path"]
                assert frame[f"{kind}_asset_sha256"] == asset["sha256"]
                assert (PACKAGE / "evidence" / case["case_id"] / asset["relative_path"]).is_file()


def test_server_rejects_more_than_one_target(tmp_path: Path) -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    store = DetectionGoldPilotPersistence(
        manifest=manifest,
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=tmp_path / "decisions",
        reviewer_session_id=REVIEWER,
    )
    store.ensure_state()
    case = manifest.cases[0]
    annotation = synthetic_annotation(case)
    extra = copy.deepcopy(annotation["player_instances"][0])
    extra["annotation_uuid"] = "extra-context-person"
    annotation["player_instances"].append(extra)
    annotation["visible_person_count"] = 2
    with pytest.raises(ValueError, match="exactly one highlighted target person"):
        save_case(store, case, annotation)


def test_atomic_boundary_completion_uses_temporary_root(tmp_path: Path) -> None:
    production_hashes = {
        path.relative_to(PACKAGE / "decisions").as_posix(): sha256_file(path)
        for path in (PACKAGE / "decisions").rglob("*")
        if path.is_file()
    }
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    ui = load_ui_config(PACKAGE / "ui_config.json")
    store = DetectionGoldPilotPersistence(
        manifest=manifest,
        ui_config=ui,
        decisions_root=tmp_path / "decisions",
        reviewer_session_id=REVIEWER,
    )
    store.ensure_state()
    for index, case in enumerate(manifest.cases):
        save_case(store, case, synthetic_annotation(case, index=index))
    before = store.state()
    completion_id = str(uuid.uuid4())
    payload = {
        "review_id": manifest.review_id,
        "reviewer_session_id": REVIEWER,
        "tranche_id": TRANCHE,
        "client_event_id": completion_id,
        "idempotency_key": completion_id,
        "expected_server_state_hash": before["server_state_hash"],
        "pending_outbox_events": 0,
        "evidence_blocker_count": 0,
        "unresolved_draft_count": 0,
        "unresolved_divergence": False,
    }
    completed = store.complete_tranche(payload)
    assert completed["event_sequence"] == 19
    assert completed["completed"] is False
    bundle = tmp_path / "decisions" / "completed_tranches" / TRANCHE
    assert validate_completion_bundle(bundle)["passed"] is True
    summary = read_json(bundle / "completed_review_summary.json")
    boundary = summary["boundary_completion"]
    assert boundary["exact_target_count"] == 18
    assert boundary["source_group_count"] == 18
    assert boundary["human_pitch_state_counts"] == {
        "BOUNDARY_UNCERTAIN": 6,
        "OFF_PITCH": 6,
        "ON_PITCH": 6,
    }
    assert boundary["footpoint_visibility_counts"] == {"FEET_NOT_VISIBLE": 6, "OBSERVED_CLEAR": 12}
    assert boundary["prior_gold_unchanged"] is True
    retried = store.complete_tranche(payload)
    assert retried["completion_ack"]["idempotent_retry"] is True
    after_hashes = {
        path.relative_to(PACKAGE / "decisions").as_posix(): sha256_file(path)
        for path in (PACKAGE / "decisions").rglob("*")
        if path.is_file()
    }
    assert after_hashes == production_hashes


def test_ui_contains_required_target_copy_and_controls() -> None:
    app = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_app.js").read_text(encoding="utf-8")
    wizard = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_wizard.js").read_text(
        encoding="utf-8"
    )
    styles = (REPO / "src/football_intelligence/review_chassis/static/styles.css").read_text(encoding="utf-8")
    copy_text = "Label the highlighted target person only. Other people are context."
    assert copy_text in app
    assert copy_text in wizard
    assert "Confirm target box" in wizard
    assert "redrawSelectedPerson" in wizard
    assert "beginRedrawSelectedVisible" in app
    assert 'label.textContent = boundaryFocusedPerson() ? "TARGET"' in app
    assert ".dgBoundaryTarget" in styles
