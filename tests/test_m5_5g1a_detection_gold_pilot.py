from __future__ import annotations

import copy
import json
import re
import uuid
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from football_intelligence.detection_gold.matching import (
    evaluate_detection_gold,
    evaluate_player_observations,
    one_to_one_match,
)
from football_intelligence.detection_gold.models import (
    CandidateRelationAnnotation,
    FootballFrameState,
    PlayerStaticAnnotation,
    TemporalFrameState,
    validate_case_annotation,
)
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.hashing import sha256_file
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.validation import validate_review_chassis_package


ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1]
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G1A_DETECTION_GOLD_FOUNDATION_ONTOLOGY_FREEZE_AND_PILOT_ANNOTATION_v1"
PACKAGE = STAGE / "10_DETECTION_GOLD_PILOT_ANNOTATION_PACKAGE"
G0_STAGE = PART2 / "M5_5G0_PLAYER_BALL_DETECTION_FORENSIC_PROVENANCE_AND_PRO_RESEARCH_HANDOFF_v1"
G0_PACK = G0_STAGE / "13_PRO_CONTEXT_PACK_FOR_CHATGPT_PRO"
RESEALED = STAGE / "01_G0_PRO_PACK_RESEALED" / "resealed_pack"
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer"
PRO_SHA256 = "7f93cb7282a2a35ecee8d6f81d888b0fd7358b06d85520a916eefbc6c94ef1d1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_annotation(case: object) -> dict:
    metadata = case.visible_metadata
    source_binding = copy.deepcopy(metadata["source_binding"])
    task_type = case.task_type
    if task_type == "detection_gold_player_static":
        return {
            "schema_version": "m5_5g1a_detection_gold_v1",
            "source_binding": source_binding,
            "visible_person_count": 0,
            "player_instances": [],
            "candidate_relations": [
                {"candidate_uuid": value, "relation": "BACKGROUND", "annotation_uuids": []}
                for value in metadata["candidate_uuids"]
            ],
            "earliest_failure_stage": "UNRESOLVED",
            "note": "",
        }
    if task_type == "detection_gold_dense_region":
        return {
            "schema_version": "m5_5g1a_detection_gold_v1",
            "source_binding": source_binding,
            "dense_region_uuid": f"dense-{case.case_id}",
            "trigger_reason": metadata["pilot_stratum"],
            "human_visible_person_count": 0,
            "visible_masks": [],
            "candidate_relations": [
                {"candidate_uuid": value, "relation": "BACKGROUND", "annotation_uuids": []}
                for value in metadata["candidate_uuids"]
            ],
            "uncertain_or_ignore": True,
            "reviewer_agreement": "NOT_REVIEWED",
            "adjudication_state": "NOT_REQUIRED",
            "note": "",
        }
    if task_type == "detection_gold_temporal_player":
        return {
            "schema_version": "m5_5g1a_detection_gold_v1",
            "source_binding": source_binding,
            "frames": [
                {
                    "frame_sequence": row["frame_sequence"],
                    "source_frame_sha256": row["source_frame_sha256"],
                    "state": "NOT_VISIBLE",
                    "current_frame_pixel_support": False,
                    "candidate_uuids": [],
                }
                for row in metadata["frame_records"]
            ],
            "contact_strip_reviewed": True,
            "stable_run_accepted": False,
            "note": "",
        }
    if task_type == "detection_gold_pitch_boundary":
        return {
            "schema_version": "m5_5g1a_detection_gold_v1",
            "source_binding": source_binding,
            "footpoint": metadata["machine_footpoint"],
            "footpoint_uncertainty_pixels": 5,
            "pitch_state": "BOUNDARY_UNCERTAIN",
            "coarse_role": "UNKNOWN",
            "primary_on_pitch_supply_eligible": False,
            "note": "",
        }
    return {
        "schema_version": "m5_5g1a_detection_gold_v1",
        "source_binding": source_binding,
        "frames": [
            {
                "frame_sequence": row["frame_sequence"],
                "source_frame_sha256": row["source_frame_sha256"],
                "state": "NOT_VISIBLE",
            }
            for row in metadata["frame_records"]
        ],
        "full_contact_strip_reviewed": True,
        "note": "",
    }


def persistence_fixture(tmp_path: Path, task_type: str = "detection_gold_pitch_boundary") -> tuple:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    case = next(item for item in manifest.cases if item.task_type == task_type)
    fixture_manifest = manifest.model_copy(
        update={"review_id": f"fixture-{task_type}", "stage_id": "fixture", "cases": [case]}
    )
    persistence = DetectionGoldPilotPersistence(
        manifest=fixture_manifest,
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=tmp_path / "decisions",
        reviewer_session_id=REVIEWER,
    )
    persistence.ensure_state()
    return persistence, case


def save_payload(persistence: DetectionGoldPilotPersistence, case: object, annotation: dict) -> dict:
    event_id = str(uuid.uuid4())
    return {
        "event_type": "DETECTION_CASE_SAVED",
        "review_id": persistence.manifest.review_id,
        "reviewer_session_id": REVIEWER,
        "case_id": case.case_id,
        "annotation": annotation,
        "client_event_id": event_id,
        "idempotency_key": event_id,
        "expected_server_state_hash": persistence.state()["server_state_hash"],
        "elapsed_active_seconds": 2,
    }


def test_g0_pack_reseal_uses_original_local_bytes() -> None:
    validation = read_json(STAGE / "01_G0_PRO_PACK_RESEALED" / "local_g0_pack_validation.json")
    manifest = read_json(STAGE / "01_G0_PRO_PACK_RESEALED" / "resealed_manifest.json")
    assert validation["passed"] is True
    assert validation["actual_file_count"] == 20
    assert validation["source_diff_present_nonempty"] is True
    assert manifest["original_nonmanifest_files_copied_byte_for_byte"] is True
    for filename in (
        "04_SOURCE_DIFF.patch",
        "17_PLAYER_FAILURE_ATLAS.jpg",
        "18_PRE_POST_NMS_AND_SCALE_ATLAS.jpg",
        "19_BALL_AND_OFF_PITCH_ATLAS.jpg",
    ):
        assert sha256_file(G0_PACK / filename) == sha256_file(RESEALED / filename)


def test_pro_decision_is_complete_and_hash_bound() -> None:
    index = read_json(STAGE / "02_PRO_DECISION_INGESTION" / "pro_decision_hash_and_index.json")
    assert index["sha256"] == PRO_SHA256
    assert index["section_count"] == 28
    assert index["final_next_stage_choice"] == "TARGETED DETECTION GOLD FIRST"


def test_frozen_schemas_forbid_identity_and_separate_observed_from_predicted() -> None:
    case = next(
        item
        for item in load_manifest(PACKAGE / "reviewer_manifest.json").cases
        if item.task_type == "detection_gold_player_static"
    )
    payload = valid_annotation(case)
    payload["persistent_player_identity"] = "forbidden"
    with pytest.raises(ValidationError):
        PlayerStaticAnnotation.model_validate(payload)
    with pytest.raises(ValidationError):
        TemporalFrameState.model_validate(
            {
                "frame_sequence": 1,
                "source_frame_sha256": "a" * 64,
                "state": "OCCLUDED_PREDICTED",
                "current_frame_pixel_support": True,
            }
        )
    with pytest.raises(ValidationError):
        FootballFrameState.model_validate(
            {
                "frame_sequence": 1,
                "source_frame_sha256": "b" * 64,
                "state": "NOT_VISIBLE",
                "centre_point": {"x": 10, "y": 10},
            }
        )


def test_candidate_relation_cardinality_and_dense_mask_coverage() -> None:
    with pytest.raises(ValidationError):
        CandidateRelationAnnotation.model_validate(
            {"candidate_uuid": "candidate", "relation": "MERGED_MULTIPLE_INSTANCES", "annotation_uuids": ["one"]}
        )
    relation = CandidateRelationAnnotation.model_validate(
        {
            "candidate_uuid": "candidate",
            "relation": "MERGED_MULTIPLE_INSTANCES",
            "annotation_uuids": ["one", "two"],
            "candidate_visible_mask_coverage": 0.75,
        }
    )
    assert relation.candidate_visible_mask_coverage == 0.75


def test_one_to_one_and_separate_error_metrics() -> None:
    box = {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 20.0}
    assert len(one_to_one_match([box], [box, box])) == 1
    metrics = evaluate_player_observations(
        gold_instances=[{"visible_body_box": box, "footpoint": {"x": 5, "y": 20}, "pitch_state": "ON_PITCH"}],
        pre_consolidation_observations=[{"visible_body_box": box}, {"visible_body_box": box}],
        final_observations=[
            {
                "visible_body_box": box,
                "footpoint": {"x": 5, "y": 20},
                "source_candidate_relation": "MERGED_MULTIPLE_INSTANCES",
                "output_relation": "CLEAN_SINGLE_INSTANCE",
                "pitch_state": "OFF_PITCH",
                "primary_on_pitch_supply_eligible": True,
                "temporal_state": "OCCLUDED_PREDICTED",
                "rendered_state": "OBSERVED",
            },
            {"visible_body_box": box, "distinct_person_suppressed": True},
        ],
    )
    assert metrics["duplicate_observation_count"] == 1
    assert metrics["merged_as_clean_count"] == 1
    assert metrics["distinct_person_suppression_count"] == 1
    assert metrics["off_pitch_false_admission_count"] == 1
    assert metrics["predicted_as_observed_count"] == 1
    frozen = evaluate_detection_gold(gold_instances=[], final_observations=[])
    assert frozen["architecture_scored_in_m5_5g1a"] is False
    assert frozen["acceptance_gate_applied"] is False


def test_exact_pilot_composition_deduplication_and_centered_ball_bursts() -> None:
    pilot = read_json(STAGE / "05_PILOT_CASE_SELECTION_AND_BINDING" / "pilot_case_manifest.json")
    dedup = read_json(STAGE / "05_PILOT_CASE_SELECTION_AND_BINDING" / "case_deduplication.json")
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    assert pilot["total_cases"] == 88
    assert pilot["counts_by_task"] == {
        "detection_gold_dense_region": 8,
        "detection_gold_football_burst": 24,
        "detection_gold_pitch_boundary": 12,
        "detection_gold_player_static": 32,
        "detection_gold_temporal_player": 12,
    }
    assert dedup["passed"] is True
    assert dedup["same_task_frame_event_duplicate_count"] == 0
    for case in (item for item in manifest.cases if item.task_type == "detection_gold_football_burst"):
        records = case.visible_metadata["frame_records"]
        assert len(records) == 9
        assert records[4]["source_frame_sha256"] == case.visible_metadata["source_binding"]["source_frame_sha256"]
        assert records[4]["frame_sequence"] == case.source_frame_sequence


def test_every_case_is_exactly_bound_and_assets_validate() -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    evidence = read_json(PACKAGE / "evidence_manifest.json")
    bindings = read_json(STAGE / "05_PILOT_CASE_SELECTION_AND_BINDING" / "case_binding_validation.json")
    assert bindings["passed"] is True
    assert len(evidence["assets"]) == 1512
    assert all(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in evidence["assets"])
    assert all(case.visible_metadata["validation_or_holdout_use_forbidden"] for case in manifest.cases)
    result = validate_review_chassis_package(
        manifest_path=PACKAGE / "reviewer_manifest.json",
        ui_config_path=PACKAGE / "ui_config.json",
        evidence_root=PACKAGE / "evidence",
        decisions_root=PACKAGE / "decisions",
    )
    assert result["passed"] is True
    assert result["hash_mismatch_count"] == 0


def test_server_rejects_partial_candidate_coverage_and_transform_drift(tmp_path: Path) -> None:
    persistence, case = persistence_fixture(tmp_path, "detection_gold_player_static")
    annotation = valid_annotation(case)
    annotation["candidate_relations"].pop()
    with pytest.raises(ValueError, match="candidate relation coverage mismatch"):
        persistence.save_detection_event(save_payload(persistence, case, annotation))
    annotation = valid_annotation(case)
    annotation["source_binding"]["panorama_transform"]["focal_to_panorama_x"] += 1
    with pytest.raises(ValueError, match="source binding mismatch"):
        persistence.save_detection_event(save_payload(persistence, case, annotation))


def test_server_rejects_wrong_frame_temporal_candidate(tmp_path: Path) -> None:
    persistence, case = persistence_fixture(tmp_path, "detection_gold_temporal_player")
    annotation = valid_annotation(case)
    first, second = case.visible_metadata["frame_records"][:2]
    wrong = second["candidates"][0]["diagnostic_uuid"]
    candidate = second["candidates"][0]
    annotation["frames"][0] = {
        "frame_sequence": first["frame_sequence"],
        "source_frame_sha256": first["source_frame_sha256"],
        "state": "OBSERVED",
        "visible_body_box": candidate["bbox_original_pixels"],
        "footpoint": {
            "x": (candidate["bbox_original_pixels"]["x1"] + candidate["bbox_original_pixels"]["x2"]) / 2,
            "y": candidate["bbox_original_pixels"]["y2"],
        },
        "current_frame_pixel_support": True,
        "candidate_uuids": [wrong],
    }
    with pytest.raises(ValueError, match="wrong-frame candidates"):
        persistence.save_detection_event(save_payload(persistence, case, annotation))


def test_persistence_is_idempotent_replayable_and_atomically_completes(tmp_path: Path) -> None:
    persistence, case = persistence_fixture(tmp_path)
    payload = save_payload(persistence, case, valid_annotation(case))
    first = persistence.save_detection_event(payload)
    duplicate = persistence.save_detection_event(payload)
    assert first["ack"]["saved_to_server"] is True
    assert duplicate["ack"]["duplicate_event"] is True
    assert len(persistence.events_path.read_text(encoding="utf-8").splitlines()) == 1
    recovery = persistence.recover_authoritative_state(write_sidecar=False)
    assert recovery["ledger_audit"]["passed"] is True
    assert recovery["ledger_audit"]["event_replay_matches_authoritative_state"] is True
    completed = persistence.complete_detection(
        {
            "expected_server_state_hash": first["ack"]["server_state_hash"],
            "client_event_id": "complete",
            "idempotency_key": "complete",
            "pending_outbox_events": 0,
            "evidence_blocker_count": 0,
            "unresolved_draft_count": 0,
            "unresolved_divergence": False,
        }
    )
    assert completed["completed"] is True
    for filename in (
        "completed_review.json",
        "completed_review_events.jsonl",
        "completed_review_manifest.json",
        "completed_review_summary.json",
    ):
        assert (persistence.decisions_root / filename).is_file()
    retried = persistence.complete_detection({"expected_server_state_hash": "stale"})
    assert retried["ack"]["duplicate_event"] is True


def test_reopen_requires_idempotency_and_is_replayable(tmp_path: Path) -> None:
    persistence, case = persistence_fixture(tmp_path)
    persistence.save_detection_event(save_payload(persistence, case, valid_annotation(case)))
    client_id = str(uuid.uuid4())
    payload = {
        "review_id": persistence.manifest.review_id,
        "reviewer_session_id": REVIEWER,
        "client_event_id": client_id,
        "idempotency_key": client_id,
        "expected_server_state_hash": persistence.state()["server_state_hash"],
        "case_id": case.case_id,
    }
    first = persistence.reopen_case(payload)
    duplicate = persistence.reopen_case(payload)
    assert first["counts"]["reviewed"] == 0
    assert duplicate["ack"]["duplicate_event"] is True
    assert persistence.recover_authoritative_state(write_sidecar=False)["ledger_audit"]["passed"] is True


def test_empty_real_root_and_temporary_persistence_are_isolated(tmp_path: Path) -> None:
    before_state = (PACKAGE / "decisions" / "review_decisions.json").read_bytes()
    before_events = (PACKAGE / "decisions" / "review_decision_events.jsonl").read_bytes()
    persistence, case = persistence_fixture(tmp_path)
    persistence.save_detection_event(save_payload(persistence, case, valid_annotation(case)))
    assert persistence.events_path.stat().st_size > 0
    assert (PACKAGE / "decisions" / "review_decisions.json").read_bytes() == before_state
    assert (PACKAGE / "decisions" / "review_decision_events.jsonl").read_bytes() == before_events == b""
    assert not any((PACKAGE / "decisions").glob("completed_review*"))


def test_ui_has_all_modules_durable_outbox_and_no_auto_truth() -> None:
    ui = read_json(PACKAGE / "ui_config.json")
    html = (REPO / "src" / "football_intelligence" / "review_chassis" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    script = (REPO / "src" / "football_intelligence" / "review_chassis" / "static" / "detection_gold_app.js").read_text(
        encoding="utf-8"
    )
    css = (REPO / "src" / "football_intelligence" / "review_chassis" / "static" / "styles.css").read_text(
        encoding="utf-8"
    )
    assert set(ui["question_contract"]["modules"]) == {
        "detection_gold_player_static",
        "detection_gold_dense_region",
        "detection_gold_temporal_player",
        "detection_gold_pitch_boundary",
        "detection_gold_football_burst",
    }
    for token in (
        "indexedDB.open",
        "/api/review/detection-gold-event",
        "Saved to server",
        "candidate_visible_mask_coverage",
        "copyGeometryToNextFrame",
        "confirmGeometryDraft",
        "blur_trail_endpoints",
        "candidate_uuids: []",
    ):
        assert token in script
    stable_run = script[script.index("function acceptStableRun") : script.index("function setFootballState")]
    assert ".sort(" not in stable_run
    assert "annotation.frames =" not in stable_run
    mask_click = script[script.index('runtime.tool === "mask"') : script.index('runtime.tool === "ball"')]
    assert "runtime.maskPoints.push(point)" in mask_click
    assert "renderAnnotationForm()" in mask_click
    assert 'data-dg-tool="fullbox"' in html
    assert 'data-dg-tool="headbox"' in html
    assert 'data-dg-tool="ellipse"' in html
    assert 'data-dg-tool="trail"' in html
    assert "@media (max-width: 800px)" in css
    detection_body = css[css.index("body.detectionGoldPresentation") : css.index(".detectionGoldShell")]
    assert "display: block" in detection_body
    assert "grid-template-columns: none" in detection_body


def test_real_browser_persistence_and_responsive_acceptance_passed() -> None:
    browser_root = STAGE / "11_BROWSER_PERSISTENCE_AND_VISUAL_REGRESSION"
    browser = read_json(browser_root / "browser_persistence_results.json")
    visual = read_json(browser_root / "visual_regression_results.json")
    persistence = read_json(browser_root / "production_persistence_exercise.json")
    assert browser["passed"] is True
    assert browser["route_and_privacy_audit"]["passed"] is True
    assert len(visual["profiles"]) == 6
    assert visual["passed"] is True
    assert browser["module_interactions"]["dense_region"]["mask_interaction"]["screenRoundTripMaxPixels"] <= 1
    assert persistence["passed"] is True
    assert browser["production_decisions_preservation"]["passed"] is True
    assert browser["real_decisions_root_opened"] is False
    assert all((browser_root / row["path"]).is_file() for row in visual["screenshots"])


def test_timing_second_review_and_safety_classification() -> None:
    timing = read_json(STAGE / "09_ANNOTATION_TIMING_AND_INTERACTION_PLAN" / "annotation_time_estimate.json")
    second_review = read_json(PACKAGE / "second_reviewer_and_adjudication_contract.json")
    summary = read_json(STAGE / "13_COMMANDS_AND_TESTS" / "build_summary.json")
    assert 30 <= timing["estimated_active_minutes"] <= 50
    assert timing["proposal_assistance_auto_labels_truth"] is False
    assert second_review["reviewer_one_answers_delivered_to_reviewer_two"] is False
    assert summary["detector_or_tracker_evaluated"] is False
    assert summary["detector_or_tracker_promoted"] is False
    assert summary["production_ready"] is False
    assert summary["classification"] == "PASS_DETECTION_GOLD_PILOT_ANNOTATION_READY"


def test_required_strata_counts_are_exact() -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    counts = Counter((case.task_type, case.visible_metadata["pilot_stratum"]) for case in manifest.cases)
    assert counts[("detection_gold_player_static", "duplicate")] == 6
    assert counts[("detection_gold_player_static", "merged")] == 6
    assert counts[("detection_gold_player_static", "missed")] == 6
    assert counts[("detection_gold_dense_region", "dense_overlap_or_candidate_cluster")] == 8
    assert counts[("detection_gold_football_burst", "likely_visible")] == 8
    assert counts[("detection_gold_football_burst", "hard_negative")] == 6
    assert counts[("detection_gold_football_burst", "near_feet_or_markings")] == 5
    assert counts[("detection_gold_football_burst", "tiny_or_blurred")] == 5


def test_all_default_complete_annotations_validate_against_frozen_schema() -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    for case in manifest.cases:
        assert validate_case_annotation(case.task_type, valid_annotation(case))["source_binding"]
