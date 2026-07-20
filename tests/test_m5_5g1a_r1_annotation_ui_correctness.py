from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

import pytest

from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1]
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PRIOR_STAGE = PART3 / "M5_5G1A_DETECTION_GOLD_FOUNDATION_ONTOLOGY_FREEZE_AND_PILOT_ANNOTATION_v1"
PRIOR_PACKAGE = PRIOR_STAGE / "10_DETECTION_GOLD_PILOT_ANNOTATION_PACKAGE"
STAGE = PART3 / "M5_5G1A_R1_ANNOTATION_UI_CORRECTNESS_AND_PILOT_LAUNCH_REPAIR_v1"
PACKAGE = STAGE / "05_CORRECTED_DETECTION_GOLD_PILOT_ANNOTATION_PACKAGE"
SCRIPT_PATH = REPO / "src" / "football_intelligence" / "review_chassis" / "static" / "detection_gold_app.js"
HTML_PATH = REPO / "src" / "football_intelligence" / "review_chassis" / "static" / "index.html"
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer_r1"
FROZEN_ONTOLOGY_HASH = "81c256cae533a983970926cb7acfa8a090ac12629166a17181c0990877e92a8b"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def script_source() -> str:
    return SCRIPT_PATH.read_text(encoding="utf-8")


def persistence_fixture(tmp_path: Path, task_type: str) -> tuple[DetectionGoldPilotPersistence, object]:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    case = next(item for item in manifest.cases if item.task_type == task_type)
    persistence = DetectionGoldPilotPersistence(
        manifest=manifest,
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=tmp_path / "decisions",
        reviewer_session_id=REVIEWER,
    )
    persistence.ensure_state()
    return persistence, case


def save_payload(
    persistence: DetectionGoldPilotPersistence,
    case: object,
    annotation: dict,
) -> dict:
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
        "elapsed_active_seconds": 1,
    }


def background_relations(case: object) -> list[dict]:
    return [
        {"candidate_uuid": value, "relation": "BACKGROUND", "annotation_uuids": []}
        for value in case.visible_metadata["candidate_uuids"]
    ]


def static_annotation(case: object) -> dict:
    return {
        "schema_version": "m5_5g1a_detection_gold_v1",
        "source_binding": copy.deepcopy(case.visible_metadata["source_binding"]),
        "visible_person_count": 0,
        "player_instances": [],
        "candidate_relations": background_relations(case),
        "earliest_failure_stage": "UNRESOLVED",
        "note": "",
    }


def dense_annotation(case: object) -> dict:
    return {
        "schema_version": "m5_5g1a_detection_gold_v1",
        "source_binding": copy.deepcopy(case.visible_metadata["source_binding"]),
        "dense_region_uuid": f"dense-{case.case_id}",
        "trigger_reason": case.visible_metadata["pilot_stratum"],
        "human_visible_person_count": 0,
        "visible_masks": [],
        "candidate_relations": background_relations(case),
        "uncertain_or_ignore": True,
        "reviewer_agreement": "NOT_REVIEWED",
        "adjudication_state": "NOT_REQUIRED",
        "note": "",
    }


def temporal_annotation(case: object) -> dict:
    return {
        "schema_version": "m5_5g1a_detection_gold_v1",
        "source_binding": copy.deepcopy(case.visible_metadata["source_binding"]),
        "frames": [
            {
                "frame_sequence": row["frame_sequence"],
                "source_frame_sha256": row["source_frame_sha256"],
                "state": "NOT_VISIBLE",
                "current_frame_pixel_support": False,
                "candidate_uuids": [],
            }
            for row in case.visible_metadata["frame_records"]
        ],
        "contact_strip_reviewed": True,
        "stable_run_accepted": False,
        "note": "",
    }


def test_r1_package_preserves_cases_evidence_and_frozen_hashes() -> None:
    prior = read_json(PRIOR_PACKAGE / "reviewer_manifest.json")
    repaired = read_json(PACKAGE / "reviewer_manifest.json")
    validation = read_json(PACKAGE / "review_package_validation.json")
    frozen = read_json(STAGE / "01_AUDIT_AND_PRIOR_STAGE_VERIFICATION" / "frozen_hash_validation.json")
    state = read_json(PACKAGE / "decisions" / "review_decisions.json")

    assert repaired["review_id"] == "m5_5g1a_detection_gold_pilot_v1_r1"
    assert repaired["cases"] == prior["cases"]
    assert sha256_file(PACKAGE / "evidence_manifest.json") == sha256_file(PRIOR_PACKAGE / "evidence_manifest.json")
    assert validation["case_count"] == 88
    assert validation["case_payload_identical"] is True
    assert validation["evidence_bytes_identical"] is True
    assert validation["fresh_event_sequence_zero"] is True
    assert frozen["freeze_hash"] == FROZEN_ONTOLOGY_HASH
    assert frozen["passed"] is True
    assert state["annotations"] == {}
    assert state["event_sequence"] == 0
    assert (PACKAGE / "decisions" / "review_decision_events.jsonl").read_bytes() == b""
    assert not any((PACKAGE / "decisions").glob("completed_review*"))


def test_ui_selects_and_edits_explicit_objects_without_latest_object_semantics() -> None:
    source = script_source()
    assert "selectedObjectByCase" in source
    assert "data-dg-object-select" in source
    assert "data-dg-object-uuid" in source
    assert "selectedObject" in source
    assert "removeSelectedAnnotation" in source
    assert "Redraw selected visible box" in source
    assert ".at(-1)" not in source
    assert "slice(-1)" not in source
    assert "latestGeometryTarget" not in source


def test_ui_candidate_binding_uses_explicit_arbitrary_target_subsets() -> None:
    source = script_source()
    assert "data-dg-target-uuid" in source
    assert "Explicit human target" in source
    assert "MERGED_MULTIPLE_INSTANCES requires an explicit subset" in source
    assert "exactly one explicitly selected human target" in source
    assert "annotation_uuids: targets" not in source
    assert "annotation_uuids: targets.slice" not in source


def test_temporal_manual_and_refined_geometry_have_empty_candidate_uuid_lists() -> None:
    source = script_source()
    accept = source[source.index("function acceptSelectedPerson") : source.index("function finishMask")]
    refine = source[source.index("function confirmGeometryDraft") : source.index("function jumpToUnresolvedFrame")]
    assert "candidate_uuids: selectedMachineCandidate ? [selectedMachineCandidate.diagnostic_uuid] : []" in accept
    assert "candidate_uuids: []" in refine
    assert "boxes.push(frame.visible_body_box)" not in source
    assert "dgTemporalObservation" in source


def test_proposal_assistance_is_conservative_and_never_auto_clean() -> None:
    source = script_source()
    instance = source[source.index("function candidateInstance") : source.index("function acceptSelectedPerson")]
    finish_mask = source[source.index("function finishMask") : source.index("function handleOverlayClick")]
    accept = source[source.index("function acceptSelectedPerson") : source.index("function finishMask")]
    assert 'visibility_state: "UNRESOLVED"' in instance
    assert 'occlusion_type: "UNKNOWN"' in instance
    assert 'pitch_state: "BOUNDARY_UNCERTAIN"' in instance
    assert 'coarse_role: "UNKNOWN"' in instance
    assert 'mask_quality: "UNCERTAIN"' in finish_mask
    assert "candidate_relations" not in accept
    assert "Create unresolved draft geometry from proposal" in source


def test_dense_coverage_scope_and_bulk_background_have_independent_safe_controls() -> None:
    source = script_source()
    coverage = source[source.index('byId("dgCandidateMaskCoverage")') : source.index('byId("dgClearCandidateBinding")')]
    bulk = source[source.index('byId("dgMarkRemainingBackground")') : source.index('byId("dgFailureStage")')]
    assert 'addEventListener("input"' in coverage
    assert "relation.candidate_visible_mask_coverage = value" in coverage
    assert "persistDraft()" in coverage
    assert "window.confirm" in bulk
    assert "remaining.length" in bulk
    assert "Existing bindings will not change" in bulk
    assert "pushHistory()" in bulk
    assert "ANNOTATE FOCAL ROI ONLY" in source
    assert "dgFocalScopeRoi" in source
    assert 'id="dgScopeBadge"' in HTML_PATH.read_text(encoding="utf-8")


def test_server_accepts_manual_temporal_observation_without_proposal(tmp_path: Path) -> None:
    persistence, case = persistence_fixture(tmp_path, "detection_gold_temporal_player")
    annotation = temporal_annotation(case)
    binding = annotation["source_binding"]
    annotation["frames"][0] = {
        "frame_sequence": case.visible_metadata["frame_records"][0]["frame_sequence"],
        "source_frame_sha256": case.visible_metadata["frame_records"][0]["source_frame_sha256"],
        "state": "OBSERVED",
        "visible_body_box": {"x1": 10, "y1": 10, "x2": 30, "y2": 50},
        "footpoint": {"x": 20, "y": 50},
        "current_frame_pixel_support": True,
        "candidate_uuids": [],
    }
    assert binding["image_width"] > 30 and binding["image_height"] > 50
    saved = persistence.save_detection_event(save_payload(persistence, case, annotation))
    assert saved["annotations"][case.case_id]["frames"][0]["candidate_uuids"] == []
    assert saved["ack"]["saved_to_server"] is True


def test_server_rejects_wrong_frame_null_and_incomplete_temporal_rows(tmp_path: Path) -> None:
    persistence, case = persistence_fixture(tmp_path, "detection_gold_temporal_player")
    records = case.visible_metadata["frame_records"]
    wrong_candidate = records[1]["candidates"][0]

    annotation = temporal_annotation(case)
    box = copy.deepcopy(wrong_candidate["bbox_original_pixels"])
    annotation["frames"][0].update(
        {
            "state": "OBSERVED",
            "visible_body_box": box,
            "footpoint": {"x": (box["x1"] + box["x2"]) / 2, "y": box["y2"]},
            "current_frame_pixel_support": True,
            "candidate_uuids": [wrong_candidate["diagnostic_uuid"]],
        }
    )
    with pytest.raises(ValueError, match="wrong-frame candidates"):
        persistence.save_detection_event(save_payload(persistence, case, annotation))

    annotation = temporal_annotation(case)
    annotation["frames"][0]["candidate_uuids"] = [None]
    with pytest.raises(ValueError):
        persistence.save_detection_event(save_payload(persistence, case, annotation))

    annotation = temporal_annotation(case)
    annotation["contact_strip_reviewed"] = False
    with pytest.raises(ValueError):
        persistence.save_detection_event(save_payload(persistence, case, annotation))


def test_server_rejects_invalid_targets_coverage_and_source_drift(tmp_path: Path) -> None:
    persistence, case = persistence_fixture(tmp_path, "detection_gold_player_static")
    annotation = static_annotation(case)
    annotation["candidate_relations"][0] = {
        "candidate_uuid": annotation["candidate_relations"][0]["candidate_uuid"],
        "relation": "CLEAN_SINGLE_INSTANCE",
        "annotation_uuids": [],
    }
    with pytest.raises(ValueError):
        persistence.save_detection_event(save_payload(persistence, case, annotation))

    annotation = static_annotation(case)
    annotation["source_binding"]["panorama_transform"]["focal_to_panorama_x"] += 1
    with pytest.raises(ValueError, match="source binding mismatch"):
        persistence.save_detection_event(save_payload(persistence, case, annotation))

    dense_persistence, dense_case = persistence_fixture(tmp_path / "dense", "detection_gold_dense_region")
    dense = dense_annotation(dense_case)
    dense["candidate_relations"][0]["candidate_visible_mask_coverage"] = 0.5
    with pytest.raises(ValueError, match="BACKGROUND candidates cannot carry"):
        dense_persistence.save_detection_event(save_payload(dense_persistence, dense_case, dense))


def test_r1_timing_is_explicitly_modelled_not_human_measured() -> None:
    timing = read_json(STAGE / "04_TIMING_AND_HUMAN_INSTRUCTIONS" / "truthful_timing_report.json")
    assert timing["status"] == "MODELLED_NOT_HUMAN_MEASURED"
    assert timing["modelled_estimated_active_minutes"] > 0
    assert timing["human_measured_active_minutes"] is None
    assert timing["hard_cases_removed"] is False
    assert timing["machine_truth_prefilled"] is False
    assert timing["actual_active_human_time_will_be_measured_during_pilot"] is True
    assert stable_hash(timing["per_module_action_counts"])
