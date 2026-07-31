from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
SCRIPT = REPO / "scripts/g7d_c3a5b_finalize_three_match_pitch_polygons.py"
SPEC = importlib.util.spec_from_file_location("c3a5b", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_geometry_accepts_dense_valid_polygon_and_rejects_invalid() -> None:
    valid = [[0, 0], [8, 0], [9, 1], [10, 4], [6, 8], [0, 7]]
    result = MODULE.polygon_geometry(valid, 20, 20)
    assert result["vertex_count"] == 6
    assert result["self_intersection_count"] == 0
    with pytest.raises(ValueError):
        MODULE.polygon_geometry([[0, 0], [10, 10], [0, 10], [10, 0]], 20, 20)
    with pytest.raises(ValueError):
        MODULE.polygon_geometry([[0, 0], [10, 0], [10, 10], [0, 0]], 20, 20)


def test_camera_segment_policy_yes_no_uncertain() -> None:
    first = [[0, 0], [4, 0], [4, 4], [0, 4]]
    second = [[1, 1], [5, 1], [5, 5], [1, 5]]
    yes = MODULE.camera_segments({"alignment_answer": "YES", "first_half_polygon_source_xy": first})
    assert yes == [
        {
            "segment_id": "MATCH_STABLE_CAMERA",
            "halves": ["FIRST_HALF", "SECOND_HALF"],
            "vertices_source_xy": first,
        }
    ]
    no = MODULE.camera_segments(
        {
            "alignment_answer": "NO",
            "first_half_polygon_source_xy": first,
            "second_half_polygon_source_xy": second,
            "second_half_closed": True,
        }
    )
    assert len(no) == 2
    with pytest.raises(ValueError):
        MODULE.camera_segments({"alignment_answer": "NO", "first_half_polygon_source_xy": first})
    with pytest.raises(ValueError):
        MODULE.camera_segments({"alignment_answer": "UNCERTAIN", "first_half_polygon_source_xy": first})


def test_final_artifacts_and_exact_human_chain() -> None:
    selected, completion = MODULE.resolve_chain()
    assert set(selected) == set(MODULE.MATCH_IDS)
    assert selected["118576"]["event"]["event_id"] == MODULE.VISIBLE_LAST_EVENT
    assert completion["value"]["completion_receipt_id"] == MODULE.COMPLETION_ID
    assert completion["value"]["all_cases_complete"] is True
    for match_id in MODULE.MATCH_IDS:
        polygon_path = PROJECT / f"matches/{match_id}/calibration/pitch_polygon_v1/pitch_polygon.json"
        polygon = read_json(polygon_path)
        event = selected[match_id]["event"]
        assert polygon["vertices_source_xy"] == event["first_half_polygon_source_xy"]
        assert polygon["camera_segments"][0]["segment_id"] == "MATCH_STABLE_CAMERA"
        assert polygon["status"] == "HUMAN_CONFIRMED"
        assert polygon["production_ready"] is False
        setup = read_json(PROJECT / f"matches/{match_id}/calibration/match_setup.json")
        pitch = setup["pitch_calibration"]
        assert pitch["polygon_sha256"] == sha256(polygon_path)
        assert pitch["search_region_status"] == "PENDING"
        assert pitch["expanded_search_region_status"] == "PENDING"
        assert pitch["production_ready"] is False
    assert (
        read_json(PROJECT / "matches/117093/calibration/pitch_polygon_v1/pitch_polygon.json")["first_half_reference"][
            "source_video_relative_path"
        ]
        == "matches/117093/source/videos/117093_panorama_1st_half-008.mp4"
    )


def test_evidence_visuals_manifest_and_handoff() -> None:
    stage = MODULE.STAGE
    artifact_manifest = read_json(stage / "01_FINAL_POLYGON_ARTIFACTS/polygon_artifact_manifest.json")
    assert len(artifact_manifest["files"]) == 15
    for record in artifact_manifest["files"]:
        path = PROJECT / record["project_relative_path"]
        assert path.stat().st_size == record["byte_size"]
        assert sha256(path) == record["sha256"]
    visuals = sorted((stage / "03_VISUAL_QA").glob("*.png"))
    assert [path.name for path in visuals] == [
        "01_THREE_MATCH_FINAL_POLYGONS.png",
        "02_SECOND_HALF_ALIGNMENT_VALIDATION.png",
    ]
    handoff = stage / "05_REVIEW_PACK/CHATGPT_HANDOFF"
    assert sorted(path.name for path in handoff.iterdir()) == [
        "01_EXECUTIVE_SUMMARY.json",
        "02_EVENT_AND_RECEIPT_CLOSURE.json",
        "03_POLYGON_AND_GEOMETRY_RESULTS.json",
        "04_MATCH_SETUP_UPDATE_RESULTS.json",
        "05_DECISION.md",
        "06_FINALIZATION_CONTRACT.md",
        "07_TESTS_SAFETY_AND_SOURCE_CHANGES.json",
        "08_FINAL_POLYGONS.png",
        "09_ALIGNMENT_VALIDATION.png",
        "10_MANIFEST.json",
    ]
    handoff_manifest = read_json(handoff / "10_MANIFEST.json")
    assert len(handoff_manifest["files"]) == 9
    for record in handoff_manifest["files"]:
        path = handoff / record["filename"]
        assert path.stat().st_size == record["byte_size"]
        assert sha256(path) == record["sha256"]


def test_setup_diff_and_immutable_truth_report() -> None:
    report = read_json(MODULE.STAGE / "02_MATCH_SETUP_UPDATES/match_setup_field_diff.json")
    for match_id in MODULE.MATCH_IDS:
        row = report["matches"][match_id]
        assert row["only_pitch_calibration_changed"] is True
        assert row["non_pitch_sha256_before"] == row["non_pitch_sha256_after"]
    validation = read_json(MODULE.STAGE / "04_TESTS_AND_LOGS/finalization_validation_report.json")
    assert validation["classification"] == MODULE.CLASSIFICATION
    assert validation["immutable_human_truth_unchanged"] is True
    assert validation["immutable_human_truth_before_sha256"] == validation["immutable_human_truth_after_sha256"]
