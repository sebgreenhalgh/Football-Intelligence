from __future__ import annotations

import hashlib
import json
import math
import runpy
import subprocess
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT / "SoccerTrack-v2"
STAGE = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B2B_128058_PITCH_POLYGON_FINALIZATION_v1"
B2A = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B2A_128058_PITCH_POLYGON_REVIEW_v1"
MATCH = PROJECT / "matches/128058"
CALIBRATION = MATCH / "calibration/pitch_polygon_v1"
EXPECTED_HEAD = "25ad330e4136a886ba85dd4a4d6bd590bb4adc27"
EVENT_ID = "d5e79c84-97a5-4f5e-9c02-a071dd7e6ca4"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_expected_head_frozen_split_and_team_convention_are_preserved() -> None:
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == EXPECTED_HEAD
    split = read(PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json")
    setup = read(MATCH / "calibration/match_setup.json")
    report = read(STAGE / "01_FINALIZATION_EVIDENCE/FINALIZATION_VALIDATION_REPORT.json")
    assert split["status"] == "FROZEN_HUMAN_APPROVED" and split["frozen"]
    assert "128058" in split["membership"]["TRAIN_DEVELOPMENT"]
    assert setup["team_mapping"]["team_1_primary_colour"] == "BLUE"
    assert setup["team_mapping"]["team_2_primary_colour"] == "WHITE"
    assert report["setup_preservation"]["only_pitch_calibration_changed"]
    assert (
        report["setup_preservation"]["non_pitch_setup_sha256_before"]
        == report["setup_preservation"]["non_pitch_setup_sha256_after"]
    )


def test_exact_immutable_event_acknowledgement_completion_and_frame_provenance() -> None:
    event_path = B2A / f"03_PITCH_POLYGON_REVIEW_PACKAGE/review_events/128058/{EVENT_ID}.json"
    acknowledgement_path = B2A / "03_PITCH_POLYGON_REVIEW_PACKAGE/review_receipts/event_acknowledgements/128058.json"
    completion_path = B2A / "03_PITCH_POLYGON_REVIEW_PACKAGE/review_receipts/completion/final.json"
    event, acknowledgement, completion = (read(path) for path in (event_path, acknowledgement_path, completion_path))
    frames = read(CALIBRATION / "source_frame_manifest.json")
    assert event["event_id"] == EVENT_ID and event["revision"] == "G7D_B2A_128058_PITCH_POLYGON_REVIEW_V1"
    assert event["alignment_answer"] == "YES" and event["first_half_closed"]
    assert event["normalization"]["closure_convention"] == "distinct_vertices_once_plus_closed_true"
    assert acknowledgement["human_event_id"] == EVENT_ID and acknowledgement["human_event_sha256"] == sha256(event_path)
    assert completion["all_cases_complete"] and completion["human_event_ids"] == [EVENT_ID]
    assert completion["human_event_sha256_values"] == [sha256(event_path)]
    assert completion["acknowledgement_receipts"][0]["receipt_sha256"] == sha256(acknowledgement_path)
    for half, frame in frames.items():
        assert event["frame_hashes"][half] == frame["frame_sha256"]
        assert event["source_dimensions"][half] == [frame["source_width"], frame["source_height"]]
        assert sha256(B2A / frame["relative_path"]) == frame["frame_sha256"]
        assert sha256(PROJECT / frame["source_video_relative_path"]) == frame["source_video_sha256"]
    assert len(sha256(completion_path)) == 64


def test_independent_geometry_and_yes_no_uncertain_camera_segment_rules() -> None:
    implementation = runpy.run_path(str(REPO / "scripts/g7d_b2b_finalize_128058_pitch_polygon.py"))
    event = read(B2A / f"03_PITCH_POLYGON_REVIEW_PACKAGE/review_events/128058/{EVENT_ID}.json")
    vertices = event["first_half_polygon_source_xy"]
    geometry = implementation["polygon_geometry"](vertices, 4096, 1080)
    assert geometry["vertex_count"] == 51 and geometry["area_pixels"] > 0 and geometry["self_intersection_count"] == 0
    assert all(math.isfinite(value) for vertex in vertices for value in vertex)
    yes = implementation["camera_segments"]("YES", vertices, None)
    no = implementation["camera_segments"]("NO", vertices, vertices)
    assert yes[0]["segment_id"] == "MATCH_STABLE_CAMERA" and len(yes) == 1
    assert [segment["segment_id"] for segment in no] == ["FIRST_HALF", "SECOND_HALF"]
    with pytest.raises(ValueError, match="FAIL_G7D_B2B_ALIGNMENT_UNCERTAIN"):
        implementation["camera_segments"]("UNCERTAIN", vertices, None)


def test_final_polygon_schema_manifest_setup_hash_and_visual_are_valid() -> None:
    polygon_path = CALIBRATION / "pitch_polygon.json"
    polygon, report, manifest = (
        read(path)
        for path in (
            polygon_path,
            CALIBRATION / "pitch_polygon_validation_report.json",
            CALIBRATION / "pitch_polygon_manifest.json",
        )
    )
    setup = read(MATCH / "calibration/match_setup.json")
    assert polygon["status"] == "HUMAN_CONFIRMED" and polygon["coordinate_space"] == "SOURCE_IMAGE_PIXELS"
    assert (
        polygon["human_review_event_id"] == EVENT_ID and polygon["closed"] and polygon["self_intersection_count"] == 0
    )
    assert polygon["second_half_alignment_answer"] == "YES" and len(polygon["camera_segments"]) == 1
    assert polygon["validation_report_sha256"] == sha256(CALIBRATION / "pitch_polygon_validation_report.json")
    assert setup["pitch_calibration"]["polygon_path"] == "calibration/pitch_polygon_v1/pitch_polygon.json"
    assert setup["pitch_calibration"]["polygon_sha256"] == sha256(polygon_path)
    assert (
        setup["pitch_calibration"]["camera_segment_count"] == 1
        and setup["pitch_calibration"]["status"] == "HUMAN_CONFIRMED"
    )
    assert report["geometry"]["first_half"]["self_intersection_count"] == 0
    assert {row["project_relative_path"] for row in manifest["files"]} == {
        "matches/128058/calibration/pitch_polygon_v1/pitch_polygon.json",
        "matches/128058/calibration/pitch_polygon_v1/pitch_polygon_validation_report.json",
        (
            "experiments/football_observation_reasoner/part 6/G7D_B2A_128058_PITCH_POLYGON_REVIEW_v1/"
            "03_PITCH_POLYGON_REVIEW_PACKAGE/review_events/128058/d5e79c84-97a5-4f5e-9c02-a071dd7e6ca4.json"
        ),
        (
            "experiments/football_observation_reasoner/part 6/G7D_B2A_128058_PITCH_POLYGON_REVIEW_v1/"
            "03_PITCH_POLYGON_REVIEW_PACKAGE/review_receipts/event_acknowledgements/128058.json"
        ),
        (
            "experiments/football_observation_reasoner/part 6/G7D_B2A_128058_PITCH_POLYGON_REVIEW_v1/"
            "03_PITCH_POLYGON_REVIEW_PACKAGE/review_receipts/completion/final.json"
        ),
        "matches/128058/calibration/pitch_polygon_v1/source_frame_manifest.json",
    }
    for row in manifest["files"]:
        path = PROJECT / row["project_relative_path"]
        assert path.stat().st_size == row["byte_size"] and sha256(path) == row["sha256"]
    assert list((STAGE / "02_VISUAL_QA").glob("*.png")) == [
        STAGE / "02_VISUAL_QA/128058_final_pitch_polygon_validation.png"
    ]


def test_handoff_manifest_is_self_contained_and_runtime_is_not_imported() -> None:
    handoff = STAGE / "03_REVIEW_PACK/CHATGPT_HANDOFF"
    expected = {
        "01_EXECUTIVE_SUMMARY.json",
        "02_EVENT_RECEIPT_AND_POLYGON_RESULTS.json",
        "03_MATCH_SETUP_AND_ARTIFACT_RESULTS.json",
        "04_DECISION.md",
        "05_FINALIZATION_CONTRACT.md",
        "06_TESTS_AND_SAFETY.json",
        "07_FINAL_POLYGON_VALIDATION.png",
        "08_MANIFEST.json",
    }
    assert {path.name for path in handoff.iterdir()} == expected
    summary, manifest = read(handoff / "01_EXECUTIVE_SUMMARY.json"), read(handoff / "08_MANIFEST.json")["files"]
    assert summary["classification"] == "PASS_G7D_B2B_128058_PITCH_POLYGON_FINALIZED"
    assert summary["next_permitted_stage"] == "G7D_B2C_RESUME_FROZEN_128058_BASELINE"
    assert len(manifest) == 7 and {row["filename"] for row in manifest} == expected - {"08_MANIFEST.json"}
    for row in manifest:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"] and sha256(path) == row["sha256"]
    source = (REPO / "scripts/g7d_b2b_finalize_128058_pitch_polygon.py").read_text(encoding="utf-8").lower()
    assert all(
        term not in source for term in ("ultralytics", "torch", "g7d_b1", "proposal_nodes", "run_all_folds", "p2", "p3")
    )
