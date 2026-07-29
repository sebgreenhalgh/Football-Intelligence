from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import runpy
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT / "SoccerTrack-v2"
STAGE = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B2A_128058_PITCH_POLYGON_REVIEW_v1"
PACKAGE = STAGE / "03_PITCH_POLYGON_REVIEW_PACKAGE"
MATCH = PROJECT / "matches/128058"
EXPECTED_HEAD = "1eadbfc08c0ea90125513ac17cbc7ee00f11ebe1"
REVISION = "G7D_B2A_128058_PITCH_POLYGON_REVIEW_V1"
REVIEW_ID = "G7D_B2A_128058_PITCH_POLYGON_REVIEW"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dense_polygon(width: int, height: int) -> list[list[float]]:
    top = [[200 + 280 * index, 120 + (index % 2)] for index in range(11)]
    right = [[width - 200 + (index % 2), 220 + 85 * index] for index in range(8)]
    bottom = [[width - 300 - 280 * index, height - 130 - (index % 2)] for index in range(10)]
    left = [[220 - (index % 2), height - 230 - 90 * index] for index in range(8)]
    return top + right + bottom + left


def payload(case: dict, event_id: str, alignment: str = "YES") -> dict:
    first, second = case["source_frames"]["first"], case["source_frames"]["second"]
    return {
        "schema_version": "football_intelligence.g7d_a.pitch_polygon_review_event.v1",
        "review_id": REVIEW_ID,
        "revision": REVISION,
        "match_id": "128058",
        "client_event_id": event_id,
        "timestamp": "2026-07-29T00:00:00Z",
        "alignment_answer": alignment,
        "first_half_polygon_source_xy": dense_polygon(first["source_width"], first["source_height"]),
        "first_half_closed": True,
        "second_half_polygon_source_xy": None,
        "second_half_closed": False,
        "frame_hashes": {"first": first["frame_sha256"], "second": second["frame_sha256"]},
        "source_dimensions": {
            "first": [first["source_width"], first["source_height"]],
            "second": [second["source_width"], second["source_height"]],
        },
        "coordinate_audit": {
            "verified": True,
            "first_half_round_trip_max_error_css_px": 0.0,
            "second_half_projection_verified": True,
        },
        "normalization": {
            "closure_convention": "distinct_vertices_once_plus_closed_true",
            "removed_exact_adjacent_vertex_indices": [],
            "removed_exact_terminal_duplicate": False,
        },
    }


def test_baseline_b2_stop_and_historical_team_convention_are_preserved() -> None:
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == EXPECTED_HEAD
    split = read(PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json")
    setup = read(MATCH / "calibration/match_setup.json")
    stop = read(STAGE / "01_INPUT_CLOSURE/b2_stop_validation.json")
    input_contract = read(
        PROJECT
        / "experiments/football_observation_reasoner/part 6"
        / "G7D_B2A_128058_Pitch_Polygon_Review_Codex_Pack/02_STAGE_INPUT_CONTRACT.json"
    )
    assert split["frozen"] and split["status"] == "FROZEN_HUMAN_APPROVED"
    assert "128058" in split["membership"]["TRAIN_DEVELOPMENT"]
    assert setup["team_mapping"]["team_1_primary_colour"] == "BLUE"
    assert setup["team_mapping"]["team_2_primary_colour"] == "WHITE"
    assert setup["pitch_calibration"]["status"] == "HUMAN_REQUIRED"
    assert setup["pitch_calibration"]["polygon_path"] is None
    assert stop["sampling_or_inference_started"] is False
    assert input_contract["review_revision"] == REVISION and input_contract["review_port"] == 8813
    assert not (MATCH / "calibration/pitch_polygon_v1/pitch_polygon.json").exists()


def test_canonical_video_and_exact_two_frame_provenance() -> None:
    resolution = read(STAGE / "01_INPUT_CLOSURE/source_video_resolution.json")["videos"]
    frames = read(STAGE / "02_REVIEW_INPUTS/source_frame_manifest.json")["frames"]
    hashes = read(MATCH / "manifests/source_file_hashes.json")
    assert set(resolution) == {"first", "second"} == set(frames)
    for half, frame in frames.items():
        video = resolution[half]
        assert video["project_relative_path"] == frame["source_video_relative_path"]
        assert hashes[video["project_relative_path"]] == video["sha256"] == frame["source_video_sha256"]
        assert sha256(PROJECT / video["project_relative_path"]) == video["sha256"]
        assert video["resolution"] == [4096, 1080] and video["frame_rate"] == "25/1"
        assert frame["selection_rule"] == "25_PERCENT_DURATION_NEAREST_FRAME_ROUND_HALF_UP"
        assert frame["source_width"] == 4096 and frame["source_height"] == 1080
        assert sha256(STAGE / frame["relative_path"]) == frame["frame_sha256"]
    assert frames["first"]["frame_index_zero_based"] == 17250
    assert frames["second"]["frame_index_zero_based"] == 16944


def test_one_case_package_has_bounded_routes_and_no_prior_human_event() -> None:
    cases = read(PACKAGE / "review_cases.json")
    contract = read(PACKAGE / "reviewer_contract.json")
    html = (PACKAGE / "index.html").read_text(encoding="utf-8")
    launcher = (PACKAGE / "launch_pitch_polygon_review.ps1").read_text(encoding="utf-8")
    assert cases["review_id"] == REVIEW_ID and cases["review_revision"] == REVISION
    assert [case["match_id"] for case in cases["cases"]] == ["128058"]
    assert contract["port"] == 8813 and contract["url"] == "http://127.0.0.1:8813/"
    assert "TEAM_1=BLUE" in html and "TEAM_2=WHITE" in html
    assert "Coordinate mapping: VERIFIED" in html and "SAVED — SERVER ACKNOWLEDGED" in html
    assert "UNCERTAIN" in html and "Saving…" in html and "$Port = 8813" in launcher
    assert not any((PACKAGE / name).exists() for name in ("review_events", "review_receipts", "review_drafts"))
    assert list((STAGE / "04_VISUAL_QA").glob("*.png")) == [
        STAGE / "04_VISUAL_QA/128058_pitch_polygon_review_inputs.png"
    ]
    process = subprocess.Popen([sys.executable, "review_server.py", "--port", "8813"], cwd=PACKAGE)
    try:
        time.sleep(0.25)
        assert urlopen("http://127.0.0.1:8813/", timeout=3).status == 200
        for half in ("first", "second"):
            response = urlopen(f"http://127.0.0.1:8813/assets/128058/{half}.png", timeout=3)
            assert response.status == 200 and response.headers["Content-Type"].startswith("image/png")
            served = PACKAGE / "_frames" / f"128058_{half}.png"
            reference = STAGE / "02_REVIEW_INPUTS" / f"{half}_half_reference.png"
            assert sha256(served) == sha256(reference)
        with pytest.raises(HTTPError) as error:
            urlopen("http://127.0.0.1:8813/assets/../review_cases.json", timeout=3)
        assert error.value.code == 404
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_coordinate_geometry_and_one_case_receipt_protocol_in_isolation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        isolated = Path(temporary)
        for name in ("review_server.py", "review_cases.json", "polygon_validation.py"):
            shutil.copy2(PACKAGE / name, isolated / name)
        process = subprocess.Popen([sys.executable, "review_server.py", "--port", "8814"], cwd=isolated)
        try:
            time.sleep(0.25)
            case = read(isolated / "review_cases.json")["cases"][0]

            def post(value: dict) -> dict:
                request = Request(
                    "http://127.0.0.1:8814/api/save",
                    data=json.dumps(value).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                return json.loads(urlopen(request, timeout=3).read())

            result = post(payload(case, "12805800-0000-4000-8000-000000000001"))
            assert result["ok"] and result["case_complete"] and result["all_cases_complete"]
            event = isolated / result["saved_path"]
            acknowledgement = isolated / result["receipt_path"]
            completion = isolated / result["completion_receipt_path"]
            assert event.exists() and acknowledgement.exists() and completion.exists()
            assert hashlib.sha256(event.read_bytes()).hexdigest() == read(acknowledgement)["human_event_sha256"]
            assert read(completion)["required_match_ids"] == ["128058"]
            repeat = post(payload(case, "12805800-0000-4000-8000-000000000001"))
            assert repeat["event_id"] == result["event_id"]
            sys.path.insert(0, str(PACKAGE))
            try:
                module = runpy.run_path(str(PACKAGE / "review_server.py"))
            finally:
                sys.path.pop(0)
            no_payload = payload(case, "12805800-0000-4000-8000-000000000004", alignment="NO")
            no_payload["second_half_polygon_source_xy"] = dense_polygon(
                case["source_frames"]["second"]["source_width"], case["source_frames"]["second"]["source_height"]
            )
            no_payload["second_half_closed"] = True
            accepted, validation = module["validate"](no_payload)
            assert accepted == no_payload and validation["ok"]
            with pytest.raises(HTTPError) as uncertain:
                post(payload(case, "12805800-0000-4000-8000-000000000002", alignment="UNCERTAIN"))
            assert uncertain.value.code == 422
            assert json.loads(uncertain.value.read())["error_code"] == "ALIGNMENT_UNCERTAIN"
            invalid = payload(case, "12805800-0000-4000-8000-000000000003")
            invalid["first_half_polygon_source_xy"] = [[-1, 0], [1, 0], [1, 1], [0, 1]]
            with pytest.raises(HTTPError) as malformed:
                post(invalid)
            assert json.loads(malformed.value.read())["error_code"] == "OUT_OF_BOUNDS_VERTEX"
        finally:
            process.terminate()
            process.wait(timeout=5)


def test_source_code_never_imports_or_runs_baseline_inference() -> None:
    source = (REPO / "scripts/g7d_b2a_build_pitch_polygon_review.py").read_text(encoding="utf-8").lower()
    assert "ultralytics" not in source and "torch" not in source and "frozenfoldwiseruntime" not in source
    assert "g7d_b1" not in source and "proposal_nodes" not in source and "run_all_folds" not in source
    assert not (MATCH / "calibration/pitch_polygon_v1/pitch_polygon.json").exists()
    calibration = read(MATCH / "calibration/match_setup.json")["pitch_calibration"]
    assert calibration["authoritative_method"] == "HUMAN_DRAWN_PER_MATCH"
    assert calibration["expanded_search_region_status"] == "PENDING_PITCH_POLYGON"
    assert calibration["polygon_path"] is None and calibration["polygon_sha256"] is None
    assert calibration["status"] == "HUMAN_REQUIRED"


def test_upload_only_handoff_is_self_contained_and_non_recursive() -> None:
    handoff = STAGE / "05_REVIEW_PACK/CHATGPT_HANDOFF"
    expected = {
        "01_EXECUTIVE_SUMMARY.json",
        "02_INPUT_AND_PROVENANCE_RESULTS.json",
        "03_REVIEWER_AND_TEST_RESULTS.json",
        "04_DECISION.md",
        "05_REVIEW_CONTRACT.md",
        "06_TESTS_AND_SAFETY.json",
        "07_REVIEW_INPUTS.png",
        "08_MANIFEST.json",
    }
    assert {path.name for path in handoff.iterdir()} == expected
    summary = read(handoff / "01_EXECUTIVE_SUMMARY.json")
    assert summary["status"] == "PASS_G7D_B2A_128058_PITCH_POLYGON_REVIEW_READY_FOR_HUMAN_REVIEW"
    assert summary["review_revision"] == REVISION and summary["human_review_case_count"] == 1
    assert summary["next_stage_after_human_completion"] == "G7D_B2B_128058_PITCH_POLYGON_FINALIZATION"
    assert len(summary["unresolved_blockers"]) == 2
    manifest = read(handoff / "08_MANIFEST.json")["files"]
    assert len(manifest) == 7 and {entry["filename"] for entry in manifest} == expected - {"08_MANIFEST.json"}
    for entry in manifest:
        path = handoff / entry["filename"]
        assert path.stat().st_size == entry["byte_size"] and sha256(path) == entry["sha256"]
    assert sha256(handoff / "07_REVIEW_INPUTS.png") == sha256(
        STAGE / "04_VISUAL_QA/128058_pitch_polygon_review_inputs.png"
    )
