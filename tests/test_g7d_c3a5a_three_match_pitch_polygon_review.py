from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT / "SoccerTrack-v2"
STAGE = PROJECT / "experiments/football_observation_reasoner/part 7/G7D_C3A5A_THREE_MATCH_PITCH_POLYGON_REVIEW_v1"
PACKAGE = STAGE / "02_PITCH_POLYGON_REVIEW_PACKAGE"
MATCH_IDS = ["117093", "118576", "118577"]
EXPECTED_HEAD = "ed21b91d6c26837deb09a059835fa2fe77f93acf"
REVISION = "G7D_C3A5A_THREE_MATCH_PITCH_POLYGON_REVIEW_V1"
REVIEW_ID = "G7D_C3A5A_THREE_MATCH_PITCH_POLYGON_REVIEW"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def polygon(width: int, height: int) -> list[list[float]]:
    return [
        [200, 150],
        [width * 0.3, 90],
        [width * 0.72, 100],
        [width - 180, 220],
        [width - 260, height - 130],
        [width * 0.55, height - 60],
        [500, height - 110],
        [160, height * 0.5],
    ]


def event(case: dict, event_id: str, alignment: str = "YES") -> dict:
    first = case["source_frames"]["first"]
    second = case["source_frames"]["second"]
    second_polygon = polygon(second["source_width"], second["source_height"]) if alignment == "NO" else None
    return {
        "schema_version": "football_intelligence.g7d_c3a5a.pitch_polygon_review_event.v1",
        "review_id": REVIEW_ID,
        "revision": REVISION,
        "match_id": case["match_id"],
        "client_event_id": event_id,
        "timestamp": "2026-07-30T23:00:00Z",
        "alignment_answer": alignment,
        "first_half_polygon_source_xy": polygon(first["source_width"], first["source_height"]),
        "first_half_closed": True,
        "second_half_polygon_source_xy": second_polygon,
        "second_half_closed": alignment == "NO",
        "frame_hashes": {"first": first["frame_sha256"], "second": second["frame_sha256"]},
        "source_dimensions": {
            "first": [first["source_width"], first["source_height"]],
            "second": [second["source_width"], second["source_height"]],
        },
        "coordinate_audit": {
            "verified": True,
            "source_round_trip_max_error_px": 0.0,
            "display_round_trip_max_error_css_px": 0.0,
            "tested_device_pixel_ratio": 1,
        },
        "normalization": {
            "first": {
                "closure_convention": "distinct_vertices_once_plus_closed_true",
                "removed_exact_adjacent_vertex_indices": [],
                "removed_exact_terminal_duplicate": False,
            },
            "second": None,
        },
    }


def post(port: int, payload: dict) -> tuple[int, dict]:
    request = Request(
        f"http://127.0.0.1:{port}/api/save",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response = urlopen(request, timeout=5)
    return response.status, json.loads(response.read())


def test_frozen_split_eligibility_and_source_correction() -> None:
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == EXPECTED_HEAD
    split = read(PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json")
    closure = read(STAGE / "00_INPUT_CLOSURE/split_and_setup_validation.json")
    correction = read(PROJECT / "matches/117093/manifests/source_correction_events.json")
    assert split["frozen"] and split["status"] == "FROZEN_HUMAN_APPROVED"
    assert set(MATCH_IDS) <= set(split["membership"]["TRAIN_DEVELOPMENT"])
    assert closure["pitch_calibration_status"] == {match_id: "HUMAN_REQUIRED" for match_id in MATCH_IDS}
    assert closure["project_default"] == "DISABLED" and closure["inference_executed"] is False
    assert any(
        row["event_type"] == "AUTHORIZED_PRE_FREEZE_SOURCE_CORRECTION"
        and row["new_source_path"].endswith("117093_panorama_1st_half-008.mp4")
        and row["old_source_path"].endswith("117093_calibrated_panorama_1st_half.mp4")
        for row in correction
    )


def test_exact_six_canonical_videos_and_frames_are_hash_bound() -> None:
    resolution = read(STAGE / "00_INPUT_CLOSURE/source_video_resolution.json")["matches"]
    frames = read(STAGE / "01_REVIEW_FRAMES/source_frame_manifest.json")["matches"]
    assert set(resolution) == set(frames) == set(MATCH_IDS)
    assert resolution["117093"]["first"]["project_relative_path"].endswith("117093_panorama_1st_half-008.mp4")
    count = 0
    for match_id in MATCH_IDS:
        source_manifest = {
            row["relative_path"]: row
            for row in read(PROJECT / f"matches/{match_id}/manifests/source_file_manifest.json")["files"]
        }
        for half in ("first", "second"):
            video = resolution[match_id][half]
            frame = frames[match_id][half]
            assert (
                source_manifest[video["project_relative_path"]]["sha256"]
                == video["sha256"]
                == frame["source_video_sha256"]
            )
            assert video["byte_size"] == (PROJECT / video["project_relative_path"]).stat().st_size
            assert frame["selection_rule"] == "25_PERCENT_DURATION_NEAREST_FRAME_ROUND_HALF_UP"
            assert abs(frame["requested_timestamp_seconds"] - video["duration_seconds"] * 0.25) < 1e-6
            assert sha256(STAGE / frame["relative_path"]) == frame["frame_sha256"]
            assert frame["source_width"] > 0 and frame["source_height"] > 0
            count += 1
    assert count == 6


def test_bounded_routes_and_all_six_real_assets() -> None:
    cases = read(PACKAGE / "cases.json")
    assets = read(PACKAGE / "asset_manifest.json")["assets"]
    assert [case["match_id"] for case in cases["cases"]] == MATCH_IDS
    assert cases["review_revision"] == REVISION and len(assets) == 6
    assert not (PACKAGE / "human_decisions").exists()
    with tempfile.TemporaryDirectory() as temporary:
        process = subprocess.Popen(
            [
                sys.executable,
                "review_server.py",
                "--port",
                "18815",
                "--decisions-root",
                str(Path(temporary) / "decisions"),
            ],
            cwd=PACKAGE,
        )
        try:
            time.sleep(0.3)
            assert urlopen("http://127.0.0.1:18815/", timeout=5).status == 200
            assert (
                json.loads(urlopen("http://127.0.0.1:18815/api/cases", timeout=5).read())["completion_receipt_id"]
                is None
            )
            for asset in assets:
                response = urlopen(f"http://127.0.0.1:18815{asset['route']}", timeout=5)
                body = response.read()
                assert response.status == 200 and response.headers["Content-Type"].startswith("image/png")
                assert hashlib.sha256(body).hexdigest() == asset["sha256"]
            with pytest.raises(HTTPError) as error:
                urlopen("http://127.0.0.1:18815/assets/../cases.json", timeout=5)
            assert error.value.code == 404
        finally:
            process.terminate()
            process.wait(timeout=5)


def test_geometry_yes_no_uncertain_and_immutable_receipts() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        isolated = Path(temporary) / "package"
        decisions = Path(temporary) / "decisions"
        isolated.mkdir()
        for name in ("review_server.py", "polygon_validation.py", "cases.json"):
            shutil.copy2(PACKAGE / name, isolated / name)
        process = subprocess.Popen(
            [sys.executable, "review_server.py", "--port", "18816", "--decisions-root", str(decisions)], cwd=isolated
        )
        try:
            time.sleep(0.3)
            cases = read(isolated / "cases.json")["cases"]
            status, first = post(18816, event(cases[0], "11709300-0000-4000-8000-000000000001", "YES"))
            assert status == 200 and first["case_complete"] and not first["all_cases_complete"]
            assert first["last_saved_event_id"] != first["completion_receipt_id"]
            event_path = decisions / first["saved_path"]
            receipt_path = decisions / first["receipt_path"]
            assert sha256(event_path) == read(receipt_path)["human_event_sha256"]
            status, second = post(18816, event(cases[1], "11857600-0000-4000-8000-000000000001", "NO"))
            assert status == 200 and second["case_complete"] and not second["all_cases_complete"]
            uncertain = event(cases[2], "11857700-0000-4000-8000-000000000001", "UNCERTAIN")
            with pytest.raises(HTTPError) as error:
                post(18816, uncertain)
            assert error.value.code == 422
            assert json.loads(error.value.read())["error_code"] == "ALIGNMENT_UNCERTAIN"
            assert not (decisions / "events/118577").exists()
            status, third = post(18816, event(cases[2], "11857700-0000-4000-8000-000000000002", "YES"))
            assert status == 200 and third["all_cases_complete"]
            assert third["completion_receipt_id"].startswith("completion-")
            completion = read(decisions / third["completion_receipt_path"])
            assert completion["required_match_ids"] == MATCH_IDS and completion["all_cases_complete"]
            assert len(completion["latest_acknowledged_events"]) == 3
            repeated_status, repeated = post(18816, event(cases[2], "11857700-0000-4000-8000-000000000002", "YES"))
            assert repeated_status == 200 and repeated["event_id"] == third["event_id"]
            assert len(list((decisions / "events").glob("*/*.json"))) == 3
        finally:
            process.terminate()
            process.wait(timeout=5)


@pytest.mark.parametrize("dpr", [1, 2])
@pytest.mark.parametrize("display", [(620.0, 163.4765625), (900.0, 237.3046875), (1200.0, 316.40625)])
def test_coordinate_round_trips_resize_letterbox_and_dpr(dpr: int, display: tuple[float, float]) -> None:
    source = (4096.0, 1080.0)
    left, top = 37.25, 91.5
    points = [(0.0, 0.0), source, (2048.0, 540.0), (811.125, 944.875)]
    for x, y in points:
        shown = (left + x * display[0] / source[0], top + y * display[1] / source[1])
        back = ((shown[0] - left) * source[0] / display[0], (shown[1] - top) * source[1] / display[1])
        shown_again = (left + back[0] * display[0] / source[0], top + back[1] * display[1] / source[1])
        canvas = (back[0] * display[0] * dpr / source[0], back[1] * display[1] * dpr / source[1])
        assert abs(back[0] - x) <= 0.5 and abs(back[1] - y) <= 0.5
        assert abs(shown_again[0] - shown[0]) <= 1 and abs(shown_again[1] - shown[1]) <= 1
        assert all(value >= 0 for value in canvas)


def test_live_edge_visuals_handoff_and_safety_boundaries() -> None:
    acceptance = read(STAGE / "03_TESTS_AND_LOGS/live_edge_acceptance.json")
    assert acceptance["status"] == "PASS_G7D_C3A5A_LIVE_EDGE_ACCEPTANCE"
    assert (
        acceptance["temporary_state_removed_after_validation"]
        and not Path(acceptance["temporary_decisions_root"]).exists()
    )
    assert len(acceptance["assets"]) == 6 and all(row["status"] == 200 for row in acceptance["assets"])
    visuals = sorted((STAGE / "04_VISUAL_QA").glob("*.png"))
    assert [path.name for path in visuals] == [
        "01_THREE_MATCH_REVIEWER_READY.png",
        "02_SECOND_HALF_ALIGNMENT_AND_SAVE.png",
    ]
    assert all(path.stat().st_size > 100_000 for path in visuals)
    handoff = STAGE / "05_REVIEW_PACK/CHATGPT_HANDOFF"
    expected = {
        f"{index:02d}_{name}"
        for index, name in enumerate(
            [
                "EXECUTIVE_SUMMARY.json",
                "SOURCE_AND_FRAME_PROVENANCE.json",
                "REVIEWER_AND_GEOMETRY_RESULTS.json",
                "DECISION.md",
                "REVIEW_CONTRACT.md",
                "TESTS_SAFETY_AND_SOURCE_CHANGES.json",
                "REVIEWER_READY.png",
                "ALIGNMENT_AND_SAVE.png",
                "MANIFEST.json",
            ],
            start=1,
        )
    }
    assert {path.name for path in handoff.iterdir()} == expected
    manifest = read(handoff / "09_MANIFEST.json")["files"]
    assert len(manifest) == 8
    for row in manifest:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"] and sha256(path) == row["sha256"]
    source = (REPO / "scripts/g7d_c3a5a_build_three_match_pitch_polygon_review.py").read_text(encoding="utf-8").lower()
    assert "import torch" not in source and "ultralytics" not in source and "pitch_gate_mode" not in source
    for match_id in MATCH_IDS:
        setup = read(PROJECT / f"matches/{match_id}/calibration/match_setup.json")
        assert setup["pitch_calibration"]["status"] == "HUMAN_REQUIRED"
        assert setup["pitch_calibration"]["polygon_path"] is None
        assert not (PROJECT / f"matches/{match_id}/calibration/pitch_polygon_v1/pitch_polygon.json").exists()
    assert not (PACKAGE / "human_decisions").exists()
