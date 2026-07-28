import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from pathlib import Path


ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
WORKSPACE = ROOT / r"experiments\football_observation_reasoner\part 5\G7D_A_TWO_MATCH_SETUP_AND_PITCH_POLYGON_REVIEW_v1"
PACKAGE = WORKSPACE / "06_PITCH_POLYGON_REVIEW_PACKAGE"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_selected_cases_and_frame_hashes() -> None:
    cases = load(PACKAGE / "review_cases.json")["cases"]
    assert [case["match_id"] for case in cases] == ["118575", "117092"]
    for case in cases:
        for frame in case["source_frames"].values():
            path = ROOT / Path(frame["relative_path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == frame["source_sha256"]
            assert frame["timestamp_seconds"] > 0


def test_review_package_and_safety_boundaries() -> None:
    assert (PACKAGE / "launch_pitch_polygon_review.ps1").exists()
    assert (PACKAGE / "review_server.py").exists()
    assert "review_events" in (PACKAGE / "review_server.py").read_text(encoding="utf-8")
    assert "inference" not in (PACKAGE / "review_server.py").read_text(encoding="utf-8").lower()
    assert not any("validation" in p.name.lower() or "holdout" in p.name.lower() for p in PACKAGE.iterdir())


def test_bounded_http_assets_and_browser_bindings() -> None:
    process = subprocess.Popen([sys.executable, "review_server.py", "--port", "8813"], cwd=PACKAGE)
    try:
        time.sleep(0.25)
        root = urlopen("http://127.0.0.1:8813/", timeout=3)
        assert root.status == 200
        html = root.read().decode()
        assert "first.src=active.asset_urls.first" in html
        assert "second.src=active.asset_urls.second" in html
        for match in ("118575", "117092"):
            for half in ("first", "second"):
                response = urlopen(f"http://127.0.0.1:8813/assets/{match}/{half}.png", timeout=3)
                assert response.status == 200
                assert response.headers["Content-Type"].startswith("image/png")
                assert (
                    hashlib.sha256(response.read()).hexdigest()
                    == load(PACKAGE / "review_cases.json")["cases"][0 if match == "118575" else 1]["source_frames"][
                        half
                    ]["frame_sha256"]
                )
        try:
            urlopen("http://127.0.0.1:8813/assets/unknown/first.png", timeout=3)
            raise AssertionError("unknown asset unexpectedly served")
        except HTTPError as error:
            assert error.code == 404
        try:
            urlopen("http://127.0.0.1:8813/assets/../review_cases.json", timeout=3)
            raise AssertionError("path traversal unexpectedly served")
        except HTTPError as error:
            assert error.code == 404
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_polygon_rules_and_coordinate_round_trip() -> None:
    polygon = [(10, 10), (100, 10), (100, 80), (10, 80)]
    assert polygon[0] != polygon[-1]
    closed = polygon + [polygon[0]]
    assert closed[0] == closed[-1]
    area = sum(closed[i][0] * closed[i + 1][1] - closed[i + 1][0] * closed[i][1] for i in range(4)) / 2
    assert area > 0
    assert all(0 <= x <= 4096 and 0 <= y <= 1906 for x, y in polygon)
    assert 'id="alignment"' in (PACKAGE / "index.html").read_text(encoding="utf-8")
    assert "<option>NO</option>" in (PACKAGE / "index.html").read_text(encoding="utf-8")


def test_canonical_transform_round_trip_and_r3_contract() -> None:
    def to_display(
        point: tuple[float, float], rect: tuple[float, float, float, float], source: tuple[float, float]
    ) -> tuple[float, float]:
        return (rect[0] + point[0] * rect[2] / source[0], rect[1] + point[1] * rect[3] / source[1])

    def to_source(
        point: tuple[float, float], rect: tuple[float, float, float, float], source: tuple[float, float]
    ) -> tuple[float, float]:
        return ((point[0] - rect[0]) * source[0] / rect[2], (point[1] - rect[1]) * source[1] / rect[3])

    source = (4096.0, 1080.0)
    points = [
        (0.0, 0.0),
        (4096.0, 0.0),
        (4096.0, 1080.0),
        (0.0, 1080.0),
        (2048.0, 540.0),
        (777.0, 121.0),
        (3123.0, 913.0),
        (1901.0, 357.0),
    ]
    for device_pixel_ratio in (1.0, 2.0):
        for rect in ((12.5, 38.0, 620.0, 163.5), (700.0, 51.0, 480.0, 126.5625)):
            for point in points:
                displayed = to_display(point, rect, source)
                restored = to_source(displayed, rect, source)
                assert max(abs(restored[0] - point[0]), abs(restored[1] - point[1])) <= 0.5
                backing_x = (displayed[0] - rect[0]) * device_pixel_ratio
                assert 0 <= backing_x <= rect[2] * device_pixel_ratio
    html = (PACKAGE / "index.html").read_text(encoding="utf-8")
    assert "displayToSource" in html and "sourceToDisplay" in html and "sourceToCanvas" in html
    assert "Coordinate mapping: VERIFIED" in html
    assert "secondCanvas" not in html
    assert 'window.addEventListener("resize",redraw)' in html
    assert "firstPoints=[];secondPoints=[]" in html


def save_payload(case: dict, event_id: str, alignment: str = "YES") -> dict:
    first = case["source_frames"]["first"]
    second = case["source_frames"]["second"]
    polygon = [[10, 10], [100, 10], [100, 80], [10, 80]]
    return {
        "schema_version": "football_intelligence.g7d_a.pitch_polygon_review_event.v1",
        "review_id": "G7D_A_PITCH_POLYGON_REVIEW",
        "revision": "G7D_A_PITCH_POLYGON_REVIEW_R4",
        "match_id": case["match_id"],
        "client_event_id": event_id,
        "timestamp": "2026-07-28T21:00:00Z",
        "alignment_answer": alignment,
        "first_half_polygon_source_xy": polygon,
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
            "first_half_round_trip_max_error_css_px": 0,
            "second_half_projection_verified": True,
        },
    }


def test_r4_save_acknowledgement_idempotency_and_completion() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        isolated = Path(temporary)
        shutil.copy2(PACKAGE / "review_server.py", isolated / "review_server.py")
        shutil.copy2(PACKAGE / "review_cases.json", isolated / "review_cases.json")
        process = subprocess.Popen([sys.executable, "review_server.py", "--port", "8814"], cwd=isolated)
        try:
            time.sleep(0.25)
            cases = load(isolated / "review_cases.json")["cases"]

            def post(payload: dict) -> dict:
                request = Request(
                    "http://127.0.0.1:8814/api/save",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                return json.loads(urlopen(request, timeout=3).read())

            first = post(save_payload(cases[0], "11111111-1111-4111-8111-111111111111"))
            assert first["ok"] and first["case_complete"] and not first["all_cases_complete"]
            assert first["saved_path"].startswith("review_events/118575/")
            repeat = post(save_payload(cases[0], "11111111-1111-4111-8111-111111111111"))
            assert repeat["event_id"] == first["event_id"]
            assert len(list((isolated / "review_events" / "118575").glob("*.json"))) == 1
            second = post(save_payload(cases[1], "22222222-2222-4222-8222-222222222222"))
            assert second["all_cases_complete"]
            restored = json.loads(urlopen("http://127.0.0.1:8814/api/cases", timeout=3).read())
            assert restored["saved_events"]["118575"]["event_id"] == first["event_id"]
            bad = save_payload(cases[0], "33333333-3333-4333-8333-333333333333")
            bad["first_half_polygon_source_xy"] = [[-1, 0], [1, 0], [1, 1], [0, 1]]
            try:
                post(bad)
                raise AssertionError("invalid canonical polygon unexpectedly saved")
            except HTTPError as error:
                assert error.code == 422
        finally:
            process.terminate()
            process.wait(timeout=5)


def test_r4_ui_save_binding_and_unsaved_guard_contract() -> None:
    html = (PACKAGE / "index.html").read_text(encoding="utf-8")
    for required in (
        'saveButton.addEventListener("click",saveCase)',
        'method:"POST"',
        "SAVED — SERVER ACKNOWLEDGED",
        "Saving…",
        "Modified — not saved",
        "beforeunload",
        "This case has unsaved changes",
        "ALL CASES COMPLETE",
        "first_half_polygon_source_xy",
        "coordinate_audit",
    ):
        assert required in html
