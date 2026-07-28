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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g7d_a_polygon_validation import normalize_client_vertices, validate_canonical_polygon


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


def dense_pitch_polygon() -> list[list[float]]:
    top = [[300 + 340 * index, 180 + (index % 2)] for index in range(11)]
    right = [[3700 + (index % 2), 260 + 95 * index] for index in range(8)]
    bottom = [[3600 - 330 * index, 980 - (index % 2)] for index in range(10)]
    left = [[300 - (index % 2), 880 - 100 * index] for index in range(8)]
    polygon = top + right + bottom + left
    assert len(polygon) == 37
    return polygon


def save_payload(case: dict, event_id: str, alignment: str = "YES", polygon: list[list[float]] | None = None) -> dict:
    first = case["source_frames"]["first"]
    second = case["source_frames"]["second"]
    polygon = polygon or [[10, 10], [100, 10], [100, 80], [10, 80]]
    return {
        "schema_version": "football_intelligence.g7d_a.pitch_polygon_review_event.v1",
        "review_id": "G7D_A_PITCH_POLYGON_REVIEW",
        "revision": "G7D_A_PITCH_POLYGON_REVIEW_R5",
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
        "normalization": {
            "closure_convention": "distinct_vertices_once_plus_closed_true",
            "removed_exact_adjacent_vertex_indices": [],
            "removed_exact_terminal_duplicate": False,
        },
    }


def test_r4_save_acknowledgement_idempotency_and_completion() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        isolated = Path(temporary)
        shutil.copy2(PACKAGE / "review_server.py", isolated / "review_server.py")
        shutil.copy2(PACKAGE / "review_cases.json", isolated / "review_cases.json")
        shutil.copy2(PACKAGE / "polygon_validation.py", isolated / "polygon_validation.py")
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

            first = post(save_payload(cases[0], "11111111-1111-4111-8111-111111111111", polygon=dense_pitch_polygon()))
            assert first["ok"] and first["case_complete"] and not first["all_cases_complete"]
            assert first["saved_path"].startswith("review_events/118575/")
            assert first["receipt_id"].startswith("ack-")
            assert (isolated / first["receipt_path"]).exists()
            repeat = post(save_payload(cases[0], "11111111-1111-4111-8111-111111111111", polygon=dense_pitch_polygon()))
            assert repeat["event_id"] == first["event_id"]
            assert len(list((isolated / "review_events" / "118575").glob("*.json"))) == 1
            second = post(save_payload(cases[1], "22222222-2222-4222-8222-222222222222", polygon=dense_pitch_polygon()))
            assert second["all_cases_complete"]
            assert second["completion_receipt_id"].startswith("completion-")
            assert (isolated / second["completion_receipt_path"]).exists()
            restored = json.loads(urlopen("http://127.0.0.1:8814/api/cases", timeout=3).read())
            assert restored["saved_events"]["118575"]["event_id"] == first["event_id"]
            bad = save_payload(cases[0], "33333333-3333-4333-8333-333333333333")
            bad["first_half_polygon_source_xy"] = [[-1, 0], [1, 0], [1, 1], [0, 1]]
            try:
                post(bad)
                raise AssertionError("invalid canonical polygon unexpectedly saved")
            except HTTPError as error:
                assert error.code == 422
                assert json.loads(error.read())["error_code"] == "OUT_OF_BOUNDS_VERTEX"
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
        "normalizeVertices",
        "Outgoing canonical payload:",
        "Server validation:",
    ):
        assert required in html


def test_r5_canonical_closure_and_field_level_geometry_validation() -> None:
    dense = dense_pitch_polygon()
    terminal_duplicate, metadata = normalize_client_vertices(dense + [dense[0]], closed=True)
    assert terminal_duplicate == dense
    assert metadata["removed_exact_terminal_duplicate"]
    adjacent_duplicate, metadata = normalize_client_vertices(dense[:8] + [dense[7]] + dense[8:], closed=True)
    assert len(adjacent_duplicate) == len(dense)
    assert metadata["removed_exact_adjacent_vertex_indices"] == [8]
    accepted = validate_canonical_polygon(dense, True, 4096, 1080)
    assert accepted["ok"] and accepted["details"]["vertex_count"] == 37
    crossing = validate_canonical_polygon([[0, 0], [100, 100], [0, 100], [100, 0]], True, 4096, 1080)
    assert crossing["error_code"] == "SELF_INTERSECTION"
    duplicate = validate_canonical_polygon([[0, 0], [100, 0], [100, 0], [0, 100]], True, 4096, 1080)
    assert duplicate["error_code"] == "DUPLICATE_OR_ZERO_LENGTH_EDGE"
    terminal = validate_canonical_polygon([[0, 0], [100, 0], [100, 100], [0, 0]], True, 4096, 1080)
    assert terminal["error_code"] == "DUPLICATE_TERMINAL_VERTEX"
    assert validate_canonical_polygon([[0, 0], [100, 0], [100, 100], [0, 200]], True, 4096, 1080)["ok"]
    assert (
        validate_canonical_polygon([[0, 0], [100, 0], [100, 100], [0, float("inf")]], True, 4096, 1080)["error_code"]
        == "NON_FINITE_VERTEX"
    )
    assert (
        validate_canonical_polygon([[0, 0], [100, 0], [100, 100], [-1, 10]], True, 4096, 1080)["error_code"]
        == "OUT_OF_BOUNDS_VERTEX"
    )
    assert (
        validate_canonical_polygon([[0, 0], [100, 0], [200, 0], [300, 0]], True, 4096, 1080)["error_code"]
        == "ZERO_AREA"
    )


def test_r6a_append_only_receipts_and_event_immutability() -> None:
    receipts = WORKSPACE / "06_PITCH_POLYGON_REVIEW_PACKAGE" / "review_receipts"
    events = PACKAGE / "review_events"
    assert sorted(path.name for path in receipts.glob("event_acknowledgements/*.json")) == [
        "117092.json",
        "118575.json",
    ]
    completion = load(receipts / "completion" / "final.json")
    assert completion["required_match_ids"] == ["118575", "117092"]
    assert completion["all_cases_complete"] is True
    assert len(completion["acknowledgement_receipts"]) == 2
    for match_id in ("118575", "117092"):
        event = next(events.joinpath(match_id).glob("*.json"))
        receipt = load(receipts / "event_acknowledgements" / f"{match_id}.json")
        assert receipt["human_event_sha256"] == hashlib.sha256(event.read_bytes()).hexdigest()
        assert receipt["case_complete"] is True
        assert receipt["creation_reason"] == "AUTHORIZED_POST_HOC_ACKNOWLEDGEMENT_RECEIPT_BACKFILL"
        assert "synthetic" not in event.read_text(encoding="utf-8").lower()
    assert not (ROOT / "matches" / "118575" / "calibration" / "pitch_polygon_v1" / "pitch_polygon.json").exists()
    assert not (ROOT / "matches" / "117092" / "calibration" / "pitch_polygon_v1" / "pitch_polygon.json").exists()
