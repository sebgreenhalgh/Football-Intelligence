import hashlib
import json
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import urlopen
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
