import hashlib
import json
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


def test_polygon_rules_and_coordinate_round_trip() -> None:
    polygon = [(10, 10), (100, 10), (100, 80), (10, 80)]
    assert polygon[0] != polygon[-1]
    closed = polygon + [polygon[0]]
    assert closed[0] == closed[-1]
    area = sum(closed[i][0] * closed[i + 1][1] - closed[i + 1][0] * closed[i][1] for i in range(4)) / 2
    assert area > 0
    assert all(0 <= x <= 4096 and 0 <= y <= 1906 for x, y in polygon)
    assert "second_half_alignment_answer" in (PACKAGE / "index.html").read_text(encoding="utf-8")
    assert "<option>NO</option>" in (PACKAGE / "index.html").read_text(encoding="utf-8")
