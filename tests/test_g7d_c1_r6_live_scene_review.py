from __future__ import annotations

import hashlib
import json
import threading
import urllib.request
from pathlib import Path

from PIL import Image, ImageStat

from football_intelligence.g7d_c1_r6_live_scene_review import REVISION, LiveSceneReviewStore, create_server

ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT.parent
    / "experiments/football_observation_reasoner/part 6"
    / "G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
)
PACKAGE = STAGE / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
EVIDENCE = STAGE / "15_R6_LIVE_FULL_FRAME_SCENE_REVIEW_REPAIR"
HANDOFF = STAGE / "16_R6_REVIEW_PACK/CHATGPT_HANDOFF"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_inputs_events_and_real_b3_overlays_are_preserved() -> None:
    document = load(PACKAGE / "review_cases.json")
    preservation = load(EVIDENCE / "EVENT_PRESERVATION.json")
    overlays = load(PACKAGE / "scene_candidate_overlays.json")
    assert document["review_revision"] == REVISION
    assert len(document["cases"]) == 24
    assert sum(len(scene["targets"]) for scene in document["cases"]) == 192
    assert preservation["candidate_event_count"] == 8
    for row in preservation["events"]:
        assert file_hash(PACKAGE / "review_events/candidate" / row["event_filename"]) == row["event_sha256"]
        assert (
            file_hash(PACKAGE / "review_receipts/acknowledgements" / row["receipt_filename"]) == row["receipt_sha256"]
        )
    assert overlays["scene_count"] == 24
    assert overlays["scenes"][0]["candidate_count"] == 43
    assert all(scene["candidate_count"] == len(scene["candidates"]) for scene in overlays["scenes"])


def test_live_package_has_one_scene_surface_real_gate_and_question_controls() -> None:
    index = (PACKAGE / "index.html").read_text(encoding="utf-8")
    app = (PACKAGE / "app.js").read_text(encoding="utf-8")
    assert index.count('id="sceneReviewSurface"') == 1
    assert 'id="showSceneBoxes" type="checkbox" checked' in index
    assert 'id="showSceneIds" type="checkbox"' in index
    assert "verifyFrameAsset" in app
    assert "sampleScenePixels" in app
    assert "SCENE_IMAGE_NOT_VISIBLE" in app
    assert "scene_candidate_overlays" in app
    assert "Can you see anyone important who has no useful box?" in app
    assert all(label in app for label in ("No", "Yes, let me mark them", "Not sure"))
    assert "PIL" not in app and "ImageDraw" not in app


def test_live_asset_route_matches_exact_frame_hash() -> None:
    store = LiveSceneReviewStore(PACKAGE)
    detail = store.target_detail("s01t08")
    assert detail is not None
    descriptor = detail["assets"]["whole_frame"]
    server = create_server(PACKAGE, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}{descriptor['url']}", timeout=10
        ) as response:
            body = response.read()
            assert response.status == 200
            assert response.headers.get_content_type() == "image/png"
            assert response.headers["X-Review-Asset-SHA256"] == descriptor["sha256"]
            assert hashlib.sha256(body).hexdigest() == descriptor["sha256"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_real_edge_acceptance_and_three_non_blank_screenshots() -> None:
    result = load(EVIDENCE / "LIVE_BROWSER_ACCEPTANCE.json")
    assert result["classification"] == "PASS_REAL_EDGE_REAL_SERVER_REAL_SCENE"
    assert result["scene_1_acknowledged_candidates_restored"] == 8
    assert result["question_1"] == "Can you see anyone important who has no useful box?"
    assert set(result["question_1_answers"]) >= {"No", "Yes, let me mark them", "Not sure"}
    assert result["visible_overlay_count"] == result["scene_candidate_count"] == 43
    assert result["non_blank_gate"]["verified"] is True
    assert result["temporary_point_added"] and result["temporary_point_removed"]
    assert result["fullscreen_verified"] and result["draft_restored_after_refresh"]
    assert result["uncaught_javascript_exception_count"] == 0
    assert {row["viewport"] for row in result["viewport_and_dpr_results"]} == {
        "1366x768",
        "1440x900",
        "1920x1080",
    }
    assert all(row["coordinate_audit"]["passed"] for row in result["viewport_and_dpr_results"])
    screenshots = sorted((EVIDENCE / "visual_qa").glob("*.png"))
    assert len(screenshots) == 3
    for path in screenshots:
        with Image.open(path).convert("RGB") as image:
            assert image.width >= 1280
            assert sum(ImageStat.Stat(image).var) / 3 > 150


def test_r5_failure_and_exact_self_contained_handoff() -> None:
    cause = load(EVIDENCE / "ROOT_CAUSE.json")
    failure = load(EVIDENCE / "R5_ACCEPTANCE_FAILURE.json")
    assert cause["acceptance_findings"]["actual_server_started"] is False
    assert cause["acceptance_findings"]["screenshots_generated_by"].startswith("PIL")
    assert failure["classification"] == "FAIL_R5_ACCEPTANCE_INVALID"
    files = sorted(path for path in HANDOFF.iterdir() if path.is_file())
    assert len(files) == 10
    manifest = load(HANDOFF / "10_MANIFEST.json")
    assert len(manifest["files"]) == 9
    assert all(row["filename"] != "10_MANIFEST.json" for row in manifest["files"])
    for row in manifest["files"]:
        path = HANDOFF / row["filename"]
        assert path.stat().st_size == row["byte_size"]
        assert file_hash(path) == row["sha256"]
    assert load(EVIDENCE / "TESTS_SAFETY_AND_SOURCE_CHANGES.json")["expected_head"] == (
        "ca284d46de169683c44c580afacb2c8e9c9d43ac"
    )
