"""Focused runtime initialization checks for the bounded C1 R3 reviewer."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

from football_intelligence.g7d_c1_r3_loaded_review import REVISION, LoadedReviewStore, create_server

EXPECTED_HEAD = "3734a2c2021bcefe3667d1c08e85440e56b693b8"
ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parent / (
    "experiments/football_observation_reasoner/part 6/G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
)
PACKAGE = STAGE / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
EVIDENCE = STAGE / "09_R3_BLANK_ASSETS_AND_WIZARD_INITIALIZATION_REPAIR"
HANDOFF = STAGE / "10_R3_REVIEW_PACK/CHATGPT_HANDOFF"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def fetch(base: str, path: str, method: str = "GET") -> tuple[int, str, bytes, dict[str, str]]:
    try:
        request = urllib.request.Request(f"{base}{path}", method=method)
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, response.headers.get_content_type(), response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get_content_type(), error.read(), dict(error.headers.items())


def test_expected_head_frozen_inputs_and_r2_calibration_are_preserved() -> None:
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == EXPECTED_HEAD
    cases = read(PACKAGE / "review_cases.json")
    preservation = read(EVIDENCE / "INPUT_PRESERVATION.json")
    assert cases["review_revision"] == REVISION
    assert len(cases["cases"]) == 24
    assert sum(len(case["targets"]) for case in cases["cases"]) == 192
    assert preservation["classification"] == "PASS"
    assert preservation["selection_sha256_before"] == preservation["selection_sha256_after"]
    assert preservation["frames_candidate_ids_source_boxes_and_selection_reasons_unchanged"] is True
    for case in cases["cases"]:
        assert sha256(PACKAGE / "assets" / case["asset_name"]) == case["frame_sha256"]
    mapping = read(PACKAGE / "target_box_calibration_status.json")
    assert mapping["review_revision"] == "G7D_C1_R2_CALIBRATED_TARGET_BOX_NOVICE_REVIEW_V1"
    assert mapping["verified"] is True and mapping["target_count"] == 192 and mapping["failure_count"] == 0


def test_root_cause_finite_state_machine_and_ready_gating_are_explicit() -> None:
    cause = read(EVIDENCE / "ROOT_CAUSE.json")
    app = (PACKAGE / "app.js").read_text(encoding="utf-8")
    contract = read(PACKAGE / "reviewer_contract.json")
    assert cause["classification"] == "PROVEN_MISSING_CALIBRATION_STATIC_ROUTE_CAUSING_UNCAUGHT_CLIENT_REFERENCE_ERROR"
    assert cause["observed_http"]["calibration_js_before_repair"] == 404
    required_states = {
        "BOOTING",
        "LOADING_CASE_LIST",
        "LOADING_SCENE",
        "LOADING_TARGET",
        "LOADING_IMAGES",
        "VERIFYING_MAPPING",
        "READY_FOR_QUESTION",
        "SAVING_DRAFT",
        "SAVING_FINAL",
        "ERROR",
    }
    assert set(contract["runtime_states"]) == required_states
    assert all(state in app for state in required_states)
    assert "if (!mappingReady()) { blockedScreen(); return; }" in app
    assert "What is inside the highlighted box?" in app
    assert "browserImage" in app and "/api/cases" in app and "/api/targets/" in app
    assert "TargetBoxCalibration" in app and "drawCropFrame" in app and "#ffcf33" in app


def test_bounded_case_detail_and_three_asset_routes() -> None:
    server = create_server(PACKAGE, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, _, body, _ = fetch(base, "/api/cases")
        cases = json.loads(body)
        assert status == 200 and len(cases["cases"]) == 24
        first = cases["cases"][0]
        status, _, body, _ = fetch(base, f"/api/scenes/{first['scene_id']}")
        scene = json.loads(body)["scene"]
        assert status == 200 and len(scene["targets"]) == 8
        status, _, body, _ = fetch(base, f"/api/targets/{scene['targets'][0]['target_id']}")
        detail = json.loads(body)
        assert status == 200 and detail["scene_id"] == first["scene_id"]
        for logical_asset, descriptor in detail["assets"].items():
            status, mime, _body, headers = fetch(base, descriptor["url"], "HEAD")
            assert status == 200 and mime == "image/png"
            assert int(headers["Content-Length"]) == descriptor["byte_size"]
            assert headers["X-Review-Asset-SHA256"] == descriptor["sha256"]
            assert headers["X-Review-Logical-Asset"] == logical_asset
        status, mime, _body, _headers = fetch(base, "/calibration.js")
        assert status == 200 and mime == "text/javascript"
        assert fetch(base, "/api/assets/../x/whole_frame")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_all_route_audit_stale_draft_filtering_and_handoff() -> None:
    audit = read(EVIDENCE / "asset_and_initialization_audit.json")
    assert audit["review_revision"] == REVISION
    assert audit["scene_count"] == 24 and audit["target_count"] == 192 and audit["asset_url_count"] == 576
    assert audit["asset_failures"] == [] and audit["all_asset_urls_pass"] is True
    assert audit["calibration_script_status"] == 200 and audit["traversal_status"] == 404
    with tempfile.TemporaryDirectory() as temporary_name:
        temporary = Path(temporary_name)
        (temporary / "review_progress/candidate").mkdir(parents=True)
        (temporary / "review_cases.json").write_bytes((PACKAGE / "review_cases.json").read_bytes())
        (temporary / "target_box_calibration_status.json").write_bytes(
            (PACKAGE / "target_box_calibration_status.json").read_bytes()
        )
        (temporary / "review_progress/candidate/s01t01.json").write_text(
            json.dumps(
                {
                    "revision": "G7D_C1_R2_CALIBRATED_TARGET_BOX_NOVICE_REVIEW_V1",
                    "target_id": "s01t01",
                    "scene_id": "scene_01_118575_118575_first_half_13",
                    "step_index": 0,
                }
            ),
            encoding="utf-8",
        )
        state = LoadedReviewStore(temporary).state()
        assert state["drafts"] == {} and state["discarded_stale_draft_count"] == 1
    manifest = read(HANDOFF / "10_MANIFEST.json")
    assert len(list(HANDOFF.iterdir())) == 10 and len(manifest["files"]) == 9
    assert all(row["filename"] != "10_MANIFEST.json" for row in manifest["files"])
    for row in manifest["files"]:
        file = HANDOFF / row["filename"]
        assert file.is_file() and file.stat().st_size == row["byte_size"] and sha256(file) == row["sha256"]
    assert len(list((EVIDENCE / "visual_qa").glob("*.png"))) == 2
    result = read(EVIDENCE / "stage_result.json")
    assert result["human_review_started"] is False and result["g7d_c2_started"] is False
