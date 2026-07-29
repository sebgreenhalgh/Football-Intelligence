"""Focused static, route, persistence, and browser-evidence checks for C1 R4."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import threading
import urllib.request
from pathlib import Path

from football_intelligence.g7d_c1_r4_stable_review import REVISION, StableBootReviewStore, create_server

ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT.parent / "experiments/football_observation_reasoner/part 6/G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
)
PACKAGE = STAGE / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
EVIDENCE = STAGE / "11_R4_UNDEFINED_HANDLER_RUNTIME_BOOT_REPAIR"
HANDOFF = STAGE / "12_R4_REVIEW_PACK/CHATGPT_HANDOFF"
EXPECTED_HEAD = "0c679af55381277620db59325f0074ef1b5fd762"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_expected_head_and_frozen_inputs_are_unchanged() -> None:
    assert (
        subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        == EXPECTED_HEAD
    )
    document = read(PACKAGE / "review_cases.json")
    assert document["review_revision"] == REVISION
    assert len(document["cases"]) == 24
    assert sum(len(case["targets"]) for case in document["cases"]) == 192
    assert len({target["target_id"] for case in document["cases"] for target in case["targets"]}) == 192
    for case in document["cases"]:
        assert sha256(PACKAGE / "assets" / case["asset_name"]) == case["frame_sha256"]
    mapping = read(PACKAGE / "target_box_calibration_status.json")
    assert mapping["verified"] is True and mapping["target_count"] == 192 and mapping["failure_count"] == 0


def test_root_cause_callback_preflight_and_scoping() -> None:
    root_cause = read(EVIDENCE / "ROOT_CAUSE.json")
    assert root_cause["classification"] == "RUNTIME_BINDING_ERROR"
    assert root_cause["declaration"]["r3_markMissedPerson_definition_count"] == 0
    assert root_cause["not_asset_or_calibration_failure"] is True
    audit = read(EVIDENCE / "callback_binding_audit.json")
    assert audit["unresolved_callback_count"] == 0
    assert audit["conflicting_duplicate_callback_count"] == 0
    assert audit["candidate_mode_eager_scene_binding"] is False
    assert audit["preflight_before_case_loading"] is True
    app = (PACKAGE / "app.js").read_text(encoding="utf-8")
    assert app.count("function markMissedPerson(") == 1
    assert app.count("function enterMissedPersonMode(") == 1
    assert app.count("function addMissedPersonMark(") == 1
    assert app.count("function removeMissedPersonMark(") == 1
    assert app.count("function exitMissedPersonMode(") == 1
    assert app.count("start();") == 1
    assert app.index("verifyRuntimeBindings()") < app.index('getJson("/api/cases")')
    for code in (
        "RUNTIME_BINDING_ERROR",
        "CASE_API_ERROR",
        "ASSET_LOAD_ERROR",
        "MAPPING_ERROR",
        "QUESTION_INITIALIZATION_ERROR",
    ):
        assert code in app


def test_all_bounded_routes_and_r4_draft_restoration() -> None:
    audit = read(EVIDENCE / "asset_route_audit.json")
    assert audit["scene_count"] == 24 and audit["target_count"] == 192
    assert audit["asset_url_count"] == 576 and audit["asset_failures"] == []
    assert all(row["status"] == 200 and row["mime_type"] == "image/png" and row["passed"] for row in audit["results"])

    server = create_server(PACKAGE, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/cases", timeout=10) as response:
            assert response.status == 200
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/app.js", timeout=10) as response:
            assert response.status == 200 and response.headers.get_content_type() == "text/javascript"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    with tempfile.TemporaryDirectory(prefix="g7d_c1_r4_draft_") as temporary_text:
        temporary = Path(temporary_text)
        (temporary / "review_cases.json").write_bytes((PACKAGE / "review_cases.json").read_bytes())
        (temporary / "target_box_calibration_status.json").write_bytes(
            (PACKAGE / "target_box_calibration_status.json").read_bytes()
        )
        store = StableBootReviewStore(temporary)
        case, target = store.cases[0], store.cases[0]["targets"][0]
        payload = {
            "schema_version": "football_intelligence.g7d_c1_r1.server_progress_draft.v1",
            "review_id": "G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS",
            "revision": REVISION,
            "draft_type": "candidate",
            "scene_id": case["scene_id"],
            "target_id": target["target_id"],
            "step_index": 1,
            "answers": {"proposal_validity": "CLEAN_SINGLE_PERSON"},
            "missed_people_source_xy": [],
            "idempotency_key": "r4-focused-refresh",
        }
        status, result = store.save_draft(payload)
        assert status == 200 and result["ok"] is True
        restored = StableBootReviewStore(temporary).state()["drafts"][target["target_id"]]
        assert restored["step_index"] == 1 and restored["answers"] == payload["answers"]


def test_real_browser_paths_visual_cap_and_handoff_manifest() -> None:
    browser = read(EVIDENCE / "browser_end_to_end_results.json")
    assert browser["candidate"] == {
        "all_canvases_drawn": True,
        "draft_saved": True,
        "mode": "candidate",
        "question_1_visible": True,
        "refresh_restored_question": 2,
        "uncaught_javascript_exception_count": 0,
    }
    assert browser["scene"] == {
        "draft_restored_after_refresh": True,
        "mode": "scene",
        "scene_mode_ready": True,
        "temporary_mark_added": True,
        "temporary_mark_removed": True,
        "uncaught_javascript_exception_count": 0,
    }
    previews = sorted((EVIDENCE / "visual_qa").glob("*.png"))
    assert [path.name for path in previews] == ["01_candidate_ready.png", "02_missed_person_mode.png"]
    assert len(list(HANDOFF.iterdir())) == 10
    manifest = read(HANDOFF / "10_MANIFEST.json")
    assert len(manifest["files"]) == 9
    assert "10_MANIFEST.json" not in {row["filename"] for row in manifest["files"]}
    for row in manifest["files"]:
        path = HANDOFF / row["filename"]
        assert path.stat().st_size == row["byte_size"] and sha256(path) == row["sha256"]
    result = read(EVIDENCE / "stage_result.json")
    assert result["classification"] == "PASS_G7D_C1_R4_STABLE_BOOT_NOVICE_REVIEWER_READY_FOR_HUMAN_REVIEW"
    assert result["human_review_started"] is False and result["g7d_c2_started"] is False
