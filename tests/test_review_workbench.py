from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from football_intelligence.review.evidence import build_visual_continuity_workbench
from football_intelligence.review.persistence import (
    ReviewPersistence,
    reconstruct_state_from_events,
)
from football_intelligence.review.schemas import (
    ALLOWED_REVIEW_TASK_TYPES,
    CONTINUITY_DECISIONS,
    CONTINUITY_QUESTION,
    ReviewCase,
    ReviewManifest,
    safety_payload,
)
from football_intelligence.review.server import ReviewServerConfig, create_server, load_manifest
from football_intelligence.review.validation import seal_completion, validate_review_package
from football_intelligence.review.workbench import KEYBOARD_SHORTCUTS, build_workbench


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_frame(path: Path, value: int) -> None:
    image = np.full((96, 160, 3), value, dtype=np.uint8)
    cv2.rectangle(image, (20 + value // 10, 20), (45 + value // 10, 70), (255, 255, 255), -1)
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok


def _build_fixture(tmp_path: Path, case_count: int = 3) -> dict[str, Path]:
    artifact_root = tmp_path / "artifacts"
    source_stage = artifact_root / "matches/128058/runs/step_m5/06a_detector_dependency_recovery"
    run_a = source_stage / "runs/portable_real_run_a"
    frame_root = artifact_root / "matches/128058/runs/step_m5/05_blind_second_window/frames/extraction_a"
    frame_root.mkdir(parents=True)
    frames = []
    for sequence in range(case_count + 2):
        path = frame_root / f"frame_{sequence:06d}.jpg"
        _write_frame(path, 30 + sequence * 20)
        frames.append(
            {
                "sequence": sequence,
                "filename": path.name,
                "relative_uri": path.name,
                "width": 160,
                "height": 96,
                "byte_sha256": _sha(path),
                "decoded_pixel_sha256": f"decoded_{sequence}",
            }
        )
    frame_manifest = _write_json(
        frame_root / "frame_manifest.json",
        {"actual_frame_count": len(frames), "frames": frames},
    )
    visible_rows = []
    candidate_rows = []
    for index in range(case_count):
        source_id = f"source_vpb_{index}"
        target_id = f"target_vpb_{index}"
        visible_rows.extend(
            [
                {
                    "visible_person_base_id": source_id,
                    "frame_sequence": index,
                    "bbox": {"x1": 20.0, "y1": 20.0, "x2": 46.0, "y2": 72.0},
                },
                {
                    "visible_person_base_id": target_id,
                    "frame_sequence": index + 1,
                    "bbox": {"x1": 26.0, "y1": 20.0, "x2": 52.0, "y2": 72.0},
                },
            ]
        )
        candidate_rows.append(
            {
                "portable_review_candidate_id": f"portable_review_{index + 1:03d}",
                "continuity_edge_id": f"edge_{index}",
                "source_frame_sequence": index,
                "target_frame_sequence": index + 1,
                "source_visible_person_base_id": source_id,
                "target_visible_person_base_id": target_id,
                "review_category": "continuity_ambiguity",
                "review_bucket": "team_colour_ambiguity",
                "uncertainty_reasons": ["frame_gap_penalty", "low_crop_quality"],
                "prefilled_acceptance": False,
                "visual_continuity_edge_is_identity": False,
                "visual_continuity_edge_is_metric": False,
                "visual_continuity_edge_is_player_slot": False,
            }
        )
    visible_path = _write_json(
        run_a / "step1/step1b4_visible_person_base_rows.json",
        {"artifact": "visible", "rows": visible_rows},
    )
    candidate_path = _write_json(
        source_stage / "review/blind_review_candidate_rows.json",
        {"artifact": "candidates", "rows": candidate_rows},
    )
    stage_root = artifact_root / "matches/128058/runs/step_m5/06b_unified_review_workbench"
    result = build_visual_continuity_workbench(
        stage_root=stage_root,
        source_stage_root=source_stage,
        frame_manifest_path=frame_manifest,
        frame_root=frame_root,
        candidate_rows_path=candidate_path,
        visible_person_base_path=visible_path,
    )
    build_workbench(Path(result["workbench_root"]))
    return {
        "stage_root": stage_root,
        "manifest": stage_root / "review/review_manifest.json",
        "evidence_root": stage_root / "review/evidence",
        "decision_root": stage_root / "review/decisions",
        "workbench_root": stage_root / "review/workbench",
    }


def _post_json(port: int, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(port: int, path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def test_review_case_schema_and_safety_reject_identity_payload(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path, case_count=1)
    manifest = load_manifest(paths["manifest"])
    case = manifest.review_cases[0]
    assert case.concise_question == CONTINUITY_QUESTION
    assert case.allowed_decisions == CONTINUITY_DECISIONS
    assert "visual_continuity_edge_review" in ALLOWED_REVIEW_TASK_TYPES
    payload = case.model_dump(mode="json")
    payload["safety_payload"] = {**safety_payload(), "identity_id": "forbidden"}
    with pytest.raises(ValueError, match="forbidden"):
        ReviewCase.model_validate(payload)


def test_visual_evidence_links_temporal_media_and_no_decision_prefill(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path, case_count=2)
    manifest = load_manifest(paths["manifest"])
    assert len(manifest.review_cases) == 2
    state = _read_json(paths["decision_root"] / "review_decisions.json")
    assert state["decisions"] == {}
    for case in manifest.review_cases:
        assets = case.evidence_manifest.evidence_assets
        assert case.evidence_manifest.temporal_evidence_available is True
        assert any(asset.media_type in {"video/mp4", "image/gif"} for asset in assets)
        assert {asset.asset_id for asset in assets} >= {
            "source_crop",
            "target_crop",
            "source_context",
            "target_context",
            "side_by_side",
            "temporal_strip",
        }
        for asset in assets:
            assert (paths["evidence_root"] / case.review_case_id / asset.relative_path).exists()


def test_durable_autosave_events_snapshots_resume_undo_and_completion(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path, case_count=3)
    manifest = load_manifest(paths["manifest"])
    persistence = ReviewPersistence(
        manifest=manifest, decision_root=paths["decision_root"], reviewer_session_id="tester"
    )
    cases = manifest.review_cases

    state = persistence.save_decision(review_case_id=cases[0].review_case_id, decision="accept_continuity")
    assert _read_json(paths["decision_root"] / "review_decisions.json")["decisions"][cases[0].review_case_id]
    persistence.save_note(review_case_id=cases[0].review_case_id, note="clear enough")
    persistence.save_decision(review_case_id=cases[1].review_case_id, decision="reject_continuity")
    persistence.save_decision(review_case_id=cases[2].review_case_id, decision="unresolved")
    events_before_undo = (paths["decision_root"] / "review_decision_events.jsonl").read_text(encoding="utf-8")
    assert len([line for line in events_before_undo.splitlines() if line.strip()]) == 4
    assert list((paths["decision_root"] / "snapshots").glob("review_state_*.json"))

    state = persistence.undo()
    assert cases[2].review_case_id not in state["decisions"]
    events_after_undo = (paths["decision_root"] / "review_decision_events.jsonl").read_text(encoding="utf-8")
    assert len(events_after_undo.splitlines()) == len(events_before_undo.splitlines()) + 1

    reconstructed = reconstruct_state_from_events(
        manifest=manifest,
        event_log_path=paths["decision_root"] / "review_decision_events.jsonl",
        reviewer_session_id="tester",
    )
    assert reconstructed["decisions"][cases[0].review_case_id] == "accept_continuity"
    assert reconstructed["notes"][cases[0].review_case_id] == "clear enough"

    persistence.save_decision(review_case_id=cases[2].review_case_id, decision="unresolved")
    completed = persistence.complete(elapsed_active_seconds=123)
    assert completed["completed"] is True
    summary = _read_json(paths["decision_root"] / "completed_review_summary.json")
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["unresolved"] == 1
    assert summary["review_duration"] == 123
    assert summary["human_approved"] is False


def test_local_server_decisions_survive_restart_export_and_seal(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path, case_count=3)
    config = ReviewServerConfig(
        manifest_path=paths["manifest"],
        evidence_root=paths["evidence_root"],
        decision_root=paths["decision_root"],
        workbench_root=paths["workbench_root"],
        port=0,
        reviewer_session_id="server-test",
    )
    server = create_server(config)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        manifest = _get_json(port, "/api/review/manifest")
        ids = [case["review_case_id"] for case in manifest["review_cases"]]
        _post_json(port, "/api/review/decision", {"review_case_id": ids[0], "decision": "accept_continuity"})
        _post_json(port, "/api/review/decision", {"review_case_id": ids[1], "decision": "reject_continuity"})
        _post_json(port, "/api/review/decision", {"review_case_id": ids[2], "decision": "unresolved"})
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()

    restarted = create_server(config)
    port = restarted.server_port
    thread = threading.Thread(target=restarted.serve_forever, daemon=True)
    thread.start()
    try:
        state = _get_json(port, "/api/review/state")
        assert state["counts"]["reviewed"] == 3
        exported = _get_json(port, "/api/review/export")
        assert exported["summary"]["accepted"] == 1
    finally:
        restarted.shutdown()
        thread.join(timeout=10)
        restarted.server_close()

    sealed = seal_completion(
        manifest_path=paths["manifest"],
        decision_root=paths["decision_root"],
        reviewer_session_id="server-test",
    )
    assert sealed["sealed"] is True


def test_workbench_static_assets_include_shortcuts_fallback_and_save_failure(tmp_path: Path) -> None:
    root = tmp_path / "workbench"
    manifest = build_workbench(root)
    app_js = (root / "app.js").read_text(encoding="utf-8")
    fallback_html = (root / "fallback.html").read_text(encoding="utf-8")
    assert manifest["raw_json_primary_interface"] is False
    assert KEYBOARD_SHORTCUTS["A"] == "accept_continuity"
    assert "Save failed" in app_js
    assert "Saving..." in app_js
    assert "fetch" in app_js
    assert "Browser-only recovery mode - decisions are not yet durably saved to the project." in fallback_html


def test_validation_and_binding_mismatch_rejections(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path, case_count=1)
    result = validate_review_package(
        manifest_path=paths["manifest"],
        evidence_root=paths["evidence_root"],
        decision_root=paths["decision_root"],
    )
    assert result["passed"] is True
    state_path = paths["decision_root"] / "review_decisions.json"
    state = _read_json(state_path)
    state["candidate_manifest_hash"] = "wrong"
    _write_json(state_path, state)
    with pytest.raises(ValueError, match="candidate manifest hash mismatch"):
        seal_completion(
            manifest_path=paths["manifest"],
            decision_root=paths["decision_root"],
            reviewer_session_id="tester",
        )
    manifest_payload = _read_json(paths["manifest"])
    manifest_payload["review_cases"][0]["evidence_hash"] = "wrong"
    tampered = _write_json(tmp_path / "tampered_manifest.json", manifest_payload)
    with pytest.raises(ValueError, match="evidence_hash"):
        ReviewManifest.model_validate(_read_json(tampered))
