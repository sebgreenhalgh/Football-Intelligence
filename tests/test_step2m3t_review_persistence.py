# ruff: noqa: E501

from __future__ import annotations

import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import football_intelligence.step2_visual_continuity.topology_sparse_pathlets as m3t  # noqa: E402
from football_intelligence.step2_visual_continuity.io import (  # noqa: E402
    STEP2M1_OUTPUT_DIR,
    STEP2M2_OUTPUT_DIR,
    STEP2M3_OUTPUT_DIR,
    STEP2M3R_OUTPUT_DIR,
    STEP2M3S_OUTPUT_DIR,
    STEP2M3T_OUTPUT_DIR,
)
from football_intelligence.step2_visual_continuity.schema import forbidden_keys_present  # noqa: E402


def review_candidate(index: int, category: str = "sparse_pathlet_boundary_review") -> dict:
    return {
        "step2m3t_review_candidate_id": f"m3t_review_{index:03d}",
        "review_subject_type": "sparse_pathlet",
        "step2m3t_review_category": category,
        "pathlet_id": f"pathlet_{index:03d}",
        "continuity_edge_id": f"edge_{index:03d}",
        "accepted_continuity_edge_ids": [f"edge_{index:03d}"],
        "min_frame_sequence": index,
        "max_frame_sequence": index + 2,
        "evidence_type": "pathlet_animation",
        "evidence_animation_gif_path": f"step2m3t_visual_evidence/pathlet_animations/pathlet_{index:03d}.gif",
        "evidence_static_strip_path": f"step2m3t_visual_evidence/pathlet_strips/pathlet_{index:03d}.jpg",
        "sampled_frame_sequences": [index, index + 1, index + 2],
        "evidence_available": True,
        "current_visual_evidence_version": m3t.M3T_CURRENT_VISUAL_EVIDENCE_VERSION,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }


def review_payload(count: int) -> dict:
    return {
        "artifact": "step2m3t_review_candidate_rows",
        "rows": [review_candidate(index) for index in range(count)],
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }


def patch_review_paths(monkeypatch, tmp_path: Path) -> Path:
    root = tmp_path / "step2m3t_sparse_pathlets"
    root.mkdir()
    monkeypatch.setattr(m3t, "STEP2M3T_OUTPUT_DIR", root)
    monkeypatch.setattr(m3t, "STEP2M3T_REVIEW_CANDIDATE_ROWS_PATH", root / "step2m3t_review_candidate_rows.json")
    monkeypatch.setattr(m3t, "STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH", root / "step2m3t_reviewed_sparse_pathlet_decisions.json")
    monkeypatch.setattr(m3t, "STEP2M3T_REVIEW_PROGRESS_SUMMARY_PATH", root / "step2m3t_review_progress_summary.json")
    monkeypatch.setattr(m3t, "STEP2M3T_REVIEW_DECISION_SUMMARY_PATH", root / "step2m3t_review_decision_summary.json")
    monkeypatch.setattr(m3t, "STEP2M3T_REVIEW_UI_HTML_PATH", root / "step2m3t_review_ui.html")
    return root


def write_review_payload(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_m3t_autosave_endpoint_writes_json(monkeypatch, tmp_path: Path) -> None:
    root = patch_review_paths(monkeypatch, tmp_path)
    write_review_payload(m3t.STEP2M3T_REVIEW_CANDIDATE_ROWS_PATH, review_payload(1))
    server = ThreadingHTTPServer(("127.0.0.1", 0), m3t.Step2M3TReviewHandler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(host, port, timeout=5)
        connection.request(
            "POST",
            "/api/step2m3t/review-decision",
            body=json.dumps(
                {
                    "step2m3t_review_candidate_id": "m3t_review_000",
                    "human_review_decision": m3t.M3T_ACCEPT_DECISION,
                }
            ),
            headers={"content-type": "application/json"},
        )
        response = connection.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert response.status == 200
    assert body["success"] is True
    assert body["reviewed_count"] == 1
    decision_path = root / "step2m3t_reviewed_sparse_pathlet_decisions.json"
    assert decision_path.exists()
    saved = json.loads(decision_path.read_text(encoding="utf-8"))
    assert len(saved["rows"]) == 1
    assert saved["rows"][0]["human_review_decision"] == m3t.M3T_ACCEPT_DECISION


def test_duplicate_m3t_saves_update_existing_row(monkeypatch, tmp_path: Path) -> None:
    patch_review_paths(monkeypatch, tmp_path)
    write_review_payload(m3t.STEP2M3T_REVIEW_CANDIDATE_ROWS_PATH, review_payload(1))
    m3t.save_m3t_review_decision({"step2m3t_review_candidate_id": "m3t_review_000", "human_review_decision": m3t.M3T_ACCEPT_DECISION})
    _decision, reviewed_payload, progress = m3t.save_m3t_review_decision(
        {"step2m3t_review_candidate_id": "m3t_review_000", "human_review_decision": m3t.M3T_REJECT_DECISION}
    )
    assert progress["reviewed_candidates"] == 1
    assert len(reviewed_payload["rows"]) == 1
    assert reviewed_payload["rows"][0]["human_review_decision"] == m3t.M3T_REJECT_DECISION


def test_local_storage_only_m3t_decisions_do_not_count(monkeypatch, tmp_path: Path) -> None:
    patch_review_paths(monkeypatch, tmp_path)
    payload = review_payload(1)
    progress = m3t.m3t_review_progress_payload(payload, m3t.read_m3t_reviewed_decisions())
    assert progress["persisted_review_decision_file_exists"] is False
    assert progress["reviewed_candidates"] == 0
    assert progress["sparse_pathlet_review_completed"] is False


def test_m3t_decision_schema_contains_safety_fields() -> None:
    candidate = review_candidate(0)
    row = m3t.m3t_review_decision_row(candidate, {"human_review_decision": m3t.M3T_ACCEPT_DECISION})
    assert row["visual_only_warning"] == "VISUAL_ONLY_NOT_METRIC"
    assert row["do_not_use_for_metrics"] is True
    assert row["match_local_only"] is True
    assert row["safe_to_apply_globally"] is False
    assert row["requires_future_match_validation"] is True
    assert row["production_ready"] is False
    assert row["no_auto_promotion"] is True
    assert row["human_approved"] is False
    assert row["approve_any_identity_tracking"] is False
    assert row["approve_any_player_slot_use"] is False
    assert row["approve_any_goalkeeper_slot_use"] is False
    assert row["approve_any_metric_use"] is False
    assert row["approve_event_or_tactical_analysis"] is False
    assert row["approve_exact_22_or_exact_two_goalkeeper_forcing"] is False
    assert row["approve_official_referee_exclusion"] is False
    assert row["approve_bad_detection_deletion"] is False
    assert row["approve_production_promotion"] is False
    assert forbidden_keys_present(row) == []


def test_completed_40_of_40_sets_sparse_pathlet_review_completed() -> None:
    payload = review_payload(40)
    rows = [
        m3t.m3t_review_decision_row(candidate, {"human_review_decision": m3t.M3T_ACCEPT_DECISION})
        for candidate in payload["rows"]
    ]
    progress = m3t.m3t_review_progress_payload(payload, {"rows": rows})
    assert progress["total_review_candidates"] == 40
    assert progress["reviewed_candidates"] == 40
    assert progress["sparse_pathlet_review_completed"] is True
    assert progress["review_decisions_visual_evidence_version_matches_current"] is True


def test_visual_evidence_version_mismatch_blocks_reviewed_handoff() -> None:
    payload = review_payload(1)
    row = m3t.m3t_review_decision_row(payload["rows"][0], {"human_review_decision": m3t.M3T_ACCEPT_DECISION})
    row["review_decisions_collected_with_visual_evidence_version"] = "old_visual_evidence"
    progress = m3t.m3t_review_progress_payload(payload, {"rows": [row]})
    assert progress["reviewed_candidates"] == 0
    assert progress["review_decisions_visual_evidence_version_matches_current"] is False
    assert any(error["issue_code"] == "step2m3t_visual_evidence_version_mismatch" for error in progress["validation_errors"])


def test_review_version_mismatch_blocks_reviewed_handoff() -> None:
    payload = review_payload(1)
    row = m3t.m3t_review_decision_row(payload["rows"][0], {"human_review_decision": m3t.M3T_ACCEPT_DECISION})
    row["review_decisions_collected_with_review_version"] = "old_review_version"
    progress = m3t.m3t_review_progress_payload(payload, {"rows": [row]})
    assert progress["reviewed_candidates"] == 0
    assert progress["review_decisions_version_matches_current"] is False
    assert any(error["issue_code"] == "step2m3t_review_version_mismatch" for error in progress["validation_errors"])


def test_m3t_review_output_paths_remain_isolated() -> None:
    m3t_root = STEP2M3T_OUTPUT_DIR.resolve()
    blocked = [
        STEP2M1_OUTPUT_DIR.resolve(),
        STEP2M2_OUTPUT_DIR.resolve(),
        STEP2M3_OUTPUT_DIR.resolve(),
        STEP2M3R_OUTPUT_DIR.resolve(),
        STEP2M3S_OUTPUT_DIR.resolve(),
    ]
    assert m3t.STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH.resolve() == m3t_root / "step2m3t_reviewed_sparse_pathlet_decisions.json"
    for path in m3t.step2m3t_output_paths().values():
        resolved = path.resolve()
        assert resolved == m3t_root or m3t_root in resolved.parents
        assert all(resolved != root and root not in resolved.parents for root in blocked)
