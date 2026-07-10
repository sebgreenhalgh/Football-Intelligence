# ruff: noqa: E501

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from http.server import ThreadingHTTPServer

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import football_intelligence.step2_visual_continuity.remediation as remediation  # noqa: E402
from football_intelligence.step2_visual_continuity.nodes import build_node_payload  # noqa: E402
from football_intelligence.step2_visual_continuity.remediation import (  # noqa: E402
    burst_frame_transform_metadata,
    burst_overlay_alignment_summary_payload,
    build_adaptation_safety_manifest,
    build_targeted_review_candidates,
    m1r_html_template,
    overlay_debug_row_valid,
    remediate_edge_for_adaptation,
    remediate_edges_for_adaptation,
    remediate_groups_for_adaptation,
    render_burst_evidence_for_candidate,
    save_m1r_review_decision,
    transform_bbox_to_rendered,
)
from test_step2m1_nodes import f3_row  # noqa: E402


def simple_edge(index: int, *, bucket: str = "safe_auto_accept_candidate", state: str = "auto_accept_candidate") -> dict:
    return {
        "continuity_edge_id": f"edge_{index:03d}",
        "source_visible_person_base_id": f"base_{index:03d}",
        "target_visible_person_base_id": f"base_{index + 1:03d}",
        "source_frame_sequence": index,
        "target_frame_sequence": index + 1,
        "frame_gap": 1,
        "edge_score_sandbox": 0.9,
        "uncertainty_score": 0.1,
        "uncertainty_reasons": [],
        "review_bucket": bucket,
        "proposed_edge_state": state,
        "final_edge_state_sandbox": "unreviewed_visual_continuity_edge",
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "no_auto_promotion": True,
    }


def m1r_candidate(index: int = 0) -> dict:
    return {
        **simple_edge(index, bucket="safe_auto_accept_audit"),
        "step2m1r_review_candidate_id": f"step2m1r_candidate_{index}",
        "review_bucket": "safe_auto_accept_audit",
        "current_overlay_version": remediation.CURRENT_BURST_OVERLAY_VERSION,
        "burst_evidence_available": True,
    }


def write_json_for_test(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def patch_m1r_review_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    paths = {
        "candidates": tmp_path / "step2m1r_targeted_review_candidate_rows.json",
        "reviewed": tmp_path / "step2m1r_reviewed_visual_continuity_decisions.json",
        "progress": tmp_path / "step2m1r_review_progress_summary.json",
        "decision": tmp_path / "step2m1r_review_decision_summary.json",
        "manifest": tmp_path / "step2m1r_adaptation_safety_manifest.json",
    }
    monkeypatch.setattr(remediation, "STEP2M1R_TARGETED_REVIEW_CANDIDATE_ROWS_PATH", paths["candidates"])
    monkeypatch.setattr(remediation, "STEP2M1R_REVIEWED_DECISIONS_PATH", paths["reviewed"])
    monkeypatch.setattr(remediation, "STEP2M1R_REVIEW_PROGRESS_SUMMARY_PATH", paths["progress"])
    monkeypatch.setattr(remediation, "STEP2M1R_REVIEW_DECISION_SUMMARY_PATH", paths["decision"])
    monkeypatch.setattr(remediation, "STEP2M1R_ADAPTATION_SAFETY_MANIFEST_PATH", paths["manifest"])
    monkeypatch.setattr(remediation, "STEP2M1R_ADAPTATION_SAFE_EDGE_ROWS_PATH", tmp_path / "missing_edges.json")
    monkeypatch.setattr(remediation, "STEP2M1R_ADAPTATION_SAFE_GROUP_ROWS_PATH", tmp_path / "missing_groups.json")
    monkeypatch.setattr(remediation, "STEP2M1R_GROUP_SPAN_REMEDIATION_SUMMARY_PATH", tmp_path / "missing_group_summary.json")
    return paths


def test_long_group_splitting_excludes_unsafe_original_and_outputs_capped_groups() -> None:
    f3_payload = {"rows": [f3_row(i, frame_sequence=frame) for i, frame in enumerate([0, 10, 20, 40, 50])]}
    node_payload = build_node_payload(f3_payload)
    edge_rows = []
    for index in range(4):
        edge_rows.append(
            {
                **simple_edge(index),
                "continuity_edge_id": f"edge_{index}_{index + 1}",
                "source_visible_person_base_id": f"base_{index:03d}",
                "target_visible_person_base_id": f"base_{index + 1:03d}",
                "final_edge_state_sandbox": "accepted_visual_continuity_edge",
            }
        )
    group_payload = {
        "rows": [
            {
                "visual_continuity_group_id": "long_group",
                "member_visible_person_base_ids": [f"base_{index:03d}" for index in range(5)],
                "accepted_continuity_edge_ids": [row["continuity_edge_id"] for row in edge_rows],
                "max_frame_span": 50,
                "max_seconds_span": 5.0,
                "group_exceeds_span_cap": True,
                "group_not_safe_for_adaptation": True,
                "production_ready": False,
                "no_auto_promotion": True,
                "human_approved": False,
            }
        ],
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }
    safe_groups, summary, _long_edges = remediate_groups_for_adaptation(group_payload, node_payload, {"rows": edge_rows})
    assert summary["groups_over_cap_before"] == 1
    assert summary["groups_over_cap_after"] == 0
    assert summary["groups_excluded_from_adaptation"] == 1
    assert safe_groups["rows"]
    assert all(row["group_not_safe_for_adaptation"] is False for row in safe_groups["rows"])
    assert all(row["max_frame_span"] <= 30 for row in safe_groups["rows"])


def test_merged_or_ambiguous_cannot_auto_accept_without_human_acceptance() -> None:
    node_payload = build_node_payload({"rows": [f3_row(0), f3_row(1)]})
    edge = simple_edge(0, bucket="merged_or_ambiguous", state="auto_accept_candidate")
    remediated = remediate_edge_for_adaptation(edge, {row["visible_person_base_id"]: row for row in node_payload["rows"]})
    assert remediated["step2m1r_remediated_proposed_edge_state"] != "auto_accept_candidate"
    assert remediated["step2m1r_adaptation_safe_positive"] is False


def test_safe_auto_accept_audit_rows_are_included_in_targeted_review() -> None:
    f3_payload = {"rows": [f3_row(i, frame_sequence=i) for i in range(25)]}
    node_payload = build_node_payload(f3_payload)
    edge_payload = {"rows": [simple_edge(i) for i in range(20)], "production_ready": False, "no_auto_promotion": True, "human_approved": False}
    _safe_payload, _safe_rows, bucket_pools = remediate_edges_for_adaptation(edge_payload, node_payload)
    review_payload = build_targeted_review_candidates(bucket_pools, set(), edge_payload)
    assert review_payload["summary"]["safe_auto_accept_audit_rows"] == 10
    assert review_payload["summary"]["targeted_review_candidate_rows"] == 10


def test_team_colour_ambiguity_alone_does_not_force_rejection_when_overlap_is_strong() -> None:
    f3_payload = {"rows": [f3_row(0), {**f3_row(1), "bbox": {"x1": 101.0, "y1": 101.0, "x2": 123.0, "y2": 171.0}, "footpoint": {"x": 112.0, "y": 171.0}}]}
    node_payload = build_node_payload(f3_payload)
    edge = {
        **simple_edge(0, bucket="team_colour_ambiguity", state="auto_reject_candidate"),
        "uncertainty_reasons": ["visual_team_context_mismatch"],
    }
    remediated = remediate_edge_for_adaptation(edge, {row["visible_person_base_id"]: row for row in node_payload["rows"]})
    assert remediated["step2m1r_remediated_proposed_edge_state"] == "needs_review_candidate"


def test_m2_adaptation_gate_remains_false_until_targeted_review_completed() -> None:
    edge_payload = {
        "adaptation_safe_edge_count": 2,
        "excluded_edge_count": 1,
        "rows": [],
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }
    group_payload = {
        "visual_continuity_group_rows": 2,
        "rows": [],
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }
    remediation_summary = {
        "groups_over_cap_after": 0,
        "groups_excluded_from_adaptation": 1,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }
    review_payload = {
        "targeted_review_completed": False,
        "summary": {"safe_auto_accept_audit_rows": 10, "burst_evidence_missing_rate": 0.0},
        "rows": [],
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }
    manifest = build_adaptation_safety_manifest(edge_payload=edge_payload, group_payload=group_payload, remediation_summary=remediation_summary, review_payload=review_payload)
    assert manifest["safe_for_step2m2_adaptation_candidate"] is False
    assert "targeted_second_review_not_completed" in manifest["unsafe_for_step2m2_adaptation_reasons"]
    assert manifest["forbidden_keys_present"] == []
    assert manifest["production_ready"] is False
    assert manifest["no_auto_promotion"] is True


def test_burst_rendering_creates_paths_and_ui_references(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    f3_payload = {"rows": [f3_row(i, frame_sequence=i) for i in range(3)]}
    node_payload = build_node_payload(f3_payload)
    frame_lookup = {}
    for frame in range(3):
        image_path = tmp_path / f"frame_{frame}.jpg"
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert cv2.imwrite(str(image_path), image)
        frame_lookup[frame] = str(image_path)
    candidate = {
        **simple_edge(0),
        "step2m1r_review_candidate_id": "step2m1r_review_test",
        "target_visible_person_base_id": "base_002",
        "target_frame_sequence": 2,
        "frame_gap": 2,
    }
    burst = render_burst_evidence_for_candidate(
        candidate,
        {row["visible_person_base_id"]: row for row in node_payload["rows"]},
        {row["frame_sequence"]: [row] for row in node_payload["rows"]},
        frame_lookup,
    )
    assert burst["burst_evidence_available"] is True
    assert burst["burst_clip_path"].endswith(".gif")
    assert burst["burst_strip_path"].endswith(".jpg")
    html = m1r_html_template({"rows": [{**candidate, **burst, "ui_assets": {"burst_clip": burst["burst_clip_path"], "burst_strip": burst["burst_strip_path"]}}]})
    assert "mini-burst evidence" in html
    assert "burst_clip" in html


def test_bbox_transform_from_original_frame_to_rendered_frame() -> None:
    metadata = burst_frame_transform_metadata(
        original_frame_width=1920,
        original_frame_height=1080,
        rendered_frame_width=960,
        rendered_frame_height=540,
    )
    transformed = transform_bbox_to_rendered({"x1": 100, "y1": 40, "x2": 300, "y2": 240}, metadata)
    assert transformed["transformed_bbox"] == {"x1": 50.0, "y1": 20.0, "x2": 150.0, "y2": 120.0}
    assert transformed["clipped"] is False
    assert transformed["bbox_transform_applied"] is True


def test_bbox_transform_with_resize_scale() -> None:
    metadata = burst_frame_transform_metadata(
        original_frame_width=1920,
        original_frame_height=1080,
        rendered_frame_width=760,
        rendered_frame_height=427,
    )
    transformed = transform_bbox_to_rendered({"x1": 960, "y1": 540, "x2": 1200, "y2": 700}, metadata)
    assert transformed["transformed_bbox"]["x1"] == 380.0
    assert transformed["transformed_bbox"]["x2"] == 475.0
    assert transformed["transformed_bbox_area"] > 0


def test_bbox_transform_with_letterbox_padding() -> None:
    metadata = burst_frame_transform_metadata(
        original_frame_width=100,
        original_frame_height=100,
        rendered_frame_width=140,
        rendered_frame_height=160,
        scale_x=1.0,
        scale_y=1.0,
        pad_x=20,
        pad_y=30,
    )
    transformed = transform_bbox_to_rendered({"x1": 10, "y1": 20, "x2": 50, "y2": 80}, metadata)
    assert transformed["transformed_bbox"] == {"x1": 30.0, "y1": 50.0, "x2": 70.0, "y2": 110.0}


def test_burst_overlay_source_and_target_boxes_only_on_their_frames(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    f3_payload = {"rows": [f3_row(i, frame_sequence=i) for i in range(3)]}
    node_payload = build_node_payload(f3_payload)
    frame_lookup = {}
    for frame in range(3):
        image_path = tmp_path / f"frame_{frame}.jpg"
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert cv2.imwrite(str(image_path), image)
        frame_lookup[frame] = str(image_path)
    candidate = {
        **simple_edge(0),
        "step2m1r_review_candidate_id": "step2m1r_review_frame_roles",
        "target_visible_person_base_id": "base_002",
        "target_frame_sequence": 2,
        "frame_gap": 2,
    }
    burst = render_burst_evidence_for_candidate(
        candidate,
        {row["visible_person_base_id"]: row for row in node_payload["rows"]},
        {row["frame_sequence"]: [row] for row in node_payload["rows"]},
        frame_lookup,
    )
    debug_rows = burst["burst_overlay_debug_rows"]
    assert all(row["frame_sequence"] == 0 for row in debug_rows if row["overlay_role"] == "source")
    assert all(row["frame_sequence"] == 2 for row in debug_rows if row["overlay_role"] == "target")


def test_intermediate_overlay_requires_candidate_id_or_interpolation_flag() -> None:
    valid_row = {
        "overlay_role": "nearby_candidate",
        "frame_sequence": 1,
        "source_frame_sequence": 0,
        "target_frame_sequence": 2,
        "actual_visible_person_base_id": "base_001",
        "interpolated_visual_aid": False,
        "interpolation_only_not_detection": False,
        "bbox_transform_applied": True,
        "coordinate_metadata": {"bbox_transform_applied": True, "rendered_frame_width": 100, "rendered_frame_height": 100},
        "transformed_bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 50},
        "transformed_bbox_area": 800,
        "footpoint_plausible": True,
    }
    invalid_row = {**valid_row, "actual_visible_person_base_id": "", "interpolated_visual_aid": False}
    interpolated_row = {**valid_row, "actual_visible_person_base_id": "", "interpolated_visual_aid": True, "interpolation_only_not_detection": True}
    assert overlay_debug_row_valid(valid_row) is True
    assert overlay_debug_row_valid(invalid_row) is False
    assert overlay_debug_row_valid(interpolated_row) is True


def test_invalid_transformed_boxes_block_overlay_review_safety() -> None:
    candidate_payload = {"rows": [{"step2m1r_review_candidate_id": "candidate_1", "source_frame_sequence": 0, "target_frame_sequence": 1}]}
    invalid_debug_row = {
        "step2m1r_review_candidate_id": "candidate_1",
        "overlay_role": "source",
        "frame_sequence": 0,
        "source_frame_sequence": 0,
        "target_frame_sequence": 1,
        "actual_visible_person_base_id": "base_000",
        "bbox_transform_applied": False,
        "coordinate_metadata": {"bbox_transform_applied": False, "rendered_frame_width": 100, "rendered_frame_height": 100},
        "transformed_bbox": {"x1": 0, "y1": 0, "x2": 0, "y2": 0},
        "transformed_bbox_area": 0,
        "footpoint_plausible": True,
        "overlay_debug_valid": False,
    }
    summary = burst_overlay_alignment_summary_payload(candidate_payload, [invalid_debug_row], [])
    assert summary["burst_overlay_alignment_safe_for_review"] is False
    assert summary["candidates_with_invalid_overlay_transforms"] == 1


def test_launcher_help_supports_m1r() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "step2m1_launch_visual_continuity_review_ui.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=True)
    assert "--m1r" in result.stdout
    assert "--sandbox" in result.stdout


def test_m1r_autosave_writes_valid_decision_and_updates_duplicate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = patch_m1r_review_paths(monkeypatch, tmp_path)
    candidate_payload = {
        "artifact": "step2m1r_targeted_review_candidate_rows",
        "summary": {"current_overlay_version": remediation.CURRENT_BURST_OVERLAY_VERSION, "burst_overlay_alignment_safe_for_review": True},
        "rows": [m1r_candidate(0)],
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }
    write_json_for_test(paths["candidates"], candidate_payload)
    first_payload = {
        "step2m1r_review_candidate_id": "step2m1r_candidate_0",
        "human_review_decision": "accept_short_window_visual_continuity_edge",
        "reviewer_name": "qa",
        "notes": "first",
    }
    _decision, reviewed, progress = save_m1r_review_decision(first_payload)
    assert reviewed["reviewed_decision_rows"] == 1
    assert progress["reviewed_candidates"] == 1
    second_payload = {**first_payload, "human_review_decision": "reject_edge", "notes": "updated"}
    _decision, reviewed, progress = save_m1r_review_decision(second_payload)
    assert reviewed["reviewed_decision_rows"] == 1
    assert reviewed["rows"][0]["human_review_decision"] == "reject_edge"
    assert reviewed["rows"][0]["notes"] == "updated"
    assert progress["rejected_count"] == 1
    assert progress["review_decisions_overlay_version_matches_current"] is True
    assert reviewed["rows"][0]["production_ready"] is False
    assert reviewed["rows"][0]["no_auto_promotion"] is True
    assert reviewed["rows"][0]["human_approved"] is False


def test_m1r_autosave_endpoint_writes_decision_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = patch_m1r_review_paths(monkeypatch, tmp_path)
    write_json_for_test(
        paths["candidates"],
        {
            "artifact": "step2m1r_targeted_review_candidate_rows",
            "summary": {"current_overlay_version": remediation.CURRENT_BURST_OVERLAY_VERSION, "burst_overlay_alignment_safe_for_review": True},
            "rows": [m1r_candidate(0)],
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), remediation.Step2M1RReviewHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/api/step2m1r/review-decision"
        request = urllib.request.Request(
            url,
            data=json.dumps(
                {
                    "step2m1r_review_candidate_id": "step2m1r_candidate_0",
                    "human_review_decision": "unsure_needs_later_review",
                    "reviewer_name": "qa",
                }
            ).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert body["success"] is True
    reviewed = json.loads(paths["reviewed"].read_text(encoding="utf-8"))
    assert len(reviewed["rows"]) == 1
    assert reviewed["rows"][0]["human_review_decision"] == "unsure_needs_later_review"
    assert reviewed["rows"][0]["review_decisions_collected_with_overlay_version"] == remediation.CURRENT_BURST_OVERLAY_VERSION


def test_m1r_review_progress_reads_decisions_key_and_edge_id_fallback() -> None:
    candidate = m1r_candidate(0)
    candidate_payload = {
        "artifact": "step2m1r_targeted_review_candidate_rows",
        "summary": {"current_overlay_version": remediation.CURRENT_BURST_OVERLAY_VERSION, "burst_overlay_alignment_safe_for_review": True},
        "rows": [candidate],
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }
    row = remediation.m1r_review_decision_row(
        candidate,
        {"human_review_decision": "accept_short_window_visual_continuity_edge", "reviewer_name": "qa"},
    )
    row_without_review_id = {key: value for key, value in row.items() if key != "step2m1r_review_candidate_id"}
    progress = remediation.m1r_review_progress_payload(candidate_payload, {"decisions": [row_without_review_id]})
    assert progress["reviewed_candidates"] == 1
    assert progress["accepted_count"] == 1
    assert progress["targeted_review_completed"] is True
    assert progress["review_decisions_overlay_version_matches_current"] is True
    assert progress["validation_errors"] == []


def test_m1r_review_progress_reads_raw_list_rows() -> None:
    candidate = m1r_candidate(1)
    candidate_payload = {
        "artifact": "step2m1r_targeted_review_candidate_rows",
        "summary": {"current_overlay_version": remediation.CURRENT_BURST_OVERLAY_VERSION, "burst_overlay_alignment_safe_for_review": True},
        "rows": [candidate],
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }
    row = remediation.m1r_review_decision_row(candidate, {"human_review_decision": "reject_edge"})
    progress = remediation.m1r_review_progress_payload(candidate_payload, [row])
    assert progress["reviewed_candidates"] == 1
    assert progress["rejected_count"] == 1
    assert progress["unsure_count"] == 0
    assert progress["targeted_review_completed"] is True


def test_overlay_version_mismatch_blocks_adaptation_even_when_review_complete() -> None:
    edge_payload = {
        "adaptation_safe_edge_count": 1,
        "excluded_edge_count": 0,
        "rows": [],
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }
    group_payload = {
        "visual_continuity_group_rows": 1,
        "rows": [],
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }
    remediation_summary = {
        "groups_over_cap_after": 0,
        "groups_excluded_from_adaptation": 0,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }
    review_payload = {
        "targeted_review_completed": True,
        "rows": [m1r_candidate(0)],
        "summary": {
            "reviewed_candidates": 1,
            "safe_auto_accept_audit_rows": 10,
            "burst_evidence_missing_rate": 0.0,
            "burst_overlay_alignment_safe_for_review": True,
            "current_overlay_version": remediation.CURRENT_BURST_OVERLAY_VERSION,
            "review_decisions_collected_with_overlay_version": ["old_overlay_version"],
            "review_decisions_overlay_version_matches_current": False,
        },
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }
    manifest = build_adaptation_safety_manifest(edge_payload=edge_payload, group_payload=group_payload, remediation_summary=remediation_summary, review_payload=review_payload)
    assert manifest["safe_for_step2m2_adaptation_candidate"] is False
    assert "targeted_review_decisions_not_collected_with_current_overlay_version" in manifest["unsafe_for_step2m2_adaptation_reasons"]
    assert manifest["forbidden_keys_present"] == []
