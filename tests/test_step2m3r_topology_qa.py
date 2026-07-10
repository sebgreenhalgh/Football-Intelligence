from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import football_intelligence.step2_visual_continuity.topology_qa as m3r  # noqa: E402
from football_intelligence.step2_visual_continuity.io import (  # noqa: E402
    STEP2M1_OUTPUT_DIR,
    STEP2M2_OUTPUT_DIR,
    STEP2M3_OUTPUT_DIR,
    STEP2M3R_OUTPUT_DIR,
)
from football_intelligence.step2_visual_continuity.schema import (  # noqa: E402
    VISUAL_ONLY_WARNING,
    assert_no_forbidden_keys,
    forbidden_keys_present,
    rows_from_payload,
)


def edge(
    index: int,
    *,
    source: str | None = None,
    target: str | None = None,
    source_frame: int | None = None,
    target_frame: int | None = None,
    bucket: str = "safe_auto_accept_candidate",
) -> dict:
    source_frame = index if source_frame is None else source_frame
    target_frame = source_frame + 1 if target_frame is None else target_frame
    return {
        "continuity_edge_id": f"edge_{index}",
        "source_visible_person_base_id": source or f"member_{index}",
        "target_visible_person_base_id": target or f"member_{index + 1}",
        "source_frame_sequence": source_frame,
        "target_frame_sequence": target_frame,
        "frame_gap": max(1, target_frame - source_frame),
        "source_review_bucket": bucket,
        "m3_acceptance_reason": "m2_adapted_auto_accept_safety_filters_passed",
        "human_review_decision_source": "",
        "human_review_decision": "",
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }


def group(
    group_id: str,
    members: list[str],
    frames: list[int],
    edge_ids: list[str],
    role_counts: dict[str, int] | None = None,
) -> dict:
    return {
        "visual_continuity_group_id": group_id,
        "member_visible_person_base_ids": members,
        "member_frame_sequences": frames,
        "accepted_continuity_edge_ids": edge_ids,
        "min_frame_sequence": min(frames),
        "max_frame_sequence": max(frames),
        "frame_span": max(frames) - min(frames),
        "seconds_span": round((max(frames) - min(frames)) / 10.0, 4),
        "role_state_counts_visual_context_only": role_counts or {"team_1_outfield_visual_context": len(members)},
    }


def test_step2m3r_output_paths_are_isolated() -> None:
    m3r_root = STEP2M3R_OUTPUT_DIR.resolve()
    blocked_roots = [STEP2M1_OUTPUT_DIR.resolve(), STEP2M2_OUTPUT_DIR.resolve(), STEP2M3_OUTPUT_DIR.resolve()]
    for path in m3r.step2m3r_output_paths().values():
        resolved = path.resolve()
        assert resolved == m3r_root or m3r_root in resolved.parents
        assert all(resolved != root and root not in resolved.parents for root in blocked_roots)


def test_duplicate_frame_members_are_flagged() -> None:
    edges = [edge(1, source="a", target="b", source_frame=10, target_frame=10)]
    row = m3r.make_group_topology_row(
        group("g1", ["a", "b"], [10, 10], ["edge_1"]),
        edges,
    )
    assert row["frames_with_multiple_members_count"] == 1
    assert row["max_members_per_frame"] == 2
    assert row["high_topology_risk"] is True
    assert "duplicate_frame_members_visual_topology" in row["topology_risk_reasons"]
    assert_no_forbidden_keys(row)


def test_branch_and_merge_topology_are_flagged() -> None:
    branch_row = m3r.make_group_topology_row(
        group("g_branch", ["a", "b", "c"], [1, 2, 2], ["edge_1", "edge_2"]),
        [
            edge(1, source="a", target="b", source_frame=1, target_frame=2),
            edge(2, source="a", target="c", source_frame=1, target_frame=2),
        ],
    )
    merge_row = m3r.make_group_topology_row(
        group("g_merge", ["a", "b", "c"], [1, 1, 2], ["edge_3", "edge_4"]),
        [
            edge(3, source="a", target="c", source_frame=1, target_frame=2),
            edge(4, source="b", target="c", source_frame=1, target_frame=2),
        ],
    )
    assert branch_row["has_branching"] is True
    assert "branching_visual_topology" in branch_row["topology_risk_reasons"]
    assert merge_row["has_merging"] is True
    assert "merging_visual_topology" in merge_row["topology_risk_reasons"]


def test_role_context_mixing_is_visual_only_and_flagged() -> None:
    row = m3r.make_group_topology_row(
        group(
            "g_role",
            ["a", "b"],
            [5, 6],
            ["edge_1"],
            role_counts={"team_1_outfield_visual_context": 1, "team_2_outfield_visual_context": 1},
        ),
        [edge(1, source="a", target="b", source_frame=5, target_frame=6)],
    )
    assert row["has_role_context_mixing"] is True
    assert row["role_context_count_visual_only"] == 2
    assert row["no_identity_tracking_performed"] is True
    assert row["no_player_slots_assigned"] is True
    assert "role_context_mixing_visual_only" in row["topology_risk_reasons"]


def test_review_queue_hard_max_and_guardrails() -> None:
    group_rows = [
        m3r.make_group_topology_row(
            group(f"g_{index}", [f"a_{index}", f"b_{index}"], [index, index], [f"edge_{index}"]),
            [edge(index, source=f"a_{index}", target=f"b_{index}", source_frame=index, target_frame=index)],
        )
        for index in range(70)
    ]
    edge_rows = [
        m3r.make_edge_topology_row(
            edge(index, bucket="role_state_mismatch"),
            group_id=f"g_{index}",
            group_row=group_rows[index],
        )
        for index in range(20)
    ]
    payload = m3r.build_topology_review_queue(group_rows, edge_rows)
    assert len(rows_from_payload(payload)) <= 60
    assert payload["production_ready"] is False
    assert payload["no_auto_promotion"] is True
    assert payload["human_approved"] is False
    assert forbidden_keys_present(payload) == []


def make_visual_evidence_inputs(tmp_path: Path) -> tuple[Path, Path]:
    if m3r.cv2 is None or m3r.np is None or m3r.imageio is None:
        pytest.skip("OpenCV/numpy/imageio unavailable for visual evidence rendering test")
    node_rows_path = tmp_path / "nodes.json"
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for frame_sequence in [1, 2]:
        image = m3r.np.full((120, 180, 3), 210, dtype=m3r.np.uint8)
        assert m3r.cv2.imwrite(str(frame_dir / f"frame_{frame_sequence}.jpg"), image)
    node_rows_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "visible_person_base_id": "a",
                        "frame_sequence": 1,
                        "frame_id": "frame_1",
                        "bbox": {"x1": 20, "y1": 20, "x2": 60, "y2": 90},
                        "step1f3_final_visual_role_state": "team_1_outfield_visual_context",
                    },
                    {
                        "visible_person_base_id": "b",
                        "frame_sequence": 2,
                        "frame_id": "frame_2",
                        "bbox": {"x1": 25, "y1": 25, "x2": 65, "y2": 95},
                        "step1f3_final_visual_role_state": "team_1_outfield_visual_context",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return node_rows_path, frame_dir


def test_m3r_visual_evidence_fields_for_group_and_edge_cards(monkeypatch, tmp_path: Path) -> None:
    node_rows_path, frame_dir = make_visual_evidence_inputs(tmp_path)
    output_dir = tmp_path / "m3r_output"
    monkeypatch.setattr(m3r, "STEP2M3R_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(
        m3r,
        "STEP2M3R_GROUP_TIMELINE_STRIPS_DIR",
        output_dir / "step2m3r_visual_evidence" / "group_timeline_strips",
    )
    monkeypatch.setattr(
        m3r,
        "STEP2M3R_EDGE_BURST_STRIPS_DIR",
        output_dir / "step2m3r_visual_evidence" / "edge_burst_strips",
    )
    monkeypatch.setattr(
        m3r,
        "STEP2M3R_GROUP_TIMELINE_ANIMATIONS_DIR",
        output_dir / "step2m3r_visual_evidence" / "group_timeline_animations",
    )
    monkeypatch.setattr(
        m3r,
        "STEP2M3R_EDGE_BURST_ANIMATIONS_DIR",
        output_dir / "step2m3r_visual_evidence" / "edge_burst_animations",
    )
    monkeypatch.setattr(m3r, "STAGE3C_FRAMES_DIR", frame_dir)
    review_payload = {
        "summary": {},
        "rows": [
            {
                "step2m3r_topology_review_candidate_id": "card_group",
                "review_subject_type": "visual_continuity_group",
                "visual_continuity_group_id": "g1",
                "min_frame_sequence": 1,
                "max_frame_sequence": 2,
            },
            {
                "step2m3r_topology_review_candidate_id": "card_edge",
                "review_subject_type": "accepted_visual_continuity_edge",
                "visual_continuity_group_id": "g1",
                "continuity_edge_id": "edge_1",
                "source_frame_sequence": 1,
                "target_frame_sequence": 2,
            },
        ],
    }
    group_row = group("g1", ["a", "b"], [1, 2], ["edge_1"])
    edge_row = edge(1, source="a", target="b", source_frame=1, target_frame=2)
    enriched = m3r.add_visual_evidence_to_review_payload(
        review_payload,
        source_groups_by_id={"g1": group_row},
        edges_by_group={"g1": [edge_row]},
        accepted_edges_by_id={"edge_1": edge_row},
        node_payload=json.loads(node_rows_path.read_text(encoding="utf-8")),
    )
    rows = rows_from_payload(enriched)
    assert rows[0]["evidence_type"] == "group_timeline_animation"
    assert rows[1]["evidence_type"] == "edge_burst_animation"
    assert all(row["evidence_available"] is True for row in rows)
    assert all(row["animation_evidence_available"] is True for row in rows)
    assert all(row["static_strip_fallback_available"] is True for row in rows)
    assert all((output_dir / row["evidence_animation_gif_path"]).exists() for row in rows)
    assert all((output_dir / row["evidence_static_strip_path"]).exists() for row in rows)
    assert enriched["summary"]["visual_evidence_available_count"] == 2
    assert enriched["summary"]["visual_evidence_missing_count"] == 0
    assert enriched["summary"]["visual_evidence_safe_for_review"] is True
    assert enriched["summary"]["animation_evidence_available_count"] == 2
    assert enriched["summary"]["animation_evidence_missing_count"] == 0
    assert enriched["summary"]["animation_evidence_safe_for_review"] is True
    assert enriched["summary"]["static_strip_fallback_available_count"] == 2
    assert enriched["summary"]["current_visual_evidence_version"] == m3r.M3R_CURRENT_VISUAL_EVIDENCE_VERSION
    assert forbidden_keys_present(enriched) == []


def test_visual_evidence_version_mismatch_blocks_handoff() -> None:
    review_payload = {
        "rows": [
            {
                "step2m3r_topology_review_candidate_id": "card_1",
                "review_subject_type": "visual_continuity_group",
                "visual_continuity_group_id": "g1",
                "evidence_available": True,
                "evidence_type": "group_timeline_animation",
                "evidence_animation_gif_path": "step2m3r_visual_evidence/group_timeline_animations/card_1.gif",
                "evidence_static_strip_path": "step2m3r_visual_evidence/group_timeline_strips/card_1.jpg",
                "animation_evidence_available": True,
                "static_strip_fallback_available": True,
                "overlay_version": m3r.M3R_CURRENT_VISUAL_EVIDENCE_VERSION,
                "current_visual_evidence_version": m3r.M3R_CURRENT_VISUAL_EVIDENCE_VERSION,
            }
        ]
    }
    reviewed_payload = {
        "rows": [
            {
                "step2m3r_topology_review_candidate_id": "card_1",
                "human_review_decision": m3r.M3R_ACCEPT_DECISION,
                "review_decisions_collected_with_review_version": m3r.M3R_CURRENT_REVIEW_VERSION,
                "review_decisions_collected_with_visual_evidence_version": "old_visual_evidence",
            }
        ]
    }
    progress = m3r.m3r_review_progress_payload(review_payload, reviewed_payload)
    handoff = m3r.build_handoff_readiness_summary(
        {"high_topology_risk_group_count": 0, "groups_over_cap_count": 0},
        {},
        review_payload,
        progress,
    )
    assert progress["review_decisions_visual_evidence_version_matches_current"] is False
    assert handoff["safe_for_visual_continuity_handoff_candidate"] is False
    assert (
        "review_decisions_visual_evidence_version_mismatch"
        in handoff["unsafe_for_visual_continuity_handoff_reasons"]
    )


def test_m3r_can_read_m3_inputs_while_writing_only_m3r_outputs(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "m3_source"
    output_dir = tmp_path / "m3r_output"
    source_dir.mkdir()
    group_path = source_dir / "groups.json"
    accepted_path = source_dir / "accepted.jsonl.gz"
    m3_freeze_path = source_dir / "freeze.json"
    m3_validation_path = source_dir / "validation.json"
    m3_group_summary_path = source_dir / "group_summary.json"
    node_rows_path, frame_dir = make_visual_evidence_inputs(tmp_path)
    group_payload = {
        "rows": [
            group("g1", ["a", "b"], [1, 2], ["edge_1"]),
        ]
    }
    group_path.write_text(json.dumps(group_payload), encoding="utf-8")
    with gzip.open(accepted_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(edge(1, source="a", target="b", source_frame=1, target_frame=2)) + "\n")
    m3_freeze_path.write_text(
        json.dumps({"step2m3_freeze_candidate_created": True, "forbidden_keys_present": []}),
        encoding="utf-8",
    )
    m3_validation_path.write_text(json.dumps({"forbidden_keys_present": []}), encoding="utf-8")
    m3_group_summary_path.write_text(json.dumps({"groups_over_cap_count": 0}), encoding="utf-8")
    before_group_text = group_path.read_text(encoding="utf-8")
    path_map = {
        "STEP2M3R_OUTPUT_DIR": output_dir,
        "STEP2M3R_GROUP_TOPOLOGY_AUDIT_ROWS_PATH": output_dir / "step2m3r_group_topology_audit_rows.json",
        "STEP2M3R_GROUP_TOPOLOGY_AUDIT_SUMMARY_PATH": output_dir / "step2m3r_group_topology_audit_summary.json",
        "STEP2M3R_GROUP_TOPOLOGY_AUDIT_SAMPLE_PATH": output_dir / "step2m3r_group_topology_audit_sample.json",
        "STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_ROWS_JSONL_GZ_PATH": output_dir
        / "step2m3r_accepted_edge_topology_audit_rows.jsonl.gz",
        "STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_SUMMARY_PATH": output_dir
        / "step2m3r_accepted_edge_topology_audit_summary.json",
        "STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_SAMPLE_PATH": output_dir
        / "step2m3r_accepted_edge_topology_audit_sample.json",
        "STEP2M3R_TOPOLOGY_REVIEW_CANDIDATE_ROWS_PATH": output_dir / "step2m3r_topology_review_candidate_rows.json",
        "STEP2M3R_TOPOLOGY_REVIEW_UI_HTML_PATH": output_dir / "step2m3r_topology_review_ui.html",
        "STEP2M3R_TOPOLOGY_REVIEW_CONTACT_SHEET_PATH": output_dir / "step2m3r_topology_review_contact_sheet.jpg",
        "STEP2M3R_GROUP_TIMELINE_STRIPS_DIR": output_dir / "step2m3r_visual_evidence" / "group_timeline_strips",
        "STEP2M3R_EDGE_BURST_STRIPS_DIR": output_dir / "step2m3r_visual_evidence" / "edge_burst_strips",
        "STEP2M3R_GROUP_TIMELINE_ANIMATIONS_DIR": output_dir
        / "step2m3r_visual_evidence"
        / "group_timeline_animations",
        "STEP2M3R_EDGE_BURST_ANIMATIONS_DIR": output_dir / "step2m3r_visual_evidence" / "edge_burst_animations",
        "STEP2M3R_REVIEWED_TOPOLOGY_DECISIONS_PATH": output_dir / "step2m3r_reviewed_topology_decisions.json",
        "STEP2M3R_REVIEW_PROGRESS_SUMMARY_PATH": output_dir / "step2m3r_review_progress_summary.json",
        "STEP2M3R_REVIEW_DECISION_SUMMARY_PATH": output_dir / "step2m3r_review_decision_summary.json",
        "STEP2M3R_HANDOFF_READINESS_SUMMARY_PATH": output_dir / "step2m3r_handoff_readiness_summary.json",
        "STEP2M3R_VALIDATION_SUMMARY_PATH": output_dir / "step2m3r_validation_summary.json",
        "STEP2M3R_SAFETY_GUARDRAIL_AUDIT_PATH": output_dir / "step2m3r_safety_guardrail_audit.json",
        "STEP2M3R_ISSUE_REGISTER_PATH": output_dir / "step2m3r_issue_register.json",
        "STEP2M3R_FREEZE_CANDIDATE_MANIFEST_PATH": output_dir / "step2m3r_freeze_candidate_manifest.json",
        "STEP2M3R_REVIEW_PACK_DIR": output_dir / "review_pack",
        "STEP2M3R_REVIEW_PACK_MANIFEST_PATH": output_dir / "review_pack" / "step2m3r_review_pack_manifest.json",
    }
    for name, path in path_map.items():
        monkeypatch.setattr(m3r, name, path)
    monkeypatch.setattr(m3r, "STEP2M3_GROUP_ROWS_PATH", group_path)
    monkeypatch.setattr(m3r, "STEP2M3_ACCEPTED_EDGES_JSONL_GZ_PATH", accepted_path)
    monkeypatch.setattr(m3r, "STEP2M3_FREEZE_CANDIDATE_MANIFEST_PATH", m3_freeze_path)
    monkeypatch.setattr(m3r, "STEP2M3_VALIDATION_SUMMARY_PATH", m3_validation_path)
    monkeypatch.setattr(m3r, "STEP2M3_GROUP_SUMMARY_PATH", m3_group_summary_path)
    monkeypatch.setattr(m3r, "STEP2M1_NODE_ROWS_PATH", node_rows_path)
    monkeypatch.setattr(m3r, "STAGE3C_FRAMES_DIR", frame_dir)
    outputs = m3r.build_step2m3r_topology_qa()
    assert group_path.read_text(encoding="utf-8") == before_group_text
    assert (output_dir / "step2m3r_validation_summary.json").exists()
    assert outputs["validation_summary"]["step2m3r_freeze_candidate_created"] is True
    review_rows = rows_from_payload(outputs["topology_review_candidates"])
    assert review_rows
    assert all(row["evidence_available"] is True for row in review_rows)
    assert all(row["animation_evidence_available"] is True for row in review_rows)
    assert all(row["static_strip_fallback_available"] is True for row in review_rows)
    assert all((output_dir / row["evidence_animation_gif_path"]).exists() for row in review_rows)
    assert all((output_dir / row["evidence_static_strip_path"]).exists() for row in review_rows)
    assert outputs["contact_sheet"]["contains_visual_evidence_thumbnails"] is True
    assert outputs["contact_sheet"]["animation_evidence_label_count"] == len(review_rows)
    ui_html = (output_dir / "step2m3r_topology_review_ui.html").read_text(encoding="utf-8")
    assert "evidence_animation_gif_path" in ui_html
    assert "evidence_static_strip_path" in ui_html
    for path in output_dir.rglob("*"):
        assert output_dir in path.resolve().parents or path.resolve() == output_dir.resolve()


def test_m3r_review_decision_schema_has_safe_defaults() -> None:
    candidate = {
        "step2m3r_topology_review_candidate_id": "card_1",
        "review_subject_type": "visual_continuity_group",
        "step2m3r_review_category": "clean_topology_control_group",
        "visual_continuity_group_id": "g1",
        "min_frame_sequence": 1,
        "max_frame_sequence": 2,
        "visual_only_warning": VISUAL_ONLY_WARNING,
    }
    row = m3r.m3r_review_decision_row(candidate, {"human_review_decision": m3r.M3R_ACCEPT_DECISION})
    assert row["production_ready"] is False
    assert row["no_auto_promotion"] is True
    assert row["human_approved"] is False
    assert row["approve_any_identity_tracking"] is False
    assert row["approve_any_player_slot_use"] is False
    assert row["approve_event_or_tactical_analysis"] is False
    assert forbidden_keys_present(row) == []
