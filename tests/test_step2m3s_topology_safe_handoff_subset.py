# ruff: noqa: E501

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import football_intelligence.step2_visual_continuity.topology_safe_handoff_subset as m3s  # noqa: E402
from football_intelligence.step2_visual_continuity.io import (  # noqa: E402
    STEP2M1_OUTPUT_DIR,
    STEP2M2_OUTPUT_DIR,
    STEP2M3_OUTPUT_DIR,
    STEP2M3R_OUTPUT_DIR,
    STEP2M3S_OUTPUT_DIR,
)
from football_intelligence.step2_visual_continuity.schema import (  # noqa: E402
    VISUAL_ONLY_WARNING,
    forbidden_keys_present,
    rows_from_payload,
)


def edge(edge_id: str, source: str, target: str, source_frame: int, target_frame: int) -> dict:
    return {
        "continuity_edge_id": edge_id,
        "source_visible_person_base_id": source,
        "target_visible_person_base_id": target,
        "source_frame_sequence": source_frame,
        "target_frame_sequence": target_frame,
        "frame_gap": max(1, target_frame - source_frame),
        "source_review_bucket": "safe_auto_accept_candidate",
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }


def group(group_id: str, edge_ids: list[str], members: list[str], frames: list[int]) -> dict:
    return {
        "visual_continuity_group_id": group_id,
        "accepted_continuity_edge_ids": edge_ids,
        "member_visible_person_base_ids": members,
        "member_frame_sequences": frames,
        "min_frame_sequence": min(frames),
        "max_frame_sequence": max(frames),
        "frame_span": max(frames) - min(frames),
        "seconds_span": round((max(frames) - min(frames)) / 10.0, 4),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }


def group_audit(group_row: dict, *, branch: bool = False, duplicate: bool = False, role: bool = False) -> dict:
    return {
        "visual_continuity_group_id": group_row["visual_continuity_group_id"],
        "min_frame_sequence": group_row["min_frame_sequence"],
        "max_frame_sequence": group_row["max_frame_sequence"],
        "frame_span": group_row["frame_span"],
        "seconds_span": group_row["seconds_span"],
        "group_over_span_cap": False,
        "high_topology_risk": branch or duplicate or role,
        "has_branching": branch,
        "has_merging": branch,
        "frames_with_multiple_members_count": 1 if duplicate else 0,
        "has_role_context_mixing": role,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }


def decision(
    candidate_id: str,
    subject: str,
    category: str,
    decision_value: str,
    *,
    group_id: str = "",
    edge_id: str = "",
) -> dict:
    return {
        "step2m3r_topology_review_candidate_id": candidate_id,
        "review_subject_type": subject,
        "step2m3r_review_category": category,
        "visual_continuity_group_id": group_id,
        "continuity_edge_id": edge_id,
        "human_review_decision": decision_value,
        "review_decisions_collected_with_visual_evidence_version": m3s.M3S_CURRENT_VISUAL_EVIDENCE_VERSION,
        "review_decisions_visual_evidence_version_matches_current": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def patch_m3s_paths(monkeypatch, tmp_path: Path) -> dict[str, Path]:
    m3_dir = tmp_path / "m3"
    m3r_dir = tmp_path / "m3r"
    m3s_dir = tmp_path / "step2m3s_topology_safe_handoff_subset"
    paths = {
        "STEP2M3_OUTPUT_DIR": m3_dir,
        "STEP2M3_ACCEPTED_EDGES_JSONL_GZ_PATH": m3_dir / "step2m3_accepted_visual_continuity_edges.jsonl.gz",
        "STEP2M3_ACCEPTED_EDGE_SUMMARY_PATH": m3_dir / "step2m3_accepted_visual_continuity_edge_summary.json",
        "STEP2M3_QUARANTINE_SUMMARY_PATH": m3_dir / "step2m3_quarantine_summary.json",
        "STEP2M3_GROUP_ROWS_PATH": m3_dir / "step2m3_adaptation_safe_visual_continuity_groups.json",
        "STEP2M3_GROUP_SUMMARY_PATH": m3_dir / "step2m3_group_summary.json",
        "STEP2M3_VALIDATION_SUMMARY_PATH": m3_dir / "step2m3_validation_summary.json",
        "STEP2M3_FREEZE_CANDIDATE_MANIFEST_PATH": m3_dir / "step2m3_freeze_candidate_manifest.json",
        "STEP2M3R_OUTPUT_DIR": m3r_dir,
        "STEP2M3R_GROUP_TOPOLOGY_AUDIT_ROWS_PATH": m3r_dir / "step2m3r_group_topology_audit_rows.json",
        "STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_ROWS_JSONL_GZ_PATH": m3r_dir
        / "step2m3r_accepted_edge_topology_audit_rows.jsonl.gz",
        "STEP2M3R_REVIEWED_TOPOLOGY_DECISIONS_PATH": m3r_dir / "step2m3r_reviewed_topology_decisions.json",
        "STEP2M3R_REVIEW_PROGRESS_SUMMARY_PATH": m3r_dir / "step2m3r_review_progress_summary.json",
        "STEP2M3R_HANDOFF_READINESS_SUMMARY_PATH": m3r_dir / "step2m3r_handoff_readiness_summary.json",
        "STEP2M3R_VALIDATION_SUMMARY_PATH": m3r_dir / "step2m3r_validation_summary.json",
        "STEP2M3R_REVIEW_DECISION_SUMMARY_PATH": m3r_dir / "step2m3r_review_decision_summary.json",
        "STEP2M3S_OUTPUT_DIR": m3s_dir,
        "STEP2M3S_REVIEWED_TOPOLOGY_DECISION_ROWS_PATH": m3s_dir / "step2m3s_reviewed_topology_decision_rows.json",
        "STEP2M3S_REVIEWED_TOPOLOGY_DECISION_SUMMARY_PATH": m3s_dir / "step2m3s_reviewed_topology_decision_summary.json",
        "STEP2M3S_HANDOFF_SAFE_GROUPS_PATH": m3s_dir / "step2m3s_handoff_safe_visual_continuity_groups.json",
        "STEP2M3S_HANDOFF_SAFE_GROUP_SAMPLE_PATH": m3s_dir / "step2m3s_handoff_safe_visual_continuity_group_sample.json",
        "STEP2M3S_HANDOFF_SAFE_GROUP_SUMMARY_PATH": m3s_dir / "step2m3s_handoff_safe_group_summary.json",
        "STEP2M3S_HANDOFF_SAFE_EDGES_JSONL_GZ_PATH": m3s_dir / "step2m3s_handoff_safe_visual_continuity_edges.jsonl.gz",
        "STEP2M3S_HANDOFF_SAFE_EDGE_SAMPLE_PATH": m3s_dir / "step2m3s_handoff_safe_visual_continuity_edge_sample.json",
        "STEP2M3S_HANDOFF_SAFE_EDGE_SUMMARY_PATH": m3s_dir / "step2m3s_handoff_safe_edge_summary.json",
        "STEP2M3S_TOPOLOGY_QUARANTINED_GROUPS_PATH": m3s_dir / "step2m3s_topology_quarantined_groups.json",
        "STEP2M3S_TOPOLOGY_QUARANTINED_EDGES_JSONL_GZ_PATH": m3s_dir / "step2m3s_topology_quarantined_edges.jsonl.gz",
        "STEP2M3S_TOPOLOGY_QUARANTINE_SUMMARY_PATH": m3s_dir / "step2m3s_topology_quarantine_summary.json",
        "STEP2M3S_HANDOFF_MANIFEST_PATH": m3s_dir / "step2m3s_handoff_manifest.json",
        "STEP2M3S_VALIDATION_SUMMARY_PATH": m3s_dir / "step2m3s_validation_summary.json",
        "STEP2M3S_SAFETY_GUARDRAIL_AUDIT_PATH": m3s_dir / "step2m3s_safety_guardrail_audit.json",
        "STEP2M3S_ISSUE_REGISTER_PATH": m3s_dir / "step2m3s_issue_register.json",
        "STEP2M3S_FREEZE_CANDIDATE_MANIFEST_PATH": m3s_dir / "step2m3s_freeze_candidate_manifest.json",
        "STEP2M3S_REVIEW_PACK_DIR": m3s_dir / "review_pack",
        "STEP2M3S_REVIEW_PACK_MANIFEST_PATH": m3s_dir / "review_pack" / "step2m3s_review_pack_manifest.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(m3s, name, path)
    return paths


def write_fixture(paths: dict[str, Path]) -> str:
    accepted_edges = [
        edge("e_reject_group", "a", "b", 1, 2),
        edge("e_accept_group", "c", "d", 10, 11),
        edge("e_unsure_group", "u", "v", 20, 21),
        edge("e_accept_edge", "x", "y", 30, 31),
        edge("e_keep_after_split", "s", "t", 40, 41),
        edge("e_reject_edge", "t", "z", 41, 42),
        edge("e_branch_1", "p", "q", 50, 51),
        edge("e_branch_2", "p", "r", 50, 51),
    ]
    groups = [
        group("g_reject", ["e_reject_group"], ["a", "b"], [1, 2]),
        group("g_accept", ["e_accept_group"], ["c", "d"], [10, 11]),
        group("g_unsure", ["e_unsure_group"], ["u", "v"], [20, 21]),
        group("g_edge", ["e_accept_edge"], ["x", "y"], [30, 31]),
        group("g_split", ["e_keep_after_split", "e_reject_edge"], ["s", "t", "z"], [40, 41, 42]),
        group("g_branch", ["e_branch_1", "e_branch_2"], ["p", "q", "r"], [50, 51, 51]),
    ]
    paths["STEP2M3_GROUP_ROWS_PATH"].parent.mkdir(parents=True, exist_ok=True)
    paths["STEP2M3_GROUP_ROWS_PATH"].write_text(json.dumps({"rows": groups}), encoding="utf-8")
    write_jsonl_gz(paths["STEP2M3_ACCEPTED_EDGES_JSONL_GZ_PATH"], accepted_edges)
    edge_to_group = {
        edge_id: group_row["visual_continuity_group_id"]
        for group_row in groups
        for edge_id in group_row["accepted_continuity_edge_ids"]
    }
    edge_audit_rows = [
        {**row, "visual_continuity_group_id": edge_to_group[row["continuity_edge_id"]]}
        for row in accepted_edges
    ]
    write_jsonl_gz(paths["STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_ROWS_JSONL_GZ_PATH"], edge_audit_rows)
    group_audit_rows = [
        group_audit(groups[0]),
        group_audit(groups[1]),
        group_audit(groups[2]),
        group_audit(groups[3]),
        group_audit(groups[4]),
        group_audit(groups[5], branch=True),
    ]
    paths["STEP2M3R_GROUP_TOPOLOGY_AUDIT_ROWS_PATH"].parent.mkdir(parents=True, exist_ok=True)
    paths["STEP2M3R_GROUP_TOPOLOGY_AUDIT_ROWS_PATH"].write_text(json.dumps({"rows": group_audit_rows}), encoding="utf-8")
    decisions = [
        decision("card_reject_group", "visual_continuity_group", "topology_risk_fallback_group", m3s.M3S_REJECT_DECISION, group_id="g_reject"),
        decision("card_accept_group", "visual_continuity_group", "topology_risk_fallback_group", m3s.M3S_ACCEPT_DECISION, group_id="g_accept"),
        decision("card_unsure_group", "visual_continuity_group", "topology_risk_fallback_group", m3s.M3S_UNSURE_DECISION, group_id="g_unsure"),
        decision("card_accept_edge", "accepted_visual_continuity_edge", "risky_accepted_edge", m3s.M3S_ACCEPT_DECISION, group_id="g_edge", edge_id="e_accept_edge"),
        decision("card_accept_split", "visual_continuity_group", "topology_risk_fallback_group", m3s.M3S_ACCEPT_DECISION, group_id="g_split"),
        decision("card_reject_edge", "accepted_visual_continuity_edge", "risky_accepted_edge", m3s.M3S_REJECT_DECISION, group_id="g_split", edge_id="e_reject_edge"),
    ]
    before = paths["STEP2M3_GROUP_ROWS_PATH"].read_text(encoding="utf-8")
    paths["STEP2M3R_REVIEWED_TOPOLOGY_DECISIONS_PATH"].write_text(json.dumps({"rows": decisions}), encoding="utf-8")
    for path, payload in [
        (paths["STEP2M3_ACCEPTED_EDGE_SUMMARY_PATH"], {"accepted_edge_count": len(accepted_edges)}),
        (paths["STEP2M3_QUARANTINE_SUMMARY_PATH"], {"quarantined_edge_count": 0}),
        (paths["STEP2M3_GROUP_SUMMARY_PATH"], {"adaptation_safe_group_count": len(groups), "groups_over_cap_count": 0}),
        (paths["STEP2M3_VALIDATION_SUMMARY_PATH"], {"forbidden_keys_present": []}),
        (paths["STEP2M3_FREEZE_CANDIDATE_MANIFEST_PATH"], {"step2m3_freeze_candidate_created": True}),
        (
            paths["STEP2M3R_REVIEW_PROGRESS_SUMMARY_PATH"],
            {
                "topology_review_completed": True,
                "current_visual_evidence_version": m3s.M3S_CURRENT_VISUAL_EVIDENCE_VERSION,
                "review_decisions_visual_evidence_version_matches_current": True,
            },
        ),
        (paths["STEP2M3R_HANDOFF_READINESS_SUMMARY_PATH"], {"safe_for_visual_continuity_handoff_candidate": True}),
        (paths["STEP2M3R_VALIDATION_SUMMARY_PATH"], {"forbidden_keys_present": []}),
        (paths["STEP2M3R_REVIEW_DECISION_SUMMARY_PATH"], {"reviewed_candidates": len(decisions)}),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return before


def test_step2m3s_output_paths_are_isolated() -> None:
    m3s_root = STEP2M3S_OUTPUT_DIR.resolve()
    blocked_roots = [STEP2M1_OUTPUT_DIR.resolve(), STEP2M2_OUTPUT_DIR.resolve(), STEP2M3_OUTPUT_DIR.resolve(), STEP2M3R_OUTPUT_DIR.resolve()]
    for path in m3s.step2m3s_output_paths().values():
        resolved = path.resolve()
        assert resolved == m3s_root or m3s_root in resolved.parents
        assert all(resolved != root and root not in resolved.parents for root in blocked_roots)


def test_m3r_decision_loader_supports_list_rows_and_decisions(tmp_path: Path) -> None:
    row = decision("card", "visual_continuity_group", "topology_risk_fallback_group", m3s.M3S_ACCEPT_DECISION, group_id="g")
    for payload in [[row], {"rows": [row]}, {"decisions": [row]}]:
        path = tmp_path / f"decisions_{len(json.dumps(payload))}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = m3s.load_m3r_reviewed_decisions(path)
        assert len(loaded) == 1
        assert loaded[0]["m3s_decision_label"] == "accepted"
        assert forbidden_keys_present(loaded) == []


def test_m3s_quarantines_rejects_unsure_and_rebuilds_safe_split(monkeypatch, tmp_path: Path) -> None:
    paths = patch_m3s_paths(monkeypatch, tmp_path)
    before_group_text = write_fixture(paths)
    outputs = m3s.build_step2m3s_topology_safe_handoff_subset()
    assert paths["STEP2M3_GROUP_ROWS_PATH"].read_text(encoding="utf-8") == before_group_text
    assert outputs["reviewed_topology_decision_summary"]["accepted_count"] == 3
    assert outputs["reviewed_topology_decision_summary"]["rejected_count"] == 2
    assert outputs["reviewed_topology_decision_summary"]["unsure_count"] == 1
    handoff_edges = m3s.read_jsonl_gz_rows(paths["STEP2M3S_HANDOFF_SAFE_EDGES_JSONL_GZ_PATH"])
    handoff_edge_ids = {row["continuity_edge_id"] for row in handoff_edges}
    assert {"e_accept_group", "e_accept_edge", "e_keep_after_split"} <= handoff_edge_ids
    assert "e_reject_group" not in handoff_edge_ids
    assert "e_unsure_group" not in handoff_edge_ids
    assert "e_reject_edge" not in handoff_edge_ids
    assert "e_branch_1" not in handoff_edge_ids
    quarantine_edges = m3s.read_jsonl_gz_rows(paths["STEP2M3S_TOPOLOGY_QUARANTINED_EDGES_JSONL_GZ_PATH"])
    reasons_by_edge = {row["continuity_edge_id"]: row["m3s_topology_quarantine_reasons"] for row in quarantine_edges}
    assert "m3r_group_rejected" in reasons_by_edge["e_reject_group"]
    assert "m3r_unsure" in reasons_by_edge["e_unsure_group"]
    assert "m3r_edge_rejected" in reasons_by_edge["e_reject_edge"]
    assert "branch_merge_topology_not_handoff_safe" in reasons_by_edge["e_branch_1"]
    handoff_groups = rows_from_payload(m3s.read_json(paths["STEP2M3S_HANDOFF_SAFE_GROUPS_PATH"]))
    assert all(row["frame_span"] <= 30 and row["seconds_span"] <= 3.0 for row in handoff_groups)
    assert outputs["validation_summary"]["production_ready"] is False
    assert outputs["validation_summary"]["no_auto_promotion"] is True
    assert outputs["validation_summary"]["human_approved"] is False
    assert outputs["freeze_candidate_manifest"]["step2m3s_freeze_candidate_created"] is True
    assert forbidden_keys_present(outputs) == []
    for path in paths["STEP2M3S_OUTPUT_DIR"].rglob("*"):
        assert paths["STEP2M3S_OUTPUT_DIR"] in path.resolve().parents or path.resolve() == paths["STEP2M3S_OUTPUT_DIR"].resolve()


def test_m3s_validation_and_review_pack(monkeypatch, tmp_path: Path) -> None:
    paths = patch_m3s_paths(monkeypatch, tmp_path)
    write_fixture(paths)
    m3s.build_step2m3s_topology_safe_handoff_subset()
    validation_outputs = m3s.validate_step2m3s_topology_safe_handoff_subset()
    manifest = m3s.write_step2m3s_review_pack()
    assert validation_outputs["issue_register"]["blocking_issue_count"] == 0
    assert validation_outputs["freeze_candidate_manifest"]["step2m3s_freeze_candidate_created"] is True
    assert manifest["step2m3s_freeze_candidate_created"] is True
    assert manifest["production_ready"] is False
    assert manifest["no_auto_promotion"] is True
    assert forbidden_keys_present(manifest) == []
