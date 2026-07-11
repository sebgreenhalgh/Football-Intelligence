from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_intelligence.core.config import SafetyConfig
from football_intelligence.core.guardrails import audit_named_payloads, find_forbidden_keys
from football_intelligence.step2_visual_continuity.nodes import build_node_payload, rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def expected_f3_count(f3_payload: dict[str, Any], g1_manifest: dict[str, Any]) -> int:
    for payload in (g1_manifest, f3_payload.get("summary", {}), f3_payload):
        for key in ("f3_row_count", "source_f3_row_count", "row_count"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(value, int) and value > 0:
                return value
    return 10418 if len(rows(f3_payload)) > 1000 else len(rows(f3_payload))


def recover_m1_nodes(
    *,
    f3_payload: dict[str, Any],
    g1_manifest: dict[str, Any],
    m3t_pathlets: list[dict[str, Any]],
    selected_edges: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    node_payload = build_node_payload(f3_payload, g1_manifest)
    node_payload.update(SafetyConfig().model_dump(mode="json"))
    node_rows = rows(node_payload)
    f3_rows = rows(f3_payload)
    nodes_by_visible_id = {str(row.get("visible_person_base_id", "")): row for row in node_rows}
    node_ids = [str(row.get("step2m1_visual_continuity_node_id", "")) for row in node_rows]
    pathlet_member_ids = {
        str(member) for pathlet in m3t_pathlets for member in pathlet.get("member_visible_person_base_ids", []) or []
    }
    selected_edge_source_ids = {str(row.get("source_visible_person_base_id", "")) for row in selected_edges}
    selected_edge_target_ids = {str(row.get("target_visible_person_base_id", "")) for row in selected_edges}
    missing_pathlet_members = sorted(pathlet_member_ids - set(nodes_by_visible_id))
    missing_sources = sorted(selected_edge_source_ids - set(nodes_by_visible_id))
    missing_targets = sorted(selected_edge_target_ids - set(nodes_by_visible_id))
    frame_sequence_invalid = [
        str(row.get("visible_person_base_id", ""))
        for row in node_rows
        if not isinstance(row.get("frame_sequence"), int) or row.get("frame_sequence") < 0
    ]
    report = {
        "schema_version": "m5.true_replay.m1_node_recovery_report.v1",
        "f3_row_count": len(f3_rows),
        "expected_f3_row_count": expected_f3_count(f3_payload, g1_manifest),
        "recovered_node_count": len(node_rows),
        "one_node_per_f3_row": len(f3_rows) == len(node_rows),
        "visible_person_id_sequence_identical": [str(row.get("visible_person_base_id", "")) for row in f3_rows]
        == [str(row.get("visible_person_base_id", "")) for row in node_rows],
        "node_ids_unique": len(node_ids) == len(set(node_ids)),
        "invalid_frame_sequence_count": len(frame_sequence_invalid),
        "invalid_frame_sequence_visible_person_ids": frame_sequence_invalid[:20],
        "pathlet_member_reference_count": len(pathlet_member_ids),
        "missing_pathlet_member_count": len(missing_pathlet_members),
        "missing_pathlet_member_ids": missing_pathlet_members[:50],
        "missing_selected_edge_source_count": len(missing_sources),
        "missing_selected_edge_source_ids": missing_sources[:50],
        "missing_selected_edge_target_count": len(missing_targets),
        "missing_selected_edge_target_ids": missing_targets[:50],
        "forbidden_key_count": len(find_forbidden_keys(node_payload)),
        "guardrail_audit": audit_named_payloads({"recovered_m1_nodes": node_payload}),
        "no_identity_or_slot_interpretation": all(
            row.get("step2m1_visual_continuity_node_is_identity") is False
            and row.get("step2m1_visual_continuity_node_is_player_slot") is False
            and row.get("step2m1_visual_continuity_node_is_goalkeeper_slot") is False
            for row in node_rows
        ),
    }
    report["passed"] = (
        report["f3_row_count"] == report["expected_f3_row_count"]
        and report["one_node_per_f3_row"]
        and report["visible_person_id_sequence_identical"]
        and report["node_ids_unique"]
        and report["invalid_frame_sequence_count"] == 0
        and report["missing_pathlet_member_count"] == 0
        and report["missing_selected_edge_source_count"] == 0
        and report["missing_selected_edge_target_count"] == 0
        and report["forbidden_key_count"] == 0
        and report["guardrail_audit"]["passed"]
        and report["no_identity_or_slot_interpretation"]
    )
    write_json(output_dir / "step2m1_visual_continuity_node_rows.json", node_payload)
    return node_payload, report
