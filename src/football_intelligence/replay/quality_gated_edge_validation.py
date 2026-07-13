from __future__ import annotations

from typing import Any

from football_intelligence.replay.entity_validity import PROBABLE_NON_PERSON, rows_from_payload
from football_intelligence.replay.portable_context import (
    forbidden_keys_present,
    guardrail_payload,
    semantic_hash,
    utc_now,
)


def validate_quality_gated_edge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = rows_from_payload(payload)
    rejected_rows = payload.get("rejected_rows", [])
    rejected_rows = rejected_rows if isinstance(rejected_rows, list) else []
    invalid_accepts = [
        row
        for row in rows
        if row.get("source_entity_validity") == PROBABLE_NON_PERSON
        or row.get("target_entity_validity") == PROBABLE_NON_PERSON
    ]
    missing_diagnostics = [
        row for row in rejected_rows if row.get("rejected_edge_available_for_diagnostics") is not True
    ]
    source_degree: dict[str, int] = {}
    target_degree: dict[str, int] = {}
    for row in rows:
        source_id = str(row.get("source_visible_person_base_id", ""))
        target_id = str(row.get("target_visible_person_base_id", ""))
        source_degree[source_id] = source_degree.get(source_id, 0) + 1
        target_degree[target_id] = target_degree.get(target_id, 0) + 1
    source_bound = int(payload.get("max_source_degree", 0) or 0)
    target_bound = int(payload.get("max_target_degree", 0) or 0)
    degree_violations = [
        {"side": "source", "visible_person_base_id": key, "degree": value}
        for key, value in source_degree.items()
        if source_bound and value > source_bound
    ] + [
        {"side": "target", "visible_person_base_id": key, "degree": value}
        for key, value in target_degree.items()
        if target_bound and value > target_bound
    ]
    input_count = int(payload.get("summary", {}).get("input_edge_count", len(rows) + len(rejected_rows)))
    coverage_ok = len(rows) + len(rejected_rows) == input_count
    forbidden = forbidden_keys_present(payload)
    passed = not invalid_accepts and not missing_diagnostics and not degree_violations and coverage_ok and not forbidden
    return guardrail_payload(
        {
            "artifact": "m5_4c_quality_gated_edge_validation",
            "created_at": utc_now(),
            "passed": passed,
            "quality_gated_edge_count": len(rows),
            "rejected_edge_count": len(rejected_rows),
            "input_edge_count": input_count,
            "all_rejected_edges_available_for_diagnostics": coverage_ok and not missing_diagnostics,
            "invalid_entity_accepted_edge_count": len(invalid_accepts),
            "degree_violation_count": len(degree_violations),
            "degree_violations": degree_violations[:50],
            "forbidden_keys_present": forbidden,
            "deterministic_output_hash": semantic_hash(
                [
                    {
                        "edge": row.get("original_continuity_edge_id"),
                        "source": row.get("source_visible_person_base_id"),
                        "target": row.get("target_visible_person_base_id"),
                    }
                    for row in rows
                ]
            ),
        }
    )
