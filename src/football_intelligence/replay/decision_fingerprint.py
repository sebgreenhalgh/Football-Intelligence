from __future__ import annotations

from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprints import semantic_hash, sha256_file
from football_intelligence.replay.contracts import (
    M3T_DECISION_ALLOWED_VALUES,
    M3T_REVIEW_VERSION,
    M3T_VISUAL_EVIDENCE_VERSION,
    M5_1_CANONICAL_M3T_DECISION_SEMANTIC_HASH,
    M5_2_RAW_M3T_DECISION_SEMANTIC_HASH,
)


APPROVAL_FLAGS = [
    "human_approved",
    "approve_any_identity_tracking",
    "approve_any_player_slot_use",
    "approve_any_goalkeeper_slot_use",
    "approve_any_metric_use",
    "approve_event_or_tactical_analysis",
    "approve_exact_22_or_exact_two_goalkeeper_forcing",
    "approve_official_referee_exclusion",
    "approve_bad_detection_deletion",
    "approve_production_promotion",
]


def rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows", [])
    return [dict(row) for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def reconcile_decision_fingerprint(
    *,
    decision_payload: dict[str, Any],
    decision_path: Path,
    review_candidates: list[dict[str, Any]],
    m3t_pathlets: list[dict[str, Any]],
    selected_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    decision_rows = rows_from_payload(decision_payload)
    candidate_ids = {str(row.get("step2m3t_review_candidate_id", "")) for row in review_candidates}
    pathlet_ids = {str(row.get("pathlet_id", "")) for row in m3t_pathlets}
    edge_ids = {str(row.get("continuity_edge_id", "")) for row in selected_edges}
    decision_candidate_ids = [str(row.get("step2m3t_review_candidate_id", "")) for row in decision_rows]
    duplicate_decision_ids = sorted(
        {item for item in decision_candidate_ids if decision_candidate_ids.count(item) > 1 and item}
    )
    missing_candidate_ids = sorted(set(decision_candidate_ids) - candidate_ids)
    pathlet_references = {str(row.get("pathlet_id", "")) for row in decision_rows if row.get("pathlet_id")}
    edge_references = {str(row.get("continuity_edge_id", "")) for row in decision_rows if row.get("continuity_edge_id")}
    accepted_edge_references = {
        str(edge_id)
        for row in decision_rows
        for edge_id in row.get("accepted_continuity_edge_ids", []) or []
        if edge_id
    }
    missing_pathlet_refs = sorted(pathlet_references - pathlet_ids)
    missing_edge_refs = sorted((edge_references | accepted_edge_references) - edge_ids)
    version_mismatches = [
        row.get("step2m3t_review_candidate_id", "")
        for row in decision_rows
        if row.get("current_review_version") != M3T_REVIEW_VERSION
        or row.get("review_decisions_collected_with_review_version") != M3T_REVIEW_VERSION
        or row.get("current_visual_evidence_version") != M3T_VISUAL_EVIDENCE_VERSION
        or row.get("review_decisions_collected_with_visual_evidence_version") != M3T_VISUAL_EVIDENCE_VERSION
        or row.get("review_decisions_visual_evidence_version_matches_current") is not True
    ]
    invalid_decisions = [
        row.get("step2m3t_review_candidate_id", "")
        for row in decision_rows
        if row.get("human_review_decision") not in M3T_DECISION_ALLOWED_VALUES
    ]
    approval_flag_violations = [
        {"decision_id": row.get("step2m3t_review_candidate_id", ""), "flag": flag, "observed": row.get(flag)}
        for row in decision_rows
        for flag in APPROVAL_FLAGS
        if row.get(flag) is not False
    ]
    raw_payload_hash = semantic_hash(decision_payload)
    canonical_rows_hash = semantic_hash(decision_rows)
    return {
        "schema_version": "m5.true_replay.m3t_decision_fingerprint_reconciliation.v1",
        "decision_byte_hash": sha256_file(decision_path),
        "raw_payload_semantic_hash": raw_payload_hash,
        "raw_row_semantic_hash": canonical_rows_hash,
        "canonical_policy_semantic_hash": canonical_rows_hash,
        "canonical_policy": {
            "payload_shape": "ordered_rows",
            "excluded_fields": [],
            "excluded_paths": ["$.artifact", "$.created_at", "$.summary"],
            "ordering_policy": "preserve_decision_row_order",
        },
        "m5_1_expected_hash": M5_1_CANONICAL_M3T_DECISION_SEMANTIC_HASH,
        "m5_2_observed_hash": M5_2_RAW_M3T_DECISION_SEMANTIC_HASH,
        "explanation": (
            "M5.1 canonicalized the ordered decision rows. M5.2 closure hashed the enclosing JSON payload, "
            "which includes wrapper metadata, so the payload hash differs while the ordered rows hash matches M5.1."
        ),
        "canonical_hash_selected_for_future_runs": canonical_rows_hash,
        "decision_row_count": len(decision_rows),
        "unique_decisions": not duplicate_decision_ids,
        "duplicate_decision_ids": duplicate_decision_ids,
        "missing_candidate_ids": missing_candidate_ids,
        "missing_pathlet_refs": missing_pathlet_refs,
        "missing_edge_refs": missing_edge_refs,
        "accepted_edge_reference_count": len(accepted_edge_references),
        "version_mismatches": version_mismatches,
        "invalid_decisions": invalid_decisions,
        "approval_flag_violations": approval_flag_violations,
        "passed": (
            canonical_rows_hash == M5_1_CANONICAL_M3T_DECISION_SEMANTIC_HASH
            and raw_payload_hash == M5_2_RAW_M3T_DECISION_SEMANTIC_HASH
            and len(decision_rows) == 40
            and not duplicate_decision_ids
            and not missing_candidate_ids
            and not missing_pathlet_refs
            and not missing_edge_refs
            and not version_mismatches
            and not invalid_decisions
            and not approval_flag_violations
        ),
    }
