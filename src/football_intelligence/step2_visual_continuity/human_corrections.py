# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID
from football_intelligence.step1_visual_reconstruction.io import STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH
from football_intelligence.step2_visual_continuity.grouping import build_group_payload
from football_intelligence.step2_visual_continuity.io import (
    STEP2M1_CORRECTION_AUDIT_ROWS_PATH,
    STEP2M1_GROUP_ROWS_SANDBOX_PATH,
    STEP2M1_REVIEW_CANDIDATE_ROWS_PATH,
    STEP2M1_REVIEWED_DECISIONS_PATH,
    STEP2M1_TRAINING_EXAMPLES_PATH,
    read_edge_candidate_payload,
    read_json,
    write_edge_candidate_artifacts,
    write_human_corrected_edge_artifacts,
    write_json,
    write_jsonl,
)
from football_intelligence.step2_visual_continuity.review_validation import (
    BULK_ACCEPT_DECISION,
    ACCEPT_DECISION,
    REJECT_DECISION,
    UNSURE_DECISION,
    validate_reviewed_decision_payload,
    write_review_progress_and_decision_summaries,
)
from football_intelligence.step2_visual_continuity.schema import (
    VISUAL_ONLY_WARNING,
    assert_no_forbidden_keys,
    guardrail_stamp,
    rows_from_payload,
    utc_iso,
    visual_stamp,
)
from football_intelligence.step2_visual_continuity.validation import build_and_write_validation_outputs


def final_edge_state_for_decision(decision: str) -> str:
    if decision == ACCEPT_DECISION:
        return "accepted_visual_continuity_edge"
    if decision == BULK_ACCEPT_DECISION:
        return "bulk_accepted_visual_continuity_edge"
    if decision == REJECT_DECISION:
        return "rejected_visual_continuity_edge"
    if decision == UNSURE_DECISION:
        return "unsure_needs_later_review"
    return "unreviewed_visual_continuity_edge"


def audit_row_for_decision(candidate: dict[str, Any], decision: dict[str, Any], corrected_edge: dict[str, Any]) -> dict[str, Any]:
    row = {
        "step2m1_review_candidate_id": decision.get("step2m1_review_candidate_id", ""),
        "continuity_edge_id": decision.get("continuity_edge_id", ""),
        "source_visible_person_base_id": decision.get("source_visible_person_base_id", ""),
        "target_visible_person_base_id": decision.get("target_visible_person_base_id", ""),
        "source_frame_sequence": decision.get("source_frame_sequence", -1),
        "target_frame_sequence": decision.get("target_frame_sequence", -1),
        "review_bucket": candidate.get("review_bucket", ""),
        "proposed_edge_state": candidate.get("proposed_edge_state", ""),
        "human_review_decision": decision.get("human_review_decision", ""),
        "final_edge_state_sandbox": corrected_edge.get("final_edge_state_sandbox", ""),
        "edge_score_sandbox": candidate.get("edge_score_sandbox", 0.0),
        "uncertainty_score": candidate.get("uncertainty_score", 0.0),
        "uncertainty_reasons": candidate.get("uncertainty_reasons", []),
        "human_confirmed": decision.get("human_confirmed") is True,
        "reviewer_name": decision.get("reviewer_name", ""),
        "notes": decision.get("notes", ""),
        "reviewed_at": decision.get("reviewed_at", ""),
    }
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def training_example_for_decision(candidate: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    row = {
        "artifact": "step2m1_visual_continuity_training_example",
        "continuity_edge_id": candidate.get("continuity_edge_id", ""),
        "step2m1_review_candidate_id": candidate.get("step2m1_review_candidate_id", ""),
        "source_visible_person_base_id": candidate.get("source_visible_person_base_id", ""),
        "target_visible_person_base_id": candidate.get("target_visible_person_base_id", ""),
        "source_frame_sequence": candidate.get("source_frame_sequence", -1),
        "target_frame_sequence": candidate.get("target_frame_sequence", -1),
        "frame_gap": candidate.get("frame_gap", 0),
        "edge_feature_summary": candidate.get("edge_feature_summary", {}),
        "review_bucket": candidate.get("review_bucket", ""),
        "proposed_edge_state": candidate.get("proposed_edge_state", ""),
        "human_visual_continuity_decision": decision.get("human_review_decision", ""),
        "human_confirmed": decision.get("human_confirmed") is True,
    }
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def apply_reviewed_decisions_payloads(
    node_payload: dict[str, Any],
    edge_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    validation, usable_reviews = validate_reviewed_decision_payload(candidate_payload, reviewed_payload)
    if not validation.get("reviewed_decisions_valid", False):
        raise ValueError(f"Step2.M1 reviewed visual-continuity decisions are invalid: {validation['validation_errors']}")
    candidates = {
        str(row.get("step2m1_review_candidate_id", "")): row
        for row in rows_from_payload(candidate_payload)
        if row.get("step2m1_review_candidate_id")
    }
    reviews_by_edge_id = {str(row.get("continuity_edge_id", "")): row for row in usable_reviews}
    corrected_edges: list[dict[str, Any]] = []
    corrected_by_edge_id: dict[str, dict[str, Any]] = {}
    for edge in rows_from_payload(edge_payload):
        corrected = dict(edge)
        review = reviews_by_edge_id.get(str(edge.get("continuity_edge_id", "")))
        if review:
            final_state = final_edge_state_for_decision(str(review.get("human_review_decision", "")))
            corrected.update(
                {
                    "human_review_decision": review.get("human_review_decision", ""),
                    "human_confirmed": True,
                    "reviewer_name": review.get("reviewer_name", ""),
                    "reviewed_at": review.get("reviewed_at", ""),
                    "review_notes": review.get("notes", ""),
                    "final_edge_state_sandbox": final_state,
                    "step2m1_review_required": final_state == "unsure_needs_later_review",
                }
            )
        else:
            corrected.setdefault("human_confirmed", False)
            corrected.setdefault("final_edge_state_sandbox", "unreviewed_visual_continuity_edge")
        visual_stamp(corrected)
        assert_no_forbidden_keys(corrected)
        corrected_edges.append(corrected)
        corrected_by_edge_id[str(edge.get("continuity_edge_id", ""))] = corrected

    audit_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    missing_audit_candidate_ids = []
    for review in usable_reviews:
        candidate = candidates.get(str(review.get("step2m1_review_candidate_id", "")))
        corrected_edge = corrected_by_edge_id.get(str(review.get("continuity_edge_id", "")))
        if not candidate or not corrected_edge:
            missing_audit_candidate_ids.append(str(review.get("step2m1_review_candidate_id", "")))
            continue
        audit_rows.append(audit_row_for_decision(candidate, review, corrected_edge))
        training_rows.append(training_example_for_decision(candidate, review))

    decision_counts = Counter(str(row.get("human_review_decision", "")) for row in usable_reviews)
    final_state_counts = Counter(str(row.get("final_edge_state_sandbox", "")) for row in corrected_edges)
    corrected_payload = guardrail_stamp(
        {
            "artifact": "step2m1_human_corrected_visual_continuity_edge_rows",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "rows": corrected_edges,
            "summary": {
                "reviewed_decision_rows": len(usable_reviews),
                "edge_rows": len(corrected_edges),
                "human_review_decision_counts": dict(sorted(decision_counts.items())),
                "final_edge_state_counts": dict(sorted(final_state_counts.items())),
                "audit_rows_expected": len(usable_reviews),
                "audit_rows_emitted": len(audit_rows),
                "reviewed_decisions_all_audited": len(audit_rows) == len(usable_reviews) and not missing_audit_candidate_ids,
                "training_examples_exported": len(training_rows),
                "visual_only_warning": VISUAL_ONLY_WARNING,
            },
        }
    )
    audit_payload = guardrail_stamp(
        {
            "artifact": "step2m1_visual_continuity_correction_audit_rows",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "rows": audit_rows,
            "summary": {
                "audit_rows": len(audit_rows),
                "reviewed_decision_rows": len(usable_reviews),
                "reviewed_decisions_all_audited": len(audit_rows) == len(usable_reviews) and not missing_audit_candidate_ids,
                "missing_audit_candidate_ids": missing_audit_candidate_ids,
                "human_review_decision_counts": dict(sorted(decision_counts.items())),
            },
        }
    )
    assert_no_forbidden_keys(corrected_payload)
    assert_no_forbidden_keys(audit_payload)
    group_payload = build_group_payload(node_payload, corrected_payload)
    return corrected_payload, audit_payload, training_rows, group_payload


def apply_and_write_reviewed_decisions(
    node_payload: dict[str, Any] | None = None,
    edge_payload: dict[str, Any] | None = None,
    candidate_payload: dict[str, Any] | None = None,
    reviewed_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    from football_intelligence.step2_visual_continuity.io import STEP2M1_NODE_ROWS_PATH

    node_payload = node_payload or read_json(STEP2M1_NODE_ROWS_PATH)
    edge_payload = edge_payload or read_edge_candidate_payload()
    candidate_payload = candidate_payload or read_json(STEP2M1_REVIEW_CANDIDATE_ROWS_PATH)
    reviewed_payload = reviewed_payload or read_json(STEP2M1_REVIEWED_DECISIONS_PATH)
    write_edge_candidate_artifacts(edge_payload)
    corrected, audit, training_rows, group_payload = apply_reviewed_decisions_payloads(
        node_payload,
        edge_payload,
        candidate_payload,
        reviewed_payload,
    )
    write_human_corrected_edge_artifacts(corrected)
    write_json(STEP2M1_CORRECTION_AUDIT_ROWS_PATH, audit)
    write_jsonl(STEP2M1_TRAINING_EXAMPLES_PATH, training_rows)
    write_json(STEP2M1_GROUP_ROWS_SANDBOX_PATH, group_payload)
    progress, decision = write_review_progress_and_decision_summaries(candidate_payload, reviewed_payload)
    f3_payload = read_json(STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH)
    validation_outputs = build_and_write_validation_outputs(
        f3_payload=f3_payload,
        node_payload=node_payload,
        edge_payload=corrected,
        group_payload=group_payload,
        review_payload=candidate_payload,
        review_progress=progress,
        review_decision=decision,
        correction_audit_payload=audit,
        corrected_edge_rows_available=True,
        post_review_validation_refreshed=True,
    )
    return corrected, audit, training_rows, group_payload, validation_outputs
