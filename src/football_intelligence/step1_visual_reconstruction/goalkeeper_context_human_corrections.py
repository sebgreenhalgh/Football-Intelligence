# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_beliefs import (
    ALLOWED_E1_GOALKEEPER_CONTEXT_BELIEFS,
    GOALKEEPER_LIKE_BELIEFS,
)
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_schema import validate_reviewed_decision_payload
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH,
    STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH,
    STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_AUDIT_ROWS_PATH,
    STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_REPORT_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    FORBIDDEN_OUTPUT_KEYS,
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    safe_float,
)


ALLOWED_E1C_FINAL_GOALKEEPER_CONTEXT_BELIEFS = set(ALLOWED_E1_GOALKEEPER_CONTEXT_BELIEFS)
CORRECTION_DECISIONS = {
    "correct_to_goalkeeper_like_team_1_context",
    "correct_to_goalkeeper_like_team_2_context",
    "correct_to_goalkeeper_like_unknown_team_context",
    "correct_to_outfield_player_like_not_goalkeeper",
    "correct_to_official_or_context_not_goalkeeper",
    "correct_to_bad_detection_or_not_person",
    "correct_to_unknown_goalkeeper_context",
}
E1C_FORBIDDEN_KEYS = set(FORBIDDEN_OUTPUT_KEYS) | {
    "track_id",
    "persistent_player_id",
    "official_exclusion",
    "official_exclusion_reason",
    "exclude_from_player_review",
    "excluded_from_player_review",
    "excluded_from_player_team_review",
    "goalkeeper_slot_id",
    "gk_slot_id",
    "assigned_goalkeeper_slot",
    "goalkeeper_identity_id",
    "expected_22_role_state",
    "expected_role_state",
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def decision_action(decision: str) -> str:
    if decision == "accept_e1_belief":
        return "human_accept_retained"
    if decision == "unsure_needs_later_review":
        return "human_unsure_downgraded_to_unknown"
    return "human_corrected_goalkeeper_context_belief"


def source_for_decision(decision: str) -> str:
    if decision == "accept_e1_belief":
        return "e1b_human_accepted"
    if decision == "unsure_needs_later_review":
        return "e1b_human_unsure_downgraded_to_unknown"
    return "e1b_human_corrected"


def team_belief_for_final(final_belief: str) -> str:
    if final_belief == "goalkeeper_like_team_1_context":
        return "team_1"
    if final_belief == "goalkeeper_like_team_2_context":
        return "team_2"
    if final_belief == "goalkeeper_like_unknown_team_context":
        return "unknown_team"
    return "not_goalkeeper"


def final_belief_for_review(e1_row: dict[str, Any], review: dict[str, Any] | None) -> str:
    if review is None:
        return str(e1_row.get("e1_goalkeeper_context_belief", "unknown_goalkeeper_context"))
    decision = str(review.get("human_review_decision", ""))
    if decision == "accept_e1_belief":
        return str(e1_row.get("e1_goalkeeper_context_belief", "unknown_goalkeeper_context"))
    if decision == "unsure_needs_later_review":
        return "unknown_goalkeeper_context"
    corrected = str(review.get("human_corrected_goalkeeper_context_belief", "unknown_goalkeeper_context"))
    return corrected if corrected in ALLOWED_E1C_FINAL_GOALKEEPER_CONTEXT_BELIEFS else "unknown_goalkeeper_context"


def confidence_from_review(e1_row: dict[str, Any], review: dict[str, Any] | None, final_belief: str) -> float:
    if review is None or review.get("human_review_decision") == "accept_e1_belief":
        return round(safe_float(e1_row.get("e1_goalkeeper_context_belief_confidence")), 4)
    score = {"high": 0.98, "medium": 0.90, "low": 0.65}.get(str(review.get("human_review_confidence", "medium")), 0.75)
    if final_belief == "unknown_goalkeeper_context":
        score = min(score, 0.65)
    if final_belief == "bad_detection_or_not_person":
        score = max(score, 0.85)
    return round(score, 4)


def final_state_for_review(e1_row: dict[str, Any], review: dict[str, Any] | None, final_belief: str) -> str:
    if review is None or review.get("human_review_decision") == "accept_e1_belief":
        return str(e1_row.get("e1_goalkeeper_context_belief_state", "review_required"))
    if review.get("human_review_decision") == "unsure_needs_later_review":
        return "review_required"
    if final_belief == "bad_detection_or_not_person":
        return "bad_detection_review_required"
    if final_belief == "unknown_goalkeeper_context":
        return "review_required"
    return "human_corrected_visual_context"


def correction_reason(decision: str, e1_belief: str, final_belief: str, review_required: bool) -> str:
    parts = [decision or "not_reviewed", f"e1={e1_belief}", f"final={final_belief}"]
    if review_required:
        parts.append("review_required")
    parts.append("visual_context_only_not_slot_identity_or_metric")
    return ";".join(parts)


def reviewed_decision_indexes(
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    validation, usable_rows = validate_reviewed_decision_payload(
        candidate_payload,
        reviewed_payload,
        reviewed_decisions_loaded=True,
    )
    candidates_by_id = {
        str(row.get("step1e1_review_candidate_id", "")): row
        for row in candidate_payload.get("rows", [])
        if row.get("step1e1_review_candidate_id")
    }
    reviews_by_visible_id: dict[str, dict[str, Any]] = {}
    for row in usable_rows:
        candidate = candidates_by_id.get(str(row.get("step1e1_review_candidate_id", "")), {})
        if str(candidate.get("visible_person_base_id", "")) != str(row.get("visible_person_base_id", "")):
            continue
        reviews_by_visible_id[str(row.get("visible_person_base_id", ""))] = row
    return validation, usable_rows, candidates_by_id, reviews_by_visible_id


def build_e1c_row(e1_row: dict[str, Any], review: dict[str, Any] | None) -> dict[str, Any]:
    e1_belief = str(e1_row.get("e1_goalkeeper_context_belief", "unknown_goalkeeper_context"))
    decision = str(review.get("human_review_decision", "")) if review else ""
    final_belief = final_belief_for_review(e1_row, review)
    if final_belief not in ALLOWED_E1C_FINAL_GOALKEEPER_CONTEXT_BELIEFS:
        final_belief = "unknown_goalkeeper_context"
    human_reviewed = review is not None
    final_confidence = confidence_from_review(e1_row, review, final_belief)
    review_required = bool(
        (review is None and e1_row.get("e1_goalkeeper_context_review_required") is True)
        or decision == "unsure_needs_later_review"
        or final_belief in {"unknown_goalkeeper_context", "bad_detection_or_not_person"}
    )
    out = dict(e1_row)
    out.update(
        {
            "e1c_final_goalkeeper_context_belief": final_belief,
            "e1c_final_goalkeeper_context_belief_state": final_state_for_review(e1_row, review, final_belief),
            "e1c_final_goalkeeper_context_belief_confidence": final_confidence,
            "e1c_goalkeeper_team_belief": team_belief_for_final(final_belief),
            "e1c_context_source": source_for_decision(decision) if review else "e1_not_reviewed_retained",
            "e1c_human_reviewed": human_reviewed,
            "e1c_human_review_decision": decision,
            "e1c_human_review_confidence": "" if review is None else str(review.get("human_review_confidence", "")),
            "e1c_review_required": review_required,
            "e1c_correction_reason": correction_reason(decision, e1_belief, final_belief, review_required),
            "e1c_human_corrected_from_e1": bool(human_reviewed and final_belief != e1_belief),
            "e1c_goalkeeper_like_visual_context": final_belief in GOALKEEPER_LIKE_BELIEFS,
            "e1c_goalkeeper_like_team_1_visual_context": final_belief == "goalkeeper_like_team_1_context",
            "e1c_goalkeeper_like_team_2_visual_context": final_belief == "goalkeeper_like_team_2_context",
            "e1c_goalkeeper_like_unknown_team_visual_context": final_belief == "goalkeeper_like_unknown_team_context",
            "e1c_outfield_player_like_not_goalkeeper": final_belief == "outfield_player_like_not_goalkeeper",
            "e1c_official_or_context_not_goalkeeper": final_belief == "official_or_context_not_goalkeeper",
            "e1c_bad_detection_or_not_person": final_belief == "bad_detection_or_not_person",
            "retained_for_future_player_team_review": True,
            "eligible_for_identity_tracking": False,
            "eligible_for_player_slot_assignment": False,
            "eligible_for_goalkeeper_slot_assignment": False,
            "eligible_for_metric_use": False,
            "visual_only_warning": VISUAL_ONLY_WARNING,
            "do_not_use_for_metrics": True,
            "production_ready": PRODUCTION_READY,
            "auto_promoted": False,
        }
    )
    return out


def audit_row_for_review(e1_row: dict[str, Any], review: dict[str, Any], corrected_row: dict[str, Any]) -> dict[str, Any]:
    decision = str(review.get("human_review_decision", ""))
    return {
        "step1e1_review_candidate_id": review.get("step1e1_review_candidate_id", ""),
        "visible_person_base_id": review.get("visible_person_base_id", ""),
        "frame_sequence": int(safe_float(review.get("frame_sequence"), -1)),
        "e1_goalkeeper_context_belief": e1_row.get("e1_goalkeeper_context_belief", ""),
        "human_review_decision": decision,
        "human_corrected_goalkeeper_context_belief": review.get("human_corrected_goalkeeper_context_belief", ""),
        "e1c_final_goalkeeper_context_belief": corrected_row.get("e1c_final_goalkeeper_context_belief", ""),
        "e1c_correction_action": decision_action(decision),
        "e1c_correction_reason": corrected_row.get("e1c_correction_reason", ""),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def forbidden_keys_present(rows: list[dict[str, Any]]) -> list[str]:
    found: set[str] = set()
    for row in rows:
        found.update(key for key in E1C_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def build_human_corrected_goalkeeper_context_payloads(
    e1_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation, usable_reviews, _candidates_by_id, reviews_by_visible_id = reviewed_decision_indexes(candidate_payload, reviewed_payload)
    if not validation.get("reviewed_decisions_valid", False):
        raise ValueError(f"E1b reviewed decisions are invalid: {validation.get('validation_errors', [])}")
    e1_rows_by_visible_id = {
        str(row.get("visible_person_base_id", "")): row
        for row in e1_payload.get("rows", [])
        if row.get("visible_person_base_id")
    }
    corrected_rows = []
    corrected_by_visible_id: dict[str, dict[str, Any]] = {}
    for e1_row in e1_payload.get("rows", []):
        visible_id = str(e1_row.get("visible_person_base_id", ""))
        corrected = build_e1c_row(e1_row, reviews_by_visible_id.get(visible_id))
        corrected_rows.append(corrected)
        corrected_by_visible_id[visible_id] = corrected

    audit_rows = []
    missing_audit_visible_ids = []
    for review in usable_reviews:
        visible_id = str(review.get("visible_person_base_id", ""))
        e1_row = e1_rows_by_visible_id.get(visible_id)
        corrected = corrected_by_visible_id.get(visible_id)
        if not e1_row or not corrected:
            missing_audit_visible_ids.append(visible_id)
            continue
        audit_rows.append(audit_row_for_review(e1_row, review, corrected))

    e1_counts = Counter(str(row.get("e1_goalkeeper_context_belief", "")) for row in e1_payload.get("rows", []))
    e1c_counts = Counter(str(row.get("e1c_final_goalkeeper_context_belief", "")) for row in corrected_rows)
    source_counts = Counter(str(row.get("e1c_context_source", "")) for row in corrected_rows)
    action_counts = Counter(str(row.get("e1c_correction_action", "")) for row in audit_rows)
    correction_rows = [
        row
        for row in audit_rows
        if row.get("e1c_correction_action") == "human_corrected_goalkeeper_context_belief"
    ]
    summary = {
        "artifact": "step1e1c_human_corrected_goalkeeper_context_summary",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slot_assignment_performed": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "e1_row_count": len(e1_payload.get("rows", [])),
        "e1c_row_count": len(corrected_rows),
        "one_row_per_e1_belief_row": len(e1_payload.get("rows", [])) == len(corrected_rows),
        "e1b_reviewed_decision_count": len(usable_reviews),
        "e1b_reviewed_decisions_valid": validation.get("reviewed_decisions_valid", False),
        "e1b_usable_human_confirmed_decision_rows": validation.get("usable_human_confirmed_decision_rows", 0),
        "e1b_human_accepted_count": action_counts.get("human_accept_retained", 0),
        "e1b_human_corrected_count": action_counts.get("human_corrected_goalkeeper_context_belief", 0),
        "e1b_human_unsure_count": action_counts.get("human_unsure_downgraded_to_unknown", 0),
        "human_goalkeeper_correction_audit_row_count": len(audit_rows),
        "audit_trail_for_every_human_review": len(audit_rows) == len(usable_reviews) and not missing_audit_visible_ids,
        "missing_audit_visible_person_base_ids": missing_audit_visible_ids,
        "e1_original_belief_counts": dict(sorted(e1_counts.items())),
        "e1c_final_belief_counts": dict(sorted(e1c_counts.items())),
        "e1c_context_source_counts": dict(sorted(source_counts.items())),
        "e1c_correction_action_counts": dict(sorted(action_counts.items())),
        "goalkeeper_context_correction_counts": dict(sorted(Counter(str(row.get("human_review_decision", "")) for row in correction_rows).items())),
        "goalkeeper_like_team_1_context_count": e1c_counts.get("goalkeeper_like_team_1_context", 0),
        "goalkeeper_like_team_2_context_count": e1c_counts.get("goalkeeper_like_team_2_context", 0),
        "goalkeeper_like_unknown_team_context_count": e1c_counts.get("goalkeeper_like_unknown_team_context", 0),
        "outfield_player_like_not_goalkeeper_count": e1c_counts.get("outfield_player_like_not_goalkeeper", 0),
        "official_or_context_not_goalkeeper_count": e1c_counts.get("official_or_context_not_goalkeeper", 0),
        "bad_detection_or_not_person_count": e1c_counts.get("bad_detection_or_not_person", 0),
        "unknown_goalkeeper_context_count": e1c_counts.get("unknown_goalkeeper_context", 0),
        "all_rows_retained_for_future_player_team_review": all(row.get("retained_for_future_player_team_review") is True for row in corrected_rows),
        "forbidden_keys_present": forbidden_keys_present(corrected_rows + audit_rows),
    }
    corrected_payload = {
        "artifact": "step1e1c_human_corrected_goalkeeper_context_rows",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slot_assignment_performed": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "sandbox_only": True,
        "human_reviewed_goalkeeper_context_correction_layer_only": True,
        "e1c_is_not_goalkeeper_slot_identity_or_metric_stage": True,
        "allowed_e1c_final_goalkeeper_context_beliefs": sorted(ALLOWED_E1C_FINAL_GOALKEEPER_CONTEXT_BELIEFS),
        "rows": corrected_rows,
        "summary": summary,
    }
    audit_payload = {
        "artifact": "step1e1c_human_goalkeeper_correction_audit_rows",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slot_assignment_performed": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "rows": audit_rows,
        "summary": {
            **summary,
            "artifact": "step1e1c_human_goalkeeper_correction_audit_summary",
            "audit_action_counts": dict(sorted(action_counts.items())),
        },
    }
    return corrected_payload, audit_payload


def human_correction_report(corrected_payload: dict[str, Any], audit_payload: dict[str, Any]) -> str:
    summary = corrected_payload.get("summary", {})
    return "\n".join(
        [
            "# Step1.E1c Human Goalkeeper/Context Correction Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Scope: human-reviewed visual goalkeeper/context correction layer only.",
            "- E1c is not canonical, not production-ready, not a slot stage, not identity tracking, and not a metric layer.",
            "- Team-specific goalkeeper-like labels are visual context labels only, not identities or slots.",
            "- Bad-detection/not-person remains a visual QA belief, not deletion.",
            "",
            "## Counts",
            "",
            f"- E1 rows: {summary.get('e1_row_count', 0)}",
            f"- E1c rows: {summary.get('e1c_row_count', 0)}",
            f"- E1b reviewed decisions: {summary.get('e1b_reviewed_decision_count', 0)}",
            f"- Human accepted: {summary.get('e1b_human_accepted_count', 0)}",
            f"- Human corrected: {summary.get('e1b_human_corrected_count', 0)}",
            f"- Human unsure: {summary.get('e1b_human_unsure_count', 0)}",
            f"- Audit rows: {summary.get('human_goalkeeper_correction_audit_row_count', 0)}",
            "",
            "## E1 Original Beliefs",
            "",
            "```json",
            json.dumps(summary.get("e1_original_belief_counts", {}), indent=2),
            "```",
            "",
            "## E1c Final Beliefs",
            "",
            "```json",
            json.dumps(summary.get("e1c_final_belief_counts", {}), indent=2),
            "```",
            "",
            "## Audit Actions",
            "",
            "```json",
            json.dumps(audit_payload.get("summary", {}).get("audit_action_counts", {}), indent=2),
            "```",
        ]
    ) + "\n"


def build_and_write_human_corrected_goalkeeper_context() -> tuple[dict[str, Any], dict[str, Any]]:
    e1_payload = read_json(STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH)
    candidate_payload = read_json(STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH)
    reviewed_payload = read_json(STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH)
    corrected_payload, audit_payload = build_human_corrected_goalkeeper_context_payloads(
        e1_payload,
        candidate_payload,
        reviewed_payload,
    )
    write_json(STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH, corrected_payload)
    write_json(STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_AUDIT_ROWS_PATH, audit_payload)
    write_text(STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_REPORT_PATH, human_correction_report(corrected_payload, audit_payload))
    return corrected_payload, audit_payload
