# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.official_context_beliefs import (
    ALLOWED_OFFICIAL_CONTEXT_BELIEFS,
    OFFICIAL_LIKE_BELIEFS,
)
from football_intelligence.step1_visual_reconstruction.official_context_review_schema import (
    validate_reviewed_decision_payload,
)
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1D1B_REVIEWED_DECISIONS_PATH,
    STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH,
    STEP1D1C_HUMAN_CORRECTION_AUDIT_ROWS_PATH,
    STEP1D1C_HUMAN_CORRECTION_REPORT_PATH,
    STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH,
    STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH,
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


ALLOWED_D1C_FINAL_OFFICIAL_CONTEXT_BELIEFS = set(ALLOWED_OFFICIAL_CONTEXT_BELIEFS)
CORRECTION_DECISIONS = {
    "correct_to_official_referee_like",
    "correct_to_assistant_or_line_official_like",
    "correct_to_non_official_context_person_like",
    "correct_to_off_pitch_context_person_like",
    "correct_to_player_like_not_official_context",
    "correct_to_bad_detection_or_not_person",
    "correct_to_unknown_official_context",
}
D1C_FORBIDDEN_KEYS = set(FORBIDDEN_OUTPUT_KEYS) | {
    "track_id",
    "persistent_player_id",
    "official_exclusion",
    "official_exclusion_reason",
    "exclude_from_player_review",
    "excluded_from_player_review",
    "excluded_from_player_team_review",
    "goalkeeper_classification",
    "goalkeeper_role",
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def decision_action(decision: str) -> str:
    if decision == "accept_d1_belief":
        return "human_accept_retained"
    if decision == "unsure_needs_later_review":
        return "human_unsure_downgraded_to_unknown"
    return "human_corrected_context_belief"


def source_for_decision(decision: str) -> str:
    if decision == "accept_d1_belief":
        return "d1b_human_accepted"
    if decision == "unsure_needs_later_review":
        return "d1b_human_unsure_downgraded_to_unknown"
    return "d1b_human_corrected"


def confidence_from_review(d1_row: dict[str, Any], review: dict[str, Any] | None, final_belief: str) -> float:
    if review is None or review.get("human_review_decision") == "accept_d1_belief":
        return round(safe_float(d1_row.get("official_context_belief_confidence")), 4)
    score = {"high": 0.98, "medium": 0.90, "low": 0.65}.get(str(review.get("human_review_confidence", "medium")), 0.75)
    if final_belief == "unknown_official_context":
        score = min(score, 0.65)
    return round(score, 4)


def final_belief_for_review(d1_row: dict[str, Any], review: dict[str, Any] | None) -> str:
    if review is None:
        return str(d1_row.get("official_context_belief", "unknown_official_context"))
    decision = str(review.get("human_review_decision", ""))
    if decision == "accept_d1_belief":
        return str(d1_row.get("official_context_belief", "unknown_official_context"))
    if decision == "unsure_needs_later_review":
        return "unknown_official_context"
    corrected = str(review.get("human_corrected_official_context_belief", "unknown_official_context"))
    return corrected if corrected in ALLOWED_D1C_FINAL_OFFICIAL_CONTEXT_BELIEFS else "unknown_official_context"


def final_state_for_review(d1_row: dict[str, Any], review: dict[str, Any] | None, final_belief: str) -> str:
    if review is None or review.get("human_review_decision") == "accept_d1_belief":
        return str(d1_row.get("official_context_belief_state", "review_required"))
    if review.get("human_review_decision") == "unsure_needs_later_review":
        return "review_required"
    if final_belief == "bad_detection_or_not_person":
        return "bad_detection_review_required"
    if final_belief == "unknown_official_context":
        return "review_required"
    return "human_corrected_visual_context"


def correction_reason(decision: str, d1_belief: str, final_belief: str, review_required: bool) -> str:
    parts = [decision or "not_reviewed", f"d1={d1_belief}", f"final={final_belief}"]
    if review_required:
        parts.append("review_required")
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
        str(row.get("step1d1_review_candidate_id", "")): row
        for row in candidate_payload.get("rows", [])
        if row.get("step1d1_review_candidate_id")
    }
    reviews_by_visible_id = {
        str(row.get("visible_person_base_id", "")): row
        for row in usable_rows
        if row.get("visible_person_base_id")
    }
    return validation, usable_rows, candidates_by_id, reviews_by_visible_id


def build_d1c_row(d1_row: dict[str, Any], review: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(d1_row)
    d1_belief = str(d1_row.get("official_context_belief", "unknown_official_context"))
    decision = str(review.get("human_review_decision", "")) if review else ""
    final_belief = final_belief_for_review(d1_row, review)
    if final_belief not in ALLOWED_D1C_FINAL_OFFICIAL_CONTEXT_BELIEFS:
        final_belief = "unknown_official_context"
    human_reviewed = review is not None
    final_confidence = confidence_from_review(d1_row, review, final_belief)
    review_required = bool(
        (review is None and d1_row.get("official_context_review_required") is True)
        or decision == "unsure_needs_later_review"
        or final_belief in {"unknown_official_context", "bad_detection_or_not_person"}
    )
    out.update(
        {
            "d1_official_context_belief": d1_belief,
            "d1_official_context_belief_state": d1_row.get("official_context_belief_state", ""),
            "d1_official_context_belief_confidence": safe_float(d1_row.get("official_context_belief_confidence")),
            "d1_official_context_review_required": d1_row.get("official_context_review_required") is True,
            "d1c_final_official_context_belief": final_belief,
            "d1c_final_official_context_belief_state": final_state_for_review(d1_row, review, final_belief),
            "d1c_final_official_context_belief_confidence": final_confidence,
            "d1c_context_source": source_for_decision(decision) if review else "d1_not_reviewed_retained",
            "d1c_human_reviewed": human_reviewed,
            "d1c_human_review_decision": decision,
            "d1c_human_review_confidence": "" if review is None else str(review.get("human_review_confidence", "")),
            "d1c_review_required": review_required,
            "d1c_correction_reason": correction_reason(decision, d1_belief, final_belief, review_required),
            "d1c_human_corrected_from_d1": bool(human_reviewed and final_belief != d1_belief),
            "d1c_bad_detection_or_not_person": final_belief == "bad_detection_or_not_person",
            "d1c_official_like_visual_context": final_belief in OFFICIAL_LIKE_BELIEFS,
            "d1c_assistant_or_line_official_like_visual_context": final_belief == "assistant_or_line_official_like",
            "retained_for_future_player_team_review": True,
            "eligible_for_step1e_goalkeeper_candidate": True,
            "eligible_for_identity_tracking": False,
            "eligible_for_player_slot_assignment": False,
            "eligible_for_metric_use": False,
            "visual_only_warning": VISUAL_ONLY_WARNING,
            "do_not_use_for_metrics": True,
            "production_ready": PRODUCTION_READY,
            "auto_promoted": False,
        }
    )
    return out


def audit_row_for_review(d1_row: dict[str, Any], review: dict[str, Any], corrected_row: dict[str, Any]) -> dict[str, Any]:
    decision = str(review.get("human_review_decision", ""))
    return {
        "step1d1_review_candidate_id": review.get("step1d1_review_candidate_id", ""),
        "visible_person_base_id": review.get("visible_person_base_id", ""),
        "frame_sequence": int(safe_float(review.get("frame_sequence"), -1)),
        "d1_official_context_belief": d1_row.get("official_context_belief", ""),
        "human_review_decision": decision,
        "human_corrected_official_context_belief": review.get("human_corrected_official_context_belief", ""),
        "d1c_final_official_context_belief": corrected_row.get("d1c_final_official_context_belief", ""),
        "d1c_correction_action": decision_action(decision),
        "d1c_correction_reason": corrected_row.get("d1c_correction_reason", ""),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def forbidden_keys_present(rows: list[dict[str, Any]]) -> list[str]:
    found: set[str] = set()
    for row in rows:
        found.update(key for key in D1C_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def build_human_corrected_official_context_payloads(
    d1_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation, usable_reviews, _candidates_by_id, reviews_by_visible_id = reviewed_decision_indexes(candidate_payload, reviewed_payload)
    if not validation.get("reviewed_decisions_valid", False):
        raise ValueError(f"D1b reviewed decisions are invalid: {validation.get('validation_errors', [])}")
    d1_rows_by_visible_id = {
        str(row.get("visible_person_base_id", "")): row
        for row in d1_payload.get("rows", [])
        if row.get("visible_person_base_id")
    }
    corrected_rows = []
    corrected_by_visible_id: dict[str, dict[str, Any]] = {}
    for d1_row in d1_payload.get("rows", []):
        visible_id = str(d1_row.get("visible_person_base_id", ""))
        corrected = build_d1c_row(d1_row, reviews_by_visible_id.get(visible_id))
        corrected_rows.append(corrected)
        corrected_by_visible_id[visible_id] = corrected

    audit_rows = []
    missing_audit_visible_ids = []
    for review in usable_reviews:
        visible_id = str(review.get("visible_person_base_id", ""))
        d1_row = d1_rows_by_visible_id.get(visible_id)
        corrected = corrected_by_visible_id.get(visible_id)
        if not d1_row or not corrected:
            missing_audit_visible_ids.append(visible_id)
            continue
        audit_rows.append(audit_row_for_review(d1_row, review, corrected))

    d1_counts = Counter(str(row.get("official_context_belief", "")) for row in d1_payload.get("rows", []))
    d1c_counts = Counter(str(row.get("d1c_final_official_context_belief", "")) for row in corrected_rows)
    source_counts = Counter(str(row.get("d1c_context_source", "")) for row in corrected_rows)
    action_counts = Counter(str(row.get("d1c_correction_action", "")) for row in audit_rows)
    correction_rows = [
        row
        for row in audit_rows
        if row.get("d1c_correction_action") == "human_corrected_context_belief"
    ]
    summary = {
        "artifact": "step1d1c_human_corrected_official_context_summary",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "d1_row_count": len(d1_payload.get("rows", [])),
        "d1c_row_count": len(corrected_rows),
        "one_row_per_d1_belief_row": len(d1_payload.get("rows", [])) == len(corrected_rows),
        "d1b_reviewed_decision_count": len(usable_reviews),
        "d1b_reviewed_decisions_valid": validation.get("reviewed_decisions_valid", False),
        "d1b_usable_human_confirmed_decision_rows": validation.get("usable_human_confirmed_decision_rows", 0),
        "d1b_human_accepted_count": action_counts.get("human_accept_retained", 0),
        "d1b_human_corrected_count": action_counts.get("human_corrected_context_belief", 0),
        "d1b_human_unsure_count": action_counts.get("human_unsure_downgraded_to_unknown", 0),
        "human_correction_audit_row_count": len(audit_rows),
        "audit_trail_for_every_human_review": len(audit_rows) == len(usable_reviews) and not missing_audit_visible_ids,
        "missing_audit_visible_person_base_ids": missing_audit_visible_ids,
        "d1_original_belief_counts": dict(sorted(d1_counts.items())),
        "d1c_final_belief_counts": dict(sorted(d1c_counts.items())),
        "d1c_context_source_counts": dict(sorted(source_counts.items())),
        "d1c_correction_action_counts": dict(sorted(action_counts.items())),
        "official_context_correction_counts": dict(sorted(Counter(str(row.get("human_review_decision", "")) for row in correction_rows).items())),
        "assistant_or_line_official_like_count": d1c_counts.get("assistant_or_line_official_like", 0),
        "official_referee_like_count": d1c_counts.get("official_referee_like", 0),
        "bad_detection_or_not_person_count": d1c_counts.get("bad_detection_or_not_person", 0),
        "all_rows_retained_for_future_player_team_review": all(row.get("retained_for_future_player_team_review") is True for row in corrected_rows),
        "forbidden_keys_present": forbidden_keys_present(corrected_rows + audit_rows),
    }
    corrected_payload = {
        "artifact": "step1d1c_human_corrected_official_context_rows",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "sandbox_only": True,
        "human_reviewed_official_context_correction_layer_only": True,
        "d1c_is_not_official_exclusion_stage": True,
        "allowed_d1c_final_official_context_beliefs": sorted(ALLOWED_D1C_FINAL_OFFICIAL_CONTEXT_BELIEFS),
        "rows": corrected_rows,
        "summary": summary,
    }
    audit_payload = {
        "artifact": "step1d1c_human_correction_audit_rows",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "rows": audit_rows,
        "summary": {
            **summary,
            "artifact": "step1d1c_human_correction_audit_summary",
            "audit_action_counts": dict(sorted(action_counts.items())),
        },
    }
    return corrected_payload, audit_payload


def human_correction_report(corrected_payload: dict[str, Any], audit_payload: dict[str, Any]) -> str:
    summary = corrected_payload.get("summary", {})
    return "\n".join(
        [
            "# Step1.D1c Human Correction Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Scope: human-reviewed visual official/context correction layer only.",
            "- D1c is not canonical, not production-ready, and not an official/referee exclusion stage.",
            "- Assistant/line-official-like is a visual context label only, not identity, role-slot, or metric evidence.",
            "- No identity tracking, player slots, expected roles, goalkeeper classification, official/referee exclusion, projected-pitch truth, tactical/physical/football metrics, project default changes, registry changes, or promotion were performed.",
            "",
            "## Counts",
            "",
            f"- D1 rows: {summary.get('d1_row_count', 0)}",
            f"- D1c rows: {summary.get('d1c_row_count', 0)}",
            f"- D1b reviewed decisions: {summary.get('d1b_reviewed_decision_count', 0)}",
            f"- Human accepted: {summary.get('d1b_human_accepted_count', 0)}",
            f"- Human corrected: {summary.get('d1b_human_corrected_count', 0)}",
            f"- Human unsure: {summary.get('d1b_human_unsure_count', 0)}",
            f"- Audit rows: {summary.get('human_correction_audit_row_count', 0)}",
            "",
            "## D1 Original Beliefs",
            "",
            "```json",
            json.dumps(summary.get("d1_original_belief_counts", {}), indent=2),
            "```",
            "",
            "## D1c Final Beliefs",
            "",
            "```json",
            json.dumps(summary.get("d1c_final_belief_counts", {}), indent=2),
            "```",
            "",
            "## Audit Actions",
            "",
            "```json",
            json.dumps(audit_payload.get("summary", {}).get("audit_action_counts", {}), indent=2),
            "```",
        ]
    ) + "\n"


def build_and_write_human_corrected_official_context() -> tuple[dict[str, Any], dict[str, Any]]:
    d1_payload = read_json(STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH)
    candidate_payload = read_json(STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH)
    reviewed_payload = read_json(STEP1D1B_REVIEWED_DECISIONS_PATH)
    corrected_payload, audit_payload = build_human_corrected_official_context_payloads(
        d1_payload,
        candidate_payload,
        reviewed_payload,
    )
    write_json(STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH, corrected_payload)
    write_json(STEP1D1C_HUMAN_CORRECTION_AUDIT_ROWS_PATH, audit_payload)
    write_text(STEP1D1C_HUMAN_CORRECTION_REPORT_PATH, human_correction_report(corrected_payload, audit_payload))
    return corrected_payload, audit_payload
