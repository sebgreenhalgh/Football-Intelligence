# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state import (
    ALLOWED_FUSED_ROLE_STATES,
    F1_FORBIDDEN_KEYS,
    role_group,
    role_team_context,
)
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_review_validation import (
    ACCEPT_DECISIONS,
    UNSURE_DECISION,
    validate_reviewed_decision_payload,
)
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH,
    STEP1F2_REVIEW_CANDIDATE_ROWS_PATH,
    STEP1F2_REVIEWED_DECISIONS_PATH,
    STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH,
    STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_AUDIT_ROWS_PATH,
    STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_REPORT_PATH,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    safe_float,
)


ALLOWED_F3_FINAL_ROLE_STATES = set(ALLOWED_FUSED_ROLE_STATES)
F3_ROLE_BOOLEAN_FIELDS = {
    "team_1_outfield_visual_context": "step1f3_team_1_outfield_visual_context",
    "team_2_outfield_visual_context": "step1f3_team_2_outfield_visual_context",
    "team_unknown_outfield_visual_context": "step1f3_team_unknown_outfield_visual_context",
    "team_1_goalkeeper_visual_context": "step1f3_team_1_goalkeeper_visual_context",
    "team_2_goalkeeper_visual_context": "step1f3_team_2_goalkeeper_visual_context",
    "goalkeeper_unknown_team_visual_context": "step1f3_goalkeeper_unknown_team_visual_context",
    "official_referee_visual_context": "step1f3_official_referee_visual_context",
    "assistant_or_line_official_visual_context": "step1f3_assistant_or_line_official_visual_context",
    "off_pitch_context_person_visual_context": "step1f3_off_pitch_context_person_visual_context",
    "bad_detection_or_not_person": "step1f3_bad_detection_or_not_person",
    "unknown_visible_person_visual_context": "step1f3_unknown_visible_person_visual_context",
}
F3_FORBIDDEN_KEYS = set(F1_FORBIDDEN_KEYS) | {
    "official_exclusion",
    "official_exclusion_reason",
    "excluded_from_player_review",
    "exclude_from_player_review",
    "excluded_from_player_team_review",
    "goalkeeper_slot_id",
    "gk_slot_id",
    "assigned_goalkeeper_slot",
    "goalkeeper_identity_id",
    "expected_22_role_state",
    "expected_role_state",
    "exact_22_role_state",
    "exact_two_goalkeeper_state",
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def f1_role_state(row: dict[str, Any]) -> str:
    role = str(row.get("step1f1_fused_visual_role_state", "unknown_visible_person_visual_context"))
    return role if role in ALLOWED_F3_FINAL_ROLE_STATES else "unknown_visible_person_visual_context"


def review_is_bulk_accept(review: dict[str, Any] | None) -> bool:
    if not review:
        return False
    return str(review.get("human_review_decision", "")) == "bulk_accept_bucket" or bool(str(review.get("bulk_accept_bucket", "")))


def final_role_for_review(f1_row: dict[str, Any], review: dict[str, Any] | None) -> str:
    original = f1_role_state(f1_row)
    if review is None:
        return original
    decision = str(review.get("human_review_decision", ""))
    if decision in ACCEPT_DECISIONS:
        return original
    if decision == UNSURE_DECISION:
        return "unknown_visible_person_visual_context"
    corrected = str(review.get("human_corrected_fused_role_state", "unknown_visible_person_visual_context"))
    return corrected if corrected in ALLOWED_F3_FINAL_ROLE_STATES else "unknown_visible_person_visual_context"


def context_source_for_review(review: dict[str, Any] | None) -> str:
    if review is None:
        return "f1_not_reviewed_retained"
    decision = str(review.get("human_review_decision", ""))
    if review_is_bulk_accept(review):
        return "f2_bulk_accepted"
    if decision == "accept_f1_role_state":
        return "f2_human_accepted"
    if decision == UNSURE_DECISION:
        return "f2_human_unsure_downgraded_to_unknown"
    return "f2_human_corrected"


def correction_action_for_review(review: dict[str, Any]) -> str:
    decision = str(review.get("human_review_decision", ""))
    if review_is_bulk_accept(review):
        return "human_bulk_accept_retained"
    if decision == "accept_f1_role_state":
        return "human_accept_retained"
    if decision == UNSURE_DECISION:
        return "human_unsure_downgraded_to_unknown"
    return "human_corrected_fused_role_state"


def confidence_from_review(f1_row: dict[str, Any], review: dict[str, Any] | None, final_role: str) -> float:
    original_confidence = round(max(0.0, min(1.0, safe_float(f1_row.get("step1f1_role_confidence"), 0.0))), 4)
    if review is None or str(review.get("human_review_decision", "")) in ACCEPT_DECISIONS:
        return original_confidence
    score = {"high": 0.98, "medium": 0.90, "low": 0.65}.get(str(review.get("human_review_confidence", "medium")), 0.90)
    if final_role == "unknown_visible_person_visual_context":
        score = min(score, 0.65)
    if final_role == "bad_detection_or_not_person":
        score = max(score, 0.85)
    return round(score, 4)


def warning_flags_for_review(f1_row: dict[str, Any], review: dict[str, Any] | None, final_role: str) -> list[str]:
    flags = list(dict.fromkeys([str(flag) for flag in f1_row.get("step1f1_warning_flags", [])] + [str(flag) for flag in f1_row.get("step1f1_conflict_flags", [])]))
    if review is not None:
        flags.append(context_source_for_review(review))
    if final_role == "unknown_visible_person_visual_context":
        flags.append("unknown_visible_person_visual_context_requires_future_review")
    return list(dict.fromkeys(flag for flag in flags if flag))


def review_required_for_final(f1_row: dict[str, Any], review: dict[str, Any] | None, final_role: str) -> bool:
    if review is None:
        return bool(f1_row.get("step1f1_review_required") is True)
    decision = str(review.get("human_review_decision", ""))
    return decision == UNSURE_DECISION or final_role == "unknown_visible_person_visual_context"


def correction_reason(decision: str, original: str, final_role: str, review_required: bool) -> str:
    parts = [decision or "not_reviewed", f"f1={original}", f"f3={final_role}"]
    if review_required:
        parts.append("review_required")
    parts.append("visual_context_only_not_slot_identity_or_metric")
    return ";".join(parts)


def reviewed_decision_indexes(
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    validation, usable_rows = validate_reviewed_decision_payload(candidate_payload, reviewed_payload)
    candidates_by_id = {
        str(row.get("step1f2_review_candidate_id", "")): row
        for row in candidate_payload.get("rows", [])
        if row.get("step1f2_review_candidate_id")
    }
    reviews_by_visible_id: dict[str, dict[str, Any]] = {}
    for row in usable_rows:
        candidate = candidates_by_id.get(str(row.get("step1f2_review_candidate_id", "")), {})
        if str(candidate.get("visible_person_base_id", "")) != str(row.get("visible_person_base_id", "")):
            continue
        reviews_by_visible_id[str(row.get("visible_person_base_id", ""))] = row
    return validation, usable_rows, candidates_by_id, reviews_by_visible_id


def build_f3_row(f1_row: dict[str, Any], review: dict[str, Any] | None) -> dict[str, Any]:
    original = f1_role_state(f1_row)
    final_role = final_role_for_review(f1_row, review)
    decision = str(review.get("human_review_decision", "")) if review else ""
    review_required = review_required_for_final(f1_row, review, final_role)
    out = dict(f1_row)
    out.update(
        {
            "step1f3_final_visual_role_state": final_role,
            "step1f3_final_visual_role_group": role_group(final_role),
            "step1f3_role_team_context": role_team_context(final_role),
            "step1f3_role_confidence": confidence_from_review(f1_row, review, final_role),
            "step1f3_context_source": context_source_for_review(review),
            "step1f3_human_reviewed": review is not None,
            "step1f3_human_review_decision": decision,
            "step1f3_human_review_confidence": "" if review is None else str(review.get("human_review_confidence", "")),
            "step1f3_review_bucket": "" if review is None else str(review.get("step1f2_review_bucket", "")),
            "step1f3_review_required": review_required,
            "step1f3_warning_flags": warning_flags_for_review(f1_row, review, final_role),
            "step1f3_correction_reason": correction_reason(decision, original, final_role, review_required),
            "step1f3_human_corrected_from_f1": bool(review is not None and final_role != original),
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
    for role, field_name in F3_ROLE_BOOLEAN_FIELDS.items():
        out[field_name] = final_role == role
    return out


def audit_row_for_review(f1_row: dict[str, Any], review: dict[str, Any], corrected_row: dict[str, Any]) -> dict[str, Any]:
    decision = str(review.get("human_review_decision", ""))
    return {
        "step1f2_review_candidate_id": review.get("step1f2_review_candidate_id", ""),
        "visible_person_base_id": review.get("visible_person_base_id", ""),
        "frame_sequence": int(safe_float(review.get("frame_sequence"), -1)),
        "review_bucket": review.get("step1f2_review_bucket", ""),
        "step1f1_fused_visual_role_state": f1_row.get("step1f1_fused_visual_role_state", ""),
        "human_review_decision": decision,
        "human_corrected_fused_role_state": review.get("human_corrected_fused_role_state", ""),
        "step1f3_final_visual_role_state": corrected_row.get("step1f3_final_visual_role_state", ""),
        "step1f3_correction_action": correction_action_for_review(review),
        "step1f3_correction_reason": corrected_row.get("step1f3_correction_reason", ""),
        "bulk_accept_bucket": review.get("bulk_accept_bucket", ""),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def forbidden_keys_present(rows: list[dict[str, Any]]) -> list[str]:
    found: set[str] = set()
    for row in rows:
        found.update(key for key in F3_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def build_human_corrected_fused_role_state_payloads(
    f1_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation, usable_reviews, _candidates_by_id, reviews_by_visible_id = reviewed_decision_indexes(candidate_payload, reviewed_payload)
    if not validation.get("reviewed_decisions_valid", False):
        raise ValueError(f"F2 reviewed decisions are invalid: {validation.get('validation_errors', [])}")
    f1_rows_by_visible_id = {
        str(row.get("visible_person_base_id", "")): row
        for row in f1_payload.get("rows", [])
        if row.get("visible_person_base_id")
    }
    corrected_rows = []
    corrected_by_visible_id: dict[str, dict[str, Any]] = {}
    for f1_row in f1_payload.get("rows", []):
        visible_id = str(f1_row.get("visible_person_base_id", ""))
        corrected = build_f3_row(f1_row, reviews_by_visible_id.get(visible_id))
        corrected_rows.append(corrected)
        corrected_by_visible_id[visible_id] = corrected

    audit_rows = []
    missing_audit_visible_ids = []
    for review in usable_reviews:
        visible_id = str(review.get("visible_person_base_id", ""))
        f1_row = f1_rows_by_visible_id.get(visible_id)
        corrected = corrected_by_visible_id.get(visible_id)
        if not f1_row or not corrected:
            missing_audit_visible_ids.append(visible_id)
            continue
        audit_rows.append(audit_row_for_review(f1_row, review, corrected))

    f1_counts = Counter(str(row.get("step1f1_fused_visual_role_state", "")) for row in f1_payload.get("rows", []))
    f3_counts = Counter(str(row.get("step1f3_final_visual_role_state", "")) for row in corrected_rows)
    group_counts = Counter(str(row.get("step1f3_final_visual_role_group", "")) for row in corrected_rows)
    source_counts = Counter(str(row.get("step1f3_context_source", "")) for row in corrected_rows)
    action_counts = Counter(str(row.get("step1f3_correction_action", "")) for row in audit_rows)
    review_required_count = sum(1 for row in corrected_rows if row.get("step1f3_review_required") is True)
    summary = {
        "artifact": "step1f3_human_corrected_fused_visual_role_state_summary",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
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
        "exact_22_forcing_performed": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "auto_promoted": False,
        "f1_row_count": len(f1_payload.get("rows", [])),
        "f3_row_count": len(corrected_rows),
        "one_row_per_f1_row": len(f1_payload.get("rows", [])) == len(corrected_rows),
        "input_visible_person_base_ids_aligned": [
            str(row.get("visible_person_base_id", "")) for row in f1_payload.get("rows", [])
        ]
        == [
            str(row.get("visible_person_base_id", "")) for row in corrected_rows
        ],
        "f2_reviewed_decision_count": len(usable_reviews),
        "f2_reviewed_decisions_valid": validation.get("reviewed_decisions_valid", False),
        "f2_usable_human_confirmed_decision_rows": validation.get("usable_human_confirmed_decision_rows", 0),
        "f2_human_accepted_count": action_counts.get("human_accept_retained", 0),
        "f2_human_corrected_count": action_counts.get("human_corrected_fused_role_state", 0),
        "f2_human_unsure_count": action_counts.get("human_unsure_downgraded_to_unknown", 0),
        "f2_bulk_accepted_count": action_counts.get("human_bulk_accept_retained", 0),
        "human_fused_role_state_correction_audit_row_count": len(audit_rows),
        "audit_trail_for_every_f2_human_decision": len(audit_rows) == len(usable_reviews) and not missing_audit_visible_ids,
        "missing_audit_visible_person_base_ids": missing_audit_visible_ids,
        "f1_role_state_counts": dict(sorted(f1_counts.items())),
        "f3_final_role_state_counts": dict(sorted(f3_counts.items())),
        "f3_final_role_group_counts": dict(sorted(group_counts.items())),
        "f3_context_source_counts": dict(sorted(source_counts.items())),
        "f3_correction_action_counts": dict(sorted(action_counts.items())),
        "f3_correction_counts_by_decision": dict(sorted(Counter(str(row.get("human_review_decision", "")) for row in audit_rows if row.get("step1f3_correction_action") == "human_corrected_fused_role_state").items())),
        "f3_review_required_rows": review_required_count,
        "f3_unknown_rows": f3_counts.get("unknown_visible_person_visual_context", 0),
        "correction_rate": round(action_counts.get("human_corrected_fused_role_state", 0) / max(1, len(usable_reviews)), 4),
        "all_rows_retained_for_future_player_team_review": all(row.get("retained_for_future_player_team_review") is True for row in corrected_rows),
        "forbidden_keys_present": forbidden_keys_present(corrected_rows + audit_rows),
    }
    corrected_payload = {
        "artifact": "step1f3_human_corrected_fused_visual_role_state_rows",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
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
        "exact_22_forcing_performed": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "auto_promoted": False,
        "sandbox_only": True,
        "human_corrected_fused_visual_role_state_layer_only": True,
        "f3_is_not_slot_identity_metric_event_or_tactical_stage": True,
        "allowed_f3_final_visual_role_states": sorted(ALLOWED_F3_FINAL_ROLE_STATES),
        "rows": corrected_rows,
        "summary": summary,
    }
    audit_payload = {
        "artifact": "step1f3_human_fused_role_state_correction_audit_rows",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
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
        "exact_22_forcing_performed": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "auto_promoted": False,
        "rows": audit_rows,
        "summary": {
            **summary,
            "artifact": "step1f3_human_fused_role_state_correction_audit_summary",
            "audit_action_counts": dict(sorted(action_counts.items())),
        },
    }
    return corrected_payload, audit_payload


def human_correction_report(corrected_payload: dict[str, Any], audit_payload: dict[str, Any]) -> str:
    summary = corrected_payload.get("summary", {})
    return "\n".join(
        [
            "# Step1.F3 Human-Corrected Fused Visual Role-State Correction Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Scope: human-reviewed fused visual role-state correction layer only.",
            "- F3 is sandbox-only, non-canonical, not production-ready, not identity tracking, not a slot stage, and not a metric/event/tactical layer.",
            "- Team and goalkeeper labels are visual context labels only, not identities, slots, exact counts, tactics, or performance claims.",
            "- Officials/referees and bad-detection rows are retained for future player/team review; no exclusion/deletion is performed.",
            "",
            "## Counts",
            "",
            f"- F1 rows: {summary.get('f1_row_count', 0)}",
            f"- F3 rows: {summary.get('f3_row_count', 0)}",
            f"- F2 reviewed decisions: {summary.get('f2_reviewed_decision_count', 0)}",
            f"- Human accepted: {summary.get('f2_human_accepted_count', 0)}",
            f"- Human corrected: {summary.get('f2_human_corrected_count', 0)}",
            f"- Human unsure: {summary.get('f2_human_unsure_count', 0)}",
            f"- Bulk accepted: {summary.get('f2_bulk_accepted_count', 0)}",
            f"- Audit rows: {summary.get('human_fused_role_state_correction_audit_row_count', 0)}",
            f"- Correction rate: {summary.get('correction_rate', 0)}",
            "",
            "## F1 Baseline Role-State Counts",
            "",
            "```json",
            json.dumps(summary.get("f1_role_state_counts", {}), indent=2),
            "```",
            "",
            "## F3 Final Role-State Counts",
            "",
            "```json",
            json.dumps(summary.get("f3_final_role_state_counts", {}), indent=2),
            "```",
            "",
            "## F3 Context Sources",
            "",
            "```json",
            json.dumps(summary.get("f3_context_source_counts", {}), indent=2),
            "```",
            "",
            "## Audit Actions",
            "",
            "```json",
            json.dumps(audit_payload.get("summary", {}).get("audit_action_counts", {}), indent=2),
            "```",
        ]
    ) + "\n"


def build_and_write_human_corrected_fused_role_state() -> tuple[dict[str, Any], dict[str, Any]]:
    f1_payload = read_json(STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH)
    candidate_payload = read_json(STEP1F2_REVIEW_CANDIDATE_ROWS_PATH)
    reviewed_payload = read_json(STEP1F2_REVIEWED_DECISIONS_PATH)
    corrected_payload, audit_payload = build_human_corrected_fused_role_state_payloads(
        f1_payload,
        candidate_payload,
        reviewed_payload,
    )
    write_json(STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH, corrected_payload)
    write_json(STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_AUDIT_ROWS_PATH, audit_payload)
    write_text(STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_REPORT_PATH, human_correction_report(corrected_payload, audit_payload))
    return corrected_payload, audit_payload
