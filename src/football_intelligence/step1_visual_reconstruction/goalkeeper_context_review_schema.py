# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_beliefs import ALLOWED_E1_GOALKEEPER_CONTEXT_BELIEFS
from football_intelligence.step1_visual_reconstruction.schema import FORBIDDEN_OUTPUT_KEYS, PRODUCTION_READY, VISUAL_ONLY_WARNING, safe_float


ALLOWED_HUMAN_GOALKEEPER_CONTEXT_DECISIONS = {
    "accept_e1_belief",
    "correct_to_goalkeeper_like_team_1_context",
    "correct_to_goalkeeper_like_team_2_context",
    "correct_to_goalkeeper_like_unknown_team_context",
    "correct_to_outfield_player_like_not_goalkeeper",
    "correct_to_official_or_context_not_goalkeeper",
    "correct_to_bad_detection_or_not_person",
    "correct_to_unknown_goalkeeper_context",
    "unsure_needs_later_review",
}

DECISION_TO_CORRECTED_BELIEF = {
    "accept_e1_belief": "",
    "correct_to_goalkeeper_like_team_1_context": "goalkeeper_like_team_1_context",
    "correct_to_goalkeeper_like_team_2_context": "goalkeeper_like_team_2_context",
    "correct_to_goalkeeper_like_unknown_team_context": "goalkeeper_like_unknown_team_context",
    "correct_to_outfield_player_like_not_goalkeeper": "outfield_player_like_not_goalkeeper",
    "correct_to_official_or_context_not_goalkeeper": "official_or_context_not_goalkeeper",
    "correct_to_bad_detection_or_not_person": "bad_detection_or_not_person",
    "correct_to_unknown_goalkeeper_context": "unknown_goalkeeper_context",
    "unsure_needs_later_review": "unknown_goalkeeper_context",
}

ALLOWED_REVIEW_CONFIDENCES = {"high", "medium", "low"}
E1B_FORBIDDEN_KEYS = set(FORBIDDEN_OUTPUT_KEYS) | {
    "track_id",
    "persistent_player_id",
    "official_exclusion",
    "official_exclusion_reason",
    "exclude_from_player_review",
    "excluded_from_player_review",
    "excluded_from_player_team_review",
    "goalkeeper_slot_id",
    "goalkeeper_identity_id",
    "expected_22_role_state",
    "expected_role_state",
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def reviewed_rows_from_payload(payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    rows = payload.get("rows", [])
    return rows if isinstance(rows, list) else []


def corrected_belief_for_decision(decision: str, e1_belief: str) -> str:
    if decision not in ALLOWED_HUMAN_GOALKEEPER_CONTEXT_DECISIONS:
        raise ValueError(f"Human review decision is not allowed: {decision}")
    if decision == "accept_e1_belief":
        return e1_belief
    return DECISION_TO_CORRECTED_BELIEF[decision]


def default_confidence_for_decision(decision: str) -> str:
    if decision == "accept_e1_belief":
        return "high"
    if decision == "unsure_needs_later_review":
        return "low"
    return "medium"


def reviewed_decision_row(
    candidate: dict[str, Any],
    human_review_decision: str,
    *,
    human_review_confidence: str | None = None,
    reviewer_name: str = "",
    notes: str = "",
    reviewed_at: str | None = None,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("step1e1_review_candidate_id", ""))
    visible_id = str(candidate.get("visible_person_base_id", ""))
    if not candidate_id:
        raise ValueError("Reviewed decision requires step1e1_review_candidate_id")
    if not visible_id:
        raise ValueError("Reviewed decision requires visible_person_base_id")
    original = str(candidate.get("e1_goalkeeper_context_belief", "unknown_goalkeeper_context"))
    corrected = corrected_belief_for_decision(human_review_decision, original)
    confidence = human_review_confidence or default_confidence_for_decision(human_review_decision)
    if confidence not in ALLOWED_REVIEW_CONFIDENCES:
        raise ValueError(f"Human review confidence is not allowed: {confidence}")
    return {
        "step1e1_review_candidate_id": candidate_id,
        "visible_person_base_id": visible_id,
        "frame_sequence": int(safe_float(candidate.get("frame_sequence"), -1)),
        "original_e1_goalkeeper_context_belief": original,
        "human_review_decision": human_review_decision,
        "human_corrected_goalkeeper_context_belief": corrected,
        "human_review_confidence": confidence,
        "reviewed_at": reviewed_at or utc_iso(),
        "reviewer_name": reviewer_name,
        "notes": notes,
        "human_confirmed": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "approve_any_goalkeeper_slot_use": False,
        "approve_any_identity_tracking": False,
        "approve_any_metric_use": False,
    }


def approval_flags_invalid(row: dict[str, Any]) -> list[str]:
    bad = []
    for key in ["approve_any_goalkeeper_slot_use", "approve_any_identity_tracking", "approve_any_metric_use"]:
        if row.get(key) is not False:
            bad.append(key)
    return bad


def validate_reviewed_decision_payload(
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any] | list[dict[str, Any]] | None,
    *,
    reviewed_decisions_loaded: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    known = {
        str(row.get("step1e1_review_candidate_id", "")): row
        for row in candidate_payload.get("rows", [])
        if row.get("step1e1_review_candidate_id")
    }
    errors: list[dict[str, Any]] = []
    usable_rows: list[dict[str, Any]] = []
    if not reviewed_decisions_loaded or reviewed_payload is None:
        return (
            {
                "reviewed_decisions_loaded": False,
                "reviewed_decisions_valid": False,
                "reviewed_decision_rows": 0,
                "validation_errors": [],
                "validation_warnings": [{"warning": "reviewed_decisions_absent"}],
                "usable_human_confirmed_decision_rows": 0,
            },
            [],
        )
    rows = reviewed_rows_from_payload(reviewed_payload)
    for index, row in enumerate(rows):
        candidate_id = str(row.get("step1e1_review_candidate_id", ""))
        candidate = known.get(candidate_id)
        if not candidate:
            errors.append({"row_index": index, "step1e1_review_candidate_id": candidate_id, "error": "unknown_review_candidate_id"})
            continue
        if not row.get("visible_person_base_id"):
            errors.append({"row_index": index, "step1e1_review_candidate_id": candidate_id, "error": "visible_person_base_id_missing"})
        if str(row.get("visible_person_base_id", "")) != str(candidate.get("visible_person_base_id", "")):
            errors.append({"row_index": index, "step1e1_review_candidate_id": candidate_id, "error": "visible_person_base_id_mismatch"})
        forbidden = sorted(key for key in E1B_FORBIDDEN_KEYS if key in row)
        if forbidden:
            errors.append({"row_index": index, "step1e1_review_candidate_id": candidate_id, "error": "forbidden_keys_present", "keys": forbidden})
        bad_approval = approval_flags_invalid(row)
        if bad_approval:
            errors.append({"row_index": index, "step1e1_review_candidate_id": candidate_id, "error": "forbidden_approval_flag_true_or_missing", "keys": bad_approval})
        if row.get("visual_only_warning") != VISUAL_ONLY_WARNING:
            errors.append({"row_index": index, "step1e1_review_candidate_id": candidate_id, "error": "visual_only_warning_missing"})
        if row.get("do_not_use_for_metrics") is not True:
            errors.append({"row_index": index, "step1e1_review_candidate_id": candidate_id, "error": "do_not_use_for_metrics_not_true"})
        if row.get("production_ready") is not False:
            errors.append({"row_index": index, "step1e1_review_candidate_id": candidate_id, "error": "production_ready_false_required"})
        decision = str(row.get("human_review_decision", ""))
        if decision not in ALLOWED_HUMAN_GOALKEEPER_CONTEXT_DECISIONS:
            errors.append({"row_index": index, "step1e1_review_candidate_id": candidate_id, "error": "human_review_decision_not_allowed", "decision": decision})
        confidence = str(row.get("human_review_confidence", ""))
        if confidence and confidence not in ALLOWED_REVIEW_CONFIDENCES:
            errors.append({"row_index": index, "step1e1_review_candidate_id": candidate_id, "error": "human_review_confidence_not_allowed", "confidence": confidence})
        corrected = str(row.get("human_corrected_goalkeeper_context_belief", ""))
        if decision in ALLOWED_HUMAN_GOALKEEPER_CONTEXT_DECISIONS:
            expected = corrected_belief_for_decision(decision, str(candidate.get("e1_goalkeeper_context_belief", "")))
            if corrected != expected:
                errors.append({"row_index": index, "step1e1_review_candidate_id": candidate_id, "error": "human_corrected_belief_does_not_match_decision", "expected": expected, "actual": corrected})
        if corrected not in ALLOWED_E1_GOALKEEPER_CONTEXT_BELIEFS:
            errors.append({"row_index": index, "step1e1_review_candidate_id": candidate_id, "error": "human_corrected_belief_not_allowed", "belief": corrected})
        if not boolish(row.get("human_confirmed")):
            continue
        usable_rows.append(
            {
                **row,
                "human_confirmed": True,
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
                "approve_any_goalkeeper_slot_use": False,
                "approve_any_identity_tracking": False,
                "approve_any_metric_use": False,
            }
        )
    counts = Counter(str(row.get("human_review_decision", "")) for row in usable_rows)
    return (
        {
            "reviewed_decisions_loaded": reviewed_decisions_loaded,
            "reviewed_decisions_valid": not errors,
            "reviewed_decision_rows": len(rows),
            "validation_errors": errors,
            "validation_warnings": [],
            "usable_human_confirmed_decision_rows": len(usable_rows),
            "human_review_decision_counts": dict(sorted(counts.items())),
        },
        usable_rows if not errors else [],
    )
