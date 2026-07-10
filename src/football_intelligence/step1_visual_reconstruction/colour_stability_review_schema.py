# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.schema import (
    FORBIDDEN_OUTPUT_KEYS,
    PRODUCTION_READY,
    VISUAL_ONLY_WARNING,
    safe_float,
)


ALLOWED_HUMAN_REVIEW_DECISIONS = {
    "accept_c2_stable_colour",
    "reject_to_unknown_ambiguous_colour",
    "reject_to_team_1_outfield_colour_like",
    "reject_to_team_2_outfield_colour_like",
    "reject_to_other_distinct_colour_like",
    "reject_to_non_outfield_context_colour",
    "crop_unusable",
    "bad_detection_or_not_person",
    "unsure_needs_later_review",
}

DECISION_TO_CORRECTED_BELIEF = {
    "accept_c2_stable_colour": "",
    "reject_to_unknown_ambiguous_colour": "unknown_ambiguous_colour",
    "reject_to_team_1_outfield_colour_like": "team_1_outfield_colour_like",
    "reject_to_team_2_outfield_colour_like": "team_2_outfield_colour_like",
    "reject_to_other_distinct_colour_like": "other_distinct_colour_like",
    "reject_to_non_outfield_context_colour": "non_outfield_context_colour",
    "crop_unusable": "crop_unusable",
    "bad_detection_or_not_person": "unknown_ambiguous_colour",
    "unsure_needs_later_review": "unknown_ambiguous_colour",
}

ALLOWED_HUMAN_CORRECTED_BELIEFS = {
    "team_1_outfield_colour_like",
    "team_2_outfield_colour_like",
    "other_distinct_colour_like",
    "non_outfield_context_colour",
    "unknown_ambiguous_colour",
    "crop_unusable",
}

ALLOWED_REVIEW_CONFIDENCES = {"high", "medium", "low"}
C2B_FORBIDDEN_KEYS = set(FORBIDDEN_OUTPUT_KEYS) | {"track_id", "persistent_player_id"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def reviewed_rows_from_payload(payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    rows = payload.get("rows", [])
    return rows if isinstance(rows, list) else []


def corrected_belief_for_decision(decision: str, c2_stable_colour_belief: str) -> str:
    if decision not in ALLOWED_HUMAN_REVIEW_DECISIONS:
        raise ValueError(f"Human review decision is not allowed: {decision}")
    if decision == "accept_c2_stable_colour":
        return c2_stable_colour_belief
    return DECISION_TO_CORRECTED_BELIEF[decision]


def default_confidence_for_decision(decision: str) -> str:
    if decision == "accept_c2_stable_colour":
        return "high"
    if decision in {"unsure_needs_later_review", "crop_unusable", "bad_detection_or_not_person"}:
        return "low"
    return "medium"


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate_reviewed_decision_payload(
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any] | list[dict[str, Any]] | None,
    *,
    reviewed_decisions_loaded: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    known = {str(row.get("c2b_review_candidate_id", "")): row for row in candidate_payload.get("rows", [])}
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
        candidate_id = str(row.get("c2b_review_candidate_id", ""))
        candidate = known.get(candidate_id)
        if not candidate:
            errors.append({"row_index": index, "c2b_review_candidate_id": candidate_id, "error": "unknown_review_candidate_id"})
            continue
        forbidden = sorted(key for key in C2B_FORBIDDEN_KEYS if key in row)
        if forbidden:
            errors.append({"row_index": index, "c2b_review_candidate_id": candidate_id, "error": "forbidden_keys_present", "keys": forbidden})
        if row.get("production_ready") is True:
            errors.append({"row_index": index, "c2b_review_candidate_id": candidate_id, "error": "production_ready_true_rejected"})
        decision = str(row.get("human_review_decision", ""))
        if decision not in ALLOWED_HUMAN_REVIEW_DECISIONS:
            errors.append({"row_index": index, "c2b_review_candidate_id": candidate_id, "error": "human_review_decision_not_allowed", "decision": decision})
        confidence = str(row.get("human_review_confidence", ""))
        if confidence and confidence not in ALLOWED_REVIEW_CONFIDENCES:
            errors.append({"row_index": index, "c2b_review_candidate_id": candidate_id, "error": "human_review_confidence_not_allowed", "confidence": confidence})
        corrected = str(row.get("human_corrected_colour_belief", ""))
        if decision in ALLOWED_HUMAN_REVIEW_DECISIONS:
            expected = corrected_belief_for_decision(decision, str(candidate.get("c2_stable_colour_belief", "")))
            if corrected != expected:
                errors.append({"row_index": index, "c2b_review_candidate_id": candidate_id, "error": "human_corrected_colour_belief_does_not_match_decision", "expected": expected, "actual": corrected})
        if corrected and corrected not in ALLOWED_HUMAN_CORRECTED_BELIEFS:
            errors.append({"row_index": index, "c2b_review_candidate_id": candidate_id, "error": "human_corrected_colour_belief_not_allowed", "belief": corrected})
        if not boolish(row.get("human_confirmed")):
            continue
        usable_rows.append({**row, "human_confirmed": True, "visual_only_warning": VISUAL_ONLY_WARNING, "do_not_use_for_metrics": True, "production_ready": PRODUCTION_READY})
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


def reviewed_decision_row(
    candidate: dict[str, Any],
    human_review_decision: str,
    *,
    human_review_confidence: str | None = None,
    reviewer_name: str = "",
    reviewer_notes: str = "",
    reviewed_at: str | None = None,
) -> dict[str, Any]:
    corrected = corrected_belief_for_decision(human_review_decision, str(candidate.get("c2_stable_colour_belief", "")))
    confidence = human_review_confidence or default_confidence_for_decision(human_review_decision)
    if confidence not in ALLOWED_REVIEW_CONFIDENCES:
        raise ValueError(f"Human review confidence is not allowed: {confidence}")
    return {
        "c2b_review_candidate_id": candidate.get("c2b_review_candidate_id", ""),
        "visible_person_base_id": candidate.get("visible_person_base_id", ""),
        "frame_sequence": int(safe_float(candidate.get("frame_sequence"), -1)),
        "c1c_seed_team_colour_belief": candidate.get("c1c_seed_team_colour_belief", ""),
        "c2_stable_colour_belief": candidate.get("c2_stable_colour_belief", ""),
        "human_review_decision": human_review_decision,
        "human_corrected_colour_belief": corrected,
        "human_review_confidence": confidence,
        "reviewer_notes": reviewer_notes,
        "reviewer_name": reviewer_name,
        "reviewed_at": reviewed_at or utc_iso(),
        "human_confirmed": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }
