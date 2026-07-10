# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.io import (
    STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH,
    STEP1E1_GOLD8_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH,
    read_json,
)
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_schema import reviewed_rows_from_payload
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    safe_float,
)


REQUIRED_REVIEW_TAGS = [
    "gold8_goalkeeper_proxy_match",
    "goalkeeper_like_belief",
    "unknown_goalkeeper_context_with_non_outfield_colour_hint",
    "bad_detection_with_goalkeeper_like_hint",
    "contradictory_official_context_goalkeeper_hints",
    "balanced_sample_outfield_player_like_not_goalkeeper",
    "balanced_sample_official_or_context_not_goalkeeper",
]

TAG_PRIORITIES = {
    "gold8_goalkeeper_proxy_match": 0,
    "goalkeeper_like_belief": 5,
    "unknown_goalkeeper_context_with_non_outfield_colour_hint": 10,
    "bad_detection_with_goalkeeper_like_hint": 15,
    "contradictory_official_context_goalkeeper_hints": 20,
    "balanced_sample_outfield_player_like_not_goalkeeper": 70,
    "balanced_sample_official_or_context_not_goalkeeper": 75,
    "review_required": 90,
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_reviewed_decisions() -> dict[str, dict[str, Any]]:
    if not STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH.exists():
        return {}
    rows = reviewed_rows_from_payload(read_json(STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH))
    return {str(row.get("step1e1_review_candidate_id", "")): row for row in rows if row.get("step1e1_review_candidate_id")}


def review_bucket(row: dict[str, Any]) -> tuple[int, str]:
    tags = set(row.get("review_reason_tags", []))
    best_priority = 99
    best_tag = "remaining_review_required"
    for tag, priority in TAG_PRIORITIES.items():
        if tag in tags and priority < best_priority:
            best_priority = priority
            best_tag = tag
    return best_priority, best_tag


def enrich_candidate(
    row: dict[str, Any],
    *,
    reviewed_by_id: dict[str, dict[str, Any]] | None = None,
    assets_by_id: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    reviewed_by_id = reviewed_by_id or {}
    assets_by_id = assets_by_id or {}
    review_id = str(row.get("step1e1_review_candidate_id", ""))
    review = reviewed_by_id.get(review_id, {})
    priority, bucket = review_bucket(row)
    return {
        **row,
        "e1b_review_bucket_priority": priority,
        "e1b_review_bucket": bucket,
        "ui_assets": assets_by_id.get(review_id, {}),
        "saved_human_review_decision": review.get("human_review_decision", ""),
        "saved_human_corrected_goalkeeper_context_belief": review.get("human_corrected_goalkeeper_context_belief", ""),
        "saved_human_review_confidence": review.get("human_review_confidence", ""),
        "saved_notes": review.get("notes", ""),
        "saved_reviewer_name": review.get("reviewer_name", ""),
        "saved_reviewed_at": review.get("reviewed_at", ""),
        "ui_is_reviewed": bool(review.get("human_confirmed") is True and review.get("human_review_decision")),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def ordered_review_candidates(
    candidate_payload: dict[str, Any] | None = None,
    *,
    reviewed_by_id: dict[str, dict[str, Any]] | None = None,
    assets_by_id: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    candidate_payload = candidate_payload or read_json(STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH)
    rows = [
        enrich_candidate(row, reviewed_by_id=reviewed_by_id, assets_by_id=assets_by_id)
        for row in candidate_payload.get("rows", [])
    ]
    rows.sort(
        key=lambda row: (
            int(row.get("e1b_review_bucket_priority", 99)),
            int(row.get("review_priority", 999)),
            int(safe_float(row.get("frame_sequence"), -1)),
            str(row.get("step1e1_review_candidate_id", "")),
        )
    )
    return rows


def reason_tag_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(tag for row in rows for tag in row.get("review_reason_tags", []))
    return {tag: counts.get(tag, 0) for tag in REQUIRED_REVIEW_TAGS}


def required_bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = reason_tag_counts(rows)
    counts["remaining_review_required"] = sum(1 for row in rows if "review_required" in row.get("review_reason_tags", []))
    return counts


def review_state_payload(
    candidate_payload: dict[str, Any] | None = None,
    *,
    assets_by_id: dict[str, dict[str, str]] | None = None,
    reviewed_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reviewed_by_id = reviewed_by_id if reviewed_by_id is not None else load_reviewed_decisions()
    state_rows = ordered_review_candidates(candidate_payload, reviewed_by_id=reviewed_by_id, assets_by_id=assets_by_id)
    first_unreviewed = next((index for index, row in enumerate(state_rows) if not row["ui_is_reviewed"]), 0)
    belief_counts = Counter(str(row.get("e1_goalkeeper_context_belief", "")) for row in state_rows)
    return {
        "artifact": "step1e1b_goalkeeper_context_review_state",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_slot_assignment_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "first_unreviewed_index": first_unreviewed,
        "total_review_candidates": len(state_rows),
        "required_bucket_counts": required_bucket_counts(state_rows),
        "e1_goalkeeper_context_belief_counts": dict(sorted(belief_counts.items())),
        "e1_eval_summary": read_json(STEP1E1_GOLD8_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH) if candidate_payload is None else {},
        "rows": state_rows,
    }
