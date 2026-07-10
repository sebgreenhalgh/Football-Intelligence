# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH,
    STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH,
    STEP1C1D_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP1C1D_REVIEW_SESSION_STATE_PATH,
    read_json,
    write_json,
)
from football_intelligence.step1_visual_reconstruction.manual_colour_seed_schema import (
    ALLOWED_MANUAL_COLOUR_LABELS,
    reviewed_rows_from_payload,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    safe_float,
)


CATEGORY_ORDER = [
    "likely_team_1_colour_seed_prefill",
    "likely_team_2_colour_seed_prefill",
    "dark_context_seed_review",
    "negative_context_seed_review",
    "other_distinct_colour_seed_review",
    "ambiguous_colour_seed_review",
    "crop_quality_failure_review",
]

NEGATIVE_MANUAL_LABELS = {
    "ambiguous_outfield_colour",
    "non_outfield_context_colour",
    "dark_context_colour",
    "other_distinct_colour",
    "crop_unusable",
    "not_a_person_or_bad_detection",
    "unsure",
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_seed_candidates(path: Path = STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH) -> list[dict[str, Any]]:
    payload = read_json(path)
    return sorted(
        payload.get("rows", []),
        key=lambda row: (
            CATEGORY_ORDER.index(str(row.get("seed_candidate_category", "")))
            if str(row.get("seed_candidate_category", "")) in CATEGORY_ORDER
            else 99,
            int(safe_float(row.get("review_priority"), 99)),
            int(safe_float(row.get("frame_sequence"), 999999)),
            str(row.get("seed_candidate_id", "")),
        ),
    )


def load_reviewed_labels(path: Path = STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = read_json(path)
    rows = reviewed_rows_from_payload(payload)
    return {str(row.get("seed_candidate_id", "")): row for row in rows if row.get("seed_candidate_id")}


def candidate_with_review(candidate: dict[str, Any], reviewed_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reviewed = reviewed_by_id.get(str(candidate.get("seed_candidate_id", "")), {})
    out = dict(candidate)
    out["saved_manual_colour_label"] = reviewed.get("manual_colour_label", "")
    out["saved_manual_label_confidence"] = reviewed.get("manual_label_confidence", "")
    out["saved_reviewer_notes"] = reviewed.get("reviewer_notes", "")
    out["saved_reviewer_name"] = reviewed.get("reviewer_name", "")
    out["saved_reviewed_at"] = reviewed.get("reviewed_at", "")
    out["saved_human_confirmed"] = bool(reviewed.get("human_confirmed") is True)
    return out


def merged_review_state(
    candidates: list[dict[str, Any]] | None = None,
    reviewed_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidates = candidates if candidates is not None else load_seed_candidates()
    reviewed_by_id = reviewed_by_id if reviewed_by_id is not None else load_reviewed_labels()
    return [candidate_with_review(candidate, reviewed_by_id) for candidate in candidates]


def is_reviewed(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("saved_human_confirmed") and candidate.get("saved_manual_colour_label") in ALLOWED_MANUAL_COLOUR_LABELS)


def filter_candidates(
    candidates: list[dict[str, Any]],
    *,
    category: str | None = None,
    reviewed_state: str | None = None,
    frame_sequence: int | None = None,
    manual_label: str | None = None,
) -> list[dict[str, Any]]:
    rows = candidates
    if category:
        rows = [row for row in rows if row.get("seed_candidate_category") == category]
    if reviewed_state == "reviewed":
        rows = [row for row in rows if is_reviewed(row)]
    elif reviewed_state == "unreviewed":
        rows = [row for row in rows if not is_reviewed(row)]
    if frame_sequence is not None:
        rows = [row for row in rows if int(safe_float(row.get("frame_sequence"), -1)) == frame_sequence]
    if manual_label:
        rows = [row for row in rows if row.get("saved_manual_colour_label") == manual_label]
    return rows


def next_unreviewed_index(candidates: list[dict[str, Any]], start_index: int = 0) -> int:
    if not candidates:
        return -1
    count = len(candidates)
    for offset in range(count):
        index = (start_index + offset) % count
        if not is_reviewed(candidates[index]):
            return index
    return -1


def progress_counts(reviewed_by_id: dict[str, dict[str, Any]], total_seed_candidates: int) -> dict[str, Any]:
    labels = Counter(
        str(row.get("manual_colour_label", ""))
        for row in reviewed_by_id.values()
        if row.get("human_confirmed") is True and row.get("manual_colour_label") in ALLOWED_MANUAL_COLOUR_LABELS
    )
    reviewed_rows = sum(labels.values())
    negative_count = sum(labels.get(label, 0) for label in NEGATIVE_MANUAL_LABELS)
    minimum = labels.get("team_1_outfield_colour_seed", 0) >= 8 and labels.get("team_2_outfield_colour_seed", 0) >= 8 and negative_count >= 4
    return {
        "total_seed_candidates": total_seed_candidates,
        "reviewed_rows": reviewed_rows,
        "unreviewed_rows": max(0, total_seed_candidates - reviewed_rows),
        "human_confirmed_team_1_seed_count": labels.get("team_1_outfield_colour_seed", 0),
        "human_confirmed_team_2_seed_count": labels.get("team_2_outfield_colour_seed", 0),
        "human_confirmed_negative_seed_count": negative_count,
        "manual_label_counts": dict(sorted(labels.items())),
        "minimum_seed_counts_satisfied": minimum,
        "c2_still_requires_c1c_seeded_validation": True,
        "production_ready": PRODUCTION_READY,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
    }


def progress_summary_payload(
    candidates: list[dict[str, Any]] | None = None,
    reviewed_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidates = candidates if candidates is not None else load_seed_candidates()
    reviewed_by_id = reviewed_by_id if reviewed_by_id is not None else load_reviewed_labels()
    return {
        "artifact": "step1c1d_review_progress_summary",
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
        **progress_counts(reviewed_by_id, len(candidates)),
    }


def session_state_payload(current_index: int = 0, reviewer_name: str = "", filters: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "artifact": "step1c1d_review_session_state",
        "updated_at": utc_iso(),
        "current_index": current_index,
        "reviewer_name": reviewer_name,
        "filters": filters or {},
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def write_progress_summary() -> dict[str, Any]:
    payload = progress_summary_payload()
    write_json(STEP1C1D_REVIEW_PROGRESS_SUMMARY_PATH, payload)
    return payload


def write_session_state(current_index: int = 0, reviewer_name: str = "", filters: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = session_state_payload(current_index=current_index, reviewer_name=reviewer_name, filters=filters)
    write_json(STEP1C1D_REVIEW_SESSION_STATE_PATH, payload)
    return payload
