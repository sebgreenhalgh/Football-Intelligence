# ruff: noqa: E501

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.step1_visual_reconstruction.colour_stability_review_eval import (
    progress_summary_payload,
    review_decision_summary_payload,
    write_review_progress_and_decision_summaries,
)
from football_intelligence.step1_visual_reconstruction.colour_stability_review_schema import (
    reviewed_decision_row,
    reviewed_rows_from_payload,
)
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH,
    STEP1C2B_REVIEWED_DECISIONS_PATH,
    read_json,
    write_json,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
)


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_review_candidates(path: Path = STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH) -> list[dict[str, Any]]:
    return read_json(path).get("rows", [])


def candidate_index(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("c2b_review_candidate_id", "")): row for row in candidates}


def load_reviewed_decisions(path: Path = STEP1C2B_REVIEWED_DECISIONS_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = reviewed_rows_from_payload(read_json(path))
    return {str(row.get("c2b_review_candidate_id", "")): row for row in rows if row.get("c2b_review_candidate_id")}


def reviewed_decision_payload(
    rows_by_id: dict[str, dict[str, Any]],
    *,
    reviewer_name: str = "",
    existing_payload: dict[str, Any] | None = None,
    candidate_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created_at = (existing_payload or {}).get("created_at", utc_iso())
    candidate_payload = candidate_payload or read_json(STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH)
    progress = progress_summary_payload(candidate_payload, rows_by_id)
    decision = review_decision_summary_payload(candidate_payload, rows_by_id)
    rows = sorted(rows_by_id.values(), key=lambda row: (int(row.get("frame_sequence", -1)), str(row.get("c2b_review_candidate_id", ""))))
    return {
        "artifact": "step1c2b_reviewed_colour_stability_decisions",
        "created_at": created_at,
        "updated_at": utc_iso(),
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
        "reviewer_name": reviewer_name,
        "rows": rows,
        "summary": {
            "reviewed_candidates": progress["reviewed_candidates"],
            "accepted_c2_count": progress["accepted_c2_count"],
            "rejected_corrected_count": progress["rejected_corrected_count"],
            "unsure_count": progress["unsure_count"],
            "c2b_approve_c2_for_next_stage_candidate": decision["c2b_approve_c2_for_next_stage_candidate"],
        },
    }


def save_reviewed_decision_payload(
    rows_by_id: dict[str, dict[str, Any]],
    *,
    reviewer_name: str = "",
    output_path: Path = STEP1C2B_REVIEWED_DECISIONS_PATH,
    candidate_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing_payload = read_json(output_path) if output_path.exists() else None
    payload = reviewed_decision_payload(
        rows_by_id,
        reviewer_name=reviewer_name,
        existing_payload=existing_payload,
        candidate_payload=candidate_payload,
    )
    write_json(output_path, payload)
    if output_path.resolve() == STEP1C2B_REVIEWED_DECISIONS_PATH.resolve():
        write_review_progress_and_decision_summaries()
    return payload


def save_single_review_decision(
    c2b_review_candidate_id: str,
    human_review_decision: str,
    *,
    human_review_confidence: str | None = None,
    reviewer_name: str = "",
    reviewer_notes: str = "",
    output_path: Path = STEP1C2B_REVIEWED_DECISIONS_PATH,
) -> dict[str, Any]:
    candidates = load_review_candidates()
    candidates_by_id = candidate_index(candidates)
    if c2b_review_candidate_id not in candidates_by_id:
        raise KeyError(f"Unknown C2b review candidate id: {c2b_review_candidate_id}")
    reviewed_by_id = load_reviewed_decisions(output_path)
    reviewed_by_id[c2b_review_candidate_id] = reviewed_decision_row(
        candidates_by_id[c2b_review_candidate_id],
        human_review_decision,
        human_review_confidence=human_review_confidence,
        reviewer_name=reviewer_name,
        reviewer_notes=reviewer_notes,
    )
    return save_reviewed_decision_payload(reviewed_by_id, reviewer_name=reviewer_name, output_path=output_path)


def export_existing_reviewed_decisions() -> dict[str, Any]:
    reviewed_by_id = load_reviewed_decisions()
    return save_reviewed_decision_payload(reviewed_by_id)
