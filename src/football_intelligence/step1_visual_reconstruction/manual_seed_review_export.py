# ruff: noqa: E501

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH,
    STEP1C1D_REVIEW_PROGRESS_SUMMARY_PATH,
    read_json,
    write_json,
)
from football_intelligence.step1_visual_reconstruction.manual_colour_seed_schema import (
    ALLOWED_MANUAL_COLOUR_LABELS,
    reviewed_rows_from_payload,
)
from football_intelligence.step1_visual_reconstruction.manual_seed_review_state import (
    load_reviewed_labels,
    load_seed_candidates,
    progress_summary_payload,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
)


DEFAULT_CONFIDENCE_BY_LABEL = {
    "team_1_outfield_colour_seed": "high",
    "team_2_outfield_colour_seed": "high",
    "ambiguous_outfield_colour": "medium",
    "non_outfield_context_colour": "medium",
    "dark_context_colour": "medium",
    "other_distinct_colour": "medium",
    "crop_unusable": "low",
    "not_a_person_or_bad_detection": "low",
    "unsure": "low",
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def candidate_index(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("seed_candidate_id", "")): row for row in candidates}


def reviewed_label_row(
    candidate: dict[str, Any],
    manual_colour_label: str,
    *,
    manual_label_confidence: str | None = None,
    reviewer_name: str = "",
    reviewer_notes: str = "",
    reviewed_at: str | None = None,
) -> dict[str, Any]:
    if manual_colour_label not in ALLOWED_MANUAL_COLOUR_LABELS:
        raise ValueError(f"Manual colour label is not allowed: {manual_colour_label}")
    confidence = manual_label_confidence or DEFAULT_CONFIDENCE_BY_LABEL.get(manual_colour_label, "medium")
    return {
        "seed_candidate_id": candidate.get("seed_candidate_id", ""),
        "visible_person_base_id": candidate.get("visible_person_base_id", ""),
        "frame_sequence": candidate.get("frame_sequence", -1),
        "crop_profile_name": candidate.get("crop_profile_name", ""),
        "prefill_suggested_manual_label": candidate.get("prefill_suggested_manual_label", ""),
        "manual_colour_label": manual_colour_label,
        "manual_label_confidence": confidence,
        "reviewer_notes": reviewer_notes,
        "reviewer_name": reviewer_name,
        "reviewed_at": reviewed_at or utc_iso(),
        "human_confirmed": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def reviewed_payload(
    rows_by_id: dict[str, dict[str, Any]],
    *,
    reviewer_name: str = "",
    existing_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created_at = (existing_payload or {}).get("created_at", utc_iso())
    rows = sorted(rows_by_id.values(), key=lambda row: (int(row.get("frame_sequence", -1)), str(row.get("seed_candidate_id", ""))))
    summary = progress_summary_payload(load_seed_candidates(), rows_by_id)
    return {
        "artifact": "step1c1c_reviewed_colour_seed_labels",
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
            "reviewed_rows": summary["reviewed_rows"],
            "human_confirmed_team_1_seed_count": summary["human_confirmed_team_1_seed_count"],
            "human_confirmed_team_2_seed_count": summary["human_confirmed_team_2_seed_count"],
            "human_confirmed_negative_seed_count": summary["human_confirmed_negative_seed_count"],
            "minimum_seed_counts_satisfied": summary["minimum_seed_counts_satisfied"],
        },
    }


def save_reviewed_payload(
    rows_by_id: dict[str, dict[str, Any]],
    *,
    reviewer_name: str = "",
    output_path: Path = STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH,
) -> dict[str, Any]:
    existing_payload = read_json(output_path) if output_path.exists() else None
    payload = reviewed_payload(rows_by_id, reviewer_name=reviewer_name, existing_payload=existing_payload)
    write_json(output_path, payload)
    progress = progress_summary_payload(load_seed_candidates(), rows_by_id)
    if output_path.resolve() == STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH.resolve():
        write_json(STEP1C1D_REVIEW_PROGRESS_SUMMARY_PATH, progress)
    return payload


def save_single_review(
    seed_candidate_id: str,
    manual_colour_label: str,
    *,
    manual_label_confidence: str | None = None,
    reviewer_name: str = "",
    reviewer_notes: str = "",
    output_path: Path = STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH,
) -> dict[str, Any]:
    candidates = load_seed_candidates()
    candidates_by_id = candidate_index(candidates)
    if seed_candidate_id not in candidates_by_id:
        raise KeyError(f"Unknown seed candidate id: {seed_candidate_id}")
    reviewed_by_id = load_reviewed_labels(output_path)
    reviewed_by_id[seed_candidate_id] = reviewed_label_row(
        candidates_by_id[seed_candidate_id],
        manual_colour_label,
        manual_label_confidence=manual_label_confidence,
        reviewer_name=reviewer_name,
        reviewer_notes=reviewer_notes,
    )
    return save_reviewed_payload(reviewed_by_id, reviewer_name=reviewer_name, output_path=output_path)


def export_existing_reviewed_seed_labels() -> dict[str, Any]:
    reviewed = load_reviewed_labels()
    payload = save_reviewed_payload(reviewed)
    return payload


def rows_from_reviewed_payload(path: Path = STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return reviewed_rows_from_payload(read_json(path))
