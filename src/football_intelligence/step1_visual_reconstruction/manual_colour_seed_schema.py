# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH,
    STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH,
    STEP1C1C_SEED_VALIDATION_SUMMARY_PATH,
    read_json,
    write_json,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    FORBIDDEN_OUTPUT_KEYS,
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
)


ALLOWED_MANUAL_COLOUR_LABELS = {
    "team_1_outfield_colour_seed",
    "team_2_outfield_colour_seed",
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


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def seed_template_row(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed_candidate_id": candidate.get("seed_candidate_id", ""),
        "visible_person_base_id": candidate.get("visible_person_base_id", ""),
        "frame_sequence": candidate.get("frame_sequence", -1),
        "crop_profile_name": candidate.get("crop_profile_name", ""),
        "prefill_suggested_manual_label": candidate.get("prefill_suggested_manual_label", ""),
        "manual_colour_label": "",
        "manual_label_confidence": "",
        "reviewer_notes": "",
        "reviewer_name": "",
        "reviewed_at": "",
        "human_confirmed": False,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def manual_seed_label_template_payload(candidate_payload: dict[str, Any]) -> dict[str, Any]:
    rows = [seed_template_row(row) for row in candidate_payload.get("rows", [])]
    return {
        "artifact": "step1c1c_manual_colour_seed_label_template",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "allowed_manual_colour_labels": sorted(ALLOWED_MANUAL_COLOUR_LABELS),
        "prefill_only": True,
        "human_confirmation_required": True,
        "rows": rows,
        "summary": {
            "seed_candidate_rows": len(candidate_payload.get("rows", [])),
            "manual_template_rows": len(rows),
        },
    }


def reviewed_rows_from_payload(payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    rows = payload.get("rows", [])
    return rows if isinstance(rows, list) else []


def row_forbidden_keys(row: dict[str, Any]) -> list[str]:
    return sorted(key for key in FORBIDDEN_OUTPUT_KEYS if key in row)


def validate_reviewed_colour_seed_payload(
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any] | list[dict[str, Any]] | None,
    *,
    reviewed_seed_labels_loaded: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    known_candidates = {str(row.get("seed_candidate_id", "")): row for row in candidate_payload.get("rows", [])}
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    usable_rows: list[dict[str, Any]] = []
    if not reviewed_seed_labels_loaded or reviewed_payload is None:
        return (
            {
                "artifact": "step1c1c_seed_validation_summary",
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
                "reviewed_seed_labels_loaded": False,
                "reviewed_seed_labels_valid": False,
                "validation_errors": [],
                "validation_warnings": [
                    {
                        "warning": "reviewed_seed_labels_absent",
                        "path": str(STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH.resolve()),
                    }
                ],
                "usable_human_confirmed_seed_rows": 0,
                "human_confirmed_seed_counts_by_manual_label": {},
                "human_confirmed_team_1_seed_count": 0,
                "human_confirmed_team_2_seed_count": 0,
                "human_confirmed_negative_seed_count": 0,
                "human_seed_set_id": "",
                "manual_review_required": True,
                "summary_reason": "No reviewed colour seed label file exists yet.",
            },
            usable_rows,
        )
    reviewed_rows = reviewed_rows_from_payload(reviewed_payload)
    for index, row in enumerate(reviewed_rows):
        seed_id = str(row.get("seed_candidate_id", ""))
        if seed_id not in known_candidates:
            errors.append({"row_index": index, "seed_candidate_id": seed_id, "error": "unknown_seed_candidate_id"})
            continue
        forbidden = row_forbidden_keys(row)
        if forbidden:
            errors.append({"row_index": index, "seed_candidate_id": seed_id, "error": "forbidden_keys_present", "keys": forbidden})
        if row.get("production_ready") is True:
            errors.append({"row_index": index, "seed_candidate_id": seed_id, "error": "production_ready_true_rejected"})
        if row.get("visual_only_warning") not in {None, VISUAL_ONLY_WARNING}:
            errors.append({"row_index": index, "seed_candidate_id": seed_id, "error": "visual_only_warning_invalid"})
        label = str(row.get("manual_colour_label", "")).strip()
        confirmed = boolish(row.get("human_confirmed"))
        if label and label not in ALLOWED_MANUAL_COLOUR_LABELS:
            errors.append({"row_index": index, "seed_candidate_id": seed_id, "error": "manual_colour_label_not_allowed", "label": label})
        if confirmed and not label:
            errors.append({"row_index": index, "seed_candidate_id": seed_id, "error": "confirmed_seed_missing_manual_colour_label"})
        if not confirmed:
            if label:
                warnings.append({"row_index": index, "seed_candidate_id": seed_id, "warning": "manual_label_ignored_without_human_confirmed_true"})
            continue
        if label in ALLOWED_MANUAL_COLOUR_LABELS:
            candidate = known_candidates[seed_id]
            usable = {
                **row,
                "manual_colour_label": label,
                "human_confirmed": True,
                "visible_person_base_id": candidate.get("visible_person_base_id", row.get("visible_person_base_id", "")),
                "seed_candidate_category": candidate.get("seed_candidate_category", ""),
                "prefill_suggested_manual_label": candidate.get("prefill_suggested_manual_label", row.get("prefill_suggested_manual_label", "")),
                "gold_visible_person_type_prefill": candidate.get("gold_visible_person_type_prefill", ""),
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
            }
            usable_rows.append(usable)
    counts = Counter(row.get("manual_colour_label", "") for row in usable_rows)
    negative_count = sum(
        counts.get(label, 0)
        for label in [
            "ambiguous_outfield_colour",
            "non_outfield_context_colour",
            "dark_context_colour",
            "other_distinct_colour",
            "crop_unusable",
            "not_a_person_or_bad_detection",
            "unsure",
        ]
    )
    valid = reviewed_seed_labels_loaded and not errors
    return (
        {
            "artifact": "step1c1c_seed_validation_summary",
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
            "reviewed_seed_labels_loaded": reviewed_seed_labels_loaded,
            "reviewed_seed_labels_valid": valid,
            "reviewed_seed_label_rows": len(reviewed_rows),
            "validation_errors": errors,
            "validation_warnings": warnings,
            "usable_human_confirmed_seed_rows": len(usable_rows),
            "human_confirmed_seed_counts_by_manual_label": dict(sorted(counts.items())),
            "human_confirmed_team_1_seed_count": counts.get("team_1_outfield_colour_seed", 0),
            "human_confirmed_team_2_seed_count": counts.get("team_2_outfield_colour_seed", 0),
            "human_confirmed_negative_seed_count": negative_count,
            "human_seed_set_id": "step1c1c_human_seed_set_v1" if usable_rows else "",
            "manual_review_required": not valid or not usable_rows,
            "summary_reason": "Reviewed seed labels passed schema validation." if valid else "Reviewed seed labels failed validation.",
        },
        usable_rows if valid else [],
    )


def build_and_write_seed_validation_summary(
    *,
    reviewed_path: Path = STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_payload = read_json(STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH)
    reviewed_exists = reviewed_path.exists()
    reviewed_payload = read_json(reviewed_path) if reviewed_exists else None
    summary, usable_rows = validate_reviewed_colour_seed_payload(
        candidate_payload,
        reviewed_payload,
        reviewed_seed_labels_loaded=reviewed_exists,
    )
    write_json(STEP1C1C_SEED_VALIDATION_SUMMARY_PATH, summary)
    return summary, usable_rows
