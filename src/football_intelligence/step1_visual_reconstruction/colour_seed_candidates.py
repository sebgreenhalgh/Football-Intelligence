# ruff: noqa: E501

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH,
    STEP1C1_COLOUR_FEATURE_ROWS_PATH,
    STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH,
    STEP1C1B_BEST_SANDBOX_BELIEF_ROWS_PATH,
    STEP1C1B_CROP_AUDIT_ROWS_PATH,
    STEP1C1B_GOLD8_CLUSTER_CONFUSION_ROWS_PATH,
    STEP1C1B_PROFILE_EVAL_SUMMARY_PATH,
    STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH,
    STEP1C1C_COLOUR_SEED_CANDIDATE_SUMMARY_PATH,
    STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_CSV_PATH,
    STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_JSON_PATH,
    read_json,
    write_json,
)
from football_intelligence.step1_visual_reconstruction.manual_colour_seed_schema import manual_seed_label_template_payload
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    safe_float,
)


SEED_CANDIDATE_CATEGORIES = [
    "likely_team_1_colour_seed_prefill",
    "likely_team_2_colour_seed_prefill",
    "ambiguous_colour_seed_review",
    "negative_context_seed_review",
    "dark_context_seed_review",
    "other_distinct_colour_seed_review",
    "crop_quality_failure_review",
]

CATEGORY_LIMITS = {
    "likely_team_1_colour_seed_prefill": 36,
    "likely_team_2_colour_seed_prefill": 36,
    "ambiguous_colour_seed_review": 30,
    "negative_context_seed_review": 20,
    "dark_context_seed_review": 20,
    "other_distinct_colour_seed_review": 20,
    "crop_quality_failure_review": 28,
}

CATEGORY_PREFILL_LABELS = {
    "likely_team_1_colour_seed_prefill": "team_1_outfield_colour_seed",
    "likely_team_2_colour_seed_prefill": "team_2_outfield_colour_seed",
    "ambiguous_colour_seed_review": "ambiguous_outfield_colour",
    "negative_context_seed_review": "non_outfield_context_colour",
    "dark_context_seed_review": "dark_context_colour",
    "other_distinct_colour_seed_review": "other_distinct_colour",
    "crop_quality_failure_review": "crop_unusable",
}

CONTEXT_CANDIDATE_TYPES = {
    "official_candidate_source",
    "referee_candidate_source",
    "staff_context_candidate_source",
    "unknown_candidate_source",
    "off_pitch_person_candidate",
    "unknown_person_candidate",
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def index_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("visible_person_base_id", "")): row for row in payload.get("rows", [])}


def best_profile_gold_prefill_rows(c1b_eval_summary: dict[str, Any], c1b_confusion_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    best_profile = str(c1b_eval_summary.get("c1b_best_profile_name", ""))
    best_strategy = str(c1b_eval_summary.get("c1b_best_prototype_strategy", ""))
    rows = {}
    for row in c1b_confusion_payload.get("rows", []):
        if row.get("profile_name") != best_profile or row.get("prototype_strategy") != best_strategy:
            continue
        visible_type = str(row.get("visible_person_type_gold", ""))
        if visible_type in {"team_1_player", "team_2_player"}:
            rows[str(row.get("visible_person_base_id", ""))] = row
    return rows


def is_context_like(row: dict[str, Any]) -> bool:
    return str(row.get("candidate_type", "")) in CONTEXT_CANDIDATE_TYPES or str(row.get("roi_status", "")) == "outside_playing_roi"


def candidate_category(
    base_row: dict[str, Any],
    c1_row: dict[str, Any],
    c1b_row: dict[str, Any],
    audit_row: dict[str, Any],
    gold_prefill_row: dict[str, Any] | None,
) -> tuple[str, str, int]:
    c1_belief = str(c1_row.get("team_colour_belief", ""))
    c1b_belief = str(c1b_row.get("team_colour_belief", ""))
    crop_quality = str(c1b_row.get("crop_quality") or c1_row.get("crop_quality", ""))
    audit_flags = set(audit_row.get("audit_issue_flags", []))
    if gold_prefill_row and gold_prefill_row.get("visible_person_type_gold") == "team_1_player" and crop_quality in {"high", "medium"} and not base_row.get("review_required") and not base_row.get("source_disagreement_review_required"):
        return "likely_team_1_colour_seed_prefill", "Gold-8 visible type suggests team_1_player; reviewer must confirm manually.", 10
    if gold_prefill_row and gold_prefill_row.get("visible_person_type_gold") == "team_2_player" and crop_quality in {"high", "medium"} and not base_row.get("review_required") and not base_row.get("source_disagreement_review_required"):
        return "likely_team_2_colour_seed_prefill", "Gold-8 visible type suggests team_2_player; reviewer must confirm manually.", 10
    if is_context_like(base_row):
        return "negative_context_seed_review", "Context/off-ROI visual row included as a negative/context seed review candidate.", 45
    if c1_belief == "dark_context_colour_like" or c1b_belief == "dark_context_colour_like":
        return "dark_context_seed_review", "Dark colour evidence included only as colour/context review, not official classification.", 35
    if c1_belief == "other_distinct_colour_like" or c1b_belief == "other_distinct_colour_like":
        return "other_distinct_colour_seed_review", "Other distinct colour evidence included for reviewer labelling, not goalkeeper classification.", 38
    if crop_quality in {"low", "unusable"} or {"small_torso_crop", "background_contaminated_crop", "mostly_green_background"} & audit_flags:
        return "crop_quality_failure_review", "Crop has low quality or contamination flags and needs manual crop-quality review.", 55
    if c1_belief in {"unknown_ambiguous_colour", "crop_unusable"} or c1b_belief in {"unknown_ambiguous_colour", "crop_unusable"}:
        return "ambiguous_colour_seed_review", "Current visual colour belief remains unknown or ambiguous.", 30
    return "ambiguous_colour_seed_review", "Visually useful row kept as ambiguous/manual decision candidate.", 60


def candidate_score(row: dict[str, Any]) -> tuple[Any, ...]:
    quality_rank = {"high": 0, "medium": 1, "low": 2, "unusable": 3}
    return (
        int(row.get("review_priority", 99)),
        quality_rank.get(str(row.get("crop_quality", "")), 9),
        -safe_float(row.get("current_c1_confidence")),
        int(safe_float(row.get("frame_sequence"), 999999)),
        str(row.get("visible_person_base_id", "")),
    )


def balanced_select(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected = []
    per_frame: Counter[int] = Counter()
    for row in sorted(rows, key=candidate_score):
        seq = int(safe_float(row.get("frame_sequence"), -1))
        if per_frame[seq] >= 6 and len(selected) < limit // 2:
            continue
        selected.append(row)
        per_frame[seq] += 1
        if len(selected) >= limit:
            break
    return selected


def seed_candidate_id(category: str, index: int, base_id: str) -> str:
    return f"step1c1c_seed_{category}_{index:03d}_{str(base_id)[-10:]}"


def seed_candidate_row(
    base_row: dict[str, Any],
    feature_row: dict[str, Any],
    c1_row: dict[str, Any],
    c1b_row: dict[str, Any],
    audit_row: dict[str, Any],
    gold_prefill_row: dict[str, Any] | None,
    category: str,
    reason: str,
    priority: int,
) -> dict[str, Any]:
    crop_profile = str(c1b_row.get("profile_name") or "c1_current")
    return {
        "seed_candidate_id": "",
        "frame_id": base_row.get("frame_id", ""),
        "frame_sequence": int(safe_float(base_row.get("frame_sequence"), -1)),
        "timestamp_seconds": safe_float(base_row.get("timestamp_seconds")),
        "visible_person_base_id": base_row.get("visible_person_base_id", ""),
        "detection_id": base_row.get("detection_id", ""),
        "source_detection_id": base_row.get("source_detection_id", ""),
        "bbox": base_row.get("bbox", {}),
        "footpoint": base_row.get("footpoint", {}),
        "state": base_row.get("state", ""),
        "candidate_type": base_row.get("candidate_type", ""),
        "original_role_source": base_row.get("original_role_source", ""),
        "roi_status": base_row.get("roi_status", ""),
        "source_role_labels": base_row.get("source_role_labels", []),
        "source_candidate_types": base_row.get("source_candidate_types", []),
        "source_model_stages": base_row.get("source_model_stages", []),
        "crop_profile_name": crop_profile,
        "torso_crop_bbox": c1b_row.get("torso_crop_bbox") or feature_row.get("torso_crop_bbox"),
        "crop_quality": c1b_row.get("crop_quality") or feature_row.get("crop_quality", ""),
        "crop_quality_reason": c1b_row.get("crop_quality_reason") or feature_row.get("crop_quality_reason", ""),
        "current_c1_team_colour_belief": c1_row.get("team_colour_belief", ""),
        "current_c1_confidence": safe_float(c1_row.get("team_colour_belief_confidence")),
        "c1b_best_sandbox_team_colour_belief": c1b_row.get("team_colour_belief", ""),
        "c1b_best_sandbox_confidence": safe_float(c1b_row.get("team_colour_belief_confidence")),
        "gold_visible_person_type_prefill": gold_prefill_row.get("visible_person_type_gold", "") if gold_prefill_row else "",
        "seed_candidate_category": category,
        "prefill_suggested_manual_label": CATEGORY_PREFILL_LABELS[category],
        "prefill_only": True,
        "human_confirmed": False,
        "reviewer_label_required": True,
        "seed_candidate_reason": reason,
        "review_priority": priority,
        "audit_issue_flags": audit_row.get("audit_issue_flags", []),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def build_colour_seed_candidate_payloads(
    base_payload: dict[str, Any],
    feature_payload: dict[str, Any],
    c1_belief_payload: dict[str, Any],
    c1b_best_payload: dict[str, Any],
    c1b_audit_payload: dict[str, Any],
    c1b_eval_summary: dict[str, Any],
    c1b_confusion_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    features_by_id = index_rows(feature_payload)
    c1_by_id = index_rows(c1_belief_payload)
    c1b_by_id = index_rows(c1b_best_payload)
    audit_by_id = index_rows(c1b_audit_payload)
    gold_prefill_by_id = best_profile_gold_prefill_rows(c1b_eval_summary, c1b_confusion_payload)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for base_row in base_payload.get("rows", []):
        base_id = str(base_row.get("visible_person_base_id", ""))
        feature_row = features_by_id.get(base_id, {})
        c1_row = c1_by_id.get(base_id, {})
        c1b_row = c1b_by_id.get(base_id, {})
        audit_row = audit_by_id.get(base_id, {})
        category, reason, priority = candidate_category(
            base_row,
            c1_row,
            c1b_row,
            audit_row,
            gold_prefill_by_id.get(base_id),
        )
        buckets[category].append(
            seed_candidate_row(
                base_row,
                feature_row,
                c1_row,
                c1b_row,
                audit_row,
                gold_prefill_by_id.get(base_id),
                category,
                reason,
                priority,
            )
        )
    rows = []
    for category in SEED_CANDIDATE_CATEGORIES:
        for row in balanced_select(buckets.get(category, []), CATEGORY_LIMITS[category]):
            row["seed_candidate_id"] = seed_candidate_id(category, len(rows) + 1, str(row.get("visible_person_base_id", "")))
            rows.append(row)
    category_counts = Counter(row.get("seed_candidate_category", "") for row in rows)
    frame_counts = Counter(int(safe_float(row.get("frame_sequence"), -1)) for row in rows)
    candidate_payload = {
        "artifact": "step1c1c_colour_seed_candidate_rows",
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
        "seed_candidates_prefill_only": True,
        "human_confirmation_required": True,
        "c1b_best_profile_name": c1b_eval_summary.get("c1b_best_profile_name", ""),
        "c1b_best_prototype_strategy": c1b_eval_summary.get("c1b_best_prototype_strategy", ""),
        "rows": rows,
        "summary": {
            "step1c1c_colour_seed_candidate_rows": len(rows),
            "seed_candidate_category_counts": dict(sorted(category_counts.items())),
            "frames_represented": len(frame_counts),
            "gold_prefill_candidate_rows": sum(1 for row in rows if row.get("gold_visible_person_type_prefill")),
            "prefill_only_rows": len(rows),
            "human_confirmed_rows": 0,
        },
    }
    summary_payload = {
        "artifact": "step1c1c_colour_seed_candidate_summary",
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
        "seed_candidate_selection_policy": [
            "Prefer observed visible rows with high/medium crop quality.",
            "Use Gold visible_person_type_gold only as prefill_only visual QA context.",
            "Include ambiguous, context, dark, other-distinct, and crop-failure review examples.",
            "Never auto-approve or auto-promote a seed.",
        ],
        "summary": candidate_payload["summary"],
    }
    template_payload = manual_seed_label_template_payload(candidate_payload)
    return candidate_payload, summary_payload, template_payload


def write_manual_seed_template_csv(template_payload: dict[str, Any]) -> None:
    rows = template_payload.get("rows", [])
    fieldnames = [
        "seed_candidate_id",
        "visible_person_base_id",
        "frame_sequence",
        "crop_profile_name",
        "prefill_suggested_manual_label",
        "manual_colour_label",
        "manual_label_confidence",
        "reviewer_notes",
        "reviewer_name",
        "reviewed_at",
        "human_confirmed",
        "visual_only_warning",
        "do_not_use_for_metrics",
        "production_ready",
    ]
    STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_and_write_colour_seed_candidates() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidate_payload, summary_payload, template_payload = build_colour_seed_candidate_payloads(
        read_json(STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH),
        read_json(STEP1C1_COLOUR_FEATURE_ROWS_PATH),
        read_json(STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH),
        read_json(STEP1C1B_BEST_SANDBOX_BELIEF_ROWS_PATH),
        read_json(STEP1C1B_CROP_AUDIT_ROWS_PATH),
        read_json(STEP1C1B_PROFILE_EVAL_SUMMARY_PATH),
        read_json(STEP1C1B_GOLD8_CLUSTER_CONFUSION_ROWS_PATH),
    )
    write_json(STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH, candidate_payload)
    write_json(STEP1C1C_COLOUR_SEED_CANDIDATE_SUMMARY_PATH, summary_payload)
    write_json(STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_JSON_PATH, template_payload)
    write_manual_seed_template_csv(template_payload)
    return candidate_payload, summary_payload, template_payload
