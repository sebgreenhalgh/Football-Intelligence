# ruff: noqa: E501

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.count_policy import COUNTABLE_ACTIONS, SHADOW_ACTIONS
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B3_COUNT_POLICY_ROWS_PATH,
    STEP1B4_RETAINED_CANDIDATE_PROVENANCE_ROWS_PATH,
    STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH,
    read_json,
    write_json,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    bbox_from_item,
    is_observed_visible_state,
    safe_float,
)


BASE_COUNTABLE_ACTIONS = COUNTABLE_ACTIONS | {"low_quality_context_candidate"}

BASE_ROW_FIELDS = [
    "frame_id",
    "frame_sequence",
    "timestamp_seconds",
    "visible_person_base_id",
    "detection_id",
    "source_detection_id",
    "bbox",
    "footpoint",
    "state",
    "observed_visible_candidate",
    "candidate_type",
    "original_role_source",
    "source_role_labels",
    "source_candidate_types",
    "source_model_stages",
    "roi_status",
    "bbox_confidence",
    "bbox_quality_score",
    "bbox_quality_reason",
    "qa_warnings",
    "qa_render_tier",
    "visual_object_group_id",
    "visual_object_group_size",
    "reconciliation_action",
    "reconciliation_confidence",
    "reconciliation_reason",
    "count_as_observed_visible_candidate_b3",
    "count_policy_reason",
    "review_required",
    "source_disagreement_review_required",
    "eligible_for_step1c_team_colour_candidate",
    "eligible_for_step1d_official_context_candidate",
    "eligible_for_step1e_goalkeeper_candidate",
    "eligible_for_identity_tracking",
    "eligible_for_player_slot_assignment",
    "eligible_for_metric_use",
    "visual_only_warning",
    "do_not_use_for_metrics",
    "production_ready",
]


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def base_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    return int(safe_float(row.get("frame_sequence"), -1)), str(row.get("detection_id", ""))


def visible_person_base_exclusion_reason(row: dict[str, Any]) -> str | None:
    action = str(row.get("reconciliation_action", ""))
    if row.get("count_as_observed_visible_candidate_b3") is not True:
        return "b3_count_policy_false"
    if not is_observed_visible_state(str(row.get("state"))):
        return "state_not_observed_clear_or_partial"
    if bbox_from_item(row) is None:
        return "invalid_bbox"
    if action in SHADOW_ACTIONS:
        return "shadow_candidate_excluded_from_visible_base"
    if action not in BASE_COUNTABLE_ACTIONS:
        return "reconciliation_action_not_countable"
    if str(row.get("qa_render_tier", "")) == "unknown_hidden_by_default":
        return "unknown_hidden_by_default_excluded"
    if row.get("visual_only_warning") != VISUAL_ONLY_WARNING:
        return "missing_visual_only_warning"
    if row.get("production_ready") is not PRODUCTION_READY:
        return "production_ready_not_false"
    return None


def visible_person_base_id(row: dict[str, Any], used: set[str]) -> str:
    seed = "|".join(
        [
            str(int(safe_float(row.get("frame_sequence"), -1))),
            str(row.get("detection_id", "")),
            str(row.get("source_detection_id", "")),
            str(row.get("visual_object_group_id", "")),
        ]
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    base_id = f"step1b4_vpb_f{int(safe_float(row.get('frame_sequence'), -1)):06d}_{digest}"
    candidate_id = base_id
    suffix = 2
    while candidate_id in used:
        candidate_id = f"{base_id}_{suffix:02d}"
        suffix += 1
    used.add(candidate_id)
    return candidate_id


def source_role_source(row: dict[str, Any]) -> str:
    value = str(row.get("original_role_source", "")).strip()
    if value:
        return value
    candidate_type = str(row.get("candidate_type", "")).strip()
    return candidate_type.removesuffix("_candidate_source") if candidate_type else "unknown"


def build_visible_person_base_row(row: dict[str, Any], base_id: str) -> dict[str, Any]:
    item = {
        "frame_id": str(row.get("frame_id", "")),
        "frame_sequence": int(safe_float(row.get("frame_sequence"), -1)),
        "timestamp_seconds": safe_float(row.get("timestamp_seconds")),
        "visible_person_base_id": base_id,
        "detection_id": str(row.get("detection_id", "")),
        "source_detection_id": str(row.get("source_detection_id", "")),
        "bbox": row.get("bbox", {}),
        "footpoint": row.get("footpoint", {}),
        "state": str(row.get("state", "")),
        "observed_visible_candidate": True,
        "candidate_type": str(row.get("candidate_type", "")),
        "original_role_source": source_role_source(row),
        "source_role_labels": as_list(row.get("source_role_labels")),
        "source_candidate_types": as_list(row.get("source_candidate_types") or row.get("candidate_type")),
        "source_model_stages": as_list(row.get("source_model_stages")),
        "roi_status": str(row.get("roi_status", "")),
        "bbox_confidence": safe_float(row.get("bbox_confidence")),
        "bbox_quality_score": safe_float(row.get("bbox_quality_score")),
        "bbox_quality_reason": str(row.get("bbox_quality_reason", "")),
        "qa_warnings": as_list(row.get("qa_warnings")),
        "qa_render_tier": str(row.get("qa_render_tier", "")),
        "visual_object_group_id": str(row.get("visual_object_group_id", "")),
        "visual_object_group_size": int(safe_float(row.get("visual_object_group_size"), 1)),
        "reconciliation_action": str(row.get("reconciliation_action", "")),
        "reconciliation_confidence": safe_float(row.get("reconciliation_confidence")),
        "reconciliation_reason": str(row.get("reconciliation_reason", "")),
        "count_as_observed_visible_candidate_b3": True,
        "count_policy_reason": str(row.get("count_policy_reason", "")),
        "review_required": bool(row.get("review_required")),
        "source_disagreement_review_required": bool(row.get("source_disagreement_review_required")),
        "eligible_for_step1c_team_colour_candidate": True,
        "eligible_for_step1d_official_context_candidate": True,
        "eligible_for_step1e_goalkeeper_candidate": True,
        "eligible_for_identity_tracking": False,
        "eligible_for_player_slot_assignment": False,
        "eligible_for_metric_use": False,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }
    return {field: item[field] for field in BASE_ROW_FIELDS}


def build_visible_person_base_payloads(count_policy_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    used_ids: set[str] = set()
    linked_ids: dict[str, str] = {}
    base_rows: list[dict[str, Any]] = []
    sorted_rows = sorted(list(count_policy_payload.get("rows", [])), key=base_sort_key)
    for row in sorted_rows:
        if visible_person_base_exclusion_reason(row) is not None:
            continue
        base_id = visible_person_base_id(row, used_ids)
        linked_ids[str(row.get("detection_id", ""))] = base_id
        base_rows.append(build_visible_person_base_row(row, base_id))

    provenance_rows = []
    for row in sorted_rows:
        reason = visible_person_base_exclusion_reason(row)
        included = reason is None
        item = dict(row)
        item["visible_person_base_included"] = included
        item["visible_person_base_exclusion_reason"] = "" if included else str(reason)
        item["linked_visible_person_base_id"] = linked_ids.get(str(row.get("detection_id", "")), "")
        item["visual_only_warning"] = VISUAL_ONLY_WARNING
        item["do_not_use_for_metrics"] = True
        item["production_ready"] = PRODUCTION_READY
        provenance_rows.append(item)

    actions = Counter(str(row.get("reconciliation_action", "")) for row in base_rows)
    exclusion_reasons = Counter(str(row.get("visible_person_base_exclusion_reason", "")) for row in provenance_rows if not row.get("visible_person_base_included"))
    base_payload = {
        "artifact": "step1b4_visible_person_base_rows",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "visible_person_base_candidate_for_step1c": True,
        "auto_promoted": False,
        "source_b3_count_policy_artifact": count_policy_payload.get("artifact", ""),
        "rows": base_rows,
        "summary": {
            "visible_person_base_rows": len(base_rows),
            "source_b3_rows": len(sorted_rows),
            "review_required_rows": sum(1 for row in base_rows if row.get("review_required") is True),
            "source_disagreement_review_required_rows": sum(1 for row in base_rows if row.get("source_disagreement_review_required") is True),
            "reconciliation_action_counts": dict(sorted(actions.items())),
        },
    }
    provenance_payload = {
        "artifact": "step1b4_retained_candidate_provenance_rows",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "all_b3_rows_preserved": True,
        "rows": provenance_rows,
        "summary": {
            "source_b3_rows": len(sorted_rows),
            "provenance_rows": len(provenance_rows),
            "visible_person_base_included_rows": len(base_rows),
            "visible_person_base_excluded_rows": len(provenance_rows) - len(base_rows),
            "exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
        },
    }
    return base_payload, provenance_payload


def build_and_write_visible_person_base() -> tuple[dict[str, Any], dict[str, Any]]:
    count_policy_payload = read_json(STEP1B3_COUNT_POLICY_ROWS_PATH)
    base_payload, provenance_payload = build_visible_person_base_payloads(count_policy_payload)
    write_json(STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH, base_payload)
    write_json(STEP1B4_RETAINED_CANDIDATE_PROVENANCE_ROWS_PATH, provenance_payload)
    return base_payload, provenance_payload
