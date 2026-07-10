# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B3_COUNT_POLICY_ROWS_PATH,
    STEP1B3_RECONCILIATION_ROWS_PATH,
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


COUNTABLE_ACTIONS = {
    "primary_observation_candidate",
    "retained_overlap_candidate",
    "context_observation_candidate",
    "off_roi_context_candidate",
}

SHADOW_ACTIONS = {"duplicate_shadow_candidate", "source_overlap_shadow_candidate"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def valid_bbox(row: dict[str, Any]) -> bool:
    return bbox_from_item(row) is not None


def count_policy_for_row(row: dict[str, Any]) -> tuple[bool, str]:
    action = str(row.get("reconciliation_action", ""))
    state = str(row.get("state", "unknown"))
    tier = str(row.get("qa_render_tier", ""))
    quality = safe_float(row.get("bbox_quality_score"))
    roi = str(row.get("roi_status", ""))
    issue_flags = set(row.get("issue_flags", []))
    gold8_supported = row.get("gold8_visible_match_support") is True

    if not is_observed_visible_state(state):
        return False, "state_not_observed_visible"
    if not valid_bbox(row):
        return False, "invalid_bbox"
    if action in SHADOW_ACTIONS:
        return False, f"{action}_not_counted"
    if tier == "unknown_hidden_by_default":
        return False, "unknown_hidden_by_default_not_counted"
    if "observed_candidate_near_excluded_gold_nonperson" in issue_flags and not gold8_supported:
        return False, "near_excluded_gold_nonperson"
    if action == "low_quality_context_candidate":
        if gold8_supported:
            return True, "low_quality_context_gold8_visible_match_support_counted"
        return False, "low_quality_context_retained_not_primary_counted"
    if action == "review_required_candidate":
        return False, "review_required_unsafe_to_count"
    if action == "off_roi_context_candidate" and quality < 0.45:
        return False, "outside_roi_not_visually_plausible"
    if roi == "outside_playing_roi" and action not in {"off_roi_context_candidate", "retained_overlap_candidate"}:
        return False, "outside_roi_not_primary_context"
    if action in COUNTABLE_ACTIONS:
        return True, f"{action}_counted"
    return False, "action_not_countable"


def apply_count_policy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        counted, reason = count_policy_for_row(row)
        item = dict(row)
        item["count_as_observed_visible_candidate_b3"] = counted
        item["count_policy_reason"] = reason
        item["count_policy_presentation_only_or_sandbox"] = True
        item["production_ready"] = PRODUCTION_READY
        item["visual_only_warning"] = VISUAL_ONLY_WARNING
        item["do_not_use_for_metrics"] = True
        out.append(item)
    return out


def build_count_policy_payload(reconciliation_payload: dict[str, Any]) -> dict[str, Any]:
    rows = apply_count_policy(list(reconciliation_payload.get("rows", [])))
    action_counts = Counter(str(row.get("reconciliation_action", "")) for row in rows)
    counted_action_counts = Counter(
        str(row.get("reconciliation_action", ""))
        for row in rows
        if row.get("count_as_observed_visible_candidate_b3") is True
    )
    reason_counts = Counter(str(row.get("count_policy_reason", "")) for row in rows)
    return {
        "artifact": "step1b3_count_policy_rows",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "count_policy_sandbox_only": True,
        "candidate_retention_preserved": True,
        "rows": rows,
        "summary": {
            "total_rows": len(rows),
            "counted_observed_visible_rows": sum(1 for row in rows if row.get("count_as_observed_visible_candidate_b3") is True),
            "not_counted_rows": sum(1 for row in rows if row.get("count_as_observed_visible_candidate_b3") is not True),
            "gold8_visible_match_support_rows": sum(1 for row in rows if row.get("gold8_visible_match_support") is True),
            "reconciliation_action_counts": dict(sorted(action_counts.items())),
            "counted_action_counts": dict(sorted(counted_action_counts.items())),
            "count_policy_reason_counts": dict(sorted(reason_counts.items())),
        },
    }


def build_and_write_count_policy_rows(reconciliation_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    reconciliation_payload = reconciliation_payload or read_json(STEP1B3_RECONCILIATION_ROWS_PATH)
    payload = build_count_policy_payload(reconciliation_payload)
    write_json(STEP1B3_COUNT_POLICY_ROWS_PATH, payload)
    return payload
