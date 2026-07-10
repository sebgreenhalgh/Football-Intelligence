from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


VISUAL_ONLY_WARNING = "VISUAL_ONLY_NOT_METRIC"
PRODUCTION_READY = False
NO_AUTO_PROMOTION = True
SANDBOX_ONLY = True

DEFAULT_MAX_FRAME_GAP = 3
MAX_FRAME_GAP_HARD_CAP = 10
TARGET_REVIEW_MIN_CANDIDATES = 60
TARGET_REVIEW_MAX_CANDIDATES = 90
HARD_MAX_REVIEW_CANDIDATES = 120
DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_FRAMES = 30
DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_SECONDS = 3.0

AUTO_ACCEPT_STATE = "auto_accept_candidate"
NEEDS_REVIEW_STATE = "needs_review_candidate"
AUTO_REJECT_STATE = "auto_reject_candidate"

ACCEPT_DECISION = "accept_short_window_visual_continuity_edge"
REJECT_DECISION = "reject_edge"
UNSURE_DECISION = "unsure_needs_later_review"
BULK_ACCEPT_DECISION = "bulk_accept_safe_bucket"
ALLOWED_REVIEW_DECISIONS = {
    ACCEPT_DECISION,
    REJECT_DECISION,
    UNSURE_DECISION,
    BULK_ACCEPT_DECISION,
}

SAFE_BULK_REVIEW_BUCKETS = {"safe_auto_accept_audit"}

FORBIDDEN_OUTPUT_KEYS = {
    "track_id",
    "identity_id",
    "player_identity_id",
    "stable_identity_id",
    "persistent_player_id",
    "player_slot_id",
    "slot_id",
    "goalkeeper_slot_id",
    "gk_slot_id",
    "assigned_goalkeeper_slot",
    "goalkeeper_identity_id",
    "expected_22_role_state",
    "pitch_x_metric",
    "pitch_y_metric",
    "speed",
    "distance",
    "fatigue",
    "player_load",
    "team_shape",
    "pass",
    "dribble",
    "tactical",
    "physical_performance",
    "event",
    "event_label",
    "football_conclusion",
}

FORBIDDEN_APPROVAL_FLAGS = [
    "approve_any_identity_tracking",
    "approve_any_player_slot_use",
    "approve_any_goalkeeper_slot_use",
    "approve_any_metric_use",
    "approve_event_or_tactical_analysis",
    "approve_exact_22_or_exact_two_goalkeeper_forcing",
    "approve_official_referee_exclusion",
    "approve_bad_detection_deletion",
    "approve_production_promotion",
]


class Step2M1SchemaError(ValueError):
    """Raised when a Step2.M1 visual-continuity payload violates its schema."""


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def clamp01(value: Any) -> float:
    return max(0.0, min(1.0, safe_float(value)))


def round_float(value: Any, digits: int = 4, default: float = 0.0) -> float:
    return round(safe_float(value, default), digits)


def visual_stamp(payload: dict[str, Any]) -> dict[str, Any]:
    payload["visual_only_warning"] = VISUAL_ONLY_WARNING
    payload["do_not_use_for_metrics"] = True
    payload["production_ready"] = PRODUCTION_READY
    payload["no_auto_promotion"] = NO_AUTO_PROMOTION
    payload.setdefault("human_approved", False)
    return payload


def guardrail_stamp(payload: dict[str, Any]) -> dict[str, Any]:
    visual_stamp(payload)
    payload.update(
        {
            "sandbox_only": SANDBOX_ONLY,
            "identity_tracking_performed": False,
            "player_slots_assigned": False,
            "goalkeeper_slots_assigned": False,
            "expected_22_role_states_created": False,
            "exact_22_forcing_performed": False,
            "exact_two_goalkeeper_forcing_performed": False,
            "official_referee_exclusion_performed": False,
            "bad_detection_rows_deleted": False,
            "metric_analysis_performed": False,
            "event_analysis_performed": False,
            "tactical_analysis_performed": False,
            "physical_performance_analysis_performed": False,
            "auto_promoted": False,
        }
    )
    return payload


def forbidden_keys_present(value: Any) -> list[str]:
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in FORBIDDEN_OUTPUT_KEYS:
                    found.add(key)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return sorted(found)


def assert_no_forbidden_keys(value: Any) -> None:
    forbidden = forbidden_keys_present(value)
    if forbidden:
        raise Step2M1SchemaError(f"Forbidden Step2.M1 keys present: {forbidden}")


def assert_visual_guardrails(row: dict[str, Any], *, require_no_auto_promotion: bool = True) -> None:
    if row.get("visual_only_warning") != VISUAL_ONLY_WARNING:
        raise Step2M1SchemaError("Step2.M1 row is missing VISUAL_ONLY_NOT_METRIC")
    if row.get("do_not_use_for_metrics") is not True:
        raise Step2M1SchemaError("Step2.M1 row has do_not_use_for_metrics != true")
    if row.get("production_ready") is not False:
        raise Step2M1SchemaError("Step2.M1 row has production_ready != false")
    if require_no_auto_promotion and row.get("no_auto_promotion") is not True:
        raise Step2M1SchemaError("Step2.M1 row has no_auto_promotion != true")


def rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows", [])
    return rows if isinstance(rows, list) else []


def bbox_from_row(row: dict[str, Any]) -> dict[str, float] | None:
    bbox = row.get("bbox", {})
    if not isinstance(bbox, dict):
        return None
    required = ("x1", "y1", "x2", "y2")
    if any(key not in bbox for key in required):
        return None
    out = {key: safe_float(bbox.get(key)) for key in required}
    if out["x2"] <= out["x1"] or out["y2"] <= out["y1"]:
        return None
    return out


def footpoint_from_row(row: dict[str, Any]) -> dict[str, Any]:
    footpoint = row.get("footpoint", {})
    if isinstance(footpoint, dict) and "x" in footpoint and "y" in footpoint:
        return {
            "x": round_float(footpoint.get("x")),
            "y": round_float(footpoint.get("y")),
            "method": str(footpoint.get("method", "source_footpoint")),
            "confidence": clamp01(footpoint.get("confidence", 0.0)),
        }
    bbox = bbox_from_row(row)
    if not bbox:
        return {"x": None, "y": None, "method": "missing_bbox", "confidence": 0.0}
    return {
        "x": round_float((bbox["x1"] + bbox["x2"]) / 2.0),
        "y": round_float(bbox["y2"]),
        "method": "bbox_bottom_center_fallback",
        "confidence": 0.65,
    }


def validate_max_frame_gap(max_frame_gap: int) -> int:
    gap = safe_int(max_frame_gap, DEFAULT_MAX_FRAME_GAP)
    if gap < 1:
        raise Step2M1SchemaError("Step2.M1 max frame gap must be at least 1")
    if gap > MAX_FRAME_GAP_HARD_CAP:
        raise Step2M1SchemaError(
            f"Step2.M1 max frame gap hard cap is {MAX_FRAME_GAP_HARD_CAP}, got {gap}"
        )
    return gap
