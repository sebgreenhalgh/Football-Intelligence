from __future__ import annotations

from typing import Any


VISUAL_ONLY_WARNING = "VISUAL_ONLY_NOT_METRIC"
PRODUCTION_READY = False
PROJECT_WIDE_DEFAULTS_CHANGED = False
STAGE3D_REGISTRIES_CHANGED = False

CANDIDATE_TYPES = {
    "person_candidate",
    "player_candidate_source",
    "official_candidate_source",
    "referee_candidate_source",
    "staff_context_candidate_source",
    "unknown_candidate_source",
    "off_pitch_person_candidate",
    "false_positive_candidate",
    "unknown_person_candidate",
}

STEP1_STATE_VALUES = {"observed_clear", "observed_partial", "unknown"}
RESERVED_STATE_VALUES = {"occluded", "carried_missing", "not_visible"}
VISIBLE_OBSERVED_STATES = {"observed_clear", "observed_partial"}

REQUIRED_ROW_FIELDS = [
    "frame_id",
    "frame_sequence",
    "timestamp_seconds",
    "detection_id",
    "source_detection_id",
    "bbox",
    "footpoint",
    "candidate_type",
    "bbox_confidence",
    "bbox_quality_score",
    "bbox_quality_reason",
    "crop_quality",
    "roi_status",
    "duplicate_group_id",
    "duplicate_action",
    "state",
    "confidence",
    "reason",
    "visual_only_warning",
    "do_not_use_for_metrics",
    "production_ready",
]

REQUIRED_BBOX_FIELDS = ["x1", "y1", "x2", "y2"]
REQUIRED_FOOTPOINT_FIELDS = ["x", "y", "method", "confidence"]

FORBIDDEN_OUTPUT_KEYS = {
    "identity_id",
    "player_identity_id",
    "stable_identity_id",
    "player_slot_id",
    "slot_id",
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
}

RESTRICTIONS = [
    VISUAL_ONLY_WARNING,
    "visual_only_person_reconstruction_foundation",
    "do_not_calculate_speed_distance_fatigue_player_load_team_shape_pass_dribble_tactical_or_physical_metrics",
    "do_not_perform_identity_tracking",
    "do_not_assign_player_slots",
    "do_not_treat_projected_2d_coordinates_as_metric_truth",
    "project_wide_defaults_changed_false",
    "stage3d_registries_changed_false",
    "production_ready_false",
]


class Step1SchemaError(ValueError):
    """Raised when a Step1 visual-only payload violates its local schema."""


def visual_stamp(payload: dict[str, Any]) -> dict[str, Any]:
    payload["visual_only_warning"] = VISUAL_ONLY_WARNING
    payload["do_not_use_for_metrics"] = True
    payload["production_ready"] = PRODUCTION_READY
    return payload


def governance_stamp(payload: dict[str, Any]) -> dict[str, Any]:
    visual_stamp(payload)
    payload["project_wide_defaults_changed"] = PROJECT_WIDE_DEFAULTS_CHANGED
    payload["stage3d_registries_changed"] = STAGE3D_REGISTRIES_CHANGED
    payload["restrictions"] = list(RESTRICTIONS)
    return payload


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def round_float(value: Any, digits: int = 3, default: float = 0.0) -> float:
    return round(safe_float(value, default), digits)


def bbox_from_item(item: dict[str, Any]) -> dict[str, float] | None:
    if isinstance(item.get("bbox"), dict):
        item = item["bbox"]
    if all(key in item for key in REQUIRED_BBOX_FIELDS):
        bbox = {key: safe_float(item.get(key)) for key in REQUIRED_BBOX_FIELDS}
        if bbox["x2"] <= bbox["x1"] or bbox["y2"] <= bbox["y1"]:
            return None
        return bbox
    return None


def bbox_area(bbox: dict[str, Any] | None) -> float:
    if not bbox:
        return 0.0
    return max(0.0, safe_float(bbox.get("x2")) - safe_float(bbox.get("x1"))) * max(
        0.0,
        safe_float(bbox.get("y2")) - safe_float(bbox.get("y1")),
    )


def bbox_iou(left: dict[str, Any] | None, right: dict[str, Any] | None) -> float:
    if not left or not right:
        return 0.0
    ix1 = max(safe_float(left.get("x1")), safe_float(right.get("x1")))
    iy1 = max(safe_float(left.get("y1")), safe_float(right.get("y1")))
    ix2 = min(safe_float(left.get("x2")), safe_float(right.get("x2")))
    iy2 = min(safe_float(left.get("y2")), safe_float(right.get("y2")))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(left) + bbox_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def footpoint_from_bbox(bbox: dict[str, Any] | None) -> dict[str, Any]:
    if not bbox:
        return {"x": None, "y": None, "method": "missing_bbox", "confidence": 0.0}
    return {
        "x": round_float((safe_float(bbox["x1"]) + safe_float(bbox["x2"])) / 2.0),
        "y": round_float(bbox["y2"]),
        "method": "bbox_bottom_center",
        "confidence": 0.85,
    }


def source_footpoint(item: dict[str, Any], bbox: dict[str, Any] | None) -> dict[str, Any]:
    if "footpoint_x" in item and "footpoint_y" in item:
        return {
            "x": round_float(item.get("footpoint_x")),
            "y": round_float(item.get("footpoint_y")),
            "method": "source_footpoint_fields",
            "confidence": 0.95,
        }
    return footpoint_from_bbox(bbox)


def is_observed_visible_state(state: str) -> bool:
    return state in VISIBLE_OBSERVED_STATES


def short_detection_label(detection_id: str, max_chars: int = 14) -> str:
    value = str(detection_id or "")
    if len(value) <= max_chars:
        return value
    return value[-max_chars:]


def validate_row(row: dict[str, Any], *, allowed_states: set[str] | None = None) -> None:
    missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
    if missing:
        raise Step1SchemaError(f"Step1 row missing required fields: {missing}")
    if row["visual_only_warning"] != VISUAL_ONLY_WARNING:
        raise Step1SchemaError("Step1 row is missing VISUAL_ONLY_NOT_METRIC")
    if row["do_not_use_for_metrics"] is not True:
        raise Step1SchemaError("Step1 row has do_not_use_for_metrics != true")
    if row["production_ready"] is not False:
        raise Step1SchemaError("Step1 row has production_ready != false")
    if row["candidate_type"] not in CANDIDATE_TYPES:
        raise Step1SchemaError(f"Unknown Step1 candidate_type: {row['candidate_type']}")
    allowed = allowed_states or STEP1_STATE_VALUES
    if row["state"] not in allowed:
        raise Step1SchemaError(f"Unknown Step1 state: {row['state']}")
    bbox = row.get("bbox")
    if not isinstance(bbox, dict) or any(field not in bbox for field in REQUIRED_BBOX_FIELDS):
        raise Step1SchemaError("Step1 row bbox must include x1/y1/x2/y2")
    footpoint = row.get("footpoint")
    if not isinstance(footpoint, dict) or any(field not in footpoint for field in REQUIRED_FOOTPOINT_FIELDS):
        raise Step1SchemaError("Step1 row footpoint must include x/y/method/confidence")
    forbidden = sorted(key for key in FORBIDDEN_OUTPUT_KEYS if key in row)
    if forbidden:
        raise Step1SchemaError(f"Forbidden Step1 row keys present: {forbidden}")


def validate_payload(payload: dict[str, Any], *, artifact: str, allowed_states: set[str] | None = None) -> None:
    if payload.get("artifact") != artifact:
        raise Step1SchemaError(f"Expected artifact {artifact}, got {payload.get('artifact')}")
    if payload.get("visual_only_warning") != VISUAL_ONLY_WARNING:
        raise Step1SchemaError("Payload is missing VISUAL_ONLY_NOT_METRIC")
    if payload.get("do_not_use_for_metrics") is not True:
        raise Step1SchemaError("Payload has do_not_use_for_metrics != true")
    if payload.get("production_ready") is not False:
        raise Step1SchemaError("Payload has production_ready != false")
    if payload.get("project_wide_defaults_changed") is not False:
        raise Step1SchemaError("Payload changed project-wide defaults")
    if payload.get("stage3d_registries_changed") is not False:
        raise Step1SchemaError("Payload changed Stage 3D registries")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise Step1SchemaError("Payload rows must be a list")
    for row in rows:
        validate_row(row, allowed_states=allowed_states)
