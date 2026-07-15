from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


SPATIAL_ANNOTATION_SCHEMA_VERSION = "football_intelligence.review_chassis.spatial_annotation.v2"
COORDINATE_SPACE = "original_image_pixels"
MIN_BBOX_SIZE_PX = 2.0
DEFAULT_SCREEN_HIT_TARGET_PX = 10.0

FORBIDDEN_BROWSER_KEYS = {
    "accepted_target",
    "accepted_target_panel",
    "accepted_target_visible_person_base_id",
    "alternative_target_panel",
    "answer",
    "candidate_construction_type",
    "candidate_id",
    "conflict_if_chosen_panel_is_not_prior_accept",
    "correct_target",
    "decision_to_output_mapping",
    "identity",
    "prior_accepted_target",
    "same_frame_alternative_target",
    "sealed_mapping",
    "source_candidate_id",
    "source_detection_id",
    "source_visible_person_base_id",
    "target_a_candidate_id",
    "target_a_visible_person_base_id",
    "target_b_candidate_id",
    "target_b_visible_person_base_id",
    "visible_person_base_id",
}


@dataclass(frozen=True)
class ImageSize:
    width: float
    height: float


@dataclass(frozen=True)
class ViewTransform:
    scale: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float) -> float:
    return round(float(value), 3)


def clamp_point(point: dict[str, Any], image_size: ImageSize) -> dict[str, float]:
    x = _as_float(point.get("x"))
    y = _as_float(point.get("y"))
    if x is None or y is None:
        raise ValueError("point requires numeric x and y")
    return {
        "x": _round(min(max(x, 0.0), image_size.width)),
        "y": _round(min(max(y, 0.0), image_size.height)),
    }


def normalize_bbox(bbox: dict[str, Any], image_size: ImageSize) -> dict[str, float]:
    values = {key: _as_float(bbox.get(key)) for key in ("x1", "y1", "x2", "y2")}
    if any(value is None for value in values.values()):
        raise ValueError("bbox requires numeric x1, y1, x2 and y2")
    x1 = min(values["x1"], values["x2"])  # type: ignore[type-var]
    x2 = max(values["x1"], values["x2"])  # type: ignore[type-var]
    y1 = min(values["y1"], values["y2"])  # type: ignore[type-var]
    y2 = max(values["y1"], values["y2"])  # type: ignore[type-var]
    x1 = min(max(float(x1), 0.0), image_size.width)
    x2 = min(max(float(x2), 0.0), image_size.width)
    y1 = min(max(float(y1), 0.0), image_size.height)
    y2 = min(max(float(y2), 0.0), image_size.height)
    return {"x1": _round(x1), "y1": _round(y1), "x2": _round(x2), "y2": _round(y2)}


def bbox_is_valid(bbox: dict[str, Any], image_size: ImageSize, *, minimum_size: float = MIN_BBOX_SIZE_PX) -> bool:
    try:
        normalized = normalize_bbox(bbox, image_size)
    except ValueError:
        return False
    return normalized["x2"] - normalized["x1"] >= minimum_size and normalized["y2"] - normalized["y1"] >= minimum_size


def move_bbox(bbox: dict[str, Any], dx: float, dy: float, image_size: ImageSize) -> dict[str, float]:
    normalized = normalize_bbox(bbox, image_size)
    width = normalized["x2"] - normalized["x1"]
    height = normalized["y2"] - normalized["y1"]
    x1 = min(max(normalized["x1"] + dx, 0.0), max(0.0, image_size.width - width))
    y1 = min(max(normalized["y1"] + dy, 0.0), max(0.0, image_size.height - height))
    return {"x1": _round(x1), "y1": _round(y1), "x2": _round(x1 + width), "y2": _round(y1 + height)}


def resize_bbox(
    bbox: dict[str, Any],
    handle: str,
    x: float,
    y: float,
    image_size: ImageSize,
    *,
    minimum_size: float = MIN_BBOX_SIZE_PX,
) -> dict[str, float]:
    normalized = normalize_bbox(bbox, image_size)
    updates = dict(normalized)
    if "w" in handle:
        updates["x1"] = min(max(x, 0.0), updates["x2"] - minimum_size)
    if "e" in handle:
        updates["x2"] = max(min(x, image_size.width), updates["x1"] + minimum_size)
    if "n" in handle:
        updates["y1"] = min(max(y, 0.0), updates["y2"] - minimum_size)
    if "s" in handle:
        updates["y2"] = max(min(y, image_size.height), updates["y1"] + minimum_size)
    return normalize_bbox(updates, image_size)


def client_to_image(point: dict[str, Any], transform: ViewTransform) -> dict[str, float]:
    x = _as_float(point.get("x"))
    y = _as_float(point.get("y"))
    if x is None or y is None:
        raise ValueError("client point requires numeric x and y")
    if transform.scale <= 0:
        raise ValueError("scale must be positive")
    return {"x": _round((x - transform.pan_x) / transform.scale), "y": _round((y - transform.pan_y) / transform.scale)}


def image_to_client(point: dict[str, Any], transform: ViewTransform) -> dict[str, float]:
    x = _as_float(point.get("x"))
    y = _as_float(point.get("y"))
    if x is None or y is None:
        raise ValueError("image point requires numeric x and y")
    return {"x": _round(x * transform.scale + transform.pan_x), "y": _round(y * transform.scale + transform.pan_y)}


def hit_test_candidates(
    candidates: list[dict[str, Any]],
    point: dict[str, Any],
    *,
    transform: ViewTransform | None = None,
    screen_hit_target_px: float = DEFAULT_SCREEN_HIT_TARGET_PX,
) -> list[dict[str, Any]]:
    x = _as_float(point.get("x"))
    y = _as_float(point.get("y"))
    if x is None or y is None:
        return []
    image_tolerance = screen_hit_target_px / max(transform.scale if transform else 1.0, 0.01)
    hits = []
    for candidate in candidates:
        bbox = candidate.get("bbox")
        if not isinstance(bbox, dict):
            continue
        x1 = _as_float(bbox.get("x1"))
        y1 = _as_float(bbox.get("y1"))
        x2 = _as_float(bbox.get("x2"))
        y2 = _as_float(bbox.get("y2"))
        if None in (x1, y1, x2, y2):
            continue
        inside = x1 - image_tolerance <= x <= x2 + image_tolerance and y1 - image_tolerance <= y <= y2 + image_tolerance
        if not inside:
            continue
        area = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
        hits.append({**candidate, "_hit_area": _round(area)})
    hits.sort(key=lambda item: (item.get("_hit_area", 0), int(item.get("anonymous_candidate_number", 999999))))
    return hits


def safe_anonymous_candidate(candidate: dict[str, Any], *, target_frame_sequence: int | None = None) -> dict[str, Any]:
    safe = {
        "anonymous_candidate_number": int(candidate["anonymous_candidate_number"]),
        "bbox": candidate["bbox"],
        "bbox_hash": candidate.get("bbox_hash"),
        "frame_sequence": target_frame_sequence or candidate.get("frame_sequence"),
        "confidence": candidate.get("confidence"),
        "class_name": candidate.get("class_name") or "person",
    }
    forbidden = scan_forbidden_browser_payload(safe)
    if forbidden["forbidden_key_count"]:
        raise ValueError(f"anonymous candidate still contains forbidden keys: {forbidden['forbidden_keys']}")
    return safe


def parse_note_payload(note: str | dict[str, Any] | None) -> dict[str, Any]:
    if note is None or note == "":
        return {}
    if isinstance(note, dict):
        return note
    try:
        parsed = json.loads(note)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_spatial_annotation_note(
    note: str | dict[str, Any] | None,
    *,
    case_id: str,
    image_size: ImageSize,
    target_frame_sequence: int | None = None,
) -> dict[str, Any]:
    payload = parse_note_payload(note)
    annotation = payload.get("spatial_annotation") if isinstance(payload.get("spatial_annotation"), dict) else payload
    if not isinstance(annotation, dict):
        annotation = {}
    bbox = annotation.get("reviewer_bbox")
    if not isinstance(bbox, dict):
        legacy_bbox = {
            "x1": annotation.get("bbox_x1"),
            "y1": annotation.get("bbox_y1"),
            "x2": annotation.get("bbox_x2"),
            "y2": annotation.get("bbox_y2"),
        }
        bbox = legacy_bbox if all(value not in (None, "") for value in legacy_bbox.values()) else None
    normalized: dict[str, Any] = {
        "schema_version": SPATIAL_ANNOTATION_SCHEMA_VERSION,
        "case_id": annotation.get("case_id") or case_id,
        "coordinate_space": COORDINATE_SPACE,
        "target_frame_sequence": annotation.get("target_frame_sequence", target_frame_sequence),
        "annotation_source": annotation.get("annotation_source") or annotation.get("source") or "none",
        "confidence": annotation.get("confidence") or "uncertain",
        "bbox_size_category": annotation.get("bbox_size_category") or "uncertain",
        "partial_or_occluded": bool(annotation.get("partial_or_occluded") in (True, "true", "True", "1", 1)),
    }
    if bbox is not None:
        normalized["reviewer_bbox"] = normalize_bbox(bbox, image_size)
        normalized.update(
            {
                "bbox_x1": normalized["reviewer_bbox"]["x1"],
                "bbox_y1": normalized["reviewer_bbox"]["y1"],
                "bbox_x2": normalized["reviewer_bbox"]["x2"],
                "bbox_y2": normalized["reviewer_bbox"]["y2"],
            }
        )
    candidate_number = annotation.get("existing_candidate_number") or annotation.get(
        "selected_anonymous_candidate_number"
    )
    if candidate_number not in (None, ""):
        normalized["existing_candidate_number"] = int(candidate_number)
        normalized["selected_anonymous_candidate_number"] = int(candidate_number)
    footpoint = annotation.get("footpoint") or {
        "x": annotation.get("footpoint_x"),
        "y": annotation.get("footpoint_y"),
    }
    if isinstance(footpoint, dict) and footpoint.get("x") not in (None, "") and footpoint.get("y") not in (None, ""):
        normalized["footpoint"] = clamp_point(footpoint, image_size)
        normalized["footpoint_x"] = normalized["footpoint"]["x"]
        normalized["footpoint_y"] = normalized["footpoint"]["y"]
    occlusion_points = annotation.get("occlusion_points")
    if not isinstance(occlusion_points, list):
        point = annotation.get("occlusion_point") or {
            "x": annotation.get("occlusion_x"),
            "y": annotation.get("occlusion_y"),
        }
        occlusion_points = [point] if isinstance(point, dict) and point.get("x") not in (None, "") else []
    normalized_points = []
    for point in occlusion_points:
        if isinstance(point, dict):
            normalized_points.append({"kind": "occlusion_location", **clamp_point(point, image_size)})
    if normalized_points:
        normalized["occlusion_points"] = normalized_points
        normalized["occlusion_location_status"] = "marked"
    else:
        normalized["occlusion_location_status"] = annotation.get("occlusion_location_status") or "not_applicable"
    if annotation.get("occlusion_not_localizable_reason"):
        normalized["occlusion_not_localizable_reason"] = annotation["occlusion_not_localizable_reason"]
    return normalized


def validate_spatial_annotation_for_decision(
    annotation: dict[str, Any],
    *,
    decision: str,
    image_size: ImageSize,
) -> dict[str, Any]:
    errors: list[str] = []
    if decision == "TARGET_VISIBLE_DRAW_BBOX":
        if not isinstance(annotation.get("reviewer_bbox"), dict) or not bbox_is_valid(
            annotation["reviewer_bbox"], image_size
        ):
            errors.append("reviewer_bbox_required")
    if decision == "TARGET_VISIBLE_SELECT_EXISTING_DETECTION":
        if annotation.get("existing_candidate_number") in (None, ""):
            errors.append("existing_candidate_number_required")
    partial = annotation.get("partial_or_occluded") is True
    has_occlusion_point = bool(annotation.get("occlusion_points"))
    has_not_localizable_reason = annotation.get("occlusion_location_status") == "not_localizable" and bool(
        str(annotation.get("occlusion_not_localizable_reason", "")).strip()
    )
    if partial and not (has_occlusion_point or has_not_localizable_reason):
        errors.append("partial_or_occluded_requires_point_or_reason")
    return {"passed": not errors, "errors": errors}


def autosave_payload(case_id: str, annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "note": json.dumps({"spatial_annotation": annotation}, indent=2, sort_keys=True, ensure_ascii=True),
        "autosave": True,
        "auto_submit_decision": False,
    }


def scan_forbidden_browser_payload(value: Any) -> dict[str, Any]:
    forbidden_keys: list[str] = []
    forbidden_values: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                child_path = f"{path}.{key}" if path else key
                if key in FORBIDDEN_BROWSER_KEYS:
                    forbidden_keys.append(child_path)
                visit(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, str):
            lowered = item.lower()
            if any(fragment in lowered for fragment in ("visible_person_base_id", "candidate_id", "answer_key")):
                forbidden_values.append(path)
            if lowered.startswith("m5_4h1_pc_") or lowered.startswith("m5_4h1_vpb_"):
                forbidden_values.append(path)

    visit(value, "")
    return {
        "forbidden_key_count": len(forbidden_keys),
        "forbidden_value_count": len(forbidden_values),
        "forbidden_keys": sorted(set(forbidden_keys)),
        "forbidden_values": sorted(set(forbidden_values)),
        "predecision_answer_key_delivered_to_client": bool(forbidden_keys or forbidden_values),
    }
