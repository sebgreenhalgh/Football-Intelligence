from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

from football_intelligence.replay.portable_context import guardrail_payload, semantic_hash, utc_now


VALID_ON_PITCH_PERSON = "valid_on_pitch_person_candidate"
VALID_OFFICIAL = "valid_official_candidate"
VALID_OFF_PITCH_PERSON = "valid_off_pitch_person"
PROBABLE_NON_PERSON = "probable_non_person_false_positive"
AMBIGUOUS_ENTITY = "ambiguous_entity_requires_review"

ENTITY_VALIDITY_STATES = {
    VALID_ON_PITCH_PERSON,
    VALID_OFFICIAL,
    VALID_OFF_PITCH_PERSON,
    PROBABLE_NON_PERSON,
    AMBIGUOUS_ENTITY,
}

COMPOUND_CONTINUITY_DECISIONS = {
    "not_applicable_invalid_entity",
    "reject_continuity",
    "accept_visual_continuity",
    "unresolved",
}


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


def round_float(value: Any, digits: int = 4) -> float:
    return round(safe_float(value), digits)


def rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows", [])
    return rows if isinstance(rows, list) else []


def bbox_from_row(row: dict[str, Any]) -> dict[str, float] | None:
    if isinstance(row.get("bbox"), dict):
        bbox = row["bbox"]
        if all(key in bbox for key in ("x1", "y1", "x2", "y2")):
            out = {key: safe_float(bbox[key]) for key in ("x1", "y1", "x2", "y2")}
            if out["x2"] > out["x1"] and out["y2"] > out["y1"]:
                return out
    if all(key in row for key in ("x1", "y1", "x2", "y2")):
        out = {key: safe_float(row[key]) for key in ("x1", "y1", "x2", "y2")}
        if out["x2"] > out["x1"] and out["y2"] > out["y1"]:
            return out
    return None


def bbox_features(bbox: dict[str, float] | None, *, frame_width: int, frame_height: int) -> dict[str, Any]:
    if bbox is None:
        return {
            "bbox_valid": False,
            "width": 0.0,
            "height": 0.0,
            "area": 0.0,
            "aspect_ratio": 0.0,
            "center_x": None,
            "center_y": None,
            "footpoint_x": None,
            "footpoint_y": None,
            "center_x_fraction": None,
            "footpoint_y_fraction": None,
            "edge_clipped_or_adjacent": False,
        }
    width = max(0.0, bbox["x2"] - bbox["x1"])
    height = max(0.0, bbox["y2"] - bbox["y1"])
    center_x = (bbox["x1"] + bbox["x2"]) / 2.0
    center_y = (bbox["y1"] + bbox["y2"]) / 2.0
    frame_width = max(1, frame_width)
    frame_height = max(1, frame_height)
    edge_adjacent = (
        bbox["x1"] <= frame_width * 0.015
        or bbox["x2"] >= frame_width * 0.985
        or bbox["y1"] <= frame_height * 0.015
        or bbox["y2"] >= frame_height * 0.985
    )
    return {
        "bbox_valid": True,
        "width": round_float(width),
        "height": round_float(height),
        "area": round_float(width * height),
        "aspect_ratio": round_float(width / height if height > 0 else 0.0),
        "center_x": round_float(center_x),
        "center_y": round_float(center_y),
        "footpoint_x": round_float(center_x),
        "footpoint_y": round_float(bbox["y2"]),
        "center_x_fraction": round_float(center_x / frame_width),
        "footpoint_y_fraction": round_float(bbox["y2"] / frame_height),
        "edge_clipped_or_adjacent": edge_adjacent,
    }


def static_signature(row: dict[str, Any]) -> str:
    bbox = bbox_from_row(row)
    if bbox is None:
        return "missing_bbox"
    features = bbox_features(bbox, frame_width=1, frame_height=1)
    seed = "|".join(
        [
            str(round(safe_float(features["center_x"]) / 4.0)),
            str(round(safe_float(features["center_y"]) / 4.0)),
            str(round(safe_float(features["width"]) / 4.0)),
            str(round(safe_float(features["height"]) / 4.0)),
        ]
    )
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def frame_dimensions(frame_manifest: dict[str, Any]) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for frame in frame_manifest.get("frames", []):
        if not isinstance(frame, dict):
            continue
        sequence = safe_int(frame.get("sequence", frame.get("frame_sequence")), -1)
        out[sequence] = (safe_int(frame.get("width"), 0), safe_int(frame.get("height"), 0))
    return out


def visible_ids_by_detection(visible_payload: dict[str, Any] | None) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    if not visible_payload:
        return {}
    for row in rows_from_payload(visible_payload):
        visible_id = str(row.get("visible_person_base_id", ""))
        for detection_id in {
            str(row.get("detection_id", "")),
            str(row.get("source_detection_id", "")),
        }:
            if detection_id and visible_id:
                mapping[detection_id].append(visible_id)
    return {key: sorted(set(value)) for key, value in mapping.items()}


def classify_entity_state(
    row: dict[str, Any],
    *,
    frame_width: int,
    frame_height: int,
    static_persistence_count: int,
) -> tuple[str, list[str]]:
    bbox = bbox_from_row(row)
    features = bbox_features(bbox, frame_width=frame_width, frame_height=frame_height)
    confidence = safe_float(row.get("confidence", row.get("bbox_confidence")), 0.0)
    class_name = str(row.get("class_name", row.get("object_type", ""))).lower()
    declared_role = " ".join([str(row.get("object_type", "")), str(row.get("role_label", ""))]).lower()
    reasons: list[str] = []

    if bbox is None or not features["bbox_valid"]:
        return AMBIGUOUS_ENTITY, ["bbox_missing_or_invalid"]
    if class_name and class_name != "person":
        return PROBABLE_NON_PERSON, [f"detector_class_is_{class_name}"]

    width = safe_float(features["width"])
    height = safe_float(features["height"])
    area = safe_float(features["area"])
    aspect = safe_float(features["aspect_ratio"])
    foot_y_fraction = safe_float(features["footpoint_y_fraction"])
    center_x_fraction = safe_float(features["center_x_fraction"])
    small_or_thin = height < 42.0 or width < 18.0 or area < 760.0
    structure_like = (small_or_thin and aspect < 0.62) or (width <= 18.0 and height <= 45.0)
    edge_adjacent = bool(features["edge_clipped_or_adjacent"]) or center_x_fraction >= 0.955
    high_static = static_persistence_count >= 4
    high_static_structure = high_static and structure_like
    off_pitch_band = foot_y_fraction <= 0.30

    if "official" in declared_role:
        reasons.append("declared_official_context")
        return VALID_OFFICIAL, reasons
    if high_static_structure and (edge_adjacent or confidence < 0.45 or off_pitch_band):
        reasons.extend(
            [
                "static_structure_like_crop",
                "static_persistence_supports_quarantine_but_does_not_promote_validity",
            ]
        )
        if edge_adjacent:
            reasons.append("edge_adjacent_image_location")
        if off_pitch_band:
            reasons.append("upper_image_band")
        return PROBABLE_NON_PERSON, reasons
    if confidence >= 0.72 and height >= 48.0 and foot_y_fraction >= 0.34:
        reasons.append("high_confidence_human_sized_image_space_box")
        return VALID_ON_PITCH_PERSON, reasons
    if confidence >= 0.45 and height >= 52.0 and foot_y_fraction >= 0.36 and not structure_like:
        reasons.append("plausible_on_pitch_person_image_space_box")
        return VALID_ON_PITCH_PERSON, reasons
    if confidence >= 0.55 and off_pitch_band and height >= 24.0 and not edge_adjacent:
        reasons.append("upper_image_band_person_like_box")
        return VALID_OFF_PITCH_PERSON, reasons

    if high_static:
        reasons.append("static_detection_requires_context_review")
    if small_or_thin:
        reasons.append("small_or_thin_bbox")
    if confidence < 0.32:
        reasons.append("low_detector_confidence")
    if off_pitch_band:
        reasons.append("upper_image_band")
    if edge_adjacent:
        reasons.append("edge_adjacent_image_location")
    if not reasons:
        reasons.append("insufficient_image_space_evidence")
    return AMBIGUOUS_ENTITY, sorted(set(reasons))


def build_entity_validity_payload(
    detector_payload: dict[str, Any],
    *,
    frame_manifest: dict[str, Any] | None = None,
    visible_person_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    detector_rows = rows_from_payload(detector_payload)
    dimensions = frame_dimensions(frame_manifest or {})
    signature_counts = Counter(static_signature(row) for row in detector_rows)
    visible_mapping = visible_ids_by_detection(visible_person_payload)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(detector_rows):
        sequence = safe_int(row.get("frame_sequence", row.get("sequence")), -1)
        frame_width, frame_height = dimensions.get(
            sequence, (safe_int(row.get("width"), 2730), safe_int(row.get("height"), 720))
        )
        signature = static_signature(row)
        state, reasons = classify_entity_state(
            row,
            frame_width=frame_width,
            frame_height=frame_height,
            static_persistence_count=signature_counts[signature],
        )
        bbox = bbox_from_row(row)
        features = bbox_features(bbox, frame_width=frame_width, frame_height=frame_height)
        detection_id = str(row.get("detection_id") or f"detector_row_{index:06d}")
        rows.append(
            {
                "entity_validity_row_id": f"entity_validity_{index:06d}",
                "detection_id": detection_id,
                "source_detection_id": str(row.get("source_detection_id", detection_id)),
                "visible_person_base_ids": visible_mapping.get(detection_id, []),
                "frame_sequence": sequence,
                "frame_file": row.get("frame_file", ""),
                "bbox": bbox or {},
                "confidence": round_float(row.get("confidence", row.get("bbox_confidence"))),
                "class_name": str(row.get("class_name", "")),
                "declared_object_type": str(row.get("object_type", "")),
                "declared_role_label": str(row.get("role_label", "")),
                "entity_validity_state": state,
                "entity_validity_reasons": reasons,
                "quarantine_classification": state if state == PROBABLE_NON_PERSON else "",
                "detector_row_deleted": False,
                "static_persistence_signature": signature,
                "static_persistence_count": int(signature_counts[signature]),
                "image_space_features": features,
                "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
                "do_not_use_for_metrics": True,
                "production_ready": False,
                "no_auto_promotion": True,
                "human_approved": False,
                "safe_to_apply_globally": False,
                "sandbox_only": True,
            }
        )
    state_counts = Counter(str(row["entity_validity_state"]) for row in rows)
    payload = guardrail_payload(
        {
            "artifact": "m5_4c_entity_validity_rows",
            "created_at": utc_now(),
            "entity_validity_rule_version": "m5.4c.image_space_entity_validity.v1",
            "classification_scope": "detector_rows_non_destructive",
            "detector_rows_input": len(detector_rows),
            "detector_rows_output": len(rows),
            "all_detector_rows_preserved": len(detector_rows) == len(rows),
            "rows": rows,
            "summary": {
                "entity_validity_rows": len(rows),
                "state_counts": dict(sorted(state_counts.items())),
                "probable_non_person_false_positive_rows": state_counts.get(PROBABLE_NON_PERSON, 0),
                "valid_on_pitch_person_candidate_rows": state_counts.get(VALID_ON_PITCH_PERSON, 0),
                "valid_official_candidate_rows": state_counts.get(VALID_OFFICIAL, 0),
                "valid_off_pitch_person_rows": state_counts.get(VALID_OFF_PITCH_PERSON, 0),
                "ambiguous_entity_requires_review_rows": state_counts.get(AMBIGUOUS_ENTITY, 0),
            },
        }
    )
    payload["entity_validity_payload_hash"] = semantic_hash(
        [{"detection_id": row["detection_id"], "state": row["entity_validity_state"]} for row in rows]
    )
    return payload


def entity_rows_by_visible_id(entity_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows_from_payload(entity_payload):
        for visible_id in row.get("visible_person_base_ids", []) or []:
            mapping[str(visible_id)] = row
    return mapping


def compound_continuity_disposition(
    *,
    source_entity_validity: str,
    target_entity_validity: str,
    continuity_decision: str | None,
) -> dict[str, Any]:
    if source_entity_validity == PROBABLE_NON_PERSON or target_entity_validity == PROBABLE_NON_PERSON:
        decision = "not_applicable_invalid_entity"
    elif continuity_decision in {"accept_visual_continuity", "accept_continuity"}:
        decision = "accept_visual_continuity"
    elif continuity_decision in {"reject_continuity", "reject_edge"}:
        decision = "reject_continuity"
    else:
        decision = "unresolved"
    return {
        "source_entity_validity": source_entity_validity,
        "target_entity_validity": target_entity_validity,
        "continuity_decision": decision,
        "continuity_decision_allowed": decision in COMPOUND_CONTINUITY_DECISIONS,
        "accepted_continuity": decision == "accept_visual_continuity",
    }
