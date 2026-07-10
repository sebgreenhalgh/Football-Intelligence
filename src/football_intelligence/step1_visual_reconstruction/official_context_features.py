# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - feature tests can run without image libraries.
    cv2 = None
    np = None

from football_intelligence.step1_visual_reconstruction.colour_features import frame_file_by_sequence
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH,
    STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH,
    STEP1D1_OFFICIAL_CONTEXT_FEATURE_ROWS_PATH,
    read_json,
    write_json,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    bbox_from_item,
    safe_float,
)


TEAM_COLOUR_LIKE_BELIEFS = {"team_1_outfield_colour_like", "team_2_outfield_colour_like"}
NON_TEAM_COLOUR_LIKE_BELIEFS = {
    "ambiguous_outfield_colour",
    "non_outfield_context_colour",
    "other_distinct_colour_like",
    "dark_context_colour_like",
    "crop_unusable",
    "unknown_ambiguous_colour",
}
SOURCE_OFFICIAL_TYPES = {"official_candidate_source", "referee_candidate_source"}
SOURCE_UNKNOWN_CONTEXT_TYPES = {
    "staff_context_candidate_source",
    "unknown_candidate_source",
    "off_pitch_person_candidate",
    "unknown_person_candidate",
}
SOURCE_PLAYER_TYPES = {"player_candidate_source", "person_candidate"}
IMAGE_HEIGHT_HINT = 720.0
IMAGE_WIDTH_HINT = 2730.0


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def official_context_feature_id(row: dict[str, Any]) -> str:
    return f"step1d1_offctx_feature_{row.get('visible_person_base_id', row.get('detection_id', 'unknown'))}"


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def crop_from_image(image: Any | None, crop_bbox: dict[str, Any] | None) -> Any | None:
    if image is None or cv2 is None or np is None or not crop_bbox:
        return None
    x1, y1, x2, y2 = (int(round(safe_float(crop_bbox.get(key)))) for key in ["x1", "y1", "x2", "y2"])
    crop = image[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
    return crop if crop.size else None


def visual_colour_flags(crop: Any | None, colour_belief: str) -> dict[str, Any]:
    if crop is None or cv2 is None or np is None:
        return {
            "dark_or_black_like_visual_flag": colour_belief == "dark_context_colour_like",
            "bright_referee_colour_like_visual_flag": False,
            "red_or_pink_like_visual_flag": False,
            "yellow_or_orange_like_visual_flag": False,
            "mixed_colour_or_overlap_warning": False,
            "feature_quality": "low",
            "feature_warning": "source_frame_or_torso_crop_unavailable_for_visual_colour_flags",
        }
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    dark_fraction = float((v < 75).mean())
    bright_fraction = float(((v > 170) & (s > 65)).mean())
    red_pink_fraction = float((((h <= 15) | (h >= 160)) & (s > 55) & (v > 90)).mean())
    yellow_orange_fraction = float(((h >= 15) & (h <= 38) & (s > 55) & (v > 95)).mean())
    green_fraction = float(((h >= 35) & (h <= 95) & (s > 45) & (v >= 45)).mean())
    dark_flag = colour_belief == "dark_context_colour_like" or dark_fraction >= 0.38
    bright_flag = bright_fraction >= 0.18 and (red_pink_fraction >= 0.06 or yellow_orange_fraction >= 0.06)
    red_flag = red_pink_fraction >= 0.08
    yellow_flag = yellow_orange_fraction >= 0.08
    mixed_warning = green_fraction >= 0.62 or (red_flag and yellow_flag) or (bright_fraction >= 0.42 and dark_fraction >= 0.28)
    quality = "high" if crop.shape[0] >= 12 and crop.shape[1] >= 8 and not mixed_warning else "medium"
    if crop.shape[0] < 8 or crop.shape[1] < 5:
        quality = "low"
    warning = "mixed_colour_or_background_overlap_visual_warning" if mixed_warning else ""
    if quality == "low":
        warning = warning or "small_or_low_information_torso_crop"
    return {
        "dark_or_black_like_visual_flag": dark_flag,
        "bright_referee_colour_like_visual_flag": bright_flag,
        "red_or_pink_like_visual_flag": red_flag,
        "yellow_or_orange_like_visual_flag": yellow_flag,
        "mixed_colour_or_overlap_warning": mixed_warning,
        "feature_quality": quality,
        "feature_warning": warning,
    }


def image_space_flags(row: dict[str, Any]) -> dict[str, bool]:
    bbox = bbox_from_item(row) or {}
    x1 = safe_float(bbox.get("x1"), 0.0)
    x2 = safe_float(bbox.get("x2"), 0.0)
    y2 = safe_float(bbox.get("y2"), 0.0)
    lower_band = y2 >= IMAGE_HEIGHT_HINT * 0.78
    near_touchline = lower_band or x1 <= IMAGE_WIDTH_HINT * 0.045 or x2 >= IMAGE_WIDTH_HINT * 0.955
    return {
        "image_space_lower_frame_band_flag": lower_band,
        "image_space_near_touchline_context_flag": near_touchline,
    }


def feature_row(
    c2c_row: dict[str, Any],
    *,
    b4_row: dict[str, Any] | None = None,
    image: Any | None = None,
) -> dict[str, Any]:
    b4_row = b4_row or {}
    candidate_type = str(c2c_row.get("candidate_type") or b4_row.get("candidate_type") or "")
    original_role_source = str(c2c_row.get("original_role_source") or b4_row.get("original_role_source") or "")
    colour_belief = str(c2c_row.get("c2c_final_colour_belief", "unknown_ambiguous_colour"))
    crop_bbox = c2c_row.get("torso_crop_bbox") or b4_row.get("torso_crop_bbox")
    crop = crop_from_image(image, crop_bbox if isinstance(crop_bbox, dict) else None)
    visual_flags = visual_colour_flags(crop, colour_belief)
    image_flags = image_space_flags(c2c_row)
    source_official = candidate_type in SOURCE_OFFICIAL_TYPES or original_role_source in {"official", "referee"}
    source_unknown_context = candidate_type in SOURCE_UNKNOWN_CONTEXT_TYPES or original_role_source in {"staff", "unknown", "context"}
    source_player = candidate_type in SOURCE_PLAYER_TYPES or original_role_source == "player"
    offroi_or_recovery = str(c2c_row.get("roi_status", "")) == "outside_playing_roi" or candidate_type in {"off_pitch_person_candidate", "staff_context_candidate_source"}
    bad_detection = boolish(c2c_row.get("c2c_bad_detection_or_not_person")) or colour_belief == "crop_unusable"
    return {
        "official_context_feature_id": official_context_feature_id(c2c_row),
        "visible_person_base_id": c2c_row.get("visible_person_base_id", ""),
        "frame_id": c2c_row.get("frame_id", ""),
        "frame_sequence": int(safe_float(c2c_row.get("frame_sequence"), -1)),
        "timestamp_seconds": safe_float(c2c_row.get("timestamp_seconds")),
        "detection_id": c2c_row.get("detection_id", ""),
        "source_detection_id": c2c_row.get("source_detection_id", ""),
        "bbox": c2c_row.get("bbox", {}),
        "footpoint": c2c_row.get("footpoint", {}),
        "state": c2c_row.get("state", ""),
        "roi_status": c2c_row.get("roi_status", ""),
        "candidate_type": candidate_type,
        "original_role_source": original_role_source,
        "c2c_final_colour_belief": colour_belief,
        "c2c_colour_source": c2c_row.get("c2c_colour_source", ""),
        "c2c_human_reviewed": boolish(c2c_row.get("c2c_human_reviewed")),
        "c2c_context_or_offroi_human_team_override": boolish(c2c_row.get("c2c_context_or_offroi_human_team_override")),
        "crop_quality": c2c_row.get("crop_quality", ""),
        "crop_quality_reason": c2c_row.get("crop_quality_reason", ""),
        "torso_crop_bbox": crop_bbox,
        **image_flags,
        "source_official_candidate_flag": source_official,
        "source_unknown_context_candidate_flag": source_unknown_context,
        "source_player_candidate_flag": source_player,
        "offroi_or_recovery_context_flag": offroi_or_recovery,
        "team_colour_like_flag": colour_belief in TEAM_COLOUR_LIKE_BELIEFS,
        "non_team_colour_like_flag": colour_belief in NON_TEAM_COLOUR_LIKE_BELIEFS,
        **visual_flags,
        "bad_detection_candidate_flag": bad_detection,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def build_official_context_feature_payload(
    c2c_payload: dict[str, Any],
    *,
    b4_payload: dict[str, Any] | None = None,
    frame_lookup: dict[int, str] | None = None,
    frame_images: dict[int, Any | None] | None = None,
) -> dict[str, Any]:
    b4_payload = b4_payload or {"rows": []}
    b4_by_id = {str(row.get("visible_person_base_id", "")): row for row in b4_payload.get("rows", [])}
    frame_lookup = frame_lookup if frame_lookup is not None else frame_file_by_sequence()
    frame_images = frame_images or {}
    image_cache: dict[int, Any | None] = dict(frame_images)
    rows = []
    for c2c_row in c2c_payload.get("rows", []):
        seq = int(safe_float(c2c_row.get("frame_sequence"), -1))
        if seq not in image_cache:
            path = frame_lookup.get(seq, "")
            image_cache[seq] = cv2.imread(path) if cv2 is not None and path and Path(path).exists() else None
        visible_id = str(c2c_row.get("visible_person_base_id", ""))
        rows.append(feature_row(c2c_row, b4_row=b4_by_id.get(visible_id), image=image_cache.get(seq)))
    source_counts = Counter(str(row.get("candidate_type", "")) for row in rows)
    return {
        "artifact": "step1d1_official_context_feature_rows",
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
        "features_are_visual_qa_hints_only": True,
        "rows": rows,
        "summary": {
            "c2c_row_count": len(c2c_payload.get("rows", [])),
            "d1_feature_row_count": len(rows),
            "one_feature_row_per_c2c_row": len(c2c_payload.get("rows", [])) == len(rows),
            "source_candidate_type_counts": dict(sorted(source_counts.items())),
            "source_official_candidate_count": sum(1 for row in rows if row.get("source_official_candidate_flag") is True),
            "source_player_candidate_count": sum(1 for row in rows if row.get("source_player_candidate_flag") is True),
            "c2c_context_offroi_human_team_override_count": sum(1 for row in rows if row.get("c2c_context_or_offroi_human_team_override") is True),
            "bad_detection_candidate_count": sum(1 for row in rows if row.get("bad_detection_candidate_flag") is True),
        },
    }


def build_and_write_official_context_features() -> dict[str, Any]:
    c2c_payload = read_json(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH)
    b4_payload = read_json(STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH)
    payload = build_official_context_feature_payload(c2c_payload, b4_payload=b4_payload)
    write_json(STEP1D1_OFFICIAL_CONTEXT_FEATURE_ROWS_PATH, payload)
    return payload
