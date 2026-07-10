# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - tests can exercise missing-frame paths without cv2.
    cv2 = None
    np = None

from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH,
    STEP1C1_COLOUR_FEATURE_ROWS_PATH,
    load_person_states,
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


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.C1 colour feature extraction. Use the project venv interpreter.")
    return cv2


def frame_file_by_sequence(state_payload: dict[str, Any] | None = None) -> dict[int, str]:
    state_payload = state_payload or load_person_states()
    return {
        int(safe_float(frame.get("frame_sequence"), -1)): str(frame.get("frame_file", ""))
        for frame in state_payload.get("frames", [])
    }


def colour_feature_id(row: dict[str, Any]) -> str:
    return f"step1c1_colour_feature_{row.get('visible_person_base_id', row.get('detection_id', 'unknown'))}"


def torso_crop_bbox(row: dict[str, Any], image_shape: tuple[int, int] | None = None) -> dict[str, float] | None:
    bbox = bbox_from_item(row)
    if not bbox:
        return None
    width = bbox["x2"] - bbox["x1"]
    height = bbox["y2"] - bbox["y1"]
    crop = {
        "x1": bbox["x1"] + width * 0.22,
        "y1": bbox["y1"] + height * 0.18,
        "x2": bbox["x2"] - width * 0.22,
        "y2": bbox["y1"] + height * 0.66,
    }
    if image_shape is not None:
        img_h, img_w = image_shape
        crop["x1"] = min(max(0.0, crop["x1"]), float(img_w - 1))
        crop["x2"] = min(max(0.0, crop["x2"]), float(img_w))
        crop["y1"] = min(max(0.0, crop["y1"]), float(img_h - 1))
        crop["y2"] = min(max(0.0, crop["y2"]), float(img_h))
    if crop["x2"] <= crop["x1"] or crop["y2"] <= crop["y1"]:
        return None
    return {key: round(value, 3) for key, value in crop.items()}


def empty_feature(row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "frame_id": row.get("frame_id", ""),
        "frame_sequence": int(safe_float(row.get("frame_sequence"), -1)),
        "timestamp_seconds": safe_float(row.get("timestamp_seconds")),
        "visible_person_base_id": row.get("visible_person_base_id", ""),
        "detection_id": row.get("detection_id", ""),
        "source_detection_id": row.get("source_detection_id", ""),
        "colour_feature_id": colour_feature_id(row),
        "bbox": row.get("bbox", {}),
        "torso_crop_bbox": None,
        "crop_width": 0,
        "crop_height": 0,
        "crop_quality": "unusable",
        "crop_quality_reason": reason,
        "dominant_hsv_bins": [],
        "dominant_lab_bins": [],
        "median_hsv": [],
        "median_lab": [],
        "saturation_summary": {},
        "brightness_summary": {},
        "green_background_fraction": 1.0,
        "dark_fraction": 0.0,
        "light_fraction": 0.0,
        "blue_like_fraction": 0.0,
        "red_or_orange_like_fraction": 0.0,
        "white_like_fraction": 0.0,
        "black_or_dark_like_fraction": 0.0,
        "feature_extraction_warning": reason,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def top_bins(values: list[str], *, limit: int = 5) -> list[dict[str, Any]]:
    total = max(1, len(values))
    return [
        {"bin": key, "fraction": round(count / total, 4)}
        for key, count in Counter(values).most_common(limit)
    ]


def fraction(mask: Any) -> float:
    return round(float(mask.mean()) if mask.size else 0.0, 4)


def quality_from_stats(width: int, height: int, green_fraction: float, usable_pixel_count: int, blur_score: float) -> tuple[str, str]:
    if width < 4 or height < 6:
        return "unusable", "torso_crop_too_small"
    if usable_pixel_count < 20:
        return "unusable", "too_few_non_background_pixels"
    if green_fraction > 0.78:
        return "low", "mostly_green_background"
    if blur_score < 8.0:
        return "low", "low_texture_or_blurry_crop"
    if green_fraction > 0.55:
        return "low", "background_contaminated_crop"
    if width < 8 or height < 10:
        return "low", "small_torso_crop"
    if green_fraction < 0.35 and blur_score >= 20.0:
        return "high", "usable_low_background_torso_crop"
    return "medium", "usable_torso_crop"


def extract_colour_feature_from_image(row: dict[str, Any], image: Any | None) -> dict[str, Any]:
    if image is None:
        return empty_feature(row, "source_frame_missing")
    cv2_module = require_cv2()
    crop_bbox = torso_crop_bbox(row, image.shape[:2])
    if crop_bbox is None:
        return empty_feature(row, "invalid_bbox_or_torso_crop")
    x1, y1, x2, y2 = (int(round(crop_bbox[key])) for key in ["x1", "y1", "x2", "y2"])
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return empty_feature(row, "empty_torso_crop")
    crop_h, crop_w = crop.shape[:2]
    hsv = cv2_module.cvtColor(crop, cv2_module.COLOR_BGR2HSV)
    lab = cv2_module.cvtColor(crop, cv2_module.COLOR_BGR2LAB)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    green_mask = (h >= 35) & (h <= 95) & (s >= 45) & (v >= 45)
    green_fraction = fraction(green_mask)
    usable_mask = ~green_mask
    if int(usable_mask.sum()) < 20:
        usable_mask = np.ones_like(green_mask, dtype=bool)
    blur_score = float(cv2_module.Laplacian(cv2_module.cvtColor(crop, cv2_module.COLOR_BGR2GRAY), cv2_module.CV_64F).var())
    quality, quality_reason = quality_from_stats(crop_w, crop_h, green_fraction, int(usable_mask.sum()), blur_score)
    hsv_pixels = hsv[usable_mask]
    lab_pixels = lab[usable_mask]
    hsv_bins = [
        f"h{int(pixel[0] // 15):02d}_s{int(pixel[1] // 86)}_v{int(pixel[2] // 86)}"
        for pixel in hsv_pixels
    ]
    lab_bins = [
        f"l{int(pixel[0] // 43)}_a{int(pixel[1] // 43)}_b{int(pixel[2] // 43)}"
        for pixel in lab_pixels
    ]
    median_hsv = [round(float(value), 3) for value in np.median(hsv_pixels, axis=0).tolist()] if len(hsv_pixels) else []
    median_lab = [round(float(value), 3) for value in np.median(lab_pixels, axis=0).tolist()] if len(lab_pixels) else []
    feature_warning = "" if quality in {"high", "medium"} else quality_reason
    return {
        "frame_id": row.get("frame_id", ""),
        "frame_sequence": int(safe_float(row.get("frame_sequence"), -1)),
        "timestamp_seconds": safe_float(row.get("timestamp_seconds")),
        "visible_person_base_id": row.get("visible_person_base_id", ""),
        "detection_id": row.get("detection_id", ""),
        "source_detection_id": row.get("source_detection_id", ""),
        "colour_feature_id": colour_feature_id(row),
        "bbox": row.get("bbox", {}),
        "torso_crop_bbox": crop_bbox,
        "crop_width": int(crop_w),
        "crop_height": int(crop_h),
        "crop_quality": quality,
        "crop_quality_reason": quality_reason,
        "dominant_hsv_bins": top_bins(hsv_bins),
        "dominant_lab_bins": top_bins(lab_bins),
        "median_hsv": median_hsv,
        "median_lab": median_lab,
        "saturation_summary": {"median": round(float(np.median(s)), 3), "mean": round(float(np.mean(s)), 3)},
        "brightness_summary": {"median": round(float(np.median(v)), 3), "mean": round(float(np.mean(v)), 3), "blur_score": round(blur_score, 3)},
        "green_background_fraction": green_fraction,
        "dark_fraction": fraction(v < 65),
        "light_fraction": fraction(v > 185),
        "blue_like_fraction": fraction((h >= 90) & (h <= 130) & (s > 45)),
        "red_or_orange_like_fraction": fraction(((h <= 18) | (h >= 165)) & (s > 45)),
        "white_like_fraction": fraction((s < 45) & (v > 160)),
        "black_or_dark_like_fraction": fraction(v < 75),
        "feature_extraction_warning": feature_warning,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def build_colour_feature_payload(
    base_payload: dict[str, Any],
    *,
    frame_lookup: dict[int, str] | None = None,
    frame_images: dict[int, Any] | None = None,
) -> dict[str, Any]:
    frame_lookup = frame_lookup if frame_lookup is not None else frame_file_by_sequence()
    frame_images = frame_images or {}
    image_cache: dict[int, Any | None] = dict(frame_images)
    rows = []
    for row in base_payload.get("rows", []):
        seq = int(safe_float(row.get("frame_sequence"), -1))
        if seq not in image_cache:
            frame_file = frame_lookup.get(seq, "")
            if not frame_file or not Path(frame_file).exists():
                image_cache[seq] = None
            else:
                image_cache[seq] = require_cv2().imread(frame_file)
        rows.append(extract_colour_feature_from_image(row, image_cache.get(seq)))
    quality_counts = Counter(str(row.get("crop_quality", "")) for row in rows)
    warning_counts = Counter(str(row.get("feature_extraction_warning", "")) for row in rows if row.get("feature_extraction_warning"))
    return {
        "artifact": "step1c1_colour_feature_rows",
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
        "source_b4_visible_person_base_rows": len(base_payload.get("rows", [])),
        "rows": rows,
        "summary": {
            "b4_visible_person_base_rows": len(base_payload.get("rows", [])),
            "step1c1_colour_feature_rows": len(rows),
            "crop_quality_counts": dict(sorted(quality_counts.items())),
            "feature_warning_counts": dict(sorted(warning_counts.items())),
        },
    }


def build_and_write_colour_features() -> dict[str, Any]:
    base_payload = read_json(STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH)
    payload = build_colour_feature_payload(base_payload)
    write_json(STEP1C1_COLOUR_FEATURE_ROWS_PATH, payload)
    return payload
