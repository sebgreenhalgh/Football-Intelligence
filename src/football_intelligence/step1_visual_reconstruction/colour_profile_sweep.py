# ruff: noqa: E501

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - tests can exercise missing-frame paths without cv2.
    cv2 = None
    np = None

from football_intelligence.step1_visual_reconstruction.colour_cluster_diagnostics import evaluate_gold8_proxy_clusters
from football_intelligence.step1_visual_reconstruction.colour_features import (
    empty_feature,
    frame_file_by_sequence,
    fraction,
    quality_from_stats,
    top_bins,
    torso_crop_bbox,
)
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH,
    STEP1C1B_COLOUR_PROFILE_SWEEP_PATH,
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
from football_intelligence.step1_visual_reconstruction.team_colour_beliefs import (
    build_colour_prototypes,
    build_team_colour_belief_payloads,
    median_triplet,
    preliminary_colour_bucket,
)


PROFILE_NAMES = [
    "c1_current",
    "torso_wider",
    "torso_lower",
    "torso_upper_only",
    "central_body_excluding_grass",
    "adaptive_non_green_core",
    "bbox_inner_third",
    "high_resolution_near_only",
]

PROTOTYPE_STRATEGIES = [
    "c1_top_chromatic",
    "warm_light_secondary_sandbox",
    "blue_vs_nonblue_chromatic_sandbox",
]

CHROMATIC_BUCKETS = {
    "blue",
    "yellow_or_orange",
    "white_or_light_colour",
    "red_or_orange",
    "purple_or_magenta",
    "other",
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.C1b colour profile sweep. Use the project venv interpreter.")
    return cv2


def clamp_crop(crop: dict[str, float], image_shape: tuple[int, int] | None) -> dict[str, float] | None:
    if image_shape is not None:
        img_h, img_w = image_shape
        crop["x1"] = min(max(0.0, crop["x1"]), float(img_w - 1))
        crop["x2"] = min(max(0.0, crop["x2"]), float(img_w))
        crop["y1"] = min(max(0.0, crop["y1"]), float(img_h - 1))
        crop["y2"] = min(max(0.0, crop["y2"]), float(img_h))
    if crop["x2"] <= crop["x1"] or crop["y2"] <= crop["y1"]:
        return None
    return {key: round(value, 3) for key, value in crop.items()}


def proportional_crop_bbox(
    row: dict[str, Any],
    image_shape: tuple[int, int] | None,
    *,
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> dict[str, float] | None:
    bbox = bbox_from_item(row)
    if not bbox:
        return None
    width = bbox["x2"] - bbox["x1"]
    height = bbox["y2"] - bbox["y1"]
    return clamp_crop(
        {
            "x1": bbox["x1"] + width * left,
            "y1": bbox["y1"] + height * top,
            "x2": bbox["x2"] - width * right,
            "y2": bbox["y1"] + height * bottom,
        },
        image_shape,
    )


def adaptive_non_green_core_bbox(row: dict[str, Any], image: Any) -> dict[str, float] | None:
    cv2_module = require_cv2()
    base = proportional_crop_bbox(row, image.shape[:2], left=0.10, right=0.10, top=0.08, bottom=0.78)
    if base is None:
        return None
    x1, y1, x2, y2 = (int(round(base[key])) for key in ["x1", "y1", "x2", "y2"])
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return base
    hsv = cv2_module.cvtColor(crop, cv2_module.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    non_green = ~((h >= 35) & (h <= 95) & (s >= 45) & (v >= 45))
    if int(non_green.sum()) < 20:
        return base
    ys, xs = np.where(non_green)
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    pad_x = max(1, int(round((max_x - min_x + 1) * 0.12)))
    pad_y = max(1, int(round((max_y - min_y + 1) * 0.10)))
    return clamp_crop(
        {
            "x1": x1 + min_x - pad_x,
            "y1": y1 + min_y - pad_y,
            "x2": x1 + max_x + pad_x + 1,
            "y2": y1 + max_y + pad_y + 1,
        },
        image.shape[:2],
    )


def profile_crop_bbox(row: dict[str, Any], image_shape: tuple[int, int] | None = None, image: Any | None = None, profile_name: str = "c1_current") -> dict[str, float] | None:
    if profile_name == "c1_current":
        return torso_crop_bbox(row, image_shape)
    if profile_name == "torso_wider":
        return proportional_crop_bbox(row, image_shape, left=0.12, right=0.12, top=0.16, bottom=0.70)
    if profile_name == "torso_lower":
        return proportional_crop_bbox(row, image_shape, left=0.20, right=0.20, top=0.28, bottom=0.80)
    if profile_name == "torso_upper_only":
        return proportional_crop_bbox(row, image_shape, left=0.20, right=0.20, top=0.08, bottom=0.46)
    if profile_name == "central_body_excluding_grass":
        return proportional_crop_bbox(row, image_shape, left=0.25, right=0.25, top=0.12, bottom=0.62)
    if profile_name == "adaptive_non_green_core":
        if image is None:
            return proportional_crop_bbox(row, image_shape, left=0.14, right=0.14, top=0.12, bottom=0.70)
        return adaptive_non_green_core_bbox(row, image)
    if profile_name == "bbox_inner_third":
        return proportional_crop_bbox(row, image_shape, left=0.333, right=0.333, top=0.15, bottom=0.76)
    if profile_name == "high_resolution_near_only":
        return proportional_crop_bbox(row, image_shape, left=0.18, right=0.18, top=0.12, bottom=0.72)
    raise ValueError(f"Unknown Step1.C1b crop profile: {profile_name}")


def profile_empty_feature(row: dict[str, Any], reason: str, profile_name: str) -> dict[str, Any]:
    feature = empty_feature(row, reason)
    feature.update(
        {
            "artifact": "step1c1b_profile_colour_feature_row_sandbox",
            "colour_feature_id": f"step1c1b_{profile_name}_colour_feature_{row.get('visible_person_base_id', row.get('detection_id', 'unknown'))}",
            "profile_name": profile_name,
            "sandbox_only": True,
            "auto_promoted": False,
        }
    )
    return feature


def high_resolution_profile_skips(row: dict[str, Any]) -> bool:
    bbox = bbox_from_item(row)
    if not bbox:
        return True
    return (bbox["x2"] - bbox["x1"]) < 11.0 or (bbox["y2"] - bbox["y1"]) < 26.0


def extract_profile_feature_from_image(row: dict[str, Any], image: Any | None, profile_name: str) -> dict[str, Any]:
    if image is None:
        return profile_empty_feature(row, "source_frame_missing", profile_name)
    if profile_name == "high_resolution_near_only" and high_resolution_profile_skips(row):
        return profile_empty_feature(row, "high_resolution_near_only_small_bbox", profile_name)
    cv2_module = require_cv2()
    crop_bbox = profile_crop_bbox(row, image.shape[:2], image, profile_name)
    if crop_bbox is None:
        return profile_empty_feature(row, "invalid_bbox_or_profile_crop", profile_name)
    x1, y1, x2, y2 = (int(round(crop_bbox[key])) for key in ["x1", "y1", "x2", "y2"])
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return profile_empty_feature(row, "empty_profile_crop", profile_name)
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
    hsv_bins = [f"h{int(pixel[0] // 15):02d}_s{int(pixel[1] // 86)}_v{int(pixel[2] // 86)}" for pixel in hsv_pixels]
    lab_bins = [f"l{int(pixel[0] // 43)}_a{int(pixel[1] // 43)}_b{int(pixel[2] // 43)}" for pixel in lab_pixels]
    feature_warning = "" if quality in {"high", "medium"} else quality_reason
    return {
        "frame_id": row.get("frame_id", ""),
        "frame_sequence": int(safe_float(row.get("frame_sequence"), -1)),
        "timestamp_seconds": safe_float(row.get("timestamp_seconds")),
        "visible_person_base_id": row.get("visible_person_base_id", ""),
        "detection_id": row.get("detection_id", ""),
        "source_detection_id": row.get("source_detection_id", ""),
        "colour_feature_id": f"step1c1b_{profile_name}_colour_feature_{row.get('visible_person_base_id', row.get('detection_id', 'unknown'))}",
        "profile_name": profile_name,
        "bbox": row.get("bbox", {}),
        "torso_crop_bbox": crop_bbox,
        "crop_width": int(crop_w),
        "crop_height": int(crop_h),
        "crop_quality": quality,
        "crop_quality_reason": quality_reason,
        "dominant_hsv_bins": top_bins(hsv_bins),
        "dominant_lab_bins": top_bins(lab_bins),
        "median_hsv": [round(float(value), 3) for value in np.median(hsv_pixels, axis=0).tolist()] if len(hsv_pixels) else [],
        "median_lab": [round(float(value), 3) for value in np.median(lab_pixels, axis=0).tolist()] if len(lab_pixels) else [],
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
        "sandbox_only": True,
        "auto_promoted": False,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def build_profile_feature_payload(
    base_payload: dict[str, Any],
    profile_name: str,
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
        rows.append(extract_profile_feature_from_image(row, image_cache.get(seq), profile_name))
    quality_counts = Counter(str(row.get("crop_quality", "")) for row in rows)
    warning_counts = Counter(str(row.get("feature_extraction_warning", "")) for row in rows if row.get("feature_extraction_warning"))
    return {
        "artifact": "step1c1b_profile_colour_feature_rows_sandbox",
        "created_at": utc_iso(),
        "profile_name": profile_name,
        "sandbox_only": True,
        "auto_promoted": False,
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
        "rows": rows,
        "summary": {
            "profile_name": profile_name,
            "b4_visible_person_base_rows": len(base_payload.get("rows", [])),
            "feature_rows": len(rows),
            "crop_quality_counts": dict(sorted(quality_counts.items())),
            "feature_warning_counts": dict(sorted(warning_counts.items())),
        },
    }


def bucketed_features(feature_payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in feature_payload.get("rows", []):
        buckets[preliminary_colour_bucket(feature)].append(feature)
    return buckets


def top_chromatic_sources(buckets: dict[str, list[dict[str, Any]]]) -> list[str]:
    return [
        key
        for key, rows in sorted(
            ((key, rows) for key, rows in buckets.items() if key in CHROMATIC_BUCKETS and rows),
            key=lambda item: (-len(item[1]), item[0]),
        )
    ]


def prototype_entry(candidate: str, source: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "colour_cluster_candidate": candidate,
        "source_bucket": source,
        "row_count": len(rows),
        "median_hsv": median_triplet(rows, "median_hsv"),
        "median_lab": median_triplet(rows, "median_lab"),
    }


def build_strategy_prototypes(feature_payload: dict[str, Any], prototype_strategy: str) -> dict[str, Any]:
    if prototype_strategy == "c1_top_chromatic":
        payload = build_colour_prototypes(feature_payload)
        payload["artifact"] = "step1c1b_colour_prototypes_sandbox"
        payload["prototype_strategy"] = prototype_strategy
        payload["sandbox_only"] = True
        payload["auto_promoted"] = False
        return payload

    buckets = bucketed_features(feature_payload)
    top_sources = top_chromatic_sources(buckets)
    if prototype_strategy == "warm_light_secondary_sandbox":
        cluster_a_sources = ["blue"] if buckets.get("blue") else top_sources[:1]
        warm_sources = ["yellow_or_orange", "white_or_light_colour", "red_or_orange"]
        cluster_b_sources = [source for source in warm_sources if source not in cluster_a_sources and buckets.get(source)]
        if not cluster_b_sources:
            cluster_b_sources = [source for source in top_sources if source not in cluster_a_sources][:1]
    elif prototype_strategy == "blue_vs_nonblue_chromatic_sandbox":
        cluster_a_sources = ["blue"] if buckets.get("blue") else top_sources[:1]
        cluster_b_sources = [source for source in top_sources if source not in cluster_a_sources]
    else:
        raise ValueError(f"Unknown Step1.C1b prototype strategy: {prototype_strategy}")

    used_sources = set(cluster_a_sources) | set(cluster_b_sources)
    prototypes = []
    for source in cluster_a_sources:
        prototypes.append(prototype_entry("outfield_colour_cluster_a", source, buckets.get(source, [])))
    for source in cluster_b_sources:
        prototypes.append(prototype_entry("outfield_colour_cluster_b", source, buckets.get(source, [])))
    prototypes.append(prototype_entry("dark_context_colour_cluster", "dark_context_colour_cluster", buckets.get("dark_context_colour_cluster", [])))
    prototypes.append(
        prototype_entry(
            "unknown_ambiguous_colour",
            "unknown_ambiguous_colour",
            buckets.get("unknown_ambiguous_colour", []) + buckets.get("crop_unusable", []) + buckets.get("green_contaminated", []),
        )
    )
    other_rows: list[dict[str, Any]] = []
    for source in top_sources:
        if source not in used_sources:
            other_rows.extend(buckets.get(source, []))
    prototypes.append(prototype_entry("other_distinct_colour_cluster", "other_distinct_colour_cluster", other_rows))
    return {
        "artifact": "step1c1b_colour_prototypes_sandbox",
        "created_at": utc_iso(),
        "prototype_strategy": prototype_strategy,
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
        "prototype_sandbox_only": True,
        "sandbox_only": True,
        "auto_promoted": False,
        "safe_team_mapping_found": False,
        "mapped_team_colour_candidate": "unknown_mapping",
        "mapping_confidence": 0.0,
        "mapping_reason": "sandbox_strategy_for_visual_crop_audit_only",
        "prototypes": prototypes,
        "summary": {
            "prototype_count": len(prototypes),
            "prototype_strategy": prototype_strategy,
            "cluster_a_source_buckets": cluster_a_sources,
            "cluster_b_source_buckets": cluster_b_sources,
            "source_bucket_counts": dict(sorted((key, len(rows)) for key, rows in buckets.items())),
        },
    }


def stamp_sandbox_payload(payload: dict[str, Any], *, profile_name: str, prototype_strategy: str, artifact: str) -> dict[str, Any]:
    payload["artifact"] = artifact
    payload["profile_name"] = profile_name
    payload["prototype_strategy"] = prototype_strategy
    payload["sandbox_only"] = True
    payload["auto_promoted"] = False
    payload["production_ready"] = PRODUCTION_READY
    payload["visual_only_warning"] = VISUAL_ONLY_WARNING
    for row in payload.get("rows", []):
        row["profile_name"] = profile_name
        row["prototype_strategy"] = prototype_strategy
        row["sandbox_only"] = True
        row["auto_promoted"] = False
        row["production_ready"] = PRODUCTION_READY
        row["visual_only_warning"] = VISUAL_ONLY_WARNING
    return payload


def build_profile_sandbox_payloads(
    base_payload: dict[str, Any],
    profile_name: str,
    prototype_strategy: str,
    *,
    frame_lookup: dict[int, str] | None = None,
    frame_images: dict[int, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    feature_payload = build_profile_feature_payload(
        base_payload,
        profile_name,
        frame_lookup=frame_lookup,
        frame_images=frame_images,
    )
    prototypes_payload = build_strategy_prototypes(feature_payload, prototype_strategy)
    prototypes_payload["profile_name"] = profile_name
    prototypes_payload["sandbox_only"] = True
    prototypes_payload["auto_promoted"] = False
    _prototypes, belief_payload, unknown_payload = build_team_colour_belief_payloads(
        base_payload,
        feature_payload,
        prototypes_payload=prototypes_payload,
    )
    stamp_sandbox_payload(
        belief_payload,
        profile_name=profile_name,
        prototype_strategy=prototype_strategy,
        artifact="step1c1b_profile_belief_rows_sandbox",
    )
    stamp_sandbox_payload(
        unknown_payload,
        profile_name=profile_name,
        prototype_strategy=prototype_strategy,
        artifact="step1c1b_unknown_ambiguous_rows_sandbox",
    )
    return feature_payload, prototypes_payload, belief_payload, unknown_payload


def high_confidence_background_contaminated_rows(feature_payload: dict[str, Any], belief_payload: dict[str, Any]) -> int:
    features_by_base_id = {str(row.get("visible_person_base_id", "")): row for row in feature_payload.get("rows", [])}
    total = 0
    for row in belief_payload.get("rows", []):
        if row.get("team_colour_belief_state") != "high_confidence_visual_colour":
            continue
        feature = features_by_base_id.get(str(row.get("visible_person_base_id", "")), {})
        if (
            feature.get("crop_quality_reason") in {"background_contaminated_crop", "mostly_green_background"}
            or safe_float(feature.get("green_background_fraction")) > 0.55
        ):
            total += 1
    return total


def build_strategy_summary(
    feature_payload: dict[str, Any],
    belief_payload: dict[str, Any],
    unknown_payload: dict[str, Any],
    *,
    profile_name: str,
    prototype_strategy: str,
    labels_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    belief_summary = belief_payload.get("summary", {})
    high_bg = high_confidence_background_contaminated_rows(feature_payload, belief_payload)
    diagnostic_summary, confusion_rows = evaluate_gold8_proxy_clusters(
        belief_payload,
        labels_payload=labels_payload,
        profile_name=profile_name,
        prototype_strategy=prototype_strategy,
        high_confidence_background_contaminated_rows=high_bg,
    )
    summary = {
        "profile_name": profile_name,
        "prototype_strategy": prototype_strategy,
        "feature_rows": len(feature_payload.get("rows", [])),
        "belief_rows": len(belief_payload.get("rows", [])),
        "crop_quality_counts": feature_payload.get("summary", {}).get("crop_quality_counts", {}),
        "feature_warning_counts": feature_payload.get("summary", {}).get("feature_warning_counts", {}),
        "unknown_ambiguous_rows": belief_summary.get("unknown_ambiguous_colour_rows", len(unknown_payload.get("rows", []))),
        "crop_unusable_rows": belief_summary.get("crop_unusable_rows", 0),
        "high_confidence_visual_colour_rows": belief_summary.get("high_confidence_visual_colour_rows", 0),
        "medium_confidence_visual_colour_rows": belief_summary.get("medium_confidence_visual_colour_rows", 0),
        "review_required_rows": belief_summary.get("review_required_rows", 0),
        "cluster_counts": belief_summary.get("cluster_counts", {}),
        "mapped_team_colour_counts": belief_summary.get("mapped_team_colour_counts", {}),
        **diagnostic_summary,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "production_ready": PRODUCTION_READY,
        "auto_promoted": False,
    }
    return summary, confusion_rows


def build_colour_profile_sweep_payload(
    base_payload: dict[str, Any],
    *,
    profile_names: list[str] | None = None,
    prototype_strategies: list[str] | None = None,
    labels_payload: dict[str, Any] | None = None,
    frame_lookup: dict[int, str] | None = None,
    frame_images: dict[int, Any] | None = None,
) -> dict[str, Any]:
    profile_names = profile_names or list(PROFILE_NAMES)
    prototype_strategies = prototype_strategies or list(PROTOTYPE_STRATEGIES)
    profiles = []
    all_confusion_rows: list[dict[str, Any]] = []
    for profile_name in profile_names:
        feature_payload = build_profile_feature_payload(
            base_payload,
            profile_name,
            frame_lookup=frame_lookup,
            frame_images=frame_images,
        )
        strategy_summaries = []
        for prototype_strategy in prototype_strategies:
            prototypes_payload = build_strategy_prototypes(feature_payload, prototype_strategy)
            prototypes_payload["profile_name"] = profile_name
            _prototypes, belief_payload, unknown_payload = build_team_colour_belief_payloads(
                base_payload,
                feature_payload,
                prototypes_payload=prototypes_payload,
            )
            stamp_sandbox_payload(
                belief_payload,
                profile_name=profile_name,
                prototype_strategy=prototype_strategy,
                artifact="step1c1b_profile_belief_rows_sandbox",
            )
            stamp_sandbox_payload(
                unknown_payload,
                profile_name=profile_name,
                prototype_strategy=prototype_strategy,
                artifact="step1c1b_unknown_ambiguous_rows_sandbox",
            )
            summary, confusion_rows = build_strategy_summary(
                feature_payload,
                belief_payload,
                unknown_payload,
                profile_name=profile_name,
                prototype_strategy=prototype_strategy,
                labels_payload=labels_payload,
            )
            strategy_summaries.append(summary)
            all_confusion_rows.extend(confusion_rows)
        profiles.append(
            {
                "profile_name": profile_name,
                "feature_rows": len(feature_payload.get("rows", [])),
                "crop_quality_counts": feature_payload.get("summary", {}).get("crop_quality_counts", {}),
                "feature_warning_counts": feature_payload.get("summary", {}).get("feature_warning_counts", {}),
                "prototype_strategy_summaries": strategy_summaries,
            }
        )
    return {
        "artifact": "step1c1b_colour_profile_sweep",
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
        "sandbox_only": True,
        "canonical_c1_outputs_overwritten": False,
        "profile_names": profile_names,
        "prototype_strategies": prototype_strategies,
        "b4_visible_person_base_rows": len(base_payload.get("rows", [])),
        "profiles": profiles,
        "gold8_cluster_confusion_rows": all_confusion_rows,
        "summary": {
            "profiles_tested": len(profile_names),
            "prototype_strategies_tested": len(prototype_strategies),
            "profile_strategy_combinations": len(profile_names) * len(prototype_strategies),
            "b4_visible_person_base_rows": len(base_payload.get("rows", [])),
            "all_profiles_preserve_b4_row_count": all(
                summary.get("belief_rows", 0) == len(base_payload.get("rows", []))
                for profile in profiles
                for summary in profile.get("prototype_strategy_summaries", [])
            ),
            "gold8_cluster_confusion_rows": len(all_confusion_rows),
        },
    }


def build_and_write_colour_profile_sweep() -> dict[str, Any]:
    base_payload = read_json(STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH)
    payload = build_colour_profile_sweep_payload(base_payload)
    write_json(STEP1C1B_COLOUR_PROFILE_SWEEP_PATH, payload)
    return payload
