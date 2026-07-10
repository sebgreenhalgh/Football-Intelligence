# ruff: noqa: E501

from __future__ import annotations

import math
from typing import Any

from football_intelligence.step2_visual_continuity.schema import (
    clamp01,
    round_float,
    safe_float,
)


GOOD_CROP_QUALITY = {"excellent", "good", "usable", "high", "medium"}
LOW_CROP_QUALITY = {"low", "poor", "unusable", "missing", "unknown"}


def bbox_area(bbox: dict[str, Any] | None) -> float:
    if not bbox:
        return 0.0
    return max(0.0, safe_float(bbox.get("x2")) - safe_float(bbox.get("x1"))) * max(
        0.0,
        safe_float(bbox.get("y2")) - safe_float(bbox.get("y1")),
    )


def bbox_center(bbox: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if not bbox:
        return None, None
    return (
        (safe_float(bbox.get("x1")) + safe_float(bbox.get("x2"))) / 2.0,
        (safe_float(bbox.get("y1")) + safe_float(bbox.get("y2"))) / 2.0,
    )


def bbox_height(bbox: dict[str, Any] | None) -> float:
    if not bbox:
        return 0.0
    return max(0.0, safe_float(bbox.get("y2")) - safe_float(bbox.get("y1")))


def bbox_aspect_ratio(bbox: dict[str, Any] | None) -> float:
    if not bbox:
        return 0.0
    width = max(0.0, safe_float(bbox.get("x2")) - safe_float(bbox.get("x1")))
    height = bbox_height(bbox)
    return width / height if height > 0 else 0.0


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


def px_delta(left: tuple[float | None, float | None], right: tuple[float | None, float | None]) -> float | None:
    if left[0] is None or left[1] is None or right[0] is None or right[1] is None:
        return None
    return math.hypot(safe_float(right[0]) - safe_float(left[0]), safe_float(right[1]) - safe_float(left[1]))


def footpoint_xy(row: dict[str, Any]) -> tuple[float | None, float | None]:
    footpoint = row.get("footpoint", {})
    if not isinstance(footpoint, dict) or footpoint.get("x") is None or footpoint.get("y") is None:
        return None, None
    return safe_float(footpoint.get("x")), safe_float(footpoint.get("y"))


def role_state_compatibility(source: dict[str, Any], target: dict[str, Any]) -> tuple[float, list[str]]:
    source_role = str(source.get("step1f3_final_visual_role_state", ""))
    target_role = str(target.get("step1f3_final_visual_role_state", ""))
    source_group = str(source.get("step1f3_final_visual_role_group", ""))
    target_group = str(target.get("step1f3_final_visual_role_group", ""))
    reasons: list[str] = []
    if source_role == target_role and source_role:
        return 1.0, reasons
    if "unknown" in source_role or "unknown" in target_role:
        reasons.append("role_state_unknown_context")
        return 0.58, reasons
    if source_group and source_group == target_group:
        reasons.append("role_group_matches_role_state_differs")
        return 0.82, reasons
    if "bad_detection" in source_role or "bad_detection" in target_role:
        reasons.append("bad_detection_proxy_adjacent")
        return 0.2, reasons
    if ("official" in source_role or "official" in target_role) and source_role != target_role:
        reasons.append("official_context_role_state_mismatch")
        return 0.28, reasons
    if ("goalkeeper" in source_role or "goalkeeper" in target_role) and source_role != target_role:
        reasons.append("goalkeeper_context_role_state_mismatch")
        return 0.35, reasons
    reasons.append("role_state_mismatch")
    return 0.42, reasons


def team_context_compatibility(source: dict[str, Any], target: dict[str, Any]) -> tuple[float, list[str]]:
    source_context = str(source.get("step1f3_role_team_context", ""))
    target_context = str(target.get("step1f3_role_team_context", ""))
    source_colour = str(source.get("c2c_final_colour_belief", ""))
    target_colour = str(target.get("c2c_final_colour_belief", ""))
    reasons: list[str] = []
    ambiguous = any("unknown" in value or "ambiguous" in value for value in [source_context, target_context, source_colour, target_colour])
    if ambiguous:
        reasons.append("team_colour_or_context_ambiguous")
        return 0.58, reasons
    if source_context and source_context == target_context:
        return 1.0, reasons
    if source_colour and source_colour == target_colour:
        return 0.9, reasons
    if "team_1" in source_colour and "team_2" in target_colour:
        reasons.append("visual_team_context_mismatch")
        return 0.25, reasons
    if "team_2" in source_colour and "team_1" in target_colour:
        reasons.append("visual_team_context_mismatch")
        return 0.25, reasons
    return 0.68, reasons


def provenance_compatibility(source: dict[str, Any], target: dict[str, Any]) -> tuple[float, list[str]]:
    comparisons = [
        ("c2c_final_colour_belief", "c2c_context_differs"),
        ("d1c_final_official_context_belief", "d1c_context_differs"),
        ("e1c_final_goalkeeper_context_belief", "e1c_context_differs"),
    ]
    scores: list[float] = []
    reasons: list[str] = []
    for key, reason in comparisons:
        left = str(source.get(key, ""))
        right = str(target.get(key, ""))
        if not left or not right:
            scores.append(0.62)
            reasons.append(f"{key}_missing")
        elif left == right:
            scores.append(1.0)
        elif "unknown" in left or "unknown" in right or "ambiguous" in left or "ambiguous" in right:
            scores.append(0.58)
            reasons.append(f"{key}_ambiguous")
        else:
            scores.append(0.32)
            reasons.append(reason)
    return sum(scores) / max(1, len(scores)), reasons


def crop_quality_penalty(source: dict[str, Any], target: dict[str, Any]) -> tuple[float, list[str]]:
    qualities = [str(source.get("crop_quality", "unknown")).lower(), str(target.get("crop_quality", "unknown")).lower()]
    if all(value in GOOD_CROP_QUALITY for value in qualities):
        return 0.0, []
    if any(value in {"unusable", "missing"} for value in qualities):
        return 0.18, ["crop_quality_unusable_or_missing"]
    if any(value in LOW_CROP_QUALITY for value in qualities):
        return 0.1, ["low_crop_quality"]
    return 0.04, ["crop_quality_context_uncertain"]


def warning_penalty(source: dict[str, Any], target: dict[str, Any]) -> tuple[float, list[str]]:
    flags = [str(flag) for flag in source.get("step1f3_warning_flags", []) + target.get("step1f3_warning_flags", [])]
    reasons = []
    penalty = 0.0
    lowered = " ".join(flags).lower()
    if "ambiguous" in lowered or "merged" in lowered:
        reasons.append("merged_or_ambiguous_warning")
        penalty += 0.12
    if "official" in lowered:
        reasons.append("official_context_warning")
        penalty += 0.08
    if "goalkeeper" in lowered:
        reasons.append("goalkeeper_context_warning")
        penalty += 0.08
    if "bad_detection" in lowered:
        reasons.append("bad_detection_warning")
        penalty += 0.12
    if source.get("step1f3_review_required") is True or target.get("step1f3_review_required") is True:
        reasons.append("step1f3_review_required")
        penalty += 0.1
    return min(0.3, penalty), reasons


def context_availability_score(source: dict[str, Any], target: dict[str, Any]) -> tuple[float, list[str]]:
    reasons = []
    available = 0
    for row, label in [(source, "source"), (target, "target")]:
        if row.get("bbox"):
            available += 1
        else:
            reasons.append(f"{label}_bbox_missing")
        fp = row.get("footpoint", {})
        if isinstance(fp, dict) and fp.get("x") is not None and fp.get("y") is not None:
            available += 1
        else:
            reasons.append(f"{label}_footpoint_missing")
    return available / 4.0, reasons


def build_edge_feature_summary(source: dict[str, Any], target: dict[str, Any], frame_gap: int) -> dict[str, Any]:
    source_bbox = source.get("bbox") if isinstance(source.get("bbox"), dict) else None
    target_bbox = target.get("bbox") if isinstance(target.get("bbox"), dict) else None
    center_delta = px_delta(bbox_center(source_bbox), bbox_center(target_bbox))
    foot_delta = px_delta(footpoint_xy(source), footpoint_xy(target))
    source_area = bbox_area(source_bbox)
    target_area = bbox_area(target_bbox)
    area_ratio = min(source_area, target_area) / max(source_area, target_area) if source_area and target_area else 0.0
    source_aspect = bbox_aspect_ratio(source_bbox)
    target_aspect = bbox_aspect_ratio(target_bbox)
    aspect_change = abs(source_aspect - target_aspect) / max(source_aspect, target_aspect, 1e-6)
    avg_height = max(1.0, (bbox_height(source_bbox) + bbox_height(target_bbox)) / 2.0)
    center_score = 1.0 - min(1.0, safe_float(center_delta, avg_height * 3.0) / (avg_height * 2.0))
    foot_score = 1.0 - min(1.0, safe_float(foot_delta, avg_height * 3.0) / (avg_height * 2.2))
    iou = bbox_iou(source_bbox, target_bbox)
    aspect_score = 1.0 - min(1.0, aspect_change)
    role_score, role_reasons = role_state_compatibility(source, target)
    team_score, team_reasons = team_context_compatibility(source, target)
    provenance_score, provenance_reasons = provenance_compatibility(source, target)
    crop_penalty, crop_reasons = crop_quality_penalty(source, target)
    flag_penalty, flag_reasons = warning_penalty(source, target)
    context_score, context_reasons = context_availability_score(source, target)
    frame_gap_penalty = min(0.18, max(0, frame_gap - 1) * 0.035)
    edge_score = (
        0.22 * center_score
        + 0.2 * foot_score
        + 0.12 * iou
        + 0.1 * area_ratio
        + 0.06 * aspect_score
        + 0.12 * role_score
        + 0.08 * team_score
        + 0.06 * provenance_score
        + 0.04 * context_score
        - crop_penalty
        - flag_penalty
        - frame_gap_penalty
    )
    reasons = role_reasons + team_reasons + provenance_reasons + crop_reasons + flag_reasons + context_reasons
    if frame_gap > 1:
        reasons.append("frame_gap_penalty")
    margin_uncertainty = 1.0 - min(1.0, abs(edge_score - 0.5) / 0.5)
    uncertainty_score = clamp01(
        0.48 * margin_uncertainty
        + 0.18 * (1.0 - context_score)
        + crop_penalty
        + flag_penalty
        + (0.16 if any("mismatch" in reason for reason in reasons) else 0.0)
        + min(0.12, max(0, frame_gap - 1) * 0.03)
    )
    return {
        "bbox_center_delta_px": None if center_delta is None else round_float(center_delta, 3),
        "footpoint_delta_px": None if foot_delta is None else round_float(foot_delta, 3),
        "bbox_iou": round_float(iou, 4),
        "bbox_area_ratio": round_float(area_ratio, 4),
        "aspect_ratio_change": round_float(aspect_change, 4),
        "role_state_compatibility": round_float(role_score, 4),
        "visual_team_context_compatibility": round_float(team_score, 4),
        "step1_c2c_d1c_e1c_compatibility": round_float(provenance_score, 4),
        "crop_quality_penalty": round_float(crop_penalty, 4),
        "warning_conflict_flag_penalty": round_float(flag_penalty, 4),
        "frame_gap_penalty": round_float(frame_gap_penalty, 4),
        "source_target_context_availability": round_float(context_score, 4),
        "edge_score_sandbox": round_float(clamp01(edge_score), 4),
        "uncertainty_score": round_float(uncertainty_score, 4),
        "uncertainty_reasons": sorted(set(reasons)),
    }
