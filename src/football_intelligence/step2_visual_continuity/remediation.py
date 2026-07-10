# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import re
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

try:
    import cv2
    import numpy as np
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None
    np = None
    Image = None

from football_intelligence.paths import CLIP_ID, MATCH_ID, ensure_dir
from football_intelligence.step1_visual_reconstruction.colour_features import frame_file_by_sequence
from football_intelligence.step2_visual_continuity.edge_features import bbox_area, bbox_height, bbox_iou, px_delta
from football_intelligence.step2_visual_continuity.io import (
    STEP2M1_OUTPUT_DIR,
    STEP2M1_GROUP_ROWS_SANDBOX_PATH,
    STEP2M1_NODE_ROWS_PATH,
    STEP2M1R_ADAPTATION_SAFE_EDGE_ROWS_JSONL_GZ_PATH,
    STEP2M1R_ADAPTATION_SAFE_EDGE_ROWS_PATH,
    STEP2M1R_ADAPTATION_SAFE_EDGE_SAMPLE_PATH,
    STEP2M1R_ADAPTATION_SAFE_EDGE_SUMMARY_PATH,
    STEP2M1R_ADAPTATION_SAFE_GROUP_ROWS_PATH,
    STEP2M1R_ADAPTATION_SAFE_GROUP_SAMPLE_PATH,
    STEP2M1R_ADAPTATION_SAFE_GROUP_SUMMARY_PATH,
    STEP2M1R_ADAPTATION_SAFETY_MANIFEST_PATH,
    STEP2M1R_BURST_OVERLAY_ALIGNMENT_SUMMARY_PATH,
    STEP2M1R_BURST_OVERLAY_DEBUG_ROWS_PATH,
    STEP2M1R_BURST_OVERLAY_QA_DIR,
    STEP2M1R_GROUP_SPAN_REMEDIATION_SUMMARY_PATH,
    STEP2M1R_REVIEW_BURST_CLIPS_DIR,
    STEP2M1R_REVIEW_BURST_COMPARISON_STRIPS_DIR,
    STEP2M1R_REVIEW_BURST_RAW_STRIPS_DIR,
    STEP2M1R_REVIEW_BURST_STRIPS_DIR,
    STEP2M1R_REVIEW_CONTACT_SHEET_PATH,
    STEP2M1R_REVIEW_DECISION_SUMMARY_PATH,
    STEP2M1R_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP2M1R_REVIEWED_DECISIONS_PATH,
    STEP2M1R_REVIEW_UI_HTML_PATH,
    STEP2M1R_SOURCE_CONTEXT_IMAGES_DIR,
    STEP2M1R_SOURCE_CROP_IMAGES_DIR,
    STEP2M1R_TARGET_CONTEXT_IMAGES_DIR,
    STEP2M1R_TARGET_CROP_IMAGES_DIR,
    STEP2M1R_TARGETED_REVIEW_CANDIDATE_ROWS_PATH,
    STEP2M1R_TARGETED_REVIEW_CANDIDATE_SAMPLE_PATH,
    STEP2M1R_TARGETED_REVIEW_CANDIDATE_SUMMARY_PATH,
    compact_edge_payload_artifacts,
    read_human_corrected_edge_payload,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step2_visual_continuity.render_review import (
    clamp_bbox,
    crop,
    draw_context,
    padded_bbox,
    placeholder,
    rel_asset_path,
    resize_fit,
    safe_stem,
    write_image,
)
from football_intelligence.step2_visual_continuity.schema import (
    ACCEPT_DECISION,
    AUTO_ACCEPT_STATE,
    AUTO_REJECT_STATE,
    DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_FRAMES,
    DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_SECONDS,
    FORBIDDEN_APPROVAL_FLAGS,
    NEEDS_REVIEW_STATE,
    REJECT_DECISION,
    UNSURE_DECISION,
    VISUAL_ONLY_WARNING,
    assert_no_forbidden_keys,
    forbidden_keys_present,
    guardrail_stamp,
    rows_from_payload,
    safe_float,
    safe_int,
    utc_iso,
    visual_stamp,
)


TARGETED_REVIEW_BUCKET_LIMITS = {
    "safe_auto_accept_audit": 10,
    "high_uncertainty_low_margin": 10,
    "merged_or_ambiguous": 10,
    "role_state_mismatch": 10,
    "long_group_boundary_split": 10,
    "official_goalkeeper_sentinel": 5,
}
M1R_HARD_MAX_REVIEW_CARDS = 60
SAFE_AUTO_ACCEPT_AUDIT_REQUIRED_FOR_M2 = 10
BURST_MISSING_RATE_BLOCK_THRESHOLD = 0.05
CURRENT_BURST_OVERLAY_VERSION = "step2m1r_burst_overlay_alignment_v2"
SAFE_FINAL_EDGE_STATES = {
    "accepted_visual_continuity_edge",
    "bulk_accepted_visual_continuity_edge",
}
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step2.M1R review evidence rendering.")
    return cv2


def nodes_by_visible_id(node_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("visible_person_base_id", "")): row
        for row in rows_from_payload(node_payload)
        if row.get("visible_person_base_id")
    }


def nodes_by_frame_sequence(node_payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for row in rows_from_payload(node_payload):
        out.setdefault(safe_int(row.get("frame_sequence"), -1), []).append(row)
    return out


def edge_by_id(edge_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("continuity_edge_id", "")): row
        for row in rows_from_payload(edge_payload)
        if row.get("continuity_edge_id")
    }


def footpoint_xy(row: dict[str, Any]) -> tuple[float | None, float | None]:
    footpoint = row.get("footpoint", {})
    if not isinstance(footpoint, dict) or footpoint.get("x") is None or footpoint.get("y") is None:
        return None, None
    return safe_float(footpoint.get("x")), safe_float(footpoint.get("y"))


def bbox_center_xy(row: dict[str, Any]) -> tuple[float | None, float | None]:
    bbox = row.get("bbox", {}) if isinstance(row.get("bbox"), dict) else {}
    if not bbox:
        return None, None
    return (safe_float(bbox.get("x1")) + safe_float(bbox.get("x2"))) / 2.0, (safe_float(bbox.get("y1")) + safe_float(bbox.get("y2"))) / 2.0


def avg_bbox_height(source: dict[str, Any], target: dict[str, Any]) -> float:
    return max(1.0, (bbox_height(source.get("bbox", {})) + bbox_height(target.get("bbox", {}))) / 2.0)


def edge_geometry(edge: dict[str, Any], node_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = node_lookup.get(str(edge.get("source_visible_person_base_id", "")), {})
    target = node_lookup.get(str(edge.get("target_visible_person_base_id", "")), {})
    source_bbox = source.get("bbox", {}) if isinstance(source.get("bbox"), dict) else {}
    target_bbox = target.get("bbox", {}) if isinstance(target.get("bbox"), dict) else {}
    source_area = bbox_area(source_bbox)
    target_area = bbox_area(target_bbox)
    area_ratio = min(source_area, target_area) / max(source_area, target_area) if source_area and target_area else 0.0
    foot_delta = px_delta(footpoint_xy(source), footpoint_xy(target))
    center_delta = px_delta(bbox_center_xy(source), bbox_center_xy(target))
    height = avg_bbox_height(source, target)
    context_available = bool(source_bbox) and bool(target_bbox) and footpoint_xy(source)[0] is not None and footpoint_xy(target)[0] is not None
    return {
        "bbox_iou": round(bbox_iou(source_bbox, target_bbox), 4),
        "footpoint_delta_px": None if foot_delta is None else round(foot_delta, 3),
        "bbox_center_delta_px": None if center_delta is None else round(center_delta, 3),
        "bbox_area_ratio": round(area_ratio, 4),
        "avg_bbox_height_px": round(height, 3),
        "context_image_inputs_available": context_available,
    }


def strong_visual_overlap(geometry: dict[str, Any]) -> bool:
    foot_delta = geometry.get("footpoint_delta_px")
    height = max(1.0, safe_float(geometry.get("avg_bbox_height_px"), 1.0))
    return (
        safe_float(geometry.get("bbox_iou")) >= 0.28
        or (
            foot_delta is not None
            and safe_float(foot_delta) <= max(35.0, height * 0.65)
            and safe_float(geometry.get("bbox_area_ratio")) >= 0.62
            and geometry.get("context_image_inputs_available") is True
        )
    )


def human_accepted(edge: dict[str, Any]) -> bool:
    return str(edge.get("final_edge_state_sandbox", "")) in SAFE_FINAL_EDGE_STATES or str(edge.get("human_review_decision", "")) in {
        "accept_short_window_visual_continuity_edge",
        "bulk_accept_safe_bucket",
    }


def high_confidence(edge: dict[str, Any]) -> bool:
    return safe_float(edge.get("edge_score_sandbox")) >= 0.72 and safe_float(edge.get("uncertainty_score")) <= 0.32


def only_colour_ambiguity(edge: dict[str, Any]) -> bool:
    reasons = [str(reason) for reason in edge.get("uncertainty_reasons", [])]
    if not reasons:
        return False
    return all("team_colour" in reason or "visual_team" in reason or "c2c" in reason for reason in reasons)


def remediated_role_bucket(edge: dict[str, Any], geometry: dict[str, Any]) -> str:
    if strong_visual_overlap(geometry):
        return "role_state_mismatch_but_strong_visual_overlap"
    return "role_state_mismatch_with_person_swap_risk"


def remediate_edge_for_adaptation(edge: dict[str, Any], node_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bucket = str(edge.get("review_bucket", ""))
    proposed_state = str(edge.get("proposed_edge_state", ""))
    geometry = edge_geometry(edge, node_lookup)
    remediated_bucket = bucket
    remediated_state = proposed_state
    exclusion_reasons: list[str] = []
    active_learning = False
    positive = False

    if bucket == "merged_or_ambiguous":
        remediated_state = AUTO_REJECT_STATE if safe_float(edge.get("edge_score_sandbox")) <= 0.35 else NEEDS_REVIEW_STATE
        active_learning = True
        if not human_accepted(edge):
            exclusion_reasons.append("merged_or_ambiguous_requires_human_acceptance")
    elif bucket == "high_uncertainty_low_margin":
        remediated_state = NEEDS_REVIEW_STATE
        active_learning = True
        if not (human_accepted(edge) and high_confidence(edge)):
            exclusion_reasons.append("high_uncertainty_requires_human_accepted_high_confidence")
    elif bucket == "role_state_mismatch":
        remediated_bucket = remediated_role_bucket(edge, geometry)
        remediated_state = NEEDS_REVIEW_STATE
        active_learning = True
        if remediated_bucket == "role_state_mismatch_with_person_swap_risk" and not human_accepted(edge):
            exclusion_reasons.append("role_state_mismatch_person_swap_risk")
    elif bucket == "team_colour_ambiguity" and strong_visual_overlap(geometry) and only_colour_ambiguity(edge):
        if proposed_state == AUTO_REJECT_STATE:
            remediated_state = NEEDS_REVIEW_STATE
        remediated_bucket = "team_colour_ambiguity_strong_visual_continuity"

    if bucket == "bad_detection_proxy_adjacent" or "bad_detection" in " ".join(str(reason) for reason in edge.get("uncertainty_reasons", [])):
        exclusion_reasons.append("bad_detection_proxy_not_used_for_adaptation_positive")

    if human_accepted(edge):
        if bucket == "high_uncertainty_low_margin":
            positive = high_confidence(edge)
        elif bucket == "merged_or_ambiguous":
            positive = True
        elif remediated_bucket == "role_state_mismatch_with_person_swap_risk":
            positive = False
        elif "bad_detection_proxy_not_used_for_adaptation_positive" not in exclusion_reasons:
            positive = True
    elif proposed_state == AUTO_ACCEPT_STATE and remediated_state == AUTO_ACCEPT_STATE and high_confidence(edge):
        positive = bucket not in {"merged_or_ambiguous", "high_uncertainty_low_margin", "bad_detection_proxy_adjacent"}

    if not positive and not exclusion_reasons:
        exclusion_reasons.append("not_adaptation_safe_positive")

    row = {
        **edge,
        "step2m1r_review_bucket": remediated_bucket,
        "step2m1r_original_review_bucket": bucket,
        "step2m1r_remediated_proposed_edge_state": remediated_state,
        "step2m1r_adaptation_safe_positive": positive,
        "step2m1r_excluded_from_adaptation": not positive,
        "step2m1r_exclusion_reasons": sorted(set(exclusion_reasons)),
        "step2m1r_active_learning_candidate": active_learning,
        "step2m1r_bbox_iou": geometry["bbox_iou"],
        "step2m1r_footpoint_delta_px": geometry["footpoint_delta_px"],
        "step2m1r_bbox_area_ratio": geometry["bbox_area_ratio"],
        "step2m1r_context_image_inputs_available": geometry["context_image_inputs_available"],
    }
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def remediate_edges_for_adaptation(edge_payload: dict[str, Any], node_payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    node_lookup = nodes_by_visible_id(node_payload)
    safe_rows: list[dict[str, Any]] = []
    bucket_pools: dict[str, list[dict[str, Any]]] = {
        "safe_auto_accept_audit": [],
        "high_uncertainty_low_margin": [],
        "merged_or_ambiguous": [],
        "role_state_mismatch": [],
        "official_goalkeeper_sentinel": [],
    }
    excluded_edge_count = 0
    original_bucket_counts: Counter[str] = Counter()
    remediated_bucket_counts: Counter[str] = Counter()
    for edge in rows_from_payload(edge_payload):
        remediated = remediate_edge_for_adaptation(edge, node_lookup)
        original_bucket = str(edge.get("review_bucket", ""))
        remediated_bucket = str(remediated.get("step2m1r_review_bucket", ""))
        original_bucket_counts[original_bucket] += 1
        remediated_bucket_counts[remediated_bucket] += 1
        if remediated.get("step2m1r_adaptation_safe_positive") is True:
            safe_rows.append(remediated)
        else:
            excluded_edge_count += 1
        if original_bucket == "safe_auto_accept_candidate" or (
            edge.get("proposed_edge_state") == AUTO_ACCEPT_STATE and high_confidence(edge)
        ):
            bucket_pools["safe_auto_accept_audit"].append({**remediated, "step2m1r_target_review_bucket": "safe_auto_accept_audit"})
        if original_bucket == "high_uncertainty_low_margin":
            bucket_pools["high_uncertainty_low_margin"].append({**remediated, "step2m1r_target_review_bucket": "high_uncertainty_low_margin"})
        if original_bucket == "merged_or_ambiguous":
            bucket_pools["merged_or_ambiguous"].append({**remediated, "step2m1r_target_review_bucket": "merged_or_ambiguous"})
        if original_bucket == "role_state_mismatch" or remediated_bucket.startswith("role_state_mismatch_"):
            bucket_pools["role_state_mismatch"].append({**remediated, "step2m1r_target_review_bucket": "role_state_mismatch"})
        if original_bucket in {"official_context_warning", "goalkeeper_context_warning"}:
            bucket_pools["official_goalkeeper_sentinel"].append({**remediated, "step2m1r_target_review_bucket": "official_goalkeeper_sentinel"})
    state_counts = Counter(str(row.get("step2m1r_remediated_proposed_edge_state", "")) for row in safe_rows)
    payload = guardrail_stamp(
        {
            "artifact": "step2m1r_adaptation_safe_visual_continuity_edge_rows",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "source_artifact": edge_payload.get("artifact", "step2m1_human_corrected_visual_continuity_edge_rows"),
            "adaptation_safe_edge_count": len(safe_rows),
            "excluded_edge_count": excluded_edge_count,
            "rows": safe_rows,
            "summary": {
                "adaptation_safe_edge_count": len(safe_rows),
                "excluded_edge_count": excluded_edge_count,
                "original_review_bucket_counts": dict(sorted(original_bucket_counts.items())),
                "remediated_review_bucket_counts": dict(sorted(remediated_bucket_counts.items())),
                "remediated_state_counts": dict(sorted(state_counts.items())),
                "visual_only_warning": VISUAL_ONLY_WARNING,
            },
        }
    )
    assert_no_forbidden_keys(payload)
    return payload, safe_rows, bucket_pools


def remediated_group_id(member_ids: list[str], source_group_id: str, segment_index: int) -> str:
    seed = "|".join([source_group_id, str(segment_index), *sorted(member_ids)])
    return f"step2m1r_vcgroup_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:14]}"


def sorted_group_members(group: dict[str, Any], node_lookup: dict[str, dict[str, Any]]) -> list[str]:
    member_ids = [str(value) for value in group.get("member_visible_person_base_ids", [])]
    return sorted(member_ids, key=lambda value: (safe_int(node_lookup.get(value, {}).get("frame_sequence"), -1), value))


def split_members_by_span(
    member_ids: list[str],
    node_lookup: dict[str, dict[str, Any]],
    *,
    max_frames: int = DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_FRAMES,
    max_seconds: float = DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_SECONDS,
) -> list[list[str]]:
    segments: list[list[str]] = []
    current: list[str] = []
    current_min_frame = 0
    current_min_seconds: float | None = None
    for member_id in member_ids:
        node = node_lookup.get(member_id, {})
        frame = safe_int(node.get("frame_sequence"), -1)
        seconds = node.get("timestamp_seconds")
        seconds_value = safe_float(seconds) if seconds is not None else None
        if not current:
            current = [member_id]
            current_min_frame = frame
            current_min_seconds = seconds_value
            continue
        frame_span = frame - current_min_frame
        seconds_span = 0.0 if current_min_seconds is None or seconds_value is None else seconds_value - current_min_seconds
        if frame_span > max_frames or seconds_span > max_seconds:
            segments.append(current)
            current = [member_id]
            current_min_frame = frame
            current_min_seconds = seconds_value
        else:
            current.append(member_id)
    if current:
        segments.append(current)
    return [segment for segment in segments if len(segment) >= 2]


def group_spans(member_ids: list[str], node_lookup: dict[str, dict[str, Any]]) -> tuple[int, float | None]:
    frames = [safe_int(node_lookup.get(member_id, {}).get("frame_sequence"), -1) for member_id in member_ids]
    timestamps = [node_lookup.get(member_id, {}).get("timestamp_seconds") for member_id in member_ids]
    numeric = [safe_float(value) for value in timestamps if value is not None]
    frame_span = max(frames) - min(frames) if frames else 0
    seconds_span = round(max(numeric) - min(numeric), 4) if len(numeric) == len(member_ids) and numeric else None
    return frame_span, seconds_span


def build_adaptation_group_row(
    member_ids: list[str],
    source_group: dict[str, Any],
    segment_index: int,
    node_lookup: dict[str, dict[str, Any]],
    edge_lookup: dict[str, dict[str, Any]],
    *,
    max_frames: int = DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_FRAMES,
    max_seconds: float = DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_SECONDS,
) -> dict[str, Any] | None:
    if len(member_ids) < 2:
        return None
    member_set = set(member_ids)
    frames = [safe_int(node_lookup.get(member_id, {}).get("frame_sequence"), -1) for member_id in member_ids]
    timestamps = [node_lookup.get(member_id, {}).get("timestamp_seconds") for member_id in member_ids]
    numeric = [safe_float(value) for value in timestamps if value is not None]
    accepted_edge_ids: list[str] = []
    for edge_id in source_group.get("accepted_continuity_edge_ids", []):
        edge = edge_lookup.get(str(edge_id), {})
        if str(edge.get("source_visible_person_base_id", "")) in member_set and str(edge.get("target_visible_person_base_id", "")) in member_set:
            accepted_edge_ids.append(str(edge_id))
    if not accepted_edge_ids:
        return None
    frame_span, seconds_span = group_spans(member_ids, node_lookup)
    row = {
        "visual_continuity_group_id": remediated_group_id(member_ids, str(source_group.get("visual_continuity_group_id", "")), segment_index),
        "step2m1r_source_visual_continuity_group_id": source_group.get("visual_continuity_group_id", ""),
        "step2m1r_remediation_action": "split_from_long_group" if source_group.get("group_exceeds_span_cap") is True else "unchanged_adaptation_safe",
        "group_kind": "capped_visual_continuity_group_sandbox_adaptation_candidate",
        "member_visible_person_base_ids": member_ids,
        "member_node_ids": [node_lookup.get(member_id, {}).get("step2m1_visual_continuity_node_id", "") for member_id in member_ids],
        "member_frame_sequences": frames,
        "accepted_continuity_edge_ids": sorted(set(accepted_edge_ids)),
        "group_member_count": len(member_ids),
        "min_frame_sequence": min(frames),
        "max_frame_sequence": max(frames),
        "max_frame_span": frame_span,
        "max_seconds_span": seconds_span if seconds_span is not None else (round(max(numeric) - min(numeric), 4) if len(numeric) == len(member_ids) and numeric else None),
        "max_visual_continuity_group_span_frames": max_frames,
        "max_visual_continuity_group_span_seconds": max_seconds,
        "group_exceeds_span_cap": frame_span > max_frames or (seconds_span is not None and seconds_span > max_seconds),
        "group_requires_future_review": False,
        "group_not_safe_for_adaptation": False,
        "visual_continuity_group_is_identity": False,
        "visual_continuity_group_is_player_slot": False,
        "visual_continuity_group_is_goalkeeper_slot": False,
        "visual_continuity_group_is_metric": False,
        "sandbox_only": True,
    }
    if row["group_exceeds_span_cap"]:
        return None
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def remediate_groups_for_adaptation(
    group_payload: dict[str, Any],
    node_payload: dict[str, Any],
    edge_payload: dict[str, Any],
    *,
    max_frames: int = DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_FRAMES,
    max_seconds: float = DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_SECONDS,
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    node_lookup = nodes_by_visible_id(node_payload)
    edge_lookup = edge_by_id(edge_payload)
    original_rows = rows_from_payload(group_payload)
    safe_rows: list[dict[str, Any]] = []
    excluded_original_group_ids: list[str] = []
    long_group_edge_ids: set[str] = set()
    groups_over_cap_before = 0
    for group in original_rows:
        member_ids = sorted_group_members(group, node_lookup)
        frame_span, seconds_span = group_spans(member_ids, node_lookup)
        over_cap = (
            group.get("group_exceeds_span_cap") is True
            or group.get("group_not_safe_for_adaptation") is True
            or frame_span > max_frames
            or (seconds_span is not None and seconds_span > max_seconds)
        )
        if over_cap:
            groups_over_cap_before += 1
            excluded_original_group_ids.append(str(group.get("visual_continuity_group_id", "")))
            long_group_edge_ids.update(str(edge_id) for edge_id in group.get("accepted_continuity_edge_ids", []))
            for index, segment in enumerate(split_members_by_span(member_ids, node_lookup, max_frames=max_frames, max_seconds=max_seconds), start=1):
                row = build_adaptation_group_row(segment, {**group, "group_exceeds_span_cap": True}, index, node_lookup, edge_lookup, max_frames=max_frames, max_seconds=max_seconds)
                if row:
                    safe_rows.append(row)
        else:
            row = build_adaptation_group_row(member_ids, {**group, "group_exceeds_span_cap": False}, 1, node_lookup, edge_lookup, max_frames=max_frames, max_seconds=max_seconds)
            if row:
                safe_rows.append(row)
            else:
                excluded_original_group_ids.append(str(group.get("visual_continuity_group_id", "")))
    frame_spans = [safe_int(row.get("max_frame_span"), 0) for row in safe_rows]
    second_spans = [safe_float(row.get("max_seconds_span"), 0.0) for row in safe_rows if row.get("max_seconds_span") is not None]
    groups_over_cap_after = sum(1 for row in safe_rows if row.get("group_exceeds_span_cap") is True)
    group_payload_out = guardrail_stamp(
        {
            "artifact": "step2m1r_adaptation_safe_visual_continuity_group_rows",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "source_artifact": group_payload.get("artifact", "step2m1_visual_continuity_group_rows_sandbox"),
            "visual_continuity_group_rows": len(safe_rows),
            "rows": safe_rows,
            "summary": {
                "adaptation_safe_group_count": len(safe_rows),
                "groups_over_cap_after": groups_over_cap_after,
                "max_group_span_frames_after": max(frame_spans) if frame_spans else 0,
                "max_group_span_seconds_after": max(second_spans) if second_spans else 0.0,
                "groups_excluded_from_adaptation": len(excluded_original_group_ids),
                "visual_only_warning": VISUAL_ONLY_WARNING,
            },
        }
    )
    summary_payload = guardrail_stamp(
        {
            "artifact": "step2m1r_group_span_remediation_summary",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "original_group_count": len(original_rows),
            "remediated_group_count": len(safe_rows) + len(excluded_original_group_ids),
            "adaptation_safe_group_count": len(safe_rows),
            "groups_over_cap_before": groups_over_cap_before,
            "groups_over_cap_after": groups_over_cap_after,
            "groups_excluded_from_adaptation": len(excluded_original_group_ids),
            "excluded_visual_continuity_group_ids_sample": excluded_original_group_ids[:40],
            "max_group_span_frames_before": max((safe_int(row.get("max_frame_span"), 0) for row in original_rows), default=0),
            "max_group_span_seconds_before": max((safe_float(row.get("max_seconds_span"), 0.0) for row in original_rows if row.get("max_seconds_span") is not None), default=0.0),
            "max_group_span_frames_after": max(frame_spans) if frame_spans else 0,
            "max_group_span_seconds_after": max(second_spans) if second_spans else 0.0,
            "max_visual_continuity_group_span_frames": max_frames,
            "max_visual_continuity_group_span_seconds": max_seconds,
        }
    )
    assert_no_forbidden_keys(group_payload_out)
    assert_no_forbidden_keys(summary_payload)
    return group_payload_out, summary_payload, long_group_edge_ids


def write_row_artifact_with_summary_and_sample(path: Path, summary_path: Path, sample_path: Path, payload: dict[str, Any], sample_limit: int = 80) -> None:
    payload_rows = rows_from_payload(payload)
    summary_payload = guardrail_stamp(
        {
            **{key: value for key, value in payload.items() if key != "rows"},
            "rows": [],
            "summary": payload.get("summary", {}),
            "total_rows": len(payload_rows),
            "sample_rows": min(sample_limit, len(payload_rows)),
        }
    )
    sample_payload_value = guardrail_stamp(
        {
            **{key: value for key, value in payload.items() if key != "rows"},
            "rows": payload_rows[:sample_limit],
            "summary": payload.get("summary", {}),
            "total_rows": len(payload_rows),
            "sample_rows": min(sample_limit, len(payload_rows)),
        }
    )
    write_json(path, payload)
    write_json(summary_path, summary_payload)
    write_json(sample_path, sample_payload_value)


def candidate_sort_key(edge: dict[str, Any]) -> tuple[float, int, str]:
    return (
        -safe_float(edge.get("uncertainty_score")),
        safe_int(edge.get("source_frame_sequence"), -1),
        str(edge.get("continuity_edge_id", "")),
    )


def make_m1r_candidate(edge: dict[str, Any], index: int, target_bucket: str) -> dict[str, Any]:
    row = {
        **edge,
        "step2m1r_review_candidate_id": f"step2m1r_review_{index:03d}_{safe_stem(str(edge.get('continuity_edge_id', 'edge')))}",
        "step2m1_review_candidate_id": str(edge.get("continuity_edge_id", "")),
        "review_card_index": index,
        "review_bucket": target_bucket,
        "step2m1r_target_review_bucket": target_bucket,
        "safe_bulk_accept_eligible": False,
        "human_confirmed": False,
        "review_decision_rule": "Accept only if the highlighted person continues through the burst without a visible person swap. Reject if the edge jumps to a nearby person. Use unsure if the burst is too blurry/occluded.",
    }
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def build_targeted_review_candidates(
    bucket_pools: dict[str, list[dict[str, Any]]],
    long_group_edge_ids: set[str],
    edge_payload: dict[str, Any],
) -> dict[str, Any]:
    selected: dict[str, dict[str, Any]] = {}
    corrected_lookup = edge_by_id(edge_payload)
    long_group_rows = [
        {**corrected_lookup[edge_id], "step2m1r_target_review_bucket": "long_group_boundary_split"}
        for edge_id in sorted(long_group_edge_ids)
        if edge_id in corrected_lookup
    ]
    pools = {
        **bucket_pools,
        "long_group_boundary_split": long_group_rows,
    }
    order = [
        "safe_auto_accept_audit",
        "high_uncertainty_low_margin",
        "merged_or_ambiguous",
        "role_state_mismatch",
        "long_group_boundary_split",
        "official_goalkeeper_sentinel",
    ]
    for bucket in order:
        limit = TARGETED_REVIEW_BUCKET_LIMITS[bucket]
        for edge in sorted(pools.get(bucket, []), key=candidate_sort_key):
            edge_id = str(edge.get("continuity_edge_id", ""))
            if edge_id in selected:
                continue
            selected[edge_id] = {**edge, "step2m1r_target_review_bucket": bucket}
            if sum(1 for row in selected.values() if row.get("step2m1r_target_review_bucket") == bucket) >= limit:
                break
            if len(selected) >= M1R_HARD_MAX_REVIEW_CARDS:
                break
        if len(selected) >= M1R_HARD_MAX_REVIEW_CARDS:
            break
    candidate_rows = [
        make_m1r_candidate(edge, index + 1, str(edge.get("step2m1r_target_review_bucket", "")))
        for index, edge in enumerate(selected.values())
    ][:M1R_HARD_MAX_REVIEW_CARDS]
    bucket_counts = Counter(str(row.get("review_bucket", "")) for row in candidate_rows)
    payload = guardrail_stamp(
        {
            "artifact": "step2m1r_targeted_review_candidate_rows",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "review_flow": "Step2.M1R targeted second review; maximum 10 minutes; A/X/U keyboard decisions only.",
            "keyboard_shortcuts": {
                "A": "accept_short_window_visual_continuity_edge",
                "X": "reject_edge",
                "U": "unsure_needs_later_review",
            },
            "review_decision_rule": "Accept only if the highlighted person continues through the burst without a visible person swap. Reject if the edge jumps to a nearby person. Use unsure if the burst is too blurry/occluded.",
            "targeted_review_card_hard_max": M1R_HARD_MAX_REVIEW_CARDS,
            "targeted_review_completed": False,
            "safe_auto_accept_audit_rows": bucket_counts.get("safe_auto_accept_audit", 0),
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "rows": candidate_rows,
            "summary": {
                "targeted_review_candidate_rows": len(candidate_rows),
                "safe_auto_accept_audit_rows": bucket_counts.get("safe_auto_accept_audit", 0),
                "bucket_counts": dict(sorted(bucket_counts.items())),
                "visual_only_warning": VISUAL_ONLY_WARNING,
            },
        }
    )
    assert_no_forbidden_keys(payload)
    return payload


def m1r_asset_paths(candidate: dict[str, Any]) -> dict[str, Path]:
    stem = safe_stem(str(candidate.get("step2m1r_review_candidate_id", candidate.get("continuity_edge_id", ""))))
    return {
        "source_context_image": STEP2M1R_SOURCE_CONTEXT_IMAGES_DIR / f"{stem}_source_context.jpg",
        "target_context_image": STEP2M1R_TARGET_CONTEXT_IMAGES_DIR / f"{stem}_target_context.jpg",
        "source_crop_image": STEP2M1R_SOURCE_CROP_IMAGES_DIR / f"{stem}_source_crop.jpg",
        "target_crop_image": STEP2M1R_TARGET_CROP_IMAGES_DIR / f"{stem}_target_crop.jpg",
        "burst_clip": STEP2M1R_REVIEW_BURST_CLIPS_DIR / f"{stem}_burst.gif",
        "burst_strip": STEP2M1R_REVIEW_BURST_STRIPS_DIR / f"{stem}_strip.jpg",
    }


def frame_sequences_for_burst(candidate: dict[str, Any], frame_lookup: dict[int, str], *, padding: int = 1) -> list[int]:
    source = safe_int(candidate.get("source_frame_sequence"), -1)
    target = safe_int(candidate.get("target_frame_sequence"), -1)
    start = min(source, target)
    end = max(source, target)
    available = set(frame_lookup)
    frames = list(range(start, end + 1))
    if start - padding in available:
        frames.insert(0, start - padding)
    if end + padding in available:
        frames.append(end + padding)
    return frames


def burst_frame_transform_metadata(
    *,
    original_frame_width: int,
    original_frame_height: int,
    rendered_frame_width: int,
    rendered_frame_height: int,
    display_width: int | None = None,
    display_height: int | None = None,
    scale_x: float | None = None,
    scale_y: float | None = None,
    pad_x: float = 0.0,
    pad_y: float = 0.0,
    coordinate_space_source: str = "original_frame",
    bbox_transform_applied: bool = False,
) -> dict[str, Any]:
    scale_x_value = scale_x if scale_x is not None else rendered_frame_width / max(1, original_frame_width)
    scale_y_value = scale_y if scale_y is not None else rendered_frame_height / max(1, original_frame_height)
    metadata = {
        "original_frame_width": original_frame_width,
        "original_frame_height": original_frame_height,
        "rendered_frame_width": rendered_frame_width,
        "rendered_frame_height": rendered_frame_height,
        "display_width": rendered_frame_width if display_width is None else display_width,
        "display_height": rendered_frame_height if display_height is None else display_height,
        "scale_x": round(scale_x_value, 8),
        "scale_y": round(scale_y_value, 8),
        "pad_x": round(pad_x, 4),
        "pad_y": round(pad_y, 4),
        "coordinate_space_source": coordinate_space_source,
        "bbox_transform_applied": bbox_transform_applied,
    }
    assert_no_forbidden_keys(metadata)
    return metadata


def render_frame_with_transform(image: Any, *, max_width: int = 760, max_height: int = 430) -> tuple[Any, dict[str, Any]]:
    cv2_module = require_cv2()
    original_height, original_width = image.shape[:2]
    scale = min(max_width / max(1, original_width), max_height / max(1, original_height))
    rendered_width = max(1, int(original_width * scale))
    rendered_height = max(1, int(original_height * scale))
    rendered = cv2_module.resize(image, (rendered_width, rendered_height), interpolation=cv2_module.INTER_AREA)
    metadata = burst_frame_transform_metadata(
        original_frame_width=original_width,
        original_frame_height=original_height,
        rendered_frame_width=rendered_width,
        rendered_frame_height=rendered_height,
        coordinate_space_source="original_frame",
    )
    return rendered, metadata


def transform_bbox_to_rendered(raw_bbox: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    transformed = {
        "x1": safe_float(raw_bbox.get("x1")) * safe_float(metadata.get("scale_x")) + safe_float(metadata.get("pad_x")),
        "y1": safe_float(raw_bbox.get("y1")) * safe_float(metadata.get("scale_y")) + safe_float(metadata.get("pad_y")),
        "x2": safe_float(raw_bbox.get("x2")) * safe_float(metadata.get("scale_x")) + safe_float(metadata.get("pad_x")),
        "y2": safe_float(raw_bbox.get("y2")) * safe_float(metadata.get("scale_y")) + safe_float(metadata.get("pad_y")),
    }
    width = safe_int(metadata.get("rendered_frame_width"), 0)
    height = safe_int(metadata.get("rendered_frame_height"), 0)
    clipped = {
        "x1": max(0.0, min(float(width), transformed["x1"])),
        "y1": max(0.0, min(float(height), transformed["y1"])),
        "x2": max(0.0, min(float(width), transformed["x2"])),
        "y2": max(0.0, min(float(height), transformed["y2"])),
    }
    was_clipped = any(abs(transformed[key] - clipped[key]) > 1e-6 for key in transformed)
    transformed_area = max(0.0, clipped["x2"] - clipped["x1"]) * max(0.0, clipped["y2"] - clipped["y1"])
    result = {
        "raw_bbox": {key: round(safe_float(raw_bbox.get(key)), 4) for key in ["x1", "y1", "x2", "y2"]},
        "transformed_bbox_unclipped": {key: round(value, 4) for key, value in transformed.items()},
        "transformed_bbox": {key: round(value, 4) for key, value in clipped.items()},
        "clipped": was_clipped,
        "transformed_bbox_area": round(transformed_area, 4),
        "bbox_transform_applied": True,
    }
    assert_no_forbidden_keys(result)
    return result


def transformed_point(point: dict[str, Any], metadata: dict[str, Any]) -> dict[str, float] | None:
    if not isinstance(point, dict) or point.get("x") is None or point.get("y") is None:
        return None
    return {
        "x": round(safe_float(point.get("x")) * safe_float(metadata.get("scale_x")) + safe_float(metadata.get("pad_x")), 4),
        "y": round(safe_float(point.get("y")) * safe_float(metadata.get("scale_y")) + safe_float(metadata.get("pad_y")), 4),
    }


def transformed_bbox_center(bbox: dict[str, Any]) -> dict[str, float]:
    return {
        "x": round((safe_float(bbox.get("x1")) + safe_float(bbox.get("x2"))) / 2.0, 4),
        "y": round((safe_float(bbox.get("y1")) + safe_float(bbox.get("y2"))) / 2.0, 4),
    }


def transformed_bbox_bottom_center(bbox: dict[str, Any]) -> dict[str, float]:
    return {
        "x": round((safe_float(bbox.get("x1")) + safe_float(bbox.get("x2"))) / 2.0, 4),
        "y": round(safe_float(bbox.get("y2")), 4),
    }


def footpoint_plausibility(node: dict[str, Any], transformed_bbox: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    point = transformed_point(node.get("footpoint", {}), metadata)
    if point is None:
        return {"footpoint_available": False, "footpoint_plausible": True, "footpoint_delta_px": None}
    bottom = transformed_bbox_bottom_center(transformed_bbox)
    delta = math.hypot(bottom["x"] - point["x"], bottom["y"] - point["y"])
    box_height = max(1.0, safe_float(transformed_bbox.get("y2")) - safe_float(transformed_bbox.get("y1")))
    return {
        "footpoint_available": True,
        "transformed_footpoint": point,
        "transformed_bbox_bottom_center": bottom,
        "footpoint_delta_px": round(delta, 4),
        "footpoint_plausible": delta <= max(12.0, box_height * 0.35),
    }


def draw_overlay_box(
    image: Any,
    *,
    raw_bbox: dict[str, Any],
    metadata: dict[str, Any],
    label: str,
    colour: tuple[int, int, int],
    node: dict[str, Any],
    overlay_role: str,
    candidate: dict[str, Any],
    frame_sequence: int,
    interpolated_visual_aid: bool = False,
    interpolation_only_not_detection: bool = False,
    thickness: int = 3,
) -> dict[str, Any]:
    cv2_module = require_cv2()
    transform = transform_bbox_to_rendered(raw_bbox, metadata)
    box = clamp_bbox(transform["transformed_bbox"], image.shape[:2])
    if box:
        cv2_module.rectangle(image, (box["x1"], box["y1"]), (box["x2"], box["y2"]), colour, thickness, cv2_module.LINE_AA)
        cv2_module.putText(image, label[:46], (box["x1"], max(18, box["y1"] - 6)), cv2_module.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2_module.LINE_AA)
    plausibility = footpoint_plausibility(node, transform["transformed_bbox"], metadata)
    debug_row = {
        "step2m1r_review_candidate_id": candidate.get("step2m1r_review_candidate_id", ""),
        "continuity_edge_id": candidate.get("continuity_edge_id", ""),
        "frame_sequence": frame_sequence,
        "source_frame_sequence": candidate.get("source_frame_sequence", -1),
        "target_frame_sequence": candidate.get("target_frame_sequence", -1),
        "overlay_role": overlay_role,
        "drawn_label": label,
        "actual_visible_person_base_id": node.get("visible_person_base_id", ""),
        "interpolated_visual_aid": interpolated_visual_aid,
        "interpolation_only_not_detection": interpolation_only_not_detection,
        "coordinate_metadata": {**metadata, "bbox_transform_applied": True},
        **transform,
        **plausibility,
    }
    debug_row["overlay_debug_valid"] = overlay_debug_row_valid(debug_row)
    visual_stamp(debug_row)
    assert_no_forbidden_keys(debug_row)
    return debug_row


def overlay_debug_row_valid(row: dict[str, Any]) -> bool:
    bbox = row.get("transformed_bbox", {})
    metadata = row.get("coordinate_metadata", {})
    if not isinstance(bbox, dict) or not isinstance(metadata, dict):
        return False
    if row.get("bbox_transform_applied") is not True or metadata.get("bbox_transform_applied") is not True:
        return False
    if safe_float(row.get("transformed_bbox_area")) <= 0:
        return False
    if row.get("footpoint_plausible") is not True:
        return False
    width = safe_float(metadata.get("rendered_frame_width"))
    height = safe_float(metadata.get("rendered_frame_height"))
    inside = (
        0 <= safe_float(bbox.get("x1")) <= width
        and 0 <= safe_float(bbox.get("x2")) <= width
        and 0 <= safe_float(bbox.get("y1")) <= height
        and 0 <= safe_float(bbox.get("y2")) <= height
    )
    if not inside:
        return False
    role = str(row.get("overlay_role", ""))
    frame = safe_int(row.get("frame_sequence"), -1)
    if role == "source" and frame != safe_int(row.get("source_frame_sequence"), -2):
        return False
    if role == "target" and frame != safe_int(row.get("target_frame_sequence"), -2):
        return False
    if role in {"intermediate_bridge_candidate", "nearby_candidate"}:
        has_detection_id = bool(row.get("actual_visible_person_base_id"))
        if not has_detection_id and row.get("interpolated_visual_aid") is not True:
            return False
    if row.get("interpolation_only_not_detection") is True and row.get("interpolated_visual_aid") is not True:
        return False
    return True


def draw_labelled_box(image: Any, bbox: dict[str, Any], label: str, colour: tuple[int, int, int], thickness: int = 3) -> None:
    cv2_module = require_cv2()
    box = clamp_bbox(bbox, image.shape[:2])
    if not box:
        return
    cv2_module.rectangle(image, (box["x1"], box["y1"]), (box["x2"], box["y2"]), colour, thickness, cv2_module.LINE_AA)
    cv2_module.putText(image, label[:36], (box["x1"], max(18, box["y1"] - 6)), cv2_module.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2_module.LINE_AA)


def interpolated_point(source: dict[str, Any], target: dict[str, Any], frame: int, source_frame: int, target_frame: int) -> tuple[float | None, float | None]:
    source_xy = footpoint_xy(source)
    target_xy = footpoint_xy(target)
    if source_xy[0] is None or target_xy[0] is None:
        return None, None
    span = max(1, target_frame - source_frame)
    weight = max(0.0, min(1.0, (frame - source_frame) / span))
    return safe_float(source_xy[0]) * (1.0 - weight) + safe_float(target_xy[0]) * weight, safe_float(source_xy[1]) * (1.0 - weight) + safe_float(target_xy[1]) * weight


def nearest_nodes_for_burst(
    frame_nodes: list[dict[str, Any]],
    source: dict[str, Any],
    target: dict[str, Any],
    frame: int,
    source_frame: int,
    target_frame: int,
) -> tuple[list[dict[str, Any]], bool]:
    point = interpolated_point(source, target, frame, source_frame, target_frame)
    if point[0] is None:
        return frame_nodes[:8], False
    scored = []
    for node in frame_nodes:
        xy = footpoint_xy(node)
        if xy[0] is None:
            continue
        scored.append((math.hypot(safe_float(xy[0]) - safe_float(point[0]), safe_float(xy[1]) - safe_float(point[1])), node))
    scored.sort(key=lambda item: item[0])
    if not scored:
        return frame_nodes[:8], False
    threshold = max(80.0, avg_bbox_height(source, target) * 1.4)
    if scored[0][0] <= threshold:
        return [scored[0][1]], True
    return [node for _score, node in scored[:8]], False


def horizontal_strip(frames: list[Any]) -> Any:
    if not frames:
        return placeholder((760, 430), "missing strip")
    cv2_module = require_cv2()
    min_height = min(frame.shape[0] for frame in frames)
    resized = []
    for frame in frames:
        if frame.shape[0] == min_height:
            resized.append(frame)
        else:
            scale = min_height / max(1, frame.shape[0])
            resized.append(cv2_module.resize(frame, (max(1, int(frame.shape[1] * scale)), min_height), interpolation=cv2_module.INTER_AREA))
    return np.hstack(resized)


def render_burst_evidence_for_candidate(
    candidate: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    frame_nodes: dict[int, list[dict[str, Any]]],
    frame_lookup: dict[int, str],
) -> dict[str, Any]:
    if cv2 is None or np is None or Image is None:
        return {
            "burst_clip_path": "",
            "burst_strip_path": "",
            "burst_frame_sequences": [],
            "intermediate_candidate_ids": [],
            "burst_evidence_available": False,
            "burst_evidence_missing_reason": "image_dependencies_unavailable",
        }
    cv2_module = require_cv2()
    source_frame = safe_int(candidate.get("source_frame_sequence"), -1)
    target_frame = safe_int(candidate.get("target_frame_sequence"), -1)
    source_node = nodes_by_id.get(str(candidate.get("source_visible_person_base_id", "")), {})
    target_node = nodes_by_id.get(str(candidate.get("target_visible_person_base_id", "")), {})
    burst_frames = frame_sequences_for_burst(candidate, frame_lookup)
    paths = m1r_asset_paths(candidate)
    for directory in [STEP2M1R_REVIEW_BURST_CLIPS_DIR, STEP2M1R_REVIEW_BURST_STRIPS_DIR]:
        ensure_dir(directory)
    rendered_frames: list[Any] = []
    raw_frames: list[Any] = []
    comparison_frames: list[Any] = []
    pil_frames: list[Any] = []
    intermediate_candidate_ids: list[str] = []
    frame_coordinate_metadata: list[dict[str, Any]] = []
    overlay_debug_rows: list[dict[str, Any]] = []
    missing_core_frame = False
    for frame in burst_frames:
        image_path = frame_lookup.get(frame, "")
        source_image = cv2_module.imread(image_path) if image_path and Path(image_path).exists() else None
        if source_image is None:
            if source_frame <= frame <= target_frame:
                missing_core_frame = True
            image = placeholder((760, 430), f"missing frame {frame}")
            metadata = burst_frame_transform_metadata(
                original_frame_width=image.shape[1],
                original_frame_height=image.shape[0],
                rendered_frame_width=image.shape[1],
                rendered_frame_height=image.shape[0],
                coordinate_space_source="rendered_frame",
            )
        else:
            image, metadata = render_frame_with_transform(source_image, max_width=760, max_height=430)
        raw_image = image.copy()
        drawn_rows: list[dict[str, Any]] = []

        def draw_node(node: dict[str, Any], role: str, label_prefix: str, colour: tuple[int, int, int], thickness: int = 3) -> None:
            bbox = node.get("bbox", {}) if isinstance(node.get("bbox"), dict) else {}
            if not bbox:
                return
            row = draw_overlay_box(
                image,
                raw_bbox=bbox,
                metadata=metadata,
                label=f"{label_prefix} {node.get('visible_person_base_id', '')}",
                colour=colour,
                node=node,
                overlay_role=role,
                candidate=candidate,
                frame_sequence=frame,
                thickness=thickness,
            )
            drawn_rows.append(row)

        if frame == source_frame:
            if safe_int(source_node.get("frame_sequence"), -999) == frame:
                draw_node(source_node, "source", "source", (0, 215, 255), 3)
        elif frame == target_frame:
            if safe_int(target_node.get("frame_sequence"), -999) == frame:
                draw_node(target_node, "target", "target", (115, 245, 145), 3)
        elif min(source_frame, target_frame) < frame < max(source_frame, target_frame):
            near_nodes, likely_bridge = nearest_nodes_for_burst(frame_nodes.get(frame, []), source_node, target_node, frame, source_frame, target_frame)
            for node in near_nodes:
                colour = (70, 170, 255) if likely_bridge else (178, 186, 195)
                role = "intermediate_bridge_candidate" if likely_bridge else "nearby_candidate"
                label_prefix = "bridge candidate" if likely_bridge else "nearby candidate"
                draw_node(node, role, label_prefix, colour, 2)
                if likely_bridge:
                    intermediate_candidate_ids.append(str(node.get("visible_person_base_id", "")))
        metadata["bbox_transform_applied"] = bool(drawn_rows)
        for row in drawn_rows:
            row["coordinate_metadata"]["bbox_transform_applied"] = bool(drawn_rows)
            row["overlay_debug_valid"] = overlay_debug_row_valid(row)
        frame_coordinate_metadata.append(
            {
                "frame_sequence": frame,
                "drawn_bbox_count": len(drawn_rows),
                **metadata,
            }
        )
        overlay_debug_rows.extend(drawn_rows)
        cv2_module.putText(image, f"frame {frame}", (12, 24), cv2_module.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2_module.LINE_AA)
        cv2_module.putText(image, VISUAL_ONLY_WARNING, (12, image.shape[0] - 12), cv2_module.FONT_HERSHEY_SIMPLEX, 0.42, (120, 220, 240), 1, cv2_module.LINE_AA)
        cv2_module.putText(raw_image, f"frame {frame} raw", (12, 24), cv2_module.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2_module.LINE_AA)
        rendered_frames.append(image)
        raw_frames.append(raw_image)
        comparison_frames.append(horizontal_strip([raw_image, image]))
        pil_frames.append(Image.fromarray(cv2_module.cvtColor(image, cv2_module.COLOR_BGR2RGB)))
    if not rendered_frames:
        return {
            "burst_clip_path": "",
            "burst_strip_path": "",
            "burst_frame_sequences": burst_frames,
            "intermediate_candidate_ids": [],
            "burst_evidence_available": False,
            "burst_evidence_missing_reason": "no_burst_frames",
            "burst_frame_coordinate_metadata": [],
            "burst_overlay_debug_rows": [],
        }
    ensure_dir(paths["burst_clip"].parent)
    pil_frames[0].save(paths["burst_clip"], save_all=True, append_images=pil_frames[1:], duration=550, loop=0)
    strip = horizontal_strip(rendered_frames[:8])
    raw_strip = horizontal_strip(raw_frames[:8])
    comparison_strip = horizontal_strip(comparison_frames[:8])
    write_image(paths["burst_strip"], strip)
    raw_strip_path = STEP2M1R_REVIEW_BURST_RAW_STRIPS_DIR / f"{paths['burst_strip'].stem}_raw.jpg"
    comparison_strip_path = STEP2M1R_REVIEW_BURST_COMPARISON_STRIPS_DIR / f"{paths['burst_strip'].stem}_comparison.jpg"
    write_image(raw_strip_path, raw_strip)
    write_image(comparison_strip_path, comparison_strip)
    invalid_debug_rows = [row for row in overlay_debug_rows if row.get("overlay_debug_valid") is not True]
    return {
        "burst_clip_path": rel_asset_path(paths["burst_clip"]),
        "burst_strip_path": rel_asset_path(paths["burst_strip"]),
        "burst_raw_strip_path": rel_asset_path(raw_strip_path),
        "burst_overlay_comparison_strip_path": rel_asset_path(comparison_strip_path),
        "burst_frame_sequences": burst_frames,
        "intermediate_candidate_ids": list(dict.fromkeys(intermediate_candidate_ids)),
        "burst_evidence_available": not missing_core_frame,
        "burst_evidence_missing_reason": "missing_core_frame" if missing_core_frame else "",
        "burst_frame_coordinate_metadata": frame_coordinate_metadata,
        "burst_overlay_debug_rows": overlay_debug_rows,
        "burst_overlay_drawn_bbox_count": len(overlay_debug_rows),
        "burst_overlay_invalid_bbox_count": len(invalid_debug_rows),
        "burst_overlay_alignment_valid": not invalid_debug_rows,
        "current_overlay_version": CURRENT_BURST_OVERLAY_VERSION,
    }


def enrich_m1r_candidate_with_node_context(candidate: dict[str, Any], node_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = node_lookup.get(str(candidate.get("source_visible_person_base_id", "")), {})
    target = node_lookup.get(str(candidate.get("target_visible_person_base_id", "")), {})
    enriched = dict(candidate)
    enriched.update(
        {
            "source_bbox": source.get("bbox", {}),
            "target_bbox": target.get("bbox", {}),
            "source_crop_quality": source.get("crop_quality", ""),
            "target_crop_quality": target.get("crop_quality", ""),
            "source_step1f3_final_visual_role_state": source.get("step1f3_final_visual_role_state", ""),
            "target_step1f3_final_visual_role_state": target.get("step1f3_final_visual_role_state", ""),
            "source_step1f3_role_team_context": source.get("step1f3_role_team_context", ""),
            "target_step1f3_role_team_context": target.get("step1f3_role_team_context", ""),
            "source_c2c_final_colour_belief": source.get("c2c_final_colour_belief", ""),
            "target_c2c_final_colour_belief": target.get("c2c_final_colour_belief", ""),
            "source_warning_flags": source.get("step1f3_warning_flags", []),
            "target_warning_flags": target.get("step1f3_warning_flags", []),
        }
    )
    return enriched


def render_m1r_still_assets(candidates: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    if cv2 is None or np is None:
        return {}
    cv2_module = require_cv2()
    frame_lookup = frame_file_by_sequence()
    for directory in [
        STEP2M1R_SOURCE_CONTEXT_IMAGES_DIR,
        STEP2M1R_TARGET_CONTEXT_IMAGES_DIR,
        STEP2M1R_SOURCE_CROP_IMAGES_DIR,
        STEP2M1R_TARGET_CROP_IMAGES_DIR,
    ]:
        ensure_dir(directory)
    image_cache: dict[int, Any | None] = {}
    assets: dict[str, dict[str, str]] = {}
    for candidate in candidates:
        paths = m1r_asset_paths(candidate)
        source_frame = safe_int(candidate.get("source_frame_sequence"), -1)
        target_frame = safe_int(candidate.get("target_frame_sequence"), -1)
        for frame_sequence in [source_frame, target_frame]:
            if frame_sequence not in image_cache:
                frame_path = frame_lookup.get(frame_sequence, "")
                image_cache[frame_sequence] = cv2_module.imread(frame_path) if frame_path and Path(frame_path).exists() else None
        source_image = image_cache.get(source_frame)
        target_image = image_cache.get(target_frame)
        source_bbox = candidate.get("source_bbox", {})
        target_bbox = candidate.get("target_bbox", {})
        if source_image is None:
            source_context = placeholder((740, 420), f"missing source frame {source_frame}")
            source_crop = placeholder((320, 420), f"missing source frame {source_frame}")
        else:
            source_context = resize_fit(draw_context(source_image, source_bbox, padded_bbox(source_bbox, source_image.shape[:2]), (0, 215, 255)), 760, 460)
            source_crop = resize_fit(crop(source_image, source_bbox, (320, 420), "source crop"), 340, 480)
        if target_image is None:
            target_context = placeholder((740, 420), f"missing target frame {target_frame}")
            target_crop = placeholder((320, 420), f"missing target frame {target_frame}")
        else:
            target_context = resize_fit(draw_context(target_image, target_bbox, padded_bbox(target_bbox, target_image.shape[:2]), (115, 245, 145)), 760, 460)
            target_crop = resize_fit(crop(target_image, target_bbox, (320, 420), "target crop"), 340, 480)
        write_image(paths["source_context_image"], source_context)
        write_image(paths["target_context_image"], target_context)
        write_image(paths["source_crop_image"], source_crop)
        write_image(paths["target_crop_image"], target_crop)
        assets[str(candidate.get("step2m1r_review_candidate_id", ""))] = {
            "source_context_image": rel_asset_path(paths["source_context_image"]),
            "target_context_image": rel_asset_path(paths["target_context_image"]),
            "source_crop_image": rel_asset_path(paths["source_crop_image"]),
            "target_crop_image": rel_asset_path(paths["target_crop_image"]),
        }
    return assets


def candidate_has_required_overlay_debug(candidate: dict[str, Any], debug_rows: list[dict[str, Any]]) -> bool:
    candidate_id = str(candidate.get("step2m1r_review_candidate_id", ""))
    rows_for_candidate = [row for row in debug_rows if str(row.get("step2m1r_review_candidate_id", "")) == candidate_id]
    if not rows_for_candidate:
        return False
    if any(row.get("overlay_debug_valid") is not True for row in rows_for_candidate):
        return False
    source_frame = safe_int(candidate.get("source_frame_sequence"), -1)
    target_frame = safe_int(candidate.get("target_frame_sequence"), -1)
    has_source = any(row.get("overlay_role") == "source" and safe_int(row.get("frame_sequence"), -2) == source_frame for row in rows_for_candidate)
    has_target = any(row.get("overlay_role") == "target" and safe_int(row.get("frame_sequence"), -2) == target_frame for row in rows_for_candidate)
    return has_source and has_target


def render_overlay_alignment_qa_samples(candidates: list[dict[str, Any]], debug_rows: list[dict[str, Any]], *, limit: int = 10) -> list[str]:
    if cv2 is None or np is None:
        return []
    cv2_module = require_cv2()
    ensure_dir(STEP2M1R_BURST_OVERLAY_QA_DIR)
    debug_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in debug_rows:
        debug_by_candidate.setdefault(str(row.get("step2m1r_review_candidate_id", "")), []).append(row)
    output_paths: list[str] = []
    for candidate in candidates[:limit]:
        candidate_id = str(candidate.get("step2m1r_review_candidate_id", ""))
        raw_strip = cv2_module.imread(str(STEP2M1_OUTPUT_DIR / candidate.get("burst_raw_strip_path", "")))
        overlay_strip = cv2_module.imread(str(STEP2M1_OUTPUT_DIR / candidate.get("burst_strip_path", "")))
        source_crop = cv2_module.imread(str(STEP2M1_OUTPUT_DIR / candidate.get("ui_assets", {}).get("source_crop_image", "")))
        target_crop = cv2_module.imread(str(STEP2M1_OUTPUT_DIR / candidate.get("ui_assets", {}).get("target_crop_image", "")))
        if raw_strip is None:
            raw_strip = placeholder((980, 220), "missing raw strip")
        if overlay_strip is None:
            overlay_strip = placeholder((980, 220), "missing overlay strip")
        if source_crop is None:
            source_crop = placeholder((220, 260), "source crop")
        if target_crop is None:
            target_crop = placeholder((220, 260), "target crop")
        raw_strip = resize_fit(raw_strip, 1380, 260)
        overlay_strip = resize_fit(overlay_strip, 1380, 260)
        source_crop = resize_fit(source_crop, 220, 260)
        target_crop = resize_fit(target_crop, 220, 260)
        canvas_width = max(1420, raw_strip.shape[1] + 40)
        canvas_height = 930
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:] = (18, 21, 24)
        canvas[44 : 44 + raw_strip.shape[0], 20 : 20 + raw_strip.shape[1]] = raw_strip
        canvas[330 : 330 + overlay_strip.shape[0], 20 : 20 + overlay_strip.shape[1]] = overlay_strip
        canvas[620 : 620 + source_crop.shape[0], 20 : 20 + source_crop.shape[1]] = source_crop
        canvas[620 : 620 + target_crop.shape[0], 260 : 260 + target_crop.shape[1]] = target_crop
        cv2_module.putText(canvas, "raw frames without overlays", (20, 28), cv2_module.FONT_HERSHEY_SIMPLEX, 0.7, (240, 244, 248), 2, cv2_module.LINE_AA)
        cv2_module.putText(canvas, "transformed overlay frames", (20, 314), cv2_module.FONT_HERSHEY_SIMPLEX, 0.7, (240, 244, 248), 2, cv2_module.LINE_AA)
        cv2_module.putText(canvas, "crop evidence and transformed bbox debug", (20, 604), cv2_module.FONT_HERSHEY_SIMPLEX, 0.7, (240, 244, 248), 2, cv2_module.LINE_AA)
        debug_lines = []
        for row in debug_by_candidate.get(candidate_id, [])[:10]:
            debug_lines.append(
                f"f{row.get('frame_sequence')} {row.get('overlay_role')} raw={row.get('raw_bbox')} -> rendered={row.get('transformed_bbox')} clipped={row.get('clipped')}"
            )
        if not debug_lines:
            debug_lines = ["no overlay debug rows"]
        for index, line in enumerate(debug_lines[:8]):
            cv2_module.putText(canvas, line[:140], (510, 646 + index * 24), cv2_module.FONT_HERSHEY_SIMPLEX, 0.43, (210, 226, 236), 1, cv2_module.LINE_AA)
        cv2_module.putText(canvas, VISUAL_ONLY_WARNING, (510, 902), cv2_module.FONT_HERSHEY_SIMPLEX, 0.48, (120, 220, 240), 1, cv2_module.LINE_AA)
        path = STEP2M1R_BURST_OVERLAY_QA_DIR / f"{safe_stem(candidate_id)}_overlay_alignment_qa.png"
        write_image(path, canvas)
        output_paths.append(rel_asset_path(path))
    return output_paths


def burst_overlay_alignment_summary_payload(
    candidate_payload: dict[str, Any],
    debug_rows: list[dict[str, Any]],
    qa_sample_paths: list[str],
) -> dict[str, Any]:
    candidate_rows = rows_from_payload(candidate_payload)
    valid_candidate_ids = []
    invalid_candidate_ids = []
    for candidate in candidate_rows:
        candidate_id = str(candidate.get("step2m1r_review_candidate_id", ""))
        if candidate_has_required_overlay_debug(candidate, debug_rows):
            valid_candidate_ids.append(candidate_id)
        else:
            invalid_candidate_ids.append(candidate_id)
    invalid_drawn_rows = [row for row in debug_rows if row.get("overlay_debug_valid") is not True]
    safe = not invalid_candidate_ids and not invalid_drawn_rows
    summary = guardrail_stamp(
        {
            "artifact": "step2m1r_burst_overlay_alignment_summary",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "current_overlay_version": CURRENT_BURST_OVERLAY_VERSION,
            "review_decisions_collected_with_overlay_version": "",
            "review_decisions_overlay_version_matches_current": False,
            "burst_review_candidates": len(candidate_rows),
            "drawn_bbox_count": len(debug_rows),
            "valid_drawn_bbox_count": len(debug_rows) - len(invalid_drawn_rows),
            "invalid_drawn_bbox_count": len(invalid_drawn_rows),
            "candidates_with_valid_overlay_transforms": len(valid_candidate_ids),
            "candidates_with_invalid_overlay_transforms": len(invalid_candidate_ids),
            "invalid_overlay_candidate_ids_sample": invalid_candidate_ids[:20],
            "burst_overlay_alignment_safe_for_review": safe,
            "overlay_qa_sample_paths": qa_sample_paths,
        }
    )
    assert_no_forbidden_keys(summary)
    return summary


def render_m1r_burst_evidence(candidate_payload: dict[str, Any], node_payload: dict[str, Any]) -> dict[str, Any]:
    node_lookup = nodes_by_visible_id(node_payload)
    frame_nodes = nodes_by_frame_sequence(node_payload)
    frame_lookup = frame_file_by_sequence()
    enriched_rows = [enrich_m1r_candidate_with_node_context(row, node_lookup) for row in rows_from_payload(candidate_payload)]
    still_assets = render_m1r_still_assets(enriched_rows)
    all_overlay_debug_rows: list[dict[str, Any]] = []
    for candidate in enriched_rows:
        candidate_id = str(candidate.get("step2m1r_review_candidate_id", ""))
        candidate["ui_assets"] = still_assets.get(candidate_id, {})
        burst = render_burst_evidence_for_candidate(candidate, node_lookup, frame_nodes, frame_lookup)
        debug_rows = burst.pop("burst_overlay_debug_rows", [])
        all_overlay_debug_rows.extend(debug_rows)
        candidate.update(burst)
        candidate["ui_assets"].update(
            {
                "burst_clip": candidate.get("burst_clip_path", ""),
                "burst_strip": candidate.get("burst_strip_path", ""),
                "burst_raw_strip": candidate.get("burst_raw_strip_path", ""),
                "burst_overlay_comparison_strip": candidate.get("burst_overlay_comparison_strip_path", ""),
            }
        )
        visual_stamp(candidate)
        assert_no_forbidden_keys(candidate)
    candidate_payload = {**candidate_payload, "rows": enriched_rows}
    qa_sample_paths = render_overlay_alignment_qa_samples(enriched_rows, all_overlay_debug_rows)
    overlay_summary = burst_overlay_alignment_summary_payload(candidate_payload, all_overlay_debug_rows, qa_sample_paths)
    debug_payload = guardrail_stamp(
        {
            "artifact": "step2m1r_burst_overlay_debug_rows",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "current_overlay_version": CURRENT_BURST_OVERLAY_VERSION,
            "rows": all_overlay_debug_rows,
            "summary": {
                "drawn_bbox_count": len(all_overlay_debug_rows),
                "valid_drawn_bbox_count": overlay_summary.get("valid_drawn_bbox_count", 0),
                "invalid_drawn_bbox_count": overlay_summary.get("invalid_drawn_bbox_count", 0),
                "burst_overlay_alignment_safe_for_review": overlay_summary.get("burst_overlay_alignment_safe_for_review", False),
            },
        }
    )
    assert_no_forbidden_keys(debug_payload)
    write_json(STEP2M1R_BURST_OVERLAY_DEBUG_ROWS_PATH, debug_payload)
    write_json(STEP2M1R_BURST_OVERLAY_ALIGNMENT_SUMMARY_PATH, overlay_summary)
    burst_missing = [row for row in enriched_rows if row.get("burst_evidence_available") is not True]
    gap_rows = [row for row in enriched_rows if safe_int(row.get("frame_gap"), 0) > 1]
    missing_gap_rows = [row for row in gap_rows if row.get("burst_evidence_available") is not True]
    candidate_payload["summary"] = {
        **candidate_payload.get("summary", {}),
        "burst_evidence_rows": len(enriched_rows),
        "burst_evidence_available_rows": len(enriched_rows) - len(burst_missing),
        "burst_evidence_missing_rows": len(burst_missing),
        "burst_evidence_missing_rate": round(len(burst_missing) / max(1, len(enriched_rows)), 4),
        "frame_gap_gt_1_rows": len(gap_rows),
        "frame_gap_gt_1_missing_burst_rows": len(missing_gap_rows),
        "current_overlay_version": CURRENT_BURST_OVERLAY_VERSION,
        "review_decisions_collected_with_overlay_version": "",
        "review_decisions_overlay_version_matches_current": False,
        "burst_overlay_alignment_safe_for_review": overlay_summary.get("burst_overlay_alignment_safe_for_review", False),
        "candidates_with_valid_overlay_transforms": overlay_summary.get("candidates_with_valid_overlay_transforms", 0),
        "candidates_with_invalid_overlay_transforms": overlay_summary.get("candidates_with_invalid_overlay_transforms", 0),
        "overlay_qa_sample_paths": qa_sample_paths,
    }
    return candidate_payload


def render_m1r_contact_sheet(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if cv2 is None or np is None:
        ensure_dir(STEP2M1R_REVIEW_CONTACT_SHEET_PATH.parent)
        STEP2M1R_REVIEW_CONTACT_SHEET_PATH.write_bytes(b"")
        return {"step2m1r_review_contact_sheet_path": str(STEP2M1R_REVIEW_CONTACT_SHEET_PATH.resolve()), "fallback_image": True}
    cv2_module = require_cv2()
    tile_w, tile_h = 300, 230
    cols = 5
    rows_needed = max(1, (len(candidates) + cols - 1) // cols)
    sheet = np.zeros((rows_needed * tile_h, cols * tile_w, 3), dtype=np.uint8)
    sheet[:] = (16, 18, 20)
    for index, candidate in enumerate(candidates):
        asset = candidate.get("ui_assets", {})
        source_path = STEP2M1_OUTPUT_DIR / asset.get("source_crop_image", "")
        target_path = STEP2M1_OUTPUT_DIR / asset.get("target_crop_image", "")
        source_crop = cv2_module.imread(str(source_path)) if source_path.exists() else placeholder((90, 130), "source")
        target_crop = cv2_module.imread(str(target_path)) if target_path.exists() else placeholder((90, 130), "target")
        source_crop = resize_fit(source_crop, 92, 125)
        target_crop = resize_fit(target_crop, 92, 125)
        x = (index % cols) * tile_w
        y = (index // cols) * tile_h
        sheet[y : y + source_crop.shape[0], x : x + source_crop.shape[1]] = source_crop
        sheet[y : y + target_crop.shape[0], x + 98 : x + 98 + target_crop.shape[1]] = target_crop
        cv2_module.putText(sheet, str(candidate.get("review_bucket", ""))[:32], (x + 4, y + 152), cv2_module.FONT_HERSHEY_SIMPLEX, 0.35, (230, 238, 245), 1, cv2_module.LINE_AA)
        cv2_module.putText(sheet, f"{candidate.get('source_frame_sequence')}->{candidate.get('target_frame_sequence')}", (x + 4, y + 176), cv2_module.FONT_HERSHEY_SIMPLEX, 0.38, (180, 230, 180), 1, cv2_module.LINE_AA)
        cv2_module.putText(sheet, f"burst={str(candidate.get('burst_evidence_available', False)).lower()}", (x + 4, y + 200), cv2_module.FONT_HERSHEY_SIMPLEX, 0.35, (110, 220, 240), 1, cv2_module.LINE_AA)
    write_image(STEP2M1R_REVIEW_CONTACT_SHEET_PATH, sheet)
    return {"step2m1r_review_contact_sheet_path": str(STEP2M1R_REVIEW_CONTACT_SHEET_PATH.resolve()), "fallback_image": False}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    temp_path = path.with_name(f"{path.stem}.tmp.{hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()[:12]}{path.suffix}")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def m1r_candidate_by_review_id(candidate_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("step2m1r_review_candidate_id", "")): row
        for row in rows_from_payload(candidate_payload)
        if row.get("step2m1r_review_candidate_id")
    }


def m1r_candidate_by_edge_id(candidate_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("continuity_edge_id", "")): row
        for row in rows_from_payload(candidate_payload)
        if row.get("continuity_edge_id")
    }


def m1r_reviewed_decision_template() -> dict[str, Any]:
    return guardrail_stamp(
        {
            "artifact": "step2m1r_reviewed_visual_continuity_decisions",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "current_overlay_version": CURRENT_BURST_OVERLAY_VERSION,
            "rows": [],
        }
    )


def normalize_m1r_decision_value(value: Any) -> str:
    decision = str(value)
    if decision == "unsure":
        return UNSURE_DECISION
    return decision


def m1r_reviewed_decision_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    payload_rows = payload.get("rows", payload.get("decisions", []))
    return [dict(row) for row in payload_rows if isinstance(row, dict)] if isinstance(payload_rows, list) else []


def normalize_m1r_reviewed_decisions_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        normalized = dict(payload)
        normalized.pop("decisions", None)
    else:
        normalized = {
            "artifact": "step2m1r_reviewed_visual_continuity_decisions",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "current_overlay_version": CURRENT_BURST_OVERLAY_VERSION,
        }
    normalized_rows = m1r_reviewed_decision_rows(payload)
    normalized["rows"] = normalized_rows
    normalized["reviewed_decision_rows"] = len(normalized_rows)
    normalized.setdefault("artifact", "step2m1r_reviewed_visual_continuity_decisions")
    normalized.setdefault("created_at", utc_iso())
    normalized.setdefault("match_id", MATCH_ID)
    normalized.setdefault("clip_id", CLIP_ID)
    normalized.setdefault("current_overlay_version", CURRENT_BURST_OVERLAY_VERSION)
    return guardrail_stamp(normalized)


def read_m1r_reviewed_decisions() -> dict[str, Any]:
    if STEP2M1R_REVIEWED_DECISIONS_PATH.exists():
        return normalize_m1r_reviewed_decisions_payload(read_json(STEP2M1R_REVIEWED_DECISIONS_PATH))
    return m1r_reviewed_decision_template()


def m1r_review_decision_row(candidate: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    decision = normalize_m1r_decision_value(payload.get("human_review_decision", ""))
    if decision not in {ACCEPT_DECISION, REJECT_DECISION, UNSURE_DECISION}:
        raise ValueError(f"Step2.M1R human review decision is not allowed: {decision}")
    row = {
        "step2m1r_review_candidate_id": candidate.get("step2m1r_review_candidate_id", ""),
        "continuity_edge_id": candidate.get("continuity_edge_id", ""),
        "source_visible_person_base_id": candidate.get("source_visible_person_base_id", ""),
        "target_visible_person_base_id": candidate.get("target_visible_person_base_id", ""),
        "source_frame_sequence": candidate.get("source_frame_sequence", -1),
        "target_frame_sequence": candidate.get("target_frame_sequence", -1),
        "review_bucket": candidate.get("review_bucket", ""),
        "human_review_decision": decision,
        "reviewer_name": str(payload.get("reviewer_name", "")),
        "notes": str(payload.get("notes", "")),
        "reviewed_at": str(payload.get("reviewed_at", "")) or utc_iso(),
        "human_confirmed": True,
        "current_overlay_version": CURRENT_BURST_OVERLAY_VERSION,
        "review_decisions_collected_with_overlay_version": CURRENT_BURST_OVERLAY_VERSION,
        "approve_any_identity_tracking": False,
        "approve_any_player_slot_use": False,
        "approve_any_goalkeeper_slot_use": False,
        "approve_any_metric_use": False,
        "approve_event_or_tactical_analysis": False,
        "approve_exact_22_or_exact_two_goalkeeper_forcing": False,
        "approve_official_referee_exclusion": False,
        "approve_bad_detection_deletion": False,
        "approve_production_promotion": False,
    }
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def m1r_candidate_for_reviewed_row(
    row: dict[str, Any],
    candidates_by_review_id: dict[str, dict[str, Any]],
    candidates_by_edge_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    candidate_id = str(row.get("step2m1r_review_candidate_id", ""))
    if candidate_id and candidate_id in candidates_by_review_id:
        return candidates_by_review_id[candidate_id]
    edge_id = str(row.get("continuity_edge_id", ""))
    if edge_id:
        return candidates_by_edge_id.get(edge_id)
    return None


def validate_m1r_reviewed_rows(candidate_payload: dict[str, Any], reviewed_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = m1r_candidate_by_review_id(candidate_payload)
    candidates_by_edge = m1r_candidate_by_edge_id(candidate_payload)
    usable: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(m1r_reviewed_decision_rows(reviewed_payload)):
        candidate_id = str(row.get("step2m1r_review_candidate_id", ""))
        candidate = m1r_candidate_for_reviewed_row(row, candidates, candidates_by_edge)
        if not candidate:
            errors.append({"row_index": index, "step2m1r_review_candidate_id": candidate_id, "continuity_edge_id": str(row.get("continuity_edge_id", "")), "error": "unknown_review_candidate"})
            continue
        canonical_candidate_id = str(candidate.get("step2m1r_review_candidate_id", candidate_id))
        if canonical_candidate_id in seen:
            errors.append({"row_index": index, "step2m1r_review_candidate_id": canonical_candidate_id, "error": "duplicate_review_candidate_id"})
        seen.add(canonical_candidate_id)
        for field in [
            "continuity_edge_id",
            "source_visible_person_base_id",
            "target_visible_person_base_id",
            "source_frame_sequence",
            "target_frame_sequence",
            "review_bucket",
        ]:
            if str(row.get(field, "")) != str(candidate.get(field, "")):
                errors.append({"row_index": index, "step2m1r_review_candidate_id": canonical_candidate_id, "error": f"{field}_mismatch"})
        decision = normalize_m1r_decision_value(row.get("human_review_decision", ""))
        if decision not in {ACCEPT_DECISION, REJECT_DECISION, UNSURE_DECISION}:
            errors.append({"row_index": index, "step2m1r_review_candidate_id": canonical_candidate_id, "error": "human_review_decision_not_allowed"})
        if row.get("human_confirmed") is not True:
            errors.append({"row_index": index, "step2m1r_review_candidate_id": canonical_candidate_id, "error": "human_confirmed_true_required"})
        for key in FORBIDDEN_APPROVAL_FLAGS:
            if row.get(key) is not False:
                errors.append({"row_index": index, "step2m1r_review_candidate_id": canonical_candidate_id, "error": "forbidden_approval_flag_true_or_missing", "key": key})
        if row.get("visual_only_warning") != VISUAL_ONLY_WARNING or row.get("do_not_use_for_metrics") is not True:
            errors.append({"row_index": index, "step2m1r_review_candidate_id": canonical_candidate_id, "error": "visual_guardrail_missing"})
        if row.get("production_ready") is not False or row.get("no_auto_promotion") is not True or row.get("human_approved") is not False:
            errors.append({"row_index": index, "step2m1r_review_candidate_id": canonical_candidate_id, "error": "promotion_guardrail_invalid"})
        forbidden = forbidden_keys_present(row)
        if forbidden:
            errors.append({"row_index": index, "step2m1r_review_candidate_id": canonical_candidate_id, "error": "forbidden_keys_present", "keys": forbidden})
        usable.append({**row, "step2m1r_review_candidate_id": canonical_candidate_id, "human_review_decision": decision})
    return usable if not errors else [], errors


def m1r_review_progress_payload(candidate_payload: dict[str, Any], reviewed_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    candidate_rows = rows_from_payload(candidate_payload)
    bucket_counts = Counter(str(row.get("review_bucket", "")) for row in candidate_rows)
    missing_burst = [row for row in candidate_rows if row.get("burst_evidence_available") is not True]
    reviewed_payload = reviewed_payload or {"rows": []}
    usable_rows, validation_errors = validate_m1r_reviewed_rows(candidate_payload, reviewed_payload)
    decision_counts = Counter(str(row.get("human_review_decision", "")) for row in usable_rows)
    accepted_count = decision_counts.get(ACCEPT_DECISION, 0)
    rejected_count = decision_counts.get(REJECT_DECISION, 0)
    unsure_count = decision_counts.get(UNSURE_DECISION, 0)
    candidates = m1r_candidate_by_review_id(candidate_payload)
    reviewed_by_bucket = Counter(str(candidates.get(str(row.get("step2m1r_review_candidate_id", "")), {}).get("review_bucket", "")) for row in usable_rows)
    bucket_progress = {
        bucket: {"total": bucket_counts.get(bucket, 0), "reviewed": reviewed_by_bucket.get(bucket, 0)}
        for bucket in sorted(bucket_counts)
    }
    collected_versions = {
        str(row.get("review_decisions_collected_with_overlay_version", ""))
        for row in usable_rows
        if row.get("review_decisions_collected_with_overlay_version")
    }
    current_overlay_version = str(candidate_payload.get("summary", {}).get("current_overlay_version", CURRENT_BURST_OVERLAY_VERSION))
    version_matches = bool(usable_rows) and collected_versions == {current_overlay_version}
    targeted_review_completed = len(usable_rows) == len(candidate_rows) and not validation_errors
    return guardrail_stamp(
        {
            "artifact": "step2m1r_review_progress_summary",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "total_review_candidates": len(candidate_rows),
            "reviewed_candidates": len(usable_rows),
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "unsure_count": unsure_count,
            "correction_rate": round((rejected_count + unsure_count) / max(1, len(usable_rows)), 4),
            "safe_auto_accept_audit_rows": bucket_counts.get("safe_auto_accept_audit", 0),
            "safe_auto_accept_audit_reviewed": reviewed_by_bucket.get("safe_auto_accept_audit", 0),
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "bucket_progress": bucket_progress,
            "burst_evidence_missing_rows": len(missing_burst),
            "burst_evidence_missing_rate": round(len(missing_burst) / max(1, len(candidate_rows)), 4),
            "current_overlay_version": current_overlay_version,
            "review_decisions_collected_with_overlay_version": sorted(collected_versions),
            "review_decisions_overlay_version_matches_current": version_matches,
            "burst_overlay_alignment_safe_for_review": candidate_payload.get("summary", {}).get("burst_overlay_alignment_safe_for_review", False),
            "candidates_with_valid_overlay_transforms": candidate_payload.get("summary", {}).get("candidates_with_valid_overlay_transforms", 0),
            "candidates_with_invalid_overlay_transforms": candidate_payload.get("summary", {}).get("candidates_with_invalid_overlay_transforms", 0),
            "targeted_review_completed": targeted_review_completed,
            "targeted_review_required_before_m2": not targeted_review_completed,
            "validation_errors": validation_errors,
            "keyboard_shortcuts": {"A": "accept", "X": "reject", "U": "unsure"},
        }
    )


def m1r_review_decision_payload(candidate_payload: dict[str, Any], reviewed_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    progress = m1r_review_progress_payload(candidate_payload, reviewed_payload)
    usable_rows, _errors = validate_m1r_reviewed_rows(candidate_payload, reviewed_payload or {"rows": []})
    return guardrail_stamp(
        {
            "artifact": "step2m1r_review_decision_summary",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "total_review_candidates": progress["total_review_candidates"],
            "reviewed_candidates": progress["reviewed_candidates"],
            "human_review_decision_counts": dict(sorted(Counter(str(row.get("human_review_decision", "")) for row in usable_rows).items())),
            "accepted_count": progress["accepted_count"],
            "rejected_count": progress["rejected_count"],
            "unsure_count": progress["unsure_count"],
            "correction_rate": progress["correction_rate"],
            "bucket_progress": progress["bucket_progress"],
            "targeted_review_completed": progress["targeted_review_completed"],
            "current_overlay_version": candidate_payload.get("summary", {}).get("current_overlay_version", CURRENT_BURST_OVERLAY_VERSION),
            "review_decisions_collected_with_overlay_version": progress["review_decisions_collected_with_overlay_version"],
            "review_decisions_overlay_version_matches_current": progress["review_decisions_overlay_version_matches_current"],
        }
    )


def refresh_m1r_progress_decision_and_gate(
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    progress = m1r_review_progress_payload(candidate_payload, reviewed_payload)
    decision_summary = m1r_review_decision_payload(candidate_payload, reviewed_payload)
    write_json_atomic(STEP2M1R_REVIEW_PROGRESS_SUMMARY_PATH, progress)
    write_json_atomic(STEP2M1R_REVIEW_DECISION_SUMMARY_PATH, decision_summary)
    manifest: dict[str, Any] | None = None
    if (
        STEP2M1R_ADAPTATION_SAFE_EDGE_ROWS_PATH.exists()
        and STEP2M1R_ADAPTATION_SAFE_GROUP_ROWS_PATH.exists()
        and STEP2M1R_GROUP_SPAN_REMEDIATION_SUMMARY_PATH.exists()
    ):
        review_payload_for_gate = {
            **candidate_payload,
            "targeted_review_completed": progress.get("targeted_review_completed", False),
            "summary": {
                **candidate_payload.get("summary", {}),
                "reviewed_candidates": progress.get("reviewed_candidates", 0),
                "accepted_count": progress.get("accepted_count", 0),
                "rejected_count": progress.get("rejected_count", 0),
                "unsure_count": progress.get("unsure_count", 0),
                "review_decisions_collected_with_overlay_version": progress.get("review_decisions_collected_with_overlay_version", []),
                "review_decisions_overlay_version_matches_current": progress.get("review_decisions_overlay_version_matches_current", False),
            },
        }
        manifest = build_adaptation_safety_manifest(
            edge_payload=read_json(STEP2M1R_ADAPTATION_SAFE_EDGE_ROWS_PATH),
            group_payload=read_json(STEP2M1R_ADAPTATION_SAFE_GROUP_ROWS_PATH),
            remediation_summary=read_json(STEP2M1R_GROUP_SPAN_REMEDIATION_SUMMARY_PATH),
            review_payload=review_payload_for_gate,
        )
        write_json_atomic(STEP2M1R_ADAPTATION_SAFETY_MANIFEST_PATH, manifest)
    return progress, decision_summary, manifest


def save_m1r_review_decision(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidate_payload = read_json(STEP2M1R_TARGETED_REVIEW_CANDIDATE_ROWS_PATH)
    by_candidate_id = m1r_candidate_by_review_id(candidate_payload)
    by_edge_id = m1r_candidate_by_edge_id(candidate_payload)
    candidate = by_candidate_id.get(str(payload.get("step2m1r_review_candidate_id", ""))) or by_edge_id.get(str(payload.get("continuity_edge_id", "")))
    if not candidate:
        raise ValueError("unknown_step2m1r_review_candidate")
    decision = m1r_review_decision_row(candidate, payload)
    reviewed_payload = read_m1r_reviewed_decisions()
    existing_rows = rows_from_payload(reviewed_payload)
    upsert_key = str(decision.get("step2m1r_review_candidate_id", ""))
    by_id = {
        str(row.get("step2m1r_review_candidate_id", "")): row
        for row in existing_rows
        if row.get("step2m1r_review_candidate_id")
    }
    by_id[upsert_key] = decision
    updated_payload = guardrail_stamp(
        {
            "artifact": "step2m1r_reviewed_visual_continuity_decisions",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "current_overlay_version": CURRENT_BURST_OVERLAY_VERSION,
            "reviewed_decision_rows": len(by_id),
            "rows": sorted(by_id.values(), key=lambda row: str(row.get("step2m1r_review_candidate_id", ""))),
        }
    )
    assert_no_forbidden_keys(updated_payload)
    write_json_atomic(STEP2M1R_REVIEWED_DECISIONS_PATH, updated_payload)
    progress, _decision_summary, _manifest = refresh_m1r_progress_decision_and_gate(candidate_payload, updated_payload)
    return decision, updated_payload, progress


def m1r_html_template(candidate_payload: dict[str, Any]) -> str:
    state_json = json.dumps(
        {
            "artifact": "step2m1r_review_ui_state",
            "visual_only_warning": VISUAL_ONLY_WARNING,
            "rows": rows_from_payload(candidate_payload),
        }
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Step2.M1R Targeted Visual Continuity Review</title>
<style>
body{{margin:0;font-family:Arial,sans-serif;background:#101315;color:#f5f7f8}}
header{{position:sticky;top:0;background:#1b2227;border-bottom:1px solid #36434b;padding:10px 14px;z-index:2}}
.wrap{{display:grid;grid-template-columns:1.15fr .85fr;gap:12px;padding:12px}}
.media{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.panel{{background:#182126;border:1px solid #33414a;border-radius:6px;padding:10px}}
.panel img{{max-width:100%;max-height:430px;display:block;margin:auto;background:#0b0d0f}}
.burst{{grid-column:1 / span 2}}
.meta{{display:grid;grid-template-columns:180px 1fr;gap:7px;font-size:13px;line-height:1.25}}
.pill{{display:inline-block;padding:4px 7px;border:1px solid #4a5962;border-radius:999px;margin:2px;color:#d6edf8}}
button{{background:#28343b;color:#f5f7f8;border:1px solid #4b5d68;border-radius:6px;padding:8px 10px;margin:4px;cursor:pointer}}
button.primary{{background:#176073;border-color:#2093ad}}
button.reject{{background:#642c2c;border-color:#9b4a4a}}
button.warn{{background:#665018;border-color:#9b7c22}}
textarea,input{{width:100%;background:#0f1316;color:#f5f7f8;border:1px solid #3b4952;border-radius:5px;padding:8px;box-sizing:border-box}}
.small{{font-size:12px;color:#aebdc5}}
.rule{{font-weight:bold;color:#f3d383;line-height:1.35}}
</style>
</head>
<body>
<header>
  <b>Step2.M1R Targeted Review</b>
  <span id="counter" class="pill"></span>
  <span id="bucket" class="pill"></span>
  <span class="pill">{VISUAL_ONLY_WARNING}</span>
  <button onclick="prev()">Previous</button>
  <button onclick="next()">Next</button>
  <button class="primary" onclick="saveDecision('accept_short_window_visual_continuity_edge')">A Accept</button>
  <button class="reject" onclick="saveDecision('reject_edge')">X Reject</button>
  <button class="warn" onclick="saveDecision('unsure_needs_later_review')">U Unsure</button>
</header>
<div class="wrap">
  <div class="media">
    <div class="panel"><div class="small">source still</div><img id="sourceContext"></div>
    <div class="panel"><div class="small">target still</div><img id="targetContext"></div>
    <div class="panel"><div class="small">source crop</div><img id="sourceCrop"></div>
    <div class="panel"><div class="small">target crop</div><img id="targetCrop"></div>
    <div class="panel burst">
      <div class="small">mini-burst evidence</div>
      <button onclick="playBurst()">Play</button><button onclick="pauseBurst()">Pause</button>
      <img id="burstClip">
      <div class="small">frame strip</div><img id="burstStrip">
      <div class="small">raw no-overlay strip</div><img id="burstRawStrip">
      <div class="small">overlay/no-overlay comparison strip</div><img id="burstComparisonStrip">
    </div>
  </div>
  <div>
    <div class="panel rule">Accept only if the highlighted person continues through the burst without a visible person swap. Reject if the edge jumps to a nearby person. Use unsure if the burst is too blurry/occluded.</div>
    <div class="panel"><div id="meta" class="meta"></div></div>
    <div class="panel">
      <label class="small">Reviewer</label><input id="reviewer" placeholder="reviewer name">
      <label class="small">Notes</label><textarea id="notes" rows="5"></textarea>
      <div id="saveStatus" class="small"></div>
    </div>
  </div>
</div>
<script>
const STATE = {state_json};
let index = 0;
let burstPlaying = true;
function row(){{ return STATE.rows[index] || {{}}; }}
function setImg(id, path){{ document.getElementById(id).src = path || ''; }}
function render(){{
  const r = row();
  const a = r.ui_assets || {{}};
  setImg('sourceContext', a.source_context_image);
  setImg('targetContext', a.target_context_image);
  setImg('sourceCrop', a.source_crop_image);
  setImg('targetCrop', a.target_crop_image);
  setImg('burstClip', a.burst_clip);
  setImg('burstStrip', a.burst_strip);
  setImg('burstRawStrip', a.burst_raw_strip);
  setImg('burstComparisonStrip', a.burst_overlay_comparison_strip);
  burstPlaying = true;
  document.getElementById('counter').textContent = `${{index+1}} / ${{STATE.rows.length}}`;
  document.getElementById('bucket').textContent = r.review_bucket || '';
  document.getElementById('saveStatus').textContent = '';
  document.getElementById('saveStatus').style.color = '#aebdc5';
  document.getElementById('meta').innerHTML = [
    ['candidate', r.step2m1r_review_candidate_id],
    ['edge', r.continuity_edge_id],
    ['frames', `${{r.source_frame_sequence}} -> ${{r.target_frame_sequence}}`],
    ['burst frames', (r.burst_frame_sequences || []).join(', ')],
    ['intermediate ids', (r.intermediate_candidate_ids || []).join(', ')],
    ['burst evidence', r.burst_evidence_available],
    ['overlay alignment valid', r.burst_overlay_alignment_valid],
    ['overlay version', r.current_overlay_version],
    ['source visible_person_base_id', r.source_visible_person_base_id],
    ['target visible_person_base_id', r.target_visible_person_base_id],
    ['source role', r.source_step1f3_final_visual_role_state],
    ['target role', r.target_step1f3_final_visual_role_state],
    ['score / uncertainty', `${{r.edge_score_sandbox}} / ${{r.uncertainty_score}}`],
    ['M1R bucket', r.step2m1r_review_bucket],
    ['reasons', (r.uncertainty_reasons || []).join(', ')],
  ].map(([k,v]) => `<div class="small">${{k}}</div><div>${{v ?? ''}}</div>`).join('');
}}
function playBurst(){{ const img = document.getElementById('burstClip'); const src = (row().ui_assets || {{}}).burst_clip || ''; img.src = src ? `${{src}}?t=${{Date.now()}}` : ''; burstPlaying = true; }}
function pauseBurst(){{ document.getElementById('burstClip').src = (row().ui_assets || {{}}).burst_strip || ''; burstPlaying = false; }}
function toggleBurst(){{ if (burstPlaying) pauseBurst(); else playBurst(); }}
function advanceAfterSave(){{ index = Math.min(STATE.rows.length - 1, index + 1); render(); }}
async function saveDecision(decision){{
  const r = row();
  const status = document.getElementById('saveStatus');
  status.textContent = 'saving...';
  status.style.color = '#f3d383';
  const payload = {{
    step2m1r_review_candidate_id:r.step2m1r_review_candidate_id,
    continuity_edge_id:r.continuity_edge_id,
    source_visible_person_base_id:r.source_visible_person_base_id,
    target_visible_person_base_id:r.target_visible_person_base_id,
    source_frame_sequence:r.source_frame_sequence,
    target_frame_sequence:r.target_frame_sequence,
    review_bucket:r.review_bucket,
    human_review_decision:decision,
    reviewer_name:document.getElementById('reviewer').value || '',
    notes:document.getElementById('notes').value || '',
    reviewed_at:new Date().toISOString()
  }};
  try {{
    const response = await fetch('/api/step2m1r/review-decision', {{method:'POST', headers:{{'content-type':'application/json'}}, body:JSON.stringify(payload)}});
    const data = await response.json();
    if (!response.ok || !data.success) throw new Error(data.error || 'save failed');
    r.saved_human_review_decision = decision;
    r.ui_is_reviewed = true;
    status.textContent = 'saved';
    status.style.color = '#8ee7a5';
    setTimeout(advanceAfterSave, 250);
  }} catch (error) {{
    status.textContent = `save failed: ${{error.message || error}}`;
    status.style.color = '#ff8f8f';
  }}
}}
function next(){{ index = Math.min(STATE.rows.length - 1, index + 1); render(); }}
function prev(){{ index = Math.max(0, index - 1); render(); }}
document.addEventListener('keydown', e => {{
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.key === ' ') {{ e.preventDefault(); toggleBurst(); }}
  else if (e.key === 'ArrowRight') next();
  else if (e.key === 'ArrowLeft') prev();
  else if (e.key.toLowerCase() === 'a') saveDecision('accept_short_window_visual_continuity_edge');
  else if (e.key.toLowerCase() === 'x') saveDecision('reject_edge');
  else if (e.key.toLowerCase() === 'u') saveDecision('unsure_needs_later_review');
}});
render();
</script>
</body>
</html>"""


def write_m1r_review_ui(candidate_payload: dict[str, Any]) -> None:
    write_text(STEP2M1R_REVIEW_UI_HTML_PATH, m1r_html_template(candidate_payload))
    render_m1r_contact_sheet(rows_from_payload(candidate_payload))


def build_adaptation_safety_manifest(
    *,
    edge_payload: dict[str, Any],
    group_payload: dict[str, Any],
    remediation_summary: dict[str, Any],
    review_payload: dict[str, Any],
) -> dict[str, Any]:
    forbidden = sorted(
        set(forbidden_keys_present(edge_payload))
        | set(forbidden_keys_present(group_payload))
        | set(forbidden_keys_present(remediation_summary))
        | set(forbidden_keys_present(review_payload))
    )
    review_summary = review_payload.get("summary", {})
    safe_auto_accept_audit_rows = safe_int(review_summary.get("safe_auto_accept_audit_rows"), 0)
    missing_burst_rate = safe_float(review_summary.get("burst_evidence_missing_rate"), 0.0)
    burst_overlay_alignment_safe = review_summary.get("burst_overlay_alignment_safe_for_review") is True
    current_overlay_version = str(review_summary.get("current_overlay_version", CURRENT_BURST_OVERLAY_VERSION))
    decisions_overlay_version_value = review_summary.get("review_decisions_collected_with_overlay_version", "")
    decisions_overlay_versions = (
        [str(value) for value in decisions_overlay_version_value]
        if isinstance(decisions_overlay_version_value, list)
        else [str(decisions_overlay_version_value)] if decisions_overlay_version_value else []
    )
    decisions_overlay_version_matches = review_summary.get("review_decisions_overlay_version_matches_current") is True
    reviewed_candidates = safe_int(review_summary.get("reviewed_candidates"), 0)
    total_review_candidates = len(rows_from_payload(review_payload))
    groups_over_cap_after = safe_int(remediation_summary.get("groups_over_cap_after"), 0)
    unsafe_reasons: list[str] = []
    safe_reasons: list[str] = []
    if groups_over_cap_after == 0:
        safe_reasons.append("adaptation_safe_group_artifact_has_no_groups_over_span_cap")
    else:
        unsafe_reasons.append("adaptation_safe_group_artifact_has_groups_over_span_cap")
    if safe_auto_accept_audit_rows >= SAFE_AUTO_ACCEPT_AUDIT_REQUIRED_FOR_M2:
        safe_reasons.append("targeted_review_set_contains_safe_auto_accept_audit_rows")
    else:
        unsafe_reasons.append("safe_auto_accept_audit_rows_below_minimum")
    if missing_burst_rate > BURST_MISSING_RATE_BLOCK_THRESHOLD:
        unsafe_reasons.append("burst_evidence_missing_for_more_than_five_percent_of_targeted_review_candidates")
    if burst_overlay_alignment_safe:
        safe_reasons.append("burst_overlay_alignment_safe_for_review")
    else:
        unsafe_reasons.append("burst_overlay_alignment_unsafe_for_review")
    if forbidden:
        unsafe_reasons.append("forbidden_keys_present")
    if reviewed_candidates != total_review_candidates:
        unsafe_reasons.append("targeted_second_review_incomplete")
    if review_payload.get("targeted_review_completed") is not True:
        unsafe_reasons.append("targeted_second_review_not_completed")
    elif not decisions_overlay_version_matches:
        unsafe_reasons.append("targeted_review_decisions_not_collected_with_current_overlay_version")
    safe_for_m2 = (
        groups_over_cap_after == 0
        and safe_auto_accept_audit_rows >= SAFE_AUTO_ACCEPT_AUDIT_REQUIRED_FOR_M2
        and not forbidden
        and missing_burst_rate <= BURST_MISSING_RATE_BLOCK_THRESHOLD
        and burst_overlay_alignment_safe
        and reviewed_candidates == total_review_candidates
        and review_payload.get("targeted_review_completed") is True
        and decisions_overlay_version_matches
    )
    manifest = guardrail_stamp(
        {
            "artifact": "step2m1r_adaptation_safety_manifest",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "safe_for_step2m2_adaptation_candidate": safe_for_m2,
            "safe_for_step2m2_adaptation_reasons": safe_reasons,
            "unsafe_for_step2m2_adaptation_reasons": unsafe_reasons,
            "adaptation_safe_edge_count": edge_payload.get("adaptation_safe_edge_count", len(rows_from_payload(edge_payload))),
            "adaptation_safe_group_count": group_payload.get("visual_continuity_group_rows", len(rows_from_payload(group_payload))),
            "excluded_edge_count": edge_payload.get("excluded_edge_count", edge_payload.get("summary", {}).get("excluded_edge_count", 0)),
            "excluded_group_count": remediation_summary.get("groups_excluded_from_adaptation", 0),
            "groups_over_cap_after": groups_over_cap_after,
            "safe_auto_accept_audit_rows": safe_auto_accept_audit_rows,
            "reviewed_candidates": reviewed_candidates,
            "total_review_candidates": total_review_candidates,
            "targeted_review_required_before_m2": review_payload.get("targeted_review_completed") is not True,
            "burst_evidence_missing_rate": missing_burst_rate,
            "burst_evidence_missing_blocks_step2m2_adaptation": missing_burst_rate > BURST_MISSING_RATE_BLOCK_THRESHOLD,
            "burst_overlay_alignment_safe_for_review": burst_overlay_alignment_safe,
            "current_overlay_version": current_overlay_version,
            "review_decisions_collected_with_overlay_version": decisions_overlay_versions,
            "review_decisions_overlay_version_matches_current": decisions_overlay_version_matches,
            "forbidden_keys_present": forbidden,
        }
    )
    assert_no_forbidden_keys(manifest)
    return manifest


def build_step2m1r_post_review_remediation() -> dict[str, Any]:
    node_payload = read_json(STEP2M1_NODE_ROWS_PATH)
    corrected_edge_payload = read_human_corrected_edge_payload()
    group_payload = read_json(STEP2M1_GROUP_ROWS_SANDBOX_PATH)
    edge_payload, _safe_edge_rows, bucket_pools = remediate_edges_for_adaptation(corrected_edge_payload, node_payload)
    compact_edge_payload_artifacts(
        edge_payload,
        legacy_json_path=STEP2M1R_ADAPTATION_SAFE_EDGE_ROWS_PATH,
        summary_path=STEP2M1R_ADAPTATION_SAFE_EDGE_SUMMARY_PATH,
        sample_path=STEP2M1R_ADAPTATION_SAFE_EDGE_SAMPLE_PATH,
        jsonl_gz_path=STEP2M1R_ADAPTATION_SAFE_EDGE_ROWS_JSONL_GZ_PATH,
    )
    safe_group_payload, group_summary, long_group_edge_ids = remediate_groups_for_adaptation(group_payload, node_payload, corrected_edge_payload)
    write_row_artifact_with_summary_and_sample(
        STEP2M1R_ADAPTATION_SAFE_GROUP_ROWS_PATH,
        STEP2M1R_ADAPTATION_SAFE_GROUP_SUMMARY_PATH,
        STEP2M1R_ADAPTATION_SAFE_GROUP_SAMPLE_PATH,
        safe_group_payload,
    )
    write_json(STEP2M1R_GROUP_SPAN_REMEDIATION_SUMMARY_PATH, group_summary)
    review_payload = build_targeted_review_candidates(bucket_pools, long_group_edge_ids, corrected_edge_payload)
    review_payload = render_m1r_burst_evidence(review_payload, node_payload)
    write_row_artifact_with_summary_and_sample(
        STEP2M1R_TARGETED_REVIEW_CANDIDATE_ROWS_PATH,
        STEP2M1R_TARGETED_REVIEW_CANDIDATE_SUMMARY_PATH,
        STEP2M1R_TARGETED_REVIEW_CANDIDATE_SAMPLE_PATH,
        review_payload,
    )
    write_m1r_review_ui(review_payload)
    reviewed_payload = read_m1r_reviewed_decisions()
    progress = m1r_review_progress_payload(review_payload, reviewed_payload)
    decision = m1r_review_decision_payload(review_payload, reviewed_payload)
    write_json(STEP2M1R_REVIEW_PROGRESS_SUMMARY_PATH, progress)
    write_json(STEP2M1R_REVIEW_DECISION_SUMMARY_PATH, decision)
    review_payload_for_gate = {
        **review_payload,
        "targeted_review_completed": progress.get("targeted_review_completed", False),
        "summary": {
            **review_payload.get("summary", {}),
            "reviewed_candidates": progress.get("reviewed_candidates", 0),
            "accepted_count": progress.get("accepted_count", 0),
            "rejected_count": progress.get("rejected_count", 0),
            "unsure_count": progress.get("unsure_count", 0),
            "review_decisions_collected_with_overlay_version": progress.get("review_decisions_collected_with_overlay_version", []),
            "review_decisions_overlay_version_matches_current": progress.get("review_decisions_overlay_version_matches_current", False),
        },
    }
    manifest = build_adaptation_safety_manifest(
        edge_payload=edge_payload,
        group_payload=safe_group_payload,
        remediation_summary=group_summary,
        review_payload=review_payload_for_gate,
    )
    write_json(STEP2M1R_ADAPTATION_SAFETY_MANIFEST_PATH, manifest)
    return {
        "adaptation_safe_edge_payload": edge_payload,
        "adaptation_safe_group_payload": safe_group_payload,
        "group_span_remediation_summary": group_summary,
        "targeted_review_payload": review_payload,
        "review_progress": progress,
        "review_decision": decision,
        "adaptation_safety_manifest": manifest,
    }


def print_step2m1r_console(outputs: dict[str, Any]) -> None:
    summary = outputs["group_span_remediation_summary"]
    review = outputs["targeted_review_payload"]
    manifest = outputs["adaptation_safety_manifest"]
    print(f"step2m1r_adaptation_safe_visual_continuity_group_rows_path: {STEP2M1R_ADAPTATION_SAFE_GROUP_ROWS_PATH.resolve()}")
    print(f"step2m1r_group_span_remediation_summary_path: {STEP2M1R_GROUP_SPAN_REMEDIATION_SUMMARY_PATH.resolve()}")
    print(f"step2m1r_targeted_review_candidate_rows_path: {STEP2M1R_TARGETED_REVIEW_CANDIDATE_ROWS_PATH.resolve()}")
    print(f"step2m1r_review_ui_html_path: {STEP2M1R_REVIEW_UI_HTML_PATH.resolve()}")
    print(f"step2m1r_review_contact_sheet_path: {STEP2M1R_REVIEW_CONTACT_SHEET_PATH.resolve()}")
    print(f"step2m1r_adaptation_safety_manifest_path: {STEP2M1R_ADAPTATION_SAFETY_MANIFEST_PATH.resolve()}")
    print(f"groups_over_cap_before: {summary.get('groups_over_cap_before', 0)}")
    print(f"groups_over_cap_after: {summary.get('groups_over_cap_after', 0)}")
    print(f"max_group_span_frames_after: {summary.get('max_group_span_frames_after', 0)}")
    print(f"max_group_span_seconds_after: {summary.get('max_group_span_seconds_after', 0.0)}")
    print(f"adaptation_safe_group_count: {summary.get('adaptation_safe_group_count', 0)}")
    print(f"groups_excluded_from_adaptation: {summary.get('groups_excluded_from_adaptation', 0)}")
    print(f"safe_auto_accept_audit_rows: {review.get('summary', {}).get('safe_auto_accept_audit_rows', 0)}")
    print(f"second_review_candidate_count: {len(rows_from_payload(review))}")
    print(f"bucket_counts: {json.dumps(review.get('summary', {}).get('bucket_counts', {}), sort_keys=True)}")
    print(f"candidates_with_valid_overlay_transforms: {review.get('summary', {}).get('candidates_with_valid_overlay_transforms', 0)}")
    print(f"candidates_with_invalid_overlay_transforms: {review.get('summary', {}).get('candidates_with_invalid_overlay_transforms', 0)}")
    print(f"burst_overlay_alignment_safe_for_review={str(review.get('summary', {}).get('burst_overlay_alignment_safe_for_review', False)).lower()}")
    print(f"step2m1r_burst_overlay_debug_rows_path: {STEP2M1R_BURST_OVERLAY_DEBUG_ROWS_PATH.resolve()}")
    print(f"step2m1r_burst_overlay_alignment_summary_path: {STEP2M1R_BURST_OVERLAY_ALIGNMENT_SUMMARY_PATH.resolve()}")
    print(f"overlay_qa_sample_paths: {json.dumps(review.get('summary', {}).get('overlay_qa_sample_paths', [])[:10])}")
    print(f"safe_for_step2m2_adaptation_candidate={str(manifest.get('safe_for_step2m2_adaptation_candidate', False)).lower()}")
    print(f"forbidden_keys_present: {manifest.get('forbidden_keys_present', [])}")
    print("production_ready=false")
    print("no_auto_promotion=true")


def prepare_step2m1r_review_ui(host: str = "127.0.0.1", port: int = 8784) -> dict[str, Any]:
    if not STEP2M1R_REVIEW_UI_HTML_PATH.exists() or not STEP2M1R_TARGETED_REVIEW_CANDIDATE_ROWS_PATH.exists():
        build_step2m1r_post_review_remediation()
    candidate_payload = read_json(STEP2M1R_TARGETED_REVIEW_CANDIDATE_ROWS_PATH)
    reviewed_payload = read_m1r_reviewed_decisions()
    progress, decision_summary, _manifest = refresh_m1r_progress_decision_and_gate(candidate_payload, reviewed_payload)
    manifest = guardrail_stamp(
        {
            "artifact": "step2m1r_review_ui_manifest",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "url": f"http://{host}:{port}/",
            "review_ui_html_path": str(STEP2M1R_REVIEW_UI_HTML_PATH.resolve()),
            "review_candidate_rows_path": str(STEP2M1R_TARGETED_REVIEW_CANDIDATE_ROWS_PATH.resolve()),
            "reviewed_decisions_path": str(STEP2M1R_REVIEWED_DECISIONS_PATH.resolve()),
            "review_progress_summary_path": str(STEP2M1R_REVIEW_PROGRESS_SUMMARY_PATH.resolve()),
            "review_decision_summary_path": str(STEP2M1R_REVIEW_DECISION_SUMMARY_PATH.resolve()),
            "review_contact_sheet_path": str(STEP2M1R_REVIEW_CONTACT_SHEET_PATH.resolve()),
            "total_review_candidates": progress.get("total_review_candidates", len(rows_from_payload(candidate_payload))),
            "reviewed_candidates": progress.get("reviewed_candidates", 0),
            "current_overlay_version": CURRENT_BURST_OVERLAY_VERSION,
            "review_decisions_overlay_version_matches_current": progress.get("review_decisions_overlay_version_matches_current", False),
            "targeted_review_completed": decision_summary.get("targeted_review_completed", False),
        }
    )
    assert_no_forbidden_keys(manifest)
    return manifest


def print_step2m1r_ui_console(manifest: dict[str, Any]) -> None:
    print(f"step2m1r_review_ui_html_path: {manifest['review_ui_html_path']}")
    print(f"step2m1r_targeted_review_candidate_rows_path: {manifest['review_candidate_rows_path']}")
    print(f"step2m1r_reviewed_visual_continuity_decisions_path: {manifest['reviewed_decisions_path']}")
    print(f"step2m1r_review_progress_summary_path: {manifest['review_progress_summary_path']}")
    print(f"step2m1r_review_decision_summary_path: {manifest['review_decision_summary_path']}")
    print(f"step2m1r_review_contact_sheet_path: {manifest['review_contact_sheet_path']}")
    print(f"total_review_candidates: {manifest['total_review_candidates']}")
    print(f"reviewed_candidates: {manifest['reviewed_candidates']}")
    print(f"current_overlay_version: {manifest['current_overlay_version']}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("no_auto_promotion=true")


class Step2M1RReviewHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path.lstrip("/"))
        file_path = STEP2M1R_REVIEW_UI_HTML_PATH if not path else (STEP2M1_OUTPUT_DIR / path).resolve()
        if not str(file_path).startswith(str(STEP2M1_OUTPUT_DIR.resolve())):
            self.send_error(403)
            return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("content-type", mimetypes.guess_type(str(file_path))[0] or "application/octet-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_decision_save(self) -> None:
        if urlparse(self.path).path != "/api/step2m1r/review-decision":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            decision, reviewed_payload, progress = save_m1r_review_decision(payload)
        except ValueError as exc:
            self._send_json({"success": False, "error": str(exc)}, status=400)
            return
        except json.JSONDecodeError as exc:
            self._send_json({"success": False, "error": f"invalid_json: {exc}"}, status=400)
            return
        self._send_json(
            {
                "success": True,
                "decision": decision,
                "reviewed_decision_rows": reviewed_payload.get("reviewed_decision_rows", len(rows_from_payload(reviewed_payload))),
                "reviewed_candidates": progress.get("reviewed_candidates", 0),
                "targeted_review_completed": progress.get("targeted_review_completed", False),
            }
        )

    def do_POST(self) -> None:  # noqa: N802
        self._handle_decision_save()

    def do_PUT(self) -> None:  # noqa: N802
        self._handle_decision_save()


def serve_step2m1r_review_ui(host: str = "127.0.0.1", port: int = 8784) -> None:
    prepare_step2m1r_review_ui(host=host, port=port)
    server = ThreadingHTTPServer((host, port), Step2M1RReviewHandler)
    print(f"Serving Step2.M1R targeted review UI at http://{host}:{port}/")
    server.serve_forever()
