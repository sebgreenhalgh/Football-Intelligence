# ruff: noqa: E501

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C1_COLOUR_FEATURE_ROWS_PATH,
    STEP1C1C_SEEDED_COLOUR_BELIEF_ROWS_SANDBOX_PATH,
    STEP1C2_SHORT_BURST_COLOUR_GROUP_ROWS_PATH,
    read_json,
    write_json,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    bbox_area,
    bbox_from_item,
    bbox_iou,
    safe_float,
)


TEAM_BELIEFS = {"team_1_outfield_colour_like", "team_2_outfield_colour_like"}
UNKNOWN_BELIEFS = {"unknown_ambiguous_colour", "ambiguous_outfield_colour", "crop_unusable"}
CONTEXT_BELIEFS = {"non_outfield_context_colour", "dark_context_colour_like", "other_distinct_colour_like"}
MAX_FRAME_GAP = 2
MAX_GROUP_SPAN_FRAMES = 7


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def point_gap(left: dict[str, Any] | None, right: dict[str, Any] | None) -> float:
    if not left or not right or left.get("x") is None or right.get("x") is None:
        return 1_000_000.0
    return float(((safe_float(left.get("x")) - safe_float(right.get("x"))) ** 2 + (safe_float(left.get("y")) - safe_float(right.get("y"))) ** 2) ** 0.5)


def bbox_center(row: dict[str, Any]) -> dict[str, Any] | None:
    bbox = bbox_from_item(row)
    if not bbox:
        return None
    return {
        "x": (safe_float(bbox.get("x1")) + safe_float(bbox.get("x2"))) / 2.0,
        "y": (safe_float(bbox.get("y1")) + safe_float(bbox.get("y2"))) / 2.0,
    }


def size_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_area = bbox_area(bbox_from_item(left))
    right_area = bbox_area(bbox_from_item(right))
    if left_area <= 0 or right_area <= 0:
        return 0.0
    return round(min(left_area, right_area) / max(left_area, right_area), 4)


def feature_index(feature_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("visible_person_base_id", "")): row for row in feature_payload.get("rows", [])}


def hsv_gap(left_feature: dict[str, Any], right_feature: dict[str, Any]) -> float:
    left = left_feature.get("median_hsv", [])
    right = right_feature.get("median_hsv", [])
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != 3 or len(right) != 3:
        return 1_000_000.0
    hue_delta = abs(safe_float(left[0]) - safe_float(right[0]))
    hue_delta = min(hue_delta, 180.0 - hue_delta) * 1.4
    sat_delta = (safe_float(left[1]) - safe_float(right[1])) * 0.35
    val_delta = (safe_float(left[2]) - safe_float(right[2])) * 0.25
    return float((hue_delta * hue_delta + sat_delta * sat_delta + val_delta * val_delta) ** 0.5)


def belief_compatible(left_belief: str, right_belief: str) -> bool:
    if left_belief == right_belief:
        return True
    if left_belief in TEAM_BELIEFS and right_belief in TEAM_BELIEFS:
        return False
    if left_belief in CONTEXT_BELIEFS and right_belief in TEAM_BELIEFS:
        return False
    if right_belief in CONTEXT_BELIEFS and left_belief in TEAM_BELIEFS:
        return False
    return left_belief in UNKNOWN_BELIEFS or right_belief in UNKNOWN_BELIEFS


def link_features(
    left: dict[str, Any],
    right: dict[str, Any],
    features_by_base_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    features_by_base_id = features_by_base_id or {}
    left_feature = features_by_base_id.get(str(left.get("visible_person_base_id", "")), {})
    right_feature = features_by_base_id.get(str(right.get("visible_person_base_id", "")), {})
    return {
        "frame_gap": int(safe_float(right.get("frame_sequence"), -1)) - int(safe_float(left.get("frame_sequence"), -1)),
        "bbox_iou": round(bbox_iou(bbox_from_item(left), bbox_from_item(right)), 4),
        "centre_pixel_gap": round(point_gap(bbox_center(left), bbox_center(right)), 3),
        "footpoint_pixel_gap": round(point_gap(left.get("footpoint"), right.get("footpoint")), 3),
        "bbox_size_ratio": size_ratio(left, right),
        "seed_belief_compatible": belief_compatible(
            str(left.get("seed_team_colour_belief", "")),
            str(right.get("seed_team_colour_belief", "")),
        ),
        "crop_hsv_gap": round(hsv_gap(left_feature, right_feature), 3),
        "same_source_detection_id": bool(
            left.get("source_detection_id")
            and right.get("source_detection_id")
            and left.get("source_detection_id") == right.get("source_detection_id")
        ),
    }


def link_score(features: dict[str, Any]) -> float:
    visual_gap = min(safe_float(features.get("centre_pixel_gap"), 1_000_000.0), safe_float(features.get("footpoint_pixel_gap"), 1_000_000.0))
    score = visual_gap - (safe_float(features.get("bbox_iou")) * 80.0)
    if features.get("same_source_detection_id"):
        score -= 35.0
    if safe_float(features.get("crop_hsv_gap"), 1_000_000.0) < 28.0:
        score -= 8.0
    return round(score, 4)


def can_link(features: dict[str, Any]) -> bool:
    if int(features.get("frame_gap", 99)) < 1 or int(features.get("frame_gap", 99)) > MAX_FRAME_GAP:
        return False
    if features.get("seed_belief_compatible") is not True:
        return False
    if safe_float(features.get("bbox_size_ratio")) < 0.35:
        return False
    if features.get("same_source_detection_id"):
        return True
    if safe_float(features.get("bbox_iou")) >= 0.06:
        return True
    if safe_float(features.get("centre_pixel_gap"), 1_000_000.0) <= 48.0:
        return True
    if safe_float(features.get("footpoint_pixel_gap"), 1_000_000.0) <= 54.0:
        return True
    return safe_float(features.get("crop_hsv_gap"), 1_000_000.0) <= 18.0 and safe_float(features.get("centre_pixel_gap"), 1_000_000.0) <= 72.0


def row_sort_key(row: dict[str, Any]) -> tuple[int, float, float, str]:
    centre = bbox_center(row) or {}
    return (
        int(safe_float(row.get("frame_sequence"), -1)),
        safe_float(centre.get("x")),
        safe_float(centre.get("y")),
        str(row.get("visible_person_base_id", "")),
    )


def grouped_member_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "visible_person_base_id": row.get("visible_person_base_id", ""),
        "frame_sequence": int(safe_float(row.get("frame_sequence"), -1)),
        "seed_team_colour_belief": row.get("seed_team_colour_belief", ""),
        "seed_team_colour_belief_confidence": safe_float(row.get("seed_team_colour_belief_confidence")),
    }


def build_group_row(group_id: str, members: list[dict[str, Any]]) -> dict[str, Any]:
    sequences = [int(safe_float(row.get("frame_sequence"), -1)) for row in members]
    belief_counts = Counter(str(row.get("seed_team_colour_belief", "")) for row in members)
    dominant_belief, dominant_count = ("", 0)
    if belief_counts:
        dominant_belief, dominant_count = sorted(belief_counts.items(), key=lambda item: (-item[1], item[0]))[0]
    review_required_count = sum(1 for row in members if row.get("seed_review_required") is True or row.get("review_required") is True)
    confidence_values = [safe_float(row.get("seed_team_colour_belief_confidence")) for row in members]
    purity = dominant_count / max(1, len(members))
    avg_confidence = sum(confidence_values) / max(1, len(confidence_values))
    group_confidence = round(max(0.05, min(0.98, avg_confidence * purity)), 4)
    return {
        "short_burst_colour_group_id": group_id,
        "group_first_frame_sequence": min(sequences),
        "group_last_frame_sequence": max(sequences),
        "group_frame_count": len(set(sequences)),
        "group_row_count": len(members),
        "visible_person_base_ids": [str(row.get("visible_person_base_id", "")) for row in members],
        "dominant_seed_team_colour_belief": dominant_belief,
        "dominant_belief_count": dominant_count,
        "group_belief_counts": dict(sorted(belief_counts.items())),
        "group_unknown_count": sum(belief_counts.get(label, 0) for label in UNKNOWN_BELIEFS),
        "group_review_required_count": review_required_count,
        "group_confidence": group_confidence,
        "group_member_sample": [grouped_member_payload(row) for row in members[:12]],
        "max_frame_gap": MAX_FRAME_GAP,
        "max_group_span_frames": MAX_GROUP_SPAN_FRAMES,
        "grouping_evidence": "adjacent_or_near_adjacent_image_space_bbox_footpoint_colour_continuity_only",
        "local_group_not_identity": True,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def build_short_burst_colour_group_payload(
    seeded_belief_payload: dict[str, Any],
    feature_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    features_by_base_id = feature_index(feature_payload or {"rows": []})
    groups: list[dict[str, Any]] = []
    row_to_group_index: dict[str, int] = {}
    for row in sorted(seeded_belief_payload.get("rows", []), key=row_sort_key):
        row_seq = int(safe_float(row.get("frame_sequence"), -1))
        best_index = -1
        best_score = 1_000_000.0
        for index, group in enumerate(groups):
            first_seq = int(group["first_frame_sequence"])
            last_seq = int(group["last_frame_sequence"])
            if row_seq - last_seq < 1 or row_seq - last_seq > MAX_FRAME_GAP:
                continue
            if row_seq - first_seq + 1 > MAX_GROUP_SPAN_FRAMES:
                continue
            features = link_features(group["members"][-1], row, features_by_base_id)
            if not can_link(features):
                continue
            score = link_score(features)
            if score < best_score:
                best_score = score
                best_index = index
        if best_index < 0:
            groups.append(
                {
                    "first_frame_sequence": row_seq,
                    "last_frame_sequence": row_seq,
                    "members": [row],
                }
            )
            row_to_group_index[str(row.get("visible_person_base_id", ""))] = len(groups) - 1
        else:
            groups[best_index]["members"].append(row)
            groups[best_index]["last_frame_sequence"] = row_seq
            row_to_group_index[str(row.get("visible_person_base_id", ""))] = best_index
    indices_by_first_seq: dict[int, int] = defaultdict(int)
    group_rows = []
    row_group_ids: dict[str, str] = {}
    for group in sorted(groups, key=lambda item: (int(item["first_frame_sequence"]), str(item["members"][0].get("visible_person_base_id", "")))):
        first_seq = int(group["first_frame_sequence"])
        group_index = indices_by_first_seq[first_seq]
        indices_by_first_seq[first_seq] += 1
        group_id = f"step1c2_sbcg_f{first_seq:06d}_{group_index:03d}"
        for member in group["members"]:
            row_group_ids[str(member.get("visible_person_base_id", ""))] = group_id
        group_rows.append(build_group_row(group_id, group["members"]))
    return {
        "artifact": "step1c2_short_burst_colour_group_rows",
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
        "short_burst_grouping_only": True,
        "max_frame_gap": MAX_FRAME_GAP,
        "max_group_span_frames": MAX_GROUP_SPAN_FRAMES,
        "row_group_ids": row_group_ids,
        "rows": group_rows,
        "summary": {
            "c1c_seeded_belief_row_count": len(seeded_belief_payload.get("rows", [])),
            "short_burst_colour_group_count": len(group_rows),
            "singleton_group_count": sum(1 for row in group_rows if row.get("group_row_count") == 1),
            "multi_row_group_count": sum(1 for row in group_rows if row.get("group_row_count", 0) > 1),
            "max_group_frame_count": max((int(row.get("group_frame_count", 0)) for row in group_rows), default=0),
            "max_group_row_count": max((int(row.get("group_row_count", 0)) for row in group_rows), default=0),
        },
    }


def build_and_write_short_burst_colour_groups() -> dict[str, Any]:
    seeded_payload = read_json(STEP1C1C_SEEDED_COLOUR_BELIEF_ROWS_SANDBOX_PATH)
    feature_payload = read_json(STEP1C1_COLOUR_FEATURE_ROWS_PATH)
    payload = build_short_burst_colour_group_payload(seeded_payload, feature_payload)
    write_json(STEP1C2_SHORT_BURST_COLOUR_GROUP_ROWS_PATH, payload)
    return payload
