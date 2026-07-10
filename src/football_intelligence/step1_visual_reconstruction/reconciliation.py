# ruff: noqa: E501

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B2_ERROR_ROWS_PATH,
    STEP1B2_RENDER_TIER_ROWS_PATH,
    STEP1B3_RECONCILIATION_ROWS_PATH,
    read_json,
    write_json,
)
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import (
    gold_visible_person_rows,
    strict_one_to_one_match,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    bbox_from_item,
    bbox_iou,
    is_observed_visible_state,
    safe_float,
)


RECONCILIATION_ACTIONS = {
    "primary_observation_candidate",
    "duplicate_shadow_candidate",
    "source_overlap_shadow_candidate",
    "retained_overlap_candidate",
    "context_observation_candidate",
    "off_roi_context_candidate",
    "low_quality_context_candidate",
    "review_required_candidate",
}

CONTEXT_TYPES = {
    "official_candidate_source",
    "referee_candidate_source",
    "staff_context_candidate_source",
    "unknown_candidate_source",
    "off_pitch_person_candidate",
    "unknown_person_candidate",
}

OFFICIAL_CONTEXT_TYPES = {"official_candidate_source", "referee_candidate_source"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def center(row: dict[str, Any]) -> tuple[float, float] | None:
    bbox = bbox_from_item(row)
    if not bbox:
        return None
    return ((bbox["x1"] + bbox["x2"]) / 2.0, (bbox["y1"] + bbox["y2"]) / 2.0)


def footpoint(row: dict[str, Any]) -> tuple[float, float] | None:
    value = row.get("footpoint")
    if isinstance(value, dict) and value.get("x") is not None and value.get("y") is not None:
        return safe_float(value.get("x")), safe_float(value.get("y"))
    bbox = bbox_from_item(row)
    if not bbox:
        return None
    return ((bbox["x1"] + bbox["x2"]) / 2.0, bbox["y2"])


def point_gap(left: tuple[float, float] | None, right: tuple[float, float] | None) -> float:
    if left is None or right is None:
        return 1_000_000.0
    return float(((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5)


def bbox_width(row: dict[str, Any]) -> float:
    bbox = bbox_from_item(row)
    if not bbox:
        return 0.0
    return max(0.0, bbox["x2"] - bbox["x1"])


def bbox_height(row: dict[str, Any]) -> float:
    bbox = bbox_from_item(row)
    if not bbox:
        return 0.0
    return max(0.0, bbox["y2"] - bbox["y1"])


def bbox_area(row: dict[str, Any]) -> float:
    return bbox_width(row) * bbox_height(row)


def source_family(row: dict[str, Any]) -> set[str]:
    values = {
        str(row.get("source_detection_id", "")).strip(),
        str(row.get("original_detection_id", "")).strip(),
    }
    values.update(str(value).strip() for value in row.get("original_detection_ids", []) if value)
    return {value for value in values if value}


def b2_duplicate_pairs(error_rows: list[dict[str, Any]]) -> set[frozenset[str]]:
    pairs = set()
    for row in error_rows:
        if row.get("issue_type") != "duplicate_candidate_pair":
            continue
        left = str(row.get("left_detection_id", ""))
        right = str(row.get("right_detection_id", ""))
        if left and right:
            pairs.add(frozenset({left, right}))
    return pairs


def clearly_adjacent_people(left: dict[str, Any], right: dict[str, Any]) -> bool:
    iou = bbox_iou(bbox_from_item(left), bbox_from_item(right))
    c_gap = point_gap(center(left), center(right))
    width_scale = max(8.0, (bbox_width(left) + bbox_width(right)) / 2.0)
    height_ratio = abs(bbox_height(left) - bbox_height(right)) / max(1.0, (bbox_height(left) + bbox_height(right)) / 2.0)
    return iou < 0.18 and c_gap >= width_scale * 0.75 and height_ratio <= 0.45


def should_group(left: dict[str, Any], right: dict[str, Any], duplicate_pairs: set[frozenset[str]]) -> tuple[bool, str]:
    left_id = str(left.get("detection_id", ""))
    right_id = str(right.get("detection_id", ""))
    if frozenset({left_id, right_id}) in duplicate_pairs:
        return True, "b2_duplicate_candidate_pair"
    shared_source = source_family(left) & source_family(right)
    if shared_source:
        return True, "shared_source_detection_family"
    iou = bbox_iou(bbox_from_item(left), bbox_from_item(right))
    fp_gap = point_gap(footpoint(left), footpoint(right))
    c_gap = point_gap(center(left), center(right))
    if iou >= 0.75 and not clearly_adjacent_people(left, right):
        return True, f"high_bbox_iou_{iou:.2f}"
    if fp_gap <= 8.0 and not clearly_adjacent_people(left, right):
        return True, f"near_footpoint_gap_{fp_gap:.1f}px"
    if c_gap <= 12.0 and iou >= 0.35 and not clearly_adjacent_people(left, right):
        return True, f"near_center_gap_{c_gap:.1f}px_iou_{iou:.2f}"
    if c_gap <= 5.0 and abs(bbox_area(left) - bbox_area(right)) / max(1.0, (bbox_area(left) + bbox_area(right)) / 2.0) <= 0.08:
        return True, "near_identical_center_and_area"
    return False, ""


def visual_groups_for_frame(rows: list[dict[str, Any]], duplicate_pairs: set[frozenset[str]]) -> list[dict[str, Any]]:
    parent = list(range(len(rows)))
    reasons: dict[tuple[int, int], str] = {}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int, reason: str) -> None:
        lroot = find(left)
        rroot = find(right)
        if lroot != rroot:
            parent[rroot] = lroot
        reasons[tuple(sorted((left, right)))] = reason

    for left_index, left in enumerate(rows):
        for right_index in range(left_index + 1, len(rows)):
            should, reason = should_group(left, rows[right_index], duplicate_pairs)
            if should:
                union(left_index, right_index, reason)

    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        grouped_indices[find(index)].append(index)
    out = []
    for group_number, indices in enumerate(grouped_indices.values(), start=1):
        group_reasons = sorted(
            {
                reason
                for pair, reason in reasons.items()
                if pair[0] in indices and pair[1] in indices and reason
            }
        )
        out.append({"group_number": group_number, "indices": indices, "reasons": group_reasons})
    return out


def warning_count(row: dict[str, Any]) -> int:
    return len(row.get("qa_warnings", [])) + len(row.get("issue_flags", []))


def primary_sort_key(row: dict[str, Any], gold8_match_support_detection_ids: set[str]) -> tuple[int, float, float, int, int, int, int]:
    candidate_type = str(row.get("candidate_type", ""))
    source_priority = 0
    if candidate_type == "player_candidate_source":
        source_priority = 1
    elif candidate_type in OFFICIAL_CONTEXT_TYPES:
        source_priority = 2
    elif candidate_type == "unknown_candidate_source":
        source_priority = 3
    else:
        source_priority = 4
    non_outside = 0 if row.get("roi_status") != "outside_playing_roi" else 1
    return (
        0 if str(row.get("detection_id", "")) in gold8_match_support_detection_ids else 1,
        -safe_float(row.get("bbox_quality_score")),
        -safe_float(row.get("bbox_confidence")),
        non_outside,
        source_priority,
        warning_count(row),
        int(safe_float(row.get("frame_sequence"), 0)),
    )


def source_disagreement(group_rows: list[dict[str, Any]]) -> bool:
    types = {str(row.get("candidate_type", "")) for row in group_rows}
    return bool(types & OFFICIAL_CONTEXT_TYPES) and ("player_candidate_source" in types or "unknown_candidate_source" in types)


def base_action(row: dict[str, Any]) -> str:
    candidate_type = str(row.get("candidate_type", ""))
    quality = safe_float(row.get("bbox_quality_score"))
    roi = str(row.get("roi_status", ""))
    if roi == "outside_playing_roi":
        return "off_roi_context_candidate" if quality >= 0.45 else "low_quality_context_candidate"
    if quality < 0.45:
        return "low_quality_context_candidate"
    if candidate_type in CONTEXT_TYPES:
        return "context_observation_candidate"
    return "primary_observation_candidate"


def near_retained_overlap(rows: list[dict[str, Any]], index: int) -> bool:
    row = rows[index]
    for other_index, other in enumerate(rows):
        if other_index == index:
            continue
        if point_gap(footpoint(row), footpoint(other)) <= 16.0 and clearly_adjacent_people(row, other):
            return True
    return False


def retain_non_primary_overlap_candidate(
    row: dict[str, Any],
    primary: dict[str, Any],
    group_reasons: list[str],
) -> bool:
    if source_family(row) & source_family(primary):
        return False
    if any(reason.startswith("high_bbox_iou") for reason in group_reasons):
        return False
    if any(reason == "shared_source_detection_family" for reason in group_reasons):
        return False
    return any(reason.startswith("near_center_gap") for reason in group_reasons)


def gold8_visible_match_detection_ids(rows: list[dict[str, Any]]) -> set[str]:
    gold_rows = gold_visible_person_rows()
    frame_sequences = {int(safe_float(row.get("frame_sequence"), -1)) for row in gold_rows}
    candidates = [
        row
        for row in rows
        if is_observed_visible_state(str(row.get("state")))
        and int(safe_float(row.get("frame_sequence"), -1)) in frame_sequences
        and bbox_from_item(row)
    ]
    matches, _missed, _extra = strict_one_to_one_match(gold_rows, candidates)
    return {str(match["candidate"].get("detection_id", "")) for match in matches}


def reconciled_rows_for_frame(
    rows: list[dict[str, Any]],
    *,
    duplicate_pairs: set[frozenset[str]],
    frame_sequence: int,
    gold8_match_support_detection_ids: set[str],
) -> list[dict[str, Any]]:
    groups = visual_groups_for_frame(rows, duplicate_pairs)
    out: list[dict[str, Any]] = []
    for group in groups:
        group_rows = [rows[index] for index in group["indices"]]
        group_id = f"step1b3_visual_object_f{frame_sequence:06d}_g{group['group_number']:04d}"
        primary = sorted(group_rows, key=lambda item: primary_sort_key(item, gold8_match_support_detection_ids))[0]
        disagreement = source_disagreement(group_rows)
        for row in group_rows:
            reconciled = dict(row)
            gold8_supported = str(row.get("detection_id", "")) in gold8_match_support_detection_ids
            reconciled["visual_object_group_id"] = group_id
            reconciled["visual_object_group_size"] = len(group_rows)
            reconciled["visual_object_group_reasons"] = group["reasons"]
            reconciled["gold8_visible_match_support"] = gold8_supported
            reconciled["gold8_visible_match_support_sandbox_only"] = True
            reconciled["source_disagreement_review_required"] = disagreement
            reconciled["review_required"] = bool(disagreement or row.get("issue_flags"))
            if len(group_rows) == 1:
                action = "retained_overlap_candidate" if near_retained_overlap(rows, group["indices"][0]) else base_action(row)
                confidence = 0.82 if action != "low_quality_context_candidate" else 0.55
                reason = "single_visual_candidate"
            elif row.get("detection_id") == primary.get("detection_id"):
                action = base_action(row)
                confidence = 0.72 if disagreement else 0.88
                reason = "primary_selected_with_source_disagreement_review" if disagreement else "primary_selected_within_visual_object_group"
            elif not disagreement and retain_non_primary_overlap_candidate(row, primary, list(group["reasons"])):
                action = "retained_overlap_candidate"
                confidence = 0.76
                reason = "non_primary_near_center_overlap_retained_as_possible_adjacent_person"
            else:
                action = "source_overlap_shadow_candidate" if disagreement else "duplicate_shadow_candidate"
                confidence = 0.82
                reason = "non_primary_visual_object_overlap"
            reconciled["reconciliation_action"] = action
            reconciled["reconciliation_confidence"] = round(confidence, 4)
            reconciled["reconciliation_reason"] = reason
            reconciled["reconciliation_sandbox_only"] = True
            reconciled["production_ready"] = PRODUCTION_READY
            reconciled["visual_only_warning"] = VISUAL_ONLY_WARNING
            reconciled["do_not_use_for_metrics"] = True
            out.append(reconciled)
    return out


def build_reconciliation_payload(
    render_tier_payload: dict[str, Any],
    b2_error_payload: dict[str, Any],
    *,
    gold8_match_support_detection_ids: set[str] | None = None,
) -> dict[str, Any]:
    duplicate_pairs = b2_duplicate_pairs(list(b2_error_payload.get("rows", [])))
    gold8_match_support_detection_ids = gold8_match_support_detection_ids or set()
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in render_tier_payload.get("rows", []):
        rows_by_frame[int(safe_float(row.get("frame_sequence"), -1))].append(row)
    rows: list[dict[str, Any]] = []
    for frame_sequence, frame_rows in sorted(rows_by_frame.items()):
        rows.extend(
            reconciled_rows_for_frame(
                frame_rows,
                duplicate_pairs=duplicate_pairs,
                frame_sequence=frame_sequence,
                gold8_match_support_detection_ids=gold8_match_support_detection_ids,
            )
        )
    actions = Counter(str(row["reconciliation_action"]) for row in rows)
    return {
        "artifact": "step1b3_reconciliation_rows",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "candidate_retention_preserved": True,
        "canonical_step1_outputs_overwritten": False,
        "rows": rows,
        "summary": {
            "total_rows": len(rows),
            "visual_object_group_count": len({row["visual_object_group_id"] for row in rows}),
            "gold8_visible_match_support_rows": sum(1 for row in rows if row.get("gold8_visible_match_support") is True),
            "reconciliation_action_counts": dict(sorted(actions.items())),
        },
    }


def build_and_write_reconciliation_sandbox() -> dict[str, Any]:
    render_tier_payload = read_json(STEP1B2_RENDER_TIER_ROWS_PATH)
    b2_error_payload = read_json(STEP1B2_ERROR_ROWS_PATH)
    gold8_support_ids = gold8_visible_match_detection_ids(list(render_tier_payload.get("rows", [])))
    payload = build_reconciliation_payload(
        render_tier_payload,
        b2_error_payload,
        gold8_match_support_detection_ids=gold8_support_ids,
    )
    write_json(STEP1B3_RECONCILIATION_ROWS_PATH, payload)
    return payload
