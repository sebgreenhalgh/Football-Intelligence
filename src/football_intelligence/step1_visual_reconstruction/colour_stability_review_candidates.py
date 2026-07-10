# ruff: noqa: E501

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C2_COLOUR_FLIP_AUDIT_ROWS_PATH,
    STEP1C2_COLOUR_STABILITY_ROWS_PATH,
    STEP1C2_GOLD8_COLOUR_STABILITY_EVAL_SUMMARY_PATH,
    STEP1C2_SHORT_BURST_COLOUR_GROUP_ROWS_PATH,
    STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH,
    STEP1C2B_REVIEW_CANDIDATE_SUMMARY_PATH,
    read_json,
    write_json,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    safe_float,
)


ALWAYS_INCLUDE_FRAMES = {59, 60, 61, 62}
ALWAYS_INCLUDE_FLIP_TYPES = {
    "unknown_to_team_colour",
    "review_required_conflict",
    "team_1_to_team_2_blocked",
    "team_2_to_team_1_blocked",
}
SAMPLED_FLIP_LIMITS = {
    "context_force_blocked": 40,
    "team_colour_to_other_distinct_blocked": 40,
}
RETAINED_TEAM_LIMIT = 40
RETAINED_UNKNOWN_CONTEXT_OTHER_LIMIT = 40

PRIORITY_BY_REASON = {
    "changed_by_c2": 10,
    "unknown_to_team_colour": 12,
    "c2_review_required": 14,
    "review_required_conflict": 16,
    "team_1_to_team_2_blocked": 18,
    "team_2_to_team_1_blocked": 18,
    "frame_59_62_manual_followup": 20,
    "context_force_blocked_sample": 42,
    "team_colour_to_other_distinct_blocked_sample": 44,
    "retained_team_1_sample": 60,
    "retained_team_2_sample": 62,
    "retained_unknown_context_other_sample": 64,
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def stable_hash(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def index_by_base_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("visible_person_base_id", "")): row for row in payload.get("rows", [])}


def sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(safe_float(row.get("frame_sequence"), -1)),
            str(row.get("flip_type", "")),
            str(row.get("visible_person_base_id", "")),
        ),
    )


def round_robin_sample(groups: dict[str, list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    sorted_groups = {key: sorted_rows(rows) for key, rows in sorted(groups.items())}
    offset = 0
    while len(selected) < limit:
        added = False
        for key in sorted(sorted_groups):
            rows = sorted_groups[key]
            if offset < len(rows):
                selected.append(rows[offset])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def add_reason(selected: dict[str, dict[str, Any]], row: dict[str, Any], reason: str) -> None:
    base_id = str(row.get("visible_person_base_id", ""))
    if not base_id:
        return
    priority = PRIORITY_BY_REASON[reason]
    if base_id not in selected:
        selected[base_id] = {"row": row, "review_reasons": [reason], "review_priority": priority}
        return
    item = selected[base_id]
    if reason not in item["review_reasons"]:
        item["review_reasons"].append(reason)
    item["review_priority"] = min(int(item["review_priority"]), priority)


def build_candidate_row(index: int, stability_row: dict[str, Any], flip_row: dict[str, Any], reasons: list[str], priority: int) -> dict[str, Any]:
    frame_sequence = int(safe_float(stability_row.get("frame_sequence"), -1))
    base_id = str(stability_row.get("visible_person_base_id", ""))
    return {
        "c2b_review_candidate_id": f"step1c2b_review_f{frame_sequence:06d}_{index:04d}_{stable_hash(base_id)}",
        "review_priority": priority,
        "review_reason": ";".join(sorted(reasons, key=lambda reason: PRIORITY_BY_REASON.get(reason, 99))),
        "review_reason_tags": sorted(reasons, key=lambda reason: PRIORITY_BY_REASON.get(reason, 99)),
        "visible_person_base_id": base_id,
        "frame_sequence": frame_sequence,
        "timestamp_seconds": safe_float(stability_row.get("timestamp_seconds")),
        "detection_id": stability_row.get("detection_id", ""),
        "source_detection_id": stability_row.get("source_detection_id", ""),
        "bbox": stability_row.get("bbox", {}),
        "footpoint": stability_row.get("footpoint", {}),
        "state": stability_row.get("state", ""),
        "roi_status": stability_row.get("roi_status", ""),
        "candidate_type": stability_row.get("candidate_type", ""),
        "original_role_source": stability_row.get("original_role_source", ""),
        "crop_quality": stability_row.get("crop_quality", ""),
        "crop_quality_reason": stability_row.get("crop_quality_reason", ""),
        "torso_crop_bbox": stability_row.get("torso_crop_bbox"),
        "c1c_seed_team_colour_belief": stability_row.get("c1c_seed_team_colour_belief", ""),
        "c1c_seed_team_colour_belief_confidence": safe_float(stability_row.get("c1c_seed_team_colour_belief_confidence")),
        "c2_stable_colour_belief": stability_row.get("c2_stable_colour_belief", ""),
        "c2_stable_colour_belief_confidence": safe_float(stability_row.get("c2_stable_colour_belief_confidence")),
        "c2_stability_action": stability_row.get("c2_stability_action", ""),
        "c2_stability_reason": stability_row.get("c2_stability_reason", ""),
        "c2_review_required": bool(stability_row.get("c2_review_required")),
        "short_burst_colour_group_id": stability_row.get("short_burst_colour_group_id", ""),
        "group_belief_counts": stability_row.get("group_belief_counts", {}),
        "flip_type": flip_row.get("flip_type", ""),
        "flip_reason": flip_row.get("flip_reason", ""),
        "local_group_not_identity": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def select_review_candidates(
    stability_payload: dict[str, Any],
    flip_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    stability_by_base = index_by_base_id(stability_payload)
    flip_by_base = index_by_base_id(flip_payload)
    selected: dict[str, dict[str, Any]] = {}
    flip_rows = flip_payload.get("rows", [])
    for flip in flip_rows:
        base_id = str(flip.get("visible_person_base_id", ""))
        row = stability_by_base.get(base_id, {})
        if not row:
            continue
        if row.get("c1c_seed_team_colour_belief") != row.get("c2_stable_colour_belief"):
            add_reason(selected, row, "changed_by_c2")
        if row.get("c2_review_required") is True:
            add_reason(selected, row, "c2_review_required")
        if int(safe_float(row.get("frame_sequence"), -1)) in ALWAYS_INCLUDE_FRAMES:
            add_reason(selected, row, "frame_59_62_manual_followup")
        flip_type = str(flip.get("flip_type", ""))
        if flip_type in ALWAYS_INCLUDE_FLIP_TYPES:
            add_reason(selected, row, flip_type)

    for flip_type, limit in SAMPLED_FLIP_LIMITS.items():
        rows = [stability_by_base[str(flip.get("visible_person_base_id", ""))] for flip in sorted_rows(flip_rows) if str(flip.get("flip_type", "")) == flip_type and str(flip.get("visible_person_base_id", "")) in stability_by_base]
        for row in rows[:limit]:
            add_reason(selected, row, f"{flip_type}_sample")

    retained_no_flip_rows = [
        stability_by_base[str(flip.get("visible_person_base_id", ""))]
        for flip in sorted_rows(flip_rows)
        if str(flip.get("flip_type", "")) == "retained_no_flip"
        and str(flip.get("visible_person_base_id", "")) in stability_by_base
    ]
    team_1_rows = [row for row in retained_no_flip_rows if row.get("c2_stable_colour_belief") == "team_1_outfield_colour_like"]
    team_2_rows = [row for row in retained_no_flip_rows if row.get("c2_stable_colour_belief") == "team_2_outfield_colour_like"]
    for row in sorted_rows(team_1_rows)[:RETAINED_TEAM_LIMIT]:
        add_reason(selected, row, "retained_team_1_sample")
    for row in sorted_rows(team_2_rows)[:RETAINED_TEAM_LIMIT]:
        add_reason(selected, row, "retained_team_2_sample")

    unknown_context_other_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retained_no_flip_rows:
        belief = str(row.get("c2_stable_colour_belief", ""))
        if belief in {"unknown_ambiguous_colour", "ambiguous_outfield_colour", "non_outfield_context_colour", "other_distinct_colour_like", "dark_context_colour_like"}:
            unknown_context_other_groups[belief].append(row)
    for row in round_robin_sample(unknown_context_other_groups, RETAINED_UNKNOWN_CONTEXT_OTHER_LIMIT):
        add_reason(selected, row, "retained_unknown_context_other_sample")

    ordered = sorted(
        selected.values(),
        key=lambda item: (
            int(item["review_priority"]),
            int(safe_float(item["row"].get("frame_sequence"), -1)),
            str(item["row"].get("visible_person_base_id", "")),
        ),
    )
    candidates = []
    for index, item in enumerate(ordered):
        row = item["row"]
        base_id = str(row.get("visible_person_base_id", ""))
        candidates.append(build_candidate_row(index, row, flip_by_base.get(base_id, {}), item["review_reasons"], int(item["review_priority"])))
    return candidates


def candidate_summary_payload(candidates: list[dict[str, Any]], eval_summary: dict[str, Any]) -> dict[str, Any]:
    reason_counts = Counter()
    for row in candidates:
        for reason in row.get("review_reason_tags", []):
            reason_counts[str(reason)] += 1
    flip_counts = Counter(str(row.get("flip_type", "")) for row in candidates)
    frame_counts = Counter(str(row.get("frame_sequence", "")) for row in candidates if int(safe_float(row.get("frame_sequence"), -1)) in ALWAYS_INCLUDE_FRAMES)
    return {
        "artifact": "step1c2b_colour_stability_review_candidate_summary",
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
        "total_review_candidates": len(candidates),
        "unique_visible_person_base_ids": len({row.get("visible_person_base_id") for row in candidates}),
        "review_reason_counts": dict(sorted(reason_counts.items())),
        "review_candidate_flip_type_counts": dict(sorted(flip_counts.items())),
        "frames_59_62_candidate_counts": dict(sorted(frame_counts.items())),
        "c2_safe_for_human_review_input": eval_summary.get("c2_safe_for_human_review", False),
        "c2b_is_approval_gate_not_canonical_output": True,
    }


def build_colour_stability_review_candidate_payload(
    stability_payload: dict[str, Any],
    flip_payload: dict[str, Any],
    group_payload: dict[str, Any] | None = None,
    eval_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del group_payload
    eval_summary = eval_summary or {}
    rows = select_review_candidates(stability_payload, flip_payload)
    return {
        "artifact": "step1c2b_colour_stability_review_candidate_rows",
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
        "rows": rows,
        "summary": candidate_summary_payload(rows, eval_summary),
    }


def build_and_write_colour_stability_review_candidates() -> dict[str, Any]:
    stability_payload = read_json(STEP1C2_COLOUR_STABILITY_ROWS_PATH)
    flip_payload = read_json(STEP1C2_COLOUR_FLIP_AUDIT_ROWS_PATH)
    group_payload = read_json(STEP1C2_SHORT_BURST_COLOUR_GROUP_ROWS_PATH)
    eval_summary = read_json(STEP1C2_GOLD8_COLOUR_STABILITY_EVAL_SUMMARY_PATH)
    payload = build_colour_stability_review_candidate_payload(stability_payload, flip_payload, group_payload, eval_summary)
    write_json(STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH, payload)
    write_json(STEP1C2B_REVIEW_CANDIDATE_SUMMARY_PATH, payload["summary"])
    return payload
