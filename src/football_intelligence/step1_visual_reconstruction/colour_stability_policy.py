# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.colour_stability_groups import TEAM_BELIEFS
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C1C_SEEDED_COLOUR_BELIEF_ROWS_SANDBOX_PATH,
    STEP1C2_COLOUR_FLIP_AUDIT_ROWS_PATH,
    STEP1C2_COLOUR_STABILITY_ROWS_PATH,
    STEP1C2_SHORT_BURST_COLOUR_GROUP_ROWS_PATH,
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
from football_intelligence.step1_visual_reconstruction.team_colour_beliefs import UNKNOWN_CONTEXT_TYPES


ALLOWED_C2_STABLE_BELIEFS = {
    "team_1_outfield_colour_like",
    "team_2_outfield_colour_like",
    "ambiguous_outfield_colour",
    "non_outfield_context_colour",
    "other_distinct_colour_like",
    "dark_context_colour_like",
    "crop_unusable",
    "unknown_ambiguous_colour",
}

ALLOWED_C2_STATES = {
    "high_confidence_stable_visual_colour",
    "medium_confidence_stable_visual_colour",
    "low_confidence_stable_visual_colour",
    "ambiguous_stable_visual_colour",
    "local_conflict_review_required",
    "crop_unusable",
    "retained_context_colour",
    "retained_other_distinct_colour",
    "review_required",
}

CONTEXT_OR_OTHER_BELIEFS = {"non_outfield_context_colour", "dark_context_colour_like", "other_distinct_colour_like"}
RETAINED_UNKNOWN_BELIEFS = {"unknown_ambiguous_colour", "ambiguous_outfield_colour"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def group_index(group_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for group in group_payload.get("rows", []):
        for base_id in group.get("visible_person_base_ids", []):
            out[str(base_id)] = group
    return out


def row_is_context_or_offroi(row: dict[str, Any]) -> bool:
    return (
        str(row.get("candidate_type", "")) in UNKNOWN_CONTEXT_TYPES
        or str(row.get("roi_status", "")) == "outside_playing_roi"
    )


def team_conflict(group: dict[str, Any]) -> bool:
    counts = group.get("group_belief_counts", {})
    return int(counts.get("team_1_outfield_colour_like", 0)) > 0 and int(counts.get("team_2_outfield_colour_like", 0)) > 0


def clear_team_consensus(group: dict[str, Any]) -> tuple[str, bool]:
    if not group:
        return "", False
    dominant = str(group.get("dominant_seed_team_colour_belief", ""))
    counts = group.get("group_belief_counts", {})
    if dominant not in TEAM_BELIEFS:
        return "", False
    if team_conflict(group):
        return dominant, False
    dominant_count = int(group.get("dominant_belief_count", 0))
    team_count = sum(int(counts.get(label, 0)) for label in TEAM_BELIEFS)
    purity = dominant_count / max(1, int(group.get("group_row_count", 0)))
    return dominant, (
        dominant_count >= 2
        and team_count == dominant_count
        and int(group.get("group_frame_count", 0)) >= 2
        and purity >= 0.60
        and safe_float(group.get("group_confidence")) >= 0.50
    )


def c2_state_for_belief(belief: str, confidence: float, review_required: bool, action: str) -> str:
    if action == "review_required_no_stabilisation":
        return "local_conflict_review_required"
    if review_required:
        return "review_required"
    if belief == "crop_unusable":
        return "crop_unusable"
    if belief in {"non_outfield_context_colour", "dark_context_colour_like"}:
        return "retained_context_colour"
    if belief == "other_distinct_colour_like":
        return "retained_other_distinct_colour"
    if belief in RETAINED_UNKNOWN_BELIEFS:
        return "ambiguous_stable_visual_colour"
    if confidence >= 0.78:
        return "high_confidence_stable_visual_colour"
    if confidence >= 0.58:
        return "medium_confidence_stable_visual_colour"
    return "low_confidence_stable_visual_colour"


def stability_decision(row: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    original = str(row.get("seed_team_colour_belief", "unknown_ambiguous_colour"))
    original_confidence = safe_float(row.get("seed_team_colour_belief_confidence"), 0.0)
    consensus, consensus_is_clear = clear_team_consensus(group)
    conflict = team_conflict(group)
    context_or_offroi = row_is_context_or_offroi(row)
    crop_quality = str(row.get("crop_quality", ""))
    stable = original if original in ALLOWED_C2_STABLE_BELIEFS else "unknown_ambiguous_colour"
    confidence = original_confidence
    action = "retained_seeded_belief"
    reason = "seeded_belief_retained_no_safe_local_change"
    review_required = False

    if original == "crop_unusable" or crop_quality == "unusable":
        stable = "crop_unusable"
        confidence = min(original_confidence, 0.20)
        action = "retained_crop_unusable"
        reason = "crop_unusable_not_stabilised"
    elif original in CONTEXT_OR_OTHER_BELIEFS:
        action = "retained_context_or_other_distinct"
        reason = "context_or_other_distinct_colour_not_forced_to_team"
        if consensus_is_clear:
            reason = "team_consensus_seen_but_context_or_other_distinct_colour_not_forced"
    elif original in RETAINED_UNKNOWN_BELIEFS:
        if consensus_is_clear and not context_or_offroi and crop_quality != "unusable":
            stable = consensus
            confidence = round(max(0.58, min(0.88, safe_float(group.get("group_confidence")) + 0.08)), 4)
            action = "stabilised_from_short_burst_consensus"
            reason = "unknown_or_ambiguous_colour_stabilised_from_clear_short_burst_consensus"
        elif conflict:
            stable = "unknown_ambiguous_colour"
            confidence = min(original_confidence, 0.35)
            action = "downgraded_to_unknown_due_to_local_conflict"
            reason = "mixed_team_colour_short_burst_conflict_not_forced"
            review_required = True
        else:
            stable = original
            confidence = min(max(original_confidence, safe_float(group.get("group_confidence"), 0.0)), 0.55)
            action = "retained_unknown_ambiguous"
            reason = "no_clear_local_team_colour_consensus"
    elif original in TEAM_BELIEFS:
        if conflict:
            action = "review_required_no_stabilisation"
            reason = "team_1_team_2_short_burst_conflict_flagged_without_flip"
            review_required = True
        elif consensus_is_clear and consensus != original:
            action = "review_required_no_stabilisation"
            reason = "strong_local_consensus_disagrees_with_seeded_team_colour_no_flip_applied"
            review_required = True
        else:
            stable = original
            confidence = max(original_confidence, min(0.92, safe_float(group.get("group_confidence"), 0.0)))
            action = "retained_seeded_belief"
            reason = "team_colour_seeded_belief_retained"
    else:
        stable = "unknown_ambiguous_colour"
        confidence = min(original_confidence, 0.35)
        action = "retained_unknown_ambiguous"
        reason = "unrecognised_seeded_colour_belief_retained_as_unknown"
    state = c2_state_for_belief(stable, confidence, review_required, action)
    return {
        "c2_stable_colour_belief": stable,
        "c2_stable_colour_belief_confidence": round(confidence, 4),
        "c2_stable_colour_belief_state": state,
        "c2_stability_action": action,
        "c2_stability_reason": reason,
        "c2_review_required": review_required,
    }


def stability_row(row: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    decision = stability_decision(row, group)
    out = {
        "visible_person_base_id": row.get("visible_person_base_id", ""),
        "frame_id": row.get("frame_id", ""),
        "frame_sequence": int(safe_float(row.get("frame_sequence"), -1)),
        "timestamp_seconds": safe_float(row.get("timestamp_seconds")),
        "detection_id": row.get("detection_id", ""),
        "source_detection_id": row.get("source_detection_id", ""),
        "bbox": row.get("bbox", {}),
        "footpoint": row.get("footpoint", {}),
        "state": row.get("state", ""),
        "roi_status": row.get("roi_status", ""),
        "candidate_type": row.get("candidate_type", ""),
        "original_role_source": row.get("original_role_source", ""),
        "crop_quality": row.get("crop_quality", ""),
        "crop_quality_reason": row.get("crop_quality_reason", ""),
        "torso_crop_bbox": row.get("torso_crop_bbox"),
        "c1c_seed_team_colour_belief": row.get("seed_team_colour_belief", ""),
        "c1c_seed_team_colour_belief_confidence": safe_float(row.get("seed_team_colour_belief_confidence")),
        "c1c_seed_team_colour_belief_state": row.get("seed_team_colour_belief_state", ""),
        "c1c_seed_team_colour_belief_reason": row.get("seed_team_colour_belief_reason", ""),
        "short_burst_colour_group_id": group.get("short_burst_colour_group_id", ""),
        "local_group_not_identity": True,
        "group_frame_count": int(group.get("group_frame_count", 0)),
        "group_belief_counts": group.get("group_belief_counts", {}),
        **decision,
        "eligible_for_step1d_official_context_candidate": True,
        "eligible_for_step1e_goalkeeper_candidate": True,
        "eligible_for_identity_tracking": False,
        "eligible_for_player_slot_assignment": False,
        "eligible_for_metric_use": False,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "auto_promoted": False,
    }
    return out


def flip_type_for_row(row: dict[str, Any]) -> tuple[str, str]:
    original = str(row.get("c1c_seed_team_colour_belief", ""))
    stable = str(row.get("c2_stable_colour_belief", ""))
    action = str(row.get("c2_stability_action", ""))
    if row.get("c2_review_required") is True or action == "review_required_no_stabilisation":
        if original == "team_1_outfield_colour_like" and stable == "team_1_outfield_colour_like":
            return "team_1_to_team_2_blocked", "team_1_to_team_2_flip_blocked_by_local_conflict_review"
        if original == "team_2_outfield_colour_like" and stable == "team_2_outfield_colour_like":
            return "team_2_to_team_1_blocked", "team_2_to_team_1_flip_blocked_by_local_conflict_review"
        return "review_required_conflict", "local_colour_conflict_requires_human_review"
    if original in RETAINED_UNKNOWN_BELIEFS and stable in TEAM_BELIEFS:
        return "unknown_to_team_colour", "unknown_or_ambiguous_colour_stabilised_from_short_burst_consensus"
    if original in TEAM_BELIEFS and stable == "unknown_ambiguous_colour":
        return "team_colour_to_unknown", "weak_or_conflicting_team_colour_downgraded_to_unknown"
    if original in {"non_outfield_context_colour", "dark_context_colour_like"} and stable == original:
        return "context_force_blocked", "context_or_offroi_colour_not_forced_to_team"
    if original == "other_distinct_colour_like" and stable == original:
        return "team_colour_to_other_distinct_blocked", "other_distinct_colour_not_forced_to_team"
    return "retained_no_flip", "c2_retained_seeded_colour_belief"


def flip_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    flip_type, flip_reason = flip_type_for_row(row)
    return {
        "visible_person_base_id": row.get("visible_person_base_id", ""),
        "frame_sequence": int(safe_float(row.get("frame_sequence"), -1)),
        "c1c_seed_team_colour_belief": row.get("c1c_seed_team_colour_belief", ""),
        "c2_stable_colour_belief": row.get("c2_stable_colour_belief", ""),
        "c2_stability_action": row.get("c2_stability_action", ""),
        "short_burst_colour_group_id": row.get("short_burst_colour_group_id", ""),
        "group_belief_counts": row.get("group_belief_counts", {}),
        "flip_type": flip_type,
        "flip_reason": flip_reason,
        "review_required": row.get("c2_review_required", False),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def context_offroi_forced_to_team_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row_is_context_or_offroi(row) and row.get("c2_stable_colour_belief") in TEAM_BELIEFS
    )


def build_colour_stability_payloads(
    seeded_payload: dict[str, Any],
    group_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    groups_by_base_id = group_index(group_payload)
    rows = [stability_row(row, groups_by_base_id.get(str(row.get("visible_person_base_id", "")), {})) for row in seeded_payload.get("rows", [])]
    flip_rows = [flip_audit_row(row) for row in rows]
    action_counts = Counter(str(row.get("c2_stability_action", "")) for row in rows)
    belief_counts = Counter(str(row.get("c2_stable_colour_belief", "")) for row in rows)
    flip_counts = Counter(str(row.get("flip_type", "")) for row in flip_rows)
    stability_payload = {
        "artifact": "step1c2_colour_stability_rows",
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
        "short_burst_groups_are_not_identities": True,
        "allowed_c2_stable_beliefs": sorted(ALLOWED_C2_STABLE_BELIEFS),
        "allowed_c2_states": sorted(ALLOWED_C2_STATES),
        "rows": rows,
        "summary": {
            "c1c_seeded_belief_row_count": len(seeded_payload.get("rows", [])),
            "c2_stability_row_count": len(rows),
            "one_row_per_c1c_seeded_belief_row": len(rows) == len(seeded_payload.get("rows", [])),
            "c2_stable_belief_counts": dict(sorted(belief_counts.items())),
            "c2_stability_action_counts": dict(sorted(action_counts.items())),
            "context_offroi_forced_to_team_count": context_offroi_forced_to_team_count(rows),
            "c2_review_required_count": sum(1 for row in rows if row.get("c2_review_required") is True),
            "short_burst_colour_group_count": len(group_payload.get("rows", [])),
        },
    }
    flip_payload = {
        "artifact": "step1c2_colour_flip_audit_rows",
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
        "rows": flip_rows,
        "summary": {
            "flip_audit_row_count": len(flip_rows),
            "flip_type_counts": dict(sorted(flip_counts.items())),
            "review_required_flip_rows": sum(1 for row in flip_rows if row.get("review_required") is True),
        },
    }
    return stability_payload, flip_payload


def build_and_write_colour_stability_policy() -> tuple[dict[str, Any], dict[str, Any]]:
    seeded_payload = read_json(STEP1C1C_SEEDED_COLOUR_BELIEF_ROWS_SANDBOX_PATH)
    group_payload = read_json(STEP1C2_SHORT_BURST_COLOUR_GROUP_ROWS_PATH)
    stability_payload, flip_payload = build_colour_stability_payloads(seeded_payload, group_payload)
    write_json(STEP1C2_COLOUR_STABILITY_ROWS_PATH, stability_payload)
    write_json(STEP1C2_COLOUR_FLIP_AUDIT_ROWS_PATH, flip_payload)
    return stability_payload, flip_payload
