from __future__ import annotations

from collections import Counter
from typing import Any

from football_intelligence.review.schemas import safety_payload
from football_intelligence.step2_visual_continuity.candidate_matching import bound_candidate_degrees

NON_PERSON = "non_person_false_positive"
UNKNOWN = "unknown_visible_person_visual_context"
TEAM_1_OUTFIELD = "team_1_outfield_visual_context"
TEAM_2_OUTFIELD = "team_2_outfield_visual_context"
TEAM_UNKNOWN_OUTFIELD = "team_unknown_outfield_visual_context"
TEAM_1_GOALKEEPER = "team_1_goalkeeper_visual_context"
TEAM_2_GOALKEEPER = "team_2_goalkeeper_visual_context"
GOALKEEPER_UNKNOWN = "goalkeeper_unknown_team_visual_context"
CENTRAL_REFEREE = "central_referee_visual_context"
ASSISTANT_NEAR = "assistant_referee_near_camera_context"
ASSISTANT_FAR = "assistant_referee_far_camera_context"
OTHER_OFF_PITCH = "other_off_pitch_person_visual_context"


def _compatible(source: str, target: str, *, allow_off_pitch: bool = False) -> tuple[bool, str]:
    if source == NON_PERSON or target == NON_PERSON:
        return False, "non_person_continuity_not_applicable"
    if source == target:
        return True, "same_visual_role_partition"
    if source == TEAM_1_OUTFIELD and target == TEAM_UNKNOWN_OUTFIELD:
        return True, "team_1_to_unknown_reviewable"
    if source == TEAM_2_OUTFIELD and target == TEAM_UNKNOWN_OUTFIELD:
        return True, "team_2_to_unknown_reviewable"
    if target == TEAM_1_OUTFIELD and source == TEAM_UNKNOWN_OUTFIELD:
        return True, "unknown_to_team_1_reviewable"
    if target == TEAM_2_OUTFIELD and source == TEAM_UNKNOWN_OUTFIELD:
        return True, "unknown_to_team_2_reviewable"
    if source == TEAM_1_GOALKEEPER and target == GOALKEEPER_UNKNOWN:
        return True, "team_1_goalkeeper_to_unknown_goalkeeper"
    if source == TEAM_2_GOALKEEPER and target == GOALKEEPER_UNKNOWN:
        return True, "team_2_goalkeeper_to_unknown_goalkeeper"
    if target == TEAM_1_GOALKEEPER and source == GOALKEEPER_UNKNOWN:
        return True, "unknown_goalkeeper_to_team_1_goalkeeper"
    if target == TEAM_2_GOALKEEPER and source == GOALKEEPER_UNKNOWN:
        return True, "unknown_goalkeeper_to_team_2_goalkeeper"
    if source == CENTRAL_REFEREE and target == UNKNOWN:
        return True, "central_referee_to_uncertain_official"
    if target == CENTRAL_REFEREE and source == UNKNOWN:
        return True, "uncertain_official_to_central_referee"
    if source == ASSISTANT_NEAR and target == UNKNOWN:
        return True, "near_assistant_to_near_or_unknown"
    if target == ASSISTANT_NEAR and source == UNKNOWN:
        return True, "unknown_to_near_assistant"
    if source == ASSISTANT_FAR and target == UNKNOWN:
        return True, "far_assistant_to_far_or_unknown"
    if target == ASSISTANT_FAR and source == UNKNOWN:
        return True, "unknown_to_far_assistant"
    if source == OTHER_OFF_PITCH or target == OTHER_OFF_PITCH:
        return (True, "off_pitch_continuity_explicitly_enabled") if allow_off_pitch else (False, "off_pitch_excluded")
    if source == UNKNOWN or target == UNKNOWN:
        return True, "unknown_strict_reviewable"
    return False, "role_partition_incompatible"


def build_role_partition_manifest() -> dict[str, Any]:
    return {
        "artifact": "m5_4e_role_partition_manifest",
        "role_partition_version": "m5.4e.role_partition.v1",
        "off_pitch_continuity_enabled": False,
        "non_person_continuity_applicable": False,
        "visual_continuity_is_real_identity": False,
        "visual_continuity_is_player_slot": False,
        "match_local_only": True,
        "sandbox_only": True,
        **safety_payload(),
    }


def apply_role_partitioning(
    *,
    candidate_rows: list[dict[str, Any]],
    role_by_visible_id: dict[str, dict[str, Any]],
    max_degree: int = 3,
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in candidate_rows:
        source_role = role_by_visible_id.get(str(row.get("source_visible_person_base_id")), {})
        target_role = role_by_visible_id.get(str(row.get("target_visible_person_base_id")), {})
        source_state = source_role.get("visual_role_context_state", "unknown_visible_person_visual_context")
        target_state = target_role.get("visual_role_context_state", "unknown_visible_person_visual_context")
        compatible, reason = _compatible(str(source_state), str(target_state))
        output = {
            **row,
            "source_visual_role_context": source_state,
            "target_visual_role_context": target_state,
            "role_partition_reason": reason,
            "visual_continuity_is_real_identity": False,
            "visual_continuity_is_player_slot": False,
            "match_local_only": True,
            "sandbox_only": True,
        }
        if compatible:
            accepted.append(output)
        else:
            rejected.append({**output, "rejection_reason": reason})
    bounded, degree_rejected = bound_candidate_degrees(accepted, max_degree=max_degree, score_key="continuity_score")
    rejected.extend({**row, "rejection_reason": "degree_bound_after_role_partition"} for row in degree_rejected)
    for index, row in enumerate(bounded):
        row["role_partitioned_continuity_candidate_id"] = f"m5_4e_rpc_{index:06d}"
    source_degree = Counter(row["source_visible_person_base_id"] for row in bounded)
    target_degree = Counter(row["target_visible_person_base_id"] for row in bounded)
    return {
        "artifact": "m5_4e_role_partitioned_candidate_rows",
        "candidate_pool_before_role_partitioning": len(candidate_rows),
        "candidate_pool_after_role_partitioning": len(bounded),
        "role_incompatible_rejected_count": len(rejected),
        "max_source_candidate_degree": max(source_degree.values() or [0]),
        "max_target_candidate_degree": max(target_degree.values() or [0]),
        "rows": bounded,
        "rejected_rows": rejected,
        **safety_payload(),
    }


def pool_size_report(before: int, after: int) -> dict[str, Any]:
    return {
        "artifact": "m5_4e_candidate_pool_size_before_after",
        "candidate_pool_before_role_partitioning": before,
        "candidate_pool_after_role_partitioning": after,
        "reduction_count": max(0, before - after),
        "reduction_fraction": round((before - after) / before, 6) if before else 0.0,
        **safety_payload(),
    }
