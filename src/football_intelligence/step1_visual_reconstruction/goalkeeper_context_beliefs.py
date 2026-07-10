# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.paths import STAGE3C13_GOLD20_MANUAL_LABELS_PATH
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import gold_visible_person_rows, strict_one_to_one_match
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH,
    STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_FEATURE_ROWS_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_REPORT_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    FORBIDDEN_OUTPUT_KEYS,
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    bbox_area,
    bbox_from_item,
    safe_float,
)


ALLOWED_E1_GOALKEEPER_CONTEXT_BELIEFS = {
    "goalkeeper_like_team_1_context",
    "goalkeeper_like_team_2_context",
    "goalkeeper_like_unknown_team_context",
    "outfield_player_like_not_goalkeeper",
    "official_or_context_not_goalkeeper",
    "bad_detection_or_not_person",
    "unknown_goalkeeper_context",
}
GOALKEEPER_LIKE_BELIEFS = {
    "goalkeeper_like_team_1_context",
    "goalkeeper_like_team_2_context",
    "goalkeeper_like_unknown_team_context",
}
TEAM_COLOUR_BELIEFS = {"team_1_outfield_colour_like", "team_2_outfield_colour_like"}
NON_OUTFIELD_COLOUR_BELIEFS = {"other_distinct_colour_like", "non_outfield_context_colour"}
OFFICIAL_OR_CONTEXT_D1C_BELIEFS = {
    "official_referee_like",
    "assistant_or_line_official_like",
    "non_official_context_person_like",
    "off_pitch_context_person_like",
}
E1_FORBIDDEN_KEYS = set(FORBIDDEN_OUTPUT_KEYS) | {
    "track_id",
    "persistent_player_id",
    "official_exclusion",
    "official_exclusion_reason",
    "exclude_from_player_review",
    "excluded_from_player_review",
    "excluded_from_player_team_review",
    "goalkeeper_slot_id",
    "expected_22_role_state",
    "expected_role_state",
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def feature_id_for_row(row: dict[str, Any]) -> str:
    raw = f"{row.get('visible_person_base_id','')}|{row.get('frame_sequence','')}|{row.get('detection_id','')}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"step1e1_gkctx_feature_f{int(safe_float(row.get('frame_sequence'), -1)):06d}_{digest}"


def review_id_for_row(row: dict[str, Any]) -> str:
    raw = f"{row.get('visible_person_base_id','')}|{row.get('e1_goalkeeper_context_belief','')}|{row.get('frame_sequence','')}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"step1e1_review_f{int(safe_float(row.get('frame_sequence'), -1)):06d}_{digest}"


def crop_area(row: dict[str, Any], c2c_row: dict[str, Any]) -> float:
    torso = c2c_row.get("torso_crop_bbox")
    if isinstance(torso, dict):
        return round(bbox_area(torso), 3)
    return round(bbox_area(bbox_from_item(row)), 3)


def goal_area_hint(row: dict[str, Any]) -> bool:
    bbox = bbox_from_item(row)
    if not bbox:
        return False
    center_x = (bbox["x1"] + bbox["x2"]) / 2.0
    bottom_y = bbox["y2"]
    return bool((center_x <= 0.22 * 2730.0 or center_x >= 0.78 * 2730.0) and bottom_y >= 125.0)


def source_goalkeeper_hint(row: dict[str, Any]) -> bool:
    source_text = f"{row.get('candidate_type','')} {row.get('original_role_source','')}".lower()
    return "goalkeeper" in source_text or "gk" in source_text


def team_belief_for_goalkeeper(feature: dict[str, Any]) -> str:
    if feature.get("source_goalkeeper_hint") is not True:
        return "unknown_team"
    colour = str(feature.get("c2c_colour_belief", ""))
    if colour == "team_1_outfield_colour_like":
        return "team_1"
    if colour == "team_2_outfield_colour_like":
        return "team_2"
    return "unknown_team"


def build_goalkeeper_context_feature_row(d1c_row: dict[str, Any], c2c_row: dict[str, Any] | None = None) -> dict[str, Any]:
    c2c_row = c2c_row or {}
    colour = str(c2c_row.get("c2c_final_colour_belief") or d1c_row.get("c2c_final_colour_belief", "unknown_ambiguous_colour"))
    d1c_context = str(d1c_row.get("d1c_final_official_context_belief", "unknown_official_context"))
    official_negative = d1c_context in OFFICIAL_OR_CONTEXT_D1C_BELIEFS
    bad_negative = d1c_context == "bad_detection_or_not_person" or boolish(d1c_row.get("d1c_bad_detection_or_not_person"))
    non_outfield = colour in NON_OUTFIELD_COLOUR_BELIEFS
    distinct_team_1 = colour != "team_1_outfield_colour_like"
    distinct_team_2 = colour != "team_2_outfield_colour_like"
    goal_hint = goal_area_hint(d1c_row)
    source_gk = source_goalkeeper_hint(d1c_row)
    ambiguity = []
    if colour == "unknown_ambiguous_colour":
        ambiguity.append("unknown_colour")
    if colour == "ambiguous_outfield_colour":
        ambiguity.append("ambiguous_outfield_colour")
    if official_negative and (non_outfield or goal_hint):
        ambiguity.append("official_context_with_goalkeeper_visual_hint")
    if d1c_context == "unknown_official_context" and non_outfield:
        ambiguity.append("unknown_official_context_with_non_outfield_colour")
    return {
        "visible_person_base_id": d1c_row.get("visible_person_base_id", ""),
        "frame_id": d1c_row.get("frame_id", ""),
        "frame_sequence": int(safe_float(d1c_row.get("frame_sequence"), -1)),
        "timestamp_seconds": safe_float(d1c_row.get("timestamp_seconds")),
        "detection_id": d1c_row.get("detection_id", ""),
        "source_detection_id": d1c_row.get("source_detection_id", ""),
        "bbox": d1c_row.get("bbox", {}),
        "footpoint": d1c_row.get("footpoint", {}),
        "state": d1c_row.get("state", ""),
        "roi_status": d1c_row.get("roi_status", ""),
        "candidate_type": d1c_row.get("candidate_type", ""),
        "original_role_source": d1c_row.get("original_role_source", ""),
        "c2c_final_colour_belief": colour,
        "c2c_colour_source": d1c_row.get("c2c_colour_source", ""),
        "c2c_human_reviewed": boolish(d1c_row.get("c2c_human_reviewed")),
        "d1c_final_official_context_belief": d1c_context,
        "d1c_context_source": d1c_row.get("d1c_context_source", ""),
        "d1c_human_reviewed": boolish(d1c_row.get("d1c_human_reviewed")),
        "d1c_bad_detection_or_not_person": bad_negative,
        "d1c_official_like_visual_context": boolish(d1c_row.get("d1c_official_like_visual_context")),
        "d1c_assistant_or_line_official_like_visual_context": boolish(d1c_row.get("d1c_assistant_or_line_official_like_visual_context")),
        "e1_goalkeeper_context_feature_id": feature_id_for_row(d1c_row),
        "crop_quality": c2c_row.get("crop_quality", ""),
        "crop_area_px": crop_area(d1c_row, c2c_row),
        "torso_colour_cluster_hint": colour,
        "c2c_colour_belief": colour,
        "d1c_context_belief": d1c_context,
        "distinct_from_team1_colour_hint": distinct_team_1,
        "distinct_from_team2_colour_hint": distinct_team_2,
        "non_outfield_colour_hint": non_outfield,
        "official_context_negative_hint": official_negative,
        "bad_detection_negative_hint": bad_negative,
        "image_space_goal_area_context_hint": goal_hint,
        "image_space_goal_area_context_hint_not_metric_truth": goal_hint,
        "source_goalkeeper_hint": source_gk,
        "source_player_hint": str(d1c_row.get("original_role_source", "")).lower() == "player",
        "source_official_hint": str(d1c_row.get("original_role_source", "")).lower() in {"official", "referee", "staff"},
        "ambiguity_flags": ambiguity,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "auto_promoted": False,
    }


def belief_decision(feature: dict[str, Any]) -> dict[str, Any]:
    warnings = []
    colour = str(feature.get("c2c_colour_belief", ""))
    d1c_context = str(feature.get("d1c_context_belief", ""))
    non_outfield = boolish(feature.get("non_outfield_colour_hint"))
    goal_hint = boolish(feature.get("image_space_goal_area_context_hint"))
    source_gk = boolish(feature.get("source_goalkeeper_hint"))
    crop_quality = str(feature.get("crop_quality", ""))
    official_negative = boolish(feature.get("official_context_negative_hint"))
    bad_negative = boolish(feature.get("bad_detection_negative_hint"))
    if goal_hint:
        warnings.append("image_space_goal_area_context_hint_not_metric_truth")
    if non_outfield:
        warnings.append("non_outfield_colour_is_visual_hint_only")
    if official_negative:
        warnings.append("official_context_negative_hint_not_exclusion")
    if bad_negative:
        return {
            "belief": "bad_detection_or_not_person",
            "state": "bad_detection_review_required",
            "confidence": 0.96,
            "review_required": True,
            "reason": "d1c_bad_detection_preserved_for_goalkeeper_context_review",
            "warnings": warnings,
        }
    if d1c_context in OFFICIAL_OR_CONTEXT_D1C_BELIEFS:
        return {
            "belief": "official_or_context_not_goalkeeper",
            "state": "official_or_context_visual_context",
            "confidence": 0.92,
            "review_required": bool(non_outfield and goal_hint),
            "reason": "d1c_official_or_context_person_retained_not_forced_goalkeeper",
            "warnings": warnings,
        }
    if source_gk:
        team = team_belief_for_goalkeeper(feature)
        belief = {
            "team_1": "goalkeeper_like_team_1_context",
            "team_2": "goalkeeper_like_team_2_context",
            "unknown_team": "goalkeeper_like_unknown_team_context",
        }[team]
        return {
            "belief": belief,
            "state": "source_goalkeeper_visual_context_review",
            "confidence": 0.78,
            "review_required": True,
            "reason": "source_goalkeeper_hint_kept_as_visual_context_not_slot",
            "warnings": warnings + ["source_role_hint_not_identity_or_slot"],
        }
    if d1c_context == "player_like_not_official_context":
        if non_outfield and goal_hint and crop_quality != "low":
            return {
                "belief": "goalkeeper_like_unknown_team_context",
                "state": "visual_goalkeeper_context_review",
                "confidence": 0.66,
                "review_required": True,
                "reason": "non_outfield_colour_and_image_space_goal_area_context_visual_hints",
                "warnings": warnings,
            }
        if colour in TEAM_COLOUR_BELIEFS:
            return {
                "belief": "outfield_player_like_not_goalkeeper",
                "state": "outfield_colour_player_visual_context",
                "confidence": 0.86,
                "review_required": False,
                "reason": "team_outfield_colour_and_player_like_d1c_context",
                "warnings": warnings,
            }
        return {
            "belief": "unknown_goalkeeper_context",
            "state": "ambiguous_goalkeeper_visual_context",
            "confidence": 0.50,
            "review_required": True,
            "reason": "player_like_context_without_clear_outfield_or_goalkeeper_visual_evidence",
            "warnings": warnings,
        }
    if d1c_context == "unknown_official_context":
        if non_outfield and goal_hint and crop_quality != "low":
            return {
                "belief": "goalkeeper_like_unknown_team_context",
                "state": "visual_goalkeeper_context_review",
                "confidence": 0.58,
                "review_required": True,
                "reason": "unknown_d1c_context_with_non_outfield_colour_and_image_space_goal_area_hint",
                "warnings": warnings,
            }
        return {
            "belief": "unknown_goalkeeper_context",
            "state": "review_required",
            "confidence": 0.42,
            "review_required": True,
            "reason": "unknown_d1c_context_not_forced_goalkeeper",
            "warnings": warnings,
        }
    return {
        "belief": "unknown_goalkeeper_context",
        "state": "review_required",
        "confidence": 0.40,
        "review_required": True,
        "reason": "insufficient_visual_context_for_goalkeeper_belief",
        "warnings": warnings,
    }


def build_goalkeeper_context_belief_row(feature: dict[str, Any]) -> dict[str, Any]:
    decision = belief_decision(feature)
    belief = str(decision["belief"])
    if belief not in ALLOWED_E1_GOALKEEPER_CONTEXT_BELIEFS:
        belief = "unknown_goalkeeper_context"
    team_belief = "not_goalkeeper"
    if belief == "goalkeeper_like_team_1_context":
        team_belief = "team_1"
    elif belief == "goalkeeper_like_team_2_context":
        team_belief = "team_2"
    elif belief == "goalkeeper_like_unknown_team_context":
        team_belief = "unknown_team"
    out = dict(feature)
    out.update(
        {
            "e1_goalkeeper_context_belief": belief,
            "e1_goalkeeper_context_belief_state": decision.get("state", "review_required"),
            "e1_goalkeeper_context_belief_confidence": round(safe_float(decision.get("confidence")), 4),
            "e1_goalkeeper_context_belief_reason": decision.get("reason", ""),
            "e1_goalkeeper_context_review_required": boolish(decision.get("review_required")),
            "e1_goalkeeper_context_warning_flags": decision.get("warnings", []),
            "e1_goalkeeper_like_visual_context": belief in GOALKEEPER_LIKE_BELIEFS,
            "e1_goalkeeper_team_belief": team_belief,
            "e1_outfield_player_like_not_goalkeeper": belief == "outfield_player_like_not_goalkeeper",
            "e1_official_or_context_not_goalkeeper": belief == "official_or_context_not_goalkeeper",
            "e1_bad_detection_or_not_person": belief == "bad_detection_or_not_person",
            "retained_for_future_player_team_review": True,
            "eligible_for_identity_tracking": False,
            "eligible_for_player_slot_assignment": False,
            "eligible_for_metric_use": False,
            "visual_only_warning": VISUAL_ONLY_WARNING,
            "do_not_use_for_metrics": True,
            "production_ready": PRODUCTION_READY,
            "auto_promoted": False,
        }
    )
    return out


def gold_goalkeeper_matched_visible_ids(belief_rows: list[dict[str, Any]]) -> set[str]:
    labels_payload = read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    gold_rows = [
        row
        for row in gold_visible_person_rows(labels_payload)
        if row.get("visible_person_type_gold") in {"gk_team_1", "gk_team_2"}
    ]
    matches, _missed, _extra = strict_one_to_one_match(gold_rows, belief_rows)
    return {str(match["candidate"].get("visible_person_base_id", "")) for match in matches}


def review_reason_tags(row: dict[str, Any], gold_goalkeeper_ids: set[str]) -> list[str]:
    tags = []
    belief = str(row.get("e1_goalkeeper_context_belief", ""))
    if str(row.get("visible_person_base_id", "")) in gold_goalkeeper_ids:
        tags.append("gold8_goalkeeper_proxy_match")
    if belief in GOALKEEPER_LIKE_BELIEFS:
        tags.append("goalkeeper_like_belief")
    if belief == "unknown_goalkeeper_context" and row.get("non_outfield_colour_hint") is True:
        tags.append("unknown_goalkeeper_context_with_non_outfield_colour_hint")
    if row.get("source_goalkeeper_hint") is True:
        tags.append("source_goalkeeper_hint")
    if belief == "bad_detection_or_not_person" and (row.get("non_outfield_colour_hint") is True or row.get("image_space_goal_area_context_hint") is True):
        tags.append("bad_detection_with_goalkeeper_like_hint")
    if row.get("official_context_negative_hint") is True and row.get("non_outfield_colour_hint") is True and row.get("image_space_goal_area_context_hint") is True:
        tags.append("contradictory_official_context_goalkeeper_hints")
    if row.get("e1_goalkeeper_context_review_required") is True:
        tags.append("review_required")
    return sorted(set(tags))


def review_priority_for_tags(tags: list[str]) -> int:
    priority = 90
    weights = {
        "gold8_goalkeeper_proxy_match": 0,
        "goalkeeper_like_belief": 5,
        "source_goalkeeper_hint": 8,
        "contradictory_official_context_goalkeeper_hints": 12,
        "unknown_goalkeeper_context_with_non_outfield_colour_hint": 16,
        "bad_detection_with_goalkeeper_like_hint": 18,
        "review_required": 35,
        "balanced_sample_outfield_player_like_not_goalkeeper": 70,
        "balanced_sample_official_or_context_not_goalkeeper": 75,
    }
    for tag in tags:
        priority = min(priority, weights.get(tag, priority))
    return priority


def build_review_candidate_payload(belief_payload: dict[str, Any], *, sample_per_group: int = 90) -> dict[str, Any]:
    belief_rows = belief_payload.get("rows", [])
    gold_goalkeeper_ids = gold_goalkeeper_matched_visible_ids(belief_rows)
    candidates: dict[str, dict[str, Any]] = {}

    def add(row: dict[str, Any], tags: list[str]) -> None:
        visible_id = str(row.get("visible_person_base_id", ""))
        if not visible_id:
            return
        priority = review_priority_for_tags(tags)
        existing = candidates.get(visible_id)
        if existing and existing.get("review_priority", 999) <= priority:
            existing["review_reason_tags"] = sorted(set(existing.get("review_reason_tags", [])) | set(tags))
            existing["review_reason"] = ";".join(existing["review_reason_tags"])
            return
        out = {
            "step1e1_review_candidate_id": review_id_for_row(row),
            "review_priority": priority,
            "review_reason": ";".join(sorted(set(tags))),
            "review_reason_tags": sorted(set(tags)),
            **row,
            "visual_only_warning": VISUAL_ONLY_WARNING,
            "do_not_use_for_metrics": True,
            "production_ready": PRODUCTION_READY,
        }
        candidates[visible_id] = out

    for row in belief_rows:
        tags = review_reason_tags(row, gold_goalkeeper_ids)
        if tags:
            add(row, tags)
    for belief, tag in [
        ("outfield_player_like_not_goalkeeper", "balanced_sample_outfield_player_like_not_goalkeeper"),
        ("official_or_context_not_goalkeeper", "balanced_sample_official_or_context_not_goalkeeper"),
    ]:
        rows = [
            row
            for row in belief_rows
            if row.get("e1_goalkeeper_context_belief") == belief
            and str(row.get("visible_person_base_id", "")) not in candidates
        ]
        rows.sort(key=lambda item: (int(safe_float(item.get("frame_sequence"), -1)), str(item.get("visible_person_base_id", ""))))
        for row in rows[:sample_per_group]:
            add(row, [tag])
    rows = sorted(candidates.values(), key=lambda row: (int(row.get("review_priority", 999)), int(safe_float(row.get("frame_sequence"), -1)), str(row.get("visible_person_base_id", ""))))
    return {
        "artifact": "step1e1_goalkeeper_context_review_candidate_rows",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "rows": rows,
        "summary": {
            "e1_review_candidate_count": len(rows),
            "unique_visible_person_base_ids": len({row.get("visible_person_base_id") for row in rows}),
            "gold8_goalkeeper_proxy_review_candidate_count": sum(1 for row in rows if "gold8_goalkeeper_proxy_match" in row.get("review_reason_tags", [])),
            "review_reason_counts": dict(sorted(Counter(tag for row in rows for tag in row.get("review_reason_tags", [])).items())),
        },
    }


def forbidden_keys_present(rows: list[dict[str, Any]]) -> list[str]:
    found: set[str] = set()
    for row in rows:
        found.update(key for key in E1_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def build_goalkeeper_context_payloads(
    d1c_payload: dict[str, Any],
    c2c_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    c2c_by_visible_id = {
        str(row.get("visible_person_base_id", "")): row
        for row in c2c_payload.get("rows", [])
        if row.get("visible_person_base_id")
    }
    feature_rows = [
        build_goalkeeper_context_feature_row(row, c2c_by_visible_id.get(str(row.get("visible_person_base_id", "")), {}))
        for row in d1c_payload.get("rows", [])
    ]
    belief_rows = [build_goalkeeper_context_belief_row(row) for row in feature_rows]
    review_payload = build_review_candidate_payload({"rows": belief_rows})
    feature_counts = Counter(str(row.get("c2c_colour_belief", "")) for row in feature_rows)
    belief_counts = Counter(str(row.get("e1_goalkeeper_context_belief", "")) for row in belief_rows)
    feature_payload = {
        "artifact": "step1e1_goalkeeper_context_feature_rows",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "rows": feature_rows,
        "summary": {
            "d1c_row_count": len(d1c_payload.get("rows", [])),
            "e1_feature_row_count": len(feature_rows),
            "one_feature_row_per_d1c_row": len(d1c_payload.get("rows", [])) == len(feature_rows),
            "c2c_colour_belief_counts": dict(sorted(feature_counts.items())),
            "non_outfield_colour_hint_count": sum(1 for row in feature_rows if row.get("non_outfield_colour_hint") is True),
            "image_space_goal_area_context_hint_count": sum(1 for row in feature_rows if row.get("image_space_goal_area_context_hint") is True),
            "source_goalkeeper_hint_count": sum(1 for row in feature_rows if row.get("source_goalkeeper_hint") is True),
        },
    }
    belief_payload = {
        "artifact": "step1e1_goalkeeper_context_belief_rows",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "e1_is_not_slot_identity_or_expected_role_stage": True,
        "allowed_e1_goalkeeper_context_beliefs": sorted(ALLOWED_E1_GOALKEEPER_CONTEXT_BELIEFS),
        "rows": belief_rows,
        "summary": {
            "d1c_row_count": len(d1c_payload.get("rows", [])),
            "e1_belief_row_count": len(belief_rows),
            "one_belief_row_per_d1c_row": len(d1c_payload.get("rows", [])) == len(belief_rows),
            "e1_goalkeeper_context_belief_counts": dict(sorted(belief_counts.items())),
            "e1_goalkeeper_like_visual_context_count": sum(1 for row in belief_rows if row.get("e1_goalkeeper_like_visual_context") is True),
            "e1_review_required_count": sum(1 for row in belief_rows if row.get("e1_goalkeeper_context_review_required") is True),
            "all_rows_retained_for_future_player_team_review": all(row.get("retained_for_future_player_team_review") is True for row in belief_rows),
            "forbidden_keys_present": forbidden_keys_present(belief_rows),
            "exact_two_goalkeeper_forcing_performed": False,
        },
    }
    return feature_payload, belief_payload, review_payload


def goalkeeper_context_report(
    feature_payload: dict[str, Any],
    belief_payload: dict[str, Any],
    review_payload: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Step1.E1 Goalkeeper Context Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Scope: visual goalkeeper/context belief sandbox only.",
            "- This is not a slot stage, identity tracking, expected-role reconstruction, or metric layer.",
            "- Goalkeeper-like labels are visual context candidates only; no exact two-goalkeeper forcing is performed.",
            "- Officials/context people are retained and not excluded from future player/team review.",
            "",
            "## Counts",
            "",
            f"- D1c rows: {belief_payload.get('summary', {}).get('d1c_row_count', 0)}",
            f"- E1 feature rows: {feature_payload.get('summary', {}).get('e1_feature_row_count', 0)}",
            f"- E1 belief rows: {belief_payload.get('summary', {}).get('e1_belief_row_count', 0)}",
            f"- Review candidates: {review_payload.get('summary', {}).get('e1_review_candidate_count', 0)}",
            f"- Goalkeeper-like visual context rows: {belief_payload.get('summary', {}).get('e1_goalkeeper_like_visual_context_count', 0)}",
            "",
            "## E1 Belief Counts",
            "",
            "```json",
            json.dumps(belief_payload.get("summary", {}).get("e1_goalkeeper_context_belief_counts", {}), indent=2),
            "```",
            "",
            "## Review Candidate Reasons",
            "",
            "```json",
            json.dumps(review_payload.get("summary", {}).get("review_reason_counts", {}), indent=2),
            "```",
        ]
    ) + "\n"


def build_and_write_goalkeeper_context_beliefs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    d1c_payload = read_json(STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH)
    c2c_payload = read_json(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH)
    feature_payload, belief_payload, review_payload = build_goalkeeper_context_payloads(d1c_payload, c2c_payload)
    write_json(STEP1E1_GOALKEEPER_CONTEXT_FEATURE_ROWS_PATH, feature_payload)
    write_json(STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH, belief_payload)
    write_json(STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH, review_payload)
    write_text(STEP1E1_GOALKEEPER_CONTEXT_REPORT_PATH, goalkeeper_context_report(feature_payload, belief_payload, review_payload))
    return feature_payload, belief_payload, review_payload
