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
    STEP1D1_OFFICIAL_CONTEXT_BELIEF_REPORT_PATH,
    STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH,
    STEP1D1_OFFICIAL_CONTEXT_FEATURE_ROWS_PATH,
    STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    safe_float,
)


ALLOWED_OFFICIAL_CONTEXT_BELIEFS = {
    "official_referee_like",
    "assistant_or_line_official_like",
    "non_official_context_person_like",
    "off_pitch_context_person_like",
    "player_like_not_official_context",
    "bad_detection_or_not_person",
    "unknown_official_context",
}
ALLOWED_OFFICIAL_CONTEXT_STATES = {
    "high_confidence_visual_context",
    "medium_confidence_visual_context",
    "low_confidence_visual_context",
    "ambiguous_visual_context",
    "review_required",
    "bad_detection_review_required",
}
OFFICIAL_LIKE_BELIEFS = {"official_referee_like", "assistant_or_line_official_like"}
CONTEXT_LIKE_BELIEFS = {
    "official_referee_like",
    "assistant_or_line_official_like",
    "non_official_context_person_like",
    "off_pitch_context_person_like",
    "unknown_official_context",
}
TEAM_COLOUR_BELIEFS = {"team_1_outfield_colour_like", "team_2_outfield_colour_like"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def review_id_for_row(row: dict[str, Any]) -> str:
    raw = f"{row.get('visible_person_base_id','')}|{row.get('official_context_belief','')}|{row.get('frame_sequence','')}"
    return f"step1d1_review_f{int(safe_float(row.get('frame_sequence'), -1)):06d}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]}"


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def belief_decision(feature: dict[str, Any]) -> dict[str, Any]:
    warning_flags: list[str] = []
    source_official = boolish(feature.get("source_official_candidate_flag"))
    source_player = boolish(feature.get("source_player_candidate_flag"))
    source_unknown_context = boolish(feature.get("source_unknown_context_candidate_flag"))
    offroi = boolish(feature.get("offroi_or_recovery_context_flag"))
    near_touchline = boolish(feature.get("image_space_near_touchline_context_flag"))
    lower_band = boolish(feature.get("image_space_lower_frame_band_flag"))
    team_colour = boolish(feature.get("team_colour_like_flag"))
    non_team_colour = boolish(feature.get("non_team_colour_like_flag"))
    dark = boolish(feature.get("dark_or_black_like_visual_flag"))
    bright = boolish(feature.get("bright_referee_colour_like_visual_flag"))
    red_or_pink = boolish(feature.get("red_or_pink_like_visual_flag"))
    yellow_or_orange = boolish(feature.get("yellow_or_orange_like_visual_flag"))
    mixed = boolish(feature.get("mixed_colour_or_overlap_warning"))
    context_override = boolish(feature.get("c2c_context_or_offroi_human_team_override"))
    bad_detection = boolish(feature.get("bad_detection_candidate_flag"))

    if context_override:
        warning_flags.append("c2c_context_offroi_human_team_override_not_player_evidence")
    if mixed:
        warning_flags.append("mixed_colour_or_overlap_visual_warning")
    if dark:
        warning_flags.append("dark_colour_is_visual_hint_not_official_proof")
    if red_or_pink or yellow_or_orange:
        warning_flags.append("bright_colour_is_visual_hint_not_official_proof")
    if near_touchline or lower_band:
        warning_flags.append("image_space_position_is_not_metric_truth")

    if bad_detection:
        return {
            "belief": "bad_detection_or_not_person",
            "state": "bad_detection_review_required",
            "confidence": 0.95,
            "review_required": True,
            "reason": "c2c_bad_detection_or_crop_unusable_preserved_for_review",
            "warning_flags": warning_flags,
        }

    official_visual_support = dark or bright or red_or_pink or yellow_or_orange or non_team_colour
    if source_official and official_visual_support:
        if near_touchline or offroi or lower_band:
            belief = "assistant_or_line_official_like"
            reason = "source_official_candidate_with_touchline_or_context_visual_support"
        else:
            belief = "official_referee_like"
            reason = "source_official_candidate_with_colour_visual_support"
        confidence = 0.82 if not mixed else 0.68
        return {
            "belief": belief,
            "state": "medium_confidence_visual_context" if confidence < 0.88 else "high_confidence_visual_context",
            "confidence": confidence,
            "review_required": True,
            "reason": reason,
            "warning_flags": warning_flags,
        }

    if source_official:
        return {
            "belief": "unknown_official_context",
            "state": "review_required",
            "confidence": 0.52,
            "review_required": True,
            "reason": "source_official_candidate_without_enough_visual_support",
            "warning_flags": warning_flags + ["source_provenance_not_role_proof"],
        }

    if context_override:
        belief = "off_pitch_context_person_like" if offroi or near_touchline else "unknown_official_context"
        return {
            "belief": belief,
            "state": "review_required",
            "confidence": 0.58,
            "review_required": True,
            "reason": "c2c_context_offroi_human_team_override_preserved_as_visual_context_review",
            "warning_flags": warning_flags,
        }

    if offroi or source_unknown_context:
        belief = "off_pitch_context_person_like" if offroi or near_touchline else "non_official_context_person_like"
        return {
            "belief": belief,
            "state": "medium_confidence_visual_context" if not mixed else "ambiguous_visual_context",
            "confidence": 0.72 if not mixed else 0.56,
            "review_required": True,
            "reason": "context_or_offroi_provenance_kept_as_visual_context_candidate",
            "warning_flags": warning_flags,
        }

    if source_player and team_colour and not mixed:
        return {
            "belief": "player_like_not_official_context",
            "state": "high_confidence_visual_context",
            "confidence": 0.88,
            "review_required": False,
            "reason": "team_colour_like_player_source_retained_as_player_like_visual_context",
            "warning_flags": warning_flags,
        }

    if source_player and (dark or bright or non_team_colour or mixed):
        return {
            "belief": "unknown_official_context",
            "state": "ambiguous_visual_context",
            "confidence": 0.48,
            "review_required": True,
            "reason": "player_source_with_mixed_or_non_team_visual_context_not_forced_official",
            "warning_flags": warning_flags,
        }

    if team_colour and not mixed:
        return {
            "belief": "player_like_not_official_context",
            "state": "medium_confidence_visual_context",
            "confidence": 0.74,
            "review_required": False,
            "reason": "team_colour_like_visual_context_retained_as_player_like_hint",
            "warning_flags": warning_flags,
        }

    return {
        "belief": "unknown_official_context",
        "state": "review_required",
        "confidence": 0.42,
        "review_required": True,
        "reason": "insufficient_visual_context_for_official_or_player_like_belief",
        "warning_flags": warning_flags,
    }


def belief_row(feature: dict[str, Any]) -> dict[str, Any]:
    decision = belief_decision(feature)
    belief = str(decision["belief"])
    state = str(decision["state"])
    if belief not in ALLOWED_OFFICIAL_CONTEXT_BELIEFS:
        belief = "unknown_official_context"
        state = "review_required"
    if state not in ALLOWED_OFFICIAL_CONTEXT_STATES:
        state = "review_required"
    return {
        "visible_person_base_id": feature.get("visible_person_base_id", ""),
        "frame_id": feature.get("frame_id", ""),
        "frame_sequence": int(safe_float(feature.get("frame_sequence"), -1)),
        "timestamp_seconds": safe_float(feature.get("timestamp_seconds")),
        "detection_id": feature.get("detection_id", ""),
        "source_detection_id": feature.get("source_detection_id", ""),
        "bbox": feature.get("bbox", {}),
        "footpoint": feature.get("footpoint", {}),
        "state": feature.get("state", ""),
        "roi_status": feature.get("roi_status", ""),
        "candidate_type": feature.get("candidate_type", ""),
        "original_role_source": feature.get("original_role_source", ""),
        "c2c_final_colour_belief": feature.get("c2c_final_colour_belief", ""),
        "c2c_colour_source": feature.get("c2c_colour_source", ""),
        "c2c_human_reviewed": boolish(feature.get("c2c_human_reviewed")),
        "c2c_context_or_offroi_human_team_override": boolish(feature.get("c2c_context_or_offroi_human_team_override")),
        "official_context_feature_id": feature.get("official_context_feature_id", ""),
        "official_context_belief": belief,
        "official_context_belief_state": state,
        "official_context_belief_confidence": round(safe_float(decision.get("confidence")), 4),
        "official_context_belief_reason": decision.get("reason", ""),
        "official_context_review_required": boolish(decision.get("review_required")),
        "official_context_warning_flags": decision.get("warning_flags", []),
        "source_official_candidate_flag": boolish(feature.get("source_official_candidate_flag")),
        "source_player_candidate_flag": boolish(feature.get("source_player_candidate_flag")),
        "offroi_or_recovery_context_flag": boolish(feature.get("offroi_or_recovery_context_flag")),
        "team_colour_like_flag": boolish(feature.get("team_colour_like_flag")),
        "non_team_colour_like_flag": boolish(feature.get("non_team_colour_like_flag")),
        "bad_detection_candidate_flag": boolish(feature.get("bad_detection_candidate_flag")),
        "retained_for_future_player_team_review": True,
        "eligible_for_step1e_goalkeeper_candidate": True,
        "eligible_for_identity_tracking": False,
        "eligible_for_player_slot_assignment": False,
        "eligible_for_metric_use": False,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "auto_promoted": False,
    }


def gold_official_matched_visible_ids(belief_rows: list[dict[str, Any]]) -> set[str]:
    labels_payload = read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    gold_rows = [
        row
        for row in gold_visible_person_rows(labels_payload)
        if row.get("visible_person_type_gold") == "official_referee"
    ]
    matches, _missed, _extra = strict_one_to_one_match(gold_rows, belief_rows)
    return {str(match["candidate"].get("visible_person_base_id", "")) for match in matches}


def review_reason_tags(row: dict[str, Any], gold_official_ids: set[str]) -> list[str]:
    tags = []
    belief = str(row.get("official_context_belief", ""))
    if belief in OFFICIAL_LIKE_BELIEFS:
        tags.append("official_like_belief")
    if belief == "off_pitch_context_person_like":
        tags.append("off_pitch_context_belief")
    if belief == "bad_detection_or_not_person":
        tags.append("bad_detection_belief")
    if row.get("source_official_candidate_flag") is True:
        tags.append("source_official_candidate")
    if row.get("c2c_context_or_offroi_human_team_override") is True:
        tags.append("c2c_context_offroi_human_team_override")
    if str(row.get("visible_person_base_id", "")) in gold_official_ids:
        tags.append("gold8_official_proxy_match")
    if row.get("c2c_final_colour_belief") in TEAM_COLOUR_BELIEFS and belief in CONTEXT_LIKE_BELIEFS - {"unknown_official_context"}:
        tags.append("team_colour_with_context_like_belief")
    if row.get("official_context_review_required") is True:
        tags.append("review_required")
    return tags


def review_priority_for_tags(tags: list[str]) -> int:
    priority = 80
    weights = {
        "gold8_official_proxy_match": 0,
        "official_like_belief": 5,
        "source_official_candidate": 8,
        "c2c_context_offroi_human_team_override": 10,
        "team_colour_with_context_like_belief": 12,
        "bad_detection_belief": 15,
        "off_pitch_context_belief": 20,
        "review_required": 35,
    }
    for tag in tags:
        priority = min(priority, weights.get(tag, priority))
    return priority


def build_review_candidate_payload(belief_payload: dict[str, Any], *, sample_per_group: int = 90) -> dict[str, Any]:
    belief_rows = belief_payload.get("rows", [])
    gold_official_ids = gold_official_matched_visible_ids(belief_rows)
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
            "step1d1_review_candidate_id": review_id_for_row(row),
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
        tags = review_reason_tags(row, gold_official_ids)
        belief = str(row.get("official_context_belief", ""))
        if tags and (
            belief in OFFICIAL_LIKE_BELIEFS
            or belief in {"off_pitch_context_person_like", "bad_detection_or_not_person"}
            or row.get("source_official_candidate_flag") is True
            or row.get("c2c_context_or_offroi_human_team_override") is True
            or str(row.get("visible_person_base_id", "")) in gold_official_ids
            or "team_colour_with_context_like_belief" in tags
        ):
            add(row, tags)
    for belief in ["player_like_not_official_context", "unknown_official_context"]:
        selected = [
            row
            for row in belief_rows
            if row.get("official_context_belief") == belief
            and str(row.get("visible_person_base_id", "")) not in candidates
        ]
        selected.sort(key=lambda item: (int(safe_float(item.get("frame_sequence"), -1)), str(item.get("visible_person_base_id", ""))))
        for row in selected[:sample_per_group]:
            add(row, [f"balanced_sample_{belief}"])
    rows = sorted(candidates.values(), key=lambda row: (int(row.get("review_priority", 999)), int(safe_float(row.get("frame_sequence"), -1)), str(row.get("visible_person_base_id", ""))))
    return {
        "artifact": "step1d1_official_context_review_candidate_rows",
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
        "summary": {
            "d1_review_candidate_count": len(rows),
            "unique_visible_person_base_ids": len({row.get("visible_person_base_id") for row in rows}),
            "gold8_official_proxy_review_candidate_count": sum(1 for row in rows if "gold8_official_proxy_match" in row.get("review_reason_tags", [])),
            "review_reason_counts": dict(sorted(Counter(tag for row in rows for tag in row.get("review_reason_tags", [])).items())),
        },
    }


def build_official_context_belief_payload(feature_payload: dict[str, Any]) -> dict[str, Any]:
    rows = [belief_row(feature) for feature in feature_payload.get("rows", [])]
    belief_counts = Counter(str(row.get("official_context_belief", "")) for row in rows)
    state_counts = Counter(str(row.get("official_context_belief_state", "")) for row in rows)
    return {
        "artifact": "step1d1_official_context_belief_rows",
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
        "d1_is_not_official_exclusion_stage": True,
        "allowed_official_context_beliefs": sorted(ALLOWED_OFFICIAL_CONTEXT_BELIEFS),
        "allowed_official_context_states": sorted(ALLOWED_OFFICIAL_CONTEXT_STATES),
        "rows": rows,
        "summary": {
            "d1_feature_row_count": len(feature_payload.get("rows", [])),
            "d1_belief_row_count": len(rows),
            "one_belief_row_per_feature_row": len(feature_payload.get("rows", [])) == len(rows),
            "official_context_belief_counts": dict(sorted(belief_counts.items())),
            "official_context_state_counts": dict(sorted(state_counts.items())),
            "review_required_count": sum(1 for row in rows if row.get("official_context_review_required") is True),
            "source_official_candidate_count": sum(1 for row in rows if row.get("source_official_candidate_flag") is True),
            "source_player_candidate_count": sum(1 for row in rows if row.get("source_player_candidate_flag") is True),
            "c2c_context_offroi_human_team_override_count": sum(1 for row in rows if row.get("c2c_context_or_offroi_human_team_override") is True),
            "official_referee_exclusion_performed": False,
            "all_rows_retained_for_future_player_team_review": all(row.get("retained_for_future_player_team_review") is True for row in rows),
        },
    }


def official_context_belief_report(belief_payload: dict[str, Any], review_payload: dict[str, Any]) -> str:
    summary = belief_payload.get("summary", {})
    return "\n".join(
        [
            "# Step1.D1 Official/Context Belief Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Scope: visual official/referee/context-person QA beliefs only.",
            "- D1 is not an official/referee exclusion stage.",
            "- No identity tracking, player slots, expected roles, goalkeeper classification, or metrics are created.",
            "",
            "## Counts",
            "",
            f"- Feature rows: {summary.get('d1_feature_row_count', 0)}",
            f"- Belief rows: {summary.get('d1_belief_row_count', 0)}",
            f"- Review-required rows: {summary.get('review_required_count', 0)}",
            f"- Review candidates: {review_payload.get('summary', {}).get('d1_review_candidate_count', 0)}",
            f"- Official/referee exclusion performed: {summary.get('official_referee_exclusion_performed', False)}",
            "",
            "## Belief Counts",
            "",
            "```json",
            json.dumps(summary.get("official_context_belief_counts", {}), indent=2),
            "```",
            "",
            "## Review Candidate Reasons",
            "",
            "```json",
            json.dumps(review_payload.get("summary", {}).get("review_reason_counts", {}), indent=2),
            "```",
        ]
    ) + "\n"


def build_and_write_official_context_beliefs() -> tuple[dict[str, Any], dict[str, Any]]:
    feature_payload = read_json(STEP1D1_OFFICIAL_CONTEXT_FEATURE_ROWS_PATH)
    belief_payload = build_official_context_belief_payload(feature_payload)
    review_payload = build_review_candidate_payload(belief_payload)
    write_json(STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH, belief_payload)
    write_json(STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH, review_payload)
    write_text(STEP1D1_OFFICIAL_CONTEXT_BELIEF_REPORT_PATH, official_context_belief_report(belief_payload, review_payload))
    return belief_payload, review_payload
