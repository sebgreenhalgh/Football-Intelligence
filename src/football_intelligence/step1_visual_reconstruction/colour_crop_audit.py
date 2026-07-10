# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.colour_cluster_diagnostics import (
    dominant_cluster,
    gold_proxy_confusion_rows,
)
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C1_COLOUR_FEATURE_ROWS_PATH,
    STEP1C1_GOLD8_COLOUR_EVAL_SUMMARY_PATH,
    STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH,
    STEP1C1B_CROP_AUDIT_ROWS_PATH,
    STEP1C1B_CROP_AUDIT_SUMMARY_PATH,
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


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def c1_gold_shared_dominant_cluster(c1_eval_summary: dict[str, Any]) -> str:
    distribution = c1_eval_summary.get("gold8_colour_eval_summary", {}).get("gold8_colour_proxy_distribution", {})
    team_1_cluster, _count_1, _purity_1 = dominant_cluster(distribution.get("team_1_player", {}))
    team_2_cluster, _count_2, _purity_2 = dominant_cluster(distribution.get("team_2_player", {}))
    if team_1_cluster and team_1_cluster == team_2_cluster:
        return team_1_cluster
    return ""


def c1_contradiction_frames(c1_eval_summary: dict[str, Any]) -> set[int]:
    diagnostics = c1_eval_summary.get("gold8_colour_eval_summary", {}).get("one_frame_colour_contradiction_diagnostics", [])
    frames = set(c1_eval_summary.get("gold8_colour_eval_summary", {}).get("frames_needing_manual_followup", []))
    for item in diagnostics:
        frames.add(item.get("frame_sequence", -1))
    return {int(safe_float(seq, -1)) for seq in frames if int(safe_float(seq, -1)) >= 0}


def recommended_profiles(flags: list[str]) -> list[str]:
    recommendations = []
    if "small_torso_crop" in flags:
        recommendations.extend(["torso_wider", "high_resolution_near_only"])
    if "background_contaminated_crop" in flags or "mostly_green_background" in flags:
        recommendations.extend(["adaptive_non_green_core", "central_body_excluding_grass"])
    if "unknown_on_gold_player_proxy" in flags:
        recommendations.extend(["torso_upper_only", "torso_lower"])
    if "dark_context_on_gold_player_proxy" in flags:
        recommendations.extend(["torso_upper_only", "adaptive_non_green_core"])
    if "frame_colour_contradiction" in flags:
        recommendations.append("manual_frame_followup")
    return list(dict.fromkeys(recommendations))


def audit_flags_for_row(
    feature: dict[str, Any],
    belief: dict[str, Any],
    *,
    gold_proxy_row: dict[str, Any] | None,
    shared_gold_proxy_cluster: str,
    contradiction_frames: set[int],
) -> list[str]:
    flags = []
    warning = str(feature.get("feature_extraction_warning", "") or feature.get("crop_quality_reason", ""))
    if warning == "small_torso_crop" or feature.get("crop_quality_reason") == "small_torso_crop":
        flags.append("small_torso_crop")
    if warning == "background_contaminated_crop" or feature.get("crop_quality_reason") == "background_contaminated_crop":
        flags.append("background_contaminated_crop")
    if warning == "mostly_green_background" or safe_float(feature.get("green_background_fraction")) > 0.78:
        flags.append("mostly_green_background")
    if gold_proxy_row and shared_gold_proxy_cluster and belief.get("team_colour_belief") == shared_gold_proxy_cluster:
        flags.append("team_1_team_2_same_cluster_proxy")
    if (
        gold_proxy_row
        and shared_gold_proxy_cluster
        and belief.get("team_colour_belief") == shared_gold_proxy_cluster
        and belief.get("team_colour_belief_state") == "high_confidence_visual_colour"
    ):
        flags.append("high_confidence_but_gold_proxy_confused")
    context_like = str(belief.get("candidate_type", "")) in {
        "official_candidate_source",
        "referee_candidate_source",
        "staff_context_candidate_source",
        "unknown_candidate_source",
        "off_pitch_person_candidate",
        "unknown_person_candidate",
    } or str(belief.get("roi_status", "")) == "outside_playing_roi"
    if context_like and safe_float(belief.get("team_colour_belief_confidence")) >= 0.78:
        flags.append("context_or_offroi_high_colour_confidence")
    if gold_proxy_row and belief.get("team_colour_belief") == "dark_context_colour_like":
        flags.append("dark_context_on_gold_player_proxy")
    if gold_proxy_row and belief.get("team_colour_belief") in {"unknown_ambiguous_colour", "crop_unusable"}:
        flags.append("unknown_on_gold_player_proxy")
    if int(safe_float(feature.get("frame_sequence"), -1)) in contradiction_frames:
        flags.append("frame_colour_contradiction")
    if any(
        flag in flags
        for flag in {
            "small_torso_crop",
            "background_contaminated_crop",
            "mostly_green_background",
            "high_confidence_but_gold_proxy_confused",
            "team_1_team_2_same_cluster_proxy",
            "dark_context_on_gold_player_proxy",
            "unknown_on_gold_player_proxy",
            "frame_colour_contradiction",
        }
    ):
        flags.append("needs_manual_crop_review")
    return list(dict.fromkeys(flags))


def build_crop_audit_payloads(
    feature_payload: dict[str, Any],
    belief_payload: dict[str, Any],
    c1_eval_summary: dict[str, Any],
    *,
    gold_proxy_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    beliefs_by_base_id = {str(row.get("visible_person_base_id", "")): row for row in belief_payload.get("rows", [])}
    confusion_by_base_id = {
        str(row.get("visible_person_base_id", "")): row
        for row in (
            gold_proxy_rows
            if gold_proxy_rows is not None
            else gold_proxy_confusion_rows(
                belief_payload,
                profile_name="c1_current",
                prototype_strategy="c1_top_chromatic",
            )
        )
    }
    shared_cluster = c1_gold_shared_dominant_cluster(c1_eval_summary)
    contradiction_frames = c1_contradiction_frames(c1_eval_summary)
    rows = []
    flag_counts: Counter[str] = Counter()
    recommendation_counts: Counter[str] = Counter()
    for feature in feature_payload.get("rows", []):
        base_id = str(feature.get("visible_person_base_id", ""))
        belief = beliefs_by_base_id.get(base_id, {})
        flags = audit_flags_for_row(
            feature,
            belief,
            gold_proxy_row=confusion_by_base_id.get(base_id),
            shared_gold_proxy_cluster=shared_cluster,
            contradiction_frames=contradiction_frames,
        )
        recommendations = recommended_profiles(flags)
        flag_counts.update(flags)
        recommendation_counts.update(recommendations)
        rows.append(
            {
                "frame_id": feature.get("frame_id", ""),
                "frame_sequence": int(safe_float(feature.get("frame_sequence"), -1)),
                "timestamp_seconds": safe_float(feature.get("timestamp_seconds")),
                "visible_person_base_id": base_id,
                "detection_id": feature.get("detection_id", ""),
                "source_detection_id": feature.get("source_detection_id", ""),
                "bbox": feature.get("bbox", {}),
                "torso_crop_bbox": feature.get("torso_crop_bbox"),
                "crop_width": int(safe_float(feature.get("crop_width"), 0)),
                "crop_height": int(safe_float(feature.get("crop_height"), 0)),
                "crop_quality": feature.get("crop_quality", ""),
                "crop_quality_reason": feature.get("crop_quality_reason", ""),
                "feature_extraction_warning": feature.get("feature_extraction_warning", ""),
                "green_background_fraction": safe_float(feature.get("green_background_fraction")),
                "blue_like_fraction": safe_float(feature.get("blue_like_fraction")),
                "red_or_orange_like_fraction": safe_float(feature.get("red_or_orange_like_fraction")),
                "white_like_fraction": safe_float(feature.get("white_like_fraction")),
                "black_or_dark_like_fraction": safe_float(feature.get("black_or_dark_like_fraction")),
                "current_team_colour_belief": belief.get("team_colour_belief", ""),
                "current_team_colour_belief_state": belief.get("team_colour_belief_state", ""),
                "current_colour_cluster_candidate": belief.get("colour_cluster_candidate", ""),
                "current_confidence": safe_float(belief.get("team_colour_belief_confidence")),
                "gold_visible_person_type_proxy": confusion_by_base_id.get(base_id, {}).get("visible_person_type_gold", ""),
                "audit_issue_flags": flags,
                "recommended_crop_profile_review": recommendations,
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
            }
        )
    audit_payload = {
        "artifact": "step1c1b_crop_audit_rows",
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
        "summary": {"step1c1b_crop_audit_rows": len(rows)},
    }
    summary_payload = {
        "artifact": "step1c1b_crop_audit_summary",
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
        "step1c1_colour_feature_rows": len(feature_payload.get("rows", [])),
        "step1c1_team_colour_belief_rows": len(belief_payload.get("rows", [])),
        "step1c1b_crop_audit_rows": len(rows),
        "audit_issue_flag_counts": dict(sorted(flag_counts.items())),
        "recommended_crop_profile_review_counts": dict(sorted(recommendation_counts.items())),
        "shared_gold_proxy_dominant_cluster": shared_cluster,
        "frames_needing_manual_followup": sorted(contradiction_frames),
        "c1_baseline": {
            "unknown_ambiguous_colour_rows": c1_eval_summary.get("unknown_ambiguous_colour_rows", 0),
            "cluster_counts": c1_eval_summary.get("cluster_counts", {}),
            "gold8_proxy_distribution": c1_eval_summary.get("gold8_colour_eval_summary", {}).get("gold8_colour_proxy_distribution", {}),
        },
    }
    return audit_payload, summary_payload


def build_and_write_crop_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    feature_payload = read_json(STEP1C1_COLOUR_FEATURE_ROWS_PATH)
    belief_payload = read_json(STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH)
    c1_eval_summary = read_json(STEP1C1_GOLD8_COLOUR_EVAL_SUMMARY_PATH)
    audit_payload, summary_payload = build_crop_audit_payloads(feature_payload, belief_payload, c1_eval_summary)
    write_json(STEP1C1B_CROP_AUDIT_ROWS_PATH, audit_payload)
    write_json(STEP1C1B_CROP_AUDIT_SUMMARY_PATH, summary_payload)
    return audit_payload, summary_payload
