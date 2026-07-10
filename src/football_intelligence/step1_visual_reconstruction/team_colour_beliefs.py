# ruff: noqa: E501

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from statistics import median
from typing import Any

from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH,
    STEP1C1_COLOUR_FEATURE_ROWS_PATH,
    STEP1C1_COLOUR_PROTOTYPES_PATH,
    STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH,
    STEP1C1_UNKNOWN_AMBIGUOUS_COLOUR_ROWS_PATH,
    read_json,
    write_json,
    write_text,
    STEP1C1_COLOUR_BELIEF_REPORT_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    safe_float,
)


UNKNOWN_CONTEXT_TYPES = {
    "official_candidate_source",
    "referee_candidate_source",
    "staff_context_candidate_source",
    "unknown_candidate_source",
    "off_pitch_person_candidate",
    "unknown_person_candidate",
}

BELIEF_VALUES = {
    "team_1_colour_like",
    "team_2_colour_like",
    "outfield_colour_cluster_a",
    "outfield_colour_cluster_b",
    "other_distinct_colour_like",
    "dark_context_colour_like",
    "unknown_ambiguous_colour",
    "crop_unusable",
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def hue_sector(feature: dict[str, Any]) -> str:
    hsv = feature.get("median_hsv", [])
    if not hsv:
        return "unknown"
    hue = safe_float(hsv[0])
    if hue <= 15 or hue >= 165:
        return "red_or_orange"
    if 16 <= hue <= 32:
        return "yellow_or_orange"
    if 33 <= hue <= 90:
        return "green_contaminated"
    if 91 <= hue <= 130:
        return "blue"
    if 131 <= hue <= 164:
        return "purple_or_magenta"
    return "other"


def feature_is_usable_for_colour(feature: dict[str, Any]) -> bool:
    return (
        feature.get("crop_quality") in {"high", "medium"}
        and safe_float(feature.get("green_background_fraction")) <= 0.62
        and bool(feature.get("median_hsv"))
    )


def preliminary_colour_bucket(feature: dict[str, Any]) -> str:
    if feature.get("crop_quality") == "unusable":
        return "crop_unusable"
    if safe_float(feature.get("black_or_dark_like_fraction")) >= 0.45:
        return "dark_context_colour_cluster"
    if not feature_is_usable_for_colour(feature):
        return "unknown_ambiguous_colour"
    if safe_float(feature.get("white_like_fraction")) >= 0.58 and safe_float(feature.get("saturation_summary", {}).get("median")) < 55:
        return "white_or_light_colour"
    return hue_sector(feature)


def median_triplet(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [row.get(key, []) for row in rows if isinstance(row.get(key), list) and len(row.get(key)) == 3]
    if not values:
        return []
    return [round(float(median([safe_float(value[index]) for value in values])), 3) for index in range(3)]


def build_colour_prototypes(feature_payload: dict[str, Any]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in feature_payload.get("rows", []):
        buckets[preliminary_colour_bucket(feature)].append(feature)
    chromatic = {
        key: rows
        for key, rows in buckets.items()
        if key not in {"unknown_ambiguous_colour", "crop_unusable", "dark_context_colour_cluster", "green_contaminated"}
    }
    top_chromatic = sorted(chromatic.items(), key=lambda item: (-len(item[1]), item[0]))
    cluster_a_source = top_chromatic[0][0] if top_chromatic else ""
    cluster_b_source = top_chromatic[1][0] if len(top_chromatic) > 1 else ""
    assignments = {
        "outfield_colour_cluster_a": buckets.get(cluster_a_source, []),
        "outfield_colour_cluster_b": buckets.get(cluster_b_source, []),
        "dark_context_colour_cluster": buckets.get("dark_context_colour_cluster", []),
        "unknown_ambiguous_colour": buckets.get("unknown_ambiguous_colour", []) + buckets.get("crop_unusable", []) + buckets.get("green_contaminated", []),
    }
    used_sources = {cluster_a_source, cluster_b_source, "dark_context_colour_cluster", "unknown_ambiguous_colour", "crop_unusable", "green_contaminated"}
    other_rows: list[dict[str, Any]] = []
    for key, rows in buckets.items():
        if key not in used_sources:
            other_rows.extend(rows)
    assignments["other_distinct_colour_cluster"] = other_rows
    prototypes = []
    for name, rows in assignments.items():
        prototypes.append(
            {
                "colour_cluster_candidate": name,
                "source_bucket": cluster_a_source if name == "outfield_colour_cluster_a" else cluster_b_source if name == "outfield_colour_cluster_b" else name,
                "row_count": len(rows),
                "median_hsv": median_triplet(rows, "median_hsv"),
                "median_lab": median_triplet(rows, "median_lab"),
            }
        )
    return {
        "artifact": "step1c1_colour_prototypes",
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
        "prototype_sandbox_only": True,
        "auto_promoted": False,
        "safe_team_mapping_found": False,
        "mapped_team_colour_candidate": "unknown_mapping",
        "mapping_confidence": 0.0,
        "mapping_reason": "no_safe_team_colour_configuration_was_used",
        "prototypes": prototypes,
        "summary": {
            "prototype_count": len(prototypes),
            "cluster_a_source_bucket": cluster_a_source,
            "cluster_b_source_bucket": cluster_b_source,
            "source_bucket_counts": dict(sorted((key, len(rows)) for key, rows in buckets.items())),
        },
    }


def prototype_source_map(prototypes_payload: dict[str, Any]) -> dict[str, str]:
    out = {}
    for proto in prototypes_payload.get("prototypes", []):
        candidate = str(proto.get("colour_cluster_candidate", ""))
        source = str(proto.get("source_bucket", ""))
        if source:
            out[source] = candidate
    return out


def cluster_confidence(feature: dict[str, Any], cluster: str) -> float:
    if cluster in {"crop_unusable", "unknown_ambiguous_colour"}:
        return 0.0 if cluster == "crop_unusable" else 0.25
    quality = str(feature.get("crop_quality", ""))
    green = safe_float(feature.get("green_background_fraction"))
    dark = safe_float(feature.get("black_or_dark_like_fraction"))
    confidence = 0.72
    if quality == "high":
        confidence += 0.12
    if green > 0.45:
        confidence -= 0.12
    if dark > 0.50 and cluster != "dark_context_colour_cluster":
        confidence -= 0.12
    return round(max(0.05, min(0.92, confidence)), 4)


def belief_state(confidence: float, belief: str, row: dict[str, Any], feature: dict[str, Any]) -> str:
    if belief == "crop_unusable":
        return "crop_unusable"
    if row.get("review_required") is True or row.get("source_disagreement_review_required") is True:
        return "review_required"
    if belief == "unknown_ambiguous_colour":
        return "ambiguous_visual_colour"
    if confidence >= 0.78 and feature.get("crop_quality") == "high":
        return "high_confidence_visual_colour"
    if confidence >= 0.58:
        return "medium_confidence_visual_colour"
    return "low_confidence_visual_colour"


def belief_for_row(row: dict[str, Any], feature: dict[str, Any], prototypes_payload: dict[str, Any]) -> dict[str, Any]:
    source_bucket = preliminary_colour_bucket(feature)
    source_map = prototype_source_map(prototypes_payload)
    cluster = source_map.get(source_bucket, "unknown_ambiguous_colour")
    if (
        cluster == "unknown_ambiguous_colour"
        and source_bucket
        not in {"unknown_ambiguous_colour", "crop_unusable", "green_contaminated"}
    ):
        cluster = "other_distinct_colour_cluster"
    if source_bucket == "crop_unusable":
        belief = "crop_unusable"
        ambiguous_reason = "crop_unusable"
    elif cluster == "dark_context_colour_cluster":
        belief = "dark_context_colour_like"
        ambiguous_reason = ""
    elif cluster == "other_distinct_colour_cluster":
        belief = "other_distinct_colour_like"
        ambiguous_reason = ""
    elif cluster in {"outfield_colour_cluster_a", "outfield_colour_cluster_b"}:
        belief = cluster
        ambiguous_reason = ""
    else:
        belief = "unknown_ambiguous_colour"
        ambiguous_reason = source_bucket
    confidence = cluster_confidence(feature, source_bucket if source_bucket == "crop_unusable" else cluster)
    context_like = str(row.get("candidate_type", "")) in UNKNOWN_CONTEXT_TYPES or str(row.get("roi_status", "")) == "outside_playing_roi"
    if context_like and belief in {"outfield_colour_cluster_a", "outfield_colour_cluster_b"} and confidence < 0.84:
        ambiguous_reason = "context_or_off_roi_not_forced_to_team_colour_cluster"
        belief = "unknown_ambiguous_colour"
        cluster = "unknown_ambiguous_colour"
        confidence = min(confidence, 0.35)
    state = belief_state(confidence, belief, row, feature)
    reason = "visual_colour_cluster_only_no_role_decision"
    if belief == "unknown_ambiguous_colour":
        reason = "insufficient_or_ambiguous_visual_colour_evidence"
    if state == "review_required":
        reason = "b4_review_required_colour_belief_needs_human_review"
    return {
        "frame_id": row.get("frame_id", ""),
        "frame_sequence": row.get("frame_sequence", -1),
        "timestamp_seconds": row.get("timestamp_seconds", 0.0),
        "visible_person_base_id": row.get("visible_person_base_id", ""),
        "detection_id": row.get("detection_id", ""),
        "source_detection_id": row.get("source_detection_id", ""),
        "bbox": row.get("bbox", {}),
        "footpoint": row.get("footpoint", {}),
        "state": row.get("state", ""),
        "candidate_type": row.get("candidate_type", ""),
        "original_role_source": row.get("original_role_source", ""),
        "source_role_labels": row.get("source_role_labels", []),
        "source_candidate_types": row.get("source_candidate_types", []),
        "source_model_stages": row.get("source_model_stages", []),
        "roi_status": row.get("roi_status", ""),
        "bbox_quality_score": row.get("bbox_quality_score", 0.0),
        "qa_warnings": row.get("qa_warnings", []),
        "review_required": bool(row.get("review_required")),
        "source_disagreement_review_required": bool(row.get("source_disagreement_review_required")),
        "colour_feature_id": feature.get("colour_feature_id", ""),
        "torso_crop_bbox": feature.get("torso_crop_bbox"),
        "crop_quality": feature.get("crop_quality", ""),
        "crop_quality_reason": feature.get("crop_quality_reason", ""),
        "colour_cluster_candidate": cluster,
        "colour_cluster_confidence": confidence,
        "mapped_team_colour_candidate": "unknown_mapping",
        "mapped_team_colour_confidence": 0.0,
        "team_colour_belief": belief,
        "team_colour_belief_confidence": confidence,
        "team_colour_belief_state": state,
        "team_colour_belief_reason": reason,
        "ambiguous_colour_reason": ambiguous_reason,
        "eligible_for_step1d_official_context_candidate": True,
        "eligible_for_step1e_goalkeeper_candidate": True,
        "eligible_for_identity_tracking": False,
        "eligible_for_player_slot_assignment": False,
        "eligible_for_metric_use": False,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def build_team_colour_belief_payloads(
    base_payload: dict[str, Any],
    feature_payload: dict[str, Any],
    prototypes_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prototypes_payload = prototypes_payload or build_colour_prototypes(feature_payload)
    features_by_id = {str(row.get("visible_person_base_id", "")): row for row in feature_payload.get("rows", [])}
    belief_rows = [
        belief_for_row(row, features_by_id.get(str(row.get("visible_person_base_id", "")), {}), prototypes_payload)
        for row in base_payload.get("rows", [])
    ]
    unknown_rows = [
        row
        for row in belief_rows
        if row.get("team_colour_belief") in {"unknown_ambiguous_colour", "crop_unusable"}
        or row.get("team_colour_belief_state") in {"ambiguous_visual_colour", "crop_unusable", "review_required"}
    ]
    belief_counts = Counter(str(row.get("team_colour_belief", "")) for row in belief_rows)
    state_counts = Counter(str(row.get("team_colour_belief_state", "")) for row in belief_rows)
    mapped_counts = Counter(str(row.get("mapped_team_colour_candidate", "")) for row in belief_rows)
    belief_payload = {
        "artifact": "step1c1_team_colour_belief_rows",
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
        "rows": belief_rows,
        "summary": {
            "b4_visible_person_base_rows": len(base_payload.get("rows", [])),
            "step1c1_team_colour_belief_rows": len(belief_rows),
            "unknown_ambiguous_colour_rows": len(unknown_rows),
            "crop_unusable_rows": state_counts.get("crop_unusable", 0),
            "high_confidence_visual_colour_rows": state_counts.get("high_confidence_visual_colour", 0),
            "medium_confidence_visual_colour_rows": state_counts.get("medium_confidence_visual_colour", 0),
            "low_confidence_visual_colour_rows": state_counts.get("low_confidence_visual_colour", 0),
            "review_required_rows": state_counts.get("review_required", 0),
            "source_disagreement_review_required_rows": sum(1 for row in belief_rows if row.get("source_disagreement_review_required") is True),
            "cluster_counts": dict(sorted(belief_counts.items())),
            "mapped_team_colour_counts": dict(sorted(mapped_counts.items())),
            "belief_state_counts": dict(sorted(state_counts.items())),
        },
    }
    unknown_payload = {
        "artifact": "step1c1_unknown_ambiguous_colour_rows",
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
        "rows": unknown_rows,
        "summary": {"unknown_ambiguous_colour_rows": len(unknown_rows)},
    }
    return prototypes_payload, belief_payload, unknown_payload


def colour_belief_report_markdown(belief_payload: dict[str, Any], prototypes_payload: dict[str, Any]) -> str:
    summary = belief_payload.get("summary", {})
    lines = [
        "# Step1.C1 Colour Belief Report",
        "",
        f"- Warning: `{VISUAL_ONLY_WARNING}`.",
        "- Scope: visual-only kit/torso colour evidence and belief candidates from Step1.B4 visible-person base rows.",
        "- No identity tracking, player slots, expected 22 roles, goalkeeper classification, official specialist classification, metric use, or football/tactical analysis was performed.",
        "- B4 provenance was preserved on every belief row.",
        "- Team mapping was not auto-promoted; configured team mapping remains unknown unless a safe reviewed config is supplied later.",
        "",
        "## Input",
        "",
        f"- B4 visible-person base rows: {summary.get('b4_visible_person_base_rows', 0)}",
        f"- Step1.C1 belief rows: {summary.get('step1c1_team_colour_belief_rows', 0)}",
        "",
        "## Cluster Counts",
        "",
        "| belief | rows |",
        "|---|---:|",
    ]
    for key, value in summary.get("cluster_counts", {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Quality And Review",
            "",
            f"- unknown/ambiguous rows: {summary.get('unknown_ambiguous_colour_rows', 0)}",
            f"- crop unusable rows: {summary.get('crop_unusable_rows', 0)}",
            f"- high confidence rows: {summary.get('high_confidence_visual_colour_rows', 0)}",
            f"- medium confidence rows: {summary.get('medium_confidence_visual_colour_rows', 0)}",
            f"- low confidence rows: {summary.get('low_confidence_visual_colour_rows', 0)}",
            f"- review-required rows: {summary.get('review_required_rows', 0)}",
            f"- source-disagreement review rows: {summary.get('source_disagreement_review_required_rows', 0)}",
            "",
            "## Prototype Policy",
            "",
            f"- sandbox prototypes: {prototypes_payload.get('prototype_sandbox_only', False)}",
            f"- safe team mapping found: {prototypes_payload.get('safe_team_mapping_found', False)}",
            f"- mapping reason: {prototypes_payload.get('mapping_reason', '')}",
            "",
            "## Known Risks",
            "",
            "- Small far-side crops can be dominated by pitch, compression, or shadow.",
            "- Dark colour beliefs are colour evidence only and are not official/referee specialist classification.",
            "- Other distinct colour beliefs are colour evidence only and are not goalkeeper classification.",
            "- Unknown/context/off-ROI rows are retained and not forced into team labels.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_and_write_team_colour_beliefs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base_payload = read_json(STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH)
    feature_payload = read_json(STEP1C1_COLOUR_FEATURE_ROWS_PATH)
    prototypes_payload, belief_payload, unknown_payload = build_team_colour_belief_payloads(base_payload, feature_payload)
    write_json(STEP1C1_COLOUR_PROTOTYPES_PATH, prototypes_payload)
    write_json(STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH, belief_payload)
    write_json(STEP1C1_UNKNOWN_AMBIGUOUS_COLOUR_ROWS_PATH, unknown_payload)
    write_text(STEP1C1_COLOUR_BELIEF_REPORT_PATH, colour_belief_report_markdown(belief_payload, prototypes_payload))
    return prototypes_payload, belief_payload, unknown_payload
