# ruff: noqa: E501

from __future__ import annotations

import math
from collections import Counter
from datetime import UTC, datetime
from statistics import median
from typing import Any

from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH,
    STEP1C1_COLOUR_FEATURE_ROWS_PATH,
    STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH,
    STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH,
    STEP1C1C_SEEDED_COLOUR_BELIEF_ROWS_SANDBOX_PATH,
    STEP1C1C_SEEDED_COLOUR_PROTOTYPES_SANDBOX_PATH,
    read_json,
    write_json,
)
from football_intelligence.step1_visual_reconstruction.manual_colour_seed_schema import (
    build_and_write_seed_validation_summary,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    safe_float,
)
from football_intelligence.step1_visual_reconstruction.team_colour_beliefs import UNKNOWN_CONTEXT_TYPES


PROTOTYPE_LABELS = {
    "team_1_outfield_colour_seed",
    "team_2_outfield_colour_seed",
    "ambiguous_outfield_colour",
    "non_outfield_context_colour",
    "dark_context_colour",
    "other_distinct_colour",
}

MANUAL_LABEL_TO_BELIEF = {
    "team_1_outfield_colour_seed": "team_1_outfield_colour_like",
    "team_2_outfield_colour_seed": "team_2_outfield_colour_like",
    "ambiguous_outfield_colour": "ambiguous_outfield_colour",
    "non_outfield_context_colour": "non_outfield_context_colour",
    "dark_context_colour": "dark_context_colour_like",
    "other_distinct_colour": "other_distinct_colour_like",
    "crop_unusable": "crop_unusable",
    "not_a_person_or_bad_detection": "unknown_ambiguous_colour",
    "unsure": "unknown_ambiguous_colour",
}

TEAM_SEED_LABELS = {"team_1_outfield_colour_seed", "team_2_outfield_colour_seed"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def index_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("visible_person_base_id", "")): row for row in payload.get("rows", [])}


def median_triplet(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [row.get(key, []) for row in rows if isinstance(row.get(key), list) and len(row.get(key)) == 3]
    if not values:
        return []
    return [round(float(median([safe_float(value[index]) for value in values])), 3) for index in range(3)]


def empty_prototypes_payload(reason: str, validation_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    validation_summary = validation_summary or {}
    return {
        "artifact": "step1c1c_seeded_colour_prototypes_sandbox",
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
        "reviewed_seed_labels_loaded": validation_summary.get("reviewed_seed_labels_loaded", False),
        "reviewed_seed_labels_valid": validation_summary.get("reviewed_seed_labels_valid", False),
        "human_seed_set_id": validation_summary.get("human_seed_set_id", ""),
        "prototypes": [],
        "summary": {"prototype_count": 0, "skipped_reason": reason},
    }


def empty_belief_payload(reason: str, validation_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    validation_summary = validation_summary or {}
    return {
        "artifact": "step1c1c_seeded_colour_belief_rows_sandbox",
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
        "seed_sandbox_only": True,
        "sandbox_only": True,
        "reviewed_seed_labels_loaded": validation_summary.get("reviewed_seed_labels_loaded", False),
        "reviewed_seed_labels_valid": validation_summary.get("reviewed_seed_labels_valid", False),
        "human_seed_set_id": validation_summary.get("human_seed_set_id", ""),
        "rows": [],
        "summary": {"seeded_colour_belief_rows": 0, "skipped_reason": reason},
    }


def build_seed_prototypes(
    candidate_payload: dict[str, Any],
    feature_payload: dict[str, Any],
    usable_seed_rows: list[dict[str, Any]],
    validation_summary: dict[str, Any],
) -> dict[str, Any]:
    candidates_by_seed_id = {str(row.get("seed_candidate_id", "")): row for row in candidate_payload.get("rows", [])}
    features_by_base_id = index_rows(feature_payload)
    grouped_features: dict[str, list[dict[str, Any]]] = {label: [] for label in PROTOTYPE_LABELS}
    seed_rows_used = []
    for seed in usable_seed_rows:
        label = str(seed.get("manual_colour_label", ""))
        if label not in PROTOTYPE_LABELS:
            continue
        candidate = candidates_by_seed_id.get(str(seed.get("seed_candidate_id", "")), {})
        feature = features_by_base_id.get(str(candidate.get("visible_person_base_id", "")), {})
        if not feature.get("median_hsv"):
            continue
        grouped_features[label].append(feature)
        seed_rows_used.append(
            {
                "seed_candidate_id": seed.get("seed_candidate_id", ""),
                "visible_person_base_id": candidate.get("visible_person_base_id", ""),
                "manual_colour_label": label,
                "human_confirmed": True,
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
            }
        )
    prototypes = []
    for label in sorted(PROTOTYPE_LABELS):
        rows = grouped_features[label]
        prototypes.append(
            {
                "manual_colour_label": label,
                "seed_team_colour_belief": MANUAL_LABEL_TO_BELIEF[label],
                "human_confirmed_seed_rows": len(rows),
                "median_hsv": median_triplet(rows, "median_hsv"),
                "median_lab": median_triplet(rows, "median_lab"),
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
            }
        )
    return {
        "artifact": "step1c1c_seeded_colour_prototypes_sandbox",
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
        "reviewed_seed_labels_loaded": validation_summary.get("reviewed_seed_labels_loaded", False),
        "reviewed_seed_labels_valid": validation_summary.get("reviewed_seed_labels_valid", False),
        "human_seed_set_id": validation_summary.get("human_seed_set_id", ""),
        "prototypes": prototypes,
        "seed_rows_used": seed_rows_used,
        "summary": {
            "prototype_count": len(prototypes),
            "prototype_seed_counts": {label: len(rows) for label, rows in sorted(grouped_features.items())},
            "human_confirmed_seed_rows_used": len(seed_rows_used),
        },
    }


def hsv_distance(left: list[Any], right: list[Any]) -> float:
    if len(left) != 3 or len(right) != 3:
        return 9999.0
    hue_delta = abs(safe_float(left[0]) - safe_float(right[0]))
    hue_delta = min(hue_delta, 180.0 - hue_delta) * 1.4
    sat_delta = (safe_float(left[1]) - safe_float(right[1])) * 0.45
    val_delta = (safe_float(left[2]) - safe_float(right[2])) * 0.35
    return math.sqrt(hue_delta * hue_delta + sat_delta * sat_delta + val_delta * val_delta)


def nearest_seed(feature: dict[str, Any], prototypes_payload: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
    feature_hsv = feature.get("median_hsv", [])
    distances: dict[str, float] = {}
    for proto in prototypes_payload.get("prototypes", []):
        label = str(proto.get("manual_colour_label", ""))
        if not proto.get("median_hsv"):
            continue
        distances[label] = round(hsv_distance(feature_hsv, proto.get("median_hsv", [])), 4)
    if not distances:
        return "", 0.0, {}
    label, distance = sorted(distances.items(), key=lambda item: (item[1], item[0]))[0]
    confidence = round(max(0.05, min(0.96, 1.0 - (distance / 140.0))), 4)
    return label, confidence, distances


def context_like(row: dict[str, Any]) -> bool:
    return str(row.get("candidate_type", "")) in UNKNOWN_CONTEXT_TYPES or str(row.get("roi_status", "")) == "outside_playing_roi"


def seeded_belief_for_row(row: dict[str, Any], feature: dict[str, Any], prototypes_payload: dict[str, Any], human_seed_set_id: str) -> dict[str, Any]:
    nearest_label, confidence, distances = nearest_seed(feature, prototypes_payload)
    crop_quality = str(feature.get("crop_quality", row.get("crop_quality", "")))
    green = safe_float(feature.get("green_background_fraction"))
    team_1_dist = distances.get("team_1_outfield_colour_seed", 9999.0)
    team_2_dist = distances.get("team_2_outfield_colour_seed", 9999.0)
    close_team_distance = abs(team_1_dist - team_2_dist) <= 12.0 or (min(team_1_dist, team_2_dist) / max(1.0, max(team_1_dist, team_2_dist)) >= 0.82)
    belief = MANUAL_LABEL_TO_BELIEF.get(nearest_label, "unknown_ambiguous_colour")
    reason = "nearest_human_reviewed_colour_seed"
    review_required = False
    if crop_quality == "unusable":
        belief = "crop_unusable"
        confidence = min(confidence, 0.15)
        reason = "crop_unusable_for_seeded_colour"
        review_required = True
    elif crop_quality == "low" or green > 0.62:
        if confidence < 0.90:
            belief = "unknown_ambiguous_colour"
            confidence = min(confidence, 0.35)
            reason = "low_or_contaminated_crop_not_forced_to_seeded_colour"
            review_required = True
    if nearest_label in TEAM_SEED_LABELS and close_team_distance:
        belief = "ambiguous_outfield_colour"
        confidence = min(confidence, 0.52)
        reason = "team_seed_distances_too_close"
        review_required = True
    if context_like(row) and belief in {"team_1_outfield_colour_like", "team_2_outfield_colour_like"}:
        belief = "unknown_ambiguous_colour"
        confidence = min(confidence, 0.35)
        reason = "context_or_off_roi_not_forced_to_team_seed_colour"
        review_required = True
    if not nearest_label:
        belief = "unknown_ambiguous_colour"
        reason = "no_usable_human_seed_prototype"
        review_required = True
    if belief in {"team_1_outfield_colour_like", "team_2_outfield_colour_like"} and confidence >= 0.78:
        state = "high_confidence_seeded_colour"
    elif belief in {"team_1_outfield_colour_like", "team_2_outfield_colour_like"} and confidence >= 0.60:
        state = "medium_confidence_seeded_colour"
    elif belief == "crop_unusable":
        state = "crop_unusable"
    else:
        state = "seed_review_required" if review_required else "ambiguous_seeded_colour"
    out = dict(row)
    out.update(
        {
            "seed_colour_feature_profile": "c1_current_colour_features",
            "nearest_seed_label_candidate": nearest_label,
            "nearest_seed_label_confidence": confidence,
            "nearest_seed_label_distances": distances,
            "seed_team_colour_belief": belief,
            "seed_team_colour_belief_confidence": confidence,
            "seed_team_colour_belief_state": state,
            "seed_team_colour_belief_reason": reason,
            "seed_review_required": review_required,
            "seed_sandbox_only": True,
            "human_seed_set_id": human_seed_set_id,
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
    )
    return out


def build_seeded_belief_payload(
    base_payload: dict[str, Any],
    c1_belief_payload: dict[str, Any],
    feature_payload: dict[str, Any],
    prototypes_payload: dict[str, Any],
    validation_summary: dict[str, Any],
) -> dict[str, Any]:
    features_by_id = index_rows(feature_payload)
    c1_by_id = index_rows(c1_belief_payload)
    human_seed_set_id = str(validation_summary.get("human_seed_set_id", ""))
    rows = []
    for base_row in base_payload.get("rows", []):
        base_id = str(base_row.get("visible_person_base_id", ""))
        source_row = c1_by_id.get(base_id, base_row)
        rows.append(seeded_belief_for_row(source_row, features_by_id.get(base_id, {}), prototypes_payload, human_seed_set_id))
    belief_counts = Counter(row.get("seed_team_colour_belief", "") for row in rows)
    state_counts = Counter(row.get("seed_team_colour_belief_state", "") for row in rows)
    context_forced = sum(
        1
        for row in rows
        if context_like(row) and row.get("seed_team_colour_belief") in {"team_1_outfield_colour_like", "team_2_outfield_colour_like"}
    )
    return {
        "artifact": "step1c1c_seeded_colour_belief_rows_sandbox",
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
        "seed_sandbox_only": True,
        "sandbox_only": True,
        "reviewed_seed_labels_loaded": validation_summary.get("reviewed_seed_labels_loaded", False),
        "reviewed_seed_labels_valid": validation_summary.get("reviewed_seed_labels_valid", False),
        "human_seed_set_id": human_seed_set_id,
        "rows": rows,
        "summary": {
            "b4_visible_person_base_rows": len(base_payload.get("rows", [])),
            "seeded_colour_belief_rows": len(rows),
            "seeded_unknown_ambiguous_rows": sum(1 for row in rows if row.get("seed_team_colour_belief") in {"unknown_ambiguous_colour", "ambiguous_outfield_colour", "crop_unusable"}),
            "seeded_belief_counts": dict(sorted(belief_counts.items())),
            "seeded_belief_state_counts": dict(sorted(state_counts.items())),
            "context_offroi_forced_to_team_count": context_forced,
        },
    }


def build_seeded_colour_sandbox_payloads() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validation_summary, usable_rows = build_and_write_seed_validation_summary()
    if not validation_summary.get("reviewed_seed_labels_loaded"):
        prototypes = empty_prototypes_payload("reviewed_seed_labels_absent", validation_summary)
        beliefs = empty_belief_payload("reviewed_seed_labels_absent", validation_summary)
        write_json(STEP1C1C_SEEDED_COLOUR_PROTOTYPES_SANDBOX_PATH, prototypes)
        write_json(STEP1C1C_SEEDED_COLOUR_BELIEF_ROWS_SANDBOX_PATH, beliefs)
        return validation_summary, prototypes, beliefs
    if not validation_summary.get("reviewed_seed_labels_valid"):
        prototypes = empty_prototypes_payload("reviewed_seed_labels_invalid", validation_summary)
        beliefs = empty_belief_payload("reviewed_seed_labels_invalid", validation_summary)
        write_json(STEP1C1C_SEEDED_COLOUR_PROTOTYPES_SANDBOX_PATH, prototypes)
        write_json(STEP1C1C_SEEDED_COLOUR_BELIEF_ROWS_SANDBOX_PATH, beliefs)
        return validation_summary, prototypes, beliefs
    candidate_payload = read_json(STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH)
    feature_payload = read_json(STEP1C1_COLOUR_FEATURE_ROWS_PATH)
    prototypes = build_seed_prototypes(candidate_payload, feature_payload, usable_rows, validation_summary)
    beliefs = build_seeded_belief_payload(
        read_json(STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH),
        read_json(STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH),
        feature_payload,
        prototypes,
        validation_summary,
    )
    write_json(STEP1C1C_SEEDED_COLOUR_PROTOTYPES_SANDBOX_PATH, prototypes)
    write_json(STEP1C1C_SEEDED_COLOUR_BELIEF_ROWS_SANDBOX_PATH, beliefs)
    return validation_summary, prototypes, beliefs
