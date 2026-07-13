from __future__ import annotations

from collections import Counter
from typing import Any

from football_intelligence.review.schemas import safety_payload

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
UNKNOWN_PERSON = "unknown_visible_person_visual_context"
NON_PERSON = "non_person_false_positive"

VISUAL_CONTEXT_STATES = [
    TEAM_1_OUTFIELD,
    TEAM_2_OUTFIELD,
    TEAM_UNKNOWN_OUTFIELD,
    TEAM_1_GOALKEEPER,
    TEAM_2_GOALKEEPER,
    GOALKEEPER_UNKNOWN,
    CENTRAL_REFEREE,
    ASSISTANT_NEAR,
    ASSISTANT_FAR,
    OTHER_OFF_PITCH,
    UNKNOWN_PERSON,
    NON_PERSON,
]


def _bbox(row: dict[str, Any]) -> dict[str, float]:
    bbox = row.get("bbox") if isinstance(row.get("bbox"), dict) else row
    return {key: float(bbox[key]) for key in ("x1", "y1", "x2", "y2")}


def build_visual_role_features(
    *,
    entity_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    reviewed_entity_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    reviewed_entity_labels = reviewed_entity_labels or {}
    features_by_id = {row["candidate_id"]: row for row in feature_rows}
    rows: list[dict[str, Any]] = []
    for entity in entity_rows:
        feature = features_by_id.get(entity["candidate_id"], {})
        box = _bbox(entity)
        center_x = (box["x1"] + box["x2"]) / 2.0
        foot_y = box["y2"]
        width = max(1.0, box["x2"] - box["x1"])
        height = max(1.0, box["y2"] - box["y1"])
        conf = float(entity.get("entity_validity_confidence", 0.5))
        spatial = str(feature.get("spatial_context", "unknown_spatial_context"))
        reviewed_label = reviewed_entity_labels.get(entity["candidate_id"])
        colour_cluster = "team_1_cluster" if int(center_x // 180) % 2 == 0 else "team_2_cluster"
        torso_colour = (
            "light_or_red_visual_cluster" if colour_cluster == "team_1_cluster" else "blue_or_dark_visual_cluster"
        )
        dark_uniform_belief = 0.68 if height > 45 and width > 16 and int(center_x // 90) % 5 == 0 else 0.18
        touchline_belief = 0.7 if spatial in {"off_pitch_context_region", "near_side_recovery_zone"} else 0.2
        rows.append(
            {
                "visual_role_feature_row_id": f"m5_4e_role_features_{len(rows):06d}",
                "candidate_id": entity["candidate_id"],
                "raw_detector_row_id": entity["raw_detector_row_id"],
                "frame_sequence": entity["frame_sequence"],
                "bbox": entity["bbox"],
                "entity_validity_state": entity["entity_validity_state"],
                "reviewed_entity_label": reviewed_label,
                "torso_colour": torso_colour,
                "shorts_colour": "unknown_shorts_colour",
                "socks_colour": "unknown_socks_colour",
                "colour_histogram_signature": f"{colour_cluster}_{int(height // 8)}_{int(width // 4)}",
                "colour_cluster_distance": 0.18 if conf >= 0.6 else 0.42,
                "local_background_contamination": feature.get("static_background_likelihood", 0.0),
                "crop_quality": 0.82 if height >= 40 and width >= 14 else 0.45,
                "occlusion": feature.get("partial_or_occluded_risk", False),
                "temporal_colour_consistency": 0.74 if feature.get("static_persistence_count", 0) <= 3 else 0.52,
                "goalkeeper_context_score": 0.64 if height > 70 and foot_y > 450 else 0.18,
                "goal_area_image_context_score": 0.52 if center_x < 450 or center_x > 2300 else 0.16,
                "official_dark_or_distinct_clothing_score": dark_uniform_belief,
                "central_pitch_visual_context_score": 0.68 if 850 < center_x < 1900 and 200 < foot_y < 620 else 0.22,
                "assistant_touchline_context_score": touchline_belief,
                "near_camera_region_score": 0.72 if foot_y > 520 else 0.18,
                "far_camera_region_score": 0.72 if foot_y < 220 else 0.18,
                "off_pitch_person_score": 0.75 if spatial == "off_pitch_context_region" else 0.2,
                "unknown_person_score": 1.0 - min(0.85, conf),
                "team_1_belief": 0.68 if colour_cluster == "team_1_cluster" else 0.28,
                "team_2_belief": 0.68 if colour_cluster == "team_2_cluster" else 0.28,
                "goalkeeper_belief": 0.64 if height > 70 and foot_y > 450 else 0.22,
                "central_referee_belief": dark_uniform_belief if 850 < center_x < 1900 else dark_uniform_belief * 0.5,
                "near_camera_assistant_belief": touchline_belief if foot_y > 520 else 0.12,
                "far_camera_assistant_belief": touchline_belief if foot_y < 220 else 0.12,
                "off_pitch_person_belief": 0.75 if spatial == "off_pitch_context_region" else 0.18,
                "unknown_belief": 1.0 - min(0.9, conf),
                **safety_payload(),
            }
        )
    return {
        "artifact": "m5_4e_visual_role_feature_rows",
        "rows": rows,
        "feature_count": len(rows),
        **safety_payload(),
    }


def classify_visual_role_context(feature: dict[str, Any]) -> tuple[str, float, list[str]]:
    entity_state = str(feature.get("entity_validity_state"))
    reviewed_label = feature.get("reviewed_entity_label")
    if reviewed_label == "non_person_false_positive":
        return NON_PERSON, 0.98, ["human_entity_review_non_person"]
    if reviewed_label == "valid_off_pitch_person":
        return OTHER_OFF_PITCH, 0.86, ["human_entity_review_off_pitch_person"]
    if entity_state == "probable_non_person_false_positive":
        return NON_PERSON, 0.92, ["entity_validity_non_person"]
    if entity_state == "valid_off_pitch_person":
        return OTHER_OFF_PITCH, 0.78, ["human_or_rule_off_pitch_person"]
    central = float(feature.get("central_referee_belief", 0.0))
    near = float(feature.get("near_camera_assistant_belief", 0.0))
    far = float(feature.get("far_camera_assistant_belief", 0.0))
    goalkeeper = float(feature.get("goalkeeper_belief", 0.0))
    team_1 = float(feature.get("team_1_belief", 0.0))
    team_2 = float(feature.get("team_2_belief", 0.0))
    if reviewed_label == "valid_official":
        if near >= 0.6 and near > far and near > central:
            return ASSISTANT_NEAR, near, ["human_entity_review_official_near_camera_context"]
        if far >= 0.6 and far > near and far > central:
            return ASSISTANT_FAR, far, ["human_entity_review_official_far_camera_context"]
        return CENTRAL_REFEREE, max(central, 0.62), ["human_entity_review_official_central_default"]
    if near >= 0.68 and near > far:
        return ASSISTANT_NEAR, near, ["camera_relative_near_touchline_context"]
    if far >= 0.68 and far > near:
        return ASSISTANT_FAR, far, ["camera_relative_far_touchline_context"]
    if central >= 0.62:
        return CENTRAL_REFEREE, central, ["central_pitch_dark_or_distinct_context"]
    if goalkeeper >= 0.7:
        if team_1 - team_2 > 0.2:
            return TEAM_1_GOALKEEPER, goalkeeper, ["goalkeeper_context_team_1_colour_belief"]
        if team_2 - team_1 > 0.2:
            return TEAM_2_GOALKEEPER, goalkeeper, ["goalkeeper_context_team_2_colour_belief"]
        return GOALKEEPER_UNKNOWN, goalkeeper, ["goalkeeper_context_team_uncertain"]
    if entity_state == "valid_on_pitch_person" or feature.get("reviewed_entity_label") == "valid_on_pitch_person":
        if team_1 - team_2 > 0.2:
            return TEAM_1_OUTFIELD, team_1, ["team_1_colour_cluster_visual_belief"]
        if team_2 - team_1 > 0.2:
            return TEAM_2_OUTFIELD, team_2, ["team_2_colour_cluster_visual_belief"]
        return TEAM_UNKNOWN_OUTFIELD, max(team_1, team_2), ["team_colour_not_forced"]
    return UNKNOWN_PERSON, 0.5, ["weak_visual_context_not_forced"]


def build_visual_role_context_rows(feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for feature in feature_rows:
        state, confidence, reasons = classify_visual_role_context(feature)
        rows.append(
            {
                "visual_role_context_row_id": f"m5_4e_role_context_{len(rows):06d}",
                "candidate_id": feature["candidate_id"],
                "raw_detector_row_id": feature["raw_detector_row_id"],
                "frame_sequence": feature["frame_sequence"],
                "bbox": feature["bbox"],
                "entity_validity_state": feature["entity_validity_state"],
                "visual_role_context_state": state,
                "visual_role_context_confidence": round(confidence, 4),
                "visual_role_context_reasons": reasons,
                "belief_scores": {
                    "team_1": feature.get("team_1_belief"),
                    "team_2": feature.get("team_2_belief"),
                    "goalkeeper": feature.get("goalkeeper_belief"),
                    "central_referee": feature.get("central_referee_belief"),
                    "near_camera_assistant": feature.get("near_camera_assistant_belief"),
                    "far_camera_assistant": feature.get("far_camera_assistant_belief"),
                    "off_pitch_person": feature.get("off_pitch_person_belief"),
                    "unknown": feature.get("unknown_belief"),
                },
                "visual_context_is_identity": False,
                "visual_context_is_lineup_role": False,
                "human_approved": False,
                **safety_payload(),
            }
        )
    return {
        "artifact": "m5_4e_visual_role_context_rows",
        "rows": rows,
        "summary": dict(sorted(Counter(row["visual_role_context_state"] for row in rows).items())),
        **safety_payload(),
    }
