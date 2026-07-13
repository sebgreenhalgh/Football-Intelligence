from __future__ import annotations

from collections import Counter
from typing import Any

from football_intelligence.review.schemas import safety_payload

VALID_ON_PITCH_PERSON = "valid_on_pitch_person"
VALID_OFFICIAL = "valid_official"
VALID_OFF_PITCH_PERSON = "valid_off_pitch_person"
PROBABLE_NON_PERSON = "probable_non_person_false_positive"
AMBIGUOUS_ENTITY = "ambiguous_entity_requires_review"

TEAM_1_OUTFIELD = "team_1_outfield_visual_context"
TEAM_2_OUTFIELD = "team_2_outfield_visual_context"
TEAM_UNKNOWN_OUTFIELD = "team_unknown_outfield_visual_context"
OFFICIAL_CONTEXT = "official_referee_visual_context"
OFF_PITCH_CONTEXT = "off_pitch_context_person_visual_context"
BAD_DETECTION = "bad_detection_or_not_person"
UNKNOWN_CONTEXT = "unknown_visible_person_visual_context"


def classify_entity_validity(feature: dict[str, Any]) -> tuple[str, float, list[str]]:
    reasons: list[str] = []
    confidence = float(feature.get("detector_confidence", 0.0))
    static_count = int(feature.get("static_persistence_count", 0))
    spatial = str(feature.get("spatial_context", "unknown_spatial_context"))
    structure_like = bool(feature.get("structure_like_shape"))
    tiny = bool(feature.get("tiny_or_distant"))
    static_like = static_count >= 4 and structure_like

    if static_like and spatial in {"off_pitch_context_region", "playing_area_roi_candidate"}:
        reasons.extend(["static_structure_like_crop", "static_persistence_with_structure_like_evidence"])
        return PROBABLE_NON_PERSON, 0.86, reasons
    if confidence < 0.20 and tiny:
        reasons.extend(["very_low_detector_confidence", "tiny_or_distant_bbox"])
        return PROBABLE_NON_PERSON, 0.68, reasons
    if spatial == "off_pitch_context_region" and confidence >= 0.42 and not structure_like:
        reasons.extend(["off_pitch_context_region", "person_like_detector_confidence"])
        return VALID_OFF_PITCH_PERSON, 0.64, reasons
    if spatial in {"playing_area_roi_candidate", "near_side_recovery_zone"} and confidence >= 0.58 and not tiny:
        reasons.extend(["on_pitch_image_space_context", "high_confidence_human_sized_image_space_box"])
        return VALID_ON_PITCH_PERSON, 0.66, reasons
    if static_count >= 4:
        reasons.append("static_persistence_requires_review_not_auto_invalid")
    if tiny:
        reasons.append("tiny_or_distant_bbox")
    if confidence <= 0.32:
        reasons.append("low_detector_confidence")
    if not reasons:
        reasons.append("insufficient_match_local_entity_evidence")
    return AMBIGUOUS_ENTITY, 0.5, reasons


def infer_visual_context(feature: dict[str, Any], entity_state: str) -> tuple[str, float, list[str]]:
    if entity_state == PROBABLE_NON_PERSON:
        return BAD_DETECTION, 0.86, ["entity_validity_probable_non_person"]
    if entity_state == VALID_OFF_PITCH_PERSON:
        return OFF_PITCH_CONTEXT, 0.62, ["entity_validity_off_pitch_person"]
    if entity_state == VALID_OFFICIAL:
        return OFFICIAL_CONTEXT, 0.62, ["entity_validity_official"]
    if entity_state == VALID_ON_PITCH_PERSON:
        return TEAM_UNKNOWN_OUTFIELD, 0.55, ["team_colour_not_forced"]
    return UNKNOWN_CONTEXT, 0.5, ["unknowns_are_not_forced"]


def build_entity_validity_rows(feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fused_rows: list[dict[str, Any]] = []
    for feature in feature_rows:
        state, confidence, reasons = classify_entity_validity(feature)
        context_state, context_confidence, context_reasons = infer_visual_context(feature, state)
        rows.append(
            {
                "entity_validity_row_id": f"m5_4d_entity_validity_{len(rows):06d}",
                "candidate_id": feature["candidate_id"],
                "raw_detector_row_id": feature["raw_detector_row_id"],
                "source_detection_id": feature.get("source_detection_id"),
                "frame_sequence": feature["frame_sequence"],
                "bbox": feature["bbox"],
                "candidate_type": "person_candidate",
                "entity_validity_state": state,
                "entity_validity_confidence": confidence,
                "entity_validity_reasons": reasons,
                "human_reviewed": False,
                "review_required": state == AMBIGUOUS_ENTITY,
                "raw_row_preserved": True,
                "detector_row_deleted": False,
                **safety_payload(),
            }
        )
        fused_rows.append(
            {
                "fused_visual_context_row_id": f"m5_4d_fused_context_{len(fused_rows):06d}",
                "candidate_id": feature["candidate_id"],
                "raw_detector_row_id": feature["raw_detector_row_id"],
                "frame_sequence": feature["frame_sequence"],
                "entity_validity_state": state,
                "visual_role_context_state": context_state,
                "visual_role_context_confidence": context_confidence,
                "visual_role_context_reasons": context_reasons,
                "team_colour_evidence_state": "team_colour_not_forced",
                "official_context_evidence_state": "not_forced",
                "goalkeeper_context_evidence_state": "not_forced",
                "uncertainty": round(1.0 - min(confidence, context_confidence), 4),
                "review_status": "requires_review"
                if state == AMBIGUOUS_ENTITY
                else "machine_candidate_not_human_approved",
                "source_provenance": {
                    "entity_feature_row_id": feature["entity_feature_row_id"],
                    "static_persistence_signature": feature["static_persistence_signature"],
                    "duplicate_group_id": feature.get("duplicate_group_id"),
                },
                "raw_row_preserved": True,
                "detector_row_deleted": False,
                **safety_payload(),
            }
        )
    return {
        "artifact": "m5_4d_entity_validity_original_rows",
        "rows": rows,
        "fused_visual_context_rows": fused_rows,
        "summary": dict(sorted(Counter(row["entity_validity_state"] for row in rows).items())),
        **safety_payload(),
    }
