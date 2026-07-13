from __future__ import annotations

from typing import Any

from football_intelligence.review.schemas import safety_payload


def select_continuity_review_candidates(
    candidate_rows: list[dict[str, Any]],
    *,
    positive_limit: int = 10,
    negative_limit: int = 10,
) -> dict[str, Any]:
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for row in sorted(candidate_rows, key=lambda item: float(item.get("continuity_score", 0.0)), reverse=True):
        features = row.get("gate_features") if isinstance(row.get("gate_features"), dict) else {}
        same_partition = row.get("source_visual_role_context") == row.get("target_visual_role_context")
        short_gap = int(row.get("frame_gap", 999)) <= 2
        low_displacement = float(features.get("center_delta_px", 999.0)) < 45.0
        scale_ok = float(features.get("bbox_area_ratio", 999.0)) < 1.6
        intermediate_support = bool(row.get("intermediate_frame_support")) or int(row.get("frame_gap", 1)) > 1
        intermediate_support = intermediate_support or float(features.get("bbox_iou", 0.0)) > 0.2
        no_competition = float(row.get("continuity_score", 0.0)) >= 0.55
        candidate = {
            **row,
            "requires_intermediate_support": True,
            "has_intermediate_support": intermediate_support,
            "positive_selector_rule_version": "m5.4e.positive_selector.v1",
        }
        if same_partition and short_gap and low_displacement and scale_ok and no_competition and intermediate_support:
            if len(positives) < positive_limit:
                positives.append({**candidate, "continuity_review_bucket": "likely_positive_continuity"})
        elif len(negatives) < negative_limit:
            negatives.append({**candidate, "continuity_review_bucket": "difficult_or_likely_negative_continuity"})
        if len(positives) >= positive_limit and len(negatives) >= negative_limit:
            break
    return {
        "artifact": "m5_4e_positive_continuity_review_selection",
        "likely_positive": positives,
        "likely_negative": negatives,
        "likely_positive_count": len(positives),
        "likely_negative_count": len(negatives),
        **safety_payload(),
    }
