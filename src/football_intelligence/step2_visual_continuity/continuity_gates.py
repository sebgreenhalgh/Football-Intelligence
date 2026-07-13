from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from football_intelligence.step2_visual_continuity.continuity_features import build_continuity_feature_snapshot


@dataclass(frozen=True)
class ContinuityGateConfig:
    max_frame_gap: int = 3
    max_candidate_degree: int = 3
    max_center_delta_per_gap_px: float = 150.0
    max_area_ratio: float = 3.2
    max_aspect_change: float = 0.55


def gate_continuity_pair(
    source: dict[str, Any],
    target: dict[str, Any],
    *,
    config: ContinuityGateConfig = ContinuityGateConfig(),
) -> dict[str, Any]:
    features = build_continuity_feature_snapshot(source, target)
    reasons: list[str] = []
    if source.get("entity_validity_state") == "probable_non_person_false_positive":
        reasons.append("source_endpoint_probable_non_person")
    if target.get("entity_validity_state") == "probable_non_person_false_positive":
        reasons.append("target_endpoint_probable_non_person")
    if not source.get("continuity_eligible", False) or not target.get("continuity_eligible", False):
        reasons.append("endpoint_not_continuity_eligible")
    frame_gap = int(features["frame_gap"])
    if frame_gap <= 0 or frame_gap > config.max_frame_gap:
        reasons.append("frame_gap_outside_short_window")
    if float(features["center_delta_px"]) > config.max_center_delta_per_gap_px * max(1, frame_gap):
        reasons.append("location_incompatible_image_space")
    if float(features["bbox_area_ratio"]) > config.max_area_ratio:
        reasons.append("bbox_scale_incompatible")
    if float(features["aspect_ratio_change"]) > config.max_aspect_change:
        reasons.append("aspect_ratio_incompatible")
    source_context = str(source.get("visual_role_context_state", ""))
    target_context = str(target.get("visual_role_context_state", ""))
    team_conflict = {
        source_context,
        target_context,
    } == {"team_1_outfield_visual_context", "team_2_outfield_visual_context"}
    if team_conflict:
        reasons.append("high_confidence_team_context_conflict")
    if "official" in source_context and "outfield" in target_context:
        reasons.append("official_player_context_conflict")
    if "official" in target_context and "outfield" in source_context:
        reasons.append("official_player_context_conflict")
    return {
        "passed": not reasons,
        "rejection_reasons": reasons,
        "features": features,
        "rule_version": "m5.4d.short_window_continuity_gates.v1",
    }
