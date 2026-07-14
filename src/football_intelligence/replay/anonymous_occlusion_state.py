from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from football_intelligence.research_handoff.stage_workspace import safety_payload


class OcclusionState(StrEnum):
    VISIBLE_CONFIRMED = "VISIBLE_CONFIRMED"
    VISIBLE_UNCERTAIN = "VISIBLE_UNCERTAIN"
    APPROACHING_OCCLUSION = "APPROACHING_OCCLUSION"
    PARTIALLY_OCCLUDED = "PARTIALLY_OCCLUDED"
    FULLY_OCCLUDED_PREDICTED = "FULLY_OCCLUDED_PREDICTED"
    MULTI_HYPOTHESIS_REENTRY = "MULTI_HYPOTHESIS_REENTRY"
    REEMERGED_UNCONFIRMED = "REEMERGED_UNCONFIRMED"
    REEMERGED_CONFIRMED = "REEMERGED_CONFIRMED"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    TERMINATED = "TERMINATED"


class ObservationNodeType(StrEnum):
    DETECTION = "DETECTION"
    OCCLUDED_NULL = "OCCLUDED_NULL"
    MERGED_OBSERVATION = "MERGED_OBSERVATION"
    FRAME_EXIT = "FRAME_EXIT"


@dataclass
class MotionState:
    footpoint_x: float
    footpoint_y: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    log_bbox_width: float = 0.0
    log_bbox_height: float = 0.0
    uncertainty: float = 1.0

    def predict(self, frames: int = 1) -> "MotionState":
        frames = max(0, int(frames))
        return MotionState(
            footpoint_x=self.footpoint_x + self.velocity_x * frames,
            footpoint_y=self.footpoint_y + self.velocity_y * frames,
            velocity_x=self.velocity_x,
            velocity_y=self.velocity_y,
            log_bbox_width=self.log_bbox_width,
            log_bbox_height=self.log_bbox_height,
            uncertainty=min(1_000_000.0, self.uncertainty + frames * max(1.0, self.uncertainty * 0.15)),
        )

    def update(
        self,
        observed_footpoint: tuple[float, float],
        bbox_width: float,
        bbox_height: float,
        alpha: float = 0.65,
    ) -> dict[str, Any]:
        predicted = self.predict(1)
        residual_x = observed_footpoint[0] - predicted.footpoint_x
        residual_y = observed_footpoint[1] - predicted.footpoint_y
        previous_x = self.footpoint_x
        previous_y = self.footpoint_y
        self.footpoint_x = predicted.footpoint_x + alpha * residual_x
        self.footpoint_y = predicted.footpoint_y + alpha * residual_y
        self.velocity_x = 0.5 * self.velocity_x + 0.5 * (self.footpoint_x - previous_x)
        self.velocity_y = 0.5 * self.velocity_y + 0.5 * (self.footpoint_y - previous_y)
        self.log_bbox_width = math.log(max(1.0, bbox_width))
        self.log_bbox_height = math.log(max(1.0, bbox_height))
        self.uncertainty = max(1.0, predicted.uncertainty * (1.0 - alpha))
        return {
            "residual": {"x": residual_x, "y": residual_y},
            "innovation_norm": math.hypot(residual_x, residual_y),
            "state_mean": self.to_dict(),
            "numerical_stability": "bounded_positive_uncertainty",
        }

    def to_dict(self) -> dict[str, float]:
        return {
            "footpoint_x": self.footpoint_x,
            "footpoint_y": self.footpoint_y,
            "velocity_x": self.velocity_x,
            "velocity_y": self.velocity_y,
            "log_bbox_width": self.log_bbox_width,
            "log_bbox_height": self.log_bbox_height,
            "uncertainty": self.uncertainty,
        }


def dynamic_ghost_lifetime(
    bbox_height: float,
    uncertainty: float = 1.0,
    *,
    continuing_occluder: bool = False,
    frame_exit: bool = False,
) -> dict[str, Any]:
    if frame_exit:
        return {"max_hidden_frames": 0, "reason": "frame_exit"}
    if bbox_height < 24:
        base = 5
        band = "small_under_24px"
    elif bbox_height <= 50:
        base = 8
        band = "medium_24_to_50px"
    else:
        base = 12
        band = "large_over_50px"
    if uncertainty > bbox_height * 2.0:
        base = max(1, base - 2)
        reason = "uncertainty_growth_reduced_lifetime"
    elif continuing_occluder:
        base += 2
        reason = "continuing_occluder_extension"
    else:
        reason = "scale_band_default"
    return {"max_hidden_frames": base, "height_band": band, "reason": reason}


@dataclass
class AnonymousTracklet:
    anonymous_tracklet_id: str
    window_id: str
    created_frame: int
    motion: MotionState
    current_state: OcclusionState = OcclusionState.VISIBLE_CONFIRMED
    last_observed_frame: int | None = None
    last_confirmed_frame: int | None = None
    hidden_frame_count: int = 0
    dynamic_max_hidden_frames: int = 0
    competing_anonymous_tracklet_ids: list[str] = field(default_factory=list)
    merged_observation_ids: list[str] = field(default_factory=list)
    active_reentry_hypotheses: list[str] = field(default_factory=list)
    review_required: bool = False
    review_reason_codes: list[str] = field(default_factory=list)
    termination_reason: str | None = None

    def transition(self, target: OcclusionState, reason_codes: list[str], threshold_result: str) -> dict[str, Any]:
        source = self.current_state
        if source == OcclusionState.MULTI_HYPOTHESIS_REENTRY and target == OcclusionState.REEMERGED_CONFIRMED:
            if "multi_observation_confirmation" not in reason_codes and "large_margin_exception" not in reason_codes:
                raise ValueError("re-entry confirmation requires multiple observations or a large-margin exception")
        self.current_state = target
        if target == OcclusionState.HUMAN_REVIEW_REQUIRED:
            self.review_required = True
            self.review_reason_codes.extend(reason_codes)
        return {
            "anonymous_tracklet_id": self.anonymous_tracklet_id,
            "source_state": source.value,
            "target_state": target.value,
            "reason_codes": reason_codes,
            "threshold_result": threshold_result,
            "review_trigger": self.review_required,
            "fallback": "remain_unresolved_or_review" if self.review_required else "continue_diagnostic_state",
            **safety_payload(),
        }

    def to_row(self) -> dict[str, Any]:
        return {
            "anonymous_tracklet_id": self.anonymous_tracklet_id,
            "window_id": self.window_id,
            "current_state": self.current_state.value,
            "created_frame": self.created_frame,
            "last_observed_frame": self.last_observed_frame,
            "last_confirmed_frame": self.last_confirmed_frame,
            "motion_state": self.motion.to_dict(),
            "hidden_frame_count": self.hidden_frame_count,
            "dynamic_max_hidden_frames": self.dynamic_max_hidden_frames,
            "competing_anonymous_tracklet_ids": self.competing_anonymous_tracklet_ids,
            "merged_observation_ids": self.merged_observation_ids,
            "active_reentry_hypotheses": self.active_reentry_hypotheses,
            "review_required": self.review_required,
            "review_reason_codes": self.review_reason_codes,
            "termination_reason": self.termination_reason,
            **safety_payload(),
        }
