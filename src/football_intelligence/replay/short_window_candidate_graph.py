from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from football_intelligence.replay.anonymous_occlusion_state import ObservationNodeType
from football_intelligence.research_handoff.stage_workspace import safety_payload


class CandidateDecision(StrEnum):
    ASSIGN_DETECTION = "ASSIGN_DETECTION"
    KEEP_OCCLUDED_NULL = "KEEP_OCCLUDED_NULL"
    SHARE_MERGED_OBSERVATION = "SHARE_MERGED_OBSERVATION"
    TERMINATE_FRAME_EXIT = "TERMINATE_FRAME_EXIT"
    ESCALATE_REVIEW = "ESCALATE_REVIEW"


@dataclass(frozen=True)
class ImageBBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ImageBBox":
        return cls(float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"]))

    @property
    def width(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def footpoint(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def iou(self, other: "ImageBBox") -> float:
        left = max(self.x1, other.x1)
        top = max(self.y1, other.y1)
        right = min(self.x2, other.x2)
        bottom = min(self.y2, other.y2)
        if right <= left or bottom <= top:
            return 0.0
        intersection = (right - left) * (bottom - top)
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0


@dataclass(frozen=True)
class CandidateObservation:
    observation_id: str
    frame_sequence: int
    bbox: ImageBBox | None
    confidence: float | None
    node_type: ObservationNodeType = ObservationNodeType.DETECTION
    source: str = "canonical_frame_detection"
    appearance_similarity: float | None = None
    contamination_risk: str = "unknown"


@dataclass(frozen=True)
class CandidateGraphConfig:
    max_candidates: int = 10
    uncertainty_multiplier: float = 2.5
    null_candidate_cost: float = 1.15
    merged_candidate_cost: float = 0.95
    appearance_tie_break_weight: float = 0.08
    equal_path_margin: float = 0.08


def euclidean(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def normalized_distance(source: ImageBBox, target: ImageBBox, *, use_footpoint: bool) -> float:
    first = source.footpoint if use_footpoint else source.center
    second = target.footpoint if use_footpoint else target.center
    normalizer = max(1.0, (source.height + target.height) / 2.0)
    return euclidean(first, second) / normalizer


def mine_local_candidates(
    source: CandidateObservation,
    candidates: list[CandidateObservation],
    *,
    config: CandidateGraphConfig | None = None,
) -> list[dict[str, Any]]:
    """High-recall local miner. Appearance is intentionally absent from inclusion logic."""

    config = config or CandidateGraphConfig()
    if source.bbox is None:
        raise ValueError("source detection candidate mining requires a source bbox")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.node_type != ObservationNodeType.DETECTION or candidate.bbox is None:
            continue
        footpoint_norm = normalized_distance(source.bbox, candidate.bbox, use_footpoint=True)
        center_norm = normalized_distance(source.bbox, candidate.bbox, use_footpoint=False)
        gate = max(config.uncertainty_multiplier, 2.0 + source.bbox.height / 60.0)
        included = footpoint_norm <= gate or center_norm <= gate
        reasons = ["scale_aware_footpoint_gate"] if included else ["outside_scale_aware_gate"]
        rows.append(
            {
                "source_observation_id": source.observation_id,
                "target_observation_id": candidate.observation_id,
                "target_frame_sequence": candidate.frame_sequence,
                "candidate_node_type": candidate.node_type.value,
                "candidate_generation_uses_appearance": False,
                "footpoint_normalized_distance": round(footpoint_norm, 6),
                "center_normalized_distance": round(center_norm, 6),
                "included": included,
                "inclusion_reasons": reasons,
                "raw_confidence": candidate.confidence,
                **safety_payload(),
            }
        )
    rows.sort(key=lambda row: (not row["included"], row["footpoint_normalized_distance"], row["target_observation_id"]))
    retained = rows[: config.max_candidates]
    retained.append(
        {
            "source_observation_id": source.observation_id,
            "target_observation_id": f"{source.observation_id}__occluded_null",
            "target_frame_sequence": source.frame_sequence,
            "candidate_node_type": ObservationNodeType.OCCLUDED_NULL.value,
            "candidate_generation_uses_appearance": False,
            "footpoint_normalized_distance": None,
            "center_normalized_distance": None,
            "included": True,
            "inclusion_reasons": ["explicit_null_candidate_preserved"],
            "raw_confidence": None,
            **safety_payload(),
        }
    )
    if len(candidates) >= 2:
        retained.append(
            {
                "source_observation_id": source.observation_id,
                "target_observation_id": f"{source.observation_id}__merged_observation",
                "target_frame_sequence": source.frame_sequence,
                "candidate_node_type": ObservationNodeType.MERGED_OBSERVATION.value,
                "candidate_generation_uses_appearance": False,
                "footpoint_normalized_distance": None,
                "center_normalized_distance": None,
                "included": True,
                "inclusion_reasons": ["two_to_one_collapse_node_preserved"],
                "raw_confidence": None,
                **safety_payload(),
            }
        )
    return retained


def assignment_cost(
    source: CandidateObservation,
    target: CandidateObservation,
    *,
    conflict_active: bool,
    use_appearance: bool,
    config: CandidateGraphConfig | None = None,
) -> dict[str, Any]:
    config = config or CandidateGraphConfig()
    if target.node_type == ObservationNodeType.OCCLUDED_NULL:
        return {
            "decision": CandidateDecision.KEEP_OCCLUDED_NULL.value,
            "total_cost": config.null_candidate_cost,
            "appearance_used": False,
            "breakdown": {"null_assignment_cost": config.null_candidate_cost},
        }
    if target.node_type == ObservationNodeType.MERGED_OBSERVATION:
        return {
            "decision": CandidateDecision.SHARE_MERGED_OBSERVATION.value,
            "total_cost": config.merged_candidate_cost,
            "appearance_used": False,
            "breakdown": {"merged_observation_cost": config.merged_candidate_cost},
        }
    if target.node_type == ObservationNodeType.FRAME_EXIT:
        return {
            "decision": CandidateDecision.TERMINATE_FRAME_EXIT.value,
            "total_cost": 0.75,
            "appearance_used": False,
            "breakdown": {"frame_exit_cost": 0.75},
        }
    if source.bbox is None or target.bbox is None:
        raise ValueError("detection assignment requires source and target bboxes")
    footpoint_norm = normalized_distance(source.bbox, target.bbox, use_footpoint=True)
    center_norm = normalized_distance(source.bbox, target.bbox, use_footpoint=False)
    bbox_iou = source.bbox.iou(target.bbox)
    log_area_ratio = abs(math.log(max(1e-6, target.bbox.area / source.bbox.area)))
    log_height_ratio = abs(math.log(max(1e-6, target.bbox.height / source.bbox.height)))
    confidence_cost = 1.0 - min(1.0, max(0.0, target.confidence if target.confidence is not None else 0.5))
    motion_innovation = (footpoint_norm + center_norm) / 2.0
    appearance_term = 0.0
    appearance_used = False
    if conflict_active and use_appearance and target.appearance_similarity is not None:
        if target.contamination_risk not in {"high", "merged_overlap"} and target.bbox.height >= 24:
            appearance_used = True
            appearance_distance = 1.0 - max(0.0, min(1.0, target.appearance_similarity))
            appearance_term = appearance_distance * config.appearance_tie_break_weight
    total = (
        footpoint_norm * 0.42
        + center_norm * 0.18
        + (1.0 - bbox_iou) * 0.16
        + log_area_ratio * 0.08
        + log_height_ratio * 0.08
        + confidence_cost * 0.04
        + motion_innovation * 0.04
        + appearance_term
    )
    return {
        "decision": CandidateDecision.ASSIGN_DETECTION.value,
        "total_cost": round(total, 6),
        "appearance_used": appearance_used,
        "breakdown": {
            "normalized_footpoint_displacement": round(footpoint_norm, 6),
            "normalized_center_displacement": round(center_norm, 6),
            "bbox_iou": round(bbox_iou, 6),
            "log_area_ratio": round(log_area_ratio, 6),
            "log_height_ratio": round(log_height_ratio, 6),
            "detector_confidence_cost": round(confidence_cost, 6),
            "motion_innovation": round(motion_innovation, 6),
            "appearance_tie_break_term": round(appearance_term, 6),
        },
    }


def one_to_one_assign(
    sources: list[CandidateObservation],
    targets: list[CandidateObservation],
    *,
    conflict_active: bool = False,
    use_appearance: bool = False,
    config: CandidateGraphConfig | None = None,
) -> dict[str, Any]:
    config = config or CandidateGraphConfig()
    if len(sources) > len(targets):
        raise ValueError("targets must include enough detections/null nodes for one-to-one assignment")
    best: tuple[float, tuple[CandidateObservation, ...], list[dict[str, Any]]] | None = None
    for permutation in itertools.permutations(targets, len(sources)):
        costs = [
            assignment_cost(
                source,
                target,
                conflict_active=conflict_active,
                use_appearance=use_appearance,
                config=config,
            )
            for source, target in zip(sources, permutation, strict=True)
        ]
        total = round(sum(cost["total_cost"] for cost in costs), 6)
        candidate = (total, permutation, costs)
        if best is None or (total, [target.observation_id for target in permutation]) < (
            best[0],
            [target.observation_id for target in best[1]],
        ):
            best = candidate
    if best is None:
        raise ValueError("no assignment candidates supplied")
    total, permutation, costs = best
    rows = []
    for source, target, cost in zip(sources, permutation, costs, strict=True):
        rows.append(
            {
                "source_observation_id": source.observation_id,
                "target_observation_id": target.observation_id,
                "decision": cost["decision"],
                "total_cost": cost["total_cost"],
                "cost_breakdown": cost["breakdown"],
                "appearance_used": cost["appearance_used"],
                "one_to_one_enforced": True,
                **safety_payload(),
            }
        )
    return {"total_cost": total, "rows": rows}


def detect_reciprocal_conflict(cost_rows: list[dict[str, Any]], *, margin_threshold: float = 0.08) -> dict[str, Any]:
    by_target: dict[str, list[dict[str, Any]]] = {}
    for row in cost_rows:
        by_target.setdefault(str(row["target_observation_id"]), []).append(row)
    competing = {
        target_id: sorted(rows, key=lambda row: row["total_cost"])
        for target_id, rows in by_target.items()
        if len(rows) > 1
    }
    small_margins = []
    for target_id, rows in competing.items():
        margin = rows[1]["total_cost"] - rows[0]["total_cost"]
        if margin <= margin_threshold:
            small_margins.append({"target_observation_id": target_id, "best_second_margin": round(margin, 6)})
    return {
        "conflict_detected": bool(competing or small_margins),
        "competing_target_count": len(competing),
        "small_margin_conflicts": small_margins,
    }


def approach_to_occlusion_signals(
    sources: list[CandidateObservation],
    targets: list[CandidateObservation],
    *,
    challenge_category_present: bool = False,
) -> dict[str, Any]:
    strong: list[str] = []
    supporting: list[str] = []
    if len(targets) < len(sources):
        strong.append("visible_detections_collapse_from_two_to_one")
    if len(sources) >= 2 and all(source.bbox is not None for source in sources):
        first, second = sources[0].bbox, sources[1].bbox
        assert first is not None and second is not None
        if first.iou(second) > 0.1:
            strong.append("predicted_bboxes_overlap")
        if euclidean(first.footpoint, second.footpoint) <= max(first.height, second.height):
            strong.append("scale_aware_footpoint_convergence")
    if len(targets) >= 2:
        supporting.append("local_candidate_density_increases")
    if any(target.confidence is not None and target.confidence < 0.3 for target in targets):
        supporting.append("detector_confidence_drop")
    if challenge_category_present:
        supporting.append("historical_challenge_category_crossing_or_crowding")
    return {
        "approaching_occlusion": bool(strong and supporting),
        "strong_signals": strong,
        "supporting_signals": supporting,
    }


def k_best_hypotheses(
    source: CandidateObservation,
    targets: list[CandidateObservation],
    *,
    k: int = 3,
    conflict_active: bool = True,
    use_appearance: bool = False,
) -> list[dict[str, Any]]:
    scored = []
    for target in targets:
        score = assignment_cost(source, target, conflict_active=conflict_active, use_appearance=use_appearance)
        scored.append((score["total_cost"], target, score))
    scored.sort(key=lambda item: (item[0], item[1].observation_id))
    rows = []
    for rank, (total, target, score) in enumerate(scored[: max(1, k)], start=1):
        rows.append(
            {
                "hypothesis_rank": rank,
                "source_observation_id": source.observation_id,
                "target_observation_id": target.observation_id,
                "path_cost": total,
                "cost_breakdown": score["breakdown"],
                "appearance_used": score["appearance_used"],
                "branch_pruning_reason": "bounded_k_best_window",
                **safety_payload(),
            }
        )
    if len(rows) >= 2:
        margin = rows[1]["path_cost"] - rows[0]["path_cost"]
        rows[0]["best_second_margin"] = round(margin, 6)
        rows[0]["near_equal_paths_preserved_as_unresolved"] = margin <= CandidateGraphConfig().equal_path_margin
    return rows
