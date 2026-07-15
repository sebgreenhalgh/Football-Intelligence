from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Any

from football_intelligence.replay.short_window_candidate_graph import ImageBBox
from football_intelligence.research_handoff.stage_workspace import safety_payload


@dataclass(frozen=True)
class FrameObservation:
    observation_id: str
    frame_sequence: int
    bbox: ImageBBox
    confidence: float | None = None
    source_provenance: str = "canonical_person_candidate"
    appearance_similarity: float | None = None
    contamination: float = 0.0

    @property
    def footpoint(self) -> tuple[float, float]:
        return self.bbox.footpoint

    def to_row(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "frame_sequence": self.frame_sequence,
            "bbox": self.bbox.__dict__,
            "footpoint": {"x": self.footpoint[0], "y": self.footpoint[1]},
            "confidence": self.confidence,
            "bbox_height_band": bbox_height_band(self.bbox.height),
            "source_provenance": self.source_provenance,
        }


@dataclass(frozen=True)
class MotionFit:
    status: str
    fitted_frame_sequences: tuple[int, ...]
    state_mean: dict[str, float]
    covariance: list[list[float]]
    innovations: tuple[dict[str, float], ...]

    @property
    def usable(self) -> bool:
        return self.status == "FIT_COMPLETE"


@dataclass(frozen=True)
class ResolverConfig:
    k_best: int = 4
    beam_width: int = 16
    max_candidates_per_track: int = 3
    normalized_motion_gate: float = 2.4
    geometry_margin_threshold: float = 0.12
    null_penalty: float = 1.1
    merged_penalty: float = 0.82
    frame_exit_penalty: float = 0.7
    appearance_weight: float = 0.06
    max_hidden_frames: int = 12


@dataclass
class _TrackState:
    footpoint_x: float
    footpoint_y: float
    velocity_x: float
    velocity_y: float
    log_width: float
    log_height: float
    covariance: float
    hidden_frames: int = 0
    terminated: bool = False
    last_bbox: ImageBBox | None = None

    def predict(self) -> tuple[float, float]:
        return self.footpoint_x + self.velocity_x, self.footpoint_y + self.velocity_y

    def clone(self) -> "_TrackState":
        return _TrackState(**self.__dict__)


@dataclass
class _Beam:
    paths: dict[str, list[dict[str, Any]]]
    states: dict[str, _TrackState]
    total_cost: float = 0.0
    cost_breakdown: dict[str, float] = field(default_factory=dict)


def bbox_height_band(height: float) -> str:
    if height < 24:
        return "small_under_24px"
    if height <= 50:
        return "medium_24_to_50px"
    return "large_over_50px"


def _linear_fit(frames: list[float], values: list[float]) -> tuple[float, float, list[float]]:
    mean_x = sum(frames) / len(frames)
    mean_y = sum(values) / len(values)
    denominator = sum((value - mean_x) ** 2 for value in frames)
    slope = (
        0.0
        if denominator == 0
        else sum((frame - mean_x) * (value - mean_y) for frame, value in zip(frames, values, strict=True)) / denominator
    )
    intercept = mean_y - slope * mean_x
    residuals = [value - (intercept + slope * frame) for frame, value in zip(frames, values, strict=True)]
    return intercept, slope, residuals


def fit_incoming_motion(observations: list[FrameObservation]) -> MotionFit:
    ordered = sorted(observations, key=lambda row: (row.frame_sequence, row.observation_id))
    frames = [row.frame_sequence for row in ordered]
    if len(set(frames)) != len(frames):
        raise ValueError("duplicated observation frame in incoming motion history")
    if len({row.observation_id for row in ordered}) != len(ordered):
        raise ValueError("duplicated observation object in incoming motion history")
    if len(ordered) < 3:
        return MotionFit(
            status="INSUFFICIENT_INCOMING_HISTORY",
            fitted_frame_sequences=tuple(frames),
            state_mean={},
            covariance=[],
            innovations=(),
        )
    x_values = [row.footpoint[0] for row in ordered]
    y_values = [row.footpoint[1] for row in ordered]
    width_values = [math.log(max(1.0, row.bbox.width)) for row in ordered]
    height_values = [math.log(max(1.0, row.bbox.height)) for row in ordered]
    _, vx, x_residuals = _linear_fit([float(value) for value in frames], x_values)
    _, vy, y_residuals = _linear_fit([float(value) for value in frames], y_values)
    _, vw, width_residuals = _linear_fit([float(value) for value in frames], width_values)
    _, vh, height_residuals = _linear_fit([float(value) for value in frames], height_values)
    last = ordered[-1]
    residual_columns = (x_residuals, y_residuals, width_residuals, height_residuals)
    variances = [max(1e-6, sum(value**2 for value in column) / len(column)) for column in residual_columns]
    covariance_diag = [
        variances[0],
        variances[1],
        max(variances[0], 1.0),
        max(variances[1], 1.0),
        variances[2],
        variances[3],
    ]
    covariance = [[0.0 for _ in covariance_diag] for _ in covariance_diag]
    for index, value in enumerate(covariance_diag):
        covariance[index][index] = round(value, 6)
    innovations = tuple(
        {
            "frame_sequence": float(row.frame_sequence),
            "footpoint_x_residual": round(x_residuals[index], 6),
            "footpoint_y_residual": round(y_residuals[index], 6),
            "innovation_norm": round(math.hypot(x_residuals[index], y_residuals[index]), 6),
        }
        for index, row in enumerate(ordered)
    )
    return MotionFit(
        status="FIT_COMPLETE",
        fitted_frame_sequences=tuple(frames),
        state_mean={
            "footpoint_x": round(last.footpoint[0], 6),
            "footpoint_y": round(last.footpoint[1], 6),
            "velocity_x": round(vx, 6),
            "velocity_y": round(vy, 6),
            "log_width": round(math.log(max(1.0, last.bbox.width)), 6),
            "log_height": round(math.log(max(1.0, last.bbox.height)), 6),
            "log_width_velocity": round(vw, 6),
            "log_height_velocity": round(vh, 6),
        },
        covariance=covariance,
        innovations=innovations,
    )


def _predicted_bbox(state: _TrackState) -> ImageBBox:
    x, y = state.predict()
    width = math.exp(state.log_width)
    height = math.exp(state.log_height)
    return ImageBBox(x - width / 2.0, y - height, x + width / 2.0, y)


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def observable_conflict_signals(
    incoming_histories: dict[str, list[FrameObservation]],
    observations_by_frame: dict[int, list[FrameObservation]],
    *,
    margin_threshold: float = 0.12,
) -> dict[str, Any]:
    fits: dict[str, MotionFit] = {}
    for track_id, history in sorted(incoming_histories.items()):
        fits[track_id] = fit_incoming_motion(history)
    usable = {key: value for key, value in fits.items() if value.usable}
    evidence: dict[str, Any] = {
        "predicted_iou": 0.0,
        "footpoint_convergence": None,
        "reciprocal_competition_count": 0,
        "relative_order_reversal": False,
        "two_to_one_observation_collapse": False,
        "area_inflation_ratio": 1.0,
        "best_second_margin": None,
        "local_candidate_density": max((len(rows) for rows in observations_by_frame.values()), default=0),
        "disappearing_compatible_observation": False,
    }
    if len(usable) >= 2:
        track_ids = sorted(usable)[:2]
        left_fit, right_fit = usable[track_ids[0]], usable[track_ids[1]]
        left_state = _state_from_fit(left_fit, incoming_histories[track_ids[0]][-1].bbox)
        right_state = _state_from_fit(right_fit, incoming_histories[track_ids[1]][-1].bbox)
        left_bbox = _predicted_bbox(left_state)
        right_bbox = _predicted_bbox(right_state)
        evidence["predicted_iou"] = round(left_bbox.iou(right_bbox), 6)
        previous_distance = _distance(
            incoming_histories[track_ids[0]][-1].footpoint,
            incoming_histories[track_ids[1]][-1].footpoint,
        )
        predicted_distance = _distance(left_bbox.footpoint, right_bbox.footpoint)
        evidence["footpoint_convergence"] = round(previous_distance - predicted_distance, 6)
        previous_order = (
            incoming_histories[track_ids[0]][-1].footpoint[0] <= incoming_histories[track_ids[1]][-1].footpoint[0]
        )
        predicted_order = left_bbox.footpoint[0] <= right_bbox.footpoint[0]
        evidence["relative_order_reversal"] = previous_order != predicted_order
        next_frame = max(row.frame_sequence for history in incoming_histories.values() for row in history) + 1
        next_rows = observations_by_frame.get(next_frame, [])
        compatible_counts = []
        candidate_costs: list[list[float]] = []
        for bbox in (left_bbox, right_bbox):
            costs = sorted(_distance(bbox.footpoint, row.footpoint) / max(1.0, bbox.height) for row in next_rows)
            candidate_costs.append(costs)
            compatible_counts.append(sum(value <= 2.4 for value in costs))
        evidence["two_to_one_observation_collapse"] = len(next_rows) == 1 and sum(compatible_counts) >= 2
        if next_rows:
            shared = 0
            for row in next_rows:
                if all(
                    _distance(bbox.footpoint, row.footpoint) / max(1.0, bbox.height) <= 2.4
                    for bbox in (left_bbox, right_bbox)
                ):
                    shared += 1
            evidence["reciprocal_competition_count"] = shared
            incoming_area = sum(history[-1].bbox.area for history in incoming_histories.values()) / len(
                incoming_histories
            )
            evidence["area_inflation_ratio"] = round(
                max(row.bbox.area for row in next_rows) / max(1.0, incoming_area), 6
            )
        all_costs = sorted(value for costs in candidate_costs for value in costs)
        if len(all_costs) >= 2:
            evidence["best_second_margin"] = round(all_costs[1] - all_costs[0], 6)
        evidence["disappearing_compatible_observation"] = any(value == 0 for value in compatible_counts)
    triggers = {
        "predicted_overlap": evidence["predicted_iou"] >= 0.05,
        "converging_footpoints": evidence["footpoint_convergence"] is not None
        and evidence["footpoint_convergence"] > 0,
        "reciprocal_competition": evidence["reciprocal_competition_count"] > 0,
        "relative_order_reversal": evidence["relative_order_reversal"],
        "two_to_one_collapse": evidence["two_to_one_observation_collapse"],
        "area_inflation": evidence["area_inflation_ratio"] >= 1.35,
        "low_assignment_margin": evidence["best_second_margin"] is not None
        and evidence["best_second_margin"] <= margin_threshold,
        "disappearing_observation": evidence["disappearing_compatible_observation"],
    }
    strong = sum(
        bool(triggers[key])
        for key in ("predicted_overlap", "reciprocal_competition", "two_to_one_collapse", "relative_order_reversal")
    )
    supporting = sum(bool(value) for value in triggers.values()) - strong
    return {
        "conflict_active": strong >= 1 and supporting >= 1,
        "numeric_evidence": evidence,
        "thresholds": {
            "predicted_iou": 0.05,
            "area_inflation_ratio": 1.35,
            "best_second_margin": margin_threshold,
            "normalized_candidate_gate": 2.4,
        },
        "triggers": triggers,
        "case_id_or_category_used": False,
        "motion_fits": {key: value.status for key, value in fits.items()},
    }


def appearance_activation_gate(
    *,
    conflict_active: bool,
    motion_compatible_candidate_count: int,
    geometry_margin: float | None,
    source_bbox_height: float,
    target_bbox_heights: list[float],
    source_contamination: float,
    target_contamination: float,
    margin_threshold: float = 0.12,
) -> dict[str, Any]:
    gates = {
        "conflict_active": conflict_active,
        "multiple_motion_compatible_candidates": motion_compatible_candidate_count >= 2,
        "geometry_margin_below_threshold": geometry_margin is not None and geometry_margin <= margin_threshold,
        "source_crop_quality_sufficient": source_bbox_height >= 24,
        "target_crop_quality_sufficient": bool(target_bbox_heights) and min(target_bbox_heights) >= 24,
        "contamination_below_threshold": max(source_contamination, target_contamination) <= 0.35,
    }
    return {
        "eligible": all(gates.values()),
        "gates": gates,
        "geometry_margin_threshold": margin_threshold,
        "contamination_threshold": 0.35,
    }


def _state_from_fit(fit: MotionFit, bbox: ImageBBox) -> _TrackState:
    mean = fit.state_mean
    return _TrackState(
        footpoint_x=mean["footpoint_x"],
        footpoint_y=mean["footpoint_y"],
        velocity_x=mean["velocity_x"],
        velocity_y=mean["velocity_y"],
        log_width=mean["log_width"],
        log_height=mean["log_height"],
        covariance=max(1.0, fit.covariance[0][0] + fit.covariance[1][1]),
        last_bbox=bbox,
    )


def _detection_cost(state: _TrackState, observation: FrameObservation) -> dict[str, float]:
    predicted = _predicted_bbox(state)
    motion = _distance(predicted.footpoint, observation.footpoint) / max(1.0, predicted.height)
    scale = abs(math.log(max(1e-6, observation.bbox.height / predicted.height)))
    confidence = 1.0 - max(0.0, min(1.0, observation.confidence if observation.confidence is not None else 0.5))
    return {
        "motion_cost": round(motion * 0.72, 6),
        "scale_cost": round(scale * 0.18, 6),
        "detection_confidence_cost": round(confidence * 0.10, 6),
    }


def _track_options(
    state: _TrackState,
    observations: list[FrameObservation],
    *,
    frame_sequence: int,
    image_size: tuple[int, int],
    config: ResolverConfig,
    appearance_enabled: bool,
) -> list[dict[str, Any]]:
    if state.terminated:
        return [{"node_type": "FRAME_EXIT", "observation": None, "base_cost": 0.0}]
    predicted = _predicted_bbox(state)
    candidates = []
    for observation in observations:
        costs = _detection_cost(state, observation)
        normalized_motion = costs["motion_cost"] / 0.72
        if normalized_motion > config.normalized_motion_gate:
            continue
        appearance = 0.0
        if appearance_enabled and observation.appearance_similarity is not None:
            appearance = (1.0 - max(0.0, min(1.0, observation.appearance_similarity))) * config.appearance_weight
        candidates.append(
            {
                "node_type": "DETECTION",
                "observation": observation,
                "base_cost": round(sum(costs.values()) + appearance, 6),
                "costs": costs | {"appearance_cost": round(appearance, 6)},
            }
        )
    candidates.sort(key=lambda row: (row["base_cost"], row["observation"].observation_id))
    options = candidates[: config.max_candidates_per_track]
    options.append(
        {
            "node_type": "OCCLUDED_NULL",
            "observation": None,
            "base_cost": round(config.null_penalty + state.hidden_frames * 0.1, 6),
            "costs": {"null_penalty": round(config.null_penalty + state.hidden_frames * 0.1, 6)},
        }
    )
    width, height = image_size
    px, py = predicted.footpoint
    near_boundary = px <= predicted.width or px >= width - predicted.width or py <= predicted.height or py >= height
    if near_boundary:
        options.append(
            {
                "node_type": "FRAME_EXIT",
                "observation": None,
                "base_cost": config.frame_exit_penalty,
                "costs": {"frame_exit_penalty": config.frame_exit_penalty},
                "boundary_evidence": {"predicted_footpoint": [px, py], "image_size": [width, height]},
            }
        )
    return options


def _merged_observable(states: list[_TrackState], observation: FrameObservation) -> bool:
    if len(states) < 2:
        return False
    predictions = [_predicted_bbox(state) for state in states]
    compatible = [
        _distance(predicted.footpoint, observation.footpoint) / max(1.0, predicted.height) <= 1.5
        for predicted in predictions
    ]
    mean_area = sum(predicted.area for predicted in predictions) / len(predictions)
    predictions_overlap = predictions[0].iou(predictions[1]) >= 0.03
    return all(compatible) and (observation.bbox.area >= mean_area * 1.25 or predictions_overlap)


def _update_state(state: _TrackState, option: dict[str, Any]) -> _TrackState:
    updated = state.clone()
    node_type = option["node_type"]
    if node_type in {"DETECTION", "MERGED_OBSERVATION"}:
        observation: FrameObservation = option["observation"]
        predicted_x, predicted_y = updated.predict()
        residual_x = observation.footpoint[0] - predicted_x
        residual_y = observation.footpoint[1] - predicted_y
        prior_x, prior_y = updated.footpoint_x, updated.footpoint_y
        updated.footpoint_x = predicted_x + residual_x * 0.65
        updated.footpoint_y = predicted_y + residual_y * 0.65
        updated.velocity_x = 0.55 * updated.velocity_x + 0.45 * (updated.footpoint_x - prior_x)
        updated.velocity_y = 0.55 * updated.velocity_y + 0.45 * (updated.footpoint_y - prior_y)
        updated.log_width = math.log(max(1.0, observation.bbox.width))
        updated.log_height = math.log(max(1.0, observation.bbox.height))
        updated.covariance = max(1.0, updated.covariance * (0.55 if node_type == "DETECTION" else 0.9))
        updated.hidden_frames = 0 if node_type == "DETECTION" else updated.hidden_frames + 1
        updated.last_bbox = observation.bbox
    elif node_type == "OCCLUDED_NULL":
        updated.footpoint_x, updated.footpoint_y = updated.predict()
        updated.covariance = min(1_000_000.0, updated.covariance * 1.25 + 1.0)
        updated.hidden_frames += 1
    elif node_type == "FRAME_EXIT":
        updated.terminated = True
    return updated


def _node_row(track_id: str, frame_sequence: int, option: dict[str, Any]) -> dict[str, Any]:
    observation = option.get("observation")
    return {
        "track_id": track_id,
        "frame_sequence": frame_sequence,
        "node_type": option["node_type"],
        "observation_id": observation.observation_id if observation is not None else None,
        "bbox": observation.bbox.__dict__ if observation is not None else None,
        "cost": option.get("base_cost", 0.0),
        "cost_breakdown": option.get("costs", {}),
        "boundary_evidence": option.get("boundary_evidence"),
    }


def resolve_joint_sequence(
    *,
    incoming_histories: dict[str, list[FrameObservation]],
    observations_by_frame: dict[int, list[FrameObservation]],
    window_frames: list[int],
    image_size: tuple[int, int],
    config: ResolverConfig | None = None,
    appearance_enabled: bool = False,
) -> dict[str, Any]:
    config = config or ResolverConfig()
    if len(window_frames) < 2 or window_frames != sorted(set(window_frames)):
        raise ValueError("window frames must be sorted and unique")
    fits = {track_id: fit_incoming_motion(history) for track_id, history in incoming_histories.items()}
    if not fits or any(not fit.usable for fit in fits.values()):
        return {
            "classification": "INSUFFICIENT_INCOMING_HISTORY",
            "motion_fits": fits,
            "hypotheses": [],
            "graph_nodes": [],
            "graph_edges": [],
        }
    start_frame = max(history[-1].frame_sequence for history in incoming_histories.values())
    initial_paths: dict[str, list[dict[str, Any]]] = {}
    states: dict[str, _TrackState] = {}
    for track_id, history in sorted(incoming_histories.items()):
        history_by_frame = {row.frame_sequence: row for row in history}
        initial_paths[track_id] = []
        for frame in window_frames:
            if frame > start_frame:
                break
            observation = history_by_frame.get(frame)
            initial_paths[track_id].append(
                {
                    "track_id": track_id,
                    "frame_sequence": frame,
                    "node_type": "DETECTION" if observation is not None else "OCCLUDED_NULL",
                    "observation_id": observation.observation_id if observation is not None else None,
                    "bbox": observation.bbox.__dict__ if observation is not None else None,
                    "cost": 0.0,
                    "cost_breakdown": {"incoming_history": 0.0},
                }
            )
        states[track_id] = _state_from_fit(fits[track_id], history[-1].bbox)
    beams = [_Beam(paths=initial_paths, states=states)]
    rejected_rows: list[dict[str, Any]] = []
    for frame_sequence in [frame for frame in window_frames if frame > start_frame]:
        expanded: list[_Beam] = []
        observations = observations_by_frame.get(frame_sequence, [])
        for beam in beams:
            track_ids = sorted(beam.states)
            option_sets = [
                _track_options(
                    beam.states[track_id],
                    observations,
                    frame_sequence=frame_sequence,
                    image_size=image_size,
                    config=config,
                    appearance_enabled=appearance_enabled,
                )
                for track_id in track_ids
            ]
            for options in itertools.product(*option_sets):
                detection_ids = [
                    option["observation"].observation_id
                    for option in options
                    if option["node_type"] == "DETECTION" and option.get("observation") is not None
                ]
                mutable_options = [dict(option) for option in options]
                exclusivity_cost = 0.0
                if len(detection_ids) != len(set(detection_ids)):
                    shared_id = next(value for value in detection_ids if detection_ids.count(value) > 1)
                    shared_observation = next(
                        option["observation"]
                        for option in options
                        if option.get("observation") is not None and option["observation"].observation_id == shared_id
                    )
                    if not _merged_observable([beam.states[track_id] for track_id in track_ids], shared_observation):
                        rejected_rows.append(
                            {
                                "frame_sequence": frame_sequence,
                                "reason": "joint_one_to_one_exclusivity_violation",
                                "observation_id": shared_id,
                            }
                        )
                        continue
                    for option in mutable_options:
                        if option.get("observation") is not None and option["observation"].observation_id == shared_id:
                            option["node_type"] = "MERGED_OBSERVATION"
                            option["base_cost"] = config.merged_penalty
                            option["costs"] = {"merged_penalty": config.merged_penalty}
                    exclusivity_cost = config.merged_penalty
                new_paths = {key: list(value) for key, value in beam.paths.items()}
                new_states = {key: value.clone() for key, value in beam.states.items()}
                increment: dict[str, float] = {
                    "motion_cost": 0.0,
                    "scale_cost": 0.0,
                    "detection_confidence_cost": 0.0,
                    "null_merged_exit_penalties": 0.0,
                    "appearance_cost": 0.0,
                    "exclusivity_cost": exclusivity_cost,
                }
                for track_id, option in zip(track_ids, mutable_options, strict=True):
                    new_paths[track_id].append(_node_row(track_id, frame_sequence, option))
                    new_states[track_id] = _update_state(new_states[track_id], option)
                    costs = option.get("costs", {})
                    increment["motion_cost"] += costs.get("motion_cost", 0.0)
                    increment["scale_cost"] += costs.get("scale_cost", 0.0)
                    increment["detection_confidence_cost"] += costs.get("detection_confidence_cost", 0.0)
                    increment["appearance_cost"] += costs.get("appearance_cost", 0.0)
                    increment["null_merged_exit_penalties"] += sum(
                        costs.get(key, 0.0) for key in ("null_penalty", "merged_penalty", "frame_exit_penalty")
                    )
                breakdown = dict(beam.cost_breakdown)
                for key, value in increment.items():
                    breakdown[key] = round(breakdown.get(key, 0.0) + value, 6)
                total = round(sum(breakdown.values()), 6)
                expanded.append(_Beam(paths=new_paths, states=new_states, total_cost=total, cost_breakdown=breakdown))
        expanded.sort(key=lambda item: (item.total_cost, _paths_fingerprint(item.paths)))
        beams = expanded[: config.beam_width]
        if not beams:
            raise ValueError(f"joint beam exhausted at frame {frame_sequence}")
    beams.sort(key=lambda item: (item.total_cost, _paths_fingerprint(item.paths)))
    selected = beams[: config.k_best]
    hypotheses = []
    for rank, beam in enumerate(selected, start=1):
        complete = all(len(path) == len(window_frames) for path in beam.paths.values())
        if not complete:
            raise AssertionError("joint hypothesis does not cover every window frame")
        hypotheses.append(
            {
                "rank": rank,
                "paths": beam.paths,
                "cost_breakdown": beam.cost_breakdown,
                "total_cost": beam.total_cost,
                "complete_window": True,
                "one_to_one_exclusivity_enforced": True,
                "appearance_contribution": beam.cost_breakdown.get("appearance_cost", 0.0),
                "pruning_reason": "deterministic_bounded_joint_beam",
            }
        )
    margin = selected[1].total_cost - selected[0].total_cost if len(selected) > 1 else None
    graph_nodes = []
    graph_edges = []
    seen_nodes = set()
    for hypothesis in hypotheses:
        for track_id, path in hypothesis["paths"].items():
            for node in path:
                node_key = (track_id, node["frame_sequence"], node["node_type"], node.get("observation_id"))
                if node_key not in seen_nodes:
                    graph_nodes.append(node)
                    seen_nodes.add(node_key)
            for source, target in zip(path, path[1:], strict=False):
                if target["frame_sequence"] - source["frame_sequence"] != 1:
                    raise AssertionError("graph edge must connect adjacent frames")
                graph_edges.append(
                    {
                        "track_id": track_id,
                        "source_frame_sequence": source["frame_sequence"],
                        "target_frame_sequence": target["frame_sequence"],
                        "source_node_type": source["node_type"],
                        "target_node_type": target["node_type"],
                        "adjacent_frame_edge": True,
                        "hypothesis_rank": hypothesis["rank"],
                    }
                )
    return {
        "classification": "RESOLVED_K_BEST",
        "motion_fits": fits,
        "hypotheses": hypotheses,
        "best_second_margin": round(margin, 6) if margin is not None else None,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "rejected_assignments": rejected_rows,
        "window_frames": window_frames,
        "frame_coverage_complete": True,
        "appearance_enabled": appearance_enabled,
        **safety_payload(),
    }


def _paths_fingerprint(paths: dict[str, list[dict[str, Any]]]) -> str:
    reduced = {
        track_id: [(row["frame_sequence"], row["node_type"], row.get("observation_id")) for row in path]
        for track_id, path in sorted(paths.items())
    }
    return hashlib.sha256(
        json.dumps(reduced, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def answer_independent_fingerprint(result: dict[str, Any]) -> str:
    payload = {
        "classification": result["classification"],
        "hypotheses": result["hypotheses"],
        "best_second_margin": result.get("best_second_margin"),
        "graph_nodes": result["graph_nodes"],
        "graph_edges": result["graph_edges"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")
    ).hexdigest()


def execute_ghost_intervals(
    *,
    case_id: str,
    hypothesis: dict[str, Any],
    max_hidden_frames: int,
) -> dict[str, Any]:
    state_rows: list[dict[str, Any]] = []
    reentry_rows: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    hidden_intervals: list[dict[str, Any]] = []
    for track_id, path in sorted(hypothesis["paths"].items()):
        active: dict[str, Any] | None = None
        seen_observation = False
        covariance = 1.0
        post_gap_detections = 0
        for node in path:
            node_type = node["node_type"]
            frame = int(node["frame_sequence"])
            if node_type == "DETECTION":
                seen_observation = True
                if active is None:
                    continue
                post_gap_detections += 1
                reentry_rows.append(
                    {
                        "case_id": case_id,
                        "track_id": track_id,
                        "frame_sequence": frame,
                        "candidate_observation_id": node.get("observation_id"),
                        "post_gap_confirmation_count": post_gap_detections,
                        "confirmed": post_gap_detections >= 2,
                    }
                )
                if post_gap_detections >= 2:
                    active["end_frame_sequence"] = frame
                    active["hidden_frame_count"] = len(active["node_types"])
                    active["post_gap_candidate_count"] = post_gap_detections
                    active["reentry_confirmed"] = True
                    hidden_intervals.append(active)
                    intervals.append(active)
                    active = None
                    covariance = 1.0
            elif node_type in {"OCCLUDED_NULL", "MERGED_OBSERVATION"}:
                if not seen_observation:
                    continue
                if active is None:
                    active = {
                        "case_id": case_id,
                        "track_id": track_id,
                        "start_frame_sequence": frame,
                        "node_types": [],
                        "expired": False,
                        "terminated": False,
                    }
                    post_gap_detections = 0
                active["node_types"].append(node_type)
                covariance = covariance * 1.25 + 1.0
                hidden_frames = len(active["node_types"])
                expired = hidden_frames > max_hidden_frames
                active["expired"] = expired
                active["terminated"] = expired
                state_rows.append(
                    {
                        "case_id": case_id,
                        "track_id": track_id,
                        "frame_sequence": frame,
                        "state": "TERMINATED" if expired else "FULLY_OCCLUDED_PREDICTED",
                        "observation_node_type": node_type,
                        "hidden_frame_count": hidden_frames,
                        "covariance": round(covariance, 6),
                        "prediction_advanced": True,
                        "dynamic_expiry_executed": expired,
                    }
                )
                if expired:
                    active["end_frame_sequence"] = frame
                    active["hidden_frame_count"] = len(active["node_types"])
                    active["post_gap_candidate_count"] = 0
                    active["reentry_confirmed"] = False
                    hidden_intervals.append(active)
                    active = None
                    seen_observation = False
                    covariance = 1.0
            elif node_type == "FRAME_EXIT":
                if active is not None:
                    active["end_frame_sequence"] = frame
                    active["hidden_frame_count"] = len(active["node_types"])
                    active["terminated"] = True
                    active["termination_reason"] = "frame_exit"
                    active["post_gap_candidate_count"] = post_gap_detections
                    active["reentry_confirmed"] = False
                    hidden_intervals.append(active)
                    if post_gap_detections:
                        intervals.append(active)
                    active = None
                seen_observation = False
        if active is not None:
            active["end_frame_sequence"] = path[-1]["frame_sequence"]
            active["hidden_frame_count"] = len(active["node_types"])
            active["terminated"] = active["expired"] or len(active["node_types"]) >= max_hidden_frames
            active["post_gap_candidate_count"] = post_gap_detections
            active["reentry_confirmed"] = False
            hidden_intervals.append(active)
            if post_gap_detections:
                intervals.append(active)
    return {
        "eligible_intervals": intervals,
        "hidden_intervals": hidden_intervals,
        "ghost_state_rows": state_rows,
        "reentry_hypotheses": reentry_rows,
        "executed": bool(hidden_intervals),
    }
