"""Definitive shared-graph sports-MOT bakeoff and executable MHSAG candidate."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable

from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.sports_mot.architecture import PitchParticipantGate, build_common_observation_graph


ORACLE_MODE = "ORACLE_OBSERVATION_ASSOCIATION"
DETECTOR_MODE = "DETECTOR_CONSTRAINED_END_TO_END"
BENCHMARK_MODES = (ORACLE_MODE, DETECTOR_MODE)
MHSAG = "MHSAG"
TIER1_ADAPTERS = (
    "CURRENT_PROJECT_BASELINE",
    "BYTETRACK",
    "OC_SORT",
    "BOT_SORT",
    "DEEP_OC_SORT",
    "CLEAN_ROOM_DEEP_EIOU",
    "DEEP_EIOU_PLUS_GTA",
    MHSAG,
)


@dataclass(frozen=True)
class AssociationConfig:
    algorithm: str
    variant: str
    motion_weight: float
    iou_weight: float
    appearance_weight: float
    confidence_weight: float
    scale_weight: float
    null_cost: float
    hard_gate_height_multiplier: float
    maximum_scale_ratio: float
    ambiguity_margin: float
    maximum_candidates: int = 8
    top_k: int = 4
    low_confidence_recovery: bool = False
    observation_centric_motion: bool = False
    expansion_iou: bool = False
    reliability_gated_appearance: bool = False
    purity_splitting: bool = False
    global_linking: bool = False
    top_k_ambiguity: bool = True

    @property
    def configuration_hash(self) -> str:
        return stable_hash(asdict(self))


BASE_CONFIGS = {
    "CURRENT_PROJECT_BASELINE": AssociationConfig(
        "CURRENT_PROJECT_BASELINE", "balanced", 1.0, 0.08, 0.0, 0.12, 0.30, 1.8, 3.2, 2.2, 0.04
    ),
    "BYTETRACK": AssociationConfig(
        "BYTETRACK", "balanced", 0.9, 0.08, 0.0, 0.35, 0.28, 1.75, 3.2, 2.2, 0.05, low_confidence_recovery=True
    ),
    "OC_SORT": AssociationConfig(
        "OC_SORT", "balanced", 1.0, 0.10, 0.0, 0.10, 0.25, 1.65, 3.0, 2.1, 0.07, observation_centric_motion=True
    ),
    "BOT_SORT": AssociationConfig(
        "BOT_SORT",
        "balanced",
        0.9,
        0.10,
        0.24,
        0.12,
        0.26,
        1.65,
        3.0,
        2.1,
        0.08,
        reliability_gated_appearance=True,
    ),
    "DEEP_OC_SORT": AssociationConfig(
        "DEEP_OC_SORT",
        "balanced",
        0.88,
        0.12,
        0.28,
        0.10,
        0.24,
        1.6,
        2.9,
        2.0,
        0.09,
        observation_centric_motion=True,
        reliability_gated_appearance=True,
    ),
    "CLEAN_ROOM_DEEP_EIOU": AssociationConfig(
        "CLEAN_ROOM_DEEP_EIOU",
        "balanced",
        0.82,
        0.24,
        0.16,
        0.12,
        0.22,
        1.55,
        2.9,
        2.0,
        0.10,
        observation_centric_motion=True,
        expansion_iou=True,
        reliability_gated_appearance=True,
    ),
    "DEEP_EIOU_PLUS_GTA": AssociationConfig(
        "DEEP_EIOU_PLUS_GTA",
        "balanced",
        0.78,
        0.28,
        0.18,
        0.10,
        0.20,
        1.5,
        2.8,
        1.95,
        0.11,
        observation_centric_motion=True,
        expansion_iou=True,
        reliability_gated_appearance=True,
        purity_splitting=True,
        global_linking=True,
    ),
    MHSAG: AssociationConfig(
        MHSAG,
        "balanced",
        0.76,
        0.30,
        0.16,
        0.16,
        0.18,
        1.42,
        2.75,
        1.9,
        0.13,
        maximum_candidates=10,
        top_k=6,
        low_confidence_recovery=True,
        observation_centric_motion=True,
        expansion_iou=True,
        reliability_gated_appearance=True,
        purity_splitting=True,
        global_linking=True,
        top_k_ambiguity=True,
    ),
}


def configuration_variants(algorithm: str) -> list[AssociationConfig]:
    base = BASE_CONFIGS[algorithm]
    conservative = replace(
        base,
        variant="conservative",
        null_cost=base.null_cost * 0.82,
        hard_gate_height_multiplier=base.hard_gate_height_multiplier * 0.92,
        ambiguity_margin=max(base.ambiguity_margin, 0.14),
    )
    coverage = replace(
        base,
        variant="coverage",
        null_cost=base.null_cost * 1.16,
        hard_gate_height_multiplier=base.hard_gate_height_multiplier * 1.06,
        ambiguity_margin=base.ambiguity_margin * 0.6,
    )
    return [conservative, base, coverage]


def _bbox(value: dict[str, Any]) -> dict[str, float]:
    box = value.get("bbox", value)
    return {key: float(box[key]) for key in ("x1", "y1", "x2", "y2")}


def _height(value: dict[str, Any]) -> float:
    box = _bbox(value)
    return max(1.0, box["y2"] - box["y1"])


def _foot(value: dict[str, Any]) -> tuple[float, float]:
    footpoint = value.get("footpoint")
    if isinstance(footpoint, dict):
        return (float(footpoint["x"]), float(footpoint["y"]))
    box = _bbox(value)
    return ((box["x1"] + box["x2"]) / 2.0, box["y2"])


def _expanded_iou(left: dict[str, Any], right: dict[str, Any], factor: float) -> float:
    def expand(value: dict[str, Any]) -> dict[str, float]:
        box = _bbox(value)
        cx = (box["x1"] + box["x2"]) / 2.0
        cy = (box["y1"] + box["y2"]) / 2.0
        half_width = (box["x2"] - box["x1"]) * factor / 2.0
        half_height = (box["y2"] - box["y1"]) * factor / 2.0
        return {"x1": cx - half_width, "y1": cy - half_height, "x2": cx + half_width, "y2": cy + half_height}

    a, b = expand(left), expand(right)
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return intersection / max(1.0, area_a + area_b - intersection)


def _appearance_distance(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    a, b = left.get("colour_descriptor"), right.get("colour_descriptor")
    if not isinstance(a, list) or not isinstance(b, list) or not a or len(a) != len(b):
        return None
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _last_nodes(path: list[str | None], nodes: dict[str, dict[str, Any]], maximum: int = 2) -> list[dict[str, Any]]:
    output = []
    for node_id in reversed(path):
        if node_id is None:
            continue
        output.append(nodes[node_id])
        if len(output) == maximum:
            break
    return list(reversed(output))


def _predicted_foot(history: list[dict[str, Any]], target_frame: int, use_velocity: bool) -> tuple[float, float]:
    last = history[-1]
    last_foot = _foot(last)
    if not use_velocity or len(history) < 2:
        return last_foot
    previous = history[-2]
    frame_gap = max(1, int(last["frame_sequence"]) - int(previous["frame_sequence"]))
    target_gap = max(1, target_frame - int(last["frame_sequence"]))
    previous_foot = _foot(previous)
    velocity = ((last_foot[0] - previous_foot[0]) / frame_gap, (last_foot[1] - previous_foot[1]) / frame_gap)
    return (last_foot[0] + velocity[0] * target_gap, last_foot[1] + velocity[1] * target_gap)


def _hard_compatible(
    history: list[dict[str, Any]], candidate: dict[str, Any] | None, config: AssociationConfig
) -> bool:
    if candidate is None or not history:
        return True
    previous = history[-1]
    frame_gap = max(1, int(candidate["frame_sequence"]) - int(previous["frame_sequence"]))
    predicted = _predicted_foot(history, int(candidate["frame_sequence"]), config.observation_centric_motion)
    displacement = math.dist(predicted, _foot(candidate))
    limit = max(32.0, config.hard_gate_height_multiplier * max(_height(previous), _height(candidate))) * frame_gap
    scale_ratio = max(_height(previous), _height(candidate)) / min(_height(previous), _height(candidate))
    return (
        displacement <= limit
        and scale_ratio <= config.maximum_scale_ratio
        and bool(previous.get("pitch_gate_eligible"))
        and bool(candidate.get("pitch_gate_eligible"))
    )


def _transition_cost(
    history: list[dict[str, Any]], candidate: dict[str, Any] | None, config: AssociationConfig
) -> float:
    if candidate is None:
        return config.null_cost
    if not history:
        return config.confidence_weight * (1.0 - float(candidate.get("confidence", 0.0)))
    previous = history[-1]
    predicted = _predicted_foot(history, int(candidate["frame_sequence"]), config.observation_centric_motion)
    displacement = math.dist(predicted, _foot(candidate))
    motion = displacement / max(24.0, 2.8 * max(_height(previous), _height(candidate)))
    scale = abs(math.log(max(1e-6, _height(candidate) / _height(previous))))
    iou = _expanded_iou(previous, candidate, 1.35 if config.expansion_iou else 1.0)
    appearance = _appearance_distance(previous, candidate)
    reliability = min(
        float(previous.get("appearance_reliability", 0.0)), float(candidate.get("appearance_reliability", 0.0))
    )
    if not config.reliability_gated_appearance or reliability < 0.55 or appearance is None:
        appearance_cost = 0.0
    else:
        appearance_cost = appearance * reliability
    confidence = float(candidate.get("confidence", 0.0))
    low_confidence_penalty = 0.0
    if confidence < 0.35 and not config.low_confidence_recovery:
        low_confidence_penalty = 0.8
    return (
        config.motion_weight * motion
        + config.iou_weight * (1.0 - iou)
        + config.appearance_weight * appearance_cost
        + config.confidence_weight * (1.0 - confidence)
        + config.scale_weight * scale
        + low_confidence_penalty
    )


def _candidate_options(
    frame_nodes: list[dict[str, Any]], history: list[dict[str, Any]], config: AssociationConfig
) -> list[dict[str, Any] | None]:
    eligible = [node for node in frame_nodes if node.get("pitch_gate_eligible")]
    if history:
        target_frame = int(eligible[0]["frame_sequence"]) if eligible else int(history[-1]["frame_sequence"]) + 1
        predicted = _predicted_foot(history, target_frame, config.observation_centric_motion)
        eligible.sort(key=lambda node: (math.dist(predicted, _foot(node)), -float(node.get("confidence", 0.0))))
    else:
        eligible.sort(key=lambda node: (-float(node.get("confidence", 0.0)), str(node["node_id"])))
    return [*eligible[: config.maximum_candidates], None]


def run_shared_graph_adapter(
    graph: dict[str, Any],
    *,
    config: AssociationConfig,
    seed_a_node_id: str,
    seed_b_node_id: str,
) -> dict[str, Any]:
    if config.algorithm not in TIER1_ADAPTERS:
        raise ValueError(f"unsupported Tier-1 adapter: {config.algorithm}")
    nodes = {str(node["node_id"]): node for node in graph["nodes"]}
    if seed_a_node_id not in nodes or seed_b_node_id not in nodes or seed_a_node_id == seed_b_node_id:
        raise ValueError("two distinct seed observations are required")
    frames = [int(frame) for frame in graph["allowed_frames"]]
    if nodes[seed_a_node_id]["frame_sequence"] != frames[0] or nodes[seed_b_node_id]["frame_sequence"] != frames[0]:
        raise ValueError("seeds must bind to the first graph frame")
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in graph["nodes"]:
        by_frame[int(node["frame_sequence"])].append(node)
    beams: list[dict[str, Any]] = [{"cost": 0.0, "A": [seed_a_node_id], "B": [seed_b_node_id]}]
    started = time.perf_counter()
    for frame in frames[1:]:
        expanded = []
        for beam in beams:
            history_a = _last_nodes(beam["A"], nodes)
            history_b = _last_nodes(beam["B"], nodes)
            options_a = _candidate_options(by_frame[frame], history_a, config)
            options_b = _candidate_options(by_frame[frame], history_b, config)
            for candidate_a in options_a:
                for candidate_b in options_b:
                    if (
                        candidate_a is not None
                        and candidate_b is not None
                        and candidate_a["node_id"] == candidate_b["node_id"]
                    ):
                        continue
                    if not _hard_compatible(history_a, candidate_a, config) or not _hard_compatible(
                        history_b, candidate_b, config
                    ):
                        continue
                    cost = (
                        beam["cost"]
                        + _transition_cost(history_a, candidate_a, config)
                        + _transition_cost(history_b, candidate_b, config)
                    )
                    expanded.append(
                        {
                            "cost": cost,
                            "A": [*beam["A"], candidate_a["node_id"] if candidate_a else None],
                            "B": [*beam["B"], candidate_b["node_id"] if candidate_b else None],
                        }
                    )
        expanded.sort(key=lambda row: (row["cost"], str(row["A"]), str(row["B"])))
        beams = expanded[: max(30, config.top_k * 8)]
        if not beams:
            return {
                "algorithm": config.algorithm,
                "configuration_hash": config.configuration_hash,
                "input_graph_hash": graph["graph_hash"],
                "status": "FAILED_NO_VALID_JOINT_PATH",
                "strand_states": [],
                "top_k_joint_paths": [],
                "runtime_seconds": round(time.perf_counter() - started, 6),
            }
    retained = beams[: config.top_k]
    best = retained[0]
    states = []
    for index, frame in enumerate(frames):
        near_best = [row for row in retained if row["cost"] - best["cost"] <= config.ambiguity_margin]
        frame_state: dict[str, Any] = {"frame_sequence": frame}
        for strand in ("A", "B"):
            alternatives = {row[strand][index] for row in near_best}
            ambiguous = config.top_k_ambiguity and len(alternatives) > 1
            node_id = None if ambiguous else best[strand][index]
            frame_state[strand] = {
                "state": "AMBIGUOUS" if ambiguous else "OBSERVED" if node_id is not None else "MISSING",
                "node_id": node_id,
                "candidate_alternative_count": len(alternatives),
            }
        states.append(frame_state)
    result = {
        "algorithm": config.algorithm,
        "adapter_version": "definitive-shared-graph-v1",
        "configuration": asdict(config),
        "configuration_hash": config.configuration_hash,
        "input_graph_hash": graph["graph_hash"],
        "status": "COMPLETED",
        "best_joint_cost": round(best["cost"], 8),
        "best_to_second_margin": round(retained[1]["cost"] - best["cost"], 8) if len(retained) > 1 else None,
        "strand_states": states,
        "top_k_joint_paths": [
            {"rank": rank, "cost": round(row["cost"], 8), "A": row["A"], "B": row["B"]}
            for rank, row in enumerate(retained, start=1)
        ],
        "one_to_one_enforced": True,
        "null_state_allowed": True,
        "ambiguous_state_allowed": True,
        "fixed_start_seeds_only": True,
        "forced_end_mapping": False,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "tracker_promoted": False,
    }
    if config.algorithm == MHSAG:
        result["mhsag"] = build_mhsag_outputs(graph, result, config)
        result["status"] = "MHSAG_EXECUTED"
    return result


def build_mhsag_outputs(graph: dict[str, Any], result: dict[str, Any], config: AssociationConfig) -> dict[str, Any]:
    nodes = {str(node["node_id"]): node for node in graph["nodes"]}
    tracklets = []
    purity_rows = []
    for strand in ("A", "B"):
        current: list[str] = []
        for state in result["strand_states"]:
            node_id = state[strand]["node_id"]
            if node_id is None:
                if current:
                    tracklets.append({"strand": strand, "node_ids": current})
                    current = []
                continue
            if current:
                previous = nodes[current[-1]]
                node = nodes[node_id]
                jump = math.dist(_foot(previous), _foot(node))
                height_limit = 2.5 * max(_height(previous), _height(node))
                appearance = _appearance_distance(previous, node)
                reliable = (
                    min(
                        float(previous.get("appearance_reliability", 0.0)),
                        float(node.get("appearance_reliability", 0.0)),
                    )
                    >= 0.55
                )
                reason = None
                if node_id in current:
                    reason = "DUPLICATE_OBSERVATION_REUSE"
                elif jump > max(36.0, height_limit):
                    reason = "MOTION_DISCONTINUITY"
                elif reliable and appearance is not None and appearance > 0.75:
                    reason = "APPEARANCE_DISCONTINUITY"
                if reason and config.purity_splitting:
                    tracklets.append({"strand": strand, "node_ids": current})
                    purity_rows.append(
                        {
                            "strand": strand,
                            "split_before_frame": node["frame_sequence"],
                            "reason": reason,
                            "jump_pixels": round(jump, 6),
                            "appearance_distance": round(appearance, 6) if appearance is not None else None,
                        }
                    )
                    current = []
            current.append(node_id)
        if current:
            tracklets.append({"strand": strand, "node_ids": current})
    tracklet_rows = []
    for index, tracklet in enumerate(tracklets, start=1):
        values = [nodes[node_id] for node_id in tracklet["node_ids"]]
        tracklet_rows.append(
            {
                "tracklet_id": f"local_tracklet_{index:03d}",
                "strand": tracklet["strand"],
                "start_frame": values[0]["frame_sequence"],
                "end_frame": values[-1]["frame_sequence"],
                "node_ids": tracklet["node_ids"],
                "pure_after_audit": True,
            }
        )
    link_rows = []
    selected_links = []
    incoming_used: set[str] = set()
    outgoing_used: set[str] = set()
    for left in tracklet_rows:
        for right in tracklet_rows:
            if left["end_frame"] >= right["start_frame"] or left["strand"] != right["strand"]:
                continue
            left_node, right_node = nodes[left["node_ids"][-1]], nodes[right["node_ids"][0]]
            gap = int(right["start_frame"]) - int(left["end_frame"])
            motion = math.dist(_foot(left_node), _foot(right_node)) / max(24.0, _height(left_node), _height(right_node))
            appearance = _appearance_distance(left_node, right_node)
            link_cost = motion + (0.0 if appearance is None else 0.2 * appearance) + 0.1 * gap
            allowed = gap <= 5 and link_cost < config.null_cost * 2.0
            row = {
                "source_tracklet_id": left["tracklet_id"],
                "target_tracklet_id": right["tracklet_id"],
                "gap_frames": gap,
                "link_cost": round(link_cost, 8),
                "link_allowed": allowed,
                "no_link_allowed": True,
                "one_to_one_required": True,
            }
            link_rows.append(row)
    for row in sorted(link_rows, key=lambda item: (item["link_cost"], item["source_tracklet_id"])):
        if not row["link_allowed"]:
            continue
        if row["source_tracklet_id"] in outgoing_used or row["target_tracklet_id"] in incoming_used:
            continue
        outgoing_used.add(row["source_tracklet_id"])
        incoming_used.add(row["target_tracklet_id"])
        selected_links.append(row)
    return {
        "status": "EXECUTED_NOT_PROMOTED",
        "short_tracklets": tracklet_rows,
        "purity_audit": purity_rows,
        "global_link_candidates": link_rows,
        "selected_min_cost_dag_links": selected_links,
        "global_no_link_count": len(tracklet_rows) - len(selected_links),
        "top_k_global_alternatives": result["top_k_joint_paths"],
        "observation_centric_motion": config.observation_centric_motion,
        "low_confidence_recovery": config.low_confidence_recovery,
        "expansion_iou": config.expansion_iou,
        "reliability_gated_appearance": config.reliability_gated_appearance,
        "one_to_one": True,
        "null_and_ambiguous_states": True,
        "persistent_identity_created": False,
    }


def build_oracle_graph(
    sequence_rows: list[dict[str, Any]], pitch_gate: PitchParticipantGate
) -> tuple[dict[str, Any], str, str]:
    observations = []
    for row in sequence_rows:
        for strand in ("A", "B"):
            value = row[strand]
            if value.get("bbox") is None:
                continue
            observations.append(
                {
                    "observation_id": f"oracle_{row['frame_sequence']}_{strand}",
                    "frame_sequence": row["frame_sequence"],
                    "bbox": value["bbox"],
                    "confidence": 1.0,
                    "source_layer": "human_gold_oracle_observation",
                    "source_row_hash": value["source_row_hash"],
                    "coordinate_space": "canonical_panorama_pixels",
                    "observation_quality": "HUMAN_GOLD",
                    "appearance_reliability": 0.0,
                }
            )
    frames = [row["frame_sequence"] for row in sequence_rows]
    graph = build_common_observation_graph(
        observations, pitch_gate=pitch_gate, allowed_frames=frames, roi=sequence_rows[0]["roi"]
    )
    graph["benchmark_mode"] = ORACLE_MODE
    graph["sequence_id"] = sequence_rows[0]["sequence_id"]
    graph["approved_polygon_hash"] = sequence_rows[0]["approved_polygon_hash"]
    graph["graph_hash"] = stable_hash({key: value for key, value in graph.items() if key != "graph_hash"})
    first_frame = frames[0]
    seed_a = f"oracle_{first_frame}_A"
    seed_b = f"oracle_{first_frame}_B"
    if seed_a not in {node["node_id"] for node in graph["nodes"]} or seed_b not in {
        node["node_id"] for node in graph["nodes"]
    }:
        raise ValueError("oracle seeds are not visible gold observations")
    return graph, seed_a, seed_b


def build_detector_graph(
    sequence_rows: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    pitch_gate: PitchParticipantGate,
) -> tuple[dict[str, Any], str, str]:
    frames = [row["frame_sequence"] for row in sequence_rows]
    graph = build_common_observation_graph(
        observations, pitch_gate=pitch_gate, allowed_frames=frames, roi=sequence_rows[0]["roi"]
    )
    graph["benchmark_mode"] = DETECTOR_MODE
    graph["sequence_id"] = sequence_rows[0]["sequence_id"]
    graph["approved_polygon_hash"] = sequence_rows[0]["approved_polygon_hash"]
    graph["graph_hash"] = stable_hash({key: value for key, value in graph.items() if key != "graph_hash"})
    seed_a = sequence_rows[0]["A"]["source_observation_id"]
    seed_b = sequence_rows[0]["B"]["source_observation_id"]
    node_ids = {node["node_id"] for node in graph["nodes"]}
    if seed_a not in node_ids or seed_b not in node_ids or seed_a == seed_b:
        raise ValueError("detector graph cannot bind both confirmed starting seeds")
    return graph, str(seed_a), str(seed_b)


def evaluate_sequence(
    *,
    result: dict[str, Any],
    graph: dict[str, Any],
    gold_rows: list[dict[str, Any]],
    benchmark_mode: str,
) -> dict[str, Any]:
    nodes = {str(node["node_id"]): node for node in graph["nodes"]}
    predictions = {int(row["frame_sequence"]): row for row in result.get("strand_states", [])}
    false_continuations = 0
    switches = 0
    losses = 0
    safe_abstentions = 0
    supply_failures = 0
    off_pitch = 0
    double_assignments = 0
    provenance_failures = 0
    correct = 0
    eligible = 0
    detector_supply = 0
    frame_rows = []
    for gold in gold_rows:
        frame = int(gold["frame_sequence"])
        prediction = predictions.get(frame, {"A": {"node_id": None}, "B": {"node_id": None}})
        a_id = prediction.get("A", {}).get("node_id")
        b_id = prediction.get("B", {}).get("node_id")
        if a_id is not None and a_id == b_id:
            double_assignments += 1
        for strand in ("A", "B"):
            truth = gold[strand]
            predicted_id = prediction.get(strand, {}).get("node_id")
            if benchmark_mode == ORACLE_MODE and truth.get("bbox") is not None:
                expected_id = f"oracle_{frame}_{strand}"
                supply_available = True
            elif benchmark_mode == DETECTOR_MODE and truth["state"] == "OBSERVED_EXISTING_DETECTION":
                expected_id = truth.get("source_observation_id")
                supply_available = expected_id in nodes
            else:
                expected_id = None
                supply_available = False
            if truth.get("bbox") is not None or truth["state"] == "MISSING_VISIBLE_NO_VALID_DETECTION":
                eligible += 1
            detector_supply += int(supply_available)
            outcome = "CORRECT_ABSTENTION"
            if expected_id is not None and predicted_id == expected_id:
                correct += 1
                outcome = "CORRECT_CONTINUATION"
            elif expected_id is not None and predicted_id is None:
                if supply_available:
                    losses += 1
                    outcome = "STRAND_LOSS_DESPITE_SUPPLY"
                else:
                    safe_abstentions += 1
                    supply_failures += 1
                    outcome = "DETECTION_SUPPLY_FAILURE"
            elif expected_id is not None and predicted_id != expected_id:
                false_continuations += 1
                switches += 1
                outcome = "ASSOCIATION_SWITCH"
            elif expected_id is None and predicted_id is None:
                safe_abstentions += int(truth["state"] in {"AMBIGUOUS", "MISSING_VISIBLE_NO_VALID_DETECTION"})
                supply_failures += int(truth["state"] == "MISSING_VISIBLE_NO_VALID_DETECTION")
                outcome = (
                    "DETECTION_SUPPLY_FAILURE"
                    if truth["state"] == "MISSING_VISIBLE_NO_VALID_DETECTION"
                    else "SAFE_ABSTENTION"
                )
            elif expected_id is None and predicted_id is not None:
                false_continuations += 1
                outcome = "FALSE_CONTINUATION_WITHOUT_GOLD_TARGET"
            if predicted_id is not None:
                node = nodes.get(str(predicted_id))
                if node is None:
                    provenance_failures += 1
                elif not node.get("pitch_gate_eligible"):
                    off_pitch += 1
            frame_rows.append(
                {
                    "frame_sequence": frame,
                    "strand": strand,
                    "gold_state": truth["state"],
                    "expected_node_id": expected_id,
                    "predicted_node_id": predicted_id,
                    "supply_available": supply_available,
                    "outcome": outcome,
                }
            )
    exact = (
        false_continuations == 0
        and losses == 0
        and off_pitch == 0
        and double_assignments == 0
        and provenance_failures == 0
    )
    deta = detector_supply / max(1, eligible)
    assa = correct / max(1, detector_supply)
    return {
        "sequence_id": gold_rows[0]["sequence_id"],
        "split": gold_rows[0]["split"],
        "benchmark_mode": benchmark_mode,
        "algorithm": result["algorithm"],
        "configuration_hash": result["configuration_hash"],
        "input_graph_hash": result["input_graph_hash"],
        "fully_exact_sequence": exact,
        "correct_strand_frames": correct,
        "eligible_strand_frames": eligible,
        "false_continuations": false_continuations,
        "identity_switches": switches,
        "strand_losses_when_supply_available": losses,
        "safe_abstentions": safe_abstentions,
        "detection_supply_failures": supply_failures,
        "off_pitch_assignments": off_pitch,
        "double_assignments": double_assignments,
        "renderer_provenance_failures": provenance_failures,
        "exact_path_coverage": correct / max(1, eligible),
        "HOTA": math.sqrt(deta * assa),
        "DetA": deta,
        "AssA": assa,
        "IDF1": 2 * correct / max(1, eligible + detector_supply),
        "runtime_seconds": result.get("runtime_seconds", 0.0),
        "frame_attribution_rows": frame_rows,
    }


def aggregate_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    count = len(values)
    summed = {
        key: sum(int(row[key]) for row in values)
        for key in (
            "false_continuations",
            "identity_switches",
            "strand_losses_when_supply_available",
            "safe_abstentions",
            "detection_supply_failures",
            "off_pitch_assignments",
            "double_assignments",
            "renderer_provenance_failures",
            "correct_strand_frames",
            "eligible_strand_frames",
        )
    }
    exact_count = sum(bool(row["fully_exact_sequence"]) for row in values)
    means = {
        key: sum(float(row[key]) for row in values) / max(1, count)
        for key in ("exact_path_coverage", "HOTA", "DetA", "AssA", "IDF1", "runtime_seconds")
    }
    return {
        "sequence_count": count,
        "fully_exact_sequences": exact_count,
        **summed,
        **means,
    }


def lexicographic_key(metrics: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(metrics["false_continuations"]),
        int(metrics["identity_switches"]),
        int(metrics["strand_losses_when_supply_available"]),
        -int(metrics["fully_exact_sequences"]),
        -float(metrics["exact_path_coverage"]),
        int(metrics["safe_abstentions"]),
        -float(metrics["AssA"]),
        -float(metrics["IDF1"]),
        -float(metrics["HOTA"]),
        float(metrics["runtime_seconds"]),
    )


def select_development_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    invalid_splits = sorted({str(row.get("split")) for row in rows if row.get("split") != "development"})
    if invalid_splits:
        raise ValueError(f"selection rows must be development-only: {invalid_splits}")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["configuration_hash"], row["benchmark_mode"])].append(row)
    summaries = []
    for (algorithm, configuration_hash, mode), group in grouped.items():
        metrics = aggregate_metrics(group)
        summaries.append(
            {
                "algorithm": algorithm,
                "configuration_hash": configuration_hash,
                "benchmark_mode": mode,
                "metrics": metrics,
                "lexicographic_key": list(lexicographic_key(metrics)),
            }
        )
    detector_summaries = [row for row in summaries if row["benchmark_mode"] == DETECTOR_MODE]
    detector_summaries.sort(
        key=lambda row: (tuple(row["lexicographic_key"]), row["algorithm"], row["configuration_hash"])
    )
    winner = detector_summaries[0] if detector_summaries else None
    development_gate = bool(
        winner
        and winner["metrics"]["false_continuations"] == 0
        and winner["metrics"]["identity_switches"] == 0
        and winner["metrics"]["strand_losses_when_supply_available"] == 0
        and winner["metrics"]["off_pitch_assignments"] == 0
        and winner["metrics"]["double_assignments"] == 0
        and winner["metrics"]["renderer_provenance_failures"] == 0
    )
    return {
        "selection_protocol": "GROUPED_SEQUENCE_LEVEL_DEVELOPMENT_AGGREGATION",
        "diagnostic_rows_used_for_selection": 0,
        "holdout_rows_used_for_selection": 0,
        "candidate_summaries": summaries,
        "selected": winner,
        "development_hard_gate_passed": development_gate,
        "lexicographic_objective": [
            "zero_switches_and_false_continuations",
            "zero_loss_despite_supply",
            "maximum_exact_path_coverage",
            "minimum_unnecessary_safe_abstention",
            "supplementary_AssA_IDF1_HOTA",
            "runtime",
        ],
    }


def grouped_leave_one_sequence_out(rows: list[dict[str, Any]]) -> dict[str, Any]:
    invalid_splits = sorted({str(row.get("split")) for row in rows if row.get("split") != "development"})
    if invalid_splits:
        raise ValueError(f"cross-validation rows must be development-only: {invalid_splits}")
    sequence_ids = sorted({str(row["sequence_id"]) for row in rows})
    if len(sequence_ids) < 3:
        raise ValueError("grouped leave-one-sequence-out requires at least three sequences")
    folds = []
    for held_out_sequence_id in sequence_ids:
        training_rows = [row for row in rows if row["sequence_id"] != held_out_sequence_id]
        validation_rows = [row for row in rows if row["sequence_id"] == held_out_sequence_id]
        selection = select_development_winner(training_rows)
        selected = selection.get("selected")
        selected_validation_rows = [
            row
            for row in validation_rows
            if selected
            and row["algorithm"] == selected["algorithm"]
            and row["configuration_hash"] == selected["configuration_hash"]
            and row["benchmark_mode"] == DETECTOR_MODE
        ]
        folds.append(
            {
                "held_out_sequence_id": held_out_sequence_id,
                "training_sequence_ids": [value for value in sequence_ids if value != held_out_sequence_id],
                "selected_algorithm": selected.get("algorithm") if selected else None,
                "selected_configuration_hash": selected.get("configuration_hash") if selected else None,
                "training_hard_gate_passed": selection["development_hard_gate_passed"],
                "validation_metrics": aggregate_metrics(selected_validation_rows),
                "group_overlap_count": 0,
            }
        )
    return {
        "protocol": "LEAVE_ONE_SEQUENCE_OUT_GROUPED_BY_TEMPORAL_SEQUENCE",
        "fold_count": len(folds),
        "sequence_ids": sequence_ids,
        "diagnostic_rows_used": 0,
        "sealed_holdout_rows_used": 0,
        "all_group_overlaps_zero": all(row["group_overlap_count"] == 0 for row in folds),
        "folds": folds,
    }


def holdout_acceptance(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "false_continuations": metrics["false_continuations"] == 0,
        "identity_switches": metrics["identity_switches"] == 0,
        "strand_losses_when_supply_available": metrics["strand_losses_when_supply_available"] == 0,
        "off_pitch_assignments": metrics["off_pitch_assignments"] == 0,
        "double_assignments": metrics["double_assignments"] == 0,
        "renderer_provenance_failures": metrics["renderer_provenance_failures"] == 0,
        "minimum_fully_exact_sequences": metrics["fully_exact_sequences"] >= 7,
        "maximum_nonexact_sequences": metrics["sequence_count"] - metrics["fully_exact_sequences"] <= 1,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "hard_gate_metrics": metrics,
        "retuning_after_result_forbidden": True,
    }


def run_cuda_probe() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; silent CPU fallback is prohibited")
    device_index = 0
    device = torch.device(f"cuda:{device_index}")
    baseline_peak_allocated = int(torch.cuda.max_memory_allocated())
    baseline_peak_reserved = int(torch.cuda.max_memory_reserved())
    started = time.perf_counter()
    left = torch.linspace(0.0, 1.0, 1024 * 512, device=device, dtype=torch.float32).to(torch.float16)
    right = torch.linspace(1.0, 0.0, 512 * 256, device=device, dtype=torch.float32).to(torch.float16)
    left = left.reshape(1024, 512)
    right = right.reshape(512, 256)
    result = left @ right
    checksum = float(result.float().mean().item())
    torch.cuda.synchronize()
    if not math.isfinite(checksum):
        raise RuntimeError("CUDA FP16 probe produced a non-finite checksum")
    return {
        "device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "fp16_executed": True,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "baseline_peak_allocated_vram_bytes": baseline_peak_allocated,
        "baseline_peak_reserved_vram_bytes": baseline_peak_reserved,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
        "result_checksum": round(checksum, 8),
        "silent_cpu_fallback": False,
    }


def ablation_configs(config: AssociationConfig) -> dict[str, AssociationConfig]:
    return {
        "without_low_confidence_recovery": replace(config, low_confidence_recovery=False),
        "without_observation_centric_motion": replace(config, observation_centric_motion=False),
        "without_appearance": replace(config, appearance_weight=0.0, reliability_gated_appearance=False),
        "without_purity_splitting": replace(config, purity_splitting=False),
        "without_global_linking": replace(config, global_linking=False),
        "without_top_k_ambiguity": replace(config, top_k_ambiguity=False, top_k=1, ambiguity_margin=0.0),
    }
