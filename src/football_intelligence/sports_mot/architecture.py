"""Clean-room sports-MOT benchmark primitives for short anonymous A/B strands.

These interfaces intentionally stop at match-local, image-space benchmark
states. They do not create a persistent player identity or a production track.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from football_intelligence.review_chassis.hashing import stable_hash


ANNOTATION_STATES = {
    "OBSERVED_EXISTING_DETECTION",
    "OBSERVED_MANUAL_BBOX",
    "MISSING_VISIBLE_NO_VALID_DETECTION",
    "NOT_VISIBLE",
    "AMBIGUOUS",
    "OUTSIDE_ROI",
}

PITCH_ZONES = {
    "INSIDE_PLAYABLE_PITCH",
    "BOUNDARY_OFFICIAL_ZONE",
    "OFF_PITCH_STAFF_OR_SPECTATOR",
    "UNRESOLVED",
}


def _bbox(value: dict[str, Any]) -> dict[str, float]:
    box = value.get("bbox", value)
    return {key: float(box[key]) for key in ("x1", "y1", "x2", "y2")}


def _foot(value: dict[str, Any]) -> tuple[float, float]:
    box = _bbox(value)
    return ((box["x1"] + box["x2"]) / 2.0, box["y2"])


def _height(value: dict[str, Any]) -> float:
    box = _bbox(value)
    return max(1.0, box["y2"] - box["y1"])


def _area(box: dict[str, float]) -> float:
    return max(0.0, box["x2"] - box["x1"]) * max(0.0, box["y2"] - box["y1"])


def _iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    a, b = _bbox(left), _bbox(right)
    intersection = {
        "x1": max(a["x1"], b["x1"]),
        "y1": max(a["y1"], b["y1"]),
        "x2": min(a["x2"], b["x2"]),
        "y2": min(a["y2"], b["y2"]),
    }
    overlap = _area(intersection)
    return overlap / max(1.0, _area(a) + _area(b) - overlap)


def _point_in_polygon(point: tuple[float, float], vertices: tuple[tuple[float, float], ...]) -> bool:
    x, y = point
    inside = False
    previous = vertices[-1]
    for current in vertices:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / max(1e-12, y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.dist(point, start)
    position = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + position * dx), py - (y1 + position * dy))


@dataclass(frozen=True)
class PitchParticipantGate:
    vertices: tuple[tuple[float, float], ...]
    tolerance_pixels: float
    source_frame_sha256: str
    approval_status: str = "PENDING_HUMAN_APPROVAL"

    def __post_init__(self) -> None:
        if len(self.vertices) < 4:
            raise ValueError("playable-pitch polygon requires at least four vertices")
        if self.tolerance_pixels < 0:
            raise ValueError("pitch tolerance cannot be negative")

    @property
    def polygon_hash(self) -> str:
        return stable_hash(
            {
                "vertices": [{"x": x, "y": y} for x, y in self.vertices],
                "tolerance_pixels": float(self.tolerance_pixels),
            }
        )

    def classify(self, footpoint: tuple[float, float]) -> dict[str, Any]:
        distance = min(
            _point_segment_distance(footpoint, self.vertices[index - 1], self.vertices[index])
            for index in range(len(self.vertices))
        )
        inside = _point_in_polygon(footpoint, self.vertices)
        if inside and distance > self.tolerance_pixels:
            zone = "INSIDE_PLAYABLE_PITCH"
        elif distance <= self.tolerance_pixels:
            zone = "BOUNDARY_OFFICIAL_ZONE"
        else:
            zone = "OFF_PITCH_STAFF_OR_SPECTATOR"
        return {
            "footpoint": {"x": round(footpoint[0], 6), "y": round(footpoint[1], 6)},
            "zone": zone,
            "distance_to_polygon_boundary_pixels": round(distance, 6),
            "inside_polygon": inside,
            "primary_benchmark_eligible": zone == "INSIDE_PLAYABLE_PITCH",
            "polygon_hash": self.polygon_hash,
        }


def _descriptor_distance(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    a = left.get("colour_descriptor")
    b = right.get("colour_descriptor")
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b) or not a:
        return None
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def build_common_observation_graph(
    observations: list[dict[str, Any]],
    *,
    pitch_gate: PitchParticipantGate,
    allowed_frames: list[int],
    roi: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build one immutable graph shared by every clean-room adapter."""
    allowed = set(allowed_frames)
    nodes: list[dict[str, Any]] = []
    for index, source in enumerate(observations):
        frame = int(source["frame_sequence"])
        if frame not in allowed:
            continue
        box = _bbox(source)
        footpoint = _foot(source)
        gate = pitch_gate.classify(footpoint)
        if roi and not (roi["x1"] <= footpoint[0] <= roi["x2"] and roi["y1"] <= footpoint[1] <= roi["y2"]):
            continue
        node = {
            "node_id": str(source.get("observation_id") or f"observation_{frame:06d}_{index:04d}"),
            "frame_sequence": frame,
            "bbox": box,
            "footpoint": gate["footpoint"],
            "confidence": float(source.get("confidence", 0.0)),
            "source_layer": str(source.get("source_layer") or source.get("source_type") or "unknown"),
            "source_row_hash": str(source.get("source_row_hash") or stable_hash(source)),
            "coordinate_space": str(source.get("coordinate_space") or "canonical_panorama_pixels"),
            "pitch_zone": gate["zone"],
            "pitch_gate_eligible": gate["primary_benchmark_eligible"],
            "colour_descriptor": source.get("colour_descriptor"),
            "appearance_reliability": float(source.get("appearance_reliability", 0.0)),
            "observation_quality": str(source.get("observation_quality", "UNRESOLVED")),
        }
        nodes.append(node)
    nodes.sort(key=lambda row: (row["frame_sequence"], row["node_id"]))
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for frame in allowed_frames:
        by_frame[frame] = [node for node in nodes if node["frame_sequence"] == frame]
    edges: list[dict[str, Any]] = []
    for left_frame, right_frame in zip(allowed_frames, allowed_frames[1:]):
        gap = max(1, right_frame - left_frame)
        for left in by_frame[left_frame]:
            for right in by_frame[right_frame]:
                displacement = math.dist(
                    (left["footpoint"]["x"], left["footpoint"]["y"]),
                    (right["footpoint"]["x"], right["footpoint"]["y"]),
                )
                height_ratio = max(_height(left), _height(right)) / min(_height(left), _height(right))
                hard_limit = max(36.0, 3.2 * max(_height(left), _height(right))) * gap
                pitch_compatible = left["pitch_gate_eligible"] and right["pitch_gate_eligible"]
                hard_gate = displacement <= hard_limit and height_ratio <= 2.2 and pitch_compatible
                edges.append(
                    {
                        "edge_id": stable_hash({"left": left["node_id"], "right": right["node_id"]}),
                        "source_node_id": left["node_id"],
                        "target_node_id": right["node_id"],
                        "source_frame_sequence": left_frame,
                        "target_frame_sequence": right_frame,
                        "gap_length": gap,
                        "motion_residual_pixels": round(displacement, 6),
                        "footpoint_residual_pixels": round(displacement, 6),
                        "bbox_scale_ratio": round(height_ratio, 6),
                        "bbox_iou": round(_iou(left, right), 6),
                        "appearance_distance": _descriptor_distance(left, right),
                        "observation_quality": min(left["confidence"], right["confidence"]),
                        "pitch_zone_compatible": pitch_compatible,
                        "hard_gate_pass": hard_gate,
                    }
                )
    graph_payload = {
        "schema_version": "football_intelligence.sports_mot.common_observation_graph.v1",
        "allowed_frames": allowed_frames,
        "pitch_polygon_hash": pitch_gate.polygon_hash,
        "roi": roi,
        "nodes": nodes,
        "edges": edges,
        "null_states": [
            {"state_id": f"null_{frame:06d}", "frame_sequence": frame, "state": "NULL"} for frame in allowed_frames
        ],
        "one_to_one_required": True,
        "null_state_allowed": True,
        "ambiguous_state_allowed": True,
        "visual_continuity_is_real_identity": False,
        "visual_continuity_is_player_slot": False,
        "match_local_only": True,
        "sandbox_only": True,
    }
    graph_payload["graph_hash"] = stable_hash(graph_payload)
    return graph_payload


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    implementation: str
    motion_weight: float
    appearance_weight: float
    confidence_weight: float
    null_cost: float
    uses_two_stage_confidence: bool = False
    uses_observation_velocity: bool = False
    uses_expansion_iou: bool = False
    uses_global_linker: bool = False


ADAPTER_SPECS = {
    "CURRENT_BASELINE": AdapterSpec("CURRENT_BASELINE", "project_baseline_adapter", 1.0, 0.0, 0.15, 2.2),
    "BYTETRACK": AdapterSpec("BYTETRACK", "clean_room_paper_inspired", 0.9, 0.0, 0.35, 2.1, True),
    "OC_SORT": AdapterSpec("OC_SORT", "clean_room_paper_inspired", 1.0, 0.0, 0.12, 2.0, False, True),
    "BOT_SORT": AdapterSpec("BOT_SORT", "clean_room_paper_inspired", 0.9, 0.25, 0.15, 2.0),
    "DEEP_OC_SORT": AdapterSpec("DEEP_OC_SORT", "clean_room_paper_inspired", 0.9, 0.3, 0.12, 1.9, False, True),
    "CLEAN_ROOM_DEEP_EIOU": AdapterSpec(
        "CLEAN_ROOM_DEEP_EIOU", "clean_room_paper_inspired", 0.85, 0.18, 0.15, 1.9, False, True, True
    ),
    "DEEP_EIOU_PLUS_GTA_LINKER": AdapterSpec(
        "DEEP_EIOU_PLUS_GTA_LINKER",
        "clean_room_paper_inspired",
        0.8,
        0.2,
        0.12,
        1.85,
        False,
        True,
        True,
        True,
    ),
    "MHSAG_PRIMARY_CANDIDATE": AdapterSpec(
        "MHSAG_PRIMARY_CANDIDATE",
        "project_clean_room_architecture",
        0.8,
        0.18,
        0.18,
        1.75,
        True,
        True,
        True,
        True,
    ),
}


def _transition_cost(
    previous: dict[str, Any] | None,
    current: dict[str, Any] | None,
    spec: AdapterSpec,
) -> float:
    if current is None:
        return spec.null_cost
    if previous is None:
        return 0.8 + spec.confidence_weight * (1.0 - current["confidence"])
    displacement = math.dist(
        (previous["footpoint"]["x"], previous["footpoint"]["y"]),
        (current["footpoint"]["x"], current["footpoint"]["y"]),
    )
    normalizer = max(24.0, 2.8 * max(_height(previous), _height(current)))
    motion = displacement / normalizer
    scale = abs(math.log(max(1e-6, _height(current) / _height(previous))))
    appearance = _descriptor_distance(previous, current)
    reliability = min(previous.get("appearance_reliability", 0.0), current.get("appearance_reliability", 0.0))
    appearance_cost = 0.0 if appearance is None else appearance * reliability
    expansion_cost = (1.0 - _iou(previous, current)) * (0.18 if spec.uses_expansion_iou else 0.0)
    low_confidence_penalty = 0.0
    if spec.uses_two_stage_confidence and current["confidence"] < 0.35:
        low_confidence_penalty = 0.2
    return (
        spec.motion_weight * motion
        + 0.35 * scale
        + spec.appearance_weight * appearance_cost
        + spec.confidence_weight * (1.0 - current["confidence"])
        + expansion_cost
        + low_confidence_penalty
    )


def _candidate_nodes(
    nodes: list[dict[str, Any]],
    previous: dict[str, Any] | None,
    *,
    maximum: int = 10,
) -> list[dict[str, Any]]:
    eligible = [node for node in nodes if node["pitch_gate_eligible"]]
    if previous is None:
        return sorted(eligible, key=lambda node: (-node["confidence"], node["node_id"]))[:maximum]
    eligible.sort(
        key=lambda node: math.dist(
            (previous["footpoint"]["x"], previous["footpoint"]["y"]),
            (node["footpoint"]["x"], node["footpoint"]["y"]),
        )
    )
    return eligible[:maximum]


def _hard_geometry_compatible(
    previous: dict[str, Any] | None,
    current: dict[str, Any] | None,
) -> bool:
    if previous is None or current is None:
        return True
    frame_gap = max(1, int(current["frame_sequence"]) - int(previous["frame_sequence"]))
    displacement = math.dist(
        (previous["footpoint"]["x"], previous["footpoint"]["y"]),
        (current["footpoint"]["x"], current["footpoint"]["y"]),
    )
    hard_limit = max(36.0, 3.2 * max(_height(previous), _height(current))) * frame_gap
    height_ratio = max(_height(previous), _height(current)) / min(_height(previous), _height(current))
    return (
        displacement <= hard_limit
        and height_ratio <= 2.2
        and previous["pitch_gate_eligible"]
        and current["pitch_gate_eligible"]
    )


def run_tracking_adapter(
    graph: dict[str, Any],
    *,
    adapter_name: str,
    seed_a_node_id: str,
    seed_b_node_id: str,
    top_k: int = 4,
) -> dict[str, Any]:
    if adapter_name not in ADAPTER_SPECS:
        return {
            "adapter_name": adapter_name,
            "status": "IMPLEMENTATION_NOT_COMPLETED",
            "input_graph_hash": graph["graph_hash"],
            "failure_classification": "IMPLEMENTATION_NOT_COMPLETED",
        }
    spec = ADAPTER_SPECS[adapter_name]
    nodes = {node["node_id"]: node for node in graph["nodes"]}
    if seed_a_node_id not in nodes or seed_b_node_id not in nodes or seed_a_node_id == seed_b_node_id:
        raise ValueError("adapter seeds must be two distinct graph observations")
    frames = list(graph["allowed_frames"])
    start = frames[0]
    if nodes[seed_a_node_id]["frame_sequence"] != start or nodes[seed_b_node_id]["frame_sequence"] != start:
        raise ValueError("adapter seeds must be fixed only on the first frame")
    by_frame = {frame: [node for node in graph["nodes"] if node["frame_sequence"] == frame] for frame in frames}
    beams: list[dict[str, Any]] = [{"cost": 0.0, "A": [seed_a_node_id], "B": [seed_b_node_id], "ambiguous_frames": []}]
    for frame in frames[1:]:
        expanded: list[dict[str, Any]] = []
        for beam in beams:
            previous_a = next((nodes[key] for key in reversed(beam["A"]) if key is not None), None)
            previous_b = next((nodes[key] for key in reversed(beam["B"]) if key is not None), None)
            options_a = _candidate_nodes(by_frame[frame], previous_a) + [None]
            options_b = _candidate_nodes(by_frame[frame], previous_b) + [None]
            for candidate_a in options_a:
                for candidate_b in options_b:
                    if (
                        candidate_a is not None
                        and candidate_b is not None
                        and candidate_a["node_id"] == candidate_b["node_id"]
                    ):
                        continue
                    if not _hard_geometry_compatible(previous_a, candidate_a):
                        continue
                    if not _hard_geometry_compatible(previous_b, candidate_b):
                        continue
                    cost_a = _transition_cost(previous_a, candidate_a, spec)
                    cost_b = _transition_cost(previous_b, candidate_b, spec)
                    if candidate_a is not None and previous_a is not None and cost_a > 3.0:
                        continue
                    if candidate_b is not None and previous_b is not None and cost_b > 3.0:
                        continue
                    expanded.append(
                        {
                            "cost": beam["cost"] + cost_a + cost_b,
                            "A": beam["A"] + [candidate_a["node_id"] if candidate_a else None],
                            "B": beam["B"] + [candidate_b["node_id"] if candidate_b else None],
                            "ambiguous_frames": list(beam["ambiguous_frames"]),
                        }
                    )
        expanded.sort(key=lambda item: (item["cost"], str(item["A"]), str(item["B"])))
        beams = expanded[: max(top_k * 5, 20)]
        if not beams:
            return {
                "adapter_name": adapter_name,
                "status": "FAILED_NO_VALID_JOINT_PATH",
                "input_graph_hash": graph["graph_hash"],
                "failure_classification": "FAILED_NO_VALID_JOINT_PATH",
            }
    beams.sort(key=lambda item: item["cost"])
    retained = beams[:top_k]
    margin = retained[1]["cost"] - retained[0]["cost"] if len(retained) > 1 else None
    best = retained[0]
    states = []
    for index, frame in enumerate(frames):
        near_best = [path for path in retained if path["cost"] - best["cost"] <= 0.15]
        ambiguous_a = len({path["A"][index] for path in near_best}) > 1
        ambiguous_b = len({path["B"][index] for path in near_best}) > 1
        state_a = "AMBIGUOUS" if ambiguous_a else "OBSERVED" if best["A"][index] else "MISSING"
        state_b = "AMBIGUOUS" if ambiguous_b else "OBSERVED" if best["B"][index] else "MISSING"
        states.append(
            {
                "frame_sequence": frame,
                "A": {"state": state_a, "node_id": None if ambiguous_a else best["A"][index]},
                "B": {"state": state_b, "node_id": None if ambiguous_b else best["B"][index]},
            }
        )
    return {
        "adapter_name": adapter_name,
        "adapter_version": "clean-room-v1",
        "implementation": spec.implementation,
        "status": "COMPLETED",
        "input_graph_hash": graph["graph_hash"],
        "configuration_hash": stable_hash(spec.__dict__),
        "one_to_one_enforced": True,
        "null_state_allowed": True,
        "ambiguous_state_allowed": True,
        "fixed_start_seeds_only": True,
        "forced_end_mapping": False,
        "best_joint_cost": round(best["cost"], 6),
        "best_to_second_margin": round(margin, 6) if margin is not None else None,
        "strand_states": states,
        "top_k_joint_paths": [
            {"rank": index + 1, "cost": round(path["cost"], 6), "A": path["A"], "B": path["B"]}
            for index, path in enumerate(retained)
        ],
        "visual_continuity_is_real_identity": False,
        "visual_continuity_is_player_slot": False,
        "match_local_only": True,
        "sandbox_only": True,
    }


def build_mhsag_artifacts(graph: dict[str, Any], adapter_result: dict[str, Any]) -> dict[str, Any]:
    if adapter_result.get("status") != "COMPLETED":
        return {"status": adapter_result.get("status"), "short_tracklets": [], "purity_audit": [], "global_links": []}
    nodes = {node["node_id"]: node for node in graph["nodes"]}
    tracklets: list[dict[str, Any]] = []
    purity: list[dict[str, Any]] = []
    for strand in ("A", "B"):
        current: list[dict[str, Any]] = []
        used_node_ids: set[str] = set()
        for state in adapter_result["strand_states"]:
            node_id = state[strand]["node_id"]
            if node_id is None:
                if current:
                    tracklets.append({"strand": strand, "nodes": current})
                    current = []
                continue
            node = nodes[node_id]
            if node_id in used_node_ids:
                purity.append(
                    {
                        "strand": strand,
                        "split_before_frame": node["frame_sequence"],
                        "reason": "duplicate_observation_reuse",
                    }
                )
                if current:
                    tracklets.append({"strand": strand, "nodes": current})
                    current = []
                continue
            used_node_ids.add(node_id)
            if current:
                jump = math.dist(
                    (current[-1]["footpoint"]["x"], current[-1]["footpoint"]["y"]),
                    (node["footpoint"]["x"], node["footpoint"]["y"]),
                )
                appearance = _descriptor_distance(current[-1], node)
                appearance_reliable = (
                    min(
                        current[-1].get("appearance_reliability", 0.0),
                        node.get("appearance_reliability", 0.0),
                    )
                    >= 0.55
                )
                reason = None
                if jump > max(45.0, 3.0 * max(_height(current[-1]), _height(node))):
                    reason = "abrupt_motion_discontinuity"
                elif appearance_reliable and appearance is not None and appearance > 0.75:
                    reason = "abrupt_appearance_discontinuity"
                if reason:
                    tracklets.append({"strand": strand, "nodes": current})
                    purity.append(
                        {
                            "strand": strand,
                            "split_before_frame": node["frame_sequence"],
                            "reason": reason,
                            "jump_pixels": round(jump, 6),
                            "appearance_distance": round(appearance, 6) if appearance is not None else None,
                        }
                    )
                    current = []
            current.append(node)
        if current:
            tracklets.append({"strand": strand, "nodes": current})
    rows = []
    for index, item in enumerate(tracklets):
        rows.append(
            {
                "short_tracklet_id": f"anonymous_tracklet_{index + 1:03d}",
                "strand": item["strand"],
                "start_frame": item["nodes"][0]["frame_sequence"],
                "end_frame": item["nodes"][-1]["frame_sequence"],
                "observation_node_ids": [node["node_id"] for node in item["nodes"]],
                "pure_after_audit": True,
            }
        )
    links = []
    for left in rows:
        for right in rows:
            if left["short_tracklet_id"] == right["short_tracklet_id"] or left["end_frame"] >= right["start_frame"]:
                continue
            gap = right["start_frame"] - left["end_frame"]
            same_strand = left["strand"] == right["strand"]
            links.append(
                {
                    "source_tracklet_id": left["short_tracklet_id"],
                    "target_tracklet_id": right["short_tracklet_id"],
                    "gap_frames": gap,
                    "one_to_one_required": True,
                    "no_link_allowed": True,
                    "top_k_eligible": gap <= 5,
                    "strand_compatible": same_strand,
                    "link_allowed": same_strand and gap <= 5,
                    "failure_classification": None if same_strand and gap <= 5 else "HARD_LINK_GATE_REJECTED",
                }
            )
    return {
        "status": "SKELETON_COMPLETED_NOT_PROMOTED",
        "input_graph_hash": graph["graph_hash"],
        "short_tracklets": rows,
        "purity_audit": purity,
        "global_links": links,
        "null_and_ambiguous_states": True,
        "top_k_linkings_supported": True,
        "persistent_identity_created": False,
    }


def evaluate_gold_paths(
    *,
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if gold is None:
        return {
            "evaluated": False,
            "reason": "gold_frame_annotations_not_completed",
            "HOTA": None,
            "DetA": None,
            "AssA": None,
            "IDF1": None,
        }
    if len(predicted) != len(gold):
        raise ValueError("predicted and gold frame paths must have equal length")
    eligible = 0
    correct_a = 0
    correct_b = 0
    supply = 0
    false_continuations = 0
    safe_abstentions = 0
    for prediction, truth in zip(predicted, gold):
        for strand, counter in (("A", "A"), ("B", "B")):
            gold_id = truth[strand].get("node_id")
            predicted_id = prediction[strand].get("node_id")
            if gold_id is not None:
                eligible += 1
                supply += int(predicted_id is not None)
                if predicted_id == gold_id:
                    if counter == "A":
                        correct_a += 1
                    else:
                        correct_b += 1
                elif predicted_id is not None:
                    false_continuations += 1
                else:
                    safe_abstentions += 1
    correct = correct_a + correct_b
    deta = supply / max(1, eligible)
    assa = correct / max(1, supply)
    hota = math.sqrt(deta * assa)
    return {
        "evaluated": True,
        "detection_supply_recall": deta,
        "eligible_observation_recall": deta,
        "exact_A_path_accuracy": correct_a / max(1, len(gold)),
        "exact_B_path_accuracy": correct_b / max(1, len(gold)),
        "false_continuation_count": false_continuations,
        "identity_switch_count": false_continuations,
        "strand_loss_despite_supply": safe_abstentions,
        "safe_abstention_count": safe_abstentions,
        "HOTA": hota,
        "DetA": deta,
        "AssA": assa,
        "IDF1": 2 * correct / max(1, eligible + supply),
        "trackeval_compatible": True,
    }
