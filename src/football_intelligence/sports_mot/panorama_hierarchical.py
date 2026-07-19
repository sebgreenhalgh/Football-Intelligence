"""Panorama-wide, match-local hierarchical association research primitives.

The module is intentionally bounded to short anonymous A/B strands. It keeps
all geometry in canonical panorama pixels, rejects sealed-holdout inputs, and
does not create persistent identity or any football metric.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.sports_mot.architecture import PitchParticipantGate


PANORAMA_VISIBILITY_STATES = {
    "INSIDE_FOCAL_ROI",
    "OUTSIDE_FOCAL_ROI_VISIBLE_IN_PANORAMA",
    "OUTSIDE_FOCAL_ROI_NOT_VISIBLE_IN_PANORAMA",
    "VISIBLE_IN_PANORAMA_NO_VALID_DETECTION",
    "PANORAMA_VISIBILITY_AMBIGUOUS",
}

MOTION_HYPOTHESES = (
    "OBSERVATION_CENTRIC_CONSTANT_VELOCITY",
    "CONSTANT_ACCELERATION",
    "ROBUST_SMOOTHED_PATH",
    "HEIGHT_ADAPTIVE_SEARCH",
    "BIDIRECTIONAL_SHORT_GAP_INTERPOLATION",
)


class HoldoutSealViolation(RuntimeError):
    """Raised before a development process can consume holdout material."""


@dataclass(frozen=True)
class DevelopmentSealGuard:
    """Permit only explicitly public diagnostic/development sequence IDs."""

    public_sequence_ids: frozenset[str]
    forbidden_split: str = "sealed_holdout"

    def require_public(self, *, sequence_id: str, split: str | None = None) -> None:
        if split == self.forbidden_split or sequence_id not in self.public_sequence_ids:
            raise HoldoutSealViolation(f"development-only guard rejected sequence: {sequence_id}")

    def filter_public_rows(self, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for row in rows:
            sequence_id = str(row.get("sequence_id", ""))
            if sequence_id not in self.public_sequence_ids:
                continue
            self.require_public(sequence_id=sequence_id, split=row.get("split"))
            output.append(row)
        return output

    def audit(self) -> dict[str, Any]:
        return {
            "schema_version": "football_intelligence.m5_5f1c.holdout_seal_guard.v1",
            "allowed_split_values": ["diagnostic", "development"],
            "public_sequence_count": len(self.public_sequence_ids),
            "holdout_labels_opened": False,
            "holdout_visual_evidence_opened": False,
            "holdout_unseal_count": 0,
            "guard_active": True,
        }


@dataclass(frozen=True)
class PMHSAGConfig:
    name: str = "panorama_balanced"
    maximum_candidates: int = 12
    top_k: int = 8
    beam_width: int = 96
    hard_gate_height_multiplier: float = 2.65
    maximum_scale_ratio: float = 1.85
    motion_weight: float = 0.72
    appearance_weight: float = 0.86
    colour_weight: float = 0.12
    confidence_weight: float = 0.05
    scale_weight: float = 0.18
    distractor_weight: float = 0.12
    no_link_cost: float = 1.35
    ambiguity_margin: float = 0.015
    micro_link_threshold: float = 1.05
    split_motion_threshold: float = 1.25
    split_appearance_threshold: float = 0.30
    global_link_weight: float = 0.08
    handoff_boundary_heights: float = 1.5
    low_confidence_minimum: float = 0.10
    maximum_gap_frames: int = 3
    panorama_handoff: bool = True
    purity_splitting: bool = True
    global_linking: bool = True
    global_linking_applied: bool = True
    yolo_backbone_appearance: bool = True
    distractor_negatives: bool = True
    multi_motion_bank: bool = True
    top_k_ambiguity: bool = True

    @property
    def configuration_hash(self) -> str:
        return stable_hash(asdict(self))


def _bbox(value: dict[str, Any]) -> dict[str, float]:
    box = value.get("bbox", value)
    return {key: float(box[key]) for key in ("x1", "y1", "x2", "y2")}


def _foot(value: dict[str, Any]) -> tuple[float, float]:
    footpoint = value.get("footpoint")
    if isinstance(footpoint, dict):
        return (float(footpoint["x"]), float(footpoint["y"]))
    box = _bbox(value)
    return ((box["x1"] + box["x2"]) / 2.0, box["y2"])


def _height(value: dict[str, Any]) -> float:
    box = _bbox(value)
    return max(1.0, box["y2"] - box["y1"])


def _area(box: dict[str, float]) -> float:
    return max(0.0, box["x2"] - box["x1"]) * max(0.0, box["y2"] - box["y1"])


def bbox_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    a, b = _bbox(left), _bbox(right)
    intersection = {
        "x1": max(a["x1"], b["x1"]),
        "y1": max(a["y1"], b["y1"]),
        "x2": min(a["x2"], b["x2"]),
        "y2": min(a["y2"], b["y2"]),
    }
    overlap = _area(intersection)
    return overlap / max(1.0, _area(a) + _area(b) - overlap)


def _vector_distance(left: Any, right: Any) -> float | None:
    if not isinstance(left, list) or not isinstance(right, list) or not left or len(left) != len(right):
        return None
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _cosine_distance(left: Any, right: Any) -> float | None:
    if not isinstance(left, list) or not isinstance(right, list) or not left or len(left) != len(right):
        return None
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    return max(0.0, min(2.0, 1.0 - dot / (left_norm * right_norm)))


def _appearance_vector(row: dict[str, Any]) -> Any:
    return row.get("yolo_backbone_compact_descriptor") or row.get("yolo_backbone_descriptor")


def _inside_roi(footpoint: tuple[float, float], roi: dict[str, float]) -> bool:
    return roi["x1"] <= footpoint[0] <= roi["x2"] and roi["y1"] <= footpoint[1] <= roi["y2"]


def _source_family(row: dict[str, Any]) -> str:
    layer = str(row.get("source_layer") or row.get("source_type") or "unknown").lower()
    if "gpu_local" in layer or "1536" in layer or "2048" in layer:
        return "local_multiscale"
    if "canonical" in layer or row.get("candidate_id"):
        return "canonical"
    return layer


def _same_cross_scale_observation(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if int(left["frame_sequence"]) != int(right["frame_sequence"]):
        return False
    if left.get("candidate_id") and left.get("candidate_id") == right.get("candidate_id"):
        return True
    if left.get("source_row_hash") and left.get("source_row_hash") == right.get("source_row_hash"):
        return True
    iou = bbox_iou(left, right)
    if iou >= 0.62:
        return True
    if {_source_family(left), _source_family(right)} != {"canonical", "local_multiscale"}:
        return False
    scale_ratio = max(_height(left), _height(right)) / min(_height(left), _height(right))
    foot_distance = math.dist(_foot(left), _foot(right)) / max(_height(left), _height(right))
    colour = _vector_distance(left.get("colour_descriptor"), right.get("colour_descriptor"))
    return iou >= 0.12 and scale_ratio <= 1.5 and foot_distance <= 0.42 and (colour is None or colour <= 0.26)


def consolidate_cross_crop_observations(
    observations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """Unify only defensible same-frame cross-scale duplicates."""
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source in observations:
        row = dict(source)
        row["bbox"] = _bbox(row)
        row["observation_id"] = str(
            row.get("observation_id")
            or row.get("consolidated_observation_id")
            or f"panorama_{row.get('candidate_id') or stable_hash(row)[:20]}"
        )
        by_frame[int(row["frame_sequence"])].append(row)

    consolidated: list[dict[str, Any]] = []
    alias_to_node: dict[str, str] = {}
    clusters: list[dict[str, Any]] = []
    for frame, frame_rows in sorted(by_frame.items()):
        parents = list(range(len(frame_rows)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            a, b = find(left), find(right)
            if a != b:
                parents[b] = a

        for left in range(len(frame_rows)):
            for right in range(left + 1, len(frame_rows)):
                if _same_cross_scale_observation(frame_rows[left], frame_rows[right]):
                    union(left, right)
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for index, row in enumerate(frame_rows):
            grouped[find(index)].append(row)
        for members in grouped.values():
            members.sort(
                key=lambda row: (
                    0 if _source_family(row) == "canonical" else 1,
                    0 if row.get("candidate_id") else 1,
                    -float(row.get("confidence", 0.0)),
                    row["observation_id"],
                )
            )
            representative = dict(members[0])
            aliases = sorted(
                {
                    alias
                    for row in members
                    for alias in [str(row["observation_id"]), *map(str, row.get("observation_aliases", []))]
                }
            )
            candidate_aliases = sorted(
                {
                    alias
                    for row in members
                    for alias in [
                        *([str(row["candidate_id"])] if row.get("candidate_id") else []),
                        *map(str, row.get("candidate_aliases", [])),
                    ]
                }
            )
            representative["observation_aliases"] = aliases
            representative["candidate_aliases"] = candidate_aliases
            representative["provenance_cluster_hash"] = stable_hash(
                {
                    "frame_sequence": frame,
                    "source_row_hashes": sorted(str(row.get("source_row_hash", "")) for row in members),
                    "aliases": aliases,
                }
            )
            representative["cross_crop_duplicate_count"] = len(members) - 1
            consolidated.append(representative)
            for alias in aliases:
                alias_to_node[alias] = representative["observation_id"]
            for alias in candidate_aliases:
                alias_to_node[alias] = representative["observation_id"]
            clusters.append(
                {
                    "frame_sequence": frame,
                    "representative_node_id": representative["observation_id"],
                    "member_node_ids": aliases,
                    "candidate_aliases": candidate_aliases,
                    "member_count": len(members),
                    "cross_crop_deduplicated": len(members) > 1,
                    "provenance_cluster_hash": representative["provenance_cluster_hash"],
                }
            )
    consolidated.sort(key=lambda row: (int(row["frame_sequence"]), row["observation_id"]))
    return consolidated, alias_to_node, clusters


def build_panorama_observation_graph(
    observations: list[dict[str, Any]],
    *,
    pitch_gate: PitchParticipantGate,
    allowed_frames: list[int],
    focal_roi: dict[str, float],
    sequence_id: str,
    split: str,
    seal_guard: DevelopmentSealGuard,
) -> dict[str, Any]:
    """Build an immutable public graph without filtering at the focal ROI."""
    seal_guard.require_public(sequence_id=sequence_id, split=split)
    allowed = set(int(frame) for frame in allowed_frames)
    consolidated, aliases, clusters = consolidate_cross_crop_observations(
        [row for row in observations if int(row["frame_sequence"]) in allowed]
    )
    nodes = []
    for source in consolidated:
        footpoint = _foot(source)
        gate = pitch_gate.classify(footpoint)
        node = {
            **source,
            "node_id": str(source["observation_id"]),
            "frame_sequence": int(source["frame_sequence"]),
            "bbox": _bbox(source),
            "footpoint": gate["footpoint"],
            "confidence": float(source.get("confidence", 0.0)),
            "source_layer": str(source.get("source_layer") or source.get("source_type") or "unknown"),
            "source_row_hash": str(source.get("source_row_hash") or stable_hash(source)),
            "coordinate_space": "canonical_panorama_pixels",
            "focal_roi_membership": "INSIDE_FOCAL_ROI" if _inside_roi(footpoint, focal_roi) else "OUTSIDE_FOCAL_ROI",
            "pitch_zone": gate["zone"],
            "pitch_gate_eligible": gate["primary_benchmark_eligible"],
            "appearance_reliability": float(source.get("appearance_reliability", 0.0)),
        }
        nodes.append(node)
    nodes.sort(key=lambda row: (row["frame_sequence"], row["node_id"]))
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_frame[node["frame_sequence"]].append(node)
    edges = []
    for left_index, left_frame in enumerate(allowed_frames):
        for right_frame in allowed_frames[left_index + 1 : left_index + 4]:
            gap = int(right_frame) - int(left_frame)
            for left in by_frame[int(left_frame)]:
                for right in by_frame[int(right_frame)]:
                    displacement = math.dist(_foot(left), _foot(right))
                    scale_ratio = max(_height(left), _height(right)) / min(_height(left), _height(right))
                    hard_limit = max(34.0, 3.1 * max(_height(left), _height(right))) * max(1, gap)
                    hard_gate = (
                        displacement <= hard_limit
                        and scale_ratio <= 2.1
                        and left["pitch_gate_eligible"]
                        and right["pitch_gate_eligible"]
                    )
                    edges.append(
                        {
                            "edge_id": stable_hash({"left": left["node_id"], "right": right["node_id"]}),
                            "source_node_id": left["node_id"],
                            "target_node_id": right["node_id"],
                            "source_frame_sequence": int(left_frame),
                            "target_frame_sequence": int(right_frame),
                            "gap_frames": gap,
                            "footpoint_displacement_pixels": round(displacement, 6),
                            "scale_ratio": round(scale_ratio, 6),
                            "bbox_iou": round(bbox_iou(left, right), 6),
                            "colour_distance": _vector_distance(
                                left.get("panorama_colour_descriptor") or left.get("colour_descriptor"),
                                right.get("panorama_colour_descriptor") or right.get("colour_descriptor"),
                            ),
                            "backbone_distance": _cosine_distance(_appearance_vector(left), _appearance_vector(right)),
                            "hard_gate_pass": hard_gate,
                        }
                    )
    payload = {
        "schema_version": "football_intelligence.m5_5f1c.panorama_observation_graph.v1",
        "sequence_id": sequence_id,
        "split": split,
        "allowed_frames": [int(frame) for frame in allowed_frames],
        "coordinate_space": "canonical_panorama_pixels",
        "focal_roi": focal_roi,
        "focal_roi_is_eligibility_gate": False,
        "pitch_polygon_hash": pitch_gate.polygon_hash,
        "nodes": nodes,
        "edges": edges,
        "alias_to_node": aliases,
        "provenance_clusters": clusters,
        "null_states": [
            {"state_id": f"null_{frame:06d}", "frame_sequence": frame, "state": "NULL"} for frame in allowed_frames
        ],
        "ambiguous_states": [
            {"state_id": f"ambiguous_{frame:06d}", "frame_sequence": frame, "state": "AMBIGUOUS"}
            for frame in allowed_frames
        ],
        "one_to_one_required": True,
        "match_local_only": True,
        "sandbox_only": True,
        "persistent_identity_created": False,
        "player_slots_created": False,
    }
    payload["graph_hash"] = stable_hash(payload)
    return payload


def extract_yolo_backbone_descriptors(
    observations: list[dict[str, Any]],
    *,
    frame_files: dict[int, Path],
    checkpoint: Path,
    required_checkpoint_sha256: str,
    batch_size: int = 32,
    image_size: int = 160,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cache ROI-pooled YOLOv8m backbone and colour descriptors on CUDA."""
    if sha256_file(checkpoint) != required_checkpoint_sha256:
        raise RuntimeError("approved YOLOv8m checkpoint hash mismatch")
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; silent CPU fallback is prohibited")
    model = YOLO(str(checkpoint))
    output = [dict(row) for row in observations]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def crop_for(row: dict[str, Any]) -> Any:
        image = cv2.imread(str(frame_files[int(row["frame_sequence"])]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot decode frame for descriptor: {row['frame_sequence']}")
        box = _bbox(row)
        pad_x = max(3, int((box["x2"] - box["x1"]) * 0.16))
        pad_y = max(3, int((box["y2"] - box["y1"]) * 0.08))
        x1, y1 = max(0, int(math.floor(box["x1"])) - pad_x), max(0, int(math.floor(box["y1"])) - pad_y)
        x2 = min(image.shape[1], int(math.ceil(box["x2"])) + pad_x)
        y2 = min(image.shape[0], int(math.ceil(box["y2"])) + pad_y)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise RuntimeError(f"empty descriptor crop: {row.get('observation_id')}")
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist_parts = []
        for channel, span in ((0, (0, 180)), (1, (0, 256)), (2, (0, 256))):
            hist = cv2.calcHist([hsv], [channel], None, [8], list(span)).reshape(-1)
            hist = hist / max(float(hist.sum()), 1.0)
            hist_parts.extend(float(value) for value in hist)
        pixels = crop.reshape(-1, 3).astype(np.float32) / 255.0
        row["panorama_colour_descriptor"] = [
            *hist_parts,
            *[float(value) for value in pixels.mean(axis=0)],
            *[float(value) for value in pixels.std(axis=0)],
        ]
        height = _height(row)
        truncation = (
            box["x1"] <= 1 or box["y1"] <= 1 or box["x2"] >= image.shape[1] - 1 or box["y2"] >= image.shape[0] - 1
        )
        row["appearance_reliability"] = round(min(1.0, height / 46.0) * (0.55 if truncation else 1.0), 6)
        return crop

    for start in range(0, len(output), batch_size):
        rows = output[start : start + batch_size]
        crops = [crop_for(row) for row in rows]
        try:
            embeddings = model.embed(
                crops,
                imgsz=image_size,
                batch=batch_size,
                device="cuda:0",
                half=True,
                verbose=False,
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("CUDA OOM during bounded descriptor extraction; CPU fallback forbidden") from exc
        for row, embedding in zip(rows, embeddings):
            vector = torch.nn.functional.normalize(embedding.float(), dim=0).cpu().tolist()
            row["yolo_backbone_descriptor"] = [round(float(value), 7) for value in vector]
            row["yolo_backbone_descriptor_hash"] = stable_hash(row["yolo_backbone_descriptor"])
    telemetry = {
        "schema_version": "football_intelligence.m5_5f1c.yolo_backbone_descriptor_runtime.v1",
        "checkpoint_sha256": required_checkpoint_sha256,
        "device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
        "fp16": True,
        "silent_cpu_fallback": False,
        "batch_size": batch_size,
        "image_size": image_size,
        "descriptor_dimension": len(output[0]["yolo_backbone_descriptor"]) if output else 0,
        "descriptor_count": len(output),
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }
    return output, telemetry


def predict_motion_bank(
    history: list[dict[str, Any]],
    target_frame: int,
    *,
    future_anchor: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return every valid full-panorama motion hypothesis."""
    if not history:
        return []
    last = history[-1]
    last_point = _foot(last)
    target_gap = max(1, target_frame - int(last["frame_sequence"]))
    hypotheses = []
    if len(history) >= 2:
        previous = history[-2]
        frame_gap = max(1, int(last["frame_sequence"]) - int(previous["frame_sequence"]))
        velocity = tuple((a - b) / frame_gap for a, b in zip(last_point, _foot(previous)))
    else:
        velocity = (0.0, 0.0)
    cv = (last_point[0] + velocity[0] * target_gap, last_point[1] + velocity[1] * target_gap)
    hypotheses.append({"name": MOTION_HYPOTHESES[0], "point": cv, "complexity_penalty": 0.0})
    if len(history) >= 3:
        p0, p1, p2 = history[-3:]
        gap1 = max(1, int(p1["frame_sequence"]) - int(p0["frame_sequence"]))
        gap2 = max(1, int(p2["frame_sequence"]) - int(p1["frame_sequence"]))
        v1 = tuple((a - b) / gap1 for a, b in zip(_foot(p1), _foot(p0)))
        v2 = tuple((a - b) / gap2 for a, b in zip(_foot(p2), _foot(p1)))
        acceleration = (v2[0] - v1[0], v2[1] - v1[1])
        point = (
            last_point[0] + v2[0] * target_gap + 0.5 * acceleration[0] * target_gap**2,
            last_point[1] + v2[1] * target_gap + 0.5 * acceleration[1] * target_gap**2,
        )
        hypotheses.append({"name": MOTION_HYPOTHESES[1], "point": point, "complexity_penalty": 0.035})
        velocities = []
        for left, right in zip(history[:-1], history[1:]):
            gap = max(1, int(right["frame_sequence"]) - int(left["frame_sequence"]))
            velocities.append(tuple((a - b) / gap for a, b in zip(_foot(right), _foot(left))))
        smooth = (
            sorted(value[0] for value in velocities)[len(velocities) // 2],
            sorted(value[1] for value in velocities)[len(velocities) // 2],
        )
        hypotheses.append(
            {
                "name": MOTION_HYPOTHESES[2],
                "point": (last_point[0] + smooth[0] * target_gap, last_point[1] + smooth[1] * target_gap),
                "complexity_penalty": 0.02,
            }
        )
    hypotheses.append(
        {
            "name": MOTION_HYPOTHESES[3],
            "point": cv,
            "complexity_penalty": 0.01,
            "search_radius_pixels": max(28.0, 1.25 * _height(last)) * target_gap,
        }
    )
    if future_anchor is not None and int(future_anchor["frame_sequence"]) > int(last["frame_sequence"]):
        total = int(future_anchor["frame_sequence"]) - int(last["frame_sequence"])
        fraction = target_gap / total
        future = _foot(future_anchor)
        hypotheses.append(
            {
                "name": MOTION_HYPOTHESES[4],
                "point": (
                    last_point[0] + (future[0] - last_point[0]) * fraction,
                    last_point[1] + (future[1] - last_point[1]) * fraction,
                ),
                "complexity_penalty": 0.025,
            }
        )
    return hypotheses


def _best_motion_cost(
    history: list[dict[str, Any]], candidate: dict[str, Any], config: PMHSAGConfig
) -> tuple[float, dict[str, Any]]:
    hypotheses = predict_motion_bank(history, int(candidate["frame_sequence"]))
    if not config.multi_motion_bank:
        hypotheses = hypotheses[:1]
    rows = []
    for hypothesis in hypotheses:
        residual = math.dist(hypothesis["point"], _foot(candidate))
        normalized = residual / max(20.0, 1.8 * max(_height(history[-1]), _height(candidate)))
        rows.append({**hypothesis, "residual_pixels": residual, "cost": normalized + hypothesis["complexity_penalty"]})
    best = min(rows, key=lambda row: (row["cost"], row["name"]))
    return float(best["cost"]), best


def _hard_compatible(history: list[dict[str, Any]], candidate: dict[str, Any] | None, config: PMHSAGConfig) -> bool:
    if candidate is None or not history:
        return True
    previous = history[-1]
    gap = max(1, int(candidate["frame_sequence"]) - int(previous["frame_sequence"]))
    hypotheses = predict_motion_bank(history, int(candidate["frame_sequence"]))
    residual = min(math.dist(row["point"], _foot(candidate)) for row in hypotheses)
    limit = max(34.0, config.hard_gate_height_multiplier * max(_height(previous), _height(candidate))) * gap
    ratio = max(_height(previous), _height(candidate)) / min(_height(previous), _height(candidate))
    return residual <= limit and ratio <= config.maximum_scale_ratio and bool(candidate.get("pitch_gate_eligible"))


def _strict_link_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    displacement = math.dist(_foot(left), _foot(right)) / max(22.0, 1.9 * max(_height(left), _height(right)))
    scale = abs(math.log(max(1e-6, _height(right) / _height(left))))
    appearance = _cosine_distance(_appearance_vector(left), _appearance_vector(right))
    if appearance is None:
        appearance = _vector_distance(
            left.get("panorama_colour_descriptor") or left.get("colour_descriptor"),
            right.get("panorama_colour_descriptor") or right.get("colour_descriptor"),
        )
    return displacement + 0.35 * scale + 0.8 * float(appearance or 0.0)


def build_pure_microtracklets(
    graph: dict[str, Any], config: PMHSAGConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Construct strict fragments, split impurity, and expose linker hashes."""
    nodes = {str(row["node_id"]): row for row in graph["nodes"]}
    frames = [int(frame) for frame in graph["allowed_frames"]]
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes.values():
        if node.get("pitch_gate_eligible"):
            by_frame[int(node["frame_sequence"])].append(node)
    successor: dict[str, str] = {}
    predecessor: dict[str, str] = {}
    for left_frame, right_frame in zip(frames, frames[1:]):
        candidates = []
        for left in by_frame[left_frame]:
            for right in by_frame[right_frame]:
                score = _strict_link_score(left, right)
                if score <= config.micro_link_threshold:
                    candidates.append((score, left["node_id"], right["node_id"]))
        candidates.sort()
        for _, left_id, right_id in candidates:
            if left_id in successor or right_id in predecessor:
                continue
            successor[left_id] = right_id
            predecessor[right_id] = left_id
    raw = []
    visited: set[str] = set()
    for node_id in sorted(nodes):
        if node_id in visited or node_id in predecessor:
            continue
        chain = []
        current = node_id
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            current = successor.get(current)
        raw.append(chain)
    for node_id in sorted(set(nodes) - visited):
        raw.append([node_id])
    raw_rows = [
        {
            "raw_tracklet_id": f"raw_micro_{index:04d}",
            "node_ids": chain,
            "start_frame": nodes[chain[0]]["frame_sequence"],
            "end_frame": nodes[chain[-1]]["frame_sequence"],
        }
        for index, chain in enumerate(raw, start=1)
    ]
    before_hash = stable_hash(raw_rows)
    pure_rows = []
    split_rows = []
    for raw_row in raw_rows:
        chunks: list[list[str]] = [[]]
        for node_id in raw_row["node_ids"]:
            if chunks[-1]:
                previous = nodes[chunks[-1][-1]]
                current = nodes[node_id]
                motion = math.dist(_foot(previous), _foot(current)) / max(
                    20.0, 1.8 * max(_height(previous), _height(current))
                )
                appearance = _cosine_distance(_appearance_vector(previous), _appearance_vector(current))
                reasons = []
                if motion > config.split_motion_threshold:
                    reasons.append("MOTION_DISCONTINUITY")
                if appearance is not None and appearance > config.split_appearance_threshold:
                    reasons.append("APPEARANCE_DISCONTINUITY")
                if node_id in chunks[-1]:
                    reasons.append("DUPLICATE_OBSERVATION_REUSE")
                if reasons and config.purity_splitting:
                    split_rows.append(
                        {
                            "raw_tracklet_id": raw_row["raw_tracklet_id"],
                            "split_before_node_id": node_id,
                            "split_before_frame": current["frame_sequence"],
                            "reasons": reasons,
                            "normalized_motion_discontinuity": round(motion, 6),
                            "appearance_discontinuity": round(appearance, 6) if appearance is not None else None,
                        }
                    )
                    chunks.append([])
            chunks[-1].append(node_id)
        for chunk in chunks:
            pure_rows.append(
                {
                    "tracklet_id": f"pure_micro_{len(pure_rows) + 1:04d}",
                    "source_raw_tracklet_id": raw_row["raw_tracklet_id"],
                    "node_ids": chunk,
                    "start_frame": nodes[chunk[0]]["frame_sequence"],
                    "end_frame": nodes[chunk[-1]]["frame_sequence"],
                    "pure_after_audit": True,
                }
            )
    after_hash = stable_hash(pure_rows)
    audit = {
        "raw_linker_input_hash": before_hash,
        "post_purity_linker_input_hash": after_hash,
        "purity_split_count": len(split_rows),
        "split_changes_linker_graph": bool(split_rows) and before_hash != after_hash,
        "final_linker_consumes_post_purity_hash": after_hash,
    }
    return pure_rows, split_rows, audit


def build_global_link_candidates(
    graph: dict[str, Any], tracklets: list[dict[str, Any]], config: PMHSAGConfig
) -> list[dict[str, Any]]:
    nodes = {str(row["node_id"]): row for row in graph["nodes"]}
    membership = {node_id: tracklet["tracklet_id"] for tracklet in tracklets for node_id in tracklet["node_ids"]}
    rows = []
    for edge in graph["edges"]:
        if not edge["hard_gate_pass"] or int(edge["gap_frames"]) > config.maximum_gap_frames:
            continue
        left, right = nodes[edge["source_node_id"]], nodes[edge["target_node_id"]]
        motion = edge["footpoint_displacement_pixels"] / max(22.0, 2.0 * max(_height(left), _height(right)))
        appearance = edge.get("backbone_distance")
        if appearance is None:
            appearance = edge.get("colour_distance") or 0.0
        source_tracklet = membership[left["node_id"]]
        target_tracklet = membership[right["node_id"]]
        internal = source_tracklet == target_tracklet
        cost = motion + 0.55 * float(appearance) + 0.06 * max(0, int(edge["gap_frames"]) - 1)
        if not internal:
            cost += 0.035
        rows.append(
            {
                "source_node_id": left["node_id"],
                "target_node_id": right["node_id"],
                "source_tracklet_id": source_tracklet,
                "target_tracklet_id": target_tracklet,
                "gap_frames": int(edge["gap_frames"]),
                "link_type": "PURE_TRACKLET_INTERNAL" if internal else "GLOBAL_TRACKLET_DAG_LINK",
                "link_cost": round(cost, 8),
                "hard_geometry_veto": False,
                "one_to_one_required": True,
                "no_link_allowed": True,
            }
        )
    rows.sort(key=lambda row: (row["source_node_id"], row["target_node_id"]))
    return rows


def dynamic_roi_handoff_rows(
    graph: dict[str, Any], result: dict[str, Any], config: PMHSAGConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = {str(row["node_id"]): row for row in graph["nodes"]}
    roi = graph["focal_roi"]
    dynamic_rows = []
    handoffs = []
    for strand in ("A", "B"):
        history = []
        previous_membership = None
        for state in result["strand_states"]:
            frame = int(state["frame_sequence"])
            node_id = state[strand].get("node_id")
            node = nodes.get(str(node_id)) if node_id is not None else None
            hypotheses = predict_motion_bank(history, frame) if history else []
            point = hypotheses[0]["point"] if hypotheses else (_foot(node) if node else (None, None))
            height = _height(history[-1]) if history else (_height(node) if node else 32.0)
            radius = max(30.0, 1.35 * height)
            near_boundary = False
            if point[0] is not None:
                near_boundary = (
                    min(
                        abs(point[0] - roi["x1"]),
                        abs(point[0] - roi["x2"]),
                        abs(point[1] - roi["y1"]),
                        abs(point[1] - roi["y2"]),
                    )
                    <= config.handoff_boundary_heights * height
                )
            membership = node.get("focal_roi_membership") if node else "NO_OBSERVATION"
            destination = "FULL_PANORAMA" if membership == "OUTSIDE_FOCAL_ROI" or near_boundary else "FOCAL_ROI"
            dynamic_rows.append(
                {
                    "sequence_id": graph["sequence_id"],
                    "frame_sequence": frame,
                    "strand": strand,
                    "predicted_footpoint": {"x": point[0], "y": point[1]},
                    "height_adaptive_search_radius_pixels": round(radius, 6),
                    "near_focal_boundary": near_boundary,
                    "requested_view": destination,
                    "selected_node_id": node_id,
                    "selected_membership": membership,
                    "focal_exit_caused_termination": False,
                }
            )
            if previous_membership and previous_membership != membership and node is not None:
                handoffs.append(
                    {
                        "sequence_id": graph["sequence_id"],
                        "frame_sequence": frame,
                        "strand": strand,
                        "source_crop": previous_membership,
                        "destination_crop": membership,
                        "destination_view": destination,
                        "selected_observation": node_id,
                        "coordinate_transform": "identity_canonical_panorama_pixels",
                        "roundtrip_error_pixels": 0.0,
                        "handoff_confidence": round(float(node.get("confidence", 0.0)), 6),
                    }
                )
            if node is not None:
                history.append(node)
                history = history[-4:]
                previous_membership = membership
    return dynamic_rows, handoffs


def _path_history(path: list[str | None], nodes: dict[str, dict[str, Any]], maximum: int = 4) -> list[dict[str, Any]]:
    output = []
    for node_id in reversed(path):
        if node_id is not None:
            output.append(nodes[node_id])
            if len(output) == maximum:
                break
    return list(reversed(output))


def _candidate_identity_cost(
    *,
    history: list[dict[str, Any]],
    candidate: dict[str, Any],
    own_seed: dict[str, Any],
    other_seed: dict[str, Any],
    distractors: list[dict[str, Any]],
    config: PMHSAGConfig,
    static_detail: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    motion, hypothesis = _best_motion_cost(history, candidate, config)
    if static_detail is None:
        own = _cosine_distance(_appearance_vector(own_seed), _appearance_vector(candidate))
        other = _cosine_distance(_appearance_vector(other_seed), _appearance_vector(candidate))
        colour = _vector_distance(
            own_seed.get("panorama_colour_descriptor") or own_seed.get("colour_descriptor"),
            candidate.get("panorama_colour_descriptor") or candidate.get("colour_descriptor"),
        )
        own = float(own or 0.0) if config.yolo_backbone_appearance else 0.0
        colour = float(colour or 0.0)
        contrast = 0.0
        if config.distractor_negatives:
            negative_distances = [
                value
                for value in (
                    _cosine_distance(_appearance_vector(row), _appearance_vector(candidate)) for row in distractors
                )
                if value is not None
            ]
            other_distance = float(other) if other is not None else 1.0
            nearest_negative = min([other_distance, *negative_distances], default=1.0)
            contrast = max(0.0, 0.18 - nearest_negative)
        static_detail = {
            "own_backbone_distance": own,
            "other_seed_distance": other,
            "colour_distance": colour,
            "distractor_contrast_penalty": contrast,
        }
    own = float(static_detail["own_backbone_distance"])
    other = static_detail["other_seed_distance"]
    colour = float(static_detail["colour_distance"])
    contrast = float(static_detail["distractor_contrast_penalty"])
    previous = history[-1]
    recent_vectors = [_appearance_vector(row) for row in history[-3:] if isinstance(_appearance_vector(row), list)]
    recent_distance = None
    candidate_vector = _appearance_vector(candidate)
    if recent_vectors and isinstance(candidate_vector, list):
        template = [sum(values) / len(values) for values in zip(*recent_vectors)]
        recent_distance = _cosine_distance(template, candidate_vector)
        if config.yolo_backbone_appearance and recent_distance is not None:
            own = 0.35 * own + 0.65 * float(recent_distance)
    scale = abs(math.log(max(1e-6, _height(candidate) / _height(previous))))
    reliability = min(
        float(own_seed.get("appearance_reliability", 0.0)),
        float(candidate.get("appearance_reliability", 0.0)),
    )
    appearance_cost = own * reliability
    cost = (
        config.motion_weight * motion
        + config.appearance_weight * appearance_cost
        + config.colour_weight * colour
        + config.scale_weight * scale
        + config.confidence_weight * (1.0 - float(candidate.get("confidence", 0.0)))
        + config.distractor_weight * contrast
    )
    return cost, {
        "motion_hypothesis": hypothesis["name"],
        "motion_residual_pixels": hypothesis["residual_pixels"],
        **static_detail,
        "recent_positive_template_distance": recent_distance,
        "combined_positive_template_distance": own,
        "appearance_reliability": reliability,
        "total_transition_cost": cost,
    }


def run_p_mhsag(
    graph: dict[str, Any],
    *,
    seed_a_node_id: str,
    seed_b_node_id: str,
    config: PMHSAGConfig,
) -> dict[str, Any]:
    """Apply post-purity tracklet DAG output as the authoritative A/B path."""
    nodes = {str(row["node_id"]): row for row in graph["nodes"]}
    aliases = graph.get("alias_to_node", {})
    seed_a_node_id = aliases.get(seed_a_node_id, seed_a_node_id)
    seed_b_node_id = aliases.get(seed_b_node_id, seed_b_node_id)
    if seed_a_node_id not in nodes or seed_b_node_id not in nodes or seed_a_node_id == seed_b_node_id:
        raise ValueError("P-MHSAG requires two distinct public seed observations")
    frames = [int(frame) for frame in graph["allowed_frames"]]
    if nodes[seed_a_node_id]["frame_sequence"] != frames[0] or nodes[seed_b_node_id]["frame_sequence"] != frames[0]:
        raise ValueError("P-MHSAG fixes seeds only at the first frame")
    tracklets, purity_rows, purity_audit = build_pure_microtracklets(graph, config)
    links = build_global_link_candidates(graph, tracklets, config)
    link_lookup = {(row["source_node_id"], row["target_node_id"]): row for row in links}
    membership = {node_id: row["tracklet_id"] for row in tracklets for node_id in row["node_ids"]}
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes.values():
        if node.get("pitch_gate_eligible") and float(node.get("confidence", 0.0)) >= config.low_confidence_minimum:
            by_frame[int(node["frame_sequence"])].append(node)
    seed_a, seed_b = nodes[seed_a_node_id], nodes[seed_b_node_id]
    first_nodes = by_frame[frames[0]]
    distractors_a = [node for node in first_nodes if node["node_id"] not in {seed_a_node_id, seed_b_node_id}]
    distractors_b = list(distractors_a)
    static_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for strand, own_seed, other_seed, distractors in (
        ("A", seed_a, seed_b, distractors_a),
        ("B", seed_b, seed_a, distractors_b),
    ):
        for candidate in nodes.values():
            own = _cosine_distance(_appearance_vector(own_seed), _appearance_vector(candidate))
            other = _cosine_distance(_appearance_vector(other_seed), _appearance_vector(candidate))
            colour = _vector_distance(
                own_seed.get("panorama_colour_descriptor") or own_seed.get("colour_descriptor"),
                candidate.get("panorama_colour_descriptor") or candidate.get("colour_descriptor"),
            )
            negative_distances = [
                value
                for value in (
                    _cosine_distance(_appearance_vector(row), _appearance_vector(candidate)) for row in distractors
                )
                if value is not None
            ]
            nearest_negative = min([float(other) if other is not None else 1.0, *negative_distances], default=1.0)
            static_identity[(strand, candidate["node_id"])] = {
                "own_backbone_distance": float(own or 0.0) if config.yolo_backbone_appearance else 0.0,
                "other_seed_distance": other,
                "colour_distance": float(colour or 0.0),
                "distractor_contrast_penalty": (
                    max(0.0, 0.18 - nearest_negative) if config.distractor_negatives else 0.0
                ),
            }
    beams = [{"cost": 0.0, "A": [seed_a_node_id], "B": [seed_b_node_id], "cost_rows": []}]
    started = time.perf_counter()
    for frame in frames[1:]:
        expanded = []
        for beam in beams:
            histories = {strand: _path_history(beam[strand], nodes) for strand in ("A", "B")}
            option_rows: dict[str, list[tuple[dict[str, Any] | None, float, dict[str, Any]]]] = {}
            for strand, own_seed, other_seed, distractors in (
                ("A", seed_a, seed_b, distractors_a),
                ("B", seed_b, seed_a, distractors_b),
            ):
                history = histories[strand]
                candidates = []
                for candidate in by_frame[frame]:
                    if not _hard_compatible(history, candidate, config):
                        continue
                    previous = history[-1]
                    link = link_lookup.get((previous["node_id"], candidate["node_id"]))
                    if config.global_linking and config.global_linking_applied and link is None:
                        continue
                    cost, detail = _candidate_identity_cost(
                        history=history,
                        candidate=candidate,
                        own_seed=own_seed,
                        other_seed=other_seed,
                        distractors=distractors,
                        config=config,
                        static_detail=static_identity[(strand, candidate["node_id"])],
                    )
                    if config.global_linking and config.global_linking_applied and link is not None:
                        cost += config.global_link_weight * float(link["link_cost"])
                        detail["tracklet_link_type"] = link["link_type"]
                        detail["global_link_applied"] = True
                    else:
                        detail["global_link_applied"] = False
                    detail["source_tracklet_id"] = membership.get(previous["node_id"])
                    detail["target_tracklet_id"] = membership.get(candidate["node_id"])
                    candidates.append((candidate, cost, detail))
                candidates.sort(key=lambda row: (row[1], -float(row[0].get("confidence", 0.0)), row[0]["node_id"]))
                candidates = candidates[: config.maximum_candidates]
                candidates.append(
                    (None, config.no_link_cost, {"state": "NULL", "total_transition_cost": config.no_link_cost})
                )
                option_rows[strand] = candidates
            for candidate_a, cost_a, detail_a in option_rows["A"]:
                for candidate_b, cost_b, detail_b in option_rows["B"]:
                    if (
                        candidate_a is not None
                        and candidate_b is not None
                        and candidate_a["node_id"] == candidate_b["node_id"]
                    ):
                        continue
                    expanded.append(
                        {
                            "cost": beam["cost"] + cost_a + cost_b,
                            "A": [*beam["A"], candidate_a["node_id"] if candidate_a else None],
                            "B": [*beam["B"], candidate_b["node_id"] if candidate_b else None],
                            "cost_rows": [
                                *beam["cost_rows"],
                                {
                                    "frame_sequence": frame,
                                    "A": detail_a,
                                    "B": detail_b,
                                },
                            ],
                        }
                    )
        expanded.sort(key=lambda row: (row["cost"], str(row["A"]), str(row["B"])))
        beams = expanded[: config.beam_width]
        if not beams:
            raise RuntimeError(f"P-MHSAG tracklet DAG has no valid joint path at frame {frame}")
    retained = beams[: config.top_k]
    best = retained[0]
    states = []
    for index, frame in enumerate(frames):
        near = [row for row in retained if row["cost"] - best["cost"] <= config.ambiguity_margin]
        state = {"frame_sequence": frame}
        for strand in ("A", "B"):
            alternatives = {row[strand][index] for row in near}
            ambiguous = config.top_k_ambiguity and len(alternatives) > 1
            node_id = None if ambiguous else best[strand][index]
            state[strand] = {
                "state": "AMBIGUOUS" if ambiguous else "OBSERVED" if node_id else "MISSING",
                "node_id": node_id,
                "tracklet_id": membership.get(node_id) if node_id else None,
                "alternative_count": len(alternatives),
            }
        states.append(state)
    result = {
        "schema_version": "football_intelligence.m5_5f1c.p_mhsag_result.v1",
        "algorithm": "P_MHSAG",
        "configuration": asdict(config),
        "configuration_hash": config.configuration_hash,
        "input_graph_hash": graph["graph_hash"],
        "post_purity_linker_input_hash": purity_audit["post_purity_linker_input_hash"],
        "authoritative_path_source": "POST_PURITY_JOINT_TRACKLET_DAG",
        "global_linking_computed": True,
        "global_linking_applied_to_final_path": bool(config.global_linking and config.global_linking_applied),
        "purity_splitting_applied_before_final_path": bool(config.purity_splitting),
        "strand_states": states,
        "top_k_joint_global_paths": [
            {"rank": rank, "cost": round(row["cost"], 8), "A": row["A"], "B": row["B"]}
            for rank, row in enumerate(retained, start=1)
        ],
        "best_to_second_margin": round(retained[1]["cost"] - best["cost"], 8) if len(retained) > 1 else None,
        "selected_transition_cost_rows": best["cost_rows"],
        "micro_tracklets": tracklets,
        "purity_split_rows": purity_rows,
        "purity_audit": purity_audit,
        "global_link_candidates": links,
        "one_to_one_enforced": True,
        "null_state_allowed": True,
        "ambiguous_state_allowed": True,
        "forced_end_mapping": False,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "tracker_promoted": False,
        "persistent_identity_created": False,
        "match_local_only": True,
        "sandbox_only": True,
    }
    dynamic, handoffs = dynamic_roi_handoff_rows(graph, result, config)
    result["dynamic_roi_rows"] = dynamic
    result["crop_handoff_rows"] = handoffs
    return result


def derive_panorama_visibility_sidecar(gold_rows: list[dict[str, Any]], graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive panorama visibility without changing historical gold rows."""
    nodes = {str(row["node_id"]): row for row in graph["nodes"]}
    aliases = graph.get("alias_to_node", {})
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes.values():
        by_frame[int(node["frame_sequence"])].append(node)
    histories: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    output = []
    for gold in gold_rows:
        frame = int(gold["frame_sequence"])
        for strand in ("A", "B"):
            truth = gold[strand]
            source_id = truth.get("source_observation_id")
            if source_id:
                node_id = aliases.get(str(source_id), str(source_id))
                node = nodes.get(node_id)
                if node is not None:
                    histories[strand].append(node)
                    histories[strand] = histories[strand][-4:]
                output.append(
                    {
                        "sequence_id": gold["sequence_id"],
                        "split": gold["split"],
                        "frame_sequence": frame,
                        "strand": strand,
                        "historical_gold_state": truth["state"],
                        "derived_panorama_state": "INSIDE_FOCAL_ROI",
                        "source_observation_id": source_id,
                        "panorama_node_id": node_id,
                        "evidence_source_row_hash": truth.get("source_row_hash"),
                        "historical_gold_mutated": False,
                    }
                )
                continue
            if truth["state"] != "OUTSIDE_ROI":
                derived = (
                    "OUTSIDE_FOCAL_ROI_NOT_VISIBLE_IN_PANORAMA"
                    if truth["state"] == "NOT_VISIBLE"
                    else "PANORAMA_VISIBILITY_AMBIGUOUS"
                )
                output.append(
                    {
                        "sequence_id": gold["sequence_id"],
                        "split": gold["split"],
                        "frame_sequence": frame,
                        "strand": strand,
                        "historical_gold_state": truth["state"],
                        "derived_panorama_state": derived,
                        "source_observation_id": None,
                        "panorama_node_id": None,
                        "historical_gold_mutated": False,
                    }
                )
                continue
            history = histories[strand]
            candidates = [node for node in by_frame[frame] if node.get("pitch_gate_eligible")]
            if history:
                hypotheses = predict_motion_bank(history, frame)
                candidates.sort(
                    key=lambda node: min(math.dist(row["point"], _foot(node)) for row in hypotheses)
                    / max(1.0, _height(node))
                )
            selected = candidates[0] if candidates else None
            normalized_residual = None
            if selected is not None and history:
                normalized_residual = min(
                    math.dist(row["point"], _foot(selected)) for row in predict_motion_bank(history, frame)
                ) / max(1.0, _height(selected))
                if normalized_residual > 1.1:
                    selected = None
            state = (
                "OUTSIDE_FOCAL_ROI_VISIBLE_IN_PANORAMA"
                if selected is not None and selected["focal_roi_membership"] == "OUTSIDE_FOCAL_ROI"
                else "VISIBLE_IN_PANORAMA_NO_VALID_DETECTION"
                if selected is None
                else "PANORAMA_VISIBILITY_AMBIGUOUS"
            )
            if selected is not None:
                histories[strand].append(selected)
                histories[strand] = histories[strand][-4:]
            output.append(
                {
                    "sequence_id": gold["sequence_id"],
                    "split": gold["split"],
                    "frame_sequence": frame,
                    "strand": strand,
                    "historical_gold_state": "OUTSIDE_ROI",
                    "derived_panorama_state": state,
                    "source_observation_id": None,
                    "panorama_node_id": selected["node_id"] if selected is not None else None,
                    "evidence_source_row_hash": selected.get("source_row_hash") if selected is not None else None,
                    "evidence_bbox": selected.get("bbox") if selected is not None else None,
                    "prediction_residual_heights": round(normalized_residual, 6)
                    if normalized_residual is not None
                    else None,
                    "historical_gold_mutated": False,
                }
            )
    if not all(row["derived_panorama_state"] in PANORAMA_VISIBILITY_STATES for row in output):
        raise AssertionError("invalid panorama visibility sidecar state")
    return output


def evaluate_panorama_paths(
    *,
    result: dict[str, Any],
    graph: dict[str, Any],
    gold_rows: list[dict[str, Any]],
    sidecar_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate public panorama-visible paths with provenance-cluster equivalence."""
    aliases = graph.get("alias_to_node", {})
    nodes = {str(row["node_id"]): row for row in graph["nodes"]}
    predictions = {int(row["frame_sequence"]): row for row in result["strand_states"]}
    sidecar = {(int(row["frame_sequence"]), row["strand"]): row for row in sidecar_rows}
    counters = defaultdict(int)
    frame_rows = []
    exact = True
    for gold in gold_rows:
        frame = int(gold["frame_sequence"])
        prediction = predictions.get(frame, {})
        selected_ids = [prediction.get(strand, {}).get("node_id") for strand in ("A", "B")]
        if selected_ids[0] is not None and selected_ids[0] == selected_ids[1]:
            counters["double_assignments"] += 1
        for strand in ("A", "B"):
            truth = gold[strand]
            derived = sidecar[(frame, strand)]
            expected = truth.get("source_observation_id")
            if expected is None and derived["derived_panorama_state"] == "OUTSIDE_FOCAL_ROI_VISIBLE_IN_PANORAMA":
                expected = derived.get("panorama_node_id")
            expected = aliases.get(str(expected), str(expected)) if expected is not None else None
            predicted = prediction.get(strand, {}).get("node_id")
            predicted = aliases.get(str(predicted), str(predicted)) if predicted is not None else None
            eligible = (
                expected is not None or derived["derived_panorama_state"] == "VISIBLE_IN_PANORAMA_NO_VALID_DETECTION"
            )
            counters["possible_strand_frames"] += 1
            counters["eligible_strand_frames"] += int(eligible)
            supply = expected in nodes if expected is not None else False
            if expected is not None and predicted == expected:
                outcome = "CORRECT_CONTINUATION"
                counters["correct_strand_frames"] += 1
            elif expected is not None and predicted is None:
                outcome = "STRAND_LOSS_DESPITE_SUPPLY" if supply else "SAFE_ABSTENTION_NO_SUPPLY"
                counters["strand_losses_when_supply_available"] += int(supply)
                counters["safe_abstentions"] += int(not supply)
                exact = False
            elif expected is not None and predicted != expected:
                outcome = "ASSOCIATION_SWITCH"
                counters["identity_switches"] += 1
                counters["false_continuations"] += 1
                exact = False
            elif expected is None and predicted is None:
                outcome = "SAFE_ABSTENTION"
                counters["safe_abstentions"] += int(
                    derived["derived_panorama_state"]
                    in {"PANORAMA_VISIBILITY_AMBIGUOUS", "VISIBLE_IN_PANORAMA_NO_VALID_DETECTION"}
                )
            else:
                outcome = "FALSE_CONTINUATION_WITHOUT_TARGET"
                counters["false_continuations"] += 1
                exact = False
            if predicted is not None:
                node = nodes.get(predicted)
                counters["provenance_failures"] += int(node is None)
                counters["off_pitch_assignments"] += int(node is not None and not node.get("pitch_gate_eligible"))
            frame_rows.append(
                {
                    "sequence_id": graph["sequence_id"],
                    "frame_sequence": frame,
                    "strand": strand,
                    "historical_gold_state": truth["state"],
                    "panorama_state": derived["derived_panorama_state"],
                    "expected_node_id": expected,
                    "predicted_node_id": predicted,
                    "supply_available": supply,
                    "outcome": outcome,
                }
            )
    eligible = counters["eligible_strand_frames"]
    correct = counters["correct_strand_frames"]
    assigned = sum(
        1 for row in frame_rows if row["predicted_node_id"] is not None and row["expected_node_id"] is not None
    )
    deta = assigned / max(1, eligible)
    assa = correct / max(1, assigned)
    return {
        **dict(counters),
        "sequence_id": graph["sequence_id"],
        "split": graph["split"],
        "fully_exact_sequence": exact,
        "exact_path_coverage": correct / max(1, eligible),
        "DetA": deta,
        "AssA": assa,
        "HOTA": math.sqrt(deta * assa),
        "IDF1": 2 * correct / max(1, eligible + assigned),
        "frame_attribution_rows": frame_rows,
    }


def aggregate_panorama_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "possible_strand_frames",
        "eligible_strand_frames",
        "correct_strand_frames",
        "identity_switches",
        "false_continuations",
        "strand_losses_when_supply_available",
        "safe_abstentions",
        "off_pitch_assignments",
        "double_assignments",
        "provenance_failures",
    )
    output = {key: sum(int(row.get(key, 0)) for row in rows) for key in keys}
    output["sequence_count"] = len(rows)
    output["fully_exact_sequences"] = sum(int(row["fully_exact_sequence"]) for row in rows)
    output["exact_path_coverage"] = output["correct_strand_frames"] / max(1, output["eligible_strand_frames"])
    output["DetA"] = sum(float(row["DetA"]) for row in rows) / max(1, len(rows))
    output["AssA"] = sum(float(row["AssA"]) for row in rows) / max(1, len(rows))
    output["HOTA"] = sum(float(row["HOTA"]) for row in rows) / max(1, len(rows))
    output["IDF1"] = sum(float(row["IDF1"]) for row in rows) / max(1, len(rows))
    output["development_hard_gate_passed"] = (
        output["identity_switches"] == 0
        and output["false_continuations"] == 0
        and output["strand_losses_when_supply_available"] == 0
        and output["off_pitch_assignments"] == 0
        and output["double_assignments"] == 0
        and output["provenance_failures"] == 0
        and output["fully_exact_sequences"] == len(rows)
    )
    return output


def grouped_development_cross_validation(
    configuration_metrics: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Select configurations without frame-level leakage across sequences."""
    sequence_ids = sorted({row["sequence_id"] for rows in configuration_metrics.values() for row in rows})

    def key(metrics: dict[str, Any]) -> tuple[Any, ...]:
        return (
            metrics["identity_switches"] + metrics["false_continuations"],
            metrics["strand_losses_when_supply_available"],
            metrics["off_pitch_assignments"] + metrics["double_assignments"] + metrics["provenance_failures"],
            -metrics["fully_exact_sequences"],
            metrics["safe_abstentions"],
            -metrics["AssA"],
            -metrics["IDF1"],
            -metrics["HOTA"],
        )

    folds = []
    for holdout in sequence_ids:
        train_summaries = {}
        for config_hash, rows in configuration_metrics.items():
            train_summaries[config_hash] = aggregate_panorama_metrics(
                [row for row in rows if row["sequence_id"] != holdout]
            )
        selected = min(train_summaries, key=lambda value: (key(train_summaries[value]), value))
        validation = next(row for row in configuration_metrics[selected] if row["sequence_id"] == holdout)
        folds.append(
            {
                "held_out_sequence_id": holdout,
                "training_sequence_ids": [value for value in sequence_ids if value != holdout],
                "selected_configuration_hash": selected,
                "validation_metrics": {
                    key: value for key, value in validation.items() if key != "frame_attribution_rows"
                },
                "group_overlap_count": 0,
            }
        )
    return {
        "protocol": "LEAVE_ONE_DEVELOPMENT_SEQUENCE_OUT",
        "fold_count": len(folds),
        "diagnostic_rows_used_for_selection": 0,
        "holdout_rows_used_for_selection": 0,
        "all_group_overlaps_zero": all(row["group_overlap_count"] == 0 for row in folds),
        "folds": folds,
    }
