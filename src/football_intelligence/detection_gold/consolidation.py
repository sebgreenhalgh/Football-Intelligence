"""Deterministic, provenance-aware consolidation of frozen detection proposals.

This module is the runtime side of M5.5G.3. It deliberately knows nothing
about annotation geometry or human labels. Evaluation lives outside this
module so consolidation rules cannot accidentally consume development gold.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from football_intelligence.review_chassis.hashing import sha256_file, stable_hash

VARIANT_NAMES = (
    "IOU_CONNECTED_COMPONENT_055",
    "GREEDY_NMS_050",
    "GREEDY_NMS_065",
    "SOFT_NMS_LINEAR_050",
    "PROVENANCE_GRAPH_STRICT_HIGHEST_SCORE",
    "PROVENANCE_GRAPH_STRICT_MEDOID",
    "PROVENANCE_GRAPH_BALANCED_HIGHEST_SCORE",
    "PROVENANCE_GRAPH_BALANCED_MEDOID",
)

OUTPUT_STATES = {
    "ACCEPT_INDEPENDENT_OBSERVATION",
    "ROUTE_DENSE_REVIEW",
    "SUPPRESS_DUPLICATE",
    "UNRESOLVED_CLUSTER",
}

FORBIDDEN_RUNTIME_KEYS = {
    "annotation_uuid",
    "case_stratum",
    "failure_label",
    "gold",
    "gold_geometry",
    "human_relation",
    "identity",
    "appearance_embedding",
    "pitch_state",
    "role",
}

REQUIRED_NODE_KEYS = {
    "source_frame_sha256",
    "proposal_uuid",
    "source_view_family",
    "inference_view_id",
    "source_view_footprint",
    "score",
    "bbox_panorama_pixels",
    "transform_hash",
    "checkpoint_runtime_hash",
    "parent_lineage_ids",
}


def consolidation_variant_specification() -> dict[str, Any]:
    """Return the immutable M5.5G.3 variant specification."""

    return {
        "schema_version": "football_intelligence.m5_5g3.consolidation_variant_specification.v1",
        "frozen_before_scoring": True,
        "detector_confidence_floor": 0.22,
        "coordinate_averaging_forbidden": True,
        "runtime_gold_features_forbidden": sorted(FORBIDDEN_RUNTIME_KEYS),
        "deterministic_order": "descending score then stable proposal UUID",
        "complete_link_cluster_preference": (
            "highest minimum pair IoU, then highest mean pair IoU, then stable member UUIDs"
        ),
        "tile_edge_margin": "max(4 panorama pixels, 0.10 proposal height)",
        "variants": [
            {
                "name": "IOU_CONNECTED_COMPONENT_055",
                "algorithm": "undirected_iou_connected_components",
                "iou_threshold": 0.55,
                "representative": "highest_score_real_member",
            },
            {
                "name": "GREEDY_NMS_050",
                "algorithm": "greedy_panorama_nms",
                "iou_threshold": 0.50,
                "representative": "highest_score_real_member",
            },
            {
                "name": "GREEDY_NMS_065",
                "algorithm": "greedy_panorama_nms",
                "iou_threshold": 0.65,
                "representative": "highest_score_real_member",
            },
            {
                "name": "SOFT_NMS_LINEAR_050",
                "algorithm": "linear_soft_nms",
                "decay_starts_at_iou": 0.50,
                "final_score_floor": 0.22,
                "representative": "highest_current_score_real_member",
            },
            {
                "name": "PROVENANCE_GRAPH_STRICT_HIGHEST_SCORE",
                "algorithm": "strict_complete_link_provenance_graph",
                "representative": "highest_score_real_member",
            },
            {
                "name": "PROVENANCE_GRAPH_STRICT_MEDOID",
                "algorithm": "strict_complete_link_provenance_graph",
                "representative": "real_member_iou_medoid",
            },
            {
                "name": "PROVENANCE_GRAPH_BALANCED_HIGHEST_SCORE",
                "algorithm": "balanced_complete_link_provenance_graph",
                "representative": "highest_score_real_member",
            },
            {
                "name": "PROVENANCE_GRAPH_BALANCED_MEDOID",
                "algorithm": "balanced_complete_link_provenance_graph",
                "representative": "real_member_iou_medoid",
            },
        ],
        "strict_graph": {
            "same_view": {
                "minimum_iou": 0.75,
                "height_ratio": [0.70, 1.43],
                "maximum_bottom_centre_distance_smaller_height": 0.20,
            },
            "cross_view_primary": {
                "minimum_iou": 0.60,
                "height_ratio": [0.65, 1.55],
                "maximum_bottom_centre_distance_smaller_height": 0.25,
            },
            "cross_view_containment": {
                "mutual_centre_containment": True,
                "minimum_iou": 0.40,
                "maximum_centre_distance_smaller_height": 0.25,
                "maximum_bottom_centre_distance_smaller_height": 0.25,
            },
            "cross_view_footprint_rule": "both centres inside footprint intersection",
        },
        "balanced_graph": {
            "same_view": {
                "minimum_iou": 0.65,
                "height_ratio": [0.60, 1.67],
                "maximum_bottom_centre_distance_smaller_height": 0.30,
            },
            "cross_view_primary": {
                "minimum_iou": 0.45,
                "height_ratio": [0.55, 1.82],
                "maximum_centre_distance_smaller_height": 0.40,
                "maximum_bottom_centre_distance_smaller_height": 0.35,
            },
            "cross_view_containment": {
                "mutual_centre_containment": True,
                "minimum_iou": 0.30,
                "maximum_bottom_centre_distance_smaller_height": 0.30,
            },
            "cross_view_footprint_rule": "both centres inside footprint intersection",
        },
        "merged_ambiguity_gate": {
            "cross_view_split_evidence": {
                "maximum_split_pair_iou": 0.15,
                "minimum_centre_separation_containing_height": 0.35,
                "minimum_score": 0.22,
            },
            "multi_mode_cluster": {
                "minimum_bottom_centre_separation_median_height": 0.45,
                "distinct_source_views_required": True,
            },
            "merged_candidate_splitting_forbidden": True,
        },
        "selection_objective": [
            "zero merged-as-clean observations",
            "minimum distinct-person suppression",
            "minimum duplicate observations",
            "maximum accepted-plus-dense coverage",
            "maximum accepted independent supply",
            "minimum background observations",
            "minimum CPU P95 runtime",
            "stable variant name",
        ],
    }


def freeze_variant_specification(spec_path: Path, hash_path: Path) -> str:
    """Atomically write and hash the frozen specification.

    Re-running the command is idempotent. An existing different specification
    is rejected rather than overwritten.
    """

    specification = consolidation_variant_specification()
    payload = json.dumps(specification, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if spec_path.exists() and spec_path.read_text(encoding="utf-8") != payload:
        raise RuntimeError("FAIL_VARIANT_SPECIFICATION: existing specification differs")
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    if not spec_path.exists():
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=spec_path.parent, delete=False, newline="\n"
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        temporary.replace(spec_path)
    digest = sha256_file(spec_path)
    hash_payload = f"{digest}  {spec_path.name}\n"
    if hash_path.exists() and hash_path.read_text(encoding="ascii") != hash_payload:
        raise RuntimeError("FAIL_VARIANT_SPECIFICATION: existing hash differs")
    hash_path.write_text(hash_payload, encoding="ascii", newline="\n")
    return digest


def _walk_forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            lowered = key_text.lower()
            if lowered in FORBIDDEN_RUNTIME_KEYS or lowered.startswith("gold_") or lowered.startswith("human_"):
                paths.append(path)
            paths.extend(_walk_forbidden_keys(nested, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            paths.extend(_walk_forbidden_keys(nested, f"{prefix}[{index}]"))
    return paths


def validate_proposal_nodes(nodes: Sequence[Mapping[str, Any]]) -> None:
    """Reject malformed provenance or any accidental gold/runtime leakage."""

    if not nodes:
        return
    forbidden = _walk_forbidden_keys(nodes)
    if forbidden:
        raise ValueError(f"forbidden runtime fields: {sorted(forbidden)}")
    source_hashes = {str(node.get("source_frame_sha256")) for node in nodes}
    if len(source_hashes) != 1:
        raise ValueError("one consolidation call must contain exactly one source frame")
    identifiers: set[str] = set()
    for node in nodes:
        missing = REQUIRED_NODE_KEYS - set(node)
        if missing:
            raise ValueError(f"proposal node missing fields: {sorted(missing)}")
        identifier = str(node["proposal_uuid"])
        if identifier in identifiers:
            raise ValueError(f"duplicate proposal UUID: {identifier}")
        identifiers.add(identifier)
        box = node["bbox_panorama_pixels"]
        footprint = node["source_view_footprint"]
        for geometry in (box, footprint):
            values = [float(geometry[key]) for key in ("x1", "y1", "x2", "y2")]
            if not all(math.isfinite(value) for value in values) or values[0] >= values[2] or values[1] >= values[3]:
                raise ValueError(f"invalid geometry for proposal {identifier}")
        if not math.isfinite(float(node["score"])):
            raise ValueError(f"invalid score for proposal {identifier}")
        if not node["parent_lineage_ids"]:
            raise ValueError(f"missing parent lineage for proposal {identifier}")


def _box(node: Mapping[str, Any]) -> Mapping[str, float]:
    return node["bbox_panorama_pixels"]


def _width(box: Mapping[str, float]) -> float:
    return max(0.0, float(box["x2"]) - float(box["x1"]))


def _height(box: Mapping[str, float]) -> float:
    return max(0.0, float(box["y2"]) - float(box["y1"]))


def _area(box: Mapping[str, float]) -> float:
    return _width(box) * _height(box)


def _centre(box: Mapping[str, float]) -> tuple[float, float]:
    return ((float(box["x1"]) + float(box["x2"])) / 2, (float(box["y1"]) + float(box["y2"])) / 2)


def _bottom_centre(box: Mapping[str, float]) -> tuple[float, float]:
    centre = _centre(box)
    return centre[0], float(box["y2"])


def _contains(box: Mapping[str, float], point: tuple[float, float]) -> bool:
    return float(box["x1"]) <= point[0] <= float(box["x2"]) and float(box["y1"]) <= point[1] <= float(box["y2"])


def proposal_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Return panorama-box IoU for two proposal nodes."""

    a, b = _box(left), _box(right)
    x1, y1 = max(float(a["x1"]), float(b["x1"])), max(float(a["y1"]), float(b["y1"]))
    x2, y2 = min(float(a["x2"]), float(b["x2"])), min(float(a["y2"]), float(b["y2"]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return intersection / max(1e-12, _area(a) + _area(b) - intersection)


def _height_ratio(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    heights = (_height(_box(left)), _height(_box(right)))
    return min(heights) / max(1e-12, max(heights))


def _normalised_distance(left: Mapping[str, Any], right: Mapping[str, Any], *, bottom: bool = False) -> float:
    point = _bottom_centre if bottom else _centre
    scale = max(1e-12, min(_height(_box(left)), _height(_box(right))))
    return math.dist(point(_box(left)), point(_box(right))) / scale


def _footprint_intersection(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float] | None:
    a, b = left["source_view_footprint"], right["source_view_footprint"]
    overlap = {
        "x1": max(float(a["x1"]), float(b["x1"])),
        "y1": max(float(a["y1"]), float(b["y1"])),
        "x2": min(float(a["x2"]), float(b["x2"])),
        "y2": min(float(a["y2"]), float(b["y2"])),
    }
    return overlap if overlap["x1"] < overlap["x2"] and overlap["y1"] < overlap["y2"] else None


def _footprints_cover_both_centres(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    overlap = _footprint_intersection(left, right)
    return bool(overlap and _contains(overlap, _centre(_box(left))) and _contains(overlap, _centre(_box(right))))


def _mutual_centre_containment(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _contains(_box(left), _centre(_box(right))) and _contains(_box(right), _centre(_box(left)))


def _strict_edge(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    iou = proposal_iou(left, right)
    ratio = _height_ratio(left, right)
    bottom = _normalised_distance(left, right, bottom=True)
    if left["inference_view_id"] == right["inference_view_id"]:
        return iou >= 0.75 and ratio >= 0.70 and bottom <= 0.20
    if not _footprints_cover_both_centres(left, right):
        return False
    primary = iou >= 0.60 and ratio >= 0.65 and bottom <= 0.25
    containment = (
        _mutual_centre_containment(left, right)
        and iou >= 0.40
        and _normalised_distance(left, right) <= 0.25
        and bottom <= 0.25
    )
    return primary or containment


def _balanced_edge(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    iou = proposal_iou(left, right)
    ratio = _height_ratio(left, right)
    bottom = _normalised_distance(left, right, bottom=True)
    if left["inference_view_id"] == right["inference_view_id"]:
        return iou >= 0.65 and ratio >= 0.60 and bottom <= 0.30
    if not _footprints_cover_both_centres(left, right):
        return False
    primary = iou >= 0.45 and ratio >= 0.55 and _normalised_distance(left, right) <= 0.40 and bottom <= 0.35
    containment = _mutual_centre_containment(left, right) and iou >= 0.30 and bottom <= 0.30
    return primary or containment


def _highest_score(members: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return sorted(members, key=lambda node: (-float(node["score"]), str(node["proposal_uuid"])))[0]


def _medoid(members: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    scored = []
    for member in members:
        total_iou = sum(proposal_iou(member, other) for other in members if other is not member)
        scored.append((total_iou, float(member["score"]), str(member["proposal_uuid"]), member))
    return sorted(scored, key=lambda row: (-row[0], -row[1], row[2]))[0][3]


def _connected_components(nodes: Sequence[Mapping[str, Any]], threshold: float) -> list[list[Mapping[str, Any]]]:
    parents = list(range(len(nodes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if proposal_iou(nodes[left], nodes[right]) >= threshold:
                a, b = find(left), find(right)
                if a != b:
                    parents[max(a, b)] = min(a, b)
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for index, node in enumerate(nodes):
        grouped[find(index)].append(node)
    return list(grouped.values())


def _greedy_nms(nodes: Sequence[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    remaining = sorted(nodes, key=lambda node: (-float(node["score"]), str(node["proposal_uuid"])))
    clusters: list[dict[str, Any]] = []
    while remaining:
        representative = remaining.pop(0)
        suppressed = [node for node in remaining if proposal_iou(representative, node) >= threshold]
        suppressed_ids = {str(node["proposal_uuid"]) for node in suppressed}
        remaining = [node for node in remaining if str(node["proposal_uuid"]) not in suppressed_ids]
        clusters.append(
            {
                "members": [representative, *suppressed],
                "representative": representative,
                "output_score": float(representative["score"]),
            }
        )
    return clusters


def _soft_nms(nodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    current = {str(node["proposal_uuid"]): float(node["score"]) for node in nodes}
    by_id = {str(node["proposal_uuid"]): node for node in nodes}
    clusters: list[dict[str, Any]] = []
    while current:
        winner_id = sorted(current, key=lambda identifier: (-current[identifier], identifier))[0]
        winner = by_id[winner_id]
        winner_score = current.pop(winner_id)
        suppressed: list[Mapping[str, Any]] = []
        for identifier in sorted(list(current)):
            iou = proposal_iou(winner, by_id[identifier])
            if iou < 0.50:
                continue
            current[identifier] *= 1.0 - iou
            if current[identifier] < 0.22:
                suppressed.append(by_id[identifier])
                del current[identifier]
        clusters.append(
            {
                "members": [winner, *suppressed],
                "representative": winner,
                "output_score": winner_score,
            }
        )
    return clusters


def _complete_link_clusters(nodes: Sequence[Mapping[str, Any]], *, balanced: bool) -> list[list[Mapping[str, Any]]]:
    edge = _balanced_edge if balanced else _strict_edge
    node_by_id = {str(node["proposal_uuid"]): node for node in nodes}
    pair_compatible: dict[tuple[str, str], bool] = {}
    pair_iou: dict[tuple[str, str], float] = {}
    identifiers = sorted(node_by_id)
    for left_index, left_id in enumerate(identifiers):
        for right_id in identifiers[left_index + 1 :]:
            key = (left_id, right_id)
            pair_compatible[key] = edge(node_by_id[left_id], node_by_id[right_id])
            pair_iou[key] = proposal_iou(node_by_id[left_id], node_by_id[right_id])

    def pair_key(left_id: str, right_id: str) -> tuple[str, str]:
        return (left_id, right_id) if left_id < right_id else (right_id, left_id)

    clusters: list[list[Mapping[str, Any]]] = []
    ordered = sorted(nodes, key=lambda node: (-float(node["score"]), str(node["proposal_uuid"])))
    for node in ordered:
        identifier = str(node["proposal_uuid"])
        compatible: list[tuple[float, float, tuple[str, ...], int]] = []
        for index, cluster in enumerate(clusters):
            member_ids = tuple(sorted(str(member["proposal_uuid"]) for member in cluster))
            keys = [pair_key(identifier, member_id) for member_id in member_ids]
            if all(pair_compatible.get(key, False) for key in keys):
                values = [pair_iou[key] for key in keys]
                compatible.append((min(values), sum(values) / len(values), member_ids, index))
        if not compatible:
            clusters.append([node])
            continue
        selected = sorted(compatible, key=lambda row: (-row[0], -row[1], row[2]))[0][3]
        clusters[selected].append(node)
    return clusters


def _build_clusters(nodes: Sequence[Mapping[str, Any]], variant_name: str) -> list[dict[str, Any]]:
    if variant_name == "IOU_CONNECTED_COMPONENT_055":
        groups = _connected_components(nodes, 0.55)
        clusters = []
        for group in groups:
            representative = _highest_score(group)
            clusters.append(
                {
                    "members": group,
                    "representative": representative,
                    "output_score": float(representative["score"]),
                }
            )
        return clusters
    if variant_name == "GREEDY_NMS_050":
        return _greedy_nms(nodes, 0.50)
    if variant_name == "GREEDY_NMS_065":
        return _greedy_nms(nodes, 0.65)
    if variant_name == "SOFT_NMS_LINEAR_050":
        return _soft_nms(nodes)
    if variant_name.startswith("PROVENANCE_GRAPH_STRICT"):
        groups = _complete_link_clusters(nodes, balanced=False)
    elif variant_name.startswith("PROVENANCE_GRAPH_BALANCED"):
        groups = _complete_link_clusters(nodes, balanced=True)
    else:
        raise ValueError(f"unknown consolidation variant: {variant_name}")
    use_medoid = variant_name.endswith("_MEDOID")
    return [
        {
            "members": group,
            "representative": _medoid(group) if use_medoid else _highest_score(group),
            "output_score": float((_medoid(group) if use_medoid else _highest_score(group))["score"]),
        }
        for group in groups
    ]


def _split_evidence_reasons(
    members: Sequence[Mapping[str, Any]], all_nodes: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    for container in members:
        container_height = max(1e-12, _height(_box(container)))
        alternatives = []
        for candidate in all_nodes:
            if candidate["source_view_family"] == container["source_view_family"]:
                continue
            if float(candidate["score"]) < 0.22 or not _contains(_box(container), _centre(_box(candidate))):
                continue
            footprint_overlap = _footprint_intersection(container, candidate)
            if footprint_overlap and _contains(footprint_overlap, _centre(_box(candidate))):
                alternatives.append(candidate)
        for left_index, left in enumerate(alternatives):
            for right in alternatives[left_index + 1 :]:
                if proposal_iou(left, right) > 0.15:
                    continue
                separation = math.dist(_centre(_box(left)), _centre(_box(right))) / container_height
                if separation < 0.35:
                    continue
                reasons.append(
                    {
                        "reason": "CROSS_VIEW_SPLIT_EVIDENCE",
                        "containing_proposal_uuid": container["proposal_uuid"],
                        "split_proposal_uuids": sorted([left["proposal_uuid"], right["proposal_uuid"]]),
                        "split_pair_iou": round(proposal_iou(left, right), 8),
                        "centre_separation_containing_height": round(separation, 8),
                    }
                )
    return reasons


def _multi_mode_reasons(members: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(members) < 2:
        return []
    median_height = sorted(_height(_box(member)) for member in members)[len(members) // 2]
    reasons: list[dict[str, Any]] = []
    for left_index, left in enumerate(members):
        for right in members[left_index + 1 :]:
            if left["inference_view_id"] == right["inference_view_id"]:
                continue
            separation = math.dist(_bottom_centre(_box(left)), _bottom_centre(_box(right))) / max(1e-12, median_height)
            if separation >= 0.45:
                reasons.append(
                    {
                        "reason": "MULTI_MODE_CLUSTER",
                        "mode_proposal_uuids": sorted([left["proposal_uuid"], right["proposal_uuid"]]),
                        "bottom_centre_separation_median_height": round(separation, 8),
                    }
                )
    return reasons


def merged_ambiguity_gate(
    members: Sequence[Mapping[str, Any]], all_nodes: Sequence[Mapping[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Route proposal-only merged ambiguity without splitting observations."""

    split = _split_evidence_reasons(members, all_nodes)
    modes = _multi_mode_reasons(members)
    reasons = sorted(split + modes, key=lambda row: stable_hash(row))
    return ("ROUTE_DENSE_REVIEW", reasons) if reasons else ("ACCEPT_INDEPENDENT_OBSERVATION", [])


def _observation_provenance_payload(
    source_hash: str,
    variant_name: str,
    gate_applied: bool,
    members: Sequence[Mapping[str, Any]],
    representative: Mapping[str, Any],
    representative_method: str,
    output_state: str,
    output_score: float,
    dense_reasons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "source_frame_sha256": source_hash,
        "consolidation_variant": variant_name,
        "merged_gate_applied": gate_applied,
        "cluster_member_proposal_uuids": sorted(str(member["proposal_uuid"]) for member in members),
        "cluster_member_proposal_hashes": sorted(_proposal_node_hash(member) for member in members),
        "representative_proposal_uuid": str(representative["proposal_uuid"]),
        "representative_proposal_hash": _proposal_node_hash(representative),
        "representative_selection_method": representative_method,
        "output_state": output_state,
        "output_score": round(float(output_score), 8),
        "dense_review_reasons": sorted((dict(row) for row in dense_reasons), key=stable_hash),
        "member_transform_hashes": sorted(str(member["transform_hash"]) for member in members),
        "member_checkpoint_runtime_hashes": sorted(str(member["checkpoint_runtime_hash"]) for member in members),
        "parent_lineage_ids": sorted({str(parent) for member in members for parent in member["parent_lineage_ids"]}),
    }


def _proposal_node_hash(node: Mapping[str, Any]) -> str:
    payload = {
        "source_frame_sha256": str(node["source_frame_sha256"]),
        "proposal_uuid": str(node["proposal_uuid"]),
        "source_view_family": str(node["source_view_family"]),
        "inference_view_id": str(node["inference_view_id"]),
        "source_view_footprint": node["source_view_footprint"],
        "crop_bounds_panorama_pixels": node.get("crop_bounds_panorama_pixels"),
        "tile_bounds_panorama_pixels": node.get("tile_bounds_panorama_pixels"),
        "raw_candidate_index": node.get("raw_candidate_index"),
        "score": float(node["score"]),
        "class_provenance": node.get("class_provenance"),
        "bbox_panorama_pixels": node["bbox_panorama_pixels"],
        "transform_hash": str(node["transform_hash"]),
        "checkpoint_runtime_hash": str(node["checkpoint_runtime_hash"]),
        "parent_lineage_ids": sorted(str(value) for value in node["parent_lineage_ids"]),
        "near_tile_or_crop_edge": node.get("near_tile_or_crop_edge"),
        "visible_in_another_overlapping_view": node.get("visible_in_another_overlapping_view"),
    }
    return stable_hash(payload)


def _suppression_provenance_payload(
    *,
    source_hash: str,
    variant_name: str,
    gate_applied: bool,
    member: Mapping[str, Any],
    representative: Mapping[str, Any],
    observation_uuid: str,
    suppression_iou: float,
) -> dict[str, Any]:
    return {
        "source_frame_sha256": source_hash,
        "consolidation_variant": variant_name,
        "merged_gate_applied": gate_applied,
        "proposal_uuid": str(member["proposal_uuid"]),
        "proposal_hash": _proposal_node_hash(member),
        "representative_proposal_uuid": str(representative["proposal_uuid"]),
        "representative_proposal_hash": _proposal_node_hash(representative),
        "observation_uuid": observation_uuid,
        "output_state": "SUPPRESS_DUPLICATE",
        "suppression_iou": round(float(suppression_iou), 8),
    }


def consolidate_proposals(
    proposal_nodes: Sequence[Mapping[str, Any]],
    variant_name: str,
    *,
    apply_merged_gate: bool,
) -> dict[str, Any]:
    """Consolidate one source frame using only proposal evidence."""

    if variant_name not in VARIANT_NAMES:
        raise ValueError(f"variant is not frozen: {variant_name}")
    validate_proposal_nodes(proposal_nodes)
    if not proposal_nodes:
        return {
            "schema_version": "football_intelligence.m5_5g3.consolidation_result.v1",
            "source_frame_sha256": None,
            "consolidation_variant": variant_name,
            "merged_gate_applied": apply_merged_gate,
            "input_proposal_count": 0,
            "observations": [],
            "duplicate_suppressions": [],
            "determinism_hash": stable_hash([]),
        }
    source_hash = str(proposal_nodes[0]["source_frame_sha256"])
    clusters = _build_clusters(proposal_nodes, variant_name)
    representative_method = (
        "REAL_MEMBER_IOU_MEDOID" if variant_name.endswith("_MEDOID") else "HIGHEST_SCORE_REAL_MEMBER"
    )
    if variant_name == "SOFT_NMS_LINEAR_050":
        representative_method = "HIGHEST_CURRENT_SCORE_REAL_MEMBER"
    observations: list[dict[str, Any]] = []
    suppressions: list[dict[str, Any]] = []
    for cluster in clusters:
        members = sorted(cluster["members"], key=lambda node: str(node["proposal_uuid"]))
        representative = cluster["representative"]
        if apply_merged_gate:
            output_state, dense_reasons = merged_ambiguity_gate(members, proposal_nodes)
        else:
            output_state, dense_reasons = "ACCEPT_INDEPENDENT_OBSERVATION", []
        provenance_payload = _observation_provenance_payload(
            source_hash,
            variant_name,
            apply_merged_gate,
            members,
            representative,
            representative_method,
            output_state,
            float(cluster["output_score"]),
            dense_reasons,
        )
        provenance_hash = stable_hash(provenance_payload)
        observation_uuid = f"observation_{stable_hash([provenance_hash, representative['proposal_uuid']])[:24]}"
        box = {key: float(_box(representative)[key]) for key in ("x1", "y1", "x2", "y2")}
        observations.append(
            {
                "observation_uuid": observation_uuid,
                "source_frame_sha256": source_hash,
                "cluster_member_proposal_uuids": provenance_payload["cluster_member_proposal_uuids"],
                "cluster_member_proposal_hashes": provenance_payload["cluster_member_proposal_hashes"],
                "representative_proposal_uuid": str(representative["proposal_uuid"]),
                "representative_proposal_hash": provenance_payload["representative_proposal_hash"],
                "representative_selection_method": representative_method,
                "consolidation_variant": variant_name,
                "merged_gate_applied": apply_merged_gate,
                "output_state": output_state,
                "score": round(float(cluster["output_score"]), 8),
                "box_panorama_pixels": box,
                "footpoint_proxy_panorama_pixels": {
                    "x": round((box["x1"] + box["x2"]) / 2, 8),
                    "y": round(box["y2"], 8),
                },
                "all_source_view_ids": sorted({str(member["inference_view_id"]) for member in members}),
                "provenance_hash": provenance_hash,
                "dense_review_reason": dense_reasons or None,
                "observed_direct": True,
                "identity_tracking_performed": False,
                "temporal_state_created": False,
            }
        )
        for member in members:
            if member["proposal_uuid"] == representative["proposal_uuid"]:
                continue
            suppression_iou = round(proposal_iou(member, representative), 8)
            suppression_payload = _suppression_provenance_payload(
                source_hash=source_hash,
                variant_name=variant_name,
                gate_applied=apply_merged_gate,
                member=member,
                representative=representative,
                observation_uuid=observation_uuid,
                suppression_iou=suppression_iou,
            )
            suppressions.append(
                {
                    "source_frame_sha256": source_hash,
                    "consolidation_variant": variant_name,
                    "merged_gate_applied": apply_merged_gate,
                    "proposal_uuid": str(member["proposal_uuid"]),
                    "proposal_hash": suppression_payload["proposal_hash"],
                    "representative_proposal_uuid": str(representative["proposal_uuid"]),
                    "representative_proposal_hash": suppression_payload["representative_proposal_hash"],
                    "observation_uuid": observation_uuid,
                    "output_state": "SUPPRESS_DUPLICATE",
                    "suppression_iou": suppression_iou,
                    "provenance_hash": stable_hash(suppression_payload),
                }
            )
    observations.sort(key=lambda row: row["observation_uuid"])
    suppressions.sort(key=lambda row: (row["observation_uuid"], row["proposal_uuid"]))
    core = {"observations": observations, "duplicate_suppressions": suppressions}
    return {
        "schema_version": "football_intelligence.m5_5g3.consolidation_result.v1",
        "source_frame_sha256": source_hash,
        "consolidation_variant": variant_name,
        "merged_gate_applied": apply_merged_gate,
        "input_proposal_count": len(proposal_nodes),
        **core,
        "determinism_hash": stable_hash(core),
    }


def validate_observation_provenance(
    result: Mapping[str, Any], proposal_nodes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Prove every proposal is assigned once and every hash is reproducible."""

    node_by_id = {str(node["proposal_uuid"]): node for node in proposal_nodes}
    errors: list[str] = []
    seen: list[str] = []
    variant_name = str(result["consolidation_variant"])
    gate_applied = bool(result["merged_gate_applied"])
    expected_clusters = {
        tuple(sorted(str(member["proposal_uuid"]) for member in cluster["members"])): cluster
        for cluster in _build_clusters(proposal_nodes, variant_name)
    }
    observed_cluster_keys = {
        tuple(sorted(str(value) for value in observation["cluster_member_proposal_uuids"]))
        for observation in result["observations"]
    }
    if observed_cluster_keys != set(expected_clusters):
        errors.append("observation clusters do not reconstruct from proposal nodes")
    for observation in result["observations"]:
        member_ids = [str(value) for value in observation["cluster_member_proposal_uuids"]]
        seen.extend(member_ids)
        if observation["representative_proposal_uuid"] not in member_ids:
            errors.append(f"representative outside cluster: {observation['observation_uuid']}")
            continue
        if not set(member_ids) <= set(node_by_id):
            errors.append(f"unknown cluster member: {observation['observation_uuid']}")
            continue
        members = [node_by_id[identifier] for identifier in member_ids]
        representative = node_by_id[observation["representative_proposal_uuid"]]
        cluster_key = tuple(sorted(member_ids))
        expected_cluster = expected_clusters.get(cluster_key)
        if expected_cluster is None:
            continue
        expected_representative = expected_cluster["representative"]
        if representative["proposal_uuid"] != expected_representative["proposal_uuid"]:
            errors.append(f"representative mismatch: {observation['observation_uuid']}")
        expected_state, expected_reasons = (
            merged_ambiguity_gate(members, proposal_nodes) if gate_applied else ("ACCEPT_INDEPENDENT_OBSERVATION", [])
        )
        if observation["output_state"] != expected_state:
            errors.append(f"output state mismatch: {observation['observation_uuid']}")
        if (observation.get("dense_review_reason") or []) != (expected_reasons or []):
            errors.append(f"dense-review reason mismatch: {observation['observation_uuid']}")
        expected_member_hashes = sorted(_proposal_node_hash(member) for member in members)
        if observation.get("cluster_member_proposal_hashes") != expected_member_hashes:
            errors.append(f"member proposal hash mismatch: {observation['observation_uuid']}")
        if observation.get("representative_proposal_hash") != _proposal_node_hash(representative):
            errors.append(f"representative proposal hash mismatch: {observation['observation_uuid']}")
        expected_score = round(float(expected_cluster["output_score"]), 8)
        if float(observation["score"]) != expected_score:
            errors.append(f"output score mismatch: {observation['observation_uuid']}")
        payload = _observation_provenance_payload(
            str(observation["source_frame_sha256"]),
            variant_name,
            gate_applied,
            members,
            representative,
            str(observation["representative_selection_method"]),
            str(observation["output_state"]),
            float(observation["score"]),
            observation.get("dense_review_reason") or [],
        )
        if stable_hash(payload) != observation["provenance_hash"]:
            errors.append(f"provenance hash mismatch: {observation['observation_uuid']}")
        if observation["box_panorama_pixels"] != {
            key: float(_box(representative)[key]) for key in ("x1", "y1", "x2", "y2")
        }:
            errors.append(f"representative box changed: {observation['observation_uuid']}")
    if sorted(seen) != sorted(node_by_id):
        errors.append("input proposals are not assigned exactly once")
    observation_by_id = {str(row["observation_uuid"]): row for row in result["observations"]}
    expected_suppressions = {}
    for observation in result["observations"]:
        representative = node_by_id[str(observation["representative_proposal_uuid"])]
        for identifier in observation["cluster_member_proposal_uuids"]:
            if identifier == representative["proposal_uuid"]:
                continue
            member = node_by_id[str(identifier)]
            suppression_iou = round(proposal_iou(member, representative), 8)
            payload = _suppression_provenance_payload(
                source_hash=str(observation["source_frame_sha256"]),
                variant_name=variant_name,
                gate_applied=gate_applied,
                member=member,
                representative=representative,
                observation_uuid=str(observation["observation_uuid"]),
                suppression_iou=suppression_iou,
            )
            expected_suppressions[str(identifier)] = {**payload, "provenance_hash": stable_hash(payload)}
    actual_suppressions = {str(row["proposal_uuid"]): row for row in result["duplicate_suppressions"]}
    if set(actual_suppressions) != set(expected_suppressions):
        errors.append("duplicate suppression ledger does not reconstruct")
    for identifier, expected in expected_suppressions.items():
        actual = actual_suppressions.get(identifier)
        if actual is None:
            continue
        comparable = {key: actual.get(key) for key in expected}
        if comparable != expected:
            errors.append(f"duplicate suppression mismatch: {identifier}")
        if str(actual.get("observation_uuid")) not in observation_by_id:
            errors.append(f"duplicate suppression observation missing: {identifier}")
    core = {
        "observations": result["observations"],
        "duplicate_suppressions": result["duplicate_suppressions"],
    }
    if result.get("determinism_hash") != stable_hash(core):
        errors.append("result determinism hash mismatch")
    return {
        "passed": not errors,
        "errors": errors,
        "input_proposal_count": len(proposal_nodes),
        "reconstructed_proposal_count": len(seen),
        "coordinate_averaging_performed": False,
    }
