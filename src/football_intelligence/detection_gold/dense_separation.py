"""Deterministic dense-region geometry for M5.5G.4.

Runtime eligibility consumes frozen proposal geometry and provenance only.
Human visible masks are accepted exclusively by the evaluation helpers in
this module; they are never inputs to :func:`evaluate_eligibility_variants`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from football_intelligence.detection_gold.consolidation import proposal_iou
from football_intelligence.review_chassis.hashing import stable_hash

TRUTH_CLASSES = (
    "CLEAN_SINGLE_PERSON",
    "DUPLICATE_SINGLE_PERSON",
    "MERGED_MULTIPLE_PEOPLE",
    "PARTIAL_SINGLE_PERSON",
    "BACKGROUND",
    "AMBIGUOUS",
)

DENSE_OUTPUT_STATES = (
    "ACCEPT_VISIBLE_INSTANCE",
    "ROUTE_HUMAN_DENSE_REVIEW",
    "UNRESOLVED_DENSE_REGION",
    "SUPPRESS_DUPLICATE_MASK",
    "REJECT_BACKGROUND_MASK",
)

ELIGIBILITY_VARIANTS = ("E0", "E1", "E2", "E3", "E4", "E5")

FORBIDDEN_RUNTIME_KEYS = {
    "annotation",
    "annotation_uuid",
    "case_stratum",
    "failure_label",
    "gold",
    "gold_geometry",
    "human",
    "human_relation",
    "human_truth",
    "mask",
    "mask_quality",
    "occluder_uuid",
    "occlusion_order",
    "pitch_state",
    "role",
    "truth_class",
    "visible_mask",
}


def dense_truth_classification_specification() -> dict[str, Any]:
    """Return thresholds frozen before dense-gold scoring."""

    return {
        "schema_version": "football_intelligence.m5_5g4.dense_truth_specification.v1",
        "frozen_before_scoring": True,
        "truth_classes": list(TRUTH_CLASSES),
        "mask_intersection_fraction_threshold": 0.10,
        "clean_mask_coverage_threshold": 0.65,
        "clean_proposal_coverage_threshold": 0.35,
        "minimum_visible_pixel_area": 16,
        "centre_containment_rule": (
            "material when mask coverage reaches threshold and either the mask centroid "
            "is in the proposal or proposal coverage reaches 0.35"
        ),
        "boundary_ignore_pixels": 1,
        "duplicate_same_mask_minimum_iou": 0.55,
        "visible_mask_is_primary_truth": True,
        "human_gold_runtime_input_forbidden": True,
        "development_only": True,
    }


def eligibility_variant_specification() -> dict[str, Any]:
    """Return the complete proposal-only gate family frozen before scoring."""

    return {
        "schema_version": "football_intelligence.m5_5g4.dense_eligibility_variants.v1",
        "frozen_before_scoring": True,
        "runtime_inputs": [
            "proposal_boxes_and_scores",
            "view_family_and_footprint",
            "tile_or_crop_provenance",
            "proposal_lineage",
            "cluster_geometry",
        ],
        "runtime_gold_features_forbidden": sorted(FORBIDDEN_RUNTIME_KEYS),
        "g3_exact_thresholds": {
            "minimum_score": 0.22,
            "cross_view_split_maximum_pair_iou": 0.15,
            "cross_view_split_minimum_centre_separation_containing_height": 0.35,
            "multi_mode_minimum_bottom_centre_separation_median_height": 0.45,
        },
        "containment_disagreement_thresholds": {
            "minimum_score": 0.22,
            "minimum_smaller_proposals": 2,
            "minimum_distinct_view_or_tile_ids": 2,
            "maximum_smaller_pair_iou": 0.15,
            "minimum_container_to_smaller_area_ratio": 1.35,
            "minimum_centre_separation_containing_height": 0.30,
        },
        "variants": {
            "E0": "unchanged G3 cross-view split OR multi-mode gate",
            "E1": "unchanged G3 cross-view split evidence",
            "E2": "unchanged G3 multi-mode cluster evidence",
            "E3": "frozen containment-disagreement evidence",
            "E4": "E1 OR E2 OR E3",
            "E5": "at least two of E1, E2 and E3",
        },
        "threshold_search_performed": False,
        "learned_gate": False,
        "development_only": True,
    }


def mask_output_consolidation_specification() -> dict[str, Any]:
    """Return deterministic promptable-mask consolidation rules."""

    return {
        "schema_version": "football_intelligence.m5_5g4.mask_output_consolidation.v1",
        "frozen_before_scoring": True,
        "minimum_current_frame_pixel_area": 16,
        "duplicate_mask_iou_threshold": 0.85,
        "duplicate_containment_threshold": 0.92,
        "real_output_mask_retained": True,
        "mask_averaging_forbidden": True,
        "appearance_or_identity_forbidden": True,
        "runtime_merged_risk_action": "ROUTE_HUMAN_DENSE_REVIEW",
        "evaluation_gold_may_measure_but_not_change_runtime_gate": True,
        "output_states": list(DENSE_OUTPUT_STATES),
        "development_only": True,
    }


def _box(row: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("bbox_panorama_pixels", "box_panorama_pixels", "bbox_original_pixels", "bbox"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    if all(key in row for key in ("x1", "y1", "x2", "y2")):
        return row
    raise ValueError("row has no supported bounding box")


def _box_area(box: Mapping[str, Any]) -> float:
    return max(0.0, float(box["x2"]) - float(box["x1"])) * max(0.0, float(box["y2"]) - float(box["y1"]))


def _centre(box: Mapping[str, Any]) -> tuple[float, float]:
    return (
        (float(box["x1"]) + float(box["x2"])) / 2,
        (float(box["y1"]) + float(box["y2"])) / 2,
    )


def _bottom_centre(box: Mapping[str, Any]) -> tuple[float, float]:
    return ((float(box["x1"]) + float(box["x2"])) / 2, float(box["y2"]))


def _contains(box: Mapping[str, Any], point: tuple[float, float]) -> bool:
    return float(box["x1"]) <= point[0] <= float(box["x2"]) and float(box["y1"]) <= point[1] <= float(box["y2"])


def _intersection_box(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float] | None:
    value = {
        "x1": max(float(left["x1"]), float(right["x1"])),
        "y1": max(float(left["y1"]), float(right["y1"])),
        "x2": min(float(left["x2"]), float(right["x2"])),
        "y2": min(float(left["y2"]), float(right["y2"])),
    }
    return value if value["x2"] > value["x1"] and value["y2"] > value["y1"] else None


def _footprint_intersection(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float] | None:
    return _intersection_box(
        left.get("source_view_footprint", _box(left)),
        right.get("source_view_footprint", _box(right)),
    )


def _walk_keys(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    found: list[tuple[tuple[str, ...], str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            found.append((path + (str(key),), normalized))
            found.extend(_walk_keys(child, path + (str(key),)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(_walk_keys(child, path + (str(index),)))
    return found


def assert_no_gold_runtime_leakage(payload: Any) -> None:
    """Raise when a runtime payload recursively includes evaluator-only keys."""

    violations = []
    for path, key in _walk_keys(payload):
        if key in FORBIDDEN_RUNTIME_KEYS or any(
            token in key for token in ("human_", "gold_", "visible_mask", "annotation_")
        ):
            violations.append(".".join(path))
    if violations:
        raise ValueError(f"gold/runtime leakage: {sorted(set(violations))}")


def polygon_area(points: Sequence[Mapping[str, Any]]) -> float:
    """Return absolute shoelace area in original-image pixels."""

    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        area += float(point["x"]) * float(following["y"]) - float(following["x"]) * float(point["y"])
    return abs(area) / 2


def _orientation(first: tuple[float, float], second: tuple[float, float], third: tuple[float, float]) -> float:
    return (second[1] - first[1]) * (third[0] - second[0]) - (second[0] - first[0]) * (third[1] - second[1])


def _segments_intersect(
    left_a: tuple[float, float],
    left_b: tuple[float, float],
    right_a: tuple[float, float],
    right_b: tuple[float, float],
) -> bool:
    values = (
        _orientation(left_a, left_b, right_a),
        _orientation(left_a, left_b, right_b),
        _orientation(right_a, right_b, left_a),
        _orientation(right_a, right_b, left_b),
    )
    return values[0] * values[1] < 0 and values[2] * values[3] < 0


def polygon_self_intersection_pairs(points: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """Return the index pairs for every pair of crossing non-adjacent edges."""

    count = len(points)
    if count < 4:
        return []
    coords = [(float(point["x"]), float(point["y"])) for point in points]
    pairs: list[tuple[int, int]] = []
    for left in range(count):
        left_next = (left + 1) % count
        for right in range(left + 1, count):
            right_next = (right + 1) % count
            if left in {right, right_next} or left_next in {right, right_next}:
                continue
            if _segments_intersect(coords[left], coords[left_next], coords[right], coords[right_next]):
                pairs.append((left, right))
    return pairs


def polygon_self_intersects(points: Sequence[Mapping[str, Any]]) -> bool:
    """Return true when non-adjacent polygon edges cross."""

    return bool(polygon_self_intersection_pairs(points))


def validate_polygon(
    points: Sequence[Mapping[str, Any]], roi: Mapping[str, Any], *, minimum_area: float = 16.0
) -> dict[str, Any]:
    """Validate one visible-mask polygon without modifying it."""

    errors: list[str] = []
    if len(points) < 3:
        errors.append("FEWER_THAN_THREE_VERTICES")
    finite = all(math.isfinite(float(point[key])) for point in points for key in ("x", "y"))
    if not finite:
        errors.append("NON_FINITE_COORDINATE")
    outside = [
        index
        for index, point in enumerate(points)
        if not (
            float(roi["x1"]) <= float(point["x"]) <= float(roi["x2"])
            and float(roi["y1"]) <= float(point["y"]) <= float(roi["y2"])
        )
    ]
    if outside:
        errors.append("EXTENDS_BEYOND_FOCAL_ROI")
    self_intersects = polygon_self_intersects(points)
    if self_intersects:
        errors.append("SELF_INTERSECTION")
    area = polygon_area(points)
    if area < minimum_area:
        errors.append("VISIBLE_AREA_BELOW_MINIMUM")
    return {
        "valid": not errors,
        "errors": errors,
        "vertex_count": len(points),
        "polygon_area_pixels": round(area, 6),
        "self_intersects": self_intersects,
        "outside_roi_vertex_indices": outside,
    }


def _raster_bounds(
    points: Sequence[Mapping[str, Any]], box: Mapping[str, Any] | None = None
) -> tuple[int, int, int, int]:
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    if box is not None:
        xs.extend((float(box["x1"]), float(box["x2"])))
        ys.extend((float(box["y1"]), float(box["y2"])))
    return (
        math.floor(min(xs)) - 1,
        math.floor(min(ys)) - 1,
        math.ceil(max(xs)) + 1,
        math.ceil(max(ys)) + 1,
    )


def _rasterize_polygon(points: Sequence[Mapping[str, Any]], bounds: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    canvas = np.zeros((max(1, y2 - y1 + 1), max(1, x2 - x1 + 1)), dtype=np.uint8)
    contour = np.asarray(
        [[round(float(point["x"]) - x1), round(float(point["y"]) - y1)] for point in points],
        dtype=np.int32,
    )
    if len(contour) >= 3:
        cv2.fillPoly(canvas, [contour], 1)
    return canvas.astype(bool)


def _box_mask(box: Mapping[str, Any], bounds: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    canvas = np.zeros((max(1, y2 - y1 + 1), max(1, x2 - x1 + 1)), dtype=bool)
    bx1 = max(0, math.floor(float(box["x1"])) - x1)
    by1 = max(0, math.floor(float(box["y1"])) - y1)
    bx2 = min(canvas.shape[1], math.ceil(float(box["x2"])) - x1 + 1)
    by2 = min(canvas.shape[0], math.ceil(float(box["y2"])) - y1 + 1)
    if bx2 > bx1 and by2 > by1:
        canvas[by1:by2, bx1:bx2] = True
    return canvas


def candidate_mask_coverage(
    candidate: Mapping[str, Any], visible_masks: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Measure one candidate box against evaluator-only visible masks."""

    box = _box(candidate)
    rows = []
    for mask in visible_masks:
        points = mask["polygon_original_pixels"]
        bounds = _raster_bounds(points, box)
        raster = _rasterize_polygon(points, bounds)
        candidate_raster = _box_mask(box, bounds)
        intersection = int(np.count_nonzero(raster & candidate_raster))
        mask_area = int(np.count_nonzero(raster))
        candidate_area = int(np.count_nonzero(candidate_raster))
        rows.append(
            {
                "candidate_uuid": str(candidate.get("proposal_uuid", candidate.get("diagnostic_uuid", "unknown"))),
                "annotation_uuid": str(mask["annotation_uuid"]),
                "intersection_pixels": intersection,
                "visible_mask_pixels": mask_area,
                "candidate_box_pixels": candidate_area,
                "candidate_visible_mask_coverage": round(intersection / max(1, mask_area), 8),
                "candidate_box_coverage": round(intersection / max(1, candidate_area), 8),
                "mask_centroid_inside_candidate": _contains(
                    box,
                    (
                        sum(float(point["x"]) for point in points) / len(points),
                        sum(float(point["y"]) for point in points) / len(points),
                    ),
                ),
            }
        )
    return rows


def classify_dense_candidate(
    coverage_rows: Sequence[Mapping[str, Any]],
    *,
    specification: Mapping[str, Any] | None = None,
    duplicate_single_person: bool = False,
) -> dict[str, Any]:
    """Assign an evaluator-only dense truth class from visible-mask overlap."""

    spec = specification or dense_truth_classification_specification()
    minimum = int(spec["minimum_visible_pixel_area"])
    material = [
        row
        for row in coverage_rows
        if int(row["intersection_pixels"]) >= minimum
        and float(row["candidate_visible_mask_coverage"]) >= float(spec["mask_intersection_fraction_threshold"])
        and (
            bool(row["mask_centroid_inside_candidate"])
            or float(row["candidate_box_coverage"]) >= float(spec["clean_proposal_coverage_threshold"])
        )
    ]
    if len(material) >= 2:
        truth_class = "MERGED_MULTIPLE_PEOPLE"
    elif len(material) == 1 and duplicate_single_person:
        truth_class = "DUPLICATE_SINGLE_PERSON"
    elif len(material) == 1:
        row = material[0]
        truth_class = (
            "CLEAN_SINGLE_PERSON"
            if float(row["candidate_visible_mask_coverage"]) >= float(spec["clean_mask_coverage_threshold"])
            else "PARTIAL_SINGLE_PERSON"
        )
    elif not coverage_rows or max((int(row["intersection_pixels"]) for row in coverage_rows), default=0) == 0:
        truth_class = "BACKGROUND"
    else:
        truth_class = "AMBIGUOUS"
    return {
        "truth_class": truth_class,
        "material_annotation_uuids": sorted(str(row["annotation_uuid"]) for row in material),
        "material_person_count": len(material),
        "human_gold_evaluator_only": True,
    }


def validate_occlusion_graph(visible_masks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate references, front-order semantics and directed cycles."""

    rows = {str(mask["annotation_uuid"]): mask for mask in visible_masks}
    edges = []
    errors = []
    adjacency: dict[str, list[str]] = defaultdict(list)
    for target_id, mask in rows.items():
        occluder_id = mask.get("occluder_uuid")
        if not occluder_id:
            continue
        occluder_id = str(occluder_id)
        if occluder_id not in rows:
            errors.append({"type": "UNKNOWN_OCCLUDER", "target": target_id, "occluder": occluder_id})
            continue
        occluder_order = int(rows[occluder_id].get("occlusion_order", 0))
        target_order = int(mask.get("occlusion_order", 0))
        if occluder_order >= target_order:
            errors.append(
                {
                    "type": "OCCLUDER_NOT_IN_FRONT",
                    "target": target_id,
                    "occluder": occluder_id,
                    "occluder_order": occluder_order,
                    "target_order": target_order,
                }
            )
        adjacency[occluder_id].append(target_id)
        edges.append({"occluder_annotation_uuid": occluder_id, "occluded_annotation_uuid": target_id})

    visiting: set[str] = set()
    visited: set[str] = set()
    cycle_nodes: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            cycle_nodes.add(node)
            return
        if node in visited:
            return
        visiting.add(node)
        for child in adjacency[node]:
            visit(child)
            if child in cycle_nodes:
                cycle_nodes.add(node)
        visiting.remove(node)
        visited.add(node)

    for node in rows:
        visit(node)
    if cycle_nodes:
        errors.append({"type": "OCCLUSION_CYCLE", "annotation_uuids": sorted(cycle_nodes)})
    return {
        "valid": not errors,
        "edges": sorted(edges, key=stable_hash),
        "errors": errors,
        "cycle_detected": bool(cycle_nodes),
    }


def _g3_split_reasons(
    members: Sequence[Mapping[str, Any]], all_nodes: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    reasons = []
    for container in members:
        box = _box(container)
        container_height = max(1e-12, float(box["y2"]) - float(box["y1"]))
        alternatives = []
        for candidate in all_nodes:
            if candidate["source_view_family"] == container["source_view_family"]:
                continue
            if float(candidate["score"]) < 0.22 or not _contains(box, _centre(_box(candidate))):
                continue
            overlap = _footprint_intersection(container, candidate)
            if overlap is not None and _contains(overlap, _centre(_box(candidate))):
                alternatives.append(candidate)
        for index, left in enumerate(alternatives):
            for right in alternatives[index + 1 :]:
                iou = proposal_iou(left, right)
                separation = math.dist(_centre(_box(left)), _centre(_box(right))) / container_height
                if iou <= 0.15 and separation >= 0.35:
                    reasons.append(
                        {
                            "reason": "CROSS_VIEW_SPLIT_EVIDENCE",
                            "containing_proposal_uuid": str(container["proposal_uuid"]),
                            "split_proposal_uuids": sorted([str(left["proposal_uuid"]), str(right["proposal_uuid"])]),
                            "split_pair_iou": round(iou, 8),
                            "centre_separation_containing_height": round(separation, 8),
                        }
                    )
    return reasons


def _g3_multi_mode_reasons(members: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(members) < 2:
        return []
    heights = sorted(float(_box(member)["y2"]) - float(_box(member)["y1"]) for member in members)
    median_height = max(1e-12, heights[len(heights) // 2])
    reasons = []
    for index, left in enumerate(members):
        for right in members[index + 1 :]:
            if left["inference_view_id"] == right["inference_view_id"]:
                continue
            separation = math.dist(_bottom_centre(_box(left)), _bottom_centre(_box(right))) / median_height
            if separation >= 0.45:
                reasons.append(
                    {
                        "reason": "MULTI_MODE_CLUSTER",
                        "mode_proposal_uuids": sorted([str(left["proposal_uuid"]), str(right["proposal_uuid"])]),
                        "bottom_centre_separation_median_height": round(separation, 8),
                    }
                )
    return reasons


def _containment_disagreement_reasons(
    members: Sequence[Mapping[str, Any]], all_nodes: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    spec = eligibility_variant_specification()["containment_disagreement_thresholds"]
    reasons = []
    for container in members:
        container_box = _box(container)
        container_area = _box_area(container_box)
        container_height = max(1e-12, float(container_box["y2"]) - float(container_box["y1"]))
        smaller = []
        for candidate in all_nodes:
            candidate_box = _box(candidate)
            candidate_area = _box_area(candidate_box)
            if candidate["proposal_uuid"] == container["proposal_uuid"]:
                continue
            if float(candidate["score"]) < float(spec["minimum_score"]):
                continue
            if not _contains(container_box, _centre(candidate_box)):
                continue
            if container_area / max(1e-12, candidate_area) < float(spec["minimum_container_to_smaller_area_ratio"]):
                continue
            smaller.append(candidate)
        for index, left in enumerate(smaller):
            for right in smaller[index + 1 :]:
                if left["inference_view_id"] == right["inference_view_id"]:
                    continue
                iou = proposal_iou(left, right)
                separation = math.dist(_centre(_box(left)), _centre(_box(right))) / container_height
                if iou > float(spec["maximum_smaller_pair_iou"]):
                    continue
                if separation < float(spec["minimum_centre_separation_containing_height"]):
                    continue
                reasons.append(
                    {
                        "reason": "CONTAINMENT_DISAGREEMENT_EVIDENCE",
                        "containing_proposal_uuid": str(container["proposal_uuid"]),
                        "smaller_proposal_uuids": sorted([str(left["proposal_uuid"]), str(right["proposal_uuid"])]),
                        "distinct_view_or_tile_ids": sorted(
                            [str(left["inference_view_id"]), str(right["inference_view_id"])]
                        ),
                        "smaller_pair_iou": round(iou, 8),
                        "centre_separation_containing_height": round(separation, 8),
                    }
                )
    return reasons


def evaluate_eligibility_variants(
    members: Sequence[Mapping[str, Any]], all_nodes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Evaluate E0-E5 using proposal evidence only."""

    runtime_payload = {"members": list(members), "all_nodes": list(all_nodes)}
    assert_no_gold_runtime_leakage(runtime_payload)
    e1_reasons = _g3_split_reasons(members, all_nodes)
    e2_reasons = _g3_multi_mode_reasons(members)
    e3_reasons = _containment_disagreement_reasons(members, all_nodes)
    fired = {"E1": bool(e1_reasons), "E2": bool(e2_reasons), "E3": bool(e3_reasons)}
    variants = {
        "E0": fired["E1"] or fired["E2"],
        "E1": fired["E1"],
        "E2": fired["E2"],
        "E3": fired["E3"],
        "E4": any(fired.values()),
        "E5": sum(fired.values()) >= 2,
    }
    result = {
        "variant_routes": {
            variant: {
                "route": route,
                "output_state": "ROUTE_HUMAN_DENSE_REVIEW" if route else "ACCEPT_VISIBLE_INSTANCE",
            }
            for variant, route in variants.items()
        },
        "evidence": {"E1": e1_reasons, "E2": e2_reasons, "E3": e3_reasons},
        "runtime_input_hash": stable_hash(runtime_payload),
        "determinism_hash": stable_hash({"variants": variants, "evidence": [e1_reasons, e2_reasons, e3_reasons]}),
        "gold_runtime_leakage": False,
    }
    assert_no_gold_runtime_leakage(
        {
            "variant_routes": result["variant_routes"],
            "evidence": result["evidence"],
            "runtime_input_hash": result["runtime_input_hash"],
        }
    )
    return result


def binary_route_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate exact development counts without population confidence claims."""

    true_positive = sum(bool(row["truth_requires_route"]) and bool(row["route"]) for row in records)
    false_negative = sum(bool(row["truth_requires_route"]) and not bool(row["route"]) for row in records)
    false_positive = sum(not bool(row["truth_requires_route"]) and bool(row["route"]) for row in records)
    true_negative = sum(not bool(row["truth_requires_route"]) and not bool(row["route"]) for row in records)
    positive = true_positive + false_negative
    negative = false_positive + true_negative
    return {
        "record_count": len(records),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "exact_route_recall": round(true_positive / max(1, positive), 8),
        "exact_false_route_rate": round(false_positive / max(1, negative), 8),
        "population_confidence_interval_reported": False,
    }


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Return binary mask IoU."""

    union = int(np.count_nonzero(left | right))
    return int(np.count_nonzero(left & right)) / max(1, union)


def consolidate_mask_outputs(
    outputs: Sequence[Mapping[str, Any]], *, specification: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Deduplicate real promptable masks without averaging or identity cues."""

    spec = specification or mask_output_consolidation_specification()
    ordered = sorted(outputs, key=lambda row: (-float(row.get("score", 0.0)), str(row["output_uuid"])))
    accepted: list[dict[str, Any]] = []
    result: list[dict[str, Any]] = []
    for output in ordered:
        mask = np.asarray(output["binary_mask"], dtype=bool)
        area = int(np.count_nonzero(mask))
        state = "ACCEPT_VISIBLE_INSTANCE"
        reason = None
        if not bool(output.get("current_frame_pixel_support", False)) or area < int(
            spec["minimum_current_frame_pixel_area"]
        ):
            state = "REJECT_BACKGROUND_MASK"
            reason = "NO_CURRENT_FRAME_PIXEL_SUPPORT_OR_AREA_BELOW_MINIMUM"
        elif bool(output.get("runtime_merged_risk", False)):
            state = "ROUTE_HUMAN_DENSE_REVIEW"
            reason = "PROPOSAL_ONLY_MERGED_RISK"
        else:
            for prior in accepted:
                prior_mask = np.asarray(prior["binary_mask"], dtype=bool)
                overlap = int(np.count_nonzero(mask & prior_mask))
                containment = overlap / max(1, min(area, int(np.count_nonzero(prior_mask))))
                if mask_iou(mask, prior_mask) >= float(spec["duplicate_mask_iou_threshold"]) or containment >= float(
                    spec["duplicate_containment_threshold"]
                ):
                    state = "SUPPRESS_DUPLICATE_MASK"
                    reason = f"DUPLICATE_OF:{prior['output_uuid']}"
                    break
        row = {
            key: value for key, value in output.items() if key not in {"binary_mask", "evaluation_gold_intersections"}
        }
        row.update({"output_state": state, "reason": reason, "mask_area_pixels": area})
        result.append(row)
        if state == "ACCEPT_VISIBLE_INSTANCE":
            accepted.append(dict(output))
    return sorted(result, key=lambda row: str(row["output_uuid"]))
