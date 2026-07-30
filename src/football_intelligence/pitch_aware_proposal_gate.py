"""Deterministic source-coordinate pitch-aware proposal gate primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

SANDBOX_DECISIONS = {"KEEP", "SUPPRESS_SANDBOX", "BOUNDARY_REVIEW", "EXCEPTION_KEEP"}
RUNTIME_INPUT_FIELDS = {
    "source_box_xyxy",
    "approximate_footpoint_xy",
    "source_width",
    "source_height",
    "perspective_band",
    "proposal_provenance",
}
FORBIDDEN_RUNTIME_FIELDS = {
    "canonical_decision",
    "analysis_flags",
    "human_label",
    "role",
    "participation",
    "pitch_state",
    "box_quality",
    "proposal_validity",
}


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Return whether a point is inside a simple polygon using ray casting."""
    x, y = map(float, point)
    inside = False
    for index, vertex in enumerate(polygon):
        previous = polygon[index - 1]
        x1, y1 = map(float, previous)
        x2, y2 = map(float, vertex)
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
    return inside


def point_segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    """Return Euclidean point-to-segment distance in source pixels."""
    px, py = map(float, point)
    x1, y1 = map(float, start)
    x2, y2 = map(float, end)
    dx, dy = x2 - x1, y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.dist((px, py), (x1, y1))
    projection = ((px - x1) * dx + (py - y1) * dy) / length_squared
    t = max(0.0, min(1.0, projection))
    return math.dist((px, py), (x1 + t * dx, y1 + t * dy))


def nearest_boundary(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> tuple[float, int, str]:
    """Return distance, segment index, and conservative pitch-boundary class."""
    candidates = []
    for index, end in enumerate(polygon):
        start = polygon[index - 1]
        distance = point_segment_distance(point, start, end)
        dx = abs(float(end[0]) - float(start[0]))
        dy = abs(float(end[1]) - float(start[1]))
        boundary_type = "TOUCHLINE" if dx >= 1.5 * dy else "GOAL_LINE"
        candidates.append((distance, index, boundary_type))
    return min(candidates, key=lambda row: (row[0], row[1]))


def segments_intersect(
    first_start: Sequence[float],
    first_end: Sequence[float],
    second_start: Sequence[float],
    second_end: Sequence[float],
) -> bool:
    """Return whether two closed line segments intersect."""

    def orientation(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
        return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (float(b[1]) - float(a[1])) * (
            float(c[0]) - float(a[0])
        )

    def on_segment(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> bool:
        return (
            min(float(a[0]), float(c[0])) - 1e-9 <= float(b[0]) <= max(float(a[0]), float(c[0])) + 1e-9
            and min(float(a[1]), float(c[1])) - 1e-9 <= float(b[1]) <= max(float(a[1]), float(c[1])) + 1e-9
        )

    o1 = orientation(first_start, first_end, second_start)
    o2 = orientation(first_start, first_end, second_end)
    o3 = orientation(second_start, second_end, first_start)
    o4 = orientation(second_start, second_end, first_end)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    return any(
        abs(value) <= 1e-9 and on_segment(a, b, c)
        for value, a, b, c in (
            (o1, first_start, second_start, first_end),
            (o2, first_start, second_end, first_end),
            (o3, second_start, first_start, second_end),
            (o4, second_start, first_end, second_end),
        )
    )


def polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    """Return the unsigned area of a polygon."""
    return (
        abs(
            sum(
                float(polygon[index - 1][0]) * float(vertex[1]) - float(vertex[0]) * float(polygon[index - 1][1])
                for index, vertex in enumerate(polygon)
            )
        )
        / 2
    )


def clip_polygon_to_box(polygon: Sequence[Sequence[float]], box: Sequence[float]) -> list[tuple[float, float]]:
    """Clip a polygon to an axis-aligned box with Sutherland-Hodgman."""
    x1, y1, x2, y2 = map(float, box)
    output = [(float(x), float(y)) for x, y in polygon]

    def clip(
        points: list[tuple[float, float]],
        inside: Any,
        intersect: Any,
    ) -> list[tuple[float, float]]:
        if not points:
            return []
        result: list[tuple[float, float]] = []
        previous = points[-1]
        previous_inside = inside(previous)
        for current in points:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    result.append(intersect(previous, current))
                result.append(current)
            elif previous_inside:
                result.append(intersect(previous, current))
            previous, previous_inside = current, current_inside
        return result

    def vertical(boundary: float, start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
        if end[0] == start[0]:
            return boundary, start[1]
        t = (boundary - start[0]) / (end[0] - start[0])
        return boundary, start[1] + t * (end[1] - start[1])

    def horizontal(boundary: float, start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
        if end[1] == start[1]:
            return start[0], boundary
        t = (boundary - start[1]) / (end[1] - start[1])
        return start[0] + t * (end[0] - start[0]), boundary

    output = clip(output, lambda p: p[0] >= x1, lambda a, b: vertical(x1, a, b))
    output = clip(output, lambda p: p[0] <= x2, lambda a, b: vertical(x2, a, b))
    output = clip(output, lambda p: p[1] >= y1, lambda a, b: horizontal(y1, a, b))
    return clip(output, lambda p: p[1] <= y2, lambda a, b: horizontal(y2, a, b))


def box_polygon_intersection_area(box: Sequence[float], polygon: Sequence[Sequence[float]]) -> float:
    return polygon_area(clip_polygon_to_box(polygon, box))


def adaptive_boundary_band(
    box_height: float,
    perspective_band: str,
    fixed_pixels: float,
    alpha: float,
    band_mode: str,
    expected_heights: Mapping[str, float],
) -> float:
    """Compute one of the predeclared bounded adaptive band forms."""
    if band_mode == "FIXED_PIXELS":
        return float(fixed_pixels)
    if band_mode == "BOX_HEIGHT":
        return max(float(fixed_pixels), float(alpha) * float(box_height))
    if band_mode == "EXPECTED_HEIGHT_BY_PERSPECTIVE":
        expected = float(expected_heights.get(perspective_band, expected_heights.get("UNKNOWN", box_height)))
        return max(float(fixed_pixels), float(alpha) * expected)
    raise ValueError(f"unsupported band mode: {band_mode}")


def candidate_geometry(
    runtime_candidate: Mapping[str, Any],
    polygon: Sequence[Sequence[float]],
    band_pixels: float,
) -> dict[str, Any]:
    """Measure one candidate entirely in source-image coordinates."""
    if FORBIDDEN_RUNTIME_FIELDS.intersection(runtime_candidate):
        raise ValueError("human-label field supplied to runtime geometry")
    box = list(map(float, runtime_candidate["source_box_xyxy"]))
    x1, y1, x2, y2 = box
    if not all(math.isfinite(value) for value in box) or not (x1 < x2 and y1 < y2):
        raise ValueError("invalid source box")
    width = float(runtime_candidate["source_width"])
    height = float(runtime_candidate["source_height"])
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError("source box outside source bounds")
    footpoint = tuple(map(float, runtime_candidate["approximate_footpoint_xy"]))
    inside = point_in_polygon(footpoint, polygon)
    distance, segment_index, boundary_type = nearest_boundary(footpoint, polygon)
    signed_distance = -distance if inside else distance
    box_area = (x2 - x1) * (y2 - y1)
    overlap_area = box_polygon_intersection_area(box, polygon)
    lower_edge = ((x1, y2), (x2, y2))
    lower_edge_crosses = any(
        segments_intersect(lower_edge[0], lower_edge[1], polygon[index - 1], vertex)
        for index, vertex in enumerate(polygon)
    )
    lower_edge_distance = min(
        nearest_boundary(lower_edge[0], polygon)[0],
        nearest_boundary(((x1 + x2) / 2, y2), polygon)[0],
        nearest_boundary(lower_edge[1], polygon)[0],
    )
    strip_height = max(1.0, 0.2 * (y2 - y1))
    bottom_strip = [x1, y2 - strip_height, x2, y2]
    bottom_strip_overlap = box_polygon_intersection_area(bottom_strip, polygon)
    uncertainty_radius = max(2.0, 0.15 * (x2 - x1), 0.05 * (y2 - y1))
    if inside:
        geometry_band = "INSIDE"
    elif distance <= band_pixels:
        geometry_band = "NEAR_BOUNDARY"
    elif distance <= 3 * band_pixels:
        geometry_band = "OUTSIDE"
    else:
        geometry_band = "FAR_OUTSIDE"
    return {
        "signed_footpoint_distance_pixels": signed_distance,
        "absolute_boundary_distance_pixels": distance,
        "inside_polygon": inside,
        "geometry_band": geometry_band,
        "adaptive_boundary_band_pixels": float(band_pixels),
        "nearest_boundary_segment_index": segment_index,
        "nearest_boundary_type": boundary_type,
        "box_polygon_intersection_pixels": overlap_area,
        "box_polygon_intersection_ratio": overlap_area / box_area,
        "lower_edge_crosses_polygon": lower_edge_crosses,
        "lower_edge_boundary_distance_pixels": lower_edge_distance,
        "bottom_strip_polygon_intersection_pixels": bottom_strip_overlap,
        "footpoint_uncertainty_radius_pixels": uncertainty_radius,
        "footpoint_uncertainty_intersects_polygon": inside or distance <= uncertainty_radius,
    }


def gate_decision(family: str, geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a deterministic implementable G0-G4 gate to measured geometry."""
    inside = bool(geometry["inside_polygon"])
    distance = float(geometry["absolute_boundary_distance_pixels"])
    band = float(geometry["adaptive_boundary_band_pixels"])
    overlap_ratio = float(geometry["box_polygon_intersection_ratio"])
    bottom_overlap = float(geometry["bottom_strip_polygon_intersection_pixels"])
    if family == "G0_KEEP_ALL":
        return {"decision": "KEEP", "reason_codes": ["KEEP_ALL_CONTROL"]}
    if family == "G1_STRICT_INSIDE":
        return {
            "decision": "KEEP" if inside else "SUPPRESS_SANDBOX",
            "reason_codes": ["FOOTPOINT_INSIDE" if inside else "STRICT_OUTSIDE_NEGATIVE_CONTROL"],
        }
    if family == "G2_INSIDE_OR_ADAPTIVE_BOUNDARY":
        if inside:
            return {"decision": "KEEP", "reason_codes": ["FOOTPOINT_INSIDE"]}
        if distance <= band:
            return {"decision": "BOUNDARY_REVIEW", "reason_codes": ["WITHIN_ADAPTIVE_BOUNDARY_BAND"]}
        return {"decision": "SUPPRESS_SANDBOX", "reason_codes": ["OUTSIDE_ADAPTIVE_BOUNDARY_BAND"]}
    if family not in {"G3_CONSERVATIVE_FAR_OUTSIDE", "G4_GEOMETRIC_EXCEPTION_GATE"}:
        raise ValueError(f"unsupported implementable family: {family}")
    if inside:
        return {"decision": "KEEP", "reason_codes": ["FOOTPOINT_INSIDE"]}
    if distance <= 3 * band:
        return {"decision": "BOUNDARY_REVIEW", "reason_codes": ["CONSERVATIVE_BOUNDARY_CORRIDOR"]}
    exceptions = []
    if overlap_ratio > 0:
        exceptions.append("BOX_INTERSECTS_POLYGON")
    if geometry["lower_edge_crosses_polygon"] or float(geometry["lower_edge_boundary_distance_pixels"]) <= band:
        exceptions.append("LOWER_EDGE_CROSSES_BOUNDARY_BAND")
    if geometry["footpoint_uncertainty_intersects_polygon"]:
        exceptions.append("FOOTPOINT_UNCERTAINTY_INTERSECTS_POLYGON")
    if family == "G4_GEOMETRIC_EXCEPTION_GATE":
        boundary_type = geometry["nearest_boundary_type"]
        if boundary_type == "TOUCHLINE" and distance <= 4 * band:
            exceptions.append("ASSISTANT_REFEREE_TOUCHLINE_CORRIDOR")
        if boundary_type == "GOAL_LINE" and distance <= 4 * band:
            exceptions.append("BEHIND_GOAL_RELEVANT_CORRIDOR")
    if exceptions:
        return {"decision": "EXCEPTION_KEEP", "reason_codes": sorted(set(exceptions))}
    if bottom_overlap == 0 and overlap_ratio < 0.05:
        return {"decision": "SUPPRESS_SANDBOX", "reason_codes": ["CONSERVATIVE_FAR_OUTSIDE"]}
    return {"decision": "KEEP", "reason_codes": ["PLAYABLE_SURFACE_OVERLAP"]}


def runtime_decide(
    family: str,
    runtime_candidate: Mapping[str, Any],
    polygon: Sequence[Sequence[float]],
    parameter: Mapping[str, Any],
    expected_heights: Mapping[str, float],
) -> dict[str, Any]:
    """Measure and decide without accepting any human-label field."""
    clean = {key: runtime_candidate[key] for key in RUNTIME_INPUT_FIELDS}
    box = clean["source_box_xyxy"]
    band = adaptive_boundary_band(
        float(box[3]) - float(box[1]),
        str(clean["perspective_band"]),
        float(parameter["fixed_pixels"]),
        float(parameter["alpha"]),
        str(parameter["band_mode"]),
        expected_heights,
    )
    geometry = candidate_geometry(clean, polygon, band)
    return {**gate_decision(family, geometry), "geometry": geometry}
