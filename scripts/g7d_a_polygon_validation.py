"""Canonical source-coordinate polygon validation for the G7D-A reviewer."""

from __future__ import annotations

import math
from typing import Any

EPSILON = 1e-6


def failure(code: str, field: str, message: str, details: dict[str, Any], location: Any = None) -> dict[str, Any]:
    return {
        "ok": False,
        "error_code": code,
        "field": field,
        "message": message,
        "details": details,
        "vertex_index_or_edge_pair": location,
    }


def normalize_client_vertices(points: list[object], closed: bool) -> tuple[list[list[float]], dict[str, Any]]:
    """Apply only mechanical R5 canonicalization; never reorder or move vertices."""
    normalized: list[list[float]] = []
    removed_adjacent: list[int] = []
    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return [], {"normalization_error": f"vertex {index} is not an x/y pair"}
        try:
            candidate = [float(point[0]), float(point[1])]
        except (TypeError, ValueError):
            return [], {"normalization_error": f"vertex {index} is not numeric"}
        if not all(math.isfinite(value) for value in candidate):
            return [], {"normalization_error": f"vertex {index} is not finite"}
        if normalized and candidate == normalized[-1]:
            removed_adjacent.append(index)
        else:
            normalized.append(candidate)
    terminal_removed = bool(closed and len(normalized) > 1 and normalized[0] == normalized[-1])
    if terminal_removed:
        normalized.pop()
    return normalized, {
        "closure_convention": "distinct_vertices_once_plus_closed_true",
        "removed_exact_adjacent_vertex_indices": removed_adjacent,
        "removed_exact_terminal_duplicate": terminal_removed,
    }


def orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a: tuple[float, float], b: tuple[float, float], point: tuple[float, float]) -> bool:
    return (
        min(a[0], b[0]) - EPSILON <= point[0] <= max(a[0], b[0]) + EPSILON
        and min(a[1], b[1]) - EPSILON <= point[1] <= max(a[1], b[1]) + EPSILON
        and abs(orient(a, b, point)) <= EPSILON
    )


def segments_intersect(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]
) -> bool:
    values = (orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b))
    signs = tuple(1 if value > EPSILON else -1 if value < -EPSILON else 0 for value in values)
    if signs[0] * signs[1] < 0 and signs[2] * signs[3] < 0:
        return True
    return (
        (signs[0] == 0 and on_segment(a, b, c))
        or (signs[1] == 0 and on_segment(a, b, d))
        or (signs[2] == 0 and on_segment(c, d, a))
        or (signs[3] == 0 and on_segment(c, d, b))
    )


def validate_canonical_polygon(
    points: object, closed: object, width: object, height: object, field: str = "first_half_polygon_source_xy"
) -> dict[str, Any]:
    if closed is not True:
        return failure("INVALID_CLOSURE", "first_half_closed", "closed must be true", {"expected": True})
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        return failure(
            "INVALID_SOURCE_DIMENSIONS", "source_dimensions", "source dimensions must be positive integers", {}
        )
    if not isinstance(points, list):
        return failure("INVALID_VERTEX_ARRAY", field, "vertices_source_xy must be an array", {})
    if len(points) < 4:
        return failure(
            "TOO_FEW_VERTICES", field, "at least four distinct vertices are required", {"count": len(points)}
        )
    vertices: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        if not isinstance(point, list) or len(point) != 2 or any(isinstance(value, bool) for value in point):
            return failure("INVALID_VERTEX", field, f"vertex {index} must be a numeric x/y pair", {}, index)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in point):
            return failure("NON_FINITE_VERTEX", field, f"vertex {index} must contain finite numbers", {}, index)
        x, y = float(point[0]), float(point[1])
        if not 0 <= x <= width or not 0 <= y <= height:
            return failure(
                "OUT_OF_BOUNDS_VERTEX",
                field,
                f"vertex {index} is outside source bounds",
                {"source_width": width, "source_height": height, "x": x, "y": y},
                index,
            )
        vertices.append((x, y))
    if vertices[0] == vertices[-1]:
        return failure(
            "DUPLICATE_TERMINAL_VERTEX",
            field,
            "first vertex must not be duplicated at the end when closed=true",
            {"closure_convention": "distinct_vertices_once_plus_closed_true"},
            [0, len(vertices) - 1],
        )
    for index, (left, right) in enumerate(zip(vertices, vertices[1:] + vertices[:1])):
        if math.dist(left, right) <= EPSILON:
            return failure(
                "DUPLICATE_OR_ZERO_LENGTH_EDGE",
                field,
                f"duplicate adjacent vertices at {index} and {(index + 1) % len(vertices)}",
                {"epsilon": EPSILON},
                [index, (index + 1) % len(vertices)],
            )
    for left_index, (a, b) in enumerate(zip(vertices, vertices[1:] + vertices[:1])):
        for right_index, (c, d) in enumerate(zip(vertices, vertices[1:] + vertices[:1])):
            if right_index <= left_index:
                continue
            if (right_index - left_index) in (1, len(vertices) - 1):
                continue
            if segments_intersect(a, b, c, d):
                return failure(
                    "SELF_INTERSECTION",
                    field,
                    f"edge {left_index} intersects edge {right_index}",
                    {"epsilon": EPSILON},
                    [left_index, right_index],
                )
    signed_area = (
        sum(
            vertex[0] * vertices[(index + 1) % len(vertices)][1] - vertices[(index + 1) % len(vertices)][0] * vertex[1]
            for index, vertex in enumerate(vertices)
        )
        / 2
    )
    if abs(signed_area) <= EPSILON:
        return failure("ZERO_AREA", field, "polygon area must be positive in magnitude", {"signed_area": signed_area})
    return {
        "ok": True,
        "error_code": None,
        "field": field,
        "message": "canonical polygon accepted",
        "details": {
            "vertex_count": len(vertices),
            "signed_area": signed_area,
            "winding": "CCW" if signed_area > 0 else "CW",
            "epsilon": EPSILON,
        },
        "vertex_index_or_edge_pair": None,
    }
