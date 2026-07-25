"""Source-coordinate pitch gating and Player Observation v1 materialization.

The runtime functions in this module deliberately accept proposal and mask
geometry only. Human pitch states, roles, and footpoints belong to evaluation
code and are rejected if they appear in a runtime payload.
"""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from football_intelligence.review_chassis.hashing import stable_hash

PLAYER_OBSERVATION_SCHEMA_VERSION = "football_intelligence.player_observation.v1"
PITCH_POLYGON_TOLERANCE_PIXELS = 10.0

ObservationState = Literal[
    "OBSERVED_BOX",
    "OBSERVED_MASK",
    "ROUTE_DENSE_REVIEW",
    "ROUTE_PITCH_BOUNDARY_REVIEW",
    "UNRESOLVED",
    "REJECT_BACKGROUND",
]
PitchRelation = Literal["UNGATED_RETAIN", "ON_PITCH", "OFF_PITCH", "BOUNDARY_UNCERTAIN"]

FORBIDDEN_RUNTIME_KEYS = {
    "annotation_uuid",
    "coarse_role",
    "evaluator",
    "gold",
    "human_footpoint",
    "human_pitch_state",
    "human_role",
    "identity",
    "pitch_state",
    "role",
    "track_id",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObservationPoint(StrictModel):
    x: float
    y: float

    @model_validator(mode="after")
    def finite(self) -> ObservationPoint:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("point coordinates must be finite")
        return self


class ObservationBox(StrictModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def ordered(self) -> ObservationBox:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("box coordinates must be finite")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must have positive area")
        return self


class FootpointUncertaintyRegion(StrictModel):
    region_type: Literal["POINT", "HORIZONTAL_INTERVAL", "MASK_CONTACT_INTERVAL"]
    x1: float
    y1: float
    x2: float
    y2: float
    support_points: list[ObservationPoint] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_region(self) -> FootpointUncertaintyRegion:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("uncertainty region coordinates must be finite")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("uncertainty region must be ordered")
        return self


class PlayerObservationV1(StrictModel):
    schema_version: Literal[PLAYER_OBSERVATION_SCHEMA_VERSION] = PLAYER_OBSERVATION_SCHEMA_VERSION
    observation_uuid: str
    source_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_index: int = Field(ge=0)
    source_view_ids: list[str] = Field(min_length=1)
    proposal_uuid_lineage: list[str] = Field(min_length=1)
    cluster_uuid: str
    visible_box: ObservationBox
    optional_visible_mask: list[ObservationPoint] | None = None
    geometry_method: Literal["F0", "F1", "F2", "F3"]
    footpoint_estimate: ObservationPoint
    footpoint_method: Literal[
        "BOX_BOTTOM_CENTRE",
        "BOX_LOWER_CONTACT_INTERVAL",
        "MASK_LOWER_CONTACT",
        "HYBRID_MASK_LOWER_CONTACT",
        "HYBRID_BOX_LOWER_CONTACT_INTERVAL",
    ]
    footpoint_uncertainty_region: FootpointUncertaintyRegion
    pitch_relation: PitchRelation
    pitch_gate_variant: Literal["P0", "P1", "P2", "P3", "P4"]
    observation_state: ObservationState
    duplicate_risk: Literal["NONE", "SUPPRESSED_LINEAGE_PRESENT", "UNRESOLVED"]
    merged_risk: Literal["NONE", "PROPOSAL_GEOMETRY_RISK", "UNRESOLVED"]
    dense_branch_state: Literal[
        "NOT_TRIGGERED",
        "FROZEN_TRIGGER_ACCEPTED_MASK",
        "FROZEN_TRIGGER_ROUTED",
        "FROZEN_TRIGGER_FAILED",
    ]
    review_route_reason: str | None = None
    role_state: Literal["UNKNOWN"] = "UNKNOWN"
    role_source: Literal["NO_FROZEN_RUNTIME_ROLE_COMPONENT"] = "NO_FROZEN_RUNTIME_ROLE_COMPONENT"
    provenance_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def enforce_observed_state(self) -> PlayerObservationV1:
        if self.observation_state == "OBSERVED_MASK" and not self.optional_visible_mask:
            raise ValueError("OBSERVED_MASK requires current-frame mask geometry")
        if self.observation_state == "ROUTE_PITCH_BOUNDARY_REVIEW" and self.pitch_relation != "BOUNDARY_UNCERTAIN":
            raise ValueError("pitch-boundary review requires an uncertain pitch relation")
        return self


def footpoint_geometry_variant_specification() -> dict[str, Any]:
    """Return constants frozen before evaluator results are loaded."""

    return {
        "schema_version": "football_intelligence.m5_5g6a.footpoint_geometry_variants.v1",
        "frozen_before_scoring": True,
        "runtime_coordinate_space": "SOURCE_PANORAMA_PIXELS",
        "human_footpoints_runtime_forbidden": True,
        "variants": {
            "F0": {
                "method": "BOX_BOTTOM_CENTRE",
                "uncertainty_region": "POINT",
            },
            "F1": {
                "method": "BOX_LOWER_CONTACT_INTERVAL",
                "horizontal_half_width_fraction_of_box": 0.18,
                "vertical_band_fraction_of_box_height": 0.05,
                "minimum_horizontal_half_width_pixels": 2.0,
                "minimum_vertical_band_pixels": 1.0,
            },
            "F2": {
                "method": "MASK_LOWER_CONTACT",
                "lower_mask_fraction": 0.18,
                "robust_horizontal_quantiles": [0.10, 0.90],
                "robust_y_quantile": 0.90,
                "minimum_uncertainty_pixels": 1.0,
                "fragmentation_penalty_pixels_per_extra_component": 2.0,
                "requires_frozen_accepted_reliable_mask": True,
            },
            "F3": {
                "method": "HYBRID",
                "accepted_reliable_mask": "F2",
                "fallback": "F1",
            },
        },
        "predicted_or_temporally_carried_geometry_forbidden": True,
    }


def pitch_gate_variant_specification() -> dict[str, Any]:
    """Return the immutable P0-P4 source-coordinate gate specification."""

    return {
        "schema_version": "football_intelligence.m5_5g6a.pitch_gate_variants.v1",
        "frozen_before_scoring": True,
        "runtime_coordinate_space": "SOURCE_PANORAMA_PIXELS",
        "approved_polygon_tolerance_pixels": PITCH_POLYGON_TOLERANCE_PIXELS,
        "boundary_margin_source": "EXISTING_APPROVED_POLYGON_TOLERANCE",
        "boundary_width_tuned_on_c2": False,
        "variants": {
            "P0": {"geometry": "F0", "decision": "RETAIN_UNGATED"},
            "P1": {"geometry": "F0", "decision": "HARD_POINT_CONTAINMENT", "boundary_margin_pixels": 0.0},
            "P2": {
                "geometry": "F1",
                "decision": "CONSERVATIVE_UNCERTAINTY_INTERSECTION",
                "boundary_margin_pixels": PITCH_POLYGON_TOLERANCE_PIXELS,
            },
            "P3": {
                "geometry": "F3",
                "decision": "CONSERVATIVE_UNCERTAINTY_INTERSECTION",
                "boundary_margin_pixels": PITCH_POLYGON_TOLERANCE_PIXELS,
            },
            "P4": {
                "geometry": ["F1", "F3"],
                "decision": "P2_P3_HARD_CLASSIFICATION_AGREEMENT_ELSE_BOUNDARY",
                "boundary_margin_pixels": PITCH_POLYGON_TOLERANCE_PIXELS,
            },
        },
        "human_pitch_state_runtime_forbidden": True,
        "boundary_performance_claimed": False,
    }


def _walk_forbidden_runtime_keys(value: Any, path: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            current = f"{path}.{key}" if path else str(key)
            lowered = str(key).lower()
            if lowered in FORBIDDEN_RUNTIME_KEYS or lowered.startswith("human_") or lowered.startswith("gold_"):
                findings.append(current)
            findings.extend(_walk_forbidden_runtime_keys(nested, current))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            findings.extend(_walk_forbidden_runtime_keys(nested, f"{path}[{index}]"))
    return findings


def validate_runtime_payload(value: Any) -> None:
    """Reject accidental evaluator truth or identity inputs."""

    findings = _walk_forbidden_runtime_keys(value)
    if findings:
        raise ValueError(f"forbidden evaluator/runtime keys: {sorted(findings)}")


def panorama_to_focal(point: Mapping[str, float], bounds: Mapping[str, float]) -> dict[str, float]:
    return {"x": float(point["x"]) - float(bounds["x1"]), "y": float(point["y"]) - float(bounds["y1"])}


def focal_to_panorama(point: Mapping[str, float], bounds: Mapping[str, float]) -> dict[str, float]:
    return {"x": float(point["x"]) + float(bounds["x1"]), "y": float(point["y"]) + float(bounds["y1"])}


def _clip_against_boundary(
    points: list[dict[str, float]],
    inside: Any,
    intersect: Any,
) -> list[dict[str, float]]:
    if not points:
        return []
    output: list[dict[str, float]] = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersect(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersect(previous, current))
        previous, previous_inside = current, current_inside
    return output


def clip_polygon_to_bounds(
    polygon: Sequence[Mapping[str, float]], bounds: Mapping[str, float]
) -> list[dict[str, float]]:
    """Clip a source-panorama polygon to a crop rectangle."""

    points = [{"x": float(point["x"]), "y": float(point["y"])} for point in polygon]
    x1, y1, x2, y2 = (float(bounds[key]) for key in ("x1", "y1", "x2", "y2"))

    def vertical(boundary: float, left: bool) -> tuple[Any, Any]:
        inside = (lambda point: point["x"] >= boundary) if left else (lambda point: point["x"] <= boundary)

        def intersect(start: Mapping[str, float], end: Mapping[str, float]) -> dict[str, float]:
            delta = float(end["x"]) - float(start["x"])
            ratio = 0.0 if abs(delta) < 1e-12 else (boundary - float(start["x"])) / delta
            return {"x": boundary, "y": float(start["y"]) + ratio * (float(end["y"]) - float(start["y"]))}

        return inside, intersect

    def horizontal(boundary: float, top: bool) -> tuple[Any, Any]:
        inside = (lambda point: point["y"] >= boundary) if top else (lambda point: point["y"] <= boundary)

        def intersect(start: Mapping[str, float], end: Mapping[str, float]) -> dict[str, float]:
            delta = float(end["y"]) - float(start["y"])
            ratio = 0.0 if abs(delta) < 1e-12 else (boundary - float(start["y"])) / delta
            return {"x": float(start["x"]) + ratio * (float(end["x"]) - float(start["x"])), "y": boundary}

        return inside, intersect

    for inside, intersect in (vertical(x1, True), vertical(x2, False), horizontal(y1, True), horizontal(y2, False)):
        points = _clip_against_boundary(points, inside, intersect)
    return points


def _point_on_segment(point: Mapping[str, float], start: Mapping[str, float], end: Mapping[str, float]) -> bool:
    cross = (float(point["y"]) - float(start["y"])) * (float(end["x"]) - float(start["x"])) - (
        float(point["x"]) - float(start["x"])
    ) * (float(end["y"]) - float(start["y"]))
    if abs(cross) > 1e-7:
        return False
    return (
        min(float(start["x"]), float(end["x"])) - 1e-7
        <= float(point["x"])
        <= max(float(start["x"]), float(end["x"])) + 1e-7
        and min(float(start["y"]), float(end["y"])) - 1e-7
        <= float(point["y"])
        <= max(float(start["y"]), float(end["y"])) + 1e-7
    )


def point_in_polygon(point: Mapping[str, float], polygon: Sequence[Mapping[str, float]]) -> bool:
    """Boundary-inclusive ray-casting point containment."""

    x, y = float(point["x"]), float(point["y"])
    inside = False
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment(point, start, end):
            return True
        y1, y2 = float(start["y"]), float(end["y"])
        if (y1 > y) == (y2 > y):
            continue
        intersection_x = float(start["x"]) + (y - y1) * (float(end["x"]) - float(start["x"])) / (y2 - y1)
        if intersection_x > x:
            inside = not inside
    return inside


def _distance_to_segment(point: Mapping[str, float], start: Mapping[str, float], end: Mapping[str, float]) -> float:
    px, py = float(point["x"]), float(point["y"])
    x1, y1, x2, y2 = float(start["x"]), float(start["y"]), float(end["x"]), float(end["y"])
    dx, dy = x2 - x1, y2 - y1
    if dx * dx + dy * dy <= 1e-12:
        return math.hypot(px - x1, py - y1)
    position = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + position * dx), py - (y1 + position * dy))


def signed_distance_to_polygon(point: Mapping[str, float], polygon: Sequence[Mapping[str, float]]) -> float:
    distance = min(
        _distance_to_segment(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )
    return distance if point_in_polygon(point, polygon) else -distance


def _orientation(first: Mapping[str, float], second: Mapping[str, float], third: Mapping[str, float]) -> float:
    return (float(second["x"]) - float(first["x"])) * (float(third["y"]) - float(first["y"])) - (
        float(second["y"]) - float(first["y"])
    ) * (float(third["x"]) - float(first["x"]))


def _segments_intersect(
    left_start: Mapping[str, float],
    left_end: Mapping[str, float],
    right_start: Mapping[str, float],
    right_end: Mapping[str, float],
) -> bool:
    values = (
        _orientation(left_start, left_end, right_start),
        _orientation(left_start, left_end, right_end),
        _orientation(right_start, right_end, left_start),
        _orientation(right_start, right_end, left_end),
    )
    if values[0] * values[1] < 0 and values[2] * values[3] < 0:
        return True
    return any(
        abs(orientation) <= 1e-9 and _point_on_segment(point, start, end)
        for orientation, point, start, end in (
            (values[0], right_start, left_start, left_end),
            (values[1], right_end, left_start, left_end),
            (values[2], left_start, right_start, right_end),
            (values[3], left_end, right_start, right_end),
        )
    )


def _box_values(box: Mapping[str, Any]) -> tuple[float, float, float, float]:
    values = tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2"))
    if not all(math.isfinite(value) for value in values) or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("invalid source-panorama box")
    return values


def estimate_footpoint(
    variant: Literal["F0", "F1", "F2", "F3"],
    box: Mapping[str, Any],
    *,
    mask_pixels: Sequence[Mapping[str, float]] | None = None,
    mask_reliable: bool = False,
    mask_component_count: int = 1,
) -> dict[str, Any]:
    """Create a runtime-only source-coordinate footpoint estimate."""

    x1, y1, x2, y2 = _box_values(box)
    width, height = x2 - x1, y2 - y1
    if variant == "F3":
        selected = "F2" if mask_reliable and mask_pixels else "F1"
        result = estimate_footpoint(
            selected,
            box,
            mask_pixels=mask_pixels,
            mask_reliable=mask_reliable,
            mask_component_count=mask_component_count,
        )
        result["geometry_method"] = "F3"
        result["footpoint_method"] = (
            "HYBRID_MASK_LOWER_CONTACT" if selected == "F2" else "HYBRID_BOX_LOWER_CONTACT_INTERVAL"
        )
        return result
    if variant == "F2":
        if not mask_reliable or not mask_pixels:
            raise ValueError("F2 requires a frozen accepted reliable current-frame mask")
        points = [{"x": float(point["x"]), "y": float(point["y"])} for point in mask_pixels]
        minimum_y, maximum_y = min(point["y"] for point in points), max(point["y"] for point in points)
        cutoff = maximum_y - 0.18 * max(1.0, maximum_y - minimum_y)
        lower = sorted((point for point in points if point["y"] >= cutoff), key=lambda point: (point["x"], point["y"]))
        xs, ys = [point["x"] for point in lower], [point["y"] for point in lower]

        def quantile(values: list[float], fraction: float) -> float:
            ordered = sorted(values)
            position = (len(ordered) - 1) * fraction
            low, high = math.floor(position), math.ceil(position)
            if low == high:
                return ordered[low]
            return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

        left, right = quantile(xs, 0.10), quantile(xs, 0.90)
        y = quantile(ys, 0.90)
        penalty = max(0, int(mask_component_count) - 1) * 2.0
        half_vertical = max(1.0, penalty)
        support = [{"x": left, "y": y - half_vertical}, {"x": right, "y": y + half_vertical}]
        return {
            "geometry_method": "F2",
            "footpoint_estimate": {"x": statistics.median(xs), "y": y},
            "footpoint_method": "MASK_LOWER_CONTACT",
            "footpoint_uncertainty_region": {
                "region_type": "MASK_CONTACT_INTERVAL",
                "x1": left,
                "y1": y - half_vertical,
                "x2": right,
                "y2": y + half_vertical,
                "support_points": support,
            },
        }
    centre_x = (x1 + x2) / 2
    if variant == "F0":
        point = {"x": centre_x, "y": y2}
        return {
            "geometry_method": "F0",
            "footpoint_estimate": point,
            "footpoint_method": "BOX_BOTTOM_CENTRE",
            "footpoint_uncertainty_region": {
                "region_type": "POINT",
                "x1": centre_x,
                "y1": y2,
                "x2": centre_x,
                "y2": y2,
                "support_points": [point],
            },
        }
    if variant != "F1":
        raise ValueError(f"unknown footpoint geometry variant: {variant}")
    half_width = max(2.0, width * 0.18)
    vertical = max(1.0, height * 0.05)
    support = [
        {"x": centre_x - half_width, "y": y2 - vertical},
        {"x": centre_x + half_width, "y": y2 - vertical},
        {"x": centre_x + half_width, "y": y2},
        {"x": centre_x - half_width, "y": y2},
    ]
    return {
        "geometry_method": "F1",
        "footpoint_estimate": {"x": centre_x, "y": y2 - vertical / 2},
        "footpoint_method": "BOX_LOWER_CONTACT_INTERVAL",
        "footpoint_uncertainty_region": {
            "region_type": "HORIZONTAL_INTERVAL",
            "x1": centre_x - half_width,
            "y1": y2 - vertical,
            "x2": centre_x + half_width,
            "y2": y2,
            "support_points": support,
        },
    }


def _conservative_relation(
    estimate: Mapping[str, Any], polygon: Sequence[Mapping[str, float]], margin: float
) -> PitchRelation:
    region = estimate["footpoint_uncertainty_region"]
    corners = [
        {"x": float(region["x1"]), "y": float(region["y1"])},
        {"x": float(region["x2"]), "y": float(region["y1"])},
        {"x": float(region["x2"]), "y": float(region["y2"])},
        {"x": float(region["x1"]), "y": float(region["y2"])},
    ]
    if region["x1"] == region["x2"] and region["y1"] == region["y2"]:
        corners = corners[:1]
    distances = [signed_distance_to_polygon(point, polygon) for point in corners]
    rectangle_edges = [(corners[index], corners[(index + 1) % len(corners)]) for index in range(len(corners))]
    polygon_edges = [(polygon[index], polygon[(index + 1) % len(polygon)]) for index in range(len(polygon))]
    boundary_intersection = len(corners) > 1 and any(
        _segments_intersect(left_start, left_end, right_start, right_end)
        for left_start, left_end in rectangle_edges
        for right_start, right_end in polygon_edges
    )
    polygon_vertex_in_region = any(
        float(region["x1"]) <= float(point["x"]) <= float(region["x2"])
        and float(region["y1"]) <= float(point["y"]) <= float(region["y2"])
        for point in polygon
    )
    if min(distances) > margin and not boundary_intersection:
        return "ON_PITCH"
    if max(distances) < -margin and not boundary_intersection and not polygon_vertex_in_region:
        return "OFF_PITCH"
    return "BOUNDARY_UNCERTAIN"


def apply_pitch_gate(
    variant: Literal["P0", "P1", "P2", "P3", "P4"],
    box: Mapping[str, Any],
    polygon: Sequence[Mapping[str, float]],
    *,
    mask_pixels: Sequence[Mapping[str, float]] | None = None,
    mask_reliable: bool = False,
    mask_component_count: int = 1,
) -> dict[str, Any]:
    """Apply one frozen gate without evaluator labels or footpoints."""

    if len(polygon) < 3:
        raise ValueError("pitch gate requires an approved source-panorama polygon")
    if variant == "P0":
        estimate = estimate_footpoint("F0", box)
        relation: PitchRelation = "UNGATED_RETAIN"
    elif variant == "P1":
        estimate = estimate_footpoint("F0", box)
        relation = "ON_PITCH" if point_in_polygon(estimate["footpoint_estimate"], polygon) else "OFF_PITCH"
    elif variant == "P2":
        estimate = estimate_footpoint("F1", box)
        relation = _conservative_relation(estimate, polygon, PITCH_POLYGON_TOLERANCE_PIXELS)
    elif variant == "P3":
        estimate = estimate_footpoint(
            "F3",
            box,
            mask_pixels=mask_pixels,
            mask_reliable=mask_reliable,
            mask_component_count=mask_component_count,
        )
        relation = _conservative_relation(estimate, polygon, PITCH_POLYGON_TOLERANCE_PIXELS)
    elif variant == "P4":
        box_estimate = estimate_footpoint("F1", box)
        box_relation = _conservative_relation(box_estimate, polygon, PITCH_POLYGON_TOLERANCE_PIXELS)
        if mask_reliable and mask_pixels:
            hybrid_estimate = estimate_footpoint(
                "F3",
                box,
                mask_pixels=mask_pixels,
                mask_reliable=mask_reliable,
                mask_component_count=mask_component_count,
            )
            hybrid_relation = _conservative_relation(hybrid_estimate, polygon, PITCH_POLYGON_TOLERANCE_PIXELS)
        else:
            # F3 is definitionally F1 when no accepted reliable mask exists.
            hybrid_estimate = {**box_estimate, "geometry_method": "F3"}
            hybrid_estimate["footpoint_method"] = "HYBRID_BOX_LOWER_CONTACT_INTERVAL"
            hybrid_relation = box_relation
        relation = (
            box_relation
            if box_relation == hybrid_relation and box_relation != "BOUNDARY_UNCERTAIN"
            else "BOUNDARY_UNCERTAIN"
        )
        estimate = hybrid_estimate
    else:
        raise ValueError(f"unknown pitch gate variant: {variant}")
    return {**estimate, "pitch_relation": relation, "pitch_gate_variant": variant}


def materialize_player_observation(
    runtime_observation: Mapping[str, Any],
    *,
    frame_index: int,
    pitch_gate_variant: Literal["P0", "P1", "P2", "P3", "P4"],
    polygon: Sequence[Mapping[str, float]],
    mask_pixels: Sequence[Mapping[str, float]] | None = None,
    mask_polygon: Sequence[Mapping[str, float]] | None = None,
    mask_reliable: bool = False,
    mask_component_count: int = 1,
    dense_branch_state: Literal[
        "NOT_TRIGGERED",
        "FROZEN_TRIGGER_ACCEPTED_MASK",
        "FROZEN_TRIGGER_ROUTED",
        "FROZEN_TRIGGER_FAILED",
    ] = "NOT_TRIGGERED",
) -> dict[str, Any]:
    """Materialize one strict, observed-only Player Observation v1 row."""

    validate_runtime_payload(runtime_observation)
    box = runtime_observation.get("box_panorama_pixels") or runtime_observation.get("visible_box")
    if not isinstance(box, Mapping):
        raise ValueError("runtime observation is missing source-panorama box geometry")
    gate = apply_pitch_gate(
        pitch_gate_variant,
        box,
        polygon,
        mask_pixels=mask_pixels,
        mask_reliable=mask_reliable,
        mask_component_count=mask_component_count,
    )
    output_state = str(runtime_observation.get("output_state", "ACCEPT_INDEPENDENT_OBSERVATION"))
    merged_risk = "PROPOSAL_GEOMETRY_RISK" if output_state == "ROUTE_DENSE_REVIEW" else "NONE"
    if output_state == "ROUTE_DENSE_REVIEW" or dense_branch_state in {"FROZEN_TRIGGER_ROUTED", "FROZEN_TRIGGER_FAILED"}:
        observation_state: ObservationState = "ROUTE_DENSE_REVIEW"
        route_reason = "proposal geometry or frozen dense branch requires review"
    elif gate["pitch_relation"] == "BOUNDARY_UNCERTAIN":
        observation_state = "ROUTE_PITCH_BOUNDARY_REVIEW"
        route_reason = "conservative source-coordinate pitch relation is uncertain"
    elif mask_reliable and mask_polygon:
        observation_state = "OBSERVED_MASK"
        route_reason = None
    else:
        observation_state = "OBSERVED_BOX"
        route_reason = None
    lineage = sorted(
        str(value)
        for value in (
            runtime_observation.get("cluster_member_proposal_uuids")
            or runtime_observation.get("proposal_uuid_lineage")
            or [runtime_observation.get("representative_proposal_uuid")]
        )
        if value
    )
    source_views = sorted(
        str(value)
        for value in (
            runtime_observation.get("all_source_view_ids") or runtime_observation.get("source_view_ids") or []
        )
    )
    if not lineage or not source_views:
        raise ValueError("runtime observation requires proposal lineage and source views")
    cluster_uuid = str(runtime_observation.get("observation_uuid") or runtime_observation.get("cluster_uuid"))
    duplicate_risk = "SUPPRESSED_LINEAGE_PRESENT" if len(lineage) > 1 else "NONE"
    observation_uuid = f"player_observation_{stable_hash([cluster_uuid, pitch_gate_variant, dense_branch_state])[:24]}"
    payload = {
        "schema_version": PLAYER_OBSERVATION_SCHEMA_VERSION,
        "observation_uuid": observation_uuid,
        "source_frame_sha256": str(runtime_observation["source_frame_sha256"]),
        "frame_index": int(frame_index),
        "source_view_ids": source_views,
        "proposal_uuid_lineage": lineage,
        "cluster_uuid": cluster_uuid,
        "visible_box": {key: float(box[key]) for key in ("x1", "y1", "x2", "y2")},
        "optional_visible_mask": [dict(point) for point in mask_polygon] if mask_reliable and mask_polygon else None,
        "geometry_method": gate["geometry_method"],
        "footpoint_estimate": gate["footpoint_estimate"],
        "footpoint_method": gate["footpoint_method"],
        "footpoint_uncertainty_region": gate["footpoint_uncertainty_region"],
        "pitch_relation": gate["pitch_relation"],
        "pitch_gate_variant": pitch_gate_variant,
        "observation_state": observation_state,
        "duplicate_risk": duplicate_risk,
        "merged_risk": merged_risk,
        "dense_branch_state": dense_branch_state,
        "review_route_reason": route_reason,
        "role_state": "UNKNOWN",
        "role_source": "NO_FROZEN_RUNTIME_ROLE_COMPONENT",
    }
    payload["provenance_hash"] = stable_hash(payload)
    return PlayerObservationV1.model_validate(payload).model_dump(mode="json", exclude_none=True)


def player_observation_json_schema() -> dict[str, Any]:
    return copy.deepcopy(PlayerObservationV1.model_json_schema())


def classify_unmatched_proposal_for_evaluation(
    box: Mapping[str, Any], partial_crowd_regions: Sequence[Mapping[str, float]]
) -> Literal["UNSCORED_CROWD", "SCORED_UNMATCHED_PROPOSAL"]:
    """Apply the evaluator-only partial-crowd completeness boundary.

    This function is intentionally separate from observation materialization;
    crowd regions may affect scoring, but never runtime acceptance or gating.
    """

    x1, y1, x2, y2 = _box_values(box)
    centre = {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2}
    for region in partial_crowd_regions:
        if float(region["x1"]) <= centre["x"] <= float(region["x2"]) and float(region["y1"]) <= centre["y"] <= float(
            region["y2"]
        ):
            return "UNSCORED_CROWD"
    return "SCORED_UNMATCHED_PROPOSAL"
