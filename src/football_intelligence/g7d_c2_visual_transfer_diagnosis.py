"""Deterministic analysis helpers for the targeted G7D-C2 review sample."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

TARGETED_WARNING = "TARGETED REVIEW SAMPLE — NOT UNBIASED ACCURACY"

PERSON_VALIDITY = {
    "CLEAN_SINGLE_PERSON",
    "PARTIAL_SINGLE_PERSON",
    "LOOSE_BACKGROUND_AROUND_PERSON",
    "MERGES_MULTIPLE_PEOPLE",
    "DUPLICATE_OF_ANOTHER_CANDIDATE",
}
SINGLE_VALIDITY = {"CLEAN_SINGLE_PERSON", "PARTIAL_SINGLE_PERSON", "LOOSE_BACKGROUND_AROUND_PERSON"}
HUMAN_ROLES = {
    "OUTFIELD_PLAYER",
    "GOALKEEPER",
    "REFEREE",
    "OTHER_OFFICIAL",
    "STAFF_OR_SPECTATOR",
    "UNKNOWN_PERSON_ROLE",
}
RELEVANT_ROLES = {"OUTFIELD_PLAYER", "GOALKEEPER", "REFEREE", "OTHER_OFFICIAL"}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """Return a Wilson 95% interval when the contracted support is available."""
    if total < 20:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def candidate_flags(decision: Mapping[str, Any]) -> dict[str, bool]:
    """Derive documented analysis flags without changing canonical labels."""
    validity = decision["proposal_validity"]
    role = decision["role"]
    quality = decision["box_quality"]
    contains_any_person = validity in PERSON_VALIDITY
    contains_single = validity in SINGLE_VALIDITY
    is_duplicate = validity == "DUPLICATE_OF_ANOTHER_CANDIDATE"
    is_merged = validity == "MERGES_MULTIPLE_PEOPLE"
    background = validity == "NO_PERSON_BACKGROUND_OR_OBJECT"
    useful = (
        validity in SINGLE_VALIDITY
        and role in HUMAN_ROLES
        and not is_duplicate
        and not is_merged
        and quality in {"GOOD_SINGLE_PERSON_BOX", "TOO_LOOSE", "TOO_TIGHT_OR_TRUNCATED"}
    )
    relevant = role in RELEVANT_ROLES and (
        decision["participation"] == "ACTIVE" or role in {"REFEREE", "OTHER_OFFICIAL"}
    )
    return {
        "contains_any_person": contains_any_person,
        "contains_single_person": contains_single,
        "contains_multiple_people": is_merged,
        "is_duplicate": is_duplicate,
        "is_background_or_object": background,
        "is_relevant_active_population": relevant,
        "box_is_useful_single_person": useful,
        "box_is_strict_good_single_person": useful and quality == "GOOD_SINGLE_PERSON_BOX",
        "box_quality_issue": quality != "GOOD_SINGLE_PERSON_BOX",
        "has_occlusion": decision["occlusion"] in {"PARTIAL", "SEVERE", "FULLY_OCCLUDED_PERSON_EXPECTED_HERE"},
    }


def summary_flags(rows: Sequence[Mapping[str, Any]], flag_names: Iterable[str]) -> dict[str, Any]:
    total = len(rows)
    metrics = {}
    for name in flag_names:
        count = sum(bool(row["analysis_flags"][name]) for row in rows)
        metrics[name] = {
            "count": count,
            "support": total,
            "targeted_rate": count / total if total else None,
            "wilson_95": wilson_interval(count, total),
            "warning": TARGETED_WARNING,
        }
    return {"support": total, "metrics": metrics, "warning": TARGETED_WARNING}


def grouped_flag_summary(
    rows: Sequence[Mapping[str, Any]], dimensions: Sequence[str], flag_names: Sequence[str]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dimension in dimensions:
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row.get(dimension, "UNKNOWN")), []).append(row)
        output[dimension] = {key: summary_flags(value, flag_names) for key, value in sorted(groups.items())}
    return output


def box_metrics(inner: Sequence[float], outer: Sequence[float]) -> dict[str, float]:
    """Calculate deterministic containment and relative-geometry measures."""
    ix1, iy1, ix2, iy2 = map(float, inner)
    ox1, oy1, ox2, oy2 = map(float, outer)
    inner_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    outer_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
    overlap = max(0.0, min(ix2, ox2) - max(ix1, ox1)) * max(0.0, min(iy2, oy2) - max(iy1, oy1))
    union = inner_area + outer_area - overlap
    inner_centre = ((ix1 + ix2) / 2, (iy1 + iy2) / 2)
    outer_centre = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)
    inner_foot = ((ix1 + ix2) / 2, iy2)
    outer_foot = ((ox1 + ox2) / 2, oy2)
    outer_height = max(oy2 - oy1, 1e-12)
    centre_distance = math.dist(inner_centre, outer_centre)
    footpoint_distance = math.dist(inner_foot, outer_foot)
    return {
        "intersection_over_inner_area": overlap / inner_area if inner_area else 0.0,
        "box_iou": overlap / union if union else 0.0,
        "inner_outer_area_ratio": inner_area / outer_area if outer_area else math.inf,
        "footpoint_distance_pixels": footpoint_distance,
        "footpoint_distance_outer_height": footpoint_distance / outer_height,
        "centre_distance_pixels": centre_distance,
        "centre_distance_outer_height": centre_distance / outer_height,
        "bottom_alignment_outer_height": abs(iy2 - oy2) / outer_height,
        "inner_bottom_region": (inner_centre[1] - oy1) / outer_height,
    }


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon test in canonical source coordinates."""
    x, y = map(float, point)
    inside = False
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        x1, y1 = map(float, previous)
        x2, y2 = map(float, current)
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def point_segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    px, py = map(float, point)
    x1, y1 = map(float, start)
    x2, y2 = map(float, end)
    dx, dy = x2 - x1, y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.dist((px, py), (x1, y1))
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_squared))
    return math.dist((px, py), (x1 + t * dx, y1 + t * dy))


def polygon_location(
    point: Sequence[float], polygon: Sequence[Sequence[float]], source_height: float
) -> dict[str, Any]:
    """Assign a documented descriptive pitch-boundary band."""
    inside = point_in_polygon(point, polygon)
    distance = min(point_segment_distance(point, polygon[index - 1], vertex) for index, vertex in enumerate(polygon))
    boundary_band = 0.015 * source_height
    far_outside_band = 0.05 * source_height
    if distance <= boundary_band:
        band = "NEAR_BOUNDARY"
    elif inside:
        band = "INSIDE_POLYGON"
    elif distance >= far_outside_band:
        band = "FAR_OUTSIDE_POLYGON"
    else:
        band = "OUTSIDE_POLYGON"
    return {
        "inside_polygon": inside,
        "boundary_distance_pixels": distance,
        "boundary_band_pixels": boundary_band,
        "far_outside_band_pixels": far_outside_band,
        "geometry_band": band,
    }


def confusion_matrix(truth: Sequence[str], predicted: Sequence[str], labels: Sequence[str]) -> list[list[int]]:
    positions = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for expected, actual in zip(truth, predicted, strict=True):
        matrix[positions[expected]][positions[actual]] += 1
    return matrix


def classification_metrics(
    truth: Sequence[str], probabilities: Sequence[Sequence[float]], labels: Sequence[str]
) -> dict[str, Any]:
    """Calculate fold-local multiclass agreement and probability diagnostics."""
    if not truth:
        return {"support": 0, "applicable": False}
    predicted = [labels[max(range(len(row)), key=row.__getitem__)] for row in probabilities]
    matrix = confusion_matrix(truth, predicted, labels)
    agreement = sum(expected == actual for expected, actual in zip(truth, predicted, strict=True)) / len(truth)
    observed = [label for label in labels if label in truth]
    recalls = []
    f1_scores = []
    for label in observed:
        index = labels.index(label)
        true_positive = matrix[index][index]
        false_negative = sum(matrix[index]) - true_positive
        false_positive = sum(row[index] for row in matrix) - true_positive
        recalls.append(true_positive / (true_positive + false_negative))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(2 * true_positive / denominator if denominator else 0.0)
    brier = 0.0
    nll = 0.0
    bins: list[dict[str, Any]] = []
    confidence_rows = []
    for expected, row, actual in zip(truth, probabilities, predicted, strict=True):
        expected_index = labels.index(expected)
        brier += sum((probability - (index == expected_index)) ** 2 for index, probability in enumerate(row))
        nll -= math.log(max(row[expected_index], 1e-15))
        confidence_rows.append((max(row), expected == actual))
    for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
        selected = [
            (confidence, correct) for confidence, correct in confidence_rows if lower <= confidence < lower + 0.2
        ]
        bins.append(
            {
                "lower": lower,
                "upper": lower + 0.2,
                "support": len(selected),
                "mean_confidence": sum(row[0] for row in selected) / len(selected) if selected else None,
                "agreement": sum(row[1] for row in selected) / len(selected) if selected else None,
            }
        )
    return {
        "support": len(truth),
        "applicable": True,
        "labels": list(labels),
        "exact_agreement": agreement,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "confusion_matrix": matrix,
        "multiclass_brier": brier / len(truth),
        "negative_log_likelihood": nll / len(truth),
        "calibration_bins": bins,
        "warning": TARGETED_WARNING,
    }


def choose_next_stage(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen smallest-stage decision rules deterministically."""
    pitch_removed = float(evidence["pitch_clutter_removed_rate"])
    pitch_loss = float(evidence["pitch_relevant_useful_loss_rate"])
    nested_burden = float(evidence["nested_separable_burden_rate"])
    temporal_score = float(evidence["crowding_temporal_support_rate"])
    semantic_error = float(evidence["useful_box_semantic_error_rate"])
    if pitch_removed >= 0.25 and pitch_loss <= 0.05:
        primary = "G7D_C3A_PITCH_AWARE_PROPOSAL_GATE_EXPERIMENT"
        reason = "Pitch/population simulation removes substantial reviewed clutter with bounded relevant-box loss."
    elif nested_burden >= 0.10:
        primary = "G7D_C3B_NESTED_CANDIDATE_SUPPRESSION_SANDBOX"
        reason = "Contained fragment or duplicate burden is substantial and conservatively separable."
    elif temporal_score >= 0.25:
        primary = "G7E_TARGETED_TEMPORAL_ANNOTATION"
        reason = "Crowding, occlusion, fragmentation, or reviewed misses support bounded temporal follow-up."
    elif semantic_error >= 0.25:
        primary = "G7D_C3_STATIC_SEMANTIC_REPAIR_EXPERIMENT"
        reason = "Useful boxes are common while fold-local semantic errors dominate."
    else:
        primary = "G7D_B4_P2_P3_STATE_REBUILD_AUDIT"
        reason = "No smaller proposal or semantic intervention passes its evidence threshold."
    conditional = (
        "G7E_TARGETED_TEMPORAL_ANNOTATION"
        if primary != "G7E_TARGETED_TEMPORAL_ANNOTATION" and temporal_score >= 0.15
        else "G7D_C3B_NESTED_CANDIDATE_SUPPRESSION_SANDBOX"
        if primary != "G7D_C3B_NESTED_CANDIDATE_SUPPRESSION_SANDBOX" and nested_burden >= 0.05
        else None
    )
    return {
        "primary_stage": primary,
        "primary_reason": reason,
        "conditional_secondary_stage": conditional,
        "decision_inputs": dict(evidence),
        "implemented": False,
        "production_change": False,
    }
