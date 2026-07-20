"""Frozen matching and metric support for future detection-gold evaluation.

This module contains no detector invocation and evaluates only caller-supplied
gold and prediction rows. M5.5G.1A freezes the behavior but does not use it to
score or select an architecture.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def bbox_iou(left: dict[str, float], right: dict[str, float]) -> float:
    ix1 = max(float(left["x1"]), float(right["x1"]))
    iy1 = max(float(left["y1"]), float(right["y1"]))
    ix2 = min(float(left["x2"]), float(right["x2"]))
    iy2 = min(float(left["y2"]), float(right["y2"]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, float(left["x2"]) - float(left["x1"])) * max(0.0, float(left["y2"]) - float(left["y1"]))
    right_area = max(0.0, float(right["x2"]) - float(right["x1"])) * max(0.0, float(right["y2"]) - float(right["y1"]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _hungarian_min_cost(cost: list[list[float]]) -> list[int]:
    """Return the assigned column for each row of a square cost matrix."""
    size = len(cost)
    if size == 0:
        return []
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for row in range(1, size + 1):
        p[0] = row
        min_value = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        column = 0
        while True:
            used[column] = True
            current_row = p[column]
            delta = math.inf
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                current = cost[current_row - 1][candidate_column - 1] - u[current_row] - v[candidate_column]
                if current < min_value[candidate_column]:
                    min_value[candidate_column] = current
                    way[candidate_column] = column
                if min_value[candidate_column] < delta:
                    delta = min_value[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    u[p[candidate_column]] += delta
                    v[candidate_column] -= delta
                else:
                    min_value[candidate_column] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            previous = way[column]
            p[column] = p[previous]
            column = previous
            if column == 0:
                break
    assignment = [-1] * size
    for column in range(1, size + 1):
        if p[column] > 0:
            assignment[p[column] - 1] = column - 1
    return assignment


def one_to_one_match(
    gold_boxes: list[dict[str, float]],
    prediction_boxes: list[dict[str, float]],
    *,
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Maximum-IoU one-to-one matching with explicit unmatched dummies."""
    size = max(len(gold_boxes), len(prediction_boxes))
    if size == 0:
        return []
    overlap = [[bbox_iou(prediction, gold) for gold in gold_boxes] for prediction in prediction_boxes]
    cost = [[1.0] * size for _ in range(size)]
    for prediction_index, row in enumerate(overlap):
        for gold_index, value in enumerate(row):
            cost[prediction_index][gold_index] = 1.0 - value
    assignment = _hungarian_min_cost(cost)
    matches = []
    for prediction_index, gold_index in enumerate(assignment[: len(prediction_boxes)]):
        if gold_index < 0 or gold_index >= len(gold_boxes):
            continue
        value = overlap[prediction_index][gold_index]
        if value >= iou_threshold:
            matches.append(
                {
                    "gold_index": gold_index,
                    "prediction_index": prediction_index,
                    "visible_box_iou": value,
                }
            )
    return sorted(matches, key=lambda row: (row["gold_index"], row["prediction_index"]))


def normalized_footpoint_error(gold: dict[str, Any], prediction: dict[str, Any]) -> float | None:
    gold_point = gold.get("footpoint")
    predicted_point = prediction.get("footpoint")
    box = gold.get("visible_body_box")
    if not isinstance(gold_point, dict) or not isinstance(predicted_point, dict) or not isinstance(box, dict):
        return None
    height = float(box["y2"]) - float(box["y1"])
    if height <= 0:
        return None
    distance = math.hypot(
        float(gold_point["x"]) - float(predicted_point["x"]),
        float(gold_point["y"]) - float(predicted_point["y"]),
    )
    return distance / height


def evaluate_player_observations(
    *,
    gold_instances: list[dict[str, Any]],
    pre_consolidation_observations: list[dict[str, Any]],
    final_observations: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute the frozen player-observation metric schema."""
    gold_boxes = [row["visible_body_box"] for row in gold_instances]
    pre_boxes = [row["visible_body_box"] for row in pre_consolidation_observations]
    final_boxes = [row["visible_body_box"] for row in final_observations]
    pre_matches = one_to_one_match(gold_boxes, pre_boxes, iou_threshold=iou_threshold)
    final_matches = one_to_one_match(gold_boxes, final_boxes, iou_threshold=iou_threshold)

    gold_overlaps: dict[int, list[int]] = defaultdict(list)
    for prediction_index, prediction in enumerate(final_boxes):
        for gold_index, gold in enumerate(gold_boxes):
            if bbox_iou(prediction, gold) >= iou_threshold:
                gold_overlaps[gold_index].append(prediction_index)
    duplicate_count = sum(max(0, len(indices) - 1) for indices in gold_overlaps.values())
    merged_as_clean_count = sum(
        row.get("source_candidate_relation") == "MERGED_MULTIPLE_INSTANCES"
        and row.get("output_relation", "CLEAN_SINGLE_INSTANCE") == "CLEAN_SINGLE_INSTANCE"
        for row in final_observations
    )
    distinct_person_suppression_count = sum(bool(row.get("distinct_person_suppressed")) for row in final_observations)
    footpoint_errors = []
    for match in final_matches:
        value = normalized_footpoint_error(
            gold_instances[match["gold_index"]], final_observations[match["prediction_index"]]
        )
        if value is not None:
            footpoint_errors.append(value)
    off_pitch_false_admission_count = sum(
        row.get("pitch_state") in {"OFF_PITCH", "BOUNDARY_UNCERTAIN"}
        and bool(row.get("primary_on_pitch_supply_eligible"))
        for row in final_observations
    )
    on_pitch_gold_count = sum(row.get("pitch_state") == "ON_PITCH" for row in gold_instances)
    matched_on_pitch = sum(
        gold_instances[match["gold_index"]].get("pitch_state") == "ON_PITCH" for match in final_matches
    )
    predicted_as_observed_count = sum(
        row.get("temporal_state") == "OCCLUDED_PREDICTED" and row.get("rendered_state") == "OBSERVED"
        for row in final_observations
    )
    return {
        "schema_version": "football_intelligence.m5_5g1a.player_metrics.v1",
        "gold_instance_count": len(gold_instances),
        "pre_consolidation_observation_count": len(pre_consolidation_observations),
        "final_observation_count": len(final_observations),
        "pre_consolidation_gold_matched_recall": len(pre_matches) / len(gold_instances) if gold_instances else 1.0,
        "final_one_to_one_recall": len(final_matches) / len(gold_instances) if gold_instances else 1.0,
        "duplicate_observation_count": duplicate_count,
        "duplicate_observation_rate": duplicate_count / len(final_observations) if final_observations else 0.0,
        "merged_as_clean_count": merged_as_clean_count,
        "merged_as_clean_rate": merged_as_clean_count / len(final_observations) if final_observations else 0.0,
        "distinct_person_suppression_count": distinct_person_suppression_count,
        "distinct_person_suppression_rate": (
            distinct_person_suppression_count / len(gold_instances) if gold_instances else 0.0
        ),
        "visible_box_iou": [match["visible_box_iou"] for match in final_matches],
        "footpoint_error_normalized_by_visible_height": footpoint_errors,
        "off_pitch_false_admission_count": off_pitch_false_admission_count,
        "on_pitch_retention": matched_on_pitch / on_pitch_gold_count if on_pitch_gold_count else 1.0,
        "predicted_as_observed_count": predicted_as_observed_count,
        "matches": final_matches,
    }


def evaluate_football_frames(
    *, gold_frames: list[dict[str, Any]], prediction_frames: list[dict[str, Any]]
) -> dict[str, Any]:
    predictions = {int(row["frame_sequence"]): row for row in prediction_frames}
    visible_states = {"VISIBLE_CLEAR", "VISIBLE_BLURRED", "PARTIALLY_OCCLUDED_VISIBLE"}
    visible_gold = 0
    visible_matched = 0
    not_visible_gold = 0
    false_alarms = 0
    centre_errors = []
    for gold in gold_frames:
        prediction = predictions.get(int(gold["frame_sequence"]), {})
        gold_visible = gold.get("state") in visible_states
        prediction_visible = prediction.get("state") in visible_states
        if gold_visible:
            visible_gold += 1
            visible_matched += prediction_visible
            if prediction_visible and gold.get("centre_point") and prediction.get("centre_point"):
                centre_errors.append(
                    math.hypot(
                        float(gold["centre_point"]["x"]) - float(prediction["centre_point"]["x"]),
                        float(gold["centre_point"]["y"]) - float(prediction["centre_point"]["y"]),
                    )
                )
        elif gold.get("state") in {"NOT_VISIBLE", "OUT_OF_FRAME"}:
            not_visible_gold += 1
            false_alarms += prediction_visible
    return {
        "schema_version": "football_intelligence.m5_5g1a.football_metrics.v1",
        "ball_visible_recall": visible_matched / visible_gold if visible_gold else 1.0,
        "ball_no_visible_false_alarm_count": false_alarms,
        "ball_no_visible_false_alarms_per_frame": false_alarms / not_visible_gold if not_visible_gold else 0.0,
        "ball_centre_error_pixels": centre_errors,
        "visible_gold_frame_count": visible_gold,
        "not_visible_gold_frame_count": not_visible_gold,
    }


def evaluate_detection_gold(
    *,
    gold_instances: list[dict[str, Any]] | None = None,
    pre_consolidation_observations: list[dict[str, Any]] | None = None,
    final_observations: list[dict[str, Any]] | None = None,
    gold_football_frames: list[dict[str, Any]] | None = None,
    predicted_football_frames: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return supported metric families without applying any acceptance gate."""
    result: dict[str, Any] = {
        "schema_version": "football_intelligence.m5_5g1a.future_metric_output.v1",
        "architecture_scored_in_m5_5g1a": False,
        "acceptance_gate_applied": False,
    }
    if gold_instances is not None:
        result["player"] = evaluate_player_observations(
            gold_instances=gold_instances,
            pre_consolidation_observations=pre_consolidation_observations or [],
            final_observations=final_observations or [],
        )
    if gold_football_frames is not None:
        result["football"] = evaluate_football_frames(
            gold_frames=gold_football_frames,
            prediction_frames=predicted_football_frames or [],
        )
    return result
