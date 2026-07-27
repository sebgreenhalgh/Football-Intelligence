"""Grouped-development evaluation for the Football Observation Reasoner.

The helpers in this module are intentionally model agnostic.  They evaluate
source-bound predictions without changing observations and keep every
denominator explicit.  IoU is deliberately absent from the primary metrics.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from football_intelligence.review_chassis.hashing import stable_hash

DEVELOPMENT_LABEL = "SINGLE_MATCH_GROUPED_DEVELOPMENT_ONLY"
CLEAN_STATE = "CLEAN_INDEPENDENT_PERSON"
UNRESOLVED_STATE = "AMBIGUOUS_UNRESOLVED"
NON_CLEAN_STATES = {
    "DUPLICATE_OF_PERSON",
    "MERGED_MULTIPLE_PEOPLE",
    "PARTIAL_PERSON",
    "BACKGROUND",
    UNRESOLVED_STATE,
}
GOALKEEPER_ROLE = "GOALKEEPER"
REFEREE_ROLE = "REFEREE"
OFFICIAL_ROLE = "OTHER_MATCH_OFFICIAL"
UNKNOWN_BY_HEAD = {
    "team": "UNKNOWN_TEAM",
    "kit": "UNKNOWN_KIT",
    "participation": "UNKNOWN_PARTICIPATION",
}
_RUNTIME_FEATURE_FIELDS = (
    "pitch_polygon_distance_features",
    "expected_scale_features",
    "visual_embedding_ref",
    "colour_kit_features",
    "shape_features",
    "mask_features",
    "neighbourhood_features",
    "proposal_provenance_features",
)
_TARGET_LEAKAGE_KEYS = {
    "annotation_uuid",
    "candidate_state_target",
    "footpoint_target_source_pixels",
    "footpoint_target_uncertainty_pixels",
    "gold_person_id",
    "gold_person_ids",
    "kit_target",
    "participation_target",
    "pitch_state_target",
    "role_target",
    "team_target",
}


def _finite_probability(value: float) -> float:
    probability = float(value)
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError("probabilities must be finite values in [0, 1]")
    return probability


def expected_calibration_error(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
    *,
    bin_count: int = 10,
) -> dict[str, Any]:
    """Return equal-width ECE and a complete reliability ledger."""

    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have equal length")
    if bin_count < 1:
        raise ValueError("bin_count must be positive")
    checked = [_finite_probability(value) for value in probabilities]
    truth = [int(bool(value)) for value in outcomes]
    bins: list[dict[str, Any]] = []
    error = 0.0
    denominator = len(checked)
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        member_indices = [
            row_index
            for row_index, probability in enumerate(checked)
            if lower <= probability <= upper and (index == bin_count - 1 or probability < upper)
        ]
        count = len(member_indices)
        mean_confidence = sum(checked[row_index] for row_index in member_indices) / count if count else None
        empirical_accuracy = sum(truth[row_index] for row_index in member_indices) / count if count else None
        absolute_gap = abs(float(mean_confidence) - float(empirical_accuracy)) if count else None
        if count and denominator:
            error += count / denominator * float(absolute_gap)
        bins.append(
            {
                "bin_index": index,
                "lower_inclusive": lower,
                "upper_inclusive": index == bin_count - 1,
                "upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
                "absolute_gap": absolute_gap,
            }
        )
    brier = (
        sum((probability - outcome) ** 2 for probability, outcome in zip(checked, truth, strict=True)) / denominator
        if denominator
        else None
    )
    return {
        "schema_version": "football_intelligence.m5_5g7a.calibration.v1",
        "development_scope": DEVELOPMENT_LABEL,
        "denominator": denominator,
        "bin_count": bin_count,
        "expected_calibration_error": error if denominator else None,
        "brier_score": brier,
        "bins": bins,
    }


def selective_risk_curve(
    confidences: Sequence[float],
    correct: Sequence[bool | int],
    *,
    coverages: Sequence[float] = (0.25, 0.5, 0.75, 0.9, 1.0),
) -> dict[str, Any]:
    """Measure error after retaining the most confident predictions."""

    if len(confidences) != len(correct):
        raise ValueError("confidences and correctness must have equal length")
    checked = [_finite_probability(value) for value in confidences]
    truth = [bool(value) for value in correct]
    ordered = sorted(range(len(checked)), key=lambda index: (-checked[index], index))
    points = []
    for requested in coverages:
        coverage = float(requested)
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage values must be in (0, 1]")
        retained = math.ceil(coverage * len(ordered)) if ordered else 0
        indices = ordered[:retained]
        risk = 1.0 - sum(truth[index] for index in indices) / retained if retained else None
        points.append(
            {
                "requested_coverage": coverage,
                "retained": retained,
                "denominator": len(ordered),
                "actual_coverage": retained / len(ordered) if ordered else 0.0,
                "risk": risk,
                "minimum_retained_confidence": min((checked[index] for index in indices), default=None),
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5g7a.selective_risk.v1",
        "development_scope": DEVELOPMENT_LABEL,
        "points": points,
    }


def categorical_head_metrics(
    node_rows: Sequence[Mapping[str, Any]],
    target_field: str,
    predicted_classes: Mapping[str, str],
    probabilities: Mapping[str, Sequence[float] | Mapping[str, float]] | Sequence[Sequence[float]],
    ordered_classes: Sequence[str],
    *,
    availability_mask_field: str | None = None,
    head_name: str | None = None,
    calibration_bin_count: int = 10,
) -> dict[str, Any]:
    """Evaluate one authorized categorical head on its explicitly masked subset.

    ``probabilities`` may be keyed by ``example_uuid`` or be a matrix aligned
    with ``node_rows``.  An empty labelled subset is a first-class
    ``NOT_EVALUABLE_NO_AUTHORIZED_LABELS`` result, which lets team and kit stay
    honestly K1-pending.
    """

    field = str(target_field).strip()
    if not field:
        raise ValueError("target_field must be non-empty")
    classes = tuple(str(value).strip() for value in ordered_classes)
    if not classes or any(not value for value in classes) or len(classes) != len(set(classes)):
        raise ValueError("ordered_classes must contain unique non-empty values")
    inferred_masks = {
        "candidate_state_target": "candidate_state",
        "role_target": "role",
        "team_target": "team",
        "kit_target": "kit",
        "pitch_state_target": "pitch",
        "participation_target": "participation",
    }
    mask_field = availability_mask_field or inferred_masks.get(field)
    resolved_head_name = str(head_name or mask_field or field.removesuffix("_target")).strip()
    if not resolved_head_name:
        raise ValueError("head_name must be non-empty")
    example_ids = [str(row.get("example_uuid", "")) for row in node_rows]
    if any(not value for value in example_ids) or len(example_ids) != len(set(example_ids)):
        raise ValueError("node_rows must have unique non-empty example_uuid values")

    labelled_indices = []
    explicit_mask_count = 0
    for index, row in enumerate(node_rows):
        target_available = row.get(field) is not None
        availability = row.get("label_availability_mask")
        if mask_field is not None and isinstance(availability, Mapping) and mask_field in availability:
            declared_available = bool(availability[mask_field])
            explicit_mask_count += 1
            if declared_available != target_available:
                raise ValueError(
                    f"{field} target availability disagrees with label_availability_mask.{mask_field} "
                    f"for {example_ids[index]}"
                )
        else:
            declared_available = target_available
        if declared_available:
            labelled_indices.append(index)

    if isinstance(probabilities, Mapping):
        probability_by_example: Mapping[str, Sequence[float] | Mapping[str, float]] = probabilities
    else:
        try:
            matrix_length = len(probabilities)
        except TypeError as exc:
            raise ValueError("probabilities must be an example mapping or a node-aligned matrix") from exc
        if matrix_length != len(node_rows):
            raise ValueError("probability matrix row count must equal node_rows length")
        probability_by_example = {example_uuid: probabilities[index] for index, example_uuid in enumerate(example_ids)}

    confusion = {target: {prediction: 0 for prediction in classes} for target in classes}
    ledger = []
    for index in labelled_indices:
        row = node_rows[index]
        example_uuid = example_ids[index]
        target = str(row[field])
        if target not in classes:
            raise ValueError(f"unknown {field} target class {target!r} for {example_uuid}")
        if example_uuid not in predicted_classes:
            raise ValueError(f"missing {resolved_head_name} prediction for {example_uuid}")
        prediction = str(predicted_classes[example_uuid])
        if prediction not in classes:
            raise ValueError(f"unknown {resolved_head_name} prediction class {prediction!r} for {example_uuid}")
        if example_uuid not in probability_by_example:
            raise ValueError(f"missing {resolved_head_name} probability vector for {example_uuid}")
        supplied = probability_by_example[example_uuid]
        if isinstance(supplied, Mapping):
            missing_classes = sorted(set(classes) - {str(key) for key in supplied})
            extra_classes = sorted({str(key) for key in supplied} - set(classes))
            if missing_classes or extra_classes:
                raise ValueError(
                    f"probability mapping classes mismatch for {example_uuid}: "
                    f"missing={missing_classes}, extra={extra_classes}"
                )
            vector = [_finite_probability(float(supplied[class_name])) for class_name in classes]
        else:
            if len(supplied) != len(classes):
                raise ValueError(f"probability vector length mismatch for {example_uuid}")
            vector = [_finite_probability(float(value)) for value in supplied]
        if not math.isclose(sum(vector), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"probabilities must sum to one for {example_uuid}")
        top_confidence = max(vector)
        predicted_probability = vector[classes.index(prediction)]
        if not math.isclose(predicted_probability, top_confidence, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"prediction is not a maximum-probability class for {example_uuid}")
        correct = prediction == target
        confusion[target][prediction] += 1
        ledger.append(
            {
                "example_uuid": example_uuid,
                "source_group_id": str(row.get("source_group_id") or "UNKNOWN"),
                "target": target,
                "prediction": prediction,
                "correct": correct,
                "top_class_confidence": top_confidence,
            }
        )

    per_class = {}
    supported_recalls = []
    for class_name in classes:
        support = sum(confusion[class_name].values())
        predicted_support = sum(confusion[target][class_name] for target in classes)
        true_positive = confusion[class_name][class_name]
        recall = true_positive / support if support else None
        precision = true_positive / predicted_support if predicted_support else None
        if recall is not None:
            supported_recalls.append(recall)
        per_class[class_name] = {
            "support": support,
            "predicted_support": predicted_support,
            "true_positive": true_positive,
            "recall": recall,
            "precision": precision,
        }
    source_rows: dict[str, list[bool]] = defaultdict(list)
    for row in ledger:
        source_rows[str(row["source_group_id"])].append(bool(row["correct"]))
    per_source_group = {
        group: {
            "denominator": len(correctness),
            "correct": sum(correctness),
            "accuracy": sum(correctness) / len(correctness),
        }
        for group, correctness in sorted(source_rows.items())
    }
    source_accuracies = [row["accuracy"] for row in per_source_group.values()]
    confidences = [float(row["top_class_confidence"]) for row in ledger]
    correctness = [bool(row["correct"]) for row in ledger]
    denominator = len(ledger)
    payload = {
        "schema_version": "football_intelligence.m5_5g7a.categorical_head_metrics.v1",
        "development_scope": DEVELOPMENT_LABEL,
        "head_name": resolved_head_name,
        "target_field": field,
        "availability_mask_field": mask_field,
        "ordered_classes": list(classes),
        "evaluation_status": "EVALUATED_AUTHORIZED_LABELLED_SUBSET"
        if denominator
        else "NOT_EVALUABLE_NO_AUTHORIZED_LABELS",
        "denominator": denominator,
        "unlabelled_or_masked_count": len(node_rows) - denominator,
        "explicit_availability_mask_row_count": explicit_mask_count,
        "accuracy": sum(correctness) / denominator if denominator else None,
        "macro_recall": sum(supported_recalls) / len(supported_recalls) if supported_recalls else None,
        "macro_recall_population": "CLASSES_WITH_NONZERO_TARGET_SUPPORT",
        "confusion_matrix": confusion,
        "per_class": per_class,
        "per_source_group": per_source_group,
        "source_group_normalized_accuracy": (
            sum(source_accuracies) / len(source_accuracies) if source_accuracies else None
        ),
        "top_class_confidence_calibration": expected_calibration_error(
            confidences,
            correctness,
            bin_count=calibration_bin_count,
        ),
        "selective_risk": selective_risk_curve(confidences, correctness),
        "k1_pending_compatible": denominator == 0,
        "iou_used_as_primary_metric": False,
        "ledger_hash": stable_hash(ledger),
    }
    payload["metrics_hash"] = stable_hash(payload)
    return payload


def candidate_outcomes(
    labelled_rows: Sequence[Mapping[str, Any]],
    predicted_states: Mapping[str, str],
    *,
    evaluator_person_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute primary outcomes against an explicit evaluator-person universe.

    Passing ``evaluator_person_ids`` is required for a full-universe claim: it
    keeps gold people with zero linked proposals in the denominator.  Omitting
    it retains a clearly labelled linked-subset diagnostic for small unit tests
    and callers that genuinely lack a separate person registry.
    """

    evaluated: list[dict[str, Any]] = []
    by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labelled_rows:
        target = row.get("candidate_state_target")
        if not target:
            continue
        example_uuid = str(row["example_uuid"])
        if example_uuid not in predicted_states:
            raise ValueError(f"missing prediction for {example_uuid}")
        prediction = str(predicted_states[example_uuid])
        gold_people = tuple(sorted(str(value) for value in row.get("gold_person_ids", []) if value))
        record = {
            "example_uuid": example_uuid,
            "source_group_id": str(row["source_group_id"]),
            "case_family": str(row.get("case_family") or "UNKNOWN"),
            "target": str(target),
            "prediction": prediction,
            "gold_person_ids": list(gold_people),
            "accepted": prediction == CLEAN_STATE,
            "correct": prediction == target,
        }
        evaluated.append(record)
        for person_id in gold_people:
            by_person[person_id].append(record)

    target_counts: dict[str, int] = defaultdict(int)
    prediction_counts: dict[str, int] = defaultdict(int)
    for row in evaluated:
        target_counts[row["target"]] += 1
        prediction_counts[row["prediction"]] += 1
    clean_rows = [row for row in evaluated if row["target"] == CLEAN_STATE]
    duplicate_rows = [row for row in evaluated if row["target"] == "DUPLICATE_OF_PERSON"]
    merged_rows = [row for row in evaluated if row["target"] == "MERGED_MULTIPLE_PEOPLE"]
    accepted_rows = [row for row in evaluated if row["accepted"]]
    linked_person_ids = set(by_person)
    if evaluator_person_ids is None:
        evaluator_ids = sorted(linked_person_ids)
        evaluator_universe_mode = "LINKED_SUBSET_ONLY"
    else:
        evaluator_ids = sorted({str(value) for value in evaluator_person_ids if str(value).strip()})
        missing_from_universe = sorted(linked_person_ids - set(evaluator_ids))
        if missing_from_universe:
            raise ValueError(f"linked gold people are absent from evaluator_person_ids: {missing_from_universe}")
        evaluator_universe_mode = "EXPLICIT_FULL_EVALUATOR_UNIVERSE"
    accepted_people = {
        person_id for person_id in evaluator_ids if any(record["accepted"] for record in by_person.get(person_id, ()))
    }
    exactly_one = sum(
        sum(record["accepted"] for record in by_person.get(person_id, ())) == 1 for person_id in evaluator_ids
    )
    duplicate_accepted = sum(row["accepted"] for row in duplicate_rows)
    merged_as_clean = sum(row["accepted"] for row in merged_rows)
    distinct_suppressed = sum(not row["accepted"] for row in clean_rows)
    clean_control_rows = [
        row
        for row in evaluated
        if str(row.get("case_family", "")).strip().lower() == "clean_control" and row["target"] == CLEAN_STATE
    ]
    preserved_clean_control_rows = [row for row in clean_control_rows if row["prediction"] == CLEAN_STATE]
    clean_control_error_rows = [row for row in clean_control_rows if row not in preserved_clean_control_rows]
    unresolved = sum(row["prediction"] == UNRESOLVED_STATE for row in evaluated)
    source_group_scores: dict[str, list[bool]] = defaultdict(list)
    for row in evaluated:
        source_group_scores[row["source_group_id"]].append(bool(row["correct"]))
    group_accuracies = [sum(values) / len(values) for values in source_group_scores.values()]
    return {
        "schema_version": "football_intelligence.m5_5g7a.candidate_outcomes.v1",
        "development_scope": DEVELOPMENT_LABEL,
        "denominators": {
            "labelled_candidates": len(evaluated),
            "clean_candidates": len(clean_rows),
            "duplicate_candidates": len(duplicate_rows),
            "merged_candidates": len(merged_rows),
            "accepted_candidates": len(accepted_rows),
            "linked_evaluator_people": len(by_person),
            "all_evaluator_people": len(evaluator_ids),
            "zero_linked_proposal_evaluator_people": len(set(evaluator_ids) - linked_person_ids),
            "source_groups": len(source_group_scores),
        },
        "evaluator_universe_mode": evaluator_universe_mode,
        "target_counts": dict(sorted(target_counts.items())),
        "prediction_counts": dict(sorted(prediction_counts.items())),
        "independent_person_supply": {
            "numerator": len(accepted_people),
            "denominator": len(evaluator_ids),
        },
        "exactly_one_observation": {"numerator": exactly_one, "denominator": len(evaluator_ids)},
        "duplicate_accepted_rate": {
            "numerator": duplicate_accepted,
            "denominator": len(duplicate_rows),
            "rate": duplicate_accepted / len(duplicate_rows) if duplicate_rows else None,
        },
        "merged_as_clean_count": merged_as_clean,
        "distinct_person_suppression": distinct_suppressed,
        "unresolved_routing": {"numerator": unresolved, "denominator": len(evaluated)},
        "clean_control_preservation": {
            "numerator": len(preserved_clean_control_rows),
            "preserved": len(preserved_clean_control_rows),
            "errors": len(clean_control_error_rows),
            "denominator": len(clean_control_rows),
            "rate": (len(preserved_clean_control_rows) / len(clean_control_rows) if clean_control_rows else None),
            "case_family": "clean_control",
            "error_example_uuids": sorted(str(row["example_uuid"]) for row in clean_control_error_rows),
        },
        "source_group_normalized_accuracy": (
            sum(group_accuracies) / len(group_accuracies) if group_accuracies else None
        ),
        "ledger_hash": stable_hash(evaluated),
    }


def _is_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _runtime_target_leakage_paths(row: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    stack: list[tuple[str, Any]] = [
        (field, row.get(field)) for field in _RUNTIME_FEATURE_FIELDS if row.get(field) is not None
    ]
    while stack:
        path, value = stack.pop()
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key).strip().lower()
                child_path = f"{path}.{key}"
                if key_text in _TARGET_LEAKAGE_KEYS:
                    paths.append(child_path)
                stack.append((child_path, item))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            stack.extend((f"{path}[{index}]", item) for index, item in enumerate(value))
    return sorted(set(paths))


def _provenance_receipt(row: Mapping[str, Any]) -> dict[str, Any]:
    source_hash = str(row.get("source_frame_sha256", "")).lower()
    artifact_hashes = row.get("source_artifact_hashes")
    artifact_source = artifact_hashes.get("source_frame") if isinstance(artifact_hashes, Mapping) else None
    checks = {
        "example_uuid_present": bool(str(row.get("example_uuid", "")).strip()),
        "candidate_uuid_present": bool(str(row.get("candidate_uuid", "")).strip()),
        "source_group_id_present": bool(str(row.get("source_group_id", "")).strip()),
        "source_frame_sha256_valid": _is_sha256(source_hash),
        "provenance_hash_valid": _is_sha256(row.get("provenance_hash", "")),
        "source_artifact_hash_bound": _is_sha256(artifact_source) and str(artifact_source).lower() == source_hash,
    }
    leakage_paths = _runtime_target_leakage_paths(row)
    checks["runtime_features_target_free"] = not leakage_paths
    return {
        "checks": checks,
        "runtime_target_leakage_paths": leakage_paths,
        "complete": all(checks.values()),
    }


def _small_far_proxy(row: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    for field in ("small_far_proxy", "small_far_side", "is_small_person", "far_side"):
        if row.get(field) is True:
            reasons.append(f"EXPLICIT_{field.upper()}")
    descriptive = " ".join(
        str(row.get(field, "")) for field in ("case_family", "universe", "visible_height_bin", "original_case_strata")
    ).upper()
    if "SMALL" in descriptive or "FAR_SIDE" in descriptive or "FARSIDE" in descriptive:
        reasons.append("DESCRIPTIVE_STRATUM_TOKEN")
    box = row.get("visible_box")
    visible_height = None
    normalized_height = None
    if isinstance(box, Mapping) and box.get("y1") is not None and box.get("y2") is not None:
        visible_height = float(box["y2"]) - float(box["y1"])
        coordinates = row.get("source_coordinates")
        image_height = coordinates.get("image_height") if isinstance(coordinates, Mapping) else None
        if image_height is not None and float(image_height) > 0.0:
            normalized_height = visible_height / float(image_height)
            if normalized_height <= 0.04:
                reasons.append("VISIBLE_HEIGHT_AT_MOST_4_PERCENT")
        elif visible_height <= 32.0:
            reasons.append("VISIBLE_HEIGHT_AT_MOST_32_PIXELS_WITHOUT_FRAME_HEIGHT")
    return {
        "is_small_far_proxy": bool(reasons),
        "proxy_reasons": sorted(set(reasons)),
        "visible_height_pixels": visible_height,
        "visible_height_fraction": normalized_height,
        "proxy_not_human_truth": True,
    }


def exhaustive_candidate_person_ledgers(
    node_rows: Sequence[Mapping[str, Any]],
    predicted_candidate_states: Mapping[str, str],
    *,
    evaluator_person_ids: Sequence[str],
    predicted_roles: Mapping[str, str] | None = None,
    predicted_pitch_states: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build exhaustive candidate and evaluator-person diagnostic ledgers.

    Every supplied node is retained, including unlabelled runtime candidates,
    and every explicit evaluator person receives a row even when no proposal is
    linked.  Small/far status is a documented geometry/descriptive proxy, not a
    hidden human label.
    """

    evaluator_ids = [str(value) for value in evaluator_person_ids]
    if any(not value.strip() for value in evaluator_ids) or len(evaluator_ids) != len(set(evaluator_ids)):
        raise ValueError("evaluator_person_ids must be unique non-empty identifiers")
    evaluator_set = set(evaluator_ids)
    candidate_ids: set[str] = set()
    candidate_rows: list[dict[str, Any]] = []
    linked_by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in node_rows:
        example_uuid = str(row.get("example_uuid", ""))
        if not example_uuid:
            raise ValueError("candidate rows require example_uuid")
        if example_uuid in candidate_ids:
            raise ValueError(f"duplicate candidate example_uuid: {example_uuid}")
        candidate_ids.add(example_uuid)
        if example_uuid not in predicted_candidate_states:
            raise ValueError(f"missing candidate-state prediction for {example_uuid}")
        target = None if row.get("candidate_state_target") is None else str(row["candidate_state_target"])
        prediction = str(predicted_candidate_states[example_uuid])
        role_target = None if row.get("role_target") is None else str(row["role_target"])
        role_prediction = (
            None
            if predicted_roles is None or example_uuid not in predicted_roles
            else str(predicted_roles[example_uuid])
        )
        pitch_target = None if row.get("pitch_state_target") is None else str(row["pitch_state_target"])
        pitch_prediction = (
            None
            if predicted_pitch_states is None or example_uuid not in predicted_pitch_states
            else str(predicted_pitch_states[example_uuid])
        )
        gold_people = sorted({str(value) for value in row.get("gold_person_ids", ()) if str(value).strip()})
        absent_people = sorted(set(gold_people) - evaluator_set)
        if absent_people:
            raise ValueError(f"linked gold people are absent from evaluator_person_ids: {absent_people}")
        accepted = prediction == CLEAN_STATE
        proxy = _small_far_proxy(row)
        provenance = _provenance_receipt(row)
        flags = {
            "small_far_side_miss": bool(proxy["is_small_far_proxy"] and target == CLEAN_STATE and not accepted),
            "duplicate_accepted": target == "DUPLICATE_OF_PERSON" and accepted,
            "distinct_person_suppressed": target == CLEAN_STATE and not accepted,
            "merged_accepted": target == "MERGED_MULTIPLE_PEOPLE" and accepted,
            "partial_as_background": target == "PARTIAL_PERSON" and prediction == "BACKGROUND",
            "background_as_partial": target == "BACKGROUND" and prediction == "PARTIAL_PERSON",
            "partial_background_confusion": (
                (target == "PARTIAL_PERSON" and prediction == "BACKGROUND")
                or (target == "BACKGROUND" and prediction == "PARTIAL_PERSON")
            ),
            "goalkeeper_as_referee_or_official": role_target == GOALKEEPER_ROLE
            and role_prediction in {REFEREE_ROLE, OFFICIAL_ROLE},
            "referee_or_official_as_goalkeeper": role_target in {REFEREE_ROLE, OFFICIAL_ROLE}
            and role_prediction == GOALKEEPER_ROLE,
            "pitch_state_mismatch": pitch_target is not None
            and pitch_prediction is not None
            and pitch_prediction != pitch_target,
            "provenance_or_leakage_defect": not provenance["complete"],
        }
        categories = sorted(name for name, enabled in flags.items() if enabled)
        record = {
            "record_kind": "CANDIDATE",
            "example_uuid": example_uuid,
            "candidate_uuid": str(row.get("candidate_uuid", "")),
            "source_group_id": str(row.get("source_group_id", "")),
            "source_frame_sha256": str(row.get("source_frame_sha256", "")),
            "universe": str(row.get("universe") or "UNKNOWN"),
            "case_family": str(row.get("case_family") or "UNKNOWN"),
            "candidate_state_target": target,
            "candidate_state_prediction": prediction,
            "candidate_state_label_available": target is not None,
            "accepted_as_independent_person": accepted,
            "candidate_state_correct": None if target is None else prediction == target,
            "role_target": role_target,
            "role_prediction": role_prediction,
            "pitch_state_target": pitch_target,
            "pitch_state_prediction": pitch_prediction,
            "gold_person_ids": gold_people,
            "label_availability_mask": dict(row.get("label_availability_mask") or {}),
            "small_far_proxy": proxy,
            "provenance": provenance,
            "outcome_flags": flags,
            "error_categories": categories,
        }
        candidate_rows.append(record)
        for person_id in gold_people:
            linked_by_person[person_id].append(record)
    candidate_rows.sort(key=lambda value: value["example_uuid"])

    person_rows = []
    for person_id in sorted(evaluator_ids):
        linked = linked_by_person.get(person_id, [])
        accepted = [row for row in linked if row["accepted_as_independent_person"]]
        clean = [row for row in linked if row["candidate_state_target"] == CLEAN_STATE]
        duplicate_accepted = [row for row in linked if row["outcome_flags"]["duplicate_accepted"]]
        merged_accepted = [row for row in linked if row["outcome_flags"]["merged_accepted"]]
        categories = []
        if not linked:
            categories.append("zero_proposal")
        if clean and not accepted:
            categories.append("distinct_person_suppressed")
        if duplicate_accepted:
            categories.append("duplicate_accepted")
        if merged_accepted:
            categories.append("merged_accepted")
        person_rows.append(
            {
                "record_kind": "EVALUATOR_PERSON",
                "evaluator_person_id": person_id,
                "linked_candidate_count": len(linked),
                "linked_candidate_example_uuids": sorted(row["example_uuid"] for row in linked),
                "accepted_candidate_count": len(accepted),
                "accepted_candidate_example_uuids": sorted(row["example_uuid"] for row in accepted),
                "zero_proposal": not linked,
                "independent_person_supplied": bool(accepted),
                "exactly_one_observation": len(accepted) == 1,
                "distinct_person_suppressed": bool(clean and not accepted),
                "duplicate_accepted_count": len(duplicate_accepted),
                "merged_accepted_count": len(merged_accepted),
                "error_categories": sorted(categories),
            }
        )
    payload = {
        "schema_version": "football_intelligence.m5_5g7a.exhaustive_evaluation_ledgers.v1",
        "development_scope": DEVELOPMENT_LABEL,
        "candidate_rows": candidate_rows,
        "person_rows": person_rows,
        "denominators": {
            "all_candidate_nodes": len(candidate_rows),
            "candidate_state_labelled_nodes": sum(row["candidate_state_label_available"] for row in candidate_rows),
            "all_evaluator_people": len(person_rows),
            "zero_proposal_evaluator_people": sum(row["zero_proposal"] for row in person_rows),
        },
        "candidate_ledger_exhaustive": len(candidate_rows) == len(node_rows),
        "person_ledger_exhaustive": len(person_rows) == len(evaluator_ids),
        "identity_tracks_created": False,
        "temporal_predictions_created": False,
    }
    payload["candidate_ledger_hash"] = stable_hash(candidate_rows)
    payload["person_ledger_hash"] = stable_hash(person_rows)
    payload["ledger_bundle_hash"] = stable_hash(payload)
    return payload


def _named_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    denominator_filter: Any,
    harmful_filter: Any,
    definition: str,
) -> dict[str, Any]:
    denominator_rows = [row for row in rows if denominator_filter(row)]
    harmful_rows = [row for row in denominator_rows if harmful_filter(row)]
    numerator = len(harmful_rows)
    denominator = len(denominator_rows)
    return {
        "definition": definition,
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
        "candidate_example_uuids": sorted(str(row["example_uuid"]) for row in harmful_rows),
    }


def candidate_confusion_audits(
    candidate_ledger_rows: Sequence[Mapping[str, Any]],
    *,
    person_ledger_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return named candidate confusions and their exact eligible subsets."""

    rows = [dict(row) for row in candidate_ledger_rows]
    labelled = [row for row in rows if row.get("candidate_state_label_available")]
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in labelled:
        confusion[str(row["candidate_state_target"])][str(row["candidate_state_prediction"])] += 1
    audits = {
        "small_far_side_miss": _named_audit(
            rows,
            denominator_filter=lambda row: row["small_far_proxy"]["is_small_far_proxy"]
            and row.get("candidate_state_target") == CLEAN_STATE,
            harmful_filter=lambda row: row["outcome_flags"]["small_far_side_miss"],
            definition="small/far proxy clean target not accepted as an independent person",
        ),
        "duplicate_accepted": _named_audit(
            rows,
            denominator_filter=lambda row: row.get("candidate_state_target") == "DUPLICATE_OF_PERSON",
            harmful_filter=lambda row: row["outcome_flags"]["duplicate_accepted"],
            definition="duplicate proposal accepted as an independent person",
        ),
        "distinct_person_suppressed": _named_audit(
            rows,
            denominator_filter=lambda row: row.get("candidate_state_target") == CLEAN_STATE,
            harmful_filter=lambda row: row["outcome_flags"]["distinct_person_suppressed"],
            definition="clean independent-person target routed to a non-clean state",
        ),
        "merged_accepted": _named_audit(
            rows,
            denominator_filter=lambda row: row.get("candidate_state_target") == "MERGED_MULTIPLE_PEOPLE",
            harmful_filter=lambda row: row["outcome_flags"]["merged_accepted"],
            definition="merged multiple-person proposal accepted as one independent person",
        ),
        "partial_as_background": _named_audit(
            rows,
            denominator_filter=lambda row: row.get("candidate_state_target") == "PARTIAL_PERSON",
            harmful_filter=lambda row: row["outcome_flags"]["partial_as_background"],
            definition="partial-person target predicted as background",
        ),
        "background_as_partial": _named_audit(
            rows,
            denominator_filter=lambda row: row.get("candidate_state_target") == "BACKGROUND",
            harmful_filter=lambda row: row["outcome_flags"]["background_as_partial"],
            definition="background target predicted as a partial person",
        ),
        "partial_background_confusion": _named_audit(
            rows,
            denominator_filter=lambda row: row.get("candidate_state_target") in {"PARTIAL_PERSON", "BACKGROUND"},
            harmful_filter=lambda row: row["outcome_flags"]["partial_background_confusion"],
            definition="partial-person/background cross-confusion in either direction",
        ),
        "goalkeeper_as_referee_or_official": _named_audit(
            rows,
            denominator_filter=lambda row: row.get("role_target") == GOALKEEPER_ROLE
            and row.get("role_prediction") is not None,
            harmful_filter=lambda row: row["outcome_flags"]["goalkeeper_as_referee_or_official"],
            definition="goalkeeper role target predicted as referee or other match official",
        ),
        "referee_or_official_as_goalkeeper": _named_audit(
            rows,
            denominator_filter=lambda row: row.get("role_target") in {REFEREE_ROLE, OFFICIAL_ROLE}
            and row.get("role_prediction") is not None,
            harmful_filter=lambda row: row["outcome_flags"]["referee_or_official_as_goalkeeper"],
            definition="referee/official role target predicted as goalkeeper",
        ),
        "goalkeeper_referee_confusion": _named_audit(
            rows,
            denominator_filter=lambda row: row.get("role_target") in {GOALKEEPER_ROLE, REFEREE_ROLE, OFFICIAL_ROLE}
            and row.get("role_prediction") is not None,
            harmful_filter=lambda row: row["outcome_flags"]["goalkeeper_as_referee_or_official"]
            or row["outcome_flags"]["referee_or_official_as_goalkeeper"],
            definition="goalkeeper versus referee/official confusion in either direction",
        ),
        "pitch_state_mismatch": _named_audit(
            rows,
            denominator_filter=lambda row: row.get("pitch_state_target") is not None
            and row.get("pitch_state_prediction") is not None,
            harmful_filter=lambda row: row["outcome_flags"]["pitch_state_mismatch"],
            definition="labelled pitch-state target differs from the predicted pitch state",
        ),
        "provenance_or_leakage_defect": _named_audit(
            rows,
            denominator_filter=lambda row: True,
            harmful_filter=lambda row: row["outcome_flags"]["provenance_or_leakage_defect"],
            definition="candidate provenance is incomplete or a runtime feature contains evaluator truth",
        ),
    }
    zero_proposal_people = sorted(
        str(row["evaluator_person_id"]) for row in person_ledger_rows if bool(row.get("zero_proposal"))
    )
    audits["zero_proposal_evaluator_person"] = {
        "definition": "evaluator person has no linked candidate proposal",
        "numerator": len(zero_proposal_people),
        "denominator": len(person_ledger_rows),
        "rate": len(zero_proposal_people) / len(person_ledger_rows) if person_ledger_rows else None,
        "evaluator_person_ids": zero_proposal_people,
    }
    payload = {
        "schema_version": "football_intelligence.m5_5g7a.named_candidate_confusion_audits.v1",
        "development_scope": DEVELOPMENT_LABEL,
        "candidate_state_confusion_matrix": {
            target: dict(sorted(predictions.items())) for target, predictions in sorted(confusion.items())
        },
        "audits": audits,
        "labelled_candidate_denominator": len(labelled),
        "all_candidate_denominator": len(rows),
        "diagnostic_only": True,
        "iou_used_as_primary_metric": False,
    }
    payload["audit_hash"] = stable_hash(payload)
    return payload


def _stratum_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labelled = [row for row in rows if row.get("candidate_state_label_available")]
    correct = [row for row in labelled if row.get("candidate_state_correct") is True]
    source_scores: dict[str, list[bool]] = defaultdict(list)
    for row in labelled:
        source_scores[str(row.get("source_group_id") or "UNKNOWN")].append(bool(row.get("candidate_state_correct")))
    source_accuracies = [sum(values) / len(values) for values in source_scores.values()]
    return {
        "population_count": len(rows),
        "labelled_count": len(labelled),
        "not_evaluable_count": len(rows) - len(labelled),
        "correct_count": len(correct),
        "accuracy": len(correct) / len(labelled) if labelled else None,
        "accepted_count": sum(bool(row.get("accepted_as_independent_person")) for row in rows),
        "target_counts": dict(sorted(Counter(str(row["candidate_state_target"]) for row in labelled).items())),
        "prediction_counts": dict(sorted(Counter(str(row["candidate_state_prediction"]) for row in labelled).items())),
        "source_group_count": len(source_scores),
        "source_group_normalized_accuracy": (
            sum(source_accuracies) / len(source_accuracies) if source_accuracies else None
        ),
    }


def candidate_stratified_metrics(
    candidate_ledger_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report candidate outcomes by every contract-required development stratum."""

    rows = [dict(row) for row in candidate_ledger_rows]
    dimensions = {
        "candidate_class": lambda row: str(row.get("candidate_state_target") or "UNLABELLED_CANDIDATE_STATE"),
        "universe": lambda row: str(row.get("universe") or "UNKNOWN"),
        "case_family": lambda row: str(row.get("case_family") or "UNKNOWN"),
        "small_far_proxy": lambda row: (
            "SMALL_FAR_PROXY" if row["small_far_proxy"]["is_small_far_proxy"] else "OTHER_SCALE_PROXY"
        ),
        "pitch_state": lambda row: str(row.get("pitch_state_target") or "UNLABELLED_PITCH_STATE"),
        "role": lambda row: str(row.get("role_target") or "UNLABELLED_ROLE"),
    }
    by_dimension: dict[str, dict[str, Any]] = {}
    for dimension, key_function in dimensions.items():
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[key_function(row)].append(row)
        by_dimension[dimension] = {key: _stratum_metrics(bucket_rows) for key, bucket_rows in sorted(buckets.items())}
    payload = {
        "schema_version": "football_intelligence.m5_5g7a.candidate_stratified_metrics.v1",
        "development_scope": DEVELOPMENT_LABEL,
        "population_count": len(rows),
        "dimensions": by_dimension,
        "required_dimensions_present": sorted(by_dimension) == sorted(dimensions),
        "small_far_is_proxy_not_human_truth": True,
        "unlabelled_strata_retained_as_not_evaluable": True,
    }
    payload["strata_hash"] = stable_hash(payload)
    return payload


def pair_relation_metrics(
    edge_rows: Sequence[Mapping[str, Any]],
    predicted_relations: Mapping[str, str],
) -> dict[str, Any]:
    """Evaluate every labelled pair with per-relation and source-normalized metrics."""

    ledger = []
    edge_ids: set[str] = set()
    for row in edge_rows:
        if not bool(row.get("target_available")) or row.get("target_relation") is None:
            continue
        edge_uuid = str(row.get("edge_uuid", ""))
        if not edge_uuid:
            raise ValueError("labelled pair rows require edge_uuid")
        if edge_uuid in edge_ids:
            raise ValueError(f"duplicate pair edge_uuid: {edge_uuid}")
        edge_ids.add(edge_uuid)
        if edge_uuid not in predicted_relations:
            raise ValueError(f"missing pair-relation prediction for {edge_uuid}")
        target = str(row["target_relation"])
        prediction = str(predicted_relations[edge_uuid])
        ledger.append(
            {
                "edge_uuid": edge_uuid,
                "source_group_id": str(row.get("source_group_id") or "UNKNOWN"),
                "source_frame_sha256": str(row.get("source_frame_sha256") or ""),
                "universe": str(row.get("universe") or "UNKNOWN"),
                "case_family": str(row.get("case_family") or "UNKNOWN"),
                "target_relation": target,
                "predicted_relation": prediction,
                "correct": target == prediction,
            }
        )
    ledger.sort(key=lambda row: row["edge_uuid"])
    relation_names = sorted(
        {str(row["target_relation"]) for row in ledger} | {str(row["predicted_relation"]) for row in ledger}
    )
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in ledger:
        confusion[str(row["target_relation"])][str(row["predicted_relation"])] += 1
    per_relation = {}
    for relation in relation_names:
        target_rows = [row for row in ledger if row["target_relation"] == relation]
        predicted_rows = [row for row in ledger if row["predicted_relation"] == relation]
        true_positive = sum(row["correct"] for row in target_rows)
        per_relation[relation] = {
            "target_support": len(target_rows),
            "predicted_support": len(predicted_rows),
            "true_positive": true_positive,
            "recall": true_positive / len(target_rows) if target_rows else None,
            "precision": true_positive / len(predicted_rows) if predicted_rows else None,
            "confusion_counts": dict(sorted(confusion.get(relation, {}).items())),
        }
    source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        source_rows[str(row["source_group_id"])].append(row)
    per_source_group = {
        group: {
            "denominator": len(rows),
            "correct": sum(row["correct"] for row in rows),
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
        }
        for group, rows in sorted(source_rows.items())
    }
    source_accuracies = [row["accuracy"] for row in per_source_group.values()]
    payload = {
        "schema_version": "football_intelligence.m5_5g7a.pair_relation_metrics.v1",
        "development_scope": DEVELOPMENT_LABEL,
        "labelled_edge_denominator": len(ledger),
        "correct_count": sum(row["correct"] for row in ledger),
        "accuracy": sum(row["correct"] for row in ledger) / len(ledger) if ledger else None,
        "confusion_matrix": {
            target: dict(sorted(predictions.items())) for target, predictions in sorted(confusion.items())
        },
        "per_relation": per_relation,
        "per_source_group": per_source_group,
        "source_group_normalized_accuracy": (
            sum(source_accuracies) / len(source_accuracies) if source_accuracies else None
        ),
        "all_labelled_edges_predicted": len(ledger) == len(edge_ids),
        "iou_used_as_primary_metric": False,
        "ledger": ledger,
    }
    payload["ledger_hash"] = stable_hash(ledger)
    payload["metrics_hash"] = stable_hash(payload)
    return payload


def k1_pending_receipt(
    node_rows: Sequence[Mapping[str, Any]],
    predictions_by_head: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Prove that unavailable K1-dependent heads remain masked and unknown."""

    field_by_head = {
        "team": "team_target",
        "kit": "kit_target",
        "participation": "participation_target",
    }
    example_ids = [str(row.get("example_uuid", "")) for row in node_rows]
    if any(not value for value in example_ids) or len(example_ids) != len(set(example_ids)):
        raise ValueError("node_rows must have unique non-empty example_uuid values")
    head_receipts = {}
    for head, target_field in field_by_head.items():
        predictions = predictions_by_head.get(head)
        if predictions is None:
            raise ValueError(f"missing prediction mapping for K1-dependent head {head}")
        target_ids = [str(row["example_uuid"]) for row in node_rows if row.get(target_field) is not None]
        mask_ids = [
            str(row["example_uuid"]) for row in node_rows if bool((row.get("label_availability_mask") or {}).get(head))
        ]
        missing_predictions = sorted(set(example_ids) - set(predictions))
        non_unknown_predictions = sorted(
            example_uuid
            for example_uuid in example_ids
            if example_uuid in predictions and str(predictions[example_uuid]) != UNKNOWN_BY_HEAD[head]
        )
        checks = {
            "target_count_zero": not target_ids,
            "availability_mask_count_zero": not mask_ids,
            "target_mask_sets_equal": set(target_ids) == set(mask_ids),
            "all_nodes_have_predictions": not missing_predictions,
            "all_predictions_route_to_unknown": not non_unknown_predictions,
        }
        head_receipts[head] = {
            "status": "NOT_EVALUABLE_K1_PENDING",
            "unknown_class": UNKNOWN_BY_HEAD[head],
            "target_count": len(target_ids),
            "availability_mask_count": len(mask_ids),
            "missing_prediction_example_uuids": missing_predictions,
            "non_unknown_prediction_example_uuids": non_unknown_predictions,
            "training_authorized": False,
            "evaluation_authorized": False,
            "checks": checks,
            "passed": all(checks.values()),
        }
    payload = {
        "schema_version": "football_intelligence.m5_5g7a.k1_pending_receipt.v1",
        "development_scope": DEVELOPMENT_LABEL,
        "status": "K1_TEAM_ROLE_KIT_PERSON_GOLD_PENDING_HUMAN_COMPLETION",
        "node_denominator": len(node_rows),
        "heads": head_receipts,
        "both_team_goalkeeper_classes_screen": "NOT_EVALUABLE_K1_PENDING",
        "warmup_player_recall_screen": "NOT_EVALUABLE_K1_PENDING",
        "warmup_staff_background_confusion_screen": "NOT_EVALUABLE_K1_PENDING",
        "team_goalkeeper_confusion_screen": "NOT_EVALUABLE_K1_PENDING",
        "team_and_kit_truth_fabricated": False,
        "geometry_candidate_baselines_blocked": False,
        "passed": all(row["passed"] for row in head_receipts.values()),
    }
    payload["receipt_hash"] = stable_hash(payload)
    return payload


def zero_harm_receipt(
    node_rows: Sequence[Mapping[str, Any]],
    node_states_before_scene_prior: Mapping[str, str],
    node_states_after_scene_prior: Mapping[str, str],
    *,
    generated_node_ids: Sequence[str] = (),
    hard_deleted_node_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Bind scene-prior safety to explicit goalkeeper and cardinality checks."""

    scene = scene_prior_safety(
        node_states_before_scene_prior,
        node_states_after_scene_prior,
        generated_node_ids=generated_node_ids,
        hard_deleted_node_ids=hard_deleted_node_ids,
    )
    goalkeeper_ids = {
        str(row["example_uuid"])
        for row in node_rows
        if row.get("role_target") == GOALKEEPER_ROLE and row.get("example_uuid") is not None
    }
    deleted_ids = set(scene["deleted_node_ids"])
    changed_to_non_clean = {
        node_id
        for node_id in goalkeeper_ids & set(node_states_before_scene_prior) & set(node_states_after_scene_prior)
        if node_states_before_scene_prior[node_id] == CLEAN_STATE
        and node_states_after_scene_prior[node_id] != CLEAN_STATE
    }
    goalkeeper_harm_ids = sorted((goalkeeper_ids & deleted_ids) | changed_to_non_clean)
    checks = {
        "zero_invented_observations": scene["invented_people_count"] == 0,
        "zero_hard_deleted_observations": scene["hard_deleted_people_count"] == 0,
        "zero_clean_person_deletion": scene["clean_person_deletion_count"] == 0,
        "zero_hard_goalkeeper_prior_deletion": not goalkeeper_harm_ids,
        "zero_hard_prediction_changes": scene["hard_prediction_change_count"] == 0,
        "no_exact_22_forcing": scene["exact_22_forcing_performed"] is False,
        "no_one_goalkeeper_per_team_forcing": scene["exactly_one_goalkeeper_per_team_forcing_performed"] is False,
    }
    payload = {
        "schema_version": "football_intelligence.m5_5g7a.zero_harm_receipt.v1",
        "development_scope": DEVELOPMENT_LABEL,
        "scene_prior_safety": scene,
        "labelled_goalkeeper_node_count": len(goalkeeper_ids),
        "hard_goalkeeper_prior_deletion_count": len(goalkeeper_harm_ids),
        "hard_goalkeeper_prior_deletion_example_uuids": goalkeeper_harm_ids,
        "checks": checks,
        "count_prior_harmful_action": not all(checks.values()),
        "passed": all(checks.values()),
    }
    payload["receipt_hash"] = stable_hash(payload)
    return payload


def scene_prior_safety(
    node_states_before: Mapping[str, str],
    node_states_after: Mapping[str, str],
    *,
    generated_node_ids: Sequence[str] = (),
    hard_deleted_node_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Audit that scene warnings did not invent or hard-delete observations."""

    before_ids = set(node_states_before)
    after_ids = set(node_states_after)
    declared_generated = set(generated_node_ids)
    declared_deleted = set(hard_deleted_node_ids)
    invented = sorted((after_ids - before_ids) | declared_generated)
    deleted = sorted((before_ids - after_ids) | declared_deleted)
    clean_deleted = sorted(
        node_id for node_id in before_ids & set(deleted) if node_states_before[node_id] == CLEAN_STATE
    )
    changed = sorted(
        node_id for node_id in before_ids & after_ids if node_states_before[node_id] != node_states_after[node_id]
    )
    result = {
        "schema_version": "football_intelligence.m5_5g7a.scene_prior_safety.v1",
        "development_scope": DEVELOPMENT_LABEL,
        "invented_people_count": len(invented),
        "hard_deleted_people_count": len(deleted),
        "clean_person_deletion_count": len(clean_deleted),
        "hard_prediction_change_count": len(changed),
        "invented_node_ids": invented,
        "deleted_node_ids": deleted,
        "changed_node_ids": changed,
        "count_prior_warning_only": not invented and not deleted and not changed,
        "exact_22_forcing_performed": False,
        "exactly_one_goalkeeper_per_team_forcing_performed": False,
        "passed": not invented and not deleted and not changed,
    }
    return result


def candidate_development_screen(
    candidate_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    *,
    selective_risk_improved: bool,
    deterministic: bool,
    provenance_complete: bool,
) -> dict[str, Any]:
    """Apply the immutable candidate-state development screen."""

    duplicate = candidate_metrics["duplicate_accepted_rate"]
    baseline_supply = baseline_metrics["independent_person_supply"]
    candidate_supply = candidate_metrics["independent_person_supply"]
    checks = {
        "merged_as_clean_zero": int(candidate_metrics["merged_as_clean_count"]) == 0,
        "duplicate_accepted_rate_at_most_0_02": duplicate["rate"] is not None and float(duplicate["rate"]) <= 0.02,
        "zero_clean_control_regression": int(candidate_metrics["clean_control_preservation"]["errors"])
        <= int(baseline_metrics["clean_control_preservation"]["errors"]),
        "no_material_independent_supply_regression": int(candidate_supply["numerator"])
        >= int(baseline_supply["numerator"]),
        "distinct_person_suppression_no_worse_than_r0": int(candidate_metrics["distinct_person_suppression"])
        <= int(baseline_metrics["distinct_person_suppression"]),
        "selective_risk_improved": bool(selective_risk_improved),
        "provenance_complete": bool(provenance_complete),
        "deterministic": bool(deterministic),
    }
    return {
        "schema_version": "football_intelligence.m5_5g7a.candidate_development_screen.v1",
        "development_scope": DEVELOPMENT_LABEL,
        "checks": checks,
        "passed": all(checks.values()),
        "thresholds_weakened": False,
    }


def required_ablation_variants() -> tuple[str, ...]:
    """Return the immutable minimum ablation matrix."""

    return (
        "GEOMETRY_ONLY",
        "VISUAL_ONLY",
        "VISUAL_PLUS_GEOMETRY",
        "VISUAL_GEOMETRY_COLOUR_KIT",
        "NODE_PLUS_PAIR_EDGES",
        "GRAPH_WITHOUT_SCENE_PRIOR",
        "GRAPH_WITH_SCENE_PRIOR",
    )


__all__ = [
    "categorical_head_metrics",
    "candidate_confusion_audits",
    "candidate_development_screen",
    "candidate_outcomes",
    "candidate_stratified_metrics",
    "exhaustive_candidate_person_ledgers",
    "expected_calibration_error",
    "k1_pending_receipt",
    "pair_relation_metrics",
    "required_ablation_variants",
    "scene_prior_safety",
    "selective_risk_curve",
    "zero_harm_receipt",
]
