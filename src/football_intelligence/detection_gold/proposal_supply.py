"""Exploratory proposal-supply diagnostics for reviewed detection gold.

The helpers in this module are deliberately detector-agnostic.  They consume
frozen human annotations and frozen provenance rows; they never execute or
configure a detector and never reinterpret a human relation label.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import median
from typing import Any

from football_intelligence.review_chassis.hashing import stable_hash

RELATIONS = {
    "CLEAN_SINGLE_INSTANCE",
    "DUPLICATE_OF_INSTANCE",
    "MERGED_MULTIPLE_INSTANCES",
    "PARTIAL_INSTANCE",
    "BACKGROUND",
    "AMBIGUOUS",
}
PERSON_SUPPORT_RELATIONS = RELATIONS - {"BACKGROUND"}
STAGE_ORDER = ("RAW", "CONFIDENCE", "PRE_NMS", "POST_NMS", "FUSED")
SUPPLY_STATES = {
    "ANY_PERSON_SUPPORT",
    "CLEAN_SINGLE_COVERAGE",
    "PARTIAL_ONLY",
    "MERGED_ONLY",
    "DUPLICATE_ONLY",
    "AMBIGUOUS_ONLY",
    "NO_REVIEWED_SUPPORT",
}


def exact_fraction(numerator: int, denominator: int) -> dict[str, int | float | None]:
    """Return an explicit numerator, denominator, and descriptive rate."""

    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": round(numerator / denominator, 8) if denominator else None,
    }


def bbox_iou(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    x1 = max(float(left["x1"]), float(right["x1"]))
    y1 = max(float(left["y1"]), float(right["y1"]))
    x2 = min(float(left["x2"]), float(right["x2"]))
    y2 = min(float(left["y2"]), float(right["y2"]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (float(left["x2"]) - float(left["x1"])) * (float(left["y2"]) - float(left["y1"]))
    right_area = (float(right["x2"]) - float(right["x1"])) * (float(right["y2"]) - float(right["y1"]))
    return intersection / max(1e-12, left_area + right_area - intersection)


def bbox_height(box: Mapping[str, float]) -> float:
    return float(box["y2"]) - float(box["y1"])


def _centre(box: Mapping[str, float]) -> tuple[float, float]:
    return (
        (float(box["x1"]) + float(box["x2"])) / 2,
        (float(box["y1"]) + float(box["y2"])) / 2,
    )


def normalized_displacements(candidate: Mapping[str, float], gold: Mapping[str, float]) -> dict[str, float]:
    """Measure centre and bottom-centre displacement in visible-height units."""

    scale = max(1e-9, bbox_height(gold))
    candidate_centre = _centre(candidate)
    gold_centre = _centre(gold)
    centre = math.dist(candidate_centre, gold_centre) / scale
    candidate_bottom = (candidate_centre[0], float(candidate["y2"]))
    gold_bottom = (gold_centre[0], float(gold["y2"]))
    return {
        "centre_displacement_visible_heights": round(math.dist(candidate_centre, gold_centre) / scale, 8),
        "bottom_centre_displacement_visible_heights": round(math.dist(candidate_bottom, gold_bottom) / scale, 8),
        "visible_height_pixels": round(scale, 8),
        "centre_displacement_pixels": round(centre * scale, 8),
    }


def box_in_bounds(box: Mapping[str, float], width: int, height: int, *, tolerance: float = 1e-6) -> bool:
    values = [float(box[key]) for key in ("x1", "y1", "x2", "y2")]
    return (
        all(math.isfinite(value) for value in values)
        and -tolerance <= values[0] < values[2] <= width + tolerance
        and -tolerance <= values[1] < values[3] <= height + tolerance
    )


def point_in_bounds(point: Mapping[str, float], width: int, height: int, *, tolerance: float = 1e-6) -> bool:
    x, y = float(point["x"]), float(point["y"])
    return (
        math.isfinite(x)
        and math.isfinite(y)
        and -tolerance <= x <= width + tolerance
        and -tolerance <= y <= height + tolerance
    )


def box_within_roi(box: Mapping[str, float], roi: Mapping[str, float], *, tolerance: float = 1e-6) -> bool:
    return (
        float(box["x1"]) >= float(roi["x1"]) - tolerance
        and float(box["y1"]) >= float(roi["y1"]) - tolerance
        and float(box["x2"]) <= float(roi["x2"]) + tolerance
        and float(box["y2"]) <= float(roi["y2"]) + tolerance
    )


def validate_relation_cardinality(relation: Mapping[str, Any], valid_targets: set[str]) -> list[str]:
    errors: list[str] = []
    label = str(relation.get("relation"))
    targets = [str(value) for value in relation.get("annotation_uuids", [])]
    if label not in RELATIONS:
        errors.append(f"unknown relation: {label}")
    if len(targets) != len(set(targets)):
        errors.append("duplicate relation target")
    if not set(targets) <= valid_targets:
        errors.append("unknown relation target")
    if label == "BACKGROUND" and targets:
        errors.append("BACKGROUND requires zero targets")
    if label in {"CLEAN_SINGLE_INSTANCE", "DUPLICATE_OF_INSTANCE", "PARTIAL_INSTANCE"} and len(targets) != 1:
        errors.append(f"{label} requires one target")
    if label == "MERGED_MULTIPLE_INSTANCES" and len(targets) < 2:
        errors.append("MERGED_MULTIPLE_INSTANCES requires at least two targets")
    return errors


def replay_detection_case_events(
    events: Sequence[Mapping[str, Any]], expected_case_ids: Sequence[str]
) -> dict[str, Any]:
    """Materialize strict case-save events while retaining legitimate resaves."""

    expected = set(expected_case_ids)
    saves = [row for row in events if row.get("event_type") == "DETECTION_CASE_SAVED"]
    completions = [row for row in events if row.get("event_type") == "DETECTION_TRANCHE_COMPLETED"]
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    sequence_values: list[int] = []
    for event in events:
        sequence_values.append(int(event.get("event_sequence", -1)))
    for event in saves:
        by_case[str(event.get("case_id"))].append(event)
    final_events = {case_id: rows[-1] for case_id, rows in by_case.items() if rows}
    resaves = {case_id: len(rows) - 1 for case_id, rows in by_case.items() if len(rows) > 1}
    checks = {
        "strict_event_count_20": len(events) == 20,
        "case_save_count_19": len(saves) == 19,
        "completion_event_count_1": len(completions) == 1,
        "event_sequences_contiguous": sequence_values == list(range(1, len(events) + 1)),
        "exact_final_case_set": set(final_events) == expected,
        "case_029_single_resave": resaves == {"m5_5g1a_case_029": 1},
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "event_count": len(events),
        "case_save_count": len(saves),
        "completion_event_count": len(completions),
        "final_case_count": len(final_events),
        "resave_counts": resaves,
        "final_events": final_events,
        "completion_event": completions[0] if len(completions) == 1 else None,
    }


def build_source_groups(case_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[str(row["source_frame_sha256"])].append(row)
    output: list[dict[str, Any]] = []
    for source_hash, rows in sorted(grouped.items()):
        case_ids = sorted(str(row["case_id"]) for row in rows)
        output.append(
            {
                "source_group_id": f"source_group_{source_hash[:16]}",
                "source_frame_sha256": source_hash,
                "case_ids": case_ids,
                "case_record_count": len(rows),
                "raw_human_person_count": sum(len(row["player_instances"]) for row in rows),
                "focal_rois": [row["focal_roi"] for row in rows],
                "duplicate_source_group": len(rows) > 1,
            }
        )
    return output


def _pair_evidence(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_box, right_box = left["visible_body_box"], right["visible_body_box"]
    mean_height = max(1e-9, (bbox_height(left_box) + bbox_height(right_box)) / 2)
    centre_distance = math.dist(_centre(left_box), _centre(right_box))
    left_foot = left.get("footpoint") or {"x": _centre(left_box)[0], "y": left_box["y2"]}
    right_foot = right.get("footpoint") or {"x": _centre(right_box)[0], "y": right_box["y2"]}
    foot_distance = math.dist(
        (float(left_foot["x"]), float(left_foot["y"])), (float(right_foot["x"]), float(right_foot["y"]))
    )
    return {
        "visible_body_iou": round(bbox_iou(left_box, right_box), 8),
        "centre_distance_visible_heights": round(centre_distance / mean_height, 8),
        "footpoint_distance_visible_heights": round(foot_distance / mean_height, 8),
        "height_ratio": round(
            min(bbox_height(left_box), bbox_height(right_box)) / max(bbox_height(left_box), bbox_height(right_box)), 8
        ),
        "same_role": left.get("coarse_role") == right.get("coarse_role"),
        "same_pitch_state": left.get("pitch_state") == right.get("pitch_state"),
    }


def cluster_cross_case_gold(case_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Conservatively merge only strong mutual-nearest same-source pairs."""

    people: list[dict[str, Any]] = []
    for case in case_rows:
        for person in case["player_instances"]:
            people.append(
                {
                    **person,
                    "case_id": case["case_id"],
                    "source_frame_sha256": case["source_frame_sha256"],
                    "source_group_id": case["source_group_id"],
                    "focal_roi": case["focal_roi"],
                }
            )
    parents = list(range(len(people)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    by_source: dict[str, list[int]] = defaultdict(list)
    for index, person in enumerate(people):
        by_source[person["source_frame_sha256"]].append(index)

    proposals: list[dict[str, Any]] = []
    for indices in by_source.values():
        case_ids = {people[index]["case_id"] for index in indices}
        if len(case_ids) < 2:
            continue
        pair_rows: list[tuple[int, int, dict[str, Any]]] = []
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                if people[left]["case_id"] == people[right]["case_id"]:
                    continue
                evidence = _pair_evidence(people[left], people[right])
                if evidence["visible_body_iou"] >= 0.15 or evidence["footpoint_distance_visible_heights"] <= 0.75:
                    pair_rows.append((left, right, evidence))
        nearest: dict[int, tuple[int, float]] = {}
        for left, right, evidence in pair_rows:
            score = evidence["footpoint_distance_visible_heights"] + (1 - evidence["visible_body_iou"])
            for source, target in ((left, right), (right, left)):
                if source not in nearest or score < nearest[source][1]:
                    nearest[source] = (target, score)
        for left, right, evidence in pair_rows:
            mutual = nearest.get(left, (None,))[0] == right and nearest.get(right, (None,))[0] == left
            compatible = evidence["same_role"] and evidence["same_pitch_state"]
            high = (
                mutual
                and compatible
                and (
                    evidence["visible_body_iou"] >= 0.5
                    or (
                        evidence["footpoint_distance_visible_heights"] <= 0.35
                        and evidence["centre_distance_visible_heights"] <= 0.45
                        and evidence["height_ratio"] >= 0.55
                    )
                )
            )
            confidence = "high" if high else "medium" if mutual and compatible else "low"
            proposal = {
                "left_case_id": people[left]["case_id"],
                "left_annotation_uuid": people[left]["annotation_uuid"],
                "right_case_id": people[right]["case_id"],
                "right_annotation_uuid": people[right]["annotation_uuid"],
                "source_group_id": people[left]["source_group_id"],
                "evidence": evidence,
                "mutual_nearest": mutual,
                "confidence": confidence,
                "canonical_merge_applied": high,
                "manual_review_required": True,
            }
            proposals.append(proposal)
            if high:
                union(left, right)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(people)):
        grouped[find(index)].append(index)
    clusters: list[dict[str, Any]] = []
    member_to_cluster: dict[tuple[str, str], str] = {}
    for indices in sorted(grouped.values(), key=lambda values: min(values)):
        members = [people[index] for index in indices]
        public_binding = sorted((member["case_id"], member["annotation_uuid"]) for member in members)
        cluster_id = f"gold_person_{stable_hash(public_binding)[:16]}"
        for member in members:
            member_to_cluster[(member["case_id"], member["annotation_uuid"])] = cluster_id
        clusters.append(
            {
                "canonical_gold_person_cluster_id": cluster_id,
                "source_group_id": members[0]["source_group_id"],
                "source_frame_sha256": members[0]["source_frame_sha256"],
                "members": [
                    {"case_id": member["case_id"], "annotation_uuid": member["annotation_uuid"]} for member in members
                ],
                "member_count": len(members),
                "case_record_count": len({member["case_id"] for member in members}),
                "coarse_roles": sorted({str(member["coarse_role"]) for member in members}),
                "pitch_states": sorted({str(member["pitch_state"]) for member in members}),
                "visibility_states": sorted({str(member["visibility_state"]) for member in members}),
                "visible_heights_pixels": [round(bbox_height(member["visible_body_box"]), 8) for member in members],
                "canonical_visible_body_box": members[0]["visible_body_box"],
                "manual_review_required": len(members) > 1,
                "single_reviewer_development_gold": True,
            }
        )
    return {
        "proposals": proposals,
        "clusters": clusters,
        "member_to_cluster": member_to_cluster,
        "raw_human_person_count": len(people),
        "canonical_gold_person_cluster_count": len(clusters),
    }


def supply_state(relations: Iterable[str]) -> dict[str, Any]:
    labels = {str(value) for value in relations if str(value) in PERSON_SUPPORT_RELATIONS}
    if not labels:
        primary = "NO_REVIEWED_SUPPORT"
    elif "CLEAN_SINGLE_INSTANCE" in labels:
        primary = "CLEAN_SINGLE_COVERAGE"
    elif labels == {"PARTIAL_INSTANCE"}:
        primary = "PARTIAL_ONLY"
    elif labels == {"MERGED_MULTIPLE_INSTANCES"}:
        primary = "MERGED_ONLY"
    elif labels == {"DUPLICATE_OF_INSTANCE"}:
        primary = "DUPLICATE_ONLY"
    elif labels == {"AMBIGUOUS"}:
        primary = "AMBIGUOUS_ONLY"
    else:
        primary = "ANY_PERSON_SUPPORT"
    return {
        "primary_supply_state": primary,
        "relation_labels": sorted(labels),
        "any_person_support": bool(labels),
        "clean_single_coverage": "CLEAN_SINGLE_INSTANCE" in labels,
        "independent_person_supply": bool(labels & {"CLEAN_SINGLE_INSTANCE", "PARTIAL_INSTANCE"}),
        "merged_support_present": "MERGED_MULTIPLE_INSTANCES" in labels,
        "duplicate_burden_present": "DUPLICATE_OF_INSTANCE" in labels,
    }


def height_bin(height: float) -> str:
    if height < 24:
        return "LT_24_PX"
    if height < 40:
        return "24_TO_LT_40_PX"
    if height < 64:
        return "40_TO_LT_64_PX"
    return "GE_64_PX"


def candidate_count_outlier_summary(case_counts: Mapping[str, int]) -> dict[str, Any]:
    values = list(case_counts.values())
    if not values:
        raise ValueError("case counts cannot be empty")
    relation_totals = sum(values)
    maximum_case = max(case_counts, key=case_counts.__getitem__)
    return {
        "case_count": len(values),
        "total_reviewed_candidate_relations": relation_totals,
        "median_candidate_relations_per_case": median(values),
        "range_candidate_relations_per_case": [min(values), max(values)],
        "maximum_case_id": maximum_case,
        "maximum_case_candidate_relations": case_counts[maximum_case],
        "maximum_case_pooled_share": exact_fraction(case_counts[maximum_case], relation_totals),
        "case_counts": dict(sorted(case_counts.items())),
        "primary_conclusions_weighted_by_candidate_count": False,
        "resampling_group": "source_frame_sha256",
    }


def relation_composition_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pooled = Counter(str(row["relation"]) for row in rows)
    per_case: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        per_case[str(row["case_id"])][str(row["relation"])] += 1
    labels = sorted(RELATIONS)
    normalized: dict[str, float] = {}
    for label in labels:
        fractions = [counts[label] / sum(counts.values()) for counts in per_case.values()]
        normalized[label] = round(sum(fractions) / len(fractions), 8) if fractions else 0.0
    return {
        "pooled_candidate_composition": {label: exact_fraction(pooled[label], len(rows)) for label in labels},
        "per_case_normalized_mean_relation_share": normalized,
        "per_case_count": len(per_case),
        "pooled_candidate_result_is_primary": False,
    }


def provisional_person_origin(stage_relations: Mapping[str, Sequence[str]]) -> str:
    raw = set(stage_relations.get("RAW", []))
    confidence = set(stage_relations.get("CONFIDENCE", []))
    pre_nms = set(stage_relations.get("PRE_NMS", []))
    post_nms = set(stage_relations.get("POST_NMS", []))
    fused = set(stage_relations.get("FUSED", []))
    if not raw:
        return "NO_VALID_RAW_PROPOSAL"
    if raw and not confidence:
        return "VALID_PROPOSAL_LOW_CONFIDENCE"
    if pre_nms & {"CLEAN_SINGLE_INSTANCE", "PARTIAL_INSTANCE"} and not post_nms & {
        "CLEAN_SINGLE_INSTANCE",
        "PARTIAL_INSTANCE",
    }:
        return "VALID_PROPOSALS_NMS_COLLAPSED"
    if "DUPLICATE_OF_INSTANCE" in fused:
        return "DUPLICATED_AFTER_VIEW_FUSION"
    return "UNRESOLVED"


def reconcile_origin(human: str, computed_counts: Mapping[str, int]) -> dict[str, Any]:
    supported = {key: value for key, value in computed_counts.items() if key != "UNRESOLVED" and value > 0}
    if len(supported) == 1:
        computed = next(iter(supported))
    elif supported:
        computed = "INSUFFICIENT_EVIDENCE"
    else:
        computed = "UNRESOLVED"
    agreement = human == computed and human != "UNRESOLVED"
    contradiction = human != "UNRESOLVED" and computed not in {human, "UNRESOLVED", "INSUFFICIENT_EVIDENCE"}
    insufficient = computed in {"UNRESOLVED", "INSUFFICIENT_EVIDENCE"} or human == "UNRESOLVED"
    return {
        "human_earliest_failure_stage": human,
        "computed_provisional_stage_origin": computed,
        "computed_person_origin_counts": dict(sorted(computed_counts.items())),
        "agreement": agreement,
        "contradiction": contradiction,
        "insufficient_evidence": insufficient,
        "review_recommendation": "MANUAL_REVIEW"
        if contradiction or (human == "UNRESOLVED" and supported)
        else "PRESERVE_AS_REVIEWED",
    }
