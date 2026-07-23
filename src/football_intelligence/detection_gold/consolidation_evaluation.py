"""Development-only evaluation for provenance-aware proposal consolidation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

from football_intelligence.detection_gold.proposal_supply import (
    bbox_area,
    bbox_height,
    box_contains_point,
    deterministic_one_to_one_supply,
    exact_fraction,
    proposal_gold_geometry,
)

INDEPENDENT_STATES = {
    "INDEPENDENT_SINGLE_SUPPORT",
    "INDEPENDENT_SUPPORT_WITH_DUPLICATE_BURDEN",
}


def _centre(box: Mapping[str, float]) -> tuple[float, float]:
    return ((float(box["x1"]) + float(box["x2"])) / 2, (float(box["y1"]) + float(box["y2"])) / 2)


def _intersection_box(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float] | None:
    result = {
        "x1": max(float(left["x1"]), float(right["x1"])),
        "y1": max(float(left["y1"]), float(right["y1"])),
        "x2": min(float(left["x2"]), float(right["x2"])),
        "y2": min(float(left["y2"]), float(right["y2"])),
    }
    return result if result["x1"] < result["x2"] and result["y1"] < result["y2"] else None


def rectangle_union_intersection_area(box: Mapping[str, float], rectangles: Sequence[Mapping[str, float]]) -> float:
    """Return box area covered by the union of axis-aligned rectangles."""

    clipped = [value for rectangle in rectangles if (value := _intersection_box(box, rectangle))]
    if not clipped:
        return 0.0
    x_values = sorted({float(value[key]) for value in clipped for key in ("x1", "x2")})
    area = 0.0
    for left, right in zip(x_values, x_values[1:], strict=False):
        if right <= left:
            continue
        intervals = sorted(
            (float(value["y1"]), float(value["y2"]))
            for value in clipped
            if float(value["x1"]) < right and float(value["x2"]) > left
        )
        if not intervals:
            continue
        covered = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start > end:
                covered += end - start
                start, end = next_start, next_end
            else:
                end = max(end, next_end)
        covered += end - start
        area += (right - left) * covered
    return area


def classify_box_against_roi_union(
    box: Mapping[str, float], rectangles: Sequence[Mapping[str, float]]
) -> dict[str, Any]:
    """Apply the frozen labelled-ROI inclusion and boundary rules."""

    centre_inside = any(box_contains_point(rectangle, _centre(box)) for rectangle in rectangles)
    overlap_area = rectangle_union_intersection_area(box, rectangles)
    overlap_fraction = overlap_area / max(1e-12, bbox_area(box))
    if centre_inside or overlap_fraction >= 0.50:
        state = "INCLUDED_IN_FALSE_OBSERVATION_EVALUATION"
    elif overlap_area > 0:
        state = "ROI_BOUNDARY_IGNORED"
    else:
        state = "OUTSIDE_LABELLED_ROI_IGNORED"
    return {
        "evaluation_roi_state": state,
        "centre_inside_labelled_roi_union": centre_inside,
        "area_fraction_inside_labelled_roi_union": round(overlap_fraction, 8),
    }


def build_evaluation_roi_manifest(
    source_groups: Sequence[Mapping[str, Any]], gold_clusters: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build source-group ROI unions before any variant is scored."""

    gold_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cluster in gold_clusters:
        gold_by_source[str(cluster["source_frame_sha256"])].append(cluster)
    rows = []
    for source in sorted(source_groups, key=lambda row: str(row["source_frame_sha256"])):
        source_hash = str(source["source_frame_sha256"])
        rectangles = [{key: float(roi[key]) for key in ("x1", "y1", "x2", "y2")} for roi in source["focal_rois"]]
        gold_rows = gold_by_source[source_hash]
        included_gold = []
        boundary_gold = []
        for cluster in gold_rows:
            classification = classify_box_against_roi_union(cluster["canonical_visible_body_box"], rectangles)
            target = included_gold if classification["evaluation_roi_state"].startswith("INCLUDED") else boundary_gold
            target.append(str(cluster["canonical_gold_person_cluster_id"]))
        rows.append(
            {
                "source_group_id": source["source_group_id"],
                "source_frame_sha256": source_hash,
                "case_ids": source["case_ids"],
                "labelled_focal_roi_rectangles": rectangles,
                "canonical_gold_person_ids_inside_union": sorted(included_gold),
                "canonical_gold_person_ids_boundary_or_outside": sorted(boundary_gold),
                "false_observation_scoring_rule": (
                    "proposal centre inside ROI union OR at least 50 percent proposal area inside ROI union"
                ),
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5g3.evaluation_roi_manifest.v1",
        "source_group_count": len(rows),
        "rows": rows,
        "outside_roi_false_observation_scoring_performed": False,
        "gold_used_by_runtime_consolidator": False,
    }


def _geometry_edge(metrics: Mapping[str, Any], *, tiny: bool) -> str | None:
    iou = float(metrics["visible_box_iou"])
    coverage = float(metrics["gold_visible_area_coverage"])
    centre = float(metrics["centre_displacement_visible_heights"])
    bottom = float(metrics["bottom_centre_displacement_visible_heights"])
    contains = bool(metrics["candidate_contains_gold_centre"])
    strong = iou >= 0.30 or (contains and coverage >= 0.50 and bottom <= 0.75)
    if tiny:
        strong = strong or (contains and centre <= 1.0 and bottom <= 1.0 and coverage >= 0.25)
    weak = iou >= 0.10 or (contains and coverage >= 0.25) or (centre <= 1.25 and bottom <= 1.25)
    return "STRONG" if strong else "WEAK" if weak else None


def proposal_gold_support_sets(
    proposals: Sequence[Mapping[str, Any]], gold_rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, list[str]]]:
    """Return evaluation-only strong and weak gold support sets."""

    result: dict[str, dict[str, list[str]]] = {}
    for proposal in proposals:
        proposal_id = str(proposal["proposal_id"])
        strong: list[str] = []
        weak: list[str] = []
        for gold in gold_rows:
            metrics = proposal_gold_geometry(proposal["bbox"], gold["bbox"])
            edge = _geometry_edge(metrics, tiny=bbox_height(gold["bbox"]) < 12)
            if edge == "STRONG":
                strong.append(str(gold["gold_person_id"]))
            elif edge == "WEAK":
                weak.append(str(gold["gold_person_id"]))
        result[proposal_id] = {"strong": sorted(strong), "weak": sorted(weak)}
    return result


def derive_pair_label(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    support_sets: Mapping[str, Mapping[str, Sequence[str]]],
    human_relations: Mapping[str, Sequence[str]],
) -> str:
    """Derive a development-only proposal pair label."""

    left_id, right_id = str(left["proposal_id"]), str(right["proposal_id"])
    left_support = set(support_sets[left_id]["strong"])
    right_support = set(support_sets[right_id]["strong"])
    relations = set(human_relations.get(left_id, ())) | set(human_relations.get(right_id, ()))
    if "MERGED_MULTIPLE_INSTANCES" in relations or len(left_support) > 1 or len(right_support) > 1:
        return "MERGED_OR_MULTI_PERSON"
    if len(left_support) == len(right_support) == 1:
        return "SAME_PERSON_ALTERNATIVES" if left_support == right_support else "DISTINCT_PEOPLE"
    if left_support and right_support and left_support.isdisjoint(right_support):
        return "DISTINCT_PEOPLE"
    if (
        not left_support
        and not right_support
        and ("BACKGROUND" in relations or (not support_sets[left_id]["weak"] and not support_sets[right_id]["weak"]))
    ):
        return "BACKGROUND_OR_UNSUPPORTED"
    return "INSUFFICIENT_EVIDENCE"


def _observation_proposals(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "proposal_id": row["observation_uuid"],
            "bbox": row["box_panorama_pixels"],
            "score": row["score"],
        }
        for row in observations
    ]


def evaluate_source_observations(
    gold_clusters: Sequence[Mapping[str, Any]],
    preconsolidation_nodes: Sequence[Mapping[str, Any]],
    consolidation_result: Mapping[str, Any],
    labelled_rois: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    """Evaluate one source while excluding outside-ROI false observations."""

    gold_rows = [
        {
            "gold_person_id": str(cluster["canonical_gold_person_cluster_id"]),
            "bbox": cluster["canonical_visible_body_box"],
        }
        for cluster in gold_clusters
        if classify_box_against_roi_union(cluster["canonical_visible_body_box"], labelled_rois)[
            "evaluation_roi_state"
        ].startswith("INCLUDED")
    ]
    metadata = {str(cluster["canonical_gold_person_cluster_id"]): cluster for cluster in gold_clusters}
    included_nodes = [
        node
        for node in preconsolidation_nodes
        if classify_box_against_roi_union(node["bbox_panorama_pixels"], labelled_rois)[
            "evaluation_roi_state"
        ].startswith("INCLUDED")
    ]
    pre_proposals = [
        {"proposal_id": node["proposal_uuid"], "bbox": node["bbox_panorama_pixels"], "score": node["score"]}
        for node in included_nodes
    ]
    pre_support = proposal_gold_support_sets(pre_proposals, gold_rows)
    pre_independent_ids = {
        gold_id for support in pre_support.values() if len(support["strong"]) == 1 for gold_id in support["strong"]
    }

    annotated_observations = []
    boundary_ledger = []
    for observation in consolidation_result["observations"]:
        roi = classify_box_against_roi_union(observation["box_panorama_pixels"], labelled_rois)
        row = {**observation, **roi}
        if roi["evaluation_roi_state"].startswith("INCLUDED"):
            annotated_observations.append(row)
        else:
            boundary_ledger.append(
                {
                    "observation_uuid": observation["observation_uuid"],
                    "output_state": observation["output_state"],
                    **roi,
                }
            )
    accepted = [row for row in annotated_observations if row["output_state"] == "ACCEPT_INDEPENDENT_OBSERVATION"]
    routed = [row for row in annotated_observations if row["output_state"] == "ROUTE_DENSE_REVIEW"]
    accepted_proposals = _observation_proposals(accepted)
    routed_proposals = _observation_proposals(routed)
    accepted_match = deterministic_one_to_one_supply(gold_rows, accepted_proposals)
    accepted_by_gold = {row["gold_person_id"]: row for row in accepted_match["person_rows"]}
    routed_support = proposal_gold_support_sets(routed_proposals, gold_rows)
    routed_person_ids = {gold_id for support in routed_support.values() for gold_id in support["strong"]}
    accepted_person_ids = {
        gold_id for gold_id, row in accepted_by_gold.items() if row["assigned_proposal_id"] is not None
    }
    duplicate_count = sum(
        max(0, int(row["strong_independent_candidate_count"]) - 1) for row in accepted_by_gold.values()
    )
    merged_as_clean_ids = set(accepted_match["merged_proposal_ids"])
    accepted_support = proposal_gold_support_sets(accepted_proposals, gold_rows)
    background_ids = {
        proposal["proposal_id"]
        for proposal in accepted_proposals
        if not accepted_support[proposal["proposal_id"]]["strong"]
        and not accepted_support[proposal["proposal_id"]]["weak"]
    }
    distinct_suppressed_ids = pre_independent_ids - accepted_person_ids
    assignment_geometries = [row["geometry"] for row in accepted_match["assignments"]]
    person_rows = []
    for gold in gold_rows:
        gold_id = str(gold["gold_person_id"])
        row = accepted_by_gold[gold_id]
        cluster = metadata[gold_id]
        person_rows.append(
            {
                "canonical_gold_person_cluster_id": gold_id,
                "source_group_id": cluster["source_group_id"],
                "case_ids": cluster.get("case_ids", []),
                "accepted_observation_uuid": row["assigned_proposal_id"],
                "accepted_exactly_one": row["supply_state"] == "INDEPENDENT_SINGLE_SUPPORT",
                "accepted_any": row["assigned_proposal_id"] is not None,
                "routed_to_dense_review": gold_id in routed_person_ids,
                "accepted_plus_dense_covered": gold_id in accepted_person_ids | routed_person_ids,
                "preconsolidation_independent_support": gold_id in pre_independent_ids,
                "distinct_person_suppressed": gold_id in distinct_suppressed_ids,
                "duplicate_final_observation_count": max(0, int(row["strong_independent_candidate_count"]) - 1),
                "visible_height_pixels": cluster.get("median_visible_height_pixels"),
                "visible_height_bin": cluster.get("visible_height_bin"),
                "visibility_states": cluster.get("visibility_states", []),
                "occlusion_types": cluster.get("occlusion_types", []),
                "pitch_states": cluster.get("pitch_states", []),
                "original_case_strata": cluster.get("original_case_strata", []),
            }
        )
    denominator = len(gold_rows)
    metrics = {
        "canonical_gold_person_count": denominator,
        "exactly_one_accepted_observation": exact_fraction(
            sum(row["accepted_exactly_one"] for row in person_rows), denominator
        ),
        "accepted_independent_supply": exact_fraction(len(accepted_person_ids), denominator),
        "no_accepted_observation": exact_fraction(denominator - len(accepted_person_ids), denominator),
        "routed_to_dense_review": exact_fraction(len(routed_person_ids), denominator),
        "accepted_plus_dense_routed_coverage": exact_fraction(
            len(accepted_person_ids | routed_person_ids), denominator
        ),
        "duplicate_final_observation_count": duplicate_count,
        "duplicate_final_observation_rate": exact_fraction(duplicate_count, denominator),
        "merged_as_clean_observation_count": len(merged_as_clean_ids),
        "merged_as_clean_observation_rate": exact_fraction(len(merged_as_clean_ids), max(1, len(accepted))),
        "preconsolidation_independent_support_count": len(pre_independent_ids),
        "distinct_person_suppression_count": len(distinct_suppressed_ids),
        "distinct_person_suppression_rate": exact_fraction(len(distinct_suppressed_ids), len(pre_independent_ids)),
        "background_accepted_observation_count": len(background_ids),
        "accepted_observation_count_inside_labelled_roi": len(accepted),
        "dense_review_observation_count_inside_labelled_roi": len(routed),
        "observation_count_error": len(accepted) - denominator,
        "absolute_observation_count_error": abs(len(accepted) - denominator),
        "preconsolidation_candidate_count_inside_labelled_roi": len(included_nodes),
        "candidate_reduction_ratio": round(1 - (len(accepted) + len(routed)) / max(1, len(included_nodes)), 8),
        "median_visible_box_iou": round(median(float(row["visible_box_iou"]) for row in assignment_geometries), 8)
        if assignment_geometries
        else None,
        "median_normalized_bottom_centre_displacement": round(
            median(float(row["bottom_centre_displacement_visible_heights"]) for row in assignment_geometries),
            8,
        )
        if assignment_geometries
        else None,
    }
    return {
        "source_frame_sha256": consolidation_result["source_frame_sha256"],
        "metrics": metrics,
        "person_rows": person_rows,
        "accepted_assignments": accepted_match["assignments"],
        "merged_as_clean_observation_uuids": sorted(merged_as_clean_ids),
        "background_accepted_observation_uuids": sorted(background_ids),
        "distinct_person_suppressed_ids": sorted(distinct_suppressed_ids),
        "boundary_and_outside_observation_ledger": boundary_ledger,
        "outside_roi_false_observation_scoring_performed": False,
    }


def _sum_metric(source_rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(int(row["metrics"][key]) for row in source_rows)


def aggregate_source_results(source_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate source evaluations with pooled and equal-source reporting."""

    person_rows = [person for source in source_rows for person in source["person_rows"]]
    denominator = len(person_rows)
    accepted = sum(row["accepted_any"] for row in person_rows)
    exact = sum(row["accepted_exactly_one"] for row in person_rows)
    routed_ids = sum(row["routed_to_dense_review"] for row in person_rows)
    covered = sum(row["accepted_plus_dense_covered"] for row in person_rows)
    pre_supported = sum(row["preconsolidation_independent_support"] for row in person_rows)
    suppressed = sum(row["distinct_person_suppressed"] for row in person_rows)
    duplicates = sum(int(row["duplicate_final_observation_count"]) for row in person_rows)
    merged = _sum_metric(source_rows, "merged_as_clean_observation_count")
    background = _sum_metric(source_rows, "background_accepted_observation_count")
    accepted_observations = _sum_metric(source_rows, "accepted_observation_count_inside_labelled_roi")
    candidate_count = _sum_metric(source_rows, "preconsolidation_candidate_count_inside_labelled_roi")
    dense_observations = _sum_metric(source_rows, "dense_review_observation_count_inside_labelled_roi")
    source_rates = [row["metrics"]["accepted_independent_supply"]["rate"] for row in source_rows if row["person_rows"]]
    ious = [
        float(assignment["geometry"]["visible_box_iou"])
        for source in source_rows
        for assignment in source["accepted_assignments"]
    ]
    bottoms = [
        float(assignment["geometry"]["bottom_centre_displacement_visible_heights"])
        for source in source_rows
        for assignment in source["accepted_assignments"]
    ]
    pooled = {
        "source_group_count": len(source_rows),
        "canonical_gold_person_count": denominator,
        "exactly_one_accepted_observation": exact_fraction(exact, denominator),
        "accepted_independent_supply": exact_fraction(accepted, denominator),
        "no_accepted_observation": exact_fraction(denominator - accepted, denominator),
        "routed_to_dense_review": exact_fraction(routed_ids, denominator),
        "accepted_plus_dense_routed_coverage": exact_fraction(covered, denominator),
        "duplicate_final_observation_count": duplicates,
        "duplicate_final_observation_rate": exact_fraction(duplicates, denominator),
        "merged_as_clean_observation_count": merged,
        "merged_as_clean_observation_rate": exact_fraction(merged, max(1, accepted_observations)),
        "preconsolidation_independent_support_count": pre_supported,
        "distinct_person_suppression_count": suppressed,
        "distinct_person_suppression_rate": exact_fraction(suppressed, pre_supported),
        "background_accepted_observation_count": background,
        "accepted_observation_count_inside_labelled_roi": accepted_observations,
        "dense_review_observation_count_inside_labelled_roi": dense_observations,
        "observation_count_error": accepted_observations - denominator,
        "candidate_reduction_ratio": round(
            1 - (accepted_observations + dense_observations) / max(1, candidate_count), 8
        ),
        "median_visible_box_iou": round(median(ious), 8) if ious else None,
        "median_normalized_bottom_centre_displacement": round(median(bottoms), 8) if bottoms else None,
        "equal_source_group_accepted_supply_rate": round(sum(source_rates) / len(source_rates), 8)
        if source_rates
        else None,
        "population_confidence_claimed": False,
    }

    def mean_rate(metric_name: str) -> float | None:
        values = [
            float(row["metrics"][metric_name]["rate"])
            for row in source_rows
            if row["metrics"][metric_name]["rate"] is not None
        ]
        return round(sum(values) / len(values), 8) if values else None

    def mean_count(metric_name: str) -> float:
        return round(sum(float(row["metrics"][metric_name]) for row in source_rows) / max(1, len(source_rows)), 8)

    source_ious = [
        float(row["metrics"]["median_visible_box_iou"])
        for row in source_rows
        if row["metrics"]["median_visible_box_iou"] is not None
    ]
    source_bottoms = [
        float(row["metrics"]["median_normalized_bottom_centre_displacement"])
        for row in source_rows
        if row["metrics"]["median_normalized_bottom_centre_displacement"] is not None
    ]
    equal_source = {
        "source_group_count": len(source_rows),
        "exactly_one_accepted_observation_rate": mean_rate("exactly_one_accepted_observation"),
        "accepted_independent_supply_rate": mean_rate("accepted_independent_supply"),
        "no_accepted_observation_rate": mean_rate("no_accepted_observation"),
        "routed_to_dense_review_rate": mean_rate("routed_to_dense_review"),
        "accepted_plus_dense_routed_coverage_rate": mean_rate("accepted_plus_dense_routed_coverage"),
        "duplicate_final_observation_rate": mean_rate("duplicate_final_observation_rate"),
        "duplicate_final_observation_mean_count": mean_count("duplicate_final_observation_count"),
        "merged_as_clean_observation_rate": mean_rate("merged_as_clean_observation_rate"),
        "merged_as_clean_observation_mean_count": mean_count("merged_as_clean_observation_count"),
        "distinct_person_suppression_rate": mean_rate("distinct_person_suppression_rate"),
        "distinct_person_suppression_mean_count": mean_count("distinct_person_suppression_count"),
        "background_accepted_observation_mean_count": mean_count("background_accepted_observation_count"),
        "observation_count_error_mean": mean_count("observation_count_error"),
        "absolute_observation_count_error_mean": mean_count("absolute_observation_count_error"),
        "candidate_reduction_ratio_mean": mean_count("candidate_reduction_ratio"),
        "median_of_source_median_visible_box_iou": round(median(source_ious), 8) if source_ious else None,
        "median_of_source_median_normalized_bottom_centre_displacement": (
            round(median(source_bottoms), 8) if source_bottoms else None
        ),
    }
    return {
        **pooled,
        "primary_aggregation": "EQUAL_SOURCE_GROUP",
        "primary_equal_source_group_metrics": equal_source,
        "pooled_person_results": pooled,
    }


def aggregate_person_subset(person_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    denominator = len(person_rows)
    return {
        "canonical_gold_person_count": denominator,
        "accepted_independent_supply": exact_fraction(sum(row["accepted_any"] for row in person_rows), denominator),
        "exactly_one_accepted_observation": exact_fraction(
            sum(row["accepted_exactly_one"] for row in person_rows), denominator
        ),
        "routed_to_dense_review": exact_fraction(
            sum(row["routed_to_dense_review"] for row in person_rows), denominator
        ),
        "accepted_plus_dense_routed_coverage": exact_fraction(
            sum(row["accepted_plus_dense_covered"] for row in person_rows), denominator
        ),
        "distinct_person_suppression": exact_fraction(
            sum(row["distinct_person_suppressed"] for row in person_rows),
            sum(row["preconsolidation_independent_support"] for row in person_rows),
        ),
    }


def screening_checks(
    aggregate: Mapping[str, Any],
    *,
    baseline_background_count: int,
    cpu_p95_milliseconds: float,
    deterministic: bool,
    provenance_exact: bool,
) -> dict[str, Any]:
    """Apply the immutable development screening criteria."""

    checks = {
        "merged_as_clean_zero": aggregate["merged_as_clean_observation_count"] == 0,
        "distinct_person_suppression_at_most_one_percent": (
            aggregate["distinct_person_suppression_rate"]["rate"] is not None
            and aggregate["distinct_person_suppression_rate"]["rate"] <= 0.01
        ),
        "duplicate_final_observation_rate_at_most_one_percent": (
            aggregate["duplicate_final_observation_rate"]["rate"] is not None
            and aggregate["duplicate_final_observation_rate"]["rate"] <= 0.01
        ),
        "accepted_independent_supply_at_least_240": aggregate["accepted_independent_supply"]["numerator"] >= 240,
        "accepted_plus_dense_coverage_at_least_288": aggregate["accepted_plus_dense_routed_coverage"]["numerator"]
        >= 288,
        "background_not_above_existing_baseline": aggregate["background_accepted_observation_count"]
        <= baseline_background_count,
        "cpu_p95_at_most_30_milliseconds": cpu_p95_milliseconds <= 30.0,
        "deterministic_repeatability": deterministic,
        "exact_provenance": provenance_exact,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "screening_only_not_final_acceptance": True,
        "hard_gate_pass_claimed": False,
    }
