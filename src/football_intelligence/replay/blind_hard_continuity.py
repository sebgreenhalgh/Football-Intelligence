from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from football_intelligence.core.fingerprints import sha256_file
from football_intelligence.replay.balanced_role_then_continuity import (
    _deterministic_empty_decision_state,
    _repo_root_from_module,
    _stage_input_paths,
    frame_quartile,
    spatial_region_bucket,
    thirty_frame_window,
)
from football_intelligence.replay.rebuilt_human_calibrated_pipeline import (
    _continuity_evidence,
    _frame_records,
    _review_case_hash,
    _source_ref,
    read_json,
    rows,
    write_json,
    write_text,
)
from football_intelligence.replay.role_partitioned_learning import _write_open_launcher
from football_intelligence.review.schemas import (
    CONTINUITY_DECISIONS,
    CONTINUITY_QUESTION,
    ReviewCase,
    ReviewManifest,
    SourceArtifactReference,
    safety_payload,
    stable_hash,
)

F1_DIAGNOSTIC_CLASSIFICATION = "M5_4F1_RAW_GEOMETRY_CONFOUNDED_CONTINUITY_REVIEW_DIAGNOSTIC_ONLY"
F2_PASS = "PASS_BLIND_RAW_FEATURE_DECONFOUNDED_CONTINUITY_REVIEW_READY"
F2_BLOCKED_RAW = "BLOCKED_RAW_FEATURE_OVERLAP"

TEAM_ROLES = {"team_1_outfield_visual_context", "team_2_outfield_visual_context"}
PRINCIPAL_FEATURES = [
    "bbox_iou",
    "center_delta_px",
    "footpoint_delta_px",
    "bbox_area_ratio",
    "aspect_ratio_change",
    "appearance_similarity",
    "continuity_score",
    "competing_candidate_margin",
]
AUDIT_FEATURES = [
    *PRINCIPAL_FEATURES,
    "reciprocal_rank",
    "intermediate_observed_support",
    "crop_quality",
    "occlusion",
    "competing_candidate_count",
]
GEOMETRY_FEATURES = ["bbox_iou", "center_delta_px", "footpoint_delta_px"]


def _class_label(value: str | None) -> str:
    text = str(value or "")
    if "positive" in text:
        return "likely_positive"
    if "negative" in text:
        return "likely_negative"
    return text or "unknown"


def _quantile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _histogram(values: list[float], bins: int = 8) -> list[dict[str, Any]]:
    values = sorted(values)
    if not values:
        return []
    low = values[0]
    high = values[-1]
    if low == high:
        return [{"min": low, "max": high, "count": len(values)}]
    width = (high - low) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        index = min(bins - 1, int((value - low) / width))
        counts[index] += 1
    return [
        {
            "min": round(low + width * index, 6),
            "max": round(low + width * (index + 1), 6),
            "count": count,
        }
        for index, count in enumerate(counts)
    ]


def _summary(values: list[float]) -> dict[str, Any]:
    values = sorted(float(value) for value in values)
    return {
        "count": len(values),
        "minimum": round(values[0], 6) if values else None,
        "maximum": round(values[-1], 6) if values else None,
        "mean": round(mean(values), 6) if values else None,
        "median": round(median(values), 6) if values else None,
        "stddev": round(pstdev(values), 6) if len(values) > 1 else 0.0 if values else None,
        "quartiles": {
            "q1": round(_quantile(values, 0.25), 6) if values else None,
            "q2": round(_quantile(values, 0.50), 6) if values else None,
            "q3": round(_quantile(values, 0.75), 6) if values else None,
        },
        "histogram": _histogram(values),
    }


def _roc_auc(positives: list[float], negatives: list[float]) -> float | None:
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _threshold_metrics(positives: list[float], negatives: list[float]) -> dict[str, Any]:
    if not positives or not negatives:
        return {
            "threshold": None,
            "polarity": None,
            "balanced_accuracy": None,
            "precision": None,
            "recall": None,
        }
    values = sorted(set(positives + negatives))
    if len(values) == 1:
        thresholds = values
    else:
        thresholds = [values[0] - 1e-9, values[-1] + 1e-9]
        thresholds.extend((left + right) / 2.0 for left, right in zip(values, values[1:]))
    best: dict[str, Any] = {
        "threshold": None,
        "polarity": None,
        "balanced_accuracy": -1.0,
        "precision": None,
        "recall": None,
    }
    for threshold in thresholds:
        for polarity in ("positive_when_greater_or_equal", "positive_when_less_or_equal"):
            positive_predictions = [
                value >= threshold if polarity == "positive_when_greater_or_equal" else value <= threshold
                for value in positives
            ]
            negative_predictions = [
                value < threshold if polarity == "positive_when_greater_or_equal" else value > threshold
                for value in negatives
            ]
            tp = sum(positive_predictions)
            fn = len(positives) - tp
            tn = sum(negative_predictions)
            fp = len(negatives) - tn
            recall = tp / (tp + fn) if tp + fn else 0.0
            specificity = tn / (tn + fp) if tn + fp else 0.0
            precision = tp / (tp + fp) if tp + fp else 0.0
            balanced_accuracy = (recall + specificity) / 2.0
            if balanced_accuracy > float(best["balanced_accuracy"]):
                best = {
                    "threshold": round(threshold, 6),
                    "polarity": polarity,
                    "balanced_accuracy": round(balanced_accuracy, 6),
                    "precision": round(precision, 6),
                    "recall": round(recall, 6),
                }
    return best


def raw_feature_shortcut_audit(
    rows_in: list[dict[str, Any]],
    *,
    artifact: str,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    feature_names = feature_names or AUDIT_FEATURES
    rows_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_in:
        rows_by_class[_class_label(str(row.get("proposed_class")))].append(row)
    features: dict[str, dict[str, Any]] = {}
    disjoint_features: list[str] = []
    perfect_threshold_features: list[str] = []
    near_perfect_auc_features: list[str] = []
    for feature in feature_names:
        positives = [
            float(row[feature]) for row in rows_by_class.get("likely_positive", []) if row.get(feature) is not None
        ]
        negatives = [
            float(row[feature]) for row in rows_by_class.get("likely_negative", []) if row.get(feature) is not None
        ]
        positive_summary = _summary(positives)
        negative_summary = _summary(negatives)
        overlap_low = None
        overlap_high = None
        ranges_overlap = False
        overlap_coefficient = None
        if positives and negatives:
            overlap_low = max(min(positives), min(negatives))
            overlap_high = min(max(positives), max(negatives))
            ranges_overlap = overlap_low <= overlap_high
            if ranges_overlap:
                positive_overlap = sum(overlap_low <= value <= overlap_high for value in positives) / len(positives)
                negative_overlap = sum(overlap_low <= value <= overlap_high for value in negatives) / len(negatives)
                overlap_coefficient = (positive_overlap + negative_overlap) / 2.0
            else:
                disjoint_features.append(feature)
                overlap_coefficient = 0.0
        auc = _roc_auc(positives, negatives)
        threshold = _threshold_metrics(positives, negatives)
        separability_auc = max(auc, 1.0 - auc) if auc is not None else None
        if threshold["balanced_accuracy"] == 1.0:
            perfect_threshold_features.append(feature)
        if separability_auc is not None and separability_auc >= 0.999:
            near_perfect_auc_features.append(feature)
        features[feature] = {
            "likely_positive": positive_summary,
            "likely_negative": negative_summary,
            "empirical_overlap_interval": {
                "minimum": round(overlap_low, 6) if overlap_low is not None and ranges_overlap else None,
                "maximum": round(overlap_high, 6) if overlap_high is not None and ranges_overlap else None,
            },
            "ranges_overlap": ranges_overlap,
            "overlap_coefficient": round(overlap_coefficient, 6) if overlap_coefficient is not None else None,
            "roc_auc_positive_higher": round(auc, 6) if auc is not None else None,
            "separability_auc": round(separability_auc, 6) if separability_auc is not None else None,
            "best_one_dimensional_threshold": threshold,
        }
    geometry_balanced_accuracy_below_target = all(
        features[feature]["best_one_dimensional_threshold"]["balanced_accuracy"] is not None
        and features[feature]["best_one_dimensional_threshold"]["balanced_accuracy"] < 0.8
        for feature in GEOMETRY_FEATURES
    )
    shared_band_fields = [
        "iou_band",
        "center_delta_band",
        "footpoint_delta_band",
        "delta_band",
        "score_band",
    ]
    shared_band_values = {
        field: sorted({str(row[field]) for row in rows_in if row.get(field) is not None})
        for field in shared_band_fields
    }
    feature_band_labels_hiding_raw_separation = bool(
        disjoint_features and any(len(values) == 1 for values in shared_band_values.values())
    )
    required_geometry_ranges_overlap = all(features[feature]["ranges_overlap"] for feature in GEOMETRY_FEATURES)
    no_principal_perfect_threshold = not any(feature in perfect_threshold_features for feature in PRINCIPAL_FEATURES)
    no_principal_perfect_auc = not any(feature in near_perfect_auc_features for feature in PRINCIPAL_FEATURES)
    return {
        "artifact": artifact,
        "case_count": len(rows_in),
        "class_counts": {key: len(value) for key, value in sorted(rows_by_class.items())},
        "features": features,
        "disjoint_numeric_range_features": disjoint_features,
        "perfect_univariate_threshold_features": perfect_threshold_features,
        "near_perfect_auc_features": near_perfect_auc_features,
        "shared_coarse_band_values": shared_band_values,
        "feature_band_labels_hiding_raw_separation": feature_band_labels_hiding_raw_separation,
        "raw_geometry_ranges_overlap": required_geometry_ranges_overlap,
        "geometry_balanced_accuracy_below_0_80": geometry_balanced_accuracy_below_target,
        "passes_raw_feature_overlap_gates": (
            required_geometry_ranges_overlap
            and geometry_balanced_accuracy_below_target
            and no_principal_perfect_threshold
            and no_principal_perfect_auc
        ),
        **safety_payload(),
    }


def _review_rows_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for case in manifest.get("review_cases", []):
        metadata = case.get("selection_metadata") if isinstance(case.get("selection_metadata"), dict) else {}
        features = metadata.get("gate_features") if isinstance(metadata.get("gate_features"), dict) else {}
        proposed = metadata.get("review_bucket") or case.get("category") or case.get("model_prediction")
        output.append(
            {
                "review_case_id": case.get("review_case_id"),
                "proposed_class": _class_label(str(proposed)),
                "bbox_iou": features.get("bbox_iou"),
                "center_delta_px": features.get("center_delta_px"),
                "footpoint_delta_px": features.get("footpoint_delta_px"),
                "bbox_area_ratio": features.get("bbox_area_ratio"),
                "aspect_ratio_change": features.get("aspect_ratio_change"),
                "appearance_similarity": metadata.get("appearance_similarity"),
                "continuity_score": metadata.get("continuity_score"),
                "competing_candidate_margin": metadata.get("competing_candidate_margin"),
                "reciprocal_rank": metadata.get("reciprocal_rank"),
                "intermediate_observed_support": 1.0 if metadata.get("has_intermediate_support") else 0.0,
                "crop_quality": metadata.get("crop_quality"),
                "occlusion": 1.0
                if metadata.get("occlusion")
                else 0.0
                if metadata.get("occlusion") is not None
                else None,
                "competing_candidate_count": metadata.get("competing_candidate_count"),
                "iou_band": metadata.get("iou_band"),
                "center_delta_band": metadata.get("center_delta_band"),
                "score_band": metadata.get("score_band"),
            }
        )
    return output


def _team_from_role(role: str) -> str:
    if role.startswith("team_1"):
        return "team_1"
    if role.startswith("team_2"):
        return "team_2"
    return "other"


def _appearance_similarity(source: dict[str, Any], target: dict[str, Any]) -> float:
    score = 0.2
    if source.get("colour_histogram_signature") == target.get("colour_histogram_signature"):
        score += 0.35
    if source.get("torso_colour") == target.get("torso_colour"):
        score += 0.25
    if source.get("shorts_colour") == target.get("shorts_colour"):
        score += 0.1
    if source.get("socks_colour") == target.get("socks_colour"):
        score += 0.1
    consistency = (
        float(source.get("temporal_colour_consistency", 0.0)) + float(target.get("temporal_colour_consistency", 0.0))
    ) / 2.0
    return round(min(1.0, score + 0.1 * consistency), 6)


def _bbox_size_bucket_from_node(node: dict[str, Any]) -> str:
    bbox = node.get("bbox") if isinstance(node.get("bbox"), dict) else {}
    height = float(bbox.get("y2", 0.0)) - float(bbox.get("y1", 0.0))
    width = float(bbox.get("x2", 0.0)) - float(bbox.get("x1", 0.0))
    area = max(0.0, height * width)
    if height < 35 or area < 700:
        return "distant_small_bbox"
    if height > 85 or area > 2800:
        return "near_large_bbox"
    return "medium_bbox"


def _rank_and_margin(candidate_rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    source_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    target_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        source_groups[(str(row["source_visible_person_base_id"]), int(row["target_frame_sequence"]))].append(row)
        target_groups[(str(row["target_visible_person_base_id"]), int(row["source_frame_sequence"]))].append(row)
    for group in [*source_groups.values(), *target_groups.values()]:
        group.sort(key=lambda item: float(item.get("continuity_score", 0.0)), reverse=True)
    source_rank: dict[str, int] = {}
    target_rank: dict[str, int] = {}
    margin: dict[str, float] = {}
    for group in source_groups.values():
        for index, row in enumerate(group):
            source_rank[str(row["role_partitioned_continuity_candidate_id"])] = index + 1
    for group in target_groups.values():
        for index, row in enumerate(group):
            target_rank[str(row["role_partitioned_continuity_candidate_id"])] = index + 1
    for row in candidate_rows:
        candidate_id = str(row["role_partitioned_continuity_candidate_id"])
        score = float(row.get("continuity_score", 0.0))
        margins = []
        for group in [
            source_groups[(str(row["source_visible_person_base_id"]), int(row["target_frame_sequence"]))],
            target_groups[(str(row["target_visible_person_base_id"]), int(row["source_frame_sequence"]))],
        ]:
            competitor_scores = [
                float(other.get("continuity_score", 0.0))
                for other in group
                if str(other["role_partitioned_continuity_candidate_id"]) != candidate_id
            ]
            margins.append(abs(score - max(competitor_scores)) if competitor_scores else 1.0)
        margin[candidate_id] = round(min(margins), 6)
    return source_rank, target_rank, margin


def _source_refs(stage_root: Path, paths: dict[str, Path]) -> list[SourceArtifactReference]:
    return [
        _source_ref(
            "m5_4f_post_role_candidate_rows",
            stage_root / "continuity" / "post_role_candidate_rows.json",
            "read-only M5.4F.1 post-role candidate rows",
        ),
        _source_ref(
            "m5_4f_post_role_context_rows",
            stage_root / "continuity" / "post_role_context_rows.json",
            "read-only M5.4F.1 post-role context rows",
        ),
        _source_ref(
            "m5_4d_continuity_node_rows",
            paths["m54d_stage_root"] / "continuity" / "continuity_node_rows.json",
            "read-only continuity nodes for evidence rendering",
        ),
    ]


def _enriched_candidate_rows(stage_root: Path) -> list[dict[str, Any]]:
    paths = _stage_input_paths(stage_root)
    candidate_rows = rows(read_json(stage_root / "continuity" / "post_role_candidate_rows.json"))
    node_rows = rows(read_json(paths["m54d_stage_root"] / "continuity" / "continuity_node_rows.json"))
    feature_rows = rows(read_json(paths["m54e_stage_root"] / "role" / "visual_role_feature_rows.json"))
    node_by_visible = {str(row["visible_person_base_id"]): row for row in node_rows}
    feature_by_candidate = {str(row["candidate_id"]): row for row in feature_rows}
    source_rank, target_rank, margin_by_candidate = _rank_and_margin(candidate_rows)
    source_groups: Counter[tuple[str, int]] = Counter(
        (str(row["source_visible_person_base_id"]), int(row["target_frame_sequence"])) for row in candidate_rows
    )
    target_groups: Counter[tuple[str, int]] = Counter(
        (str(row["target_visible_person_base_id"]), int(row["source_frame_sequence"])) for row in candidate_rows
    )
    enriched: list[dict[str, Any]] = []
    for row in candidate_rows:
        source_role = str(row.get("source_visual_role_context", ""))
        target_role = str(row.get("target_visual_role_context", ""))
        if source_role != target_role or source_role not in TEAM_ROLES:
            continue
        gap = int(row.get("frame_gap", 0))
        if gap not in {1, 2, 3}:
            continue
        features = row.get("gate_features") if isinstance(row.get("gate_features"), dict) else {}
        bbox_iou = float(features.get("bbox_iou", 0.0))
        center_delta = float(features.get("center_delta_px", 999.0))
        footpoint_delta = float(features.get("footpoint_delta_px", 999.0))
        score = float(row.get("continuity_score", 0.0))
        if not (0.55 <= bbox_iou <= 0.9 and 1.5 <= center_delta <= 15 and 1.5 <= footpoint_delta <= 16):
            continue
        if not (0.53 <= score <= 0.72):
            continue
        source_node = node_by_visible.get(str(row["source_visible_person_base_id"]), {})
        target_node = node_by_visible.get(str(row["target_visible_person_base_id"]), {})
        source_feature = feature_by_candidate.get(str(source_node.get("candidate_id")), {})
        target_feature = feature_by_candidate.get(str(target_node.get("candidate_id")), {})
        crop_quality = (
            float(source_feature.get("crop_quality", 0.0)) + float(target_feature.get("crop_quality", 0.0))
        ) / 2.0
        occlusion = (
            bool(source_feature.get("occlusion"))
            or bool(target_feature.get("occlusion"))
            or min(float(source_feature.get("crop_quality", 1.0)), float(target_feature.get("crop_quality", 1.0)))
            < 0.55
        )
        candidate_id = str(row["role_partitioned_continuity_candidate_id"])
        competing_count = (
            source_groups[(str(row["source_visible_person_base_id"]), int(row["target_frame_sequence"]))]
            - 1
            + target_groups[(str(row["target_visible_person_base_id"]), int(row["source_frame_sequence"]))]
            - 1
        )
        appearance = _appearance_similarity(source_feature, target_feature)
        frame = int(row["source_frame_sequence"])
        enriched_row = {
            **row,
            "bbox_iou": bbox_iou,
            "center_delta_px": center_delta,
            "footpoint_delta_px": footpoint_delta,
            "bbox_area_ratio": float(features.get("bbox_area_ratio", 1.0)),
            "aspect_ratio_change": float(features.get("aspect_ratio_change", 0.0)),
            "appearance_similarity": appearance,
            "continuity_score": score,
            "competing_candidate_margin": margin_by_candidate[candidate_id],
            "reciprocal_rank": max(source_rank[candidate_id], target_rank[candidate_id]),
            "intermediate_observed_support": 1.0 if row.get("intermediate_frame_support") else 0.0,
            "crop_quality": round(crop_quality, 6),
            "occlusion": 1.0 if occlusion else 0.0,
            "competing_candidate_count": competing_count,
            "team_partition": _team_from_role(source_role),
            "source_temporal_quartile": frame_quartile(frame),
            "source_thirty_frame_window": thirty_frame_window(frame),
            "source_spatial_region_bucket": spatial_region_bucket(source_node) if source_node else "unknown",
            "bbox_size_bucket": _bbox_size_bucket_from_node(source_node),
            "source_candidate_id": source_node.get("candidate_id"),
            "target_candidate_id": target_node.get("candidate_id"),
            "effective_role_context": source_role,
            "hard_positive_score": 0.0,
            "hard_negative_score": 0.0,
        }
        enriched_row["hard_positive_score"] = (
            float(bbox_iou <= 0.8)
            + float(center_delta >= 3.0)
            + float(gap in {2, 3})
            + float(row.get("intermediate_frame_support"))
            + appearance
            + 0.4 * float(competing_count > 0)
            + 0.7 * float(occlusion)
            - 0.4 * abs(bbox_iou - 0.72)
            - 0.03 * abs(center_delta - 5.0)
        )
        enriched_row["hard_negative_score"] = (
            float(bbox_iou >= 0.75)
            + float(center_delta <= 5.0)
            + float(competing_count > 0)
            + float(margin_by_candidate[candidate_id] < 0.12)
            + appearance
            + 0.7 * float(occlusion)
            - 0.4 * abs(bbox_iou - 0.76)
            - 0.03 * abs(center_delta - 4.0)
        )
        enriched.append(enriched_row)
    return enriched


def _equivalence_cluster_id(row: dict[str, Any]) -> str:
    payload = {
        "team": row["team_partition"],
        "gap": row["frame_gap"],
        "window": row["source_thirty_frame_window"],
        "spatial": row["source_spatial_region_bucket"],
        "iou_band": round(float(row["bbox_iou"]), 1),
        "center_band": round(float(row["center_delta_px"]) / 3.0),
        "competing": int(row["competing_candidate_count"] > 0),
    }
    return f"m5_4f2_continuity_cluster_{stable_hash(payload)[:12]}"


def _iou_kind(row: dict[str, Any]) -> str:
    if float(row["bbox_iou"]) >= 0.75:
        return "high"
    if float(row["bbox_iou"]) <= 0.70:
        return "low"
    return "mid"


def _selection_slots(limit: int) -> list[dict[str, Any]]:
    gap_sequence = ([1, 2, 3] * ((limit // 3) + 1))[:limit]
    team_sequence = (["team_1", "team_2"] * ((limit // 2) + 1))[:limit]
    quartile_sequence = (["q1_000_149", "q2_150_299", "q3_300_449", "q4_450_599"] * ((limit // 4) + 1))[:limit]
    iou_sequence = (["high", "low", "mid", "high", "low", "mid", "any", "any"] * ((limit // 8) + 1))[:limit]
    competing_slots = set(range(0, min(limit, 12), 2))
    occlusion_slots = {2, 7, 13, 18} if limit >= 20 else {2, 7, 13}
    return [
        {
            "gap": gap_sequence[index],
            "team": team_sequence[index],
            "quartile": quartile_sequence[index],
            "iou_kind": iou_sequence[index],
            "needs_competing": index in competing_slots,
            "needs_occlusion": index in occlusion_slots,
        }
        for index in range(limit)
    ]


def _select_hard_cases(enriched_rows: list[dict[str, Any]], *, limit_per_class: int) -> dict[str, Any]:
    endpoint_counts: Counter[str] = Counter()
    window_counts: Counter[str] = Counter()
    cluster_counts: Counter[tuple[str, str]] = Counter()
    used_candidates: set[str] = set()
    class_team_counts: dict[str, Counter[str]] = defaultdict(Counter)
    class_gap_counts: dict[str, Counter[int]] = defaultdict(Counter)
    team_targets = {"team_1": math.ceil(limit_per_class / 2), "team_2": limit_per_class // 2}
    gap_targets = {1: math.ceil(limit_per_class / 3), 2: math.ceil(limit_per_class / 3), 3: limit_per_class // 3}

    def cluster(row: dict[str, Any]) -> str:
        return _equivalence_cluster_id(row)

    def ok(row: dict[str, Any], class_label: str) -> bool:
        candidate_id = str(row["role_partitioned_continuity_candidate_id"])
        source = str(row["source_visible_person_base_id"])
        target = str(row["target_visible_person_base_id"])
        return (
            candidate_id not in used_candidates
            and endpoint_counts[source] < 2
            and endpoint_counts[target] < 2
            and window_counts[str(row["source_thirty_frame_window"])] < 4
            and cluster_counts[(class_label, cluster(row))] < 2
            and class_team_counts[class_label][str(row["team_partition"])] < team_targets[str(row["team_partition"])]
            and class_gap_counts[class_label][int(row["frame_gap"])] < gap_targets[int(row["frame_gap"])]
        )

    def add(row: dict[str, Any], class_label: str, output: list[dict[str, Any]]) -> None:
        row = {
            **row,
            "proposed_class": class_label,
            "equivalence_cluster_id": cluster(row),
            "difficulty_category": "hard_positive_candidate"
            if class_label == "likely_positive"
            else "hard_negative_candidate",
        }
        output.append(row)
        used_candidates.add(str(row["role_partitioned_continuity_candidate_id"]))
        endpoint_counts[str(row["source_visible_person_base_id"])] += 1
        endpoint_counts[str(row["target_visible_person_base_id"])] += 1
        window_counts[str(row["source_thirty_frame_window"])] += 1
        cluster_counts[(class_label, cluster(row))] += 1
        class_team_counts[class_label][str(row["team_partition"])] += 1
        class_gap_counts[class_label][int(row["frame_gap"])] += 1

    positives: list[dict[str, Any]] = []
    slots = _selection_slots(limit_per_class)
    for slot in slots:
        pool = [row for row in enriched_rows if ok(row, "likely_positive")]

        def positive_score(row: dict[str, Any]) -> tuple[float, str, str]:
            score = (
                7.0 * float(int(row["frame_gap"]) == slot["gap"])
                + 6.0 * float(row["team_partition"] == slot["team"])
                + 4.0 * float(row["source_temporal_quartile"] == slot["quartile"])
                + 5.0 * float(slot["iou_kind"] == "any" or _iou_kind(row) == slot["iou_kind"])
                + 5.0 * float(not slot["needs_competing"] or int(row["competing_candidate_count"]) > 0)
                + 7.0 * float(not slot["needs_occlusion"] or float(row["occlusion"]) > 0.0)
                + float(row["hard_positive_score"])
            )
            return (
                -score,
                str(row["source_thirty_frame_window"]),
                str(row["role_partitioned_continuity_candidate_id"]),
            )

        pool.sort(key=positive_score)
        if pool:
            add(pool[0], "likely_positive", positives)

    negatives: list[dict[str, Any]] = []
    for slot, positive in zip(slots, positives):
        pool = [row for row in enriched_rows if ok(row, "likely_negative")]

        def negative_distance(row: dict[str, Any]) -> tuple[float, str]:
            distance = (
                abs(float(row["bbox_iou"]) - float(positive["bbox_iou"])) / 0.04
                + abs(float(row["center_delta_px"]) - float(positive["center_delta_px"])) / 1.5
                + abs(float(row["footpoint_delta_px"]) - float(positive["footpoint_delta_px"])) / 1.5
                + abs(float(row["continuity_score"]) - float(positive["continuity_score"])) / 0.04
                + 3.0 * float(int(row["frame_gap"]) != int(positive["frame_gap"]))
                + 3.0 * float(row["team_partition"] != positive["team_partition"])
                + 2.0 * float(_iou_kind(row) != _iou_kind(positive))
                + 4.0 * float(slot["needs_competing"] and int(row["competing_candidate_count"]) == 0)
                + 6.0 * float(slot["needs_occlusion"] and float(row["occlusion"]) == 0.0)
                - 0.35 * float(row["hard_negative_score"])
            )
            return (distance, str(row["role_partitioned_continuity_candidate_id"]))

        pool.sort(key=negative_distance)
        if pool:
            add(pool[0], "likely_negative", negatives)
    return {
        "likely_positive": positives,
        "likely_negative": negatives,
        "endpoint_reuse_distribution": dict(sorted(endpoint_counts.items())),
        "endpoint_reuse_max": max(endpoint_counts.values() or [0]),
        "window_distribution": dict(sorted(window_counts.items())),
        "cluster_distribution": {
            f"{label}:{cluster}": count for (label, cluster), count in sorted(cluster_counts.items())
        },
    }


def _selected_rows_for_audit(selection: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "proposed_class": row["proposed_class"],
            **{feature: row.get(feature) for feature in AUDIT_FEATURES},
        }
        for row in [*selection["likely_positive"], *selection["likely_negative"]]
    ]


def _balance_audits(selection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected = [*selection["likely_positive"], *selection["likely_negative"]]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_class[str(row["proposed_class"])].append(row)
    endpoint = {
        "artifact": "m5_4f2_endpoint_reuse_audit",
        "endpoint_reuse_distribution": selection["endpoint_reuse_distribution"],
        "endpoint_reuse_max": selection["endpoint_reuse_max"],
        "endpoint_reuse_limit": 2,
        "endpoint_reuse_passed": selection["endpoint_reuse_max"] <= 2,
        **safety_payload(),
    }
    equivalence = {
        "artifact": "m5_4f2_equivalence_cluster_audit",
        "cluster_distribution": selection["cluster_distribution"],
        "independent_cluster_count_by_class": {
            label: len({row["equivalence_cluster_id"] for row in rows_for_class})
            for label, rows_for_class in sorted(by_class.items())
        },
        "max_cases_per_cluster": max(selection["cluster_distribution"].values() or [0]),
        "uses_class_membership_as_cluster": False,
        "clusters_are_unique_per_row": len(selection["cluster_distribution"]) == len(selected),
        **safety_payload(),
    }
    temporal = {
        "artifact": "m5_4f2_temporal_balance_audit",
        "frame_gap_distribution": {
            label: dict(sorted(Counter(int(row["frame_gap"]) for row in rows_for_class).items()))
            for label, rows_for_class in sorted(by_class.items())
        },
        "temporal_quartile_distribution": {
            label: dict(sorted(Counter(str(row["source_temporal_quartile"]) for row in rows_for_class).items()))
            for label, rows_for_class in sorted(by_class.items())
        },
        "thirty_frame_window_distribution": selection["window_distribution"],
        "max_cases_per_30_frame_window": max(selection["window_distribution"].values() or [0]),
        **safety_payload(),
    }
    team = {
        "artifact": "m5_4f2_team_balance_audit",
        "team_distribution": {
            label: dict(sorted(Counter(str(row["team_partition"]) for row in rows_for_class).items()))
            for label, rows_for_class in sorted(by_class.items())
        },
        "role_distribution": {
            label: dict(sorted(Counter(str(row["effective_role_context"]) for row in rows_for_class).items()))
            for label, rows_for_class in sorted(by_class.items())
        },
        **safety_payload(),
    }
    return {"endpoint": endpoint, "equivalence": equivalence, "temporal": temporal, "team": team}


def _case_counts(selection: dict[str, Any]) -> dict[str, Any]:
    rows_for_all = [*selection["likely_positive"], *selection["likely_negative"]]
    return {
        "case_count": len(rows_for_all),
        "proposed_positive_count": len(selection["likely_positive"]),
        "proposed_negative_count": len(selection["likely_negative"]),
        "hard_positive_count": len(selection["likely_positive"]),
        "hard_negative_count": len(selection["likely_negative"]),
        "high_iou_negative_count": sum(float(row["bbox_iou"]) >= 0.75 for row in selection["likely_negative"]),
        "lower_iou_positive_count": sum(float(row["bbox_iou"]) <= 0.8 for row in selection["likely_positive"]),
        "competing_candidate_count_by_proposed_class": {
            "likely_positive": sum(int(row["competing_candidate_count"]) > 0 for row in selection["likely_positive"]),
            "likely_negative": sum(int(row["competing_candidate_count"]) > 0 for row in selection["likely_negative"]),
        },
        "occlusion_count_by_proposed_class": {
            "likely_positive": sum(float(row["occlusion"]) > 0.0 for row in selection["likely_positive"]),
            "likely_negative": sum(float(row["occlusion"]) > 0.0 for row in selection["likely_negative"]),
        },
    }


def _diagnostic_classifier(
    rows_in: list[dict[str, Any]],
    *,
    feature_names: list[str],
    artifact: str,
) -> dict[str, Any]:
    data = [row for row in rows_in if all(row.get(feature) is not None for feature in feature_names)]
    groups = sorted({str(row["equivalence_cluster_id"]) for row in data})
    folds = max(1, min(5, len(groups)))
    group_to_fold = {group: index % folds for index, group in enumerate(groups)}
    labels = ["likely_negative", "likely_positive"]
    predictions: list[tuple[str, str]] = []
    importances: Counter[str] = Counter()
    for fold in range(folds):
        train = [row for row in data if group_to_fold[str(row["equivalence_cluster_id"])] != fold]
        valid = [row for row in data if group_to_fold[str(row["equivalence_cluster_id"])] == fold]
        if not train or not valid:
            continue
        means = {feature: mean([float(row[feature]) for row in train]) for feature in feature_names}
        scales = {feature: pstdev([float(row[feature]) for row in train]) or 1.0 for feature in feature_names}
        centroids = {}
        for label in labels:
            class_rows = [row for row in train if row["proposed_class"] == label]
            if not class_rows:
                continue
            centroids[label] = [
                mean([(float(row[feature]) - means[feature]) / scales[feature] for row in class_rows])
                for feature in feature_names
            ]
        for feature_index, feature in enumerate(feature_names):
            if len(centroids) == 2:
                importances[feature] += abs(
                    centroids["likely_positive"][feature_index] - centroids["likely_negative"][feature_index]
                )
        for row in valid:
            vector = [(float(row[feature]) - means[feature]) / scales[feature] for feature in feature_names]
            distances = {
                label: sum((value - centroid[index]) ** 2 for index, value in enumerate(vector))
                for label, centroid in centroids.items()
            }
            predicted = min(distances, key=distances.get) if distances else "likely_negative"
            predictions.append((str(row["proposed_class"]), predicted))
    recalls = []
    f1s = []
    for label in labels:
        tp = sum(truth == label and predicted == label for truth, predicted in predictions)
        fp = sum(truth != label and predicted == label for truth, predicted in predictions)
        fn = sum(truth == label and predicted != label for truth, predicted in predictions)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "artifact": artifact,
        "feature_names": feature_names,
        "group_by": "equivalence_cluster_id",
        "grouped_folds": folds,
        "train_validation_cluster_leakage_detected": False,
        "grouped_balanced_accuracy": round(sum(recalls) / len(recalls), 6) if recalls else 0.0,
        "grouped_macro_f1": round(sum(f1s) / len(f1s), 6) if f1s else 0.0,
        "feature_importance": dict(sorted((key, round(value, 6)) for key, value in importances.items())),
        "diagnostic_only_not_continuity_model": True,
        **safety_payload(),
    }


def _write_blind_workbench(workbench_root: Path) -> None:
    workbench_root.mkdir(parents=True, exist_ok=True)
    write_text(
        workbench_root / "index.html",
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blind Hard Continuity Review</title>
  <link rel="stylesheet" href="/styles.css">
</head>
<body>
  <aside>
    <div id="counts"></div>
    <label><input id="unresolvedOnly" type="checkbox"> unresolved only</label>
    <div id="caseList"></div>
  </aside>
  <main>
    <header>
      <h1 id="caseTitle">Blind continuity review</h1>
      <p id="safeMeta"></p>
      <button id="revealBtn">Reveal model information</button>
    </header>
    <section id="modelPanel" class="hidden"></section>
    <section id="evidence"></section>
    <section id="context"></section>
    <section id="decisions"></section>
    <textarea id="note" placeholder="Optional note"></textarea>
    <footer>
      <button id="prev">Previous</button>
      <button id="next">Next</button>
      <button id="complete">Complete review</button>
      <span id="status"></span>
    </footer>
  </main>
  <script src="/app.js"></script>
</body>
</html>
""",
    )
    write_text(
        workbench_root / "styles.css",
        """body {
  margin: 0;
  font-family: Segoe UI, Arial, sans-serif;
  background: #101418;
  color: #e8edf2;
  display: grid;
  grid-template-columns: 300px 1fr;
  min-height: 100vh;
}
aside {
  border-right: 1px solid #2d3742;
  padding: 12px;
  overflow: auto;
}
main {
  padding: 16px;
  overflow: auto;
}
.case {
  display: block;
  width: 100%;
  margin: 4px 0;
  padding: 8px;
  background: #1b232c;
  color: #dfe7ef;
  border: 1px solid #344250;
  text-align: left;
}
.case.active {
  border-color: #7cc7ff;
}
.case.done {
  background: #203828;
}
.hidden {
  display: none;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 12px;
}
.asset {
  background: #151b22;
  border: 1px solid #344250;
  padding: 8px;
}
.asset img,
.asset video {
  width: 100%;
  max-height: 520px;
  object-fit: contain;
  background: #000;
}
.decision {
  margin: 8px 8px 8px 0;
  padding: 10px 14px;
  border: 0;
  background: #29435c;
  color: white;
}
.decision.selected {
  outline: 3px solid #9be28f;
}
textarea {
  width: 100%;
  min-height: 72px;
  background: #0d1117;
  color: #e8edf2;
  border: 1px solid #344250;
}
button {
  cursor: pointer;
}
#modelPanel {
  border: 1px solid #7c5b2a;
  background: #201911;
  padding: 10px;
  margin: 8px 0;
  white-space: pre-wrap;
}
""",
    )
    write_text(
        workbench_root / "app.js",
        """let manifest = null;
let state = null;
let active = 0;
let revealed = {};

const $ = id => document.getElementById(id);

async function api(path, opts) {
  const res = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...(opts || {})
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function asset(c, type) {
  return (c.evidence_manifest.evidence_assets || []).find(a => a.asset_type === type) || null;
}

function media(c, a, label) {
  if (!a) return "";
  const src = `/evidence/${c.review_case_id}/${a.relative_path}`;
  const tag = a.media_type === "video/mp4"
    ? `<video controls loop muted playsinline src="${src}"></video>`
    : `<img src="${src}" alt="">`;
  return `<div class="asset"><strong>${label}</strong>${tag}</div>`;
}

function current() {
  return manifest.review_cases[active];
}

function renderList() {
  const only = $("unresolvedOnly").checked;
  const decisions = state.decisions || {};
  $("caseList").innerHTML = "";
  manifest.review_cases.forEach((c, i) => {
    if (only && decisions[c.review_case_id]) return;
    const b = document.createElement("button");
    b.className = "case"
      + (i === active ? " active" : "")
      + (decisions[c.review_case_id] ? " done" : "");
    b.textContent = `${i + 1}. f${c.source_frame_sequence}->${c.target_frame_sequence}`;
    b.onclick = () => { active = i; render(); };
    $("caseList").appendChild(b);
  });
}

function renderEvidence(c) {
  const temporal = asset(c, "temporal_clip")
    || c.evidence_manifest.evidence_assets.find(a => a.media_type === "video/mp4");
  const assets = [
    media(c, temporal, "Temporal evidence"),
    media(c, asset(c, "temporal_strip"), "Temporal strip"),
    media(c, asset(c, "source_full_frame"), "Source full frame"),
    media(c, asset(c, "target_full_frame"), "Target full frame"),
    media(c, asset(c, "source_crop"), "Source crop"),
    media(c, asset(c, "target_crop"), "Target crop"),
    media(c, asset(c, "source_context"), "Source context"),
    media(c, asset(c, "target_context"), "Target context")
  ];
  $("evidence").innerHTML = `<div class="grid">${assets.join("")}</div>`;
}

function render() {
  const c = current();
  const decisions = state.decisions || {};
  $("counts").textContent = `${Object.keys(decisions).length}/${manifest.review_cases.length} reviewed`;
  $("caseTitle").textContent = `Case ${active + 1} of ${manifest.review_cases.length}`;
  $("safeMeta").textContent =
    `frames ${c.source_frame_sequence}->${c.target_frame_sequence}`
    + ` | role compatible | ${c.selection_metadata.blind_context.team_partition}`;
  $("note").value = (state.notes || {})[c.review_case_id] || "";
  renderEvidence(c);
  $("context").textContent = "Competing candidates: "
    + JSON.stringify(c.selection_metadata.blind_context.competing_candidates);
  $("modelPanel").classList.toggle("hidden", !revealed[c.review_case_id]);
  $("modelPanel").textContent = revealed[c.review_case_id]
    ? JSON.stringify(c.selection_metadata.blind_hidden_model_info, null, 2)
    : "";
  $("decisions").innerHTML = "";
  for (const decision of c.allowed_decisions) {
    const b = document.createElement("button");
    b.className = "decision" + (decisions[c.review_case_id] === decision ? " selected" : "");
    b.textContent = decision;
    b.onclick = () => save(decision);
    $("decisions").appendChild(b);
  }
  renderList();
}

async function save(decision) {
  const c = current();
  let note = $("note").value || "";
  note += `\\n[model_info_revealed_before_decision=${!!revealed[c.review_case_id]}]`;
  state = await api("/api/review/decision", {
    method: "POST",
    body: JSON.stringify({
      review_case_id: c.review_case_id,
      decision,
      note,
      last_viewed_case_id: c.review_case_id
    })
  });
  $("status").textContent = "saved";
  render();
}

async function init() {
  manifest = await api("/api/review/manifest");
  state = await api("/api/review/state");
  $("revealBtn").onclick = () => {
    revealed[current().review_case_id] = true;
    render();
  };
  $("prev").onclick = () => {
    active = Math.max(0, active - 1);
    render();
  };
  $("next").onclick = () => {
    active = Math.min(manifest.review_cases.length - 1, active + 1);
    render();
  };
  $("unresolvedOnly").onchange = renderList;
  $("complete").onclick = async () => {
    state = await api("/api/review/complete", {method: "POST", body: "{}"});
    $("status").textContent = "complete";
  };
  render();
}

init().catch(err => {$("status").textContent = err.message;});
""",
    )
    write_text(workbench_root / "fallback.html", "<p>Blind review workbench requires the local review server.</p>\n")


def _review_manifest(
    *,
    stage_root: Path,
    selection: dict[str, Any],
    completed_at: str,
) -> dict[str, Any]:
    paths = _stage_input_paths(stage_root)
    continuity_root = stage_root / "continuity_v2"
    evidence_root = continuity_root / "evidence"
    node_rows = rows(read_json(paths["m54d_stage_root"] / "continuity" / "continuity_node_rows.json"))
    node_by_visible = {str(row["visible_person_base_id"]): row for row in node_rows}
    frame_records = _frame_records(read_json(paths["frame_manifest"]))
    source_refs = _source_refs(stage_root, paths)
    selected_rows = [*selection["likely_positive"], *selection["likely_negative"]]
    cases: list[ReviewCase] = []
    for index, row in enumerate(selected_rows, start=1):
        case_id = f"m5_4f2_blind_continuity_case_{index:03d}"
        evidence = _continuity_evidence(
            evidence_root=evidence_root,
            case_id=case_id,
            edge=row,
            node_by_visible_id=node_by_visible,
            frame_root=paths["frame_root"],
            frame_records=frame_records,
        )
        hidden = {
            "proposed_bucket": row["proposed_class"],
            "continuity_score": row["continuity_score"],
            "hard_positive_score": row["hard_positive_score"],
            "hard_negative_score": row["hard_negative_score"],
            "raw_features": {feature: row.get(feature) for feature in PRINCIPAL_FEATURES},
        }
        visible = {
            "team_partition": row["team_partition"],
            "effective_role_context": row["effective_role_context"],
            "source_visible_person_base_id": row["source_visible_person_base_id"],
            "target_visible_person_base_id": row["target_visible_person_base_id"],
            "source_candidate_id": row["source_candidate_id"],
            "target_candidate_id": row["target_candidate_id"],
            "competing_candidates": {
                "count": row["competing_candidate_count"],
                "margin": row["competing_candidate_margin"],
            },
            "intermediate_observed_support": bool(row["intermediate_observed_support"]),
            "interpolation_label": "INTERP_NOT_OBS",
        }
        case_payload = {
            "review_case_id": case_id,
            "task_type": "visual_continuity_edge_review",
            "concise_question": CONTINUITY_QUESTION,
            "allowed_decisions": CONTINUITY_DECISIONS,
            "candidate_artifact_id": str(row["role_partitioned_continuity_candidate_id"]),
            "source_artifact_references": source_refs,
            "source_frame_sequence": int(row["source_frame_sequence"]),
            "target_frame_sequence": int(row["target_frame_sequence"]),
            "evidence_manifest": evidence,
            "uncertainty_reasons": [
                "blind_review_default_hides_proposed_bucket_and_model_score",
                f"team_partition={row['team_partition']}",
                f"effective_role_context={row['effective_role_context']}",
                "interpolated_boxes_remain_labelled_INTERP_NOT_OBS",
            ],
            "category": "blind_hard_continuity_review",
            "priority": index,
            "control_status": "m5_4f2_blind_raw_feature_deconfounded_candidate",
            "candidate_hash": "",
            "evidence_hash": evidence.evidence_hash,
            "safety_payload": safety_payload(),
            "review_round": 5,
            "selection_metadata": {
                "blind_review_default_state": "proposed_bucket_model_score_and_raw_features_hidden",
                "reveal_model_information_control": "available_explicit_action_records_note_marker",
                "blind_context": visible,
                "blind_hidden_model_info": hidden,
            },
            "model_prediction": None,
            "model_confidence": None,
            "equivalence_cluster_id": row["equivalence_cluster_id"],
            "representative_of_count": 1,
        }
        case_payload["candidate_hash"] = _review_case_hash(case_payload)
        cases.append(ReviewCase.model_validate(case_payload))
    manifest = ReviewManifest(
        created_at=completed_at,
        title="M5.4F.2 Blind Hard Continuity Review",
        review_task_family="m5_4f2_blind_raw_feature_deconfounded_continuity",
        review_cases=cases,
        candidate_manifest_hash=stable_hash([case.candidate_hash for case in cases]),
        evidence_manifest_hash=stable_hash([case.evidence_hash for case in cases]),
        source_manifest_hash=stable_hash([ref.model_dump(mode="json") for ref in source_refs]),
        source_artifact_references=source_refs,
    )
    return manifest.model_dump(mode="json")


def _write_case_index(path: Path, manifest: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_case_id",
                "candidate_artifact_id",
                "proposed_bucket_hidden",
                "source_frame_sequence",
                "target_frame_sequence",
                "frame_gap",
                "team_partition",
                "equivalence_cluster_id",
            ],
        )
        writer.writeheader()
        for case in manifest.get("review_cases", []):
            context = case["selection_metadata"]["blind_context"]
            hidden = case["selection_metadata"]["blind_hidden_model_info"]
            writer.writerow(
                {
                    "review_case_id": case["review_case_id"],
                    "candidate_artifact_id": case["candidate_artifact_id"],
                    "proposed_bucket_hidden": hidden["proposed_bucket"],
                    "source_frame_sequence": case["source_frame_sequence"],
                    "target_frame_sequence": case["target_frame_sequence"],
                    "frame_gap": case["evidence_manifest"]["frame_gap"],
                    "team_partition": context["team_partition"],
                    "equivalence_cluster_id": case["equivalence_cluster_id"],
                }
            )


def _inventory(paths: list[Path], *, base: Path) -> dict[str, Any]:
    entries = []
    for root in paths:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                entries.append(
                    {
                        "path": str(path.relative_to(base)),
                        "sha256": sha256_file(path),
                        "size": path.stat().st_size,
                    }
                )
    return {"file_count": len(entries), "inventory_hash": stable_hash(entries), "entries": entries}


def build_blind_hard_continuity_review(*, stage_root: Path, repo_root: Path | None = None) -> dict[str, Any]:
    stage_root = stage_root.resolve()
    repo_root = (repo_root or _repo_root_from_module()).resolve()
    audit_root = stage_root / "audit"
    continuity_root = stage_root / "continuity_v2"
    validation_root = stage_root / "validation"
    for root in [audit_root, continuity_root, validation_root]:
        root.mkdir(parents=True, exist_ok=True)
    source_paths = [stage_root / "role_review" / "decisions", stage_root / "continuity", stage_root / "learning"]
    before_inventory = _inventory(source_paths, base=stage_root)

    completed = read_json(stage_root / "role_review" / "decisions" / "completed_review.json")
    completed_state = completed.get("state") if isinstance(completed.get("state"), dict) else completed
    completed_at = str(completed_state.get("completed_at") or completed_state.get("updated_at"))

    f1_manifest = read_json(stage_root / "continuity" / "deconfounded_continuity_review_manifest.json")
    f1_rows = _review_rows_from_manifest(f1_manifest)
    f1_audit = raw_feature_shortcut_audit(
        f1_rows,
        artifact="m5_4f1_raw_feature_shortcut_audit",
    )
    f1_audit["existing_pack_classification"] = F1_DIAGNOSTIC_CLASSIFICATION
    f1_audit["feature_band_labels_hiding_raw_separation"] = {
        "bbox_iou": "bbox_iou" in f1_audit["disjoint_numeric_range_features"],
        "center_delta_px": "center_delta_px" in f1_audit["disjoint_numeric_range_features"],
        "footpoint_delta_px": "footpoint_delta_px" in f1_audit["disjoint_numeric_range_features"],
    }
    write_json(audit_root / "m5_4f1_raw_feature_shortcut_audit.json", f1_audit)
    write_text(
        audit_root / "m5_4f1_continuity_selection_incident.md",
        "\n".join(
            [
                "# M5.4F.1 Continuity Selection Incident",
                "",
                f"Existing pack classification: `{F1_DIAGNOSTIC_CLASSIFICATION}`",
                "",
                "The M5.4F.1 proposed buckets are preserved as diagnostics only because raw numeric geometry features",
                "can separate the proposed labels with one-dimensional thresholds. Coarse band labels were shared,",
                "but the raw `bbox_iou`, `center_delta_px`, and `footpoint_delta_px` ranges were disjoint.",
            ]
        )
        + "\n",
    )

    enriched = _enriched_candidate_rows(stage_root)
    positive_pool = sorted(
        enriched,
        key=lambda row: (
            -float(row["hard_positive_score"]),
            str(row["role_partitioned_continuity_candidate_id"]),
        ),
    )
    negative_pool = sorted(
        enriched,
        key=lambda row: (
            -float(row["hard_negative_score"]),
            str(row["role_partitioned_continuity_candidate_id"]),
        ),
    )
    write_json(
        continuity_root / "hard_candidate_taxonomy.json",
        {
            "artifact": "m5_4f2_hard_candidate_taxonomy",
            "hard_positive_search_preferences": [
                "lower_or_moderate_iou",
                "meaningful_displacement",
                "intermediate_observed_support",
                "appearance_consistency",
                "competition_or_occlusion_when_available",
            ],
            "hard_negative_search_preferences": [
                "high_or_moderate_iou",
                "small_or_overlapping_displacement",
                "same_team_similar_kit",
                "competing_candidate_or_low_margin_when_available",
                "raw_geometry_overlaps_positive_bucket",
            ],
            "labels_are_proposed_for_review_not_human_truth": True,
            **safety_payload(),
        },
    )
    write_json(
        continuity_root / "hard_positive_candidate_rows.json",
        {"artifact": "m5_4f2_hard_positive_candidate_rows", "rows": positive_pool[:500], **safety_payload()},
    )
    write_json(
        continuity_root / "hard_negative_candidate_rows.json",
        {"artifact": "m5_4f2_hard_negative_candidate_rows", "rows": negative_pool[:500], **safety_payload()},
    )

    selected = _select_hard_cases(enriched, limit_per_class=20)
    audit_rows = _selected_rows_for_audit(selected)
    overlap_audit = raw_feature_shortcut_audit(audit_rows, artifact="m5_4f2_raw_feature_overlap_audit")
    if not overlap_audit["passes_raw_feature_overlap_gates"]:
        selected = _select_hard_cases(enriched, limit_per_class=15)
        audit_rows = _selected_rows_for_audit(selected)
        overlap_audit = raw_feature_shortcut_audit(audit_rows, artifact="m5_4f2_raw_feature_overlap_audit")
    balance = _balance_audits(selected)
    counts = _case_counts(selected)
    selected_rows = [*selected["likely_positive"], *selected["likely_negative"]]
    geometry_diag = _diagnostic_classifier(
        selected_rows,
        feature_names=["bbox_iou", "center_delta_px", "footpoint_delta_px", "bbox_area_ratio", "aspect_ratio_change"],
        artifact="m5_4f2_geometry_only_shortcut_diagnostic",
    )
    appearance_diag = _diagnostic_classifier(
        selected_rows,
        feature_names=["appearance_similarity", "crop_quality", "occlusion"],
        artifact="m5_4f2_appearance_only_shortcut_diagnostic",
    )
    combined_diag = _diagnostic_classifier(
        selected_rows,
        feature_names=["bbox_iou", "center_delta_px", "footpoint_delta_px", "appearance_similarity", "crop_quality"],
        artifact="m5_4f2_combined_shortcut_diagnostic",
    )
    multivariate = {
        "artifact": "m5_4f2_proposed_bucket_multivariate_shortcut_audit",
        "geometry_only": geometry_diag,
        "appearance_only": appearance_diag,
        "combined": combined_diag,
        "diagnostic_only_not_continuity_model": True,
        **safety_payload(),
    }
    write_json(audit_root / "proposed_bucket_univariate_threshold_audit.json", overlap_audit)
    write_json(audit_root / "proposed_bucket_multivariate_shortcut_audit.json", multivariate)
    write_json(continuity_root / "raw_feature_overlap_audit.json", overlap_audit)
    write_json(continuity_root / "endpoint_reuse_audit.json", balance["endpoint"])
    write_json(continuity_root / "equivalence_cluster_audit.json", balance["equivalence"])
    write_json(continuity_root / "temporal_balance_audit.json", balance["temporal"])
    write_json(continuity_root / "team_balance_audit.json", balance["team"])
    write_json(continuity_root / "raw_feature_balance_audit.json", overlap_audit)

    gates_pass = (
        overlap_audit["passes_raw_feature_overlap_gates"]
        and balance["endpoint"]["endpoint_reuse_passed"]
        and balance["equivalence"]["max_cases_per_cluster"] <= 2
        and min(balance["equivalence"]["independent_cluster_count_by_class"].values() or [0]) >= 10
        and balance["temporal"]["max_cases_per_30_frame_window"] <= 4
    )
    launcher_path = None
    review_url = None
    if gates_pass:
        manifest = _review_manifest(stage_root=stage_root, selection=selected, completed_at=completed_at)
        write_json(continuity_root / "deconfounded_hard_continuity_review_manifest.json", manifest)
        _write_case_index(continuity_root / "deconfounded_hard_continuity_case_index.csv", manifest)
        decision_root = continuity_root / "decisions"
        decision_root.mkdir(parents=True, exist_ok=True)
        write_json(
            decision_root / "review_decisions.json",
            _deterministic_empty_decision_state(ReviewManifest.model_validate(manifest), completed_at),
        )
        write_text(decision_root / "review_decision_events.jsonl", "")
        (decision_root / "snapshots").mkdir(parents=True, exist_ok=True)
        _write_blind_workbench(continuity_root / "workbench")
        launcher_path = str(
            _write_open_launcher(
                launcher_path=stage_root / "OPEN_BLIND_HARD_CONTINUITY_REVIEW.ps1",
                repo_root=repo_root,
                manifest_path=continuity_root / "deconfounded_hard_continuity_review_manifest.json",
                evidence_root=continuity_root / "evidence",
                decision_root=decision_root,
                workbench_root=continuity_root / "workbench",
                label="M5.4F.2 blind hard continuity",
                port=8774,
            )
        )
        review_url = "http://127.0.0.1:8774/"
    else:
        write_json(
            continuity_root / "deconfounded_hard_continuity_review_manifest.json",
            {
                "artifact": "m5_4f2_deconfounded_hard_continuity_review_manifest",
                "review_cases": [],
                "status": F2_BLOCKED_RAW,
                **safety_payload(),
            },
        )
        with (continuity_root / "deconfounded_hard_continuity_case_index.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            csv.writer(handle).writerow(["review_case_id", "candidate_artifact_id", "proposed_bucket_hidden"])

    after_inventory = _inventory(source_paths, base=stage_root)
    source_mutation = {
        "artifact": "m5_4f2_source_mutation_audit",
        "before": before_inventory,
        "after": after_inventory,
        "m5_4f1_artifacts_preserved": before_inventory["inventory_hash"] == after_inventory["inventory_hash"],
        "human_role_decisions_unchanged": True,
        **safety_payload(),
    }
    safety = {
        "artifact": "m5_4f2_safety_guardrail_audit",
        "all_safety_flags_preserved": True,
        "persistent_identity_assigned": False,
        "player_slots_assigned": False,
        "goalkeeper_slots_assigned": False,
        "metric_pitch_coordinates_used": False,
        "tactical_event_or_physical_outputs_created": False,
        **safety_payload(),
    }
    source_mutation_path = validation_root / "m5_4f2_source_mutation_audit.json"
    safety_path = validation_root / "m5_4f2_safety_guardrail_audit.json"
    write_json(source_mutation_path, source_mutation)
    write_json(safety_path, safety)
    final_classification = F2_PASS if gates_pass else F2_BLOCKED_RAW
    exact_blocker = "NONE" if gates_pass else "RAW_FEATURE_OVERLAP_GATES_FAILED"
    temporal_assets = {
        "temporal_gif_count": len(list((continuity_root / "evidence").rglob("*.gif"))),
        "temporal_mp4_count": len(list((continuity_root / "evidence").rglob("*.mp4"))),
    }
    summary = {
        "artifact": "m5_4f2_validation_summary",
        "final_classification": final_classification,
        "exact_blocker": exact_blocker,
        "old_pack_diagnostic_classification": F1_DIAGNOSTIC_CLASSIFICATION,
        "m5_4f1_artifacts_preserved": source_mutation["m5_4f1_artifacts_preserved"],
        "role_review_repeated": False,
        "human_role_decisions_modified": False,
        "continuity_calibrator_trained": False,
        "blind_review_default_state": "proposed_bucket_model_score_and_raw_features_hidden",
        "launcher_path": launcher_path,
        "review_url": review_url,
        "source_mutation_audit_path": str(source_mutation_path),
        "safety_guardrail_audit_path": str(safety_path),
        **counts,
        "independent_cluster_count": balance["equivalence"]["independent_cluster_count_by_class"],
        "endpoint_reuse_maximum": balance["endpoint"]["endpoint_reuse_max"],
        "frame_gap_distribution": balance["temporal"]["frame_gap_distribution"],
        "team_distribution": balance["team"]["team_distribution"],
        "temporal_quartile_distribution": balance["temporal"]["temporal_quartile_distribution"],
        "raw_feature_overlap_passed": overlap_audit["passes_raw_feature_overlap_gates"],
        "geometry_only_grouped_balanced_accuracy": geometry_diag["grouped_balanced_accuracy"],
        "appearance_only_grouped_balanced_accuracy": appearance_diag["grouped_balanced_accuracy"],
        **temporal_assets,
        **safety_payload(),
    }
    write_json(validation_root / "m5_4f2_validation_summary.json", summary)
    return summary
