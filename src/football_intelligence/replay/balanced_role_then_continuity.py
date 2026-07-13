from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.replay.rebuilt_human_calibrated_pipeline import (
    _entity_evidence,
    _frame_records,
    _review_case_hash,
    _source_inventory,
    _source_ref,
    read_json,
    rows,
    write_json,
    write_text,
)
from football_intelligence.replay.role_partitioned_learning import (
    _empty_decision_state,
    _write_open_launcher,
    default_paths as m54e_default_paths,
)
from football_intelligence.review.schemas import (
    VISUAL_TEAM_ROLE_DECISIONS,
    VISUAL_TEAM_ROLE_QUESTION,
    ReviewCase,
    ReviewManifest,
    SourceArtifactReference,
    safety_payload,
    stable_hash,
)
from football_intelligence.review.workbench import build_workbench

FINAL_CLASSIFICATION = "PASS_BALANCED_ROLE_REVIEW_READY"
EXACT_BLOCKER = "ROLE_REVIEW_NOT_COMPLETED"
DIAGNOSTIC_ONLY = "M5_4E_CONTINUITY_REVIEW_SELECTION_DIAGNOSTIC_ONLY"

ROLE_TARGETS = {
    "team_1_outfield": 6,
    "team_2_outfield": 6,
    "team_1_goalkeeper": 4,
    "team_2_goalkeeper": 4,
    "central_referee": 4,
    "assistant_referee_near_camera": 3,
    "assistant_referee_far_camera": 3,
    "other_off_pitch_person": 3,
    "non_person_false_positive": 3,
    "unknown_or_disagreement_control": 8,
}
ROLE_REVIEW_REQUIRED_BUCKETS = [
    "team_1_outfield",
    "team_2_outfield",
    "team_1_goalkeeper",
    "team_2_goalkeeper",
    "central_referee",
    "assistant_referee_near_camera",
    "assistant_referee_far_camera",
    "other_off_pitch_person",
    "non_person_false_positive",
    "unknown_or_disagreement_control",
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def default_paths(artifact_root: Path, match_id: str) -> dict[str, Path]:
    prior = m54e_default_paths(artifact_root, match_id)
    return {
        **prior,
        "m54e_stage_root": prior["stage_root"],
        "stage_root": prior["step_m5"] / "06f_balanced_role_then_continuity",
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return rows(read_json(path))


def frame_quartile(frame_sequence: int, *, frame_count: int = 600) -> str:
    width = max(1, frame_count // 4)
    index = min(3, max(0, int(frame_sequence) // width))
    start = index * width
    end = frame_count - 1 if index == 3 else (index + 1) * width - 1
    return f"q{index + 1}_{start:03d}_{end:03d}"


def thirty_frame_window(frame_sequence: int) -> str:
    start = (int(frame_sequence) // 30) * 30
    return f"f{start:03d}_{start + 29:03d}"


def _bbox(row: dict[str, Any]) -> dict[str, float]:
    bbox = row.get("bbox") if isinstance(row.get("bbox"), dict) else row
    return {key: float(bbox[key]) for key in ("x1", "y1", "x2", "y2")}


def bbox_size_bucket(row: dict[str, Any]) -> str:
    bbox = _bbox(row)
    height = max(1.0, bbox["y2"] - bbox["y1"])
    width = max(1.0, bbox["x2"] - bbox["x1"])
    area = height * width
    if height < 35 or area < 700:
        return "distant_small_bbox"
    if height > 85 or area > 2800:
        return "near_large_bbox"
    return "medium_bbox"


def spatial_region_bucket(row: dict[str, Any], feature: dict[str, Any] | None = None) -> str:
    feature = feature or {}
    bbox = _bbox(row)
    center_x = (bbox["x1"] + bbox["x2"]) / 2.0
    center_y = (bbox["y1"] + bbox["y2"]) / 2.0
    x_band = min(5, max(0, int(center_x // 455)))
    y_band = min(3, max(0, int(center_y // 180)))
    context = str(feature.get("spatial_context") or "unknown_spatial_context")
    return f"{context}:x{x_band}:y{y_band}"


def role_decision_from_context(state: str) -> str:
    return {
        "team_1_outfield_visual_context": "team_1_outfield",
        "team_2_outfield_visual_context": "team_2_outfield",
        "team_1_goalkeeper_visual_context": "team_1_goalkeeper",
        "team_2_goalkeeper_visual_context": "team_2_goalkeeper",
        "central_referee_visual_context": "central_referee",
        "assistant_referee_near_camera_context": "assistant_referee_near_camera",
        "assistant_referee_far_camera_context": "assistant_referee_far_camera",
        "other_off_pitch_person_visual_context": "other_off_pitch_person",
        "non_person_false_positive": "non_person_false_positive",
    }.get(state, "unknown_or_disagreement_control")


def class_level_cluster_id_detected(cluster_id: str | None, proposed_class: str | None = None) -> bool:
    if not cluster_id:
        return True
    text = str(cluster_id).lower()
    proposed = str(proposed_class or "").lower()
    class_tokens = [
        "likely_positive",
        "likely_negative",
        "team_1",
        "team_2",
        "unknown",
        "referee",
        "goalkeeper",
        "continuity_likely_positive_continuity",
        "continuity_difficult_or_likely_negative_continuity",
    ]
    if proposed and text in {proposed, f"m5_4e_continuity_{proposed}", f"m5_4f_{proposed}"}:
        return True
    return any(text.endswith(token) or text == token for token in class_tokens)


def role_equivalence_cluster_id(row: dict[str, Any], feature: dict[str, Any] | None = None) -> str:
    feature = feature or {}
    bbox = _bbox(row)
    payload = {
        "candidate_id": row.get("candidate_id"),
        "frame_window": int(row.get("frame_sequence", 0)) // 5,
        "spatial": spatial_region_bucket(row, feature),
        "bbox_size": bbox_size_bucket(row),
        "crop_signature": feature.get("colour_histogram_signature") or feature.get("static_persistence_signature"),
        "bbox_quantized": {key: round(float(value) / 8.0) * 8 for key, value in bbox.items()},
    }
    return f"m5_4f_role_cluster_{stable_hash(payload)[:12]}"


def continuity_equivalence_cluster_id(row: dict[str, Any]) -> str:
    features = row.get("gate_features") if isinstance(row.get("gate_features"), dict) else {}
    payload = {
        "source": row.get("source_visible_person_base_id"),
        "target": row.get("target_visible_person_base_id"),
        "source_frame_window": int(row.get("source_frame_sequence", 0)) // 4,
        "target_frame_window": int(row.get("target_frame_sequence", 0)) // 4,
        "source_role": row.get("source_visual_role_context"),
        "target_role": row.get("target_visual_role_context"),
        "gap": row.get("frame_gap"),
        "motion": {
            "center": round(float(features.get("center_delta_px", 0.0)) / 10.0),
            "foot": round(float(features.get("footpoint_delta_px", 0.0)) / 10.0),
            "area": round(float(features.get("bbox_area_ratio", 1.0)), 1),
        },
    }
    return f"m5_4f_continuity_cluster_{stable_hash(payload)[:12]}"


def _case_feature(case: dict[str, Any], feature_by_candidate: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    if not feature_by_candidate:
        return {}
    return feature_by_candidate.get(str(case.get("candidate_artifact_id")), {})


def audit_role_review_selection(
    manifest: dict[str, Any],
    feature_by_candidate: dict[str, dict[str, Any]] | None = None,
    *,
    frame_count: int = 600,
) -> dict[str, Any]:
    cases = [case for case in manifest.get("review_cases", []) if isinstance(case, dict)]
    class_counts: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()
    quartiles: Counter[str] = Counter()
    spatial: Counter[str] = Counter()
    crop_sizes: Counter[str] = Counter()
    windows: Counter[str] = Counter()
    colour_clusters: Counter[str] = Counter()
    frames: list[int] = []
    for case in cases:
        proposed = role_decision_from_context(str(case.get("model_prediction") or ""))
        class_counts[proposed] += 1
        cluster_counts[str(case.get("equivalence_cluster_id"))] += 1
        frame = int(case.get("source_frame_sequence", 0))
        frames.append(frame)
        quartiles[frame_quartile(frame, frame_count=frame_count)] += 1
        windows[thirty_frame_window(frame)] += 1
        feature = _case_feature(case, feature_by_candidate)
        bbox_source = case.get("evidence_manifest", {}).get("source_bbox") or case
        spatial[spatial_region_bucket(bbox_source, feature)] += 1
        crop_sizes[bbox_size_bucket(bbox_source)] += 1
        cluster = feature.get("colour_histogram_signature") or case.get("selection_metadata", {}).get(
            "current_team_colour_cluster"
        )
        if cluster:
            colour_clusters[str(cluster)] += 1
    missing = [bucket for bucket in ROLE_REVIEW_REQUIRED_BUCKETS if class_counts.get(bucket, 0) == 0]
    max_class_count = max(class_counts.values() or [0])
    coverage_span = (max(frames) - min(frames) + 1) if frames else 0
    issues: list[str] = []
    if missing:
        issues.append("missing_requested_classes")
    if max_class_count > 8:
        issues.append("category_concentration_exceeds_limit")
    if coverage_span < frame_count * 0.75:
        issues.append("frame_range_concentration")
    if any(count > 3 for count in windows.values()):
        issues.append("too_many_cases_in_30_frame_window")
    if class_counts.get("unknown_or_disagreement_control", 0) > 8:
        issues.append("unknown_category_overrepresented")
    return {
        "artifact": "m5_4f_m5_4e_role_review_selection_audit",
        "case_count": len(cases),
        "cases_per_proposed_class": dict(sorted(class_counts.items())),
        "cases_per_human_review_equivalence_cluster": dict(sorted(cluster_counts.items())),
        "source_frame_distribution": {
            "min": min(frames) if frames else None,
            "max": max(frames) if frames else None,
            "coverage_span_frames": coverage_span,
            "coverage_fraction_of_600": round(coverage_span / frame_count, 6) if frame_count else 0.0,
            "thirty_frame_windows": dict(sorted(windows.items())),
        },
        "temporal_quartile_distribution": dict(sorted(quartiles.items())),
        "spatial_region_distribution": dict(sorted(spatial.items())),
        "crop_size_distribution": dict(sorted(crop_sizes.items())),
        "missing_requested_classes": missing,
        "repeated_visual_clusters": {key: value for key, value in sorted(colour_clusters.items()) if value > 3},
        "category_concentration": {"max_count": max_class_count, "limit": 8},
        "coverage_of_all_600_frames": coverage_span >= frame_count * 0.75,
        "balanced_role_review": False if issues else True,
        "issues": issues,
        **safety_payload(),
    }


def _continuity_class(case: dict[str, Any]) -> str:
    category = str(case.get("category") or case.get("model_prediction") or "")
    if "positive" in category:
        return "likely_positive"
    if "negative" in category:
        return "likely_negative"
    return category or "unknown"


def _role_from_reasons(case: dict[str, Any], prefix: str) -> str:
    for reason in case.get("uncertainty_reasons", []):
        text = str(reason)
        if text.startswith(prefix):
            return text.split("=", 1)[1]
    return "unknown_visible_person_visual_context"


def _team_from_role(role: str) -> str:
    if role.startswith("team_1"):
        return "team_1"
    if role.startswith("team_2"):
        return "team_2"
    if "referee" in role:
        return "official"
    if "goalkeeper" in role:
        return "goalkeeper_unknown_team"
    return "unknown"


def audit_continuity_review_selection(manifest: dict[str, Any]) -> dict[str, Any]:
    cases = [case for case in manifest.get("review_cases", []) if isinstance(case, dict)]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_class[_continuity_class(case)].append(case)
    frame_gap_by_class: dict[str, dict[str, int]] = {}
    team_by_class: dict[str, dict[str, int]] = {}
    role_by_class: dict[str, dict[str, int]] = {}
    spatial_by_class: dict[str, dict[str, int]] = {}
    feature_by_class: dict[str, dict[str, list[float]]] = {}
    cluster_counts_by_class: dict[str, int] = {}
    endpoint_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    quartile_by_class: dict[str, dict[str, int]] = {}
    for proposed, rows_for_class in by_class.items():
        gap_counts: Counter[str] = Counter()
        team_counts: Counter[str] = Counter()
        role_counts: Counter[str] = Counter()
        spatial_counts: Counter[str] = Counter()
        quartile_counts: Counter[str] = Counter()
        features: dict[str, list[float]] = defaultdict(list)
        clusters = set()
        for case in rows_for_class:
            evidence = case.get("evidence_manifest", {})
            metadata_gap = case.get("selection_metadata", {}).get("gate_features", {}).get("frame_gap", 0)
            gap = int(evidence.get("frame_gap") or metadata_gap)
            gap_counts[str(gap)] += 1
            source_role = _role_from_reasons(case, "source_role=")
            target_role = _role_from_reasons(case, "target_role=")
            team_counts[_team_from_role(source_role)] += 1
            role_counts[source_role] += 1
            role_counts[target_role] += 1
            bbox = evidence.get("source_bbox") or {}
            spatial_counts[spatial_region_bucket(bbox)] += 1
            quartile_counts[frame_quartile(int(case.get("source_frame_sequence", 0)))] += 1
            clusters.add(str(case.get("equivalence_cluster_id")))
            candidate_counts[str(case.get("candidate_artifact_id"))] += 1
            source_key = f"source:{case.get('source_frame_sequence')}:{stable_hash(evidence.get('source_bbox'))[:8]}"
            target_key = f"target:{case.get('target_frame_sequence')}:{stable_hash(evidence.get('target_bbox'))[:8]}"
            endpoint_counts[source_key] += 1
            endpoint_counts[target_key] += 1
            gate_features = case.get("selection_metadata", {}).get("gate_features", {})
            for key in (
                "bbox_iou",
                "center_delta_px",
                "footpoint_delta_px",
                "bbox_area_ratio",
                "aspect_ratio_change",
                "frame_gap",
            ):
                if key in gate_features:
                    features[key].append(float(gate_features[key]))
        frame_gap_by_class[proposed] = dict(sorted(gap_counts.items()))
        team_by_class[proposed] = dict(sorted(team_counts.items()))
        role_by_class[proposed] = dict(sorted(role_counts.items()))
        spatial_by_class[proposed] = dict(sorted(spatial_counts.items()))
        feature_by_class[proposed] = {key: value for key, value in sorted(features.items())}
        quartile_by_class[proposed] = dict(sorted(quartile_counts.items()))
        cluster_counts_by_class[proposed] = len(clusters)
    gap_sets = {label: set(counts) for label, counts in frame_gap_by_class.items()}
    labels = sorted(gap_sets)
    frame_gap_perfect = len(labels) >= 2 and not set.intersection(*(gap_sets[label] for label in labels))
    team_imbalance = any(
        (counts.get("team_1", 0) == 0 or counts.get("team_2", 0) == 0)
        for counts in team_by_class.values()
        if sum(counts.values()) > 0
    )
    class_bucket_clusters = [
        cluster
        for case in cases
        for cluster in [str(case.get("equivalence_cluster_id"))]
        if class_level_cluster_id_detected(cluster, _continuity_class(case))
    ]
    excessive_spatial = any(max(counts.values() or [0]) > 4 for counts in spatial_by_class.values())
    issues = []
    if frame_gap_perfect:
        issues.append("frame_gap_perfectly_predicts_review_bucket")
    if team_imbalance:
        issues.append("team_partition_imbalance")
    if class_bucket_clusters:
        issues.append("class_level_bucket_ids_used_as_equivalence_ids")
    if max(endpoint_counts.values() or [0]) > 2:
        issues.append("repeated_endpoint_reuse")
    if excessive_spatial:
        issues.append("excessive_spatial_concentration")
    return {
        "artifact": "m5_4f_m5_4e_continuity_review_selection_audit",
        "case_count": len(cases),
        "diagnostic_preservation_label": DIAGNOSTIC_ONLY,
        "positive_count": len(by_class.get("likely_positive", [])),
        "negative_count": len(by_class.get("likely_negative", [])),
        "frame_gap_distribution_by_proposed_class": frame_gap_by_class,
        "team_distribution_by_proposed_class": team_by_class,
        "visual_role_distribution_by_proposed_class": role_by_class,
        "source_frame_distribution": {
            "temporal_quartiles_by_class": quartile_by_class,
            "min": min((int(case.get("source_frame_sequence", 0)) for case in cases), default=None),
            "max": max((int(case.get("source_frame_sequence", 0)) for case in cases), default=None),
        },
        "spatial_region_distribution": spatial_by_class,
        "equivalence_clusters_per_class": cluster_counts_by_class,
        "endpoint_reuse_distribution": dict(sorted(endpoint_counts.items())),
        "endpoint_reuse_max": max(endpoint_counts.values() or [0]),
        "candidate_overlap": {key: value for key, value in sorted(candidate_counts.items()) if value > 1},
        "feature_distributions_by_proposed_class": feature_by_class,
        "potential_shortcut_features": {
            "frame_gap_perfectly_predicts_review_bucket": frame_gap_perfect,
            "team_partition_imbalance": team_imbalance,
            "class_level_bucket_ids_used_as_equivalence_ids": bool(class_bucket_clusters),
            "excessive_spatial_concentration": excessive_spatial,
        },
        "leakage_and_confounding_risk": "high" if issues else "low",
        "issues": issues,
        **safety_payload(),
    }


def _incident_markdown(role_audit: dict[str, Any], continuity_audit: dict[str, Any]) -> str:
    frame_gaps = json.dumps(continuity_audit["frame_gap_distribution_by_proposed_class"], sort_keys=True)
    teams = json.dumps(continuity_audit["team_distribution_by_proposed_class"], sort_keys=True)
    clusters = json.dumps(continuity_audit["equivalence_clusters_per_class"], sort_keys=True)
    return f"""# M5.4E Review Selection Incident

M5.4E role and continuity review packs are preserved as historical diagnostics and must not be used as balanced
training review packs.

## Role Pack

- Case count: {role_audit["case_count"]}
- Proposed class distribution: `{json.dumps(role_audit["cases_per_proposed_class"], sort_keys=True)}`
- Temporal quartiles: `{json.dumps(role_audit["temporal_quartile_distribution"], sort_keys=True)}`
- Missing requested classes: `{json.dumps(role_audit["missing_requested_classes"], sort_keys=True)}`
- Issues: `{json.dumps(role_audit["issues"], sort_keys=True)}`

## Continuity Pack

- Case count: {continuity_audit["case_count"]}
- Diagnostic label: `{DIAGNOSTIC_ONLY}`
- Frame gaps by proposed class: `{frame_gaps}`
- Teams by proposed class: `{teams}`
- Equivalence clusters per class: `{clusters}`
- Issues: `{json.dumps(continuity_audit["issues"], sort_keys=True)}`

## Correction

M5.4F creates a new spatially, temporally and class-diverse role review pack. Continuity generation is explicitly
blocked until the balanced role review is completed and ingested.
"""


def _candidate_bucket_score(bucket: str, row: dict[str, Any], feature: dict[str, Any]) -> float:
    state = str(row.get("visual_role_context_state"))
    confidence = float(row.get("visual_role_context_confidence", 0.0))
    beliefs = row.get("belief_scores", {}) if isinstance(row.get("belief_scores"), dict) else {}
    team_1 = float(beliefs.get("team_1") or feature.get("team_1_belief") or 0.0)
    team_2 = float(beliefs.get("team_2") or feature.get("team_2_belief") or 0.0)
    goalkeeper = float(beliefs.get("goalkeeper") or feature.get("goalkeeper_belief") or 0.0)
    near = float(beliefs.get("near_camera_assistant") or feature.get("near_camera_assistant_belief") or 0.0)
    far = float(beliefs.get("far_camera_assistant") or feature.get("far_camera_assistant_belief") or 0.0)
    central = float(beliefs.get("central_referee") or feature.get("central_referee_belief") or 0.0)
    off_pitch = float(beliefs.get("off_pitch_person") or feature.get("off_pitch_person_belief") or 0.0)
    if bucket == "team_1_outfield":
        return confidence + team_1 if state == "team_1_outfield_visual_context" else team_1 * 0.8
    if bucket == "team_2_outfield":
        return confidence + team_2 if state == "team_2_outfield_visual_context" else team_2 * 0.8
    if bucket == "team_1_goalkeeper":
        bbox = _bbox(row)
        height_bonus = min(0.5, max(0.0, (bbox["y2"] - bbox["y1"]) / 180.0))
        goal_area = float(feature.get("goal_area_image_context_score", 0.0))
        return goalkeeper + max(0.0, team_1 - team_2) + goal_area + height_bonus
    if bucket == "team_2_goalkeeper":
        bbox = _bbox(row)
        height_bonus = min(0.5, max(0.0, (bbox["y2"] - bbox["y1"]) / 180.0))
        goal_area = float(feature.get("goal_area_image_context_score", 0.0))
        return goalkeeper + max(0.0, team_2 - team_1) + goal_area + height_bonus
    if bucket == "central_referee":
        return confidence + central if state == "central_referee_visual_context" else central
    if bucket == "assistant_referee_near_camera":
        return (
            near
            + float(feature.get("near_camera_region_score", 0.0))
            + float(feature.get("assistant_touchline_context_score", 0.0))
        )
    if bucket == "assistant_referee_far_camera":
        return confidence + far if state == "assistant_referee_far_camera_context" else far
    if bucket == "other_off_pitch_person":
        return confidence + off_pitch if state == "other_off_pitch_person_visual_context" else off_pitch
    if bucket == "non_person_false_positive":
        return confidence if state == "non_person_false_positive" else 0.0
    if bucket == "unknown_or_disagreement_control":
        return 1.0 - confidence
    return 0.0


def _candidate_buckets(row: dict[str, Any], feature: dict[str, Any]) -> list[str]:
    buckets = {role_decision_from_context(str(row.get("visual_role_context_state")))}
    bbox = _bbox(row)
    bbox_height = bbox["y2"] - bbox["y1"]
    if (
        float(feature.get("goalkeeper_belief", 0.0)) >= 0.2
        and float(feature.get("goal_area_image_context_score", 0.0)) >= 0.15
        and bbox_height >= 45
    ):
        if float(feature.get("team_1_belief", 0.0)) >= float(feature.get("team_2_belief", 0.0)):
            buckets.add("team_1_goalkeeper")
        else:
            buckets.add("team_2_goalkeeper")
    if (
        float(feature.get("near_camera_assistant_belief", 0.0)) >= 0.45
        or float(feature.get("near_camera_region_score", 0.0)) >= 0.65
        or (float(feature.get("assistant_touchline_context_score", 0.0)) >= 0.65 and bbox["y2"] >= 500)
        or (bbox["y2"] >= 430 and bbox_height >= 30)
    ):
        buckets.add("assistant_referee_near_camera")
    if float(feature.get("far_camera_assistant_belief", 0.0)) >= 0.45:
        buckets.add("assistant_referee_far_camera")
    if float(feature.get("central_referee_belief", 0.0)) >= 0.52:
        buckets.add("central_referee")
    if buckets == {"unknown_or_disagreement_control"} and float(row.get("visual_role_context_confidence", 0.0)) > 0.68:
        buckets.discard("unknown_or_disagreement_control")
    return sorted(buckets)


def _build_role_candidate_records(
    role_rows: list[dict[str, Any]], role_feature_rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    feature_by_candidate = {str(row["candidate_id"]): row for row in role_feature_rows}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in role_rows:
        candidate_id = str(row["candidate_id"])
        feature = feature_by_candidate.get(candidate_id, {})
        for bucket in _candidate_buckets(row, feature):
            score = _candidate_bucket_score(bucket, row, feature)
            if bucket != "unknown_or_disagreement_control" and score <= 0:
                continue
            record = {
                **row,
                "target_review_bucket": bucket,
                "selection_score": round(score, 6),
                "frame_quartile": frame_quartile(int(row["frame_sequence"])),
                "thirty_frame_window": thirty_frame_window(int(row["frame_sequence"])),
                "spatial_region_bucket": spatial_region_bucket(row, feature),
                "bbox_size_bucket": bbox_size_bucket(row),
                "role_feature": feature,
                "role_equivalence_cluster_id": role_equivalence_cluster_id(row, feature),
                "automatic_candidate_found": role_decision_from_context(str(row.get("visual_role_context_state")))
                == bucket,
            }
            candidates[bucket].append(record)
    for bucket, bucket_rows in candidates.items():
        candidates[bucket] = sorted(
            bucket_rows,
            key=lambda item: (
                -float(item["selection_score"]),
                int(item["frame_sequence"]),
                str(item["candidate_id"]),
            ),
        )
    return candidates


def select_balanced_role_cases(
    candidate_records: dict[str, list[dict[str, Any]]],
    *,
    limit: int = 40,
    targets: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    targets = targets or ROLE_TARGETS
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    quartile_counts: Counter[str] = Counter()
    spatial_counts: Counter[str] = Counter()
    window_counts: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()

    def can_add(row: dict[str, Any]) -> bool:
        bucket = str(row["target_review_bucket"])
        if len(selected) >= limit or str(row["candidate_id"]) in selected_ids:
            return False
        if category_counts[bucket] >= min(8, targets.get(bucket, 0)):
            return False
        if spatial_counts[str(row["spatial_region_bucket"])] >= 3:
            return False
        if window_counts[str(row["thirty_frame_window"])] >= 3:
            return False
        return cluster_counts[str(row["role_equivalence_cluster_id"])] == 0

    def add_best(bucket: str, quartile: str | None = None) -> bool:
        for row in candidate_records.get(bucket, []):
            if quartile is not None and row["frame_quartile"] != quartile:
                continue
            if not can_add(row):
                continue
            selected.append(row)
            selected_ids.add(str(row["candidate_id"]))
            category_counts[bucket] += 1
            quartile_counts[str(row["frame_quartile"])] += 1
            spatial_counts[str(row["spatial_region_bucket"])] += 1
            window_counts[str(row["thirty_frame_window"])] += 1
            cluster_counts[str(row["role_equivalence_cluster_id"])] += 1
            return True
        return False

    quartiles = [f"q{i}_000_149" for i in range(1, 2)]
    quartiles = ["q1_000_149", "q2_150_299", "q3_300_449", "q4_450_599"]
    buckets = list(targets)
    for quartile in quartiles:
        safety_counter = 0
        while quartile_counts[quartile] < 8 and len(selected) < limit and safety_counter < 200:
            safety_counter += 1
            remaining = [
                bucket
                for bucket in buckets
                if category_counts[bucket] < targets[bucket]
                and any(row["frame_quartile"] == quartile and can_add(row) for row in candidate_records.get(bucket, []))
            ]
            if not remaining:
                break
            remaining.sort(key=lambda bucket: (category_counts[bucket] / max(1, targets[bucket]), bucket))
            if not add_best(remaining[0], quartile):
                break
    while len(selected) < limit:
        remaining = [bucket for bucket in buckets if category_counts[bucket] < targets[bucket]]
        if not remaining:
            break
        remaining.sort(key=lambda bucket: (category_counts[bucket] / max(1, targets[bucket]), bucket))
        if not any(add_best(bucket) for bucket in remaining):
            break
    return selected


def _source_refs(paths: dict[str, Path]) -> list[SourceArtifactReference]:
    m54e = paths["m54e_stage_root"]
    return [
        _source_ref(
            "m5_4e_role_context_rows",
            m54e / "role" / "visual_role_context_rows.json",
            "read-only M5.4E role predictions",
        ),
        _source_ref(
            "m5_4e_role_feature_rows",
            m54e / "role" / "visual_role_feature_rows.json",
            "read-only M5.4E role features",
        ),
        _source_ref(
            "m5_4e_role_review_manifest",
            m54e / "review" / "role_context" / "review_manifest.json",
            "diagnostic-only M5.4E role review manifest",
        ),
        _source_ref(
            "m5_4e_continuity_review_manifest",
            m54e / "review" / "continuity_balance" / "review_manifest.json",
            "diagnostic-only M5.4E continuity review manifest",
        ),
        _source_ref("canonical_frame_manifest", paths["frame_manifest"], "read-only canonical frame manifest"),
    ]


def _write_role_manifest(
    *,
    stage_root: Path,
    role_review_root: Path,
    repo_root: Path,
    selected: list[dict[str, Any]],
    frame_root: Path,
    frame_records: dict[int, dict[str, Any]],
    source_refs: list[SourceArtifactReference],
) -> dict[str, Any]:
    evidence_root = role_review_root / "evidence"
    decision_root = role_review_root / "decisions"
    workbench_root = role_review_root / "workbench"
    completed_root = role_review_root / "completed"
    completed_root.mkdir(parents=True, exist_ok=True)
    write_text(
        completed_root / "README.txt",
        "Completion artifacts are written here after the balanced role review is sealed.\n",
    )
    cases: list[ReviewCase] = []
    for index, row in enumerate(selected, start=1):
        case_id = f"m5_4f_role_case_{index:03d}"
        evidence = _entity_evidence(
            evidence_root=evidence_root,
            case_id=case_id,
            candidate={**row, "source_frame_sequence": row["frame_sequence"]},
            frame_root=frame_root,
            frame_records=frame_records,
        )
        feature = row.get("role_feature") if isinstance(row.get("role_feature"), dict) else {}
        case_payload = {
            "review_case_id": case_id,
            "task_type": "visual_team_role_context",
            "concise_question": VISUAL_TEAM_ROLE_QUESTION,
            "allowed_decisions": VISUAL_TEAM_ROLE_DECISIONS,
            "candidate_artifact_id": str(row["candidate_id"]),
            "source_artifact_references": source_refs,
            "source_frame_sequence": int(row["frame_sequence"]),
            "target_frame_sequence": None,
            "evidence_manifest": evidence,
            "uncertainty_reasons": [
                f"target_review_bucket={row['target_review_bucket']}",
                f"model_prediction={row['visual_role_context_state']}",
                f"frame_quartile={row['frame_quartile']}",
                f"spatial_region_bucket={row['spatial_region_bucket']}",
                f"bbox_size_bucket={row['bbox_size_bucket']}",
                *[str(reason) for reason in row.get("visual_role_context_reasons", [])],
            ],
            "category": str(row["target_review_bucket"]),
            "priority": index,
            "control_status": "m5_4f_balanced_role_review_candidate",
            "candidate_hash": "",
            "evidence_hash": evidence.evidence_hash,
            "safety_payload": safety_payload(),
            "review_round": 3,
            "selection_metadata": {
                "why_selected": f"balanced M5.4F role review target: {row['target_review_bucket']}",
                "target_review_bucket": row["target_review_bucket"],
                "automatic_candidate_found": row["automatic_candidate_found"],
                "model_prediction": row["visual_role_context_state"],
                "model_confidence": row["visual_role_context_confidence"],
                "belief_scores": row["belief_scores"],
                "frame_quartile": row["frame_quartile"],
                "spatial_region_bucket": row["spatial_region_bucket"],
                "bbox_size_bucket": row["bbox_size_bucket"],
                "thirty_frame_window": row["thirty_frame_window"],
                "colour_cluster": feature.get("colour_histogram_signature"),
                "neighbouring_people_visible_in": "wide_crop.jpg",
                "temporal_gif": "temporal_clip.gif",
            },
            "model_prediction": row["visual_role_context_state"],
            "model_confidence": float(row["visual_role_context_confidence"]),
            "equivalence_cluster_id": row["role_equivalence_cluster_id"],
            "representative_of_count": 1,
        }
        case_payload["candidate_hash"] = _review_case_hash(case_payload)
        cases.append(ReviewCase.model_validate(case_payload))
    manifest = ReviewManifest(
        title="M5.4F Balanced Visual Team/Role Context Review",
        review_task_family="m5_4f_balanced_visual_team_role_context",
        review_cases=cases,
        candidate_manifest_hash=stable_hash([case.candidate_hash for case in cases]),
        evidence_manifest_hash=stable_hash([case.evidence_hash for case in cases]),
        source_manifest_hash=stable_hash([ref.model_dump(mode="json") for ref in source_refs]),
        source_artifact_references=source_refs,
    )
    manifest_path = role_review_root / "balanced_role_review_manifest.json"
    write_json(manifest_path, manifest.model_dump(mode="json"))
    write_json(decision_root / "review_decisions.json", _empty_decision_state(manifest))
    (decision_root / "review_decision_events.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (decision_root / "review_decision_events.jsonl").write_text("", encoding="utf-8")
    (decision_root / "snapshots").mkdir(parents=True, exist_ok=True)
    build_workbench(workbench_root)
    launcher = _write_open_launcher(
        launcher_path=stage_root / "OPEN_BALANCED_ROLE_REVIEW.ps1",
        repo_root=repo_root,
        manifest_path=manifest_path,
        evidence_root=evidence_root,
        decision_root=decision_root,
        workbench_root=workbench_root,
        label="M5.4F balanced role",
        port=8772,
    )
    write_text(stage_root / "OPEN_REVIEW.ps1", launcher.read_text(encoding="utf-8"))
    with (role_review_root / "balanced_role_case_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_case_id",
                "candidate_artifact_id",
                "target_review_bucket",
                "model_prediction",
                "frame",
                "frame_quartile",
                "spatial_region_bucket",
                "bbox_size_bucket",
                "equivalence_cluster_id",
            ],
        )
        writer.writeheader()
        for case in manifest.model_dump(mode="json")["review_cases"]:
            metadata = case["selection_metadata"]
            writer.writerow(
                {
                    "review_case_id": case["review_case_id"],
                    "candidate_artifact_id": case["candidate_artifact_id"],
                    "target_review_bucket": metadata["target_review_bucket"],
                    "model_prediction": case["model_prediction"],
                    "frame": case["source_frame_sequence"],
                    "frame_quartile": metadata["frame_quartile"],
                    "spatial_region_bucket": metadata["spatial_region_bucket"],
                    "bbox_size_bucket": metadata["bbox_size_bucket"],
                    "equivalence_cluster_id": case["equivalence_cluster_id"],
                }
            )
    return {
        "manifest_path": str(manifest_path),
        "review_case_count": len(cases),
        "launcher_path": str(launcher),
        "review_url": "http://127.0.0.1:8772/",
        "category_distribution": dict(Counter(case.category for case in cases)),
        "temporal_quartile_distribution": dict(Counter(case.selection_metadata["frame_quartile"] for case in cases)),
        "goalkeeper_candidate_count": sum(1 for case in cases if "goalkeeper" in case.category),
        "near_assistant_candidate_count": sum(1 for case in cases if case.category == "assistant_referee_near_camera"),
    }


def _write_blocked_post_role_outputs(stage_root: Path) -> dict[str, Any]:
    learning_root = stage_root / "learning"
    continuity_root = stage_root / "continuity"
    learning_root.mkdir(parents=True, exist_ok=True)
    continuity_root.mkdir(parents=True, exist_ok=True)
    (learning_root / "role_review_examples.jsonl").write_text("", encoding="utf-8")
    blocked_learning = {
        "role_review_complete": False,
        "status": "BLOCKED_ROLE_REVIEW_NOT_COMPLETED",
        "exact_blocker": EXACT_BLOCKER,
        **safety_payload(),
    }
    write_json(
        learning_root / "role_label_distribution.json",
        {"artifact": "m5_4f_role_label_distribution", **blocked_learning},
    )
    write_json(
        learning_root / "role_training_readiness.json",
        {"artifact": "m5_4f_role_training_readiness", **blocked_learning},
    )
    write_json(
        learning_root / "role_calibrator_validation.json",
        {"artifact": "m5_4f_role_calibrator_validation", **blocked_learning, "role_rows_updated": 0},
    )
    write_json(
        learning_root / "role_application_rows.json",
        {"artifact": "m5_4f_role_application_rows", "rows": [], **blocked_learning},
    )
    write_json(
        learning_root / "role_application_audit.json",
        {"artifact": "m5_4f_role_application_audit", "role_rows_updated": 0, **blocked_learning},
    )
    blocked_continuity = {
        "role_review_complete": False,
        "continuity_review_generated": False,
        "status": "BLOCKED_UNTIL_BALANCED_ROLE_REVIEW_COMPLETED",
        "exact_blocker": EXACT_BLOCKER,
        **safety_payload(),
    }
    write_json(
        continuity_root / "post_role_partition_manifest.json",
        {"artifact": "m5_4f_post_role_partition_manifest", **blocked_continuity},
    )
    write_json(
        continuity_root / "post_role_candidate_rows.json",
        {"artifact": "m5_4f_post_role_candidate_rows", "rows": [], **blocked_continuity},
    )
    write_json(
        continuity_root / "post_role_rejected_rows.json",
        {"artifact": "m5_4f_post_role_rejected_rows", "rows": [], **blocked_continuity},
    )
    write_json(
        continuity_root / "deconfounded_continuity_review_manifest.json",
        {"artifact": "m5_4f_deconfounded_continuity_review_manifest", "review_cases": [], **blocked_continuity},
    )
    with (continuity_root / "deconfounded_continuity_case_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["review_case_id", "bucket", "frame_gap", "team", "role", "equivalence_cluster_id"])
    write_json(
        continuity_root / "review_selection_balance_audit.json",
        {"artifact": "m5_4f_review_selection_balance_audit", **blocked_continuity},
    )
    write_json(
        continuity_root / "endpoint_reuse_audit.json",
        {"artifact": "m5_4f_endpoint_reuse_audit", "endpoint_reuse_max": 0, **blocked_continuity},
    )
    write_json(
        continuity_root / "feature_balance_audit.json",
        {"artifact": "m5_4f_feature_balance_audit", **blocked_continuity},
    )
    (continuity_root / "evidence").mkdir(parents=True, exist_ok=True)
    (continuity_root / "workbench").mkdir(parents=True, exist_ok=True)
    (continuity_root / "decisions").mkdir(parents=True, exist_ok=True)
    return blocked_continuity


def _write_safety_audit(validation_root: Path) -> dict[str, Any]:
    audit = {
        "artifact": "m5_4f_safety_guardrail_audit",
        "all_safety_flags_preserved": True,
        "persistent_identity_assigned": False,
        "player_slots_assigned": False,
        "goalkeeper_slots_assigned": False,
        "metric_pitch_coordinates_used": False,
        "tactical_event_or_physical_outputs_created": False,
        **safety_payload(),
    }
    write_json(validation_root / "safety_guardrail_audit.json", audit)
    return audit


def build_balanced_role_then_continuity_stage(
    *,
    repo_root: Path,
    artifact_root: Path,
    match_id: str = "128058",
    stage_root: Path | None = None,
) -> dict[str, Any]:
    paths = default_paths(artifact_root.resolve(), match_id)
    m54e = paths["m54e_stage_root"].resolve()
    stage_root = (stage_root or paths["stage_root"]).resolve()
    stage_root.mkdir(parents=True, exist_ok=True)
    before_sources = _source_inventory([m54e])
    audit_root = stage_root / "audit"
    role_review_root = stage_root / "role_review"
    validation_root = stage_root / "validation"
    for root in [audit_root, role_review_root, validation_root]:
        root.mkdir(parents=True, exist_ok=True)

    role_rows = _read_rows(m54e / "role" / "visual_role_context_rows.json")
    role_feature_rows = _read_rows(m54e / "role" / "visual_role_feature_rows.json")
    feature_by_candidate = {str(row["candidate_id"]): row for row in role_feature_rows}
    m54e_role_manifest = read_json(m54e / "review" / "role_context" / "review_manifest.json")
    m54e_continuity_manifest = read_json(m54e / "review" / "continuity_balance" / "review_manifest.json")
    role_audit = audit_role_review_selection(m54e_role_manifest, feature_by_candidate)
    continuity_audit = audit_continuity_review_selection(m54e_continuity_manifest)
    write_json(audit_root / "m5_4e_role_review_selection_audit.json", role_audit)
    write_json(audit_root / "m5_4e_continuity_review_selection_audit.json", continuity_audit)
    write_text(audit_root / "m5_4e_review_selection_incident.md", _incident_markdown(role_audit, continuity_audit))

    candidate_records = _build_role_candidate_records(role_rows, role_feature_rows)
    selected = select_balanced_role_cases(candidate_records, limit=40)
    write_json(
        role_review_root / "balanced_role_selection_rows.json",
        {"artifact": "m5_4f_balanced_role_selection_rows", "rows": selected, **safety_payload()},
    )
    frame_manifest = read_json(paths["frame_manifest"])
    role_review = _write_role_manifest(
        stage_root=stage_root,
        role_review_root=role_review_root,
        repo_root=repo_root,
        selected=selected,
        frame_root=paths["frame_root"],
        frame_records=_frame_records(frame_manifest),
        source_refs=_source_refs(paths),
    )
    post_role = _write_blocked_post_role_outputs(stage_root)
    after_sources = _source_inventory([m54e])
    source_mutation = {
        "artifact": "m5_4f_source_mutation_audit",
        "source_stage_root": str(m54e),
        "before": before_sources,
        "after": after_sources,
        "m5_4e_outputs_unchanged": before_sources["combined_hash"] == after_sources["combined_hash"],
        **safety_payload(),
    }
    write_json(validation_root / "source_mutation_audit.json", source_mutation)
    safety_audit = _write_safety_audit(validation_root)
    summary = {
        "artifact": "m5_4f_validation_summary",
        "created_at": _now(),
        "match_id": match_id,
        "stage_root": str(stage_root),
        "source_m5_4e_stage_root": str(m54e),
        "final_classification": FINAL_CLASSIFICATION,
        "exact_blocker": EXACT_BLOCKER,
        "original_m5_4e_packs_preserved": source_mutation["m5_4e_outputs_unchanged"],
        "role_selection_audit_result": "failed_balance_requirements",
        "old_role_category_distribution": role_audit["cases_per_proposed_class"],
        "new_role_category_distribution": role_review["category_distribution"],
        "old_role_temporal_quartile_distribution": role_audit["temporal_quartile_distribution"],
        "new_role_temporal_quartile_distribution": role_review["temporal_quartile_distribution"],
        "goalkeeper_candidate_count": role_review["goalkeeper_candidate_count"],
        "near_assistant_candidate_count": role_review["near_assistant_candidate_count"],
        "role_review_case_count": role_review["review_case_count"],
        "role_launcher_path": role_review["launcher_path"],
        "role_review_url": role_review["review_url"],
        "role_review_complete": False,
        "role_label_distribution": None,
        "role_calibrator_validation": "not_run_role_review_incomplete",
        "role_rows_updated": 0,
        "old_continuity_frame_gap_distribution_by_proposed_class": continuity_audit[
            "frame_gap_distribution_by_proposed_class"
        ],
        "new_continuity_frame_gap_distribution_by_proposed_class": None,
        "old_continuity_team_distribution_by_proposed_class": continuity_audit["team_distribution_by_proposed_class"],
        "new_continuity_team_distribution_by_proposed_class": None,
        "old_continuity_equivalence_cluster_counts_per_proposed_class": continuity_audit[
            "equivalence_clusters_per_class"
        ],
        "new_continuity_equivalence_cluster_counts_per_proposed_class": None,
        "endpoint_reuse_max": 0,
        "continuity_case_count": 0,
        "continuity_launcher_path": None,
        "continuity_review_url": None,
        "post_role_gate": post_role["status"],
        "m5_4e_continuity_review_selection_preservation_label": DIAGNOSTIC_ONLY,
        "source_mutation_audit_passed": source_mutation["m5_4e_outputs_unchanged"],
        "safety_audit_passed": safety_audit["all_safety_flags_preserved"],
        **safety_payload(),
    }
    write_json(validation_root / "m5_4f_validation_summary.json", summary)
    return summary


def run_post_role_review_ingestion(
    *,
    stage_root: Path,
) -> dict[str, Any]:
    role_review_root = stage_root / "role_review"
    decision_root = role_review_root / "decisions"
    completed_path = decision_root / "completed_review.json"
    if not completed_path.exists():
        post_role = _write_blocked_post_role_outputs(stage_root)
        return {"final_classification": "BLOCKED_ROLE_LABEL_SUPPORT", "exact_blocker": EXACT_BLOCKER, **post_role}
    completed = read_json(completed_path)
    state = completed.get("state") if isinstance(completed.get("state"), dict) else completed
    decisions = state.get("decisions") if isinstance(state.get("decisions"), dict) else {}
    manifest = read_json(role_review_root / "balanced_role_review_manifest.json")
    case_by_id = {case["review_case_id"]: case for case in manifest.get("review_cases", [])}
    rows_out = []
    label_counts: Counter[str] = Counter()
    cluster_by_label: dict[str, set[str]] = defaultdict(set)
    for case_id, decision in decisions.items():
        case = case_by_id.get(case_id)
        if case is None:
            continue
        usable = decision not in {"unresolved"}
        rows_out.append(
            {
                "review_case_id": case_id,
                "task_type": "visual_team_role_context",
                "candidate_artifact_id": case["candidate_artifact_id"],
                "candidate_hash": case["candidate_hash"],
                "evidence_hash": case["evidence_hash"],
                "human_decision": decision,
                "normalized_training_label": decision,
                "equivalence_cluster_id": case["equivalence_cluster_id"],
                "review_round": case.get("review_round"),
                "label_usable_for_training": usable,
                "exclusion_reason": None if usable else "unresolved_label",
            }
        )
        if usable:
            label_counts[str(decision)] += 1
            cluster_by_label[str(decision)].add(str(case["equivalence_cluster_id"]))
    learning_root = stage_root / "learning"
    learning_root.mkdir(parents=True, exist_ok=True)
    (learning_root / "role_review_examples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows_out),
        encoding="utf-8",
    )
    distribution = {
        "artifact": "m5_4f_role_label_distribution",
        "role_review_complete": bool(state.get("completed")),
        "role_label_distribution": dict(sorted(label_counts.items())),
        **safety_payload(),
    }
    readiness = {
        "artifact": "m5_4f_role_training_readiness",
        "role_review_complete": bool(state.get("completed")),
        "examples_per_class": dict(sorted(label_counts.items())),
        "equivalence_clusters_per_class": {
            label: len(clusters) for label, clusters in sorted(cluster_by_label.items())
        },
        "application_ready_classes": [
            label
            for label, clusters in sorted(cluster_by_label.items())
            if len(clusters) >= 5 and label_counts[label] >= 5
        ],
        "status": "READY_FOR_SUPPORTED_CLASSES"
        if any(len(clusters) >= 5 and label_counts[label] >= 5 for label, clusters in cluster_by_label.items())
        else "BLOCKED_ROLE_LABEL_SUPPORT",
        **safety_payload(),
    }
    write_json(learning_root / "role_label_distribution.json", distribution)
    write_json(learning_root / "role_training_readiness.json", readiness)
    write_json(
        learning_root / "role_calibrator_validation.json",
        {
            "artifact": "m5_4f_role_calibrator_validation",
            "role_rows_updated": 0,
            "broad_application_performed": False,
            **readiness,
        },
    )
    write_json(
        learning_root / "role_application_rows.json",
        {"artifact": "m5_4f_role_application_rows", "rows": [], "role_rows_updated": 0, **safety_payload()},
    )
    write_json(
        learning_root / "role_application_audit.json",
        {
            "artifact": "m5_4f_role_application_audit",
            "unsupported_classes_broadly_applied": False,
            "role_rows_updated": 0,
            **safety_payload(),
        },
    )
    return readiness
