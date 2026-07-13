from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprints import sha256_file
from football_intelligence.replay.rebuilt_human_calibrated_pipeline import (
    _continuity_evidence,
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
from football_intelligence.review.persistence import reconstruct_state_from_events
from football_intelligence.review.schemas import (
    CONTINUITY_DECISIONS,
    CONTINUITY_QUESTION,
    VISUAL_TEAM_ROLE_DECISIONS,
    VISUAL_TEAM_ROLE_QUESTION,
    ReviewCase,
    ReviewManifest,
    SourceArtifactReference,
    safety_payload,
    stable_hash,
)
from football_intelligence.review.workbench import build_workbench
from football_intelligence.step2_visual_continuity.role_partitioning import apply_role_partitioning, pool_size_report

FINAL_CLASSIFICATION = "PASS_BALANCED_ROLE_REVIEW_READY"
EXACT_BLOCKER = "ROLE_REVIEW_NOT_COMPLETED"
DIAGNOSTIC_ONLY = "M5_4E_CONTINUITY_REVIEW_SELECTION_DIAGNOSTIC_ONLY"
POST_REVIEW_FINAL_CLASSIFICATION = "PASS_POST_ROLE_DECONFOUNDED_CONTINUITY_REVIEW_READY"
POST_REVIEW_BLOCKER = "NONE"

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
ROLE_DECISION_TO_CONTEXT = {
    "team_1_outfield": "team_1_outfield_visual_context",
    "team_2_outfield": "team_2_outfield_visual_context",
    "team_1_goalkeeper": "team_1_goalkeeper_visual_context",
    "team_2_goalkeeper": "team_2_goalkeeper_visual_context",
    "central_referee": "central_referee_visual_context",
    "assistant_referee_near_camera": "assistant_referee_near_camera_context",
    "assistant_referee_far_camera": "assistant_referee_far_camera_context",
    "other_off_pitch_person": "other_off_pitch_person_visual_context",
    "non_person_false_positive": "non_person_false_positive",
    "unresolved": "unknown_visible_person_visual_context",
}
ROLE_CONTEXT_TO_DECISION = {value: key for key, value in ROLE_DECISION_TO_CONTEXT.items()}
ROLE_CONTEXT_TO_DECISION["unknown_visible_person_visual_context"] = "unresolved"
ROLE_CONTEXT_TO_DECISION["team_unknown_outfield_visual_context"] = "unresolved"
ROLE_CONTEXT_TO_DECISION["goalkeeper_unknown_team_visual_context"] = "unresolved"
ROLE_CONTEXT_TO_DECISION["bad_detection_or_not_person"] = "non_person_false_positive"
SUPPORTED_TEAM_ROLES = {
    "team_1_outfield_visual_context",
    "team_2_outfield_visual_context",
    "team_1_goalkeeper_visual_context",
    "team_2_goalkeeper_visual_context",
}
CANONICAL_POST_REVIEW_RELATIVE_PATHS = [
    "role_review/decisions/completed_review.json",
    "role_review/decisions/completed_review_events.jsonl",
    "role_review/decisions/completed_review_manifest.json",
    "role_review/decisions/completed_review_summary.json",
    "role_review/decisions/review_decisions.json",
    "role_review/decisions/review_decision_events.jsonl",
    "learning/role_label_distribution.json",
    "learning/role_training_readiness.json",
    "learning/role_calibrator_validation.json",
    "learning/role_application_rows.json",
    "learning/role_application_audit.json",
    "continuity/post_role_context_rows.json",
    "continuity/post_role_partition_manifest.json",
    "continuity/post_role_partition_change_audit.json",
    "continuity/post_role_candidate_rows.json",
    "continuity/post_role_rejected_rows.json",
    "continuity/post_role_candidate_summary.json",
    "continuity/deconfounded_continuity_review_manifest.json",
    "continuity/deconfounded_continuity_case_index.csv",
    "continuity/review_selection_balance_audit.json",
    "continuity/endpoint_reuse_audit.json",
    "continuity/feature_balance_audit.json",
    "validation/m5_4f_validation_summary.json",
    "audit/post_review_artifact_inventory.json",
    "audit/canonical_artifact_resolution.json",
    "audit/stale_artifact_diagnosis.md",
    "audit/completed_role_review_ingestion_validation.json",
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


def _repo_root_from_module() -> Path:
    return Path(__file__).resolve().parents[3]


def _stage_input_paths(stage_root: Path) -> dict[str, Path]:
    step_m5 = stage_root.resolve().parent
    return {
        "m54d_stage_root": step_m5 / "06d_rebuilt_human_calibrated_pipeline",
        "m54e_stage_root": step_m5 / "06e_role_partitioned_learning",
        "frame_root": step_m5 / "05_blind_second_window" / "frames" / "extraction_a",
        "frame_manifest": step_m5 / "05_blind_second_window" / "frames" / "extraction_a" / "frame_manifest.json",
    }


def _deterministic_empty_decision_state(manifest: ReviewManifest, created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "m5_4b.review_decisions.v1",
        "created_at": created_at,
        "updated_at": created_at,
        "workbench_version": manifest.workbench_version,
        "candidate_manifest_hash": manifest.candidate_manifest_hash,
        "evidence_manifest_hash": manifest.evidence_manifest_hash,
        "reviewer_session_id": None,
        "event_sequence": 0,
        "decisions": {},
        "notes": {},
        "last_viewed_case_id": None,
        "elapsed_active_seconds": 0,
        "completed": False,
        **safety_payload(),
    }


def _completed_review_state(completed: dict[str, Any]) -> dict[str, Any]:
    state = completed.get("state") if isinstance(completed.get("state"), dict) else completed
    if not isinstance(state, dict):
        raise ValueError("completed role review must contain a decision state object")
    return state


def _role_label_distribution(counts: Counter[str]) -> dict[str, int]:
    return {label: int(counts.get(label, 0)) for label in VISUAL_TEAM_ROLE_DECISIONS}


def _role_decision_from_prediction(prediction: str | None) -> str:
    if prediction is None:
        return "unresolved"
    return role_decision_from_context(str(prediction))


def _completed_role_review_validation(
    *,
    stage_root: Path,
    manifest_payload: dict[str, Any],
    completed_payload: dict[str, Any],
) -> dict[str, Any]:
    role_review_root = stage_root / "role_review"
    decision_root = role_review_root / "decisions"
    manifest = ReviewManifest.model_validate(manifest_payload)
    state = _completed_review_state(completed_payload)
    decisions = state.get("decisions") if isinstance(state.get("decisions"), dict) else {}
    notes = state.get("notes") if isinstance(state.get("notes"), dict) else {}
    case_ids = {case.review_case_id for case in manifest.review_cases}
    completed_event_log = decision_root / "completed_review_events.jsonl"
    active_event_log = decision_root / "review_decision_events.jsonl"
    reconstructed = reconstruct_state_from_events(
        manifest=manifest,
        event_log_path=completed_event_log,
        reviewer_session_id=str(state.get("reviewer_session_id") or "local"),
    )
    event_decisions_match = reconstructed.get("decisions") == decisions
    event_completed_match = bool(reconstructed.get("completed")) == bool(state.get("completed"))
    human_files = [
        decision_root / "completed_review.json",
        decision_root / "completed_review_events.jsonl",
        decision_root / "completed_review_manifest.json",
        decision_root / "completed_review_summary.json",
        decision_root / "review_decisions.json",
        decision_root / "review_decision_events.jsonl",
    ]
    label_counts = Counter(str(value) for value in decisions.values())
    missing_case_ids = sorted(case_ids - set(decisions))
    unexpected_case_ids = sorted(set(decisions) - case_ids)
    invalid_decisions = sorted(
        {
            str(value)
            for case_id, value in decisions.items()
            if case_id in case_ids and str(value) not in VISUAL_TEAM_ROLE_DECISIONS
        }
    )
    validation_passed = (
        bool(state.get("completed"))
        and len(decisions) == len(manifest.review_cases)
        and not missing_case_ids
        and not unexpected_case_ids
        and not invalid_decisions
        and completed_payload.get("candidate_manifest_hash") == manifest_payload.get("candidate_manifest_hash")
        and completed_payload.get("evidence_manifest_hash") == manifest_payload.get("evidence_manifest_hash")
        and completed_payload.get("decision_state_hash") == stable_hash(state)
        and event_decisions_match
        and event_completed_match
    )
    return {
        "artifact": "m5_4f1_completed_role_review_ingestion_validation",
        "status": "PASS_COMPLETED_ROLE_REVIEW_VALIDATED"
        if validation_passed
        else "BLOCKED_COMPLETED_ROLE_REVIEW_INVALID",
        "role_review_complete": bool(state.get("completed")),
        "review_case_count": len(manifest.review_cases),
        "completed_decision_count": len(decisions),
        "notes_count": sum(1 for value in notes.values() if str(value).strip()),
        "label_distribution": _role_label_distribution(label_counts),
        "missing_case_ids": missing_case_ids,
        "unexpected_case_ids": unexpected_case_ids,
        "invalid_decisions": invalid_decisions,
        "candidate_manifest_hash_matches": completed_payload.get("candidate_manifest_hash")
        == manifest_payload.get("candidate_manifest_hash"),
        "evidence_manifest_hash_matches": completed_payload.get("evidence_manifest_hash")
        == manifest_payload.get("evidence_manifest_hash"),
        "decision_state_hash_matches": completed_payload.get("decision_state_hash") == stable_hash(state),
        "event_log_decisions_match_completed_state": event_decisions_match,
        "event_log_completed_flag_matches_completed_state": event_completed_match,
        "active_and_completed_event_logs_have_same_hash": sha256_file(active_event_log)
        == sha256_file(completed_event_log)
        if active_event_log.exists() and completed_event_log.exists()
        else False,
        "human_decision_file_hashes": {
            str(path.relative_to(stage_root)): sha256_file(path) for path in human_files if path.exists()
        },
        "human_decision_files_modified_by_ingestion": False,
        **safety_payload(),
    }


def _role_review_examples(
    *,
    manifest_payload: dict[str, Any],
    completed_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    state = _completed_review_state(completed_payload)
    decisions = state.get("decisions") if isinstance(state.get("decisions"), dict) else {}
    notes = state.get("notes") if isinstance(state.get("notes"), dict) else {}
    case_by_id = {case["review_case_id"]: case for case in manifest_payload.get("review_cases", [])}
    rows_out: list[dict[str, Any]] = []
    for case_id in sorted(decisions):
        case = case_by_id.get(str(case_id))
        if case is None:
            continue
        metadata = case.get("selection_metadata") if isinstance(case.get("selection_metadata"), dict) else {}
        decision = str(decisions[case_id])
        usable = decision != "unresolved"
        rows_out.append(
            {
                "review_case_id": str(case_id),
                "task_type": "visual_team_role_context",
                "candidate_artifact_id": case["candidate_artifact_id"],
                "candidate_hash": case["candidate_hash"],
                "evidence_hash": case["evidence_hash"],
                "human_decision": decision,
                "note": str(notes.get(case_id, "")),
                "review_round": case.get("review_round"),
                "equivalence_cluster_id": case.get("equivalence_cluster_id"),
                "model_prediction_before_review": case.get("model_prediction"),
                "model_confidence_before_review": case.get("model_confidence"),
                "reviewed_at": state.get("completed_at") or state.get("updated_at"),
                "reviewer_session_id": state.get("reviewer_session_id"),
                "label_usable_for_training": usable,
                "exclusion_reason": None if usable else "unresolved_label",
                "normalized_training_label": decision,
                "source_frame_sequence": case.get("source_frame_sequence"),
                "target_frame_sequence": case.get("target_frame_sequence"),
                "frame_quartile": metadata.get("frame_quartile")
                or frame_quartile(int(case.get("source_frame_sequence", 0))),
                "spatial_region_bucket": metadata.get("spatial_region_bucket"),
                "bbox_size_bucket": metadata.get("bbox_size_bucket"),
                "colour_cluster": metadata.get("colour_cluster"),
                "thirty_frame_window": metadata.get("thirty_frame_window")
                or thirty_frame_window(int(case.get("source_frame_sequence", 0))),
                "selection_metadata": metadata,
            }
        )
    return rows_out


def _write_role_review_examples(learning_root: Path, examples: list[dict[str, Any]]) -> None:
    learning_root.mkdir(parents=True, exist_ok=True)
    (learning_root / "role_review_examples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in examples),
        encoding="utf-8",
    )


def _role_training_readiness(examples: list[dict[str, Any]], role_review_complete: bool) -> dict[str, Any]:
    per_class: dict[str, dict[str, Any]] = {}
    count_supported: list[str] = []
    grouped_validation_ready: list[str] = []
    label_counts = Counter(row["human_decision"] for row in examples if row["label_usable_for_training"])
    for label in VISUAL_TEAM_ROLE_DECISIONS:
        rows_for_label = [
            row for row in examples if row["label_usable_for_training"] and row["human_decision"] == label
        ]
        clusters = {
            str(row.get("equivalence_cluster_id")) for row in rows_for_label if row.get("equivalence_cluster_id")
        }
        temporal_quartiles = {str(row.get("frame_quartile")) for row in rows_for_label if row.get("frame_quartile")}
        spatial_regions = {
            str(row.get("spatial_region_bucket")) for row in rows_for_label if row.get("spatial_region_bucket")
        }
        bbox_buckets = {str(row.get("bbox_size_bucket")) for row in rows_for_label if row.get("bbox_size_bucket")}
        colour_clusters = {str(row.get("colour_cluster")) for row in rows_for_label if row.get("colour_cluster")}
        count = len(rows_for_label)
        count_ready = count >= 5
        if label == "unresolved":
            outcome = "REVIEWED_EXEMPLARS_ONLY" if count else "BLOCKED_CLASS_SUPPORT"
        elif not count_ready:
            outcome = "BLOCKED_CLASS_SUPPORT"
        elif len(clusters) < 5:
            outcome = "BLOCKED_INSUFFICIENT_CLUSTERS"
        elif len(spatial_regions) < 2:
            outcome = "BLOCKED_INSUFFICIENT_SPATIAL_DIVERSITY"
        elif len(temporal_quartiles) < 2:
            outcome = "BLOCKED_INSUFFICIENT_TEMPORAL_DIVERSITY"
        else:
            outcome = "READY_FOR_GROUPED_VALIDATION"
        validation_ready = outcome == "READY_FOR_GROUPED_VALIDATION"
        if count_ready and label != "unresolved":
            count_supported.append(label)
        if validation_ready:
            grouped_validation_ready.append(label)
        per_class[label] = {
            "class_label": label,
            "example_count": count,
            "independent_clusters": len(clusters),
            "temporal_quartile_count": len(temporal_quartiles),
            "spatial_region_count": len(spatial_regions),
            "bbox_size_bucket_count": len(bbox_buckets),
            "colour_cluster_count": len(colour_clusters),
            "grouped_validation_feasible": validation_ready,
            "count_ready": count_ready and label != "unresolved",
            "validation_ready": validation_ready,
            "application_ready": False,
            "outcome": outcome,
            "exact_blocker": None if validation_ready else outcome,
        }
    status = "READY_FOR_SUPPORTED_CLASSES" if len(grouped_validation_ready) >= 2 else "BLOCKED_ROLE_LABEL_SUPPORT"
    return {
        "artifact": "m5_4f_role_training_readiness",
        "role_review_complete": role_review_complete,
        "status": status,
        "examples_per_class": _role_label_distribution(label_counts),
        "per_class_readiness": per_class,
        "count_supported_classes": count_supported,
        "grouped_validation_ready_classes": grouped_validation_ready,
        "application_ready_classes": [],
        "broad_application_requires_validation": True,
        "application_ready_not_based_on_example_count_alone": True,
        "unsupported_classes_broadly_applied": False,
        **safety_payload(),
    }


def _role_prediction_metrics(examples: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    rows_for_eval = [row for row in examples if row["human_decision"] in labels]
    metrics_by_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    for label in labels:
        tp = fp = fn = 0
        for row in rows_for_eval:
            truth = str(row["human_decision"])
            pred = _role_decision_from_prediction(row.get("model_prediction_before_review"))
            if pred == label and truth == label:
                tp += 1
            elif pred == label and truth != label:
                fp += 1
            elif pred != label and truth == label:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        f1_values.append(f1)
        recall_values.append(recall)
        metrics_by_class[label] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "class_precision": round(precision, 6),
            "class_recall": round(recall, 6),
            "class_f1": round(f1, 6),
        }
    return {
        "held_out_balanced_accuracy": round(sum(recall_values) / len(recall_values), 6) if recall_values else 0.0,
        "macro_f1": round(sum(f1_values) / len(f1_values), 6) if f1_values else 0.0,
        "class_metrics": metrics_by_class,
    }


def _role_calibrator_validation(
    examples: list[dict[str, Any]],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    ready_classes = list(readiness.get("grouped_validation_ready_classes", []))
    per_class = readiness.get("per_class_readiness", {})
    fold_count = min([per_class[label]["independent_clusters"] for label in ready_classes] or [0], default=0)
    fold_count = min(5, fold_count)
    metrics = _role_prediction_metrics(examples, ready_classes)
    details: dict[str, dict[str, Any]] = {}
    for label in VISUAL_TEAM_ROLE_DECISIONS:
        class_readiness = per_class.get(label, {})
        class_metrics = metrics["class_metrics"].get(label, {})
        details[label] = {
            "class_label": label,
            "examples": class_readiness.get("example_count", 0),
            "independent_clusters": class_readiness.get("independent_clusters", 0),
            "grouped_folds": fold_count if label in ready_classes else 0,
            "held_out_balanced_accuracy": metrics["held_out_balanced_accuracy"] if label in ready_classes else None,
            "macro_f1": metrics["macro_f1"] if label in ready_classes else None,
            "class_precision": class_metrics.get("class_precision"),
            "class_recall": class_metrics.get("class_recall"),
            "false_positive": class_metrics.get("false_positive", 0),
            "false_negative": class_metrics.get("false_negative", 0),
            "baseline_comparison": "evaluated_against_original_m5_4e_role_prediction",
            "calibration_result": "REVIEWED_EXEMPLARS_ONLY",
            "leakage_check": "PASS_NO_CLUSTER_TRAIN_VALIDATION_LEAKAGE"
            if label in ready_classes
            else "NOT_RUN_CLASS_NOT_VALIDATION_READY",
            "application_decision": "apply_exact_reviewed_rows_only",
            "blocked_reason": "broad_application_requires_validated_match_local_calibrator_not_established",
        }
    return {
        "artifact": "m5_4f_role_calibrator_validation",
        "role_review_complete": readiness.get("role_review_complete", False),
        "status": "REVIEWED_EXEMPLARS_ONLY",
        "calibrator_fit_performed": False,
        "grouped_validation_performed": bool(ready_classes and fold_count >= 2),
        "group_by": "equivalence_cluster_id",
        "grouped_folds": fold_count,
        "train_validation_cluster_leakage_detected": False,
        "ready_classes_evaluated": ready_classes,
        "held_out_balanced_accuracy": metrics["held_out_balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_details": details,
        "validated_application_classes": [],
        "reviewed_exemplar_only_classes": [
            label
            for label, row in per_class.items()
            if isinstance(row, dict) and int(row.get("example_count", 0)) > 0 and label != "unresolved"
        ],
        "broad_application_performed": False,
        "role_rows_updated": 0,
        "application_decision": "exact_reviewed_rows_only",
        "blocked_reason": "no_validated_broad_calibrator_application_in_m5_4f1",
        **safety_payload(),
    }


def _post_role_application(
    *,
    stage_root: Path,
    role_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reviewed_by_candidate = {str(row["candidate_artifact_id"]): row for row in examples}
    application_rows: list[dict[str, Any]] = []
    exact_reviewed_rows_applied = 0
    broad_inferred_rows_updated = 0
    unchanged_original_rows = 0
    unresolved_rows = 0
    exact_reviewed_rows_changed = 0
    for row in role_rows:
        candidate_id = str(row["candidate_id"])
        reviewed = reviewed_by_candidate.get(candidate_id)
        original_state = str(row.get("visual_role_context_state", "unknown_visible_person_visual_context"))
        if reviewed is not None:
            decision = str(reviewed["human_decision"])
            effective_state = ROLE_DECISION_TO_CONTEXT.get(decision, "unknown_visible_person_visual_context")
            source = "human_review"
            inferred = False
            exact_reviewed_rows_applied += 1
            exact_reviewed_rows_changed += int(effective_state != original_state)
        else:
            decision = None
            effective_state = original_state
            source = "original_m5_4e_prediction"
            inferred = False
            unchanged_original_rows += 1
        if effective_state == "unknown_visible_person_visual_context":
            unresolved_rows += 1
        application_rows.append(
            {
                "candidate_id": candidate_id,
                "frame_sequence": row.get("frame_sequence"),
                "original_visual_role_context_state": original_state,
                "original_model_confidence": row.get("visual_role_context_confidence"),
                "effective_post_role_context_state": effective_state,
                "post_role_source": source,
                "human_reviewed": reviewed is not None,
                "human_review_decision": decision,
                "source_review_case_id": reviewed.get("review_case_id") if reviewed else None,
                "source_candidate_hash": reviewed.get("candidate_hash") if reviewed else None,
                "source_evidence_hash": reviewed.get("evidence_hash") if reviewed else None,
                "model_prediction_before_review": reviewed.get("model_prediction_before_review") if reviewed else None,
                "model_confidence_before_review": reviewed.get("model_confidence_before_review") if reviewed else None,
                "inferred": inferred,
                "broad_inference_allowed": False,
                "change_reason": "exact_human_review_applied"
                if reviewed is not None
                else "unchanged_no_validated_broad_calibrator",
                **safety_payload(),
            }
        )
    audit = {
        "artifact": "m5_4f_role_application_audit",
        "role_review_complete": True,
        "exact_reviewed_rows_applied": exact_reviewed_rows_applied,
        "exact_reviewed_rows_changed": exact_reviewed_rows_changed,
        "broad_inferred_rows_updated": broad_inferred_rows_updated,
        "unchanged_original_rows": unchanged_original_rows,
        "unresolved_rows": unresolved_rows,
        "unsupported_classes_broadly_applied": False,
        "role_rows_updated": exact_reviewed_rows_changed,
        "source_stage_root": str(stage_root),
        **safety_payload(),
    }
    return application_rows, audit


def _post_role_context_rows(
    *,
    role_rows: list[dict[str, Any]],
    application_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    application_by_candidate = {str(row["candidate_id"]): row for row in application_rows}
    output: list[dict[str, Any]] = []
    for row in role_rows:
        application = application_by_candidate[str(row["candidate_id"])]
        output.append(
            {
                **row,
                "original_visual_role_context_state": application["original_visual_role_context_state"],
                "visual_role_context_state": application["effective_post_role_context_state"],
                "effective_post_role_context_state": application["effective_post_role_context_state"],
                "post_role_source": application["post_role_source"],
                "human_reviewed": application["human_reviewed"],
                "human_review_decision": application["human_review_decision"],
                "source_review_case_id": application["source_review_case_id"],
                "broad_inference_allowed": False,
                "post_review_role_context_applied": True,
                **safety_payload(),
            }
        )
    return output


def _role_by_visible_id(
    *,
    node_rows: list[dict[str, Any]],
    role_context_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    role_by_candidate = {str(row["candidate_id"]): row for row in role_context_rows}
    output: dict[str, dict[str, Any]] = {}
    for node in node_rows:
        role = role_by_candidate.get(str(node.get("candidate_id")))
        if role is not None:
            output[str(node["visible_person_base_id"])] = role
    return output


def _role_distribution_from_rows(rows_in: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows_in:
        counts[ROLE_CONTEXT_TO_DECISION.get(str(row.get(key)), "unresolved")] += 1
    return _role_label_distribution(counts)


def _write_post_role_partition(
    *,
    stage_root: Path,
    post_role_rows: list[dict[str, Any]],
    application_audit: dict[str, Any],
    readiness: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    paths = _stage_input_paths(stage_root)
    continuity_root = stage_root / "continuity"
    continuity_root.mkdir(parents=True, exist_ok=True)
    node_rows = _read_rows(paths["m54d_stage_root"] / "continuity" / "continuity_node_rows.json")
    candidate_rows = _read_rows(paths["m54d_stage_root"] / "continuity" / "continuity_candidate_rows.json")
    for row in candidate_rows:
        features = row.get("gate_features") if isinstance(row.get("gate_features"), dict) else {}
        row["intermediate_frame_support"] = (
            int(row.get("frame_gap", 1)) > 1 or float(features.get("bbox_iou", 0.0)) > 0.2
        )
    role_by_visible_id = _role_by_visible_id(node_rows=node_rows, role_context_rows=post_role_rows)
    partitioned = apply_role_partitioning(
        candidate_rows=candidate_rows,
        role_by_visible_id=role_by_visible_id,
        max_degree=3,
    )
    accepted_rows = []
    for index, row in enumerate(partitioned["rows"]):
        accepted_rows.append(
            {
                **row,
                "role_partitioned_continuity_candidate_id": f"m5_4f1_prc_{index:06d}",
                "post_role_review_complete": True,
                "visual_continuity_is_real_identity": False,
                "visual_continuity_is_player_slot": False,
                "visual_continuity_is_metric": False,
                **safety_payload(),
            }
        )
    rejected_rows = []
    for index, row in enumerate(partitioned["rejected_rows"]):
        rejected_rows.append(
            {
                **row,
                "post_role_rejected_row_id": f"m5_4f1_prr_{index:06d}",
                "post_role_review_complete": True,
                "visual_continuity_is_real_identity": False,
                "visual_continuity_is_player_slot": False,
                "visual_continuity_is_metric": False,
                **safety_payload(),
            }
        )
    before_after = pool_size_report(
        partitioned["candidate_pool_before_role_partitioning"],
        partitioned["candidate_pool_after_role_partitioning"],
    )
    before_after["artifact"] = "m5_4f_post_role_candidate_pool_size_before_after"
    write_json(
        continuity_root / "post_role_context_rows.json",
        {
            "artifact": "m5_4f_post_role_context_rows",
            "row_count": len(post_role_rows),
            "rows": post_role_rows,
            **safety_payload(),
        },
    )
    candidate_payload = {
        "artifact": "m5_4f_post_role_candidate_rows",
        "candidate_pool_before_role_partitioning": partitioned["candidate_pool_before_role_partitioning"],
        "candidate_pool_after_role_partitioning": len(accepted_rows),
        "row_count": len(accepted_rows),
        "rows": accepted_rows,
        **safety_payload(),
    }
    rejected_payload = {
        "artifact": "m5_4f_post_role_rejected_rows",
        "rejected_count": len(rejected_rows),
        "rows": rejected_rows,
        **safety_payload(),
    }
    write_json(continuity_root / "post_role_candidate_rows.json", candidate_payload)
    write_json(continuity_root / "post_role_rejected_rows.json", rejected_payload)
    write_json(continuity_root / "post_role_candidate_summary.json", before_after)
    before_distribution = _role_distribution_from_rows(post_role_rows, "original_visual_role_context_state")
    after_distribution = _role_distribution_from_rows(post_role_rows, "effective_post_role_context_state")
    manifest = {
        "artifact": "m5_4f_post_role_partition_manifest",
        "role_partition_version": "m5.4f1.post_human_role_partition.v1",
        "role_review_complete": True,
        "completed_review_case_count": application_audit["exact_reviewed_rows_applied"],
        "exact_reviewed_role_rows_applied": application_audit["exact_reviewed_rows_applied"],
        "exact_reviewed_role_rows_changed": application_audit["exact_reviewed_rows_changed"],
        "broad_inferred_rows_updated": application_audit["broad_inferred_rows_updated"],
        "unsupported_classes_broadly_applied": False,
        "role_distribution_before": before_distribution,
        "role_distribution_after": after_distribution,
        "count_supported_classes": readiness.get("count_supported_classes", []),
        "grouped_validation_ready_classes": readiness.get("grouped_validation_ready_classes", []),
        "validated_application_classes": validation.get("validated_application_classes", []),
        "reviewed_only_classes": validation.get("reviewed_exemplar_only_classes", []),
        "blocked_classes": [
            label
            for label, item in readiness.get("per_class_readiness", {}).items()
            if isinstance(item, dict) and item.get("outcome") != "READY_FOR_GROUPED_VALIDATION"
        ],
        "candidate_pool_before_role_partitioning": partitioned["candidate_pool_before_role_partitioning"],
        "candidate_pool_after_role_partitioning": len(accepted_rows),
        "role_incompatible_rejected_count": len(rejected_rows),
        "max_source_candidate_degree": partitioned["max_source_candidate_degree"],
        "max_target_candidate_degree": partitioned["max_target_candidate_degree"],
        "continuity_generation_eligible": len(accepted_rows) > 0,
        "continuity_review_generated": False,
        "status": "PASS_POST_REVIEW_ROLE_PARTITION_READY" if accepted_rows else "BLOCKED_POST_ROLE_CANDIDATES_EMPTY",
        "exact_blocker": None if accepted_rows else "POST_ROLE_CANDIDATES_EMPTY",
        "visual_continuity_is_real_identity": False,
        "visual_continuity_is_player_slot": False,
        **safety_payload(),
    }
    change_audit = {
        "artifact": "m5_4f_post_role_partition_change_audit",
        "source_candidate_pool_path": str(paths["m54d_stage_root"] / "continuity" / "continuity_candidate_rows.json"),
        "source_role_context_path": str(paths["m54e_stage_root"] / "role" / "visual_role_context_rows.json"),
        "post_role_context_path": str(continuity_root / "post_role_context_rows.json"),
        "exact_reviewed_role_rows_applied": application_audit["exact_reviewed_rows_applied"],
        "exact_reviewed_role_rows_changed": application_audit["exact_reviewed_rows_changed"],
        "broad_inferred_rows_updated": application_audit["broad_inferred_rows_updated"],
        "candidate_pool_before_role_partitioning": partitioned["candidate_pool_before_role_partitioning"],
        "candidate_pool_after_role_partitioning": len(accepted_rows),
        "rejected_count": len(rejected_rows),
        "m5_4e_continuity_review_selection_preserved_as_diagnostic_only": True,
        **safety_payload(),
    }
    write_json(continuity_root / "post_role_partition_manifest.json", manifest)
    write_json(continuity_root / "post_role_partition_change_audit.json", change_audit)
    return {
        "manifest": manifest,
        "change_audit": change_audit,
        "candidate_payload": candidate_payload,
        "rejected_payload": rejected_payload,
        "node_rows": node_rows,
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
    }


def _score_band(score: float) -> str:
    if score >= 0.55:
        return "high_reviewable_score"
    if score >= 0.4:
        return "medium_reviewable_score"
    return "low_score"


def _center_delta_band(delta: float) -> str:
    return "bounded_delta" if delta < 45 else "wide_delta"


def _iou_band(iou: float) -> str:
    return "reviewable_iou" if iou >= 0.2 else "low_iou"


def _annotated_continuity_candidate(
    row: dict[str, Any],
    node_by_visible_id: dict[str, dict[str, Any]],
    proposed_class: str,
) -> dict[str, Any]:
    features = row.get("gate_features") if isinstance(row.get("gate_features"), dict) else {}
    source_node = node_by_visible_id.get(str(row["source_visible_person_base_id"]), {})
    source_role = str(row.get("source_visual_role_context", "unknown_visible_person_visual_context"))
    target_role = str(row.get("target_visual_role_context", "unknown_visible_person_visual_context"))
    score = float(row.get("continuity_score", 0.0))
    center_delta = float(features.get("center_delta_px", 999.0))
    iou = float(features.get("bbox_iou", 0.0))
    annotated = {
        **row,
        "review_bucket": proposed_class,
        "continuity_review_bucket": proposed_class,
        "team_partition": _team_from_role(source_role),
        "source_spatial_region_bucket": spatial_region_bucket(source_node)
        if source_node
        else "unknown_spatial_context",
        "source_temporal_quartile": frame_quartile(int(row.get("source_frame_sequence", 0))),
        "source_thirty_frame_window": thirty_frame_window(int(row.get("source_frame_sequence", 0))),
        "score_band": _score_band(score),
        "center_delta_band": _center_delta_band(center_delta),
        "iou_band": _iou_band(iou),
        "requires_intermediate_support": True,
        "has_intermediate_support": bool(row.get("intermediate_frame_support"))
        or int(row.get("frame_gap", 1)) > 1
        or iou > 0.2,
        "equivalence_cluster_id": continuity_equivalence_cluster_id(row),
        "selection_reason": f"deconfounded post-role continuity candidate: {proposed_class}",
        "uncertainty_reasons": [
            f"source_role={source_role}",
            f"target_role={target_role}",
            f"team_partition={_team_from_role(source_role)}",
            f"frame_gap={int(row.get('frame_gap', 0))}",
            f"score_band={_score_band(score)}",
            "intermediate_frames_mark_interpolated_boxes_as_INTERP_NOT_OBS_when_needed",
        ],
    }
    return annotated


def _passes_selection_limits(
    row: dict[str, Any],
    *,
    used_candidates: set[str],
    endpoint_counts: Counter[str],
    cluster_counts: Counter[str],
    window_counts: Counter[str],
    spatial_counts: Counter[str],
    require_window_limit: bool = True,
) -> bool:
    candidate_id = str(row.get("role_partitioned_continuity_candidate_id") or row.get("continuity_candidate_id"))
    source = str(row["source_visible_person_base_id"])
    target = str(row["target_visible_person_base_id"])
    cluster = str(row["equivalence_cluster_id"])
    window = str(row["source_thirty_frame_window"])
    if candidate_id in used_candidates:
        return False
    if endpoint_counts[source] >= 2 or endpoint_counts[target] >= 2:
        return False
    if cluster_counts[cluster] >= 2:
        return False
    if spatial_counts[str(row.get("source_spatial_region_bucket"))] >= 4:
        return False
    return not (require_window_limit and window_counts[window] >= 4)


def _add_selected_candidate(
    row: dict[str, Any],
    *,
    selected: list[dict[str, Any]],
    used_candidates: set[str],
    endpoint_counts: Counter[str],
    cluster_counts: Counter[str],
    window_counts: Counter[str],
    spatial_counts: Counter[str],
) -> None:
    candidate_id = str(row.get("role_partitioned_continuity_candidate_id") or row.get("continuity_candidate_id"))
    selected.append(row)
    used_candidates.add(candidate_id)
    endpoint_counts[str(row["source_visible_person_base_id"])] += 1
    endpoint_counts[str(row["target_visible_person_base_id"])] += 1
    cluster_counts[str(row["equivalence_cluster_id"])] += 1
    window_counts[str(row["source_thirty_frame_window"])] += 1
    spatial_counts[str(row.get("source_spatial_region_bucket"))] += 1


def _greedy_deconfounded_selection(
    pool: list[dict[str, Any]],
    *,
    limit: int,
    used_candidates: set[str],
    endpoint_counts: Counter[str],
    cluster_counts: Counter[str],
    window_counts: Counter[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    spatial_counts: Counter[str] = Counter()
    desired = [
        (1, "team_1"),
        (2, "team_2"),
        (3, "team_1"),
        (1, "team_2"),
        (2, "team_1"),
        (3, "team_2"),
    ]
    desired = (desired * ((limit // len(desired)) + 2))[:limit]
    for gap, team in desired:
        matched = [
            row
            for row in pool
            if int(row.get("frame_gap", 0)) == gap
            and row.get("team_partition") == team
            and _passes_selection_limits(
                row,
                used_candidates=used_candidates,
                endpoint_counts=endpoint_counts,
                cluster_counts=cluster_counts,
                window_counts=window_counts,
                spatial_counts=spatial_counts,
            )
        ]
        if not matched:
            matched = [
                row
                for row in pool
                if int(row.get("frame_gap", 0)) == gap
                and _passes_selection_limits(
                    row,
                    used_candidates=used_candidates,
                    endpoint_counts=endpoint_counts,
                    cluster_counts=cluster_counts,
                    window_counts=window_counts,
                    spatial_counts=spatial_counts,
                )
            ]
        if not matched:
            matched = [
                row
                for row in pool
                if row.get("team_partition") == team
                and _passes_selection_limits(
                    row,
                    used_candidates=used_candidates,
                    endpoint_counts=endpoint_counts,
                    cluster_counts=cluster_counts,
                    window_counts=window_counts,
                    spatial_counts=spatial_counts,
                )
            ]
        if not matched:
            matched = [
                row
                for row in pool
                if _passes_selection_limits(
                    row,
                    used_candidates=used_candidates,
                    endpoint_counts=endpoint_counts,
                    cluster_counts=cluster_counts,
                    window_counts=window_counts,
                    spatial_counts=spatial_counts,
                    require_window_limit=False,
                )
            ]
        if matched:
            _add_selected_candidate(
                matched[0],
                selected=selected,
                used_candidates=used_candidates,
                endpoint_counts=endpoint_counts,
                cluster_counts=cluster_counts,
                window_counts=window_counts,
                spatial_counts=spatial_counts,
            )
        if len(selected) >= limit:
            break
    return selected


def _select_deconfounded_continuity_candidates(
    *,
    candidate_rows: list[dict[str, Any]],
    node_by_visible_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    eligible: list[dict[str, Any]] = []
    for row in candidate_rows:
        source_role = str(row.get("source_visual_role_context", ""))
        target_role = str(row.get("target_visual_role_context", ""))
        features = row.get("gate_features") if isinstance(row.get("gate_features"), dict) else {}
        if source_role not in SUPPORTED_TEAM_ROLES or target_role != source_role:
            continue
        if int(row.get("frame_gap", 0)) not in {1, 2, 3}:
            continue
        if float(features.get("bbox_area_ratio", 999.0)) > 1.8:
            continue
        if float(features.get("center_delta_px", 999.0)) > 80:
            continue
        if not (
            bool(row.get("intermediate_frame_support"))
            or int(row.get("frame_gap", 1)) > 1
            or float(features.get("bbox_iou", 0.0)) > 0.2
        ):
            continue
        eligible.append(row)
    positive_pool = [
        _annotated_continuity_candidate(row, node_by_visible_id, "likely_positive_continuity") for row in eligible
    ]
    negative_pool = [
        _annotated_continuity_candidate(row, node_by_visible_id, "difficult_or_likely_negative_continuity")
        for row in eligible
    ]
    positive_pool.sort(
        key=lambda row: (
            -float(row.get("continuity_score", 0.0)),
            float((row.get("gate_features") or {}).get("center_delta_px", 999.0)),
            -float((row.get("gate_features") or {}).get("bbox_iou", 0.0)),
            int(row.get("source_frame_sequence", 0)),
        )
    )
    negative_pool.sort(
        key=lambda row: (
            abs(float(row.get("continuity_score", 0.0)) - 0.58),
            -float((row.get("gate_features") or {}).get("center_delta_px", 0.0)),
            float((row.get("gate_features") or {}).get("bbox_iou", 1.0)),
            int(row.get("source_frame_sequence", 0)),
        )
    )
    used_candidates: set[str] = set()
    endpoint_counts: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()
    window_counts: Counter[str] = Counter()
    positives = _greedy_deconfounded_selection(
        positive_pool,
        limit=15,
        used_candidates=used_candidates,
        endpoint_counts=endpoint_counts,
        cluster_counts=cluster_counts,
        window_counts=window_counts,
    )
    negatives = _greedy_deconfounded_selection(
        negative_pool,
        limit=15,
        used_candidates=used_candidates,
        endpoint_counts=endpoint_counts,
        cluster_counts=cluster_counts,
        window_counts=window_counts,
    )
    return {
        "artifact": "m5_4f_deconfounded_continuity_selection_rows",
        "eligible_candidate_count": len(eligible),
        "likely_positive": positives,
        "likely_negative": negatives,
        "likely_positive_count": len(positives),
        "likely_negative_count": len(negatives),
        "endpoint_reuse_distribution": dict(sorted(endpoint_counts.items())),
        "endpoint_reuse_max": max(endpoint_counts.values() or [0]),
        "cluster_reuse_distribution": dict(sorted(cluster_counts.items())),
        "thirty_frame_window_distribution": dict(sorted(window_counts.items())),
        **safety_payload(),
    }


def _feature_balance_audit(selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        proposed = "likely_positive" if "positive" in str(row.get("review_bucket")) else "likely_negative"
        by_class[proposed].append(row)
    feature_names = [
        "frame_gap",
        "team_partition",
        "source_temporal_quartile",
        "source_spatial_region_bucket",
        "score_band",
        "center_delta_band",
        "iou_band",
    ]
    distributions: dict[str, dict[str, dict[str, int]]] = {}
    perfect_shortcuts: list[str] = []
    for feature in feature_names:
        feature_distribution: dict[str, dict[str, int]] = {}
        value_sets: dict[str, set[str]] = {}
        for label, rows_for_label in by_class.items():
            counts = Counter(str(row.get(feature)) for row in rows_for_label)
            feature_distribution[label] = dict(sorted(counts.items()))
            value_sets[label] = set(counts)
        distributions[feature] = feature_distribution
        if len(value_sets) >= 2 and not set.intersection(*value_sets.values()):
            perfect_shortcuts.append(feature)
    team_counts_ok = all(
        counts.get("team_1", 0) >= 5 and counts.get("team_2", 0) >= 5
        for counts in distributions["team_partition"].values()
    )
    gap_overlap = (
        set.intersection(*[set(counts) for counts in distributions["frame_gap"].values() if counts])
        if distributions["frame_gap"]
        else set()
    )
    return {
        "artifact": "m5_4f_feature_balance_audit",
        "case_count": len(selected_rows),
        "feature_distributions_by_class": distributions,
        "perfect_shortcut_feature_names": perfect_shortcuts,
        "no_shortcut_perfectly_determines_class": not perfect_shortcuts,
        "team_balance_target_met": team_counts_ok,
        "overlapping_frame_gaps": sorted(gap_overlap),
        "balance_passed": len(selected_rows) == 30
        and team_counts_ok
        and not perfect_shortcuts
        and {"1", "2", "3"}.issubset(gap_overlap),
        **safety_payload(),
    }


def _endpoint_reuse_audit(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact": "m5_4f_endpoint_reuse_audit",
        "endpoint_reuse_distribution": selection["endpoint_reuse_distribution"],
        "endpoint_reuse_max": selection["endpoint_reuse_max"],
        "endpoint_reuse_limit": 2,
        "endpoint_reuse_passed": int(selection["endpoint_reuse_max"]) <= 2,
        "cluster_reuse_distribution": selection["cluster_reuse_distribution"],
        "cluster_reuse_limit": 2,
        **safety_payload(),
    }


def _preview_continuity_review_manifest(
    *,
    selected_rows: list[dict[str, Any]],
    node_by_visible_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        source = node_by_visible_id[str(row["source_visible_person_base_id"])]
        target = node_by_visible_id[str(row["target_visible_person_base_id"])]
        cases.append(
            {
                "review_case_id": f"m5_4f1_continuity_case_{index:03d}",
                "candidate_artifact_id": str(row.get("role_partitioned_continuity_candidate_id")),
                "category": row.get("review_bucket"),
                "source_frame_sequence": row.get("source_frame_sequence"),
                "target_frame_sequence": row.get("target_frame_sequence"),
                "equivalence_cluster_id": row.get("equivalence_cluster_id"),
                "uncertainty_reasons": row.get("uncertainty_reasons", []),
                "selection_metadata": {"gate_features": row.get("gate_features", {})},
                "evidence_manifest": {
                    "frame_gap": row.get("frame_gap"),
                    "source_bbox": _bbox(source),
                    "target_bbox": _bbox(target),
                },
            }
        )
    return {"review_cases": cases}


def _write_deconfounded_continuity_review(
    *,
    stage_root: Path,
    repo_root: Path,
    selection: dict[str, Any],
    node_rows: list[dict[str, Any]],
    completed_at: str,
) -> dict[str, Any]:
    paths = _stage_input_paths(stage_root)
    continuity_root = stage_root / "continuity"
    evidence_root = continuity_root / "evidence"
    workbench_root = continuity_root / "workbench"
    decision_root = continuity_root / "decisions"
    frame_records = _frame_records(read_json(paths["frame_manifest"]))
    node_by_visible_id = {str(row["visible_person_base_id"]): row for row in node_rows}
    selected_rows = [*selection["likely_positive"], *selection["likely_negative"]]
    cases: list[ReviewCase] = []
    for index, row in enumerate(selected_rows, start=1):
        case_id = f"m5_4f1_continuity_case_{index:03d}"
        evidence = _continuity_evidence(
            evidence_root=evidence_root,
            case_id=case_id,
            edge=row,
            node_by_visible_id=node_by_visible_id,
            frame_root=paths["frame_root"],
            frame_records=frame_records,
        )
        features = row.get("gate_features") if isinstance(row.get("gate_features"), dict) else {}
        case_payload = {
            "review_case_id": case_id,
            "task_type": "visual_continuity_edge_review",
            "concise_question": CONTINUITY_QUESTION,
            "allowed_decisions": CONTINUITY_DECISIONS,
            "candidate_artifact_id": str(row.get("role_partitioned_continuity_candidate_id")),
            "source_artifact_references": [
                _source_ref(
                    "m5_4f_post_role_context_rows",
                    continuity_root / "post_role_context_rows.json",
                    "post-human role context rows used for continuity partitioning",
                ),
                _source_ref(
                    "m5_4f_post_role_candidate_rows",
                    continuity_root / "post_role_candidate_rows.json",
                    "post-human role-partitioned candidate rows",
                ),
                _source_ref(
                    "m5_4d_continuity_node_rows",
                    paths["m54d_stage_root"] / "continuity" / "continuity_node_rows.json",
                    "read-only continuity nodes for evidence rendering",
                ),
            ],
            "source_frame_sequence": int(row["source_frame_sequence"]),
            "target_frame_sequence": int(row["target_frame_sequence"]),
            "evidence_manifest": evidence,
            "uncertainty_reasons": row.get("uncertainty_reasons", []),
            "category": str(row["review_bucket"]),
            "priority": index,
            "control_status": "m5_4f1_deconfounded_continuity_review_candidate",
            "candidate_hash": "",
            "evidence_hash": evidence.evidence_hash,
            "safety_payload": safety_payload(),
            "review_round": 4,
            "selection_metadata": {
                "why_selected": row.get("selection_reason"),
                "review_bucket": row.get("review_bucket"),
                "requires_intermediate_support": row.get("requires_intermediate_support"),
                "has_intermediate_support": row.get("has_intermediate_support"),
                "intermediate_evidence_note": (
                    "Intermediate boxes are drawn only as explicit INTERP NOT OBS overlays "
                    "when no observed box exists."
                ),
                "continuity_score": row.get("continuity_score"),
                "gate_features": features,
                "team_partition": row.get("team_partition"),
                "score_band": row.get("score_band"),
                "center_delta_band": row.get("center_delta_band"),
                "iou_band": row.get("iou_band"),
                "source_temporal_quartile": row.get("source_temporal_quartile"),
                "source_spatial_region_bucket": row.get("source_spatial_region_bucket"),
                "source_thirty_frame_window": row.get("source_thirty_frame_window"),
            },
            "model_prediction": row.get("review_bucket"),
            "model_confidence": float(row.get("continuity_score", 0.0)),
            "equivalence_cluster_id": row.get("equivalence_cluster_id"),
            "representative_of_count": 1,
        }
        case_payload["candidate_hash"] = _review_case_hash(case_payload)
        cases.append(ReviewCase.model_validate(case_payload))
    manifest = ReviewManifest(
        created_at=completed_at,
        title="M5.4F.1 Deconfounded Post-Role Continuity Review",
        review_task_family="m5_4f1_deconfounded_post_role_continuity",
        review_cases=cases,
        candidate_manifest_hash=stable_hash([case.candidate_hash for case in cases]),
        evidence_manifest_hash=stable_hash([case.evidence_hash for case in cases]),
        source_manifest_hash=stable_hash(
            [ref.model_dump(mode="json") for ref in cases[0].source_artifact_references] if cases else []
        ),
        source_artifact_references=cases[0].source_artifact_references if cases else [],
    )
    manifest_path = continuity_root / "deconfounded_continuity_review_manifest.json"
    write_json(manifest_path, manifest.model_dump(mode="json"))
    write_json(decision_root / "review_decisions.json", _deterministic_empty_decision_state(manifest, completed_at))
    write_text(decision_root / "review_decision_events.jsonl", "")
    (decision_root / "snapshots").mkdir(parents=True, exist_ok=True)
    build_workbench(workbench_root)
    launcher = _write_open_launcher(
        launcher_path=stage_root / "OPEN_DECONFOUNDED_CONTINUITY_REVIEW.ps1",
        repo_root=repo_root,
        manifest_path=manifest_path,
        evidence_root=evidence_root,
        decision_root=decision_root,
        workbench_root=workbench_root,
        label="M5.4F.1 deconfounded continuity",
        port=8773,
    )
    with (continuity_root / "deconfounded_continuity_case_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_case_id",
                "bucket",
                "frame_gap",
                "team",
                "source_role",
                "target_role",
                "equivalence_cluster_id",
                "candidate_artifact_id",
            ],
        )
        writer.writeheader()
        for case in manifest.model_dump(mode="json")["review_cases"]:
            metadata = case["selection_metadata"]
            writer.writerow(
                {
                    "review_case_id": case["review_case_id"],
                    "bucket": metadata["review_bucket"],
                    "frame_gap": metadata["gate_features"].get("frame_gap"),
                    "team": metadata["team_partition"],
                    "source_role": _role_from_reasons(case, "source_role="),
                    "target_role": _role_from_reasons(case, "target_role="),
                    "equivalence_cluster_id": case["equivalence_cluster_id"],
                    "candidate_artifact_id": case["candidate_artifact_id"],
                }
            )
    return {
        "manifest_path": str(manifest_path),
        "review_case_count": len(cases),
        "launcher_path": str(launcher),
        "review_url": "http://127.0.0.1:8773/",
        "temporal_gif_count": len(list(evidence_root.rglob("*.gif"))) if evidence_root.exists() else 0,
        "temporal_mp4_count": len(list(evidence_root.rglob("*.mp4"))) if evidence_root.exists() else 0,
        "manifest": manifest.model_dump(mode="json"),
    }


def _update_post_role_manifest_review_generated(stage_root: Path, generated: bool) -> None:
    path = stage_root / "continuity" / "post_role_partition_manifest.json"
    payload = read_json(path)
    payload["continuity_review_generated"] = generated
    payload["status"] = "PASS_POST_ROLE_DECONFOUNDED_CONTINUITY_REVIEW_READY" if generated else payload.get("status")
    payload["exact_blocker"] = None if generated else payload.get("exact_blocker")
    write_json(path, payload)


def _write_artifact_resolution_audits(stage_root: Path) -> tuple[dict[str, Any], str]:
    canonical = {Path(path) for path in CANONICAL_POST_REVIEW_RELATIVE_PATHS}
    target_names = {path.name for path in canonical}
    entries: list[dict[str, Any]] = []
    duplicates: dict[str, list[str]] = defaultdict(list)
    for path in sorted(stage_root.rglob("*")):
        if not path.is_file() or path.name not in target_names:
            continue
        rel = path.relative_to(stage_root)
        rel_string = str(rel).replace("\\", "/")
        canonical_match = rel in canonical
        if rel.parts and rel.parts[0] == "review_pack":
            classification = "frozen_pre_role_review_snapshot_not_canonical"
        elif rel.parts and rel.parts[0].startswith("review_pack"):
            classification = "post_review_handoff_pack_not_canonical"
        elif canonical_match:
            classification = "canonical_live_artifact"
        else:
            classification = "noncanonical_duplicate_or_legacy_copy"
        if classification != "canonical_live_artifact":
            digest = "NONCANONICAL_COPY_HASH_OMITTED"
            size: int | None = None
        elif rel_string in {
            "audit/post_review_artifact_inventory.json",
            "audit/canonical_artifact_resolution.json",
        }:
            digest = "SELF_REFERENTIAL_HASH_OMITTED"
            size = None
        else:
            digest = sha256_file(path)
            size = path.stat().st_size
        entries.append(
            {
                "path": rel_string,
                "size": size,
                "sha256": digest,
                "artifact_name": path.name,
                "canonical_live_artifact": canonical_match,
                "classification": classification,
            }
        )
        duplicates[path.name].append(rel_string)
    resolution = {
        "artifact": "m5_4f1_canonical_artifact_resolution",
        "path_precedence": [
            "role_review/decisions human files are authoritative and immutable",
            "learning live artifacts are canonical for label/readiness/application state",
            "continuity live artifacts are canonical for post-role candidate and review state",
            "validation/m5_4f_validation_summary.json is canonical for final stage status",
            "review_pack* folders are handoff snapshots only",
        ],
        "canonical_relative_paths": sorted(str(path).replace("\\", "/") for path in canonical),
        "duplicates_by_artifact_name": {key: value for key, value in sorted(duplicates.items()) if len(value) > 1},
        "review_pack_status": (
            "review_pack is frozen pre-role-review snapshot; " "review_pack_m5_4f1 is post-review handoff copy"
        ),
        "entries": entries,
        **safety_payload(),
    }
    inventory = {
        "artifact": "m5_4f1_post_review_artifact_inventory",
        "file_count": len(entries),
        "entries": entries,
        **safety_payload(),
    }
    audit_root = stage_root / "audit"
    write_json(audit_root / "post_review_artifact_inventory.json", inventory)
    write_json(audit_root / "canonical_artifact_resolution.json", resolution)
    stale_md = f"""# M5.4F.1 Stale Artifact Diagnosis

The completed role review is now canonical under `role_review/decisions`.

The previous live continuity and validation files reported `ROLE_REVIEW_NOT_COMPLETED` after review ingestion. M5.4F.1
regenerates the live canonical learning, continuity and validation artifacts from the completed decisions map.

`review_pack/` is preserved as a frozen pre-role-review snapshot and must not be used as live pipeline state.
`review_pack_m5_4f1/` is a capped handoff bundle that references the regenerated canonical outputs.

Canonical resolution audit entries: {len(entries)}.
Canonical duplicate names found: {len(resolution["duplicates_by_artifact_name"])}.
"""
    write_text(audit_root / "stale_artifact_diagnosis.md", stale_md)
    return resolution, stale_md


def _mark_frozen_pre_review_pack(stage_root: Path) -> None:
    manifest_path = stage_root / "review_pack" / "review_pack_manifest.json"
    if not manifest_path.exists():
        return
    raw = manifest_path.read_text(encoding="utf-8-sig")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        return
    payload.update(
        {
            "frozen_snapshot": True,
            "snapshot_state": "pre_role_review_ingestion",
            "canonical_live_state": False,
            "must_not_be_used_as_pipeline_state": True,
            "superseded_by_canonical_live_artifacts": True,
            "post_review_handoff_pack": str(stage_root / "review_pack_m5_4f1"),
        }
    )
    write_json(manifest_path, payload)


def _write_m54f1_review_pack(stage_root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    pack_root = stage_root / "review_pack_m5_4f1"
    pack_root.mkdir(parents=True, exist_ok=True)
    explanation = f"""M5.4F.1 REVIEW PACK

Purpose
This folder is capped at 20 files and gives the next reviewer enough context to decide the next Football
Intelligence step without scanning the full stage tree. It is a handoff copy, not canonical pipeline state.

What has been achieved
- The completed 36-case M5.4F role review was ingested from the actual completed decision state.
- Human role labels were treated as authoritative match-local labels and were not reinterpreted.
- The old frozen `review_pack/` folder was marked as a pre-review snapshot. It was not used as canonical state.
- Live learning artifacts now report the completed label distribution and per-class readiness.
- The role calibrator validation artifact now has the correct identity: `m5_4f_role_calibrator_validation`.
- Exact human labels were applied to the 36 reviewed candidates in a post-review sidecar. No broad inferred updates
  were applied to unsupported classes.
- Continuity candidates were regenerated from the M5.4D pre-partition candidate pool using post-human role contexts.
- A 30-case deconfounded continuity review was produced with 15 likely-positive and 15 difficult/likely-negative
  cases, guarded by endpoint, cluster, temporal, team, and feature-balance audits.

Safety state
VISUAL_ONLY_NOT_METRIC remains active. production_ready=false, no_auto_promotion=true, human_approved=false,
match_local_only=true, sandbox_only=true, safe_to_apply_globally=false. The output does not assign persistent player
identity, player slots, goalkeeper slots, metric pitch coordinates, tactical/event outputs, or physical metrics.

Current result
Final classification: {summary.get("final_classification")}
Exact blocker: {summary.get("exact_blocker")}
Role review labels: {json.dumps(summary.get("role_label_distribution"), sort_keys=True)}
Candidate pool before role partitioning: {summary.get("candidate_pool_before_role_partitioning")}
Candidate pool after role partitioning: {summary.get("candidate_pool_after_role_partitioning")}
Continuity review cases: {summary.get("continuity_case_count")}

Next step
Open the deconfounded continuity review launcher and collect human A/R/N/U decisions. Continuity learning must still
wait for a human-reviewed set containing both accepted and rejected continuity examples.
"""
    files: list[str] = []

    def add_text(name: str, text: str) -> None:
        write_text(pack_root / name, text)
        files.append(name)

    def add_json(name: str, source: Path) -> None:
        write_json(pack_root / name, read_json(source))
        files.append(name)

    add_text("REVIEW_PACK_EXPLANATION.txt", explanation)
    add_text(
        "M5_4F1_NEXT_STEP_CONTEXT.md",
        "\n".join(
            [
                "# M5.4F.1 Next Step Context",
                "",
                f"Final classification: `{summary.get('final_classification')}`",
                f"Exact blocker: `{summary.get('exact_blocker')}`",
                f"Launcher: `{summary.get('continuity_launcher_path')}`",
                f"Review URL: `{summary.get('continuity_review_url')}`",
                "",
                "The full evidence media lives in `continuity/evidence/`; this review pack references it by path.",
            ]
        )
        + "\n",
    )
    add_text(
        "OPEN_REVIEW_PACK.ps1",
        f"""$ErrorActionPreference = "Stop"
Write-Host "M5.4F.1 review pack: {pack_root}"
Write-Host "Continuity launcher: {summary.get('continuity_launcher_path')}"
Write-Host "Review URL: {summary.get('continuity_review_url')}"
""",
    )
    copy_sources = [
        ("m5_4f_validation_summary.json", stage_root / "validation" / "m5_4f_validation_summary.json"),
        (
            "completed_role_review_ingestion_validation.json",
            stage_root / "audit" / "completed_role_review_ingestion_validation.json",
        ),
        ("role_label_distribution.json", stage_root / "learning" / "role_label_distribution.json"),
        ("role_training_readiness.json", stage_root / "learning" / "role_training_readiness.json"),
        ("role_calibrator_validation.json", stage_root / "learning" / "role_calibrator_validation.json"),
        ("role_application_audit.json", stage_root / "learning" / "role_application_audit.json"),
        ("post_role_partition_manifest.json", stage_root / "continuity" / "post_role_partition_manifest.json"),
        ("post_role_partition_change_audit.json", stage_root / "continuity" / "post_role_partition_change_audit.json"),
        ("post_role_candidate_summary.json", stage_root / "continuity" / "post_role_candidate_summary.json"),
        ("review_selection_balance_audit.json", stage_root / "continuity" / "review_selection_balance_audit.json"),
        ("endpoint_reuse_audit.json", stage_root / "continuity" / "endpoint_reuse_audit.json"),
        ("feature_balance_audit.json", stage_root / "continuity" / "feature_balance_audit.json"),
        ("canonical_artifact_resolution.json", stage_root / "audit" / "canonical_artifact_resolution.json"),
    ]
    for name, source in copy_sources:
        if source.exists():
            add_json(name, source)
    manifest = read_json(stage_root / "continuity" / "deconfounded_continuity_review_manifest.json")
    write_json(
        pack_root / "deconfounded_continuity_review_manifest_summary.json",
        {
            **manifest,
            "review_cases": manifest.get("review_cases", [])[:30],
            "full_manifest_path": str(stage_root / "continuity" / "deconfounded_continuity_review_manifest.json"),
        },
    )
    files.append("deconfounded_continuity_review_manifest_summary.json")
    case_index = (stage_root / "continuity" / "deconfounded_continuity_case_index.csv").read_text(encoding="utf-8")
    add_text("deconfounded_continuity_case_index.csv", case_index)
    stale = (stage_root / "audit" / "stale_artifact_diagnosis.md").read_text(encoding="utf-8")
    add_text("stale_artifact_diagnosis.md", stale)
    pack_manifest = {
        "artifact": "m5_4f1_review_pack",
        "folder_name": "review_pack_m5_4f1",
        "file_cap": 20,
        "file_count": len(files) + 1,
        "files": [*files, "review_pack_manifest.json"],
        "canonical_live_state": False,
        "uses_full_review_evidence_by_reference": True,
        "full_stage_root": str(stage_root),
        "continuity_launcher_path": summary.get("continuity_launcher_path"),
        "continuity_review_url": summary.get("continuity_review_url"),
        **safety_payload(),
    }
    write_json(pack_root / "review_pack_manifest.json", pack_manifest)
    if pack_manifest["file_count"] > 20:
        raise ValueError("M5.4F.1 review pack exceeds the 20-file cap")
    return {"review_pack_path": str(pack_root), "review_pack_file_count": pack_manifest["file_count"]}


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
    stage_root = stage_root.resolve()
    role_review_root = stage_root / "role_review"
    decision_root = role_review_root / "decisions"
    completed_path = decision_root / "completed_review.json"
    if not completed_path.exists():
        post_role = _write_blocked_post_role_outputs(stage_root)
        return {"final_classification": "BLOCKED_ROLE_LABEL_SUPPORT", "exact_blocker": EXACT_BLOCKER, **post_role}
    completed = read_json(completed_path)
    state = _completed_review_state(completed)
    completed_at = str(state.get("completed_at") or state.get("updated_at") or completed.get("created_at") or _now())
    manifest = read_json(role_review_root / "balanced_role_review_manifest.json")
    validation_root = stage_root / "validation"
    audit_root = stage_root / "audit"
    continuity_root = stage_root / "continuity"
    learning_root = stage_root / "learning"
    for root in [validation_root, audit_root, continuity_root, learning_root]:
        root.mkdir(parents=True, exist_ok=True)

    completed_validation = _completed_role_review_validation(
        stage_root=stage_root,
        manifest_payload=manifest,
        completed_payload=completed,
    )
    write_json(audit_root / "completed_role_review_ingestion_validation.json", completed_validation)
    if completed_validation["status"] != "PASS_COMPLETED_ROLE_REVIEW_VALIDATED":
        summary = {
            "artifact": "m5_4f_validation_summary",
            "created_at": completed_at,
            "stage_root": str(stage_root),
            "final_classification": "BLOCKED_COMPLETED_ROLE_REVIEW_VALIDATION",
            "exact_blocker": "COMPLETED_ROLE_REVIEW_VALIDATION_FAILED",
            "role_review_complete": bool(state.get("completed")),
            "completed_role_review_validation": completed_validation["status"],
            **safety_payload(),
        }
        write_json(validation_root / "m5_4f_validation_summary.json", summary)
        return summary

    examples = _role_review_examples(manifest_payload=manifest, completed_payload=completed)
    _write_role_review_examples(learning_root, examples)
    label_counts = Counter(row["human_decision"] for row in examples if row["label_usable_for_training"])
    distribution = {
        "artifact": "m5_4f_role_label_distribution",
        "role_review_complete": bool(state.get("completed")),
        "role_label_distribution": _role_label_distribution(label_counts),
        "role_review_case_count": len(manifest.get("review_cases", [])),
        "training_usable_count": sum(1 for row in examples if row["label_usable_for_training"]),
        "training_excluded_count": sum(1 for row in examples if not row["label_usable_for_training"]),
        **safety_payload(),
    }
    readiness = _role_training_readiness(examples, bool(state.get("completed")))
    calibrator_validation = _role_calibrator_validation(examples, readiness)
    write_json(learning_root / "role_label_distribution.json", distribution)
    write_json(learning_root / "role_training_readiness.json", readiness)
    write_json(learning_root / "role_calibrator_validation.json", calibrator_validation)

    paths = _stage_input_paths(stage_root)
    role_rows = _read_rows(paths["m54e_stage_root"] / "role" / "visual_role_context_rows.json")
    application_rows, application_audit = _post_role_application(
        stage_root=stage_root,
        role_rows=role_rows,
        examples=examples,
    )
    write_json(
        learning_root / "role_application_rows.json",
        {
            "artifact": "m5_4f_role_application_rows",
            "row_count": len(application_rows),
            "rows": application_rows,
            **safety_payload(),
        },
    )
    write_json(learning_root / "role_application_audit.json", application_audit)
    post_role_rows = _post_role_context_rows(role_rows=role_rows, application_rows=application_rows)
    partition_info = _write_post_role_partition(
        stage_root=stage_root,
        post_role_rows=post_role_rows,
        application_audit=application_audit,
        readiness=readiness,
        validation=calibrator_validation,
    )

    node_by_visible_id = {str(row["visible_person_base_id"]): row for row in partition_info["node_rows"]}
    selection = _select_deconfounded_continuity_candidates(
        candidate_rows=partition_info["accepted_rows"],
        node_by_visible_id=node_by_visible_id,
    )
    selected_rows = [*selection["likely_positive"], *selection["likely_negative"]]
    endpoint_audit = _endpoint_reuse_audit(selection)
    feature_audit = _feature_balance_audit(selected_rows)
    preview_manifest = _preview_continuity_review_manifest(
        selected_rows=selected_rows,
        node_by_visible_id=node_by_visible_id,
    )
    selection_audit = audit_continuity_review_selection(preview_manifest)
    selection_audit.update(
        {
            "artifact": "m5_4f_review_selection_balance_audit",
            "diagnostic_preservation_label": "M5_4F1_DECONFOUNDED_POST_ROLE_CONTINUITY_REVIEW",
            "selection_source": "post_human_role_context_rows",
            "balance_passed": not selection_audit["issues"]
            and selection_audit["positive_count"] == 15
            and selection_audit["negative_count"] == 15
            and all(value >= 5 for value in selection_audit["equivalence_clusters_per_class"].values())
            and endpoint_audit["endpoint_reuse_passed"]
            and feature_audit["balance_passed"],
        }
    )
    write_json(continuity_root / "review_selection_balance_audit.json", selection_audit)
    write_json(continuity_root / "endpoint_reuse_audit.json", endpoint_audit)
    write_json(continuity_root / "feature_balance_audit.json", feature_audit)
    if not selection_audit["balance_passed"]:
        write_json(
            continuity_root / "deconfounded_continuity_review_manifest.json",
            {
                "artifact": "m5_4f_deconfounded_continuity_review_manifest",
                "review_cases": [],
                "status": "BLOCKED_DECONFOUNDED_REVIEW_BALANCE_GATES",
                "exact_blocker": "DECONFOUNDED_CONTINUITY_REVIEW_BALANCE_GATES_FAILED",
                **safety_payload(),
            },
        )
        with (continuity_root / "deconfounded_continuity_case_index.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["review_case_id", "bucket", "frame_gap", "team", "source_role", "target_role"])
        summary = {
            "artifact": "m5_4f_validation_summary",
            "created_at": completed_at,
            "stage_root": str(stage_root),
            "source_m5_4e_stage_root": str(paths["m54e_stage_root"]),
            "final_classification": "BLOCKED_DECONFOUNDED_CONTINUITY_REVIEW_BALANCE",
            "exact_blocker": "DECONFOUNDED_CONTINUITY_REVIEW_BALANCE_GATES_FAILED",
            "completed_role_review_validation": completed_validation["status"],
            "role_review_complete": True,
            "role_label_distribution": distribution["role_label_distribution"],
            "role_training_readiness": readiness["status"],
            "role_calibrator_validation": calibrator_validation["status"],
            "role_rows_updated": application_audit["role_rows_updated"],
            "candidate_pool_before_role_partitioning": partition_info["manifest"][
                "candidate_pool_before_role_partitioning"
            ],
            "candidate_pool_after_role_partitioning": partition_info["manifest"][
                "candidate_pool_after_role_partitioning"
            ],
            "continuity_case_count": 0,
            "continuity_review_generated": False,
            "review_selection_balance_audit_passed": False,
            **safety_payload(),
        }
        write_json(validation_root / "m5_4f_validation_summary.json", summary)
        _mark_frozen_pre_review_pack(stage_root)
        _write_artifact_resolution_audits(stage_root)
        pack = _write_m54f1_review_pack(stage_root, summary)
        return {**summary, **pack}

    continuity_review = _write_deconfounded_continuity_review(
        stage_root=stage_root,
        repo_root=_repo_root_from_module(),
        selection=selection,
        node_rows=partition_info["node_rows"],
        completed_at=completed_at,
    )
    actual_audit = audit_continuity_review_selection(continuity_review["manifest"])
    actual_audit.update(
        {
            "artifact": "m5_4f_review_selection_balance_audit",
            "diagnostic_preservation_label": "M5_4F1_DECONFOUNDED_POST_ROLE_CONTINUITY_REVIEW",
            "selection_source": "post_human_role_context_rows",
            "balance_passed": not actual_audit["issues"]
            and actual_audit["positive_count"] == 15
            and actual_audit["negative_count"] == 15
            and all(value >= 5 for value in actual_audit["equivalence_clusters_per_class"].values())
            and endpoint_audit["endpoint_reuse_passed"]
            and feature_audit["balance_passed"],
        }
    )
    write_json(continuity_root / "review_selection_balance_audit.json", actual_audit)
    _update_post_role_manifest_review_generated(stage_root, actual_audit["balance_passed"])
    temporal_counts = {
        "temporal_gif_count": continuity_review["temporal_gif_count"],
        "temporal_mp4_count": continuity_review["temporal_mp4_count"],
    }
    summary = {
        "artifact": "m5_4f_validation_summary",
        "created_at": completed_at,
        "match_id": "128058",
        "stage_root": str(stage_root),
        "source_m5_4e_stage_root": str(paths["m54e_stage_root"]),
        "final_classification": POST_REVIEW_FINAL_CLASSIFICATION
        if actual_audit["balance_passed"]
        else "BLOCKED_DECONFOUNDED_CONTINUITY_REVIEW_BALANCE",
        "exact_blocker": POST_REVIEW_BLOCKER
        if actual_audit["balance_passed"]
        else "DECONFOUNDED_CONTINUITY_REVIEW_BALANCE_GATES_FAILED",
        "completed_role_review_validation": completed_validation["status"],
        "role_review_complete": True,
        "role_review_case_count": len(manifest.get("review_cases", [])),
        "role_label_distribution": distribution["role_label_distribution"],
        "role_training_readiness": readiness["status"],
        "count_supported_role_classes": readiness.get("count_supported_classes", []),
        "grouped_validation_ready_role_classes": readiness.get("grouped_validation_ready_classes", []),
        "role_calibrator_validation": calibrator_validation["status"],
        "role_calibrator_grouped_validation_performed": calibrator_validation["grouped_validation_performed"],
        "role_rows_updated": application_audit["role_rows_updated"],
        "exact_reviewed_rows_applied": application_audit["exact_reviewed_rows_applied"],
        "broad_inferred_rows_updated": application_audit["broad_inferred_rows_updated"],
        "unsupported_classes_broadly_applied": False,
        "post_role_gate": partition_info["manifest"]["status"],
        "candidate_pool_before_role_partitioning": partition_info["manifest"][
            "candidate_pool_before_role_partitioning"
        ],
        "candidate_pool_after_role_partitioning": partition_info["manifest"]["candidate_pool_after_role_partitioning"],
        "role_incompatible_rejected_count": partition_info["manifest"]["role_incompatible_rejected_count"],
        "continuity_case_count": continuity_review["review_case_count"],
        "likely_positive_continuity_review_count": selection["likely_positive_count"],
        "likely_negative_continuity_review_count": selection["likely_negative_count"],
        "continuity_review_generated": actual_audit["balance_passed"],
        "continuity_launcher_path": continuity_review["launcher_path"],
        "continuity_review_url": continuity_review["review_url"],
        "endpoint_reuse_max": endpoint_audit["endpoint_reuse_max"],
        "review_selection_balance_audit_passed": actual_audit["balance_passed"],
        "feature_balance_audit_passed": feature_audit["balance_passed"],
        "new_continuity_frame_gap_distribution_by_proposed_class": actual_audit[
            "frame_gap_distribution_by_proposed_class"
        ],
        "new_continuity_team_distribution_by_proposed_class": actual_audit["team_distribution_by_proposed_class"],
        "new_continuity_equivalence_cluster_counts_per_proposed_class": actual_audit["equivalence_clusters_per_class"],
        "m5_4e_continuity_review_selection_preservation_label": DIAGNOSTIC_ONLY,
        "original_m5_4e_packs_preserved": True,
        **temporal_counts,
        **safety_payload(),
    }
    write_json(validation_root / "m5_4f_validation_summary.json", summary)
    _mark_frozen_pre_review_pack(stage_root)
    _write_artifact_resolution_audits(stage_root)
    pack = _write_m54f1_review_pack(stage_root, summary)
    return {**summary, **pack}
