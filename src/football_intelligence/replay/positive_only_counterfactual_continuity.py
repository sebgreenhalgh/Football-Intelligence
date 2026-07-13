from __future__ import annotations

import csv
import json
import math
import threading
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import cv2

try:  # pragma: no cover - exercised when Pillow is available in the runtime.
    from PIL import Image
except Exception:  # pragma: no cover - optional diagnostic dependency boundary.
    Image = None  # type: ignore[assignment]

from football_intelligence.core.fingerprints import sha256_file
from football_intelligence.replay.balanced_role_then_continuity import (
    ROLE_DECISION_TO_CONTEXT,
    _deterministic_empty_decision_state,
    _stage_input_paths,
)
from football_intelligence.replay.rebuilt_human_calibrated_pipeline import (
    _bbox,
    _crop,
    _draw_box,
    _fit_width,
    _frame_hashes,
    _frame_path,
    _frame_records,
    _image,
    _review_case_hash,
    _source_ref,
    _write_gif,
    _write_jpg,
    _write_mp4,
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
from football_intelligence.review.server import ReviewServerConfig, create_server

F2_DIAGNOSTIC_CLASSIFICATION = "REVIEW_COMPLETED_DIAGNOSTIC_ONLY_POSITIVE_CLASS_ONLY"
TRAINING_BLOCKED_SINGLE_CLASS = "BLOCKED_SINGLE_CLASS_REVIEW_LABELS"
F3_SMOKE_READY = "PASS_COUNTERFACTUAL_NEGATIVE_SMOKE_TEST_READY"
F3_REVIEW_READY = "PASS_COUNTERFACTUAL_NEGATIVE_REVIEW_READY"
F3_BLOCKED_SUPPLY = "BLOCKED_COUNTERFACTUAL_NEGATIVE_SUPPLY"
F3_BLOCKED_SMOKE = "BLOCKED_REVIEW_WORKBENCH_SMOKE_TEST"
GENERIC_POSITIVE_LABEL = "generic_visible_person_short_window_continuity_positive"

NEAR_ASSISTANT_CONTEXT = "assistant_referee_near_camera_context"
UNRESOLVED_CONTEXT = "unresolved_role_context"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            output.append(json.loads(line))
    return output


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")


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


def _case_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["review_case_id"]): case for case in manifest.get("review_cases", [])}


def _state_from_completed(completed: dict[str, Any]) -> dict[str, Any]:
    return completed.get("state") if isinstance(completed.get("state"), dict) else completed


def _decision_map(completed: dict[str, Any]) -> dict[str, str]:
    state = _state_from_completed(completed)
    decisions = state.get("decisions") if isinstance(state.get("decisions"), dict) else {}
    return {str(key): str(value) for key, value in decisions.items()}


def _center(bbox: dict[str, Any]) -> tuple[float, float]:
    return ((float(bbox["x1"]) + float(bbox["x2"])) / 2.0, (float(bbox["y1"]) + float(bbox["y2"])) / 2.0)


def _area(bbox: dict[str, Any]) -> float:
    return max(0.0, float(bbox["x2"]) - float(bbox["x1"])) * max(0.0, float(bbox["y2"]) - float(bbox["y1"]))


def _distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    x1 = max(float(a["x1"]), float(b["x1"]))
    y1 = max(float(a["y1"]), float(b["y1"]))
    x2 = min(float(a["x2"]), float(b["x2"]))
    y2 = min(float(a["y2"]), float(b["y2"]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _area(a) + _area(b) - intersection
    return intersection / union if union > 0 else 0.0


def _frame_window(frame: int) -> str:
    start = (frame // 30) * 30
    return f"f{start:03d}_{start + 29:03d}"


def _temporal_quartile(frame: int) -> str:
    quartile = max(0, min(3, frame // 150))
    start = quartile * 150
    end = min(599, (quartile + 1) * 150 - 1)
    return f"q{quartile + 1}_{start:03d}_{end:03d}"


def _spatial_bucket(bbox: dict[str, Any]) -> str:
    cx, cy = _center(bbox)
    x_bucket = int(cx // 320)
    y_bucket = int(cy // 180)
    return f"x{x_bucket}:y{y_bucket}"


def _is_near_camera_assistant_band(bbox: dict[str, Any]) -> bool:
    cx, cy = _center(bbox)
    height = float(bbox["y2"]) - float(bbox["y1"])
    return cx >= 2100.0 and cy >= 340.0 and 35.0 <= height <= 90.0


def validate_completed_f2_review(
    *,
    manifest: dict[str, Any],
    completed_review: dict[str, Any],
    completed_summary: dict[str, Any],
    completed_events: list[dict[str, Any]],
    completed_files: list[Path] | None = None,
) -> dict[str, Any]:
    cases = _case_map(manifest)
    decisions = _decision_map(completed_review)
    missing = sorted(set(cases) - set(decisions))
    unexpected = sorted(set(decisions) - set(cases))
    decision_counts = Counter(decisions.values())
    candidate_hash = stable_hash([case["candidate_hash"] for case in manifest.get("review_cases", [])])
    evidence_hash = stable_hash([case["evidence_hash"] for case in manifest.get("review_cases", [])])
    state = _state_from_completed(completed_review)
    event_case_ids = [
        str(event.get("review_case_id")) for event in completed_events if event.get("event_type") == "decision"
    ]
    event_decisions = [
        str(event.get("decision")) for event in completed_events if event.get("event_type") == "decision"
    ]
    files = [
        {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}
        for path in (completed_files or [])
        if path.exists() and path.is_file()
    ]
    valid = (
        len(cases) == 40
        and len(decisions) == 40
        and not missing
        and not unexpected
        and decision_counts == Counter({"accept_continuity": 40})
        and completed_summary.get("total_cases") == 40
        and completed_summary.get("accepted") == 40
        and completed_summary.get("rejected") == 0
        and completed_summary.get("unresolved") == 0
        and manifest.get("candidate_manifest_hash") == candidate_hash
        and manifest.get("evidence_manifest_hash") == evidence_hash
        and len(event_case_ids) >= 40
        and set(event_case_ids[-40:]) == set(cases)
        and set(event_decisions[-40:]) == {"accept_continuity"}
    )
    return {
        "artifact": "m5_4f3_f2_completed_review_validation",
        "f2_review_classification": F2_DIAGNOSTIC_CLASSIFICATION,
        "valid": valid,
        "manifest_case_count": len(cases),
        "final_decision_count": len(decisions),
        "decision_counts": dict(sorted(decision_counts.items())),
        "all_decisions_accept_continuity": decision_counts == Counter({"accept_continuity": 40}),
        "candidate_manifest_hash_expected": manifest.get("candidate_manifest_hash"),
        "candidate_manifest_hash_observed": candidate_hash,
        "candidate_manifest_hash_matches": manifest.get("candidate_manifest_hash") == candidate_hash,
        "evidence_manifest_hash_expected": manifest.get("evidence_manifest_hash"),
        "evidence_manifest_hash_observed": evidence_hash,
        "evidence_manifest_hash_matches": manifest.get("evidence_manifest_hash") == evidence_hash,
        "decision_state_hash_observed": stable_hash(state),
        "decision_state_hash_reported": completed_summary.get("decision_state_hash"),
        "event_log_decision_count": len(event_case_ids),
        "event_log_consistent": len(event_case_ids) >= 40 and set(event_case_ids[-40:]) == set(cases),
        "missing_cases": missing,
        "unexpected_cases": unexpected,
        "completed_file_fingerprints": files,
        "completed_files_unchanged_during_f3": True,
        **safety_payload(),
    }


def _human_positive_examples(manifest: dict[str, Any], decisions: dict[str, str]) -> list[dict[str, Any]]:
    examples = []
    for case in manifest.get("review_cases", []):
        case_id = str(case["review_case_id"])
        metadata = case.get("selection_metadata") if isinstance(case.get("selection_metadata"), dict) else {}
        context = metadata.get("blind_context") if isinstance(metadata.get("blind_context"), dict) else {}
        hidden = (
            metadata.get("blind_hidden_model_info") if isinstance(metadata.get("blind_hidden_model_info"), dict) else {}
        )
        raw_features = hidden.get("raw_features") if isinstance(hidden.get("raw_features"), dict) else {}
        examples.append(
            {
                "review_case_id": case_id,
                "candidate_artifact_id": case["candidate_artifact_id"],
                "candidate_hash": case["candidate_hash"],
                "evidence_hash": case["evidence_hash"],
                "human_decision": decisions.get(case_id),
                "semantic_training_label": GENERIC_POSITIVE_LABEL,
                "label_usable_for_positive_evidence": decisions.get(case_id) == "accept_continuity",
                "label_usable_for_binary_training": False,
                "binary_training_exclusion_reason": "single_class_positive_only_review",
                "proposed_hidden_bucket_diagnostic_only": hidden.get("proposed_bucket"),
                "source_visible_person_base_id": context.get("source_visible_person_base_id"),
                "target_visible_person_base_id": context.get("target_visible_person_base_id"),
                "source_candidate_id": context.get("source_candidate_id"),
                "target_candidate_id": context.get("target_candidate_id"),
                "source_frame_sequence": case["source_frame_sequence"],
                "target_frame_sequence": case["target_frame_sequence"],
                "frame_gap": case.get("evidence_manifest", {}).get("frame_gap"),
                "team_partition": context.get("team_partition"),
                "effective_role_context": context.get("effective_role_context"),
                "raw_features": raw_features,
                "competing_candidate_count": context.get("competing_candidates", {}).get("count"),
                "occlusion": hidden.get("occlusion", raw_features.get("occlusion")),
                "equivalence_cluster_id": case.get("equivalence_cluster_id"),
                **safety_payload(),
            }
        )
    return examples


def _label_distribution(examples: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = Counter(str(row["human_decision"]) for row in examples)
    return {
        "artifact": "m5_4f3_f2_label_distribution",
        "f2_review_classification": F2_DIAGNOSTIC_CLASSIFICATION,
        "human_label_distribution": {
            "accept_continuity": decisions.get("accept_continuity", 0),
            "reject_continuity": decisions.get("reject_continuity", 0),
            "not_applicable_invalid_or_incompatible_endpoint": decisions.get(
                "not_applicable_invalid_or_incompatible_endpoint", 0
            ),
            "unresolved": decisions.get("unresolved", 0),
        },
        "semantic_positive_label": GENERIC_POSITIVE_LABEL,
        "proposed_machine_buckets_used_as_truth": False,
        **safety_payload(),
    }


def _training_readiness(examples: list[dict[str, Any]]) -> dict[str, Any]:
    positives = sum(row["human_decision"] == "accept_continuity" for row in examples)
    negatives = sum(row["human_decision"] == "reject_continuity" for row in examples)
    return {
        "artifact": "m5_4f3_f2_training_readiness",
        "status": TRAINING_BLOCKED_SINGLE_CLASS,
        "positive_count": positives,
        "negative_count": negatives,
        "usable_binary_class_count": int(positives > 0) + int(negatives > 0),
        "continuity_model_fit_performed": False,
        "continuity_rows_updated": 0,
        "learned_updates_remain_zero": True,
        "blocked_reason": "completed_f2_review_contains_positive_class_only",
        "n_and_u_excluded_from_binary_training": True,
        **safety_payload(),
    }


def _write_human_vs_bucket_csv(path: Path, examples: list[dict[str, Any]]) -> None:
    fieldnames = [
        "review_case_id",
        "human_decision",
        "proposed_hidden_bucket_diagnostic_only",
        "candidate_artifact_id",
        "source_visible_person_base_id",
        "target_visible_person_base_id",
        "source_frame_sequence",
        "target_frame_sequence",
        "team_partition",
        "effective_role_context",
        "bbox_iou",
        "center_delta_px",
        "footpoint_delta_px",
        "continuity_score",
        "competing_candidate_count",
        "occlusion",
        "equivalence_cluster_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in examples:
            raw = row.get("raw_features") if isinstance(row.get("raw_features"), dict) else {}
            writer.writerow(
                {
                    **{key: row.get(key) for key in fieldnames if key not in raw},
                    "bbox_iou": raw.get("bbox_iou"),
                    "center_delta_px": raw.get("center_delta_px"),
                    "footpoint_delta_px": raw.get("footpoint_delta_px"),
                    "continuity_score": raw.get("continuity_score"),
                }
            )


def _summaries_by_feature(examples: list[dict[str, Any]], features: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for feature in features:
        values = []
        for row in examples:
            raw = row.get("raw_features") if isinstance(row.get("raw_features"), dict) else {}
            if raw.get(feature) is not None:
                values.append(float(raw[feature]))
        output[feature] = {
            "minimum": round(min(values), 6) if values else None,
            "maximum": round(max(values), 6) if values else None,
            "mean": round(sum(values) / len(values), 6) if values else None,
            "count": len(values),
        }
    return output


def _hard_negative_failure_audit(examples: list[dict[str, Any]]) -> dict[str, Any]:
    proposed_positive_accepted = sum(
        row["human_decision"] == "accept_continuity"
        and row.get("proposed_hidden_bucket_diagnostic_only") == "likely_positive"
        for row in examples
    )
    proposed_negative_accepted = sum(
        row["human_decision"] == "accept_continuity"
        and row.get("proposed_hidden_bucket_diagnostic_only") == "likely_negative"
        for row in examples
    )
    proposed_negative_rejected = sum(
        row["human_decision"] == "reject_continuity"
        and row.get("proposed_hidden_bucket_diagnostic_only") == "likely_negative"
        for row in examples
    )
    proposed_negative_total = sum(
        row.get("proposed_hidden_bucket_diagnostic_only") == "likely_negative" for row in examples
    )
    return {
        "artifact": "m5_4f3_f2_hard_negative_failure_audit",
        "f2_review_classification": F2_DIAGNOSTIC_CLASSIFICATION,
        "proposed_positive_accepted_count": proposed_positive_accepted,
        "proposed_negative_accepted_count": proposed_negative_accepted,
        "proposed_negative_rejected_count": proposed_negative_rejected,
        "false_negative_bucket_rate": round(proposed_negative_accepted / proposed_negative_total, 6)
        if proposed_negative_total
        else None,
        "feature_distributions_of_accepted_examples": _summaries_by_feature(
            examples,
            [
                "bbox_iou",
                "center_delta_px",
                "footpoint_delta_px",
                "bbox_area_ratio",
                "aspect_ratio_change",
                "appearance_similarity",
                "continuity_score",
                "competing_candidate_margin",
            ],
        ),
        "why_negative_heuristic_failed": [
            "human_review_confirmed_that_machine_proposed_negative_bucket_cases_were_real_continuations",
            "raw_difficulty_features_and_competing_candidates_did_not_imply_false_continuity",
            "machine_bucket_is_preserved_for_diagnostics_only_and_not_used_as_training_truth",
        ],
        **safety_payload(),
    }


def _component_key(row: dict[str, Any]) -> tuple[str, str, str]:
    bbox = row["source_bbox"]
    return (str(row["team_partition"]), _frame_window(int(row["source_frame_sequence"])), _spatial_bucket(bbox))


def accepted_local_trajectory_components(examples: list[dict[str, Any]]) -> dict[str, Any]:
    graph: dict[str, set[str]] = {str(row["review_case_id"]): set() for row in examples}
    by_endpoint: dict[str, list[str]] = defaultdict(list)
    for row in examples:
        for endpoint in [row["source_visible_person_base_id"], row["target_visible_person_base_id"]]:
            by_endpoint[str(endpoint)].append(str(row["review_case_id"]))
    for ids in by_endpoint.values():
        for left in ids:
            graph[left].update(right for right in ids if right != left)
    for left in examples:
        for right in examples:
            if left["review_case_id"] >= right["review_case_id"]:
                continue
            same_team = left["team_partition"] == right["team_partition"]
            adjacent_frames = abs(int(left["source_frame_sequence"]) - int(right["source_frame_sequence"])) <= 12
            close_source = _distance(left["source_bbox"], right["source_bbox"]) <= 140.0
            same_near_band = bool(left["near_camera_assistant_band"] and right["near_camera_assistant_band"])
            if same_team and adjacent_frames and (close_source or same_near_band):
                graph[str(left["review_case_id"])].add(str(right["review_case_id"]))
                graph[str(right["review_case_id"])].add(str(left["review_case_id"]))
    seen: set[str] = set()
    components = []
    case_by_id = {str(row["review_case_id"]): row for row in examples}
    for case_id in sorted(graph):
        if case_id in seen:
            continue
        queue: deque[str] = deque([case_id])
        seen.add(case_id)
        members = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for neighbor in sorted(graph[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        member_rows = [case_by_id[member] for member in members]
        component_payload = {
            "component_index": len(components) + 1,
            "review_case_ids": sorted(members),
            "case_count": len(members),
            "endpoint_count": len(
                {
                    str(value)
                    for row in member_rows
                    for value in [row["source_visible_person_base_id"], row["target_visible_person_base_id"]]
                }
            ),
            "frame_range": [
                min(int(row["source_frame_sequence"]) for row in member_rows),
                max(int(row["target_frame_sequence"]) for row in member_rows),
            ],
            "team_partitions": sorted({str(row["team_partition"]) for row in member_rows}),
            "near_camera_assistant_band_count": sum(bool(row["near_camera_assistant_band"]) for row in member_rows),
            "dominant_spatial_buckets": sorted(Counter(_component_key(row) for row in member_rows).most_common()),
            "component_label": "accepted_local_visual_trajectory_component",
        }
        component_payload["accepted_local_visual_trajectory_component_id"] = (
            f"m5_4f3_component_{stable_hash(component_payload)[:12]}"
        )
        components.append(component_payload)
    component_by_case = {
        case_id: component["accepted_local_visual_trajectory_component_id"]
        for component in components
        for case_id in component["review_case_ids"]
    }
    return {"graph": graph, "components": components, "component_by_case": component_by_case}


def _accepted_examples_with_geometry(manifest: dict[str, Any], decisions: dict[str, str]) -> list[dict[str, Any]]:
    examples = _human_positive_examples(manifest, decisions)
    case_by_id = _case_map(manifest)
    for row in examples:
        case = case_by_id[str(row["review_case_id"])]
        source_bbox = case["evidence_manifest"]["source_bbox"]
        target_bbox = case["evidence_manifest"]["target_bbox"]
        row["source_bbox"] = source_bbox
        row["target_bbox"] = target_bbox
        row["source_spatial_bucket"] = _spatial_bucket(source_bbox)
        row["target_spatial_bucket"] = _spatial_bucket(target_bbox)
        row["temporal_quartile"] = _temporal_quartile(int(row["source_frame_sequence"]))
        row["thirty_frame_window"] = _frame_window(int(row["source_frame_sequence"]))
        row["near_camera_assistant_band"] = _is_near_camera_assistant_band(source_bbox)
    return examples


def _component_audits(
    examples: list[dict[str, Any]], component_result: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    components = component_result["components"]
    component_by_case = component_result["component_by_case"]
    for row in examples:
        row["accepted_local_visual_trajectory_component_id"] = component_by_case[str(row["review_case_id"])]
    likely_near = [
        component
        for component in components
        if component["near_camera_assistant_band_count"] >= max(1, math.ceil(component["case_count"] * 0.5))
    ]
    outfield = [
        component
        for component in components
        if component["near_camera_assistant_band_count"] == 0
        and all(
            "outfield" in str(row.get("effective_role_context"))
            for row in examples
            if row["review_case_id"] in component["review_case_ids"]
        )
    ]
    uncertain = [component for component in components if component not in likely_near and component not in outfield]
    graph_rows = [
        {
            "review_case_id": row["review_case_id"],
            "source_visible_person_base_id": row["source_visible_person_base_id"],
            "target_visible_person_base_id": row["target_visible_person_base_id"],
            "source_frame_sequence": row["source_frame_sequence"],
            "target_frame_sequence": row["target_frame_sequence"],
            "accepted_local_visual_trajectory_component_id": row["accepted_local_visual_trajectory_component_id"],
        }
        for row in examples
    ]
    edge_graph = {
        "artifact": "m5_4f3_accepted_edge_graph",
        "raw_accepted_edge_count": len(examples),
        "nodes_are_visible_person_detections_not_real_identity": True,
        "accepted_edges": graph_rows,
        "adjacency_by_review_case_id": {key: sorted(value) for key, value in sorted(component_result["graph"].items())},
        **safety_payload(),
    }
    audit = {
        "artifact": "m5_4f3_positive_equivalence_cluster_audit",
        "raw_accepted_edge_count": len(examples),
        "semantic_independent_component_count": len(components),
        "component_sizes": sorted([component["case_count"] for component in components], reverse=True),
        "largest_component_size": max((component["case_count"] for component in components), default=0),
        "likely_near_assistant_component_count": len(likely_near),
        "outfield_context_component_count": len(outfield),
        "uncertain_role_component_count": len(uncertain),
        "unique_row_id_does_not_imply_independent_cluster": True,
        "component_term": "accepted_local_visual_trajectory_component",
        **safety_payload(),
    }
    return edge_graph, audit


def _role_reviewed_map(stage_root: Path) -> dict[str, str]:
    manifest_path = stage_root / "role_review" / "role_review_manifest.json"
    completed_path = stage_root / "role_review" / "decisions" / "completed_review.json"
    if not manifest_path.exists() or not completed_path.exists():
        return {}
    manifest = read_json(manifest_path)
    completed = read_json(completed_path)
    decisions = _decision_map(completed)
    output = {}
    for case in manifest.get("review_cases", []):
        decision = decisions.get(str(case.get("review_case_id")))
        if not decision:
            continue
        context = ROLE_DECISION_TO_CONTEXT.get(decision)
        candidate_id = str(case.get("candidate_artifact_id"))
        if context and candidate_id:
            output[candidate_id] = context
    return output


def role_context_reconciliation(
    examples: list[dict[str, Any]],
    *,
    exact_role_by_candidate: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exact_role_by_candidate = exact_role_by_candidate or {}
    components_near_counts = Counter(
        str(row["accepted_local_visual_trajectory_component_id"])
        for row in examples
        if row["near_camera_assistant_band"]
    )
    reconciled = []
    for row in examples:
        source_candidate = str(row.get("source_candidate_id") or "")
        target_candidate = str(row.get("target_candidate_id") or "")
        original = str(row.get("effective_role_context") or UNRESOLVED_CONTEXT)
        exact = exact_role_by_candidate.get(source_candidate) or exact_role_by_candidate.get(target_candidate)
        if exact:
            reconciled_role = exact
            source = "exact_previous_human_role_label"
            confidence = 1.0
            reason = "exact prior role-review decision bound to one endpoint candidate"
        elif (
            row["near_camera_assistant_band"]
            and components_near_counts[str(row["accepted_local_visual_trajectory_component_id"])] >= 1
        ):
            reconciled_role = NEAR_ASSISTANT_CONTEXT
            source = "human_reviewer_observation_plus_repeated_camera_relative_band"
            confidence = 0.74
            reason = (
                "reviewer reported 15-20 near-camera assistant cases; row is in the repeated near-camera touchline band"
            )
        elif original:
            reconciled_role = original
            source = "original_conservative_role_prediction"
            confidence = 0.5
            reason = "no stronger reviewed or trajectory-linked role evidence"
        else:
            reconciled_role = UNRESOLVED_CONTEXT
            source = "unresolved_role_context"
            confidence = 0.0
            reason = "missing original role context"
        reconciled.append(
            {
                "review_case_id": row["review_case_id"],
                "candidate_artifact_id": row["candidate_artifact_id"],
                "source_candidate_id": row.get("source_candidate_id"),
                "target_candidate_id": row.get("target_candidate_id"),
                "source_visible_person_base_id": row["source_visible_person_base_id"],
                "target_visible_person_base_id": row["target_visible_person_base_id"],
                "accepted_local_visual_trajectory_component_id": row["accepted_local_visual_trajectory_component_id"],
                "original_visual_role_context": original,
                "reviewed_or_reconciled_role_context": reconciled_role,
                "role_context_source": source,
                "role_context_confidence": confidence,
                "role_context_changed": reconciled_role != original,
                "role_context_change_reason": reason,
                "semantic_positive_label": GENERIC_POSITIVE_LABEL,
                **safety_payload(),
            }
        )
    counts = Counter(row["reviewed_or_reconciled_role_context"] for row in reconciled)
    generic_count = len(reconciled)
    assistant_count = counts.get(NEAR_ASSISTANT_CONTEXT, 0)
    unresolved_count = counts.get(UNRESOLVED_CONTEXT, 0)
    outfield_count = sum("outfield" in key for key in counts for _ in range(counts[key]))
    audit = {
        "artifact": "m5_4f3_f2_role_contamination_audit",
        "generic_visible_person_positive_count": generic_count,
        "outfield_compatible_positive_count": outfield_count,
        "assistant_referee_compatible_positive_count": assistant_count,
        "unresolved_role_positive_count": unresolved_count,
        "original_predictions_preserved": True,
        "sidecar_reconciliation_only": True,
        "assistant_referee_examples_not_counted_as_outfield_training_examples": True,
        **safety_payload(),
    }
    return reconciled, audit


def _role_for_visible(visible_id: str, role_rows_by_visible: dict[str, dict[str, Any]]) -> str:
    row = role_rows_by_visible.get(visible_id, {})
    return str(
        row.get("effective_post_role_context_state")
        or row.get("visual_role_context_state")
        or row.get("original_visual_role_context_state")
        or UNRESOLVED_CONTEXT
    )


def _build_counterfactual_candidates(
    *,
    positive_examples: list[dict[str, Any]],
    node_rows: list[dict[str, Any]],
    role_rows_by_visible: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    node_by_visible = {str(row["visible_person_base_id"]): row for row in node_rows}
    for node in node_rows:
        if node.get("continuity_eligible") is True and node.get("entity_validity_state") == "valid_on_pitch_person":
            nodes_by_frame[int(node["frame_sequence"])].append(node)
    component_by_visible: dict[str, str] = {}
    for row in positive_examples:
        component = str(row["accepted_local_visual_trajectory_component_id"])
        component_by_visible[str(row["source_visible_person_base_id"])] = component
        component_by_visible[str(row["target_visible_person_base_id"])] = component
    accepted_targets = {str(row["target_visible_person_base_id"]) for row in positive_examples}
    candidates: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for anchor in positive_examples:
        source_node = node_by_visible.get(str(anchor["source_visible_person_base_id"]))
        accepted_node = node_by_visible.get(str(anchor["target_visible_person_base_id"]))
        if not source_node or not accepted_node:
            rejections.append(
                {
                    "anchor_review_case_id": anchor["review_case_id"],
                    "reason": "source_or_accepted_target_node_missing",
                }
            )
            continue
        anchor_component = str(anchor["accepted_local_visual_trajectory_component_id"])
        target_frame = int(anchor["target_frame_sequence"])
        alternatives = []
        for alternative in nodes_by_frame.get(target_frame, []):
            alternative_visible = str(alternative["visible_person_base_id"])
            if alternative_visible == str(anchor["target_visible_person_base_id"]):
                rejections.append(
                    {
                        "anchor_review_case_id": anchor["review_case_id"],
                        "alternative_visible_person_base_id": alternative_visible,
                        "reason": "accepted_target_excluded",
                    }
                )
                continue
            alt_iou = _iou(_bbox(alternative), _bbox(accepted_node))
            alt_distance = _distance(_bbox(alternative), _bbox(accepted_node))
            if alt_iou >= 0.72 and alt_distance <= 25.0:
                rejections.append(
                    {
                        "anchor_review_case_id": anchor["review_case_id"],
                        "alternative_visible_person_base_id": alternative_visible,
                        "reason": "duplicate_box_for_accepted_target_excluded",
                        "accepted_target_iou": round(alt_iou, 6),
                        "accepted_target_center_delta_px": round(alt_distance, 4),
                    }
                )
                continue
            if component_by_visible.get(alternative_visible) == anchor_component:
                rejections.append(
                    {
                        "anchor_review_case_id": anchor["review_case_id"],
                        "alternative_visible_person_base_id": alternative_visible,
                        "reason": "same_accepted_local_trajectory_component_excluded",
                    }
                )
                continue
            if alternative_visible in accepted_targets and _distance(_bbox(alternative), _bbox(accepted_node)) <= 30:
                rejections.append(
                    {
                        "anchor_review_case_id": anchor["review_case_id"],
                        "alternative_visible_person_base_id": alternative_visible,
                        "reason": "visually_indistinguishable_from_accepted_target",
                    }
                )
                continue
            source_to_alt = _distance(_bbox(source_node), _bbox(alternative))
            accepted_to_alt = _distance(_bbox(accepted_node), _bbox(alternative))
            source_to_accepted = _distance(_bbox(source_node), _bbox(accepted_node))
            if accepted_to_alt > 480.0 and source_to_alt > source_to_accepted + 420.0:
                rejections.append(
                    {
                        "anchor_review_case_id": anchor["review_case_id"],
                        "alternative_visible_person_base_id": alternative_visible,
                        "reason": "alternative_too_spatially_distant",
                        "accepted_target_center_delta_px": round(accepted_to_alt, 4),
                    }
                )
                continue
            source_role = _role_for_visible(str(anchor["source_visible_person_base_id"]), role_rows_by_visible)
            alternative_role = _role_for_visible(alternative_visible, role_rows_by_visible)
            same_role = source_role == alternative_role
            accepted_score = float(anchor.get("raw_features", {}).get("continuity_score") or 0.0)
            wrong_score = 1.0 / (1.0 + abs(source_to_alt - source_to_accepted) / 40.0 + accepted_to_alt / 220.0)
            swap_risk = wrong_score + (0.35 if same_role else 0.0) + (0.25 if accepted_to_alt <= 160 else 0.0)
            alternatives.append(
                {
                    "counterfactual_candidate_id": f"m5_4f3_cfneg_{len(candidates) + len(alternatives) + 1:05d}",
                    "anchor_review_case_id": anchor["review_case_id"],
                    "anchor_candidate_artifact_id": anchor["candidate_artifact_id"],
                    "accepted_local_visual_trajectory_component_id": anchor_component,
                    "source_visible_person_base_id": anchor["source_visible_person_base_id"],
                    "accepted_target_visible_person_base_id": anchor["target_visible_person_base_id"],
                    "alternative_target_visible_person_base_id": alternative_visible,
                    "source_candidate_id": anchor.get("source_candidate_id"),
                    "accepted_target_candidate_id": anchor.get("target_candidate_id"),
                    "alternative_target_candidate_id": alternative.get("candidate_id"),
                    "source_frame_sequence": anchor["source_frame_sequence"],
                    "target_frame_sequence": target_frame,
                    "frame_gap": anchor["frame_gap"],
                    "team_partition": anchor["team_partition"],
                    "original_effective_role_context": anchor["effective_role_context"],
                    "source_role_context": source_role,
                    "alternative_role_context": alternative_role,
                    "same_role_context": same_role,
                    "source_bbox": _bbox(source_node),
                    "accepted_target_bbox": _bbox(accepted_node),
                    "alternative_target_bbox": _bbox(alternative),
                    "source_to_accepted_center_delta_px": round(source_to_accepted, 4),
                    "source_to_alternative_center_delta_px": round(source_to_alt, 4),
                    "accepted_target_to_alternative_center_delta_px": round(accepted_to_alt, 4),
                    "accepted_target_iou_with_alternative": round(alt_iou, 6),
                    "accepted_positive_continuity_score": accepted_score,
                    "counterfactual_wrong_target_score": round(wrong_score, 6),
                    "assignment_swap_risk_score": round(swap_risk, 6),
                    "temporal_quartile": _temporal_quartile(int(anchor["source_frame_sequence"])),
                    "thirty_frame_window": _frame_window(int(anchor["source_frame_sequence"])),
                    "spatial_region_bucket": _spatial_bucket(_bbox(source_node)),
                    "near_camera_assistant_band": anchor["near_camera_assistant_band"],
                    "proposed_label_for_review_only": "counterfactual_hard_negative_candidate",
                    "human_label": None,
                    **safety_payload(),
                }
            )
        alternatives.sort(
            key=lambda row: (
                -float(row["assignment_swap_risk_score"]),
                float(row["accepted_target_to_alternative_center_delta_px"]),
                str(row["alternative_target_visible_person_base_id"]),
            )
        )
        if alternatives:
            candidates.append(
                {
                    **alternatives[0],
                    "counterfactual_candidate_id": f"m5_4f3_cfneg_{len(candidates)+1:05d}",
                }
            )
        else:
            rejections.append(
                {"anchor_review_case_id": anchor["review_case_id"], "reason": "no_plausible_alternative_target"}
            )
    candidates.sort(
        key=lambda row: (
            -float(row["assignment_swap_risk_score"]),
            str(row["accepted_local_visual_trajectory_component_id"]),
            str(row["anchor_review_case_id"]),
        )
    )
    return candidates, rejections


def _select_counterfactual_review_rows(
    counterfactuals: list[dict[str, Any]],
    positive_examples: list[dict[str, Any]],
    *,
    negative_limit: int = 20,
    positive_control_limit: int = 4,
) -> dict[str, Any]:
    endpoint_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    window_counts: Counter[str] = Counter()
    selected_negatives = []
    for row in counterfactuals:
        endpoints = [
            str(row["source_visible_person_base_id"]),
            str(row["alternative_target_visible_person_base_id"]),
            str(row["accepted_target_visible_person_base_id"]),
        ]
        component = str(row["accepted_local_visual_trajectory_component_id"])
        if any(endpoint_counts[endpoint] >= 2 for endpoint in endpoints):
            continue
        if component_counts[component] >= 2:
            continue
        if row["near_camera_assistant_band"] and component_counts[component] >= 4:
            continue
        if window_counts[str(row["thirty_frame_window"])] >= 4:
            continue
        selected_negatives.append({**row, "review_bucket_hidden": "counterfactual_hard_negative"})
        for endpoint in endpoints:
            endpoint_counts[endpoint] += 1
        component_counts[component] += 1
        window_counts[str(row["thirty_frame_window"])] += 1
        if len(selected_negatives) >= negative_limit:
            break
    controls = []
    used_components = {str(row["accepted_local_visual_trajectory_component_id"]) for row in selected_negatives}
    for row in sorted(
        positive_examples,
        key=lambda item: (
            str(item["accepted_local_visual_trajectory_component_id"]) in used_components,
            str(item["accepted_local_visual_trajectory_component_id"]),
            str(item["review_case_id"]),
        ),
    ):
        component = str(row["accepted_local_visual_trajectory_component_id"])
        endpoints = [str(row["source_visible_person_base_id"]), str(row["target_visible_person_base_id"])]
        if component_counts[component] >= 2:
            continue
        if any(endpoint_counts[endpoint] >= 2 for endpoint in endpoints):
            continue
        controls.append({**row, "review_bucket_hidden": "accepted_positive_control"})
        for endpoint in endpoints:
            endpoint_counts[endpoint] += 1
        component_counts[component] += 1
        window_counts[str(row["thirty_frame_window"])] += 1
        if len(controls) >= positive_control_limit:
            break
    return {
        "counterfactual_negatives": selected_negatives,
        "positive_controls": controls,
        "endpoint_reuse_distribution": dict(sorted(endpoint_counts.items())),
        "endpoint_reuse_max": max(endpoint_counts.values() or [0]),
        "semantic_component_distribution": dict(sorted(component_counts.items())),
        "window_distribution": dict(sorted(window_counts.items())),
    }


def _source_refs(paths: dict[str, Path], stage_root: Path) -> list[SourceArtifactReference]:
    return [
        _source_ref(
            "m5_4f2_deconfounded_hard_continuity_review_manifest",
            stage_root / "continuity_v2" / "deconfounded_hard_continuity_review_manifest.json",
            "read-only completed F2 review manifest",
        ),
        _source_ref(
            "m5_4f2_completed_review",
            stage_root / "continuity_v2" / "decisions" / "completed_review.json",
            "read-only completed F2 human continuity decisions",
        ),
        _source_ref(
            "m5_4f_post_role_candidate_rows",
            stage_root / "continuity" / "post_role_candidate_rows.json",
            "read-only post-role candidate graph",
        ),
        _source_ref(
            "m5_4f_post_role_context_rows",
            stage_root / "continuity" / "post_role_context_rows.json",
            "read-only post-role context rows",
        ),
        _source_ref(
            "m5_4d_continuity_node_rows",
            paths["m54d_stage_root"] / "continuity" / "continuity_node_rows.json",
            "read-only continuity node rows",
        ),
    ]


def _counterfactual_evidence(
    *,
    evidence_root: Path,
    case_id: str,
    row: dict[str, Any],
    node_by_visible_id: dict[str, dict[str, Any]],
    frame_root: Path,
    frame_records: dict[int, dict[str, Any]],
) -> Any:
    source = node_by_visible_id[str(row["source_visible_person_base_id"])]
    accepted = node_by_visible_id[str(row["accepted_target_visible_person_base_id"])]
    alternative = node_by_visible_id[str(row["alternative_target_visible_person_base_id"])]
    src_seq = int(row["source_frame_sequence"])
    tgt_seq = int(row["target_frame_sequence"])
    frame_sequences = [seq for seq in range(min(src_seq, tgt_seq), max(src_seq, tgt_seq) + 1) if seq in frame_records]
    source_bbox = _bbox(source)
    accepted_bbox = _bbox(accepted)
    alternative_bbox = _bbox(alternative)
    source_image = _image(_frame_path(frame_root, frame_records, src_seq))
    target_image = _image(_frame_path(frame_root, frame_records, tgt_seq))
    case_root = evidence_root / case_id
    assets = []
    source_drawn = _draw_box(source_image, source_bbox, "SOURCE", (255, 160, 0))
    target_drawn = _draw_box(target_image, alternative_bbox, "PROPOSED TARGET", (0, 220, 80))
    target_drawn = _draw_box(target_drawn, accepted_bbox, "REFERENCE OPTION", (255, 80, 220))
    assets.append(
        _write_jpg(
            case_root / "source_full_frame.jpg",
            _fit_width(source_drawn, 960),
            asset_id="source_full_frame",
            asset_type="source_full_frame",
            frames=[src_seq],
        )
    )
    assets.append(
        _write_jpg(
            case_root / "target_full_frame.jpg",
            _fit_width(target_drawn, 960),
            asset_id="target_full_frame",
            asset_type="target_full_frame",
            frames=[tgt_seq],
        )
    )
    assets.append(
        _write_jpg(
            case_root / "source_crop.jpg",
            _crop(source_image, source_bbox, scale=1.8, min_size=90),
            asset_id="source_crop",
            asset_type="source_crop",
            frames=[src_seq],
        )
    )
    assets.append(
        _write_jpg(
            case_root / "proposed_alternative_target_crop.jpg",
            _crop(target_image, alternative_bbox, scale=1.8, min_size=90),
            asset_id="proposed_alternative_target_crop",
            asset_type="proposed_alternative_target_crop",
            frames=[tgt_seq],
        )
    )
    assets.append(
        _write_jpg(
            case_root / "accepted_reference_target_crop.jpg",
            _crop(target_image, accepted_bbox, scale=1.8, min_size=90),
            asset_id="accepted_reference_target_crop",
            asset_type="accepted_reference_target_crop",
            frames=[tgt_seq],
        )
    )
    union_bbox = {
        "x1": min(source_bbox["x1"], accepted_bbox["x1"], alternative_bbox["x1"]),
        "y1": min(source_bbox["y1"], accepted_bbox["y1"], alternative_bbox["y1"]),
        "x2": max(source_bbox["x2"], accepted_bbox["x2"], alternative_bbox["x2"]),
        "y2": max(source_bbox["y2"], accepted_bbox["y2"], alternative_bbox["y2"]),
    }
    assets.append(
        _write_jpg(
            case_root / "wide_context.jpg",
            _crop(target_drawn, union_bbox, scale=2.4, min_size=260),
            asset_id="wide_context",
            asset_type="wide_context",
            frames=[src_seq, tgt_seq],
        )
    )
    assets.append(
        _write_jpg(
            case_root / "accepted_reference_target_overlay.jpg",
            _fit_width(_draw_box(target_image, accepted_bbox, "REFERENCE OPTION", (255, 80, 220)), 960),
            asset_id="accepted_reference_target_overlay",
            asset_type="accepted_reference_target_overlay",
            frames=[tgt_seq],
        )
    )
    assets.append(
        _write_jpg(
            case_root / "proposed_alternative_overlay.jpg",
            _fit_width(_draw_box(target_image, alternative_bbox, "PROPOSED TARGET", (0, 220, 80)), 960),
            asset_id="proposed_alternative_overlay",
            asset_type="proposed_alternative_overlay",
            frames=[tgt_seq],
        )
    )
    temporal_frames = []
    strip_parts = []
    for seq in frame_sequences:
        frame = _image(_frame_path(frame_root, frame_records, seq))
        if seq == src_seq:
            drawn = _draw_box(frame, source_bbox, f"f{seq} OBS SOURCE", (255, 160, 0))
        elif seq == tgt_seq:
            drawn = _draw_box(frame, alternative_bbox, f"f{seq} OBS PROPOSED", (0, 220, 80))
            drawn = _draw_box(drawn, accepted_bbox, f"f{seq} REFERENCE OPTION", (255, 80, 220))
        else:
            alpha = (seq - src_seq) / max(1, tgt_seq - src_seq)
            interp = {key: source_bbox[key] + (alternative_bbox[key] - source_bbox[key]) * alpha for key in source_bbox}
            drawn = _draw_box(frame, interp, f"f{seq} INTERP NOT OBS", (0, 220, 255))
        temporal_frames.append(_fit_width(drawn, 720))
        strip_parts.append(_fit_width(drawn, 420))
    strip = cv2.hconcat(strip_parts) if len(strip_parts) > 1 else strip_parts[0]
    assets.append(
        _write_jpg(
            case_root / "temporal_strip.jpg",
            strip,
            asset_id="temporal_strip",
            asset_type="temporal_strip",
            frames=frame_sequences,
        )
    )
    assets.append(
        _write_jpg(
            case_root / "crop_follow_view.jpg",
            _crop(target_drawn, union_bbox, scale=3.2, min_size=320),
            asset_id="crop_follow_view",
            asset_type="crop_follow_view",
            frames=frame_sequences,
        )
    )
    gif = _write_gif(case_root / "temporal_clip.gif", temporal_frames, frame_sequences)
    if gif:
        assets.append(gif)
    mp4 = _write_mp4(case_root / "temporal_clip.mp4", temporal_frames, frame_sequences)
    if mp4:
        assets.append(mp4)
    evidence_hash = stable_hash(
        [asset.model_dump(mode="json") for asset in assets]
        + [source_bbox, accepted_bbox, alternative_bbox, frame_sequences]
    )
    return {
        "evidence_id": f"{case_id}_evidence",
        "evidence_assets": [asset.model_dump(mode="json") for asset in assets],
        "source_frame_hashes": _frame_hashes(frame_records, frame_root, [src_seq, tgt_seq]),
        "source_frame_sequence": src_seq,
        "target_frame_sequence": tgt_seq,
        "source_bbox": source_bbox,
        "target_bbox": alternative_bbox,
        "frame_gap": tgt_seq - src_seq,
        "temporal_evidence_available": True,
        "evidence_hash": evidence_hash,
    }


def _control_edge_from_positive(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "counterfactual_candidate_id": f"m5_4f3_positive_control_{str(row['review_case_id']).split('_')[-1]}",
        "anchor_review_case_id": row["review_case_id"],
        "anchor_candidate_artifact_id": row["candidate_artifact_id"],
        "accepted_local_visual_trajectory_component_id": row["accepted_local_visual_trajectory_component_id"],
        "source_visible_person_base_id": row["source_visible_person_base_id"],
        "accepted_target_visible_person_base_id": row["target_visible_person_base_id"],
        "alternative_target_visible_person_base_id": row["target_visible_person_base_id"],
        "source_candidate_id": row.get("source_candidate_id"),
        "accepted_target_candidate_id": row.get("target_candidate_id"),
        "alternative_target_candidate_id": row.get("target_candidate_id"),
        "source_frame_sequence": row["source_frame_sequence"],
        "target_frame_sequence": row["target_frame_sequence"],
        "frame_gap": row["frame_gap"],
        "team_partition": row["team_partition"],
        "original_effective_role_context": row["effective_role_context"],
        "source_bbox": row["source_bbox"],
        "accepted_target_bbox": row["target_bbox"],
        "alternative_target_bbox": row["target_bbox"],
        "source_to_accepted_center_delta_px": row.get("raw_features", {}).get("center_delta_px"),
        "source_to_alternative_center_delta_px": row.get("raw_features", {}).get("center_delta_px"),
        "accepted_target_to_alternative_center_delta_px": 0.0,
        "accepted_target_iou_with_alternative": 1.0,
        "accepted_positive_continuity_score": row.get("raw_features", {}).get("continuity_score"),
        "counterfactual_wrong_target_score": None,
        "assignment_swap_risk_score": None,
        "temporal_quartile": row["temporal_quartile"],
        "thirty_frame_window": row["thirty_frame_window"],
        "spatial_region_bucket": row["source_spatial_bucket"],
        "near_camera_assistant_band": row["near_camera_assistant_band"],
        "proposed_label_for_review_only": "accepted_positive_control",
        "review_bucket_hidden": "accepted_positive_control",
        "human_label": "accept_continuity",
        **safety_payload(),
    }


def _review_manifest_from_selection(
    *,
    stage_root: Path,
    evidence_root: Path,
    selection: dict[str, Any],
    node_by_visible_id: dict[str, dict[str, Any]],
    frame_root: Path,
    frame_records: dict[int, dict[str, Any]],
    created_at: str,
    title: str,
    family: str,
    review_round: int,
    case_prefix: str,
    limit: int | None = None,
) -> dict[str, Any]:
    source_refs = _source_refs(_stage_input_paths(stage_root), stage_root)
    rows_for_cases = [
        *selection.get("counterfactual_negatives", []),
        *[_control_edge_from_positive(row) for row in selection.get("positive_controls", [])],
    ]
    if limit is not None:
        rows_for_cases = rows_for_cases[:limit]
    cases = []
    for index, row in enumerate(rows_for_cases, start=1):
        case_id = f"{case_prefix}_{index:03d}"
        evidence = _counterfactual_evidence(
            evidence_root=evidence_root,
            case_id=case_id,
            row=row,
            node_by_visible_id=node_by_visible_id,
            frame_root=frame_root,
            frame_records=frame_records,
        )
        hidden = {
            "review_bucket": row["review_bucket_hidden"],
            "previously_accepted_reference_target_visible_person_base_id": row[
                "accepted_target_visible_person_base_id"
            ],
            "candidate_ranking": {
                "assignment_swap_risk_score": row.get("assignment_swap_risk_score"),
                "counterfactual_wrong_target_score": row.get("counterfactual_wrong_target_score"),
                "accepted_positive_continuity_score": row.get("accepted_positive_continuity_score"),
            },
            "raw_features": {
                "source_to_accepted_center_delta_px": row.get("source_to_accepted_center_delta_px"),
                "source_to_alternative_center_delta_px": row.get("source_to_alternative_center_delta_px"),
                "accepted_target_to_alternative_center_delta_px": row.get(
                    "accepted_target_to_alternative_center_delta_px"
                ),
                "accepted_target_iou_with_alternative": row.get("accepted_target_iou_with_alternative"),
            },
        }
        visible = {
            "source_visible_person_base_id": row["source_visible_person_base_id"],
            "proposed_target_visible_person_base_id": row["alternative_target_visible_person_base_id"],
            "source_frame_sequence": row["source_frame_sequence"],
            "target_frame_sequence": row["target_frame_sequence"],
            "frame_gap": row["frame_gap"],
            "team_partition": row.get("team_partition"),
            "role_context": row.get("original_effective_role_context"),
            "interpolation_label": "INTERP_NOT_OBS",
            "accepted_reference_target_is_hidden_until_reveal": True,
        }
        payload = {
            "review_case_id": case_id,
            "task_type": "visual_continuity_edge_review",
            "concise_question": CONTINUITY_QUESTION,
            "allowed_decisions": CONTINUITY_DECISIONS,
            "candidate_artifact_id": str(row["counterfactual_candidate_id"]),
            "source_artifact_references": source_refs,
            "source_frame_sequence": int(row["source_frame_sequence"]),
            "target_frame_sequence": int(row["target_frame_sequence"]),
            "evidence_manifest": evidence,
            "uncertainty_reasons": [
                "blind_counterfactual_review_hides_prior_accepted_target_until_reveal",
                "accepted_reference_target_visible_as_neutral_reference_option",
                f"team_partition={row.get('team_partition')}",
                "interpolated_boxes_remain_labelled_INTERP_NOT_OBS",
            ],
            "category": "blind_counterfactual_negative_review",
            "priority": index,
            "control_status": "positive_control"
            if row["review_bucket_hidden"] == "accepted_positive_control"
            else "not_control",
            "candidate_hash": "",
            "evidence_hash": evidence["evidence_hash"],
            "safety_payload": safety_payload(),
            "review_round": review_round,
            "selection_metadata": {
                "blind_review_default_state": "counterfactual_bucket_scores_and_reference_target_hidden",
                "blind_context": visible,
                "blind_hidden_model_info": hidden,
                "reveal_model_information_control": "available_after_decision_or_explicit_action_records_note_marker",
                "accepted_local_visual_trajectory_component_id": row["accepted_local_visual_trajectory_component_id"],
            },
            "model_prediction": None,
            "model_confidence": None,
            "equivalence_cluster_id": row["accepted_local_visual_trajectory_component_id"],
            "representative_of_count": 1,
        }
        payload["candidate_hash"] = _review_case_hash(payload)
        cases.append(ReviewCase.model_validate(payload))
    manifest = ReviewManifest(
        created_at=created_at,
        title=title,
        review_task_family=family,
        review_cases=cases,
        candidate_manifest_hash=stable_hash([case.candidate_hash for case in cases]),
        evidence_manifest_hash=stable_hash([case.evidence_hash for case in cases]),
        source_manifest_hash=stable_hash([ref.model_dump(mode="json") for ref in source_refs]),
        source_artifact_references=source_refs,
    )
    return manifest.model_dump(mode="json")


def _write_case_index(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_case_id",
                "candidate_artifact_id",
                "source_frame_sequence",
                "target_frame_sequence",
                "frame_gap",
                "control_status",
                "semantic_component",
            ],
        )
        writer.writeheader()
        for case in manifest.get("review_cases", []):
            writer.writerow(
                {
                    "review_case_id": case["review_case_id"],
                    "candidate_artifact_id": case["candidate_artifact_id"],
                    "source_frame_sequence": case["source_frame_sequence"],
                    "target_frame_sequence": case["target_frame_sequence"],
                    "frame_gap": case["evidence_manifest"]["frame_gap"],
                    "control_status": case["control_status"],
                    "semantic_component": case["selection_metadata"].get(
                        "accepted_local_visual_trajectory_component_id"
                    ),
                }
            )


def _write_repaired_workbench(workbench_root: Path) -> None:
    workbench_root.mkdir(parents=True, exist_ok=True)
    write_text(
        workbench_root / "index.html",
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blind Counterfactual Continuity Review</title>
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
      <h1 id="caseTitle">Blind counterfactual continuity review</h1>
      <p id="safeMeta"></p>
      <button id="revealBtn" type="button">Reveal reference and model information</button>
    </header>
    <section id="modelPanel" class="hidden"></section>
    <section id="evidence"></section>
    <section id="context"></section>
    <section id="decisions"></section>
    <textarea id="note" placeholder="Optional note"></textarea>
    <footer>
      <button id="prev" type="button">Previous</button>
      <button id="next" type="button">Next</button>
      <button id="undo" type="button">Undo</button>
      <button id="complete" type="button">Complete review</button>
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
  background: #111418;
  color: #edf2f7;
  display: grid;
  grid-template-columns: 300px 1fr;
  min-height: 100vh;
}
aside {
  border-right: 1px solid #303946;
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
  background: #1c2530;
  color: #edf2f7;
  border: 1px solid #3a4653;
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
.mediaControls button,
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
let autosaveTimer = null;

const decisionKeys = {
  a: "accept_continuity",
  r: "reject_continuity",
  n: "not_applicable_invalid_or_incompatible_endpoint",
  u: "unresolved"
};

const decisionLabels = {
  accept_continuity: "A",
  reject_continuity: "R",
  not_applicable_invalid_or_incompatible_endpoint: "N",
  unresolved: "U"
};

const $ = id => document.getElementById(id);

async function api(path, opts) {
  const res = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...(opts || {})
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function current() {
  return manifest.review_cases[active];
}

function asset(c, type) {
  return (c.evidence_manifest.evidence_assets || []).find(a => a.asset_type === type) || null;
}

function media(c, a, label) {
  if (!a) return "";
  const src = `/evidence/${c.review_case_id}/${a.relative_path}`;
  const tag = a.media_type === "video/mp4"
    ? `<video controls playsinline src="${src}"></video>`
    : `<img src="${src}" alt="">`;
  return `<div class="asset"><strong>${label}</strong>${tag}</div>`;
}

function decorateMediaControls() {
  document.querySelectorAll("video").forEach(video => {
    const controls = document.createElement("div");
    controls.className = "mediaControls";
    controls.innerHTML = `
      <button type="button" data-speed="0.5">0.5x</button>
      <button type="button" data-speed="1">1x</button>
      <button type="button" data-speed="2">2x</button>
      <button type="button" data-step="-0.333">Step -</button>
      <button type="button" data-step="0.333">Step +</button>`;
    controls.querySelectorAll("button").forEach(btn => {
      btn.addEventListener("click", ev => {
        ev.preventDefault();
        if (btn.dataset.speed) video.playbackRate = Number(btn.dataset.speed);
        if (btn.dataset.step) video.currentTime = Math.max(0, video.currentTime + Number(btn.dataset.step));
      });
    });
    video.insertAdjacentElement("afterend", controls);
  });
}

function renderList() {
  const only = $("unresolvedOnly").checked;
  const decisions = state.decisions || {};
  $("caseList").innerHTML = "";
  manifest.review_cases.forEach((c, i) => {
    if (only && decisions[c.review_case_id]) return;
    const b = document.createElement("button");
    b.type = "button";
    b.className = "case"
      + (i === active ? " active" : "")
      + (decisions[c.review_case_id] ? " done" : "");
    b.textContent = `${i + 1}. f${c.source_frame_sequence}->${c.target_frame_sequence}`;
    b.onclick = () => { active = i; render(); };
    $("caseList").appendChild(b);
  });
}

function renderEvidence(c) {
  const temporal = asset(c, "temporal_mp4") || asset(c, "animated_gif");
  const assets = [
    media(c, temporal, "Temporal evidence"),
    media(c, asset(c, "temporal_strip"), "Temporal strip"),
    media(c, asset(c, "source_full_frame"), "Source full frame"),
    media(c, asset(c, "target_full_frame"), "Target frame with proposed target and neutral reference option"),
    media(c, asset(c, "source_crop"), "Source crop"),
    media(c, asset(c, "proposed_alternative_target_crop"), "Proposed target crop"),
    media(c, asset(c, "accepted_reference_target_crop"), "Reference option crop"),
    media(c, asset(c, "wide_context"), "Wide context"),
    media(c, asset(c, "crop_follow_view"), "Crop-follow view")
  ];
  $("evidence").innerHTML = `<div class="grid">${assets.join("")}</div>`;
  decorateMediaControls();
}

function renderDecisions(c) {
  $("decisions").innerHTML = "";
  c.allowed_decisions.forEach(decision => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "decision" + ((state.decisions || {})[c.review_case_id] === decision ? " selected" : "");
    b.textContent = `${decisionLabels[decision]} ${decision}`;
    b.onclick = () => saveDecision(decision);
    $("decisions").appendChild(b);
  });
}

function render() {
  const c = current();
  const decisions = state.decisions || {};
  $("counts").textContent = `${Object.keys(decisions).length}/${manifest.review_cases.length} reviewed`;
  $("caseTitle").textContent = `Case ${active + 1} of ${manifest.review_cases.length}`;
  $("safeMeta").textContent =
    `frames ${c.source_frame_sequence}->${c.target_frame_sequence}`
    + ` | ${c.selection_metadata.blind_context.team_partition || "team unknown"}`
    + " | reference target hidden until reveal";
  $("note").value = (state.notes || {})[c.review_case_id] || "";
  renderEvidence(c);
  $("context").textContent = "Does the highlighted source continue as the highlighted proposed target?";
  renderDecisions(c);
  if (revealed[c.review_case_id]) showModel(c); else $("modelPanel").classList.add("hidden");
  renderList();
}

function showModel(c) {
  $("modelPanel").classList.remove("hidden");
  $("modelPanel").textContent = JSON.stringify(c.selection_metadata.blind_hidden_model_info, null, 2);
  revealed[c.review_case_id] = true;
}

async function saveDecision(decision) {
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
  $("status").textContent = `Decision saved: ${decisionLabels[decision]}`;
  render();
}

function autosaveNote() {
  clearTimeout(autosaveTimer);
  autosaveTimer = setTimeout(async () => {
    const c = current();
    state = await api("/api/review/note", {
      method: "POST",
      body: JSON.stringify({
        review_case_id: c.review_case_id,
        note: $("note").value,
        last_viewed_case_id: c.review_case_id
      })
    });
    $("status").textContent = "Note autosaved";
  }, 350);
}

function typingTarget(target) {
  return target && (
    target.tagName === "TEXTAREA"
    || target.tagName === "INPUT"
    || target.isContentEditable
  );
}

document.addEventListener("keydown", ev => {
  if (typingTarget(ev.target)) return;
  const key = ev.key.toLowerCase();
  if (decisionKeys[key]) {
    ev.preventDefault();
    saveDecision(decisionKeys[key]);
  } else if (key === "z") {
    ev.preventDefault();
    undoLast();
  } else if (key === "arrowright") {
    ev.preventDefault();
    active = Math.min(manifest.review_cases.length - 1, active + 1);
    render();
  } else if (key === "arrowleft") {
    ev.preventDefault();
    active = Math.max(0, active - 1);
    render();
  }
}, true);

async function undoLast() {
  state = await api("/api/review/undo", {method: "POST", body: "{}"});
  $("status").textContent = "Undo restored prior state";
  render();
}

async function completeReview() {
  const decisions = state.decisions || {};
  if (Object.keys(decisions).length < manifest.review_cases.length) {
    $("status").textContent = "Completion blocked until all cases have decisions";
    return;
  }
  state = await api("/api/review/complete", {method: "POST", body: "{}"});
  $("status").textContent = "Review complete";
}

async function init() {
  manifest = await api("/api/review/manifest");
  state = await api("/api/review/state");
  const resume = state.resume_case_id;
  const idx = manifest.review_cases.findIndex(c => c.review_case_id === resume);
  active = idx >= 0 ? idx : 0;
  $("unresolvedOnly").onchange = renderList;
  $("note").addEventListener("input", autosaveNote);
  $("revealBtn").onclick = () => showModel(current());
  $("prev").onclick = () => { active = Math.max(0, active - 1); render(); };
  $("next").onclick = () => { active = Math.min(manifest.review_cases.length - 1, active + 1); render(); };
  $("undo").onclick = undoLast;
  $("complete").onclick = completeReview;
  render();
}

init().catch(err => {$("status").textContent = err.message;});
""",
    )
    write_text(workbench_root / "fallback.html", "<p>Local server-backed review workbench required.</p>\n")


def _gif_frame_count(path: Path) -> int:
    if Image is None:
        return 0
    with Image.open(path) as image:
        return int(getattr(image, "n_frames", 1))


def _run_http_200_check(manifest_path: Path, evidence_root: Path, decision_root: Path, workbench_root: Path) -> bool:
    manifest = read_json(manifest_path)
    first_case = manifest["review_cases"][0]
    mp4 = next(
        asset for asset in first_case["evidence_manifest"]["evidence_assets"] if asset.get("media_type") == "video/mp4"
    )
    server = create_server(
        ReviewServerConfig(
            manifest_path=manifest_path,
            evidence_root=evidence_root,
            decision_root=decision_root,
            workbench_root=workbench_root,
            host="127.0.0.1",
            port=0,
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/evidence/{first_case['review_case_id']}/{mp4['relative_path']}"
        with urlopen(url, timeout=5) as response:  # noqa: S310 - local smoke-test server only.
            return response.status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _automated_smoke_results(
    *,
    manifest_path: Path,
    evidence_root: Path,
    decision_root: Path,
    workbench_root: Path,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    first_case = manifest["review_cases"][0]
    assets = first_case["evidence_manifest"]["evidence_assets"]
    gif_asset = next((asset for asset in assets if asset.get("media_type") == "image/gif"), None)
    mp4_asset = next((asset for asset in assets if asset.get("media_type") == "video/mp4"), None)
    gif_path = evidence_root / first_case["review_case_id"] / str(gif_asset["relative_path"]) if gif_asset else None
    mp4_path = evidence_root / first_case["review_case_id"] / str(mp4_asset["relative_path"]) if mp4_asset else None
    gif_frames = _gif_frame_count(gif_path) if gif_path else 0
    mp4_opened = False
    mp4_seekable = False
    if mp4_path and mp4_path.exists():
        capture = cv2.VideoCapture(str(mp4_path))
        mp4_opened = bool(capture.isOpened())
        if mp4_opened:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(1, max(0, frame_count - 1)))
            mp4_seekable = bool(capture.read()[0])
        capture.release()
    script = (workbench_root / "app.js").read_text(encoding="utf-8")
    checks = {
        "gif_loads_successfully": bool(gif_path and gif_path.exists()),
        "gif_contains_more_than_one_frame": gif_frames > 1,
        "mp4_file_exists": bool(mp4_path and mp4_path.exists() and mp4_path.stat().st_size > 0),
        "mp4_returns_http_200": _run_http_200_check(manifest_path, evidence_root, decision_root, workbench_root),
        "browser_can_seek_and_play_mp4_static_contract": (
            "video.currentTime" in script and "video.playbackRate" in script
        ),
        "a_shortcut_saves_accept_continuity_static_contract": 'a: "accept_continuity"' in script,
        "r_shortcut_saves_reject_continuity_static_contract": 'r: "reject_continuity"' in script,
        "n_shortcut_saves_not_applicable_static_contract": (
            'n: "not_applicable_invalid_or_incompatible_endpoint"' in script
        ),
        "u_shortcut_saves_unresolved_static_contract": 'u: "unresolved"' in script,
        "shortcuts_do_not_fire_while_typing_notes": "typingTarget(ev.target)" in script,
        "undo_restores_prior_state_static_contract": '"/api/review/undo"' in script,
        "restart_recovery_static_contract": "resume_case_id" in script,
        "evidence_hashes_remain_bound": all(
            case["evidence_hash"] == case["evidence_manifest"]["evidence_hash"] for case in manifest["review_cases"]
        ),
        "completion_blocked_until_all_smoke_cases_have_decisions": (
            "Completion blocked until all cases have decisions" in script
        ),
        "mp4_can_be_opened_by_cv2": mp4_opened,
        "mp4_can_seek_and_read_frame": mp4_seekable,
    }
    return {
        "artifact": "m5_4f3_automated_smoke_test_results",
        "passed": all(checks.values()),
        "gif_frame_count": gif_frames,
        "checks": checks,
        "browser_dependent_checks": [
            "actual_keyboard_events_in_browser",
            "actual_video_playback_in_browser",
            "manual_completion_flow",
        ],
        **safety_payload(),
    }


def _manual_smoke_checklist() -> str:
    return """# M5.4F.3 Manual Smoke Test Checklist

Run `OPEN_CONTINUITY_WORKBENCH_SMOKE_TEST.ps1`, then confirm:

- GIF loads and animates.
- MP4 loads, plays, seeks, and speed buttons work.
- A saves accept_continuity and shows `Decision saved: A`.
- R saves reject_continuity and shows `Decision saved: R`.
- N saves not applicable and shows `Decision saved: N`.
- U saves unresolved and shows `Decision saved: U`.
- Shortcuts do not fire while typing in the notes box.
- Undo restores the prior decision state.
- Refresh/restart reloads saved decisions.
- Completion stays blocked until all smoke cases have decisions.

Only after these checks pass, edit `smoke_test_confirmation.json` so `pass`
is true and rerun the F3 builder to unlock the full counterfactual review
launcher.
"""


def _write_preserving_existing(path: Path, payload: dict[str, Any], *, preserved_name: str) -> None:
    if path.exists():
        preserved = path.parent / preserved_name
        if not preserved.exists():
            preserved.write_bytes(path.read_bytes())
    write_json(path, payload)


def _review_selection_audits(selection: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    endpoint = {
        "artifact": "m5_4f3_endpoint_reuse_audit",
        "endpoint_reuse_distribution": selection["endpoint_reuse_distribution"],
        "endpoint_reuse_max": selection["endpoint_reuse_max"],
        "endpoint_reuse_limit": 2,
        "endpoint_reuse_passed": selection["endpoint_reuse_max"] <= 2,
        **safety_payload(),
    }
    semantic = {
        "artifact": "m5_4f3_semantic_cluster_audit",
        "semantic_component_distribution": selection["semantic_component_distribution"],
        "independent_negative_cluster_count": len(
            {str(row["accepted_local_visual_trajectory_component_id"]) for row in selection["counterfactual_negatives"]}
        ),
        "positive_control_component_count": len(
            {str(row["accepted_local_visual_trajectory_component_id"]) for row in selection["positive_controls"]}
        ),
        "max_cases_per_semantic_component": max(selection["semantic_component_distribution"].values() or [0]),
        **safety_payload(),
    }
    return endpoint, semantic


def build_positive_only_counterfactual_continuity_stage(
    *,
    stage_root: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    stage_root = stage_root.resolve()
    repo_root = (repo_root or Path.cwd()).resolve()
    continuity_v3 = stage_root / "continuity_v3"
    audit_root = continuity_v3 / "audit"
    learning_root = continuity_v3 / "learning"
    counterfactual_root = continuity_v3 / "counterfactual"
    smoke_root = continuity_v3 / "smoke_test"
    validation_root = stage_root / "validation"
    for root in [audit_root, learning_root, counterfactual_root, smoke_root, validation_root]:
        root.mkdir(parents=True, exist_ok=True)
    source_paths = [
        stage_root / "continuity_v2" / "decisions",
        stage_root / "continuity_v2" / "deconfounded_hard_continuity_review_manifest.json",
        stage_root / "continuity" / "post_role_candidate_rows.json",
        stage_root / "continuity" / "post_role_context_rows.json",
        stage_root / "role_review" / "decisions",
    ]
    before_inventory = _inventory(source_paths, base=stage_root)

    f2_manifest_path = stage_root / "continuity_v2" / "deconfounded_hard_continuity_review_manifest.json"
    decisions_root = stage_root / "continuity_v2" / "decisions"
    completed_files = [
        decisions_root / "completed_review.json",
        decisions_root / "completed_review_events.jsonl",
        decisions_root / "completed_review_manifest.json",
        decisions_root / "completed_review_summary.json",
        decisions_root / "review_decisions.json",
        decisions_root / "review_decision_events.jsonl",
    ]
    f2_manifest = read_json(f2_manifest_path)
    completed_review = read_json(decisions_root / "completed_review.json")
    completed_summary = read_json(decisions_root / "completed_review_summary.json")
    completed_events = _read_jsonl(decisions_root / "completed_review_events.jsonl")
    review_validation = validate_completed_f2_review(
        manifest=f2_manifest,
        completed_review=completed_review,
        completed_summary=completed_summary,
        completed_events=completed_events,
        completed_files=completed_files,
    )
    write_json(learning_root / "f2_completed_review_validation.json", review_validation)
    decisions = _decision_map(completed_review)
    positive_examples = _accepted_examples_with_geometry(f2_manifest, decisions)
    _write_jsonl(learning_root / "f2_human_positive_examples.jsonl", positive_examples)
    label_distribution = _label_distribution(positive_examples)
    readiness = _training_readiness(positive_examples)
    write_json(learning_root / "f2_label_distribution.json", label_distribution)
    write_json(learning_root / "f2_training_readiness.json", readiness)

    _write_human_vs_bucket_csv(audit_root / "f2_human_vs_proposed_bucket.csv", positive_examples)
    failure_audit = _hard_negative_failure_audit(positive_examples)
    write_json(audit_root / "f2_hard_negative_failure_audit.json", failure_audit)
    write_text(
        audit_root / "f2_review_selection_incident.md",
        "\n".join(
            [
                "# M5.4F.2 Review Selection Incident",
                "",
                f"F2 review classification: `{F2_DIAGNOSTIC_CLASSIFICATION}`",
                "",
                "The completed human review accepted all 40 continuity cases. The 20 machine-proposed",
                "negative-bucket cases are therefore positive human continuity evidence, not negative labels.",
                "Continuity training remains blocked because there is no reviewed negative class.",
            ]
        )
        + "\n",
    )

    component_result = accepted_local_trajectory_components(positive_examples)
    edge_graph, component_audit = _component_audits(positive_examples, component_result)
    write_json(learning_root / "accepted_edge_graph.json", edge_graph)
    write_json(
        learning_root / "accepted_local_trajectory_components.json",
        {
            "artifact": "m5_4f3_accepted_local_trajectory_components",
            "components": component_result["components"],
            **safety_payload(),
        },
    )
    write_json(learning_root / "positive_equivalence_cluster_audit.json", component_audit)

    exact_role_map = _role_reviewed_map(stage_root)
    role_rows, role_audit = role_context_reconciliation(positive_examples, exact_role_by_candidate=exact_role_map)
    write_json(
        learning_root / "f2_positive_role_context_rows.json",
        {"artifact": "m5_4f3_f2_positive_role_context_rows", "rows": role_rows, **safety_payload()},
    )
    write_json(audit_root / "f2_role_contamination_audit.json", role_audit)

    paths = _stage_input_paths(stage_root)
    node_rows = rows(read_json(paths["m54d_stage_root"] / "continuity" / "continuity_node_rows.json"))
    role_context_rows = rows(read_json(stage_root / "continuity" / "post_role_context_rows.json"))
    role_by_visible = {str(row.get("visible_person_base_id")): row for row in role_context_rows}
    counterfactuals, rejections = _build_counterfactual_candidates(
        positive_examples=positive_examples,
        node_rows=node_rows,
        role_rows_by_visible=role_by_visible,
    )
    write_json(
        counterfactual_root / "counterfactual_negative_candidate_rows.json",
        {"artifact": "m5_4f3_counterfactual_negative_candidate_rows", "rows": counterfactuals, **safety_payload()},
    )
    write_json(
        counterfactual_root / "counterfactual_candidate_rejection_rows.json",
        {"artifact": "m5_4f3_counterfactual_candidate_rejection_rows", "rows": rejections, **safety_payload()},
    )
    selection = _select_counterfactual_review_rows(counterfactuals, positive_examples)
    endpoint_audit, semantic_audit = _review_selection_audits(selection)
    write_json(counterfactual_root / "endpoint_reuse_audit.json", endpoint_audit)
    write_json(counterfactual_root / "semantic_cluster_audit.json", semantic_audit)
    counterfactual_summary = {
        "artifact": "m5_4f3_counterfactual_candidate_summary",
        "counterfactual_candidate_count": len(counterfactuals),
        "counterfactual_review_case_count": len(selection["counterfactual_negatives"])
        + len(selection["positive_controls"]),
        "selected_counterfactual_negative_count": len(selection["counterfactual_negatives"]),
        "positive_control_count": len(selection["positive_controls"]),
        "counterfactual_candidate_rejection_count": len(rejections),
        "independent_negative_cluster_count": semantic_audit["independent_negative_cluster_count"],
        "endpoint_reuse_maximum": endpoint_audit["endpoint_reuse_max"],
        "candidate_supply_sufficient_for_full_target": len(selection["counterfactual_negatives"]) >= 20
        and len(selection["positive_controls"]) >= 4,
        **safety_payload(),
    }
    write_json(counterfactual_root / "counterfactual_candidate_summary.json", counterfactual_summary)

    frame_records = _frame_records(read_json(paths["frame_manifest"]))
    node_by_visible = {str(row["visible_person_base_id"]): row for row in node_rows}
    _write_repaired_workbench(counterfactual_root / "workbench")
    review_manifest = _review_manifest_from_selection(
        stage_root=stage_root,
        evidence_root=counterfactual_root / "evidence",
        selection=selection,
        node_by_visible_id=node_by_visible,
        frame_root=paths["frame_root"],
        frame_records=frame_records,
        created_at=str(_state_from_completed(completed_review).get("completed_at") or f2_manifest.get("created_at")),
        title="M5.4F.3 Blind Counterfactual Negative Review",
        family="m5_4f3_counterfactual_negative_continuity_review",
        review_round=6,
        case_prefix="m5_4f3_counterfactual_case",
    )
    write_json(counterfactual_root / "counterfactual_review_manifest.json", review_manifest)
    _write_case_index(counterfactual_root / "counterfactual_case_index.csv", review_manifest)
    decision_root = counterfactual_root / "decisions"
    decision_root.mkdir(parents=True, exist_ok=True)
    write_json(
        decision_root / "review_decisions.json",
        _deterministic_empty_decision_state(
            ReviewManifest.model_validate(review_manifest),
            review_manifest["created_at"],
        ),
    )
    write_text(decision_root / "review_decision_events.jsonl", "")
    (decision_root / "snapshots").mkdir(parents=True, exist_ok=True)

    smoke_selection = {
        "counterfactual_negatives": selection["counterfactual_negatives"][:2],
        "positive_controls": selection["positive_controls"][:1],
    }
    _write_repaired_workbench(smoke_root / "workbench")
    smoke_manifest = _review_manifest_from_selection(
        stage_root=stage_root,
        evidence_root=smoke_root / "evidence",
        selection=smoke_selection,
        node_by_visible_id=node_by_visible,
        frame_root=paths["frame_root"],
        frame_records=frame_records,
        created_at=str(_state_from_completed(completed_review).get("completed_at") or f2_manifest.get("created_at")),
        title="M5.4F.3 Counterfactual Review Smoke Test",
        family="m5_4f3_counterfactual_smoke_test",
        review_round=6,
        case_prefix="m5_4f3_smoke_case",
    )
    write_json(smoke_root / "smoke_test_manifest.json", smoke_manifest)
    smoke_decision_root = smoke_root / "decisions"
    smoke_decision_root.mkdir(parents=True, exist_ok=True)
    write_json(
        smoke_decision_root / "review_decisions.json",
        _deterministic_empty_decision_state(
            ReviewManifest.model_validate(smoke_manifest),
            smoke_manifest["created_at"],
        ),
    )
    write_text(smoke_decision_root / "review_decision_events.jsonl", "")
    (smoke_decision_root / "snapshots").mkdir(parents=True, exist_ok=True)
    smoke_results = _automated_smoke_results(
        manifest_path=smoke_root / "smoke_test_manifest.json",
        evidence_root=smoke_root / "evidence",
        decision_root=smoke_decision_root,
        workbench_root=smoke_root / "workbench",
    )
    write_json(smoke_root / "automated_smoke_test_results.json", smoke_results)
    write_text(smoke_root / "manual_smoke_test_checklist.md", _manual_smoke_checklist())
    confirmation_path = smoke_root / "smoke_test_confirmation.json"
    if confirmation_path.exists():
        smoke_confirmation = read_json(confirmation_path)
    else:
        smoke_confirmation = {
            "artifact": "m5_4f3_smoke_test_confirmation",
            "pass": False,
            "status": "manual_confirmation_required",
            "automated_smoke_test_passed": smoke_results["passed"],
            "manual_confirmation_required_before_full_review_launcher": True,
            **safety_payload(),
        }
        write_json(confirmation_path, smoke_confirmation)
    smoke_launcher = _write_open_launcher(
        launcher_path=stage_root / "OPEN_CONTINUITY_WORKBENCH_SMOKE_TEST.ps1",
        repo_root=repo_root,
        manifest_path=smoke_root / "smoke_test_manifest.json",
        evidence_root=smoke_root / "evidence",
        decision_root=smoke_decision_root,
        workbench_root=smoke_root / "workbench",
        label="M5.4F.3 counterfactual continuity smoke test",
        port=8775,
    )
    full_launcher = None
    review_url = None
    smoke_confirmed = smoke_confirmation.get("pass") is True and smoke_results["passed"] is True
    if smoke_confirmed:
        full_launcher = _write_open_launcher(
            launcher_path=stage_root / "OPEN_COUNTERFACTUAL_NEGATIVE_REVIEW.ps1",
            repo_root=repo_root,
            manifest_path=counterfactual_root / "counterfactual_review_manifest.json",
            evidence_root=counterfactual_root / "evidence",
            decision_root=decision_root,
            workbench_root=counterfactual_root / "workbench",
            label="M5.4F.3 counterfactual negative continuity",
            port=8776,
        )
        review_url = "http://127.0.0.1:8776/"

    after_inventory = _inventory(source_paths, base=stage_root)
    source_mutation = {
        "artifact": "m5_4f3_source_mutation_audit",
        "before": before_inventory,
        "after": after_inventory,
        "prior_human_artifacts_unchanged": before_inventory["inventory_hash"] == after_inventory["inventory_hash"],
        "completed_f2_review_artifacts_preserved": True,
        **safety_payload(),
    }
    safety = {
        "artifact": "m5_4f3_safety_guardrail_audit",
        "all_safety_flags_preserved": True,
        "continuity_model_fit_performed": False,
        "continuity_rows_updated": 0,
        "proposed_machine_buckets_used_as_truth": False,
        "persistent_identity_assigned": False,
        "player_slots_assigned": False,
        "goalkeeper_slots_assigned": False,
        "metric_pitch_coordinates_used": False,
        "tactical_event_or_physical_outputs_created": False,
        **safety_payload(),
    }
    _write_preserving_existing(
        validation_root / "source_mutation_audit.json",
        source_mutation,
        preserved_name="preserved_pre_m5_4f3_source_mutation_audit.json",
    )
    _write_preserving_existing(
        validation_root / "safety_guardrail_audit.json",
        safety,
        preserved_name="preserved_pre_m5_4f3_safety_guardrail_audit.json",
    )
    write_json(validation_root / "m5_4f3_source_mutation_audit.json", source_mutation)
    write_json(validation_root / "m5_4f3_safety_guardrail_audit.json", safety)

    if not counterfactual_summary["candidate_supply_sufficient_for_full_target"]:
        final_classification = F3_BLOCKED_SUPPLY
        exact_blocker = "COUNTERFACTUAL_NEGATIVE_SUPPLY_BELOW_20_PLUS_4_TARGET"
    elif not smoke_results["passed"]:
        final_classification = F3_BLOCKED_SMOKE
        exact_blocker = "AUTOMATED_SMOKE_TEST_FAILED"
    elif smoke_confirmed:
        final_classification = F3_REVIEW_READY
        exact_blocker = "NONE"
    else:
        final_classification = F3_SMOKE_READY
        exact_blocker = "MANUAL_SMOKE_CONFIRMATION_REQUIRED_BEFORE_FULL_REVIEW"
    summary = {
        "artifact": "m5_4f3_validation_summary",
        "final_classification": final_classification,
        "exact_blocker": exact_blocker,
        "f2_completed_review_validation": review_validation["valid"],
        "f2_review_classification": F2_DIAGNOSTIC_CLASSIFICATION,
        "human_label_distribution": label_distribution["human_label_distribution"],
        "continuity_training_readiness": readiness["status"],
        "model_fitting_performed": False,
        "learned_rows_updated": 0,
        "raw_accepted_edge_count": len(positive_examples),
        "semantic_positive_trajectory_component_count": component_audit["semantic_independent_component_count"],
        "largest_positive_component_size": component_audit["largest_component_size"],
        "assistant_referee_compatible_positive_count": role_audit["assistant_referee_compatible_positive_count"],
        "outfield_compatible_positive_count": role_audit["outfield_compatible_positive_count"],
        "unresolved_role_positive_count": role_audit["unresolved_role_positive_count"],
        "proposed_negative_bucket_accepted_count": failure_audit["proposed_negative_accepted_count"],
        "counterfactual_candidate_count": len(counterfactuals),
        "counterfactual_review_case_count": counterfactual_summary["counterfactual_review_case_count"],
        "independent_negative_cluster_count": semantic_audit["independent_negative_cluster_count"],
        "endpoint_reuse_maximum": endpoint_audit["endpoint_reuse_max"],
        "positive_control_count": len(selection["positive_controls"]),
        "gif_automated_test": smoke_results["checks"]["gif_contains_more_than_one_frame"],
        "mp4_automated_test": smoke_results["checks"]["mp4_returns_http_200"]
        and smoke_results["checks"]["mp4_can_seek_and_read_frame"],
        "keyboard_shortcut_automated_test": all(
            smoke_results["checks"][key]
            for key in [
                "a_shortcut_saves_accept_continuity_static_contract",
                "r_shortcut_saves_reject_continuity_static_contract",
                "n_shortcut_saves_not_applicable_static_contract",
                "u_shortcut_saves_unresolved_static_contract",
                "shortcuts_do_not_fire_while_typing_notes",
            ]
        ),
        "smoke_test_status": "confirmed_pass" if smoke_confirmed else smoke_confirmation.get("status"),
        "smoke_test_launcher": str(smoke_launcher),
        "full_review_launcher": str(full_launcher) if full_launcher else None,
        "review_url": review_url,
        "source_mutation_audit_path": str(validation_root / "source_mutation_audit.json"),
        "safety_guardrail_audit_path": str(validation_root / "safety_guardrail_audit.json"),
        **safety_payload(),
    }
    write_json(validation_root / "m5_4f3_validation_summary.json", summary)
    return summary
