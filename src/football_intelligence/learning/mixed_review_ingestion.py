from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ENTITY_LABELS = [
    "valid_on_pitch_person",
    "valid_official",
    "valid_off_pitch_person",
    "non_person_false_positive",
    "unresolved",
]
CONTINUITY_LABELS = [
    "accept_continuity",
    "reject_continuity",
    "unresolved",
    "not_applicable_invalid_or_incompatible_endpoint",
    "not_applicable_invalid_endpoint",
]


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _event_metadata(event_log_path: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    if not event_log_path.exists():
        return metadata
    for line in event_log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event_type") != "decision":
            continue
        case_id = str(event.get("review_case_id") or "")
        if case_id:
            metadata[case_id] = {
                "reviewed_at": event.get("timestamp"),
                "reviewer_session_id": event.get("reviewer_session_id"),
                "event_id": event.get("event_id"),
                "event_sequence": event.get("event_sequence"),
            }
    return metadata


def _decision_state(completed_review: dict[str, Any]) -> dict[str, Any]:
    state = completed_review.get("state") if isinstance(completed_review.get("state"), dict) else completed_review
    decisions = state.get("decisions")
    if not isinstance(decisions, dict):
        raise ValueError("completed review does not contain a decisions map")
    return state


def _normalise_continuity_label(case: dict[str, Any], decision: str, note: str) -> str:
    if decision == "not_applicable_invalid_or_incompatible_endpoint":
        return decision
    if decision != "unresolved":
        return decision
    text = f"{note} {' '.join(case.get('uncertainty_reasons', []))}".lower()
    if "invalid endpoint" in text or "not applicable" in text or "non-person" in text:
        return "not_applicable_invalid_or_incompatible_endpoint"
    return decision


def ingest_mixed_review(
    *,
    completed_review_path: Path,
    review_manifest_path: Path,
    event_log_path: Path,
    requested_manifest_path: Path | None = None,
) -> dict[str, Any]:
    completed = read_json(completed_review_path)
    manifest = read_json(review_manifest_path)
    state = _decision_state(completed)
    decisions = state["decisions"]
    notes = state.get("notes", {}) if isinstance(state.get("notes"), dict) else {}
    events = _event_metadata(event_log_path)
    cases = {str(case["review_case_id"]): case for case in manifest.get("review_cases", [])}
    examples: list[dict[str, Any]] = []
    binding_errors: list[dict[str, Any]] = []
    for case_id, decision in sorted(decisions.items()):
        case = cases.get(str(case_id))
        if case is None:
            binding_errors.append({"review_case_id": case_id, "error": "decision_without_manifest_case"})
            continue
        task_type = str(case.get("task_type"))
        note = str(notes.get(case_id, ""))
        human_decision = str(decision)
        label = human_decision
        if task_type == "visual_continuity_edge_review":
            label = _normalise_continuity_label(case, human_decision, note)
        usable = label != "unresolved"
        if task_type == "visual_continuity_edge_review" and label.startswith("not_applicable_invalid"):
            usable = False
        if task_type not in {"entity_validity", "visual_continuity_edge_review"}:
            usable = False
        examples.append(
            {
                "review_case_id": case_id,
                "task_type": task_type,
                "candidate_artifact_id": case.get("candidate_artifact_id"),
                "candidate_hash": case.get("candidate_hash"),
                "evidence_hash": case.get("evidence_hash"),
                "human_decision": human_decision,
                "normalized_training_label": label,
                "note": note,
                "review_round": case.get("review_round"),
                "source_frame_sequence": case.get("source_frame_sequence"),
                "target_frame_sequence": case.get("target_frame_sequence"),
                "category": case.get("category"),
                "equivalence_cluster_id": case.get("equivalence_cluster_id"),
                "selection_metadata": case.get("selection_metadata")
                if isinstance(case.get("selection_metadata"), dict)
                else {},
                "model_prediction_before_review": case.get("model_prediction"),
                "model_confidence_before_review": case.get("model_confidence"),
                "reviewed_at": events.get(case_id, {}).get("reviewed_at")
                or state.get("completed_at")
                or completed.get("created_at"),
                "reviewer_session_id": events.get(case_id, {}).get("reviewer_session_id")
                or state.get("reviewer_session_id")
                or completed.get("reviewer_session_id"),
                "label_usable_for_training": usable,
                "exclusion_reason": None if usable else _exclusion_reason(task_type, label),
            }
        )
    entity_examples = [row for row in examples if row["task_type"] == "entity_validity"]
    continuity_examples = [row for row in examples if row["task_type"] == "visual_continuity_edge_review"]
    distribution = label_distribution(entity_examples, continuity_examples)
    return {
        "artifact": "m5_4e_mixed_review_ingestion",
        "examples": examples,
        "entity_examples": entity_examples,
        "continuity_examples": continuity_examples,
        "distribution": distribution,
        "binding_validation": {
            "artifact": "m5_4e_round_1_review_binding_validation",
            "completed_review_path": str(completed_review_path),
            "review_manifest_path": str(review_manifest_path),
            "requested_manifest_path": str(requested_manifest_path) if requested_manifest_path else None,
            "requested_manifest_path_exists": requested_manifest_path.exists() if requested_manifest_path else None,
            "used_canonical_manifest_because_requested_missing": bool(
                requested_manifest_path and not requested_manifest_path.exists()
            ),
            "decision_count": len(decisions),
            "manifest_case_count": len(cases),
            "bound_case_count": len(examples),
            "binding_error_count": len(binding_errors),
            "binding_errors": binding_errors,
            "summary_counters_used_for_label_inventory": False,
            "passed": not binding_errors and len(examples) == len(decisions),
        },
    }


def _exclusion_reason(task_type: str, label: str) -> str:
    if label == "unresolved":
        return "unresolved_label"
    if label.startswith("not_applicable_invalid"):
        return "continuity_not_applicable_invalid_endpoint"
    if task_type not in {"entity_validity", "visual_continuity_edge_review"}:
        return "unsupported_task_type_for_training"
    return "not_training_usable"


def label_distribution(
    entity_examples: list[dict[str, Any]],
    continuity_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    entity_counts = Counter(row["normalized_training_label"] for row in entity_examples)
    continuity_counts = Counter(row["normalized_training_label"] for row in continuity_examples)
    usable = sum(1 for row in entity_examples + continuity_examples if row["label_usable_for_training"])
    excluded = len(entity_examples) + len(continuity_examples) - usable
    return {
        "artifact": "m5_4e_round_1_label_distribution",
        "entity_case_count": len(entity_examples),
        "continuity_case_count": len(continuity_examples),
        "entity_label_distribution": {label: entity_counts.get(label, 0) for label in ENTITY_LABELS},
        "continuity_label_distribution": {label: continuity_counts.get(label, 0) for label in CONTINUITY_LABELS},
        "decision_counts_by_task_type": {
            "entity_validity": len(entity_examples),
            "visual_continuity_edge_review": len(continuity_examples),
        },
        "decision_counts_by_label": dict(sorted((entity_counts + continuity_counts).items())),
        "training_usable_count": usable,
        "training_excluded_count": excluded,
        "summary_counters_used_for_label_inventory": False,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
