from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_review_state(decision_root: Path) -> dict[str, Any]:
    path = decision_root / "review_decisions.json"
    if not path.exists():
        return {"decisions": {}, "notes": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"decisions": {}, "notes": {}}


def examples_from_review_manifest(
    *,
    manifest: dict[str, Any],
    decision_state: dict[str, Any],
    feature_by_candidate: dict[str, dict[str, Any]],
    edge_by_candidate: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions = decision_state.get("decisions", {}) if isinstance(decision_state.get("decisions"), dict) else {}
    entity_examples: list[dict[str, Any]] = []
    continuity_examples: list[dict[str, Any]] = []
    for case in manifest.get("review_cases", []):
        case_id = case.get("review_case_id")
        if case_id not in decisions:
            continue
        label = decisions[case_id]
        candidate_id = str(case.get("candidate_artifact_id"))
        common = {
            "review_case_id": case_id,
            "candidate_hash": case.get("candidate_hash"),
            "evidence_hash": case.get("evidence_hash"),
            "human_label": label,
            "reviewer_confidence": "human_selected",
            "round_number": case.get("review_round"),
            "equivalence_cluster_id": case.get("equivalence_cluster_id"),
        }
        if case.get("task_type") == "entity_validity":
            entity_examples.append(
                {**common, "candidate_id": candidate_id, "feature_snapshot": feature_by_candidate.get(candidate_id, {})}
            )
        elif case.get("task_type") == "visual_continuity_edge_review":
            continuity_examples.append(
                {**common, "edge_id": candidate_id, "feature_snapshot": edge_by_candidate.get(candidate_id, {})}
            )
    return entity_examples, continuity_examples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
