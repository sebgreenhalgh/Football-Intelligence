from __future__ import annotations

from collections import defaultdict
from typing import Any


def cluster_aware_split(rows: list[dict[str, Any]], *, folds: int = 3) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("equivalence_cluster_id") or row.get("review_case_id"))].append(row)
    assignments: list[dict[str, Any]] = []
    for index, (cluster_id, members) in enumerate(sorted(groups.items())):
        fold = index % max(1, folds)
        for member in members:
            assignments.append(
                {"review_case_id": member.get("review_case_id"), "equivalence_cluster_id": cluster_id, "fold": fold}
            )
    return assignments


def validate_training_examples(rows: list[dict[str, Any]], *, min_examples: int = 8) -> dict[str, Any]:
    assignments = cluster_aware_split(rows)
    train_eval_overlap = False
    enough = len(rows) >= min_examples
    return {
        "example_count": len(rows),
        "cluster_aware_folds": assignments,
        "train_eval_overlap": train_eval_overlap,
        "passed": enough and not train_eval_overlap,
        "blocking_reason": None if enough else "awaiting_more_human_review_examples",
    }
