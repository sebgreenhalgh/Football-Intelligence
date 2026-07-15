"""Deterministic counterbalancing for anonymous multi-hypothesis review."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Sequence


def _stable_key(seed: str, position: int, case_id: str) -> str:
    """Keep balancing independent of candidate IDs and answer metadata."""
    payload = f"{seed}\0{position}\0{case_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _permutations_for_count(count: int) -> tuple[tuple[int, ...], ...]:
    if count == 2:
        return ((0, 1), (1, 0))
    if count == 3:
        return ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    raise ValueError("only two- and three-hypothesis cases can be blinded")


def assign_counterbalanced_permutations(
    cases: Sequence[dict[str, Any]], *, stage_seed: str
) -> dict[str, tuple[int, ...]]:
    """Assign globally balanced permutations to an ordered frozen case list.

    The input deliberately contains only anonymous case identity and hypothesis
    count.  In particular, target IDs, scores, ranks and human decisions cannot
    influence the permutation assignment.
    """
    if not cases:
        return {}
    seen: set[str] = set()
    by_count: dict[int, list[tuple[int, str]]] = {2: [], 3: []}
    for position, case in enumerate(cases):
        case_id = str(case["case_id"])
        count = int(case["hypothesis_count"])
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        if count not in by_count:
            raise ValueError(f"unsupported hypothesis count for {case_id}: {count}")
        seen.add(case_id)
        by_count[count].append((position, case_id))

    assignments: dict[str, tuple[int, ...]] = {}
    for count, rows in by_count.items():
        options = _permutations_for_count(count)
        ordered = sorted(rows, key=lambda row: _stable_key(stage_seed, row[0], row[1]))
        for ordinal, (_, case_id) in enumerate(ordered):
            assignments[case_id] = options[ordinal % len(options)]
    return assignments


def apply_permutation(frozen_hypotheses: Sequence[dict[str, Any]], permutation: Sequence[int]) -> list[dict[str, Any]]:
    """Return displayed hypotheses in A/B/C order without changing their data."""
    if len(frozen_hypotheses) != len(permutation):
        raise ValueError("permutation length must match frozen hypothesis count")
    if sorted(permutation) != list(range(len(frozen_hypotheses))):
        raise ValueError("permutation must contain every frozen hypothesis exactly once")
    return [dict(frozen_hypotheses[index]) for index in permutation]


def audit_counterbalance(cases: Sequence[dict[str, Any]], assignments: dict[str, Sequence[int]]) -> dict[str, Any]:
    """Summarize balance without exposing the case-level mapping."""
    by_count: dict[str, dict[str, int]] = {}
    for count in (2, 3):
        labels = {index: 0 for index in range(count)}
        rows = [case for case in cases if int(case["hypothesis_count"]) == count]
        for case in rows:
            permutation = assignments[str(case["case_id"])]
            labels[int(permutation[0])] += 1
        counts = {f"frozen_rank_{index + 1}_as_A": value for index, value in labels.items()}
        by_count[str(count)] = {
            "case_count": len(rows),
            **counts,
            "difference_between_extremes": max(labels.values(), default=0) - min(labels.values(), default=0),
        }
    all_counts = Counter(int(assignments[str(case["case_id"])][0]) for case in cases)
    balanced_by_shape = all(
        details["difference_between_extremes"] <= 1 for details in by_count.values() if details["case_count"]
    )
    return {
        "case_count": len(cases),
        "hypothesis_count_distribution": dict(Counter(int(case["hypothesis_count"]) for case in cases)),
        "rank_as_A_counts": {str(index + 1): all_counts.get(index, 0) for index in range(3)},
        "by_hypothesis_count": by_count,
        "globally_balanced_within_one": balanced_by_shape,
        "global_rank_counts_are_not_comparable_across_two_and_three_path_cases": True,
        "permutation_policy": "two-path parity and three-path cyclic permutations over frozen list order",
    }
