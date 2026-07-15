from __future__ import annotations

from pathlib import Path

from football_intelligence.review_chassis.blinding import (
    apply_permutation,
    assign_counterbalanced_permutations,
    audit_counterbalance,
)


def _cases(counts: list[int]) -> list[dict[str, object]]:
    return [
        {"case_id": f"case_{index:03d}", "hypothesis_count": count, "stratum": "gap_1" if index % 2 else "gap_2"}
        for index, count in enumerate(counts)
    ]


def test_two_path_permutations_are_deterministic_and_balanced() -> None:
    cases = _cases([2] * 9)
    first = assign_counterbalanced_permutations(cases, stage_seed="test-seed")
    second = assign_counterbalanced_permutations(cases, stage_seed="test-seed")
    assert first == second
    audit = audit_counterbalance(cases, first)
    assert audit["by_hypothesis_count"]["2"]["difference_between_extremes"] <= 1
    assert all(details["difference_between_extremes"] <= 1 for details in audit["two_path_by_stratum"].values())
    assert audit["globally_balanced_within_one"] is True


def test_three_path_permutations_are_cyclic_and_balanced() -> None:
    cases = _cases([3] * 7)
    assignments = assign_counterbalanced_permutations(cases, stage_seed="test-seed")
    audit = audit_counterbalance(cases, assignments)
    assert audit["by_hypothesis_count"]["3"]["difference_between_extremes"] <= 1
    assert {tuple(value) for value in assignments.values()} <= {(0, 1, 2), (1, 2, 0), (2, 0, 1)}


def test_permutation_uses_frozen_case_order_not_target_id_order() -> None:
    cases = _cases([2, 3, 2, 3])
    altered = [dict(case, target_candidate_id=f"different_{index}") for index, case in enumerate(cases)]
    assert assign_counterbalanced_permutations(cases, stage_seed="test-seed") == assign_counterbalanced_permutations(
        altered, stage_seed="test-seed"
    )


def test_apply_permutation_preserves_only_display_order() -> None:
    frozen = [{"name": "first"}, {"name": "second"}, {"name": "third"}]
    assert [item["name"] for item in apply_permutation(frozen, (2, 0, 1))] == ["third", "first", "second"]


def test_generic_client_has_no_legacy_answer_key_bootstrap() -> None:
    app = Path(__file__).parents[1] / "src" / "football_intelligence" / "review_chassis" / "static" / "app.js"
    text = app.read_text(encoding="utf-8")
    assert "post_decision_answer_key" not in text
    assert "allowed.has(option.value)" in text
