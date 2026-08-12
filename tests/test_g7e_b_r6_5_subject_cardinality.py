"""R6.5 branch-aware subject-cardinality regression tests."""

from __future__ import annotations

import copy

import pytest

from football_intelligence.g7e_b_r6_action_reducer import (
    R6_ADDITIONAL_SUBJECT_SOURCE,
    R6_LEGACY_ADDITIONAL_SUBJECT_SOURCE,
    r6_subject_cardinality_error,
)


BURST = "r6_5_subject_cardinality"


def draft(
    focus: str,
    subject_count: int,
    *,
    context: str | None = None,
    multi: str | None = None,
    uncertain: str | None = None,
    extra_source: str = R6_ADDITIONAL_SUBJECT_SOURCE,
    branch_answered: bool = True,
) -> dict:
    subjects = [
        {
            "subject_token": f"SUBJECT_{'ABCD'[index]}",
            "subject_definition_source": "INITIAL_BRANCH",
        }
        for index in range(subject_count)
    ]
    initial = 2 if focus == "MORE_THAN_ONE_RELEVANT_PERSON" and multi == "ADD_SUBJECT_B" else 1
    if focus == "NO_RELEVANT_PERSON":
        initial = 2 if context == "YES_MORE_THAN_ONE_PERSON" else 1 if context == "YES_ONE_PERSON" else 0
    if focus == "NOT_SURE":
        initial = 1 if uncertain == "UNCERTAIN_SUBJECT_A" else 0
    for subject in subjects[initial:]:
        subject["subject_definition_source"] = extra_source
    additional = f"{BURST}|additional_subject"
    return {
        "burst_id": BURST,
        "answers": {
            "original_focus_box_answer": focus,
            "context_subject_answer": context,
            "multi_subject_b": multi,
            "uncertain_focus_path": uncertain,
        },
        "subjects": subjects,
        "question_lifecycle": {additional: "ANSWERED" if branch_answered else "SKIPPED_NOT_APPLICABLE"},
        "action_journal": [
            {
                "action_type": "ANSWER_QUESTION",
                "question_instance_key": additional,
            }
        ]
        if branch_answered
        else [],
    }


@pytest.mark.parametrize("focus", ["ONE_RELEVANT_MATCH_PERSON", "PART_OF_ONE_RELEVANT_MATCH_PERSON"])
def test_single_person_focus_accepts_only_server_provenanced_additional_subject(focus: str) -> None:
    valid = draft(focus, 2)
    assert r6_subject_cardinality_error(valid) is None

    unexplained = copy.deepcopy(valid)
    unexplained["subjects"][1]["subject_definition_source"] = "INITIAL_BRANCH"
    assert "provenance" in str(r6_subject_cardinality_error(unexplained)).lower()


def test_legacy_r6_additional_subject_source_keeps_paused_drafts_resumable() -> None:
    assert (
        r6_subject_cardinality_error(
            draft("PART_OF_ONE_RELEVANT_MATCH_PERSON", 2, extra_source=R6_LEGACY_ADDITIONAL_SUBJECT_SOURCE)
        )
        is None
    )


@pytest.mark.parametrize(
    ("context", "subject_count", "expected"),
    [("NO", 0, None), ("YES_ONE_PERSON", 2, None), ("YES_MORE_THAN_ONE_PERSON", 2, None)],
)
def test_context_subject_cardinality_is_branch_aware(context: str, subject_count: int, expected: str | None) -> None:
    assert r6_subject_cardinality_error(draft("NO_RELEVANT_PERSON", subject_count, context=context)) == expected


def test_context_one_person_rejects_unexplained_second_subject() -> None:
    invalid = draft("NO_RELEVANT_PERSON", 2, context="YES_ONE_PERSON", extra_source="INITIAL_BRANCH")
    assert "provenance" in str(r6_subject_cardinality_error(invalid)).lower()


def test_no_subject_route_and_three_subject_limit_remain_strict() -> None:
    assert r6_subject_cardinality_error(draft("NO_RELEVANT_PERSON", 0, context="NO")) is None
    assert r6_subject_cardinality_error(draft("NO_RELEVANT_PERSON", 1, context="NO")) == (
        "The selected initial branch permits no subjects."
    )
    assert r6_subject_cardinality_error(draft("ONE_RELEVANT_MATCH_PERSON", 3)) is None
    assert r6_subject_cardinality_error(draft("ONE_RELEVANT_MATCH_PERSON", 4)) == (
        "A burst may contain at most three ordered subjects."
    )


def test_stale_or_invalidated_additional_branch_cannot_keep_an_extra_subject() -> None:
    invalid = draft("ONE_RELEVANT_MATCH_PERSON", 2, branch_answered=False)
    assert "answered canonical additional-subject" in str(r6_subject_cardinality_error(invalid))


@pytest.mark.parametrize(
    ("focus", "count", "context", "multi", "uncertain", "expected"),
    [
        ("ONE_RELEVANT_MATCH_PERSON", 0, None, None, None, False),
        ("ONE_RELEVANT_MATCH_PERSON", 1, None, None, None, True),
        ("ONE_RELEVANT_MATCH_PERSON", 2, None, None, None, True),
        ("PART_OF_ONE_RELEVANT_MATCH_PERSON", 2, None, None, None, True),
        ("MORE_THAN_ONE_RELEVANT_PERSON", 1, None, "ADD_SUBJECT_B", None, False),
        ("MORE_THAN_ONE_RELEVANT_PERSON", 2, None, "ADD_SUBJECT_B", None, True),
        ("MORE_THAN_ONE_RELEVANT_PERSON", 3, None, "ADD_SUBJECT_B", None, True),
        ("NO_RELEVANT_PERSON", 0, "NO", None, None, True),
        ("NO_RELEVANT_PERSON", 1, "NO", None, None, False),
        ("NO_RELEVANT_PERSON", 1, "YES_ONE_PERSON", None, None, True),
        ("NO_RELEVANT_PERSON", 2, "YES_ONE_PERSON", None, None, True),
        ("NO_RELEVANT_PERSON", 2, "YES_MORE_THAN_ONE_PERSON", None, None, True),
        ("NOT_SURE", 0, None, None, "NO_SUBJECT", True),
        ("NOT_SURE", 1, None, None, "UNCERTAIN_SUBJECT_A", True),
        ("NOT_SURE", 2, None, None, "UNCERTAIN_SUBJECT_A", True),
    ],
)
def test_subject_cardinality_matrix_has_no_eligibility_mismatches(
    focus: str,
    count: int,
    context: str | None,
    multi: str | None,
    uncertain: str | None,
    expected: bool,
) -> None:
    eligible = (
        r6_subject_cardinality_error(draft(focus, count, context=context, multi=multi, uncertain=uncertain)) is None
    )
    assert eligible is expected
