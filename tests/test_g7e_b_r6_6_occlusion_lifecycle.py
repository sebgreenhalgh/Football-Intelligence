"""R6.6 subject-level occlusion lifecycle and compiler alignment tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from football_intelligence.g7e_b_r5_reviewer_state import (
    R5_REVIEW_REVISION,
    R5_WORKING_DRAFT_SCHEMA,
    compile_final_event as compile_r5_final_event,
)
from football_intelligence.g7e_b_r6_action_reducer import (
    _all_summary_fields_answered,
    _reconcile_branches,
    compile_final_event,
    validate_r6_invariants,
)
from football_intelligence.temporal_review import TemporalReviewStore


PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
PART8 = PROJECT / "experiments/football_observation_reasoner/part 8"
PACKAGE = (
    PART8
    / "G7E_B_R6_5_SUBJECT_CARDINALITY_AND_FINAL_COMPILER_ALIGNMENT_v1"
    / "03_SUBJECT_CARDINALITY_IMPLEMENTATION/temporal_reviewer_r6_5"
)
REAL_ROOT = (
    PART7
    / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
    / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
)
BURST = "g7e_a_118577_14"
OCCLUSION_KEY = f"{BURST}|SUBJECT_B|occlusion"
HUMAN_VALUES = ["NONE", "ENTERING_OCCLUSION", "OCCLUDED", "EXITING_OCCLUSION", "UNCERTAIN"]
PHASE_PATTERNS = [
    ["NONE"] * 9,
    ["ENTERING_OCCLUSION"] * 9,
    ["ENTERING_OCCLUSION"] * 4 + ["OCCLUDED"] + ["EXITING_OCCLUSION"] * 4,
    ["EXITING_OCCLUSION"] * 9,
    ["UNCERTAIN"] * 9,
    ["NONE", "ENTERING_OCCLUSION", "OCCLUDED", "EXITING_OCCLUSION", "UNCERTAIN"] + ["NONE"] * 4,
]


@pytest.fixture
def reviewer(tmp_path: Path) -> TemporalReviewStore:
    return TemporalReviewStore(PACKAGE, tmp_path / "real", tmp_path / "practice", acceptance_mode=True)


@pytest.fixture
def real_draft() -> dict:
    return json.loads((REAL_ROOT / "drafts" / f"{BURST}.json").read_text(encoding="utf-8"))


def compile_draft(reviewer: TemporalReviewStore, draft: dict) -> tuple[dict | None, list[dict]]:
    return compile_final_event(
        draft,
        reviewer.canonical_contract,
        reviewer.canonical_contract_sha256,
        reviewer.action_contract_sha256,
        reviewer.by_id[BURST],
    )


@pytest.mark.parametrize("human_value", HUMAN_VALUES)
@pytest.mark.parametrize("phases", PHASE_PATTERNS)
def test_all_human_occlusion_values_are_independent_of_valid_frame_phase_patterns(
    reviewer: TemporalReviewStore,
    real_draft: dict,
    human_value: str,
    phases: list[str],
) -> None:
    draft = copy.deepcopy(real_draft)
    draft["answered_domain_values"][OCCLUSION_KEY] = human_value
    for observation, phase in zip(draft["subjects"][1]["frame_observations"], phases, strict=True):
        observation["occlusion_phase"] = phase

    summary_eligible = _all_summary_fields_answered(draft, reviewer.canonical_contract)
    event, errors = compile_draft(reviewer, draft)

    assert summary_eligible is True
    assert errors == []
    assert event is not None
    assert [row["occlusion_phase"] for row in event["subjects"][1]["frame_observations"]] == phases


@pytest.mark.parametrize("lifecycle", ["ACTIVE", "SKIPPED_NOT_APPLICABLE"])
def test_applicable_occlusion_requires_answered_lifecycle(
    reviewer: TemporalReviewStore, real_draft: dict, lifecycle: str
) -> None:
    draft = copy.deepcopy(real_draft)
    draft["question_lifecycle"][OCCLUSION_KEY] = lifecycle
    draft["answered_domain_values"].pop(OCCLUSION_KEY, None)

    event, errors = compile_draft(reviewer, draft)

    assert _all_summary_fields_answered(draft, reviewer.canonical_contract) is False
    assert event is None
    assert errors[0]["error_code"] == "R6_SUMMARY_NOT_SERVER_AUTHORIZED"


def test_invalid_human_occlusion_enum_fails_summary_and_compiler(
    reviewer: TemporalReviewStore, real_draft: dict
) -> None:
    draft = copy.deepcopy(real_draft)
    draft["answered_domain_values"][OCCLUSION_KEY] = "SYNTHETIC_OCCLUSION_VALUE"

    event, errors = compile_draft(reviewer, draft)

    assert _all_summary_fields_answered(draft, reviewer.canonical_contract) is False
    assert any(
        error == f"INVALID_DOMAIN_ENUM:{OCCLUSION_KEY}"
        for error in validate_r6_invariants(draft, reviewer.canonical_contract)
    )
    assert event is None
    assert errors[0]["error_code"] == "R6_SUMMARY_NOT_SERVER_AUTHORIZED"


def test_invalid_frame_phase_remains_independently_rejected(reviewer: TemporalReviewStore, real_draft: dict) -> None:
    draft = copy.deepcopy(real_draft)
    draft["subjects"][1]["frame_observations"][0]["occlusion_phase"] = "INVALID_FRAME_PHASE"

    event, errors = compile_draft(reviewer, draft)

    assert _all_summary_fields_answered(draft, reviewer.canonical_contract) is False
    assert event is None
    assert errors[0]["error_code"] == "R6_SUMMARY_NOT_SERVER_AUTHORIZED"


def test_r5_compiler_independently_rejects_invalid_frame_phase(reviewer: TemporalReviewStore, real_draft: dict) -> None:
    draft = copy.deepcopy(real_draft)
    draft["schema_version"] = R5_WORKING_DRAFT_SCHEMA
    draft["review_revision"] = R5_REVIEW_REVISION
    draft["r6_subject_cardinality_provenance_verified"] = True
    draft["subjects"][1]["frame_observations"][0]["occlusion_phase"] = "INVALID_FRAME_PHASE"

    event, errors = compile_r5_final_event(
        draft,
        reviewer.canonical_contract,
        reviewer.canonical_contract_sha256,
        reviewer.by_id[BURST],
    )

    assert event is None
    assert any(row["error_code"] == "FINAL_OCCLUSION_PHASE_REQUIRED" for row in errors)


def test_non_applicable_occlusion_is_skipped_and_requires_a_new_answer_if_restored(
    reviewer: TemporalReviewStore, real_draft: dict
) -> None:
    draft = copy.deepcopy(real_draft)
    for observation in draft["subjects"][1]["frame_observations"]:
        observation["visibility"] = "VISIBLE_COMPLETE"
        observation["occlusion_phase"] = "NONE"
    _reconcile_branches(draft, reviewer.canonical_contract, "r6-6-hide", "VISIBILITY_CHANGED")

    assert draft["question_lifecycle"][OCCLUSION_KEY] == "SKIPPED_NOT_APPLICABLE"
    assert OCCLUSION_KEY not in draft["answered_domain_values"]

    draft["subjects"][1]["frame_observations"][0]["visibility"] = "VISIBLE_PARTIAL"
    draft["subjects"][1]["frame_observations"][0]["occlusion_phase"] = "ENTERING_OCCLUSION"
    _reconcile_branches(draft, reviewer.canonical_contract, "r6-6-show", "VISIBILITY_CHANGED")

    assert _all_summary_fields_answered(draft, reviewer.canonical_contract) is False
    event, errors = compile_draft(reviewer, draft)
    assert event is None
    assert errors[0]["error_code"] == "R6_SUMMARY_NOT_SERVER_AUTHORIZED"
