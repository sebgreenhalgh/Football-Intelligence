"""R6.7.2 provisional additional-subject rollback contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from football_intelligence.g7e_b_r5_reviewer_state import question_key
from football_intelligence.g7e_b_r6_action_reducer import (
    R6_ADDITIONAL_SUBJECT_SOURCE,
    _answer_question,
    _all_summary_fields_answered,
    applicable_question_sequence,
    compile_final_event,
    rollback_provisional_additional_subject,
)
from football_intelligence.temporal_review import TemporalReviewStore, canonical_digest


PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
PART8 = PROJECT / "experiments/football_observation_reasoner/part 8"
STAGE = PART8 / "G7E_B_R6_7_1_REAL_MODE_RELEASE_GATE_CLOSURE_v1"
PACKAGE = STAGE / "03_REAL_MODE_RELEASE_GATE_IMPLEMENTATION/temporal_reviewer_r6_7_1"
BUILDER = STAGE / "build_r6_7_1.py"
REAL_ROOT = (
    PART7
    / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
    / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
)
BURST = "g7e_a_117092_10"
ADDITIONAL = question_key(BURST, "additional_subject")


@pytest.fixture(scope="module", autouse=True)
def fresh_package() -> None:
    subprocess.run([sys.executable, str(BUILDER)], check=True)


@pytest.fixture
def reviewer(tmp_path: Path) -> TemporalReviewStore:
    return TemporalReviewStore(PACKAGE, tmp_path / "real", tmp_path / "practice", acceptance_mode=True)


@pytest.fixture
def real_draft() -> dict:
    return json.loads((REAL_ROOT / "drafts" / f"{BURST}.json").read_text(encoding="utf-8"))


def reviewed_subject(subject: dict) -> None:
    subject["anchor_source_xy"] = [1.0, 1.0]


def normalized_base(real_draft: dict) -> dict:
    draft = copy.deepcopy(real_draft)
    assert rollback_provisional_additional_subject(draft) is True
    return draft


@pytest.mark.parametrize("answer", ["CONTINUE", "NOT_SURE"])
def test_unentered_additional_subject_rolls_back_when_latest_answer_changes(
    reviewer: TemporalReviewStore, real_draft: dict, answer: str
) -> None:
    draft = normalized_base(real_draft)
    _answer_question(draft, reviewer.by_id[BURST], reviewer.canonical_contract, ADDITIONAL, "ADD_SUBJECT")
    assert [subject["subject_token"] for subject in draft["subjects"]] == ["SUBJECT_A", "SUBJECT_B"]
    _answer_question(draft, reviewer.by_id[BURST], reviewer.canonical_contract, ADDITIONAL, answer)

    assert rollback_provisional_additional_subject(draft) is True
    assert [subject["subject_token"] for subject in draft["subjects"]] == ["SUBJECT_A"]
    assert all("|SUBJECT_B|" not in key for key in applicable_question_sequence(draft, reviewer.canonical_contract))


def test_forward_into_subject_b_keeps_it_and_anchor_is_next(reviewer: TemporalReviewStore, real_draft: dict) -> None:
    draft = normalized_base(real_draft)
    _answer_question(draft, reviewer.by_id[BURST], reviewer.canonical_contract, ADDITIONAL, "ADD_SUBJECT")
    sequence = applicable_question_sequence(draft, reviewer.canonical_contract)

    assert question_key(BURST, "anchor", "SUBJECT_B") in sequence
    assert rollback_provisional_additional_subject(draft) is False


def test_reviewed_additional_subject_survives_continue_and_only_provisional_c_is_removed(
    reviewer: TemporalReviewStore, real_draft: dict
) -> None:
    draft = normalized_base(real_draft)
    _answer_question(draft, reviewer.by_id[BURST], reviewer.canonical_contract, ADDITIONAL, "ADD_SUBJECT")
    reviewed_subject(draft["subjects"][1])
    _answer_question(draft, reviewer.by_id[BURST], reviewer.canonical_contract, ADDITIONAL, "CONTINUE")
    assert rollback_provisional_additional_subject(draft) is False
    assert [subject["subject_token"] for subject in draft["subjects"]] == ["SUBJECT_A", "SUBJECT_B"]

    _answer_question(draft, reviewer.by_id[BURST], reviewer.canonical_contract, ADDITIONAL, "ADD_SUBJECT")
    assert draft["subjects"][2]["subject_definition_source"] == R6_ADDITIONAL_SUBJECT_SOURCE
    _answer_question(draft, reviewer.by_id[BURST], reviewer.canonical_contract, ADDITIONAL, "CONTINUE")
    assert rollback_provisional_additional_subject(draft) is True
    assert [subject["subject_token"] for subject in draft["subjects"]] == ["SUBJECT_A", "SUBJECT_B"]


def test_three_reviewed_subjects_remain_and_maximum_is_not_weakened(
    reviewer: TemporalReviewStore, real_draft: dict
) -> None:
    draft = normalized_base(real_draft)
    _answer_question(draft, reviewer.by_id[BURST], reviewer.canonical_contract, ADDITIONAL, "ADD_SUBJECT")
    reviewed_subject(draft["subjects"][1])
    _answer_question(draft, reviewer.by_id[BURST], reviewer.canonical_contract, ADDITIONAL, "ADD_SUBJECT")
    reviewed_subject(draft["subjects"][2])
    _answer_question(draft, reviewer.by_id[BURST], reviewer.canonical_contract, ADDITIONAL, "CONTINUE")

    assert rollback_provisional_additional_subject(draft) is False
    assert len(draft["subjects"]) == 3


def test_exact_revision_108_copy_normalizes_for_summary_and_exactly_once_save(tmp_path: Path, real_draft: dict) -> None:
    root = tmp_path / "real"
    shutil.copytree(REAL_ROOT, root)
    draft_path = root / "drafts" / f"{BURST}.json"
    original_bytes = draft_path.read_bytes()
    original = json.loads(original_bytes)
    human_digest = canonical_digest(
        {
            key: original[key]
            for key in ("answers", "answered_domain_values", "subjects", "missed_person_marks", "click_transactions")
        }
    )
    reviewer = TemporalReviewStore(PACKAGE, root, tmp_path / "practice", acceptance_mode=True)
    restored = reviewer.draft("real", BURST)
    request = {
        "mode": "real", "burst_id": BURST, "draft_version": original["draft_version"],
        "draft_content_sha256": original["draft_content_sha256"],
        "optimistic_lock_token": original["optimistic_lock_token"],
    }
    preflight = reviewer.final_save_preflight(request, "real")
    event, errors = compile_final_event(
        original, reviewer.canonical_contract, reviewer.canonical_contract_sha256,
        reviewer.action_contract_sha256, reviewer.by_id[BURST],
    )

    assert draft_path.read_bytes() == original_bytes
    assert restored is not None
    assert [subject["subject_token"] for subject in restored["subjects"]] == ["SUBJECT_A"]
    assert restored["summary_ready"] is True
    assert restored["draft_version"] == original["draft_version"]
    assert restored["draft_content_sha256"] == original["draft_content_sha256"]
    assert preflight["ok"] is True and errors == [] and event is not None
    assert [subject["subject_token"] for subject in event["subjects"]] == ["SUBJECT_A"]
    assert _all_summary_fields_answered(
        (lambda draft: (rollback_provisional_additional_subject(draft), draft)[1])(copy.deepcopy(original)),
        reviewer.canonical_contract,
    ) is True
    save_request = {
        **request,
        "proposed_event_id": preflight["proposed_event_id"],
        "idempotency_key": preflight["idempotency_key"],
    }
    saved = reviewer.save_event(save_request, "real")
    repeated = reviewer.save_event(save_request, "real")
    assert saved["status"] == repeated["status"] == "SERVER_ACKNOWLEDGED"
    assert repeated["recovered_existing_event"] is True
    event_paths = [
        path
        for path in (root / "events").rglob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["burst_id"] == BURST
    ]
    acknowledgement_paths = [
        path
        for path in (root / "receipts/acknowledgements").glob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["burst_id"] == BURST
    ]
    assert len(event_paths) == len(acknowledgement_paths) == 1
    assert canonical_digest(
        {
            key: original[key]
            for key in ("answers", "answered_domain_values", "subjects", "missed_person_marks", "click_transactions")
        }
    ) == human_digest
    assert len(original["missed_person_marks"]) == 8
