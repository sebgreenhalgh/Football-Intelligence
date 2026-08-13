"""R6.7 immutable subject-level occlusion-answer persistence tests."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from football_intelligence.g7e_b_r6_action_reducer import _all_summary_fields_answered, compile_final_event
from football_intelligence.temporal_review import TemporalReviewStore, canonical_digest


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
SUBJECT = "SUBJECT_B"
OCCLUSION_KEY = f"{BURST}|{SUBJECT}|occlusion"
HUMAN_VALUES = ["NONE", "ENTERING_OCCLUSION", "OCCLUDED", "EXITING_OCCLUSION", "UNCERTAIN"]


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
def test_applicable_subject_event_preserves_each_canonical_human_occlusion_answer(
    reviewer: TemporalReviewStore, real_draft: dict, human_value: str
) -> None:
    draft = copy.deepcopy(real_draft)
    draft["answered_domain_values"][OCCLUSION_KEY] = human_value

    event, errors = compile_draft(reviewer, draft)

    assert _all_summary_fields_answered(draft, reviewer.canonical_contract) is True
    assert errors == []
    assert event is not None
    subject = next(row for row in event["subjects"] if row["subject_token"] == SUBJECT)
    assert subject["occlusion_sequence_answer"] == human_value
    assert [row["occlusion_phase"] for row in subject["frame_observations"]] == ["ENTERING_OCCLUSION"] * 9
    assert "occlusion_sequence_answer" not in event["subjects"][0]


def test_human_answer_changes_compiled_event_identity_without_rewriting_frame_phases(
    reviewer: TemporalReviewStore, real_draft: dict
) -> None:
    first = copy.deepcopy(real_draft)
    second = copy.deepcopy(real_draft)
    first["answered_domain_values"][OCCLUSION_KEY] = "OCCLUDED"
    second["answered_domain_values"][OCCLUSION_KEY] = "EXITING_OCCLUSION"

    first_event, first_errors = compile_draft(reviewer, first)
    second_event, second_errors = compile_draft(reviewer, second)

    assert first_errors == second_errors == []
    assert first_event is not None and second_event is not None
    assert canonical_digest(first_event) != canonical_digest(second_event)
    first_subject = next(row for row in first_event["subjects"] if row["subject_token"] == SUBJECT)
    second_subject = next(row for row in second_event["subjects"] if row["subject_token"] == SUBJECT)
    assert first_subject["occlusion_sequence_answer"] == "OCCLUDED"
    assert second_subject["occlusion_sequence_answer"] == "EXITING_OCCLUSION"
    assert first_subject["frame_observations"] == second_subject["frame_observations"]


@pytest.mark.parametrize("answer", [None, "INVALID_OCCLUSION_VALUE"])
def test_applicable_subject_without_valid_canonical_answer_cannot_compile(
    reviewer: TemporalReviewStore, real_draft: dict, answer: str | None
) -> None:
    draft = copy.deepcopy(real_draft)
    if answer is None:
        draft["answered_domain_values"].pop(OCCLUSION_KEY)
    else:
        draft["answered_domain_values"][OCCLUSION_KEY] = answer

    event, errors = compile_draft(reviewer, draft)

    assert event is None
    assert errors[0]["error_code"] == "R6_SUMMARY_NOT_SERVER_AUTHORIZED"


def test_temp_real_copy_persists_answer_before_draft_deletion_and_keeps_human_content(tmp_path: Path) -> None:
    copied_root = tmp_path / "real"
    shutil.copytree(REAL_ROOT, copied_root)
    historical_bytes = {
        path.relative_to(copied_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in list((copied_root / "events").rglob("*.json"))
        + list((copied_root / "receipts/acknowledgements").glob("*.json"))
    }
    acknowledgement_count = len(list((copied_root / "receipts/acknowledgements").glob("*.json")))
    before_draft = copied_root / "drafts" / f"{BURST}.json"
    before = json.loads(before_draft.read_text(encoding="utf-8"))
    before_human = canonical_digest(
        {
            key: before[key]
            for key in ("answers", "answered_domain_values", "subjects", "missed_person_marks", "click_transactions")
        }
    )
    reviewer = TemporalReviewStore(PACKAGE, copied_root, tmp_path / "practice", acceptance_mode=True)
    request = {
        "mode": "real",
        "burst_id": BURST,
        "draft_version": before["draft_version"],
        "draft_content_sha256": before["draft_content_sha256"],
        "optimistic_lock_token": before["optimistic_lock_token"],
    }

    preflight = reviewer.final_save_preflight(request, "real")
    assert preflight["ok"] is True
    save_request = {
        **request,
        "proposed_event_id": preflight["proposed_event_id"],
        "idempotency_key": preflight["idempotency_key"],
    }
    saved = reviewer.save_event(
        save_request,
        "real",
    )
    repeated = reviewer.save_event(
        save_request,
        "real",
    )

    assert saved["status"] == repeated["status"] == "SERVER_ACKNOWLEDGED"
    assert repeated["recovered_existing_event"] is True
    assert not before_draft.exists()
    event = reviewer.acknowledged_event("real", saved["event_id"])["event"]
    subject = next(row for row in event["subjects"] if row["subject_token"] == SUBJECT)
    assert subject["occlusion_sequence_answer"] == "OCCLUDED"
    assert [row["occlusion_phase"] for row in subject["frame_observations"]] == ["ENTERING_OCCLUSION"] * 9
    assert len(event["whole_burst_missed_person_marks"]) == 9
    assert canonical_digest(
        {
            key: before[key]
            for key in ("answers", "answered_domain_values", "subjects", "missed_person_marks", "click_transactions")
        }
    ) == before_human
    assert len(list((copied_root / "events/TRANCHE_2").glob("*.json"))) == 10
    assert len(list((copied_root / "receipts/acknowledgements").glob("*.json"))) == acknowledgement_count + 1
    assert {
        relative: hashlib.sha256((copied_root / relative).read_bytes()).hexdigest()
        for relative in historical_bytes
    } == historical_bytes
