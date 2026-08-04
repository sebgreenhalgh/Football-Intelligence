"""Focused R6 server-authoritative reducer and exact no-subject route tests."""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from pathlib import Path

import pytest

from football_intelligence.g7e_b_r6_action_reducer import R6_REVIEW_REVISION
from football_intelligence.temporal_review import (
    InterruptedAcknowledgement,
    ReviewValidationError,
    TemporalReviewStore,
)

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PART7 / "G7E_B_R6_SERVER_AUTHORITATIVE_ACTION_REDUCER_AND_EXACT_BRANCH_REPAIR_v1"
PACKAGE = STAGE / "03_SERVER_AUTHORITATIVE_ACTION_REDUCER/temporal_reviewer_r6"
FAILED_BURST = "g7e_a_117092_16"


def store(tmp_path: Path) -> TemporalReviewStore:
    return TemporalReviewStore(PACKAGE, tmp_path / "real", tmp_path / "practice", acceptance_mode=True)


def dispatch(
    reviewer: TemporalReviewStore,
    draft: dict,
    action_type: str,
    payload: dict | None = None,
    *,
    mode: str = "real",
    question_key: str | None = None,
    action_id: str | None = None,
) -> dict:
    action_id = action_id or str(uuid.uuid4())
    response = reviewer.apply_browser_action(
        {
            "schema_version": "football_intelligence.g7e_b_r6.browser_action.v1",
            "action_id": action_id,
            "idempotency_key": action_id,
            "review_revision": R6_REVIEW_REVISION,
            "contract_hash": reviewer.action_contract_sha256,
            "mode": mode,
            "tranche_id": draft.get("tranche_id"),
            "burst_id": draft["burst_id"],
            "expected_draft_revision": draft["draft_version"],
            "expected_draft_sha256": draft["draft_content_sha256"],
            "question_instance_key": question_key or draft["current_question_instance_key"],
            "action_type": action_type,
            "payload": payload or {},
            "client_timestamp": "2026-08-03T00:00:00Z",
        },
        mode,
    )
    return response["draft"]


def answer(reviewer: TemporalReviewStore, draft: dict, value: str) -> dict:
    return dispatch(reviewer, draft, "ANSWER_QUESTION", {"value": value})


def forward(reviewer: TemporalReviewStore, draft: dict) -> dict:
    return dispatch(reviewer, draft, "NAVIGATE_FORWARD")


def exact_no_subject_route(reviewer: TemporalReviewStore, marks: int) -> dict:
    draft = reviewer.initialize_draft("real", FAILED_BURST)
    draft = answer(reviewer, draft, "NO_RELEVANT_PERSON")
    draft = forward(reviewer, draft)
    assert draft["current_question"] == "context_subject"
    draft = answer(reviewer, draft, "NO")
    draft = forward(reviewer, draft)
    assert draft["current_question"] == "missed_check"
    draft = answer(reviewer, draft, "YES" if marks else "NO")
    draft = forward(reviewer, draft)
    if marks:
        assert draft["current_question"] == "missed_mark"
        for index in range(marks):
            draft = dispatch(
                reviewer,
                draft,
                "ADD_MISSED_PERSON_MARK",
                {"frame_sequence": index % 9, "source_xy": [100.0 + index, 200.0 + index]},
            )
        draft = dispatch(reviewer, draft, "COMPLETE_MISSED_PERSON_MARKING")
        draft = forward(reviewer, draft)
    assert draft["current_question"] == "summary"
    assert draft["summary_ready"] is True
    return draft


def test_exact_no_subject_yes_route_atomically_preserves_27_marks(tmp_path: Path) -> None:
    reviewer = store(tmp_path)
    draft = exact_no_subject_route(reviewer, 27)
    assert len(draft["missed_person_marks"]) == 27
    assert draft["question_lifecycle"][f"{FAILED_BURST}|missed_check"] == "ANSWERED"
    assert draft["question_lifecycle"][f"{FAILED_BURST}|missed_mark"] == "ANSWERED"
    assert draft["question_lifecycle"][f"{FAILED_BURST}|additional_subject"] == "SKIPPED_NOT_APPLICABLE"
    request = {
        "mode": "real",
        "burst_id": FAILED_BURST,
        "draft_version": draft["draft_version"],
        "draft_content_sha256": draft["draft_content_sha256"],
        "optimistic_lock_token": draft["optimistic_lock_token"],
    }
    preflight = reviewer.final_save_preflight(request, "real")
    assert preflight["ok"] and preflight["status"] == "READY_TO_PERSIST"


@pytest.mark.parametrize("value", ["NO", "NOT_SURE"])
def test_no_subject_non_marking_routes_reach_server_authorized_summary(tmp_path: Path, value: str) -> None:
    reviewer = store(tmp_path)
    draft = reviewer.initialize_draft("real", FAILED_BURST)
    draft = answer(reviewer, draft, "NO_RELEVANT_PERSON")
    draft = forward(reviewer, draft)
    draft = answer(reviewer, draft, "NO")
    draft = forward(reviewer, draft)
    draft = answer(reviewer, draft, value)
    draft = forward(reviewer, draft)
    assert draft["summary_ready"] is True and draft["current_question"] == "summary"
    assert draft["missed_person_marks"] == []


def test_answer_and_lifecycle_are_atomic_and_direct_draft_posts_are_forbidden(tmp_path: Path) -> None:
    reviewer = store(tmp_path)
    draft = reviewer.initialize_draft("real", FAILED_BURST)
    before = (tmp_path / "real/drafts" / f"{FAILED_BURST}.json").read_bytes()
    stale = copy.deepcopy(draft)
    stale["answers"]["original_focus_box_answer"] = "NO_RELEVANT_PERSON"
    with pytest.raises(ReviewValidationError) as failure:
        reviewer.save_draft(stale, "real")
    assert failure.value.error_code == "DIRECT_DRAFT_MUTATION_FORBIDDEN"
    assert (tmp_path / "real/drafts" / f"{FAILED_BURST}.json").read_bytes() == before
    updated = answer(reviewer, draft, "NO_RELEVANT_PERSON")
    key = f"{FAILED_BURST}|original_focus"
    assert updated["answered_domain_values"][key] == "NO_RELEVANT_PERSON"
    assert updated["question_lifecycle"][key] == "ANSWERED"


def test_duplicate_action_is_idempotent_and_stale_concurrent_action_fails(tmp_path: Path) -> None:
    reviewer = store(tmp_path)
    draft = reviewer.initialize_draft("real", FAILED_BURST)
    action_id = str(uuid.uuid4())
    action = {
        "schema_version": "football_intelligence.g7e_b_r6.browser_action.v1",
        "action_id": action_id,
        "idempotency_key": action_id,
        "review_revision": R6_REVIEW_REVISION,
        "contract_hash": reviewer.action_contract_sha256,
        "mode": "real",
        "tranche_id": draft["tranche_id"],
        "burst_id": FAILED_BURST,
        "expected_draft_revision": draft["draft_version"],
        "expected_draft_sha256": draft["draft_content_sha256"],
        "question_instance_key": draft["current_question_instance_key"],
        "action_type": "ANSWER_QUESTION",
        "payload": {"value": "NO_RELEVANT_PERSON"},
        "client_timestamp": "2026-08-03T00:00:00Z",
    }
    first = reviewer.apply_browser_action(action, "real")
    second = reviewer.apply_browser_action(action, "real")
    assert second["idempotent_replay"] is True
    assert first["draft"]["draft_content_sha256"] == second["draft"]["draft_content_sha256"]
    conflicting_duplicate = copy.deepcopy(action)
    conflicting_duplicate["payload"] = {"value": "ONE_RELEVANT_MATCH_PERSON"}
    with pytest.raises(ValueError, match="different semantic content"):
        reviewer.apply_browser_action(conflicting_duplicate, "real")
    stale = copy.deepcopy(action)
    stale["action_id"] = str(uuid.uuid4())
    stale["idempotency_key"] = stale["action_id"]
    with pytest.raises(ValueError, match="STALE_DRAFT"):
        reviewer.apply_browser_action(stale, "real")


def test_fault_and_race_recovery_is_deterministic_and_atomic(tmp_path: Path) -> None:
    reviewer = store(tmp_path)
    draft = reviewer.initialize_draft("real", FAILED_BURST)
    with pytest.raises(ValueError, match="current question is not answered"):
        forward(reviewer, draft)

    draft = answer(reviewer, draft, "NO_RELEVANT_PERSON")
    restored = store(tmp_path).draft("real", FAILED_BURST)
    assert restored is not None
    assert restored["draft_content_sha256"] == draft["draft_content_sha256"]

    stale_semantic_id = str(uuid.uuid4())
    stale_semantic = {
        "schema_version": "football_intelligence.g7e_b_r6.browser_action.v1",
        "action_id": stale_semantic_id,
        "idempotency_key": stale_semantic_id,
        "review_revision": R6_REVIEW_REVISION,
        "contract_hash": reviewer.action_contract_sha256,
        "mode": "real",
        "tranche_id": draft["tranche_id"],
        "burst_id": FAILED_BURST,
        "expected_draft_revision": draft["draft_version"] - 1,
        "expected_draft_sha256": "0" * 64,
        "question_instance_key": f"{FAILED_BURST}|original_focus",
        "action_type": "ANSWER_QUESTION",
        "payload": {"value": "NO_RELEVANT_PERSON"},
        "client_timestamp": "2026-08-03T00:00:00Z",
    }
    with pytest.raises(ValueError, match="STALE_DRAFT"):
        reviewer.apply_browser_action(stale_semantic, "real")

    draft = forward(reviewer, draft)
    draft = answer(reviewer, draft, "NO")
    draft = forward(reviewer, draft)
    draft = answer(reviewer, draft, "YES")
    draft = forward(reviewer, draft)
    draft = dispatch(
        reviewer,
        draft,
        "ADD_MISSED_PERSON_MARK",
        {"mark_id": "rapid-mark", "frame_sequence": 4, "source_xy": [300.0, 400.0]},
    )
    draft = dispatch(reviewer, draft, "REMOVE_MISSED_PERSON_MARK", {"mark_id": "rapid-mark"})
    draft = dispatch(
        reviewer,
        draft,
        "ADD_MISSED_PERSON_MARK",
        {"mark_id": "retained-mark", "frame_sequence": 4, "source_xy": [500.0, 600.0]},
    )

    complete_id = str(uuid.uuid4())
    complete_action = {
        "schema_version": "football_intelligence.g7e_b_r6.browser_action.v1",
        "action_id": complete_id,
        "idempotency_key": complete_id,
        "review_revision": R6_REVIEW_REVISION,
        "contract_hash": reviewer.action_contract_sha256,
        "mode": "real",
        "tranche_id": draft["tranche_id"],
        "burst_id": FAILED_BURST,
        "expected_draft_revision": draft["draft_version"],
        "expected_draft_sha256": draft["draft_content_sha256"],
        "question_instance_key": draft["current_question_instance_key"],
        "action_type": "COMPLETE_MISSED_PERSON_MARKING",
        "payload": {},
        "client_timestamp": "2026-08-03T00:00:00Z",
    }
    first = reviewer.apply_browser_action(complete_action, "real")
    duplicate = reviewer.apply_browser_action(complete_action, "real")
    assert duplicate["idempotent_replay"] is True
    assert first["draft"]["draft_content_sha256"] == duplicate["draft"]["draft_content_sha256"]
    draft = forward(reviewer, first["draft"])
    old_request = {
        "mode": "real",
        "burst_id": FAILED_BURST,
        "draft_version": draft["draft_version"] - 1,
        "draft_content_sha256": first["draft"]["draft_content_sha256"],
        "optimistic_lock_token": first["draft"]["optimistic_lock_token"],
    }
    assert reviewer.final_save_preflight(old_request, "real")["ok"] is False
    request = {
        "mode": "real",
        "burst_id": FAILED_BURST,
        "draft_version": draft["draft_version"],
        "draft_content_sha256": draft["draft_content_sha256"],
        "optimistic_lock_token": draft["optimistic_lock_token"],
    }
    preflight = reviewer.final_save_preflight(request, "real")
    save = {
        **request,
        "proposed_event_id": preflight["proposed_event_id"],
        "idempotency_key": preflight["idempotency_key"],
        "simulate_interrupt_after_event": True,
    }
    with pytest.raises(InterruptedAcknowledgement):
        reviewer.save_event(save, "real")
    save.pop("simulate_interrupt_after_event")
    recovered = reviewer.save_event(save, "real")
    repeated = reviewer.save_event(save, "real")
    assert recovered["status"] == repeated["status"] == "SERVER_ACKNOWLEDGED"
    assert repeated["recovered_existing_event"] is True
    assert len(list((tmp_path / "real/events").rglob("*.json"))) == 1
    assert len(list((tmp_path / "real/receipts/acknowledgements").glob("*.json"))) == 1


def test_production_bundle_has_no_direct_canonical_mutation_adapter() -> None:
    source = (PACKAGE / "review.js").read_text(encoding="utf-8")
    assert 'api("/api/action", envelope)' in source
    assert 'api("/api/draft"' not in source
    assert ".answered_domain_values[" not in source
    assert ".question_lifecycle[" not in source
    assert 'productionActionOrigin: "REAL_DOM_ACTIONS"' in source
    assert "if (app.pending || app.readOnly) return null" in source
    assert "(app.current.candidates || []).forEach" in source
    assert "app.current.candidates.source_box_xyxy" not in source


def test_r6_package_preserves_exact_corpus_and_candidate_hashes() -> None:
    r5 = json.loads(
        (
            PART7
            / (
                "G7E_B_R5_REVIEWER_STATE_MACHINE_AND_FULL_CORPUS_STABILIZATION_v1/"
                "02_CANONICAL_STATE_CONTRACT/temporal_reviewer_r5/review_cases.json"
            )
        ).read_text(encoding="utf-8")
    )
    r6 = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))

    def frozen(doc: dict[str, object]) -> str:
        return hashlib.sha256(
            json.dumps(
                [(row["burst_id"], row["frames"], row["frame_candidates"]) for row in doc["cases"]],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    assert frozen(r5) == frozen(r6)
    assert len(r6["cases"]) == 120
    assert sum(len(row["frames"]) for row in r6["cases"]) == 1080
