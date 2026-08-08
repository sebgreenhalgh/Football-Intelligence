"""Bounded R6.3 stale-resync, duplicate-Done, and hot-path contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import uuid

import pytest

from football_intelligence.g7e_b_r6_action_reducer import R6_REVIEW_REVISION
from football_intelligence.temporal_review import StaleDraftError, TemporalReviewStore
from football_intelligence.temporal_reviewer.persistence import ActionTransaction, recover_action_transactions

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
PACKAGE = PART7 / "G7E_B_R6_SERVER_AUTHORITATIVE_ACTION_REDUCER_AND_EXACT_BRANCH_REPAIR_v1/03_SERVER_AUTHORITATIVE_ACTION_REDUCER/temporal_reviewer_r6"
BURST = "g7e_a_117092_16"


def make_store(tmp_path: Path) -> TemporalReviewStore:
    return TemporalReviewStore(PACKAGE, tmp_path / "real", tmp_path / "practice", acceptance_mode=True)


def action(store: TemporalReviewStore, draft: dict, action_type: str, payload: dict | None = None) -> dict:
    action_id = str(uuid.uuid4())
    return {
        "schema_version": "football_intelligence.g7e_b_r6.browser_action.v1",
        "action_id": action_id,
        "idempotency_key": action_id,
        "review_revision": R6_REVIEW_REVISION,
        "contract_hash": store.action_contract_sha256,
        "mode": "real",
        "tranche_id": draft["tranche_id"],
        "burst_id": draft["burst_id"],
        "expected_draft_revision": draft["draft_version"],
        "expected_draft_sha256": draft["draft_content_sha256"],
        "question_instance_key": draft["current_question_instance_key"],
        "action_type": action_type,
        "payload": payload or {},
        "client_timestamp": "2026-08-08T00:00:00Z",
    }


def dispatch(store: TemporalReviewStore, draft: dict, action_type: str, payload: dict | None = None) -> dict:
    return store.apply_browser_action(action(store, draft, action_type, payload), "real")["draft"]


def prepare_missed_mark(store: TemporalReviewStore) -> dict:
    draft = store.initialize_draft("real", BURST)
    draft = dispatch(store, draft, "ANSWER_QUESTION", {"value": "NO_RELEVANT_PERSON"})
    draft = dispatch(store, draft, "NAVIGATE_FORWARD")
    draft = dispatch(store, draft, "ANSWER_QUESTION", {"value": "NO"})
    draft = dispatch(store, draft, "NAVIGATE_FORWARD")
    draft = dispatch(store, draft, "ANSWER_QUESTION", {"value": "YES"})
    return dispatch(store, draft, "NAVIGATE_FORWARD")


def test_stale_revision_and_hash_return_canonical_without_replay(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    draft = store.initialize_draft("real", BURST)
    accepted = store.apply_browser_action(action(store, draft, "ANSWER_QUESTION", {"value": "NO_RELEVANT_PERSON"}), "real")["draft"]
    stale = action(store, draft, "ANSWER_QUESTION", {"value": "ONE_RELEVANT_MATCH_PERSON"})
    with pytest.raises(StaleDraftError) as failure:
        store.apply_browser_action(stale, "real")
    assert failure.value.error_code == "STALE_DRAFT_REVISION"
    assert failure.value.canonical_draft["draft_version"] == accepted["draft_version"]
    assert failure.value.canonical_draft["answered_domain_values"] == accepted["answered_domain_values"]
    assert "ONE_RELEVANT_MATCH_PERSON" not in json.dumps(failure.value.canonical_draft)

    stale_hash = action(store, accepted, "NAVIGATE_FORWARD")
    stale_hash["expected_draft_sha256"] = "0" * 64
    with pytest.raises(StaleDraftError) as hash_failure:
        store.apply_browser_action(stale_hash, "real")
    assert hash_failure.value.error_code == "STALE_DRAFT_HASH"
    assert hash_failure.value.canonical_draft["draft_content_sha256"] == accepted["draft_content_sha256"]


def test_fresh_duplicate_done_is_canonical_noop(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    draft = prepare_missed_mark(store)
    draft = dispatch(store, draft, "ADD_MISSED_PERSON_MARK", {"frame_sequence": 4, "source_xy": [100.0, 200.0]})
    first_action = action(store, draft, "COMPLETE_MISSED_PERSON_MARKING")
    first = store.apply_browser_action(first_action, "real")
    complete = first["draft"]
    second = store.apply_browser_action(action(store, complete, "COMPLETE_MISSED_PERSON_MARKING"), "real")
    assert second["canonical_noop"] is True
    assert second["draft"]["draft_version"] == complete["draft_version"]
    assert second["draft"]["draft_content_sha256"] == complete["draft_content_sha256"]


def test_startup_recovery_validates_committed_without_rematerializing(tmp_path: Path) -> None:
    root = tmp_path / "root"
    action_id = str(uuid.uuid4())
    draft = b'{"burst_id":"history","draft_version":0}\n'
    ActionTransaction(root, action_id).commit(
        draft_relative="drafts/history.json",
        draft_bytes=draft,
        receipt_relative=f"receipts/actions/action-ack-{action_id}.json",
        receipt_bytes=b'{"receipt":true}\n',
        ledger_relative=f"action_idempotency/{action_id}.json",
        ledger_bytes=b'{"ledger":true}\n',
        transaction_context={
            "previous_draft_revision": 0,
            "previous_draft_sha256": "0" * 64,
            "action_envelope_sha256": "1" * 64,
            "action_semantic_sha256": "2" * 64,
            "next_draft_revision": 0,
            "next_draft_sha256": "3" * 64,
        },
    )
    before = (root / "drafts/history.json").read_bytes()
    result = recover_action_transactions(root)
    assert result == {"inspected": 1, "recovered": 0, "committed": 1}
    assert (root / "drafts/history.json").read_bytes() == before


def test_runtime_contracts_are_bound_to_cached_gate_and_resync_ui() -> None:
    server = (PROJECT / "SoccerTrack-v2/src/football_intelligence/temporal_review.py").read_text(encoding="utf-8")
    browser = (PROJECT / "SoccerTrack-v2/src/football_intelligence/g7e_b_r6_temporal_review.js").read_text(encoding="utf-8")
    assert "self._cached_release_gate_status = self._verify_release_gate()" in server
    assert "recover_action_transactions(root)" not in server
    assert "canonical_draft" in server and "rejected_action_replayed" in server
    assert "Server state restored. Your rejected click was not replayed." in browser
    assert "verifiedImageCache" in browser and "requestAnimationFrame" in browser
