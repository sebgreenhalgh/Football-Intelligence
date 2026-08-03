"""Focused tests for the G7E-B R5 lifecycle and release candidate."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from football_intelligence.g7e_b_r5_reviewer_state import (
    R5_REVIEW_REVISION,
    R5_WORKING_DRAFT_SCHEMA,
    load_contract,
    synthetic_complete_draft,
)
from football_intelligence.temporal_review import ReviewValidationError, TemporalReviewStore

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PART7 / "G7E_B_R5_REVIEWER_STATE_MACHINE_AND_FULL_CORPUS_STABILIZATION_v1"
PACKAGE = STAGE / "02_CANONICAL_STATE_CONTRACT/temporal_reviewer_r5"
REAL_ROOT = (
    PART7
    / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
    / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
)
BURST_1_EVENT = REAL_ROOT / "events/TRANCHE_1/302bff13-c381-5307-93f8-f2e83b282287.json"
BURST_1_ACK = REAL_ROOT / "receipts/acknowledgements/ack-302bff13-c381-5307-93f8-f2e83b282287.json"
BURST_2 = "g7e_a_118575_18"
PASS = "PASS_G7E_B_R5_REVIEWER_RELEASE_CANDIDATE_READY_FOR_REAL_TRANCHE_RESUME"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_store(tmp_path: Path) -> TemporalReviewStore:
    return TemporalReviewStore(
        PACKAGE,
        decisions_root=tmp_path / "real",
        practice_root=tmp_path / "practice",
        acceptance_mode=True,
    )


def test_release_gate_and_exact_corpus_are_hash_bound(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    gate = store.r5_release_gate_status()
    assert store.review_revision == R5_REVIEW_REVISION
    assert gate["valid"] is True
    assert gate["release_classification"] == PASS
    assert len(store.cases) == 120
    assert len(store.practice_cases) == 3
    assert sum(len(case["frames"]) for case in store.cases) == 1080


def test_contract_is_the_complete_ui_domain_and_relationship_source_of_truth() -> None:
    contract, digest = load_contract(PACKAGE / "canonical_reviewer_state_contract.json")
    assert digest == sha256(PACKAGE / "canonical_reviewer_state_contract.json")
    assert set(contract["domain_labels"]) == set(contract["domain_enums"])
    for domain, values in contract["domain_enums"].items():
        assert set(contract["domain_labels"][domain]) == set(values)
        assert all(contract["domain_labels"][domain][value] for value in values)
    families = contract["relationship_compatibility"]["question_families"]
    assert set(families) == {
        "MULTIPLE_BOX_RELATIONSHIP",
        "SINGLE_MERGED_BOX_CONFIRMATION",
        "FRAGMENT_MEANING",
    }
    for definition in families.values():
        assert definition["question"] and definition["help"] and definition["options"]
        assert {value for value, _ in definition["options"]} <= set(definition["allowed_relationships"])
    reviewer_js = (PACKAGE / "review.js").read_text(encoding="utf-8")
    assert 'const addValue = IS_R5 ? "ADD_SUBJECT" : "ADD"' in reviewer_js


def test_sparse_initialization_has_only_unanswered_question_one(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    draft = store.initialize_draft("real", BURST_2)
    assert draft["schema_version"] == R5_WORKING_DRAFT_SCHEMA
    assert draft["current_question"] == "original_focus"
    assert set(draft["question_lifecycle"].values()) == {"ACTIVE"}
    assert draft["answered_domain_values"] == {}
    assert draft["answers"] == {}
    assert draft["subjects"] == []
    assert all(value is not None for value in draft["answered_domain_values"].values())
    assert store.draft("real", BURST_2) == {key: value for key, value in draft.items() if key != "server_file_sha256"}


def test_null_unreached_answer_is_rejected_without_mutation(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    draft = store.initialize_draft("real", BURST_2)
    before = (tmp_path / "real/drafts" / f"{BURST_2}.json").read_bytes()
    invalid = copy.deepcopy(draft)
    invalid["answered_domain_values"][invalid["current_question_instance_key"]] = None
    invalid.pop("server_file_sha256", None)
    with pytest.raises(ReviewValidationError) as failure:
        store.save_draft(invalid, "real")
    assert any(row["error_code"] == "NULL_DOMAIN_ANSWER" for row in failure.value.errors)
    assert (tmp_path / "real/drafts" / f"{BURST_2}.json").read_bytes() == before


def test_incomplete_draft_cannot_compile_final_event(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    draft = store.initialize_draft("real", BURST_2)
    result = store.final_save_preflight(
        {
            "burst_id": BURST_2,
            "draft_version": draft["draft_version"],
            "draft_content_sha256": draft["draft_content_sha256"],
            "optimistic_lock_token": draft["optimistic_lock_token"],
        },
        "real",
    )
    assert result["ok"] is False
    assert result["status"] == "FINAL_SAVE_ERROR"
    assert result["error_code"] == "FINAL_EVENT_COMPILATION_FAILED"
    assert not list((tmp_path / "real/events").rglob("*.json"))


def test_complete_draft_preflight_and_save_are_atomic_and_idempotent(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    initial = store.initialize_draft("real", BURST_2)
    case = store.by_id[BURST_2]
    complete = synthetic_complete_draft(
        case,
        "real",
        store.canonical_contract,
        store.canonical_contract_sha256,
    )
    complete["draft_version"] = initial["draft_version"]
    complete["optimistic_lock_token"] = initial["optimistic_lock_token"]
    saved_draft = store.save_draft(complete, "real")
    request = {
        "burst_id": BURST_2,
        "draft_version": saved_draft["draft_version"],
        "draft_content_sha256": saved_draft["draft_content_sha256"],
        "optimistic_lock_token": saved_draft["optimistic_lock_token"],
    }
    first_preflight = store.final_save_preflight(request, "real")
    second_preflight = store.final_save_preflight(request, "real")
    assert first_preflight == second_preflight
    assert first_preflight["status"] == "READY_TO_PERSIST"
    final_request = {
        **request,
        "proposed_event_id": first_preflight["proposed_event_id"],
        "idempotency_key": first_preflight["idempotency_key"],
    }
    first = store.save_event(final_request, "real")
    second = store.save_event(final_request, "real")
    assert first["status"] == "SERVER_ACKNOWLEDGED"
    assert second["recovered_existing_event"] is True
    assert first["event_id"] == second["event_id"]
    assert len(list((tmp_path / "real/events").rglob("*.json"))) == 1
    assert len(list((tmp_path / "real/receipts/acknowledgements").glob("*.json"))) == 1


def test_release_evidence_proves_full_state_and_fault_soaks() -> None:
    transition = read_json(STAGE / "04_TRANSITION_AND_FAULT_SOAK/transition_soak_results.json")
    corpus = read_json(STAGE / "05_FULL_CORPUS_RELEASE_SOAK/full_120_burst_initialization_results.json")
    frames = read_json(STAGE / "05_FULL_CORPUS_RELEASE_SOAK/frame_1080_step_audit.json")
    tranches = read_json(STAGE / "05_FULL_CORPUS_RELEASE_SOAK/full_six_tranche_soak_results.json")
    faults = read_json(STAGE / "04_TRANSITION_AND_FAULT_SOAK/fault_injection_results.json")
    assert transition["passed"] and transition["transition_sequence_count"] == 50_000
    assert corpus["passed"] and corpus["bursts"] == 120
    assert frames["passed"] and frames["frame_references"] == 1080
    assert tranches["passed"] and tranches["acknowledgement_count"] == 120
    assert tranches["tranche_receipt_count"] == 6
    assert tranches["global_receipt_count"] == 1
    assert faults["passed"]


def test_real_event_chain_is_append_only_and_original_bytes_are_unchanged() -> None:
    assert sha256(BURST_1_EVENT) == "0b033c5af85107840b3c2a257d9aa836ca88b1b745f7ab28e444ff5f87234727"
    assert sha256(BURST_1_ACK) == "21a3a80ea572d41520d17b2b01ad9d29b3d9c5a491b068de50c688dfd891a62e"
    events = list((REAL_ROOT / "events").rglob("*.json"))
    acknowledgements = list((REAL_ROOT / "receipts/acknowledgements").glob("*.json"))
    assert len(events) == len(acknowledgements) >= 1
    event_ids = [read_json(path)["event_id"] for path in events]
    assert len(event_ids) == len(set(event_ids))
    assert all((REAL_ROOT / f"receipts/acknowledgements/ack-{event_id}.json").is_file() for event_id in event_ids)


def test_real_edge_acceptance_and_exact_three_visuals() -> None:
    edge = read_json(STAGE / "06_REAL_STATE_MIGRATION_AND_ACCEPTANCE/edge_real_and_temporary_acceptance.json")
    assert edge["passed"]
    assert edge["actual_local_server"] is True
    assert edge["browser"] == "Microsoft Edge"
    branches = edge["branch_browser_acceptance"]
    assert branches["actual_reviewer"] is True
    assert branches["question_family_count"] == 20
    assert branches["relationship_families_rendered"] == 3
    assert branches["all_passed"] is True
    assert all(row["passed"] for row in branches["rows"])
    visuals = sorted((STAGE / "07_VISUAL_QA").glob("*.png"))
    assert len(visuals) == 3
    assert all(path.stat().st_size > 100_000 for path in visuals)


def test_chatgpt_handoff_exact_twelve_file_manifest() -> None:
    handoff = STAGE / "09_REVIEW_PACK/CHATGPT_HANDOFF"
    files = sorted(path for path in handoff.iterdir() if path.is_file())
    assert len(files) == 12
    manifest = read_json(handoff / "12_MANIFEST.json")
    assert len(manifest["files"]) == 11
    assert {row["filename"] for row in manifest["files"]} == {
        path.name for path in files if path.name != "12_MANIFEST.json"
    }
    for row in manifest["files"]:
        path = handoff / row["filename"]
        assert row["byte_size"] == path.stat().st_size
        assert row["sha256"] == sha256(path)
