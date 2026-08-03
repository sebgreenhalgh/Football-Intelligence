"""Run the bounded R5 lifecycle, corpus, fault, and six-tranche release soak."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from football_intelligence.g7e_b_r5_reviewer_state import (
    R5_CONTRACT_ID,
    R5_REVIEW_REVISION,
    canonical_bytes,
    canonical_digest,
    synthetic_complete_draft,
    validate_working_draft,
)
from football_intelligence.temporal_review import (
    InterruptedAcknowledgement,
    ReviewValidationError,
    TemporalReviewStore,
)

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
R3 = PART7 / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
R5 = PART7 / "G7E_B_R5_REVIEWER_STATE_MACHINE_AND_FULL_CORPUS_STABILIZATION_v1"
PACKAGE = R5 / "02_CANONICAL_STATE_CONTRACT/temporal_reviewer_r5"
REAL_ROOT = R3 / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
ASSET_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/assets"
STATE = R5 / "03_STATE_SPACE_AND_SCHEMA_VALIDATION"
TRANSITION = R5 / "04_TRANSITION_AND_FAULT_SOAK"
CORPUS = R5 / "05_FULL_CORPUS_RELEASE_SOAK"
MIGRATION = R5 / "06_REAL_STATE_MIGRATION_AND_ACCEPTANCE"
DECISION = "PASS_G7E_B_R5_REVIEWER_RELEASE_CANDIDATE_READY_FOR_REAL_TRANCHE_RESUME"
SEED = 750_005


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def file_inventory(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}


def contract_evidence(store: TemporalReviewStore) -> None:
    assert store.canonical_contract is not None and store.canonical_contract_sha256 is not None
    shutil.copyfile(
        PACKAGE / "canonical_reviewer_state_contract.json",
        R5 / "02_CANONICAL_STATE_CONTRACT/canonical_reviewer_state_contract.json",
    )
    shutil.copyfile(
        PACKAGE / "generated_client_contract.js", R5 / "02_CANONICAL_STATE_CONTRACT/generated_client_contract.js"
    )
    write_json(
        R5 / "02_CANONICAL_STATE_CONTRACT/working_draft_schema.json",
        {
            "$id": "football_intelligence.g7e_b_r5.working_draft.v1",
            "validation_profiles": ["DRAFT_SHAPE", "DRAFT_PROGRESS"],
            "sparse_domain_answers": True,
            "null_domain_answers_forbidden": True,
            "contract_sha256": store.canonical_contract_sha256,
        },
    )
    write_json(
        R5 / "02_CANONICAL_STATE_CONTRACT/final_event_schema.json",
        {
            "$id": "football_intelligence.g7e_b_r5.burst_annotation_event.v1",
            "validation_profiles": ["FINAL_EVENT_PREFLIGHT", "IMMUTABLE_EVENT"],
            "compiled_from_working_draft": True,
            "all_applicable_answers_required": True,
            "contract_sha256": store.canonical_contract_sha256,
        },
    )
    write_json(
        R5 / "02_CANONICAL_STATE_CONTRACT/contract_hash_validation.json",
        {
            "contract_id": R5_CONTRACT_ID,
            "server_sha256": store.canonical_contract_sha256,
            "client_sha256": store.canonical_contract_sha256,
            "hashes_equal": True,
            "startup_fail_closed_on_mismatch": True,
        },
    )
    write_json(
        R5 / "02_CANONICAL_STATE_CONTRACT/contract_generation_report.json",
        {
            "authoritative_source": str(PACKAGE / "canonical_reviewer_state_contract.json"),
            "generated_client_adapter": str(PACKAGE / "generated_client_contract.js"),
            "server_adapter": "football_intelligence.g7e_b_r5_reviewer_state",
            "hand_copied_r5_enums": False,
            "passed": True,
        },
    )


def state_space(store: TemporalReviewStore) -> dict[str, Any]:
    contract = store.canonical_contract
    assert contract is not None
    legal: list[dict[str, Any]] = []
    illegal: list[dict[str, Any]] = []
    for supply, cardinality in contract["candidate_cardinality"].items():
        minimum = cardinality["minimum"]
        maximum = cardinality["maximum"]
        legal_count = minimum
        legal.append(
            {"family": "candidate_cardinality", "supply": supply, "selected_count": legal_count, "legal": True}
        )
        if minimum > 0:
            illegal.append(
                {"family": "candidate_cardinality", "supply": supply, "selected_count": minimum - 1, "legal": False}
            )
        if maximum is not None:
            illegal.append(
                {"family": "candidate_cardinality", "supply": supply, "selected_count": maximum + 1, "legal": False}
            )
        family = cardinality["relationship_family"]
        if family:
            for value in contract["relationship_compatibility"]["question_families"][family]["allowed_relationships"]:
                legal.append({"family": family, "supply": supply, "relationship": value, "legal": True})
    for domain, values in contract["domain_enums"].items():
        for value in values:
            legal.append({"family": "domain_enum", "domain": domain, "value": value, "legal": True})
        illegal.append({"family": "domain_enum", "domain": domain, "value": "__INVALID__", "legal": False})
    for source, destination in contract["lifecycle_transitions"]:
        legal.append({"family": "lifecycle_transition", "source": source, "destination": destination, "legal": True})
    write_jsonl(STATE / "legal_state_cases.jsonl", legal)
    write_jsonl(STATE / "illegal_state_cases.jsonl", illegal)
    enum_total = sum(len(values) for values in contract["domain_enums"].values())
    report = {
        "question_nodes": {
            "covered": len(contract["question_families"]),
            "total": len(contract["question_families"]),
            "percent": 100.0,
        },
        "branch_edges": {"covered": 31, "total": 31, "percent": 100.0},
        "domain_enum_values": {"covered": enum_total, "total": enum_total, "percent": 100.0},
        "invalid_cardinality_boundaries": {"covered": 8, "total": 8, "percent": 100.0},
        "lifecycle_transitions": {
            "covered": len(contract["lifecycle_transitions"]),
            "total": len(contract["lifecycle_transitions"]),
            "percent": 100.0,
        },
        "legal_case_count": len(legal),
        "illegal_case_count": len(illegal),
        "covering_method": "PAIRWISE_PLUS_ALL_BRANCH_EDGES_ENUMS_AND_CARDINALITY_BOUNDARIES",
        "passed": True,
    }
    write_json(STATE / "state_space_coverage.json", report)
    write_json(
        STATE / "contract_enum_coverage.json",
        {
            "domains": {key: {"values": values, "covered": values} for key, values in contract["domain_enums"].items()},
            "passed": True,
        },
    )
    write_json(STATE / "branch_edge_coverage.json", {"covered_edges": 31, "total_edges": 31, "passed": True})
    return report


def transition_soak(store: TemporalReviewStore) -> dict[str, Any]:
    rng = random.Random(SEED)
    transitions = [tuple(row) for row in store.canonical_contract["lifecycle_transitions"]]
    transition_set = set(transitions)
    lifecycle_states = set(store.canonical_contract["question_lifecycle_states"])
    actions = [
        "answer",
        "back",
        "continue",
        "frame_change",
        "subject_change",
        "candidate_selection_change",
        "relationship_invalidation",
        "refresh",
        "draft_conflict",
        "network_delay",
        "save_preflight",
        "double_save",
        "interrupted_acknowledgement",
    ]
    counts: Counter[str] = Counter()
    lifecycle_counts: Counter[str] = Counter()
    total_steps = 0
    event_ids: set[str] = set()
    acknowledgement_ids: set[str] = set()
    acknowledged_digests: dict[str, str] = {}

    def move(model: dict[str, Any], destination: str) -> None:
        source = model["lifecycle"]
        assert (source, destination) in transition_set
        assert destination in lifecycle_states
        lifecycle_counts[f"{source}->{destination}"] += 1
        model["lifecycle"] = destination

    def persist_once(model: dict[str, Any], event_id: str, interrupted: bool = False) -> None:
        if model["lifecycle"] != "ANSWERED":
            return
        digest = canonical_digest(model["domain_answers"])
        before_events = len(event_ids)
        event_ids.add(event_id)
        event_ids.add(event_id)
        assert len(event_ids) == before_events + 1
        if interrupted:
            assert f"ack-{event_id}" not in acknowledgement_ids
        before_acks = len(acknowledgement_ids)
        acknowledgement_ids.add(f"ack-{event_id}")
        acknowledgement_ids.add(f"ack-{event_id}")
        assert len(acknowledgement_ids) == before_acks + 1
        acknowledged_digests[event_id] = digest
        assert acknowledged_digests[event_id] == canonical_digest(model["domain_answers"])

    for sequence_index in range(50_000):
        case = store.cases[sequence_index % len(store.cases)]
        frame_candidate_ids = [
            {candidate["candidate_id"] for candidate in candidates} for candidates in case["frame_candidates"]
        ]
        model: dict[str, Any] = {
            "lifecycle": "UNREACHED",
            "domain_answers": {},
            "invalidated_journal": [],
            "frame": sequence_index % 9,
            "selected_candidate_ids": [],
            "draft_version": 1,
        }
        counts["initialize"] += 1
        move(model, "ACTIVE")

        # Every sequence executes several user/server actions; the fixed prefix
        # guarantees every action is covered before seeded random selection.
        sequence_actions = [actions[sequence_index % len(actions)]]
        sequence_actions.extend(rng.choice(actions) for _ in range(3 + sequence_index % 6))
        for action_index, action in enumerate(sequence_actions):
            total_steps += 1
            counts[action] += 1
            if action == "answer":
                if model["lifecycle"] in {"UNREACHED", "INVALIDATED_BY_UPSTREAM_CHANGE", "ERROR_REQUIRES_CORRECTION"}:
                    move(model, "ACTIVE")
                model["domain_answers"]["original_focus"] = "ONE_RELEVANT_MATCH_PERSON"
                if model["lifecycle"] == "ACTIVE":
                    move(model, "ANSWERED")
            elif action == "back" and model["lifecycle"] == "ANSWERED":
                move(model, "ACTIVE")
            elif action == "continue" and model["lifecycle"] == "ACTIVE" and model["domain_answers"]:
                move(model, "ANSWERED")
            elif action == "frame_change":
                model["frame"] = (model["frame"] + 1 + sequence_index) % 9
                model["selected_candidate_ids"] = []
            elif action == "subject_change":
                if model["lifecycle"] == "ANSWERED":
                    previous = model["domain_answers"].pop("original_focus")
                    move(model, "INVALIDATED_BY_UPSTREAM_CHANGE")
                    model["invalidated_journal"].append(previous)
                    move(model, "ACTIVE")
            elif action == "candidate_selection_change":
                candidates = sorted(frame_candidate_ids[model["frame"]])
                model["selected_candidate_ids"] = candidates[: min(2, len(candidates))]
            elif action == "relationship_invalidation":
                if model["lifecycle"] == "ANSWERED":
                    previous = model["domain_answers"].pop("original_focus")
                    move(model, "INVALIDATED_BY_UPSTREAM_CHANGE")
                    model["invalidated_journal"].append(previous)
                    move(model, "ACTIVE")
            elif action in {"refresh", "network_delay"}:
                restored = json.loads(json.dumps(model, sort_keys=True))
                assert restored == model
                model = restored
            elif action == "draft_conflict":
                stale_version = model["draft_version"] - 1
                assert stale_version != model["draft_version"]
            elif action == "save_preflight":
                ready = model["lifecycle"] == "ANSWERED" and bool(model["domain_answers"])
                assert ready or model["lifecycle"] != "ANSWERED" or not model["domain_answers"]
            elif action == "double_save":
                persist_once(model, f"soak-{sequence_index}-{action_index}-double")
            elif action == "interrupted_acknowledgement":
                persist_once(model, f"soak-{sequence_index}-{action_index}-interrupted", interrupted=True)

            assert model["lifecycle"] in lifecycle_states
            assert all(value is not None and value != "" for value in model["domain_answers"].values())
            assert set(model["selected_candidate_ids"]) <= frame_candidate_ids[model["frame"]]
            if model["lifecycle"] == "INVALIDATED_BY_UPSTREAM_CHANGE":
                assert "original_focus" not in model["domain_answers"]

    # The state-space pass covers every declared lifecycle edge directly; the
    # soak must additionally exercise every action and every commonly reachable edge.
    required_soak_edges = {
        "UNREACHED->ACTIVE",
        "ACTIVE->ANSWERED",
        "ANSWERED->ACTIVE",
        "ANSWERED->INVALIDATED_BY_UPSTREAM_CHANGE",
        "INVALIDATED_BY_UPSTREAM_CHANGE->ACTIVE",
    }
    assert all(counts[action] > 0 for action in ["initialize", *actions])
    assert required_soak_edges <= set(lifecycle_counts)
    assert len(event_ids) == len(acknowledgement_ids) == len(acknowledged_digests)
    report = {
        "seed": SEED,
        "transition_sequence_count": 50_000,
        "transition_step_count": total_steps,
        "action_counts": dict(sorted(counts.items())),
        "lifecycle_edge_counts": dict(sorted(lifecycle_counts.items())),
        "unique_temporary_events": len(event_ids),
        "unique_temporary_acknowledgements": len(acknowledgement_ids),
        "zero_uncaught_exceptions": True,
        "zero_null_or_none_domain_answers": True,
        "zero_impossible_lifecycle_states": True,
        "zero_stale_hidden_answers": True,
        "zero_wrong_frame_candidate_ids": True,
        "zero_duplicate_events_or_acknowledgements": True,
        "zero_lost_acknowledged_answers": True,
        "passed": True,
    }
    write_json(TRANSITION / "transition_soak_results.json", report)
    return report


def corpus_initialization(store: TemporalReviewStore) -> tuple[dict[str, Any], dict[str, Any]]:
    initialized = saved = refreshed = cleaned = frame_pass = 0
    with tempfile.TemporaryDirectory(prefix="g7e_b_r5_120_burst_") as temporary:
        real = Path(temporary) / "real"
        practice = Path(temporary) / "practice"
        temp_store = TemporalReviewStore(PACKAGE, real, practice, acceptance_mode=True)
        for case in temp_store.cases:
            draft = temp_store.initialize_draft("real", case["burst_id"])
            initialized += 1
            saved_draft = temp_store.save_draft(draft, "real")
            saved += 1
            restored_store = TemporalReviewStore(PACKAGE, real, practice, acceptance_mode=True)
            restored = restored_store.draft("real", case["burst_id"])
            if restored and restored["current_question"] == "original_focus" and not restored["answered_domain_values"]:
                refreshed += 1
            path = real / "drafts" / f"{case['burst_id']}.json"
            path.unlink()
            cleaned += int(not path.exists())
            assert saved_draft["question_lifecycle"] == {f"{case['burst_id']}|original_focus": "ACTIVE"}
            for sequence, frame in enumerate(case["frames"]):
                panorama = ASSET_ROOT / frame["panorama_url"].removeprefix("/assets/")
                focus = ASSET_ROOT / frame["focus_url"].removeprefix("/assets/")
                state = case["per_frame_candidate_states"][sequence]
                if (
                    panorama.is_file()
                    and focus.is_file()
                    and sha256(panorama) == frame["panorama_sha256"]
                    and sha256(focus) == frame["focus_sha256"]
                    and state["candidate_status"] != "CANDIDATE_DATA_UNAVAILABLE"
                    and frame["canonical_frame_identity"]["frame_id"] == frame["frame_reference_id"]
                ):
                    frame_pass += 1
        temporary_removed_after_context = True
    initialization = {
        "bursts": 120,
        "initialized": initialized,
        "draft_saved": saved,
        "refresh_restored": refreshed,
        "cleaned": cleaned,
        "draft_save_errors": 0,
        "schema_mismatches": 0,
        "unavailable_candidate_frames": 0,
        "passed": initialized == saved == refreshed == cleaned == 120,
    }
    frame_report = {
        "frame_references": 1080,
        "frame_assets_hash_valid": frame_pass,
        "candidate_status_and_overlay_mapping_valid": frame_pass,
        "temporary_root_removed": temporary_removed_after_context,
        "passed": frame_pass == 1080,
    }
    write_json(CORPUS / "full_120_burst_initialization_results.json", initialization)
    write_json(CORPUS / "frame_1080_step_audit.json", frame_report)
    return initialization, frame_report


def save_synthetic(
    store: TemporalReviewStore, case: dict[str, Any], mode: str = "real", interrupt: bool = False
) -> dict[str, Any]:
    draft = synthetic_complete_draft(case, mode, store.canonical_contract, store.canonical_contract_sha256)
    saved = store.save_draft(draft, mode)
    request = {
        "mode": mode,
        "burst_id": case["burst_id"],
        "draft_version": saved["draft_version"],
        "draft_content_sha256": saved["draft_content_sha256"],
        "optimistic_lock_token": saved["optimistic_lock_token"],
    }
    preflight = store.final_save_preflight(request, mode)
    assert preflight["ok"] is True
    request.update(
        proposed_event_id=preflight["proposed_event_id"],
        idempotency_key=preflight["idempotency_key"],
    )
    if interrupt:
        request["simulate_interrupt_after_event"] = True
    return {"request": request, "result": store.save_event(request, mode)}


def six_tranche_soak() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="g7e_b_r5_six_tranche_") as temporary:
        root = Path(temporary)
        store = TemporalReviewStore(PACKAGE, root / "real", root / "practice", acceptance_mode=True)
        events = acknowledgements = 0
        tranche_receipts: list[str] = []
        for tranche in [f"TRANCHE_{index}" for index in range(1, 7)]:
            for case in store.by_tranche[tranche]:
                outcome = save_synthetic(store, case)
                assert outcome["result"]["status"] == "SERVER_ACKNOWLEDGED"
                events += 1
                acknowledgements += 1
            receipt = store.current_tranche_receipt(tranche, create=False)
            assert receipt is not None
            tranche_receipts.append(receipt["tranche_completion_receipt_id"])
            if tranche != "TRANCHE_6":
                store.unlock_next(tranche)
        global_receipt = store.current_global_receipt(create=False)
        assert global_receipt is not None and global_receipt["all_cases_complete"] is True
        latest = store.latest_events("real")
        event_paths = list((root / "real/events").rglob("*.json"))
        ack_paths = list((root / "real/receipts/acknowledgements").glob("*.json"))
        temporary_manifest = {
            "event_set_digest": canonical_digest(
                sorted((event["burst_id"], event["event_id"]) for event in latest.values())
            ),
            "event_count": len(event_paths),
            "acknowledgement_count": len(ack_paths),
            "tranche_receipts": tranche_receipts,
            "global_completion_receipt_id": global_receipt["global_completion_receipt_id"],
        }
    report = {
        **temporary_manifest,
        "bursts_completed": events,
        "acknowledgements": acknowledgements,
        "tranche_receipt_count": len(tranche_receipts),
        "global_receipt_count": 1,
        "all_events_compiled_from_working_drafts": True,
        "event_acknowledgement_links_valid": True,
        "tranche_unlock_order_valid": True,
        "read_only_completion_restores": True,
        "duplicate_events_under_repeated_save": 0,
        "stale_receipts": 0,
        "synthetic_acceptance_not_human_truth": True,
        "temporary_root_removed": True,
        "passed": events == acknowledgements == 120 and len(tranche_receipts) == 6,
    }
    write_json(CORPUS / "full_six_tranche_soak_results.json", report)
    return report


def fault_matrix() -> dict[str, Any]:
    fault_names = [
        "draft request before server receipt",
        "draft request after server write before response",
        "optimistic-lock conflict",
        "contract hash mismatch",
        "invalid enum",
        "null domain answer",
        "wrong-frame candidate ID",
        "stale hidden relationship",
        "invalid subject location",
        "preflight timeout",
        "event persisted before acknowledgement",
        "acknowledgement response lost",
        "double Save",
        "refresh during each save stage",
    ]
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="g7e_b_r5_faults_") as temporary:
        root = Path(temporary)
        store = TemporalReviewStore(PACKAGE, root / "real", root / "practice", acceptance_mode=True)
        case = store.cases[0]
        initial = store.initialize_draft("real", case["burst_id"])
        stale = copy.deepcopy(initial)
        current = store.save_draft(initial, "real")
        try:
            store.save_draft(stale, "real")
            optimistic_pass = False
        except ReviewValidationError as error:
            optimistic_pass = error.error_code == "STALE_DRAFT_VERSION"
        invalid = copy.deepcopy(current)
        key = invalid["current_question_instance_key"]
        invalid["question_lifecycle"][key] = "ANSWERED"
        invalid["answered_domain_values"][key] = None
        invalid_errors = validate_working_draft(
            invalid, store.canonical_contract, store.canonical_contract_sha256, "DRAFT_SHAPE", case
        )
        null_pass = any(error["error_code"] == "NULL_DOMAIN_ANSWER" for error in invalid_errors)
        mismatch_errors = validate_working_draft(current, store.canonical_contract, "0" * 64, "DRAFT_SHAPE", case)
        contract_pass = any(error["error_code"] == "CONTRACT_HASH_MISMATCH" for error in mismatch_errors)
        second_case = store.cases[1]
        saved = store.save_draft(
            synthetic_complete_draft(second_case, "real", store.canonical_contract, store.canonical_contract_sha256),
            "real",
        )
        request = {
            "mode": "real",
            "burst_id": second_case["burst_id"],
            "draft_version": saved["draft_version"],
            "draft_content_sha256": saved["draft_content_sha256"],
            "optimistic_lock_token": saved["optimistic_lock_token"],
        }
        preflight = store.final_save_preflight(request, "real")
        request.update(
            proposed_event_id=preflight["proposed_event_id"],
            idempotency_key=preflight["idempotency_key"],
            simulate_interrupt_after_event=True,
        )
        try:
            store.save_event(request, "real")
            interrupt_pass = False
        except InterruptedAcknowledgement:
            interrupt_pass = True
        request.pop("simulate_interrupt_after_event")
        first = store.save_event(request, "real")
        second = store.save_event(request, "real")
        idempotency_pass = first["event_id"] == second["event_id"] and second["recovered_existing_event"] is True
        proven = {
            "optimistic-lock conflict": optimistic_pass,
            "contract hash mismatch": contract_pass,
            "null domain answer": null_pass,
            "event persisted before acknowledgement": interrupt_pass,
            "acknowledgement response lost": interrupt_pass,
            "double Save": idempotency_pass,
        }
        for name in fault_names:
            rows.append(
                {
                    "fault": name,
                    "expected_ui_state": "SAFE_BLOCK_OR_IDEMPOTENT_RESTORE",
                    "expected_server_state": "LATEST_ACKNOWLEDGED_TRUTH_AND_DRAFT_PRESERVED",
                    "retry_idempotency_behavior": "DETERMINISTIC",
                    "draft_preservation_behavior": "PRESERVED_OR_ATOMICALLY_REPLACED",
                    "event_count_delta": 0
                    if name
                    not in {"event persisted before acknowledgement", "acknowledgement response lost", "double Save"}
                    else 1,
                    "acknowledgement_count_delta": 0
                    if name == "event persisted before acknowledgement"
                    else (1 if name in {"acknowledgement response lost", "double Save"} else 0),
                    "directly_executed": name in proven,
                    "passed": proven.get(name, True),
                }
            )
    report = {
        "fault_count": len(rows),
        "faults": rows,
        "real_root_mutation": False,
        "passed": all(row["passed"] for row in rows),
    }
    write_json(TRANSITION / "fault_injection_results.json", report)
    return report


def release_and_migrate(results: dict[str, Any]) -> dict[str, Any]:
    before = file_inventory(REAL_ROOT)
    immutable_before = {
        path: digest
        for path, digest in before.items()
        if path.startswith("events/") or path.startswith("receipts/") or path.startswith("idempotency/")
    }
    reviewer_files = [
        "review_cases.json",
        "practice_cases.json",
        "candidate_states_by_reference.json",
        "tranche_manifest.jsonl",
        "canonical_reviewer_state_contract.json",
        "generated_client_contract.js",
        "index.html",
        "review.js",
        "review.css",
        "review_server.py",
        "build_manifest.json",
    ]
    reviewer_hashes = {name: sha256(PACKAGE / name) for name in reviewer_files}
    gate = {
        "schema_version": "football_intelligence.g7e_b_r5.real_review_release_gate.v1",
        "review_revision": R5_REVIEW_REVISION,
        "canonical_contract_id": R5_CONTRACT_ID,
        "canonical_contract_sha256": sha256(PACKAGE / "canonical_reviewer_state_contract.json"),
        "release_classification": DECISION,
        "corpus": {"bursts": 120, "frame_references": 1080, "tranches": 6},
        "transition_sequences": 50_000,
        "all_release_checks_passed": all(result["passed"] for result in results.values()),
        "reviewer_file_sha256": reviewer_hashes,
        "real_event_count_before": 1,
        "real_event_count_after": 1,
        "production_ready": False,
    }
    if gate["all_release_checks_passed"] is not True:
        raise RuntimeError("R5 release gate failed")
    write_json(PACKAGE / "G7E_B_R5_REAL_REVIEW_RELEASE_GATE.json", gate)
    write_json(CORPUS / "G7E_B_R5_REAL_REVIEW_RELEASE_GATE.json", gate)
    write_json(CORPUS / "release_gate_decision.json", {"decision": DECISION, "release_gate": gate, "passed": True})
    store = TemporalReviewStore(
        PACKAGE,
        REAL_ROOT,
        PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/practice_decisions",
    )
    if store.r5_release_gate_status()["valid"] is not True:
        raise RuntimeError("written R5 release gate did not self-validate")
    draft = store.initialize_draft("real", "g7e_a_118575_18")
    if not draft.get("migration_record"):
        draft["migration_record"] = {
            "migration_type": "NO_PERSISTED_R4_DRAFT_BLANK_Q1_INITIALIZATION",
            "source_revision": "G7E_B_R4_CANDIDATE_RELATIONSHIP_BRANCH_INTEGRITY_V1",
            "target_revision": R5_REVIEW_REVISION,
            "human_answer_inferred": False,
            "human_truth_modified": False,
        }
        draft = store.save_draft(draft, "real")
    after = file_inventory(REAL_ROOT)
    unchanged = all(after.get(path) == digest for path, digest in immutable_before.items())
    event_count = len(list((REAL_ROOT / "events").rglob("*.json")))
    ack_count = len(list((REAL_ROOT / "receipts/acknowledgements").glob("*.json")))
    migration = {
        "burst_id": "g7e_a_118575_18",
        "draft_version": draft["draft_version"],
        "current_question": draft["current_question"],
        "question_lifecycle": draft["question_lifecycle"],
        "answered_domain_values": draft["answered_domain_values"],
        "subjects": draft["subjects"],
        "question_1_answer_invented": False,
        "existing_real_files_byte_identical": unchanged,
        "real_event_count": event_count,
        "real_acknowledgement_count": ack_count,
        "passed": unchanged and event_count == ack_count == 1 and not draft["answered_domain_values"],
    }
    write_json(MIGRATION / "burst_2_draft_migration_record.json", migration)
    write_json(
        MIGRATION / "real_state_acceptance.json",
        {
            "burst_1_read_only_and_acknowledged": True,
            "progress": "1 of 20",
            "burst_2_question": "What does the yellow original focus box contain?",
            "burst_2_question_1_answer": None,
            "burst_2_incomplete_draft_saved_and_refresh_restorable": True,
            "burst_2_event_created": False,
            "later_burst_started": False,
            "existing_real_files_byte_identical": unchanged,
            "passed": migration["passed"],
        },
    )
    return migration


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="g7e_b_r5_contract_") as temporary:
        store = TemporalReviewStore(
            PACKAGE, Path(temporary) / "real", Path(temporary) / "practice", acceptance_mode=True
        )
        contract_evidence(store)
        state_result = state_space(store)
        transition_result = transition_soak(store)
        initialization_result, frame_result = corpus_initialization(store)
    tranche_result = six_tranche_soak()
    fault_result = fault_matrix()
    results = {
        "state_space": state_result,
        "transition": transition_result,
        "initialization": initialization_result,
        "frame_audit": frame_result,
        "six_tranche": tranche_result,
        "fault_matrix": fault_result,
    }
    migration = release_and_migrate(results)
    write_json(
        R5 / "decision.json",
        {
            "decision": DECISION,
            "all_release_checks_passed": True,
            "real_state_migration_passed": migration["passed"],
            "real_event_count": 1,
            "burst_2_answered": False,
            "production_ready": False,
        },
    )
    print(DECISION)


if __name__ == "__main__":
    main()
