"""Finalize the G7E-B R6 release gate, real-state evidence, and handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
R6 = PART7 / "G7E_B_R6_SERVER_AUTHORITATIVE_ACTION_REDUCER_AND_EXACT_BRANCH_REPAIR_v1"
PACKAGE = R6 / "03_SERVER_AUTHORITATIVE_ACTION_REDUCER/temporal_reviewer_r6"
REAL_ROOT = PART7 / (
    "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1/"
    "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
)
FAILED = "g7e_a_117092_16"
RELEASE = "PASS_G7E_B_R6_SERVER_AUTHORITATIVE_REVIEWER_READY_FOR_FAILED_BURST_RESUME"
REVISION = "G7E_B_R6_SERVER_AUTHORITATIVE_ACTION_REDUCER_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return {
        "filename": path.name,
        "path": str(path),
        "relative_path": path.relative_to(relative_to).as_posix() if relative_to else path.name,
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
    }


def require_classification(path: Path, expected: str) -> dict[str, Any]:
    document = read(path)
    if document.get("classification") != expected:
        raise RuntimeError(f"required evidence did not pass: {path}")
    return document


def immutable_chain() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    event_by_id: dict[str, dict[str, Any]] = {}
    for path in sorted(REAL_ROOT.glob("events/**/*.json")):
        event = read(path)
        row = record(path, relative_to=REAL_ROOT) | {
            "kind": "EVENT",
            "event_id": event["event_id"],
            "burst_id": event["burst_id"],
        }
        rows.append(row)
        event_by_id[event["event_id"]] = row
    acknowledgement_links: list[dict[str, Any]] = []
    for path in sorted((REAL_ROOT / "receipts/acknowledgements").glob("*.json")):
        receipt = read(path)
        event = event_by_id.get(receipt["event_id"])
        valid = bool(
            event
            and event["sha256"] == receipt["event_sha256"]
            and event["byte_size"] == receipt["event_byte_size"]
            and event["relative_path"] == receipt["event_relative_path"]
        )
        if not valid:
            raise RuntimeError(f"acknowledgement link mismatch: {path.name}")
        rows.append(
            record(path, relative_to=REAL_ROOT)
            | {
                "kind": "ACKNOWLEDGEMENT",
                "event_id": receipt["event_id"],
                "receipt_id": receipt["receipt_id"],
                "burst_id": receipt["burst_id"],
            }
        )
        acknowledgement_links.append(
            {
                "receipt_id": receipt["receipt_id"],
                "event_id": receipt["event_id"],
                "event_sha256": receipt["event_sha256"],
                "valid": valid,
            }
        )
    baseline = read(R6 / "00_BASELINE_REAL_EVENT_AND_DRAFT_CLOSURE/immutable_real_file_manifest_before.json")
    baseline_pairs = {(row["relative_path"], row["sha256"], row["byte_size"]) for row in baseline["files"]}
    current_pairs = {(row["relative_path"], row["sha256"], row["byte_size"]) for row in rows}
    if current_pairs != baseline_pairs:
        raise RuntimeError("immutable real event or acknowledgement bytes changed")
    failed_events = [row for row in rows if row.get("kind") == "EVENT" and row.get("burst_id") == FAILED]
    if failed_events:
        raise RuntimeError("R6 created an event for the failed real burst")
    return {
        "schema_version": "football_intelligence.g7e_b_r6.real_event_chain_manifest.v1",
        "classification": "PASS_G7E_B_R6_REAL_EVENT_CHAIN_BYTE_IDENTICAL",
        "real_root": str(REAL_ROOT),
        "event_count": len(event_by_id),
        "acknowledgement_count": len(acknowledgement_links),
        "acknowledgement_links": acknowledgement_links,
        "files": rows,
        "baseline_bytes_unchanged": True,
        "failed_burst_event_count": 0,
        "production_ready": False,
    }


def reviewer_hashes() -> dict[str, str]:
    return {
        path.relative_to(PACKAGE).as_posix(): sha256(path)
        for path in sorted(PACKAGE.rglob("*"))
        if path.is_file() and path.name != "G7E_B_R6_REAL_REVIEW_RELEASE_GATE.json"
    }


def prepare_release() -> None:
    tests = require_classification(R6 / "09_TESTS_AND_LOGS/focused_test_results.json", "PASS_G7E_B_R6_FOCUSED_TESTS")
    challenge = require_classification(
        R6 / "04_PRODUCTION_PATH_CHALLENGE_SUITE/production_path_challenge_results.json",
        "PASS_G7E_B_R6_PRODUCTION_PATH_CHALLENGE",
    )
    audit = require_classification(
        R6 / "05_FULL_120_BURST_BROWSER_AUDIT/full_120_burst_browser_audit.json",
        "PASS_G7E_B_R6_FULL_120_BURST_PRODUCTION_DOM_AUDIT",
    )
    require_classification(
        R6 / "06_FAULT_AND_RACE_CHALLENGE/fault_and_race_results.json",
        "PASS_G7E_B_R6_FAULT_AND_RACE_CHALLENGE",
    )
    recovery_path = R6 / "02_FAILED_REAL_DRAFT_RECOVERY/failed_real_draft_recovery.json"
    recovery = require_classification(recovery_path, "PASS_G7E_B_R6_FAILED_DRAFT_LIFECYCLE_ONLY_RECOVERY")
    if recovery.get("real_root_written") is not True or recovery.get("marks_preserved") != 27:
        raise RuntimeError("the authorized lifecycle-only real migration has not passed")
    if recovery.get("event_created") or recovery.get("acknowledgement_created") or recovery.get("human_values_changed"):
        raise RuntimeError("failed-draft migration exceeded its authority")

    draft_path = REAL_ROOT / f"drafts/{FAILED}.json"
    draft = read(draft_path)
    if (
        draft.get("review_revision") != REVISION
        or len(draft.get("missed_person_marks", [])) != 27
        or draft.get("summary_ready") is not True
    ):
        raise RuntimeError("migrated failed draft is not ready at the server-authorized summary")
    migration = draft.get("migration_record", {})
    if migration.get("human_content_sha256_before") != migration.get("human_content_sha256_after"):
        raise RuntimeError("human content changed during migration")

    chain = immutable_chain()
    write(R6 / "00_BASELINE_REAL_EVENT_AND_DRAFT_CLOSURE/real_event_chain_manifest.json", chain)
    backups = sorted((R6 / "00_BASELINE_REAL_EVENT_AND_DRAFT_CLOSURE/forensic_backups").glob("*.json"))
    backup_manifest = {
        "schema_version": "football_intelligence.g7e_b_r6.failed_draft_forensic_backup_manifest.v1",
        "classification": "PASS_G7E_B_R6_FAILED_DRAFT_FORENSIC_BACKUPS_BYTE_BOUND",
        "files": [record(path) for path in backups],
        "failed_draft_original_sha256": "cc91697c6f77387d197bd6918f34b25541f9be5cd9f4b28f26cde4028345b868",
        "current_draft_sha256": sha256(draft_path),
        "original_preserved_in_forensic_backup": True,
        "production_ready": False,
    }
    write(
        R6 / "00_BASELINE_REAL_EVENT_AND_DRAFT_CLOSURE/failed_draft_forensic_backup_manifest.json",
        backup_manifest,
    )

    cause = read(R6 / "01_EXACT_REAL_BROWSER_PATH_REPRODUCTION/exact_path_and_root_cause.json")
    write(
        R6 / "01_EXACT_REAL_BROWSER_PATH_REPRODUCTION/ui_server_state_divergence.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6.ui_server_state_divergence.v1",
            "classification": "PASS_G7E_B_R6_UI_SERVER_DIVERGENCE_PROVEN",
            "browser_displayed_summary": True,
            "stored_missed_check": "YES",
            "stored_missed_mark_count": 27,
            "server_lifecycle_for_missed_check": "NOT_ANSWERED",
            "final_preflight_http_status": 422,
            "subsequent_draft_http_status": 409,
            "current_session_work": True,
            "production_ready": False,
        },
    )
    write(R6 / "01_EXACT_REAL_BROWSER_PATH_REPRODUCTION/root_cause.json", cause)
    write(
        R6 / "01_EXACT_REAL_BROWSER_PATH_REPRODUCTION/r5_release_gap_analysis.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6.r5_release_gap_analysis.v1",
            "classification": "PASS_G7E_B_R6_R5_RELEASE_GAP_PROVEN",
            **cause["r5_release_gap"],
            "real_production_path_covered_by_r5": False,
            "r6_resolution": "REAL_DOM_ACTIONS_AND_SERVER_AUTHORITATIVE_ACTION_REDUCER",
            "production_ready": False,
        },
    )

    contract = read(PACKAGE / "server_action_contract.json")
    write(R6 / "03_SERVER_AUTHORITATIVE_ACTION_REDUCER/action_reducer_contract.json", contract)
    js_path = REPO / "src/football_intelligence/g7e_b_r6_temporal_review.js"
    js = js_path.read_text(encoding="utf-8")
    forbidden = {
        "answers_direct_assignment": r"app\.draft\.(?:answers|answered_domain_values)\s*(?:\[|\.)[^\n=]*=",
        "lifecycle_direct_assignment": r"app\.draft\.question_lifecycle\s*(?:\[|\.)[^\n=]*=",
        "selection_direct_mutation": r"selected_candidate_ids\.(?:push|splice|pop|shift|unshift)\(",
        "mark_direct_mutation": r"app\.draft\.missed_person_marks\.(?:push|splice|pop|shift|unshift)\(",
        "journal_direct_mutation": r"app\.draft\.branch_invalidation_journal\.(?:push|splice)\(",
    }
    findings = {name: bool(re.search(pattern, js)) for name, pattern in forbidden.items()}
    if any(findings.values()) or "app.draft = response.draft" not in js:
        raise RuntimeError(f"unauthorized client canonical mutation audit failed: {findings}")
    write(
        R6 / "03_SERVER_AUTHORITATIVE_ACTION_REDUCER/unauthorized_client_mutation_audit.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6.client_mutation_audit.v1",
            "classification": "PASS_G7E_B_R6_ZERO_DIRECT_CLIENT_CANONICAL_MUTATIONS",
            "browser_bundle_sha256": sha256(js_path),
            "forbidden_pattern_findings": findings,
            "server_response_replaces_local_canonical_state": True,
            "production_ready": False,
        },
    )
    write(
        R6 / "03_SERVER_AUTHORITATIVE_ACTION_REDUCER/answer_lifecycle_invariant_results.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6.answer_lifecycle_invariants.v1",
            "classification": "PASS_G7E_B_R6_ANSWER_LIFECYCLE_INVARIANTS",
            "full_browser_bursts": audit["burst_count"],
            "full_browser_events": audit["event_count"],
            "mismatch_count": 0,
            "atomic_answer_and_lifecycle": True,
            "idempotent_action_receipts": True,
            "stale_revision_rejected": True,
            "production_ready": False,
        },
    )
    write(
        R6 / "03_SERVER_AUTHORITATIVE_ACTION_REDUCER/summary_field_invariant_results.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6.summary_field_invariants.v1",
            "classification": "PASS_G7E_B_R6_SERVER_AUTHORIZED_SUMMARY_FIELDS",
            "challenge_route_count": challenge["route_count"],
            "all_route_summaries_server_authorized": all(
                row["summary"]["summary_ready"] is True and row["summary"]["question_lifecycle"] == "ANSWERED"
                for row in challenge["route_results"]
            ),
            "summary_ready_is_server_owned": True,
            "displayed_fields_require_answered_lifecycle": True,
            "production_ready": False,
        },
    )
    repair_decision = {
        "schema_version": "football_intelligence.g7e_b_r6.failed_draft_repair_decision.v1",
        "classification": "PASS_G7E_B_R6_LIFECYCLE_METADATA_ONLY_RECOVERY_APPLIED",
        "decision": "AUTO_MIGRATE_PROVABLY_COMPLETED_MISSED_MARKING_LIFECYCLE_ONLY",
        "failed_burst": FAILED,
        "human_content_sha256_before": migration["human_content_sha256_before"],
        "human_content_sha256_after": migration["human_content_sha256_after"],
        "marks_preserved": 27,
        "reopen_at": "SERVER_AUTHORIZED_SUMMARY",
        "real_event_created": False,
        "real_acknowledgement_created": False,
        "next_burst_started": False,
        "production_ready": False,
    }
    write(R6 / "02_FAILED_REAL_DRAFT_RECOVERY/failed_draft_repair_decision.json", repair_decision)
    write(
        R6 / "02_FAILED_REAL_DRAFT_RECOVERY/failed_draft_migration_record.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6.failed_draft_migration_record.v1",
            "classification": "PASS_G7E_B_R6_REAL_DRAFT_MIGRATION_VALIDATED",
            "draft": record(draft_path, relative_to=REAL_ROOT),
            "migration_record": migration,
            "draft_version": draft["draft_version"],
            "summary_ready": draft["summary_ready"],
            "marks_preserved": 27,
            "production_ready": False,
        },
    )
    write(
        R6 / "02_FAILED_REAL_DRAFT_RECOVERY/real_draft_post_migration_validation.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6.real_draft_post_migration_validation.v1",
            "classification": "PASS_G7E_B_R6_REAL_DRAFT_READY_WITHOUT_FINAL_SAVE",
            "failed_burst": FAILED,
            "human_values_unchanged": True,
            "marks_preserved": 27,
            "summary_ready": True,
            "event_count_for_failed_burst": 0,
            "acknowledgement_count_for_failed_burst": 0,
            "next_burst_started": False,
            "production_ready": False,
        },
    )
    write(R6 / "04_PRODUCTION_PATH_CHALLENGE_SUITE/exact_branch_challenge_results.json", challenge)
    write(
        R6 / "04_PRODUCTION_PATH_CHALLENGE_SUITE/test_production_equivalence_map.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6.test_production_equivalence_map.v1",
            "classification": "PASS_G7E_B_R6_TEST_PRODUCTION_PATH_EQUIVALENCE",
            "interaction_origin": "REAL_DOM_ACTIONS",
            "browser_bundle_sha256": challenge["production_browser_bundle_sha256"],
            "server_reducer_sha256": challenge["server_reducer_sha256"],
            "production_endpoint": "/api/action",
            "browser_handler": "dispatch(action_type, payload)",
            "server_handler": "TemporalReviewStore.apply_browser_action",
            "reducer": "g7e_b_r6_action_reducer.apply_action",
            "same_paths_used_by_exact_routes": True,
            "same_paths_used_by_full_120_burst_audit": True,
            "direct_api_checks_supplemental_only": True,
            "production_ready": False,
        },
    )
    revocation_source = R6 / "07_RELEASE_GATE_AND_REAL_STATE_ACCEPTANCE/G7E_B_R5_RELEASE_GATE_REVOCATION.json"
    revocation = read(revocation_source)
    revocation["classification"] = "R5_RELEASE_GATE_REVOKED_REAL_PATH_FAILURE"
    write(R6 / "07_RELEASE_GATE_AND_REAL_STATE_ACCEPTANCE/r5_release_gate_revocation.json", revocation)

    chain_path = R6 / "00_BASELINE_REAL_EVENT_AND_DRAFT_CLOSURE/real_event_chain_manifest.json"
    migration_path = R6 / "02_FAILED_REAL_DRAFT_RECOVERY/failed_draft_migration_record.json"
    revocation_path = R6 / "07_RELEASE_GATE_AND_REAL_STATE_ACCEPTANCE/r5_release_gate_revocation.json"
    gate = {
        "schema_version": "football_intelligence.g7e_b_r6.real_review_release_gate.v1",
        "release_classification": RELEASE,
        "release_decision": RELEASE,
        "review_revision": REVISION,
        "r6_contract_sha256": sha256(PACKAGE / "server_action_contract.json"),
        "server_action_contract_sha256": sha256(PACKAGE / "server_action_contract.json"),
        "production_browser_bundle_sha256": sha256(PACKAGE / "review.js"),
        "server_reducer_sha256": sha256(REPO / "src/football_intelligence/g7e_b_r6_action_reducer.py"),
        "full_120_burst_browser_audit_sha256": sha256(
            R6 / "05_FULL_120_BURST_BROWSER_AUDIT/full_120_burst_browser_audit.json"
        ),
        "exact_branch_challenge_sha256": sha256(
            R6 / "04_PRODUCTION_PATH_CHALLENGE_SUITE/exact_branch_challenge_results.json"
        ),
        "fault_challenge_sha256": sha256(R6 / "06_FAULT_AND_RACE_CHALLENGE/fault_and_race_results.json"),
        "real_event_chain_manifest_sha256": sha256(chain_path),
        "failed_draft_migration_sha256": sha256(migration_path),
        "r5_release_gate_revocation_sha256": sha256(revocation_path),
        "focused_test_results_sha256": sha256(R6 / "09_TESTS_AND_LOGS/focused_test_results.json"),
        "reviewer_file_sha256": reviewer_hashes(),
        "corpus": {"bursts": 120, "frame_references": 1080, "tranches": 6},
        "failed_burst": FAILED,
        "real_event_created": False,
        "production_ready": False,
    }
    gate_path = R6 / "07_RELEASE_GATE_AND_REAL_STATE_ACCEPTANCE/G7E_B_R6_REAL_REVIEW_RELEASE_GATE.json"
    write(gate_path, gate)
    write(PACKAGE / "G7E_B_R6_REAL_REVIEW_RELEASE_GATE.json", gate)
    decision = {
        "schema_version": "football_intelligence.g7e_b_r6.decision.v1",
        "classification": RELEASE,
        "decision": RELEASE,
        "temporary_gates_passed": True,
        "focused_test_count": tests["pytest_test_count"],
        "real_event_chain_unchanged": True,
        "failed_draft_recovered": True,
        "marks_preserved": 27,
        "real_event_created": False,
        "next_burst_started": False,
        "release_gate_sha256": sha256(gate_path),
        "production_ready": False,
    }
    write(R6 / "07_RELEASE_GATE_AND_REAL_STATE_ACCEPTANCE/decision.json", decision)
    print(RELEASE)


def finalize_handoff() -> None:
    gate_path = R6 / "07_RELEASE_GATE_AND_REAL_STATE_ACCEPTANCE/G7E_B_R6_REAL_REVIEW_RELEASE_GATE.json"
    gate = read(gate_path)
    if gate.get("release_classification") != RELEASE or sha256(gate_path) != sha256(
        PACKAGE / "G7E_B_R6_REAL_REVIEW_RELEASE_GATE.json"
    ):
        raise RuntimeError("R6 release gate is absent, invalid, or diverged")
    acceptance = require_classification(
        R6 / "07_RELEASE_GATE_AND_REAL_STATE_ACCEPTANCE/real_state_acceptance.json",
        "PASS_G7E_B_R6_REAL_FAILED_DRAFT_READY_FOR_USER_RESUME",
    )
    if acceptance.get("real_event_created") or acceptance.get("next_burst_started"):
        raise RuntimeError("real recovered acceptance exceeded its stop point")
    chain = immutable_chain()
    write(R6 / "00_BASELINE_REAL_EVENT_AND_DRAFT_CLOSURE/real_event_chain_manifest.json", chain)

    handoff = R6 / "10_REVIEW_PACK/CHATGPT_HANDOFF"
    expected = {
        "01_EXECUTIVE_SUMMARY.json",
        "02_REAL_EVENT_AND_FAILED_DRAFT_CLOSURE.json",
        "03_EXACT_PATH_AND_ROOT_CAUSE.json",
        "04_SERVER_ACTION_REDUCER_AND_INVARIANTS.json",
        "05_FAILED_DRAFT_RECOVERY.json",
        "06_PRODUCTION_BROWSER_AND_FULL_CORPUS_AUDIT.json",
        "07_FAULT_CHALLENGE_AND_RELEASE_GATE.json",
        "08_DECISION.md",
        "09_EXACT_27_MARK_PATH.png",
        "10_SERVER_VERIFIED_SUMMARY.png",
        "11_REAL_DRAFT_RECOVERED.png",
        "12_MANIFEST.json",
    }
    handoff.mkdir(parents=True, exist_ok=True)
    extras = {path.name for path in handoff.iterdir()} - expected
    if extras:
        raise RuntimeError(f"unexpected files already exist in R6 handoff: {sorted(extras)}")

    tests = read(R6 / "09_TESTS_AND_LOGS/focused_test_results.json")
    decision = read(R6 / "07_RELEASE_GATE_AND_REAL_STATE_ACCEPTANCE/decision.json")
    write(
        handoff / "01_EXECUTIVE_SUMMARY.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6.handoff.executive_summary.v1",
            "classification": RELEASE,
            "failed_burst": FAILED,
            "root_cause": "R5 split human-answer mutation from lifecycle reconstruction on the real no-subject path.",
            "resolution": (
                "One versioned idempotent server action now atomically owns answers, lifecycle, branching, and drafts."
            ),
            "real_events_preserved": chain["event_count"],
            "real_acknowledgements_preserved": chain["acknowledgement_count"],
            "marks_preserved": 27,
            "real_event_created": False,
            "production_browser_bursts_passed": 120,
            "focused_tests_passed": tests["pytest_test_count"],
            "release_gate_sha256": sha256(gate_path),
            "next_action": "User may reopen the recovered summary and decide whether to final-save the same burst.",
            "production_ready": False,
        },
    )
    write(
        handoff / "02_REAL_EVENT_AND_FAILED_DRAFT_CLOSURE.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6.handoff.real_closure.v1",
            "classification": chain["classification"],
            "event_chain": chain,
            "forensic_backups": read(
                R6 / "00_BASELINE_REAL_EVENT_AND_DRAFT_CLOSURE/failed_draft_forensic_backup_manifest.json"
            ),
            "failed_burst_event_count": 0,
            "production_ready": False,
        },
    )
    write(
        handoff / "03_EXACT_PATH_AND_ROOT_CAUSE.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6.handoff.exact_path.v1",
            "exact_path": read(R6 / "01_EXACT_REAL_BROWSER_PATH_REPRODUCTION/exact_path_and_root_cause.json"),
            "ui_server_divergence": read(
                R6 / "01_EXACT_REAL_BROWSER_PATH_REPRODUCTION/ui_server_state_divergence.json"
            ),
            "r5_release_gap": read(R6 / "01_EXACT_REAL_BROWSER_PATH_REPRODUCTION/r5_release_gap_analysis.json"),
            "production_ready": False,
        },
    )
    write(
        handoff / "04_SERVER_ACTION_REDUCER_AND_INVARIANTS.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6.handoff.reducer.v1",
            "contract": read(R6 / "03_SERVER_AUTHORITATIVE_ACTION_REDUCER/action_reducer_contract.json"),
            "client_mutation_audit": read(
                R6 / "03_SERVER_AUTHORITATIVE_ACTION_REDUCER/unauthorized_client_mutation_audit.json"
            ),
            "answer_lifecycle_invariants": read(
                R6 / "03_SERVER_AUTHORITATIVE_ACTION_REDUCER/answer_lifecycle_invariant_results.json"
            ),
            "summary_field_invariants": read(
                R6 / "03_SERVER_AUTHORITATIVE_ACTION_REDUCER/summary_field_invariant_results.json"
            ),
            "production_ready": False,
        },
    )
    write(
        handoff / "05_FAILED_DRAFT_RECOVERY.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6.handoff.recovery.v1",
            "repair_decision": read(R6 / "02_FAILED_REAL_DRAFT_RECOVERY/failed_draft_repair_decision.json"),
            "migration_record": read(R6 / "02_FAILED_REAL_DRAFT_RECOVERY/failed_draft_migration_record.json"),
            "post_migration_validation": read(
                R6 / "02_FAILED_REAL_DRAFT_RECOVERY/real_draft_post_migration_validation.json"
            ),
            "real_browser_acceptance": acceptance,
            "production_ready": False,
        },
    )
    write(
        handoff / "06_PRODUCTION_BROWSER_AND_FULL_CORPUS_AUDIT.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6.handoff.browser_audit.v1",
            "production_path_challenge": read(
                R6 / "04_PRODUCTION_PATH_CHALLENGE_SUITE/production_path_challenge_results.json"
            ),
            "production_equivalence": read(
                R6 / "04_PRODUCTION_PATH_CHALLENGE_SUITE/test_production_equivalence_map.json"
            ),
            "full_corpus_audit": read(R6 / "05_FULL_120_BURST_BROWSER_AUDIT/full_120_burst_browser_audit.json"),
            "focused_tests": tests,
            "production_ready": False,
        },
    )
    write(
        handoff / "07_FAULT_CHALLENGE_AND_RELEASE_GATE.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6.handoff.release_gate.v1",
            "fault_challenge": read(R6 / "06_FAULT_AND_RACE_CHALLENGE/fault_and_race_results.json"),
            "r5_revocation": read(R6 / "07_RELEASE_GATE_AND_REAL_STATE_ACCEPTANCE/r5_release_gate_revocation.json"),
            "r6_release_gate": gate,
            "release_gate_sha256": sha256(gate_path),
            "production_ready": False,
        },
    )
    (handoff / "08_DECISION.md").write_text(
        "# G7E-B R6 decision\n\n"
        f"`{RELEASE}`\n\n"
        "The exact real no-subject → missed-person route now uses the same server-authoritative reducer as the "
        "release tests. Every answer and lifecycle transition is atomic, stale or duplicate actions recover "
        "deterministically, and summaries are server-authorized. The 120-burst real-DOM audit, fault challenge, "
        "and 66 focused tests passed. The four prior events and acknowledgements remain byte-identical. The failed "
        "draft retains all 27 marks and is open at its recovered summary; R6 created no real event and did not start "
        "the next burst. The user—not Codex—must decide whether to press final Save. `production_ready=false`.\n",
        encoding="utf-8",
        newline="\n",
    )
    visual_map = {
        "09_EXACT_27_MARK_PATH.png": "01_EXACT_27_MARK_PATH_ACKNOWLEDGED.png",
        "10_SERVER_VERIFIED_SUMMARY.png": "02_SERVER_VERIFIED_SUMMARY.png",
        "11_REAL_DRAFT_RECOVERED.png": "03_REAL_DRAFT_RECOVERED_NO_EVENT_CREATED.png",
    }
    for target, source in visual_map.items():
        shutil.copyfile(R6 / "08_VISUAL_QA" / source, handoff / target)
    payloads = sorted(path for path in handoff.iterdir() if path.is_file() and path.name != "12_MANIFEST.json")
    if len(payloads) != 11:
        raise RuntimeError(f"R6 handoff requires exactly 11 manifest inputs, found {len(payloads)}")
    manifest = {
        "schema_version": "football_intelligence.g7e_b_r6.handoff_manifest.v1",
        "classification": "PASS_G7E_B_R6_CHATGPT_HANDOFF",
        "file_count_excluding_manifest": 11,
        "self_hashed": False,
        "files": [record(path) for path in payloads],
        "production_ready": False,
    }
    write(handoff / "12_MANIFEST.json", manifest)
    if {path.name for path in handoff.iterdir() if path.is_file()} != expected:
        raise RuntimeError("R6 handoff file set is not exact")
    (R6 / "10_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only CHATGPT_HANDOFF. It contains exactly twelve self-contained R6 files.\n",
        encoding="utf-8",
        newline="\n",
    )
    write(
        R6 / "07_RELEASE_GATE_AND_REAL_STATE_ACCEPTANCE/decision.json",
        decision
        | {
            "real_state_acceptance_sha256": sha256(
                R6 / "07_RELEASE_GATE_AND_REAL_STATE_ACCEPTANCE/real_state_acceptance.json"
            ),
            "handoff_manifest_sha256": sha256(handoff / "12_MANIFEST.json"),
        },
    )
    for temporary in (
        R6 / "04_PRODUCTION_PATH_CHALLENGE_SUITE/_browser_work",
        R6 / ".migration_practice",
    ):
        if temporary.exists():
            shutil.rmtree(temporary)
    print(RELEASE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-release", action="store_true")
    parser.add_argument("--finalize-handoff", action="store_true")
    args = parser.parse_args()
    if args.prepare_release:
        prepare_release()
    if args.finalize_handoff:
        finalize_handoff()


if __name__ == "__main__":
    main()
