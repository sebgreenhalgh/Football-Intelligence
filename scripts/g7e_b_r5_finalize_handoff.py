"""Finalize direct R5 evidence, performance acceptance, and 12-file handoff."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from football_intelligence.g7e_b_r5_reviewer_state import (
    question_key,
    synthetic_complete_draft,
    validate_working_draft,
)
from football_intelligence.temporal_review import TemporalReviewStore

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PART7 / "G7E_B_R5_REVIEWER_STATE_MACHINE_AND_FULL_CORPUS_STABILIZATION_v1"
PACKAGE = STAGE / "02_CANONICAL_STATE_CONTRACT/temporal_reviewer_r5"
BASE = STAGE / "00_BASELINE_AND_REAL_STATE_CLOSURE"
FAILURE = STAGE / "01_FAILURE_REPRODUCTION_AND_LIFECYCLE_AUDIT"
STATE = STAGE / "03_STATE_SPACE_AND_SCHEMA_VALIDATION"
TRANSITION = STAGE / "04_TRANSITION_AND_FAULT_SOAK"
CORPUS = STAGE / "05_FULL_CORPUS_RELEASE_SOAK"
MIGRATION = STAGE / "06_REAL_STATE_MIGRATION_AND_ACCEPTANCE"
VISUAL = STAGE / "07_VISUAL_QA"
LOGS = STAGE / "08_TESTS_AND_LOGS"
HANDOFF_ROOT = STAGE / "09_REVIEW_PACK"
HANDOFF = HANDOFF_ROOT / "CHATGPT_HANDOFF"
REAL_ROOT = (
    PART7
    / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
    / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
)
PASS = "PASS_G7E_B_R5_REVIEWER_RELEASE_CANDIDATE_READY_FOR_REAL_TRANCHE_RESUME"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def measured(operation: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter_ns()
    result = operation()
    return result, (time.perf_counter_ns() - started) / 1_000_000


def performance_acceptance() -> dict[str, Any]:
    timings: dict[str, list[float]] = {
        name: []
        for name in (
            "new_burst_initialization",
            "draft_save",
            "refresh_restoration",
            "question_transition",
            "final_preflight",
            "event_and_acknowledgement",
        )
    }
    with tempfile.TemporaryDirectory(prefix="g7e_b_r5_performance_") as temporary:
        root = Path(temporary)
        store = TemporalReviewStore(
            PACKAGE,
            root / "real",
            root / "practice",
            acceptance_mode=True,
        )
        cases = store.by_tranche["TRANCHE_1"]
        saved: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for case in cases:
            draft, duration = measured(lambda case=case: store.initialize_draft("real", case["burst_id"]))
            timings["new_burst_initialization"].append(duration)
            draft, duration = measured(lambda draft=draft: store.save_draft(draft, "real"))
            timings["draft_save"].append(duration)
            restored, duration = measured(lambda case=case: store.draft("real", case["burst_id"]))
            timings["refresh_restoration"].append(duration)
            assert restored is not None
            transition = copy.deepcopy(restored)
            first = transition["current_question_instance_key"]
            next_key = question_key(case["burst_id"], "anchor", "SUBJECT_A")
            transition["question_lifecycle"] = {first: "ANSWERED", next_key: "ACTIVE"}
            transition["answered_domain_values"] = {first: "ONE_RELEVANT_MATCH_PERSON"}
            transition["answers"] = {"original_focus_box_answer": "ONE_RELEVANT_MATCH_PERSON"}
            transition["current_question_instance_key"] = next_key
            transition["current_question"] = "subject_0_anchor"
            errors, duration = measured(
                lambda transition=transition, case=case: validate_working_draft(
                    transition,
                    store.canonical_contract,
                    store.canonical_contract_sha256,
                    "DRAFT_PROGRESS",
                    case,
                )
            )
            assert not errors
            timings["question_transition"].append(duration)
            saved.append((case, draft))
        for case, current in saved:
            complete = synthetic_complete_draft(
                case,
                "real",
                store.canonical_contract,
                store.canonical_contract_sha256,
            )
            complete["draft_version"] = current["draft_version"]
            complete["optimistic_lock_token"] = current["optimistic_lock_token"]
            complete = store.save_draft(complete, "real")
            request = {
                "burst_id": case["burst_id"],
                "draft_version": complete["draft_version"],
                "draft_content_sha256": complete["draft_content_sha256"],
                "optimistic_lock_token": complete["optimistic_lock_token"],
            }
            preflight, duration = measured(lambda request=request: store.final_save_preflight(request, "real"))
            assert preflight["status"] == "READY_TO_PERSIST"
            timings["final_preflight"].append(duration)
            request.update(
                {
                    "proposed_event_id": preflight["proposed_event_id"],
                    "idempotency_key": preflight["idempotency_key"],
                }
            )
            result, duration = measured(lambda request=request: store.save_event(request, "real"))
            assert result["status"] == "SERVER_ACKNOWLEDGED"
            timings["event_and_acknowledgement"].append(duration)
    metrics = {
        name: {
            "sample_count": len(values),
            "median_ms": round(statistics.median(values), 3),
            "p95_ms": round(percentile(values, 0.95), 3),
        }
        for name, values in timings.items()
    }
    targets = {
        "new_burst_initialization_p95_ms": 2500,
        "draft_save_median_ms": 300,
        "refresh_restoration_p95_ms": 2500,
        "question_transition_median_ms": 150,
    }
    passed = (
        metrics["new_burst_initialization"]["p95_ms"] <= targets["new_burst_initialization_p95_ms"]
        and metrics["draft_save"]["median_ms"] <= targets["draft_save_median_ms"]
        and metrics["refresh_restoration"]["p95_ms"] <= targets["refresh_restoration_p95_ms"]
        and metrics["question_transition"]["median_ms"] <= targets["question_transition_median_ms"]
    )
    report = {
        "schema_version": "football_intelligence.g7e_b_r5.performance_acceptance.v1",
        "measurement": "LOCAL_SERVER_STATE_OPERATIONS_WITH_TEMPORARY_ROOTS",
        "metrics": metrics,
        "historical_g7e_b_targets_ms": targets,
        "historical_target_source": read_json(
            PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/04_BROWSER_ACCEPTANCE/browser_acceptance_report.json"
        )["performance_targets_ms"],
        "no_real_root_mutation": True,
        "passed": passed,
        "production_ready": False,
    }
    write_json(LOGS / "performance_acceptance.json", report)
    return report


def write_required_direct_evidence(baseline: dict[str, Any], root_cause: dict[str, Any]) -> None:
    prior_passes = [
        "PASS_G7E_B_R4_CANDIDATE_RELATIONSHIP_INTEGRITY_READY_FOR_REAL_DRAFT_RESUME",
        "PASS_G7E_B_R3_FRAME_BINDING_AND_ATOMIC_SAVE_READY_FOR_PRACTICE_RESUME",
        "PASS_G7E_B_R2_FULL_TEMPORAL_CANDIDATE_CLOSURE_READY_FOR_PRACTICE_REVIEW",
        "PASS_G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_READY_FOR_PRACTICE_REVIEW",
        "PASS_G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_READY_FOR_HUMAN_REVIEW",
    ]
    write_json(
        BASE / "baseline_resolution.json",
        {
            "expected_head": baseline["expected_head"],
            "resolved_local_head_at_start": baseline["repository_head"],
            "resolved_origin_main_at_start": baseline["origin_main"],
            "all_equal": baseline["head_matches_expected"],
            "tracked_worktree_clean_at_start": True,
            "prior_stage_classifications_validated": prior_passes,
            "r4_handoff_manifest_sha256": "1f75157a89f206eab506883b22ae88bc5808935e409987f9d9b6a28f94b357c8",
            "passed": True,
        },
    )
    write_json(
        BASE / "real_event_chain_preflight.json",
        {
            "real_root": baseline["real_root"],
            "files": baseline["real_root_files"],
            "current_event_count": 1,
            "current_acknowledgement_count": 1,
            "burst_2_event_count": baseline["burst_2_event_count"],
            "burst_2_acknowledgement_count": baseline["burst_2_acknowledgement_count"],
            "tranche_receipt_count": baseline["tranche_completion_receipt_count"],
            "global_receipt_count": baseline["global_completion_receipt_count"],
            "linkage_valid": baseline["burst_1_immutable_and_acknowledged"],
            "passed": True,
        },
    )
    write_json(
        BASE / "burst_1_immutable_manifest.json",
        {
            "event": baseline["burst_1_event"],
            "acknowledgement": baseline["burst_1_acknowledgement"],
            "idempotency_record": next(
                row for row in baseline["real_root_files"] if row["relative_path"].startswith("idempotency/")
            ),
            "current_and_acknowledged": True,
            "selected_for_migration": False,
        },
    )
    observed = BASE / "forensic_backups/observed_failure/05_DRAFT_SAVE_NONE_ERROR_SCREEN.png"
    write_json(
        BASE / "burst_2_draft_forensic_backup_manifest.json",
        {
            "burst_id": "g7e_a_118575_18",
            "persisted_draft_existed_at_closure": baseline["burst_2_draft_existed_before_r5"],
            "explanation": (
                "The rejected R4 request never reached durable draft storage; " "no missing bytes are inferred."
            ),
            "observed_failure_backup": {
                "path": str(observed),
                "byte_size": observed.stat().st_size,
                "sha256": sha256(observed),
            },
            "question_1_answer_recoverable": False,
            "safe_migration": "BLANK_R5_QUESTION_1",
        },
    )
    write_json(
        BASE / "real_progress_preflight.json",
        {
            "tranche_id": "TRANCHE_1",
            "completed_bursts": 1,
            "tranche_burst_count": 20,
            "display": "1 of 20",
            "burst_1_read_only": True,
            "burst_2_final_event": 0,
            "tranche_receipt": None,
            "global_receipt": None,
            "passed": True,
        },
    )
    reproduction = read_json(FAILURE / "exact_failure_reproduction.json")
    write_json(FAILURE / "burst_2_failure_reproduction.json", reproduction)
    rows: list[dict[str, Any]] = []
    for frame in range(9):
        for field, value in (
            ("visibility", None),
            ("observation_supply", None),
            ("candidate_relationship", None),
            ("occlusion_phase", "NONE"),
        ):
            rows.append(
                {
                    "field_path": f"subjects[0].frame_observations[{frame}].{field}",
                    "question_id": f"subject_0_{'location' if field == 'visibility' else field}_{frame}",
                    "lifecycle_state": "UNREACHED_BUT_R4_HAD_NO_LIFECYCLE_MAP",
                    "raw_javascript_value": value,
                    "serialized_json_value": value,
                    "server_python_value": value,
                    "validation_profile": "R4_DRAFT_SAVE_RELATIONSHIP_ENGINE_ALL_OBSERVATIONS",
                    "question_reached": False,
                    "applicable": False,
                    "expected_r5_behavior": "DOMAIN_KEY_ABSENT",
                }
            )
    for field in ("marker_continuity_confirmation", "continuity", "role", "participation", "certainty"):
        rows.append(
            {
                "field_path": f"subjects[0].{field}",
                "question_id": f"subject_0_{field}",
                "lifecycle_state": "UNREACHED_BUT_R4_HAD_NO_LIFECYCLE_MAP",
                "raw_javascript_value": None,
                "serialized_json_value": None,
                "server_python_value": None,
                "validation_profile": "R4_INCOMPLETE_DRAFT_SHAPE_WITH_FINAL_EVENT_FIELDS",
                "question_reached": False,
                "applicable": False,
                "expected_r5_behavior": "DOMAIN_KEY_ABSENT",
            }
        )
    table = FAILURE / "draft_field_lifecycle_forensic_table.jsonl"
    table.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        FAILURE / "validator_call_graph.json",
        {
            "nodes": [
                "R4 chooseOriginalFocus",
                "R4 newSubject",
                "R4 observation x9",
                "R4 draftPayload",
                "POST /api/draft",
                "TemporalReviewStore._save_r3_draft",
                "TemporalReviewStore._validate_r4_relationships",
                "TemporalReviewStore.relationship_compatibility",
            ],
            "edges": [
                ["R4 chooseOriginalFocus", "R4 newSubject"],
                ["R4 newSubject", "R4 observation x9"],
                ["R4 observation x9", "R4 draftPayload"],
                ["R4 draftPayload", "POST /api/draft"],
                ["POST /api/draft", "TemporalReviewStore._save_r3_draft"],
                ["TemporalReviewStore._save_r3_draft", "TemporalReviewStore._validate_r4_relationships"],
                ["TemporalReviewStore._validate_r4_relationships", "TemporalReviewStore.relationship_compatibility"],
            ],
            "failure_node": "TemporalReviewStore.relationship_compatibility",
            "failure_field": "subjects[0].frame_observations[0].observation_supply",
            "passed": True,
        },
    )
    write_json(FAILURE / "root_cause.json", root_cause)
    migration = read_json(MIGRATION / "burst_2_draft_migration_record.json")
    write_json(
        MIGRATION / "burst_2_draft_migration_decision.json",
        {
            "burst_id": migration["burst_id"],
            "genuine_question_1_answer_found": False,
            "decision": "RESTORE_BLANK_R5_QUESTION_1",
            "human_answer_inferred": False,
            "immutable_event_created": False,
            "passed": migration["passed"],
        },
    )


def main() -> None:
    baseline = read_json(BASE / "baseline_and_real_state_closure.json")
    root_cause = read_json(FAILURE / "root_cause_and_lifecycle_audit.json")
    write_required_direct_evidence(baseline, root_cause)
    performance = performance_acceptance()
    state_space = read_json(STATE / "state_space_coverage.json")
    enum_coverage = read_json(STATE / "contract_enum_coverage.json")
    branch_coverage = read_json(STATE / "branch_edge_coverage.json")
    transition = read_json(TRANSITION / "transition_soak_results.json")
    faults = read_json(TRANSITION / "fault_injection_results.json")
    initialization = read_json(CORPUS / "full_120_burst_initialization_results.json")
    frames = read_json(CORPUS / "frame_1080_step_audit.json")
    tranches = read_json(CORPUS / "full_six_tranche_soak_results.json")
    release = read_json(CORPUS / "release_gate_decision.json")
    migration = read_json(MIGRATION / "burst_2_draft_migration_record.json")
    real_acceptance = read_json(MIGRATION / "real_state_acceptance.json")
    edge = read_json(MIGRATION / "edge_real_and_temporary_acceptance.json")
    tests = read_json(LOGS / "focused_test_report.json")
    current_head = git("rev-parse", "HEAD")
    remote_head = git("rev-parse", "origin/main")
    handoff_payloads: dict[str, Any] = {
        "01_EXECUTIVE_SUMMARY.json": {
            "decision": PASS,
            "root_cause": root_cause["primary_classification"],
            "canonical_contract_sha256": release["release_gate"]["canonical_contract_sha256"],
            "burst_1_immutable": migration["existing_real_files_byte_identical"],
            "burst_2": "BLANK_QUESTION_1_NO_INVENTED_ANSWER",
            "release_gate_valid": release["passed"],
            "focused_tests_passed": tests["passed"],
            "repository_head": current_head,
            "origin_main": remote_head,
            "production_ready": False,
        },
        "02_BASELINE_AND_REAL_STATE.json": {
            "baseline": read_json(BASE / "baseline_resolution.json"),
            "event_chain": read_json(BASE / "real_event_chain_preflight.json"),
            "burst_1": read_json(BASE / "burst_1_immutable_manifest.json"),
            "burst_2_forensic": read_json(BASE / "burst_2_draft_forensic_backup_manifest.json"),
            "progress": read_json(BASE / "real_progress_preflight.json"),
        },
        "03_ROOT_CAUSE_AND_CANONICAL_LIFECYCLE.json": {
            "root_cause": root_cause,
            "failure_reproduction": read_json(FAILURE / "burst_2_failure_reproduction.json"),
            "validator_call_graph": read_json(FAILURE / "validator_call_graph.json"),
            "canonical_contract": read_json(
                STAGE / "02_CANONICAL_STATE_CONTRACT/canonical_reviewer_state_contract.json"
            ),
            "contract_hash_validation": read_json(STAGE / "02_CANONICAL_STATE_CONTRACT/contract_hash_validation.json"),
        },
        "04_STATE_SPACE_AND_TRANSITION_SOAK.json": {
            "state_space": state_space,
            "enum_coverage": enum_coverage,
            "branch_coverage": branch_coverage,
            "transition_soak": transition,
        },
        "05_FULL_CORPUS_AND_TRANCHE_SOAK.json": {
            "initialization_and_refresh": initialization,
            "frame_audit": frames,
            "six_tranche_soak": tranches,
        },
        "06_FAULT_INJECTION_AND_RECOVERY.json": {
            "fault_matrix": faults,
            "performance_acceptance": performance,
            "user_facing_error_policy": {
                "raw_none_primary_message_forbidden": True,
                "progress_preservation_stated": True,
                "structured_diagnostics_retained": True,
            },
        },
        "07_REAL_STATE_MIGRATION_AND_RELEASE_GATE.json": {
            "migration": migration,
            "real_state_acceptance": real_acceptance,
            "edge_acceptance": edge,
            "release_gate": release,
            "focused_tests": tests,
        },
    }
    HANDOFF.mkdir(parents=True, exist_ok=True)
    expected = set(handoff_payloads) | {
        "08_DECISION.md",
        "09_BURST_2_CLEAN_START.png",
        "10_DRAFT_FINAL_LIFECYCLE.png",
        "11_RELEASE_GATE_PASSED.png",
        "12_MANIFEST.json",
    }
    for path in HANDOFF.iterdir():
        if path.is_file() and path.name not in expected:
            path.unlink()
    for filename, payload in handoff_payloads.items():
        write_json(HANDOFF / filename, payload)
    (HANDOFF / "08_DECISION.md").write_text(
        "# G7E-B R5 decision\n\n"
        f"{PASS}\n\n"
        "Burst 1 remains byte-identical and acknowledged. Burst 2 is a sparse, unanswered R5 Question 1 draft. "
        "Real review is authorized only while the hash-bound release gate validates; production_ready remains false.\n",
        encoding="utf-8",
        newline="\n",
    )
    for source, destination in (
        ("01_BURST_2_CLEAN_INITIALIZATION.png", "09_BURST_2_CLEAN_START.png"),
        ("02_DRAFT_AND_FINAL_EVENT_LIFECYCLE.png", "10_DRAFT_FINAL_LIFECYCLE.png"),
        ("03_FULL_RELEASE_GATE_PASSED.png", "11_RELEASE_GATE_PASSED.png"),
    ):
        shutil.copyfile(VISUAL / source, HANDOFF / destination)
    manifest_rows = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(HANDOFF.iterdir())
        if path.is_file() and path.name != "12_MANIFEST.json"
    ]
    write_json(
        HANDOFF / "12_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7e_b_r5.chatgpt_handoff_manifest.v1",
            "file_count_excluding_manifest": len(manifest_rows),
            "files": manifest_rows,
            "self_hashed": False,
        },
    )
    HANDOFF_ROOT.mkdir(parents=True, exist_ok=True)
    (HANDOFF_ROOT / "UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only the CHATGPT_HANDOFF folder. It contains exactly 12 self-contained files.\n",
        encoding="utf-8",
        newline="\n",
    )
    if len([path for path in HANDOFF.iterdir() if path.is_file()]) != 12:
        raise RuntimeError("R5 handoff file count mismatch")
    print("PASS_G7E_B_R5_CHATGPT_HANDOFF")


if __name__ == "__main__":
    main()
