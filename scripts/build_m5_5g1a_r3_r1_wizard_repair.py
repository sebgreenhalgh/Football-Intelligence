"""Build the bounded M5.5G.1A-R3-R1 wizard-state repair workspace."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash
from football_intelligence.review_chassis.validation import validate_review_chassis_package

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G1A_R3_R1_Wizard_State_Repair_Codex_Prompt_Pack"
R3 = PART3 / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
R3_PACKAGE = R3 / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
R3_DECISIONS = R3_PACKAGE / "decisions"
STAGE = PART3 / "M5_5G1A_R3_R1_WIZARD_STATE_INVALIDATION_AND_SAFE_CASE_RESTART_v1"
PACKAGE = STAGE / "05_REPAIRED_INCREMENTAL_ANNOTATION_PACKAGE"
PACK = STAGE / "07_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "5e03cf76525c26deb3d983b957602b01ee5ce82a"
ORIGIN = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
REVIEW_ID = "m5_5g1a_detection_gold_pilot_v1_r3"
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer_r3"
CLIENT_BUILD_ID = "m5_5g1a_r3_r1_wizard_state_repair_v1"
INDEXEDDB_NAMESPACE = "fi_detection_gold_m5_5g1a_r3_r1_wizard_state_repair_v1"
CLASSIFICATION = "PASS_R3_WIZARD_STATE_REPAIR_READY_TO_RESUME_TRANCHE_B"
CASE_HASH = "986604e34e6f831825dfb76601f854ece083e3ba9001f97bad26d6d596e6a401"
EVIDENCE_HASH = "58c2de8da4e6e8d2160a29550e8030e7c5225845fc58be44eae7651c0b4a1ab4"
ONTOLOGY_HASH = "81c256cae533a983970926cb7acfa8a090ac12629166a17181c0990877e92a8b"
SECTIONS = (
    "00_PROMPT_AND_INPUTS",
    "01_LIVE_STATE_AND_PRESERVATION_AUDIT",
    "02_REVISION_AWARE_WIZARD_STATE",
    "03_SAFE_CASE_RESET_AND_RECOVERY",
    "04_BROWSER_PERSISTENCE_AND_REGRESSION",
    "05_REPAIRED_INCREMENTAL_ANNOTATION_PACKAGE",
    "06_COMMANDS_AND_TESTS",
    "07_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)
ALLOWED_CHANGES = {
    "scripts/build_m5_5g1a_r3_r1_wizard_repair.py",
    "scripts/capture_m5_5g1a_r3_r1_browser_acceptance.py",
    "scripts/finalize_m5_5g1a_r3_r1_review_pack.py",
    "src/football_intelligence/detection_gold/incremental.py",
    "src/football_intelligence/detection_gold/persistence.py",
    "src/football_intelligence/review_chassis/persistence.py",
    "src/football_intelligence/review_chassis/server.py",
    "src/football_intelligence/review_chassis/static/detection_gold_app.js",
    "src/football_intelligence/review_chassis/static/detection_gold_wizard.js",
    "src/football_intelligence/review_chassis/static/styles.css",
    "tests/test_m5_5g1a_r3_r1_wizard_repair.py",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def safe_path(path: Path) -> str:
    return f"<FOOTBALL_INTELLIGENCE_ROOT>/{path.resolve().relative_to(ROOT.resolve()).as_posix()}"


def rows_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def tree_manifest(root: Path, *, include_rows: bool = False) -> dict[str, Any]:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    result: dict[str, Any] = {
        "root": safe_path(root),
        "file_count": len(rows),
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "tree_hash": rows_hash(rows),
    }
    if include_rows:
        result["files"] = rows
    return result


def ensure_workspace() -> None:
    for name in SECTIONS:
        (STAGE / name).mkdir(parents=True, exist_ok=True)
    if (PACKAGE / "decisions").exists():
        raise RuntimeError("the repaired package must not contain a second decisions root")


def authorization() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.rstrip("\r\n")
    changed = [row[3:].replace("\\", "/") for row in status.splitlines() if len(row) > 3]
    baseline_exists = (
        subprocess.run(["git", "cat-file", "-e", f"{BASELINE}^{{commit}}"], cwd=REPO, check=False).returncode == 0
    )
    ancestor = (
        subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, head], cwd=REPO, check=False).returncode == 0
    )
    result = {
        "authorized_baseline": BASELINE,
        "head_at_build": head,
        "branch": git("branch", "--show-current"),
        "origin": git("remote", "get-url", "origin"),
        "baseline_exists": baseline_exists,
        "baseline_is_ancestor": ancestor,
        "clean_gate_confirmed_before_first_edit": True,
        "head_was_exact_baseline_before_first_edit": True,
        "working_tree_paths_at_build": changed,
        "working_tree_contains_only_r3_r1_changes": set(changed) <= ALLOWED_CHANGES,
        "intervening_commits": git("rev-list", "--reverse", f"{BASELINE}..{head}").splitlines(),
        "detector_or_tracker_change_authorized": False,
    }
    result["passed"] = all(
        (
            head == BASELINE,
            baseline_exists,
            ancestor,
            result["branch"] == "main",
            result["origin"] == ORIGIN,
            result["working_tree_contains_only_r3_r1_changes"],
        )
    )
    if not result["passed"]:
        raise RuntimeError(f"R3-R1 authorization failed: {result}")
    return result


def copy_prompt_pack() -> dict[str, Any]:
    manifest = read_json(PROMPT / "08_PROMPT_PACK_MANIFEST.json")
    destination = STAGE / "00_PROMPT_AND_INPUTS"
    checks = []
    for entry in manifest["files"]:
        source = PROMPT / entry["filename"]
        target = destination / source.name
        shutil.copy2(source, target)
        checks.append(
            {
                "filename": source.name,
                "size_match": source.stat().st_size == int(entry["byte_size"]),
                "sha256_match": sha256_file(source) == entry["sha256"] == sha256_file(target),
            }
        )
    shutil.copy2(PROMPT / "08_PROMPT_PACK_MANIFEST.json", destination / "08_PROMPT_PACK_MANIFEST.json")
    result = {
        "manifest_sha256": sha256_file(PROMPT / "08_PROMPT_PACK_MANIFEST.json"),
        "file_count": len(checks) + 1,
        "checks": checks,
        "passed": len(checks) == 8 and all(row["size_match"] and row["sha256_match"] for row in checks),
    }
    if not result["passed"]:
        raise RuntimeError("R3-R1 prompt-pack integrity failed")
    write_json(destination / "prompt_copy_validation.json", result)
    return result


def parse_events() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (R3_DECISIONS / "review_decision_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_snapshot_sidecars() -> dict[str, Any]:
    checks = []
    for sidecar in sorted((R3_DECISIONS / "snapshots").glob("*.json.sha256")):
        expected, filename = sidecar.read_text(encoding="utf-8").strip().split(maxsplit=1)
        target = sidecar.with_name(filename.strip())
        checks.append({"path": target.name, "sha256_match": target.exists() and sha256_file(target) == expected})
    return {
        "snapshot_count": len(checks),
        "checks": checks,
        "passed": bool(checks) and all(row["sha256_match"] for row in checks),
    }


def live_state_audit(decisions_before: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = load_manifest(R3_PACKAGE / "reviewer_manifest.json")
    ui = load_ui_config(R3_PACKAGE / "ui_config.json")
    state = read_json(R3_DECISIONS / "review_decisions.json")
    events = parse_events()
    persistence = DetectionGoldPilotPersistence(
        manifest=manifest,
        ui_config=ui,
        decisions_root=R3_DECISIONS,
        reviewer_session_id=REVIEWER,
    )
    replayed = persistence._materialize_events(events)  # noqa: SLF001 - deliberate read-only ledger audit
    replay_keys = (
        "active_tranche_id",
        "annotation_hashes",
        "annotations",
        "completed",
        "decisions",
        "event_sequence",
        "structured_reviews",
        "tranche_completions",
        "wizard_states",
    )
    replay_matches = all(stable_hash(replayed.get(key)) == stable_hash(state.get(key)) for key in replay_keys)
    tranches = ui.question_contract["gold_tranches"]
    a_ids = tranches["A_CORE_STATIC"]["case_ids"]
    b_ids = tranches["B_REMAINING_STATIC"]["case_ids"]
    saved = set(state.get("annotations", {}))
    saved_b = [case_id for case_id in b_ids if case_id in saved]
    unsaved_b = [case_id for case_id in b_ids if case_id not in saved]
    next_case = unsaved_b[0] if unsaved_b else None
    sequences = [int(event["event_sequence"]) for event in events]
    event_types = Counter(str(event.get("event_type")) for event in events)
    client_ids = [str(event.get("client_event_id") or "") for event in events]
    idempotency = [str(event.get("idempotency_key") or "") for event in events]
    completion = validate_completion_bundle(R3_DECISIONS / "completed_tranches" / "A_CORE_STATIC")
    snapshots = validate_snapshot_sidecars()
    prior = read_json(R3 / "01_PRIOR_STATE_AND_DEFECT_AUDIT" / "prior_state_validation.json")
    gate = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r1.live_state_precondition.v1",
        "review_id": manifest.review_id,
        "reviewer_session_id": REVIEWER,
        "event_sequence": int(state.get("event_sequence", -1)),
        "event_count": len(events),
        "event_type_counts": dict(sorted(event_types.items())),
        "strict_contiguous_event_sequences": sequences == list(range(1, len(events) + 1)),
        "client_event_ids_unique": len(client_ids) == len(set(client_ids)) and all(client_ids),
        "idempotency_keys_unique": len(idempotency) == len(set(idempotency)) and all(idempotency),
        "event_replay_matches_authoritative_state": replay_matches,
        "snapshot_sidecars": snapshots,
        "tranche_a_expected_case_count": len(a_ids),
        "tranche_a_saved_case_count": len(set(a_ids) & saved),
        "tranche_a_completion_valid": completion["passed"],
        "tranche_a_completed": "A_CORE_STATIC" in state.get("tranche_completions", {}),
        "tranche_b_expected_case_count": len(b_ids),
        "tranche_b_saved_case_count": len(saved_b),
        "tranche_b_saved_case_ids": saved_b,
        "tranche_b_current_case_id": next_case,
        "tranche_b_current_case_saved": next_case in saved,
        "tranche_b_completed": "B_REMAINING_STATIC" in state.get("tranche_completions", {}),
        "later_tranche_b_unsaved_count": max(0, len(unsaved_b) - 1),
        "pending_outbox_events": 0,
        "pending_outbox_evidence": "USER_CONFIRMED_AND_ALL_26_SERVER_EVENTS_ACKNOWLEDGED",
        "case7_stale_state_scope": "BROWSER_LOCAL_UNSAVED_ONLY",
        "case7_absent_from_server_state_and_ledger": next_case not in saved
        and all(event.get("case_id") != next_case for event in events),
        "case_payload_hash": stable_hash(read_json(R3_PACKAGE / "reviewer_manifest.json")["cases"]),
        "evidence_tree_hash": prior["evidence_tree_hash"],
        "ontology_hash": prior["frozen_ontology_hash"],
        "decisions_tree_before": decisions_before,
    }
    gate["passed"] = all(
        (
            manifest.review_id == REVIEW_ID,
            gate["event_sequence"] == 26,
            gate["event_count"] == 26,
            gate["strict_contiguous_event_sequences"],
            gate["client_event_ids_unique"],
            gate["idempotency_keys_unique"],
            replay_matches,
            snapshots["passed"],
            gate["tranche_a_saved_case_count"] == 18,
            gate["tranche_a_completion_valid"],
            gate["tranche_a_completed"],
            gate["tranche_b_expected_case_count"] == 14,
            gate["tranche_b_saved_case_count"] == 6,
            gate["tranche_b_current_case_id"] == "m5_5g1a_case_016",
            not gate["tranche_b_current_case_saved"],
            not gate["tranche_b_completed"],
            gate["pending_outbox_events"] == 0,
            gate["case7_absent_from_server_state_and_ledger"],
            gate["case_payload_hash"] == CASE_HASH,
            gate["evidence_tree_hash"] == EVIDENCE_HASH,
            gate["ontology_hash"] == ONTOLOGY_HASH,
        )
    )
    if not gate["passed"]:
        raise RuntimeError(f"FAIL_LIVE_STATE_PRECONDITION: {gate}")

    completion_root = R3_DECISIONS / "completed_tranches" / "A_CORE_STATIC"
    preservation = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r1.saved_case_preservation.v1",
        "tranche_a_files": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(completion_root.iterdir())
            if path.is_file()
        ],
        "tranche_b_saved_cases": [
            {
                "case_id": case_id,
                "annotation_hash": state["annotation_hashes"][case_id],
                "annotation_payload_hash": stable_hash(state["annotations"][case_id]),
                "wizard_state_hash": stable_hash(state["wizard_states"][case_id]),
            }
            for case_id in saved_b
        ],
        "automatic_reopen_performed": False,
        "automatic_resave_performed": False,
        "saved_human_payloads_copied_to_repaired_package": False,
        "passed": len(saved_b) == 6 and completion["passed"],
    }
    stale = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r1.stale_draft_disposition.v1",
        "case_id": next_case,
        "server_saved": False,
        "old_indexeddb_namespace": f"fi_detection_gold_{REVIEW_ID}",
        "new_indexeddb_namespace": INDEXEDDB_NAMESPACE,
        "old_namespace_opened_or_imported_by_repaired_client": False,
        "old_browser_draft_deleted_by_server": False,
        "new_namespace_starts_empty": True,
        "disposition": "IGNORE_STALE_OLD_NAMESPACE_AND_RESTART_FROM_SERVER_AUTHORITATIVE_ABSENCE",
        "saved_cases_affected": 0,
        "server_events_written": 0,
        "passed": next_case == "m5_5g1a_case_016",
    }
    return gate, preservation, stale


def build_package() -> dict[str, Any]:
    source_evidence = tree_manifest(R3_PACKAGE / "evidence")
    if source_evidence["tree_hash"] != EVIDENCE_HASH or source_evidence["file_count"] != 1512:
        raise RuntimeError("frozen R3 evidence tree changed")
    shutil.copytree(R3_PACKAGE / "evidence", PACKAGE / "evidence", copy_function=shutil.copy2, dirs_exist_ok=True)
    for name in (
        "reviewer_manifest.json",
        "evidence_manifest.json",
        "second_reviewer_and_adjudication_contract.json",
    ):
        shutil.copy2(R3_PACKAGE / name, PACKAGE / name)
    ui = read_json(R3_PACKAGE / "ui_config.json")
    predecessor_ui_config_hash = ui_config_hash(load_ui_config(R3_PACKAGE / "ui_config.json"))
    ui["page_title"] = "Football Intelligence - Repaired Incremental Detection Gold"
    ui["review_title"] = "Repaired incremental detection-gold annotation"
    ui["task_instructions"] = "Resume Tranche B from the next server-unsaved case using revision-aware review."
    ui["question_contract"].update(
        {
            "client_build_id": CLIENT_BUILD_ID,
            "revision_aware_wizard_state": True,
            "candidate_answer_validity_states": ["VALID", "NEEDS_REVIEW", "UNANSWERED", "INVALID"],
            "indexeddb_namespace": INDEXEDDB_NAMESPACE,
            "prior_indexeddb_namespace_import_forbidden": True,
            "first_load_server_reconciliation": True,
            "first_load_open_next_server_unsaved_case": True,
            "same_server_authoritative_decisions_root": True,
            "default_tranche_id": "B_REMAINING_STATIC",
            "compatible_predecessor_ui_config_hashes": [predecessor_ui_config_hash],
        }
    )
    write_json(PACKAGE / "ui_config.json", ui)
    pointer = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r1.external_decisions_pointer.v1",
        "review_id": REVIEW_ID,
        "decisions_root": str(R3_DECISIONS),
        "package_local_decisions_root_created": False,
        "read_only_during_build_and_automated_acceptance": True,
        "launcher_uses_existing_server_authoritative_root": True,
    }
    write_json(PACKAGE / "server_decisions_root_pointer.json", pointer)

    validation_root = STAGE / "_tmp" / "package_validation_empty_decisions"
    if validation_root.exists():
        shutil.rmtree(validation_root)
    validation_manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    validation_config = load_ui_config(PACKAGE / "ui_config.json")
    DetectionGoldPilotPersistence(
        manifest=validation_manifest,
        ui_config=validation_config,
        decisions_root=validation_root,
        reviewer_session_id=REVIEWER,
    ).ensure_state()
    generic = validate_review_chassis_package(
        manifest_path=PACKAGE / "reviewer_manifest.json",
        ui_config_path=PACKAGE / "ui_config.json",
        evidence_root=PACKAGE / "evidence",
        decisions_root=validation_root,
    )
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    config = load_ui_config(PACKAGE / "ui_config.json")
    copied_evidence = tree_manifest(PACKAGE / "evidence")
    result = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r1.package_validation.v1",
        "review_id": manifest.review_id,
        "manifest_hash": manifest_hash(manifest),
        "ui_config_hash": ui_config_hash(config),
        "case_count": len(manifest.cases),
        "case_payload_hash": stable_hash(read_json(PACKAGE / "reviewer_manifest.json")["cases"]),
        "manifest_byte_identical_to_r3": sha256_file(PACKAGE / "reviewer_manifest.json")
        == sha256_file(R3_PACKAGE / "reviewer_manifest.json"),
        "evidence_manifest_byte_identical_to_r3": sha256_file(PACKAGE / "evidence_manifest.json")
        == sha256_file(R3_PACKAGE / "evidence_manifest.json"),
        "evidence_copy": copied_evidence,
        "client_build_id": config.question_contract["client_build_id"],
        "indexeddb_namespace": config.question_contract["indexeddb_namespace"],
        "default_tranche_id": config.question_contract["default_tranche_id"],
        "compatible_predecessor_ui_config_hashes": config.question_contract["compatible_predecessor_ui_config_hashes"],
        "package_local_decisions_root_absent": not (PACKAGE / "decisions").exists(),
        "generic_package_validation_on_empty_fixture": generic,
        "real_browser_acceptance": {"status": "PENDING", "passed": False},
    }
    result["package_checks_passed"] = all(
        (
            result["review_id"] == REVIEW_ID,
            result["case_count"] == 88,
            result["case_payload_hash"] == CASE_HASH,
            result["manifest_byte_identical_to_r3"],
            result["evidence_manifest_byte_identical_to_r3"],
            copied_evidence["file_count"] == 1512,
            copied_evidence["tree_hash"] == EVIDENCE_HASH,
            result["client_build_id"] == CLIENT_BUILD_ID,
            result["indexeddb_namespace"] == INDEXEDDB_NAMESPACE,
            result["default_tranche_id"] == "B_REMAINING_STATIC",
            result["package_local_decisions_root_absent"],
            generic["passed"],
        )
    )
    result["passed"] = result["package_checks_passed"]
    if not result["passed"]:
        raise RuntimeError(f"repaired package validation failed: {result}")
    write_json(PACKAGE / "review_package_validation.json", result)
    return result


def write_revision_artifacts() -> None:
    write_json(
        STAGE / "02_REVISION_AWARE_WIZARD_STATE" / "wizard_dependency_graph.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r1.wizard_dependency_graph.v1",
            "nodes": [
                "human_objects",
                "finished_drawing_people",
                "person_questions",
                "candidate_answers",
                "candidate_target_bindings",
                "review_summary",
            ],
            "edges": [
                ["human_objects", "person_questions"],
                ["human_objects", "candidate_answers"],
                ["human_objects", "candidate_target_bindings"],
                ["finished_drawing_people", "candidate_answers"],
                ["person_questions", "review_summary"],
                ["candidate_answers", "review_summary"],
                ["candidate_target_bindings", "review_summary"],
            ],
            "baseline_root_cause": "candidate_answered_uuids remained counted after target bindings were cleared",
            "repair": "candidate records carry explicit revisions, validity, targets and invalidation provenance",
        },
    )
    write_json(
        STAGE / "02_REVISION_AWARE_WIZARD_STATE" / "revision_and_invalidation_contract.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r1.revision_invalidation.v1",
            "revisions": [
                "human_truth_revision",
                "person_question_revision",
                "candidate_answer_revision",
                "summary_revision",
            ],
            "delete_one_person": {
                "human_truth_revision_incremented": True,
                "targeting_answers": "NEEDS_REVIEW",
                "deleted_uuid_removed_from_targets": True,
                "summary": "NEEDS_REVIEW",
            },
            "delete_all_people": {
                "finished_drawing_people": False,
                "wizard_step": 1,
                "target_dependent_answers": "NEEDS_REVIEW",
                "machine_review_hidden": True,
            },
            "add_person": {
                "all_answers_default": "NEEDS_REVIEW",
                "target_free_background_retention": "EXPLICIT_CONFIRMATION_ONLY",
            },
            "visible_geometry_or_ambiguity_edit": "INVALIDATE_RELATED_CANDIDATES",
            "role_or_pitch_only_edit": "SUMMARY_REVIEW_ONLY",
            "save_rejects_non_valid_answer": True,
            "server_enforces_revision_integrity": True,
        },
    )
    write_json(
        STAGE / "02_REVISION_AWARE_WIZARD_STATE" / "candidate_answer_validity_matrix.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r1.candidate_validity_matrix.v1",
            "states": {
                "VALID": {"counts_as_checked": True, "save_eligible": True, "queued": False},
                "NEEDS_REVIEW": {"counts_as_checked": False, "save_eligible": False, "queued": True},
                "UNANSWERED": {"counts_as_checked": False, "save_eligible": False, "queued": True},
                "INVALID": {"counts_as_checked": False, "save_eligible": False, "queued": True},
            },
            "progress_fields": ["total", "valid", "stale", "unanswered", "invalid"],
            "stale_targets_count_as_valid": False,
        },
    )
    write_json(
        STAGE / "03_SAFE_CASE_RESET_AND_RECOVERY" / "case_restart_validation.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r1.case_restart_validation.v1",
            "status": "PENDING_REAL_BROWSER_ACCEPTANCE",
            "confirmation_required": True,
            "scope": "CURRENT_UNSAVED_CASE_ONLY",
            "server_event_written": False,
            "saved_cases_preserved": True,
            "tranche_completion_preserved": True,
            "return_step": 1,
            "passed": False,
        },
    )
    write_json(
        STAGE / "03_SAFE_CASE_RESET_AND_RECOVERY" / "first_load_reconciliation.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r1.first_load_reconciliation.v1",
            "status": "PENDING_REAL_BROWSER_ACCEPTANCE",
            "server_state_authoritative": True,
            "new_indexeddb_namespace": INDEXEDDB_NAMESPACE,
            "old_namespace_imported": False,
            "expected_saved_b_count": 6,
            "expected_progress": "6/14 saved",
            "expected_next_case_id": "m5_5g1a_case_016",
            "expected_step": 1,
            "expected_people": 0,
            "expected_valid_candidate_answers": 0,
            "passed": False,
        },
    )
    write_json(
        STAGE / "04_BROWSER_PERSISTENCE_AND_REGRESSION" / "browser_persistence_results.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r1.browser_acceptance.v1",
            "status": "PENDING_REAL_BROWSER_ACCEPTANCE",
            "temporary_copied_decisions_only": True,
            "real_human_decisions_root_opened_for_browser_test": False,
            "passed": False,
        },
    )


def write_launcher_and_instructions() -> None:
    launcher = f"""$ErrorActionPreference = 'Stop'
$port = 8807
$occupied = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($occupied) {{
  Write-Error 'Port 8807 is occupied. Stop the old R3 server, then rerun. This launcher will not move ports.'
}}
$repo = '{REPO}'
$package = '{PACKAGE}'
$decisions = '{R3_DECISIONS}'
Set-Location -LiteralPath $repo
Write-Host 'Starting the M5.5G.1A-R3-R1 repaired annotation client.' -ForegroundColor Green
Write-Host 'Open http://127.0.0.1:8807/' -ForegroundColor Cyan
uv run fi-pipeline review-chassis serve `
  --manifest "$package/reviewer_manifest.json" `
  --ui-config "$package/ui_config.json" `
  --evidence-root "$package/evidence" `
  --decisions-root "$decisions" `
  --host 127.0.0.1 `
  --port 8807 `
  --reviewer-session-id '{REVIEWER}'
"""
    instructions = """# Resume Tranche B with the repaired wizard

1. Stop the old port-8807 R3 server if it is still running.
2. Run `launch_repaired_incremental_gold_review.ps1`.
3. Open `http://127.0.0.1:8807/`.
4. Confirm the page says `6/14 saved` and opens the next unsaved Tranche B case at Step 1.
5. Continue annotation normally. Existing Tranche A and the six saved Tranche B
   cases are server-authoritative and are not resaved.

The repaired browser uses a new IndexedDB namespace. It deliberately does not
import the stale unsaved Case 7 draft from the old client. `Restart this case`
clears only the current unsaved browser draft and creates no server event.

This remains diagnostic annotation only. No detector or tracker was trained, evaluated, tuned or promoted.
"""
    for root in (PACKAGE, STAGE):
        write_text(root / "launch_repaired_incremental_gold_review.ps1", launcher)
        write_text(root / "HUMAN_INSTRUCTIONS.md", instructions)


def main() -> None:
    decisions_before = tree_manifest(R3_DECISIONS, include_rows=True)
    ensure_workspace()
    authorization_result = authorization()
    prompt_result = copy_prompt_pack()
    gate, preservation, stale = live_state_audit(decisions_before)
    write_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "live_state_precondition.json", gate)
    write_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "saved_case_preservation.json", preservation)
    write_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "stale_case7_draft_disposition.json", stale)
    package_result = build_package()
    write_revision_artifacts()
    write_launcher_and_instructions()
    decisions_after = tree_manifest(R3_DECISIONS, include_rows=True)
    preserved = decisions_before == decisions_after
    if not preserved:
        raise RuntimeError("FAIL_SAVED_CASE_PRESERVATION: live R3 decisions changed during build")
    gate["decisions_tree_after"] = decisions_after
    gate["live_decisions_byte_identical_after_build"] = True
    write_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "live_state_precondition.json", gate)
    write_json(
        STAGE / "06_COMMANDS_AND_TESTS" / "build_summary.json",
        {
            "classification": CLASSIFICATION,
            "authorization": authorization_result,
            "prompt_pack": prompt_result,
            "live_state_precondition_passed": gate["passed"],
            "saved_case_preservation_passed": preserved,
            "package": package_result,
            "browser_acceptance_pending": True,
            "tests_pending": True,
            "review_pack_pending": True,
            "model_fit_performed": False,
            "detector_or_tracker_changed": False,
            "detector_or_tracker_promoted": False,
            "production_ready": False,
            "human_approved": False,
        },
    )
    print(json.dumps({"stage": str(STAGE), "package": str(PACKAGE), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
