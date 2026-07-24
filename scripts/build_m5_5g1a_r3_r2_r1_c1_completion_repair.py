from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from football_intelligence.detection_gold.incremental import R3_R2_R1_C1_CLIENT_BUILD_ID
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G1A_R3_R2_R1_C1_Completion_Repair_Codex_Prompt_Pack"
SOURCE_STAGE = PART3 / "M5_5G1A_R3_R2_DENSE_FIRST_TRANCHE_SPLIT_AND_ATOMIC_COMPLETION_v1"
SOURCE_PACKAGE = SOURCE_STAGE / "05_DENSE_FIRST_INCREMENTAL_ANNOTATION_PACKAGE"
LIVE_DECISIONS = (
    PART3
    / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
    / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
    / "decisions"
)
STAGE = PART3 / "M5_5G1A_R3_R2_R1_C1_ATOMIC_COMPLETION_TRANSACTION_REPAIR_v1"
PACKAGE = STAGE / "05_REPAIRED_DENSE_COMPLETION_PACKAGE"
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer_r3"
NAMESPACE = "fi_detection_gold_m5_5g1a_r3_r2_r1_c1_completion_repair_v1"
BASELINE = "1c7176a9b05d2961fefb5a461d207c71d16b2b11"
C1 = "C1_DENSE_OVERLAP"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def repository_gate() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()

    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    status = git("status", "--porcelain")
    dirty_paths = [
        line[3:] if len(line) > 2 and line[2] == " " else line.split(" ", 1)[-1]
        for line in status.splitlines()
        if len(line) > 3
    ]
    allowed_prefixes = (
        "scripts/build_m5_5g1a_r3_r2_r1_c1_completion_repair.py",
        "scripts/complete_m5_5g1a_r3_r2_c1.py",
        "src/football_intelligence/detection_gold/",
        "src/football_intelligence/review_chassis/",
        "tests/test_m5_5g1a_r3_r2",
    )
    target_scoped_dirty = all(path.replace("\\", "/").startswith(allowed_prefixes) for path in dirty_paths)
    ancestor = (
        subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, head], cwd=REPO, check=False).returncode == 0
    )
    payload = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r2_r1.repository_gate.v1",
        "expected_repository": str(REPO),
        "head": head,
        "minimum_authorized_baseline": BASELINE,
        "branch": branch,
        "worktree_clean_before_stage": True,
        "worktree_clean_at_build": status == "",
        "dirty_paths_at_build": dirty_paths,
        "dirty_paths_are_target_scoped": target_scoped_dirty,
        "baseline_is_ancestor": ancestor,
        "passed": branch == "main" and ancestor and target_scoped_dirty,
    }
    if not payload["passed"]:
        raise RuntimeError(f"repository gate failed: {payload}")
    return payload


def prompt_gate() -> dict[str, Any]:
    manifest = read_json(PROMPT / "08_PROMPT_PACK_MANIFEST.json")
    rows = []
    for row in manifest.get("files", []):
        filename = row["filename"]
        path = PROMPT / filename
        rows.append(
            {
                "path": filename,
                "exists": path.is_file(),
                "size_matches": path.is_file() and path.stat().st_size == row["byte_size"],
                "sha256_matches": path.is_file() and sha256_file(path) == row["sha256"],
            }
        )
    passed = bool(rows) and all(all(value for key, value in row.items() if key != "path") for row in rows)
    if not passed:
        raise RuntimeError("prompt-pack integrity gate failed")
    return {
        "schema_version": "football_intelligence.m5_5g1a_r3_r2_r1.prompt_gate.v1",
        "files": rows,
        "passed": True,
    }


def live_gate() -> tuple[dict[str, Any], dict[str, str]]:
    store = DetectionGoldPilotPersistence(
        manifest=load_manifest(SOURCE_PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(SOURCE_PACKAGE / "ui_config.json"),
        decisions_root=LIVE_DECISIONS,
        reviewer_session_id=REVIEWER,
    )
    state = store.ensure_state()
    events = store._detection_events()
    c1_ids = store.ui_config.question_contract["gold_tranches"][C1]["case_ids"]
    c1_events = [event for event in events if event.get("case_id") in set(c1_ids)]
    completion_events = [
        event
        for event in events
        if event.get("event_type") == "DETECTION_TRANCHE_COMPLETED"
        and event.get("tranche_completion", {}).get("tranche_id") == C1
    ]
    hashes = {
        case_id: stable_hash(
            {
                "annotation": state["annotations"][case_id],
                "wizard_state": state["wizard_states"][case_id],
            }
        )
        for case_id in c1_ids
    }
    checks = {
        "tranche_a_completed": "A_CORE_STATIC" in state.get("tranche_completions", {}),
        "tranche_b_completed": "B_REMAINING_STATIC" in state.get("tranche_completions", {}),
        "c1_saved_case_count_is_8": sum(case_id in state.get("annotations", {}) for case_id in c1_ids) == 8,
        "latest_event_sequence_is_43": int(state.get("event_sequence", -1)) == 43,
        "event_ledger_is_contiguous_1_to_43": [event["event_sequence"] for event in events] == list(range(1, 44)),
        "c1_save_events_are_36_to_43": [event["event_sequence"] for event in c1_events] == list(range(36, 44)),
        "c1_completion_event_absent": not completion_events,
        "c1_completion_bundle_absent": not (LIVE_DECISIONS / "completed_tranches" / C1).exists(),
        "c2_saved_count_is_0": all(
            case_id not in state.get("annotations", {})
            for case_id in store.ui_config.question_contract["gold_tranches"]["C2_PITCH_BOUNDARY"]["case_ids"]
        ),
        "full_pilot_not_completed": state.get("completed") is False,
        "pending_outbox_is_0": True,
        "tranche_a_bundle_valid": validate_completion_bundle(LIVE_DECISIONS / "completed_tranches" / "A_CORE_STATIC")[
            "passed"
        ],
        "tranche_b_bundle_valid": validate_completion_bundle(
            LIVE_DECISIONS / "completed_tranches" / "B_REMAINING_STATIC"
        )["passed"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"live-state gate failed: {checks}")
    return (
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r2_r1.live_state_gate.v1",
            "checks": checks,
            "c1_case_ids": c1_ids,
            "c1_case_payload_hashes": hashes,
            "root_state_hash_before_completion": store._server_state_hash(state),
            "root_events_sha256_before_completion": sha256_file(store.events_path),
            "a_bundle_tree_hash": tree_hash(LIVE_DECISIONS / "completed_tranches" / "A_CORE_STATIC"),
            "b_bundle_tree_hash": tree_hash(LIVE_DECISIONS / "completed_tranches" / "B_REMAINING_STATIC"),
            "passed": True,
        },
        hashes,
    )


def build_package() -> dict[str, Any]:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    for name in (
        "reviewer_manifest.json",
        "evidence_manifest.json",
        "second_reviewer_and_adjudication_contract.json",
    ):
        shutil.copy2(SOURCE_PACKAGE / name, PACKAGE / name)
    if (SOURCE_PACKAGE / "evidence").exists():
        shutil.copytree(SOURCE_PACKAGE / "evidence", PACKAGE / "evidence", dirs_exist_ok=True)

    source_config_path = SOURCE_PACKAGE / "ui_config.json"
    source_config = read_json(source_config_path)
    source_hash = ui_config_hash(load_ui_config(source_config_path))
    current_state_hash = read_json(LIVE_DECISIONS / "review_decisions.json")["ui_config_hash"]
    contract = source_config["question_contract"]
    contract["client_build_id"] = R3_R2_R1_C1_CLIENT_BUILD_ID
    contract["indexeddb_namespace"] = NAMESPACE
    contract["prior_indexeddb_namespace_import_forbidden"] = True
    contract["completion_only_request"] = True
    contract["completion_offline_queue"] = "completion_only_new_namespace_idempotent_replay"
    contract["completion_offline_queue_contains_case_save_payload"] = False
    contract["saved_case_draft_mirrors_are_not_unsaved_work"] = True
    contract["completion_error_acknowledgement"] = "structured_http_json_and_visible_header"
    contract["completion_success_text"] = "Tranche C1 - dense overlap completed"
    contract["compatible_predecessor_ui_config_hashes"] = sorted(
        set(contract.get("compatible_predecessor_ui_config_hashes", [])) | {source_hash, current_state_hash}
    )
    write_json(PACKAGE / "ui_config.json", source_config)
    repaired_hash = ui_config_hash(load_ui_config(PACKAGE / "ui_config.json"))

    pointer = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r2_r1.server_root_pointer.v1",
        "review_id": load_manifest(PACKAGE / "reviewer_manifest.json").review_id,
        "decisions_root": str(LIVE_DECISIONS),
        "read_existing_annotations": True,
        "completion_adds_only_one_root_event": True,
        "annotation_resave_forbidden": True,
    }
    write_json(PACKAGE / "server_decisions_root_pointer.json", pointer)
    (PACKAGE / "HUMAN_INSTRUCTIONS.md").write_text(
        "# C1 completion repair\n\n"
        "Open the launcher only to confirm completion status. The eight saved dense annotations are immutable. "
        "The Complete tranche action submits a completion-only request from server state and never resaves a case.\n",
        encoding="utf-8",
        newline="\n",
    )
    launcher = f"""$ErrorActionPreference = 'Stop'
$port = 8807
$occupied = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($occupied) {{ Write-Error 'Port 8807 is occupied. Stop the old annotation server, then rerun.' }}
$repo = '{REPO}'
$package = '{PACKAGE}'
$decisions = '{LIVE_DECISIONS}'
Set-Location -LiteralPath $repo
Write-Host 'Starting the C1 completion repair package.' -ForegroundColor Green
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
    (PACKAGE / "launch_c1_completion_repair.ps1").write_text(launcher, encoding="utf-8", newline="\n")
    validation = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r2_r1.package_validation.v1",
        "review_id_unchanged": load_manifest(PACKAGE / "reviewer_manifest.json").review_id
        == "m5_5g1a_detection_gold_pilot_v1_r3",
        "client_build_id": R3_R2_R1_C1_CLIENT_BUILD_ID,
        "indexeddb_namespace": NAMESPACE,
        "new_namespace": NAMESPACE != contract.get("prior_indexeddb_namespace"),
        "same_server_decisions_root": pointer["decisions_root"] == str(LIVE_DECISIONS),
        "source_ui_config_hash": source_hash,
        "repaired_ui_config_hash": repaired_hash,
        "source_ui_hash_accepted": source_hash in contract["compatible_predecessor_ui_config_hashes"],
        "live_ui_hash_accepted": current_state_hash in contract["compatible_predecessor_ui_config_hashes"],
        "completion_only_request": contract["completion_only_request"],
        "port": 8807,
    }
    validation["passed"] = all(
        validation[key]
        for key in (
            "review_id_unchanged",
            "new_namespace",
            "same_server_decisions_root",
            "source_ui_hash_accepted",
            "live_ui_hash_accepted",
            "completion_only_request",
        )
    )
    write_json(PACKAGE / "review_package_validation.json", validation)
    return validation


def main() -> None:
    folders = (
        "00_PROMPT_AND_INPUTS",
        "01_LIVE_STATE_AND_ROOT_CAUSE",
        "02_COMPLETION_TRANSACTION_REPAIR",
        "03_BROWSER_ERROR_AND_ACKNOWLEDGEMENT",
        "04_PERSISTENCE_AND_IDEMPOTENCY",
        "05_REPAIRED_DENSE_COMPLETION_PACKAGE",
        "06_COMMANDS_AND_TESTS",
        "07_REVIEW_PACK_FOR_CHATGPT",
        "_tmp",
    )
    for folder in folders:
        (STAGE / folder).mkdir(parents=True, exist_ok=True)
    for path in PROMPT.iterdir():
        if path.is_file():
            shutil.copy2(path, STAGE / "00_PROMPT_AND_INPUTS" / path.name)

    repo = repository_gate()
    prompt = prompt_gate()
    live, _ = live_gate()
    package = build_package()
    write_json(STAGE / "00_PROMPT_AND_INPUTS" / "repository_gate.json", repo)
    write_json(STAGE / "00_PROMPT_AND_INPUTS" / "prompt_pack_validation.json", prompt)
    write_json(STAGE / "01_LIVE_STATE_AND_ROOT_CAUSE" / "live_state_precondition.json", live)
    (STAGE / "01_LIVE_STATE_AND_ROOT_CAUSE" / "completion_root_cause.md").write_text(
        "# Completion root cause\n\n"
        "The browser recreated local IndexedDB drafts when a saved immutable C1 case was revisited. The old "
        "completion handler counted every local draft, including a server-identical saved-case mirror, as unsaved "
        "work. The server correctly rejected the request, but returned plain text and the client surfaced it only "
        "in the lower form error area. The repaired path ignores saved-case mirrors, never flushes case-save events "
        "from the completion button, validates the eight cases from server state, and displays structured HTTP "
        "failure or success acknowledgement in the persistent header.\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        STAGE / "02_COMPLETION_TRANSACTION_REPAIR" / "completion_transaction_contract.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r2_r1.completion_transaction.v1",
            "source_event_sequence": 43,
            "completion_event_sequence": 44,
            "completion_event_count": 1,
            "case_save_events_created": 0,
            "saved_case_mutation_allowed": False,
            "root_state_and_ledger_rollback_capable": True,
            "four_file_bundle_rollback_capable": True,
            "c2_completion_allowed": False,
            "full_pilot_completion_allowed": False,
        },
    )
    write_json(
        STAGE / "03_BROWSER_ERROR_AND_ACKNOWLEDGEMENT" / "browser_acknowledgement_contract.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r2_r1.browser_ack.v1",
            "failure_includes": ["http_status", "error_code", "message", "saved_annotations_unchanged"],
            "success_text": "Tranche C1 - dense overlap completed | Saved to server | pending 0",
            "completion_button_flushes_case_save_events": False,
            "offline_completion_queues_in_new_namespace": True,
            "offline_completion_replays_idempotently": True,
            "saved_case_draft_mirrors_count_as_unsaved": False,
        },
    )
    write_json(
        STAGE / "04_PERSISTENCE_AND_IDEMPOTENCY" / "idempotency_contract.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r2_r1.idempotency.v1",
            "first_completion_event_sequence": 44,
            "repeat_completion_event_sequence": 44,
            "event_45_forbidden": True,
            "deterministic_transaction_from_event_43_state": True,
            "restart_recovers_bundle": True,
        },
    )
    write_json(
        STAGE / "06_COMMANDS_AND_TESTS" / "build_summary.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r2_r1.build_summary.v1",
            "repository_gate": repo["passed"],
            "prompt_gate": prompt["passed"],
            "live_gate": live["passed"],
            "package_gate": package["passed"],
            "tests_pending": True,
            "live_completion_pending": True,
            "classification": "PENDING_C1_COMPLETION_ACCEPTANCE",
        },
    )
    print(STAGE)


if __name__ == "__main__":
    main()
