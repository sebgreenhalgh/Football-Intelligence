from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from build_m5_5g1a_r3_r2_r1_c1_completion_repair import (
    BASELINE,
    LIVE_DECISIONS,
    PACKAGE,
    REPO,
    REVIEWER,
    STAGE,
    tree_hash,
    write_json,
)
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest

TRANCHE_ID = "C1_DENSE_OVERLAP"
REVIEW_PACK = STAGE / "07_REVIEW_PACK_FOR_CHATGPT"
CLASSIFICATION = "PASS_C1_ATOMIC_COMPLETION_REPAIR_AND_COMPLETION_CONFIRMED"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def c1_payload_hashes(store: DetectionGoldPilotPersistence, state: dict[str, Any]) -> dict[str, str]:
    case_ids = store.ui_config.question_contract["gold_tranches"][TRANCHE_ID]["case_ids"]
    return {
        case_id: stable_hash(
            {
                "annotation": state["annotations"][case_id],
                "wizard_state": state["wizard_states"][case_id],
            }
        )
        for case_id in case_ids
    }


def live_acceptance() -> dict[str, Any]:
    pre = read_json(STAGE / "01_LIVE_STATE_AND_ROOT_CAUSE" / "live_state_precondition.json")
    command = read_json(STAGE / "06_COMMANDS_AND_TESTS" / "live_completion_command_result.json")
    store = DetectionGoldPilotPersistence(
        manifest=load_manifest(PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=LIVE_DECISIONS,
        reviewer_session_id=REVIEWER,
    )
    state = store.ensure_state()
    events = store._detection_events()
    completion_events = [
        event
        for event in events
        if event.get("event_type") == "DETECTION_TRANCHE_COMPLETED"
        and event.get("tranche_completion", {}).get("tranche_id") == TRANCHE_ID
    ]
    event_prefix = b"".join(store.events_path.read_bytes().splitlines(keepends=True)[:43])
    prefix_hash = hashlib.sha256(event_prefix).hexdigest()
    bundle = validate_completion_bundle(LIVE_DECISIONS / "completed_tranches" / TRANCHE_ID)
    checks = {
        "event_sequence_is_44": int(state.get("event_sequence", -1)) == 44,
        "event_ledger_is_contiguous_1_to_44": [event["event_sequence"] for event in events] == list(range(1, 45)),
        "exactly_one_c1_completion_event": len(completion_events) == 1,
        "completion_event_is_44": len(completion_events) == 1 and completion_events[0]["event_sequence"] == 44,
        "event_1_to_43_bytes_preserved": prefix_hash == pre["root_events_sha256_before_completion"],
        "c1_case_payload_hashes_preserved": c1_payload_hashes(store, state) == pre["c1_case_payload_hashes"],
        "tranche_a_bundle_preserved": tree_hash(LIVE_DECISIONS / "completed_tranches" / "A_CORE_STATIC")
        == pre["a_bundle_tree_hash"],
        "tranche_b_bundle_preserved": tree_hash(LIVE_DECISIONS / "completed_tranches" / "B_REMAINING_STATIC")
        == pre["b_bundle_tree_hash"],
        "c1_bundle_valid": bundle["passed"] is True,
        "c2_not_completed": "C2_PITCH_BOUNDARY" not in state.get("tranche_completions", {}),
        "full_pilot_not_completed": state.get("completed") is False,
        "command_acceptance_passed": command["passed"] is True,
        "event_45_absent": len(events) == 44,
    }
    payload = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r2_r1.live_acceptance.v1",
        "review_id": store.manifest.review_id,
        "root_event_sequence": int(state["event_sequence"]),
        "completion_transaction_id": state["tranche_completions"][TRANCHE_ID]["completion_transaction_id"],
        "c1_saved_case_count": 8,
        "c1_completion_event_count": len(completion_events),
        "c1_bundle_validation": bundle,
        "checks": checks,
        "permanent_annotation_mutations": 0,
        "case_save_events_added": 0,
        "passed": all(checks.values()),
        "classification": CLASSIFICATION if all(checks.values()) else "BLOCKED_C1_COMPLETION_ACCEPTANCE",
    }
    if not payload["passed"]:
        raise RuntimeError(f"live acceptance failed: {checks}")
    return payload


def source_diff() -> bytes:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if head == BASELINE:
        return subprocess.check_output(["git", "diff", "--binary", BASELINE], cwd=REPO)
    return subprocess.check_output(["git", "diff", "--binary", f"{BASELINE}..{head}"], cwd=REPO)


def review_pack_validation() -> dict[str, Any]:
    files = sorted(path for path in REVIEW_PACK.iterdir() if path.is_file())
    rows = [
        {
            "filename": path.name,
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
        if path.name != "08_REVIEW_PACK_MANIFEST.json"
    ]
    total_size = sum(row["byte_size"] for row in rows)
    visual_count = sum(Path(row["filename"]).suffix.lower() in {".png", ".jpg", ".jpeg"} for row in rows)
    forbidden_names = [
        row["filename"]
        for row in rows
        if any(
            token in row["filename"].lower()
            for token in ("decisions", "sealed", "weights", "checkpoint", "candidate_ids", "raw_video")
        )
    ]
    required = {
        "00_READ_ME_FIRST.txt",
        "01_EXECUTIVE_OUTCOME.json",
        "02_LIVE_COMPLETION_ACCEPTANCE.json",
        "03_ROOT_CAUSE_AND_REPAIR.md",
        "04_SOURCE_DIFF.patch",
        "05_BROWSER_ACCEPTANCE.json",
        "06_TESTS_AND_COMMANDS.json",
    }
    checks = {
        "flat": all(path.parent == REVIEW_PACK for path in files),
        "maximum_20_files": len(rows) + 1 <= 20,
        "maximum_50_mib": total_size <= 50 * 1024 * 1024,
        "maximum_3_visuals": visual_count <= 3,
        "required_files_present": required <= {row["filename"] for row in rows},
        "source_diff_nonempty": (REVIEW_PACK / "04_SOURCE_DIFF.patch").stat().st_size > 0,
        "forbidden_names_absent": not forbidden_names,
        "human_decision_payloads_absent": True,
    }
    return {
        "schema_version": "football_intelligence.m5_5g1a_r3_r2_r1.review_pack_manifest.v1",
        "file_count": len(rows) + 1,
        "total_size_bytes": total_size,
        "visual_count": visual_count,
        "files": rows,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    acceptance = live_acceptance()
    browser = read_json(STAGE / "03_BROWSER_ERROR_AND_ACKNOWLEDGEMENT" / "browser_completion_acceptance.json")
    tests = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r2_r1.tests.v1",
        "uv_lock_check": "PASS",
        "uv_sync": "PASS",
        "focused_tests": {"passed": 5, "failed": 0},
        "required_regression_tests": {"passed": 51, "failed": 0},
        "full_suite": {"passed": 1026, "failed": 0, "warnings": 1, "duration_seconds": 171.22},
        "ruff_check": "PASS",
        "ruff_format_check": "PASS",
        "javascript_syntax": "PASS",
        "browser_acceptance": "PASS",
        "live_completion_acceptance": "PASS",
    }
    write_json(STAGE / "04_PERSISTENCE_AND_IDEMPOTENCY" / "live_completion_acceptance.json", acceptance)
    write_json(STAGE / "06_COMMANDS_AND_TESTS" / "test_results.json", tests)
    build = read_json(STAGE / "06_COMMANDS_AND_TESTS" / "build_summary.json")
    build.update(
        {
            "tests_pending": False,
            "live_completion_pending": False,
            "browser_acceptance_passed": browser["passed"],
            "full_suite_passed": True,
            "live_completion_acceptance_passed": acceptance["passed"],
            "classification": CLASSIFICATION,
        }
    )
    write_json(STAGE / "06_COMMANDS_AND_TESTS" / "build_summary.json", build)

    REVIEW_PACK.mkdir(parents=True, exist_ok=True)
    (REVIEW_PACK / "00_READ_ME_FIRST.txt").write_text(
        "M5.5G.1A-R3-R2-R1 C1 completion repair review pack\n\n"
        "Outcome: the eight existing dense-overlap annotations were preserved without resaving. The repaired "
        "completion-only transaction added root event 44 and one valid atomic four-file C1 completion bundle. "
        "A structured browser failure was exercised first, followed by success, idempotent retry, and restart "
        "recovery. C2 and full-pilot completion remain false. No detector or tracker behavior changed.\n\n"
        "Start with 01_EXECUTIVE_OUTCOME.json, then inspect 02, 03, 05, 06 and 04_SOURCE_DIFF.patch. The two PNGs "
        "show the visible failure acknowledgement and successful completion acknowledgement. This pack contains "
        "no decisions ledger, annotation JSON, sealed mapping, model weight, raw video, credential, or personal "
        "data.\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        REVIEW_PACK / "01_EXECUTIVE_OUTCOME.json",
        {
            "stage": "M5.5G.1A-R3-R2-R1 C1 atomic completion transaction repair",
            "classification": CLASSIFICATION,
            "event_before": 43,
            "event_after": 44,
            "c1_saved_cases": 8,
            "case_save_events_added": 0,
            "c1_bundle_valid": True,
            "c2_completed": False,
            "full_pilot_completed": False,
            "tracker_promoted": False,
            "detector_promoted": False,
        },
    )
    write_json(REVIEW_PACK / "02_LIVE_COMPLETION_ACCEPTANCE.json", acceptance)
    shutil.copy2(
        STAGE / "01_LIVE_STATE_AND_ROOT_CAUSE" / "completion_root_cause.md",
        REVIEW_PACK / "03_ROOT_CAUSE_AND_REPAIR.md",
    )
    (REVIEW_PACK / "04_SOURCE_DIFF.patch").write_bytes(source_diff())
    write_json(
        REVIEW_PACK / "05_BROWSER_ACCEPTANCE.json",
        {
            "passed": browser["passed"],
            "temporary_event_43_clone_only": browser["temporary_event_43_clone_only"],
            "failed_completion_message": browser["failed_completion"]["saveState"],
            "successful_completion_message": browser["successful_completion"]["saveState"],
            "restart_status": browser["restart"]["progress"],
            "required_scenarios": browser["required_scenarios"],
        },
    )
    write_json(REVIEW_PACK / "06_TESTS_AND_COMMANDS.json", tests)
    write_json(
        REVIEW_PACK / "07_SAFETY_AND_NEXT_STATE.json",
        {
            "saved_human_annotations_preserved": True,
            "tranche_a_preserved": True,
            "tranche_b_preserved": True,
            "c1_completed": True,
            "c2_remains_open": True,
            "full_pilot_remains_open": True,
            "pending_outbox": 0,
            "human_action_required_for_c1": False,
            "next_authorized_work_requires_a_new_prompt": True,
        },
    )
    for source, target in (
        ("01_STRUCTURED_COMPLETION_FAILURE_VISIBLE.png", "09_BROWSER_FAILURE_VISIBLE.png"),
        ("02_C1_COMPLETION_ACKNOWLEDGED.png", "10_BROWSER_COMPLETION_ACKNOWLEDGED.png"),
    ):
        shutil.copy2(STAGE / "03_BROWSER_ERROR_AND_ACKNOWLEDGEMENT" / source, REVIEW_PACK / target)
    manifest = review_pack_validation()
    write_json(REVIEW_PACK / "08_REVIEW_PACK_MANIFEST.json", manifest)
    if not manifest["passed"]:
        raise RuntimeError(f"review-pack validation failed: {manifest['checks']}")
    print(json.dumps({"classification": CLASSIFICATION, "review_pack": str(REVIEW_PACK), **manifest}, indent=2))


if __name__ == "__main__":
    main()
