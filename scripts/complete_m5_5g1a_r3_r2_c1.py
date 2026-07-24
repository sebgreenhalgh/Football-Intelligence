from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G1A_R3_R2_R1_C1_ATOMIC_COMPLETION_TRANSACTION_REPAIR_v1"
PACKAGE = STAGE / "05_REPAIRED_DENSE_COMPLETION_PACKAGE"
LIVE_DECISIONS = (
    PART3
    / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
    / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
    / "decisions"
)
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer_r3"
TRANCHE_ID = "C1_DENSE_OVERLAP"
COMMAND = "complete_existing_c1_from_server_state"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def case_payload_hashes(store: DetectionGoldPilotPersistence, state: dict[str, Any]) -> dict[str, str]:
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


def audit_precondition(store: DetectionGoldPilotPersistence) -> tuple[dict[str, Any], dict[str, Any]]:
    state = store.ensure_state()
    events = store._detection_events()
    case_ids = store.ui_config.question_contract["gold_tranches"][TRANCHE_ID]["case_ids"]
    completion_events = [
        event
        for event in events
        if event.get("event_type") == "DETECTION_TRANCHE_COMPLETED"
        and event.get("tranche_completion", {}).get("tranche_id") == TRANCHE_ID
    ]
    already_completed = TRANCHE_ID in state.get("tranche_completions", {})
    checks = {
        "review_id_matches": state.get("review_id") == store.manifest.review_id,
        "tranche_a_completed": "A_CORE_STATIC" in state.get("tranche_completions", {}),
        "tranche_b_completed": "B_REMAINING_STATIC" in state.get("tranche_completions", {}),
        "c1_saved_case_count_is_8": sum(case_id in state.get("annotations", {}) for case_id in case_ids) == 8,
        "event_sequence_is_43_or_completed_44": int(state.get("event_sequence", -1))
        == (44 if already_completed else 43),
        "event_ledger_contiguous": [event["event_sequence"] for event in events]
        == list(range(1, int(state.get("event_sequence", 0)) + 1)),
        "c1_completion_count_is_expected": len(completion_events) == (1 if already_completed else 0),
        "c1_bundle_presence_is_expected": (store.decisions_root / "completed_tranches" / TRANCHE_ID).exists()
        is already_completed,
        "full_pilot_not_completed": state.get("completed") is False,
        "c2_not_completed": "C2_PITCH_BOUNDARY" not in state.get("tranche_completions", {}),
    }
    if not all(checks.values()):
        raise RuntimeError(f"C1 completion precondition failed: {checks}")
    return state, {"checks": checks, "already_completed": already_completed}


def run(args: argparse.Namespace) -> dict[str, Any]:
    decisions_root = args.decisions_root.resolve()
    store = DetectionGoldPilotPersistence(
        manifest=load_manifest(args.package / "reviewer_manifest.json"),
        ui_config=load_ui_config(args.package / "ui_config.json"),
        decisions_root=decisions_root,
        reviewer_session_id=REVIEWER,
    )
    before, precondition = audit_precondition(store)
    before_case_hashes = case_payload_hashes(store, before)
    before_event_bytes = store.events_path.read_bytes()
    before_a_hash = tree_hash(decisions_root / "completed_tranches" / "A_CORE_STATIC")
    before_b_hash = tree_hash(decisions_root / "completed_tranches" / "B_REMAINING_STATIC")
    if not args.execute:
        return {
            "schema_version": "football_intelligence.m5_5g1a_r3_r2_r1.one_time_command.v1",
            "command": COMMAND,
            "mode": "dry_run",
            "decisions_root": str(decisions_root),
            "precondition": precondition,
            "would_add_event_sequence": None if precondition["already_completed"] else 44,
            "would_resave_annotations": False,
            "passed": True,
        }

    request = {
        "review_id": store.manifest.review_id,
        "reviewer_session_id": REVIEWER,
        "tranche_id": TRANCHE_ID,
        "client_event_id": "m5_5g1a_r3_r2_r1_c1_completion_event_44",
        "idempotency_key": f"{store.manifest.review_id}:complete-tranche:{TRANCHE_ID}",
        "expected_server_state_hash": store._server_state_hash(before),
        "pending_outbox_events": 0,
        "evidence_blocker_count": 0,
        "unresolved_draft_count": 0,
        "unresolved_divergence": False,
        "input_source": "complete_existing_c1_from_server_state",
    }
    first = store.complete_tranche(request)
    repeat = store.complete_tranche(request)
    after = store.ensure_state()
    after_events = store._detection_events()
    bundle_root = decisions_root / "completed_tranches" / TRANCHE_ID
    completion_events = [
        event
        for event in after_events
        if event.get("event_type") == "DETECTION_TRANCHE_COMPLETED"
        and event.get("tranche_completion", {}).get("tranche_id") == TRANCHE_ID
    ]
    checks = {
        "event_sequence_is_44": int(after.get("event_sequence", -1)) == 44,
        "exactly_one_c1_completion_event": len(completion_events) == 1,
        "completion_event_is_44": completion_events[0]["event_sequence"] == 44,
        "event_1_to_43_bytes_preserved": store.events_path.read_bytes().startswith(before_event_bytes),
        "c1_case_payload_hashes_preserved": case_payload_hashes(store, after) == before_case_hashes,
        "tranche_a_bundle_preserved": tree_hash(decisions_root / "completed_tranches" / "A_CORE_STATIC")
        == before_a_hash,
        "tranche_b_bundle_preserved": tree_hash(decisions_root / "completed_tranches" / "B_REMAINING_STATIC")
        == before_b_hash,
        "c1_bundle_valid": validate_completion_bundle(bundle_root)["passed"],
        "repeat_is_idempotent": repeat["ack"]["duplicate_event"] is True,
        "repeat_did_not_create_event_45": len(after_events) == 44,
        "c2_not_completed": "C2_PITCH_BOUNDARY" not in after.get("tranche_completions", {}),
        "full_pilot_not_completed": after.get("completed") is False,
        "first_ack_valid": first.get("completion_ack", {}).get("bundle_valid") is True,
    }
    result = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r2_r1.one_time_command.v1",
        "command": COMMAND,
        "mode": "execute",
        "decisions_root": str(decisions_root),
        "precondition": precondition,
        "completion_transaction_id": first["completion_ack"]["completion_transaction_id"],
        "checks": checks,
        "event_sequence_after": int(after["event_sequence"]),
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        raise RuntimeError(f"C1 completion acceptance failed: {checks}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete C1 from immutable server state without resaving cases.")
    parser.add_argument("command", choices=[COMMAND])
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--decisions-root", type=Path, default=LIVE_DECISIONS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    output = args.output or STAGE / "06_COMMANDS_AND_TESTS" / (
        "live_completion_command_result.json" if args.execute else "completion_command_dry_run.json"
    )
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
