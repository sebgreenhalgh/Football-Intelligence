"""Forensically diagnose and safely migrate the failed G7E-B R2 practice draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.temporal_review import TemporalReviewStore

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
R2 = (
    PROJECT
    / "experiments/football_observation_reasoner/part 7"
    / "G7E_B_R2_FULL_TEMPORAL_CANDIDATE_CLOSURE_AND_REVIEWER_REPAIR_v1"
)
B0 = PROJECT / "experiments/football_observation_reasoner/part 7" / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1"
STAGE = (
    PROJECT
    / "experiments/football_observation_reasoner/part 7"
    / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
)
PACKAGE = R2 / "06_REVIEWER_REPAIR/temporal_reviewer_r2"
R3_PACKAGE = STAGE / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3"
PRACTICE_ROOT = B0 / "03_TEMPORAL_REVIEWER/practice_decisions"
REAL_ROOT = PACKAGE / "human_decisions"
SOURCE_DRAFT = PRACTICE_ROOT / "drafts/g7e_a_118576_01.json"
BACKUP = STAGE / "00_INPUT_EVENT_AND_DRAFT_CLOSURE/forensic_backups/g7e_a_118576_01.original.json"
TEMP_MIGRATED = STAGE / "02_DRAFT_REPAIR/g7e_a_118576_01.r3_migrated.temporary.json"
R3_REVISION = "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_V1"
MIGRATION_ID = "G7E_B_R3_PRACTICE_DRAFT_FRAME_BINDING_MIGRATION_V1"


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_digest(value: Any) -> str:
    packed = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(packed).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_bytes(value))


def count(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.glob(pattern)) if root.is_dir() else 0


def inventory(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    return [
        {
            "path": str(path),
            "byte_size": path.stat().st_size,
            "sha256": sha256(path),
            "last_modified_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def unique_frames() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    path = R2 / "01_UNIQUE_FRAME_INDEX/unique_temporal_frame_index.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[row["unique_frame_id"]] = row
    return rows


def load_case(package: Path = PACKAGE) -> dict[str, Any]:
    practice = read_json(package / "practice_cases.json")
    return next(case for case in practice["cases"] if case["burst_id"] == "g7e_a_118576_01")


def identities(case: dict[str, Any]) -> list[dict[str, Any]]:
    by_unique = unique_frames()
    result = []
    for frame, state in zip(case["frames"], case["per_frame_candidate_states"], strict=True):
        unique = by_unique[state["unique_frame_id"]]
        if unique["frame_pixel_sha256"] != frame["source_frame_pixel_sha256"]:
            raise RuntimeError("unique-frame pixel hash mismatch")
        result.append(
            {
                "burst_id": case["burst_id"],
                "frame_id": frame["frame_reference_id"],
                "unique_frame_id": state["unique_frame_id"],
                "frame_index": unique["frame_index_zero_based"],
                "frame_pixel_sha256": frame["source_frame_pixel_sha256"],
            }
        )
    return result


def event_payload(draft: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "practice",
        "burst_id": draft["burst_id"],
        "original_focus_box_answer": draft["answers"]["original_focus_box_answer"],
        "context_subject_answer": draft["answers"].get("context_subject_answer", "NOT_APPLICABLE"),
        "subjects": draft["subjects"],
        "candidate_mappings": draft["candidate_mappings"],
        "whole_burst_missed_person_answer": draft["answers"]["missed_check"],
        "whole_burst_missed_person_marks": draft["missed_person_marks"],
        "source_frame_hashes": [frame["source_frame_pixel_sha256"] for frame in case["frames"]],
        "candidate_runtime_contract": case["candidate_runtime_contract"],
        "unique_frame_candidate_status": case["unique_frame_candidate_status"],
        "per_frame_candidate_states": case["per_frame_candidate_states"],
        "summary_confirmed": True,
        "draft_version": draft.get("draft_version"),
        "draft_content_sha256": draft.get("draft_content_sha256"),
        "optimistic_lock_token": draft.get("optimistic_lock_token"),
        "click_transactions": draft.get("click_transactions", []),
    }


def preflight() -> dict[str, Any]:
    state = {
        "schema_version": "football_intelligence.g7e_b_r3.storage_root_preflight.v1",
        "repository_head": os.popen(f'git -C "{REPO}" rev-parse HEAD').read().strip(),
        "roots": {
            "practice": str(PRACTICE_ROOT),
            "real": str(REAL_ROOT),
            "r2_temporary_acceptance": str(R2 / "07_BROWSER_ACCEPTANCE/_temporary_r2_acceptance"),
            "r3_temporary_acceptance": str(STAGE / "05_BROWSER_ACCEPTANCE/_temporary_r3_acceptance"),
        },
        "counts": {
            "practice_drafts": count(PRACTICE_ROOT, "drafts/*.json"),
            "practice_events": count(PRACTICE_ROOT, "events/*/*.json"),
            "practice_acknowledgements": count(PRACTICE_ROOT, "receipts/acknowledgements/*.json"),
            "real_drafts": count(REAL_ROOT, "drafts/*.json"),
            "real_events": count(REAL_ROOT, "events/*/*.json"),
            "real_acknowledgements": count(REAL_ROOT, "receipts/acknowledgements/*.json"),
            "real_tranche_receipts": count(REAL_ROOT, "receipts/tranche_completion/*.json"),
            "real_global_receipts": count(REAL_ROOT, "receipts/global_completion/*.json"),
        },
        "practice_files": inventory(PRACTICE_ROOT),
        "real_files": inventory(REAL_ROOT),
        "temporary_acceptance_roots_present": {
            "r2": (R2 / "07_BROWSER_ACCEPTANCE/_temporary_r2_acceptance").exists(),
            "r3": (STAGE / "05_BROWSER_ACCEPTANCE/_temporary_r3_acceptance").exists(),
        },
        "production_ready": False,
    }
    real_keys = (
        "real_events",
        "real_acknowledgements",
        "real_tranche_receipts",
        "real_global_receipts",
    )
    state["passed"] = state["repository_head"] == "c9360bdf09cc2d78e693e571f9ae294f67a1af2e" and all(
        state["counts"][key] == 0 for key in real_keys
    )
    if not state["passed"]:
        raise SystemExit("FAIL_G7E_B_R3_REAL_EVENT_PREFLIGHT")
    return state


def prove_bindings(draft: dict[str, Any], case: dict[str, Any], frame_identities: list[dict[str, Any]]):
    candidates = {
        candidate["candidate_id"]: (sequence, candidate)
        for sequence, rows in enumerate(case["frame_candidates"])
        for candidate in rows
    }
    mappings = {mapping["candidate_id"]: mapping for mapping in draft["candidate_mappings"]}
    forensic_rows = []
    proof_failures = []
    for subject_index, subject in enumerate(draft["subjects"]):
        for sequence, observation in enumerate(subject["frame_observations"]):
            expected = frame_identities[sequence]
            selected = observation.get("selected_candidate_ids", [])
            selected_proof = []
            for candidate_id in selected:
                candidate_record = candidates.get(candidate_id)
                mapping = mappings.get(candidate_id)
                valid = bool(
                    candidate_record
                    and candidate_record[0] == sequence
                    and mapping
                    and mapping.get("frame_sequence") == sequence
                    and mapping.get("frame_reference_id") == expected["frame_id"]
                )
                selected_proof.append({"candidate_id": candidate_id, "exact_frame_proof": valid})
            location_present = isinstance(observation.get("subject_location_source_x"), (int, float))
            intent_proven = bool(selected) and all(row["exact_frame_proof"] for row in selected_proof)
            if location_present and not intent_proven:
                proof_failures.append({"subject_token": subject["subject_token"], "frame_sequence": sequence})
            forensic_rows.append(
                {
                    "subject_token": subject["subject_token"],
                    "subject_index": subject_index,
                    "observation_frame_id": expected["frame_id"],
                    "observation_unique_frame_id": expected["unique_frame_id"],
                    "observation_frame_index": expected["frame_index"],
                    "visibility_state": observation.get("visibility"),
                    "location_present": location_present,
                    "location_declared_frame_id": observation.get("frame_reference_id"),
                    "location_declared_unique_frame_id": observation.get("unique_frame_id"),
                    "location_declared_frame_index": observation.get("frame_index"),
                    "source_coordinates": [
                        observation.get("subject_location_source_x"),
                        observation.get("subject_location_source_y"),
                    ]
                    if location_present
                    else None,
                    "draft_question_provenance": f"subject_{subject_index}_location_{sequence}",
                    "last_modified_sequence": None,
                    "selected_candidate_frame_proof": selected_proof,
                    "validation_result": "MISSING_CANONICAL_FRAME_IDENTITY",
                    "intent_proven": intent_proven,
                }
            )
    return forensic_rows, proof_failures


def migrate(draft: dict[str, Any], case: dict[str, Any], frame_identities: list[dict[str, Any]]) -> dict[str, Any]:
    migrated = json.loads(json.dumps(draft))
    migrated["schema_version"] = "football_intelligence.g7e_b_r3.temporal_review_draft.v1"
    migrated["review_revision"] = R3_REVISION
    migrated["draft_version"] = 1
    migrated["click_transactions"] = []
    migrated["prior_final_save_error"] = {
        "error_code": "FINAL_SAVE_ERROR",
        "message": "subject location frame mismatch",
        "source_revision": draft["review_revision"],
        "preserved_for_targeted_restore": True,
    }
    migrated["targeted_correction"] = None
    migrated["action_journal"] = []
    for subject_index, subject in enumerate(migrated["subjects"]):
        anchor_sequence = subject["anchor_frame_sequence"]
        subject["anchor_canonical_frame_identity"] = frame_identities[anchor_sequence]
        for sequence, observation in enumerate(subject["frame_observations"]):
            identity = frame_identities[sequence]
            observation["frame_reference_id"] = identity["frame_id"]
            observation["canonical_frame_identity"] = identity
            observation["candidate_selection_binding"] = {
                "action_type": "CANDIDATE_SELECTION",
                "canonical_frame_identity": identity,
                "question_id": f"subject_{subject_index}_supply_{sequence}",
                "selected_candidate_ids": observation.get("selected_candidate_ids", []),
                "binding_provenance": "EXACT_SELECTED_CANDIDATE_UNIQUE_FRAME",
                "migration_record_id": MIGRATION_ID,
            }
            x = observation.get("subject_location_source_x")
            y = observation.get("subject_location_source_y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                observation["location_binding"] = {
                    "action_type": (
                        "APPROXIMATE_HIDDEN_LOCATION"
                        if observation.get("approximate_hidden_location")
                        else "SUBJECT_LOCATION"
                    ),
                    "canonical_frame_identity": identity,
                    "question_id": f"subject_{subject_index}_location_{sequence}",
                    "source_xy": [x, y],
                    "binding_provenance": "EXACT_SELECTED_CANDIDATE_UNIQUE_FRAME",
                    "migration_record_id": MIGRATION_ID,
                }
            migrated["action_journal"].append(
                {
                    "action": "BIND_EXISTING_FRAME_LOCAL_METADATA",
                    "subject_token": subject["subject_token"],
                    "frame_id": identity["frame_id"],
                    "question_id": f"subject_{subject_index}_location_{sequence}",
                    "migration_record_id": MIGRATION_ID,
                }
            )
    for mapping in migrated["candidate_mappings"]:
        mapping["canonical_frame_identity"] = frame_identities[mapping["frame_sequence"]]
    for mark in migrated["missed_person_marks"]:
        identity = frame_identities[mark["frame_sequence"]]
        mark["canonical_frame_identity"] = identity
        mark["mark_binding"] = {
            "action_type": "MISSED_PERSON_MARK",
            "canonical_frame_identity": identity,
            "question_id": "missed_mark",
            "source_xy": mark["source_xy"],
            "binding_provenance": "EXISTING_FRAME_REFERENCE_AND_SEQUENCE_AGREE",
            "migration_record_id": MIGRATION_ID,
        }
    migrated["updated_at_utc"] = now()
    migrated["draft_content_sha256"] = canonical_digest(migrated)
    migrated["optimistic_lock_token"] = canonical_digest(
        {
            "review_revision": R3_REVISION,
            "burst_id": migrated["burst_id"],
            "draft_version": migrated["draft_version"],
            "draft_content_sha256": migrated["draft_content_sha256"],
        }
    )
    return migrated


def analyze() -> None:
    root_state = preflight()
    if not BACKUP.is_file() or sha256(BACKUP) != sha256(SOURCE_DRAFT):
        raise SystemExit("FAIL_G7E_B_R3_INPUT_PROVENANCE")
    draft = read_json(BACKUP)
    case = load_case()
    frame_identities = identities(case)
    store = TemporalReviewStore(PACKAGE, STAGE / "_forensic_tmp/real", STAGE / "_forensic_tmp/practice")
    reproduced = None
    try:
        store._validate_event(event_payload(draft, case), "practice")  # noqa: SLF001
    except ValueError as exc:
        reproduced = str(exc)
    if reproduced != "subject location frame mismatch":
        raise SystemExit("FAIL_G7E_B_R3_FAILURE_REPRODUCTION")
    rows, failures = prove_bindings(draft, case, frame_identities)
    if failures:
        raise SystemExit("FAIL_G7E_B_R3_DRAFT_REPAIR")
    migrated = migrate(draft, case, frame_identities)
    TEMP_MIGRATED.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(TEMP_MIGRATED, canonical_bytes(migrated))
    if not R3_PACKAGE.is_dir():
        raise SystemExit("FAIL_G7E_B_R3_FRAME_BINDING: R3 package missing")
    r3_case = load_case(R3_PACKAGE)
    r3_store = TemporalReviewStore(R3_PACKAGE, STAGE / "_r3_validation_tmp/real", STAGE / "_r3_validation_tmp/practice")
    r3_store._validate_draft(migrated)  # noqa: SLF001
    r3_store._validate_r3_frame_bindings(migrated, r3_case, final=True)  # noqa: SLF001
    r3_store._validate_event(event_payload(migrated, r3_case), "practice")  # noqa: SLF001
    zero_counts = root_state["counts"]
    write_json(STAGE / "00_INPUT_EVENT_AND_DRAFT_CLOSURE/storage_root_preflight.json", root_state)
    write_json(
        STAGE / "00_INPUT_EVENT_AND_DRAFT_CLOSURE/practice_draft_forensic_backup_manifest.json",
        {
            "schema_version": "football_intelligence.g7e_b_r3.forensic_backup_manifest.v1",
            "source_path": str(SOURCE_DRAFT),
            "backup_path": str(BACKUP),
            "byte_size": BACKUP.stat().st_size,
            "sha256": sha256(BACKUP),
            "schema": draft["schema_version"],
            "review_revision": draft["review_revision"],
            "burst_id": draft["burst_id"],
            "current_question": draft["current_question"],
            "current_frame_sequence": draft["current_frame_sequence"],
            "updated_at_utc": draft["updated_at_utc"],
            "practice_event_count": zero_counts["practice_events"],
            "practice_acknowledgement_count": zero_counts["practice_acknowledgements"],
            "source_and_backup_hash_match": True,
            "immutable_forensic_copy": True,
        },
    )
    write_json(
        STAGE / "01_FORENSIC_ROOT_CAUSE/final_save_failure_reproduction.json",
        {
            "schema_version": "football_intelligence.g7e_b_r3.failure_reproduction.v1",
            "draft_sha256": sha256(BACKUP),
            "temporary_copy_used": True,
            "actual_draft_mutated": False,
            "observed_error": reproduced,
            "expected_error": "subject location frame mismatch",
            "reproduced": True,
        },
    )
    table = STAGE / "01_FORENSIC_ROOT_CAUSE/frame_binding_forensic_table.jsonl"
    atomic_write(table, b"".join(canonical_bytes(row) for row in rows))
    write_json(
        STAGE / "01_FORENSIC_ROOT_CAUSE/root_cause.json",
        {
            "schema_version": "football_intelligence.g7e_b_r3.root_cause.v1",
            "classification": "OTHER_PROVEN_ROOT_CAUSE",
            "specific_root_cause": "FRAME_IDENTITY_OMITTED_FROM_FRAME_LOCAL_SUBJECT_OBSERVATIONS",
            "client_creation_site": "g7e_b_r2_temporal_review.js observation()",
            "client_serialization_site": "eventPayload() passes mutable subjects through without adding frame identity",
            "server_detection_site": "TemporalReviewStore._validate_r1_event requires observation.frame_reference_id",
            "mismatched_subject_observation_count": len(rows),
            "source_coordinates_moved": False,
            "intent_proof": (
                "Every location has selected candidate IDs that exist only in the exact mapped frame; "
                "no contradiction exists."
            ),
            "stale_active_frame_capture_proven": False,
            "async_navigation_race_proven": False,
            "draft_restore_rebind_proven": False,
            "array_position_only_used_as_repair_proof": False,
            "passed": True,
        },
    )
    before_hash = sha256(BACKUP)
    after_hash = sha256(TEMP_MIGRATED)
    write_json(
        STAGE / "02_DRAFT_REPAIR/practice_draft_repair_decision.json",
        {
            "schema_version": "football_intelligence.g7e_b_r3.draft_repair_decision.v1",
            "decision": "DETERMINISTIC_METADATA_ONLY_MIGRATION",
            "intent_proven_for_all_locations": True,
            "proof_source": "EXACT_SELECTED_CANDIDATE_UNIQUE_FRAME",
            "coordinates_unchanged": True,
            "answers_unchanged": True,
            "user_repetition_required": False,
            "actual_draft_updated": False,
            "temporary_migrated_draft_sha256": after_hash,
        },
    )
    write_json(
        STAGE / "02_DRAFT_REPAIR/practice_draft_migration_record.json",
        {
            "schema_version": "football_intelligence.g7e_b_r3.practice_draft_migration.v1",
            "migration_record_id": MIGRATION_ID,
            "source_draft_sha256": before_hash,
            "temporary_migrated_draft_sha256": after_hash,
            "source_coordinates_changed": 0,
            "human_answers_changed": 0,
            "subject_observation_bindings_added": len(rows),
            "candidate_mapping_bindings_added": len(draft["candidate_mappings"]),
            "missed_person_mark_bindings_added": len(draft["missed_person_marks"]),
            "click_transactions_fabricated": 0,
            "actual_draft_update_pending_temporary_acceptance": True,
            "append_only_forensic_backup": str(BACKUP),
        },
    )
    print("PASS_G7E_B_R3_FORENSIC_ROOT_CAUSE_AND_TEMPORARY_MIGRATION")


def apply_actual() -> None:
    if sha256(SOURCE_DRAFT) != sha256(BACKUP):
        raise SystemExit("FAIL_G7E_B_R3_DRAFT_REPAIR: source changed after forensic backup")
    report_path = STAGE / "05_BROWSER_ACCEPTANCE/browser_acceptance_report.json"
    if not report_path.is_file():
        raise SystemExit("FAIL_G7E_B_R3_DRAFT_REPAIR: browser acceptance missing")
    report = read_json(report_path)
    if report.get("decision") != "PASS_G7E_B_R3_REAL_EDGE_ACCEPTANCE":
        raise SystemExit("FAIL_G7E_B_R3_DRAFT_REPAIR: temporary acceptance did not pass")
    if report.get("temporary_migrated_draft_sha256") != sha256(TEMP_MIGRATED):
        raise SystemExit("FAIL_G7E_B_R3_DRAFT_REPAIR: accepted migration hash mismatch")
    before = sha256(SOURCE_DRAFT)
    atomic_write(SOURCE_DRAFT, TEMP_MIGRATED.read_bytes())
    after = sha256(SOURCE_DRAFT)
    if after != sha256(TEMP_MIGRATED):
        raise SystemExit("FAIL_G7E_B_R3_DRAFT_REPAIR: actual migration write failed")
    write_json(
        STAGE / "02_DRAFT_REPAIR/actual_practice_draft_migration_application.json",
        {
            "schema_version": "football_intelligence.g7e_b_r3.actual_migration_application.v1",
            "source_path": str(SOURCE_DRAFT),
            "before_sha256": before,
            "after_sha256": after,
            "temporary_acceptance_report_sha256": sha256(report_path),
            "coordinates_changed": 0,
            "human_answers_changed": 0,
            "actual_final_practice_event_created": False,
            "applied_at_utc": now(),
            "passed": True,
        },
    )
    print("PASS_G7E_B_R3_ACTUAL_PRACTICE_DRAFT_MIGRATED_WITHOUT_FINAL_SAVE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-actual", action="store_true")
    args = parser.parse_args()
    apply_actual() if args.apply_actual else analyze()


if __name__ == "__main__":
    main()
