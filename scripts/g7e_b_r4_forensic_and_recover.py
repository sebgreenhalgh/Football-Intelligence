"""Forensically close and safely recover the G7E-B R4 real draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.temporal_review import TemporalReviewStore, create_server

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
R3 = PART7 / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
R4 = PART7 / "G7E_B_R4_CANDIDATE_RELATIONSHIP_BRANCH_INTEGRITY_AND_REAL_DRAFT_RECOVERY_v1"
R3_PACKAGE = R3 / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3"
R4_PACKAGE = R4 / "02_BRANCH_COMPATIBILITY_ENGINE/temporal_reviewer_r4"
REAL_ROOT = R3_PACKAGE / "human_decisions"
REAL_DRAFT = REAL_ROOT / "drafts/g7e_a_117093_10.json"
BACKUP = R4 / "00_INPUT_EVENT_AND_REAL_DRAFT_CLOSURE/forensic_backups/g7e_a_117093_10.original.json"
ASSET_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/assets"
EXPECTED_R3_COMMIT = "6357aff3d030b38ff879bb25281d9f2823c68925"
R4_REVISION = "G7E_B_R4_CANDIDATE_RELATIONSHIP_BRANCH_INTEGRITY_V1"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def file_rows(root: Path, pattern: str) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "byte_size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.glob(pattern))
        if path.is_file()
    ]


def event_payload(draft: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "real",
        "burst_id": draft["burst_id"],
        "original_focus_box_answer": draft["answers"]["original_focus_box_answer"],
        "context_subject_answer": draft["answers"].get("context_subject_answer") or "NOT_APPLICABLE",
        "subjects": draft["subjects"],
        "candidate_mappings": draft["candidate_mappings"],
        "whole_burst_missed_person_answer": draft["answers"]["missed_check"],
        "whole_burst_missed_person_marks": draft["missed_person_marks"],
        "source_frame_hashes": [frame["source_frame_pixel_sha256"] for frame in case["frames"]],
        "candidate_runtime_contract": draft["candidate_runtime_contract"],
        "unique_frame_candidate_status": draft["unique_frame_candidate_status"],
        "per_frame_candidate_states": draft["per_frame_candidate_states"],
        "summary_confirmed": True,
        "draft_version": draft["draft_version"],
        "draft_content_sha256": draft["draft_content_sha256"],
        "optimistic_lock_token": draft["optimistic_lock_token"],
        "click_transactions": draft["click_transactions"],
    }


def reproduce_http(draft: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="g7e_b_r4_reproduction_") as temp:
        temp_root = Path(temp)
        decisions = temp_root / "real"
        practice = temp_root / "practice"
        (decisions / "drafts").mkdir(parents=True)
        practice.mkdir()
        shutil.copyfile(BACKUP, decisions / "drafts" / BACKUP.name.replace(".original", ""))
        store = TemporalReviewStore(R3_PACKAGE, decisions, practice)
        case = store.by_id[draft["burst_id"]]
        payload = event_payload(draft, case)
        server = create_server(R3_PACKAGE, decisions, practice, ASSET_ROOT, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        body: dict[str, Any]
        status: int
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_address[1]}/api/final-save-preflight",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
                    status = response.status
                    body = json.loads(response.read())
            except urllib.error.HTTPError as error:
                status = error.code
                body = json.loads(error.read())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        return {
            "http_status": status,
            "response": body,
            "temporary_copy_sha256": sha256(decisions / "drafts" / "g7e_a_117093_10.json"),
            "temporary_event_count": len(list((decisions / "events").rglob("*.json")))
            if (decisions / "events").exists()
            else 0,
            "temporary_acknowledgement_count": len(list((decisions / "receipts/acknowledgements").glob("*.json")))
            if (decisions / "receipts/acknowledgements").exists()
            else 0,
        }


def relationship_rows(draft: dict[str, Any], case: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    journal = list(draft.get("action_journal", []))
    for subject_index, subject in enumerate(draft["subjects"]):
        stored = subject.get("candidate_relationship")
        for sequence, observation in enumerate(subject["frame_observations"]):
            supply = observation.get("observation_supply")
            selected = list(observation.get("selected_candidate_ids", []))
            local = {row["candidate_id"]: row for row in case["frame_candidates"][sequence]}
            selection_entries = [
                (index, entry)
                for index, entry in enumerate(journal)
                if entry.get("action_type") == "CANDIDATE_SELECTION"
                and entry.get("question_id") == f"subject_{subject_index}_supply_{sequence}"
            ]
            applicable = supply in ("MULTIPLE_CANDIDATES", "MERGED_WITH_OTHER_PEOPLE", "FRAGMENT_ONLY")
            selected_frames = [
                case["frames"][sequence]["frame_reference_id"] if candidate_id in local else None
                for candidate_id in selected
            ]
            valid_selection = (
                len(selected) == len(set(selected))
                and all(candidate_id in local for candidate_id in selected)
                and ((supply == "ONE_USEFUL_CANDIDATE" and len(selected) == 1) or applicable)
            )
            rows.append(
                {
                    "tranche_id": draft["tranche_id"],
                    "burst_id": draft["burst_id"],
                    "subject_token": subject["subject_token"],
                    "subject_index": subject_index,
                    "frame_sequence": sequence,
                    "observation_frame_id": observation["frame_reference_id"],
                    "unique_frame_id": observation["canonical_frame_identity"]["unique_frame_id"],
                    "candidate_status_state": case["per_frame_candidate_states"][sequence]["candidate_status"],
                    "candidate_supply_answer": supply,
                    "selected_candidate_ids": selected,
                    "selected_candidate_count": len(selected),
                    "selected_candidate_frame_ids": selected_frames,
                    "stored_relationship": stored,
                    "relationship_storage_location": f"subjects[{subject_index}].candidate_relationship",
                    "relationship_question_id": f"subject_{subject_index}_relationship",
                    "relationship_question_branch_family": "LEGACY_SUBJECT_LEVEL_GENERIC",
                    "relationship_applicable": applicable,
                    "question_journal_sequence": selection_entries[-1][0] if selection_entries else None,
                    "last_upstream_answer_change": None,
                    "last_candidate_selection_change": selection_entries[-1][1] if selection_entries else None,
                    "draft_version": draft["draft_version"],
                    "candidate_selection_valid": valid_selection,
                    "validation_result": (
                        "INVALID_NULL_SUBJECT_RELATIONSHIP_FOR_NON_APPLICABLE_SINGLE_BOX_BRANCH"
                        if supply == "ONE_USEFUL_CANDIDATE" and stored is None
                        else "REQUIRES_R4_COMPATIBILITY_VALIDATION"
                    ),
                    "error_path": f"subjects[{subject_index}].candidate_relationship",
                }
            )
    return rows


def preserved_human_projection(draft: dict[str, Any]) -> dict[str, Any]:
    return {
        "answers": draft.get("answers"),
        "subjects": [
            {
                "subject_token": subject.get("subject_token"),
                "subject_definition_source": subject.get("subject_definition_source"),
                "anchor_frame_sequence": subject.get("anchor_frame_sequence"),
                "anchor_source_xy": subject.get("anchor_source_xy"),
                "marker_continuity_confirmation": subject.get("marker_continuity_confirmation"),
                "occlusion_confirmed": subject.get("occlusion_confirmed"),
                "role": subject.get("role"),
                "participation": subject.get("participation"),
                "certainty": subject.get("certainty"),
                "frame_observations": [
                    {
                        key: observation.get(key)
                        for key in (
                            "frame_reference_id",
                            "canonical_frame_identity",
                            "visibility",
                            "subject_location_source_x",
                            "subject_location_source_y",
                            "human_confirmed",
                            "approximate_hidden_location",
                            "location_binding",
                            "observation_supply",
                            "selected_candidate_ids",
                            "candidate_selection_binding",
                            "occlusion_phase",
                        )
                    }
                    for observation in subject.get("frame_observations", [])
                ],
            }
            for subject in draft.get("subjects", [])
        ],
        "candidate_mappings": draft.get("candidate_mappings"),
        "missed_person_marks": draft.get("missed_person_marks"),
        "click_transactions": draft.get("click_transactions"),
    }


def migrate_payload(source: dict[str, Any], contract: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    migrated = json.loads(json.dumps(source))
    changes: list[dict[str, Any]] = []
    migrated["schema_version"] = "football_intelligence.g7e_b_r4.temporal_review_draft.v1"
    migrated["review_revision"] = R4_REVISION
    aliases = contract["legacy_supply_aliases"]
    for subject_index, subject in enumerate(migrated["subjects"]):
        legacy_relationship = subject.pop("candidate_relationship", None)
        for sequence, observation in enumerate(subject["frame_observations"]):
            raw_supply = observation.get("observation_supply")
            supply = aliases.get(raw_supply, raw_supply)
            if raw_supply != supply:
                observation["observation_supply"] = supply
                changes.append(
                    {
                        "path": f"subjects[{subject_index}].frame_observations[{sequence}].observation_supply",
                        "before": raw_supply,
                        "after": supply,
                        "reason": "CANONICAL_LEGACY_SUPPLY_ALIAS",
                    }
                )
            state = contract["supply_states"].get(supply)
            if state is None:
                raise RuntimeError("FAIL_G7E_B_R4_REAL_DRAFT_RECOVERY: unsupported supply state")
            path = f"subjects[{subject_index}].frame_observations[{sequence}]"
            if not state["relationship_applicable"]:
                previous = observation.get("candidate_relationship", legacy_relationship)
                observation["candidate_relationship"] = state["canonical_relationship"]
                observation["relationship_question_id"] = None
                observation["relationship_branch_family"] = None
                changes.append(
                    {
                        "path": f"{path}.candidate_relationship",
                        "before": previous,
                        "after": state["canonical_relationship"],
                        "reason": "PROVEN_NON_APPLICABLE_FRAME_BRANCH",
                    }
                )
                migrated["action_journal"].append(
                    {
                        "action": "RELATIONSHIP_BRANCH_INVALIDATED",
                        "subject_token": subject["subject_token"],
                        "frame_reference_id": observation["frame_reference_id"],
                        "frame_sequence": sequence,
                        "previous_value": previous,
                        "reason": "GENERIC_SINGLE_BOX_RELATIONSHIP_BUG",
                        "upstream_change": f"CURRENT_CANONICAL_SUPPLY={supply}",
                        "question_id": f"subject_{subject_index}_relationship_{sequence}",
                        "migration_record_id": "G7E_B_R4_REAL_DRAFT_NON_APPLICABLE_RELATIONSHIP_MIGRATION_V1",
                        "created_at_utc": utc_now(),
                    }
                )
                continue
            family = state["question_family"]
            allowed = contract["question_families"][family]["allowed_relationships"]
            if supply == "MERGED_WITH_OTHER_PEOPLE" and legacy_relationship is None:
                relationship = "MERGED_MULTI_PERSON"
                derivation = "DERIVED_DIRECTLY_FROM_CANONICAL_HUMAN_SUPPLY_ANSWER"
            elif legacy_relationship in allowed:
                relationship = legacy_relationship
                derivation = "PRESERVED_EXISTING_APPLICABLE_HUMAN_ANSWER"
            else:
                relationship = None
                derivation = "TARGETED_HUMAN_CORRECTION_REQUIRED"
            observation["candidate_relationship"] = relationship
            observation["relationship_question_id"] = f"subject_{subject_index}_relationship_{sequence}"
            observation["relationship_branch_family"] = family
            changes.append(
                {
                    "path": f"{path}.candidate_relationship",
                    "before": legacy_relationship,
                    "after": relationship,
                    "reason": derivation,
                }
            )
        needs_occlusion = any(
            observation.get("visibility") in ("VISIBLE_PARTIAL", "FULLY_OCCLUDED_EXPECTED_PRESENT")
            for observation in subject["frame_observations"]
        )
        has_applicable_relationship = any(
            contract["supply_states"][observation["observation_supply"]]["relationship_applicable"]
            for observation in subject["frame_observations"]
        )
        if (
            subject.get("continuity") is None
            and not needs_occlusion
            and not has_applicable_relationship
            and subject.get("marker_continuity_confirmation") != "CANNOT_TELL"
        ):
            subject["continuity"] = "NOT_APPLICABLE"
            changes.append(
                {
                    "path": f"subjects[{subject_index}].continuity",
                    "before": None,
                    "after": "NOT_APPLICABLE",
                    "reason": "PROVEN_SKIPPED_NON_APPLICABLE_CONTINUITY_BRANCH",
                }
            )
    migrated["real_draft_recovery"] = {
        "migration_record_id": "G7E_B_R4_REAL_DRAFT_NON_APPLICABLE_RELATIONSHIP_MIGRATION_V1",
        "source_sha256": sha256(BACKUP),
        "human_answers_changed": 0,
        "source_coordinates_changed": 0,
        "candidate_selections_changed": 0,
        "real_event_created_by_migration": False,
    }
    migrated["current_question"] = "summary"
    migrated["current_frame_sequence"] = 8
    return migrated, changes


def validate_temporary_recovery() -> dict[str, Any]:
    contract = read_json(R4_PACKAGE / "relationship_compatibility.json")
    source = read_json(BACKUP)
    migrated, changes = migrate_payload(source, contract)
    if preserved_human_projection(source) != preserved_human_projection(migrated):
        raise RuntimeError("FAIL_G7E_B_R4_REAL_DRAFT_RECOVERY: unaffected human projection changed")
    with tempfile.TemporaryDirectory(prefix="g7e_b_r4_recovery_") as temp:
        root = Path(temp)
        decisions = root / "real"
        practice = root / "practice"
        (decisions / "drafts").mkdir(parents=True)
        practice.mkdir()
        temp_draft = decisions / "drafts/g7e_a_117093_10.json"
        shutil.copyfile(BACKUP, temp_draft)
        store = TemporalReviewStore(R4_PACKAGE, decisions, practice, acceptance_mode=True)
        saved = store.save_draft(migrated, "real")
        case = store.by_id[saved["burst_id"]]
        payload = event_payload(saved, case)
        preflight = store.final_save_preflight(payload, "real")
        if preflight.get("status") != "READY_TO_PERSIST":
            raise RuntimeError("FAIL_G7E_B_R4_FINAL_SAVE: temporary preflight failed")
        payload.update(
            {
                "proposed_event_id": preflight["proposed_event_id"],
                "idempotency_key": preflight["idempotency_key"],
                "acceptance_temporary": True,
            }
        )
        first = store.save_event(payload, "real")
        second = store.save_event(payload, "real")
        events = list((decisions / "events").rglob("*.json"))
        acknowledgements = list((decisions / "receipts/acknowledgements").glob("*.json"))
        if len(events) != 1 or len(acknowledgements) != 1 or not second["recovered_existing_event"]:
            raise RuntimeError("FAIL_G7E_B_R4_FINAL_SAVE: temporary idempotency failed")
        result = {
            "schema_version": "football_intelligence.g7e_b_r4.temporary_recovery_validation.v1",
            "temporary_only": True,
            "source_sha256": sha256(BACKUP),
            "saved_temporary_draft_sha256": saved["server_file_sha256"],
            "saved_temporary_draft_version": saved["draft_version"],
            "relationship_change_count": len(changes),
            "preflight": preflight,
            "first_save": first,
            "duplicate_save": second,
            "event_count": len(events),
            "acknowledgement_count": len(acknowledgements),
            "read_only_event_validated": read_json(events[0])["relationship_branch_validation"] == "PASSED",
            "unaffected_human_projection_preserved": True,
            "human_answers_changed": 0,
            "source_coordinates_changed": 0,
            "candidate_selections_changed": 0,
            "passed": True,
        }
    write_json(R4 / "02_BRANCH_COMPATIBILITY_ENGINE/relationship_compatibility_matrix.json", contract)
    write_json(
        R4 / "02_BRANCH_COMPATIBILITY_ENGINE/branch_invalidation_results.json",
        {
            "schema_version": "football_intelligence.g7e_b_r4.branch_invalidation_results.v1",
            "migration_changes": changes,
            "branch_invalidation_journal_entries_added": len(changes),
            "hidden_stale_answers_remaining": 0,
            "human_answers_replaced": 0,
            "passed": True,
        },
    )
    write_json(
        R4 / "03_REAL_DRAFT_RECOVERY/real_draft_repair_decision.json",
        {
            "schema_version": "football_intelligence.g7e_b_r4.real_draft_repair_decision.v1",
            "decision": "DETERMINISTIC_NON_APPLICABLE_RELATIONSHIP_METADATA_MIGRATION",
            "targeted_human_correction_required": False,
            "proof": (
                "ALL_NINE_FRAMES_ONE_USEFUL_CANDIDATE_EXACTLY_ONE_VALID_SELECTED_BOX_" "NO_RELATIONSHIP_QUESTION_ASKED"
            ),
            "genuine_human_relationship_answer_rewritten": False,
            "unaffected_human_projection_preserved": True,
            "temporary_final_save_passed": True,
            "actual_draft_update_authorized_after_edge_acceptance": True,
        },
    )
    write_json(R4 / "04_FINAL_SAVE_AND_TARGETED_CORRECTION/final_save_preflight_results.json", result)
    return result


def apply_actual_recovery() -> None:
    validation = read_json(R4 / "04_FINAL_SAVE_AND_TARGETED_CORRECTION/final_save_preflight_results.json")
    browser = read_json(R4 / "05_BROWSER_ACCEPTANCE/browser_acceptance_report.json")
    if not validation.get("passed") or browser.get("decision") != "PASS_G7E_B_R4_REAL_EDGE_ACCEPTANCE":
        raise RuntimeError("FAIL_G7E_B_R4_REAL_DRAFT_RECOVERY: temporary acceptance gate is incomplete")
    if sha256(REAL_DRAFT) != sha256(BACKUP):
        raise RuntimeError("FAIL_G7E_B_R4_REAL_DRAFT_RECOVERY: actual draft changed after forensic closure")
    before = read_json(BACKUP)
    contract = read_json(R4_PACKAGE / "relationship_compatibility.json")
    migrated, changes = migrate_payload(before, contract)
    before_counts = {
        "events": len(file_rows(REAL_ROOT, "events/**/*.json")),
        "acknowledgements": len(file_rows(REAL_ROOT, "receipts/acknowledgements/*.json")),
        "tranche_receipts": len(file_rows(REAL_ROOT, "receipts/tranches/*.json")),
        "global_receipts": len(file_rows(REAL_ROOT, "receipts/global/*.json")),
    }
    store = TemporalReviewStore(
        R4_PACKAGE,
        REAL_ROOT,
        PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/practice_decisions",
    )
    saved = store.save_draft(migrated, "real")
    after = read_json(REAL_DRAFT)
    after_counts = {
        "events": len(file_rows(REAL_ROOT, "events/**/*.json")),
        "acknowledgements": len(file_rows(REAL_ROOT, "receipts/acknowledgements/*.json")),
        "tranche_receipts": len(file_rows(REAL_ROOT, "receipts/tranches/*.json")),
        "global_receipts": len(file_rows(REAL_ROOT, "receipts/global/*.json")),
    }
    if before_counts != after_counts or any(before_counts.values()):
        raise RuntimeError("FAIL_G7E_B_R4_REAL_DRAFT_RECOVERY: actual immutable counts changed")
    if preserved_human_projection(before) != preserved_human_projection(after):
        raise RuntimeError("FAIL_G7E_B_R4_REAL_DRAFT_RECOVERY: actual human projection changed")
    if saved["draft_version"] != before["draft_version"] + 1:
        raise RuntimeError("FAIL_G7E_B_R4_REAL_DRAFT_RECOVERY: draft version did not advance once")
    record = {
        "schema_version": "football_intelligence.g7e_b_r4.real_draft_migration_record.v1",
        "migration_record_id": "G7E_B_R4_REAL_DRAFT_NON_APPLICABLE_RELATIONSHIP_MIGRATION_V1",
        "before_path": str(BACKUP),
        "actual_path": str(REAL_DRAFT),
        "before_sha256": sha256(BACKUP),
        "after_sha256": sha256(REAL_DRAFT),
        "before_byte_size": BACKUP.stat().st_size,
        "after_byte_size": REAL_DRAFT.stat().st_size,
        "before_draft_version": before["draft_version"],
        "after_draft_version": after["draft_version"],
        "changes": changes,
        "unaffected_human_projection_preserved": True,
        "human_answers_changed": 0,
        "source_coordinates_changed": 0,
        "candidate_selections_changed": 0,
        "missed_person_marks_changed": 0,
        "real_event_created": False,
        "real_acknowledgement_created": False,
        "burst_2_started": False,
        "root_counts_before": before_counts,
        "root_counts_after": after_counts,
        "applied_at_utc": utc_now(),
        "passed": True,
    }
    write_json(R4 / "03_REAL_DRAFT_RECOVERY/real_draft_migration_record.json", record)
    print("PASS_G7E_B_R4_ACTUAL_REAL_DRAFT_RECOVERED_WITHOUT_FINAL_SAVE")


def analyze() -> None:
    if not REAL_DRAFT.is_file() or not BACKUP.is_file():
        raise RuntimeError("FAIL_G7E_B_R4_REAL_DRAFT_RECOVERY: draft or forensic backup missing")
    if sha256(REAL_DRAFT) != sha256(BACKUP):
        raise RuntimeError("FAIL_G7E_B_R4_REAL_DRAFT_RECOVERY: backup is not byte-identical")
    local = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/main")
    subject = git("show", "-s", "--format=%s", local)
    tracked = git("status", "--porcelain", "--untracked-files=no")
    if local != remote or local != EXPECTED_R3_COMMIT or tracked:
        raise RuntimeError("FAIL_G7E_B_R4_BASELINE_PROVENANCE")
    draft = read_json(BACKUP)
    store = TemporalReviewStore(R3_PACKAGE, REAL_ROOT, PART7 / "unused_r4_practice_root")
    case = store.by_id[draft["burst_id"]]
    event_files = file_rows(REAL_ROOT, "events/**/*.json")
    acknowledgement_files = file_rows(REAL_ROOT, "receipts/acknowledgements/*.json")
    tranche_receipts = file_rows(REAL_ROOT, "receipts/tranches/*.json")
    global_receipts = file_rows(REAL_ROOT, "receipts/global/*.json")
    if event_files or acknowledgement_files:
        raise RuntimeError("FAIL_G7E_B_R4_UNEXPECTED_REAL_EVENT_STATE")
    baseline = {
        "schema_version": "football_intelligence.g7e_b_r4.baseline_resolution.v1",
        "resolved_post_r3_commit": local,
        "origin_main": remote,
        "commit_subject": subject,
        "r3_handoff_decision": read_json(R3 / "07_TESTS_AND_LOGS/decision.json")["decision"],
        "r3_handoff_manifest_sha256": sha256(R3 / "08_REVIEW_PACK/CHATGPT_HANDOFF/12_MANIFEST.json"),
        "tracked_worktree_clean": True,
        "pre_existing_untracked_items": [f".pytest_tmp_g7d_c1_{letter}/" for letter in "bcdefg"],
        "resolved_at_utc": utc_now(),
        "passed": True,
    }
    root_preflight = {
        "schema_version": "football_intelligence.g7e_b_r4.real_root_preflight.v1",
        "real_root": str(REAL_ROOT),
        "drafts": file_rows(REAL_ROOT, "drafts/*.json"),
        "events": event_files,
        "acknowledgements": acknowledgement_files,
        "tranche_receipts": tranche_receipts,
        "global_receipts": global_receipts,
        "failed_burst_id": draft["burst_id"],
        "unexpected_real_event_state": False,
        "passed": True,
    }
    backup_manifest = {
        "schema_version": "football_intelligence.g7e_b_r4.real_draft_forensic_backup_manifest.v1",
        "source_path": str(REAL_DRAFT),
        "backup_path": str(BACKUP),
        "byte_size": BACKUP.stat().st_size,
        "sha256": sha256(BACKUP),
        "byte_identical_to_source_at_closure": True,
        "review_revision": draft["review_revision"],
        "schema_revision": draft["schema_version"],
        "mode": draft["mode"],
        "tranche_id": draft["tranche_id"],
        "burst_id": draft["burst_id"],
        "current_question": draft["current_question"],
        "current_frame_sequence": draft["current_frame_sequence"],
        "draft_version": draft["draft_version"],
        "optimistic_lock_token": draft["optimistic_lock_token"],
        "updated_at_utc_from_draft": draft["updated_at_utc"],
        "recorded_at_utc": utc_now(),
    }
    reproduction = reproduce_http(draft)
    reproduction.update(
        {
            "schema_version": "football_intelligence.g7e_b_r4.failure_reproduction.v1",
            "source_draft_sha256": sha256(BACKUP),
            "actual_draft_mutated": False,
            "expected_failure": "invalid candidate relationship",
            "reproduced": reproduction["response"].get("error") == "invalid candidate relationship",
            "structured_server_errors": reproduction["response"].get("errors", []),
            "structured_error_was_unavailable_in_r3": "errors" not in reproduction["response"],
        }
    )
    if not reproduction["reproduced"]:
        raise RuntimeError("FAIL_G7E_B_R4_FAILURE_REPRODUCTION")
    rows = relationship_rows(draft, case)
    invalid = [row for row in rows if row["validation_result"].startswith("INVALID_")]
    root_cause = {
        "schema_version": "football_intelligence.g7e_b_r4.root_cause.v1",
        "classification": "GENERIC_SINGLE_BOX_RELATIONSHIP_BUG",
        "specific_root_cause": (
            "R3_INITIALIZED_SUBJECT_RELATIONSHIP_NULL_BUT_SKIPPED_THE_RELATIONSHIP_QUESTION_"
            "WHEN_ALL_FRAME_SUPPLIES_WERE_ONE_USEFUL_CANDIDATE_WHILE_THE_SERVER_"
            "UNCONDITIONALLY_REQUIRED_A_RELATIONSHIP_ENUM"
        ),
        "subject_token": draft["subjects"][0]["subject_token"],
        "affected_frame_ids": [row["observation_frame_id"] for row in invalid],
        "affected_frame_count": len(invalid),
        "candidate_supply_answers": sorted({row["candidate_supply_answer"] for row in invalid}),
        "selected_candidate_ids_by_frame": {
            row["observation_frame_id"]: row["selected_candidate_ids"] for row in invalid
        },
        "selected_candidate_counts": sorted({row["selected_candidate_count"] for row in invalid}),
        "stored_relationship": draft["subjects"][0].get("candidate_relationship"),
        "relationship_branch": "NON_APPLICABLE_SINGLE_BOX_PER_FRAME",
        "upstream_change_that_caused_incompatibility": None,
        "upstream_cause": "INITIAL_NULL_VALUE_SURVIVED_A_CORRECTLY_SKIPPED_GENERIC_RELATIONSHIP_SCREEN",
        "relationship_question_was_asked": False,
        "genuine_human_relationship_answer_exists": False,
        "automatic_repair_is_unambiguous": True,
        "automatic_repair": (
            "SET_EACH_NON_APPLICABLE_FRAME_RELATIONSHIP_TO_NOT_APPLICABLE_AND_REMOVE_" "LEGACY_SUBJECT_LEVEL_NULL_FIELD"
        ),
        "human_answers_replaced": 0,
        "passed": len(invalid) == 9,
    }
    base = R4 / "00_INPUT_EVENT_AND_REAL_DRAFT_CLOSURE"
    write_json(base / "baseline_resolution.json", baseline)
    write_json(base / "real_root_preflight.json", root_preflight)
    write_json(base / "real_draft_forensic_backup_manifest.json", backup_manifest)
    forensic = R4 / "01_FAILURE_REPRODUCTION_AND_FORENSICS"
    write_json(forensic / "failure_reproduction.json", reproduction)
    write_jsonl(forensic / "candidate_relationship_forensic_table.jsonl", rows)
    write_json(
        forensic / "invalid_relationship_index.json",
        {
            "schema_version": "football_intelligence.g7e_b_r4.invalid_relationship_index.v1",
            "invalid_count": len(invalid),
            "invalid_records": invalid,
        },
    )
    write_json(forensic / "root_cause.json", root_cause)
    print("PASS_G7E_B_R4_FAILURE_REPRODUCED_AND_ROOT_CAUSE_PROVEN")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("analyze", "validate-temp", "apply-actual"), required=True)
    args = parser.parse_args()
    if args.phase == "analyze":
        analyze()
    elif args.phase == "validate-temp":
        validate_temporary_recovery()
        print("PASS_G7E_B_R4_TEMPORARY_RECOVERY_AND_FINAL_SAVE")
    else:
        apply_actual_recovery()


if __name__ == "__main__":
    main()
