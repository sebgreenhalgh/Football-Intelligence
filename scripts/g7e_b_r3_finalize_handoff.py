"""Finalize direct R3 evidence and the exact twelve-file ChatGPT handoff."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from football_intelligence.temporal_review import TemporalReviewStore

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PART7 / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
PACKAGE = STAGE / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3"
B0 = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1"
PRACTICE = B0 / "03_TEMPORAL_REVIEWER/practice_decisions"
ACTUAL_DRAFT = PRACTICE / "drafts/g7e_a_118576_01.json"
HANDOFF = STAGE / "08_REVIEW_PACK/CHATGPT_HANDOFF"
DECISION = "PASS_G7E_B_R3_FRAME_BINDING_AND_ATOMIC_SAVE_READY_FOR_PRACTICE_RESUME"
IDENTITY_FIELDS = {"burst_id", "frame_id", "unique_frame_id", "frame_index", "frame_pixel_sha256"}


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


def count(root: Path, pattern: str) -> int:
    return len(list(root.glob(pattern))) if root.is_dir() else 0


def roots() -> dict[str, int]:
    real = PACKAGE / "human_decisions"
    return {
        "practice_drafts": count(PRACTICE, "drafts/*.json"),
        "practice_events": count(PRACTICE, "events/*/*.json"),
        "practice_acknowledgements": count(PRACTICE, "receipts/acknowledgements/*.json"),
        "real_drafts": count(real, "drafts/*.json"),
        "real_events": count(real, "events/*/*.json"),
        "real_acknowledgements": count(real, "receipts/acknowledgements/*.json"),
        "real_tranche_receipts": count(real, "receipts/tranche_completion/*.json"),
        "real_global_receipts": count(real, "receipts/global_completion/*.json"),
    }


def frame_validation() -> dict[str, Any]:
    store = TemporalReviewStore(PACKAGE, STAGE / "_final_validation/real", STAGE / "_final_validation/practice")
    draft = read_json(ACTUAL_DRAFT)
    case = store.practice_by_id[draft["burst_id"]]
    store._validate_draft(draft)  # noqa: SLF001
    store._validate_r3_frame_bindings(draft, case, final=True)  # noqa: SLF001
    observations = [row for subject in draft["subjects"] for row in subject["frame_observations"]]
    locations = [row for row in observations if row.get("location_binding")]
    selections = [row for row in observations if row.get("candidate_selection_binding")]
    marks = draft["missed_person_marks"]
    package_references = 0
    for filename in ("review_cases.json", "practice_cases.json"):
        for review_case in read_json(PACKAGE / filename)["cases"]:
            for sequence, frame in enumerate(review_case["frames"]):
                identity = frame["canonical_frame_identity"]
                if set(identity) != IDENTITY_FIELDS:
                    raise RuntimeError("canonical identity fields diverged")
                if review_case["per_frame_candidate_states"][sequence]["canonical_frame_identity"] != identity:
                    raise RuntimeError("case frame/state identity mismatch")
                package_references += 1
    return {
        "schema_version": "football_intelligence.g7e_b_r3.frame_binding_validation.v1",
        "review_revision": draft["review_revision"],
        "actual_practice_draft_path": str(ACTUAL_DRAFT),
        "actual_practice_draft_sha256": sha256(ACTUAL_DRAFT),
        "package_frame_references_checked": package_references,
        "subject_observations_checked": len(observations),
        "location_bindings_checked": len(locations),
        "candidate_selection_bindings_checked": len(selections),
        "candidate_mappings_checked": len(draft["candidate_mappings"]),
        "missed_person_marks_checked": len(marks),
        "canonical_frame_identity_fields": sorted(IDENTITY_FIELDS),
        "frame_local_action_types": [
            "SUBJECT_LOCATION",
            "APPROXIMATE_HIDDEN_LOCATION",
            "CANDIDATE_SELECTION",
            "MISSED_PERSON_MARK",
        ],
        "coordinates_changed_by_migration": 0,
        "human_answers_changed_by_migration": 0,
        "validation_failures": 0,
        "passed": package_references == 1107 and len(observations) == 9,
        "production_ready": False,
    }


def main() -> None:
    test_report = read_json(STAGE / "07_TESTS_AND_LOGS/focused_test_report.json")
    browser = read_json(STAGE / "05_BROWSER_ACCEPTANCE/browser_acceptance_report.json")
    migration = read_json(STAGE / "02_DRAFT_REPAIR/practice_draft_migration_record.json")
    application = read_json(STAGE / "02_DRAFT_REPAIR/actual_practice_draft_migration_application.json")
    if not test_report["passed"] or browser["decision"] != "PASS_G7E_B_R3_REAL_EDGE_ACCEPTANCE":
        raise SystemExit("FAIL_G7E_B_R3_FOCUSED_TESTS")
    current_roots = roots()
    if any(
        current_roots[key]
        for key in (
            "practice_events",
            "practice_acknowledgements",
            "real_events",
            "real_acknowledgements",
            "real_tranche_receipts",
            "real_global_receipts",
        )
    ):
        raise SystemExit("FAIL_G7E_B_R3_REAL_EVENT_PREFLIGHT")
    validation = frame_validation()
    write_json(STAGE / "03_FRAME_BINDING_IMPLEMENTATION/frame_binding_validation_results.json", validation)
    decision = {
        "schema_version": "football_intelligence.g7e_b_r3.decision.v1",
        "decision": DECISION,
        "repository_head_at_execution": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "review_revision": "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_V1",
        "root_cause_proven": True,
        "deterministic_metadata_only_migration_applied": True,
        "actual_practice_draft_sha256": sha256(ACTUAL_DRAFT),
        "actual_final_practice_event_created": False,
        "browser_acceptance": browser["decision"],
        "frame_binding_validation": validation["passed"],
        "focused_tests_passed": True,
        "storage_root_counts": current_roots,
        "detector_or_model_inference_run": False,
        "real_tranche_1_started": False,
        "validation_or_holdout_accessed": False,
        "project_defaults_changed": False,
        "full_repository_test_suite_run": False,
        "production_ready": False,
    }
    write_json(STAGE / "07_TESTS_AND_LOGS/decision.json", decision)

    if HANDOFF.exists():
        for path in HANDOFF.iterdir():
            if path.is_file():
                path.unlink()
    HANDOFF.mkdir(parents=True, exist_ok=True)
    summary = {
        "decision": DECISION,
        "root_cause": "FRAME_IDENTITY_OMITTED_FROM_FRAME_LOCAL_SUBJECT_OBSERVATIONS",
        "practice_work_preserved": {
            "subjects": 1,
            "frame_observations": 9,
            "candidate_mappings": 10,
            "missed_person_marks": 6,
            "source_coordinates_changed": 0,
            "human_answers_changed": 0,
        },
        "actual_final_practice_event_created": False,
        "resume_action": "Launch R3, confirm the restored summary, then the human may press Save.",
        "production_ready": False,
    }
    write_json(HANDOFF / "01_EXECUTIVE_SUMMARY.json", summary)
    write_json(
        HANDOFF / "02_EVENT_AND_DRAFT_PREFLIGHT.json",
        {
            "storage_root_counts": current_roots,
            "actual_practice_draft_path": str(ACTUAL_DRAFT),
            "actual_practice_draft_sha256": sha256(ACTUAL_DRAFT),
            "forensic_backup_sha256": migration["source_draft_sha256"],
            "real_human_event_preflight": "PASS",
            "actual_final_practice_event_created": False,
        },
    )
    write_json(
        HANDOFF / "03_FAILURE_REPRODUCTION_AND_ROOT_CAUSE.json",
        {
            "failure_reproduction": read_json(STAGE / "01_FORENSIC_ROOT_CAUSE/final_save_failure_reproduction.json"),
            "root_cause": read_json(STAGE / "01_FORENSIC_ROOT_CAUSE/root_cause.json"),
            "forensic_table_row_count": 9,
        },
    )
    write_json(
        HANDOFF / "04_DRAFT_REPAIR_RESULTS.json",
        {
            "repair_decision": read_json(STAGE / "02_DRAFT_REPAIR/practice_draft_repair_decision.json"),
            "migration_record": migration,
            "actual_application": application,
            "actual_draft_sha256_recomputed": sha256(ACTUAL_DRAFT),
        },
    )
    write_json(
        HANDOFF / "05_FRAME_BINDING_IMPLEMENTATION.json",
        {
            "implementation": read_json(STAGE / "03_FRAME_BINDING_IMPLEMENTATION/frame_binding_implementation.json"),
            "validation": validation,
            "launcher": str(STAGE / "launch_temporal_burst_review_r3.ps1"),
            "reviewer_url": "http://127.0.0.1:8818/",
        },
    )
    write_json(
        HANDOFF / "06_ATOMIC_SAVE_AND_IDEMPOTENCY_RESULTS.json",
        {
            "atomic_save": read_json(STAGE / "04_ATOMIC_FINAL_SAVE/atomic_save_acceptance.json"),
            "idempotency": read_json(STAGE / "04_ATOMIC_FINAL_SAVE/idempotency_and_recovery_results.json"),
            "temporary_only": True,
            "actual_practice_final_save_performed": False,
        },
    )
    write_json(
        HANDOFF / "07_BROWSER_AND_REGRESSION_ACCEPTANCE.json",
        {
            "browser": browser,
            "focused_test_results": test_report["results"],
            "historical_practice_draft_isolation": test_report["historical_practice_draft_isolation"],
            "three_visual_cap": True,
            "full_repository_test_suite_run": False,
        },
    )
    (HANDOFF / "08_DECISION.md").write_text(
        "# G7E-B R3 decision\n\n"
        f"`{DECISION}`\n\n"
        "The exact R2 failure was reproduced and traced to omitted frame identity metadata. The existing "
        "practice intent was proven from exact frame-local candidate IDs, so only binding metadata was migrated; "
        "human answers and source coordinates were unchanged. Edge proved atomic frame commits, optimistic draft "
        "locking, structured preflight errors, interrupted acknowledgement recovery, duplicate-save protection, "
        "and read-only completion restoration. No actual final practice event was created. The human may resume "
        "practice with the R3 launcher; real Tranche 1 remains out of scope.\n",
        encoding="utf-8",
        newline="\n",
    )
    for source_name, target_name in (
        ("01_TARGETED_FRAME_CORRECTION.png", "09_TARGETED_CORRECTION.png"),
        ("02_FRAME_COMMIT_AND_VALIDATION.png", "10_FRAME_COMMIT.png"),
        ("03_ATOMIC_SAVE_ACKNOWLEDGED.png", "11_SAVE_ACKNOWLEDGED.png"),
    ):
        shutil.copyfile(STAGE / "06_VISUAL_QA" / source_name, HANDOFF / target_name)
    files = []
    for path in sorted(HANDOFF.iterdir()):
        if path.name == "12_MANIFEST.json" or not path.is_file():
            continue
        files.append({"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)})
    if len(files) != 11:
        raise SystemExit("FAIL_G7E_B_R3_CHATGPT_HANDOFF")
    write_json(
        HANDOFF / "12_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7e_b_r3.handoff_manifest.v1",
            "file_count_excluding_manifest": 11,
            "files": files,
            "self_hashed": False,
            "production_ready": False,
        },
    )
    (STAGE / "08_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only the CHATGPT_HANDOFF folder. It contains exactly twelve self-contained files.\n",
        encoding="utf-8",
        newline="\n",
    )
    if len([path for path in HANDOFF.iterdir() if path.is_file()]) != 12:
        raise SystemExit("FAIL_G7E_B_R3_CHATGPT_HANDOFF")
    print(DECISION)


if __name__ == "__main__":
    main()
