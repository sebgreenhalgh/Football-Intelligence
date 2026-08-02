"""Create the exact self-contained twelve-file G7E-B R4 handoff."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PART7 / "G7E_B_R4_CANDIDATE_RELATIONSHIP_BRANCH_INTEGRITY_AND_REAL_DRAFT_RECOVERY_v1"
R3 = PART7 / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
REAL_ROOT = R3 / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
REAL_DRAFT = REAL_ROOT / "drafts/g7e_a_117093_10.json"
HANDOFF = STAGE / "08_REVIEW_PACK/CHATGPT_HANDOFF"
DECISION = "PASS_G7E_B_R4_CANDIDATE_RELATIONSHIP_INTEGRITY_READY_FOR_REAL_DRAFT_RESUME"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def immutable_counts(root: Path) -> dict[str, int]:
    return {
        "events": len(list(root.glob("events/*/*.json"))),
        "acknowledgements": len(list(root.glob("receipts/acknowledgements/*.json"))),
        "tranche_receipts": len(list(root.glob("receipts/tranche_completion/*.json"))),
        "global_receipts": len(list(root.glob("receipts/global_completion/*.json"))),
    }


def main() -> None:
    tests = read_json(STAGE / "07_TESTS_AND_LOGS/focused_test_report.json")
    browser = read_json(STAGE / "05_BROWSER_ACCEPTANCE/browser_acceptance_report.json")
    migration = read_json(STAGE / "03_REAL_DRAFT_RECOVERY/real_draft_migration_record.json")
    if not tests["passed"] or browser["decision"] != "PASS_G7E_B_R4_REAL_EDGE_ACCEPTANCE":
        raise SystemExit("FAIL_G7E_B_R4_CHATGPT_HANDOFF: acceptance gate is incomplete")
    counts = immutable_counts(REAL_ROOT)
    if any(counts.values()) or migration["burst_2_started"] is not False:
        raise SystemExit("FAIL_G7E_B_R4_CHATGPT_HANDOFF: real human root advanced")
    if sha256(REAL_DRAFT) != migration["after_sha256"]:
        raise SystemExit("FAIL_G7E_B_R4_CHATGPT_HANDOFF: migrated real draft hash mismatch")

    HANDOFF.mkdir(parents=True, exist_ok=True)
    for path in HANDOFF.iterdir():
        if path.is_file():
            path.unlink()
    baseline = read_json(STAGE / "00_INPUT_EVENT_AND_REAL_DRAFT_CLOSURE/baseline_resolution.json")
    preflight = read_json(STAGE / "00_INPUT_EVENT_AND_REAL_DRAFT_CLOSURE/real_root_preflight.json")
    backup = read_json(STAGE / "00_INPUT_EVENT_AND_REAL_DRAFT_CLOSURE/real_draft_forensic_backup_manifest.json")
    failure = read_json(STAGE / "01_FAILURE_REPRODUCTION_AND_FORENSICS/failure_reproduction.json")
    root_cause = read_json(STAGE / "01_FAILURE_REPRODUCTION_AND_FORENSICS/root_cause.json")
    invalid = read_json(STAGE / "01_FAILURE_REPRODUCTION_AND_FORENSICS/invalid_relationship_index.json")
    matrix = read_json(STAGE / "02_BRANCH_COMPATIBILITY_ENGINE/relationship_compatibility_matrix.json")
    invalidation = read_json(STAGE / "02_BRANCH_COMPATIBILITY_ENGINE/branch_invalidation_results.json")
    implementation = read_json(STAGE / "02_BRANCH_COMPATIBILITY_ENGINE/relationship_engine_implementation.json")
    recovery = read_json(STAGE / "03_REAL_DRAFT_RECOVERY/real_draft_repair_decision.json")
    final_save = read_json(STAGE / "04_FINAL_SAVE_AND_TARGETED_CORRECTION/final_save_preflight_results.json")
    local_head = git("rev-parse", "HEAD")
    origin_main = git("rev-parse", "origin/main")
    executive = {
        "schema_version": "football_intelligence.g7e_b_r4.executive_summary.v1",
        "decision": DECISION,
        "root_cause": root_cause["classification"],
        "real_burst_id": "g7e_a_117093_10",
        "real_draft_status": "RECOVERED_AT_SUMMARY_AWAITING_USER_FINAL_SAVE",
        "real_draft_version": migration["after_draft_version"],
        "real_draft_sha256": migration["after_sha256"],
        "human_answers_changed": 0,
        "source_coordinates_changed": 0,
        "candidate_selections_changed": 0,
        "actual_immutable_counts": counts,
        "burst_2_started": False,
        "browser_acceptance": browser["decision"],
        "focused_tests_passed": True,
        "local_head_at_packaging": local_head,
        "origin_main_at_packaging": origin_main,
        "production_ready": False,
    }
    write_json(HANDOFF / "01_EXECUTIVE_SUMMARY.json", executive)
    write_json(
        HANDOFF / "02_BASELINE_EVENT_AND_REAL_DRAFT_PREFLIGHT.json",
        {
            "schema_version": "football_intelligence.g7e_b_r4.handoff.baseline_and_preflight.v1",
            "baseline_resolution": baseline,
            "real_root_preflight": preflight,
            "forensic_backup_manifest": backup,
            "actual_post_migration_immutable_counts": counts,
        },
    )
    write_json(
        HANDOFF / "03_FAILURE_REPRODUCTION_AND_ROOT_CAUSE.json",
        {
            "schema_version": "football_intelligence.g7e_b_r4.handoff.failure_and_root_cause.v1",
            "failure_reproduction": failure,
            "root_cause": root_cause,
            "invalid_relationship_index": invalid,
        },
    )
    write_json(
        HANDOFF / "04_BRANCH_COMPATIBILITY_AND_INVALIDATION.json",
        {
            "schema_version": "football_intelligence.g7e_b_r4.handoff.branch_compatibility.v1",
            "canonical_matrix": matrix,
            "implementation": implementation,
            "branch_invalidation_results": invalidation,
            "edge_canonical_branch_results": browser["canonical_branch_results"],
            "edge_upstream_invalidation": browser["upstream_invalidation"],
            "edge_refresh_restoration": browser["refresh_restoration"],
        },
    )
    write_json(
        HANDOFF / "05_REAL_DRAFT_RECOVERY_RESULTS.json",
        {
            "schema_version": "football_intelligence.g7e_b_r4.handoff.real_draft_recovery.v1",
            "repair_decision": recovery,
            "migration_record": migration,
            "actual_immutable_counts": counts,
            "actual_draft_hash_verified": True,
            "user_must_press_final_save": True,
        },
    )
    write_json(
        HANDOFF / "06_FINAL_SAVE_AND_TARGETED_CORRECTION.json",
        {
            "schema_version": "football_intelligence.g7e_b_r4.handoff.final_save.v1",
            "temporary_final_save": final_save,
            "all_error_preflight": browser["exact_stale_relationship_reproduction"],
            "acknowledged_event_read_only_reload": browser["acknowledged_event_read_only_reload"],
            "actual_final_save_pressed_by_codex": False,
            "targeted_human_correction_required": False,
        },
    )
    write_json(
        HANDOFF / "07_BROWSER_AND_REGRESSION_ACCEPTANCE.json",
        {
            "schema_version": "football_intelligence.g7e_b_r4.handoff.browser_and_regression.v1",
            "browser_acceptance": browser,
            "focused_test_report": tests,
            "full_repository_test_suite_run": False,
            "detector_or_temporal_inference_run": False,
            "validation_or_holdout_accessed": False,
            "project_defaults_changed": False,
        },
    )
    decision_text = (
        f"# {DECISION}\n\n"
        "The R3 server required a generic subject-level relationship even though every frame had exactly one useful "
        "candidate and the relationship question was correctly skipped. The nine non-applicable frame fields are now "
        "canonically `NOT_APPLICABLE`; the skipped continuity field is likewise marked `NOT_APPLICABLE`. No genuine "
        "human answer, coordinate, candidate selection, role, participation, certainty, or missed-person mark "
        "changed.\n\n"
        "Launch `launch_temporal_burst_review_r4.ps1`, choose real review, inspect the restored summary for "
        "`g7e_a_117093_10`, and press **Save burst** yourself. Codex did not create a real event or begin Burst 2.\n"
    )
    (HANDOFF / "08_DECISION.md").write_text(decision_text, encoding="utf-8", newline="\n")
    for source_name, destination_name in (
        ("01_AFFECTED_FRAME_AND_SELECTED_BOXES.png", "09_AFFECTED_FRAME.png"),
        ("02_BRANCH_SPECIFIC_RELATIONSHIP.png", "10_BRANCH_SPECIFIC_QUESTION.png"),
        ("03_REAL_DRAFT_RECOVERED_READY_TO_RESUME.png", "11_REAL_DRAFT_READY.png"),
    ):
        shutil.copyfile(STAGE / "06_VISUAL_QA" / source_name, HANDOFF / destination_name)
    rows = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(HANDOFF.iterdir())
        if path.is_file() and path.name != "12_MANIFEST.json"
    ]
    write_json(
        HANDOFF / "12_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7e_b_r4.handoff_manifest.v1",
            "file_count_excluding_manifest": len(rows),
            "files": rows,
            "manifest_self_hashed": False,
        },
    )
    upload = STAGE / "08_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt"
    upload.write_text(
        "Upload only the CHATGPT_HANDOFF folder. It contains exactly 12 self-contained files.\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        STAGE / "decision.json",
        {
            "schema_version": "football_intelligence.g7e_b_r4.decision.v1",
            "decision": DECISION,
            "real_draft_sha256": migration["after_sha256"],
            "real_event_created": False,
            "burst_2_started": False,
            "production_ready": False,
        },
    )
    if len(list(HANDOFF.iterdir())) != 12 or len(rows) != 11:
        raise SystemExit("FAIL_G7E_B_R4_CHATGPT_HANDOFF: file count mismatch")
    print(DECISION)


if __name__ == "__main__":
    main()
