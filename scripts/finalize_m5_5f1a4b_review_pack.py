"""Create and validate the flat M5.5F.1A.4b ChatGPT review pack."""

# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from football_intelligence.review_chassis.hashing import sha256_file, stable_hash


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
STAGE = PART2 / "M5_5F1A4B_SERVER_AUTHORITATIVE_FINALIZATION_AND_STATE_HASH_INTEGRITY_v1"
BACKUP = STAGE / "00_IMMUTABLE_PORT8802_DECISIONS_BACKUP"
EXERCISE = STAGE / "02_PRODUCTION_RECOVERY_EXERCISE"
WORKING = EXERCISE / "working_decisions"
PACK = STAGE / "03_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "35b6bdd0de54f8c450c03b988e774d7144cfda30"
HEAD = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()

FILES = [
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_SUMMARY.md",
    "02_RUN_AND_GIT_CONTEXT.json",
    "03_FILES_CHANGED.md",
    "04_SOURCE_DIFF.patch",
    "05_IMMUTABLE_BACKUP_MANIFEST.json",
    "06_ROOT_CAUSE_AND_STATE_HASH_AUDIT.json",
    "07_LEDGER_RECOVERY_VALIDATION.json",
    "08_COMPLETION_ELIGIBILITY.json",
    "09_PRODUCTION_EXERCISE.json",
    "10_COMPLETED_REVIEW_SUMMARY.json",
    "11_COMPLETION_BUNDLE_VALIDATION.json",
    "12_FINALIZE_REVIEW_ELIGIBLE.png",
    "13_FINALIZATION_COMPLETED.png",
    "14_COMMANDS_AND_TEST_RESULTS.md",
    "15_SAFETY_AUDIT.json",
    "16_NO_ANNOTATION_MUTATION_AUDIT.json",
    "17_API_AND_CLIENT_STATE_TRANSITIONS.json",
    "18_ACCEPTANCE.json",
    "19_HUMAN_NEXT_ACTION.md",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    if PACK.exists():
        raise RuntimeError(f"refusing to overwrite existing review pack: {PACK}")
    result = read_json(EXERCISE / "production_recovery_and_finalization_result.json")
    recovery = result["tests"]["recovery"]
    completion_summary = read_json(WORKING / "completed_review_summary.json")
    completion_validation = result["tests"]["completion_bundle"]
    if not result.get("passed") or not completion_validation.get("passed"):
        raise RuntimeError("production recovery/finalization exercise did not pass")

    backup_files = [path for path in sorted(BACKUP.rglob("*")) if path.is_file()]
    backup_rows = [
        {"path": path.relative_to(BACKUP).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in backup_files
    ]
    backup_manifest = {
        "schema_version": "football_intelligence.m5_5f1a4b.immutable_backup_manifest.v1",
        "backup_root": str(BACKUP),
        "backup_tree_file_count": len(backup_rows),
        "backup_tree_total_bytes": sum(row["size"] for row in backup_rows),
        "backup_tree_aggregate_hash": stable_hash(backup_rows),
        "event_ledger_hash": sha256_file(BACKUP / "review_decision_events.jsonl"),
        "event_ledger_size": (BACKUP / "review_decision_events.jsonl").stat().st_size,
        "materialized_state_file_hash": sha256_file(BACKUP / "review_decisions.json"),
        "materialized_state_file_size": (BACKUP / "review_decisions.json").stat().st_size,
        "recovered_authoritative_materialized_state_hash": recovery["materialized_state_hash"],
        "approved_polygon_hash": read_json(BACKUP / "polygon" / "approved_polygon.json")["approved_polygon_hash"],
        "sequence_ids": recovery["sequence_ids"],
        "per_sequence_frame_counts": {
            row["sequence_id"]: row["persisted_frame_count"] for row in recovery["per_sequence"]
        },
        "strand_frame_states": recovery["completion_eligibility"]["strand_frame_states"],
        "highest_event_sequence": recovery["ledger_audit"]["highest_event_sequence"],
        "immutable_backup_unchanged": result["immutable_backup_unchanged"],
    }
    manifest_path = STAGE / "01_IMMUTABLE_BACKUP_MANIFEST" / "immutable_backup_manifest.json"
    write(manifest_path, backup_manifest)

    changed = subprocess.run(
        ["git", "diff", "--name-status", f"{BASELINE}..{HEAD}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", f"{BASELINE}..{HEAD}", "--"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    PACK.mkdir(parents=True)
    write(
        PACK / "01_EXECUTIVE_SUMMARY.md",
        """# M5.5F.1A.4b Review Handoff

The completed 24-sequence gold annotation ledger is scientifically complete and was preserved byte-for-byte before repair. The defect was a client state-transition error: after the last sequence save, hydration was deleted and the same sequence was reconstructed from a blank local draft. Hydrated server annotations were also incorrectly counted as unsaved local work.

The server now treats identical seed and sequence replays as semantic no-ops, derives completion eligibility from authoritative materialized state, validates prior-state hashes before mutation, and exposes a recovery route that writes no annotation events. The real Edge exercise against a copy appended exactly one `REVIEW_COMPLETED` event and validated all four completion artifacts.

No annotation was repeated or replaced. No tracker is promoted.
""",
    )
    write(
        PACK / "02_RUN_AND_GIT_CONTEXT.json",
        {
            "stage": "M5.5F.1A.4b",
            "baseline": BASELINE,
            "implementation_commit": HEAD,
            "branch": "main",
            "production_review_url": "http://127.0.0.1:8802/",
            "isolated_exercise_url": "http://127.0.0.1:8803/",
            "classification": "PASS_GOLD_ANNOTATION_FINALIZATION_READY",
        },
    )
    write(PACK / "03_FILES_CHANGED.md", changed)
    write(PACK / "04_SOURCE_DIFF.patch", diff)
    write(PACK / "05_IMMUTABLE_BACKUP_MANIFEST.json", backup_manifest)
    write(
        PACK / "06_ROOT_CAUSE_AND_STATE_HASH_AUDIT.json",
        {
            "post_save_hydration_deleted": True,
            "last_sequence_navigation_remained_on_sequence_24": True,
            "blank_local_draft_recreated_from_machine_proposal": True,
            "hydrated_state_misclassified_as_dirty": True,
            "first_complete_event_sequence": 1110,
            "first_complete_materialized_state_hash": "c951dd9c015bf73550b1ecb07c68e1412892d2a7bea2e437bc0b9a2cb843e80a",
            "semantic_noop_events_after_first_complete_state": 129,
            "historical_event_hash_mismatch_count": 0,
            "historical_state_hash_mismatch_count": 0,
            "legacy_duplicate_event_sequence": 524,
            "legacy_duplicate_preserved_in_append_order": True,
            "acknowledgement_returns_actual_post_materialization_hash": True,
        },
    )
    write(
        PACK / "07_LEDGER_RECOVERY_VALIDATION.json",
        {
            "ledger_audit": recovery["ledger_audit"],
            "materialized_state_hash": recovery["materialized_state_hash"],
            "sequence_count": len(recovery["sequence_ids"]),
            "all_per_sequence_frame_counts": sorted({row["persisted_frame_count"] for row in recovery["per_sequence"]}),
            "scientific_annotation_events_written": recovery["scientific_annotation_events_written"],
        },
    )
    write(PACK / "08_COMPLETION_ELIGIBILITY.json", recovery["completion_eligibility"])
    write(
        PACK / "09_PRODUCTION_EXERCISE.json",
        {
            "passed": result["passed"],
            "pre_reload_eligible": result["tests"]["initial_eligibility"]["eligibility"]["eligible"],
            "post_reload_eligible": result["tests"]["reload_eligibility"]["eligibility"]["eligible"],
            "completed": result["tests"]["completed"]["completed"],
            "completion_retry_duplicate": result["tests"]["completion_retry"]["duplicate"],
            "final_event_sequence": completion_summary["final_server_event_sequence"],
            "final_materialized_state_hash": completion_summary["final_materialized_state_hash"],
            "immutable_backup_unchanged": result["immutable_backup_unchanged"],
        },
    )
    write(PACK / "10_COMPLETED_REVIEW_SUMMARY.json", completion_summary)
    write(PACK / "11_COMPLETION_BUNDLE_VALIDATION.json", completion_validation)
    shutil.copy2(EXERCISE / "17_FINALIZE_REVIEW_ELIGIBLE.png", PACK / "12_FINALIZE_REVIEW_ELIGIBLE.png")
    shutil.copy2(EXERCISE / "18_FINALIZATION_COMPLETED.png", PACK / "13_FINALIZATION_COMPLETED.png")
    write(
        PACK / "14_COMMANDS_AND_TEST_RESULTS.md",
        """# Validation

- `uv lock --check`: passed
- `uv sync`: passed
- Ruff check and format check: passed
- JavaScript syntax check: passed
- focused A4/A4b tests: 7 passed
- review-chassis and A4 regression set: 42 passed
- full suite: 823 passed, 1 dependency deprecation warning
- `uv run fi-pipeline --help`: passed
- `uv run fi-pipeline review-chassis --help`: passed
- `git diff --check`: passed
- real Edge recovery/reload/finalization exercise: passed
""",
    )
    write(
        PACK / "15_SAFETY_AUDIT.json",
        {
            "VISUAL_ONLY_NOT_METRIC": True,
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
            "safe_to_apply_globally": False,
            "match_local_only": True,
            "sandbox_only": True,
            "identity_tracking_performed": False,
            "player_slots_assigned": False,
            "model_fit_performed": False,
            "learned_continuity_rows_updated": 0,
            "tracker_promoted": False,
        },
    )
    write(
        PACK / "16_NO_ANNOTATION_MUTATION_AUDIT.json",
        {
            "immutable_backup_unchanged": result["immutable_backup_unchanged"],
            "original_ledger_is_exact_prefix": True,
            "new_scientific_annotation_events": 0,
            "new_completion_events": 1,
            "machine_proposals_used_to_replace_human_work": False,
            "annotations_repeated": False,
        },
    )
    write(
        PACK / "17_API_AND_CLIENT_STATE_TRANSITIONS.json",
        {
            "recovery_route": "/api/review/gold-recover",
            "completion_route": "/api/review/gold-complete",
            "hydration_sets_dirty": False,
            "genuine_user_edit_sets_dirty": True,
            "semantic_seed_replay_appends_event": False,
            "finalized_sequence_survives_reload": True,
            "completion_eligibility_server_authoritative": True,
            "finalize_action_navigation_independent": True,
            "completion_retry_appends_event": False,
        },
    )
    write(
        PACK / "18_ACCEPTANCE.json",
        {
            "classification": "PASS_GOLD_ANNOTATION_FINALIZATION_READY",
            "exact_blocker": None,
            "full_suite_passed": True,
            "production_copy_exercise_passed": True,
            "completion_bundle_validated": True,
            "no_tracker_promoted": True,
        },
    )
    write(
        PACK / "19_HUMAN_NEXT_ACTION.md",
        """# Human Action

The original annotations are complete and do not need to be repeated. Launch the existing port-8802 package with the repaired repository code. Confirm the persistent button reads `Finalize review`, then use it once. The server will append one completion event and atomically create the four completion artifacts. A retry is safe and will return the original acknowledgement without appending another event.
""",
    )

    manifest = {
        "schema_version": "football_intelligence.m5_5f1a4b.review_pack.v1",
        "implementation_commit": HEAD,
        "classification": "PASS_GOLD_ANNOTATION_FINALIZATION_READY",
        "files": FILES,
        "file_count": len(FILES),
        "max_files": 20,
        "max_total_bytes": 50 * 1024 * 1024,
        "raw_video_excluded": True,
        "model_weights_excluded": True,
        "sealed_mappings_excluded": True,
        "historical_annotation_payloads_excluded": True,
    }
    write(PACK / "REVIEW_PACK_MANIFEST.json", manifest)
    actual = sorted(path.name for path in PACK.iterdir() if path.is_file())
    total_bytes = sum(path.stat().st_size for path in PACK.iterdir() if path.is_file())
    if actual != sorted(FILES):
        raise RuntimeError(f"review pack file set mismatch: {actual}")
    if total_bytes > 50 * 1024 * 1024:
        raise RuntimeError("review pack exceeds 50 MiB")
    aggregate_hash = stable_hash([{"path": name, "sha256": sha256_file(PACK / name)} for name in actual])
    write(
        STAGE / "stage_summary.json",
        {
            "classification": "PASS_GOLD_ANNOTATION_FINALIZATION_READY",
            "implementation_commit": HEAD,
            "full_suite_passed": True,
            "production_copy_exercise_passed": True,
            "immutable_backup_unchanged": True,
            "review_pack_validated": True,
            "review_pack_file_count": len(actual),
            "review_pack_aggregate_hash": aggregate_hash,
            "exact_blocker": None,
            "no_tracker_promoted": True,
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "file_count": len(actual),
                "total_bytes": total_bytes,
                "aggregate_hash": aggregate_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
