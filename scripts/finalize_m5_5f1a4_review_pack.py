"""Create and validate the flat M5.5F.1A.4 ChatGPT review pack."""

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
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F1A4_SERVER_PERSISTENCE_CRASH_SAFE_GOLD_ANNOTATION_AND_REANNOTATION_ACCELERATION_v1"
)
PACKAGE = STAGE / "07_CRASH_SAFE_GOLD_ANNOTATION_PACKAGE"
PACK = STAGE / "12_REVIEW_PACK_FOR_CHATGPT"
BROWSER = STAGE / "09_BROWSER_CRASH_RESTART_AND_OFFLINE_TESTS" / "browser_evidence"
BASELINE = "eb250e8d2c5ed226abac86544f5d9d3d27ea0e96"
HEAD = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()

FILES = [
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_SUMMARY.md",
    "02_RUN_AND_GIT_CONTEXT.json",
    "03_FILES_CHANGED.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TEST_RESULTS.md",
    "06_OUTPUT_ARTIFACT_INDEX.json",
    "07_FAILURE_AND_RECOVERABILITY.json",
    "08_EVENT_API_AND_SERVER_MATERIALIZER.json",
    "09_BROWSER_DURABLE_OUTBOX.json",
    "10_HYDRATION_AND_RECONCILIATION.json",
    "11_SEQUENCE_SAVE_AND_COMPLETION_GATES.json",
    "12_CRASH_RESTART_AND_OFFLINE_VALIDATION.json",
    "13_REANNOTATION_ACCELERATION.json",
    "14_PRODUCTION_PERSISTENCE_EXERCISE.json",
    "15_SAFETY_AND_MUTATION_AUDIT.json",
    "16_ACCEPTANCE_AND_NEXT_STAGE.json",
    "17_PERSISTENCE_INSPECTOR_UI.png",
    "18_CRASH_RECOVERY_VALIDATION_VISUAL.jpg",
    "19_HUMAN_REVIEW_INSTRUCTIONS.md",
]


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if PACK.exists():
        raise RuntimeError(f"refusing to overwrite existing pack: {PACK}")
    PACK.mkdir(parents=True)
    browser_result = read_json(STAGE / "09_BROWSER_CRASH_RESTART_AND_OFFLINE_TESTS" / "crash_restart_results.json")
    package_validation = read_json(PACKAGE / "review_package_validation.json")
    full_suite = {
        "passed": False,
        "passed_tests": 818,
        "failed_tests": 1,
        "command": "uv run pytest -q",
        "blocker": "historical M5.5F.1A.2 package is not fresh: its committed test requires an empty decisions map but the pre-existing package contains 24 decisions; prior artifact and test were left untouched",
    }
    changed = subprocess.run(
        ["git", "diff", "--name-status", f"{BASELINE}..HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    diff = subprocess.run(
        ["git", "diff", f"{BASELINE}..HEAD", "--"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    write(
        PACK / "01_EXECUTIVE_SUMMARY.md",
        """# M5.5F.1A.4 Review Handoff

This stage repairs the gold annotation persistence boundary. Seed, frame, pair, stable-run, note, undo and sequence-save actions are represented as durable server events. The browser queues before network transmission in IndexedDB, falls back to localStorage, hydrates from server materialized state, and blocks completion until the server is authoritative and reconciled.

The browser smoke on `http://127.0.0.1:8802/` passed for seed persistence, frame persistence, reload hydration, offline queueing, server restart and queue flush. The new package remains fresh with an empty decisions root; smoke writes used an isolated temporary root.

The implementation is visual-only, match-local, sandbox-only and not production-ready. Full-suite validation has one unrelated historical-fixture failure documented in `07_FAILURE_AND_RECOVERABILITY.json`.
""",
    )
    write(
        PACK / "02_RUN_AND_GIT_CONTEXT.json",
        {
            "stage": "M5.5F.1A.4",
            "baseline": BASELINE,
            "implementation_commit": HEAD,
            "branch": "main",
            "review_url": "http://127.0.0.1:8802/",
            "review_id": "m5_5f1a4_crash_safe_gold_annotation_v1",
            "reviewer_session_id": "m5_5f1a4_crash_safe_gold_annotation_reviewer",
            "package_case_count": 25,
            "annotation_sequence_count": 24,
            "browser_smoke_passed": True,
        },
    )
    write(PACK / "03_FILES_CHANGED.md", changed)
    write(PACK / "04_SOURCE_DIFF.patch", diff)
    write(
        PACK / "05_COMMANDS_AND_TEST_RESULTS.md",
        """# Validation

- `uv lock --check`: passed
- `uv sync`: passed
- `uv run fi-pipeline --help`: passed
- `uv run fi-pipeline review-chassis --help`: passed
- Ruff check: passed
- Ruff format check: passed
- JavaScript syntax check: passed
- `git diff --check`: passed
- focused persistence/review tests: 26 passed
- full suite: 818 passed, 1 failed; historical fixture blocker documented below
- real Edge/CDP 8802 persistence exercise: passed
""",
    )
    package_files = []
    for path in sorted((PACKAGE / "decisions").rglob("*")):
        if path.is_file() and path.name not in {"review_decision_events.jsonl"}:
            package_files.append(
                {"path": path.relative_to(PACKAGE).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)}
            )
    write(
        PACK / "06_OUTPUT_ARTIFACT_INDEX.json",
        {
            "package_validation": package_validation,
            "package_files": package_files,
            "stage_outputs": [
                "01_AUTHORIZATION_AND_FAILURE_INGESTION",
                "02_PERSISTENCE_ROOT_CAUSE",
                "03_SERVER_EVENT_API_AND_MATERIALIZER",
                "04_BROWSER_DURABLE_OUTBOX",
                "05_STATE_HYDRATION_AND_RECONCILIATION",
                "06_SEQUENCE_SAVE_AND_COMPLETION_GATES",
                "08_REANNOTATION_ACCELERATION",
                "09_BROWSER_CRASH_RESTART_AND_OFFLINE_TESTS",
                "10_SCIENTIFIC_INTEGRITY_AND_RECOVERY_VALIDATION",
            ],
        },
    )
    write(
        PACK / "07_FAILURE_AND_RECOVERABILITY.json",
        {
            "prior_browser_export": {
                "approved_polygon": "recoverable",
                "frame_annotations": "not recoverable",
                "indexeddb": "absent",
                "server_events": "absent",
            },
            "root_cause": [
                "client-only draft persistence",
                "seed and frame actions skipped server event append",
                "no durable outbox",
                "no server materialized gold state",
            ],
            "historical_fixture_blocker": full_suite["blocker"],
            "new_stage_decisions_root_empty": True,
        },
    )
    write(
        PACK / "08_EVENT_API_AND_SERVER_MATERIALIZER.json",
        {
            "event_route": "/api/review/gold-event",
            "completion_route": "/api/review/gold-complete",
            "event_types": [
                "SEED_CONFIRMED",
                "SEED_SWAPPED",
                "SEED_CORRECTED",
                "SEED_REJECTED",
                "FRAME_STATE_SET",
                "PAIR_ACCEPTED",
                "STABLE_RUN_ACCEPTED",
                "MANUAL_BBOX_SET",
                "NOTE_UPDATED",
                "UNDO",
                "SEQUENCE_SAVED",
                "REVIEW_COMPLETED",
            ],
            "idempotency": True,
            "atomic_append": True,
            "replay_materialization": True,
            "hash_chain": True,
            "monotonic_server_sequence": True,
        },
    )
    write(
        PACK / "09_BROWSER_DURABLE_OUTBOX.json",
        {
            "primary": "IndexedDB",
            "fallback": "localStorage",
            "enqueue_before_network": True,
            "retain_until_ack": True,
            "retry_on_reconnect": True,
            "offline_status": "Offline — queued locally",
            "duplicate_retry_observed": True,
        },
    )
    write(
        PACK / "10_HYDRATION_AND_RECONCILIATION.json",
        {
            "server_authoritative": True,
            "hydrate_on_load": True,
            "reload_test_passed": True,
            "restart_test_passed": True,
            "divergence_blocks": True,
            "pending_count_after_recovery": 0,
        },
    )
    write(
        PACK / "11_SEQUENCE_SAVE_AND_COMPLETION_GATES.json",
        {
            "explicit_frame_events": True,
            "seed_required": True,
            "sequence_save_after_flush": True,
            "navigation_after_ack": True,
            "completion_route_server_authoritative": True,
            "all_24_sequences_required": True,
            "outbox_empty_required": True,
            "evidence_clear_required": True,
            "polygon_approved_required": True,
        },
    )
    write(PACK / "12_CRASH_RESTART_AND_OFFLINE_VALIDATION.json", browser_result)
    write(
        PACK / "13_REANNOTATION_ACCELERATION.json",
        {
            "stable_run_preview": True,
            "contact_strip": True,
            "next_unannotated": True,
            "next_uncertain": True,
            "keyboard_pair_acceptance": True,
            "no_auto_accept": True,
            "notes_optional": True,
        },
    )
    write(
        PACK / "14_PRODUCTION_PERSISTENCE_EXERCISE.json",
        {
            "browser_smoke_passed": True,
            "seed_event_acknowledged": True,
            "frame_event_acknowledged": True,
            "reload_hydrated": True,
            "server_restart_preserved": True,
            "offline_queue_flushed": True,
            "duplicate_event_not_reappended": True,
            "new_package_decisions_root_empty": True,
        },
    )
    write(
        PACK / "15_SAFETY_AND_MUTATION_AUDIT.json",
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
            "detector_rerun": False,
            "historical_artifacts_mutated": False,
            "prior_stage_mutated": False,
            "raw_video_included_in_pack": False,
            "weights_included_in_pack": False,
            "sealed_mapping_included_in_pack": False,
        },
    )
    write(
        PACK / "16_ACCEPTANCE_AND_NEXT_STAGE.json",
        {
            "classification": "BLOCKED_HISTORICAL_FIXTURE_MUTATION",
            "implementation_ready": True,
            "new_package_browser_ready": True,
            "exact_blocker": full_suite["blocker"],
            "next_stage": "Use an immutable empty-decision fixture or temporary decisions root in the historical M5.5F.1A.2 regression test, without changing its scientific assertions; then rerun the full suite.",
            "no_tracker_promoted": True,
        },
    )
    shutil.copy2(BROWSER / "17_PERSISTENCE_INSPECTOR_UI.png", PACK / "17_PERSISTENCE_INSPECTOR_UI.png")
    shutil.copy2(BROWSER / "18_CRASH_RECOVERY_VALIDATION_VISUAL.jpg", PACK / "18_CRASH_RECOVERY_VALIDATION_VISUAL.jpg")
    write(
        PACK / "19_HUMAN_REVIEW_INSTRUCTIONS.md",
        """# Human Action

1. The prior frame annotations are not recoverable; only the approved polygon was recoverable.
2. Do not use port 8801. Use port 8802 only after confirming the stage is accepted.
3. Verify the header says `Saved to server` after the first seed or frame action.
4. Reload after the first frame and confirm the annotation remains.
5. Only then reannotate with stable-run acceleration; use the contact strip and next-unannotated controls, but inspect each frame.
6. Stop if the server event sequence does not increase.
7. Notes are optional.
8. No tracker is promoted by this stage.
""",
    )
    manifest = {
        "schema_version": "m5_5f1a4.review_pack.v1",
        "review_pack_name": "M5.5F.1A.4 ChatGPT handoff",
        "implementation_commit": HEAD,
        "files": FILES,
        "file_count": len(FILES),
        "visual_files": ["17_PERSISTENCE_INSPECTOR_UI.png", "18_CRASH_RECOVERY_VALIDATION_VISUAL.jpg"],
        "max_files": 20,
        "max_total_bytes": 50 * 1024 * 1024,
        "sealed_mapping_excluded": True,
        "raw_video_excluded": True,
        "model_weights_excluded": True,
        "answer_keys_excluded": True,
        "personal_data_excluded": True,
        "classification": "BLOCKED_HISTORICAL_FIXTURE_MUTATION",
    }
    write(PACK / "REVIEW_PACK_MANIFEST.json", manifest)
    actual = sorted(path.name for path in PACK.iterdir() if path.is_file())
    total = sum(path.stat().st_size for path in PACK.iterdir() if path.is_file())
    if actual != sorted(FILES):
        raise RuntimeError(f"pack file set mismatch: {actual}")
    if total > 50 * 1024 * 1024:
        raise RuntimeError("review pack exceeds 50 MiB")
    write(
        STAGE / "stage_summary.json",
        {
            "classification": "BLOCKED_HISTORICAL_FIXTURE_MUTATION",
            "implementation_commit": HEAD,
            "browser_validation_passed": True,
            "package_validation_passed": bool(package_validation.get("passed")),
            "review_pack_validated": True,
            "full_suite_passed": False,
            "exact_blocker": full_suite["blocker"],
            "review_url": "http://127.0.0.1:8802/",
            "new_package_decisions_root_empty": True,
            "no_tracker_promoted": True,
        },
    )
    write(STAGE / "10_SCIENTIFIC_INTEGRITY_AND_RECOVERY_VALIDATION" / "browser_validation_summary.json", browser_result)
    write(
        STAGE / "11_COMMANDS_AND_TESTS" / "final_validation.json",
        {
            "focused_tests": {"passed": True, "count": 26},
            "full_suite": full_suite,
            "lock_check": True,
            "sync": True,
            "cli_help": True,
            "ruff": True,
            "javascript_syntax": True,
            "review_pack": {"passed": True, "file_count": len(actual)},
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "file_count": len(actual),
                "total_bytes": total,
                "aggregate_hash": stable_hash([{"path": name, "sha256": sha256_file(PACK / name)} for name in actual]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
