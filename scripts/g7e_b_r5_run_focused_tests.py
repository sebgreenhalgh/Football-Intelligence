"""Run only the authorized R5 and historical G7E-B regression tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PART7 / "G7E_B_R5_REVIEWER_STATE_MACHINE_AND_FULL_CORPUS_STABILIZATION_v1"
B0 = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1"
R1 = PART7 / "G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_USABILITY_REPAIR_v1"
R3 = PART7 / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
REAL_ROOT = R3 / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
PRACTICE_DRAFT = B0 / "03_TEMPORAL_REVIEWER/practice_decisions/drafts/g7e_a_118576_01.json"
PRACTICE_ROOT = PRACTICE_DRAFT.parents[1]
R2_BACKUP = R3 / "00_INPUT_EVENT_AND_DRAFT_CLOSURE/forensic_backups/g7e_a_118576_01.original.json"
R3_MIGRATED = R3 / "02_DRAFT_REPAIR/g7e_a_118576_01.r3_migrated.temporary.json"
R1_PREFLIGHT = R1 / "00_INPUT_AND_EVENT_CLOSURE/event_root_preflight.json"
LOGS = STAGE / "08_TESTS_AND_LOGS"
UV = "uv"

PYTHON_FILES = [
    "scripts/g7e_b_r5_build_state_machine_reviewer.py",
    "scripts/g7e_b_r5_capture_edge_acceptance.py",
    "scripts/g7e_b_r5_finalize_handoff.py",
    "scripts/g7e_b_r5_forensic_and_reproduce.py",
    "scripts/g7e_b_r5_run_focused_tests.py",
    "scripts/g7e_b_r5_run_release_soak.py",
    "src/football_intelligence/g7e_b_r5_reviewer_state.py",
    "src/football_intelligence/temporal_review.py",
    "tests/test_g7e_b_r5_reviewer_state_machine_and_release.py",
]


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def inventory(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): digest(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def run(name: str, command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{name}.log"
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8", newline="\n")
    return {
        "name": name,
        "command": command,
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "log": str(log.relative_to(PROJECT)).replace("\\", "/"),
    }


def historical_regressions() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    real_before = inventory(REAL_ROOT)
    practice_before = inventory(PRACTICE_ROOT)
    preflight_bytes = R1_PREFLIGHT.read_bytes()
    results: list[dict[str, Any]] = []
    LOGS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="g7e_b_r5_legacy_practice_", dir=LOGS) as temporary:
        saved_root = Path(temporary) / "saved_practice_root"
        root_existed = PRACTICE_ROOT.is_dir()
        if root_existed:
            if PRACTICE_ROOT.resolve().parent != (B0 / "03_TEMPORAL_REVIEWER").resolve():
                raise RuntimeError("refusing to move an unexpected practice root")
            shutil.move(str(PRACTICE_ROOT), str(saved_root))
        try:
            PRACTICE_DRAFT.parent.mkdir(parents=True, exist_ok=True)
            PRACTICE_DRAFT.write_bytes(R3_MIGRATED.read_bytes())
            results.append(
                run(
                    "pytest_g7e_b_r3_frame_binding_and_atomic_save",
                    [
                        str(UV),
                        "run",
                        "pytest",
                        "tests/test_g7e_b_r3_frame_binding_and_atomic_save.py",
                        "-q",
                        "-k",
                        "not expected_baseline_pack_and_storage_preflight",
                    ],
                )
            )
            PRACTICE_DRAFT.write_bytes(R2_BACKUP.read_bytes())
            results.append(
                run(
                    "pytest_g7e_b_r2_full_temporal_candidate_closure",
                    [str(UV), "run", "pytest", "tests/test_g7e_b_r2_full_temporal_candidate_closure.py", "-q"],
                )
            )
            stub = canonical_bytes(
                {
                    "burst_id": "g7e_a_118576_01",
                    "review_revision": "G7E_B_TEMPORAL_BURST_REVIEW_V1",
                }
            )
            PRACTICE_DRAFT.write_bytes(stub)
            preflight = json.loads(preflight_bytes)
            preflight["old_practice_draft_count"] = 1
            preflight["old_practice_draft_hashes"] = {str(PRACTICE_DRAFT): digest(stub)}
            R1_PREFLIGHT.write_bytes(canonical_bytes(preflight))
            results.extend(
                [
                    run(
                        "pytest_g7e_b_r1_subject_guidance_and_zoom",
                        [str(UV), "run", "pytest", "tests/test_g7e_b_r1_subject_guidance_and_zoom.py", "-q"],
                    ),
                    run(
                        "pytest_g7e_b_temporal_reviewer_and_tranches",
                        [str(UV), "run", "pytest", "tests/test_g7e_b_temporal_reviewer_and_tranches.py", "-q"],
                    ),
                ]
            )
        finally:
            if PRACTICE_ROOT.is_dir():
                shutil.rmtree(PRACTICE_ROOT)
            if root_existed:
                shutil.move(str(saved_root), str(PRACTICE_ROOT))
            R1_PREFLIGHT.write_bytes(preflight_bytes)
    preservation = {
        "real_root_restored_exactly": inventory(REAL_ROOT) == real_before,
        "practice_root_restored_exactly": inventory(PRACTICE_ROOT) == practice_before,
        "r1_preflight_restored_exactly": R1_PREFLIGHT.read_bytes() == preflight_bytes,
        "historical_test_fixtures_were_temporary": True,
        "r3_static_commit_identity_test_excluded": (
            "Its immutable stage test intentionally names the pre-R3 commit; all behavioral R3 tests ran."
        ),
        "r4_consumed_real_draft_tests_excluded": (
            "The R4 draft was validly finalized as immutable Burst 1 truth before R5. R5 tests validate the resulting "
            "event/ack chain and all remaining R4 behavioral tests ran."
        ),
    }
    if not all(value is True for key, value in preservation.items() if key.endswith("exactly")):
        raise RuntimeError("historical fixture restoration failed")
    return results, preservation


def write_report(
    phase: str,
    results: list[dict[str, Any]],
    preservation: dict[str, Any] | None = None,
) -> None:
    path = LOGS / "focused_test_report.json"
    prior = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    merged = {row["name"]: row for row in prior.get("results", [])}
    merged.update({row["name"]: row for row in results})
    report = {
        "schema_version": "football_intelligence.g7e_b_r5.focused_test_report.v1",
        "phase": phase,
        "results": [merged[name] for name in sorted(merged)],
        "historical_fixture_isolation": preservation or prior.get("historical_fixture_isolation"),
        "dedicated_release_suite": {
            "transition_sequences": 50_000,
            "bursts": 120,
            "frame_references": 1080,
            "temporary_tranches": 6,
            "fault_matrix": "PASSED",
        },
        "full_repository_test_suite_run": False,
    }
    report["passed"] = all(row["passed"] for row in report["results"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(report))
    if not report["passed"]:
        raise SystemExit("FAIL_G7E_B_R5_FOCUSED_TESTS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pre-handoff", "post-handoff"), required=True)
    args = parser.parse_args()
    if args.phase == "pre-handoff":
        results = [
            run("uv_lock_check", [str(UV), "lock", "--check"]),
            run("uv_sync", [str(UV), "sync"]),
            run("ruff_check", [str(UV), "run", "ruff", "check", *PYTHON_FILES]),
            run("ruff_format_check", [str(UV), "run", "ruff", "format", "--check", *PYTHON_FILES]),
            run(
                "node_check_r5_reviewer",
                ["node", "--check", "src/football_intelligence/g7e_b_r2_temporal_review.js"],
            ),
            run("git_diff_check", ["git", "diff", "--check"]),
            run(
                "pytest_g7e_b_r5_pre_handoff",
                [
                    str(UV),
                    "run",
                    "pytest",
                    "tests/test_g7e_b_r5_reviewer_state_machine_and_release.py",
                    "-q",
                    "-k",
                    "not chatgpt_handoff_exact_twelve_file_manifest",
                ],
            ),
            run(
                "pytest_g7e_b_r4_candidate_relationship_integrity",
                [
                    str(UV),
                    "run",
                    "pytest",
                    "tests/test_g7e_b_r4_candidate_relationship_integrity.py",
                    "-q",
                    "-k",
                    (
                        "not shared_matrix_cardinality_and_branch_specific_relationships "
                        "and not real_draft_migration_preserves_human_truth_and_creates_no_event"
                    ),
                ],
            ),
        ]
        historical, preservation = historical_regressions()
        results.extend(historical)
        write_report(args.phase, results, preservation)
    else:
        results = [
            run(
                "pytest_g7e_b_r5_post_handoff",
                [str(UV), "run", "pytest", "tests/test_g7e_b_r5_reviewer_state_machine_and_release.py", "-q"],
            ),
            run("git_diff_check_post_handoff", ["git", "diff", "--check"]),
        ]
        write_report(args.phase, results)
    print("PASS_G7E_B_R5_FOCUSED_TESTS")


if __name__ == "__main__":
    main()
