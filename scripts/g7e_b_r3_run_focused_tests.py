"""Run the bounded R3 and historical temporal-review tests with draft isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PART7 / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
B0 = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1"
R1 = PART7 / "G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_USABILITY_REPAIR_v1"
PRACTICE_DRAFT = B0 / "03_TEMPORAL_REVIEWER/practice_decisions/drafts/g7e_a_118576_01.json"
FORENSIC_BACKUP = STAGE / "00_INPUT_EVENT_AND_DRAFT_CLOSURE/forensic_backups/g7e_a_118576_01.original.json"
R1_PREFLIGHT = R1 / "00_INPUT_AND_EVENT_CLOSURE/event_root_preflight.json"
LOGS = STAGE / "07_TESTS_AND_LOGS"

PYTHON_FILES = [
    "scripts/g7e_b_r3_forensic_and_migrate.py",
    "scripts/g7e_b_r3_build_frame_bound_reviewer.py",
    "scripts/g7e_b_r3_capture_edge_acceptance.py",
    "scripts/g7e_b_r3_finalize_handoff.py",
    "scripts/g7e_b_r3_run_focused_tests.py",
    "src/football_intelligence/temporal_review.py",
    "tests/test_g7e_b_r3_frame_binding_and_atomic_save.py",
    "tests/test_g7e_b_r2_full_temporal_candidate_closure.py",
    "tests/test_g7e_b_temporal_reviewer_and_tranches.py",
]


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
    current = PRACTICE_DRAFT.read_bytes()
    preflight_bytes = R1_PREFLIGHT.read_bytes()
    current_hash = digest(current)
    results: list[dict[str, Any]] = []
    try:
        PRACTICE_DRAFT.write_bytes(FORENSIC_BACKUP.read_bytes())
        results.append(
            run(
                "pytest_g7e_b_r2_full_temporal_candidate_closure",
                ["uv", "run", "pytest", "tests/test_g7e_b_r2_full_temporal_candidate_closure.py", "-q"],
            )
        )
        stub = canonical_bytes({"burst_id": "g7e_a_118576_01", "review_revision": "G7E_B_TEMPORAL_BURST_REVIEW_V1"})
        PRACTICE_DRAFT.write_bytes(stub)
        preflight = json.loads(preflight_bytes)
        preflight["old_practice_draft_count"] = 1
        preflight["old_practice_draft_hashes"] = {str(PRACTICE_DRAFT): digest(stub)}
        R1_PREFLIGHT.write_bytes(canonical_bytes(preflight))
        results.extend(
            [
                run(
                    "pytest_g7e_b_r1_subject_guidance_and_zoom",
                    ["uv", "run", "pytest", "tests/test_g7e_b_r1_subject_guidance_and_zoom.py", "-q"],
                ),
                run(
                    "pytest_g7e_b_temporal_reviewer_and_tranches",
                    ["uv", "run", "pytest", "tests/test_g7e_b_temporal_reviewer_and_tranches.py", "-q"],
                ),
            ]
        )
    finally:
        PRACTICE_DRAFT.write_bytes(current)
        R1_PREFLIGHT.write_bytes(preflight_bytes)
    return (
        results,
        {
            "actual_r3_practice_draft_sha256_before": current_hash,
            "actual_r3_practice_draft_sha256_after": digest(PRACTICE_DRAFT.read_bytes()),
            "r1_preflight_sha256_before": digest(preflight_bytes),
            "r1_preflight_sha256_after": digest(R1_PREFLIGHT.read_bytes()),
            "byte_identical_restoration": PRACTICE_DRAFT.read_bytes() == current
            and R1_PREFLIGHT.read_bytes() == preflight_bytes,
            "historical_drafts_used_only_during_focused_regression": True,
        },
    )


def write_report(phase: str, results: list[dict[str, Any]], preservation: dict[str, Any] | None = None) -> None:
    path = LOGS / "focused_test_report.json"
    prior = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    merged = {row["name"]: row for row in prior.get("results", [])}
    merged.update({row["name"]: row for row in results})
    report = {
        "schema_version": "football_intelligence.g7e_b_r3.focused_test_report.v1",
        "phase": phase,
        "results": [merged[name] for name in sorted(merged)],
        "historical_practice_draft_isolation": preservation or prior.get("historical_practice_draft_isolation"),
        "full_repository_test_suite_run": False,
    }
    report["passed"] = all(row["passed"] for row in report["results"])
    path.write_bytes(canonical_bytes(report))
    if not report["passed"]:
        raise SystemExit("FAIL_G7E_B_R3_FOCUSED_TESTS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pre-handoff", "post-handoff"), required=True)
    args = parser.parse_args()
    if args.phase == "pre-handoff":
        results = [
            run("uv_lock_check", ["uv", "lock", "--check"]),
            run("uv_sync", ["uv", "sync"]),
            run("ruff_check", ["uv", "run", "ruff", "check", *PYTHON_FILES]),
            run("ruff_format_check", ["uv", "run", "ruff", "format", "--check", *PYTHON_FILES]),
            run(
                "node_check_r3_reviewer",
                ["node", "--check", "src/football_intelligence/g7e_b_r2_temporal_review.js"],
            ),
            run("git_diff_check", ["git", "diff", "--check"]),
            run(
                "pytest_g7e_b_r3_pre_handoff",
                [
                    "uv",
                    "run",
                    "pytest",
                    "tests/test_g7e_b_r3_frame_binding_and_atomic_save.py",
                    "-q",
                    "-k",
                    "not edge_acceptance_visual_cap_root_isolation_and_handoff and not scope_and_decision",
                ],
            ),
        ]
        historical, preservation = historical_regressions()
        results.extend(historical)
        write_report(args.phase, results, preservation)
    else:
        results = [
            run(
                "pytest_g7e_b_r3_post_handoff",
                ["uv", "run", "pytest", "tests/test_g7e_b_r3_frame_binding_and_atomic_save.py", "-q"],
            ),
            run("git_diff_check_post_handoff", ["git", "diff", "--check"]),
        ]
        write_report(args.phase, results)
    print("PASS_G7E_B_R3_FOCUSED_TESTS")


if __name__ == "__main__":
    main()
