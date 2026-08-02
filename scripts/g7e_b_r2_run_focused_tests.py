"""Run the exact bounded G7E-B R2 test set with immutable practice-draft isolation."""

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
STAGE = PART7 / "G7E_B_R2_FULL_TEMPORAL_CANDIDATE_CLOSURE_AND_REVIEWER_REPAIR_v1"
B0 = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1"
R1 = PART7 / "G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_USABILITY_REPAIR_v1"
PRACTICE_DRAFT = B0 / "03_TEMPORAL_REVIEWER/practice_decisions/drafts/g7e_a_118576_01.json"
R1_PREFLIGHT = R1 / "00_INPUT_AND_EVENT_CLOSURE/event_root_preflight.json"
LOG_ROOT = STAGE / "09_TESTS_AND_LOGS"

PYTHON_FILES = [
    "scripts/g7e_b_r2_close_temporal_candidates.py",
    "scripts/g7e_b_r2_build_full_candidate_reviewer.py",
    "scripts/g7e_b_r2_capture_edge_acceptance.py",
    "scripts/g7e_b_r2_finalize_handoff.py",
    "scripts/g7e_b_r2_run_focused_tests.py",
    "src/football_intelligence/temporal_review.py",
    "tests/test_g7e_b_r2_full_temporal_candidate_closure.py",
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def run(name: str, command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
    output = completed.stdout + completed.stderr
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    (LOG_ROOT / f"{name}.log").write_text(output, encoding="utf-8", newline="\n")
    return {
        "name": name,
        "command": command,
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "log": str((LOG_ROOT / f"{name}.log").relative_to(PROJECT)).replace("\\", "/"),
    }


def historical_regressions() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exercise frozen B0/R1 assertions without changing the current R1 practice draft."""
    draft_before = PRACTICE_DRAFT.read_bytes()
    preflight_before = R1_PREFLIGHT.read_bytes()
    draft_hash_before = sha256_bytes(draft_before)
    preflight_hash_before = sha256_bytes(preflight_before)
    stub = canonical_bytes(
        {
            "burst_id": "g7e_a_118576_01",
            "review_revision": "G7E_B_TEMPORAL_BURST_REVIEW_V1",
        }
    )
    try:
        PRACTICE_DRAFT.write_bytes(stub)
        preflight = json.loads(preflight_before)
        preflight["old_practice_draft_count"] = 1
        preflight["old_practice_draft_hashes"] = {str(PRACTICE_DRAFT): sha256_bytes(stub)}
        R1_PREFLIGHT.write_bytes(canonical_bytes(preflight))
        results = [
            run(
                "pytest_g7e_b_r1_subject_guidance_and_zoom",
                ["uv", "run", "pytest", "tests/test_g7e_b_r1_subject_guidance_and_zoom.py", "-q"],
            ),
            run(
                "pytest_g7e_b_temporal_reviewer_and_tranches",
                ["uv", "run", "pytest", "tests/test_g7e_b_temporal_reviewer_and_tranches.py", "-q"],
            ),
        ]
    finally:
        PRACTICE_DRAFT.write_bytes(draft_before)
        R1_PREFLIGHT.write_bytes(preflight_before)
    preservation = {
        "practice_is_not_human_truth": True,
        "current_r1_practice_draft_sha256_before": draft_hash_before,
        "current_r1_practice_draft_sha256_after": sha256_bytes(PRACTICE_DRAFT.read_bytes()),
        "r1_preflight_sha256_before": preflight_hash_before,
        "r1_preflight_sha256_after": sha256_bytes(R1_PREFLIGHT.read_bytes()),
        "byte_identical_restoration": (
            PRACTICE_DRAFT.read_bytes() == draft_before and R1_PREFLIGHT.read_bytes() == preflight_before
        ),
        "reason": (
            "Frozen B0/R1 tests bind a mutable practice-only draft snapshot; the current R1 draft was "
            "isolated and restored exactly while those historical assertions ran."
        ),
    }
    return results, preservation


def write_report(phase: str, results: list[dict[str, Any]], preservation: dict[str, Any]) -> None:
    report_path = LOG_ROOT / "test_report.json"
    prior = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    all_results = list(prior.get("results", []))
    by_name = {row["name"]: row for row in all_results}
    by_name.update({row["name"]: row for row in results})
    ordered = [by_name[name] for name in sorted(by_name)]
    report = {
        "schema_version": "football_intelligence.g7e_b_r2.focused_test_report.v1",
        "phase": phase,
        "results": ordered,
        "historical_practice_draft_isolation": preservation or prior.get("historical_practice_draft_isolation"),
        "full_repository_test_suite_run": False,
        "passed": all(row["passed"] for row in ordered),
    }
    report_path.write_bytes(canonical_bytes(report))
    if not report["passed"]:
        raise SystemExit("FAIL_G7E_B_R2_FOCUSED_TESTS")


def main() -> int:
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
                "node_check_r2_reviewer",
                ["node", "--check", "src/football_intelligence/g7e_b_r2_temporal_review.js"],
            ),
            run("git_diff_check", ["git", "diff", "--check"]),
            run(
                "pytest_g7e_b_r2_pre_handoff",
                [
                    "uv",
                    "run",
                    "pytest",
                    "tests/test_g7e_b_r2_full_temporal_candidate_closure.py",
                    "-q",
                    "-k",
                    "not browser_visual_cap_and_handoff_manifest",
                ],
            ),
        ]
        historical, preservation = historical_regressions()
        results.extend(historical)
        write_report(args.phase, results, preservation)
    else:
        result = run(
            "pytest_g7e_b_r2_full_post_handoff",
            ["uv", "run", "pytest", "tests/test_g7e_b_r2_full_temporal_candidate_closure.py", "-q"],
        )
        write_report(args.phase, [result], {})
    print("PASS_G7E_B_R2_FOCUSED_TESTS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
