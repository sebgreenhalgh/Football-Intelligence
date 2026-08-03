"""Run only the contracted G7E-B R6 release checks and retain their output."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
R6 = (
    PROJECT / "experiments/football_observation_reasoner/part 7/"
    "G7E_B_R6_SERVER_AUTHORITATIVE_ACTION_REDUCER_AND_EXACT_BRANCH_REPAIR_v1"
)
PACKAGE_JS = R6 / "03_SERVER_AUTHORITATIVE_ACTION_REDUCER/temporal_reviewer_r6/review.js"

PYTHON_FILES = [
    "src/football_intelligence/g7e_b_r5_reviewer_state.py",
    "src/football_intelligence/temporal_review.py",
    "src/football_intelligence/g7e_b_r6_action_reducer.py",
    "scripts/g7e_b_r6_build_server_action_reviewer.py",
    "scripts/g7e_b_r6_capture_edge_acceptance.py",
    "scripts/g7e_b_r6_forensic_and_migrate.py",
    "scripts/g7e_b_r6_finalize_release.py",
    "scripts/g7e_b_r6_run_focused_tests.py",
    "tests/test_g7e_b_r6_server_authoritative_action_reducer.py",
    "tests/test_g7e_b_r5_reviewer_state_machine_and_release.py",
    "tests/test_g7e_b_r4_candidate_relationship_integrity.py",
    "tests/test_g7e_b_r3_frame_binding_and_atomic_save.py",
    "tests/test_g7e_b_r2_full_temporal_candidate_closure.py",
    "tests/test_g7e_b_r1_subject_guidance_and_zoom.py",
    "tests/test_g7e_b_temporal_reviewer_and_tranches.py",
]

TEST_FILES = [
    "tests/test_g7e_b_r6_server_authoritative_action_reducer.py",
    "tests/test_g7e_b_r5_reviewer_state_machine_and_release.py",
    "tests/test_g7e_b_r4_candidate_relationship_integrity.py",
    "tests/test_g7e_b_r3_frame_binding_and_atomic_save.py",
    "tests/test_g7e_b_r2_full_temporal_candidate_closure.py",
    "tests/test_g7e_b_r1_subject_guidance_and_zoom.py",
    "tests/test_g7e_b_temporal_reviewer_and_tranches.py",
]


def run(label: str, command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
    row: dict[str, object] = {
        "label": label,
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
    }
    if completed.returncode:
        raise RuntimeError(json.dumps(row, indent=2))
    return row


def main() -> None:
    checks = [
        run("uv_lock_check", ["uv", "lock", "--check"]),
        run("uv_sync", ["uv", "sync"]),
        run("ruff_check", ["uv", "run", "ruff", "check", *PYTHON_FILES]),
        run("ruff_format_check", ["uv", "run", "ruff", "format", "--check", *PYTHON_FILES]),
        run("node_check_source", ["node", "--check", "src/football_intelligence/g7e_b_r6_temporal_review.js"]),
        run("node_check_package", ["node", "--check", str(PACKAGE_JS)]),
    ]
    for test_file in TEST_FILES:
        checks.append(run(f"pytest:{test_file}", ["uv", "run", "pytest", test_file, "-q"]))
    checks.append(run("git_diff_check", ["git", "diff", "--check"]))
    passed_tests = sum(
        int(str(row["stdout"]).split(" passed", maxsplit=1)[0].rsplit(maxsplit=1)[-1])
        for row in checks
        if str(row["label"]).startswith("pytest:")
    )
    result = {
        "schema_version": "football_intelligence.g7e_b_r6.focused_test_results.v1",
        "classification": "PASS_G7E_B_R6_FOCUSED_TESTS",
        "check_count": len(checks),
        "pytest_file_count": len(TEST_FILES),
        "pytest_test_count": passed_tests,
        "all_passed": all(bool(row["passed"]) for row in checks),
        "full_repository_suite_run": False,
        "checks": checks,
        "production_ready": False,
    }
    target = R6 / "09_TESTS_AND_LOGS/focused_test_results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"PASS_G7E_B_R6_FOCUSED_TESTS {passed_tests}")


if __name__ == "__main__":
    main()
