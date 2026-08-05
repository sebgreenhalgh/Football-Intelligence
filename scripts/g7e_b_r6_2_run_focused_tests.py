"""Run and record the bounded R6.2 and inherited reviewer test matrix."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
from uuid import uuid4

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
STAGE = PROJECT / (
    "experiments/football_observation_reasoner/part 8/" "G7E_B_R6_2_PRECISION_ZOOM_PAN_AND_COORDINATE_SAFE_MARKING_v1"
)
OUTPUT = STAGE / "09_FOCUSED_REGRESSION_TESTS/focused_test_report.json"

PYTEST_PATHS = (
    "tests/test_g7e_b_r6_2_precision_navigation.py",
    "tests/test_g7e_b_r6_1_final_byte_runtime.py",
    "tests/test_g7e_b_r6_server_authoritative_action_reducer.py",
    "tests/test_g7e_b_r5_reviewer_state_machine_and_release.py",
    "tests/test_g7e_b_r4_candidate_relationship_integrity.py",
    "tests/test_g7e_b_r3_frame_binding_and_atomic_save.py",
    "tests/test_g7e_b_r2_full_temporal_candidate_closure.py",
    "tests/test_g7e_b_r1_subject_guidance_and_zoom.py",
    "tests/test_g7e_b_temporal_reviewer_and_tranches.py",
    "tests/unit/core/test_fingerprints.py",
    "tests/unit/core/test_path_roots.py",
)
RUFF_PATHS = (
    "src/football_intelligence/temporal_reviewer",
    "src/football_intelligence/temporal_review.py",
    "src/football_intelligence/g7e_b_r6_action_reducer.py",
    "scripts/g7e_b_r6_capture_edge_acceptance.py",
    "scripts/g7e_b_r6_2_build_precision_reviewer.py",
    "scripts/g7e_b_r6_2_capture_edge_acceptance.py",
    "scripts/g7e_b_r6_2_finalize_release.py",
    "scripts/g7e_b_r6_2_reproduce_navigation_limit.py",
    "scripts/g7e_b_r6_2_run_focused_tests.py",
    "scripts/g7e_b_r6_2_verify_real_resume.py",
    "tests/test_g7e_b_r6_2_precision_navigation.py",
    "tests/test_g7e_b_r6_1_final_byte_runtime.py",
    "tests/test_g7e_b_r6_server_authoritative_action_reducer.py",
)


def run(label: str, command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
    record = {
        "label": label,
        "command": command,
        "exit_code": result.returncode,
        "stdout_tail": result.stdout[-8_000:],
        "stderr_tail": result.stderr[-8_000:],
    }
    if result.returncode:
        raise RuntimeError(f"{label} failed: {record}")
    return record


def main() -> None:
    python = str(Path(sys.executable).resolve())
    ruff = str((REPO / ".venv/Scripts/ruff.exe").resolve()) if sys.platform == "win32" else "ruff"
    uv = shutil.which("uv") or "uv"
    pytest_base = PROJECT / f"_r62_pytest_{uuid4().hex}"
    records = [
        run(
            "r6_2_and_inherited_pytest",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(pytest_base),
                *PYTEST_PATHS,
            ],
        ),
        run("reviewer_ruff_check", [ruff, "check", *RUFF_PATHS]),
        run("reviewer_ruff_format_check", [ruff, "format", "--check", *RUFF_PATHS]),
        run("browser_javascript_syntax", ["node", "--check", "src/football_intelligence/g7e_b_r6_temporal_review.js"]),
        run("viewport_javascript_syntax", ["node", "--check", "src/football_intelligence/g7e_b_r6_2_viewport.js"]),
        run("dependency_lock", [uv, "lock", "--check"]),
        run("repository_data_boundary", [python, "scripts/check_repository_data_boundaries.py"]),
        run("git_diff_check", ["git", "diff", "--check", "HEAD"]),
    ]
    shutil.rmtree(pytest_base, ignore_errors=True)
    if "96 passed" not in records[0]["stdout_tail"]:
        raise RuntimeError(f"focused pytest count was not the frozen 96-test matrix: {records[0]['stdout_tail']}")
    document = {
        "schema_version": "football_intelligence.g7e_b_r6_2.focused_tests.v1",
        "classification": "PASS_G7E_B_R6_2_FOCUSED_AND_INHERITED_TESTS",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "pytest_expected_count": 96,
        "pytest_paths": list(PYTEST_PATHS),
        "checks": records,
        "intentionally_excluded": [
            "GPU and neural-inference suites",
            "tests requiring validation, holdout, or unrelated private media",
            "the unrelated full repository suite",
        ],
        "production_ready": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(document["classification"])


if __name__ == "__main__":
    main()
