"""Run G7F-C R2 navigation loading-state and R1 regression acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydantic

from football_intelligence.dense_person_reviewer import DensePersonHTTPServer, DensePersonReviewerConfig
from football_intelligence.gold_eval.core import inventory_tree
from g7f_c_r1_run_acceptance import run_edge_acceptance as run_r1_edge_acceptance
from g7f_c_r1_run_acceptance import verify_frozen


PASS = "PASS_G7F_C_R2_REVIEWER_NAVIGATION_LOADING_STATE_REPAIR_READY_FOR_CALIBRATION_RESUME"
RELEASE = "G7F_C_DENSE_PERSON_REVIEWER_R2"
BASELINE = "b842dbd2836240b3d8b04c0ff137e920a7440eca"
FI_PYTHON = Path(r"C:\Users\sebgr\anaconda3\envs\fi-reviewer\python.exe")
BASE_SITE_PACKAGES = Path(r"C:\Users\sebgr\anaconda3\Lib\site-packages")


def paths(repo: Path) -> dict[str, Path]:
    part9 = repo.parent / "experiments/football_observation_reasoner/part 9"
    return {
        "repo": repo,
        "source": part9 / "G7F_C_DENSE_PERSON_GOLD_DISCRIMINATION_SET_v1",
        "repair": part9 / "G7F_C_R2_REVIEWER_NAVIGATION_LOADING_STATE_REPAIR_v1",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def original_bindings(source: Path) -> dict[str, str]:
    return read_json(source / "05_REVIEWER/reviewer_binding_hashes.json")


def reviewer_config(all_paths: dict[str, Path], decisions_root: Path, *, port: int = 0) -> DensePersonReviewerConfig:
    source = all_paths["source"]
    return DensePersonReviewerConfig(
        selection_manifest_path=source / "01_SELECTION/dense_gold_selection_manifest.json",
        assets_root=source / "05_REVIEWER/assets",
        decisions_root=decisions_root,
        binding_hashes=original_bindings(source),
        reviewer_release=RELEASE,
        reveal_payload_path=source / "05_REVIEWER/sealed_candidate_reveal_payloads.json",
        pass_kind="FIRST_PASS",
        allowed_selection_statuses=("CALIBRATION_ONLY",),
        port=port,
    )


def run_focused_tests(all_paths: dict[str, Path], temp_root: Path) -> dict[str, Any]:
    args = [
        "tests/test_g7f_c_r2_reviewer_navigation.py",
        "tests/test_g7f_c_r1_reviewer_interaction.py",
        "tests/test_g7f_c_dense_person_gold.py",
        "-q",
        f"--basetemp={temp_root / 'pytest_temp'}",
        "-o",
        f"cache_dir={temp_root / 'pytest_cache'}",
    ]
    code = (
        "import json,sys; "
        f"sys.path.append({str(BASE_SITE_PACKAGES)!r}); "
        "import pytest; "
        f"raise SystemExit(pytest.main(json.loads({json.dumps(json.dumps(args))})))"
    )
    result = subprocess.run(
        [str(FI_PYTHON), "-c", code],
        cwd=all_paths["repo"],
        env={**os.environ, "PYTHONPATH": str(all_paths["repo"] / "src")},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"focused tests failed:\n{result.stdout}\n{result.stderr}")
    return {"exit_code": result.returncode, "stdout": result.stdout.strip(), "interpreter": str(FI_PYTHON)}


def run_navigation_edge_acceptance(all_paths: dict[str, Path], temp_root: Path) -> dict[str, Any]:
    source_decisions = all_paths["source"] / "06_DENSE_DECISIONS"
    decisions = temp_root / "navigation_decisions"
    copied = []
    for relative in (
        "events/first_pass__DG-001.json",
        "acknowledgements/first_pass__DG-001.json",
    ):
        source_path = source_decisions / relative
        target_path = decisions / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)
        copied.append({"relative_path": relative, "sha256": sha256_file(target_path)})
    before = inventory_tree(decisions, include_files=True)
    server = DensePersonHTTPServer(reviewer_config(all_paths, decisions, port=0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    try:
        result = subprocess.run(
            [
                "node",
                str(all_paths["repo"] / "scripts/g7f_c_r2_edge_navigation_acceptance.js"),
                f"http://127.0.0.1:{port}/",
            ],
            cwd=all_paths["repo"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"R2 live Edge navigation acceptance failed:\n{result.stdout}\n{result.stderr}")
        payload = json.loads(result.stdout)
        if not payload.get("passed"):
            raise RuntimeError("R2 live Edge navigation acceptance did not pass")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    after = inventory_tree(decisions, include_files=True)
    if before != after:
        raise RuntimeError("R2 navigation browser test mutated its seeded finalized event/ack root")
    payload["seeded_finalized_dg001_files"] = copied
    payload["temporary_decisions_byte_identical"] = True
    return payload


def decision_categories(inventory: dict[str, Any]) -> dict[str, int]:
    categories = Counter(row["relative_path"].split("/", 1)[0] for row in inventory["files"])
    return {
        "acknowledgements": categories["acknowledgements"],
        "action_receipts": categories["action_receipts"],
        "drafts": categories["drafts"],
        "events": categories["events"],
        "reveal_logs": categories["reveal_logs"],
    }


def real_decision_integrity(all_paths: dict[str, Path]) -> dict[str, Any]:
    before = read_json(all_paths["repair"] / "00_PRECHANGE_REAL_DECISIONS_INVENTORY.json")
    real_root = all_paths["source"] / "06_DENSE_DECISIONS"
    after = inventory_tree(real_root, include_files=True)
    summary_matches = all(
        (
            before["file_count"] == after["file_count"],
            before["total_bytes"] == after["total_bytes"],
            before["ordered_inventory_sha256"] == after["ordered_inventory_sha256"],
        )
    )
    if not summary_matches:
        raise RuntimeError(f"real decision tree changed: before={before}, after={after}")
    after_files = {row["relative_path"]: row for row in after["files"]}
    for expected in before["critical_live_files"]:
        actual = after_files.get(expected["relative_path"])
        if actual is None or actual["byte_size"] != expected["byte_size"] or actual["sha256"] != expected["sha256"]:
            raise RuntimeError(f"critical real decision changed: {expected['relative_path']}")

    dg001_event = real_root / "events/first_pass__DG-001.json"
    dg001_ack = real_root / "acknowledgements/first_pass__DG-001.json"
    event_before, ack_before = dg001_event.read_bytes(), dg001_ack.read_bytes()
    server = DensePersonHTTPServer(reviewer_config(all_paths, real_root))
    try:
        state = server.store.state("DG-001", "FIRST_PASS")
    finally:
        server.server_close()
    if dg001_event.read_bytes() != event_before or dg001_ack.read_bytes() != ack_before:
        raise RuntimeError("loading finalized DG-001 changed event or acknowledgement bytes")
    if not state["finalized"] or not state["read_only"]:
        raise RuntimeError("DG-001 did not load as finalized/read-only")
    return {
        "schema_version": "football_intelligence.g7f_c_r2.real_decisions_integrity.v1",
        "before": before,
        "after": after,
        "after_categories": decision_categories(after),
        "byte_identical": True,
        "dg001_finalized": True,
        "dg001_read_only": True,
        "dg001_event_and_ack_unchanged_after_load": True,
        "calibration_work_reset": False,
        "production_ready": False,
    }


def build_release(all_paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, str]]:
    repo = all_paths["repo"]
    repair = all_paths["repair"]
    launcher = repair / "05_REVIEWER/launch_calibration_reviewer_r2.ps1"
    requirements = repair / "05_REVIEWER/REQUIREMENTS.md"
    release_files = [
        repo / "src/football_intelligence/dense_person_gold.py",
        repo / "src/football_intelligence/dense_person_reviewer.py",
        repo / "src/football_intelligence/dense_person_reviewer_static/index.html",
        repo / "src/football_intelligence/dense_person_reviewer_static/app.js",
        repo / "src/football_intelligence/dense_person_reviewer_static/styles.css",
        repo / "scripts/g7f_c_r2_run_reviewer.py",
        launcher,
        requirements,
    ]
    release = {
        "reviewer_release": RELEASE,
        "repository_baseline_commit": git(repo, "rev-parse", "HEAD"),
        "baseline_matches_expected": git(repo, "rev-parse", "HEAD") == BASELINE,
        "source_state": "AUTHORIZED_UNCOMMITTED_R2_REPAIR",
        "workflow_authorization": "CALIBRATION_ONLY",
        "files": [
            {"path": str(path), "byte_size": path.stat().st_size, "sha256": sha256_file(path)} for path in release_files
        ],
        "production_ready": False,
    }
    release_path = repair / "05_REVIEWER/reviewer_release_manifest.json"
    write_json(release_path, release)
    bindings = original_bindings(all_paths["source"])
    bindings["reviewer_release_manifest_sha256"] = sha256_file(release_path)
    write_json(repair / "05_REVIEWER/reviewer_binding_hashes.json", bindings)
    return release, bindings


def launcher_acceptance(all_paths: dict[str, Path]) -> dict[str, Any]:
    launcher = all_paths["repair"] / "05_REVIEWER/launch_calibration_reviewer_r2.ps1"
    source = launcher.read_text(encoding="utf-8")
    checks = {
        "explicit_fi_reviewer_interpreter": str(FI_PYTHON) in source,
        "broken_dot_venv_absent": ".venv" not in source,
        "explicit_pythonpath": "$env:PYTHONPATH" in source,
        "does_not_install_packages": "& $python -m pip" not in source,
        "check_only_supported": "[switch]$CheckOnly" in source,
    }
    if not all(checks.values()):
        raise RuntimeError(f"R2 launcher checks failed: {checks}")
    result = subprocess.run(
        [
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "-CheckOnly",
        ],
        cwd=all_paths["repo"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"R2 launcher check failed: {result.stdout}\n{result.stderr}")
    return {
        "launcher_path": str(launcher),
        "launcher_sha256": sha256_file(launcher),
        "checks": checks,
        "runtime_check": json.loads(result.stdout.strip().splitlines()[-1]),
        "python": {"executable": str(FI_PYTHON), "version": sys.version.split()[0]},
        "dependencies": {
            "hashlib": "stdlib",
            "numpy": np.__version__,
            "opencv_python": cv2.__version__,
            "pydantic": pydantic.__version__,
        },
        "packages_installed": False,
        "production_ready": False,
    }


def write_handoff(
    all_paths: dict[str, Path],
    *,
    frozen: dict[str, Any],
    integrity: dict[str, Any],
    tests: dict[str, Any],
    release: dict[str, Any],
    bindings: dict[str, str],
    environment: dict[str, Any],
) -> Path:
    repo = all_paths["repo"]
    handoff = all_paths["repair"] / "10_REVIEW_PACK/CHATGPT_HANDOFF"
    handoff.mkdir(parents=True, exist_ok=True)
    write_json(
        handoff / "00_EXECUTIVE_SUMMARY.json",
        {
            "stage": "G7F_C_R2_REVIEWER_NAVIGATION_LOADING_STATE_REPAIR",
            "decision": PASS,
            "root_cause_repaired": True,
            "production_code_change": "renderControls() after state.loading=false in loadImage finally",
            "real_decisions_byte_identical": True,
            "dg001_remains_finalized": True,
            "annotation_resumed_automatically": False,
            "scored_annotation_authorized": False,
            "production_ready": False,
        },
    )
    write_json(handoff / "01_REAL_DECISIONS_BEFORE_AFTER_INTEGRITY.json", integrity)
    write_json(handoff / "02_FROZEN_CONTRACT_HASHES.json", frozen)
    write_json(handoff / "03_TEST_AND_LIVE_EDGE_ACCEPTANCE.json", tests)
    write_json(
        handoff / "04_REVIEWER_R2_RELEASE_AND_ENVIRONMENT.json",
        {
            "reviewer_release": RELEASE,
            "release_manifest": release,
            "release_manifest_sha256": sha256_file(all_paths["repair"] / "05_REVIEWER/reviewer_release_manifest.json"),
            "binding_hashes": bindings,
            "launcher": environment,
            "existing_v1_r1_events_remain_valid": True,
            "truth_schema_migration": False,
            "production_ready": False,
        },
    )
    diff = subprocess.check_output(
        [
            "git",
            "diff",
            "--no-ext-diff",
            "--",
            "src/football_intelligence/dense_person_reviewer_static/app.js",
        ],
        cwd=repo,
        text=True,
    )
    (handoff / "05_PRODUCTION_SOURCE_DIFF.patch").write_text(diff, encoding="utf-8", newline="\n")
    (handoff / "06_DECISION.md").write_text(
        f"# Decision\n\n`{PASS}`\n\n"
        "DG-001 remains immutable. Navigation is enabled after image loading according to queue position, "
        "including on finalized frames. Calibration may resume only when an operator explicitly starts R2. "
        "Scored annotation remains unauthorized.\n\n`production_ready=false`\n",
        encoding="utf-8",
        newline="\n",
    )
    payload_files = sorted(path for path in handoff.iterdir() if path.is_file() and path.name != "10_MANIFEST.json")
    if len(payload_files) != 7:
        raise RuntimeError(f"R2 handoff must contain seven payload files before manifest: {payload_files}")
    write_json(
        handoff / "10_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7f_c_r2.handoff_manifest.v1",
            "files": [
                {"name": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
                for path in payload_files
            ],
            "file_count_excluding_manifest": len(payload_files),
            "decision": PASS,
            "production_ready": False,
        },
    )
    return handoff


def run(phase: str, repo: Path) -> dict[str, Any]:
    all_paths = paths(repo.resolve())
    head = git(repo, "rev-parse", "HEAD")
    if head != BASELINE or git(repo, "rev-parse", "origin/main") != BASELINE:
        raise RuntimeError(f"R2 baseline mismatch: HEAD={head} origin/main={git(repo, 'rev-parse', 'origin/main')}")
    frozen = verify_frozen(all_paths["source"])
    acceptance_root = all_paths["repair"] / "07_ACCEPTANCE"
    acceptance_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="g7f_c_r2_", dir=acceptance_root) as temp_name:
        temp_root = Path(temp_name)
        focused = run_focused_tests(all_paths, temp_root)
        navigation = run_navigation_edge_acceptance(all_paths, temp_root)
        r1_regression = run_r1_edge_acceptance(
            all_paths,
            temp_root / "r1_regression",
            acceptance_root / "r1_interaction_regression.png",
        )
    integrity = real_decision_integrity(all_paths)
    tests = {
        "focused_tests": focused,
        "r2_live_edge_navigation": navigation,
        "r1_live_edge_interaction_regression": r1_regression,
        "r2_navigation_checks_passed": all(row["passed"] for row in navigation["checks"]),
        "r1_interaction_checks_passed": all(row["passed"] for row in r1_regression["checks"]),
        "production_ready": False,
    }
    if phase == "preflight":
        return {
            "phase": phase,
            "decision": "PREFLIGHT_PASS",
            "focused": focused,
            "r2_edge_checks": len(navigation["checks"]),
            "r1_edge_checks": len(r1_regression["checks"]),
            "real_decisions_byte_identical": integrity["byte_identical"],
            "production_ready": False,
        }

    release, bindings = build_release(all_paths)
    environment = launcher_acceptance(all_paths)
    handoff = write_handoff(
        all_paths,
        frozen=frozen,
        integrity=integrity,
        tests=tests,
        release=release,
        bindings=bindings,
        environment=environment,
    )
    return {
        "decision": PASS,
        "handoff": str(handoff),
        "reviewer_release": RELEASE,
        "real_decisions_byte_identical": True,
        "annotation_resumed_automatically": False,
        "scored_annotation_authorized": False,
        "production_ready": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "finalize"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(run(args.phase, args.repo), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
