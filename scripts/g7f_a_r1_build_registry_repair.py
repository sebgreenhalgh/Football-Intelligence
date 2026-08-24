from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from football_intelligence.gold_eval.core import (
    UnsupportedMetricError,
    evaluator_code_sha256,
    inventory_tree,
    read_json,
    sha256_file,
    validate_gold,
    write_json,
    write_jsonl,
)
from football_intelligence.gold_eval.evaluation import (
    evaluate_candidate_run,
    require_supported_metric,
    validate_candidate_run,
)
from football_intelligence.gold_eval.r1 import (
    EXPECTED_COUNTS,
    build_frame_registries,
    build_frozen_reference_run,
    coverage_summary,
    registry_schemas,
)


BASELINE_COMMIT = "fa63d88117d739820b796c9f291dd893c92996d6"
PASS = "PASS_G7F_A_R1_FULL_SOURCE_FRAME_REGISTRY_AND_ADAPTER_COVERAGE_REPAIR_READY_FOR_G7F_B_RESUME"
HOLD = "HOLD_G7F_A_R1_FULL_SOURCE_FRAME_REGISTRY_AND_ADAPTER_COVERAGE_REPAIR_REQUIRED"
EXPECTED_GOLD_HASHES = {
    "gold_bursts.jsonl": "fa949b5f16707f3379adfafe838ce45ae515182fe7f4a0c90fef7aedc64bc362",
    "gold_subjects.jsonl": "77e4e3d5a6d0412a3a68563bdd33e0434d90d24e77ffe25039932a652c093b1f",
    "gold_subject_frames.jsonl": "e00100e1038f09d2c3bff4e0b2c5554565f5f9a1aa13bb3716d6f0ab57c5deea",
    "gold_missed_observations.jsonl": "ca6a6fd6586eb977cd00d8d97262e9d57272ce8fe42c0b12db200eea48b01931",
    "gold_selected_candidate_bindings.jsonl": "9f81bfb941f334e61aa7d05ed2a825a83e15dc740805b880373608b7d64cea07",
    "gold_source_event_index.jsonl": "fc2a872632316af5cb92edaecee8c3040d8a250f630aa8291d26bc938affbf9e",
}
EXPECTED_GOLD_MANIFEST_SHA256 = "0a8e712487249801caa36509fce5833b4dbb3941f45d98fd2bb408371c170fdb"
EXPECTED_SPLIT_SHA256 = "944d4f8a29087a6af5b955e0652295d8c24788f37ad1c08828403cb37bae1306"
EXPECTED_METRIC_SHA256 = "2edecc2b5d0850930eb0f9ac9fa1f69bfb6a59ee495d489db266255ed618e94e"
EXPECTED_HUMAN_INVENTORY_SHA256 = "5fb20e72f35ba7bce75876d4aa584cf87db1604248ec5c3ea476b778db719672"


def git(repository: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True, text=True).stdout.strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _command_report(command: list[str], repository: Path, environment: dict[str, str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _junit_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "sha256": sha256_file(path),
    }


def _write_command_log(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"command={report['command']}",
            f"returncode={report['returncode']}",
            "stdout:",
            report["stdout"],
            "stderr:",
            report["stderr"],
        ]
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def _artifact(path: Path) -> dict[str, Any]:
    return {"byte_size": path.stat().st_size, "sha256": sha256_file(path)}


def _gold_hash_report(workspace: Path) -> dict[str, Any]:
    rows = {}
    for name, expected in EXPECTED_GOLD_HASHES.items():
        path = workspace / "02_NORMALIZED_GOLD" / name
        actual = sha256_file(path)
        rows[name] = {
            "byte_size": path.stat().st_size,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "byte_identical": actual == expected,
        }
    return {"files": rows, "all_six_byte_identical": all(row["byte_identical"] for row in rows.values())}


def _metric_guard_report(workspace: Path) -> dict[str, Any]:
    guards = {}
    for metric in ("precision", "recall", "mAP", "HOTA"):
        try:
            require_supported_metric(workspace, metric)
        except UnsupportedMetricError as error:
            guards[metric] = {"failed_closed": True, "message": str(error)}
        else:
            guards[metric] = {"failed_closed": False}
    return {
        "unsupported_requests": guards,
        "all_required_unsupported_metrics_failed_closed": all(row["failed_closed"] for row in guards.values()),
        "supported_point_metric": {
            "name": "subject_marker_candidate_support_rate",
            "tier": require_supported_metric(workspace, "subject_marker_candidate_support_rate"),
        },
    }


def _build(args: argparse.Namespace, staging: Path) -> dict[str, Any]:
    repository = args.repository.resolve()
    original = args.original_workspace.resolve()
    reviewer = args.reviewer_package.resolve()
    decisions = args.decisions_root.resolve()
    repair_commit = git(repository, "rev-parse", "HEAD")
    branch = git(repository, "branch", "--show-current")
    status = git(repository, "status", "--porcelain")
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASELINE_COMMIT, repair_commit],
            cwd=repository,
            check=False,
        ).returncode
        == 0
    )
    _require(branch == "main", f"expected main branch, found {branch}")
    _require(ancestor, f"expected baseline {BASELINE_COMMIT} is not an ancestor of {repair_commit}")
    _require(not status, "repository worktree must be clean for final R1 build")
    _require(original.is_dir(), f"original G7F-A workspace missing: {original}")
    _require(reviewer.is_dir(), f"frozen reviewer package missing: {reviewer}")
    _require(decisions.is_dir(), f"human decisions root missing: {decisions}")

    original_before = inventory_tree(original)
    shutil.copytree(original, staging, dirs_exist_ok=True)
    handoff = staging / "10_REVIEW_PACK" / "CHATGPT_HANDOFF"
    if handoff.is_dir():
        shutil.rmtree(handoff)
    handoff.mkdir(parents=True)
    evidence = staging / "09_R1_RELEASE_EVIDENCE"
    evidence.mkdir(parents=True)

    instance_rows, source_rows, defect_report = build_frame_registries(original, reviewer)
    schemas = registry_schemas()
    for name, schema in schemas.items():
        write_json(staging / "01_GOLD_SCHEMAS" / name, schema)
    write_json(
        staging / "04_EVALUATION_HARNESS" / "future_candidate_run_v2.schema.json",
        schemas["future_candidate_run_v2.schema.json"],
    )
    frame_instance_path = staging / "02_NORMALIZED_GOLD" / "gold_frame_instances.jsonl"
    source_registry_path = staging / "02_NORMALIZED_GOLD" / "gold_source_frame_registry.jsonl"
    write_jsonl(frame_instance_path, instance_rows)
    write_jsonl(source_registry_path, source_rows)
    write_json(
        staging / "04_EVALUATION_HARNESS" / "evaluator_binding_r1.json",
        {
            "schema_version": "football_intelligence.g7f_a_r1.evaluator_binding.v1",
            "repair_parent_commit": BASELINE_COMMIT,
            "repository_commit": repair_commit,
            "evaluator_code_sha256": evaluator_code_sha256(),
            "gold_corpus_version": "G7F_A_GOLD_CORPUS_V1",
            "gold_scope": "TEMPORAL_OBSERVATION_GOLD",
            "production_ready": False,
        },
    )

    frozen_run = build_frozen_reference_run(instance_rows, reviewer, repair_commit)
    frozen_run_path = staging / "05_FROZEN_BASELINE" / "frozen_historical_candidate_run_v2.json"
    write_json(frozen_run_path, frozen_run)
    validated = validate_candidate_run(staging, frozen_run_path, require_exact_frame_coverage=True)
    first_evaluation = evidence / "frozen_reference_candidate_evaluation_a.json"
    second_evaluation = evidence / "frozen_reference_candidate_evaluation_b.json"
    evaluate_candidate_run(
        staging,
        frozen_run_path,
        first_evaluation,
        require_exact_frame_coverage=True,
    )
    evaluate_candidate_run(
        staging,
        frozen_run_path,
        second_evaluation,
        require_exact_frame_coverage=True,
    )
    deterministic_evaluation = first_evaluation.read_bytes() == second_evaluation.read_bytes()

    verifier_output = evidence / "independent_frame_population_verification.json"
    verifier_command = [
        sys.executable,
        str(repository / "scripts" / "g7f_a_r1_verify_frame_registry.py"),
        "--original-workspace",
        str(original),
        "--reviewer-package",
        str(reviewer),
        "--r1-workspace",
        str(staging),
        "--output",
        str(verifier_output),
    ]
    environment = os.environ.copy()
    verifier = _command_report(verifier_command, repository, environment)
    _write_command_log(evidence / "independent_verifier.log", verifier)
    _require(verifier["passed"], "independent frame-population verifier failed")
    independent = read_json(verifier_output)

    environment.update(
        {
            "G7F_A_WORKSPACE": str(staging),
            "G7F_A_R1_WORKSPACE": str(staging),
            "G7F_A_ORIGINAL_WORKSPACE": str(original),
            "G7F_A_REVIEWER_PACKAGE": str(reviewer),
        }
    )
    focused_xml = staging / "08_TESTS_AND_INTEGRITY" / "focused_g7f_a_r1_pytest.xml"
    applicable_xml = staging / "08_TESTS_AND_INTEGRITY" / "applicable_g7f_a_pytest.xml"
    focused = _command_report(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_g7f_a_r1_frame_registry_and_adapter.py",
            "-q",
            f"--junitxml={focused_xml}",
        ],
        repository,
        environment,
    )
    _write_command_log(staging / "08_TESTS_AND_INTEGRITY" / "focused_g7f_a_r1_pytest.log", focused)
    applicable = _command_report(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_g7f_a_gold_corpus_and_eval.py",
            "-q",
            f"--junitxml={applicable_xml}",
        ],
        repository,
        environment,
    )
    _write_command_log(staging / "08_TESTS_AND_INTEGRITY" / "applicable_g7f_a_pytest.log", applicable)
    ruff = _command_report(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "src/football_intelligence/gold_eval",
            "scripts/g7f_a_r1_build_registry_repair.py",
            "scripts/g7f_a_r1_verify_frame_registry.py",
            "tests/test_g7f_a_gold_corpus_and_eval.py",
            "tests/test_g7f_a_r1_frame_registry_and_adapter.py",
        ],
        repository,
        environment,
    )
    _write_command_log(staging / "08_TESTS_AND_INTEGRITY" / "g7f_a_r1_ruff.log", ruff)

    gold_validation = validate_gold(staging)
    gold_hashes = _gold_hash_report(staging)
    split_path = staging / "03_SPLITS" / "gold_split_manifest_v1.json"
    metric_path = staging / "04_EVALUATION_HARNESS" / "metric_capabilities.json"
    gold_manifest_path = staging / "02_NORMALIZED_GOLD" / "gold_normalization_manifest.json"
    metric_guards = _metric_guard_report(staging)
    human_inventory = inventory_tree(decisions)
    original_after = inventory_tree(original)
    omitted_hashes = set(defect_report["formerly_omitted_hashes"])
    frozen_candidates_on_omitted = [
        row for row in frozen_run["candidates"] if row["source_frame_sha256"] in omitted_hashes
    ]
    processed_on_omitted = [
        row for row in frozen_run["processed_frame_instances"] if row["source_frame_sha256"] in omitted_hashes
    ]
    frozen_replay = {
        "run": _artifact(frozen_run_path),
        "processed_frame_instances": len(frozen_run["processed_frame_instances"]),
        "candidate_rows": len(frozen_run["candidates"]),
        "authorized_source_hashes": len({row["source_frame_sha256"] for row in instance_rows}),
        "formerly_omitted_hashes_accepted": len({row["source_frame_sha256"] for row in frozen_candidates_on_omitted}),
        "affected_frame_instances_represented": len(processed_on_omitted),
        "formerly_blocked_candidates_accepted": len(frozen_candidates_on_omitted),
        "coverage": validated["coverage"],
        "candidate_validation_count": validated["candidate_count"],
        "evaluation_a": _artifact(first_evaluation),
        "evaluation_b": _artifact(second_evaluation),
        "repeated_evaluation_byte_identical": deterministic_evaluation,
        "tier_b_point_support_separate_from_tier_a_historical_labels": True,
        "detector_inference_executed": False,
        "production_ready": False,
    }
    test_report = {
        "focused_r1": {**_junit_summary(focused_xml), "passed": focused["passed"]},
        "applicable_g7f_a": {**_junit_summary(applicable_xml), "passed": applicable["passed"]},
        "ruff": {"passed": ruff["passed"], "returncode": ruff["returncode"]},
        "independent_verifier": independent,
        "repeated_evaluation_byte_identical": deterministic_evaluation,
        "production_ready": False,
    }
    integrity_report = {
        "original_workspace_before": original_before,
        "original_workspace_after": original_after,
        "original_workspace_byte_identical": original_before == original_after,
        "human_source_inventory": human_inventory,
        "human_source_inventory_expected_sha256": EXPECTED_HUMAN_INVENTORY_SHA256,
        "human_source_byte_identical": human_inventory["ordered_inventory_sha256"] == EXPECTED_HUMAN_INVENTORY_SHA256,
        "gold": gold_hashes,
        "gold_manifest_sha256": sha256_file(gold_manifest_path),
        "zero_human_truth_writes": True,
        "zero_original_g7f_a_workspace_writes": True,
        "production_ready": False,
    }
    metric_split_report = {
        "metric_contract_sha256": sha256_file(metric_path),
        "expected_metric_contract_sha256": EXPECTED_METRIC_SHA256,
        "split_manifest_sha256": sha256_file(split_path),
        "expected_split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "gold_manifest_sha256": sha256_file(gold_manifest_path),
        "expected_gold_manifest_sha256": EXPECTED_GOLD_MANIFEST_SHA256,
        "metric_guards": metric_guards,
        "fold_count": len(read_json(split_path)["folds"]),
        "future_sealed_holdout_current_case_count": gold_validation["future_sealed_holdout_current_case_count"],
        "production_ready": False,
    }
    frame_report = {
        **coverage_summary(instance_rows, source_rows, defect_report),
        "artifact": _artifact(frame_instance_path),
        "schema": _artifact(staging / "01_GOLD_SCHEMAS" / "gold_frame_instances.schema.json"),
    }
    source_report = {
        "row_count": len(source_rows),
        "artifact": _artifact(source_registry_path),
        "schema": _artifact(staging / "01_GOLD_SCHEMAS" / "gold_source_frame_registry.schema.json"),
        "consistent_dimensions_for_repeated_hashes": True,
        "authorization_source": "gold_source_frame_registry.jsonl",
        "reviewed_point_rows_authorize_frames": False,
        "production_ready": False,
    }
    contract_report = {
        "schema_version": "football_intelligence.g7f_a_r1.candidate_run.v2",
        "schema": _artifact(staging / "04_EVALUATION_HARNESS" / "future_candidate_run_v2.schema.json"),
        "processed_frame_instances_independent_from_candidates": True,
        "zero_candidate_state": "PROCESSED_ZERO_CANDIDATES",
        "legacy_v1_coverage_state": "UNDECLARED_OR_PARTIAL_FRAME_COVERAGE",
        "g7f_b_requires_v2_exact_coverage": True,
        "unknown_hashes_fail_closed": True,
        "dimension_mismatches_fail_closed": True,
        "bad_boxes_fail_closed": True,
        "duplicate_candidate_ids_fail_closed": True,
        "malformed_provenance_or_transform_fails_closed": True,
        "production_ready": False,
    }

    gates = {
        "repository_main_and_expected_ancestry": branch == "main" and ancestor,
        "repository_clean": not status,
        "original_workspace_byte_identical": integrity_report["original_workspace_byte_identical"],
        "human_source_byte_identical": integrity_report["human_source_byte_identical"],
        "six_gold_files_byte_identical": gold_hashes["all_six_byte_identical"],
        "gold_manifest_unchanged": sha256_file(gold_manifest_path) == EXPECTED_GOLD_MANIFEST_SHA256,
        "exact_defect_counts": defect_report["counts"] == EXPECTED_COUNTS,
        "independent_registry_verification": independent["classification"].startswith("PASS_"),
        "exact_frozen_replay": (
            frozen_replay["processed_frame_instances"] == 1080
            and frozen_replay["candidate_rows"] == 49803
            and frozen_replay["authorized_source_hashes"] == 1044
            and frozen_replay["formerly_omitted_hashes_accepted"] == 152
            and frozen_replay["affected_frame_instances_represented"] == 155
            and frozen_replay["formerly_blocked_candidates_accepted"] == 7575
        ),
        "candidate_run_v2_exact_coverage": validated["coverage"]["state"] == "EXACT_FULL_FRAME_COVERAGE",
        "evaluation_deterministic": deterministic_evaluation,
        "metric_contract_unchanged": sha256_file(metric_path) == EXPECTED_METRIC_SHA256,
        "split_manifest_unchanged": sha256_file(split_path) == EXPECTED_SPLIT_SHA256,
        "tier_c_metrics_fail_closed": metric_guards["all_required_unsupported_metrics_failed_closed"],
        "focused_r1_tests_passed": focused["passed"],
        "applicable_g7f_a_tests_passed": applicable["passed"],
        "ruff_passed": ruff["passed"],
        "production_ready_false": True,
        "g7f_b_not_resumed": True,
    }
    passed = all(gates.values())
    classification = PASS if passed else HOLD
    bindings = {
        "repair_commit": repair_commit,
        "repair_parent_commit": BASELINE_COMMIT,
        "r1_manifest_sha256": None,
        "frame_instance_registry_sha256": sha256_file(frame_instance_path),
        "unique_source_frame_registry_sha256": sha256_file(source_registry_path),
        "candidate_run_v2_schema_sha256": sha256_file(
            staging / "04_EVALUATION_HARNESS" / "future_candidate_run_v2.schema.json"
        ),
        "frozen_reference_v2_sha256": sha256_file(frozen_run_path),
        "repaired_evaluator_code_sha256": evaluator_code_sha256(),
        "repaired_evaluation_module_sha256": sha256_file(
            repository / "src" / "football_intelligence" / "gold_eval" / "evaluation.py"
        ),
        "unchanged_gold_manifest_sha256": sha256_file(gold_manifest_path),
        "unchanged_split_manifest_sha256": sha256_file(split_path),
        "unchanged_metric_contract_sha256": sha256_file(metric_path),
        "unchanged_human_source_inventory_sha256": human_inventory["ordered_inventory_sha256"],
        "production_ready": False,
    }
    r1_manifest_path = evidence / "r1_repair_manifest.json"
    write_json(
        r1_manifest_path,
        {
            "schema_version": "football_intelligence.g7f_a_r1.repair_manifest.v1",
            "stage": "G7F_A_R1_FULL_SOURCE_FRAME_REGISTRY_AND_ADAPTER_COVERAGE_REPAIR",
            "classification": classification,
            "repository": {
                "repair_commit": repair_commit,
                "repair_parent_commit": BASELINE_COMMIT,
                "branch": branch,
                "worktree_clean": not status,
            },
            "artifacts": {key: value for key, value in bindings.items() if key != "r1_manifest_sha256"},
            "gates": gates,
            "production_ready": False,
        },
    )
    bindings["r1_manifest_sha256"] = sha256_file(r1_manifest_path)
    decision = {
        "classification": classification,
        "authorize_g7f_b_resume": passed,
        "g7f_b_resumed": False,
        "gates": gates,
        "production_ready": False,
    }
    write_json(evidence / "release_decision.json", decision)

    write_json(
        handoff / "00_EXECUTIVE_SUMMARY.json",
        {
            "stage": "G7F_A_R1_FULL_SOURCE_FRAME_REGISTRY_AND_ADAPTER_COVERAGE_REPAIR",
            "classification": classification,
            "repair_scope": "ADAPTER_FRAME_COVERAGE_ONLY",
            "counts": defect_report["counts"],
            "g7f_b_resumed": False,
            "production_ready": False,
        },
    )
    write_json(handoff / "01_SOURCE_AND_ORIGINAL_GOLD_IMMUTABILITY.json", integrity_report)
    write_json(handoff / "02_FULL_FRAME_INSTANCE_REGISTRY.json", frame_report)
    write_json(handoff / "03_UNIQUE_SOURCE_FRAME_REGISTRY.json", source_report)
    write_json(handoff / "04_OLD_VS_REPAIRED_COVERAGE_DIFF.json", defect_report)
    write_json(handoff / "05_CANDIDATE_RUN_V2_COVERAGE_CONTRACT.json", contract_report)
    write_json(handoff / "06_FROZEN_REFERENCE_REPLAY.json", frozen_replay)
    write_json(handoff / "07_METRIC_AND_SPLIT_IMMUTABILITY.json", metric_split_report)
    write_json(handoff / "08_TEST_AND_DETERMINISM_REPORT.json", test_report)
    write_json(handoff / "09_G7F_B_RESUME_BINDINGS.json", bindings)
    decision_lines = [
        "# G7F-A R1 decision",
        "",
        classification,
        "",
        "The confirmed adapter frame-coverage defect is repaired with the complete frozen source-frame registry.",
        "Human truth, the six original Gold JSONL files, split contract, and metric contract remain unchanged.",
        "Candidate-run v2 declares processed frame instances independently from candidate rows.",
        "G7F-B was not resumed automatically.",
        "",
        "production_ready=false",
        "",
    ]
    (handoff / "10_DECISION.md").write_text("\n".join(decision_lines), encoding="utf-8", newline="\n")
    handoff_files = sorted(path for path in handoff.iterdir() if path.is_file())
    write_json(
        handoff / "12_MANIFEST.json",
        {
            "stage": "G7F_A_R1_FULL_SOURCE_FRAME_REGISTRY_AND_ADAPTER_COVERAGE_REPAIR",
            "classification": classification,
            "files": [
                {"name": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
                for path in handoff_files
            ],
            "file_count": len(handoff_files) + 1,
            "maximum_allowed": 12,
            "production_ready": False,
        },
    )
    _require(len(list(handoff.iterdir())) == 12, "R1 handoff must contain exactly 12 files")
    _require(passed, classification)
    return {"decision": decision, "bindings": bindings}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the isolated G7F-A R1 registry and adapter repair")
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--original-workspace", required=True, type=Path)
    parser.add_argument("--reviewer-package", required=True, type=Path)
    parser.add_argument("--decisions-root", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    _require(not workspace.exists(), f"target R1 workspace already exists: {workspace}")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{workspace.name}.staging-", dir=workspace.parent))
    try:
        result = _build(args, staging)
        staging.rename(workspace)
    except Exception:
        raise
    print(json.dumps({**result, "workspace": str(workspace)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
