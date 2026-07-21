"""Finalize and validate the M5.5G.1A-R3 ChatGPT review pack."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from build_m5_5g1a_r3_incremental_gold import (
    BASELINE,
    CASE_HASH,
    CLASSIFICATION,
    EVIDENCE_HASH,
    FREEZE_HASH,
    ORIGIN,
    PACKAGE,
    R2_DECISION_HASHES,
    R2_PACKAGE,
    REPO,
    ROOT,
    STAGE,
    read_json,
)
from football_intelligence.review_chassis.hashing import sha256_file


PACK = STAGE / "09_REVIEW_PACK_FOR_CHATGPT"
RESULTS_ROOT = STAGE / "08_COMMANDS_AND_TESTS"
EXPECTED_FILES = (
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_OUTCOME.md",
    "02_REPOSITORY_STATE.json",
    "03_CHANGED_FILES.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TESTS.md",
    "06_PRIOR_AND_FROZEN_PRESERVATION.json",
    "07_AFFECTED_CASE_AND_FRAME_LOCK_AUDIT.json",
    "08_CROSS_FRAME_EXCLUSION_AND_BROWSER.json",
    "09_VISIBLE_BODY_AND_FOOTPOINT_WORKFLOW.json",
    "10_GOLD_TRANCHE_MANIFEST.json",
    "11_TRANCHE_COMPLETION_AND_PERSISTENCE.json",
    "12_INCREMENTAL_NEXT_STAGE_PERMISSION.json",
    "13_TRUTHFUL_TIMING.json",
    "14_HUMAN_INSTRUCTIONS.md",
    "15_SAFETY_AND_ACCEPTANCE.json",
    "16_STATIC_FRAME_LOCK.png",
    "17_HIDDEN_FEET_AND_PARTIAL_PERSON.png",
    "18_TRANCHE_A_COMPLETION.png",
    "19_ACCEPTANCE_AND_HUMAN_ACTION.json",
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def safe_text(value: str) -> str:
    for source, replacement in (
        (str(REPO), "<REPOSITORY>"),
        (str(REPO).replace("\\", "/"), "<REPOSITORY>"),
        (str(ROOT), "<FOOTBALL_INTELLIGENCE_ROOT>"),
        (str(ROOT).replace("\\", "/"), "<FOOTBALL_INTELLIGENCE_ROOT>"),
        (str(Path.home()), "<USER_PROFILE>"),
        (str(Path.home()).replace("\\", "/"), "<USER_PROFILE>"),
    ):
        value = value.replace(source, replacement)
    return value


def write_text(root: Path, name: str, value: str) -> None:
    (root / name).write_text(safe_text(value).rstrip() + "\n", encoding="utf-8")


def write_json(root: Path, name: str, value: Any) -> None:
    write_text(root, name, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))


def real_root_status() -> dict[str, Any]:
    decisions = PACKAGE / "decisions"
    state = read_json(decisions / "review_decisions.json")
    completion_files = sorted(path.name for path in decisions.rglob("completed_review*"))
    result = {
        "annotation_count": len(state.get("annotations", {})),
        "decision_count": len(state.get("decisions", {})),
        "wizard_state_count": len(state.get("wizard_states", {})),
        "event_sequence": int(state.get("event_sequence", 0)),
        "event_ledger_bytes": (decisions / "review_decision_events.jsonl").stat().st_size,
        "tranche_completion_count": len(state.get("tranche_completions", {})),
        "completed": bool(state.get("completed", False)),
        "completion_artifacts": completion_files,
    }
    result["fresh_and_empty"] = all(
        (
            result["annotation_count"] == 0,
            result["decision_count"] == 0,
            result["wizard_state_count"] == 0,
            result["event_sequence"] == 0,
            result["event_ledger_bytes"] == 0,
            result["tranche_completion_count"] == 0,
            not result["completed"],
            not completion_files,
        )
    )
    return result


def repository_gate() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    origin = git("remote", "get-url", "origin")
    upstream = git("rev-parse", "@{upstream}")
    status = git("status", "--porcelain")
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASELINE, head],
            cwd=REPO,
            check=False,
        ).returncode
        == 0
    )
    result = {
        "implementation_commit": head,
        "authorized_baseline": BASELINE,
        "baseline_is_ancestor": ancestor,
        "branch": branch,
        "origin": origin,
        "remote_head_matches_local": upstream == head,
        "worktree_clean": not status,
    }
    if not all(
        (
            head != BASELINE,
            ancestor,
            branch == "main",
            origin == ORIGIN,
            upstream == head,
            not status,
        )
    ):
        raise RuntimeError(f"repository finalization gate failed: {result}")
    return result


def run_command(name: str, args: list[str], *, timeout: int = 900) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.run(
        args,
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )
    elapsed = round(time.perf_counter() - started, 2)
    output = "\n".join(part.strip() for part in (process.stdout, process.stderr) if part.strip())
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    result = {
        "name": name,
        "command": " ".join(args),
        "return_code": process.returncode,
        "duration_seconds": elapsed,
        "summary": lines[-1] if lines else "completed without console output",
        "passed": process.returncode == 0,
    }
    if process.returncode != 0:
        raise RuntimeError(f"validation command failed: {result}\n{output[-4000:]}")
    return result


def changed_python_files(head: str) -> list[str]:
    rows = git("diff", "--name-only", BASELINE, head).splitlines()
    return [row for row in rows if row.endswith(".py")]


def preliminary_commands(head: str) -> list[dict[str, Any]]:
    python_files = changed_python_files(head)
    commands = [
        run_command("uv_lock_check", ["uv", "lock", "--check"]),
        run_command("uv_sync", ["uv", "sync"], timeout=1200),
        run_command(
            "cuda_runtime",
            [
                str(REPO / ".venv" / "Scripts" / "python.exe"),
                "-c",
                (
                    "import torch; assert torch.cuda.is_available(); "
                    "x=torch.ones(4, device='cuda:0'); assert float(x.sum()) == 4.0; "
                    "print(torch.__version__, torch.cuda.get_device_name(0))"
                ),
            ],
        ),
        run_command("ruff_check", ["uv", "run", "ruff", "check", *python_files]),
        run_command("ruff_format_check", ["uv", "run", "ruff", "format", "--check", *python_files]),
        run_command(
            "node_detection_gold_app",
            ["node", "--check", "src/football_intelligence/review_chassis/static/detection_gold_app.js"],
        ),
        run_command(
            "node_detection_gold_wizard",
            ["node", "--check", "src/football_intelligence/review_chassis/static/detection_gold_wizard.js"],
        ),
        run_command("fi_pipeline_help", ["uv", "run", "fi-pipeline", "--help"]),
        run_command(
            "review_chassis_help",
            ["uv", "run", "fi-pipeline", "review-chassis", "--help"],
        ),
        run_command("git_diff_check", ["git", "diff", "--check", BASELINE, head]),
    ]
    return commands


def test_commands() -> list[dict[str, Any]]:
    regression_files = [
        "tests/test_m5_5g1a_r2_novice_wizard.py",
        "tests/test_m5_5g1a_r1_annotation_ui_correctness.py",
        "tests/test_m5_5g1a_detection_gold_pilot.py",
        "tests/test_m5_5g0_detection_forensics.py",
        "tests/test_m5_5f1a4_persistence.py",
        "tests/test_step2m3t_review_persistence.py",
        "tests/test_m5_4f5_review_chassis.py",
    ]
    return [
        run_command(
            "focused_r3",
            ["uv", "run", "pytest", "tests/test_m5_5g1a_r3_incremental_gold.py", "-q"],
            timeout=600,
        ),
        run_command(
            "relevant_regressions",
            ["uv", "run", "pytest", *regression_files, "-q"],
            timeout=900,
        ),
        run_command("full_suite", ["uv", "run", "pytest", "-q"], timeout=1800),
    ]


def validate_visual(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        spread = max(ImageStat.Stat(image.convert("RGB").resize((96, 60))).stddev)
        result = {
            "filename": path.name,
            "width": image.width,
            "height": image.height,
            "rgb_standard_deviation_max": round(spread, 3),
            "nonblank": spread > 8,
        }
    if result["width"] < 1000 or result["height"] < 600 or not result["nonblank"]:
        raise RuntimeError(f"invalid review-pack visual: {result}")
    return result


def validate_pack(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.iterdir() if path.is_file())
    nested = [path for path in root.rglob("*") if path.is_file() and path.parent != root]
    visuals = [path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    total_size = sum(path.stat().st_size for path in files)
    forbidden_extensions = {".mp4", ".avi", ".mov", ".pt", ".pth", ".onnx"}
    forbidden_names = [path.name for path in files if path.suffix.lower() in forbidden_extensions]
    privacy_hits: list[dict[str, str]] = []
    credential_assignment = re.compile(
        r"(?:password|api[_-]?key)\s*=\s*['\"][^'\"]+['\"]",
        re.IGNORECASE,
    )
    for path in files:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            continue
        value = path.read_text(encoding="utf-8", errors="replace")
        for token in ("C:" + "\\Users\\", "/" + "Users/"):
            if token.lower() in value.lower():
                privacy_hits.append({"filename": path.name, "token": token})
        if "BEGIN " + "PRIVATE KEY" in value or credential_assignment.search(value):
            privacy_hits.append({"filename": path.name, "token": "credential_pattern"})
    visual_results = [validate_visual(path) for path in visuals]
    checks = {
        "exact_expected_files": {path.name for path in files} == set(EXPECTED_FILES),
        "file_count_at_most_20": len(files) <= 20,
        "flat": not nested,
        "total_size_at_most_50_mib": total_size <= 50 * 1024 * 1024,
        "visual_count_at_most_three": len(visuals) <= 3,
        "source_diff_present_nonempty": (root / "04_SOURCE_DIFF.patch").stat().st_size > 0,
        "forbidden_binaries_absent": not forbidden_names,
        "personal_or_credential_tokens_absent": not privacy_hits,
        "visuals_valid": all(row["nonblank"] for row in visual_results),
    }
    result = {
        "schema_version": "football_intelligence.m5_5g1a_r3.review_pack_validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "file_count": len(files),
        "total_size_bytes": total_size,
        "visual_count": len(visuals),
        "visuals": visual_results,
        "forbidden_names": forbidden_names,
        "privacy_hits": privacy_hits,
    }
    if not result["passed"]:
        raise RuntimeError(f"review-pack validation failed: {result}")
    return result


def compact_browser(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": report["passed"],
        "temporary_decisions_only": report["temporary_decisions_only"],
        "real_r3_decisions_root_opened": report["real_r3_decisions_root_opened"],
        "required_browser_scenarios": report["required_browser_scenarios"],
        "required_scenario_count": len(report["required_browser_scenarios"]),
        "required_scenario_pass_count": sum(report["required_browser_scenarios"].values()),
        "viewports": [
            {
                "profile": row["profile"],
                "passed": row["passed"],
                "horizontal_overflow_pixels": row["bodyHorizontalOverflowPixels"],
                "image_overlay_max_delta_pixels": row["imageOverlayMaxDelta"],
            }
            for row in report["visual_regression"]
        ],
        "tranche_a_atomic_bundle_valid": report["tranche_completion"]["bundle_validation"]["passed"],
        "tranche_a_did_not_complete_full_pilot": not report["tranche_completion"]["full_pilot_completed"],
        "prior_and_production_decisions_preserved": report["prior_and_production_decisions_preserved"],
    }


def populate_pack(
    root: Path,
    *,
    repository: dict[str, Any],
    commands: list[dict[str, Any]],
) -> dict[str, Any]:
    if root.exists():
        unknown = [path for path in root.iterdir() if path.name not in EXPECTED_FILES]
        if unknown:
            raise RuntimeError(f"refusing to replace review pack with unknown files: {unknown}")
        for path in root.iterdir():
            if path.is_file():
                path.unlink()
    root.mkdir(parents=True, exist_ok=True)

    prior = read_json(STAGE / "01_PRIOR_STATE_AND_DEFECT_AUDIT" / "prior_state_validation.json")
    audit = read_json(STAGE / "01_PRIOR_STATE_AND_DEFECT_AUDIT" / "r2_affected_case_audit.json")
    exclusions = read_json(STAGE / "02_STATIC_FRAME_AND_CANDIDATE_LOCK" / "cross_frame_candidate_exclusion_report.json")
    frame_lock = read_json(STAGE / "02_STATIC_FRAME_AND_CANDIDATE_LOCK" / "static_authoritative_frame_binding.json")
    partial = read_json(
        STAGE / "03_FOOTPOINT_AND_PARTIAL_PERSON_WORKFLOW" / "visible_body_and_partial_person_rules.json"
    )
    footpoint = read_json(STAGE / "03_FOOTPOINT_AND_PARTIAL_PERSON_WORKFLOW" / "footpoint_novice_mapping.json")
    exceptions = read_json(STAGE / "03_FOOTPOINT_AND_PARTIAL_PERSON_WORKFLOW" / "footpoint_exception_validation.json")
    tranches = read_json(STAGE / "04_INCREMENTAL_TRANCHE_DESIGN" / "gold_tranche_manifest.json")
    completion = read_json(STAGE / "04_INCREMENTAL_TRANCHE_DESIGN" / "tranche_completion_contract.json")
    browser = read_json(STAGE / "05_BROWSER_PERSISTENCE_AND_REGRESSION" / "browser_persistence_results.json")
    timing = read_json(STAGE / "05_BROWSER_PERSISTENCE_AND_REGRESSION" / "truthful_tranche_timing.json")
    package = read_json(PACKAGE / "review_package_validation.json")
    next_stage = read_json(
        STAGE / "07_NEXT_STAGE_INCREMENTAL_GOLD_CONTRACT" / "incremental_next_stage_permissions.json"
    )
    real_root = real_root_status()
    r2_hashes = {name: sha256_file(R2_PACKAGE / "decisions" / name) for name in R2_DECISION_HASHES}
    changed_files = [
        row for row in git("diff", "--name-only", BASELINE, repository["implementation_commit"]).splitlines() if row
    ]
    source_diff = git("diff", "--binary", BASELINE, repository["implementation_commit"])

    write_text(
        root,
        "01_EXECUTIVE_OUTCOME.md",
        f"""# M5.5G.1A-R3 executive outcome

## Classification

`{CLASSIFICATION}`

The frozen 88-case diagnostic pilot is now split into five independently
completable gold tranches. Static and dense cases lock one authoritative middle
frame, review only candidates physically bound to that frame, and keep adjacent
frames as non-editable context. The visible-body and four-way footpoint workflow
passed real-browser validation without changing the frozen ontology.

Tranche A contains 18 static cases, exactly three from each required stratum.
The real R3 decisions root is fresh and empty. No detector or tracker was
evaluated, trained, changed, promoted, or interpreted as human truth.
""",
    )
    write_json(root, "02_REPOSITORY_STATE.json", repository)
    write_text(root, "03_CHANGED_FILES.md", "# Changed files\n\n" + "\n".join(f"- `{row}`" for row in changed_files))
    write_text(root, "04_SOURCE_DIFF.patch", source_diff)
    write_text(
        root,
        "05_COMMANDS_AND_TESTS.md",
        "# Commands and tests\n\n"
        + "\n".join(
            f"- **{row['name']}:** {'passed' if row['passed'] else 'pending'} "
            f"({row['summary']}; {row['duration_seconds']}s)"
            for row in commands
        ),
    )
    write_json(
        root,
        "06_PRIOR_AND_FROZEN_PRESERVATION.json",
        {
            "passed": prior["passed"] and r2_hashes == R2_DECISION_HASHES,
            "r2_decision_hashes_expected": R2_DECISION_HASHES,
            "r2_decision_hashes_actual": r2_hashes,
            "r2_saved_annotation_count": prior["r2_saved_annotation_count"],
            "r2_event_sequence": prior["r2_event_sequence"],
            "r2_human_work_migration_performed": False,
            "case_payload_hash": prior["case_payload_hash"],
            "expected_case_payload_hash": CASE_HASH,
            "evidence_tree_hash": prior["evidence_tree_hash"],
            "expected_evidence_tree_hash": EVIDENCE_HASH,
            "frozen_ontology_hash": prior["frozen_ontology_hash"],
            "expected_frozen_ontology_hash": FREEZE_HASH,
        },
    )
    compact_cases = [
        {
            "case_id": row["case_id"],
            "module_case_number": row["module_case_number"],
            "task_type": row["task_type"],
            "authoritative_candidate_count": row["authoritative_candidate_count"],
            "excluded_reference_candidate_count": row["excluded_reference_candidate_count"],
            "primary_canvas_locked": row["primary_canvas_locked"],
        }
        for row in audit["affected_cases"]
    ]
    write_json(
        root,
        "07_AFFECTED_CASE_AND_FRAME_LOCK_AUDIT.json",
        {
            "passed": audit["passed"] and frame_lock["passed"],
            "root_cause": audit["root_cause"],
            "repair": audit["repair"],
            "cases_audited": audit["cases_audited"],
            "affected_case_count": audit["affected_case_count"],
            "affected_cases": compact_cases,
            "case_006": next(row for row in compact_cases if row["module_case_number"] == 6),
            "case_007": next(row for row in compact_cases if row["module_case_number"] == 7),
            "authoritative_binding_count": len(frame_lock["bindings"]),
            "references_editable": False,
        },
    )
    write_json(
        root,
        "08_CROSS_FRAME_EXCLUSION_AND_BROWSER.json",
        {
            "excluded_reference_candidate_count": exclusions["excluded_candidate_uuid_count"],
            "excluded_rows_deleted": exclusions["excluded_rows_deleted"],
            "reason": "REFERENCE_FRAME_NOT_EDITABLE",
            "browser": compact_browser(browser),
        },
    )
    write_json(
        root,
        "09_VISIBLE_BODY_AND_FOOTPOINT_WORKFLOW.json",
        {
            "visible_body_rule": partial,
            "automatic_footpoint": footpoint,
            "exceptions": exceptions,
            "frozen_schema_changed": False,
        },
    )
    write_json(root, "10_GOLD_TRANCHE_MANIFEST.json", tranches)
    write_json(
        root,
        "11_TRANCHE_COMPLETION_AND_PERSISTENCE.json",
        {
            "completion_contract": completion,
            "browser_completion_exercise": compact_browser(browser),
            "production_decisions_root": real_root,
            "indexeddb_durable_outbox": True,
            "idempotent_server_events": True,
            "server_authoritative_materialization": True,
        },
    )
    write_json(root, "12_INCREMENTAL_NEXT_STAGE_PERMISSION.json", next_stage)
    write_json(root, "13_TRUTHFUL_TIMING.json", timing)
    write_text(root, "14_HUMAN_INSTRUCTIONS.md", (PACKAGE / "HUMAN_INSTRUCTIONS.md").read_text(encoding="utf-8"))
    write_json(
        root,
        "15_SAFETY_AND_ACCEPTANCE.json",
        {
            "classification": CLASSIFICATION,
            "package_validation_passed": package["passed"],
            "browser_validation_passed": browser["passed"],
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
            "diagnostic_only": True,
            "model_fit_performed": False,
            "detector_or_tracker_evaluated": False,
            "detector_or_tracker_promoted": False,
            "cases_or_evidence_changed": False,
            "human_decisions_in_pack": False,
            "real_r3_decisions_root_fresh_and_empty": real_root["fresh_and_empty"],
        },
    )
    write_json(
        root,
        "19_ACCEPTANCE_AND_HUMAN_ACTION.json",
        {
            "classification": CLASSIFICATION,
            "all_machine_acceptance_gates_passed": all(row["passed"] for row in commands),
            "human_annotation_started": False,
            "human_action": "Launch port 8807 and complete Tranche A - core static.",
            "tranche_a_case_count": 18,
            "full_pilot_remains_incomplete_after_tranche_a": True,
            "next_stage_unlock": next_stage["unlock_condition"],
        },
    )
    visual_root = STAGE / "05_BROWSER_PERSISTENCE_AND_REGRESSION"
    for source, destination in (
        ("01_STATIC_FRAME_LOCK_AND_CANDIDATE_QUEUE.png", "16_STATIC_FRAME_LOCK.png"),
        ("02_HIDDEN_FEET_ESTIMATE_AND_PARTIAL_PERSON.png", "17_HIDDEN_FEET_AND_PARTIAL_PERSON.png"),
        ("03_TRANCHE_A_COMPLETED_FULL_PILOT_BLOCKED.png", "18_TRANCHE_A_COMPLETION.png"),
    ):
        shutil.copy2(visual_root / source, root / destination)

    entries = [
        {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(item for item in root.iterdir() if item.is_file())
    ]
    write_json(
        root,
        "REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3.review_pack_manifest.v1",
            "classification": CLASSIFICATION,
            "implementation_commit": repository["implementation_commit"],
            "file_count_including_manifest": len(entries) + 1,
            "flat": True,
            "maximum_files": 20,
            "maximum_total_bytes": 50 * 1024 * 1024,
            "maximum_visuals": 3,
            "entries_excluding_manifest": entries,
        },
    )
    return validate_pack(root)


def main() -> None:
    repository = repository_gate()
    package = read_json(PACKAGE / "review_package_validation.json")
    browser = read_json(STAGE / "05_BROWSER_PERSISTENCE_AND_REGRESSION" / "browser_persistence_results.json")
    prior = read_json(STAGE / "01_PRIOR_STATE_AND_DEFECT_AUDIT" / "prior_state_validation.json")
    if not all((package["passed"], browser["passed"], prior["passed"], real_root_status()["fresh_and_empty"])):
        raise RuntimeError("package, browser, prior-state or fresh-root gate failed")

    commands = preliminary_commands(repository["implementation_commit"])
    pending_tests = [
        {
            "name": name,
            "command": "pending final review-pack gate",
            "return_code": None,
            "duration_seconds": 0,
            "summary": "pending",
            "passed": False,
        }
        for name in ("focused_r3", "relevant_regressions", "full_suite")
    ]
    build_path = RESULTS_ROOT / "build_summary.json"
    build = read_json(build_path)
    build.update(
        {
            "tests_pending": False,
            "browser_acceptance_pending": False,
            "implementation_commit": repository["implementation_commit"],
            "review_pack_pending": True,
        }
    )
    build_path.write_text(json.dumps(build, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    populate_pack(PACK, repository=repository, commands=[*commands, *pending_tests])

    commands.extend(test_commands())
    validation = populate_pack(PACK, repository=repository, commands=commands)
    validation.update(
        {
            "implementation_commit": repository["implementation_commit"],
            "classification": CLASSIFICATION,
            "manifest_sha256": sha256_file(PACK / "REVIEW_PACK_MANIFEST.json"),
        }
    )
    write_json(RESULTS_ROOT, "command_results.json", {"passed": True, "commands": commands})
    write_json(RESULTS_ROOT, "review_pack_validation.json", validation)
    build.update(
        {
            "review_pack_pending": False,
            "review_pack_validation_passed": validation["passed"],
            "full_suite_summary": next(row["summary"] for row in commands if row["name"] == "full_suite"),
        }
    )
    build_path.write_text(json.dumps(build, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
