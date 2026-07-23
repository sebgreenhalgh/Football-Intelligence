"""Finalize and validate the M5.5G.1A-R3-R1 ChatGPT review pack."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from build_m5_5g1a_r3_r1_wizard_repair import (
    BASELINE,
    CASE_HASH,
    CLASSIFICATION,
    EVIDENCE_HASH,
    ONTOLOGY_HASH,
    ORIGIN,
    PACKAGE,
    REPO,
    ROOT,
    STAGE,
    read_json,
)
from football_intelligence.review_chassis.hashing import sha256_file

PACK = STAGE / "07_REVIEW_PACK_FOR_CHATGPT"
RESULTS = STAGE / "06_COMMANDS_AND_TESTS"
BROWSER_ROOT = STAGE / "04_BROWSER_PERSISTENCE_AND_REGRESSION"
EXPECTED_FILES = (
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_OUTCOME.md",
    "02_REPOSITORY_AND_LIVE_STATE.json",
    "03_CHANGED_FILES.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TESTS.md",
    "06_SAVED_WORK_PRESERVATION.json",
    "07_REVISION_AND_INVALIDATION.json",
    "08_PROGRESS_AND_SUMMARY.json",
    "09_RESTART_AND_FIRST_LOAD.json",
    "10_BROWSER_AND_PERSISTENCE.json",
    "11_PACKAGE_AND_LAUNCHER.json",
    "12_SAFETY_AND_ACCEPTANCE.json",
    "13_STALE_WARNING_AFTER_DELETION.png",
    "14_DELETE_ALL_RETURNS_STEP1.png",
    "15_CLEAN_RESTARTED_CASE7.png",
    "16_HUMAN_ACTION_AND_NEXT_STAGE.json",
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
    replacements = (
        (str(REPO), "<REPOSITORY>"),
        (str(ROOT), "<FOOTBALL_INTELLIGENCE_ROOT>"),
        (str(Path.home()), "<USER_PROFILE>"),
    )
    for source, replacement in replacements:
        value = value.replace(source, replacement).replace(source.replace("\\", "/"), replacement)
    return value


def write_text(name: str, value: str) -> None:
    (PACK / name).write_text(safe_text(value).rstrip() + "\n", encoding="utf-8")


def write_json(name: str, value: Any) -> None:
    write_text(name, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))


def repository_gate() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    upstream = git("rev-parse", "@{upstream}")
    result = {
        "implementation_commit": head,
        "authorized_baseline": BASELINE,
        "baseline_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASELINE, head],
            cwd=REPO,
            check=False,
        ).returncode
        == 0,
        "branch": git("branch", "--show-current"),
        "origin": git("remote", "get-url", "origin"),
        "remote_head_matches_local": upstream == head,
        "worktree_clean": not git("status", "--porcelain"),
    }
    result["passed"] = all(
        (
            head != BASELINE,
            result["baseline_is_ancestor"],
            result["branch"] == "main",
            result["origin"] == ORIGIN,
            result["remote_head_matches_local"],
            result["worktree_clean"],
        )
    )
    if not result["passed"]:
        raise RuntimeError(f"repository finalization gate failed: {result}")
    return result


def run_command(name: str, command: list[str], *, timeout: int = 1800) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )
    output = "\n".join(part.strip() for part in (process.stdout, process.stderr) if part.strip())
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    row = {
        "name": name,
        "command": " ".join(command),
        "return_code": process.returncode,
        "duration_seconds": round(time.perf_counter() - started, 2),
        "summary": " | ".join(lines[-3:]) if lines else "completed without console output",
        "passed": process.returncode == 0,
    }
    if not row["passed"]:
        raise RuntimeError(f"validation command failed: {row}\n{output[-5000:]}")
    return row


def changed_python_files(head: str) -> list[str]:
    return [path for path in git("diff", "--name-only", BASELINE, head).splitlines() if path.endswith(".py")]


def validation_commands(head: str) -> list[dict[str, Any]]:
    python_files = changed_python_files(head)
    regressions = [
        "tests/test_m5_5g1a_r3_incremental_gold.py",
        "tests/test_m5_5g1a_r2_novice_wizard.py",
        "tests/test_m5_5g1a_r1_annotation_ui_correctness.py",
        "tests/test_m5_5g1a_detection_gold_pilot.py",
        "tests/test_m5_5g2a_proposal_supply.py",
        "tests/test_m5_5g0_detection_forensics.py",
        "tests/test_m5_5f1a4_persistence.py",
        "tests/test_step2m3t_review_persistence.py",
    ]
    return [
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
            "node_app",
            ["node", "--check", "src/football_intelligence/review_chassis/static/detection_gold_app.js"],
        ),
        run_command(
            "node_wizard",
            ["node", "--check", "src/football_intelligence/review_chassis/static/detection_gold_wizard.js"],
        ),
        run_command("focused_r3_r1", ["uv", "run", "pytest", "tests/test_m5_5g1a_r3_r1_wizard_repair.py", "-q"]),
        run_command("relevant_regressions", ["uv", "run", "pytest", *regressions, "-q"]),
        run_command("full_suite", ["uv", "run", "pytest", "-q"], timeout=2400),
        run_command("pipeline_help", ["uv", "run", "fi-pipeline", "--help"]),
        run_command("review_help", ["uv", "run", "fi-pipeline", "review-chassis", "--help"]),
        run_command("diff_check", ["git", "diff", "--check", BASELINE, head]),
    ]


def compact_browser(report: dict[str, Any]) -> dict[str, Any]:
    scenarios = report["required_scenarios"]
    return {
        "passed": report["passed"],
        "browser": report["browser"],
        "temporary_copied_decisions_only": report["temporary_copied_decisions_only"],
        "real_human_decisions_root_opened": report["real_human_decisions_root_opened_for_browser_test"],
        "source_decisions_preserved": report["source_decisions_preserved"],
        "scenario_count": len(scenarios),
        "scenario_pass_count": sum(bool(value) for value in scenarios.values()),
        "scenarios": scenarios,
        "viewports": [
            {
                "profile": row["profile"],
                "passed": row["passed"],
                "horizontal_overflow_pixels": row["bodyHorizontalOverflowPixels"],
                "image_overlay_max_delta_pixels": row["imageOverlayMaxDelta"],
            }
            for row in report["visual_regression"]
        ],
        "deletion_progress": report["deletion_progress"],
        "stale_api_http_status": report["stale_api_rejection"]["http_status"],
        "temporary_final_b_saved_count": report["temporary_final_b_saved_count"],
    }


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


def validate_pack() -> dict[str, Any]:
    files = sorted(path for path in PACK.iterdir() if path.is_file())
    nested = [path for path in PACK.rglob("*") if path.is_file() and path.parent != PACK]
    visuals = [path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    total_bytes = sum(path.stat().st_size for path in files)
    forbidden_extensions = {".mp4", ".avi", ".mov", ".pt", ".pth", ".onnx"}
    forbidden_names = [path.name for path in files if path.suffix.lower() in forbidden_extensions]
    privacy_hits: list[dict[str, str]] = []
    credential = re.compile(r"(?:password|api[_-]?key)\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE)
    for path in files:
        if path in visuals:
            continue
        value = path.read_text(encoding="utf-8", errors="replace")
        for token in ("C:" + "\\Users\\", "/" + "Users/"):
            if token.lower() in value.lower():
                privacy_hits.append({"filename": path.name, "token": token})
        if "BEGIN " + "PRIVATE KEY" in value or credential.search(value):
            privacy_hits.append({"filename": path.name, "token": "credential_pattern"})
    visual_results = [validate_visual(path) for path in visuals]
    checks = {
        "exact_expected_files": {path.name for path in files} == set(EXPECTED_FILES),
        "file_count_at_most_20": len(files) <= 20,
        "flat": not nested,
        "total_size_at_most_50_mib": total_bytes <= 50 * 1024 * 1024,
        "visual_count_at_most_three": len(visuals) <= 3,
        "source_diff_present_nonempty": (PACK / "04_SOURCE_DIFF.patch").stat().st_size > 0,
        "forbidden_binaries_absent": not forbidden_names,
        "personal_or_credential_tokens_absent": not privacy_hits,
        "visuals_valid": all(row["nonblank"] for row in visual_results),
        "human_decision_payloads_absent": not any("decision_payload" in path.name for path in files),
    }
    result = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r1.review_pack_validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "file_count": len(files),
        "total_size_bytes": total_bytes,
        "visual_count": len(visuals),
        "visuals": visual_results,
        "forbidden_names": forbidden_names,
        "privacy_hits": privacy_hits,
    }
    if not result["passed"]:
        raise RuntimeError(f"review-pack validation failed: {result}")
    return result


def populate_pack(repository: dict[str, Any], commands: list[dict[str, Any]]) -> dict[str, Any]:
    if PACK.exists():
        unknown = [path for path in PACK.iterdir() if path.name not in EXPECTED_FILES]
        if unknown:
            raise RuntimeError(f"refusing to replace review pack with unknown files: {unknown}")
        for path in PACK.iterdir():
            if path.is_file():
                path.unlink()
    PACK.mkdir(parents=True, exist_ok=True)

    live = read_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "live_state_precondition.json")
    preservation = read_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "saved_case_preservation.json")
    stale = read_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "stale_case7_draft_disposition.json")
    revision = read_json(STAGE / "02_REVISION_AWARE_WIZARD_STATE" / "revision_and_invalidation_contract.json")
    invalidation = read_json(STAGE / "02_REVISION_AWARE_WIZARD_STATE" / "wizard_dependency_graph.json")
    progress = read_json(STAGE / "02_REVISION_AWARE_WIZARD_STATE" / "candidate_answer_validity_matrix.json")
    restart = read_json(STAGE / "03_SAFE_CASE_RESET_AND_RECOVERY" / "case_restart_validation.json")
    first_load = read_json(STAGE / "03_SAFE_CASE_RESET_AND_RECOVERY" / "first_load_reconciliation.json")
    browser = read_json(BROWSER_ROOT / "browser_persistence_results.json")
    package = read_json(PACKAGE / "review_package_validation.json")
    changed = git("diff", "--name-only", BASELINE, repository["implementation_commit"]).splitlines()

    write_text(
        "01_EXECUTIVE_OUTCOME.md",
        f"""# M5.5G.1A-R3-R1 executive outcome

## Classification

`{CLASSIFICATION}`

The repaired wizard restores the six server-saved Tranche B cases, discards
only the unsaved stale Case 7 browser draft, and opens Case 7 at a clean Step 1.
Revision-aware candidate answers can no longer remain silently valid after
people are added, deleted, redrawn, or reclassified. Safe restart affects only
the current unsaved browser draft and writes no server event.

All real-browser recovery scenarios passed against a temporary copy of the
ledger. The original Tranche A completion and all six saved Tranche B decisions
remain byte-for-byte unchanged.
""",
    )
    write_json("02_REPOSITORY_AND_LIVE_STATE.json", {"repository": repository, "live_state": live})
    write_text("03_CHANGED_FILES.md", "# Changed files\n\n" + "\n".join(f"- `{row}`" for row in changed))
    write_text("04_SOURCE_DIFF.patch", git("diff", "--binary", BASELINE, repository["implementation_commit"]))
    write_text(
        "05_COMMANDS_AND_TESTS.md",
        "# Commands and tests\n\n"
        + "\n".join(f"- **{row['name']}:** passed ({row['summary']}; {row['duration_seconds']}s)" for row in commands),
    )
    write_json(
        "06_SAVED_WORK_PRESERVATION.json",
        {
            "live_gate_passed": live["passed"],
            "tranche_a_completed": live["tranche_a_completed"],
            "tranche_a_saved_case_count": live["tranche_a_saved_case_count"],
            "tranche_b_saved_case_count": live["tranche_b_saved_case_count"],
            "saved_b_case_ids": live["tranche_b_saved_case_ids"],
            "current_case_id": live["tranche_b_current_case_id"],
            "current_case_saved": live["tranche_b_current_case_saved"],
            "pending_outbox_events": live["pending_outbox_events"],
            "saved_case_preservation": preservation,
            "source_decisions_preserved_in_browser": browser["source_decisions_preserved"],
        },
    )
    write_json(
        "07_REVISION_AND_INVALIDATION.json",
        {"revision_model": revision, "invalidation_matrix": invalidation},
    )
    write_json(
        "08_PROGRESS_AND_SUMMARY.json",
        {"progress_semantics": progress, "browser_deletion_progress": browser["deletion_progress"]},
    )
    write_json(
        "09_RESTART_AND_FIRST_LOAD.json",
        {
            "stale_draft_disposition": stale,
            "safe_restart": restart,
            "first_load_reconciliation": first_load,
        },
    )
    write_json("10_BROWSER_AND_PERSISTENCE.json", compact_browser(browser))
    write_json(
        "11_PACKAGE_AND_LAUNCHER.json",
        {
            "package_validation": package,
            "launcher": "05_REPAIRED_INCREMENTAL_ANNOTATION_PACKAGE/launch_repaired_incremental_gold_review.ps1",
            "url": "http://127.0.0.1:8807/",
            "same_server_decisions_root": True,
            "new_client_build_id": package["client_build_id"],
            "new_indexeddb_namespace": package["indexeddb_namespace"],
        },
    )
    write_json(
        "12_SAFETY_AND_ACCEPTANCE.json",
        {
            "classification": CLASSIFICATION,
            "case_payload_hash": CASE_HASH,
            "evidence_tree_hash": EVIDENCE_HASH,
            "ontology_hash": ONTOLOGY_HASH,
            "model_fit_performed": False,
            "detector_or_tracker_changed": False,
            "detector_or_tracker_promoted": False,
            "existing_human_decisions_rewritten": False,
            "historical_artifacts_mutated": False,
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
        },
    )
    for source, destination in (
        ("01_STALE_WARNING_AFTER_PERSON_DELETION.png", "13_STALE_WARNING_AFTER_DELETION.png"),
        ("02_DELETE_ALL_RETURNS_STEP1.png", "14_DELETE_ALL_RETURNS_STEP1.png"),
        ("03_CLEAN_RESTARTED_CASE7_6_OF_14.png", "15_CLEAN_RESTARTED_CASE7.png"),
    ):
        shutil.copy2(BROWSER_ROOT / source, PACK / destination)
    write_json(
        "16_HUMAN_ACTION_AND_NEXT_STAGE.json",
        {
            "classification": CLASSIFICATION,
            "human_action": "Launch the repaired package at port 8807 and resume Tranche B from Case 7.",
            "expected_first_load": {
                "progress": "6/14 saved",
                "case": "Visible people 16",
                "step": 1,
                "people": 0,
            },
            "saved_work_will_not_be_resaved": True,
            "next_stage_blocked_until_tranche_b_completed": True,
        },
    )

    entries = [
        {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(item for item in PACK.iterdir() if item.is_file())
    ]
    write_json(
        "REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r1.review_pack_manifest.v1",
            "classification": CLASSIFICATION,
            "implementation_commit": repository["implementation_commit"],
            "file_count_including_manifest": len(entries) + 1,
            "flat": True,
            "maximum_files": 20,
            "maximum_total_bytes": 50 * 1024 * 1024,
            "maximum_visuals": 3,
            "human_decision_payloads_included": False,
            "entries_excluding_manifest": entries,
        },
    )
    return validate_pack()


def main() -> None:
    repository = repository_gate()
    browser = read_json(BROWSER_ROOT / "browser_persistence_results.json")
    package = read_json(PACKAGE / "review_package_validation.json")
    live = read_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "live_state_precondition.json")
    if not all((browser["passed"], package["passed"], live["passed"], browser["source_decisions_preserved"])):
        raise RuntimeError("live-state, package, browser, or preservation gate failed")

    commands = validation_commands(repository["implementation_commit"])
    validation = populate_pack(repository, commands)
    validation.update(
        {
            "implementation_commit": repository["implementation_commit"],
            "classification": CLASSIFICATION,
            "manifest_sha256": sha256_file(PACK / "REVIEW_PACK_MANIFEST.json"),
        }
    )
    (RESULTS / "command_results.json").write_text(
        json.dumps({"passed": True, "commands": commands}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (RESULTS / "review_pack_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_path = RESULTS / "build_summary.json"
    build = read_json(build_path)
    build.update(
        {
            "implementation_commit": repository["implementation_commit"],
            "package": package,
            "review_pack_pending": False,
            "review_pack_validation_passed": validation["passed"],
            "tests_pending": False,
            "full_suite_summary": next(row["summary"] for row in commands if row["name"] == "full_suite"),
            "classification": CLASSIFICATION,
        }
    )
    build_path.write_text(json.dumps(build, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
