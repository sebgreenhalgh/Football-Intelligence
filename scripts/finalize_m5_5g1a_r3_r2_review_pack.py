"""Finalize and validate the M5.5G.1A-R3-R2 ChatGPT review pack."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from build_m5_5g1a_r3_r2_dense_first_split import (
    BASELINE,
    CASE_HASH,
    CLASSIFICATION,
    EVIDENCE_HASH,
    ONTOLOGY_HASH,
    ORIGIN,
    PACK,
    PACKAGE,
    REPO,
    ROOT,
    STAGE,
    read_json,
)
from football_intelligence.review_chassis.hashing import sha256_file

RESULTS = STAGE / "07_COMMANDS_AND_TESTS"
BROWSER = STAGE / "04_BROWSER_PERSISTENCE_AND_COMPLETION"
EXPECTED_FILES = (
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_OUTCOME.md",
    "02_REPOSITORY_AND_LIVE_STATE.json",
    "03_TRANCHE_SPLIT.json",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TESTS.md",
    "06_A_B_PRESERVATION.json",
    "07_DENSE_NOVICE_WORKFLOW.md",
    "08_MASK_REVISION_AND_INVALIDATION.json",
    "09_C1_COMPLETION_SEMANTICS.json",
    "10_FIRST_LOAD_RECONCILIATION.json",
    "11_BROWSER_AND_PERSISTENCE.json",
    "12_TIMING.json",
    "13_NEXT_STAGE_PERMISSION.json",
    "14_SAFETY_AND_ACCEPTANCE.json",
    "15_DENSE_THREE_MASKS.png",
    "16_MASK_INVALIDATION.png",
    "17_C1_ATOMIC_COMPLETION.png",
    "18_HUMAN_ACTION.json",
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
        (str(ROOT), "<FOOTBALL_INTELLIGENCE_ROOT>"),
        (str(Path.home()), "<USER_PROFILE>"),
    ):
        value = value.replace(source, replacement).replace(source.replace("\\", "/"), replacement)
    return value


def write_text(name: str, value: str) -> None:
    (PACK / name).write_text(safe_text(value).rstrip() + "\n", encoding="utf-8")


def write_json(name: str, value: Any) -> None:
    write_text(name, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))


def repository_gate(*, precommit: bool) -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    origin = git("remote", "get-url", "origin")
    baseline_ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASELINE, head],
            cwd=REPO,
            check=False,
        ).returncode
        == 0
    )
    if precommit:
        staged = git("diff", "--cached", "--name-only", BASELINE).splitlines()
        unstaged = git("diff", "--name-only").splitlines()
        result = {
            "implementation_commit": "PENDING_COMMIT",
            "authorized_baseline": BASELINE,
            "head_at_precommit": head,
            "head_is_authorized_baseline": head == BASELINE,
            "baseline_is_ancestor": baseline_ancestor,
            "branch": branch,
            "origin": origin,
            "staged_paths": staged,
            "unstaged_paths": unstaged,
            "remote_head_matches_local": git("rev-parse", "@{upstream}") == head,
            "worktree_clean": False,
            "mode": "PRECOMMIT_STAGED",
        }
        result["passed"] = all(
            (
                result["head_is_authorized_baseline"],
                baseline_ancestor,
                branch == "main",
                origin == ORIGIN,
                bool(staged),
                not unstaged,
                result["remote_head_matches_local"],
            )
        )
    else:
        upstream = git("rev-parse", "@{upstream}")
        result = {
            "implementation_commit": head,
            "authorized_baseline": BASELINE,
            "baseline_is_ancestor": baseline_ancestor,
            "branch": branch,
            "origin": origin,
            "remote_head_matches_local": upstream == head,
            "worktree_clean": not git("status", "--porcelain"),
            "mode": "FINAL_COMMITTED",
        }
        result["passed"] = all(
            (
                head != BASELINE,
                baseline_ancestor,
                branch == "main",
                origin == ORIGIN,
                result["remote_head_matches_local"],
                result["worktree_clean"],
            )
        )
    if not result["passed"]:
        raise RuntimeError(f"repository finalization gate failed: {result}")
    return result


def source_diff(*, precommit: bool, implementation_commit: str) -> str:
    if precommit:
        return git("diff", "--cached", "--binary", BASELINE)
    return git("diff", "--binary", BASELINE, implementation_commit)


def changed_python_files(*, precommit: bool, implementation_commit: str) -> list[str]:
    if precommit:
        rows = git("diff", "--cached", "--name-only", BASELINE).splitlines()
    else:
        rows = git("diff", "--name-only", BASELINE, implementation_commit).splitlines()
    return [row for row in rows if row.endswith(".py")]


def run_command(name: str, command: list[str], *, timeout: int = 2400) -> dict[str, Any]:
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
        "summary": " | ".join(lines[-4:]) if lines else "completed without console output",
        "passed": process.returncode == 0,
    }
    if not row["passed"]:
        raise RuntimeError(f"validation command failed: {row}\n{output[-8000:]}")
    return row


def validation_commands(
    *,
    precommit: bool,
    implementation_commit: str,
) -> list[dict[str, Any]]:
    python_files = changed_python_files(
        precommit=precommit,
        implementation_commit=implementation_commit,
    )
    regressions = [
        "tests/test_m5_5g1a_detection_gold_pilot.py",
        "tests/test_m5_5g1a_r1_annotation_ui_correctness.py",
        "tests/test_m5_5g1a_r2_novice_wizard.py",
        "tests/test_m5_5g1a_r3_incremental_gold.py",
        "tests/test_m5_5g1a_r3_r1_wizard_repair.py",
        "tests/test_m5_5g2b_proposal_supply.py",
        "tests/test_m5_5g3_consolidation.py",
    ]
    diff_command = (
        ["git", "diff", "--check", "--cached"]
        if precommit
        else [
            "git",
            "diff",
            "--check",
            BASELINE,
            implementation_commit,
        ]
    )
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
        run_command(
            "ruff_format_check",
            ["uv", "run", "ruff", "format", "--check", *python_files],
        ),
        run_command(
            "node_app",
            ["node", "--check", "src/football_intelligence/review_chassis/static/detection_gold_app.js"],
        ),
        run_command(
            "node_wizard",
            [
                "node",
                "--check",
                "src/football_intelligence/review_chassis/static/detection_gold_wizard.js",
            ],
        ),
        run_command("relevant_regressions", ["uv", "run", "pytest", *regressions, "-q"]),
        run_command("pipeline_help", ["uv", "run", "fi-pipeline", "--help"]),
        run_command("review_help", ["uv", "run", "fi-pipeline", "review-chassis", "--help"]),
        run_command("diff_check", diff_command),
    ]
    return commands


def prepare_pack_status(repository: dict[str, Any]) -> None:
    package = read_json(PACKAGE / "review_package_validation.json")
    build_path = RESULTS / "build_summary.json"
    build = read_json(build_path)
    build.update(
        {
            "implementation_commit": repository["implementation_commit"],
            "package": package,
            "review_pack_pending": False,
            "tests_pending": False,
            "classification": CLASSIFICATION,
        }
    )
    build_path.write_text(json.dumps(build, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_browser(report: dict[str, Any]) -> dict[str, Any]:
    scenarios = report["required_scenarios"]
    return {
        "passed": report["passed"],
        "browser": report["browser"],
        "temporary_copied_decisions_only": report["temporary_copied_decisions_only"],
        "real_human_decisions_root_opened": report["real_human_decisions_root_opened"],
        "source_decisions_preserved": report["source_decisions_preserved"],
        "scenario_count": len(scenarios),
        "scenario_pass_count": sum(bool(value) for value in scenarios.values()),
        "scenarios": scenarios,
        "viewport_count": len(report["visual_regression"]),
        "viewport_pass_count": sum(bool(row["passed"]) for row in report["visual_regression"]),
        "candidate_count_in_exercise": report["candidate_count"],
        "mask_revision": report["mask_revision"],
        "tranche_completions_after_restart": report["tranche_completions_after_restart"],
        "full_completion_attempt": report["full_completion_attempt"],
    }


def populate_pack(
    repository: dict[str, Any],
    commands: list[dict[str, Any]],
    *,
    precommit: bool,
) -> None:
    if PACK.exists():
        unknown = [path for path in PACK.iterdir() if path.name not in EXPECTED_FILES]
        if unknown:
            raise RuntimeError(f"refusing to replace review pack with unknown files: {unknown}")
        for path in PACK.iterdir():
            if path.is_file():
                path.unlink()
    PACK.mkdir(parents=True, exist_ok=True)
    live = read_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "live_state_precondition.json")
    preservation = read_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "a_b_preservation.json")
    split = read_json(STAGE / "02_TRANCHE_MANIFEST_SPLIT" / "tranche_split_validation.json")
    mapping = read_json(STAGE / "02_TRANCHE_MANIFEST_SPLIT" / "old_to_new_tranche_mapping.json")
    wizard = read_json(STAGE / "03_DENSE_WIZARD_VALIDATION" / "dense_wizard_validation.json")
    invalidation = read_json(STAGE / "03_DENSE_WIZARD_VALIDATION" / "mask_revision_invalidation.json")
    completion = read_json(BROWSER / "c1_completion_contract.json")
    first_load = read_json(BROWSER / "first_load_reconciliation.json")
    browser = read_json(BROWSER / "browser_persistence_results.json")
    timing = read_json(BROWSER / "truthful_dense_timing.json")
    permission = read_json(STAGE / "06_NEXT_STAGE_DENSE_PERMISSION" / "next_stage_dense_permission.json")
    package = read_json(PACKAGE / "review_package_validation.json")
    diff = source_diff(
        precommit=precommit,
        implementation_commit=repository["implementation_commit"],
    )
    write_text(
        "01_EXECUTIVE_OUTCOME.md",
        f"""# M5.5G.1A-R3-R2 executive outcome

## Classification

`{CLASSIFICATION}`

The completed 32-case static gold work remains byte-identical. The former
mixed Tranche C is now split into an eight-case dense C1 tranche and a
twelve-case pitch/boundary C2 tranche without changing any of the 88 case
payloads or 1,512 evidence assets.

The browser opens directly at C1 with a fresh local draft namespace. The dense
wizard records visible masks, front/back relationships, explicit occluders and
candidate-to-mask coverage. Editing or deleting a mask invalidates dependent
machine-box answers. C1 has an independent atomic completion bundle; C2 and
full-pilot completion remain separate.

Browser completion evidence was produced only in a temporary copy of the live
decisions root. It is a persistence exercise, not human annotation or gold.
""",
    )
    write_json("02_REPOSITORY_AND_LIVE_STATE.json", {"repository": repository, "live_state": live})
    write_json("03_TRANCHE_SPLIT.json", {"validation": split, "old_to_new_mapping": mapping})
    write_text("04_SOURCE_DIFF.patch", diff)
    write_text(
        "05_COMMANDS_AND_TESTS.md",
        "# Commands and tests\n\n"
        + "\n".join(f"- **{row['name']}:** passed ({row['summary']}; {row['duration_seconds']}s)" for row in commands),
    )
    write_json("06_A_B_PRESERVATION.json", preservation)
    write_text(
        "07_DENSE_NOVICE_WORKFLOW.md",
        """# Dense novice workflow

1. **Trace each visible person.** Trace visible pixels in the locked Current
   frame and focal ROI only. Do not draw through another person or infer hidden
   body pixels.
2. **Answer short overlap questions.** Record mask quality, whether another
   person is in front, which person is the occluder, and image-edge truncation.
3. **Check the machine boxes.** Review one candidate at a time, bind useful or
   merged boxes to explicit masks, and record visible-mask coverage.
4. **Review and save.** Saving remains server-gated and revision-aware.

Previous and Next frames are reference-only. Technical UUIDs stay under
Advanced details.
""",
    )
    write_json(
        "08_MASK_REVISION_AND_INVALIDATION.json",
        {
            "dense_wizard_validation": wizard,
            "contract": invalidation,
            "browser_exercise": browser["mask_revision"],
        },
    )
    write_json(
        "09_C1_COMPLETION_SEMANTICS.json",
        {
            "contract": completion,
            "temporary_completion_result": browser["tranche_completions_after_restart"],
            "full_completion_attempt": browser["full_completion_attempt"],
            "synthetic_acceptance_fixture_only": True,
        },
    )
    write_json("10_FIRST_LOAD_RECONCILIATION.json", first_load)
    write_json("11_BROWSER_AND_PERSISTENCE.json", compact_browser(browser))
    write_json("12_TIMING.json", timing)
    write_json("13_NEXT_STAGE_PERMISSION.json", permission)
    write_json(
        "14_SAFETY_AND_ACCEPTANCE.json",
        {
            "classification": CLASSIFICATION,
            "case_payload_hash": CASE_HASH,
            "evidence_tree_hash": EVIDENCE_HASH,
            "ontology_hash": ONTOLOGY_HASH,
            "review_package_validation": package,
            "model_fit_performed": False,
            "detector_or_consolidator_or_tracker_changed": False,
            "detector_or_consolidator_or_tracker_promoted": False,
            "existing_human_decisions_rewritten": False,
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
            "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        },
    )
    for source, destination in (
        ("01_DENSE_THREE_MASKS_AND_OVERLAP.png", "15_DENSE_THREE_MASKS.png"),
        ("02_MASK_EDIT_INVALIDATES_CANDIDATE.png", "16_MASK_INVALIDATION.png"),
        ("03_C1_COMPLETED_ATOMICALLY.png", "17_C1_ATOMIC_COMPLETION.png"),
    ):
        shutil.copy2(BROWSER / source, PACK / destination)
    write_json(
        "18_HUMAN_ACTION.json",
        {
            "classification": CLASSIFICATION,
            "human_action": "Launch the dense-first package and annotate the eight C1 cases.",
            "launcher": "05_DENSE_FIRST_INCREMENTAL_ANNOTATION_PACKAGE/launch_dense_first_review.ps1",
            "url": "http://127.0.0.1:8807/",
            "expected_first_load": {"tranche": "C1_DENSE_OVERLAP", "progress": "0/8 saved"},
            "c1_human_annotation_completed": False,
            "next_stage_currently_permitted": False,
        },
    )
    entries = [
        {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(item for item in PACK.iterdir() if item.is_file())
    ]
    write_json(
        "REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r2.review_pack_manifest.v1",
            "classification": CLASSIFICATION,
            "implementation_commit": repository["implementation_commit"],
            "file_count_including_manifest": len(entries) + 1,
            "flat": True,
            "maximum_files": 20,
            "maximum_total_bytes": 50 * 1024 * 1024,
            "maximum_visuals": 3,
            "human_decision_payloads_included": False,
            "manifest_self_hash_included": False,
            "entries_excluding_manifest": entries,
        },
    )


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
    credential = re.compile(r"(?:password|api[_-]?key)\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE)
    privacy_hits: list[dict[str, str]] = []
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
        "human_decision_payloads_absent": not any("completed_review" in path.name for path in files),
        "manifest_self_hash_absent": "REVIEW_PACK_MANIFEST.json"
        not in {
            entry["filename"] for entry in read_json(PACK / "REVIEW_PACK_MANIFEST.json")["entries_excluding_manifest"]
        },
    }
    result = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r2.review_pack_validation.v1",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", action="store_true")
    args = parser.parse_args()
    repository = repository_gate(precommit=args.precommit)
    browser = read_json(BROWSER / "browser_persistence_results.json")
    package = read_json(PACKAGE / "review_package_validation.json")
    live = read_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "live_state_precondition.json")
    if not all((browser["passed"], package["passed"], live["passed"], browser["source_decisions_preserved"])):
        raise RuntimeError("live-state, package, browser, or preservation gate failed")

    prepare_pack_status(repository)
    populate_pack(repository, [], precommit=args.precommit)
    commands = validation_commands(
        precommit=args.precommit,
        implementation_commit=repository["implementation_commit"],
    )
    commands.append(
        run_command(
            "focused_r3_r2_pack_aware",
            ["uv", "run", "pytest", "tests/test_m5_5g1a_r3_r2_dense_first_split.py", "-q"],
        )
    )
    commands.append(run_command("full_suite_pack_aware", ["uv", "run", "pytest", "-q"]))
    populate_pack(repository, commands, precommit=args.precommit)
    validation = validate_pack()
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
            "full_suite_summary": next(row["summary"] for row in commands if row["name"] == "full_suite_pack_aware"),
            "classification": CLASSIFICATION,
        }
    )
    build_path.write_text(json.dumps(build, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
