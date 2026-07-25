"""Finalize and validate the M5.5G.1A-R3-R4 ChatGPT review pack."""

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

from build_m5_5g1a_r3_r4_c2_pitch_boundary import (
    ALLOWED_CHANGES,
    BASELINE,
    CASE_HASH,
    CLASSIFICATION,
    EVIDENCE_HASH,
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
    "00_READ_ME_FIRST.md",
    "01_EXECUTIVE_OUTCOME.json",
    "02_REPOSITORY_AND_LIVE_STATE.json",
    "03_PRIOR_TRANCHE_PRESERVATION.json",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TESTS.json",
    "06_C2_MEMBERSHIP.json",
    "07_PITCH_POLYGON_AND_TRANSFORMS.json",
    "08_NOVICE_WORKFLOW.json",
    "09_ROLE_AND_PITCH_SEMANTICS.json",
    "10_FOOTPOINT_UNCERTAINTY.json",
    "11_CANDIDATE_INDEPENDENCE_AND_INVALIDATION.json",
    "12_PERSISTENCE_AND_COMPLETION.json",
    "13_TIMING.json",
    "14_NEXT_STAGE_AND_SAFETY.json",
    "15_C2_START_0_OF_12.png",
    "16_SUBSTITUTE_PLAYER_OFF_PITCH.png",
    "17_BOUNDARY_UNCERTAIN_POLYGON_BAND.png",
    "18_REVIEW_PACK_MANIFEST.json",
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout.strip()


def safe_text(value: str) -> str:
    replacements = (
        (str(REPO), "<REPOSITORY>"),
        (str(ROOT), "<FOOTBALL_INTELLIGENCE_ROOT>"),
        (str(Path.home()), "<USER_PROFILE>"),
    )
    for source, replacement in replacements:
        value = value.replace(source, replacement)
        value = value.replace(source.replace("\\", "/"), replacement)
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
        untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
        result = {
            "implementation_commit": "PENDING_COMMIT",
            "authorized_baseline": BASELINE,
            "head_at_precommit": head,
            "head_is_authorized_baseline": head == BASELINE,
            "baseline_is_ancestor": baseline_ancestor,
            "branch": branch,
            "origin": origin,
            "staged_paths": staged,
            "staged_paths_authorized": set(staged) <= ALLOWED_CHANGES,
            "unstaged_paths": unstaged,
            "untracked_paths": untracked,
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
                result["staged_paths_authorized"],
                "scripts/finalize_m5_5g1a_r3_r4_review_pack.py" in staged,
                not unstaged,
                not untracked,
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
        paths = git("diff", "--cached", "--name-only", BASELINE).splitlines()
    else:
        paths = git("diff", "--name-only", BASELINE, implementation_commit).splitlines()
    return [path for path in paths if path.endswith(".py")]


def run_command(name: str, command: list[str], *, timeout: int = 2400) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    output = "\n".join(part.strip() for part in (process.stdout, process.stderr) if part.strip())
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    result = {
        "name": name,
        "command": " ".join(command),
        "return_code": process.returncode,
        "duration_seconds": round(time.perf_counter() - started, 2),
        "summary": " | ".join(lines[-4:]) if lines else "completed without console output",
        "passed": process.returncode == 0,
    }
    if not result["passed"]:
        raise RuntimeError(f"validation command failed: {result}\n{output[-8000:]}")
    return result


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
        "tests/test_m5_5g1a_r3_r2_dense_first_split.py",
        "tests/test_m5_5g1a_r3_r2_r1_c1_completion_repair.py",
        "tests/test_m5_5g4_r2_corrected_dense_gold.py",
        "tests/test_m5_5g5a_promptable_masks.py",
    ]
    diff_command = (
        ["git", "diff", "--check", "--cached"]
        if precommit
        else ["git", "diff", "--check", BASELINE, implementation_commit]
    )
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
        run_command(
            "ruff_format_check",
            ["uv", "run", "ruff", "format", "--check", *python_files],
        ),
        run_command(
            "node_detection_gold_app",
            ["node", "--check", "src/football_intelligence/review_chassis/static/detection_gold_app.js"],
        ),
        run_command(
            "node_detection_gold_wizard",
            [
                "node",
                "--check",
                "src/football_intelligence/review_chassis/static/detection_gold_wizard.js",
            ],
        ),
        run_command(
            "focused_r3_r4",
            ["uv", "run", "pytest", "tests/test_m5_5g1a_r3_r4_c2_pitch_boundary.py", "-q"],
        ),
        run_command("annotation_completion_g5a_regressions", ["uv", "run", "pytest", *regressions, "-q"]),
        run_command("full_suite", ["uv", "run", "pytest", "-q"], timeout=3600),
        run_command("pipeline_help", ["uv", "run", "fi-pipeline", "--help"]),
        run_command("review_chassis_help", ["uv", "run", "fi-pipeline", "review-chassis", "--help"]),
        run_command("diff_check", diff_command),
    ]


def prepare_pack_status(repository: dict[str, Any]) -> None:
    package = read_json(PACKAGE / "review_package_validation.json")
    build_path = RESULTS / "build_summary.json"
    build = read_json(build_path)
    build.update(
        {
            "implementation_commit": repository["implementation_commit"],
            "review_pack_pending": False,
            "tests_pending": False,
            "review_package_passed": package["passed"],
            "classification": CLASSIFICATION,
        }
    )
    build_path.write_text(json.dumps(build, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_browser(report: dict[str, Any]) -> dict[str, Any]:
    viewports = []
    for row in report["visual_regression"]:
        viewports.append(
            {
                "profile": row["profile"],
                "physical_viewport": row["physical_viewport"],
                "effective_browser_zoom_percent": row["effective_browser_zoom_percent"],
                "image_natural_dimensions": [row["imageNaturalWidth"], row["imageNaturalHeight"]],
                "image_overlay_max_delta_css_pixels": row["imageOverlayMaxDelta"],
                "body_horizontal_overflow_pixels": row["bodyHorizontalOverflowPixels"],
                "passed": row["passed"],
            }
        )
    return {
        "passed": report["passed"],
        "status": report["status"],
        "browser": report["browser"],
        "url": report["url"],
        "temporary_copied_decisions_only": report["temporary_copied_decisions_only"],
        "real_human_decisions_root_opened": report["real_human_decisions_root_opened"],
        "live_decisions_preserved": report["live_decisions_preserved"],
        "scenario_count": len(report["required_scenarios"]),
        "scenario_pass_count": sum(bool(value) for value in report["required_scenarios"].values()),
        "required_scenarios": report["required_scenarios"],
        "viewport_count": len(viewports),
        "viewport_pass_count": sum(bool(row["passed"]) for row in viewports),
        "viewports": viewports,
        "actual_human_active_minutes": report["actual_human_active_minutes"],
        "automation_time_claimed_as_human_time": report["automation_time_claimed_as_human_time"],
    }


def clear_known_pack_files() -> None:
    if PACK.exists():
        unknown = [path for path in PACK.iterdir() if path.name not in EXPECTED_FILES]
        if unknown:
            raise RuntimeError(f"refusing to replace review pack with unknown files: {unknown}")
        for path in PACK.iterdir():
            if path.is_file():
                path.unlink()
    PACK.mkdir(parents=True, exist_ok=True)


def populate_pack(
    repository: dict[str, Any],
    commands: list[dict[str, Any]],
    *,
    precommit: bool,
) -> None:
    clear_known_pack_files()
    live = read_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "live_state_precondition.json")
    preservation = read_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "prior_tranche_preservation.json")
    membership = read_json(STAGE / "02_C2_CASE_AND_PITCH_POLYGON_VALIDATION" / "c2_case_membership_validation.json")
    polygon = read_json(
        STAGE / "02_C2_CASE_AND_PITCH_POLYGON_VALIDATION" / "pitch_polygon_and_transform_validation.json"
    )
    schema = read_json(STAGE / "03_FOCUSED_C2_NOVICE_WORKFLOW" / "c2_annotation_schema_binding.json")
    workflow = read_json(STAGE / "03_FOCUSED_C2_NOVICE_WORKFLOW" / "novice_pitch_workflow_validation.json")
    semantics = read_json(STAGE / "03_FOCUSED_C2_NOVICE_WORKFLOW" / "role_pitch_semantic_examples.json")
    footpoint = read_json(STAGE / "03_FOCUSED_C2_NOVICE_WORKFLOW" / "footpoint_uncertainty_validation.json")
    candidate = read_json(STAGE / "03_FOCUSED_C2_NOVICE_WORKFLOW" / "candidate_independence_validation.json")
    invalidation = read_json(STAGE / "03_FOCUSED_C2_NOVICE_WORKFLOW" / "revision_invalidation_results.json")
    browser = read_json(BROWSER / "browser_persistence_results.json")
    completion = read_json(BROWSER / "c2_completion_contract.json")
    timing = read_json(BROWSER / "truthful_c2_timing.json")
    permission = read_json(STAGE / "06_NEXT_STAGE_PERMISSION" / "next_stage_permission.json")
    package = read_json(PACKAGE / "review_package_validation.json")
    diff = source_diff(
        precommit=precommit,
        implementation_commit=repository["implementation_commit"],
    )

    write_text(
        "00_READ_ME_FIRST.md",
        f"""# M5.5G.1A-R3-R4 review pack

Classification: `{CLASSIFICATION}`

This flat pack documents the C2 pitch/boundary annotation implementation and
real-browser acceptance. It contains no human C2 decisions: C2 remains at
0/12 until the reviewer performs the real annotation. Browser completion was
exercised only against a temporary copied decisions root.

Start with `01_EXECUTIVE_OUTCOME.json`, inspect `04_SOURCE_DIFF.patch`, then
use the three screenshots as direct evidence of the production C2 interface.
""",
    )
    write_json(
        "01_EXECUTIVE_OUTCOME.json",
        {
            "classification": CLASSIFICATION,
            "implementation_commit": repository["implementation_commit"],
            "c2_case_count": 12,
            "c2_real_saved_case_count": 0,
            "human_annotation_required": True,
            "review_package_ready": package["passed"],
            "browser_acceptance_passed": browser["passed"],
            "detector_or_promptable_mask_inference_performed": False,
            "pitch_gate_implemented_or_tuned": False,
            "component_promoted": False,
        },
    )
    write_json("02_REPOSITORY_AND_LIVE_STATE.json", {"repository": repository, "live_state": live})
    write_json("03_PRIOR_TRANCHE_PRESERVATION.json", preservation)
    write_text("04_SOURCE_DIFF.patch", diff)
    write_json(
        "05_COMMANDS_AND_TESTS.json",
        {
            "passed": bool(commands) and all(row["passed"] for row in commands),
            "command_count": len(commands),
            "commands": commands,
        },
    )
    write_json(
        "06_C2_MEMBERSHIP.json",
        {
            "passed": membership["passed"],
            "case_count": len(membership["actual_case_ids"]),
            "case_range": "anonymous C2 cases 1 through 12",
            "all_88_case_payloads_unchanged": membership["all_88_case_payloads_unchanged"],
            "case_payload_hash": membership["case_payload_hash"],
            "evidence_tree_hash": membership["evidence_tree_hash"],
            "authoritative_binding_checks_passed": all(row["passed"] for row in membership["all_rows"]),
        },
    )
    write_json("07_PITCH_POLYGON_AND_TRANSFORMS.json", polygon)
    write_json("08_NOVICE_WORKFLOW.json", {"schema": schema, "workflow": workflow})
    write_json("09_ROLE_AND_PITCH_SEMANTICS.json", semantics)
    write_json("10_FOOTPOINT_UNCERTAINTY.json", footpoint)
    write_json(
        "11_CANDIDATE_INDEPENDENCE_AND_INVALIDATION.json",
        {"candidate_independence": candidate, "revision_invalidation": invalidation},
    )
    write_json(
        "12_PERSISTENCE_AND_COMPLETION.json",
        {
            "browser": compact_browser(browser),
            "completion_contract": completion,
            "review_package_validation": package,
            "synthetic_acceptance_fixture_only": True,
        },
    )
    write_json("13_TIMING.json", timing)
    write_json(
        "14_NEXT_STAGE_AND_SAFETY.json",
        {
            "next_stage_permission": permission,
            "classification": CLASSIFICATION,
            "case_payload_hash": CASE_HASH,
            "evidence_tree_hash": EVIDENCE_HASH,
            "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
            "sandbox_only": True,
            "no_auto_promotion": True,
            "production_ready": False,
            "human_approved": False,
            "identity_or_tracking_performed": False,
            "detector_inference_performed": False,
            "promptable_mask_inference_performed": False,
            "pitch_gate_implemented_or_tuned": False,
            "frozen_light_hq_sam_candidate_modified": False,
            "prior_gold_mutated": False,
        },
    )
    for source, destination in (
        ("01_C2_START_0_OF_12.png", "15_C2_START_0_OF_12.png"),
        ("02_SUBSTITUTE_PLAYER_OFF_PITCH.png", "16_SUBSTITUTE_PLAYER_OFF_PITCH.png"),
        ("03_BOUNDARY_UNCERTAIN_POLYGON_BAND.png", "17_BOUNDARY_UNCERTAIN_POLYGON_BAND.png"),
    ):
        shutil.copy2(BROWSER / source, PACK / destination)

    entries = [
        {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(item for item in PACK.iterdir() if item.is_file())
    ]
    write_json(
        "18_REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r4.review_pack_manifest.v1",
            "classification": CLASSIFICATION,
            "implementation_commit": repository["implementation_commit"],
            "file_count_including_manifest": len(entries) + 1,
            "flat": True,
            "maximum_files": 20,
            "maximum_total_bytes": 50 * 1024 * 1024,
            "maximum_visuals": 3,
            "human_decision_payloads_included": False,
            "hidden_expected_answers_or_private_mappings_included": False,
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
    manifest = read_json(PACK / "18_REVIEW_PACK_MANIFEST.json")
    manifest_entries = manifest["entries_excluding_manifest"]
    manifest_hashes_match = all(
        (PACK / row["filename"]).is_file()
        and (PACK / row["filename"]).stat().st_size == row["size_bytes"]
        and sha256_file(PACK / row["filename"]) == row["sha256"]
        for row in manifest_entries
    )
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
        "manifest_self_hash_absent": "18_REVIEW_PACK_MANIFEST.json"
        not in {row["filename"] for row in manifest_entries},
        "manifest_hashes_match": manifest_hashes_match,
        "manifest_entry_count_matches": len(manifest_entries) + 1 == len(files),
    }
    result = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r4.review_pack_validation.v1",
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
    preservation = read_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "prior_tranche_preservation.json")
    if not all(
        (
            browser["passed"],
            package["passed"],
            live["passed"],
            preservation["passed"],
            browser["live_decisions_preserved"],
            browser["real_human_decisions_root_opened"] is False,
        )
    ):
        raise RuntimeError("live-state, package, browser, or preservation gate failed")

    prepare_pack_status(repository)
    populate_pack(repository, [], precommit=args.precommit)
    commands = validation_commands(
        precommit=args.precommit,
        implementation_commit=repository["implementation_commit"],
    )
    populate_pack(repository, commands, precommit=args.precommit)
    validation = validate_pack()
    validation.update(
        {
            "implementation_commit": repository["implementation_commit"],
            "classification": CLASSIFICATION,
            "manifest_sha256": sha256_file(PACK / "18_REVIEW_PACK_MANIFEST.json"),
        }
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
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
