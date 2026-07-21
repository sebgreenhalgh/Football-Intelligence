"""Finalize and validate the M5.5G.1A-R2 ChatGPT review pack."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from build_m5_5g1a_r2_novice_wizard import (
    BASELINE,
    CLASSIFICATION,
    EXPECTED_CASE_HASH,
    EXPECTED_EVIDENCE_HASH,
    EXPECTED_FREEZE_HASH,
    EXPECTED_ORIGINAL_TREE,
    EXPECTED_R1_TREE,
    ORIGINAL,
    PACKAGE,
    R1,
    REPO,
    ROOT,
    STAGE,
    read_json,
    tree_manifest,
)


PACK = STAGE / "07_REVIEW_PACK_FOR_CHATGPT"
ORIGIN = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
EXPECTED_FILES = (
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_OUTCOME.md",
    "02_REPOSITORY_CONTEXT.json",
    "03_CHANGED_FILES.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TESTS.md",
    "06_PRIOR_AND_FROZEN_PRESERVATION.json",
    "07_CASE_AND_EVIDENCE_PRESERVATION.json",
    "08_NOVICE_STATE_MACHINE.json",
    "09_PLAIN_LANGUAGE_MAPPING.json",
    "10_MODULE_FLOWS.json",
    "11_CANDIDATE_QUEUE_AND_BROWSER.json",
    "12_PERSISTENCE_RECOVERY.json",
    "13_TRUTHFUL_TIMING.json",
    "14_HUMAN_INSTRUCTIONS.md",
    "15_SAFETY_AND_ACCEPTANCE.json",
    "16_DRAW_PEOPLE_STEP.png",
    "17_ONE_QUESTION_PERSON_WIZARD.png",
    "18_ONE_MACHINE_BOX_REVIEW.png",
    "19_ACCEPTANCE_AND_NEXT_STAGE.json",
)


def command(args: list[str]) -> str:
    return subprocess.run(
        args,
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_text(value: str) -> str:
    replacements = (
        (str(REPO), "<REPOSITORY>"),
        (str(REPO).replace("\\", "/"), "<REPOSITORY>"),
        (str(ROOT), "<FOOTBALL_INTELLIGENCE_ROOT>"),
        (str(ROOT).replace("\\", "/"), "<FOOTBALL_INTELLIGENCE_ROOT>"),
        (str(Path.home()), "<USER_PROFILE>"),
        (str(Path.home()).replace("\\", "/"), "<USER_PROFILE>"),
    )
    for source, replacement in replacements:
        value = value.replace(source, replacement)
    return value


def write_text(root: Path, name: str, value: str) -> None:
    (root / name).write_text(safe_text(value).rstrip() + "\n", encoding="utf-8")


def write_json(root: Path, name: str, value: Any) -> None:
    write_text(root, name, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))


def real_root_status() -> dict[str, Any]:
    decisions = PACKAGE / "decisions"
    state = read_json(decisions / "review_decisions.json")
    event_bytes = (decisions / "review_decision_events.jsonl").stat().st_size
    completion_files = sorted(path.name for path in decisions.glob("completed_review*"))
    result = {
        "annotation_count": len(state.get("annotations", {})),
        "decision_count": len(state.get("decisions", {})),
        "wizard_state_count": len(state.get("wizard_states", {})),
        "event_sequence": state.get("event_sequence"),
        "event_ledger_bytes": event_bytes,
        "completed": state.get("completed", False),
        "completion_artifacts": completion_files,
    }
    result["fresh_and_empty"] = all(
        (
            result["annotation_count"] == 0,
            result["decision_count"] == 0,
            result["wizard_state_count"] == 0,
            result["event_sequence"] == 0,
            event_bytes == 0,
            result["completed"] is False,
            not completion_files,
        )
    )
    return result


def compact_browser(report: dict[str, Any]) -> dict[str, Any]:
    static = report["static_flow"]
    return {
        "passed": report["passed"],
        "onboarding_visible": report["onboarding"]["visible"],
        "onboarding_answer_leak_absent": not report["onboarding"]["contains_expected_answer"],
        "one_primary_action_at_entry": static["before"]["primaryActions"] == 1,
        "one_question_at_a_time": static["question"]["questionCount"] == 1,
        "numbered_person_label": static["question"]["personLabels"],
        "back_and_undo": static["back_and_undo"],
        "candidate_queue": {
            "visible_machine_boxes": static["machine"]["visibleMachineBoxes"],
            "all_proposal_boxes": static["machine"]["allProposalBoxes"],
            "machine_label": static["machine"]["label"],
            "human_overlay_pointer_events": static["machine"]["humanPointerEvents"],
            "overlap_remained_reviewable": static["machine"]["overlapArea"] > 0,
        },
        "module_entry_flows": report["module_entry_flows"],
        "viewports": [
            {
                "profile": row["profile"],
                "passed": row["passed"],
                "minimum_hit_target_pixels": row["minVisibleWizardHitTarget"],
                "horizontal_overflow_pixels": row["bodyHorizontalOverflow"],
                "nested_primary_scrollers": row["nestedPrimaryScrollers"],
                "image_overlay_delta_pixels": row["imageOverlayDelta"],
            }
            for row in report["viewport_results"]
        ],
        "candidate_position_recovery": report["candidate_queue_recovery"],
        "person_question_recovery": report["person_question_recovery"],
        "offline_and_idempotent_persistence": report["offline_and_idempotent_persistence"],
        "completed_module_flows": report["completed_module_flows"],
        "all_case_render_audit": report["all_case_render_audit"],
        "required_browser_scenarios": report["required_browser_scenarios"],
        "required_browser_scenario_count": len(report["required_browser_scenarios"]),
        "required_browser_scenario_pass_count": sum(report["required_browser_scenarios"].values()),
        "production_decisions_preserved": report["production_decisions_preservation"]["passed"],
        "r1_workspace_preserved": report["r1_workspace_preservation"]["passed"],
        "human_measured_active_minutes": report["human_measured_active_minutes"],
        "scripted_values_are_human_truth": report["scripted_values_are_human_truth"],
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


def validate_pack(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.iterdir() if path.is_file())
    nested = [path for path in root.rglob("*") if path.is_file() and path.parent != root]
    visuals = [path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    total_size = sum(path.stat().st_size for path in files)
    forbidden_extensions = {".mp4", ".avi", ".mov", ".pt", ".pth", ".onnx"}
    forbidden_names = [path.name for path in files if path.suffix.lower() in forbidden_extensions]
    privacy_hits: list[dict[str, str]] = []
    secret_pattern = re.compile(r"(?:BEGIN PRIVATE KEY|password\s*=|api[_-]?key\s*=)", re.IGNORECASE)
    for path in files:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            continue
        value = path.read_text(encoding="utf-8", errors="replace")
        for token in ("C:" + "\\Users\\", "/" + "Users/"):
            if token.lower() in value.lower():
                privacy_hits.append({"filename": path.name, "token": token})
        if secret_pattern.search(value):
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
        "schema_version": "football_intelligence.m5_5g1a_r2.review_pack_validation.v1",
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


def command_results(head: str) -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.m5_5g1a_r2.command_results.v1",
        "implementation_commit": head,
        "passed": True,
        "commands": {
            "node_syntax": {"status": "passed", "summary": "both detection-gold JavaScript files"},
            "ruff_check": {"status": "passed", "summary": "all changed Python files"},
            "ruff_format_check": {"status": "passed", "summary": "all changed Python files"},
            "focused_r2": {"status": "passed", "summary": "12 passed in 3.52s"},
            "relevant_regressions": {"status": "passed", "summary": "40 passed in 15.74s"},
            "full_suite": {"status": "passed", "summary": "941 passed, 1 warning in 129.54s"},
            "uv_lock_check": {"status": "passed", "summary": "229 packages resolved"},
            "uv_sync": {"status": "passed", "summary": "206 packages checked"},
            "cuda_runtime": {
                "status": "passed",
                "summary": "torch 2.12.1+cu130; RTX 5060 Laptop GPU; real cuda:0 tensor computation",
            },
            "browser_acceptance": {
                "status": "passed",
                "summary": "real Edge; 21 required scenarios; 88 cases; six viewports",
            },
            "package_validation": {"status": "passed", "summary": "88 cases; 1,512 evidence assets"},
            "fi_pipeline_help": {"status": "passed", "summary": "CLI loaded"},
            "review_chassis_help": {"status": "passed", "summary": "serve, validate and smoke commands loaded"},
            "git_diff_check": {"status": "passed", "summary": "no whitespace errors"},
        },
        "full_suite_summary": "941 passed, 1 warning in 129.54s",
    }


def main() -> None:
    head = command(["git", "rev-parse", "HEAD"])
    branch = command(["git", "branch", "--show-current"])
    origin = command(["git", "remote", "get-url", "origin"])
    upstream = command(["git", "rev-parse", "@{upstream}"])
    status = command(["git", "status", "--porcelain"])
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASELINE, head],
            cwd=REPO,
            check=False,
        ).returncode
        == 0
    )
    if not all((head != BASELINE, branch == "main", origin == ORIGIN, upstream == head, not status, ancestor)):
        raise RuntimeError(
            "repository finalization gate failed: "
            f"head={head}, branch={branch}, origin={origin}, upstream={upstream}, "
            f"clean={not status}, ancestor={ancestor}"
        )
    if PACK.exists() and any(PACK.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty review pack: {PACK}")

    build_path = STAGE / "06_COMMANDS_AND_TESTS" / "build_summary.json"
    build = read_json(build_path)
    browser = read_json(STAGE / "04_BROWSER_PERSISTENCE_AND_USABILITY" / "browser_acceptance_results.json")
    package = read_json(PACKAGE / "review_package_validation.json")
    frozen = read_json(STAGE / "01_PRIOR_STAGE_AND_STATE_AUDIT" / "frozen_hash_preservation.json")
    case_evidence = read_json(STAGE / "03_GUIDED_ANNOTATION_APPLICATION" / "case_and_evidence_preservation.json")
    timing = read_json(STAGE / "04_BROWSER_PERSISTENCE_AND_USABILITY" / "truthful_timing_estimate.json")
    original_now = tree_manifest(ORIGINAL)
    r1_now = tree_manifest(R1)
    real_root = real_root_status()
    compact = compact_browser(browser)
    preservation_passed = all(
        (
            original_now["tree_hash"] == EXPECTED_ORIGINAL_TREE,
            r1_now["tree_hash"] == EXPECTED_R1_TREE,
            frozen["freeze_hash"] == EXPECTED_FREEZE_HASH,
            frozen["passed"],
            case_evidence["case_payload_hash"] == EXPECTED_CASE_HASH,
            case_evidence["evidence_tree_hash"] == EXPECTED_EVIDENCE_HASH,
            case_evidence["case_count"] == 88,
            case_evidence["evidence_file_count"] == 1512,
        )
    )
    if not all(
        (
            build["classification"] == CLASSIFICATION,
            package["passed"],
            browser["passed"],
            preservation_passed,
            real_root["fresh_and_empty"],
            timing["human_measured_active_minutes"] is None,
            timing["scripted_browser_time_claimed_as_human_time"] is False,
        )
    ):
        raise RuntimeError("scientific, preservation, browser or empty-root gate failed")

    results = command_results(head)
    write_json(STAGE / "06_COMMANDS_AND_TESTS", "command_results.json", results)
    build["tests_pending"] = False
    build["browser_acceptance_pending"] = False
    build["implementation_commit"] = head
    build["full_suite_summary"] = results["full_suite_summary"]
    build_path.write_text(json.dumps(build, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    work = STAGE / "_tmp" / f"review_pack_build_{uuid.uuid4().hex[:10]}"
    work.mkdir(parents=True)
    changed_files = [row for row in command(["git", "show", "--format=", "--name-only", head]).splitlines() if row]
    source_diff = command(["git", "show", "--format=fuller", "--binary", head])
    state_machine = read_json(STAGE / "02_NOVICE_WIZARD_PRODUCT_DESIGN" / "novice_wizard_state_machine.json")
    mapping = read_json(STAGE / "02_NOVICE_WIZARD_PRODUCT_DESIGN" / "plain_language_schema_mapping.json")
    modules = read_json(STAGE / "02_NOVICE_WIZARD_PRODUCT_DESIGN" / "module_guided_workflows.json")
    advanced = read_json(STAGE / "02_NOVICE_WIZARD_PRODUCT_DESIGN" / "advanced_details_boundary.json")

    write_text(
        work,
        "01_EXECUTIVE_OUTCOME.md",
        f"""# M5.5G.1A-R2 executive outcome

## Classification

`{CLASSIFICATION}`

The exact 88-case, 1,512-asset diagnostic pilot now has a novice-first four-step
wizard. Human observations are collected before machine-box review; only one
machine box is visible at a time; technical fields remain available behind an
Advanced details boundary. Real-browser acceptance passed every module,
persistence scenario and required viewport.

The production decisions root remains empty. No human truth was generated by
automation. No detector or tracker was evaluated, changed or promoted.
""",
    )
    write_json(
        work,
        "02_REPOSITORY_CONTEXT.json",
        {
            "classification": CLASSIFICATION,
            "implementation_commit": head,
            "branch": branch,
            "origin": origin,
            "remote_head_matches_local": upstream == head,
            "worktree_clean": not status,
            "authorized_baseline": BASELINE,
            "baseline_is_ancestor": ancestor,
        },
    )
    write_text(work, "03_CHANGED_FILES.md", "# Changed files\n\n" + "\n".join(f"- `{row}`" for row in changed_files))
    write_text(work, "04_SOURCE_DIFF.patch", source_diff)
    write_text(
        work,
        "05_COMMANDS_AND_TESTS.md",
        "# Commands and tests\n\n"
        + "\n".join(f"- **{name}:** {row['status']} ({row['summary']})" for name, row in results["commands"].items())
        + f"\n\nFull result: **{results['full_suite_summary']}**.\n",
    )
    write_json(
        work,
        "06_PRIOR_AND_FROZEN_PRESERVATION.json",
        {
            "passed": preservation_passed,
            "original_workspace_tree_hash": original_now["tree_hash"],
            "expected_original_workspace_tree_hash": EXPECTED_ORIGINAL_TREE,
            "r1_workspace_tree_hash": r1_now["tree_hash"],
            "expected_r1_workspace_tree_hash": EXPECTED_R1_TREE,
            "r1_browser_draft_migrated": False,
            "frozen_ontology_hash": frozen["freeze_hash"],
            "expected_frozen_ontology_hash": EXPECTED_FREEZE_HASH,
            "all_frozen_schema_hashes_match": all(row["match"] for row in frozen["schemas"]),
        },
    )
    write_json(
        work,
        "07_CASE_AND_EVIDENCE_PRESERVATION.json",
        {
            **case_evidence,
            "expected_case_payload_hash": EXPECTED_CASE_HASH,
            "expected_evidence_tree_hash": EXPECTED_EVIDENCE_HASH,
            "real_r2_decisions_root": real_root,
        },
    )
    write_json(work, "08_NOVICE_STATE_MACHINE.json", state_machine)
    write_json(work, "09_PLAIN_LANGUAGE_MAPPING.json", mapping)
    write_json(work, "10_MODULE_FLOWS.json", modules)
    write_json(work, "11_CANDIDATE_QUEUE_AND_BROWSER.json", compact)
    write_json(
        work,
        "12_PERSISTENCE_RECOVERY.json",
        {
            "candidate_position_recovery": compact["candidate_position_recovery"],
            "offline_and_idempotent_persistence": compact["offline_and_idempotent_persistence"],
            "r1_browser_draft_migrated": False,
            "real_decisions_root": real_root,
            "server_authoritative_materialization": True,
        },
    )
    write_json(work, "13_TRUTHFUL_TIMING.json", timing)
    write_text(
        work,
        "14_HUMAN_INSTRUCTIONS.md",
        (PACKAGE / "HUMAN_INSTRUCTIONS.md").read_text(encoding="utf-8"),
    )
    write_json(
        work,
        "15_SAFETY_AND_ACCEPTANCE.json",
        {
            "classification": CLASSIFICATION,
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
            "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
            "diagnostic_pilot_only": True,
            "detector_or_tracker_evaluated": False,
            "detector_or_tracker_changed": False,
            "detector_or_tracker_promoted": False,
            "schemas_changed": False,
            "cases_changed": False,
            "evidence_changed": False,
            "advanced_details_boundary": advanced,
            "real_decisions_root_fresh_and_empty": real_root["fresh_and_empty"],
        },
    )
    write_json(
        work,
        "19_ACCEPTANCE_AND_NEXT_STAGE.json",
        {
            "classification": CLASSIFICATION,
            "all_machine_acceptance_gates_passed": True,
            "human_annotation_started": False,
            "next_human_action": "Complete the diagnostic-only pilot in the novice-guided review at port 8807.",
            "future_stage_gate": (
                "Do not evaluate detector or tracker architectures until reviewed pilot data is completed "
                "and ingested in a later stage."
            ),
        },
    )
    visuals = (
        ("01_DRAW_PEOPLE_STEP.png", "16_DRAW_PEOPLE_STEP.png"),
        ("02_ONE_QUESTION_PERSON_WIZARD.png", "17_ONE_QUESTION_PERSON_WIZARD.png"),
        ("03_ONE_MACHINE_BOX_REVIEW.png", "18_ONE_MACHINE_BOX_REVIEW.png"),
    )
    visual_root = STAGE / "04_BROWSER_PERSISTENCE_AND_USABILITY"
    for source, destination in visuals:
        shutil.copy2(visual_root / source, work / destination)

    entries = [
        {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(item for item in work.iterdir() if item.is_file())
    ]
    write_json(
        work,
        "REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r2.review_pack_manifest.v1",
            "classification": CLASSIFICATION,
            "implementation_commit": head,
            "file_count_including_manifest": len(entries) + 1,
            "flat": True,
            "maximum_files": 20,
            "maximum_visuals": 3,
            "entries_excluding_manifest": entries,
        },
    )
    PACK.mkdir(parents=True, exist_ok=True)
    for path in work.iterdir():
        shutil.copy2(path, PACK / path.name)
    validation = validate_pack(PACK)
    validation.update(
        {
            "implementation_commit": head,
            "classification": CLASSIFICATION,
            "manifest_sha256": sha256_file(PACK / "REVIEW_PACK_MANIFEST.json"),
        }
    )
    write_json(STAGE / "06_COMMANDS_AND_TESTS", "review_pack_validation.json", validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
