"""Create and validate the final M5.5G.1A-R1 ChatGPT review pack."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 3"
    / "M5_5G1A_R1_ANNOTATION_UI_CORRECTNESS_AND_PILOT_LAUNCH_REPAIR_v1"
)
PACKAGE = STAGE / "05_CORRECTED_DETECTION_GOLD_PILOT_ANNOTATION_PACKAGE"
PACK = STAGE / "07_REVIEW_PACK_FOR_CHATGPT"
PRIOR_STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 3"
    / "M5_5G1A_DETECTION_GOLD_FOUNDATION_ONTOLOGY_FREEZE_AND_PILOT_ANNOTATION_v1"
)
BASELINE = "893e15959d43bee2e3ff9f609f71d4768c3cca5d"
ORIGIN = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
CLASSIFICATION = "PASS_DETECTION_GOLD_PILOT_R1_READY"
EXPECTED_FILES = (
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_OUTCOME.md",
    "02_REPOSITORY_CONTEXT.json",
    "03_CHANGED_FILES.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TESTS.md",
    "06_CONFIRMED_DEFECT_DISPOSITION.json",
    "07_PRIOR_AND_FROZEN_PRESERVATION.json",
    "08_CASE_AND_EVIDENCE_PRESERVATION.json",
    "09_EXPLICIT_SELECTION_AND_BINDING.json",
    "10_TEMPORAL_MANUAL_GEOMETRY.json",
    "11_DENSE_COVERAGE_PERSISTENCE.json",
    "12_BROWSER_AND_CRASH_RECOVERY.json",
    "13_TRUTHFUL_TIMING.json",
    "14_HUMAN_INSTRUCTIONS.md",
    "15_SAFETY_AND_ACCEPTANCE.json",
    "16_EXPLICIT_OBJECT_AND_TARGET_SELECTION.png",
    "17_DENSE_NONLATEST_MASK_AND_COVERAGE.png",
    "18_TEMPORAL_MANUAL_AND_REFINED_GEOMETRY.png",
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


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


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


def compact_browser(report: dict[str, Any]) -> dict[str, Any]:
    scenarios = report["scenario_results"]
    player = report["player_selection_and_binding"]
    temporal = report["temporal_manual_and_refined"]
    recovery = report["crash_recovery"]
    return {
        "passed": report["passed"],
        "scenario_count": len(scenarios),
        "all_scenarios_passed": all(scenarios.values()),
        "scenario_results": scenarios,
        "viewport_profiles": [
            {
                "profile": row["profile"],
                "passed": row["passed"],
                "focal_scope_badge_visible": row["focal_scope_badge_visible"],
                "panorama_roi_count": row["panorama_scope"]["roiCount"],
                "body_horizontal_overflow_pixels": row["bodyHorizontalOverflowPixels"],
                "image_overlay_max_delta": row["imageOverlayMaxDelta"],
            }
            for row in report["scope_and_viewports"]
        ],
        "explicit_selection": {
            "three_people_created": player["created"]["created"] == 3,
            "duplicate_bound_to_first": player["duplicate_targets_first_instance"],
            "merged_bound_to_explicit_two_of_three": player["merged_targets_explicit_two_of_three"],
            "middle_removal_cleared_affected_binding": player["affected_merged_binding_cleared"],
        },
        "temporal": {
            "manual_state": temporal["manual_frame_state"],
            "manual_candidate_uuids": temporal["manual_frame_candidate_uuids"],
            "refined_state": temporal["refined_frame_state"],
            "refined_candidate_uuids": temporal["refined_frame_candidate_uuids"],
            "wrong_frame_rejection_passed": report["wrong_frame_rejection"]["passed"],
        },
        "dense_coverage": {
            "before_reload": report["dense_selection_and_coverage"]["coverage_before_reload"],
            "after_reload": report["dense_reload_coverage"],
            "after_browser_restart": report["dense_browser_restart_coverage"],
            "after_server_restart": recovery["dense_coverage_after_server_restart"],
        },
        "crash_recovery": {
            "offline_outbox_replayed": recovery["passed"],
            "duplicate_ack": recovery["duplicate_ack"].get("duplicate_event"),
            "completion_blocked": recovery["completion_status_code"] >= 400,
        },
        "production_decisions_preserved": report["production_decisions_preservation"]["passed"],
        "prior_stage_preserved": report["prior_stage_preservation"]["passed"],
        "human_measured_active_minutes": report["human_measured_active_minutes"],
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
    privacy_hits = []
    forbidden_text = (
        "C:" + "\\Users\\",
        "/" + "Users/",
        "BEGIN " + "PRIVATE KEY",
        "pass" + "word=",
    )
    for path in files:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in forbidden_text:
            if token.lower() in text.lower():
                privacy_hits.append({"filename": path.name, "token": token})
    visual_results = [validate_visual(path) for path in visuals]
    checks = {
        "exact_expected_files": set(path.name for path in files) == set(EXPECTED_FILES),
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
        "schema_version": "football_intelligence.m5_5g1a_r1.review_pack_validation.v1",
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

    build = read_json(STAGE / "06_COMMANDS_AND_TESTS" / "build_summary.json")
    commands = read_json(STAGE / "06_COMMANDS_AND_TESTS" / "command_results.json")
    prior = read_json(STAGE / "01_AUDIT_AND_PRIOR_STAGE_VERIFICATION" / "prior_stage_preservation_validation.json")
    frozen = read_json(STAGE / "01_AUDIT_AND_PRIOR_STAGE_VERIFICATION" / "frozen_hash_validation.json")
    package = read_json(PACKAGE / "review_package_validation.json")
    defects = read_json(STAGE / "02_UI_CORRECTNESS_REPAIR" / "confirmed_defect_disposition.json")
    design = read_json(STAGE / "02_UI_CORRECTNESS_REPAIR" / "explicit_selection_and_validation_design.json")
    browser = read_json(STAGE / "03_BROWSER_AND_PERSISTENCE_REGRESSION" / "browser_acceptance_results.json")
    timing = read_json(STAGE / "04_TIMING_AND_HUMAN_INSTRUCTIONS" / "truthful_timing_report.json")
    production_state = read_json(PACKAGE / "decisions" / "review_decisions.json")
    real_root_empty = not production_state["annotations"] and production_state["event_sequence"] == 0
    events_empty = (PACKAGE / "decisions" / "review_decision_events.jsonl").stat().st_size == 0
    completion_absent = not any((PACKAGE / "decisions").glob("completed_review*"))
    if not all(
        (
            build["classification"] == CLASSIFICATION,
            commands["passed"],
            prior["passed"],
            frozen["passed"],
            package["passed"],
            defects["all_confirmed_defects_addressed"],
            browser["passed"],
            timing["human_measured_active_minutes"] is None,
            real_root_empty,
            events_empty,
            completion_absent,
        )
    ):
        raise RuntimeError("scientific or validation gate failed before review-pack finalization")

    work = STAGE / "_tmp" / f"review_pack_build_{uuid.uuid4().hex[:10]}"
    work.mkdir(parents=True)
    changed_files = command(["git", "show", "--format=", "--name-only", head]).splitlines()
    source_diff = command(["git", "show", "--format=fuller", "--binary", head])
    compact = compact_browser(browser)

    write_text(
        work,
        "01_EXECUTIVE_OUTCOME.md",
        f"""# M5.5G.1A-R1 executive outcome

## Classification

`{CLASSIFICATION}`

All seven confirmed launch-blocking defects were repaired without changing the
frozen ontology, schemas, case payloads or evidence bytes. Real-browser
acceptance passed all {len(browser['scenario_results'])} scenarios across six
required viewport profiles. The corrected production decisions root remains
empty at `0/88`; no human labels were generated by automation.

No detector or tracker was evaluated, changed or promoted.
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
    write_text(
        work,
        "03_CHANGED_FILES.md",
        "# Changed files\n\n" + "\n".join(f"- `{path}`" for path in changed_files),
    )
    write_text(work, "04_SOURCE_DIFF.patch", source_diff)
    write_text(
        work,
        "05_COMMANDS_AND_TESTS.md",
        "# Commands and tests\n\n"
        + "\n".join(
            f"- **{name}:** {row['status']} ({row.get('summary', 'completed')})"
            for name, row in commands["commands"].items()
        )
        + f"\n\nFull result: **{commands['full_suite_summary']}**.\n",
    )
    write_json(work, "06_CONFIRMED_DEFECT_DISPOSITION.json", defects)
    write_json(
        work,
        "07_PRIOR_AND_FROZEN_PRESERVATION.json",
        {
            "prior_stage_byte_identical": prior["byte_identical"],
            "prior_stage_tree_hash_before": prior["before"]["tree_hash"],
            "prior_stage_tree_hash_after": prior["after"]["tree_hash"],
            "frozen_ontology_hash": frozen["freeze_hash"],
            "expected_frozen_ontology_hash": frozen["expected_freeze_hash"],
            "all_frozen_schema_hashes_match": all(row["match"] for row in frozen["schema_checks"]),
        },
    )
    write_json(
        work,
        "08_CASE_AND_EVIDENCE_PRESERVATION.json",
        {
            "case_count": package["case_count"],
            "case_order_identical": package["case_order_identical"],
            "case_payload_identical": package["case_payload_identical"],
            "case_payload_hash_before": package["case_payload_hash_before"],
            "case_payload_hash_after": package["case_payload_hash_after"],
            "evidence_file_count": package["evidence_after"]["file_count"],
            "evidence_bytes_identical": package["evidence_bytes_identical"],
            "evidence_tree_hash_before": package["evidence_before"]["tree_hash"],
            "evidence_tree_hash_after": package["evidence_after"]["tree_hash"],
            "fresh_r1_decisions_root_empty": real_root_empty and events_empty and completion_absent,
        },
    )
    write_json(
        work,
        "09_EXPLICIT_SELECTION_AND_BINDING.json",
        {
            "design": design,
            "browser_result": compact["explicit_selection"],
            "proposal_defaults": player_defaults(browser),
        },
    )
    write_json(work, "10_TEMPORAL_MANUAL_GEOMETRY.json", compact["temporal"])
    write_json(work, "11_DENSE_COVERAGE_PERSISTENCE.json", compact["dense_coverage"])
    write_json(work, "12_BROWSER_AND_CRASH_RECOVERY.json", compact)
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
            "detector_changed_or_evaluated": False,
            "tracker_changed_or_evaluated": False,
            "detector_or_tracker_promoted": False,
            "schema_migration_performed": False,
            "pilot_validation_or_holdout_use_forbidden": True,
            "original_stage_preserved": prior["passed"],
            "corrected_real_decisions_root_empty": real_root_empty and events_empty and completion_absent,
        },
    )
    visuals = (
        (
            STAGE / "03_BROWSER_AND_PERSISTENCE_REGRESSION" / "01_EXPLICIT_OBJECT_AND_TARGET_SELECTION.png",
            "16_EXPLICIT_OBJECT_AND_TARGET_SELECTION.png",
        ),
        (
            STAGE / "03_BROWSER_AND_PERSISTENCE_REGRESSION" / "02_DENSE_NONLATEST_MASK_AND_COVERAGE.png",
            "17_DENSE_NONLATEST_MASK_AND_COVERAGE.png",
        ),
        (
            STAGE / "03_BROWSER_AND_PERSISTENCE_REGRESSION" / "03_TEMPORAL_MANUAL_AND_REFINED_GEOMETRY.png",
            "18_TEMPORAL_MANUAL_AND_REFINED_GEOMETRY.png",
        ),
    )
    for source, name in visuals:
        shutil.copy2(source, work / name)

    entries = []
    for path in sorted(item for item in work.iterdir() if item.is_file()):
        entries.append({"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(
        work,
        "REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r1.review_pack_manifest.v1",
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


def player_defaults(browser: dict[str, Any]) -> dict[str, Any]:
    rows = browser["player_selection_and_binding"]["conservative_defaults"]["instances"]
    return {
        "all_instances_started_unresolved": all(row["visibility_state"] == "UNRESOLVED" for row in rows),
        "all_instances_started_unknown_occlusion": all(row["occlusion_type"] == "UNKNOWN" for row in rows),
        "all_instances_started_boundary_uncertain": all(row["pitch_state"] == "BOUNDARY_UNCERTAIN" for row in rows),
        "all_instances_started_unknown_role": all(row["coarse_role"] == "UNKNOWN" for row in rows),
        "automatic_candidate_relation_count": browser["player_selection_and_binding"]["conservative_defaults"][
            "candidate_relation_count"
        ],
    }


if __name__ == "__main__":
    main()
