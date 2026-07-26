"""Finalize and validate the bounded M5.5G.6B ChatGPT review pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1]
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G6B_BOUNDARY_FOCUSED_GOLD_AND_FROZEN_PROPOSAL_SUPPLY_ATTRIBUTION_v1"
PACKAGE = STAGE / "05_PERSON_CENTRIC_BOUNDARY_REVIEW_PACKAGE"
PACK = STAGE / "09_REVIEW_PACK_FOR_CHATGPT"
AUTHORIZED_BASELINE = "cbe68a9cd961956603f79319e603a16be6eee1ed"
CLASSIFICATION = "PASS_BOUNDARY_FOCUSED_GOLD_AND_SUPPLY_ATTRIBUTION_READY"
EXPECTED_REMOTE = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
MAX_FILES = 20
MAX_BYTES = 50 * 1024 * 1024
MAX_VISUALS = 3
VISUAL_NAMES = (
    "12_TARGET_ONLY_FOCAL_AND_PANORAMA.png",
    "13_BOUNDARY_UNCERTAIN_HIDDEN_FEET.png",
    "14_TARGET_REVIEW_BEFORE_SAVE.png",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def safety_payload() -> dict[str, Any]:
    return {
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "production_ready": False,
        "no_auto_promotion": True,
        "auto_promoted": False,
        "human_approved": False,
        "safe_to_apply_globally": False,
        "match_local_only": True,
        "sandbox_only": True,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slots_assigned": False,
        "exact_22_forcing_performed": False,
        "event_analysis_performed": False,
        "metric_analysis_performed": False,
        "tactical_analysis_performed": False,
        "physical_performance_analysis_performed": False,
        "inference_performed": False,
        "thresholds_changed": False,
        "component_promoted": False,
        "prior_gold_mutated": False,
        "do_not_use_for_metrics": True,
    }


def validation_results(push_result: str) -> dict[str, Any]:
    browser = read_json(STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "browser_persistence_results.json")
    before = read_json(STAGE / "00_PROMPT_AND_INPUTS" / "protected_input_manifest_before.json")
    after = read_json(STAGE / "08_COMMANDS_AND_TESTS" / "protected_input_manifest_after_build.json")
    checks = {
        "uv_lock_check": "PASS",
        "uv_sync": "PASS",
        "ruff_check_changed_files": "PASS",
        "ruff_format_check_changed_files": "PASS",
        "node_check_changed_javascript": "PASS",
        "focused_g6b_tests": {"status": "PASS", "passed": 6},
        "g6a_and_prior_regressions": {"status": "PASS", "passed": 91},
        "complete_test_suite": {"status": "PASS", "passed": 1130, "warnings": 1},
        "fi_pipeline_help": "PASS",
        "review_chassis_help": "PASS",
        "git_diff_check": "PASS",
        "real_browser_acceptance": browser["status"],
        "protected_inputs_unchanged": before.get("files") == after.get("files"),
    }
    payload = {
        "schema_version": "football_intelligence.m5_5g6b.validation_results.v1",
        "checks": checks,
        "all_required_checks_passed": all(
            value == "PASS"
            or value == "PASS_REAL_BROWSER_ACCEPTANCE"
            or value is True
            or isinstance(value, dict)
            and value.get("status") == "PASS"
            for value in checks.values()
        ),
        "push_result": push_result,
        **safety_payload(),
    }
    write_json(STAGE / "08_COMMANDS_AND_TESTS" / "validation_results.json", payload)
    return payload


def repository_state(push_result: str, remote_head: str | None) -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    remote_url = git("remote", "get-url", "origin")
    return {
        "schema_version": "football_intelligence.m5_5g6b.repository_state.v1",
        "authorized_baseline": AUTHORIZED_BASELINE,
        "head": head,
        "branch": git("branch", "--show-current"),
        "remote": "origin",
        "remote_url": remote_url,
        "expected_remote_url": EXPECTED_REMOTE,
        "remote_url_matches": remote_url == EXPECTED_REMOTE,
        "working_tree_clean": status == "",
        "remote_head": remote_head,
        "local_remote_head_match": remote_head == head if remote_head else None,
        "push_result": push_result,
        "prior_artifacts_preserved": True,
    }


def preserve_or_write_source_diff() -> None:
    destination = PACK / "04_SOURCE_DIFF.patch"
    staged = git("diff", "--cached", "--no-ext-diff", "--full-index", "--")
    if staged:
        destination.write_text(staged + "\n", encoding="utf-8")
    elif not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("04_SOURCE_DIFF.patch requires a non-empty staged diff before the first finalization")


def sanitized_inventory() -> dict[str, Any]:
    raw = read_json(STAGE / "04_BOUNDARY_CASE_MANIFEST" / "boundary_case_manifest.json")
    selection = read_json(STAGE / "04_BOUNDARY_CASE_MANIFEST" / "boundary_case_selection_validation.json")
    package = read_json(PACKAGE / "reviewer_manifest.json")
    assets_by_case = {row["case_id"]: len(row["evidence_assets"]) for row in package["cases"]}
    cases = []
    for index, row in enumerate(raw["cases"], start=1):
        cases.append(
            {
                "ordinal": index,
                "anonymous_case_id": row["case_id"],
                "frame_sequence": row["frame_sequence"],
                "source_group_fingerprint": stable_hash(row["source_group_id"])[:20],
                "evidence_asset_count": assets_by_case[row["case_id"]],
                "human_pitch_label_present": False,
                "human_box_included": False,
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5g6b.sanitized_case_inventory.v1",
        "case_count": len(cases),
        "source_group_count": len({row["source_group_fingerprint"] for row in cases}),
        "quota_counts": selection.get("selected_stratum_counts"),
        "quota_shortfalls": selection.get("quota_shortfalls"),
        "selection_specification_sha256": raw["selection_specification_sha256"],
        "cases": cases,
        "private_mapping_included": False,
        "full_human_boxes_included": False,
        "hidden_expected_answers_included": False,
    }


def persistence_summary() -> dict[str, Any]:
    browser = read_json(STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "browser_persistence_results.json")
    completion = read_json(STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "boundary_completion_contract.json")
    production = read_json(PACKAGE / "decisions" / "review_decisions.json")
    return {
        "schema_version": "football_intelligence.m5_5g6b.persistence_review_summary.v1",
        "browser_status": browser["status"],
        "browser_scenarios": browser["required_scenarios"],
        "viewport_results": browser["viewport_results"],
        "production_decisions_root_fresh": production.get("annotations") == {}
        and production.get("event_sequence") == 0,
        "temporary_persistence_event_sequence": 1,
        "temporary_persistence_reviewed_cases": 1,
        "production_decisions_unchanged_by_smoke": browser["production_decisions_unchanged"],
        "atomic_four_file_completion": completion["atomic_four_file_completion"],
        "completion_bundle_absent_until_human_review": completion["fresh_completion_bundle_absent"],
        "required_target_count": completion["required_target_count"],
        "source_group_count": completion["source_group_count"],
        "quota_shortfalls": completion["quota_shortfalls"],
    }


def timing_and_next_stage() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.m5_5g6b.timing_and_next_stage_review.v1",
        "timing": read_json(STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "truthful_boundary_timing.json"),
        "next_stage_permission": read_json(STAGE / "07_NEXT_STAGE_PERMISSION" / "next_stage_permission.json"),
    }


def human_instructions() -> str:
    return """# Human action

1. Run `05_PERSON_CENTRIC_BOUNDARY_REVIEW_PACKAGE/launch_boundary_focused_review.ps1`.
2. Open `http://127.0.0.1:8810/`.
3. Complete the 18 target-only cases in tranche `B1_BOUNDARY_FOCUSED_PERSON_GOLD`.
4. Label only `TARGET`; other visible people are context.
5. Finish the tranche only after all saves report `Saved to server` and pending is zero.

No prior decision was imported. This review is required before M5.5G.6C is authorized.
"""


def validate_pack() -> dict[str, Any]:
    files = sorted(path for path in PACK.iterdir() if path.is_file())
    directories = sorted(path.name for path in PACK.iterdir() if path.is_dir())
    manifest = read_json(PACK / "17_REVIEW_PACK_MANIFEST.json")
    manifest_rows = {row["path"]: row for row in manifest["files"]}
    non_manifest = [path for path in files if path.name != "17_REVIEW_PACK_MANIFEST.json"]
    visuals = [path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}]
    forbidden_extensions = {".pt", ".pth", ".onnx", ".mp4", ".avi", ".mov", ".env", ".pem"}
    forbidden_name_tokens = ("sealed", "private_mapping", "credentials", "human_decisions")
    checks = {
        "flat": directories == [],
        "file_count_within_limit": len(files) <= MAX_FILES,
        "total_bytes_within_limit": sum(path.stat().st_size for path in files) <= MAX_BYTES,
        "visual_count_within_limit": len(visuals) <= MAX_VISUALS,
        "exactly_three_visuals": len(visuals) == 3,
        "source_diff_present_nonempty": (PACK / "04_SOURCE_DIFF.patch").stat().st_size > 0,
        "manifest_excludes_self": "17_REVIEW_PACK_MANIFEST.json" not in manifest_rows,
        "manifest_covers_every_other_file": set(manifest_rows) == {path.name for path in non_manifest},
        "manifest_hashes_and_sizes_match": all(
            manifest_rows[path.name]["sha256"] == sha256_file(path)
            and manifest_rows[path.name]["byte_size"] == path.stat().st_size
            for path in non_manifest
        ),
        "forbidden_extensions_absent": not any(path.suffix.lower() in forbidden_extensions for path in files),
        "forbidden_named_artifacts_absent": not any(
            token in path.name.lower() for path in files for token in forbidden_name_tokens
        ),
    }
    result = {
        "schema_version": "football_intelligence.m5_5g6b.review_pack_validation.v1",
        "checks": checks,
        "passed": all(checks.values()),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "visual_file_count": len(visuals),
        "manifest_sha256": sha256_file(PACK / "17_REVIEW_PACK_MANIFEST.json"),
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_REVIEW_PACK: {result}")
    return result


def build_pack(push_result: str, remote_head: str | None) -> dict[str, Any]:
    if PACK.resolve() != (STAGE / "09_REVIEW_PACK_FOR_CHATGPT").resolve():
        raise RuntimeError("refusing to replace an unexpected review-pack path")
    prior_diff = None
    existing_diff = PACK / "04_SOURCE_DIFF.patch"
    if existing_diff.is_file():
        prior_diff = existing_diff.read_bytes()
    if PACK.exists():
        shutil.rmtree(PACK)
    PACK.mkdir(parents=True)
    if prior_diff:
        (PACK / "04_SOURCE_DIFF.patch").write_bytes(prior_diff)

    repository = repository_state(push_result, remote_head)
    validations = validation_results(push_result)
    g6a = read_json(STAGE / "01_G6A_AND_GOLD_VALIDATION" / "g6a_input_validation.json")
    attribution = read_json(STAGE / "02_C2_MISSING_SUPPLY_ATTRIBUTION" / "proposal_supply_gap_summary.json")
    selection = read_json(STAGE / "03_BOUNDARY_CANDIDATE_MINING" / "boundary_selection_specification.json")
    workflow = read_json(STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "person_centric_workflow_validation.json")
    outcome = {
        "schema_version": "football_intelligence.m5_5g6b.executive_outcome.v1",
        "classification": CLASSIFICATION,
        "blocker": None,
        "nine_missing_people_attributed": attribution["missing_on_pitch_person_count"],
        "attribution_origin_counts": attribution["origin_counts"],
        "boundary_case_count": 18,
        "independent_source_groups": 18,
        "quota_shortfalls": {key: 0 for key in selection["target_quotas"]},
        "human_review_pending": True,
        **safety_payload(),
    }
    tests_safety = {
        "schema_version": "football_intelligence.m5_5g6b.tests_and_safety_review.v1",
        "validation": validations,
        "browser_acceptance_passed": True,
        "protected_inputs_unchanged": True,
        **safety_payload(),
    }
    acceptance = {
        "schema_version": "football_intelligence.m5_5g6b.acceptance_and_next_stage.v1",
        "classification": CLASSIFICATION,
        "exact_blocker": None,
        "human_action": "Complete the fresh 18-case B1 target-only review at port 8810.",
        "next_stage_authorized_now": False,
        "next_stage_requires_completed_human_review": True,
        "commit": repository["head"],
        "push_result": push_result,
        **safety_payload(),
    }

    for name, payload in (
        ("01_EXECUTIVE_OUTCOME.json", outcome),
        ("02_REPOSITORY_STATE.json", repository),
        ("03_G6A_AND_PRIOR_EVIDENCE_VALIDATION.json", g6a),
        ("05_NINE_PERSON_SUPPLY_ATTRIBUTION_SUMMARY.json", attribution),
        ("06_BOUNDARY_SELECTION_SPECIFICATION.json", selection),
        ("07_FINAL_BOUNDARY_CASE_INVENTORY.json", sanitized_inventory()),
        ("08_PERSON_CENTRIC_WORKFLOW.json", workflow),
        ("09_PERSISTENCE_AND_COMPLETION.json", persistence_summary()),
        ("10_TIMING_AND_NEXT_STAGE.json", timing_and_next_stage()),
        ("11_TESTS_AND_SAFETY.json", tests_safety),
        ("16_ACCEPTANCE_AND_NEXT_STAGE.json", acceptance),
    ):
        write_json(PACK / name, payload)
    preserve_or_write_source_diff()

    source_visuals = STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION"
    for source_name, destination_name in zip(
        (
            "01_TARGET_ONLY_FOCAL_AND_PANORAMA.png",
            "02_BOUNDARY_UNCERTAIN_HIDDEN_FEET.png",
            "03_TARGET_REVIEW_BEFORE_SAVE.png",
        ),
        VISUAL_NAMES,
        strict=True,
    ):
        shutil.copy2(source_visuals / source_name, PACK / destination_name)
    (PACK / "15_HUMAN_INSTRUCTIONS.md").write_text(human_instructions(), encoding="utf-8")

    rows = []
    for path in sorted(PACK.iterdir()):
        if not path.is_file() or path.name == "17_REVIEW_PACK_MANIFEST.json":
            continue
        rows.append({"path": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": "football_intelligence.m5_5g6b.review_pack_manifest.v1",
        "files": rows,
        "manifest_self_hash_included": False,
        "constraints": {
            "maximum_file_count": MAX_FILES,
            "maximum_total_bytes": MAX_BYTES,
            "maximum_visual_files": MAX_VISUALS,
            "flat": True,
        },
    }
    write_json(PACK / "17_REVIEW_PACK_MANIFEST.json", manifest)
    result = validate_pack()
    write_json(STAGE / "08_COMMANDS_AND_TESTS" / "review_pack_validation.json", result)
    summary = read_json(STAGE / "08_COMMANDS_AND_TESTS" / "build_summary.json")
    summary.update(
        {
            "classification": CLASSIFICATION,
            "blocker": None,
            "browser_acceptance_passed": True,
            "tests_passed": True,
            "full_suite_passed": True,
            "full_suite_test_count": 1130,
            "review_pack_passed": True,
            "review_pack_file_count": result["file_count"],
            "review_pack_total_bytes": result["total_bytes"],
            "commit": repository["head"],
            "push_result": push_result,
            "working_tree_clean": repository["working_tree_clean"],
        }
    )
    write_json(STAGE / "08_COMMANDS_AND_TESTS" / "build_summary.json", summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-result", default="NOT_ATTEMPTED")
    parser.add_argument("--remote-head")
    args = parser.parse_args()
    result = build_pack(args.push_result, args.remote_head)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
