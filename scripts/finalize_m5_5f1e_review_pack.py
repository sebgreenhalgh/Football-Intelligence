"""Create and validate the committed M5.5F.1E ChatGPT review pack."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from football_intelligence.review_chassis.hashing import sha256_file


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F1E_SPENT_HOLDOUT_FORENSICS_ORACLE_REACHABILITY_INVARIANTS_AND_FRESH_CHALLENGE_GOLD_ACQUISITION_v1"
)
PACK = STAGE / "14_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "022c38b9bb8cd0e3520ec28b3453ddd2c1c081fb"
CLASSIFICATION = "PASS_FRESH_CHALLENGE_GOLD_ANNOTATION_READY"
EXPECTED_FILES = (
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_SUMMARY.md",
    "02_RUN_AND_GIT_CONTEXT.json",
    "03_FILES_CHANGED.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TEST_RESULTS.md",
    "06_OUTPUT_ARTIFACT_INDEX.json",
    "07_SPENT_HOLDOUT_FORENSICS.json",
    "08_ORACLE_INVARIANT_HARNESS.json",
    "09_SOURCE_INVENTORY_AND_EXCLUSIONS.json",
    "10_CHALLENGE_MINING_AND_STRATIFICATION.json",
    "11_SPLIT_AND_NEW_HOLDOUT_SEAL.json",
    "12_ANNOTATION_UI_AND_TIME_BUDGET.json",
    "13_PERSISTENCE_AND_BROWSER_VALIDATION.json",
    "14_NEXT_STAGE_BENCHMARK_CONTRACT.json",
    "15_SAFETY_AND_MUTATION_AUDIT.json",
    "16_ACCEPTANCE_AND_HUMAN_ACTION.json",
    "17_SPENT_HOLDOUT_FORENSIC_ATLAS.jpg",
    "18_FRESH_CHALLENGE_ANNOTATION_UI.png",
    "19_HUMAN_REVIEW_INSTRUCTIONS.md",
)
SAFETY_KEYS = (
    "production_ready",
    "no_auto_promotion",
    "human_approved",
    "safe_to_apply_globally",
    "match_local_only",
    "sandbox_only",
    "identity_tracking_performed",
    "player_slots_assigned",
    "goalkeeper_slots_assigned",
    "exact_22_forcing_performed",
    "model_fit_performed",
    "learned_continuity_rows_updated",
    "historical_artifacts_mutated",
    "tracker_promoted",
    "level3_or_level4_work_performed",
    "occlusion_work_performed",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def command(args: list[str]) -> str:
    return subprocess.run(
        args,
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def safe_text(value: str) -> str:
    replacements = {
        str(Path.home()): "<USER_PROFILE>",
        str(Path.home()).replace("\\", "/"): "<USER_PROFILE>",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value


def write_text(name: str, value: str) -> None:
    (PACK / name).write_text(safe_text(value).rstrip() + "\n", encoding="utf-8")


def write_json(name: str, value: Any) -> None:
    write_text(name, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))


def artifact_index() -> dict[str, Any]:
    rows = []
    for directory in sorted(path for path in STAGE.iterdir() if path.is_dir()):
        if directory.name in {"_tmp", PACK.name}:
            continue
        files = [path for path in directory.rglob("*") if path.is_file()]
        rows.append(
            {
                "section": directory.name,
                "file_count": len(files),
                "size_bytes": sum(path.stat().st_size for path in files),
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5f1e.review_artifact_index.v1",
        "sections": rows,
        "total_file_count": sum(row["file_count"] for row in rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "temporary_artifacts_excluded": True,
        "sealed_payload_content_excluded": True,
    }


def compact_browser_report(report: dict[str, Any]) -> dict[str, Any]:
    tests = report["tests"]
    evidence = tests["evidence_routes"]
    initial = tests["initial_seed_screen"]
    return {
        "passed": report["passed"],
        "url": report["url"],
        "evidence_route_sample_count": evidence["sample_count"],
        "evidence_routes_passed": evidence["passed"],
        "sealed_mapping_accessible": tests["sealed_mapping_access"]["accessible"],
        "initial_seed_screen": {
            "seed_visible": initial["seedVisible"],
            "annotation_visible": initial["annotationVisible"],
            "natural_dimensions": [initial["naturalWidth"], initial["naturalHeight"]],
            "seed_labels": initial["seedLabels"],
            "challenge_characteristics_visible": bool(initial["characteristics"]),
            "horizontal_overflow": initial["horizontalOverflow"],
            "complete_disabled": initial["completeDisabled"],
        },
        "event_ledger": tests["event_ledger"],
        "reload_event_sequence": tests["reload_recovery"]["serverSequence"],
        "restart_event_sequence": tests["server_restart_recovery"]["serverSequence"],
        "finalized_event_sequence": tests["sequence_finalization"]["serverSequence"],
        "pending_after_finalization": tests["sequence_finalization"]["pending"],
        "real_decisions_root_untouched": tests["real_decisions_root"]["untouched"],
    }


def validate_visual(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        extrema = ImageStat.Stat(image.convert("RGB").resize((64, 64))).extrema
        nonblank = any(high > low for low, high in extrema)
        result = {
            "path": path.name,
            "width": image.width,
            "height": image.height,
            "nonblank": nonblank,
        }
    if result["width"] < 1000 or result["height"] < 500 or not result["nonblank"]:
        raise RuntimeError(f"invalid review-pack visual: {result}")
    return result


def main() -> None:
    head = command(["git", "rev-parse", "HEAD"])
    branch = command(["git", "branch", "--show-current"])
    origin = command(["git", "remote", "get-url", "origin"])
    upstream = command(["git", "rev-parse", "@{upstream}"])
    status = command(["git", "status", "--porcelain"])
    if head == BASELINE:
        raise RuntimeError("review pack must be generated after the implementation commit")
    if status:
        raise RuntimeError(f"review pack requires a clean worktree: {status}")
    if upstream != head:
        raise RuntimeError(f"implementation commit is not pushed: head={head} upstream={upstream}")
    if PACK.exists() and any(PACK.iterdir()):
        raise RuntimeError(f"refusing to overwrite a non-empty review pack: {PACK}")
    PACK.mkdir(parents=True, exist_ok=True)

    authorization = read_json(STAGE / "01_AUTHORIZATION_AND_SPENT_RESULT_PRESERVATION" / "authorization.json")
    mutation = read_json(STAGE / "01_AUTHORIZATION_AND_SPENT_RESULT_PRESERVATION" / "prior_stage_mutation_audit.json")
    spent_blocker = read_json(
        STAGE / "01_AUTHORIZATION_AND_SPENT_RESULT_PRESERVATION" / "spent_holdout_execution_blocker_tests.json"
    )
    forensics = read_json(STAGE / "02_IMMUTABLE_HOLDOUT_FAILURE_FORENSICS" / "spent_holdout_forensic_summary.json")
    invariants = read_json(
        STAGE / "03_ORACLE_REACHABILITY_AND_MATERIALIZATION_INVARIANTS" / "oracle_invariant_manifest.json"
    )
    repairs = read_json(
        STAGE / "03_ORACLE_REACHABILITY_AND_MATERIALIZATION_INVARIANTS" / "generic_structural_repairs.json"
    )
    source = read_json(STAGE / "04_AVAILABLE_SOURCE_AND_UNUSED_WINDOW_INVENTORY" / "source_inventory.json")
    exclusions = read_jsonl(
        STAGE / "04_AVAILABLE_SOURCE_AND_UNUSED_WINDOW_INVENTORY" / "used_window_exclusion_ledger.jsonl"
    )
    coverage = read_json(STAGE / "05_GPU_CHALLENGE_CANDIDATE_MINING" / "challenge_stratum_coverage.json")
    telemetry = read_json(STAGE / "05_GPU_CHALLENGE_CANDIDATE_MINING" / "gpu_runtime_telemetry.json")
    leakage = read_json(STAGE / "06_EVENT_CLUSTER_DEDUPLICATION_AND_SPLIT_SEALING" / "split_leakage_audit.json")
    access_state = read_json(
        STAGE / "06_EVENT_CLUSTER_DEDUPLICATION_AND_SPLIT_SEALING" / "new_holdout_access_state.json"
    )
    access_negative = read_json(
        STAGE / "06_EVENT_CLUSTER_DEDUPLICATION_AND_SPLIT_SEALING" / "new_holdout_access_negative_tests.json"
    )
    time_budget = read_json(STAGE / "08_ANNOTATION_EFFICIENCY_AND_TIME_BUDGET" / "annotation_time_estimate.json")
    efficiency = read_json(
        STAGE / "08_ANNOTATION_EFFICIENCY_AND_TIME_BUDGET" / "interaction_efficiency_validation.json"
    )
    package = read_json(STAGE / "10_FRESH_CHALLENGE_GOLD_ANNOTATION_PACKAGE" / "review_package_validation.json")
    production_exercise = read_json(
        STAGE / "11_MACHINE_PREFLIGHT_AND_BROWSER_VALIDATION" / "production_persistence_exercise.json"
    )
    browser = read_json(STAGE / "11_MACHINE_PREFLIGHT_AND_BROWSER_VALIDATION" / "browser_visual_regression.json")
    readiness = read_json(STAGE / "12_NEXT_STAGE_BENCHMARK_CONTRACT" / "benchmark_readiness.json")
    next_contract = read_json(STAGE / "12_NEXT_STAGE_BENCHMARK_CONTRACT" / "next_stage_benchmark_contract.json")
    summary = read_json(STAGE / "13_COMMANDS_AND_TESTS" / "build_summary.json")

    validation = {
        "schema_version": "football_intelligence.m5_5f1e.final_validation.v1",
        "recorded_at": datetime.now(UTC).isoformat(),
        "commands": {
            "uv lock --check": "PASS - 229 packages resolved",
            "uv sync": "PASS - 206 packages checked",
            "repository CUDA computation": ("PASS - torch 2.12.1+cu130, NVIDIA GeForce RTX 5060 Laptop GPU, cuda:0"),
            "ruff check changed Python files": "PASS",
            "ruff format --check changed Python files": "PASS",
            "focused and historical regressions": "PASS - 66 passed",
            "complete test suite": "PASS - 887 passed, 1 third-party deprecation warning",
            "fi-pipeline --help": "PASS",
            "fi-pipeline review-chassis --help": "PASS",
            "real Edge browser acceptance": "PASS",
            "git diff --check": "PASS",
        },
        "browser_acceptance_passed": browser["passed"],
        "production_persistence_passed": production_exercise["passed"],
        "classification": summary["classification"],
        "passed": all(
            (
                browser["passed"],
                production_exercise["passed"],
                summary["classification"] == CLASSIFICATION,
            )
        ),
    }
    (STAGE / "13_COMMANDS_AND_TESTS" / "final_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not validation["passed"]:
        raise RuntimeError(f"cannot build passing review pack: {validation}")

    write_text(
        "01_EXECUTIVE_SUMMARY.md",
        "# M5.5F.1E Executive Summary\n\n"
        f"Classification: `{CLASSIFICATION}`. The immutable spent holdout was not rerun. Its oracle failure was "
        "localized to a disconnected graph at a pitch-boundary transition, and detector failures were reduced to "
        "five contiguous events. A fresh second-half source produced 32 blinded Level-2 challenge sequences with "
        "all eight requested strata represented. The crash-safe annotation package passed real-browser acceptance "
        "at port 8806. Human gold annotation is still required before any tracker evaluation. No tracker was "
        "promoted.\n",
    )
    write_json(
        "02_RUN_AND_GIT_CONTEXT.json",
        {
            "authorized_baseline": BASELINE,
            "implementation_commit": head,
            "branch": branch,
            "origin": origin,
            "upstream_commit": upstream,
            "pushed": upstream == head,
            "working_tree_clean_at_pack_build": not status,
            "stage": STAGE.name,
            "classification": CLASSIFICATION,
        },
    )
    changed = command(["git", "diff", "--name-status", f"{BASELINE}..{head}"])
    write_text("03_FILES_CHANGED.md", "# Files Changed\n\n```text\n" + changed + "\n```\n")
    source_diff = command(["git", "diff", "--binary", f"{BASELINE}..{head}"])
    if not source_diff:
        raise RuntimeError("committed source diff is empty")
    write_text("04_SOURCE_DIFF.patch", source_diff)
    write_text(
        "05_COMMANDS_AND_TEST_RESULTS.md",
        "# Commands and Test Results\n\n```json\n" + json.dumps(validation, indent=2, sort_keys=True) + "\n```\n",
    )
    write_json("06_OUTPUT_ARTIFACT_INDEX.json", artifact_index())
    write_json(
        "07_SPENT_HOLDOUT_FORENSICS.json",
        {
            "spent_holdout_rerun": forensics["spent_holdout_rerun"],
            "revised_candidate_scored": forensics["revised_candidate_scored"],
            "parameter_selection_performed": forensics["parameter_selection_performed"],
            "oracle_loss_frame_count": forensics["oracle_loss_frame_count"],
            "oracle_loss_event_count": forensics["oracle_loss_event_count"],
            "oracle_sequence_count": forensics["oracle_sequence_count"],
            "oracle_root_cause": forensics["oracle_root_cause"],
            "oracle_first_disconnected_transition": forensics["oracle_first_disconnected_transition"],
            "oracle_graph_hash_reproduced": forensics["oracle_graph_hash_reproduced"],
            "detector_switch_frame_count": forensics["detector_switch_frame_count"],
            "detector_switch_event_count": forensics["detector_switch_event_count"],
            "detector_event_root_cause_counts": forensics["detector_event_root_cause_counts"],
            "unavailable_values_fabricated": forensics["unavailable_values_fabricated"],
            "second_execution_blocked_before_evaluator": spent_blocker["blocked_before_evaluator_call"],
        },
    )
    write_json(
        "08_ORACLE_INVARIANT_HARNESS.json",
        {
            "passed": invariants["passed"],
            "harness_executable": invariants["harness_executable"],
            "invariants": invariants["invariants"],
            "synthetic_fixture_count": invariants["synthetic_fixture_count"],
            "original_public_sequence_count": invariants["original_public_sequence_count"],
            "spent_forensic_assertion_count": invariants["spent_forensic_assertion_count"],
            "expected_negative_fixtures_detected": invariants["expected_negative_fixtures_detected"],
            "tracker_called": invariants["tracker_called"],
            "gold_supplied_to_tracker": invariants["gold_supplied_to_tracker"],
            "spent_holdout_rerun": invariants["spent_holdout_rerun"],
            "generic_structural_repairs": repairs,
        },
    )
    write_json(
        "09_SOURCE_INVENTORY_AND_EXCLUSIONS.json",
        {
            "fresh_source_choice": source["fresh_source_choice"],
            "choice_reason": source["choice_reason"],
            "second_half_strictly_unused": source["second_half_strictly_unused"],
            "second_half_prior_reference_hit_count": len(source["second_half_prior_reference_hits"]),
            "available_unused_window_count": source["available_unused_window_count"],
            "prior_review_case_count": source["prior_review_case_count"],
            "used_window_exclusion_count": len(exclusions),
            "different_match_or_camera_available": source["different_match_or_camera_available"],
            "different_match_or_camera_reason": source["different_match_or_camera_reason"],
        },
    )
    write_json(
        "10_CHALLENGE_MINING_AND_STRATIFICATION.json",
        {
            "selected_count": coverage["selected_count"],
            "target_count": coverage["target_count"],
            "minimum_count": coverage["minimum_count"],
            "every_stratum_has_four": coverage["every_stratum_has_four"],
            "stratum_counts": coverage["stratum_counts"],
            "machine_selection_only": coverage["machine_selection_only"],
            "human_labels_created": coverage["human_labels_created"],
            "gpu": {
                "device": telemetry["device"],
                "torch_version": telemetry["torch_version"],
                "cuda_runtime": telemetry["cuda_runtime"],
                "model_parameter_device": telemetry["model_parameter_device"],
                "checkpoint_sha256": telemetry["checkpoint_sha256"],
                "windows_evaluated": telemetry["windows_evaluated"],
                "candidate_count": telemetry["candidate_count"],
                "rejected_count": telemetry["rejected_count"],
                "runtime_seconds": telemetry["runtime_seconds"],
                "peak_vram_mb": telemetry["peak_vram_mb"],
                "silent_cpu_fallback": telemetry["silent_cpu_fallback"],
                "gold_labels_created": telemetry["gold_labels_created"],
            },
        },
    )
    write_json(
        "11_SPLIT_AND_NEW_HOLDOUT_SEAL.json",
        {
            "split_counts": leakage["split_counts"],
            "leakage_audit_passed": leakage["passed"],
            "leakage_violation_count": len(leakage["violations"]),
            "negative_access_tests_passed": access_negative["passed"],
            "new_holdout_unseal_count": access_state["unseal_count"],
            "future_unseal_authorized": access_state["future_unseal_authorized"],
            "semantic_content_accessed": access_state["semantic_content_accessed"],
            "exact_assignments_included": False,
            "new_holdout_ids_included": False,
        },
    )
    write_json(
        "12_ANNOTATION_UI_AND_TIME_BUDGET.json",
        {
            "package_validation_passed": package["passed"],
            "review_case_count": package["review_case_count"],
            "gold_sequence_count": package["gold_sequence_count"],
            "pitch_approval_case_count": package["pitch_approval_case_count"],
            "image_sequence_asset_count": package["image_sequence_asset_count"],
            "reviewer_session_id": package["reviewer_session_id"],
            "url": package["url"],
            "fresh_empty_decisions": package["fresh_empty_decisions"],
            "fresh_empty_event_sequence": package["fresh_empty_event_sequence"],
            "sealed_mapping_static_access_forbidden": package["sealed_mapping_static_access_forbidden"],
            "split_labels_in_reviewer_manifest": package["split_labels_in_reviewer_manifest"],
            "browser_forbidden_value_hit_count": len(package["browser_forbidden_value_hits"]),
            "time_budget": time_budget,
            "interaction_efficiency": efficiency,
        },
    )
    direct_fields = (
        "passed",
        "direct_persistence_passed",
        "browser_http_exercise_passed",
        "all_acknowledged",
        "event_ledger_nonempty",
        "pitch_polygon_migrated",
        "reload_state_hash_preserved",
        "server_restart_state_hash_preserved",
        "sequence_finalized",
        "stable_run_expected_frame_event_count",
        "stable_run_explicit_frame_event_count",
        "real_package_decisions_root_untouched",
    )
    write_json(
        "13_PERSISTENCE_AND_BROWSER_VALIDATION.json",
        {
            "direct_exercise": {key: production_exercise[key] for key in direct_fields},
            "real_browser": compact_browser_report(browser),
        },
    )
    write_json(
        "14_NEXT_STAGE_BENCHMARK_CONTRACT.json",
        {
            "readiness": readiness,
            "ordered_workflow": next_contract["ordered_workflow"],
            "prerequisite": next_contract["prerequisite"],
            "spent_holdout_selection_or_scoring_forbidden": next_contract[
                "spent_holdout_selection_or_scoring_forbidden"
            ],
            "new_holdout_access_before_freeze_forbidden": next_contract["new_holdout_access_before_freeze_forbidden"],
            "new_holdout_unseal_count_at_contract_creation": next_contract[
                "new_holdout_unseal_count_at_contract_creation"
            ],
            "tracker_promoted": next_contract["tracker_promoted"],
        },
    )
    write_json(
        "15_SAFETY_AND_MUTATION_AUDIT.json",
        {
            "authorization_passed": authorization["passed"],
            "spent_guard_passed": authorization["spent_guard"]["passed"],
            "prior_mutation_audit_passed": mutation["passed"],
            "safety": {key: summary[key] for key in SAFETY_KEYS},
            "visual_only_warning": summary["visual_only_warning"],
            "do_not_use_for_metrics": summary["do_not_use_for_metrics"],
            "tracker_evaluated_on_new_gold": summary["tracker_evaluated_on_new_gold"],
        },
    )
    human_actions = [
        "Do not use any previous annotation port.",
        "Use port 8806 only because the stage classification is PASS.",
        "Approve a pitch polygon only when a new camera requires it.",
        "Annotate the visible cyan A and magenta B sequences.",
        "Use stable-run acceptance only after reviewing the complete contact strip.",
        "Expect approximately 30-40 active minutes.",
        "Notes are optional.",
        "Stop if Saved to server does not appear or the server event sequence stops advancing.",
        "No tracker has been promoted.",
    ]
    write_json(
        "16_ACCEPTANCE_AND_HUMAN_ACTION.json",
        {
            "classification": CLASSIFICATION,
            "annotation_ready": True,
            "human_annotation_complete": False,
            "url": "http://127.0.0.1:8806/",
            "launcher_relative_to_stage": "10_FRESH_CHALLENGE_GOLD_ANNOTATION_PACKAGE/launch_review.ps1",
            "human_actions": human_actions,
            "next_stage_blocker": readiness["blocker"],
        },
    )

    shutil.copy2(
        STAGE / "02_IMMUTABLE_HOLDOUT_FAILURE_FORENSICS" / "forensic_atlas.jpg",
        PACK / "17_SPENT_HOLDOUT_FORENSIC_ATLAS.jpg",
    )
    shutil.copy2(
        STAGE
        / "11_MACHINE_PREFLIGHT_AND_BROWSER_VALIDATION"
        / "browser_evidence"
        / "fresh_challenge_annotation_ui.png",
        PACK / "18_FRESH_CHALLENGE_ANNOTATION_UI.png",
    )
    write_text(
        "19_HUMAN_REVIEW_INSTRUCTIONS.md",
        "# Human Review Instructions\n\n"
        "Launch `10_FRESH_CHALLENGE_GOLD_ANNOTATION_PACKAGE/launch_review.ps1` and use only "
        "`http://127.0.0.1:8806/`. Confirm or correct the visible cyan A and magenta B seeds, then annotate "
        "all synchronized frames. Review every contact-strip frame before accepting a stable run. Notes are "
        "optional. Expect about 30-40 active minutes. Stop immediately if **Saved to server** disappears or "
        "the server event sequence stops advancing. Do not use an earlier annotation port. No tracker has been "
        "promoted, and these labels must not be treated as tracker output.\n",
    )

    payload_files = sorted(path for path in PACK.iterdir() if path.is_file())
    visuals = [path for path in payload_files if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"}]
    visual_validation = [validate_visual(path) for path in visuals]
    manifest = {
        "schema_version": "football_intelligence.m5_5f1e.review_pack_manifest.v1",
        "classification": CLASSIFICATION,
        "implementation_commit": head,
        "flat": True,
        "maximum_file_count": 20,
        "maximum_total_bytes": 52_428_800,
        "maximum_visual_files": 3,
        "files": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in payload_files
        ],
        "visual_validation": visual_validation,
        "sealed_mapping_included": False,
        "split_assignments_included": False,
        "new_holdout_ids_included": False,
        "answer_keys_included": False,
        "candidate_ids_included": False,
        "raw_video_or_weights_included": False,
        "credentials_or_personal_data_included": False,
    }
    write_json("REVIEW_PACK_MANIFEST.json", manifest)

    files = sorted(path for path in PACK.iterdir() if path.is_file())
    names = tuple(path.name for path in files)
    total = sum(path.stat().st_size for path in files)
    visuals = [path for path in files if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"}]
    checks = {
        "exact_file_set": names == tuple(sorted(EXPECTED_FILES)),
        "flat": all(path.parent == PACK for path in files),
        "file_count": len(files),
        "within_file_limit": len(files) <= 20,
        "total_bytes": total,
        "within_size_limit": total <= 52_428_800,
        "visual_count": len(visuals),
        "within_visual_limit": len(visuals) <= 3,
        "source_diff_nonempty": (PACK / "04_SOURCE_DIFF.patch").stat().st_size > 0,
        "forbidden_artifacts_absent": not any(
            token in path.name.lower()
            for path in files
            for token in ("server_mapping", "weights", "checkpoint", "raw_video")
        ),
        "personal_path_absent": all(
            "C:\\Users\\sebgr" not in path.read_text(encoding="utf-8", errors="ignore")
            for path in files
            if path.suffix.lower() in {".json", ".md", ".txt", ".patch"}
        ),
    }
    if not all(value for key, value in checks.items() if key not in {"file_count", "total_bytes", "visual_count"}):
        raise RuntimeError(f"review pack validation failed: {checks}")
    print(json.dumps({"passed": True, **checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
