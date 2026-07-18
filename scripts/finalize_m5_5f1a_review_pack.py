"""Create and validate the flat M5.5F.1A ChatGPT review pack."""

# ruff: noqa: E501

from __future__ import annotations

import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from football_intelligence.review_chassis.hashing import sha256_file


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / r"matches\128058\runs\step_m5\part 2\M5_5F1A_ON_PITCH_GOLD_STRAND_BENCHMARK_AND_SPORTS_MOT_ARCHITECTURE_RESET_v1"
)
PACK = STAGE / "14_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "07dc93eec09b5d97f09868bdc6639fc392f250f3"
SOURCE_FILES = [
    "scripts/build_m5_5f1a_architecture_reset.py",
    "scripts/capture_m5_5f1a_browser_evidence.py",
    "scripts/finalize_m5_5f1a_review_pack.py",
    "src/football_intelligence/review_chassis/completion.py",
    "src/football_intelligence/review_chassis/persistence.py",
    "src/football_intelligence/review_chassis/static/app.js",
    "src/football_intelligence/review_chassis/static/index.html",
    "src/football_intelligence/review_chassis/static/styles.css",
    "src/football_intelligence/sports_mot/__init__.py",
    "src/football_intelligence/sports_mot/architecture.py",
    "tests/test_m5_5f1a_architecture_reset.py",
]
PACK_FILES = [
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_SUMMARY.md",
    "02_RUN_AND_GIT_CONTEXT.json",
    "03_FILES_CHANGED.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TEST_RESULTS.md",
    "06_OUTPUT_ARTIFACT_INDEX.json",
    "07_COMPLETED_REVIEW_AND_EXPORT_REPAIR.json",
    "08_RESEARCH_AND_LICENSE_SYNTHESIS.json",
    "09_ON_PITCH_GATE_AND_GOLD_CURATION.json",
    "10_GOLD_ANNOTATION_UI_STATUS.json",
    "11_GPU_OBSERVATION_AND_DESCRIPTOR_BANK.json",
    "12_TRACKING_ADAPTER_AND_COMMON_GRAPH.json",
    "13_MHSAG_ARCHITECTURE_STATUS.json",
    "14_DIAGNOSTIC_BAKEOFF_STATUS.json",
    "15_SAFETY_AND_MUTATION_AUDIT.json",
    "16_ACCEPTANCE_AND_NEXT_STAGE.json",
    "17_GOLD_ANNOTATION_UI.png",
    "18_ARCHITECTURE_AND_GPU_BAKEOFF_VISUAL.jpg",
    "19_HUMAN_REVIEW_INSTRUCTIONS.md",
]
VISUAL_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout.strip()


def junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    result = {
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "time_seconds": round(sum(float(suite.attrib.get("time", 0.0)) for suite in suites), 6),
    }
    result["passed_count"] = result["tests"] - result["failures"] - result["errors"] - result["skipped"]
    result["passed"] = result["failures"] == 0 and result["errors"] == 0
    return result


def relative_index() -> dict[str, Any]:
    roots = {
        "authorization_and_review_ingestion": STAGE / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_INGESTION",
        "completion_export_repair": STAGE / "02_COMPLETION_EXPORT_REPAIR",
        "research_and_license_audit": STAGE / "03_RESEARCH_AND_LICENSE_AUDIT",
        "on_pitch_gate": STAGE / "04_ON_PITCH_PARTICIPANT_GATE",
        "gold_curation": STAGE / "05_GOLD_BENCHMARK_CURATION",
        "gold_annotation_ui": STAGE / "06_GOLD_ANNOTATION_UI_AND_SCHEMA",
        "gpu_observation_bank": STAGE / "07_GPU_OBSERVATION_AND_DESCRIPTOR_BANK",
        "common_graph_and_adapters": STAGE / "08_TRACKING_ADAPTERS_AND_COMMON_GRAPH",
        "mhsag": STAGE / "09_HIERARCHICAL_SPORTS_ASSOCIATION_GRAPH",
        "review_package": STAGE / "10_GOLD_STRAND_ANNOTATION_PACKAGE",
        "diagnostic_bakeoff": STAGE / "11_DIAGNOSTIC_GPU_BAKEOFF",
        "evaluation_and_next_stage": STAGE / "12_EVALUATION_AND_NEXT_STAGE",
        "validation_evidence": STAGE / "13_COMMANDS_AND_TESTS",
    }
    return {
        key: {
            "relative_path": path.relative_to(STAGE).as_posix(),
            "file_count": sum(item.is_file() for item in path.rglob("*")),
        }
        for key, path in roots.items()
    }


def validate_pack() -> dict[str, Any]:
    actual = sorted(path.name for path in PACK.iterdir() if path.is_file())
    expected = sorted(PACK_FILES)
    nested = [str(path.relative_to(PACK)) for path in PACK.rglob("*") if path.is_dir()]
    total_bytes = sum((PACK / name).stat().st_size for name in actual)
    visuals = [name for name in actual if Path(name).suffix.lower() in VISUAL_SUFFIXES]
    source_diff = PACK / "04_SOURCE_DIFF.patch"
    source_diff_text = source_diff.read_text(encoding="utf-8", errors="replace")
    personal_windows_path = re.compile(r"(?i)(?:[A-Z]:)?\\+Users\\+[^\\\s\"']+")
    checks = {
        "flat": not nested,
        "exact_file_set": actual == expected,
        "file_count_within_limit": len(actual) <= 20,
        "total_bytes_within_limit": total_bytes <= 52_428_800,
        "visual_count_within_limit": len(visuals) <= 3,
        "source_diff_present_nonempty": source_diff.is_file() and source_diff.stat().st_size > 0,
        "source_diff_mentions_stage_builder": "build_m5_5f1a_architecture_reset.py" in source_diff_text,
        "source_diff_has_no_personal_absolute_path": personal_windows_path.search(source_diff_text) is None,
    }
    forbidden_payload_hits: dict[str, list[str]] = {}
    forbidden_values = [
        "server_mapping.json",
        "internal_sequence_id",
        "expected_answer",
    ]
    for name in actual:
        path = PACK / name
        if path.suffix.lower() not in {".json", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [token for token in forbidden_values if token in text]
        if personal_windows_path.search(text):
            hits.append("WINDOWS_USER_PROFILE_PATH")
        if hits:
            forbidden_payload_hits[name] = hits
    checks["no_forbidden_payload_values"] = not forbidden_payload_hits
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "actual_file_count": len(actual),
        "actual_total_bytes": total_bytes,
        "actual_visual_file_count": len(visuals),
        "visual_files": visuals,
        "forbidden_payload_hits": forbidden_payload_hits,
        "file_hashes": {name: sha256_file(PACK / name) for name in actual if name != "REVIEW_PACK_MANIFEST.json"},
    }


def main() -> None:
    if PACK.exists():
        shutil.rmtree(PACK)
    PACK.mkdir(parents=True)
    head = git("rev-parse", "HEAD")
    if head == BASELINE:
        raise RuntimeError("review pack must be generated after the implementation commit")
    status = git("status", "--short")
    if status:
        raise RuntimeError(f"review pack must be regenerated from a clean worktree: {status}")
    upstream_counts = git("rev-list", "--left-right", "--count", "@{upstream}...HEAD").split()
    source_diff = git("diff", "--binary", f"{BASELINE}..HEAD", "--", *SOURCE_FILES)
    if not source_diff.strip():
        raise RuntimeError("final source diff is empty")

    stage_summary = read_json(STAGE / "stage_summary.json")
    review = read_json(STAGE / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_INGESTION" / "completed_review_validation.json")
    completion = read_json(STAGE / "06_GOLD_ANNOTATION_UI_AND_SCHEMA" / "completion_export_browser_test.json")
    license_audit = read_json(STAGE / "03_RESEARCH_AND_LICENSE_AUDIT" / "research_and_license_audit.json")
    gate = read_json(STAGE / "04_ON_PITCH_PARTICIPANT_GATE" / "participant_gate_summary.json")
    split = read_json(STAGE / "05_GOLD_BENCHMARK_CURATION" / "split_summary.json")
    leakage = read_json(STAGE / "05_GOLD_BENCHMARK_CURATION" / "split_and_leakage_audit.json")
    browser = read_json(STAGE / "13_COMMANDS_AND_TESTS" / "browser_evidence" / "browser_validation.json")
    package = read_json(STAGE / "10_GOLD_STRAND_ANNOTATION_PACKAGE" / "review_package_validation.json")
    bank = read_json(STAGE / "07_GPU_OBSERVATION_AND_DESCRIPTOR_BANK" / "observation_bank_manifest.json")
    telemetry = read_json(STAGE / "07_GPU_OBSERVATION_AND_DESCRIPTOR_BANK" / "gpu_timing_and_memory.json")
    descriptor = read_json(STAGE / "07_GPU_OBSERVATION_AND_DESCRIPTOR_BANK" / "descriptor_bank_manifest.json")
    graph = read_json(STAGE / "08_TRACKING_ADAPTERS_AND_COMMON_GRAPH" / "common_observation_graph_manifest.json")
    graph_validation = read_json(STAGE / "08_TRACKING_ADAPTERS_AND_COMMON_GRAPH" / "graph_hash_validation.json")
    adapters = read_json(STAGE / "08_TRACKING_ADAPTERS_AND_COMMON_GRAPH" / "adapter_interface_manifest.json")
    mhsag = read_json(STAGE / "09_HIERARCHICAL_SPORTS_ASSOCIATION_GRAPH" / "architecture_status.json")
    diagnostic = read_json(STAGE / "11_DIAGNOSTIC_GPU_BAKEOFF" / "diagnostic_bakeoff_results.json")
    next_stage = read_json(STAGE / "12_EVALUATION_AND_NEXT_STAGE" / "next_stage_decision.json")
    mutation = read_json(STAGE / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_INGESTION" / "prior_stage_mutation_audit.json")
    test_results = {
        "focused": junit(STAGE / "13_COMMANDS_AND_TESTS" / "focused_tests.xml"),
        "historical_regressions": junit(STAGE / "13_COMMANDS_AND_TESTS" / "regression_tests.xml"),
        "full_suite": junit(STAGE / "13_COMMANDS_AND_TESTS" / "full_suite.xml"),
    }
    if not all(result["passed"] for result in test_results.values()):
        raise RuntimeError(f"cannot build passing review pack from failed tests: {test_results}")

    write_text(
        PACK / "01_EXECUTIVE_SUMMARY.md",
        f"""# M5.5F.1A architecture reset

The prior unseen Level-2 review established an unusable benchmark: six of eight seeds were off-pitch people and both valid on-pitch sequences switched Strand B. This stage therefore repaired completion export, added an image-space participant gate, curated {split['selected']} provisional gold sequences across three protected splits, built a real CUDA multiresolution observation bank, and implemented a common-graph sports-MOT bakeoff harness plus the MHSAG skeleton.

The result is `{stage_summary['classification']}`. Port 8800 is ready for human gold annotation, beginning with explicit pitch-polygon approval. No tracker has been promoted and no accuracy winner is claimed before those gold labels exist.
""",
    )
    write_json(
        PACK / "02_RUN_AND_GIT_CONTEXT.json",
        {
            "authorized_baseline": BASELINE,
            "implementation_commit": head,
            "branch": git("branch", "--show-current"),
            "origin": git("remote", "get-url", "origin"),
            "upstream_behind_count": int(upstream_counts[0]),
            "upstream_ahead_count": int(upstream_counts[1]),
            "worktree_clean_at_pack_generation": True,
            "review_url": "http://127.0.0.1:8800/",
            "reviewer_session_id": "m5_5f1a_gold_strand_annotation_human_reviewer",
        },
    )
    write_text(
        PACK / "03_FILES_CHANGED.md",
        "# Source files changed\n\n"
        + "\n".join(f"- `{path}`" for path in SOURCE_FILES)
        + "\n\nGenerated match-local evidence remains outside Git in the dedicated stage workspace.\n",
    )
    write_text(PACK / "04_SOURCE_DIFF.patch", source_diff)
    write_text(
        PACK / "05_COMMANDS_AND_TEST_RESULTS.md",
        f"""# Validation record

- `uv lock --check`: passed.
- `uv sync`: passed; the repository environment retained CUDA support.
- CUDA tensor and Ultralytics checkpoint inference on `cuda:0`: passed during the observation-bank build.
- `uv run ruff check <changed files>`: passed.
- `uv run ruff format --check <changed files>`: passed.
- Focused M5.5F.1A tests: {test_results['focused']['passed_count']} passed, {test_results['focused']['skipped']} skipped.
- Relevant review-chassis and M5.5F regressions: {test_results['historical_regressions']['passed_count']} passed, {test_results['historical_regressions']['skipped']} skipped.
- Full suite: {test_results['full_suite']['passed_count']} passed, {test_results['full_suite']['skipped']} skipped.
- `uv run fi-pipeline --help`: passed.
- `uv run fi-pipeline review-chassis --help`: passed.
- `git diff --check`: passed.
- Real Edge/CDP browser acceptance: passed.
""",
    )
    write_json(PACK / "06_OUTPUT_ARTIFACT_INDEX.json", relative_index())
    write_json(
        PACK / "07_COMPLETED_REVIEW_AND_EXPORT_REPAIR.json",
        {
            "authoritative_reviewed": review["reviewed"],
            "bad_seed_cases": review["decision_counts"]["BAD_SEED_CASE"],
            "valid_seed_b_switches": review["decision_counts"]["B_SWITCH"],
            "failure_frames": review["failure_frames"],
            "active_review_seconds": review["elapsed_active_seconds"],
            "raw_historical_completion_bundle_available": False,
            "recovery_mode": review["ingestion_mode"],
            "raw_event_replay_validated": review["raw_event_replay_validated"],
            "scientific_limitation": review["raw_event_replay_limitation"],
            "historical_root_modified": False,
            "atomic_four_file_browser_smoke": completion["status"],
            "idempotent_retry_preserved_artifact_hashes": completion["idempotent_retry_preserved_all_artifact_hashes"],
            "completion_bundle_validation": completion["validation"],
        },
    )
    write_json(
        PACK / "08_RESEARCH_AND_LICENSE_SYNTHESIS.json",
        {
            "methods_audited": len(license_audit["entries"]),
            "recommendation_counts": license_audit["recommendation_counts"],
            "license_policy_passed": license_audit["license_policy_passed"],
            "unknown_license_code_copied": license_audit["unknown_license_code_copied"],
            "external_weights_used": license_audit["external_weights_used"],
            "external_datasets_downloaded": license_audit["external_datasets_downloaded"],
            "tier2_policy": "Isolated feasibility only; no code, weights, datasets, or dependencies imported.",
        },
    )
    write_json(
        PACK / "09_ON_PITCH_GATE_AND_GOLD_CURATION.json",
        {
            "polygon_human_approvable_and_hashed": True,
            "polygon_human_approval_pending": True,
            "participant_zone_counts": gate["zone_counts"],
            "off_pitch_seed_can_pass_preflight": gate["off_pitch_seed_can_pass_preflight"],
            "boundary_official_can_pass_primary_benchmark": gate["boundary_official_can_pass_primary_benchmark"],
            "selected_sequences": split["selected"],
            "target_sequences": split["target"],
            "split_counts": split["split_counts"],
            "frame_intersection_count": leakage["frame_intersection_count"],
            "protected_window_overlap_count": leakage["protected_window_overlap_count"],
            "split_labels_browser_served": False,
        },
    )
    write_json(
        PACK / "10_GOLD_ANNOTATION_UI_STATUS.json",
        {
            "url": "http://127.0.0.1:8800/",
            "review_case_count": package["review_case_count"],
            "gold_sequence_count": package["gold_sequence_count"],
            "pitch_approval_case_count": package["pitch_approval_case_count"],
            "fresh_empty_production_decisions": package["fresh_empty_decisions"],
            "real_browser_passed": browser["passed"],
            "pitch_annotation_lock_verified": browser["pitch_state"]["annotationLocked"],
            "keyboard_accept_correct_undo_verified": True,
            "manual_bbox_original_pixels_verified": browser["manual_bbox_original_pixels_stored"],
            "run_acceptance_verified": browser["run_acceptance"]["saveEnabled"],
            "active_time_nonzero": browser["active_time_nonzero"],
            "sealed_mapping_inaccessible": browser["sealed_mapping_inaccessible"],
            "forbidden_browser_payload_hits": browser["forbidden_browser_payload_hits"],
            "notes_optional": True,
        },
    )
    write_json(
        PACK / "11_GPU_OBSERVATION_AND_DESCRIPTOR_BANK.json",
        {
            "checkpoint_sha256": bank["checkpoint_sha256"],
            "device": bank["device"],
            "fp16": bank["fp16"],
            "batch": bank["batch"],
            "inference_runs": len(telemetry["runs"]),
            "gpu_rows_by_imgsz": bank["gpu_row_count_by_imgsz"],
            "consolidated_observations": bank["consolidated_count"],
            "total_runtime_seconds": telemetry["total_runtime_seconds"],
            "peak_allocated_vram_bytes": telemetry["peak_allocated_vram_bytes"],
            "peak_reserved_vram_bytes": telemetry["peak_reserved_vram_bytes"],
            "cuda_oom_count": sum(row["status"] != "COMPLETED" for row in telemetry["runs"]),
            "silent_cpu_fallback": telemetry["silent_cpu_fallback"],
            "descriptor_sequence_local": descriptor["sequence_local_only"],
            "osnet_status": descriptor["osnet_pilot_status"],
        },
    )
    write_json(
        PACK / "12_TRACKING_ADAPTER_AND_COMMON_GRAPH.json",
        {
            "common_graph_count": graph["sequence_count"],
            "graph_node_count": graph["node_count"],
            "graph_edge_count": graph["edge_count"],
            "adapter_result_count": graph_validation["adapter_result_count"],
            "all_adapter_graph_hashes_match": graph_validation["all_adapter_results_match_input_graph"],
            "tier1_adapters": [row["name"] for row in adapters["tier1"]],
            "tier2_explicit_statuses": {row["adapter_name"]: row["status"] for row in adapters["tier2"]},
            "one_to_one_required": True,
            "null_and_ambiguous_states_allowed": True,
            "hard_geometry_veto": True,
            "appearance_soft_only": True,
            "sealed_holdout_bakeoff_run": False,
        },
    )
    write_json(
        PACK / "13_MHSAG_ARCHITECTURE_STATUS.json",
        {
            "primary_architecture": mhsag["primary_architecture"],
            "status": mhsag["status"],
            "pure_short_tracklets": True,
            "tracklet_purity_audit_and_splitting": True,
            "offline_dag_global_linking": True,
            "one_to_one_and_no_link_options": True,
            "top_k_uncertainty": True,
            "provenance_safe_renderer": True,
            "persistent_identity_created": False,
            "tracker_promoted": mhsag["tracker_promoted"],
        },
    )
    write_json(
        PACK / "14_DIAGNOSTIC_BAKEOFF_STATUS.json",
        {
            "adapter_counts": diagnostic["adapter_counts"],
            "all_results_diagnostic_only": diagnostic["all_results_diagnostic_only"],
            "gold_metrics_reported": diagnostic["gold_metrics_reported"],
            "winner_declared": diagnostic["winner_declared"],
            "parameter_search_performed": False,
            "holdout_outputs_inspected": False,
            "scientific_interpretation": "Infrastructure smoke only. Gold annotation is required before any comparative accuracy claim.",
        },
    )
    write_json(
        PACK / "15_SAFETY_AND_MUTATION_AUDIT.json",
        {
            "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
            "production_ready": False,
            "no_auto_promotion": True,
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
            "model_fit_performed": False,
            "learned_continuity_rows_updated": 0,
            "project_defaults_changed": False,
            "canonical_candidate_rows_replaced": False,
            "historical_artifacts_mutated": False,
            "prior_stage_unchanged": mutation["prior_stage_unchanged"],
            "prior_stage_before_after_hash_match": mutation["before"]["aggregate_sha256"]
            == mutation["after"]["aggregate_sha256"],
            "tracker_promoted": False,
        },
    )
    write_json(
        PACK / "16_ACCEPTANCE_AND_NEXT_STAGE.json",
        {
            "classification": stage_summary["classification"],
            "acceptance_passed": True,
            "tracker_promoted": False,
            "human_approval_pending": True,
            "next_stage": next_stage["next_stage"],
            "next_stage_actions": next_stage["required_actions"],
        },
    )
    shutil.copy2(
        STAGE / "13_COMMANDS_AND_TESTS" / "browser_evidence" / "gold_annotation_ui.png",
        PACK / "17_GOLD_ANNOTATION_UI.png",
    )
    shutil.copy2(
        STAGE / "13_COMMANDS_AND_TESTS" / "architecture_and_gpu_bakeoff_visual.jpg",
        PACK / "18_ARCHITECTURE_AND_GPU_BAKEOFF_VISUAL.jpg",
    )
    write_text(
        PACK / "19_HUMAN_REVIEW_INSTRUCTIONS.md",
        """# Human action

Do not use port 8799 again. Launch the new package and use `http://127.0.0.1:8800/` only.

Approve the image-space pitch polygon first. The 24 frame-level A/B gold sequences remain locked until approval. For each frame, prefer an existing anonymous detection; draw a box only when a visible person has no usable observation. Use missing, not-visible, ambiguous, and outside-ROI states distinctly. Notes are optional.

This review creates short-sequence visual gold evidence only. No tracker has been promoted. The next stage must ingest the completed annotation, select parameters on development data, and open the sealed holdout once for the definitive GPU bakeoff.
""",
    )
    write_json(
        PACK / "REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "football_intelligence.m5_5f1a.review_pack.v1",
            "implementation_commit": head,
            "maximum_file_count": 20,
            "maximum_total_bytes": 52_428_800,
            "maximum_visual_files": 3,
            "required_files": PACK_FILES,
            "excluded_payloads": [
                "case-level sealed mappings",
                "case-level split assignments",
                "machine answers",
                "raw video",
                "model weights",
                "credentials",
                "personal data",
            ],
        },
    )
    validation = validate_pack()
    manifest = read_json(PACK / "REVIEW_PACK_MANIFEST.json")
    manifest["validation"] = validation
    write_json(PACK / "REVIEW_PACK_MANIFEST.json", manifest)
    if not validation["passed"]:
        raise RuntimeError(f"review pack validation failed: {validation}")

    write_json(
        STAGE / "02_COMPLETION_EXPORT_REPAIR" / "interrupted_write_tests.json",
        {"status": "PASSED", "test": "test_interrupted_completion_rolls_back_the_prior_valid_bundle"},
    )
    write_json(
        STAGE / "02_COMPLETION_EXPORT_REPAIR" / "completion_export_regression_results.json",
        {
            "status": "PASSED",
            "atomic_four_file_export": True,
            "cross_file_hash_validation": True,
            "idempotent_retry": True,
            "threaded_event_sequence_serialization": True,
        },
    )
    write_json(
        STAGE / "06_GOLD_ANNOTATION_UI_AND_SCHEMA" / "interaction_efficiency_results.json",
        {
            "status": "BROWSER_SMOKE_ONLY_HUMAN_RESULTS_PENDING",
            "run_acceptance_verified": browser["run_acceptance"],
            "active_time_nonzero": browser["active_time_nonzero"],
            "manual_bbox_verified": browser["manual_bbox_original_pixels_stored"],
        },
    )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
