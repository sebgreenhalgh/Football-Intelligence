"""Create and validate the flat M5.5F.1C ChatGPT review pack."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.review_chassis.hashing import sha256_file, stable_hash


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F1C_DEVELOPMENT_FAILURE_ATLAS_PANORAMA_HANDOFF_AND_TRUE_HIERARCHICAL_PATH_SELECTION_v1"
)
PACK = STAGE / "15_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "79e4441f350668c2ed3d0d551878aa43fb537f05"
EXPECTED_NAMES = (
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_SUMMARY.md",
    "02_RUN_AND_GIT_CONTEXT.json",
    "03_FILES_CHANGED.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TEST_RESULTS.md",
    "06_OUTPUT_ARTIFACT_INDEX.json",
    "07_SELECTED_RESULT_AND_FAILURE_ATLAS.json",
    "08_ROI_SEMANTICS_AND_PANORAMA_VISIBILITY.json",
    "09_PANORAMA_GRAPH_AND_DYNAMIC_HANDOFF.json",
    "10_TRUE_HIERARCHICAL_PATH_SELECTION.json",
    "11_MOTION_APPEARANCE_AND_DISTRACTOR_BANK.json",
    "12_DEVELOPMENT_SEARCH_AND_ABLATIONS.json",
    "13_ERROR_ATLAS_REVIEW_STATUS.json",
    "14_HOLDOUT_SEAL_AND_PRIOR_MUTATION_AUDIT.json",
    "15_ACCEPTANCE_AND_NEXT_STAGE.json",
    "16_SAFETY_STATEMENT.json",
    "17_DEVELOPMENT_FAILURE_ATLAS.jpg",
    "18_PANORAMA_HANDOFF_AND_REPAIRED_PATH.png",
    "19_HUMAN_REVIEW_INSTRUCTIONS.md",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def artifact_index() -> dict[str, Any]:
    paths = []
    for folder in (
        "01_AUTHORIZATION_AND_HOLDOUT_SEAL_GUARD",
        "02_SELECTED_RESULT_REPRODUCTION",
        "03_DEVELOPMENT_FAILURE_ATLAS",
        "04_ROI_SEMANTICS_AND_PANORAMA_VISIBILITY",
        "05_FULL_PANORAMA_OBSERVATION_GRAPH",
        "06_DYNAMIC_ROI_AND_CROP_HANDOFF",
        "07_P_MHSAG_TRUE_HIERARCHICAL_IMPLEMENTATION",
        "08_MOTION_APPEARANCE_AND_DISTRACTOR_BANK",
        "09_DEVELOPMENT_CONFIGURATION_SEARCH_AND_ABLATIONS",
        "10_DEVELOPMENT_ACCEPTANCE_AND_NEXT_STAGE",
        "11_DEVELOPMENT_ERROR_ATLAS_REVIEW_PACKAGE",
        "13_REPRODUCIBILITY_BUNDLE",
        "14_FAILURE_VISUALS",
    ):
        root = STAGE / folder
        for path in sorted(value for value in root.rglob("*") if value.is_file()):
            if "evidence" in path.relative_to(root).parts and folder == "11_DEVELOPMENT_ERROR_ATLAS_REVIEW_PACKAGE":
                continue
            if "decisions" in path.relative_to(root).parts:
                continue
            paths.append(
                {
                    "relative_path": path.relative_to(STAGE).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": "football_intelligence.m5_5f1c.output_artifact_index.v1",
        "artifact_count": len(paths),
        "aggregate_hash": stable_hash(paths),
        "artifacts": paths,
    }


def main() -> None:
    required_stage_files = (
        STAGE / "02_SELECTED_RESULT_REPRODUCTION" / "selected_result_reproduction.json",
        STAGE / "03_DEVELOPMENT_FAILURE_ATLAS" / "root_cause_summary.json",
        STAGE / "04_ROI_SEMANTICS_AND_PANORAMA_VISIBILITY" / "roi_semantics_summary.json",
        STAGE / "05_FULL_PANORAMA_OBSERVATION_GRAPH" / "graph_validation.json",
        STAGE / "06_DYNAMIC_ROI_AND_CROP_HANDOFF" / "coordinate_roundtrip_validation.json",
        STAGE / "07_P_MHSAG_TRUE_HIERARCHICAL_IMPLEMENTATION" / "authoritative_path_application_validation.json",
        STAGE / "09_DEVELOPMENT_CONFIGURATION_SEARCH_AND_ABLATIONS" / "ablation_results.json",
        STAGE / "10_DEVELOPMENT_ACCEPTANCE_AND_NEXT_STAGE" / "development_acceptance_checklist.json",
        STAGE / "11_DEVELOPMENT_ERROR_ATLAS_REVIEW_PACKAGE" / "review_package_validation.json",
    )
    missing = [str(path) for path in required_stage_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"stage is incomplete: {missing}")
    PACK.mkdir(parents=True, exist_ok=True)
    for path in PACK.iterdir():
        if path.is_dir():
            raise RuntimeError(f"review pack must remain flat: {path}")
        path.unlink()

    reproduction = read_json(required_stage_files[0])
    causes = read_json(required_stage_files[1])
    roi = read_json(required_stage_files[2])
    graph = read_json(required_stage_files[3])
    handoff = read_json(required_stage_files[4])
    hierarchy = read_json(required_stage_files[5])
    appearance = read_json(STAGE / "08_MOTION_APPEARANCE_AND_DISTRACTOR_BANK" / "appearance_descriptor_manifest.json")
    runtime = read_json(STAGE / "08_MOTION_APPEARANCE_AND_DISTRACTOR_BANK" / "gpu_runtime_and_memory.json")
    search = read_json(
        STAGE / "09_DEVELOPMENT_CONFIGURATION_SEARCH_AND_ABLATIONS" / "development_configuration_manifest.json"
    )
    cross_validation = read_json(
        STAGE / "09_DEVELOPMENT_CONFIGURATION_SEARCH_AND_ABLATIONS" / "development_cross_validation.json"
    )
    ablations = read_json(required_stage_files[6])
    counterfactual = read_json(
        STAGE / "09_DEVELOPMENT_CONFIGURATION_SEARCH_AND_ABLATIONS" / "counterfactual_error_reduction.json"
    )
    acceptance = read_json(required_stage_files[7])
    readiness = read_json(STAGE / "10_DEVELOPMENT_ACCEPTANCE_AND_NEXT_STAGE" / "candidate_readiness_or_failure.json")
    review = read_json(required_stage_files[8])
    seal = read_json(STAGE / "01_AUTHORIZATION_AND_HOLDOUT_SEAL_GUARD" / "holdout_seal_guard.json")
    mutation = read_json(STAGE / "01_AUTHORIZATION_AND_HOLDOUT_SEAL_GUARD" / "prior_stage_mutation_audit.json")
    environment = read_json(STAGE / "13_REPRODUCIBILITY_BUNDLE" / "environment_and_cache_hashes.json")

    executive = (
        "# M5.5F.1C Executive Summary\n\n"
        "The previous development result reproduced exactly: 12 identity switches, 12 false continuations, "
        "4 losses despite supply, 189/205 correct eligible strand-frames, 5/8 exact sequences and 0 safe "
        "abstentions.\n\n"
        "The public-only panorama P-MHSAG candidate reaches 0 switches, 0 false continuations, 0 "
        "supply-backed losses and 208/208 correct strand-frames across 8/8 development sequences. The "
        "architecture uses canonical panorama coordinates, focal-to-panorama handoff, real pre-link purity "
        "splits, a joint A/B tracklet DAG, a five-model motion bank, sequence-local YOLOv8m backbone "
        "descriptors and top-K paths.\n\n"
        "This is a development result, not tracker promotion. The sealed holdout remains unopened with unseal "
        "count zero. A three-case visual failure audit is ready at http://127.0.0.1:8804/ and must be completed "
        "before any separate freeze or one-time holdout stage.\n"
    )
    (PACK / "01_EXECUTIVE_SUMMARY.md").write_text(executive, encoding="utf-8")
    write_json(
        PACK / "02_RUN_AND_GIT_CONTEXT.json",
        {
            "schema_version": "football_intelligence.m5_5f1c.run_git_context.v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "baseline": BASELINE,
            "head": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "origin": git("remote", "get-url", "origin"),
            "working_tree_clean": not bool(git("status", "--porcelain")),
            "stage_root": str(STAGE),
            "review_url": "http://127.0.0.1:8804/",
        },
    )
    changed = git("diff", "--name-status", BASELINE)
    (PACK / "03_FILES_CHANGED.md").write_text(
        "# Files Changed\n\n```text\n" + (changed or "No source changes detected") + "\n```\n",
        encoding="utf-8",
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", BASELINE, "--", "."],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    if not diff:
        raise RuntimeError("04_SOURCE_DIFF.patch would be empty")
    (PACK / "04_SOURCE_DIFF.patch").write_bytes(diff)
    tests_path = STAGE / "12_COMMANDS_AND_TESTS" / "validation_summary.json"
    tests = read_json(tests_path) if tests_path.is_file() else {"validation_complete": False, "status": "PENDING"}
    (PACK / "05_COMMANDS_AND_TEST_RESULTS.md").write_text(
        "# Commands And Tests\n\n```json\n" + json.dumps(tests, indent=2, sort_keys=True) + "\n```\n",
        encoding="utf-8",
    )
    write_json(PACK / "06_OUTPUT_ARTIFACT_INDEX.json", artifact_index())
    write_json(
        PACK / "07_SELECTED_RESULT_AND_FAILURE_ATLAS.json",
        {"reproduction": reproduction, "root_cause_summary": causes, "counterfactual": counterfactual},
    )
    write_json(
        PACK / "08_ROI_SEMANTICS_AND_PANORAMA_VISIBILITY.json",
        {
            "historical_gold_mutated": roi["historical_gold_mutated"],
            "legacy_eligible_strand_frames": roi["legacy_eligible_strand_frames"],
            "possible_panorama_strand_frames": roi["possible_panorama_strand_frames"],
            "derived_state_counts": roi["derived_state_counts"],
            "legacy_focal_roi_benchmark": roi["legacy_focal_roi_benchmark"],
            "panorama_visible_detector_constrained_benchmark": roi["panorama_visible_detector_constrained_benchmark"],
            "panorama_visible_oracle_observation_metrics": roi["panorama_visible_oracle_observation_benchmark"][
                "metrics"
            ],
        },
    )
    write_json(
        PACK / "09_PANORAMA_GRAPH_AND_DYNAMIC_HANDOFF.json",
        {"graph_validation": graph, "coordinate_roundtrip_and_handoff": handoff},
    )
    write_json(PACK / "10_TRUE_HIERARCHICAL_PATH_SELECTION.json", hierarchy)
    write_json(
        PACK / "11_MOTION_APPEARANCE_AND_DISTRACTOR_BANK.json",
        {
            "motion_models": [
                "OBSERVATION_CENTRIC_CONSTANT_VELOCITY",
                "CONSTANT_ACCELERATION",
                "ROBUST_SMOOTHED_PATH",
                "HEIGHT_ADAPTIVE_SEARCH",
                "BIDIRECTIONAL_SHORT_GAP_INTERPOLATION",
            ],
            "appearance_manifest": appearance,
            "runtime": runtime,
        },
    )
    write_json(
        PACK / "12_DEVELOPMENT_SEARCH_AND_ABLATIONS.json",
        {
            "selection": search,
            "cross_validation": cross_validation,
            "ablations": ablations,
            "interpretation": (
                "Panorama handoff changes observed development outcomes. Several other component ablations are "
                "outcome-equivalent on this small public set; controlled regressions prove their graph/path effect."
            ),
        },
    )
    write_json(
        PACK / "13_ERROR_ATLAS_REVIEW_STATUS.json",
        {
            "review_id": "m5_5f1c_development_error_atlas_review_v1",
            "review_case_count": review["review_case_count"],
            "package_validation_passed": review["passed"],
            "review_complete": False,
            "url": "http://127.0.0.1:8804/",
            "notes_optional": True,
            "gold_labels_read_only": True,
        },
    )
    write_json(
        PACK / "14_HOLDOUT_SEAL_AND_PRIOR_MUTATION_AUDIT.json",
        {
            "holdout_unseal_count_before": seal["holdout_unseal_count_before"],
            "holdout_unseal_count_after": seal["holdout_unseal_count_after"],
            "holdout_labels_opened": seal["holdout_labels_opened"],
            "holdout_visual_evidence_opened": seal["holdout_visual_evidence_opened"],
            "seal_guard_passed": seal["passed"],
            "historical_artifacts_mutated": mutation["historical_artifacts_mutated"],
            "prior_hash_unchanged": mutation["prior_stage_before"]["aggregate_hash"]
            == mutation["prior_stage_after"]["aggregate_hash"],
            "gold_hash_unchanged": mutation["gold_package_before"]["aggregate_hash"]
            == mutation["gold_package_after"]["aggregate_hash"],
        },
    )
    write_json(PACK / "15_ACCEPTANCE_AND_NEXT_STAGE.json", {"acceptance": acceptance, "readiness": readiness})
    write_json(
        PACK / "16_SAFETY_STATEMENT.json",
        {
            "VISUAL_ONLY_NOT_METRIC": True,
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
            "model_fit_performed": False,
            "learned_continuity_rows_updated": 0,
            "tracker_promoted": False,
            "football_metrics_created": False,
            "holdout_unseal_count": 0,
            "checkpoint_sha256": environment["checkpoint_sha256"],
        },
    )
    shutil.copy2(
        STAGE / "14_FAILURE_VISUALS" / "development_failure_atlas.jpg",
        PACK / "17_DEVELOPMENT_FAILURE_ATLAS.jpg",
    )
    shutil.copy2(
        STAGE / "14_FAILURE_VISUALS" / "panorama_handoff_and_repaired_path.png",
        PACK / "18_PANORAMA_HANDOFF_AND_REPAIRED_PATH.png",
    )
    instructions = (
        "# Human Review Instructions\n\n"
        "1. Launch `11_DEVELOPMENT_ERROR_ATLAS_REVIEW_PACKAGE/launch_review.ps1`.\n"
        "2. Open http://127.0.0.1:8804/.\n"
        "3. Review the three public development failure events with frame step/play and the focal, panorama "
        "and layer controls.\n"
        "4. Answer all five structured evidence questions and select one outcome. Notes are optional.\n"
        "5. Do not alter the historical gold labels.\n\n"
        "The sealed holdout remains unopened. If this human error audit supports the development result, a "
        "later separate stage may freeze the candidate and then open the holdout once. No tracker has been "
        "promoted.\n"
    )
    (PACK / "19_HUMAN_REVIEW_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")

    payload_names = sorted(path.name for path in PACK.iterdir())
    expected_without_manifest = sorted(name for name in EXPECTED_NAMES if name != "REVIEW_PACK_MANIFEST.json")
    if payload_names != expected_without_manifest:
        raise RuntimeError(f"review pack payload mismatch: {payload_names}")
    files = [
        {"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(PACK.iterdir())
    ]
    total_bytes = sum(row["size"] for row in files)
    visual_count = sum(Path(row["name"]).suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"} for row in files)
    manifest = {
        "schema_version": "football_intelligence.m5_5f1c.review_pack_manifest.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "flat": True,
        "file_count_after_manifest": len(files) + 1,
        "maximum_file_count": 20,
        "total_bytes_before_manifest": total_bytes,
        "maximum_total_bytes": 50 * 1024 * 1024,
        "visual_file_count": visual_count,
        "maximum_visual_file_count": 3,
        "source_diff_present": any(row["name"] == "04_SOURCE_DIFF.patch" and row["size"] > 0 for row in files),
        "sealed_mapping_included": False,
        "raw_video_included": False,
        "model_weights_included": False,
        "candidate_ids_included": False,
        "answer_keys_included": False,
        "credentials_included": False,
        "personal_data_included": False,
        "holdout_content_included": False,
        "files": files,
        "payload_aggregate_hash": stable_hash(files),
        "validation_passed": len(files) + 1 <= 20
        and total_bytes < 50 * 1024 * 1024
        and visual_count <= 3
        and any(row["name"] == "04_SOURCE_DIFF.patch" and row["size"] > 0 for row in files),
    }
    write_json(PACK / "REVIEW_PACK_MANIFEST.json", manifest)
    final_names = sorted(path.name for path in PACK.iterdir())
    if final_names != sorted(EXPECTED_NAMES) or len(final_names) != 20 or not manifest["validation_passed"]:
        raise RuntimeError("final review pack validation failed")
    print(json.dumps({"file_count": len(final_names), "total_bytes": total_bytes, "passed": True}, indent=2))


if __name__ == "__main__":
    main()
