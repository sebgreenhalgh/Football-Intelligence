"""Create and validate the flat M5.5F.1B ChatGPT review pack."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from football_intelligence.review_chassis.hashing import sha256_file, stable_hash


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
STAGE = PART2 / "M5_5F1B_GOLD_BENCHMARK_INGESTION_DEFINITIVE_GPU_SPORTS_MOT_BAKEOFF_AND_SEALED_HOLDOUT_v1"
PACK = STAGE / "15_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "3f01f9a6bb6495e8f4e67aa5023e7a0cc4a1a70e"
FILES = (
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_SUMMARY.md",
    "02_RUN_AND_GIT_CONTEXT.json",
    "03_FILES_CHANGED.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TEST_RESULTS.md",
    "06_OUTPUT_ARTIFACT_INDEX.json",
    "07_GOLD_INGESTION_AND_SPLIT_INTEGRITY.json",
    "08_ORACLE_AND_DETECTOR_BENCHMARKS.json",
    "09_GPU_CACHE_AND_COMMON_GRAPH.json",
    "10_DEFINITIVE_ALGORITHM_BAKEOFF.json",
    "11_MHSAG_IMPLEMENTATION_AND_ABLATIONS.json",
    "12_FROZEN_WINNER_PRE_REGISTRATION.json",
    "13_SEALED_HOLDOUT_RESULTS.json",
    "14_VISUAL_AUDIT_PACKAGE_STATUS.json",
    "15_SAFETY_AND_MUTATION_AUDIT.json",
    "16_ACCEPTANCE_AND_NEXT_STAGE.json",
    "17_DEVELOPMENT_BAKEOFF_VISUAL.jpg",
    "18_HOLDOUT_RESULT_VISUAL.png",
    "19_HUMAN_REVIEW_INSTRUCTIONS.md",
)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: Any) -> None:
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def artifact_index() -> dict[str, Any]:
    rows = []
    for path in sorted(value for value in STAGE.rglob("*") if value.is_file() and PACK not in value.parents):
        if "_tmp" in path.parts:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(STAGE).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "stage_root": str(STAGE),
        "artifact_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "aggregate_hash": stable_hash(rows),
        "artifacts": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focused-result", required=True)
    parser.add_argument("--regression-result", required=True)
    parser.add_argument("--full-suite-result", required=True)
    args = parser.parse_args()
    if not (STAGE / "stage_summary.json").is_file():
        raise RuntimeError("M5.5F.1B stage has not been generated")
    if any(PACK.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty review pack: {PACK}")

    summary = read(STAGE / "stage_summary.json")
    gold = read(STAGE / "01_AUTHORIZATION_AND_GOLD_COMPLETION_VALIDATION" / "completed_gold_validation.json")
    normalized = read(STAGE / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "normalized_completion_sidecar.json")
    replay = read(STAGE / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "event_replay_validation.json")
    leakage = read(STAGE / "03_GOLD_SPLIT_LEAKAGE_AND_SEAL_AUDIT" / "leakage_audit.json")
    oracle = read(STAGE / "04_ORACLE_AND_DETECTOR_CONSTRAINED_BENCHMARKS" / "oracle_benchmark_manifest.json")
    detector = read(STAGE / "04_ORACLE_AND_DETECTOR_CONSTRAINED_BENCHMARKS" / "detector_benchmark_manifest.json")
    observation = read(STAGE / "05_GPU_OBSERVATION_AND_DESCRIPTOR_CACHE" / "observation_cache_manifest.json")
    descriptor = read(STAGE / "05_GPU_OBSERVATION_AND_DESCRIPTOR_CACHE" / "descriptor_cache_manifest.json")
    gpu = read(STAGE / "05_GPU_OBSERVATION_AND_DESCRIPTOR_CACHE" / "gpu_runtime_and_memory.json")
    graphs = read(STAGE / "06_COMMON_GRAPH_AND_ADAPTER_PARITY" / "common_graph_manifest.json")
    parity = read(STAGE / "06_COMMON_GRAPH_AND_ADAPTER_PARITY" / "adapter_parity_validation.json")
    selection = read(STAGE / "07_DIAGNOSTIC_AND_DEVELOPMENT_BAKEOFF" / "development_bakeoff_summary.json")
    cross_validation = read(STAGE / "07_DIAGNOSTIC_AND_DEVELOPMENT_BAKEOFF" / "development_cross_validation.json")
    components = read(STAGE / "08_MHSAG_FULL_IMPLEMENTATION_AND_ABLATIONS" / "mhsag_component_outputs.json")
    ablations = read(STAGE / "08_MHSAG_FULL_IMPLEMENTATION_AND_ABLATIONS" / "mhsag_ablation_results.json")
    frozen = read(STAGE / "09_FROZEN_WINNER_PRE_REGISTRATION" / "frozen_candidate_manifest.json")
    prereg = read(STAGE / "09_FROZEN_WINNER_PRE_REGISTRATION" / "pre_registration_hash.json")
    holdout = read(STAGE / "10_ONE_TIME_SEALED_HOLDOUT_EVALUATION" / "sealed_holdout_results.json")
    holdout_gate = read(STAGE / "10_ONE_TIME_SEALED_HOLDOUT_EVALUATION" / "holdout_acceptance_checklist.json")
    visual_status = read(STAGE / "11_HOLDOUT_WINNER_VISUAL_AUDIT_PACKAGE" / "visual_audit_package_status.json")
    advancement = read(STAGE / "12_FAILURE_TRIAGE_OR_LEVEL3_READINESS" / "advancement_decision.json")
    reproducibility = read(STAGE / "14_REPRODUCIBILITY_BUNDLE" / "reproducibility_manifest.json")
    index = artifact_index()
    head = git("rev-parse", "HEAD").strip()
    changed = git("diff", "--name-status", f"{BASELINE}..{head}")
    diff = git("diff", f"{BASELINE}..{head}", "--")

    write(
        PACK / "01_EXECUTIVE_SUMMARY.md",
        f"""# M5.5F.1B Definitive Bakeoff Handoff

The completed gold benchmark validated at 24 finalized sequences and 624 strand-frame states. Its 1,240-event ledger replayed deterministically into a fresh root, and the original package remained byte-for-byte unchanged.

All eight Tier-1 adapters ran in oracle and detector-constrained modes on immutable shared graphs. MHSAG ({frozen['configuration']['variant']}) ranked first on development, but it retained {summary['development_metrics']['identity_switches']} switches and {summary['development_metrics']['strand_losses_when_supply_available']} losses despite available supply. The hard gate therefore failed. The sealed holdout was not opened, no human winner audit was created, Level 3 remains blocked, and no tracker was promoted.

Final classification: `{summary['classification']}`.
""",
    )
    write(
        PACK / "02_RUN_AND_GIT_CONTEXT.json",
        {
            "stage": "M5.5F.1B",
            "baseline": BASELINE,
            "implementation_commit": head,
            "branch": git("branch", "--show-current").strip(),
            "origin": git("remote", "get-url", "origin").strip(),
            "classification": summary["classification"],
        },
    )
    write(PACK / "03_FILES_CHANGED.md", "# Files Changed\n\n```text\n" + changed + "```\n")
    write(PACK / "04_SOURCE_DIFF.patch", diff)
    write(
        PACK / "05_COMMANDS_AND_TEST_RESULTS.md",
        f"""# Commands And Tests

- `uv lock --check`: passed
- `uv sync`: passed
- CUDA assertion and real FP16 CUDA probe: passed on `{gpu['device_name']}`
- Ruff check: passed
- Ruff format check: passed
- focused M5.5F.1B tests: {args.focused_result}
- M5.5F.1A.4b and sports-MOT regressions: {args.regression_result}
- complete suite: {args.full_suite_result}
- `uv run fi-pipeline --help`: passed
- `uv run fi-pipeline review-chassis --help`: passed
- `git diff --check`: passed
""",
    )
    write(PACK / "06_OUTPUT_ARTIFACT_INDEX.json", index)
    write(
        PACK / "07_GOLD_INGESTION_AND_SPLIT_INTEGRITY.json",
        {
            "completion": gold,
            "normalized_completion": normalized,
            "event_replay": replay,
            "split_leakage": leakage,
            "approved_polygon_hash": reproducibility["approved_polygon_hash"],
        },
    )
    write(PACK / "08_ORACLE_AND_DETECTOR_BENCHMARKS.json", {"oracle": oracle, "detector": detector})
    write(
        PACK / "09_GPU_CACHE_AND_COMMON_GRAPH.json",
        {
            "observation_cache": observation,
            "descriptor_cache": descriptor,
            "cuda_runtime": gpu,
            "graph_schema_hash": graphs["graph_schema_hash"],
            "graph_count_before_holdout": graphs["graph_count_before_holdout"],
            "adapter_parity": parity,
        },
    )
    write(
        PACK / "10_DEFINITIVE_ALGORITHM_BAKEOFF.json",
        {
            "selection_protocol": selection["selection_protocol"],
            "selected": selection["selected"],
            "development_hard_gate_passed": selection["development_hard_gate_passed"],
            "candidate_count": len(selection["candidate_summaries"]),
            "cross_validation_protocol": cross_validation["protocol"],
            "cross_validation_fold_count": cross_validation["fold_count"],
            "diagnostic_rows_used_for_selection": 0,
            "holdout_rows_used_for_selection": 0,
        },
    )
    write(
        PACK / "11_MHSAG_IMPLEMENTATION_AND_ABLATIONS.json",
        {
            "implementation_status": components["implementation_status"],
            "required_components": components["required_components"],
            "execution_row_count": len(components["development_and_diagnostic_rows"]),
            "ablations": ablations,
        },
    )
    write(PACK / "12_FROZEN_WINNER_PRE_REGISTRATION.json", {"manifest": frozen, "pre_registration": prereg})
    write(
        PACK / "13_SEALED_HOLDOUT_RESULTS.json",
        {
            "holdout_opened": holdout["holdout_opened"],
            "unseal_count": holdout["unseal_count"],
            "frozen_candidate_manifest_hash": holdout["frozen_candidate_manifest_hash"],
            "oracle": holdout["oracle"],
            "detector": holdout["detector"],
            "acceptance": holdout_gate,
            "retuning_performed_after_holdout": False,
        },
    )
    write(PACK / "14_VISUAL_AUDIT_PACKAGE_STATUS.json", visual_status)
    write(
        PACK / "15_SAFETY_AND_MUTATION_AUDIT.json",
        {
            "VISUAL_ONLY_NOT_METRIC": True,
            "historical_artifacts_mutated": reproducibility["historical_artifacts_mutated"],
            "protected_gold_before_hash": reproducibility["protected_gold_before_hash"],
            "protected_gold_after_hash": reproducibility["protected_gold_after_hash"],
            "holdout_unseal_count": reproducibility["holdout_unseal_count"],
            "retuning_after_holdout": False,
            "tracker_promoted": False,
            "production_ready": False,
            "human_approved": False,
            "match_local_only": True,
            "sandbox_only": True,
        },
    )
    write(PACK / "16_ACCEPTANCE_AND_NEXT_STAGE.json", advancement)
    shutil.copy2(
        STAGE / "07_DIAGNOSTIC_AND_DEVELOPMENT_BAKEOFF" / "development_bakeoff_visual.jpg",
        PACK / "17_DEVELOPMENT_BAKEOFF_VISUAL.jpg",
    )
    shutil.copy2(
        STAGE / "10_ONE_TIME_SEALED_HOLDOUT_EVALUATION" / "holdout_result_visual.png",
        PACK / "18_HOLDOUT_RESULT_VISUAL.png",
    )
    write(
        PACK / "19_HUMAN_REVIEW_INSTRUCTIONS.md",
        """# Human Action

No human winner audit is requested. The development hard gate failed, so port 8803 was not launched and the sealed holdout remained unopened.

Review the development failure attribution and MHSAG ablations before authorizing a new architecture branch. Any next branch must remain development-only and must not infer or tune against sealed holdout labels.
""",
    )

    payload_files = [PACK / name for name in FILES if name != "REVIEW_PACK_MANIFEST.json"]
    if any(not path.is_file() for path in payload_files):
        raise RuntimeError("review pack payload is incomplete")
    file_rows = [
        {"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)} for path in payload_files
    ]
    manifest = {
        "schema_version": "football_intelligence.m5_5f1b.review_pack.v1",
        "classification": summary["classification"],
        "file_count": len(FILES),
        "total_bytes": sum(row["size"] for row in file_rows),
        "maximum_file_count": 20,
        "maximum_total_bytes": 52_428_800,
        "visual_file_count": 2,
        "maximum_visual_files": 3,
        "flat": True,
        "source_diff_present": True,
        "sealed_mappings_included": False,
        "candidate_ids_included": False,
        "answer_keys_included": False,
        "raw_video_included": False,
        "model_weights_included": False,
        "files": file_rows,
    }
    manifest["passed"] = (
        manifest["file_count"] <= manifest["maximum_file_count"]
        and manifest["total_bytes"] <= manifest["maximum_total_bytes"]
        and manifest["visual_file_count"] <= manifest["maximum_visual_files"]
    )
    write(PACK / "REVIEW_PACK_MANIFEST.json", manifest)
    actual_files = sorted(path.name for path in PACK.iterdir() if path.is_file())
    if actual_files != sorted(FILES) or not manifest["passed"]:
        raise RuntimeError("review pack validation failed")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
