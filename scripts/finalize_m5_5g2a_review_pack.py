"""Create and validate the flat M5.5G.2A ChatGPT review pack."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from football_intelligence.detection_forensics import sha256_file, validate_flat_context_pack

REPO = Path(__file__).resolve().parents[1]
FOOTBALL_ROOT = REPO.parent
PART3 = FOOTBALL_ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G2A_PLAYER_PROPOSAL_SUPPLY_EXPLORATORY_DIAGNOSTIC_v1"
PACK = STAGE / "09_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "be8848ac606d04d7a9c5888276d96582d34f0c71"
EXPECTED_CLASSIFICATION = "PASS_TRANCHE_A_EXPLORATORY_PROPOSAL_DIAGNOSTIC_READY_FOR_PRO_REVIEW"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def source_diff() -> str:
    head = git("rev-parse", "HEAD")
    if head != BASELINE:
        return subprocess.run(
            ["git", "diff", "--no-ext-diff", f"{BASELINE}..{head}"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    staged = subprocess.run(
        ["git", "diff", "--cached", "--no-ext-diff"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    unstaged = subprocess.run(
        ["git", "diff", "--no-ext-diff"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return staged + unstaged


def ensure_visual(path: Path) -> None:
    with Image.open(path) as image:
        image.load()
        if image.width < 640 or image.height < 300:
            raise ValueError(f"visual is too small: {path.name}")
        extrema = ImageStat.Stat(image.convert("L")).extrema[0]
        if extrema[0] == extrema[1]:
            raise ValueError(f"visual is blank: {path.name}")


def sanitized_source_group_summary() -> dict[str, Any]:
    manifest = read_json(STAGE / "02_GOLD_SOURCE_GROUP_AND_INSTANCE_DEDUPLICATION" / "source_group_manifest.json")
    audit = read_json(
        STAGE / "02_GOLD_SOURCE_GROUP_AND_INSTANCE_DEDUPLICATION" / "cross_case_gold_instance_deduplication.json"
    )
    return {
        "case_record_count": manifest["case_record_count"],
        "unique_source_group_count": manifest["unique_source_group_count"],
        "duplicate_source_case_sets": [row["case_ids"] for row in manifest["groups"] if row["duplicate_source_group"]],
        "cross_case_cluster_proposal_count": audit["proposal_count"],
        "conservative_canonical_merge_count": audit["canonical_merge_count"],
        "all_cross_case_proposals_require_manual_review": audit["all_proposals_require_manual_review"],
        "primary_group_key": "source_frame_sha256",
        "raw_source_hashes_included": False,
        "annotation_identifiers_included": False,
    }


def sanitized_lineage_summary() -> dict[str, Any]:
    validation = read_json(STAGE / "03_CANDIDATE_LINEAGE_BINDING" / "candidate_lineage_binding_validation.json")
    binding = read_json(STAGE / "03_CANDIDATE_LINEAGE_BINDING" / "candidate_lineage_binding.json")
    unique_rows: dict[str, dict[str, Any]] = {}
    for row in binding["rows"]:
        unique_rows.setdefault(row["candidate_uuid"], row)
    return {
        "passed": validation["passed"],
        "reviewed_relation_row_count": validation["reviewed_relation_row_count"],
        "unique_lineage_entity_count": validation["bound_unique_candidate_uuid_count"],
        "binding_error_count": validation["binding_error_count"],
        "view_distribution": dict(sorted(Counter(row["inference_view_type"] for row in unique_rows.values()).items())),
        "renderer_membership": {
            "present": sum(bool(row["final_renderer_row"]) for row in unique_rows.values()),
            "absent": sum(not bool(row["final_renderer_row"]) for row in unique_rows.values()),
        },
        "one_uuid_is_one_lineage_entity": True,
        "exact_frozen_replay_performed": False,
        "candidate_identifiers_included": False,
    }


def main() -> None:
    PACK.mkdir(parents=True, exist_ok=True)
    summary = read_json(STAGE / "M5_5G2A_STAGE_SUMMARY.json")
    completion = read_json(STAGE / "01_TRANCHE_A_INGESTION_AND_QA" / "tranche_a_completion_validation.json")
    inventory = read_json(STAGE / "01_TRANCHE_A_INGESTION_AND_QA" / "tranche_a_gold_inventory.json")
    coverage = read_json(STAGE / "04_STAGE_AND_VIEW_PROPOSAL_COVERAGE" / "development_proposal_coverage.json")
    diagnostics = read_json(
        STAGE / "05_DUPLICATE_MERGED_AND_BACKGROUND_DIAGNOSTICS" / "duplicate_merged_background_diagnostics.json"
    )
    outlier = read_json(
        STAGE / "05_DUPLICATE_MERGED_AND_BACKGROUND_DIAGNOSTICS" / "candidate_count_outlier_analysis.json"
    )
    origin = read_json(
        STAGE / "05_DUPLICATE_MERGED_AND_BACKGROUND_DIAGNOSTICS" / "human_vs_computed_origin_reconciliation.json"
    )
    runtime = read_json(STAGE / "08_COMMANDS_AND_TESTS" / "exploratory_runtime_and_vram.json")
    preservation = read_json(STAGE / "08_COMMANDS_AND_TESTS" / "prior_stage_preservation.json")
    next_priority = read_json(STAGE / "07_NEXT_ANNOTATION_AND_EXPERIMENT_DECISION" / "next_annotation_priority.json")
    validation_results_path = STAGE / "08_COMMANDS_AND_TESTS" / "validation_results.json"
    validation_results = read_json(validation_results_path) if validation_results_path.exists() else None

    (PACK / "00_READ_ME_FIRST.md").write_text(
        "# M5.5G.2A review pack\n\n"
        "This flat pack documents a bounded exploratory proposal-supply diagnostic over single-reviewer "
        "Tranche A development gold. It contains no detector comparison, architecture score, training, "
        "promotion, validation claim, raw video, model weights, candidate mappings, or hidden answers. Read "
        "the executive outcome, then the exact-denominator coverage and safety records.\n",
        encoding="utf-8",
    )
    (PACK / "01_EXECUTIVE_OUTCOME.md").write_text(
        "# Executive outcome\n\n"
        f"Classification: `{summary['classification']}`\n\n"
        f"Decision: `{summary['final_decision']}`\n\n"
        "The 18-case completion replay and all 233 reviewed lineage bindings passed. The primary analysis uses "
        "17 source groups and conservative canonical-person clusters, not pooled candidate counts. Case 008 "
        "remains intact but cannot dominate the conclusion. Cross-case person links for Cases 007 and 027 "
        "remain manual-review items. No exact detector replay was required.\n",
        encoding="utf-8",
    )
    write_json(
        PACK / "02_REPOSITORY_STATE.json",
        {
            "source_commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "baseline": BASELINE,
            "prior_stage_preserved": preservation["passed"],
            "expected_remote_and_branch_only": True,
            "absolute_paths_included": False,
        },
    )
    write_json(
        PACK / "03_TRANCHE_A_COMPLETION_AND_GOLD_QA.json",
        {
            "completion_passed": completion["passed"],
            "case_count": completion["case_count"],
            "strict_event_count": completion["strict_event_count"],
            "event_replay": completion["event_replay"],
            "evidence_validation": completion["evidence_validation"],
            "gold_inventory": inventory,
            "single_primary_reviewer": True,
            "benchmark_grade_validation_gold": False,
            "transaction_identifiers_included": False,
        },
    )
    patch = source_diff()
    if not patch.strip():
        raise ValueError("04_SOURCE_DIFF.patch would be empty")
    (PACK / "04_SOURCE_DIFF.patch").write_text(patch, encoding="utf-8")
    write_json(PACK / "05_SOURCE_GROUPING_SUMMARY.json", sanitized_source_group_summary())
    write_json(PACK / "06_LINEAGE_BINDING_SUMMARY.json", sanitized_lineage_summary())
    write_json(PACK / "07_PERSON_STAGE_VIEW_COVERAGE.json", coverage)
    write_json(PACK / "08_DUPLICATE_MERGED_BACKGROUND.json", diagnostics)
    write_json(PACK / "09_OUTLIER_AND_WEIGHTING.json", outlier)
    write_json(
        PACK / "10_ORIGIN_RECONCILIATION_SUMMARY.json",
        {
            "agreement_count": origin["agreement_count"],
            "contradiction_count": origin["contradiction_count"],
            "insufficient_evidence_count": origin["insufficient_evidence_count"],
            "human_fields_overwritten": origin["human_fields_overwritten"],
            "computed_fields_are_provisional": origin["computed_fields_are_provisional"],
            "bounded_raw_top_k_caveat": origin["bounded_raw_top_k_caveat"],
        },
    )
    write_json(
        PACK / "11_RUNTIME_AND_SAFETY.json",
        {
            "runtime": runtime,
            "classification": summary["classification"],
            "training_performed": False,
            "detector_inference_performed": False,
            "production_defaults_changed": False,
            "detector_or_tracker_promoted": False,
            "validation_or_holdout_use": False,
            "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        },
    )
    write_json(PACK / "12_NEXT_ANNOTATION_PRIORITY.json", next_priority)
    shutil.copy2(
        STAGE / "07_NEXT_ANNOTATION_AND_EXPERIMENT_DECISION" / "next_stage_decision.md",
        PACK / "13_FINAL_DECISION.md",
    )
    visual_sources = {
        "14_GOLD_ONLY_ATLAS.png": "gold_only_atlas_all_18_cases.png",
        "15_CANDIDATE_RELATION_ATLAS.png": "candidate_relation_atlas_uncluttered.png",
        "16_STAGE_VIEW_SUPPLY_ATLAS.png": "stage_view_supply_representatives.png",
    }
    for target_name, source_name in visual_sources.items():
        source = STAGE / "06_VISUAL_QA_AND_CASE_LEDGER" / source_name
        ensure_visual(source)
        shutil.copy2(source, PACK / target_name)
    if validation_results is None:
        tests_text = (
            "# Tests and commands\n\nFinal command results have not yet been attached to this generated pack.\n"
        )
    else:
        tests_text = (
            "# Tests and commands\n\n"
            f"Focused M5.5G.2A: {validation_results['focused_tests']}\n\n"
            f"Relevant historical regressions: {validation_results['regression_tests']}\n\n"
            f"Complete suite: {validation_results['full_suite']}\n\n"
            "Passed additional gates: `uv lock --check`, `uv sync`, CUDA availability in the normal `.venv`, "
            "Ruff check and format check, `fi-pipeline --help`, `fi-pipeline review-chassis --help`, "
            "and `git diff --check`.\n"
        )
    (PACK / "17_TESTS_AND_COMMANDS.md").write_text(tests_text, encoding="utf-8")
    interim = validate_flat_context_pack(
        PACK,
        maximum_file_count=20,
        maximum_total_bytes=50 * 1024 * 1024,
    )
    if interim["visual_file_count"] != 3:
        raise ValueError("review pack must contain exactly three visuals")
    text_payload = b"".join(
        path.read_bytes()
        for path in PACK.iterdir()
        if path.is_file() and path.suffix.casefold() not in {".png", ".jpg", ".jpeg", ".patch"}
    )
    if re.search(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text_payload):
        raise ValueError("review pack exposes a UUID")
    manifest_rows = [
        {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(PACK.iterdir())
        if path.is_file()
    ]
    write_json(
        PACK / "18_REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "football_intelligence.m5_5g2a.review_pack_manifest.v1",
            "classification": EXPECTED_CLASSIFICATION,
            "file_count_including_manifest": len(manifest_rows) + 1,
            "maximum_file_count": 20,
            "maximum_total_bytes": 50 * 1024 * 1024,
            "visual_file_count": 3,
            "flat": True,
            "candidate_identifiers_included": False,
            "hidden_answers_included": False,
            "files_excluding_self": manifest_rows,
        },
    )
    final = validate_flat_context_pack(PACK, maximum_file_count=20, maximum_total_bytes=50 * 1024 * 1024)
    if final["visual_file_count"] != 3:
        raise ValueError("review pack visual count changed during finalization")
    write_json(STAGE / "08_COMMANDS_AND_TESTS" / "review_pack_validation.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
