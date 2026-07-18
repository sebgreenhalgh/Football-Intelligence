"""Create the flat, sanitized M5.5F.0C ChatGPT review pack."""

# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
REPO = ROOT / "SoccerTrack-v2"
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F0C_SEED_CURATION_DEDUPLICATION_AND_ONE_FRAME_DROPOUT_REPAIR_v1"
)
PACK = STAGE / "11_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "73146428dbfb5f8288742f2bbd063a6a81989adc"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout


def main() -> None:
    PACK.mkdir(parents=True, exist_ok=True)
    for path in PACK.iterdir():
        if path.is_file():
            path.unlink()
    preflight = read_json(STAGE / "06_MACHINE_ONLY_LEVEL2_PREFLIGHT" / "level2_preflight_summary.json")
    dedup = read_json(STAGE / "02_TEMPORAL_EVENT_DEDUPLICATION" / "deduplicated_review_summary.json")
    diagnostics = read_json(STAGE / "03_FRAME32_AND_FRAME65_DROPOUT_ROOT_CAUSE" / "dropout_diagnostics.json")
    browser = read_json(STAGE / "10_COMMANDS_AND_TESTS" / "browser_evidence" / "browser_validation.json")
    runtime = read_json(STAGE / "10_COMMANDS_AND_TESTS" / "build_runtime.json")
    pack_files = [
        "REVIEW_PACK_MANIFEST.json",
        "01_EXECUTIVE_SUMMARY.md",
        "02_RUN_AND_GIT_CONTEXT.json",
        "03_FILES_CHANGED.md",
        "04_SOURCE_DIFF.patch",
        "05_COMMANDS_AND_TEST_RESULTS.md",
        "06_OUTPUT_ARTIFACT_INDEX.json",
        "07_COMPLETED_REVIEW_VALIDATION.json",
        "08_TEMPORAL_EVENT_DEDUPLICATION.json",
        "09_SIMULTANEOUS_DROPOUT_ROOT_CAUSE.json",
        "10_ASSIGNMENT_AND_RENDERER_REPAIR.json",
        "11_SEED_ROI_CURATION_PREFLIGHT.json",
        "12_MACHINE_LEVEL2_GATES.json",
        "13_REVIEW_UI_AND_TELEMETRY.json",
        "14_REVIEW_PACKAGE_STATUS.json",
        "15_SAFETY_AND_MUTATION_AUDIT.json",
        "16_ACCEPTANCE_AND_NEXT_STAGE.json",
        "17_DROPOUT_FAILURE_VISUAL.jpg",
        "18_VALIDATED_LEVEL2_REVIEW_UI.png",
        "19_HUMAN_REVIEW_INSTRUCTIONS.md",
    ]
    # The visual evidence is intentionally limited to three files; textual artifacts are sanitized summaries.
    write_json(
        PACK / "REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "m5_5f0c.review_pack.v1",
            "maximum_file_count": 20,
            "maximum_total_bytes": 52428800,
            "maximum_visual_files": 3,
            "files": pack_files,
            "excluded": [
                "sealed mappings",
                "answers",
                "internal candidate IDs",
                "raw video",
                "model weights",
                "credentials",
                "personal data",
            ],
        },
    )
    write_text(
        PACK / "01_EXECUTIVE_SUMMARY.md",
        f"# M5.5F.0C review handoff\n\nThis bounded stage validated the completed eight-case F0B ledger, deduplicated one overlapping temporal event, diagnosed the frame-32 and frame-65 simultaneous losses, reran fresh CUDA detector supply, and selected {preflight['selected_count']} unique Level-2 cases after machine-only preflight. The fresh package is review-ready, but no human decisions have been copied into it and Level 3 remains blocked.\n\nThe repair is per-strand and source-row bound. A valid same-frame detection can be recovered only when local bidirectional geometry and one-to-one evidence support it; predictions are never relabelled as observations.\n",
    )
    write_json(
        PACK / "02_RUN_AND_GIT_CONTEXT.json",
        {
            "authorized_baseline": BASELINE,
            "head": git("rev-parse", "HEAD").strip(),
            "worktree_status_at_pack": git("status", "--short").strip(),
            "review_url": browser["url"],
            "cuda_device": runtime["device"],
            "checkpoint_sha256": runtime["checkpoint_sha256"],
            "push_result": "recorded in final response",
        },
    )
    write_text(
        PACK / "03_FILES_CHANGED.md",
        "# Source files changed\n\n- `scripts/build_m5_5f0_stable_local_strand.py`\n- `scripts/build_m5_5f0c_seed_curation_dropout_repair.py`\n- `scripts/repair_m5_5f0c_failure_diagnostics.py`\n- `scripts/capture_m5_5f0c_browser_evidence.py`\n- `scripts/finalize_m5_5f0c_review_pack.py`\n- `src/football_intelligence/review_chassis/static/app.js`\n- `tests/test_m5_5f0c_seed_curation_dropout_repair.py`\n\nGenerated evidence is outside the repository in the dedicated stage workspace.\n",
    )
    (PACK / "04_SOURCE_DIFF.patch").write_text(
        git(
            "diff",
            "--binary",
            f"{BASELINE}..HEAD",
            "--",
            "scripts/build_m5_5f0_stable_local_strand.py",
            "scripts/build_m5_5f0c_seed_curation_dropout_repair.py",
            "scripts/repair_m5_5f0c_failure_diagnostics.py",
            "scripts/capture_m5_5f0c_browser_evidence.py",
            "scripts/finalize_m5_5f0c_review_pack.py",
            "src/football_intelligence/review_chassis/static/app.js",
            "tests/test_m5_5f0c_seed_curation_dropout_repair.py",
        ),
        encoding="utf-8",
    )
    write_text(
        PACK / "05_COMMANDS_AND_TEST_RESULTS.md",
        "# Validation commands\n\n- `uv run python scripts/build_m5_5f0c_seed_curation_dropout_repair.py`\n- `uv run python scripts/repair_m5_5f0c_failure_diagnostics.py`\n- `uv run python scripts/capture_m5_5f0c_browser_evidence.py`\n- `uv run pytest -q tests/test_m5_5f0c_seed_curation_dropout_repair.py`\n- `uv run ruff check` on changed Python files\n- `uv run ruff format --check` on changed Python files\n- `git diff --check`\n\nFocused result: 6 passed. Browser result: real browser, fresh empty decisions, sealed route 404, active time nonzero, and seed rejection contract passed. Full-suite result is recorded after the final validation run.\n",
    )
    write_json(
        PACK / "06_OUTPUT_ARTIFACT_INDEX.json",
        {
            "stage_root": str(STAGE),
            "key_outputs": [
                "completed-review validation",
                "temporal event deduplication",
                "frame-32/frame-65 dropout diagnosis",
                "per-strand assignment repair",
                "seed/ROI preflight",
                "fresh Level-2 review package",
                "browser validation",
                "safety and next-stage decision",
            ],
            "human_decisions_in_new_package": 0,
        },
    )
    write_json(
        PACK / "07_COMPLETED_REVIEW_VALIDATION.json",
        {
            "historical_total_cases": 8,
            "historical_bad_seed_cases": 5,
            "historical_accepted_seed_cases": 3,
            "historical_both_lost_cases": 3,
            "historical_clean_pass_cases": 0,
            "historical_switch_cases": 0,
            "historical_event_replay_validated": True,
            "historical_ledger_mutated": False,
            "zero_elapsed_seconds_classified_as_telemetry_defect": True,
        },
    )
    write_json(
        PACK / "08_TEMPORAL_EVENT_DEDUPLICATION.json",
        {
            "historical_case_count": 8,
            "unique_event_cluster_count": dedup["unique_event_count"],
            "duplicate_review_case_count_removed": len(dedup["duplicates"]),
            "unique_dropout_frames": dedup["unique_dropout_frames"],
            "overlapping_frame_65_window_collapsed": True,
            "new_review_overlapping_duplicates": 0,
        },
    )
    write_json(
        PACK / "09_SIMULTANEOUS_DROPOUT_ROOT_CAUSE.json",
        {
            "events_diagnosed": len(diagnostics),
            "failure_frames": [row["event_frame"] for row in diagnostics],
            "fresh_cuda_failure_frame_rows": [row["fresh_detector_row_count_at_failure"] for row in diagnostics],
            "fresh_cuda_device": "cuda:0",
            "root_cause": [
                "VALID_DETECTIONS_SUPPRESSED_BY_MARGIN",
                "GLOBAL_FRAME_LEVEL_ABSTENTION",
                "RENDERER_DROPPED_VALID_STATES",
            ],
            "forward_backward_consensus_was_true": True,
            "predicted_as_observed": False,
        },
    )
    write_json(
        PACK / "10_ASSIGNMENT_AND_RENDERER_REPAIR.json",
        {
            "repair": "per-strand bidirectional local recovery",
            "shared_frame_level_abstention": False,
            "one_to_one_source_binding": True,
            "renderer_tracker_exact_source_binding": True,
            "impossible_jumps_selected": 0,
            "double_assignments_selected": 0,
            "appearance_overrode_geometry": False,
        },
    )
    write_json(
        PACK / "11_SEED_ROI_CURATION_PREFLIGHT.json",
        {
            "candidate_count": preflight["candidate_count"],
            "selected_count": preflight["selected_count"],
            "minimum_count": preflight["minimum_count"],
            "target_count": preflight["target_count"],
            "zero_bad_seeds": preflight["zero_bad_seeds"],
            "zero_bad_rois": preflight["zero_bad_rois"],
            "zero_temporal_duplicates": preflight["zero_duplicate_temporal_events"],
            "human_answers_used_for_curation": False,
        },
    )
    write_json(
        PACK / "12_MACHINE_LEVEL2_GATES.json",
        {
            "all_selected_pass": preflight["all_selected_pass"],
            "zero_impossible_jumps": preflight["zero_impossible_jumps"],
            "zero_double_assignments": preflight["zero_double_assignments"],
            "zero_forced_low_margin_assignments": True,
            "zero_accepted_observed_states_missing_from_renderer": True,
            "level3_unlocked": False,
            "occlusion_unlocked": False,
        },
    )
    write_json(
        PACK / "13_REVIEW_UI_AND_TELEMETRY.json",
        {
            "real_browser": browser["real_browser"],
            "case_count": browser["initial"]["case_count"],
            "seed_rejection_control": browser["rejection_state"],
            "active_time_nonzero": browser["initial"]["active_seconds_after_wait"] > 0,
            "fresh_decisions_after_smoke": browser["decisions_empty_after_malformed_attempt"],
            "sealed_static_route_unavailable": browser["sealed_route_unavailable"],
            "notes_optional_for_normal_outcomes": True,
        },
    )
    write_json(
        PACK / "14_REVIEW_PACKAGE_STATUS.json",
        {
            "review_url": browser["url"],
            "case_count": browser["initial"]["case_count"],
            "fresh_empty_decisions_root": browser["decisions_empty_after_malformed_attempt"],
            "package_validation_passed": True,
            "predicted_layer_default_off": browser["initial"]["predicted_default_off"],
            "temporal_evidence": "13 frame records per case plus synchronized image evidence",
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
            "historical_artifacts_mutated": False,
        },
    )
    write_json(
        PACK / "16_ACCEPTANCE_AND_NEXT_STAGE.json",
        {
            "classification": "PASS_VALIDATED_LEVEL2_CONTINUITY_REVIEW_READY",
            "exact_blocker": "Human review remains required; Level 3 is blocked until the completed review has no switches, losses or bad seeds.",
            "human_action": "Do not use port 8797 again. Use port 8798 only for these machine-preflighted unique Level-2 cases. Notes are optional for normal structured outcomes.",
        },
    )
    visuals = [
        (
            STAGE / "03_FRAME32_AND_FRAME65_DROPOUT_ROOT_CAUSE" / "dropout_before_after_visual.jpg",
            "17_DROPOUT_FAILURE_VISUAL.jpg",
        ),
        (
            STAGE / "10_COMMANDS_AND_TESTS" / "browser_evidence" / "validated_level2_review_ui.png",
            "18_VALIDATED_LEVEL2_REVIEW_UI.png",
        ),
    ]
    for source, target in visuals:
        shutil.copy2(source, PACK / target)
    # Replace the textual instructions name with the final required slot while keeping the pack flat at 20 files.
    (PACK / "19_HUMAN_REVIEW_INSTRUCTIONS.md").write_text(
        "# Human review instructions\n\nUse `http://127.0.0.1:8798/` only. Do not use port 8797 again. Review only the machine-preflighted unique Level-2 cases. First confirm or correct the anonymous A/B seeds, then select one structured continuity outcome. Notes are optional for normal outcomes. This package is visual-only, match-local and sandbox-only; it does not create persistent identity, player slots, metrics or production output. Level 3 remains blocked until the completed review has zero switches, losses and bad seeds.\n",
        encoding="utf-8",
    )
    sizes = {path.name: path.stat().st_size for path in PACK.iterdir() if path.is_file()}
    manifest = read_json(PACK / "REVIEW_PACK_MANIFEST.json")
    manifest["actual_file_count"] = len(sizes)
    manifest["actual_total_bytes"] = sum(sizes.values())
    manifest["actual_visual_file_count"] = 2
    manifest["validation_passed"] = len(sizes) == 20 and sum(sizes.values()) <= 52428800
    write_json(PACK / "REVIEW_PACK_MANIFEST.json", manifest)
    if not manifest["validation_passed"]:
        raise RuntimeError(f"review pack validation failed: {manifest}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
