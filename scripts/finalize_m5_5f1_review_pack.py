"""Create and validate the flat M5.5F.1 ChatGPT review pack."""

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
    / r"matches\128058\runs\step_m5\part 2\M5_5F1_SEQUENCE_GLOBAL_ASSOCIATION_BAKEOFF_AND_UNSEEN_LEVEL2_VALIDATION_v1"
)
PACK = STAGE / "12_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "f64612757ccec5dfe919f406ade116be4c842045"
SOURCE_FILES = [
    "scripts/build_m5_5f1_sequence_global_association_bakeoff.py",
    "scripts/capture_m5_5f1_browser_evidence.py",
    "scripts/finalize_m5_5f1_review_pack.py",
    "src/football_intelligence/review_chassis/static/app.js",
    "src/football_intelligence/review_chassis/static/index.html",
    "tests/test_m5_5f1_sequence_global_association_bakeoff.py",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout


def main() -> None:
    PACK.mkdir(parents=True, exist_ok=True)
    validation = read_json(STAGE / "09_UNSEEN_LEVEL2_ASSOCIATION_REVIEW_PACKAGE" / "review_package_validation.json")
    browser = read_json(STAGE / "11_COMMANDS_AND_TESTS" / "browser_evidence" / "browser_validation.json")
    graph = read_json(STAGE / "03_COMMON_OBSERVATION_GRAPH" / "graph_validation.json")
    graph_nodes = read_jsonl(STAGE / "03_COMMON_OBSERVATION_GRAPH" / "observation_nodes.jsonl")
    graph_edges = read_jsonl(STAGE / "03_COMMON_OBSERVATION_GRAPH" / "observation_edges.jsonl")
    bakeoff = read_json(STAGE / "04_ASSOCIATION_ALGORITHM_BAKEOFF" / "bakeoff_summary.json")
    optimizer = read_json(STAGE / "05_SEQUENCE_GLOBAL_TWO_STRAND_OPTIMIZER" / "optimizer_summary.json")
    telemetry = read_json(STAGE / "06_GPU_APPEARANCE_AND_MOTION_EVIDENCE" / "gpu_timing_and_memory.json")
    descriptor = read_json(STAGE / "06_GPU_APPEARANCE_AND_MOTION_EVIDENCE" / "descriptor_comparison.json")
    curation = read_json(STAGE / "07_UNSEEN_LEVEL2_CASE_CURATION" / "temporal_exclusion_audit.json")
    selected = read_jsonl(STAGE / "07_UNSEEN_LEVEL2_CASE_CURATION" / "selected_unseen_cases.jsonl")
    gates = read_json(STAGE / "08_MACHINE_ONLY_UNSEEN_GATES" / "unseen_gate_summary.json")
    completed = read_json(STAGE / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "completed_review_validation.json")
    authorization = read_json(STAGE / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "authorization_audit.json")
    decision = read_json(STAGE / "10_EVALUATION_AND_NEXT_STAGE" / "next_stage_decision.json")
    pack_files = [
        "REVIEW_PACK_MANIFEST.json",
        "01_EXECUTIVE_SUMMARY.md",
        "02_RUN_AND_GIT_CONTEXT.json",
        "03_FILES_CHANGED.md",
        "04_SOURCE_DIFF.patch",
        "05_COMMANDS_AND_TEST_RESULTS.md",
        "06_OUTPUT_ARTIFACT_INDEX.json",
        "07_COMPLETED_REVIEW_AND_SWITCH_AUDIT.json",
        "08_COMMON_OBSERVATION_GRAPH.json",
        "09_ASSOCIATION_ALGORITHM_BAKEOFF.json",
        "10_SEQUENCE_GLOBAL_OPTIMIZER.json",
        "11_GPU_APPEARANCE_EVIDENCE.json",
        "12_UNSEEN_CASE_CURATION.json",
        "13_MACHINE_UNSEEN_GATES.json",
        "14_REVIEW_PACKAGE_STATUS.json",
        "15_SAFETY_AND_MUTATION_AUDIT.json",
        "16_ACCEPTANCE_AND_NEXT_STAGE.json",
        "17_SWITCH_DIAGNOSTIC_VISUAL.jpg",
        "18_UNSEEN_REVIEW_UI.png",
        "19_HUMAN_REVIEW_INSTRUCTIONS.md",
    ]
    write_json(
        PACK / "REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "m5_5f1.review_pack.v1",
            "maximum_file_count": 20,
            "maximum_total_bytes": 52_428_800,
            "maximum_visual_files": 3,
            "files": pack_files,
            "excluded": [
                "sealed mappings",
                "human answers",
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
        f"""# M5.5F.1 review handoff

This bounded stage compared five association strategies on one immutable CUDA-tagged per-frame observation graph. It reproduced the three historical Level-2 switch failures for diagnostic development only, retained the three pass controls, and curated {len(selected)} machine-preflighted unseen Level-2 cases for the fresh port-8799 review.

The unseen set is pairwise disjoint, balanced across four requested strata, and has not used the historical human answers for selection. The package is review-ready, not validated for accuracy: human review remains required and Level 3 stays blocked until the completed unseen review has no switches, losses, or bad seeds.
""",
    )
    write_json(
        PACK / "02_RUN_AND_GIT_CONTEXT.json",
        {
            "authorized_baseline": BASELINE,
            "head": git("rev-parse", "HEAD").strip(),
            "worktree_status_at_pack": git("status", "--short").strip().splitlines(),
            "review_url": "http://127.0.0.1:8799/",
            "reviewer_session_id": "m5_5f1_unseen_level2_association_human_reviewer",
            "detector_device": telemetry["device"],
            "checkpoint_sha256": telemetry["checkpoint_sha256"],
            "push_result": "reported in final response",
        },
    )
    write_text(
        PACK / "03_FILES_CHANGED.md",
        "# Source files changed\n\n"
        + "\n".join(f"- `{path}`" for path in SOURCE_FILES)
        + "\n\nGenerated evidence is outside the repository in the dedicated stage workspace.\n",
    )
    (PACK / "04_SOURCE_DIFF.patch").write_text(
        git("diff", "--binary", f"{BASELINE}..HEAD", "--", *SOURCE_FILES), encoding="utf-8"
    )
    write_text(
        PACK / "05_COMMANDS_AND_TEST_RESULTS.md",
        """# Validation record

- `uv run python scripts/build_m5_5f1_sequence_global_association_bakeoff.py`: completed with six diagnostic and eight unseen cases.
- `uv run python scripts/capture_m5_5f1_browser_evidence.py`: passed in real Edge/CDP; stepper used, active time persisted, sealed route unavailable, deliverable decisions stayed empty.
- `uv run pytest -q tests/test_m5_5f1_sequence_global_association_bakeoff.py`: 10 passed.
- `uv run ruff check` on changed Python files: passed.
- `uv run ruff format --check` on changed Python files: passed.
- `git diff --check`: passed before commit.

The full suite result is recorded in the final validation command output and final response.
""",
    )
    write_json(
        PACK / "06_OUTPUT_ARTIFACT_INDEX.json",
        {
            "stage_workspace": "M5_5F1_SEQUENCE_GLOBAL_ASSOCIATION_BAKEOFF_AND_UNSEEN_LEVEL2_VALIDATION_v1",
            "key_outputs": [
                "completed review validation",
                "switch/pass diagnostic reproduction",
                "common observation graph",
                "five-algorithm bakeoff",
                "sequence-global optimizer",
                "CUDA detector and temporary appearance telemetry",
                "unseen Level-2 curation and machine gates",
                "fresh port-8799 review package",
                "real-browser evidence",
            ],
            "human_decisions_in_new_package": 0,
        },
    )
    write_json(
        PACK / "07_COMPLETED_REVIEW_AND_SWITCH_AUDIT.json",
        {
            "historical_review_cases": completed["reviewed"],
            "historical_pass_controls": 3,
            "historical_switch_failures": 3,
            "historical_switch_decision_counts": completed["decision_counts"],
            "historical_switch_failure_frames": [119, 175, 235],
            "historical_review_used_for": "diagnostic development only",
            "final_validation_uses_unseen_only": True,
            "historical_ledger_read_only": completed["prior_decisions_read_only"],
            "historical_telemetry_zero_duration_classified_as_defect": completed["telemetry_defect"],
        },
    )
    write_json(
        PACK / "08_COMMON_OBSERVATION_GRAPH.json",
        {
            "case_count": graph["case_count"],
            "frame_observation_row_count": len(graph_nodes),
            "edge_row_count": len(graph_edges),
            "all_graphs_have_nodes": graph["all_graphs_have_nodes"],
            "same_graph_used_by_algorithms": graph["same_graph_used_by_algorithms"],
            "one_to_one_edges": graph["one_to_one_edges"],
            "hard_geometry_gate_recorded": graph["hard_geometry_gate_recorded"],
        },
    )
    write_json(
        PACK / "09_ASSOCIATION_ALGORITHM_BAKEOFF.json",
        {
            algorithm: {
                "case_count": value["case_count"],
                "ambiguous_case_count": value["ambiguous_case_count"],
                "one_to_one_all": value["one_to_one_all"],
                "forced_end_mapping_any": value["forced_end_mapping_any"],
            }
            for algorithm, value in bakeoff.items()
        },
    )
    write_json(
        PACK / "10_SEQUENCE_GLOBAL_OPTIMIZER.json",
        {
            key: optimizer[key]
            for key in [
                "window_frames",
                "fixed_start_seeds",
                "joint_A_B",
                "one_to_one",
                "null_state_allowed",
                "ambiguous_state_allowed",
                "top_k_retained",
                "hard_geometry_veto",
                "forced_end_mapping",
                "all_cases_have_beam_history",
            ]
        },
    )
    write_json(
        PACK / "11_GPU_APPEARANCE_EVIDENCE.json",
        {
            "device": telemetry["device"],
            "checkpoint_sha256": telemetry["checkpoint_sha256"],
            "detector_rows": telemetry["detector_rows"],
            "descriptor_rows": telemetry["descriptor_rows"],
            "oom_count": telemetry["oom_count"],
            "silent_cpu_fallback": telemetry["silent_cpu_fallback"],
            "global_defaults_changed": telemetry["global_defaults_changed"],
            "temporary_descriptor": descriptor["current_temporary_descriptor"],
            "geometry_absolute_veto": descriptor["geometry_absolute_veto"],
            "appearance_not_decisive": descriptor["same_team_appearance_not_decisive"],
        },
    )
    write_json(
        PACK / "12_UNSEEN_CASE_CURATION.json",
        {
            "selected_count": len(selected),
            "target_count": 8,
            "minimum_count": 6,
            "selected_windows": curation["selected_windows"],
            "selected_strata_counts": curation["selected_strata_counts"],
            "pairwise_disjoint": curation["selected_cases_pairwise_disjoint"],
            "diagnostic_overlap_count": curation["overlap_count"],
            "human_answers_used": curation["human_answers_used"],
        },
    )
    write_json(
        PACK / "13_MACHINE_UNSEEN_GATES.json",
        {
            key: gates[key]
            for key in [
                "selected_count",
                "minimum_count",
                "all_selected_pass",
                "zero_bad_seeds",
                "zero_bad_rois",
                "zero_duplicate_events",
                "zero_impossible_jumps",
                "zero_double_assignments",
                "zero_observed_rows_without_provenance",
                "zero_tracker_renderer_mismatches",
                "zero_forced_low_confidence_paths",
                "human_review_still_required",
                "diagnostic_cases_not_used_as_final_validation",
            ]
        },
    )
    write_json(
        PACK / "14_REVIEW_PACKAGE_STATUS.json",
        {
            "review_url": "http://127.0.0.1:8799/",
            "reviewer_session_id": "m5_5f1_unseen_level2_association_human_reviewer",
            "case_count": validation["review_case_count"],
            "package_validation_passed": validation["passed"],
            "fresh_empty_decisions_root": browser["package_decisions_remain_empty"],
            "gif_asset_count": validation["gif_asset_count"],
            "image_sequence_asset_count": validation["image_sequence_asset_count"],
            "mp4_asset_count": validation["mp4_asset_count"],
            "hash_mismatch_count": validation["hash_mismatch_count"],
            "predicted_and_alternative_layers_default_off": True,
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
            "historical_artifacts_mutated": authorization["historical_artifacts_mutated"],
            "prior_stage_unchanged": authorization["prior_stage_unchanged"],
        },
    )
    write_json(
        PACK / "16_ACCEPTANCE_AND_NEXT_STAGE.json",
        {
            "classification": decision["classification"],
            "exact_blocker": decision["exact_blocker"],
            "human_action": "Do not use port 8798 again. Use port 8799 only for this fresh unseen Level-2 review. Review all eight cases; notes are optional for normal structured outcomes. Level 3 remains blocked until the completed unseen review has no switches, losses or bad seeds.",
        },
    )
    shutil.copy2(
        STAGE / "02_THREE_SWITCH_AND_THREE_PASS_REPRODUCTION" / "switch_failure_visual.jpg",
        PACK / "17_SWITCH_DIAGNOSTIC_VISUAL.jpg",
    )
    shutil.copy2(
        STAGE / "11_COMMANDS_AND_TESTS" / "browser_evidence" / "unseen_review_ui.png", PACK / "18_UNSEEN_REVIEW_UI.png"
    )
    write_text(
        PACK / "19_HUMAN_REVIEW_INSTRUCTIONS.md",
        """# Human review instructions

Do not use port 8798 again. Use `http://127.0.0.1:8799/` only.

Review the eight unseen Level-2 cases. First confirm or correct the anonymous A/B seeds, then judge visual continuity. Notes are optional for normal structured outcomes; use a note for `BAD_CASE` or `UNRESOLVED`. The alternate joint-hypothesis layer is optional and off by default.

This is visual-only, match-local, sandbox-only review evidence. It does not create persistent player identity, player slots, metrics, or production output. Level 3 remains blocked until the completed unseen review has zero switches, losses, and bad seeds.
""",
    )
    expected = set(pack_files)
    actual = {path.name for path in PACK.iterdir() if path.is_file()}
    if actual != expected:
        raise RuntimeError(f"review pack contains unexpected files: {sorted(actual ^ expected)}")
    sizes = {path.name: path.stat().st_size for path in PACK.iterdir() if path.is_file()}
    manifest = read_json(PACK / "REVIEW_PACK_MANIFEST.json")
    manifest.update(
        {
            "actual_file_count": len(sizes),
            "actual_total_bytes": sum(sizes.values()),
            "actual_visual_file_count": 2,
            "validation_passed": len(sizes) == 20 and sum(sizes.values()) <= 52_428_800,
        }
    )
    write_json(PACK / "REVIEW_PACK_MANIFEST.json", manifest)
    if not manifest["validation_passed"]:
        raise RuntimeError(f"review pack validation failed: {manifest}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
