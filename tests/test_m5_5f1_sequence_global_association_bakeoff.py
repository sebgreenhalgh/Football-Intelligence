from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F1_SEQUENCE_GLOBAL_ASSOCIATION_BAKEOFF_AND_UNSEEN_LEVEL2_VALIDATION_v1"
)
PACKAGE = STAGE / "09_UNSEEN_LEVEL2_ASSOCIATION_REVIEW_PACKAGE"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_completed_review_and_switch_failures_are_bound_exactly() -> None:
    review = read_json(STAGE / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "completed_review_validation.json")
    assert review["reviewed"] == 6
    assert review["remaining"] == 0
    assert review["decision_counts"] == {"A_SWITCH": 2, "B_SWITCH": 1, "PASS": 3}
    assert review["all_seed_actions_confirm"] is True
    failures = read_jsonl(STAGE / "02_THREE_SWITCH_AND_THREE_PASS_REPRODUCTION" / "switch_failure_rows.jsonl")
    passes = read_jsonl(STAGE / "02_THREE_SWITCH_AND_THREE_PASS_REPRODUCTION" / "pass_control_rows.jsonl")
    assert {row["failure_frame"] for row in failures} == {119, 175, 235}
    assert len(passes) == 3
    assert all(row["human_label_used_for"] == "diagnostic_only_not_parameter_selection" for row in failures)


def test_all_algorithms_use_one_immutable_observation_graph() -> None:
    graph = read_json(STAGE / "03_COMMON_OBSERVATION_GRAPH" / "graph_validation.json")
    bakeoff = read_json(STAGE / "04_ASSOCIATION_ALGORITHM_BAKEOFF" / "bakeoff_summary.json")
    assert graph["same_graph_used_by_algorithms"] is True
    assert graph["one_to_one_edges"] is True
    assert graph["hard_geometry_gate_recorded"] is True
    assert set(bakeoff) == {
        "CURRENT_REPAIRED_LOCAL",
        "OBSERVATION_CENTRIC_MOTION",
        "TWO_STAGE_CONFIDENCE_ASSOCIATION",
        "ADAPTIVE_MOTION_APPEARANCE",
        "JOINT_SEQUENCE_GLOBAL_TWO_STRAND",
    }
    digests = {tuple(value["graph_digests"]) for value in bakeoff.values()}
    assert len(digests) == 1
    assert all(value["one_to_one_all"] for value in bakeoff.values())
    assert all(not value["forced_end_mapping_any"] for value in bakeoff.values())


def test_global_optimizer_is_joint_nullable_top_k_and_geometry_gated() -> None:
    summary = read_json(STAGE / "05_SEQUENCE_GLOBAL_TWO_STRAND_OPTIMIZER" / "optimizer_summary.json")
    assert summary["fixed_start_seeds"] is True
    assert summary["joint_A_B"] is True
    assert summary["one_to_one"] is True
    assert summary["null_state_allowed"] is True
    assert summary["ambiguous_state_allowed"] is True
    assert summary["top_k_retained"] is True
    assert summary["hard_geometry_veto"] is True
    assert summary["forced_end_mapping"] is False
    assert summary["all_cases_have_beam_history"] is True


def test_gpu_detector_and_temporary_appearance_have_no_fallback() -> None:
    telemetry = read_json(STAGE / "06_GPU_APPEARANCE_AND_MOTION_EVIDENCE" / "gpu_timing_and_memory.json")
    assert telemetry["checkpoint_sha256"] == "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
    assert telemetry["device"] == "cuda:0"
    assert telemetry["oom_count"] == 0
    assert telemetry["silent_cpu_fallback"] is False
    assert telemetry["detector_rows"] > 0
    assert telemetry["descriptor_rows"] > 0
    descriptor = read_json(STAGE / "06_GPU_APPEARANCE_AND_MOTION_EVIDENCE" / "descriptor_comparison.json")
    assert descriptor["geometry_absolute_veto"] is True
    assert descriptor["same_team_appearance_not_decisive"] is True


def test_unseen_cases_are_pairwise_disjoint_and_stratified() -> None:
    audit = read_json(STAGE / "07_UNSEEN_LEVEL2_CASE_CURATION" / "temporal_exclusion_audit.json")
    selected = read_jsonl(STAGE / "07_UNSEEN_LEVEL2_CASE_CURATION" / "selected_unseen_cases.jsonl")
    assert len(selected) == 8
    assert audit["overlap_count"] == 0
    assert audit["selected_cases_pairwise_disjoint"] is True
    assert audit["human_answers_used"] is False
    assert audit["selected_strata_counts"] == {
        "cross_team_distractor": 2,
        "easy_separated_pair": 2,
        "moderate_motion_scale_change": 2,
        "same_team_nearby_distractor": 2,
    }
    excluded = [tuple(window) for window in audit["excluded_windows"]]
    assert all(
        all(end < excluded_start or start > excluded_end for excluded_start, excluded_end in excluded)
        for start, end in audit["selected_windows"]
    )


def test_unseen_machine_gates_pass_without_human_selection() -> None:
    summary = read_json(STAGE / "08_MACHINE_ONLY_UNSEEN_GATES" / "unseen_gate_summary.json")
    rows = read_jsonl(STAGE / "08_MACHINE_ONLY_UNSEEN_GATES" / "machine_gate_rows.jsonl")
    assert summary["selected_count"] == 8
    assert summary["minimum_count"] == 6
    assert summary["all_selected_pass"] is True
    assert summary["diagnostic_cases_not_used_as_final_validation"] is True
    assert all(row["passed"] for row in rows if row["case_id"].startswith("m5_5f1_unseen_level2_case_"))
    assert all(row["observed_source_rows_have_provenance"] for row in rows)
    assert all(row["impossible_jumps"] == 0 for row in rows)


def test_fresh_package_is_premium_gif_backed_and_empty() -> None:
    manifest = read_json(PACKAGE / "reviewer_manifest.json")
    ui = read_json(PACKAGE / "ui_config.json")
    decisions = read_json(PACKAGE / "decisions" / "review_decisions.json")
    validation = read_json(PACKAGE / "review_package_validation.json")
    assert manifest["review_id"] == "m5_5f1_unseen_level2_association_review_v1"
    assert len(manifest["cases"]) == 8
    assert all(len(case["visible_metadata"]["frame_records"]) == 13 for case in manifest["cases"])
    assert all(case["visible_metadata"]["benchmark_level"] == 2 for case in manifest["cases"])
    assert decisions["decisions"] == {}
    assert decisions["structured_reviews"] == {}
    assert ui["presentation_mode"] == "stable_local_strand_continuity"
    assert ui["question_contract"]["alternative_hypothesis_toggle_enabled"] is True
    assert ui["question_contract"]["alternative_hypothesis_default_off"] is True
    assert validation["passed"] is True
    assert validation["gif_asset_count"] == 16
    assert validation["hash_mismatch_count"] == 0
    assert validation["mp4_asset_count"] == 0
    assert "8799" in (PACKAGE / "launch_review.ps1").read_text(encoding="utf-8")


def test_browser_smoke_proves_privacy_stepper_and_active_time() -> None:
    browser = read_json(STAGE / "11_COMMANDS_AND_TESTS" / "browser_evidence" / "browser_validation.json")
    assert browser["real_browser"] is True
    assert browser["initial"]["case_count"] == 8
    assert browser["initial"]["predicted_default_off"] is True
    assert browser["initial"]["alternative_toggle_default_off"] is True
    assert browser["initial"]["alternative_layer_hidden"] is True
    assert browser["initial"]["answer_or_algorithm_leak"] is False
    assert browser["stepper_used"] is True
    assert browser["saved_decision_count_in_smoke_root"] == 1
    assert browser["active_time_nonzero"] is True
    assert browser["sealed_route_unavailable"] is True
    assert browser["package_decisions_remain_empty"] is True


def test_shared_ui_records_active_seconds_and_alternative_layer() -> None:
    app = (REPO / "src" / "football_intelligence" / "review_chassis" / "static" / "app.js").read_text(encoding="utf-8")
    index = (REPO / "src" / "football_intelligence" / "review_chassis" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "Math.ceil(activeTimeAccumulated)" in app
    assert "elapsed_active_seconds: activeTimeNow()" in app
    assert "premiumAlternativeToggle" in app
    assert "premiumAlternativeLayer" in index


def test_safety_flags_and_prior_artifact_preservation_are_explicit() -> None:
    validation = read_json(PACKAGE / "review_package_validation.json")
    authorization = read_json(STAGE / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "authorization_audit.json")
    next_stage = read_json(STAGE / "10_EVALUATION_AND_NEXT_STAGE" / "next_stage_decision.json")
    assert validation["human_approved"] is False
    assert validation["production_ready"] is False
    assert validation["no_auto_promotion"] is True
    assert validation["safe_to_apply_globally"] is False
    assert authorization["prior_stage_unchanged"] is True
    assert authorization["historical_artifacts_mutated"] is False
    assert next_stage["classification"] == "PASS_UNSEEN_LEVEL2_ASSOCIATION_REVIEW_READY"
    assert "Level 3 remains blocked" in next_stage["exact_blocker"]
