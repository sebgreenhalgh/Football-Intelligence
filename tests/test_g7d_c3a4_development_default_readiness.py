from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from football_intelligence.proposal_gate_hook import DEFAULT_PITCH_GATE_MODE, PitchGateMode

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT / "SoccerTrack-v2"
STAGE = PROJECT / "experiments/football_observation_reasoner/part 7" / "G7D_C3A4_DEVELOPMENT_DEFAULT_READINESS_AUDIT_v1"
EXPECTED_HEAD = "186dc876c9f08c509e4831917702ac51002ed0e6"
DECISION = "PASS_G7D_C3A4_DEFERRED_FOR_ADDITIONAL_COVERAGE"


def read(relative: str):
    return json.loads((STAGE / relative).read_text(encoding="utf-8-sig"))


def test_expected_head_branch_and_default_remain_unchanged() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_HEAD, "HEAD"], cwd=REPO, check=True)
    assert (
        subprocess.run(
            ["git", "branch", "--show-current"], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.strip()
        == "main"
    )
    assert DEFAULT_PITCH_GATE_MODE is PitchGateMode.DISABLED
    hook = (REPO / "src/football_intelligence/proposal_gate_hook.py").read_text(encoding="utf-8")
    assert "DEFAULT_PITCH_GATE_MODE = PitchGateMode.DISABLED" in hook


def test_exact_c3a3_frozen_evidence_is_closed() -> None:
    closure = read("01_INPUT_CLOSURE/input_closure.json")
    assert closure["classification"] == "PASS_G7D_C3A4_FROZEN_EVIDENCE_CLOSURE"
    assert all(closure["checks"].values())
    assert closure["frames"] == 96
    assert closure["control_candidates"] == 5940
    assert closure["retained_candidates"] == 4252
    assert closure["suppressed_candidates"] == 1688
    assert closure["candidate_fold_outputs"] == 21260
    assert round(closure["runtime_seconds"], 3) == 464.570
    assert closure["runtime_envelope_seconds"] == [370.13335, 500.76865]
    assert round(closure["c3a2_measured_runtime_reduction_fraction"], 4) == 0.3743
    assert round(closure["c3a2_measured_speedup_factor"], 3) == 1.598
    assert closure["reviewed_safety"] == {
        "active_players_goalkeepers": "77/77",
        "officials": "10/10",
        "unsafe_missed_person_neighbourhood_losses": 0,
        "useful_relevant": "87/87",
    }


def test_exact_train_development_inventory_only() -> None:
    coverage = read("02_COVERAGE_AUDIT/coverage_matrix.json")
    assert coverage["train_match_count"] == 6
    assert [row["match_id"] for row in coverage["matches"]] == [
        "117092",
        "117093",
        "118575",
        "118576",
        "118577",
        "128058",
    ]
    assert {row["split"] for row in coverage["matches"]} == {"TRAIN_DEVELOPMENT"}
    assert coverage["validation_or_holdout_content_accessed"] is False


def test_polygon_paths_hashes_and_camera_segments() -> None:
    coverage = read("02_COVERAGE_AUDIT/coverage_matrix.json")
    by_match = {row["match_id"]: row for row in coverage["matches"]}
    expected = {
        "117092": "92ca8040eedd3b0ec0bb685648691f0c314d8527f3fa8f2db1823b4461e4b338",
        "118575": "fbd7f3a473acc197b4c893d90bbaa4c5d484d1e883e8df1ac4601daf4396dec1",
        "128058": "24ad1e4d143527e5a3e92cded1b5d8b10526d67b5b0d1f8b02289a91e8c65307",
    }
    for match, sha256 in expected.items():
        polygon = by_match[match]["polygon"]
        path = PROJECT / polygon["path"]
        assert polygon["status"] == "HUMAN_CONFIRMED"
        assert polygon["declared_sha256"] == polygon["actual_sha256"] == sha256
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256
        assert polygon["camera_segment_count"] == 1
        assert polygon["second_half_alignment"] == "YES"
    for match in ("117093", "118576", "118577"):
        polygon = by_match[match]["polygon"]
        assert polygon["status"] == "HUMAN_REQUIRED"
        assert polygon["path"] is None and polygon["declared_sha256"] is None


def test_exact_runtime_polygon_and_condition_coverage() -> None:
    coverage = read("02_COVERAGE_AUDIT/coverage_matrix.json")
    assert coverage["authoritative_polygon_count"] == 3
    assert coverage["authoritative_polygon_fraction"] == 0.5
    assert coverage["authoritative_polygon_matches_in_runtime"] == 3
    assert coverage["all_authoritative_polygon_matches_in_runtime"] is True
    assert set(coverage["evaluated_matches"]) == {"117092", "118575", "128058"}
    assert set(coverage["unevaluated_matches"]) == {"117093", "118576", "118577"}
    by_match = {row["match_id"]: row for row in coverage["matches"]}
    assert by_match["117092"]["conditions"]["lighting"] == "NIGHT"
    assert by_match["117092"]["conditions"]["unusual_conditions"] == "FLOODLIGHT_GLARE, LOW_LIGHT"
    for match in ("117092", "118575", "128058"):
        assert by_match[match]["runtime_evidence"]["frames"] == 32


def test_edge_case_classifications_preserve_known_limitations() -> None:
    matrix = read("03_EDGE_CASE_AUDIT/edge_case_matrix.json")
    edges = {row["edge_case"]: row for row in matrix["edges"]}
    assert edges["assistant referee near the touchline"]["classification"] == "COVERED_AND_PASSING"
    assert edges["assistant referee near the touchline"]["support"] == 9
    assert edges["active player just outside the pitch"]["support"] == 1
    assert edges["boundary-uncertain person"]["support"] == 21
    assert edges["low-light glare"]["classification"] == "COVERED_AND_PASSING"
    assert edges["goalkeeper behind the goal line"]["classification"] == "NOT_COVERED"
    assert edges["goalkeeper behind the goal line"]["support"] == 0
    assert edges["player retrieving the ball"]["classification"] == "NOT_COVERED"
    assert edges["multiple camera segments"]["classification"] == "NOT_COVERED"
    assert edges["missing or invalid polygon"]["classification"] == "PARTIALLY_COVERED"
    assert edges["extreme panorama distortion"]["classification"] == "NOT_COVERED"
    assert edges["dense or crowded scenes"]["classification"] == "PARTIALLY_COVERED"
    assert matrix["known_coverage_limitation"] == "NO_GOAL_LINE_NEAREST_GOALKEEPER_CASE"


def test_predeclared_promotion_rule_deterministically_defers() -> None:
    criteria = read("05_DECISION/promotion_criteria.json")
    decision = read("05_DECISION/decision.json")
    assert [row["criterion"] for row in criteria["criteria"] if row["pass"]] == [1, 2, 3, 4, 6, 7, 8]
    assert criteria["failed_criteria"] == [5]
    assert decision["classification"] == DECISION
    assert decision["decision"] == "DEFER_FOR_ADDITIONAL_COVERAGE"
    assert decision["primary_blocker"] == "NO_GOAL_LINE_NEAREST_GOALKEEPER_CASE"
    assert decision["default_changed"] is False and decision["project_default"] == "DISABLED"


def test_additional_coverage_plan_is_exact_and_bounded() -> None:
    plan = read("05_DECISION/additional_coverage_plan.json")
    expected = ["117093", "118576", "118577"]
    assert plan["matches_needing_human_confirmed_polygons"] == expected
    assert plan["matches_needing_bounded_replay"] == expected
    assert plan["limits"] == {
        "additional_matches": 3,
        "maximum_frames_per_match": 16,
        "maximum_frames_total": 48,
        "maximum_scene_checks_total": 12,
        "maximum_targeted_candidate_decisions_total": 60,
    }
    assert sum(plan["allocation"]["frames"].values()) == 48
    assert sum(plan["allocation"]["targeted_candidate_decisions"].values()) == 60
    assert sum(plan["allocation"]["scene_checks"].values()) == 12
    assert plan["pass_fail_thresholds"]["goal_line_nearest_or_behind_goal_goalkeeper_support_minimum"] == 1
    assert plan["pass_fail_thresholds"]["uncovered_high_severity_case_after_round"] == 0


def test_draft_policy_is_inactive_scoped_and_fail_closed() -> None:
    policy = read("04_DEFAULT_POLICY_DRAFT/development_default_policy_draft.json")
    assert policy["policy_id"] == "G7D_C3A4_DEVELOPMENT_DEFAULT_POLICY_DRAFT_V1"
    assert policy["status"] == "DRAFT_NOT_ACTIVE"
    assert policy["active"] is False and policy["component_promoted"] is False
    assert policy["project_default_before_and_after"] == "DISABLED"
    assert policy["applies_only_to"] == "TRAIN_DEVELOPMENT_EXPERIMENTS"
    assert policy["fail_closed"]["result"] == "DISABLED"
    assert policy["fail_closed"]["silent_active_fallback"] is False
    assert set(policy["excluded"]) == {
        "VALIDATION_MODEL_SELECTION",
        "SEALED_HOLDOUT",
        "PRODUCTION",
        "HISTORICAL_FROZEN_OUTPUTS",
    }
    assert policy["production_ready"] is False


def test_one_visual_nine_file_handoff_and_no_inference_or_default_mutation() -> None:
    visual = STAGE / "06_VISUAL_QA/DEVELOPMENT_DEFAULT_READINESS_MATRIX.png"
    assert visual.is_file() and visual.stat().st_size > 0
    handoff = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    files = {path.name for path in handoff.iterdir() if path.is_file()}
    assert files == {
        "01_EXECUTIVE_SUMMARY.json",
        "02_MATCH_AND_POLYGON_COVERAGE.json",
        "03_EDGE_CASE_COVERAGE.json",
        "04_PROMOTION_CRITERIA_AND_DECISION.json",
        "05_ADDITIONAL_COVERAGE_PLAN.json",
        "06_DEVELOPMENT_DEFAULT_POLICY_DRAFT.md",
        "07_TESTS_SAFETY_AND_SOURCE_CHANGES.json",
        "08_READINESS_MATRIX.png",
        "09_MANIFEST.json",
    }
    manifest = read("07_REVIEW_PACK/CHATGPT_HANDOFF/09_MANIFEST.json")
    assert manifest["file_count"] == 8
    assert {row["filename"] for row in manifest["files"]} == files - {"09_MANIFEST.json"}
    for row in manifest["files"]:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    safety = read("06_TESTS_AND_LOGS/source_changes_and_safety.json")
    assert safety["inference_run"] is False
    assert safety["training_or_tuning_run"] is False
    assert safety["validation_or_holdout_content_accessed"] is False
    assert safety["runtime_default_changed"] is False
    assert safety["full_suite_run"] is False
