from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
STAGE = (
    PROJECT
    / "experiments/football_observation_reasoner/part 7"
    / "G7D_C3A5D_ADDITIONAL_COVERAGE_FINALIZATION_AND_DEFAULT_DECISION_v1"
)
SOURCE = (
    PROJECT
    / "experiments/football_observation_reasoner/part 7"
    / "G7D_C3A5C_ADDITIONAL_COVERAGE_REPLAY_AND_REVIEW_v1"
    / "04_ADDITIONAL_COVERAGE_REVIEW_PACKAGE/human_decisions"
)
DECISION = "PASS_G7D_C3A5D_DEVELOPMENT_DEFAULT_PROMOTION_APPROVED"


def load(relative: str) -> dict:
    return json.loads((STAGE / relative).read_text(encoding="utf-8"))


def jsonl(relative: str) -> list[dict]:
    return [json.loads(line) for line in (STAGE / relative).read_text(encoding="utf-8").splitlines()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_exact_human_event_chain_and_visible_event_resolution() -> None:
    report = load("01_HUMAN_REVIEW_CLOSURE/event_chain_validation.json")
    assert report["classification"] == "PASS_G7D_C3A5D_HUMAN_EVENT_CHAIN"
    assert (report["candidate_event_count"], report["scene_event_count"]) == (60, 12)
    assert report["latest_acknowledged_event_count"] == report["acknowledgement_receipt_count"] == 72
    assert report["completion_receipt_count"] == 1
    assert report["completion_receipt_id"] == "completion-35fed8f25691bc05701601fe"
    assert report["all_cases_complete"] is True
    assert len(report["event_and_receipt_manifest"]) == 145
    assert report["synthetic_or_temporary_event_count"] == 0
    assert len(jsonl("01_HUMAN_REVIEW_CLOSURE/candidate_human_labels.jsonl")) == 60
    assert len(jsonl("01_HUMAN_REVIEW_CLOSURE/scene_human_labels.jsonl")) == 12
    assert report["visible_last_event_resolution"] == {
        "event_id": "796d658e-980b-456b-a6b0-391f96a1f72d",
        "event_type": "scene",
        "stable_id": "scene_12_118577_stable_control",
        "server_sequence": 72,
    }


def test_human_event_and_receipt_bytes_are_unchanged() -> None:
    report = load("00_INPUT_AND_EVENT_CLOSURE/human_source_preservation.json")
    assert report["classification"] == "PASS_G7D_C3A5D_IMMUTABLE_HUMAN_TRUTH_PRESERVED"
    assert report["before_file_count"] == report["after_file_count"] == 145
    assert report["byte_identical"] is True
    for row in report["after"]:
        path = PROJECT / row["project_relative_path"]
        assert path.stat().st_size == row["byte_size"]
        assert sha256(path) == row["sha256"]
    assert len(list((SOURCE / "events/candidate").glob("*.json"))) == 60
    assert len(list((SOURCE / "events/scene").glob("*.json"))) == 12


def test_normalized_labels_are_canonical_and_do_not_infer_team() -> None:
    candidates = jsonl("01_HUMAN_REVIEW_CLOSURE/candidate_human_labels.jsonl")
    scenes = jsonl("01_HUMAN_REVIEW_CLOSURE/scene_human_labels.jsonl")
    marks = jsonl("01_HUMAN_REVIEW_CLOSURE/missed_person_marks.jsonl")
    assert len(candidates) == 60 and len(scenes) == 12 and len(marks) == 3
    assert len({row["target_id"] for row in candidates}) == 60
    assert all("team" not in json.dumps(row["canonical_decision"]).lower() for row in candidates)
    assert all(row["team_inferred"] is False and row["human_role"] is None for row in marks)
    assert all(row["full_frame_coverage_confirmed"] is True for row in scenes)


def test_additional_candidate_safety_is_exact_zero_loss() -> None:
    report = load("02_ADDITIONAL_CANDIDATE_SAFETY/candidate_gate_safety.json")
    assert report["reviewed_candidate_count"] == 60
    assert {key: value["reviewed"] for key, value in report["by_gate_decision"].items()} == {
        "KEEP": 16,
        "BOUNDARY_REVIEW": 16,
        "SUPPRESS_SANDBOX": 25,
        "EXCEPTION_KEEP": 3,
    }
    assert set(report["critical_suppression_counts"].values()) == {0}
    suppressed = report["by_gate_decision"]["SUPPRESS_SANDBOX"]
    assert suppressed["background_or_object"] == 21
    assert suppressed["out_of_scope_people"] == 4
    assert report["boundary_review_useful_people"] == 6
    assert report["exception_keep_useful_people"] == 1


def test_goalkeeper_outside_player_and_official_evidence_is_safe() -> None:
    report = load("03_EDGE_CASE_AND_SCENE_SAFETY/scene_edge_case_summary.json")
    goalkeeper = report["goalkeeper_at_or_behind_endline"]
    assert goalkeeper["positive_scene_count"] == 3
    assert goalkeeper["required_minimum_met"] is True
    assert goalkeeper["unsafe_suppression_count"] == 0
    assert len(goalkeeper["safe_retained_associations"]) == 3
    assert all(
        all(link["gate_decision"] != "SUPPRESS_SANDBOX" for link in row["retained_candidate_links"])
        for row in goalkeeper["safe_retained_associations"]
    )
    outside = report["player_temporarily_outside_or_retrieving_ball"]
    assert outside["positive_scene_count"] == 2
    assert outside["direct_human_candidate_support"]["gate_decision"] == "BOUNDARY_REVIEW"
    official = report["relevant_official_near_touchline"]
    assert official["positive_scene_count"] == 12
    assert official["direct_reviewed_official_count"] == official["direct_reviewed_officials_retained"] == 6


def test_all_additional_missed_person_neighbourhoods_are_preserved() -> None:
    report = load("03_EDGE_CASE_AND_SCENE_SAFETY/missed_person_neighbourhood_safety.json")
    assert report["mark_count"] == report["preserved_neighbourhood_count"] == 3
    assert report["proposal_supply_miss_before_gate_count"] == 0
    assert report["unsafe_all_nearby_suppressed_count"] == 0
    assert all(row["retained_nearby_candidate_count"] > 0 for row in report["marks"])


def test_combined_evidence_has_exact_252_36_and_six_matches() -> None:
    report = load("04_COMBINED_DEVELOPMENT_EVIDENCE/combined_six_match_evidence.json")
    assert report["candidate_reviews"] == {"prior": 192, "additional": 60, "combined": 252}
    assert report["whole_scene_reviews"] == {"prior": 24, "additional": 12, "combined": 36}
    assert report["all_six_matches_polygon_and_runtime_valid"] is True
    assert {row["match_id"] for row in report["match_matrix"]} == {
        "117092",
        "117093",
        "118575",
        "118576",
        "118577",
        "128058",
    }
    assert all(row["polygon_status"] == "HUMAN_CONFIRMED" for row in report["match_matrix"])
    assert all(row["runtime_evidence"]["frames"] > 0 for row in report["match_matrix"])
    assert report["missed_person_neighbourhoods"]["unsafe_all_nearby_suppressed"] == 0


def test_updated_edge_matrix_has_no_uncovered_high_severity_case() -> None:
    report = load("04_COMBINED_DEVELOPMENT_EVIDENCE/updated_edge_case_matrix.json")
    assert report["promotion_edge_criterion_pass"] is True
    assert report["uncovered_high_severity_cases"] == []
    assert len(report["edges"]) == 10
    allowed = {"COVERED_AND_PASSING", "NOT_APPLICABLE"}
    assert {row["classification"] for row in report["edges"]} <= allowed


def test_all_eight_criteria_pass_and_decision_is_deterministic() -> None:
    criteria = load("05_PROMOTION_DECISION/promotion_criteria.json")
    decision = load("05_PROMOTION_DECISION/decision.json")
    assert criteria["predeclared_c3a4_criteria_unchanged"] is True
    assert criteria["passed_criteria"] == 8 and criteria["failed_criteria"] == []
    assert [row["criterion"] for row in criteria["criteria"]] == list(range(1, 9))
    assert all(row["pass"] is True for row in criteria["criteria"])
    assert decision["classification"] == DECISION
    assert decision["default_changed"] is False and decision["project_default"] == "DISABLED"
    assert decision["production_ready"] is False


def test_policy_v2_is_inactive_bounded_and_fail_closed() -> None:
    policy = load("05_PROMOTION_DECISION/development_default_policy_draft_v2.json")
    safety = load("07_TESTS_AND_LOGS/source_changes_and_safety.json")
    assert policy["policy_id"] == "G7D_C3A5D_DEVELOPMENT_DEFAULT_POLICY_DRAFT_V2"
    assert policy["status"] == "DRAFT_NOT_ACTIVE" and policy["active"] is False
    assert policy["project_default_before_and_after"] == "DISABLED"
    assert policy["applies_only_to"] == "TRAIN_DEVELOPMENT"
    assert policy["required_polygon_status"] == "HUMAN_CONFIRMED"
    assert policy["required_camera_segment_policy"] == "MATCH_STABLE_CAMERA"
    assert policy["fail_closed"]["result"] == "DISABLED"
    assert policy["fail_closed"]["silent_active_fallback"] is False
    assert policy["excluded"] == ["VALIDATION", "SEALED_HOLDOUT", "PRODUCTION", "HISTORICAL_FROZEN_OUTPUTS"]
    assert policy["production_ready"] is False
    assert safety["runtime_default_changed"] is False
    assert safety["detector_feature_fold_or_pitch_gate_inference_run"] is False
    assert safety["training_tuning_or_recalibration_run"] is False
    assert safety["validation_or_holdout_access"] is False
    hook = (ROOT / "src/football_intelligence/proposal_gate_hook.py").read_text(encoding="utf-8")
    assert "DEFAULT_PITCH_GATE_MODE = PitchGateMode.DISABLED" in hook


def test_exact_two_visuals_and_ten_file_handoff_manifest() -> None:
    visuals = sorted((STAGE / "06_VISUAL_QA").glob("*.png"))
    assert [path.name for path in visuals] == [
        "01_ADDITIONAL_COVERAGE_SAFETY.png",
        "02_FINAL_PROMOTION_READINESS_MATRIX.png",
    ]
    handoff = STAGE / "08_REVIEW_PACK/CHATGPT_HANDOFF"
    assert len(list(handoff.iterdir())) == 10
    manifest = json.loads((handoff / "10_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["file_count"] == 9
    assert "10_MANIFEST.json" not in {row["filename"] for row in manifest["files"]}
    for row in manifest["files"]:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"]
        assert sha256(path) == row["sha256"]
