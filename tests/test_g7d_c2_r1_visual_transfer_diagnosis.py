from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from football_intelligence.g7d_c2_visual_transfer_diagnosis import (
    TARGETED_WARNING,
    box_metrics,
    candidate_flags,
    choose_next_stage,
    classification_metrics,
    polygon_location,
)


STAGE = Path(
    r"C:\Users\sebgr\Documents\football-intelligence\experiments\football_observation_reasoner\part 6"
    r"\G7D_C2_R1_RESUME_VISUAL_TRANSFER_DIAGNOSIS_FINALIZATION_v1"
)


def load(relative: str) -> dict:
    return json.loads((STAGE / relative).read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def test_analysis_helpers_preserve_documented_semantics() -> None:
    flags = candidate_flags(
        {
            "proposal_validity": "CLEAN_SINGLE_PERSON",
            "role": "OUTFIELD_PLAYER",
            "box_quality": "GOOD_SINGLE_PERSON_BOX",
            "participation": "ACTIVE",
            "occlusion": "NONE",
        }
    )
    assert flags["box_is_useful_single_person"]
    assert flags["is_relevant_active_population"]
    assert not flags["is_background_or_object"]

    geometry = box_metrics([2, 2, 4, 4], [0, 0, 10, 10])
    assert geometry["intersection_over_inner_area"] == pytest.approx(1.0)
    assert geometry["inner_outer_area_ratio"] == pytest.approx(0.04)

    inside = polygon_location([5, 5], [[0, 0], [10, 0], [10, 10], [0, 10]], 100)
    outside = polygon_location([30, 30], [[0, 0], [10, 0], [10, 10], [0, 10]], 100)
    assert inside["inside_polygon"]
    assert outside["geometry_band"] == "FAR_OUTSIDE_POLYGON"


def test_fold_metrics_and_decision_are_fold_local_and_deterministic() -> None:
    metrics = classification_metrics(
        ["A", "B"],
        [[0.9, 0.1], [0.2, 0.8]],
        ["A", "B"],
    )
    assert metrics["exact_agreement"] == 1.0
    assert metrics["warning"] == TARGETED_WARNING
    evidence = {
        "pitch_clutter_removed_rate": 0.578,
        "pitch_relevant_useful_loss_rate": 0.0,
        "nested_separable_burden_rate": 0.02,
        "crowding_temporal_support_rate": 0.75,
        "useful_box_semantic_error_rate": 0.94,
    }
    decision = choose_next_stage(evidence)
    assert decision["primary_stage"] == "G7D_C3A_PITCH_AWARE_PROPOSAL_GATE_EXPERIMENT"
    assert decision["conditional_secondary_stage"] == "G7E_TARGETED_TEMPORAL_ANNOTATION"
    assert not decision["implemented"]


def test_continuation_and_human_chain_are_exact() -> None:
    continuation = load("00_CONTINUATION_PROVENANCE/continuation_validation.json")
    closure = load("01_HUMAN_REVIEW_CLOSURE/human_review_validation_report.json")
    assert continuation["classification"] == "PASS_G7D_C2_R1_FROZEN_PROVENANCE"
    assert continuation["partial_analysis_outputs_found"] is False
    assert continuation["completion_receipt_id"] == "completion-r8-bbbaabc5fdbff19754baee53"
    assert continuation["latest_event_set_digest"] == (
        "bbbaabc5fdbff19754baee53dce8342a91f49c92967bf319398b1ba30e7b4e08"
    )
    assert continuation["latest_counts"] == {"candidate": 192, "scene": 24, "total": 216}
    assert continuation["all_acknowledgements_valid"] is True
    assert closure["candidate_count"] == 192
    assert closure["scene_count"] == 24


def test_analysis_is_descriptive_and_keeps_all_folds_separate() -> None:
    policy = load("03_PITCH_AND_POPULATION_DIAGNOSIS/pitch_polygon_filter_simulation.json")
    nested = load("04_NESTED_CANDIDATE_DIAGNOSIS/nested_candidate_summary.json")
    scene = load("05_SCENE_DIAGNOSIS/scene_coverage_summary.json")
    folds = load("07_FOLDWISE_SEMANTIC_DIAGNOSIS/foldwise_semantic_diagnosis.json")
    report = load("11_FINALIZATION_EVIDENCE/finalization_validation_report.json")
    assert policy["applied_to_production"] is False
    assert policy["policies"]["C_FAR_OUTSIDE_WITH_RELEVANT_EXCEPTIONS"]["reviewed_useful_relevant_lost"] == 0
    assert nested["reviewed_useful_targets_lost"] == 0
    assert scene["scene_count"] == 24 and scene["total_missed_person_marks"] == 22
    assert folds["aggregation"] == "NONE"
    assert sorted(folds["folds"]) == ["0", "1", "2", "3", "4"]
    assert all(
        set(folds["folds"][str(index)]) == {"candidate_state", "role", "team", "participation", "pitch"}
        for index in range(5)
    )
    assert report["inference_rerun"] is False
    assert report["thresholds_changed"] is False
    assert report["validation_or_holdout_access"] is False
    assert report["production_filter_or_suppression_applied"] is False
    assert report["tests"] == "PASS_FOCUSED_TESTS"


def test_three_real_visuals_and_exact_twelve_file_handoff() -> None:
    visuals = sorted((STAGE / "10_VISUAL_EVIDENCE").glob("*.png"))
    assert len(visuals) == 3
    for path in visuals:
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 50_000

    handoff = STAGE / "12_REVIEW_PACK/CHATGPT_HANDOFF"
    files = sorted(path for path in handoff.iterdir() if path.is_file())
    assert len(files) == 12
    manifest = load("12_REVIEW_PACK/CHATGPT_HANDOFF/12_MANIFEST.json")
    assert manifest["file_count"] == 11
    assert manifest["self_hash_omitted"] is True
    assert {row["filename"] for row in manifest["files"]} == {
        path.name for path in files if path.name != "12_MANIFEST.json"
    }
    for row in manifest["files"]:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"]
        assert digest(path) == row["sha256"]


def test_recommendation_is_bounded_and_not_implemented() -> None:
    recommendation = load("09_NEXT_STAGE_RECOMMENDATION/next_stage_recommendation.json")
    assert recommendation["primary_stage"] == "G7D_C3A_PITCH_AWARE_PROPOSAL_GATE_EXPERIMENT"
    assert recommendation["conditional_secondary_stage"] == "G7E_TARGETED_TEMPORAL_ANNOTATION"
    assert recommendation["implemented"] is False
    assert recommendation["production_change"] is False
    workload = recommendation["bounded_workload"]
    assert (workload["burst_count_min"], workload["burst_count_max"]) == (120, 180)
    assert (workload["frames_per_burst_min"], workload["frames_per_burst_max"]) == (5, 11)
