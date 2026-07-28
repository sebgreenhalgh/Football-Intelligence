from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from football_intelligence.football_observation_reasoner.g7b_stage import (
    BASELINE_COMMIT,
    CANDIDATE_CLASSES,
    EXPECTED_K1_DISTRIBUTIONS,
    KIT_CLASSES,
    PARTICIPATION_CLASSES,
    ROLE_CLASSES,
    STAGE_ID,
    StageLocations,
    TEAM_CLASSES,
    build_k1_join,
    derive_primary_truth,
    node_tabular_features,
    review_pack_validation,
    read_json,
    tree_hash,
    tree_records,
    write_json,
    validate_k1_and_g7a,
)


def _node(candidate_uuid: str, state: str) -> dict[str, object]:
    return {
        "candidate_uuid": candidate_uuid,
        "example_uuid": f"example-{candidate_uuid}",
        "source_group_id": "group-1",
        "source_frame_sha256": "frame-sha",
        "visible_box": {"x1": 10, "y1": 20, "x2": 30, "y2": 60},
        "candidate_state_target": state,
        "gold_person_ids": ["person-1"] if state != "BACKGROUND" else [],
    }


def test_stage_contract_keeps_all_observation_axes_separate() -> None:
    assert STAGE_ID == "M5_5G7B_K1_SUPERVISED_MULTITASK_AND_HIERARCHICAL_OBSERVATION_SELECTION_v1"
    assert BASELINE_COMMIT == "5aa841dd8107ebc5a2f2bb50831d3ed2c326bed9"
    assert "GOALKEEPER" in ROLE_CLASSES
    assert TEAM_CLASSES[:2] == ("TEAM_1", "TEAM_2")
    assert "WARMUP_OR_BIB" in KIT_CLASSES
    assert "OFF_PITCH_SUBSTITUTE_OR_WARMING" in PARTICIPATION_CLASSES
    assert "BACKGROUND" in CANDIDATE_CLASSES
    assert EXPECTED_K1_DISTRIBUTIONS["team_affiliation"]["UNKNOWN_TEAM"] == 41
    assert EXPECTED_K1_DISTRIBUTIONS["kit_state"]["WARMUP_OR_BIB"] == 33


def test_actual_frozen_k1_transaction_state_tree_distributions_and_g7a_artifacts_validate() -> None:
    repository = Path(__file__).resolve().parents[1]
    part4 = repository.parent / "matches" / "128058" / "runs" / "step_m5" / "part 4"
    prompt = part4 / "M5_5G7B_K1_Hierarchical_Reasoner_Codex_Prompt_Pack"
    g7a = part4 / "M5_5G7A_FOOTBALL_OBSERVATION_REASONER_V0_ARCHITECTURE_DATASET_AND_BASELINES_v1"
    k1 = part4 / "M5_5G7A_K1_TEAM_ROLE_KIT_PERSON_GOLD_COMPLETION_VALIDATION_v1"
    if not (prompt.is_dir() and g7a.is_dir() and k1.is_dir()):
        pytest.skip("external immutable G7A/K1 package is not available in this checkout")
    validation = validate_k1_and_g7a(StageLocations(repository, prompt, g7a, k1, part4 / "unused-test-output"))
    assert validation["passed"] is True
    assert validation["k1"]["accepted_decision_count"] == 128
    assert validation["k1"]["decision_state_hash"] == (
        "92f90bbdb7fa194f7d10fedeb6e53d0ee34c7064bcdd8b87cd55bd166f55c274"
    )
    assert validation["k1"]["payload_tree_hash"] == ("27f04937b155386a7214a2f9600fef238a5ea05fe54444f5f62ca4db79615a58")
    manifest = read_json(k1 / "K1_COMPLETION_BUNDLE_MANIFEST.json")
    assert manifest["completion_transaction_id"] == "k1_completion_34cd4442fa494684bdb259e23b348c8a"
    assert validation["g7a"]["node_count"] == 2812
    assert validation["g7a"]["edge_count"] == 24566


def test_k1_join_never_invents_candidate_state() -> None:
    cases = [
        {
            "case_id": "case-1",
            "source_group_id": "group-1",
            "source_frame_sha256": "frame-sha",
            "target_binding_sha256": "binding-sha",
            "target_crop_sha256": "crop-sha",
            "target": {"bbox_original_pixels": {"x1": 10, "y1": 20, "x2": 30, "y2": 60}},
        }
    ]
    decisions = [
        {
            "case_id": "case-1",
            "annotation": {
                "role": "OUTFIELD_PLAYER",
                "team_affiliation": "UNKNOWN_TEAM",
                "kit_state": "WARMUP_OR_BIB",
                "pitch_state": "OFF_PITCH",
                "participation_state": "OFF_PITCH_SUBSTITUTE_OR_WARMING",
                "certainty": "CERTAIN",
            },
        }
    ]
    ledger, people, summary = build_k1_join(
        decisions,
        cases,
        [_node("clean", "CLEAN_INDEPENDENT_PERSON"), _node("background", "BACKGROUND")],
        {"group-1": 2},
    )
    assert len(ledger) == len(people) == 1
    assert people[0]["candidate_state_target"] is None
    assert people[0]["candidate_state_target_available"] is False
    assert summary["candidate_state_values_created_from_k1"] == 0
    assert summary["merged_or_background_propagation_rows"] == 0
    assert {row["candidate_state_source"] for row in summary["propagation_rows"]} == {"PRIOR_G7A_ONLY"}
    assert all(row["team"] == "UNKNOWN_TEAM" for row in summary["propagation_rows"])


def test_node_features_exclude_polygon_membership_and_pitch_distance() -> None:
    row = {
        "source_coordinates": {"image_width": 100, "image_height": 80},
        "visible_box": {"x1": 10, "y1": 12, "x2": 30, "y2": 62},
        "score": 0.8,
        "pitch_polygon_distance_features": {
            "pitch_relation": "ON_PITCH",
            "signed_distance_pixels": 1234.0,
            "footpoint_uncertainty_pixels": 0.0,
        },
    }
    changed = dict(row)
    changed["pitch_polygon_distance_features"] = {
        "pitch_relation": "OFF_PITCH",
        "signed_distance_pixels": -9999.0,
        "footpoint_uncertainty_pixels": 888.0,
    }
    left = node_tabular_features(row)
    right = node_tabular_features(changed)
    assert left.shape == (32,)
    assert np.array_equal(left, right)


def test_primary_population_logic_is_geometry_and_scope_conservative() -> None:
    base = {
        "role": "OUTFIELD_PLAYER",
        "pitch_state": "ON_PITCH",
        "participation_state": "ACTIVE_ON_PITCH",
        "kit_state": "MATCH_OUTFIELD_KIT",
    }
    assert derive_primary_truth(base) == "ACTIVE_OBSERVATION"
    assert derive_primary_truth({**base, "pitch_state": "OFF_PITCH"}) == "BOUNDARY_OR_PARTICIPATION_UNRESOLVED"
    assert (
        derive_primary_truth(
            {
                **base,
                "pitch_state": "OFF_PITCH",
                "participation_state": "OFF_PITCH_SUBSTITUTE_OR_WARMING",
                "kit_state": "WARMUP_OR_BIB",
            }
        )
        == "OUT_OF_SCOPE_PERSON"
    )
    assert derive_primary_truth({**base, "role": "STAFF_OR_SPECTATOR"}) == "OUT_OF_SCOPE_PERSON"


def test_review_pack_enforces_flat_compact_limits(tmp_path: Path) -> None:
    (tmp_path / "04_SOURCE_DIFF.patch").write_text("diff --git a/a b/a\n", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")
    records = tree_records(tmp_path)
    write_json(
        tmp_path / "REVIEW_PACK_MANIFEST.json",
        {
            "payload_file_count": len(records),
            "payload_total_bytes": sum(row["bytes"] for row in records),
            "payload_tree_hash": tree_hash(records),
            "files": records,
        },
    )
    result = review_pack_validation(tmp_path)
    assert result["passed"] is True
    (tmp_path / "summary.json").write_text('{"tampered": true}\n', encoding="utf-8")
    assert review_pack_validation(tmp_path)["passed"] is False
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "forbidden.pt").write_bytes(b"weights")
    assert review_pack_validation(tmp_path)["passed"] is False


def test_stage_builder_does_not_feed_evaluator_gold_to_runtime_selector() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "build_m5_5g7b_k1_hierarchical_reasoner.py").read_text(
        encoding="utf-8"
    )
    selector_source = source[source.index("def _hierarchical_selection(") : source.index("def _node_ablation(")]
    assert '"distinct_person_hypothesis_count": 1' in selector_source
    assert '"clean_control": False' in selector_source
    assert "gold_person_ids" not in selector_source
    assert "candidate_state_target" not in selector_source


def test_visual_refresh_mode_reuses_sealed_ledgers_without_model_fitting() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "build_m5_5g7b_k1_hierarchical_reasoner.py").read_text(
        encoding="utf-8"
    )
    refresh_source = source[source.index("def refresh_visuals_only(") : source.index("def finalize_only(")]
    assert "k1_oof_prediction_ledger.jsonl" in refresh_source
    assert "pairwise_model_results.json" in refresh_source
    assert "solver_decision_ledger.jsonl" in refresh_source
    assert "_fit_nested_calibration" not in refresh_source
    assert "_pairwise_evaluation" not in refresh_source
    assert "_run_ablations" not in refresh_source
    assert '"model_fitting_performed": False' in refresh_source
