from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G6C_PITCH_GATE_REEVALUATION_AND_PROPOSAL_SUPPLY_RECOVERY_DECISION_v1"
SCRIPT = REPO / "scripts" / "build_m5_5g6c_pitch_gate_recovery_decision.py"


def load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("m5_5g6c_builder_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="module")
def replay() -> dict[str, Any]:
    builder = load_builder()
    completed, manifest, b1_validation = builder.validate_b1_completion()
    universe = builder.evaluator_universe_contract()
    specifications = builder.validate_frozen_specifications(manifest)
    c2, _, _ = builder.replay_c2()
    b1, _, _ = builder.replay_b1(completed, manifest)
    combined = builder.combined_gate_decision(c2, b1)
    phenotypes, phenotype_summary, _ = builder.phenotype_nine_misses()
    experiment, experiment_contract = builder.proposal_recovery_decision(phenotypes, phenotype_summary)
    observation = builder.player_observation_status(combined)
    shortlist = builder.development_shortlist(combined, experiment)
    return {
        "builder": builder,
        "b1_validation": b1_validation,
        "universe": universe,
        "specifications": specifications,
        "c2": c2,
        "b1": b1,
        "combined": combined,
        "phenotypes": phenotypes,
        "phenotype_summary": phenotype_summary,
        "experiment": experiment,
        "experiment_contract": experiment_contract,
        "observation": observation,
        "shortlist": shortlist,
    }


def test_b1_completion_and_source_bindings_replay_exactly(replay: dict[str, Any]) -> None:
    validation = replay["b1_validation"]
    assert validation["passed"] is True
    assert validation["completion_transaction_id"] == (
        "tranche_B1_BOUNDARY_FOCUSED_PERSON_GOLD_59a9cb2aa61a80669f02182feb2dc672"
    )
    assert validation["root_event_sequence"] == validation["root_event_count"] == 19
    assert validation["event_type_counts"] == {
        "DETECTION_CASE_SAVED": 18,
        "DETECTION_TRANCHE_COMPLETED": 1,
    }
    assert validation["case_count"] == validation["distinct_source_group_count"] == 18
    assert validation["exact_counts"] == {
        "roles": {"GOALKEEPER": 2, "OFFICIAL": 8, "PLAYER": 8},
        "pitch_states": {"BOUNDARY_UNCERTAIN": 8, "OFF_PITCH": 2, "ON_PITCH": 8},
        "footpoints": {"OBSERVED_CLEAR": 18},
        "relations": {"CLEAN_SINGLE_INSTANCE": 18},
    }
    assert validation["snapshot_count"] == 19
    assert validation["snapshot_replay_valid"] is True
    assert validation["source_bindings_valid"] is True
    assert validation["prior_gold_unchanged"] is True


def test_c2_and_b1_are_separate_evaluator_universes(replay: dict[str, Any]) -> None:
    universe = replay["universe"]
    assert universe["pooling_into_single_accuracy_forbidden"] is True
    assert universe["single_overall_accuracy_reported"] is False
    assert universe["universes"]["C2_BROAD_CLEAR_PERSON_DEVELOPMENT_SET"] == {
        "cases": 12,
        "people": 96,
        "on_pitch": 45,
        "off_pitch": 51,
        "boundary_uncertain": 0,
        "uses": [
            "on-pitch retention",
            "labelled off-pitch leakage",
            "role retention",
            "feet-not-visible handling",
            "general frozen proposal supply",
        ],
    }
    assert universe["universes"]["B1_TARGETED_BOUNDARY_STRESS_SET"]["cases"] == 18
    assert universe["universes"]["B1_TARGETED_BOUNDARY_STRESS_SET"]["population_representative"] is False


def test_frozen_specs_and_c2_runtime_replay_are_exact(replay: dict[str, Any]) -> None:
    specifications = replay["specifications"]
    assert specifications["passed"] is True
    assert specifications["footpoint_specification_sha256"] == (
        "d82e1a8315dd285f0c32ebc9966d0a8d46e791034a006bf9a41d6a4e4c7d55c2"
    )
    assert specifications["pitch_gate_specification_sha256"] == (
        "d31938676d2718b3c222e83b79c7c1435af80d62b6b25ac28335ff76427aab8c"
    )
    assert all(specifications["checks"].values())
    assert specifications["threshold_or_margin_changed"] is False

    c2 = replay["c2"]
    assert c2["passed"] is True
    assert c2["runtime_ledger_exact"] is True
    assert c2["variant_outputs_exact"] is True
    assert c2["frozen_runtime_row_count"] == c2["replayed_runtime_row_count"] == 235
    assert c2["human_truth_entered_runtime"] is False
    assert [row["pitch_gate_variant"] for row in c2["variants"]] == ["P1", "P2", "P3", "P4"]


def test_b1_gate_results_and_combined_shortlist_are_exact(replay: dict[str, Any]) -> None:
    variants = {row["pitch_gate_variant"]: row for row in replay["b1"]["variants"]}
    assert variants["P1"]["on_pitch"] == {"denominator": 8, "retained": 8, "hard_off": 0, "routed": 0}
    assert variants["P1"]["off_pitch"] == {"denominator": 2, "rejected": 0, "leaked": 2, "routed": 0}
    assert variants["P1"]["boundary_uncertain"] == {
        "denominator": 8,
        "routed": 0,
        "hard_on": 1,
        "hard_off": 7,
    }
    for variant in ("P2", "P3", "P4"):
        assert variants[variant]["on_pitch"] == {
            "denominator": 8,
            "retained": 2,
            "hard_off": 0,
            "routed": 6,
        }
        assert variants[variant]["off_pitch"] == {
            "denominator": 2,
            "rejected": 0,
            "leaked": 0,
            "routed": 2,
        }
        assert variants[variant]["boundary_uncertain"] == {
            "denominator": 8,
            "routed": 8,
            "hard_on": 0,
            "hard_off": 0,
        }
    assert replay["b1"]["shortlisted_variants"] == []
    assert replay["b1"]["human_truth_entered_runtime"] is False
    assert replay["combined"]["shortlisted_pitch_gate_variants"] == []
    assert replay["combined"]["decision"] == "NO_FROZEN_PITCH_GATE_VARIANT_PASSES_BOTH_UNIVERSES"
    assert all(row["combined_screen_passed"] is False for row in replay["combined"]["variants"])


def test_nine_miss_phenotypes_select_exactly_one_recovery_experiment(replay: dict[str, Any]) -> None:
    rows = replay["phenotypes"]
    summary = replay["phenotype_summary"]
    assert len(rows) == summary["missing_person_count"] == 9
    assert summary["phenotype_counts"] == {"SMALL_FAR_SIDE": 9}
    assert summary["origin_counts"] == {"NO_RAW_PROPOSAL": 7, "RAW_LOCALIZATION_BAD": 2}
    assert 22.0 <= summary["visible_height_pixels"]["minimum"]
    assert summary["visible_height_pixels"]["maximum"] <= 33.1
    assert summary["provenance_coordinate_repair_indicated"] is False
    assert all(row["generic_iou_only_decision"] is False for row in rows)
    assert all(row["human_truth_entered_runtime"] is False for row in rows)

    experiment = replay["experiment"]
    contract = replay["experiment_contract"]
    assert experiment["exactly_one_experiment_selected"] is True
    assert experiment["selected_experiment"] == "R-A_HIGH_RESOLUTION_SMALL_PERSON_PROPOSAL_BAKEOFF"
    assert experiment["selected_experiment_id"] == "R-A1_FROZEN_G2B_HIGH_RESOLUTION_VIEW_MATRIX"
    assert set(experiment["rejected_options"]) == {
        "R-B_OCCLUSION_PARTIAL_PERSON_PROPOSAL_BAKEOFF",
        "R-C_GENERAL_NEW_DETECTOR_FAMILY_PROPOSAL_BAKEOFF",
        "R-D_ANNOTATION_FIRST_EXPANSION",
        "R-E_PROVENANCE_COORDINATE_REPAIR",
    }
    assert contract["target_count"] == 9
    assert contract["control_count"] == 18
    assert contract["future_permitted_inference"]["authorized_in_g6c"] is False
    assert contract["future_permitted_inference"]["threshold_search"] is False
    assert contract["future_permitted_inference"]["crop_or_tile_change"] is False
    assert contract["future_permitted_inference"]["nms_or_fusion_change"] is False
    assert contract["experiment_executed"] is False


def test_player_observation_schema_is_frozen_but_runtime_candidate_is_blocked(replay: dict[str, Any]) -> None:
    observation = replay["observation"]
    assert observation["schema_and_materializer_status"] == "READY_DEVELOPMENT_ONLY"
    assert observation["pitch_gate_status"] == "BLOCKED_NO_VARIANT_PASSES_COMBINED_SCREEN"
    assert observation["pitch_gate_candidates"] == []
    assert observation["proposal_supply_status"] == "BELOW_FROZEN_OBSERVATION_SCREEN"
    assert observation["fused_proposal_support_ceiling"] == {
        "supported_on_pitch_people": 36,
        "denominator": 45,
    }
    assert observation["observed_state_contamination_count"] == 0
    assert observation["provenance_failure_count"] == 0
    assert observation["player_observation_v1_complete"] is False
    assert observation["new_inference_performed"] is False


def test_protected_inputs_are_unchanged_and_review_pack_is_bounded(replay: dict[str, Any]) -> None:
    builder = replay["builder"]
    before = read_json(STAGE / "00_PROMPT_AND_INPUTS" / "protected_input_manifest_before.json")
    after = read_json(STAGE / "11_COMMANDS_AND_TESTS" / "protected_input_manifest_after.json")
    assert before == after == builder.protected_manifest()

    pack = STAGE / "12_REVIEW_PACK_FOR_CHATGPT"
    manifest = read_json(pack / "18_REVIEW_PACK_MANIFEST.json")
    assert manifest["passed"] is True
    assert manifest["file_count_including_manifest"] <= 20
    assert manifest["total_bytes_excluding_manifest"] <= 50 * 1024 * 1024
    assert manifest["visual_file_count"] == 3
    assert all(manifest["checks"].values())
    assert (pack / "04_SOURCE_DIFF.patch").stat().st_size > 0
    assert all(path.is_file() and path.parent == pack for path in pack.iterdir())


def test_stage_keeps_all_safety_prohibitions(replay: dict[str, Any]) -> None:
    serialized = json.dumps(replay, default=str, sort_keys=True)
    assert "FREEZE_PLAYER_OBSERVATION_SCHEMA_ONLY_AUTHORIZE_PROPOSAL_RECOVERY_EXPERIMENT" in serialized
    for payload_name in ("combined", "observation", "experiment", "experiment_contract"):
        payload = replay[payload_name]
        assert payload["production_ready"] is False
        assert payload["no_auto_promotion"] is True
        assert payload["human_approved"] is False
        assert payload["safe_to_apply_globally"] is False
        assert payload["match_local_only"] is True
        assert payload["sandbox_only"] is True
        assert payload["identity_tracking_performed"] is False
        assert payload["metric_analysis_performed"] is False
        assert payload["auto_promoted"] is False
