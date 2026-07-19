from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.sports_mot.frozen_holdout import (
    HoldoutGovernanceError,
    ImmutablePrimaryResultTransaction,
    OneTimeSemanticAccessController,
    compare_determinism_runs,
    evaluate_machine_gates,
    retry_policy,
    validate_preregistration,
)


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "SoccerTrack-v2"
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
F1C = PART2 / "M5_5F1C_DEVELOPMENT_FAILURE_ATLAS_PANORAMA_HANDOFF_AND_TRUE_HIERARCHICAL_PATH_SELECTION_v1"
STAGE = PART2 / "M5_5F1D_FROZEN_P_MHSAG_PREREGISTRATION_ONE_TIME_SEALED_HOLDOUT_AND_ROBUSTNESS_AUDIT_v1"
CONFIGURATION_HASH = "60854fda0a73e6df74d9fcbb157c211e2850d3860f657600cb01212d888b88a7"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def candidate_and_preregistration() -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = {
        "candidate_source_commit": "cf4d0222e2e8aabf1c462286fc71788e0acd9fc6",
        "configuration_hash": CONFIGURATION_HASH,
    }
    candidate_hash = stable_hash(candidate)
    prereg = {
        "frozen_candidate_manifest_hash": candidate_hash,
        "exact_execution_command": "execute",
        "oracle_mode_command": "oracle",
        "detector_mode_command": "detector",
        "legacy_focal_supplementary_command": "focal",
        "holdout_sequence_count": 8,
        "expected_output_schemas": {},
        "machine_hard_gates": {},
        "failure_attribution_rules": [],
        "one_time_access_policy": {},
        "same_config_retry_policy": {},
        "pre_registered_shadow_stress_matrix": [],
        "conditional_visual_audit_policy": {},
        "no_retune_statement": True,
        "configuration_hash": CONFIGURATION_HASH,
        "execution_harness_source_hashes": [],
    }
    return candidate, prereg


def passing_metrics() -> dict[str, Any]:
    return {
        "sequence_count": 8,
        "fully_exact_sequences": 8,
        "identity_switches": 0,
        "false_continuations": 0,
        "strand_losses_when_supply_available": 0,
        "off_pitch_assignments": 0,
        "double_assignments": 0,
        "provenance_failures": 0,
    }


def test_pre_registration_requires_complete_frozen_binding() -> None:
    candidate, prereg = candidate_and_preregistration()
    result = validate_preregistration(
        prereg,
        expected_candidate_manifest_hash=stable_hash(candidate),
        expected_configuration_hash=CONFIGURATION_HASH,
    )
    assert result["passed"]
    assert result["pre_registration_hash"] == stable_hash(prereg)


def test_holdout_is_inaccessible_before_unseal(tmp_path: Path) -> None:
    controller = OneTimeSemanticAccessController(tmp_path / "event.json", tmp_path / "state.json")
    called = False

    def resolver() -> None:
        nonlocal called
        called = True

    result = controller.reject_pre_unseal_access(resolver)
    assert result["passed"]
    assert not called
    assert controller.unseal_count == 0


def test_atomic_one_time_unseal_and_second_request_rejection(tmp_path: Path) -> None:
    controller = OneTimeSemanticAccessController(tmp_path / "event.json", tmp_path / "state.json")
    candidate, prereg = candidate_and_preregistration()
    resolved = controller.unseal(
        preregistration=prereg,
        preregistration_hash=stable_hash(prereg),
        candidate_manifest=candidate,
        candidate_manifest_hash=stable_hash(candidate),
        sealed_manifest_hash="a" * 64,
        sealed_container_hash="b" * 64,
        actor={"process_id": 1, "session": "test"},
        resolver=lambda: {"authorized": True},
    )
    assert resolved == {"authorized": True}
    assert controller.unseal_count == 1
    assert read_json(tmp_path / "event.json")["unseal_count_after"] == 1
    with pytest.raises(HoldoutGovernanceError, match="already been unsealed"):
        controller.unseal(
            preregistration=prereg,
            preregistration_hash=stable_hash(prereg),
            candidate_manifest=candidate,
            candidate_manifest_hash=stable_hash(candidate),
            sealed_manifest_hash="a" * 64,
            sealed_container_hash="b" * 64,
            actor={},
            resolver=lambda: None,
        )


def test_invalid_pre_registration_never_writes_unseal_event(tmp_path: Path) -> None:
    controller = OneTimeSemanticAccessController(tmp_path / "event.json", tmp_path / "state.json")
    candidate, prereg = candidate_and_preregistration()
    prereg["no_retune_statement"] = False
    with pytest.raises(HoldoutGovernanceError, match="invalid pre-registration"):
        controller.unseal(
            preregistration=prereg,
            preregistration_hash=stable_hash(prereg),
            candidate_manifest=candidate,
            candidate_manifest_hash=stable_hash(candidate),
            sealed_manifest_hash="a" * 64,
            sealed_container_hash="b" * 64,
            actor={},
            resolver=lambda: None,
        )
    assert not (tmp_path / "event.json").exists()


def test_three_run_canary_requires_discrete_and_float_equivalence() -> None:
    base = {
        "strand_states": {"s": ["A", "B"]},
        "observation_source_choices": {"s": ["n1", "n2"]},
        "error_attribution": {"s": []},
        "fully_exact_sequences": 8,
        "graph_hashes": {"s": "g"},
        "descriptor_cache_hash": "d",
        "configuration_hash": CONFIGURATION_HASH,
        "joint_path_costs": [1.0, 2.0],
    }
    result = compare_determinism_runs([base, {**base}, {**base, "joint_path_costs": [1.0000001, 2.0]}])
    assert result["passed"]
    assert result["maximum_floating_cost_delta"] <= 1e-6


def test_canary_rejects_changed_discrete_path() -> None:
    base = {
        "strand_states": {"s": ["A"]},
        "observation_source_choices": {},
        "error_attribution": {},
        "fully_exact_sequences": 8,
        "graph_hashes": {},
        "descriptor_cache_hash": "d",
        "configuration_hash": CONFIGURATION_HASH,
        "joint_path_costs": [1.0],
    }
    changed = {**base, "strand_states": {"s": ["B"]}}
    assert not compare_determinism_runs([base, changed])["passed"]


def test_primary_result_transaction_is_immutable(tmp_path: Path) -> None:
    artifacts = {
        "oracle_holdout_results.json": {"mode": "oracle"},
        "detector_holdout_results.json": {"mode": "detector"},
        "legacy_focal_holdout_results.json": {"mode": "focal"},
    }
    transaction = ImmutablePrimaryResultTransaction(tmp_path)
    committed = transaction.commit(artifacts, context={"configuration_hash": CONFIGURATION_HASH})
    assert committed["status"] == "IMMUTABLE_FIRST_VALID_PRIMARY_RESULT"
    assert transaction.validate()["passed"]
    with pytest.raises(HoldoutGovernanceError, match="already exists"):
        transaction.commit(artifacts, context={})


def test_primary_result_transaction_detects_mutation(tmp_path: Path) -> None:
    transaction = ImmutablePrimaryResultTransaction(tmp_path)
    transaction.commit(
        {
            "oracle_holdout_results.json": {},
            "detector_holdout_results.json": {},
            "legacy_focal_holdout_results.json": {},
        },
        context={},
    )
    (tmp_path / "detector_holdout_results.json").write_text("{}\n", encoding="utf-8")
    assert not transaction.validate()["passed"]


def test_machine_hard_gate_passes_only_complete_oracle_and_detector() -> None:
    oracle = {"metrics": passing_metrics(), "frame_attribution_rows": []}
    detector = {"metrics": passing_metrics(), "frame_attribution_rows": []}
    result = evaluate_machine_gates(oracle, detector)
    assert result["passed"]


def test_machine_hard_gate_allows_one_safe_detector_abstention_sequence() -> None:
    oracle = {"metrics": passing_metrics(), "frame_attribution_rows": []}
    metrics = {**passing_metrics(), "fully_exact_sequences": 7}
    detector = {
        "metrics": metrics,
        "frame_attribution_rows": [{"outcome": "SAFE_ABSTENTION_NO_SUPPLY"}],
    }
    assert evaluate_machine_gates(oracle, detector)["passed"]


@pytest.mark.parametrize(
    "field",
    [
        "identity_switches",
        "false_continuations",
        "strand_losses_when_supply_available",
        "off_pitch_assignments",
        "double_assignments",
        "provenance_failures",
    ],
)
def test_detector_hard_error_blocks_machine_gate(field: str) -> None:
    oracle = {"metrics": passing_metrics(), "frame_attribution_rows": []}
    detector = {
        "metrics": {**passing_metrics(), field: 1, "fully_exact_sequences": 7},
        "frame_attribution_rows": [{"outcome": "ASSOCIATION_SWITCH"}],
    }
    assert not evaluate_machine_gates(oracle, detector)["passed"]


def test_scientific_underperformance_never_authorizes_retry() -> None:
    result = retry_policy("SCIENTIFIC_UNDERPERFORMANCE", sequence_score_committed=False, valid_result_exists=False)
    assert not result["same_config_retry_allowed"]
    assert not result["scientific_underperformance_retry_allowed"]


def test_runtime_retry_requires_no_committed_score_or_valid_result() -> None:
    assert retry_policy("CUDA_OOM", sequence_score_committed=False, valid_result_exists=False)[
        "same_config_retry_allowed"
    ]
    assert not retry_policy("CUDA_OOM", sequence_score_committed=True, valid_result_exists=False)[
        "same_config_retry_allowed"
    ]
    assert not retry_policy("PROCESS_CRASH", sequence_score_committed=False, valid_result_exists=True)[
        "same_config_retry_allowed"
    ]


def test_completed_f1c_audit_is_three_repair_correct() -> None:
    decisions = read_json(F1C / "11_DEVELOPMENT_ERROR_ATLAS_REVIEW_PACKAGE" / "decisions" / "completed_review.json")
    rows = decisions["state"]["decisions"]
    assert len(rows) == 3
    assert set(rows.values()) == {"REPAIR_CORRECT"}
    assert decisions["decision_state_hash"] == "9499f26f7dbb687a8089d0aef737e2e1cd90ffb86fd429092a81ab41511073a4"


def test_frozen_candidate_source_file_still_matches_f1c_reproducibility_manifest() -> None:
    manifest = read_json(F1C / "13_REPRODUCIBILITY_BUNDLE" / "reproducibility_manifest.json")
    source = next(row for row in manifest["source_files"] if row["path"].endswith("panorama_hierarchical.py"))
    assert (
        sha256_file(REPO / "src" / "football_intelligence" / "sports_mot" / "panorama_hierarchical.py")
        == source["sha256"]
    )


def test_prepared_stage_has_three_clean_process_canaries_when_present() -> None:
    comparison = STAGE / "03_DEVELOPMENT_DETERMINISM_AND_REPRODUCIBILITY_CANARY" / "development_canary_comparison.json"
    if comparison.exists():
        payload = read_json(comparison)
        assert payload["passed"]
        assert payload["run_count"] == 3


def test_visual_audit_is_conditional_and_exactly_eight_when_present() -> None:
    decision = STAGE / "09_CONDITIONAL_VISUAL_AUDIT_CONSTRUCTION" / "visual_audit_creation_decision.json"
    if not decision.exists():
        return
    payload = read_json(decision)
    manifest_path = STAGE / "10_HOLDOUT_VISUAL_AUDIT_PACKAGE" / "reviewer_manifest.json"
    if payload["primary_machine_gate_passed"]:
        assert len(read_json(manifest_path)["cases"]) == 8
    else:
        assert not manifest_path.exists()


def test_safety_prevents_promotion_or_new_stage_work() -> None:
    source = (REPO / "scripts" / "run_m5_5f1d_frozen_holdout.py").read_text(encoding="utf-8")
    assert '"tracker_promoted": False' in source
    assert '"model_fit_performed": False' in source
    assert "learned_continuity_rows_updated" in source
    assert "run_level_3" not in source
    assert "run_occlusion" not in source
