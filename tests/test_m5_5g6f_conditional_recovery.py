from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from football_intelligence.detection_forensics import sha256_file, stable_hash

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_m5_5g6f_conditional_recovery.py"
PART3 = REPO.parent / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G6F_CONDITIONAL_LOW_CONFIDENCE_CROSS_VIEW_RECOVERY_AND_DUPLICATE_CONTROL_v1"


def load_builder() -> ModuleType:
    specification = importlib.util.spec_from_file_location("m5_5g6f_test_builder", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return load_builder()


def read_json(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"run the cached-only G6F builder first: {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    assert path.is_file(), f"run the cached-only G6F builder first: {path}"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_frozen_matrix_is_bounded_hashed_and_truth_free() -> None:
    matrix_path = STAGE / "02_FROZEN_TRIGGER_AND_ADMISSION_MATRIX" / "frozen_trigger_admission_matrix.json"
    matrix = read_json(matrix_path)
    digest = (STAGE / "02_FROZEN_TRIGGER_AND_ADMISSION_MATRIX" / "frozen_trigger_admission_matrix.sha256").read_text(
        encoding="ascii"
    )
    assert matrix["variant_count"] == len(matrix["variants"]) == 12
    assert matrix["variant_count"] <= matrix["maximum_variant_count"]
    assert {row["score_band"] for row in matrix["variants"]} == {"B0", "B1", "B2"}
    assert {row["cross_view_mode"] for row in matrix["variants"]} == {"X1", "X2", "X3"}
    assert {row["admission_mode"] for row in matrix["variants"]} == {"A1", "A2", "A3"}
    assert matrix["human_truth_runtime_forbidden"] is True
    assert matrix["matrix_payload_hash"] == stable_hash({k: v for k, v in matrix.items() if k != "matrix_payload_hash"})
    assert digest == f"{sha256_file(matrix_path)}  {matrix_path.name}\n"


def test_runtime_is_physically_locked_before_evaluator_join() -> None:
    root = STAGE / "04_CONDITIONAL_RECOVERY_BAKEOFF"
    lock = read_json(root / "pre_evaluator_runtime_lock.json")
    receipt = read_json(root / "evaluator_join_receipt.json")
    runtime = read_json(root / "runtime_variant_materialization.json")
    assert lock["evaluator_loaded"] is False
    assert lock["human_truth_runtime_use"] is False
    assert lock["runtime_materialization_sha256"] == sha256_file(root / "runtime_variant_materialization.json")
    assert receipt["pre_evaluator_runtime_lock_sha256"] == sha256_file(root / "pre_evaluator_runtime_lock.json")
    assert receipt["joined_after_runtime_materialization"] is True
    assert runtime["materialized_before_evaluator_join"] is True
    assert runtime["runtime_gold_features_used"] is False


def test_cached_rows_and_runtime_provenance_are_exact() -> None:
    validation = read_json(STAGE / "01_CACHED_ROW_AND_UNIVERSE_VALIDATION" / "g6e_and_cached_row_validation.json")
    assert validation["passed"] is True
    assert validation["cached_rows"]["stage_row_counts"] == {
        "RAW": 147000,
        "CONFIDENCE_SURVIVING": 28185,
        "POST_NMS": 7526,
        "FUSED": 2327,
    }
    checks = validation["cached_rows"]["checks"]
    assert checks["runtime_all_cuda"] is True
    assert checks["runtime_all_fp16"] is True
    assert checks["runtime_no_cpu_fallback"] is True
    assert checks["runtime_nms_exact"] is True
    assert checks["runtime_roundtrip_exact"] is True
    assert validation["g6e_inputs"]["raw_reconciliation_root_cause"].startswith(
        "HISTORICAL_C2_REVIEW_PAYLOAD_STAGE_PRIORITY"
    )


def test_machine_trigger_replay_is_deterministic_and_truth_free() -> None:
    runtime_only = read_json(STAGE / "03_MACHINE_ONLY_TRIGGER_REPLAY" / "machine_trigger_results.runtime_only.json")
    scored = read_json(STAGE / "03_MACHINE_ONLY_TRIGGER_REPLAY" / "machine_trigger_results.json")
    assert runtime_only["truth_free"] is True
    assert runtime_only["deterministic"] is True
    assert runtime_only["new_inference_performed"] is False
    counts = {band: row["triggered_source_count"] for band, row in runtime_only["band_results"].items()}
    assert counts == {"B0": 45, "B1": 34, "B2": 7}
    for band in ("B0", "B1", "B2"):
        assert (
            scored["band_results"][band]["runtime_payload_hash"]
            == runtime_only["band_results"][band]["runtime_payload_hash"]
        )
        assert scored["band_results"][band]["evaluator_diagnostics"]["human_truth_runtime_use"] is False


def test_recovery_is_exact_member_only_and_never_replaces_baseline() -> None:
    runtime = read_json(STAGE / "04_CONDITIONAL_RECOVERY_BAKEOFF" / "runtime_variant_materialization.json")
    assert len(runtime["variants"]) == 12
    rows = [
        candidate
        for variant in runtime["variants"]
        for source in variant["source_rows"]
        for candidate in [*source["admitted_recovery"], *source["routed_recovery"]]
    ]
    assert rows
    assert all(row["selected_exact_evidence_id"] in row["parent_evidence_ids"] for row in rows)
    assert all(row["coordinate_averaging_performed"] is False for row in rows)
    assert all(row["baseline_replacement_performed"] is False for row in rows)
    assert all(row["runtime_gold_features_used"] is False for row in rows)


def test_development_screen_is_unweakened_and_negative() -> None:
    shortlist = read_json(STAGE / "09_DEVELOPMENT_SHORTLIST_AND_DECISION" / "development_shortlist.json")
    decision = read_json(STAGE / "09_DEVELOPMENT_SHORTLIST_AND_DECISION" / "final_decision.json")
    assert shortlist["screen_not_weakened"] is True
    assert shortlist["passing_variant_count"] == 0
    assert shortlist["development_candidate_frozen"] is None
    assert shortlist["all_nine_target_maximum"] < 9
    assert all(not row["passes_full_development_screen"] for row in shortlist["shortlist"])
    assert decision["choice"] == "AUTHORIZE_OFFICIAL_SMALL_PERSON_DETECTOR_FAMILY_BAKEOFF"
    assert decision["component_promoted"] is False


def test_c2_transition_ledger_preserves_baseline_and_exposes_reachability() -> None:
    rows = read_jsonl(STAGE / "05_C2_TRANSITION_AND_OBSERVATION_DIAGNOSIS" / "c2_transition_ledger.jsonl")
    diagnosis = read_json(
        STAGE / "05_C2_TRANSITION_AND_OBSERVATION_DIAGNOSIS" / "observation_materialization_diagnosis.json"
    )
    assert len(rows) == 45
    assert sum(row["suppressed"] for row in rows) == 0
    assert sum(row["anonymous_g6d_target_id"] is not None for row in rows) == 9
    assert diagnosis["g6d_targets_without_s0_raw_anchor"] == 7
    assert diagnosis["g6d_target_origin_counts"] == {"NO_RAW_PROPOSAL": 7, "RAW_LOCALIZATION_BAD": 2}
    assert diagnosis["coordinate_averaging_performed"] is False
    assert diagnosis["baseline_replacement_performed"] is False


def test_static_dense_and_b1_are_reported_without_new_inference() -> None:
    regression = read_json(STAGE / "06_STATIC_DENSE_AND_B1_REGRESSION" / "static_dense_regression.json")
    assert regression["variant_count"] == len(regression["variants"]) == 12
    assert regression["frozen_dense_branch_unchanged"] is True
    assert regression["new_promptable_or_detector_inference"] is False
    assert regression["unscored_crowd_remains_unscored"] is True
    for row in regression["variants"]:
        assert row["b1"]["pitch_gate_tuned"] is False
        assert row["dense"]["frozen_dense_mask_branch_unchanged"] is True
        assert row["dense"]["promptable_inference_rerun"] is False


def test_off_pitch_and_crowd_burden_is_separate_and_conservative() -> None:
    burden = read_json(STAGE / "07_OFF_PITCH_CROWD_AND_RUNTIME_BURDEN" / "off_pitch_and_crowd_burden.json")
    assert burden["clear_off_pitch_people"] == 51
    assert burden["unmatched_indistinct_crowd_policy"] == "UNSCORED_CROWD"
    assert burden["unmatched_rows_scored_as_background_false_positive"] is False
    assert burden["off_pitch_output_counts_as_on_pitch_supply"] is False
    assert burden["human_pitch_state_runtime_use"] is False


def test_runtime_burden_is_historical_estimate_only() -> None:
    burden = read_json(STAGE / "07_OFF_PITCH_CROWD_AND_RUNTIME_BURDEN" / "runtime_burden_estimate.json")
    assert burden["basis"] == "VERIFIED_HISTORICAL_G6E_CUDA_TIMINGS"
    assert burden["estimates_not_new_runtime_measurements"] is True
    assert burden["new_gpu_inference_performed"] is False
    assert burden["global_c0"]["source_count"] == 49
    assert burden["global_c0"]["s3_tile_view_count"] == 196
    assert burden["score_bands"]["B2"]["reduction_at_least_40_percent"] is True


def test_protected_history_is_byte_exact() -> None:
    before = read_json(STAGE / "01_CACHED_ROW_AND_UNIVERSE_VALIDATION" / "protected_inputs_before.json")
    after = read_json(STAGE / "10_COMMANDS_AND_TESTS" / "protected_inputs_after.json")

    def compact(payload: dict[str, Any]) -> list[tuple[str, int, str]]:
        return sorted((str(row["path"]), int(row["bytes"]), str(row["sha256"])) for row in payload["files"])

    assert compact(before) == compact(after)
    assert before["tree_sha256"] == after["tree_sha256"]


def test_stage_safety_and_no_promotion() -> None:
    summary = read_json(STAGE / "stage_summary.json")
    assert summary["classification"] == "PASS_CONDITIONAL_CROSS_VIEW_RECOVERY_BAKEOFF_READY_FOR_PRO_REVIEW"
    assert summary["new_inference_performed"] is False
    assert summary["global_confidence_changed"] is False
    assert summary["nms_changed"] is False
    assert summary["tile_geometry_changed"] is False
    assert summary["fusion_defaults_changed"] is False
    assert summary["pitch_gate_settings_changed"] is False
    assert summary["component_promoted"] is False
    assert summary["production_ready"] is False


def test_review_pack_is_flat_bounded_visual_and_hash_valid(builder: ModuleType) -> None:
    validation = builder.validate_review_pack()
    assert validation["passed"] is True
    assert validation["file_count"] <= 20
    assert validation["visual_file_count"] == 3
    assert validation["total_bytes"] <= 50 * 1024 * 1024
    assert (STAGE / "11_REVIEW_PACK_FOR_CHATGPT" / "04_SOURCE_DIFF.patch").is_file()
