from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from football_intelligence.detection_forensics import EXPECTED_CHECKPOINT_SHA256, sha256_file

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_m5_5g6e_c0_reintegration.py"
PART3 = REPO.parent / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G6E_C0_PROPOSAL_REINTEGRATION_AND_PLAYER_OBSERVATION_V1_FULL_UNIVERSE_VALIDATION_v1"


def load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("m5_5g6e_test_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return load_builder()


def read_json(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"run the bounded G6E builder first: {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    assert path.is_file(), f"run the bounded G6E builder first: {path}"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_exact_c0_checkpoint_matrix_and_fusion(builder: ModuleType) -> None:
    validation = read_json(STAGE / "01_G6D_AND_PRIOR_ARTIFACT_VALIDATION" / "g6d_and_prior_artifact_validation.json")
    assert sha256_file(builder.CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256
    assert validation["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert validation["matrix_sha256"] == builder.MATRIX_SHA256
    assert validation["g3_consolidation_variant"] == "IOU_CONNECTED_COMPONENT_055"
    assert validation["passed"] is True


def test_exact_c0_tile_geometry_and_transforms(builder: ModuleType) -> None:
    matrix = read_json(STAGE / "03_FULL_UNIVERSE_C0_REPLAY" / "missing_c2_frozen_matrix.json")
    assert matrix["runtime"] == {
        "agnostic_nms": False,
        "augment": False,
        "classes": [0],
        "conf": 0.22,
        "imgsz": 1280,
        "iou": 0.7,
        "max_det": 80,
    }
    families = {row["physical_source_id"] for row in matrix["physical_execution_plan"]}
    assert families == builder.C0_FAMILIES
    tiles = [
        row
        for row in matrix["physical_execution_plan"]
        if row["physical_source_id"] == "S3_OVERLAPPING_HIGH_RESOLUTION_TILES"
    ]
    assert len(tiles) == 12
    assert all(row["imgsz"] == 1536 for row in tiles)
    assert all(row["crop_bounds_panorama_pixels"]["y2"] == 720.0 for row in tiles)


def test_g6c_g6d_row_level_reconciliation_is_explicit() -> None:
    summary = read_json(STAGE / "02_RAW_STAGE_PROVENANCE_RECONCILIATION" / "raw_stage_reconciliation_summary.json")
    rows = read_jsonl(STAGE / "02_RAW_STAGE_PROVENANCE_RECONCILIATION" / "raw_stage_reconciliation_ledger.jsonl")
    assert len(rows) == summary["target_count"] == 9
    assert summary["historical_origin_counts"] == {"NO_RAW_PROPOSAL": 7, "RAW_LOCALIZATION_BAD": 2}
    assert summary["exact_s0_raw_support_count"] == 9
    assert summary["exact_s0_confidence_independent_support_count"] == 9
    assert summary["exact_s0_post_nms_independent_support_count"] == 0
    assert summary["exact_s3_post_nms_independent_support_count"] == 9
    assert summary["exact_c0_fused_independent_support_count"] == 9
    equivalence = summary["shared_source_semantic_equivalence"]
    assert equivalence["passed"] is True
    assert all(stage["semantic_rows_exact"] for stage in equivalence["stages"].values())
    assert all(row["questions"]["raw_candidate_existed_under_same_runtime"] for row in rows)
    errors = read_json(STAGE / "08_VISUAL_QA_AND_ERROR_LEDGER" / "reintegration_error_ledger.json")
    nms = next(row for row in errors["entries"] if row["classification"] == "LOST_AT_NMS")
    assert nms["full_c0_c2_count"] == 0
    assert nms["s0_only_nine_target_count"] == 9
    assert nms["s3_recovered_all_nine_before_fusion"] is True


def test_historical_inputs_are_unchanged() -> None:
    before = read_json(STAGE / "01_G6D_AND_PRIOR_ARTIFACT_VALIDATION" / "protected_inputs_before.json")
    after = read_json(STAGE / "11_COMMANDS_AND_TESTS" / "protected_inputs_after.json")
    assert after["matches_before"] is True
    assert before["tree_sha256"] == after["tree_sha256"]
    assert all(Path(row["path"]).is_file() for row in before["files"])


def test_full_universes_are_frozen_before_scoring() -> None:
    contract = read_json(STAGE / "03_FULL_UNIVERSE_C0_REPLAY" / "full_universe_contract.json")
    replay = read_json(STAGE / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_full_universe_replay_manifest.json")
    assert contract["frozen_before_scoring"] is True
    assert contract["universes"]["C2"]["people"] == 96
    assert contract["universes"]["B1"]["people"] == 18
    assert contract["universes"]["STATIC"]["people"] == 300
    assert contract["universes"]["DENSE"]["people"] == 73
    assert contract["dense_masks"] == {"scoreable": 71, "unreliable": 2}
    assert replay["full_universe_hash"] == contract["full_universe_hash"]
    assert replay["universe_hashes_frozen_before_scoring"] == {
        name: row["universe_hash"] for name, row in contract["universes"].items()
    }


def test_no_human_runtime_leakage_and_exact_stage_capture() -> None:
    replay = read_json(STAGE / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_full_universe_replay_manifest.json")
    assert replay["human_labels_joined_after_proposal_generation"] is True
    assert replay["runtime_gold_features_used"] is False
    assert replay["nms_exact"] is True
    assert replay["coordinate_roundtrip_exact"] is True
    assert set(replay["aggregate_stage_supply"]["C2"]) == {
        "RAW",
        "CONFIDENCE_SURVIVING",
        "POST_NMS",
        "FUSED",
        "OBSERVATION",
    }


def test_static_and_dense_regression_are_fully_reported() -> None:
    static = read_json(STAGE / "04_STATIC_AND_DENSE_REGRESSION" / "c0_static_results.json")
    dense = read_json(STAGE / "04_STATIC_AND_DENSE_REGRESSION" / "c0_dense_results.json")
    assert static["proposal_regression_count"] == 0
    assert static["clean_control_regression_count"] == 0
    assert static["source_group_regressions"] == []
    assert static["small_person_supply"]["baseline"] == static["small_person_supply"]["c0"]
    assert static["partial_or_occluded_supply"]["baseline"] == static["partial_or_occluded_supply"]["c0"]
    assert dense["people"] == 73
    assert dense["scoreable_masks"] == 71
    assert dense["unreliable_masks"] == 2
    assert dense["promptable_inference_rerun"] is False
    assert dense["new_dense_triggers_created"] is False
    assert dense["merged_as_clean"] == 0


def test_player_observation_schema_is_observed_only_and_provenance_complete() -> None:
    manifest = read_json(
        STAGE / "05_PLAYER_OBSERVATION_V1_REINTEGRATION" / "player_observation_v1_reintegration_manifest.json"
    )
    rows = read_jsonl(STAGE / "05_PLAYER_OBSERVATION_V1_REINTEGRATION" / "player_observation_v1_runtime_rows.jsonl")
    assert manifest["observation_schema"] == "football_intelligence.player_observation.v1"
    assert manifest["observed_only"] is True
    assert manifest["predicted_or_temporal_states_forbidden"] is True
    assert manifest["provenance_complete"] is True
    serialized = json.dumps(rows).lower()
    assert '"observation_state": "predicted' not in serialized
    assert '"track_id"' not in serialized
    assert all(row["evaluator_join_after_runtime"]["human_truth_entered_runtime"] is False for row in rows)


def test_p0_p4_are_unchanged_and_c2_b1_joins_are_independent() -> None:
    gate = read_json(STAGE / "06_PITCH_GATE_DIAGNOSTIC_REPLAY" / "pitch_gate_diagnostic_replay.json")
    assert set(gate["variants"]) == {"P0", "P1", "P2", "P3", "P4"}
    assert gate["pitch_gate_settings_changed"] is False
    assert gate["c2_and_b1_evaluator_joins_independent"] is True
    assert gate["unchanged_gate_variants_passing_frozen_c2_b1_screen"] == []
    assert gate["pitch_gate_unresolved"] is True
    assert all(row["observed_state_contamination"] == 0 for row in gate["variants"].values())
    assert all(row["provenance_failures"] == 0 for row in gate["variants"].values())


def test_runtime_vram_and_repeatability() -> None:
    runtime = read_json(STAGE / "07_RUNTIME_VRAM_AND_OPERATIONAL_BURDEN" / "runtime_and_vram.json")
    assert runtime["source_group_count"] == 49
    assert runtime["view_count"] == 245
    assert runtime["tile_view_count"] == 196
    assert runtime["cuda_only"] is True
    assert runtime["fp16_every_view"] is True
    assert runtime["silent_cpu_fallback"] is False
    assert runtime["deterministic_repeat"] is True
    assert runtime["peak_vram_within_6_5_gib"] is True
    assert runtime["nms_replay_exact"] is True
    assert runtime["coordinate_roundtrip_exact"] is True
    assert all(row["operational_claim"] is False for row in runtime["bounded_non_operational_extrapolations"].values())


def test_frozen_settings_safety_and_negative_scientific_decision() -> None:
    result = read_json(
        STAGE / "05_PLAYER_OBSERVATION_V1_REINTEGRATION" / "player_observation_v1_reintegration_results.json"
    )
    final = read_json(STAGE / "10_NEXT_STAGE_DECISION" / "final_decision.json")
    assert result["proposal_screen_passed"] is False
    assert result["observation_supply_screen_passed"] is False
    assert final["choice"] == "REJECT_C0_DUE_FULL_UNIVERSE_REGRESSION"
    assert final["detector_settings_changed"] is False
    assert final["pitch_gate_settings_changed"] is False
    assert final["light_hq_sam_behavior_changed"] is False
    assert final["no_component_promoted"] is True


def test_review_pack_is_flat_private_bounded_and_hash_valid(builder: ModuleType) -> None:
    root = STAGE / "12_REVIEW_PACK_FOR_CHATGPT"
    validation = builder.validate_review_pack(root)
    assert validation["passed"] is True
    assert validation["file_count"] <= 20
    assert validation["total_bytes"] <= 50 * 1024 * 1024
    assert validation["visual_count"] == 3
    assert validation["leaks"] == []
    assert (root / "04_SOURCE_DIFF.patch").is_file()
