from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from football_intelligence.detection_forensics import EXPECTED_CHECKPOINT_SHA256, sha256_file
from football_intelligence.detection_gold.proposal_supply import deterministic_one_to_one_supply

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_m5_5g6d_high_resolution_proposal_bakeoff.py"
PART3 = REPO.parent / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G6D_R_A1_HIGH_RESOLUTION_SMALL_PERSON_PROPOSAL_BAKEOFF_v1"


def load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("m5_5g6d_test_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return load_builder()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def require_generated(path: Path) -> Path:
    assert path.is_file(), f"run the bounded G6D builder first: {path}"
    return path


def test_exact_target_control_universes(builder: ModuleType) -> None:
    _, validation = builder.validate_g6c_contract()
    assert validation["target_count"] == 9
    assert validation["target_universe_hash"] == builder.TARGET_HASH
    assert validation["control_count"] == 18
    assert validation["control_universe_hash"] == builder.CONTROL_HASH
    assert validation["passed"] is True


def test_exact_checkpoint_and_runtime() -> None:
    checkpoint = REPO / "models" / "model=yolov8m-imgsz=2048.pt"
    assert checkpoint.is_file()
    assert sha256_file(checkpoint) == EXPECTED_CHECKPOINT_SHA256
    validation = read_json(
        require_generated(
            STAGE / "02_CHECKPOINT_RUNTIME_AND_VIEW_MATRIX_FREEZE" / "checkpoint_and_runtime_validation.json"
        )
    )
    assert validation["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert validation["device"] == "cuda:0"
    assert validation["batch"] == 1
    assert validation["passed"] is True


def test_matrix_frozen_before_inference_and_no_gold_crop() -> None:
    matrix_path = require_generated(STAGE / "02_CHECKPOINT_RUNTIME_AND_VIEW_MATRIX_FREEZE" / "frozen_view_matrix.json")
    matrix = read_json(matrix_path)
    recorded_hash = (
        (STAGE / "02_CHECKPOINT_RUNTIME_AND_VIEW_MATRIX_FREEZE" / "frozen_view_matrix.sha256")
        .read_text(encoding="ascii")
        .strip()
    )
    assert sha256_file(matrix_path) == recorded_hash
    assert matrix["matrix_frozen_before_inference"] is True
    assert matrix["evaluator_results_loaded_at_freeze"] is False
    assert matrix["human_geometry_used_to_construct_runtime_crops"] is False
    assert matrix["physical_execution_count"] == 81
    assert {row["crop_origin"] for row in matrix["physical_execution_plan"]} == {
        "full_panorama",
        "frozen_g2b_tile_grid",
        "immutable_pre_existing_case_focal_bounds",
    }


def test_matrix_runtime_and_fusion_constants_are_unchanged(builder: ModuleType) -> None:
    matrix = read_json(
        require_generated(STAGE / "02_CHECKPOINT_RUNTIME_AND_VIEW_MATRIX_FREEZE" / "frozen_view_matrix.json")
    )
    runtime = matrix["canonical_runtime"]
    assert runtime == {
        "agnostic_nms": False,
        "augment": False,
        "batch": 1,
        "classes": [0],
        "conf": 0.22,
        "device": "cuda:0",
        "iou": 0.7,
        "max_det": 80,
    }
    assert {row["imgsz"] for row in matrix["physical_execution_plan"]} == {1280, 1536, 2048}
    assert matrix["fusion_variant"] == builder.FUSION_VARIANT == "IOU_CONNECTED_COMPONENT_055"


def test_source_coordinate_round_trips_are_exact() -> None:
    manifest = read_json(require_generated(STAGE / "03_CUDA_PROPOSAL_REPLAY" / "cuda_inference_manifest.json"))
    assert manifest["checks"]["coordinate_roundtrip_exact"] is True
    assert manifest["primary"]["coordinate_roundtrip_every_view"] is True
    assert manifest["repeat"]["coordinate_roundtrip_every_view"] is True


def test_all_required_pipeline_stages_are_captured() -> None:
    target = read_json(require_generated(STAGE / "04_STAGE_AND_VIEW_TARGET_SUPPORT" / "target_stage_view_support.json"))
    observed = {row["pipeline_stage"] for person in target["people"] for row in person["rows"]}
    assert observed == {"RAW", "CONFIDENCE_SURVIVING", "POST_NMS", "FUSED"}
    assert all(len(person["rows"]) == 7 * 4 for person in target["people"])


def test_tiny_person_matching_is_one_to_one_and_not_iou_only() -> None:
    gold = [
        {"gold_person_id": "A", "bbox": {"x1": 100.0, "y1": 100.0, "x2": 106.0, "y2": 109.0}},
        {"gold_person_id": "B", "bbox": {"x1": 130.0, "y1": 100.0, "x2": 136.0, "y2": 109.0}},
    ]
    proposals = [
        {"proposal_id": "pa", "bbox": {"x1": 101.0, "y1": 100.0, "x2": 105.0, "y2": 109.0}},
        {"proposal_id": "pb", "bbox": {"x1": 131.0, "y1": 100.0, "x2": 135.0, "y2": 109.0}},
    ]
    result = deterministic_one_to_one_supply(gold, proposals, tiny_height_pixels=12.0)
    assert result["one_to_one"] is True
    assert len(result["assignments"]) == 2
    assert result["merged_proposals_assigned_independently"] is False


def test_merged_support_never_counts_as_two_independent_people() -> None:
    gold = [
        {"gold_person_id": "A", "bbox": {"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 40.0}},
        {"gold_person_id": "B", "bbox": {"x1": 20.0, "y1": 10.0, "x2": 30.0, "y2": 40.0}},
    ]
    proposals = [{"proposal_id": "merged", "bbox": {"x1": 9.0, "y1": 9.0, "x2": 31.0, "y2": 41.0}}]
    result = deterministic_one_to_one_supply(gold, proposals)
    assert result["merged_proposal_ids"] == ["merged"]
    assert result["assignments"] == []
    assert {row["supply_state"] for row in result["person_rows"]} == {"MERGED_ONLY_SUPPORT"}


def test_paired_controls_and_frozen_fusion_screens() -> None:
    paired = read_json(
        require_generated(STAGE / "05_FROZEN_FUSION_AND_CONTROL_EVALUATION" / "paired_control_regression.json")
    )
    fusion = read_json(
        require_generated(STAGE / "05_FROZEN_FUSION_AND_CONTROL_EVALUATION" / "frozen_fusion_results.json")
    )
    assert len(paired["rows"]) == 18 * 6
    assert all(summary["denominator"] == 18 for summary in paired["summaries"].values())
    assert set(fusion["configuration_results"]) == {"C0", "C1", "C2", "C3", "C4", "C5"}
    assert all(result["merged_as_clean_observations"] == 0 for result in fusion["configuration_results"].values())


def test_determinism_oom_vram_and_no_cpu_fallback() -> None:
    runtime = read_json(require_generated(STAGE / "06_RUNTIME_VRAM_AND_DETERMINISM" / "runtime_and_vram.json"))
    manifest = read_json(require_generated(STAGE / "03_CUDA_PROPOSAL_REPLAY" / "cuda_inference_manifest.json"))
    assert runtime["deterministic_repeatability"] is True
    assert runtime["silent_cpu_fallback"] is False
    assert runtime["cuda_oom_count"] == 0
    assert manifest["primary"]["view_count"] == manifest["repeat"]["view_count"] == 81
    assert all(row["maximum_peak_allocated_vram_mib"] <= 6.5 * 1024 for row in runtime["combinations"].values())


def test_no_pitch_human_label_or_role_runtime_leakage() -> None:
    fusion = read_json(
        require_generated(STAGE / "05_FROZEN_FUSION_AND_CONTROL_EVALUATION" / "frozen_fusion_results.json")
    )
    matrix = read_json(
        require_generated(STAGE / "02_CHECKPOINT_RUNTIME_AND_VIEW_MATRIX_FREEZE" / "frozen_view_matrix.json")
    )
    assert fusion["human_truth_used_in_runtime_or_fusion"] is False
    assert matrix["human_geometry_used_to_construct_runtime_crops"] is False
    assert all(not result["pitch_or_role_input_used"] for result in fusion["configuration_results"].values())


def test_review_pack_is_flat_bounded_and_hash_valid(builder: ModuleType) -> None:
    root = STAGE / "11_REVIEW_PACK_FOR_CHATGPT"
    validation = builder.validate_review_pack(root)
    manifest = read_json(root / "16_REVIEW_PACK_MANIFEST.json")
    assert validation["passed"] is True
    assert validation["file_count"] <= 20
    assert validation["total_bytes"] <= 50 * 1024 * 1024
    assert validation["visual_count"] == 3
    assert manifest["total_bytes_including_manifest"] == validation["total_bytes"]


def test_no_component_is_promoted() -> None:
    summary = read_json(require_generated(STAGE / "stage_summary.json"))
    assert summary["component_promoted"] is False
    assert summary["detector_promoted"] is False
    assert summary["tracker_promoted"] is False
    assert summary["project_defaults_changed"] is False
