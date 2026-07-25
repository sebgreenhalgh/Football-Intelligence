from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from football_intelligence.detection_gold.promptable_masks import (
    decode_packed_mask,
    deduplicate_masks,
    evaluator_pitch_state,
    fixed_context_crop,
    official_source_allowed,
    prompt_payload_forbidden_values,
    rasterize_polygon,
)


REPO = Path(__file__).resolve().parents[1]
PART3 = REPO.parent / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G5A_AUTHORIZED_PROMPTABLE_MASK_BAKEOFF_AND_DENSE_BRANCH_DECISION_v1"
REVIEW_PACK = STAGE / "11_REVIEW_PACK_FOR_CHATGPT"


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_packed_masks_empty_polygons_and_fixed_deduplication() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 3:7] = 1
    packed = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
    decoded = decode_packed_mask(
        {
            "height": 8,
            "width": 8,
            "packed_bits_base64": base64.b64encode(packed).decode("ascii"),
        }
    )
    assert np.array_equal(decoded, mask.astype(bool))
    assert not rasterize_polygon([], {"x1": 0, "y1": 0, "x2": 8, "y2": 8}).any()

    kept, suppressed = deduplicate_masks(
        [
            {"output_mask_id": "high", "official_score": 0.9, "mask": decoded},
            {"output_mask_id": "lower", "official_score": 0.8, "mask": decoded.copy()},
        ]
    )
    assert [row["output_mask_id"] for row in kept] == ["high"]
    assert suppressed[0]["duplicate_of"] == "high"


def test_official_source_allowlist_and_runtime_leakage_audit() -> None:
    assert official_source_allowed("https://github.com/facebookresearch/sam2")
    assert official_source_allowed("https://dl.fbaipublicfiles.com/segment_anything_2/model.pt")
    assert official_source_allowed("https://huggingface.co/lkeab/hq-sam/resolve/revision/model.pth")
    assert not official_source_allowed("https://example.invalid/community-quantized-model.pt")

    safe = [{"prompt_type": "R0", "box": [1, 2, 3, 4]}]
    leaked = [{"prompt_type": "R0", "pitch_state": "OFF_PITCH"}]
    forbidden = {"pitch_state", "OFF_PITCH", "target_annotation_uuid"}
    assert prompt_payload_forbidden_values(safe, forbidden) == []
    assert prompt_payload_forbidden_values(leaked, forbidden)


def test_fixed_crop_and_evaluator_pitch_state_are_deterministic() -> None:
    crop = fixed_context_crop(
        [{"x1": 100, "y1": 100, "x2": 200, "y2": 300}],
        1000,
        500,
        context_fraction=0.25,
    )
    assert crop == {"x1": 75, "y1": 50, "x2": 225, "y2": 350}

    polygon = [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 100, "y": 100}, {"x": 0, "y": 100}]
    assert evaluator_pitch_state({"x1": 40, "y1": 20, "x2": 60, "y2": 80}, polygon)["state"] == "ON_PITCH"
    assert evaluator_pitch_state({"x1": 110, "y1": 20, "x2": 130, "y2": 80}, polygon)["state"] == "OFF_PITCH"


def test_dense_gold_and_protected_inputs_validate_exactly() -> None:
    validation = _read_json(
        STAGE / "01_DENSE_GOLD_V2_AND_RUNTIME_INPUT_VALIDATION" / "dense_gold_v2_input_validation.json"
    )
    protected = _read_json(STAGE / "00_PROMPT_AND_INPUTS" / "protected_input_manifest_before.json")

    assert validation["passed"] is True
    assert all(validation["checks"].values())
    assert validation["dataset_id"] == "C1_DENSE_GOLD_V2_APPLIED_OVERLAY"
    assert validation["dataset_hash"] == "fa14afb2f1e8c4327f8daf2d52030156a79134c836820e70f167599cf400d762"
    assert validation["inventory"]["person_instance_count"] == 73
    assert validation["inventory"]["trusted_scoreable_visible_mask_count"] == 71
    assert validation["inventory"]["unreliable_visible_mask_geometry_count"] == 2
    assert len(validation["unreliable_annotation_uuids"]) == 2
    assert protected["passed"] is True
    assert protected["tree_hash"] == "4f7acf9a1cfe80db720ebd9d286adebd3c68302f85416b1dc70a12c9197a8266"


def test_frozen_prompt_matrix_separates_runtime_and_assistance() -> None:
    specification = _read_json(STAGE / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "frozen_crop_prompt_specification.json")
    leakage = _read_json(STAGE / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "runtime_gold_leakage_audit.json")

    assert specification["frozen_before_inference"] is True
    assert specification["image_crop_count"] == 16
    assert specification["runtime_prompt_count"] == 604
    assert specification["annotation_assistance_prompt_count"] == 142
    assert specification["annotation_assistance_expected_prompt_count"] == 142
    assert len(specification["unreliable_human_masks_excluded_from_h0"]) == 2
    assert {row["prompt_type"] for row in specification["runtime_prompts"]} == {"R0", "R1", "R2", "R3"}
    assert {row["prompt_type"] for row in specification["annotation_assistance_prompts"]} == {"H0"}
    assert leakage["passed"] is True
    assert leakage["forbidden_runtime_hits"] == []


def test_pitch_addendum_is_evaluator_only_and_preserves_off_pitch_people() -> None:
    sidecar = _read_json(
        STAGE / "01_DENSE_GOLD_V2_AND_RUNTIME_INPUT_VALIDATION" / "evaluator_only_pitch_state_sidecar.json"
    )
    comparison = _read_json(STAGE / "06_MASK_OUTPUT_CONSOLIDATION_AND_EVALUATION" / "model_comparison_summary.json")

    assert sidecar["runtime_prompt_crop_or_gate_use"] is False
    assert sidecar["pitch_gate_implemented_or_tuned"] is False
    assert sidecar["counts"] == {"BOUNDARY_UNCERTAIN": 9, "OFF_PITCH": 27, "ON_PITCH": 37}
    assert sidecar["off_pitch_only_cases"] == ["m5_5g1a_case_040"]
    assert all(row["coarse_role"] for row in sidecar["rows"])
    assert all(row["primary_on_pitch_denominator"] == 37 for row in comparison["runtime_branches"])
    assert all(row["off_pitch_denominator_descriptive_only"] == 27 for row in comparison["runtime_branches"])
    assert all(row["primary_on_pitch_scoreable_count"] == 37 for row in comparison["annotation_assistance_branches"])
    assert all(row["scoreable_count_all_pitch_states"] == 71 for row in comparison["annotation_assistance_branches"])


def test_all_six_official_candidates_run_on_cuda_without_fallback() -> None:
    authorization = _read_json(
        STAGE / "02_OFFICIAL_MODEL_LICENCE_AND_WEIGHT_PROVENANCE" / "model_authorization_matrix.json"
    )
    preflight = _read_json(STAGE / "08_RUNTIME_VRAM_AND_FAILURE_LEDGER" / "hardware_preflight.json")
    inference = _read_json(STAGE / "05_PROMPTABLE_INFERENCE" / "promptable_inference_manifest.json")

    assert authorization["passed"] is True
    assert authorization["executed_candidate_count"] == 6
    assert all(row["admitted"] for row in authorization["candidates"])
    assert preflight["passed"] is True
    assert preflight["admitted_candidate_count"] == 6
    assert preflight["silent_cpu_fallback_count"] == 0
    assert inference["passed"] is True
    assert inference["successful_candidate_count"] == 6
    assert inference["all_cuda_no_fallback"] is True
    assert inference["all_repeatable"] is True
    assert all(row["peak_allocated_vram_bytes"] <= int(6.5 * 1024**3) for row in inference["rows"])


def test_official_top_k_is_consolidated_and_merged_masks_are_never_clean() -> None:
    consolidation = _read_json(
        STAGE / "06_MASK_OUTPUT_CONSOLIDATION_AND_EVALUATION" / "mask_output_consolidation_spec.json"
    )
    runtime = _read_jsonl(STAGE / "06_MASK_OUTPUT_CONSOLIDATION_AND_EVALUATION" / "runtime_prompt_results.jsonl")

    assert consolidation["official_multimask_top_k"] == 3
    assert consolidation["all_official_multimask_outputs_enter_runtime_consolidation"] is True
    assert consolidation["merged_output_action"] == "ROUTE_UNRESOLVED_AND_NEVER_ACCEPT_AS_CLEAN"
    assert sum(row["merged_output_routed_count"] for row in runtime) > 0
    assert sum(row["merged_as_clean_output_count"] for row in runtime) == 0
    assert sum(row["accepted_duplicate_mask_count"] for row in runtime) == 0
    assert any(row["suppressed_or_extra_duplicate_mask_count"] > 0 for row in runtime)


def test_shortlist_and_safety_are_bounded_development_only() -> None:
    shortlist = _read_json(STAGE / "09_NEXT_STAGE_DECISION" / "development_shortlist.json")
    summary = _read_json(STAGE / "10_COMMANDS_AND_TESTS" / "build_summary.json")

    assert shortlist["decision_code"] == "C"
    assert shortlist["decision"] == "FREEZE_LIGHTWEIGHT_RUNTIME_BRANCH_ONLY"
    assert shortlist["runtime_branch"]["candidate_id"] == "light_hq_sam_vit_tiny"
    assert shortlist["runtime_branch"]["crop_type"] == "C1"
    assert shortlist["runtime_branch"]["prompt_type"] == "R0"
    assert shortlist["annotation_assistance_branch"] is None
    assert shortlist["maximum_one_per_role"] is True
    for payload in (shortlist, summary):
        assert payload["production_ready"] is False
        assert payload["safe_to_apply_globally"] is False
        assert payload["training_performed"] is False
        assert payload["fine_tuning_performed"] is False
        assert payload["threshold_tuning_performed"] is False
        assert payload["production_component_promoted"] is False


def test_review_pack_is_flat_bounded_and_excludes_sensitive_payloads() -> None:
    validation = _read_json(STAGE / "10_COMMANDS_AND_TESTS" / "review_pack_validation.json")
    files = [path for path in REVIEW_PACK.iterdir() if path.is_file()]

    assert validation["passed"] is True
    assert validation["file_count_including_manifest"] == len(files) <= 20
    assert validation["visual_file_count"] == 3
    assert validation["total_bytes_excluding_manifest"] <= 50 * 1024 * 1024
    assert all(path.parent == REVIEW_PACK for path in files)
    assert (REVIEW_PACK / "04_SOURCE_DIFF.patch").stat().st_size > 0
    assert not any(path.suffix.lower() in {".pt", ".pth", ".mp4", ".avi"} for path in files)
    json_text = "\n".join(path.read_text(encoding="utf-8") for path in files if path.suffix == ".json")
    assert "polygon_original_pixels" not in json_text
    assert "packed_bits_base64" not in json_text
