from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from football_intelligence.detection_gold.consolidation import merged_ambiguity_gate
from football_intelligence.detection_gold.dense_separation import (
    ELIGIBILITY_VARIANTS,
    binary_route_metrics,
    candidate_mask_coverage,
    classify_dense_candidate,
    consolidate_mask_outputs,
    dense_truth_classification_specification,
    eligibility_variant_specification,
    evaluate_eligibility_variants,
    mask_output_consolidation_specification,
    polygon_area,
    polygon_self_intersection_pairs,
    polygon_self_intersects,
    validate_occlusion_graph,
    validate_polygon,
)
from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.review_chassis.hashing import sha256_file

REPO = Path(__file__).resolve().parents[1]
PART3 = REPO.parent / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G4_CONDITIONAL_DENSE_REGION_INSTANCE_SEPARATION_DEVELOPMENT_v1"


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _node(
    proposal_uuid: str,
    box: tuple[float, float, float, float],
    *,
    family: str,
    view: str,
    score: float = 0.9,
) -> dict[str, object]:
    return {
        "source_frame_sha256": "a" * 64,
        "proposal_uuid": proposal_uuid,
        "source_view_family": family,
        "inference_view_id": view,
        "source_view_footprint": {"x1": 0.0, "y1": 0.0, "x2": 200.0, "y2": 200.0},
        "score": score,
        "bbox_panorama_pixels": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
        "transform_hash": "b" * 64,
        "checkpoint_runtime_hash": "c" * 64,
        "parent_lineage_ids": [f"raw:{proposal_uuid}"],
    }


def _mask(annotation_uuid: str, points: list[tuple[float, float]], **extra: object) -> dict[str, object]:
    return {
        "annotation_uuid": annotation_uuid,
        "polygon_original_pixels": [{"x": x, "y": y} for x, y in points],
        "mask_quality": "PRECISE",
        "occlusion_order": 0,
        **extra,
    }


def test_polygon_validity_self_intersection_and_roi_containment() -> None:
    square = _mask("square", [(2, 2), (8, 2), (8, 8), (2, 8)])
    bow_tie = _mask("bow", [(2, 2), (8, 8), (8, 2), (2, 8)])
    roi = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}

    assert polygon_area(square["polygon_original_pixels"]) == 36
    assert not polygon_self_intersects(square["polygon_original_pixels"])
    assert validate_polygon(square["polygon_original_pixels"], roi, minimum_area=4)["valid"]
    assert polygon_self_intersects(bow_tie["polygon_original_pixels"])
    assert polygon_self_intersection_pairs(bow_tie["polygon_original_pixels"]) == [(0, 2)]
    assert "SELF_INTERSECTION" in validate_polygon(bow_tie["polygon_original_pixels"], roi, minimum_area=1)["errors"]


def test_candidate_mask_coverage_and_merged_truth() -> None:
    masks = [
        _mask("left", [(1, 1), (5, 1), (5, 9), (1, 9)]),
        _mask("right", [(7, 1), (11, 1), (11, 9), (7, 9)]),
    ]
    candidate = _node("container", (0, 0, 12, 10), family="FULL", view="full")
    coverage = candidate_mask_coverage(candidate, masks)

    assert len(coverage) == 2
    assert all(0 <= row["candidate_visible_mask_coverage"] <= 1 for row in coverage)
    assert classify_dense_candidate(coverage)["truth_class"] == "MERGED_MULTIPLE_PEOPLE"


def test_dense_truth_specification_is_frozen_before_scoring() -> None:
    specification = dense_truth_classification_specification()

    assert specification["frozen_before_scoring"] is True
    assert specification["human_gold_runtime_input_forbidden"] is True
    assert specification["mask_intersection_fraction_threshold"] == 0.10


def test_occlusion_graph_flags_bad_order_and_cycle() -> None:
    masks = [
        _mask("a", [(0, 0), (4, 0), (4, 4), (0, 4)], occluder_uuid="b", occlusion_order=0),
        _mask("b", [(1, 1), (5, 1), (5, 5), (1, 5)], occluder_uuid="a", occlusion_order=0),
    ]
    result = validate_occlusion_graph(masks)

    assert result["valid"] is False
    assert result["cycle_detected"] is True
    assert {row["type"] for row in result["errors"]} == {
        "OCCLUDER_NOT_IN_FRONT",
        "OCCLUSION_CYCLE",
    }


def test_runtime_gate_rejects_nested_human_gold() -> None:
    nodes = [
        {
            **_node("one", (0, 0, 20, 40), family="FULL", view="full"),
            "nested": {"human_truth": True},
        }
    ]

    with pytest.raises(ValueError, match="gold/runtime leakage"):
        evaluate_eligibility_variants(nodes, nodes)


def test_all_frozen_variants_are_deterministic_and_e0_matches_g3() -> None:
    container = _node("container", (0, 0, 100, 100), family="FULL", view="full")
    left = _node("left", (10, 20, 35, 80), family="TILE", view="tile-left")
    right = _node("right", (65, 20, 90, 80), family="TILE", view="tile-right")
    nodes = [container, left, right]

    first = evaluate_eligibility_variants([container], nodes)
    second = evaluate_eligibility_variants([container], deepcopy(nodes))
    g3_state, _ = merged_ambiguity_gate([container], nodes)

    assert set(first["variant_routes"]) == set(ELIGIBILITY_VARIANTS)
    assert first["variant_routes"]["E0"]["route"] == (g3_state == "ROUTE_DENSE_REVIEW")
    assert first["variant_routes"]["E1"]["route"] is True
    assert first["variant_routes"]["E3"]["route"] is True
    assert first["variant_routes"]["E5"]["route"] is True
    assert first["determinism_hash"] == second["determinism_hash"]


def test_eligibility_specification_contains_no_post_score_variant() -> None:
    specification = eligibility_variant_specification()

    assert specification["frozen_before_scoring"] is True
    assert tuple(specification["variants"]) == ELIGIBILITY_VARIANTS
    assert specification["threshold_search_performed"] is False
    assert specification["learned_gate"] is False


def test_exact_route_metrics_do_not_claim_population_confidence() -> None:
    result = binary_route_metrics(
        [
            {"truth_requires_route": True, "route": True},
            {"truth_requires_route": True, "route": False},
            {"truth_requires_route": False, "route": True},
            {"truth_requires_route": False, "route": False},
        ]
    )

    assert result["true_positive"] == 1
    assert result["false_negative"] == 1
    assert result["false_positive"] == 1
    assert result["true_negative"] == 1
    assert result["population_confidence_interval_reported"] is False


def test_mask_output_deduplication_and_merged_risk_routing() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:10, 2:10] = True
    duplicate = mask.copy()
    merged = np.zeros((20, 20), dtype=bool)
    merged[2:16, 2:16] = True
    outputs = [
        {
            "output_uuid": "accepted",
            "score": 0.9,
            "binary_mask": mask,
            "current_frame_pixel_support": True,
            "runtime_merged_risk": False,
            "prompt_source_uuid": "proposal-a",
        },
        {
            "output_uuid": "duplicate",
            "score": 0.8,
            "binary_mask": duplicate,
            "current_frame_pixel_support": True,
            "runtime_merged_risk": False,
            "prompt_source_uuid": "proposal-b",
        },
        {
            "output_uuid": "merged",
            "score": 0.7,
            "binary_mask": merged,
            "current_frame_pixel_support": True,
            "runtime_merged_risk": True,
            "prompt_source_uuid": "proposal-c",
        },
    ]
    result = {row["output_uuid"]: row for row in consolidate_mask_outputs(outputs)}

    assert result["accepted"]["output_state"] == "ACCEPT_VISIBLE_INSTANCE"
    assert result["duplicate"]["output_state"] == "SUPPRESS_DUPLICATE_MASK"
    assert result["merged"]["output_state"] == "ROUTE_HUMAN_DENSE_REVIEW"
    assert mask_output_consolidation_specification()["mask_averaging_forbidden"] is True


def test_specification_hash_is_stable() -> None:
    specification = eligibility_variant_specification()

    assert stable_hash(specification) == stable_hash(deepcopy(specification))


def test_generated_c1_completion_bundle_and_prior_stage_preservation() -> None:
    validation = _read_json(
        STAGE / "01_C1_COMPLETION_INGESTION_AND_MASK_QA" / "c1_completion_and_dense_gold_validation.json"
    )
    preservation = _read_json(STAGE / "09_COMMANDS_AND_TESTS" / "prior_stage_preservation.json")

    assert validation["passed"] is True
    assert all(validation["checks"].values())
    assert len(validation["case_ids"]) == 8
    assert all(row["candidate_relation_cardinality_exact"] for row in validation["case_validation"])
    assert preservation["passed"] is True
    assert preservation["before"] == preservation["after"]
    assert preservation["historical_artifacts_mutated"] is False
    assert preservation["human_masks_mutated"] is False


def test_generated_dense_gold_qa_is_actionable_without_mutation() -> None:
    quality = _read_json(STAGE / "01_C1_COMPLETION_INGESTION_AND_MASK_QA" / "dense_gold_quality_flags.json")
    queue = _read_json(STAGE / "01_C1_COMPLETION_INGESTION_AND_MASK_QA" / "dense_gold_manual_review_queue.json")

    assert quality["case_count"] == 8
    assert quality["visible_mask_count"] == 73
    assert quality["masks_modified"] is False
    assert quality["development_gold_usable_for_bounded_evaluation"] is False
    crossing_errors = [
        error
        for row in quality["rows"]
        for error in row["material_errors"]
        if "SELF_INTERSECTION" in error.get("flags", [])
    ]
    assert crossing_errors
    assert all(error["self_intersection_edge_pairs"] for error in crossing_errors)
    assert all(row["action"] == "REVIEW_ONLY_DO_NOT_AUTO_CORRECT" for row in queue["rows"])


def test_generated_g3_parity_oracle_and_eligibility_boundary() -> None:
    baseline = _read_json(STAGE / "03_BOX_ONLY_AND_MASK_ORACLE_BASELINES" / "box_only_dense_baseline.json")
    oracle = _read_json(STAGE / "03_BOX_ONLY_AND_MASK_ORACLE_BASELINES" / "human_mask_oracle_upper_bound.json")
    eligibility = _read_json(STAGE / "04_RUNTIME_ELIGIBILITY_GATE" / "dense_eligibility_results.json")

    assert baseline["g3_baseline_parity"]["matches_frozen_g3_report"] is True
    assert baseline["g3_baseline_parity"]["variant"] == "IOU_CONNECTED_COMPONENT_055"
    assert oracle["label"] == "HUMAN_MASK_ORACLE_NOT_RUNTIME"
    assert oracle["human_gold_runtime_input"] is False
    assert oracle["model_inference_performed"] is False
    assert set(eligibility["variants"]) == set(ELIGIBILITY_VARIANTS)
    assert eligibility["runtime_gate_input_includes_human_gold"] is False
    assert eligibility["shortlisted_variants"] == []
    assert all(row["screening_checks"]["no_gold_runtime_leakage"] for row in eligibility["variants"].values())


def test_generated_promptable_skip_and_runtime_vram_screens() -> None:
    provenance = _read_json(
        STAGE / "05_PROMPTABLE_MASK_RESEARCH_BRANCH" / "promptable_weight_and_licence_provenance.json"
    )
    experiment = _read_json(STAGE / "05_PROMPTABLE_MASK_RESEARCH_BRANCH" / "promptable_mask_experiment_manifest.json")
    runtime = _read_json(STAGE / "06_INSTANCE_OUTPUT_AND_ERROR_EVALUATION" / "runtime_and_vram.json")

    assert provenance["status"] == "SKIPPED_NO_AUTHORIZED_LOCAL_PROMPTABLE_WEIGHT"
    assert provenance["candidate_local_weight_count"] == 0
    assert provenance["download_performed"] is False
    assert experiment["experiment_performed"] is False
    assert experiment["human_prompts_hidden_in_p1_p2"] is True
    assert runtime["cuda_available"] is True
    assert runtime["silent_cpu_fallback"] is False
    assert runtime["c1_exact_replay_view_count"] == 40
    assert runtime["peak_allocated_vram_mib"] > 0
    assert set(runtime["eligibility_gate_cpu"]) == set(ELIGIBILITY_VARIANTS)


def test_generated_safety_and_flat_review_pack_limits() -> None:
    outcome = _read_json(STAGE / "08_NEXT_STAGE_DECISION" / "executive_outcome.json")
    pack = STAGE / "10_REVIEW_PACK_FOR_CHATGPT"
    manifest = _read_json(pack / "19_REVIEW_PACK_MANIFEST.json")
    files = list(pack.iterdir())

    assert outcome["classification"] == "PASS_CONDITIONAL_DENSE_INSTANCE_SEPARATION_DEVELOPMENT_READY_FOR_PRO_REVIEW"
    assert outcome["final_decision"] == "REPAIR_DENSE_GOLD_OR_PROVENANCE"
    for field in (
        "training_performed",
        "fine_tuning_performed",
        "learned_gate_or_classifier_created",
        "detector_or_consolidator_or_segmenter_or_tracker_promoted",
        "production_defaults_changed",
        "identity_tracking_performed",
        "temporal_state_created",
    ):
        assert outcome[field] is False
    assert len(files) <= 20
    assert sum(path.stat().st_size for path in files) <= 50 * 1024 * 1024
    assert all(path.is_file() for path in files)
    assert len([path for path in files if path.suffix.lower() == ".png"]) <= 3
    assert (pack / "04_SOURCE_DIFF.patch").stat().st_size > 0
    assert manifest["file_count_including_manifest"] == len(files)
    assert manifest["total_size_bytes_including_manifest"] == sum(path.stat().st_size for path in files)
    assert "19_REVIEW_PACK_MANIFEST.json" not in {row["filename"] for row in manifest["files"]}
    assert all(sha256_file(pack / row["filename"]) == row["sha256"] for row in manifest["files"])
