from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_intelligence.detection_gold.consolidation import (
    VARIANT_NAMES,
    consolidate_proposals,
    consolidation_variant_specification,
    freeze_variant_specification,
    merged_ambiguity_gate,
    validate_observation_provenance,
    validate_proposal_nodes,
)
from football_intelligence.detection_gold.consolidation_evaluation import (
    aggregate_source_results,
    classify_box_against_roi_union,
    evaluate_source_observations,
    screening_checks,
)
from football_intelligence.review_chassis.hashing import sha256_file

REPO = Path(__file__).resolve().parents[1]
PART3 = REPO.parent / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G3_PROVENANCE_AWARE_CROSS_VIEW_CONSOLIDATION_AND_MERGED_AMBIGUITY_GATE_DEVELOPMENT_v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def proposal(
    identifier: str,
    box: tuple[float, float, float, float],
    *,
    score: float = 0.9,
    view_id: str = "full-1280",
    family: str = "FULL_PANORAMA_1280",
    footprint: tuple[float, float, float, float] = (0, 0, 100, 100),
) -> dict:
    return {
        "source_frame_sha256": "a" * 64,
        "proposal_uuid": identifier,
        "source_view_family": family,
        "inference_view_id": view_id,
        "source_view_footprint": dict(zip(("x1", "y1", "x2", "y2"), footprint, strict=True)),
        "score": score,
        "bbox_panorama_pixels": dict(zip(("x1", "y1", "x2", "y2"), box, strict=True)),
        "transform_hash": f"transform-{view_id}",
        "checkpoint_runtime_hash": "runtime-fixed",
        "parent_lineage_ids": [f"raw:{identifier}"],
    }


def observation(identifier: str, box: tuple[float, float, float, float]) -> dict:
    return {
        "observation_uuid": identifier,
        "box_panorama_pixels": dict(zip(("x1", "y1", "x2", "y2"), box, strict=True)),
        "score": 0.9,
        "output_state": "ACCEPT_INDEPENDENT_OBSERVATION",
    }


def gold(identifier: str, box: tuple[float, float, float, float]) -> dict:
    return {
        "canonical_gold_person_cluster_id": identifier,
        "source_group_id": "source-1",
        "canonical_visible_body_box": dict(zip(("x1", "y1", "x2", "y2"), box, strict=True)),
    }


def evaluation_result(gold_rows: list[dict], pre_nodes: list[dict], observations: list[dict]) -> dict:
    return evaluate_source_observations(
        gold_rows,
        pre_nodes,
        {
            "source_frame_sha256": "a" * 64,
            "observations": observations,
        },
        [{"x1": -100, "y1": -100, "x2": 200, "y2": 200}],
    )


def test_evaluation_roi_boundaries_do_not_score_outside_boxes_as_false() -> None:
    roi = [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}]
    included = classify_box_against_roi_union({"x1": 8, "y1": 0, "x2": 12, "y2": 10}, roi)
    boundary = classify_box_against_roi_union({"x1": 9, "y1": 0, "x2": 13, "y2": 10}, roi)
    outside = classify_box_against_roi_union({"x1": 20, "y1": 0, "x2": 24, "y2": 10}, roi)
    assert included["evaluation_roi_state"] == "INCLUDED_IN_FALSE_OBSERVATION_EVALUATION"
    assert included["area_fraction_inside_labelled_roi_union"] == 0.5
    assert boundary["evaluation_roi_state"] == "ROI_BOUNDARY_IGNORED"
    assert outside["evaluation_roi_state"] == "OUTSIDE_LABELLED_ROI_IGNORED"


def test_proposal_nodes_require_complete_exact_provenance() -> None:
    node = proposal("one", (0, 0, 10, 20))
    validate_proposal_nodes([node])
    invalid = {**node, "parent_lineage_ids": []}
    with pytest.raises(ValueError, match="missing parent lineage"):
        validate_proposal_nodes([invalid])


@pytest.mark.parametrize(
    "forbidden",
    ["gold_geometry", "annotation_uuid", "role", "pitch_state", "case_stratum", "failure_label"],
)
def test_runtime_rejects_gold_and_human_feature_leakage(forbidden: str) -> None:
    node = {**proposal("one", (0, 0, 10, 20)), forbidden: "not-runtime-evidence"}
    with pytest.raises(ValueError, match="forbidden runtime fields"):
        consolidate_proposals([node], VARIANT_NAMES[0], apply_merged_gate=False)


def test_variant_specification_freezes_idempotently_before_scoring(tmp_path: Path) -> None:
    spec_path = tmp_path / "specification.json"
    hash_path = tmp_path / "specification.sha256"
    first = freeze_variant_specification(spec_path, hash_path)
    second = freeze_variant_specification(spec_path, hash_path)
    specification = consolidation_variant_specification()
    assert first == second == sha256_file(spec_path)
    assert specification["frozen_before_scoring"] is True
    assert [row["name"] for row in specification["variants"]] == list(VARIANT_NAMES)
    assert len(specification["variants"]) == 8


def test_connected_component_baseline_preserves_v0_transitive_parity() -> None:
    nodes = [
        proposal("a", (0, 0, 10, 20), score=0.95),
        proposal("b", (2, 0, 12, 20), score=0.90),
        proposal("c", (4, 0, 14, 20), score=0.85),
    ]
    result = consolidate_proposals(nodes, "IOU_CONNECTED_COMPONENT_055", apply_merged_gate=False)
    assert len(result["observations"]) == 1
    assert result["observations"][0]["representative_proposal_uuid"] == "a"
    assert result["observations"][0]["cluster_member_proposal_uuids"] == ["a", "b", "c"]


@pytest.mark.parametrize("variant", ["GREEDY_NMS_050", "GREEDY_NMS_065", "SOFT_NMS_LINEAR_050"])
def test_nms_variants_are_deterministic_under_input_permutation(variant: str) -> None:
    nodes = [
        proposal("a", (0, 0, 10, 20), score=0.90),
        proposal("b", (1, 0, 11, 20), score=0.90),
        proposal("c", (40, 0, 50, 20), score=0.80),
    ]
    forward = consolidate_proposals(nodes, variant, apply_merged_gate=False)
    reverse = consolidate_proposals(list(reversed(nodes)), variant, apply_merged_gate=False)
    assert forward == reverse


def test_complete_link_prevents_transitive_chain_distinct_person_collapse() -> None:
    nodes = [
        proposal("a", (0, 0, 10, 20), score=0.95),
        proposal("b", (2, 0, 12, 20), score=0.90),
        proposal("c", (4, 0, 14, 20), score=0.85),
    ]
    graph = consolidate_proposals(nodes, "PROVENANCE_GRAPH_BALANCED_HIGHEST_SCORE", apply_merged_gate=False)
    assert len(graph["observations"]) == 2
    assert sorted(len(row["cluster_member_proposal_uuids"]) for row in graph["observations"]) == [1, 2]


def test_medoid_is_a_real_member_and_never_an_averaged_box() -> None:
    nodes = [
        proposal("left", (0, 0, 10, 20), score=0.95),
        proposal("middle", (1, 0, 11, 20), score=0.80),
        proposal("right", (2, 0, 12, 20), score=0.90),
    ]
    result = consolidate_proposals(nodes, "PROVENANCE_GRAPH_BALANCED_MEDOID", apply_merged_gate=False)
    row = result["observations"][0]
    assert row["representative_proposal_uuid"] == "middle"
    assert row["box_panorama_pixels"] == nodes[1]["bbox_panorama_pixels"]
    assert row["box_panorama_pixels"] in [node["bbox_panorama_pixels"] for node in nodes]
    assert validate_observation_provenance(result, nodes)["coordinate_averaging_performed"] is False


def test_merged_gate_uses_cross_view_split_evidence_without_splitting() -> None:
    container = proposal("container", (0, 0, 40, 20), family="FULL_PANORAMA_1280")
    left = proposal(
        "left",
        (2, 0, 12, 20),
        view_id="tile-left",
        family="OVERLAPPING_HIGH_RESOLUTION_TILES",
    )
    right = proposal(
        "right",
        (26, 0, 36, 20),
        view_id="tile-right",
        family="OVERLAPPING_HIGH_RESOLUTION_TILES",
    )
    state, reasons = merged_ambiguity_gate([container], [container, left, right])
    assert state == "ROUTE_DENSE_REVIEW"
    assert {row["reason"] for row in reasons} == {"CROSS_VIEW_SPLIT_EVIDENCE"}
    result = consolidate_proposals(
        [container, left, right], "PROVENANCE_GRAPH_BALANCED_HIGHEST_SCORE", apply_merged_gate=True
    )
    all_boxes = [node["bbox_panorama_pixels"] for node in (container, left, right)]
    assert all(row["box_panorama_pixels"] in all_boxes for row in result["observations"])
    assert sorted(
        identifier for row in result["observations"] for identifier in row["cluster_member_proposal_uuids"]
    ) == ["container", "left", "right"]


def test_stale_or_invalid_provenance_is_rejected() -> None:
    nodes = [proposal("one", (0, 0, 10, 20)), proposal("two", (1, 0, 11, 20))]
    result = consolidate_proposals(nodes, "GREEDY_NMS_050", apply_merged_gate=False)
    stale_nonrepresentative = {
        **nodes[1],
        "bbox_panorama_pixels": {"x1": 1.25, "y1": 0, "x2": 11.25, "y2": 20},
    }
    stale = [nodes[0], stale_nonrepresentative]
    validation = validate_observation_provenance(result, stale)
    assert validation["passed"] is False
    assert any("member proposal hash mismatch" in error for error in validation["errors"])


def test_duplicate_observation_metric_is_separate_from_supply() -> None:
    target = gold("person", (0, 0, 10, 20))
    pre = [proposal("one", (0, 0, 10, 20)), proposal("two", (0.5, 0, 10.5, 20))]
    result = evaluation_result(
        [target], pre, [observation("one", (0, 0, 10, 20)), observation("two", (0.5, 0, 10.5, 20))]
    )
    assert result["metrics"]["accepted_independent_supply"]["numerator"] == 1
    assert result["metrics"]["duplicate_final_observation_count"] == 1


def test_merged_as_clean_metric_counts_one_box_supporting_two_people() -> None:
    targets = [gold("a", (0, 0, 10, 20)), gold("b", (11, 0, 21, 20))]
    merged = proposal("merged", (0, 0, 21, 20))
    result = evaluation_result(targets, [merged], [observation("merged", (0, 0, 21, 20))])
    assert result["metrics"]["merged_as_clean_observation_count"] == 1


def test_distinct_person_suppression_metric_uses_preconsolidation_upper_bound() -> None:
    targets = [gold("a", (0, 0, 10, 20)), gold("b", (30, 0, 40, 20))]
    pre = [proposal("a", (0, 0, 10, 20)), proposal("b", (30, 0, 40, 20))]
    result = evaluation_result(targets, pre, [observation("a", (0, 0, 10, 20))])
    assert result["metrics"]["distinct_person_suppression_count"] == 1
    assert result["metrics"]["distinct_person_suppression_rate"]["rate"] == 0.5


def test_equal_source_aggregation_is_reported_separately_from_pooled_people() -> None:
    def source_row(source_hash: str, count: int, accepted: bool) -> dict:
        people = [
            {
                "accepted_any": accepted,
                "accepted_exactly_one": accepted,
                "routed_to_dense_review": False,
                "accepted_plus_dense_covered": accepted,
                "preconsolidation_independent_support": True,
                "distinct_person_suppressed": not accepted,
                "duplicate_final_observation_count": 0,
            }
            for _ in range(count)
        ]
        return {
            "source_frame_sha256": source_hash,
            "person_rows": people,
            "accepted_assignments": [],
            "metrics": {
                "merged_as_clean_observation_count": 0,
                "merged_as_clean_observation_rate": {"rate": 0.0},
                "background_accepted_observation_count": 0,
                "accepted_observation_count_inside_labelled_roi": count if accepted else 0,
                "preconsolidation_candidate_count_inside_labelled_roi": count,
                "dense_review_observation_count_inside_labelled_roi": 0,
                "accepted_independent_supply": {"rate": 1.0 if accepted else 0.0},
                "exactly_one_accepted_observation": {"rate": 1.0 if accepted else 0.0},
                "no_accepted_observation": {"rate": 0.0 if accepted else 1.0},
                "routed_to_dense_review": {"rate": 0.0},
                "accepted_plus_dense_routed_coverage": {"rate": 1.0 if accepted else 0.0},
                "duplicate_final_observation_count": 0,
                "duplicate_final_observation_rate": {"rate": 0.0},
                "distinct_person_suppression_count": 0 if accepted else count,
                "distinct_person_suppression_rate": {"rate": 0.0 if accepted else 1.0},
                "observation_count_error": 0 if accepted else -count,
                "absolute_observation_count_error": 0 if accepted else count,
                "candidate_reduction_ratio": 0.0 if accepted else 1.0,
                "median_visible_box_iou": None,
                "median_normalized_bottom_centre_displacement": None,
            },
        }

    aggregate = aggregate_source_results([source_row("a", 100, True), source_row("b", 1, False)])
    assert aggregate["accepted_independent_supply"]["numerator"] == 100
    assert aggregate["accepted_independent_supply"]["denominator"] == 101
    assert aggregate["equal_source_group_accepted_supply_rate"] == 0.5
    assert aggregate["primary_aggregation"] == "EQUAL_SOURCE_GROUP"
    assert aggregate["primary_equal_source_group_metrics"]["accepted_independent_supply_rate"] == 0.5
    assert aggregate["pooled_person_results"]["accepted_independent_supply"]["numerator"] == 100


def test_screening_is_bounded_and_never_claims_final_acceptance() -> None:
    aggregate = {
        "merged_as_clean_observation_count": 0,
        "distinct_person_suppression_rate": {"rate": 0.01},
        "duplicate_final_observation_rate": {"rate": 0.01},
        "accepted_independent_supply": {"numerator": 240},
        "accepted_plus_dense_routed_coverage": {"numerator": 288},
        "background_accepted_observation_count": 2,
    }
    result = screening_checks(
        aggregate,
        baseline_background_count=2,
        cpu_p95_milliseconds=30.0,
        deterministic=True,
        provenance_exact=True,
    )
    assert result["passed"] is True
    assert result["screening_only_not_final_acceptance"] is True
    assert result["hard_gate_pass_claimed"] is False


def test_g2b_artifacts_case008_folds_and_safety_outputs_validate() -> None:
    validation = read_json(STAGE / "01_G2B_INGESTION_AND_PRECONSOLIDATION_AUDIT" / "g2b_input_validation.json")
    results = read_json(STAGE / "04_DUPLICATE_CLUSTERING_AND_REPRESENTATIVE_SELECTION" / "consolidation_results.json")
    folds = read_json(STAGE / "06_PERSON_OBSERVATION_EVALUATION" / "source_fold_stability.json")
    summary = read_json(STAGE / "M5_5G3_STAGE_SUMMARY.json")
    assert validation["passed"] is True
    assert all(validation["artifact_hash_checks"].values())
    assert all(
        result["sensitivities"]["case_008_sensitivity"]["with_case_008"]["canonical_gold_person_count"]
        >= result["sensitivities"]["case_008_sensitivity"]["without_case_008"]["canonical_gold_person_count"]
        for result in results["with_merged_gate"].values()
    )
    assert len(folds["fold_pareto_variants"]) == 5
    assert all(len(rows) == 5 for rows in folds["variant_rows"].values())
    assert folds["threshold_tuning_per_fold"] is False
    assert summary["training_performed"] is False
    assert summary["identity_tracking_performed"] is False
    assert summary["detector_tracker_or_consolidator_promoted"] is False
    assert summary["production_defaults_changed"] is False


def test_review_pack_is_flat_bounded_hashed_and_contains_three_real_atlases() -> None:
    pack = STAGE / "10_REVIEW_PACK_FOR_CHATGPT"
    manifest = read_json(pack / "19_REVIEW_PACK_MANIFEST.json")
    assert manifest["passed"] is True
    assert manifest["file_count_including_manifest"] <= 20
    assert manifest["total_bytes_excluding_manifest"] <= 50 * 1024 * 1024
    assert manifest["visual_count"] == 3
    source_diff = (pack / "04_SOURCE_DIFF.patch").read_text(encoding="utf-8")
    assert source_diff
    assert {
        "scripts/build_m5_5g3_consolidation.py",
        "src/football_intelligence/detection_gold/consolidation.py",
        "src/football_intelligence/detection_gold/consolidation_evaluation.py",
        "tests/test_m5_5g3_consolidation.py",
    } <= {line.removeprefix("+++ b/") for line in source_diff.splitlines() if line.startswith("+++ b/")}
    assert not [path for path in pack.iterdir() if path.is_dir()]
    assert "19_REVIEW_PACK_MANIFEST.json" not in {row["name"] for row in manifest["files"]}
