from __future__ import annotations

from copy import deepcopy

import pytest

from football_intelligence.football_observation_reasoner.g7b_pairwise import (
    PAIR_CLASSES,
    PairwiseOOFConfig,
    append_supplied_oof_node_probability_summaries,
    canonicalize_g7a_edge_features,
    canonicalize_pair_feature_mapping,
    grouped_oof_pairwise_evaluation,
    pair_metrics_and_screen,
)


def _pair_features(value: float, *, left: float = 0.2, right: float = 0.8) -> dict[str, object]:
    return {
        "bbox_iou": value,
        "left_containment_fraction": left,
        "right_containment_fraction": right,
        "visual_embedding_cosine_similarity": 1.0 - value / 5.0,
        "mask_iou": None,
        "same_source_view": True,
        "human_identity_feature_used": False,
        "feature_hash": "not-a-runtime-feature",
    }


def _edge(
    identifier: str,
    group: str,
    left: str,
    right: str,
    relation: str,
    value: float,
) -> dict[str, object]:
    return {
        "edge_uuid": identifier,
        "source_group_id": group,
        "left_candidate_uuid": left,
        "right_candidate_uuid": right,
        "target_relation": relation,
        "target_available": True,
        "identity_relation_created": False,
        "pair_features": _pair_features(value),
    }


def _synthetic_population() -> tuple[list[dict[str, object]], dict[str, int], dict[str, tuple[float, ...]]]:
    edges: list[dict[str, object]] = []
    folds = {"group_0": 0, "group_1": 1}
    node_features: dict[str, tuple[float, ...]] = {}
    class_value = {
        "SAME_PERSON_DUPLICATE": 0.9,
        "DISTINCT_PEOPLE": 0.1,
        "MERGED_CONTAINS_BOTH": 0.7,
        "INSUFFICIENT_EVIDENCE": 0.3,
    }
    for group_index, group in enumerate(folds):
        for repeat in range(2):
            for class_index, relation in enumerate(PAIR_CLASSES):
                edge_index = len(edges)
                left = f"node_{edge_index:03d}_left"
                right = f"node_{edge_index:03d}_right"
                value = class_value[relation] + 0.005 * repeat
                edges.append(_edge(f"edge_{edge_index:03d}", group, left, right, relation, value))
                node_features[left] = (value, float(class_index), float(group_index), 0.25)
                node_features[right] = (value + 0.01, float(class_index), float(group_index), 0.75)
    return edges, folds, node_features


def test_g7a_directional_features_are_canonicalized_invariant_to_endpoint_order() -> None:
    original = _pair_features(0.4, left=0.1, right=0.9)
    swapped = deepcopy(original)
    swapped["left_containment_fraction"] = original["right_containment_fraction"]
    swapped["right_containment_fraction"] = original["left_containment_fraction"]
    assert canonicalize_pair_feature_mapping(original) == canonicalize_pair_feature_mapping(swapped)
    canonical = canonicalize_pair_feature_mapping(original)
    assert canonical["endpoint_containment_fraction__min"] == 0.1
    assert canonical["endpoint_containment_fraction__max"] == 0.9
    assert "left_containment_fraction" not in canonical
    assert "right_containment_fraction" not in canonical


def test_pair_feature_contract_rejects_identity_temporal_target_and_unpaired_directional_evidence() -> None:
    with pytest.raises(ValueError, match="identity"):
        canonicalize_pair_feature_mapping({"bbox_iou": 0.4, "human_identity_feature_used": True})
    with pytest.raises(ValueError, match="prohibited"):
        canonicalize_pair_feature_mapping({"bbox_iou": 0.4, "track_id_similarity": 1.0})
    with pytest.raises(ValueError, match="leaks"):
        canonicalize_pair_feature_mapping({"bbox_iou": 0.4, "target_relation_probability": 1.0})
    with pytest.raises(ValueError, match="counterpart"):
        canonicalize_pair_feature_mapping({"bbox_iou": 0.4, "left_quality": 0.2})


def test_supplied_node_probability_join_is_symmetric_and_proves_group_exclusion() -> None:
    edge = _edge("edge", "held", "a", "b", "DISTINCT_PEOPLE", 0.2)
    canonical = canonicalize_g7a_edge_features([edge])
    supplied = {
        "a": {
            "source_group_id": "held",
            "held_out_fold": 0,
            "provenance_kind": "OUTER_GROUP_OOF",
            "model_fit_source_group_ids": ["train"],
            "probabilities": {"candidate::clean": 0.8, "candidate::duplicate": 0.2},
        },
        "b": {
            "source_group_id": "held",
            "held_out_fold": 0,
            "provenance_kind": "OUTER_GROUP_OOF",
            "model_fit_source_group_ids": ["train"],
            "probabilities": {"candidate::clean": 0.3, "candidate::duplicate": 0.7},
        },
    }
    joined = append_supplied_oof_node_probability_summaries(canonical, [edge], supplied, {"held": 0})
    swapped = deepcopy(edge)
    swapped["left_candidate_uuid"], swapped["right_candidate_uuid"] = "b", "a"
    swapped_joined = append_supplied_oof_node_probability_summaries(canonical, [swapped], supplied, {"held": 0})
    assert joined["features_by_edge_uuid"] == swapped_joined["features_by_edge_uuid"]
    assert joined["audit"]["caller_supplied_only"] is True
    assert joined["audit"]["all_edge_groups_excluded_from_node_model_fits"] is True
    row = joined["features_by_edge_uuid"]["edge"]
    assert row["oof_node_probability::candidate::clean::mean"] == pytest.approx(0.55)
    assert row["oof_node_probability::candidate::clean::abs_difference"] == pytest.approx(0.5)


def test_supplied_node_probability_join_fails_closed_on_group_leakage_or_wrong_fold() -> None:
    edge = _edge("edge", "held", "a", "b", "DISTINCT_PEOPLE", 0.2)
    canonical = canonicalize_g7a_edge_features([edge])
    base = {
        candidate: {
            "source_group_id": "held",
            "held_out_fold": 0,
            "provenance_kind": "INNER_GROUP_OOF",
            "model_fit_source_group_ids": ["train"],
            "probabilities": {"clean": 0.75, "duplicate": 0.25},
        }
        for candidate in ("a", "b")
    }
    leaking = deepcopy(base)
    leaking["a"]["model_fit_source_group_ids"] = ["held", "train"]
    with pytest.raises(ValueError, match="leaks source group"):
        append_supplied_oof_node_probability_summaries(canonical, [edge], leaking, {"held": 0})
    outer_leaking = deepcopy(base)
    outer_leaking["a"]["model_fit_source_group_ids"] = ["outer_test", "train"]
    with pytest.raises(ValueError, match="outer_test"):
        append_supplied_oof_node_probability_summaries(
            canonical,
            [edge],
            outer_leaking,
            {"held": 0},
            additional_excluded_source_group_ids=("outer_test",),
        )
    wrong_fold = deepcopy(base)
    wrong_fold["b"]["held_out_fold"] = 1
    with pytest.raises(ValueError, match="held-out fold mismatch"):
        append_supplied_oof_node_probability_summaries(canonical, [edge], wrong_fold, {"held": 0})


def test_grouped_oof_p1_p2_p3_cover_every_edge_and_preserve_all_positive_training_pairs() -> None:
    edges, folds, node_features = _synthetic_population()
    canonical = canonicalize_g7a_edge_features(edges)
    result = grouped_oof_pairwise_evaluation(
        edges,
        canonical,
        folds,
        node_features,
        config=PairwiseOOFConfig(p2_epochs=8, p2_hidden_dim=12, seed=91),
    )
    assert result["all_variants_source_group_leakage_free"] is True
    assert result["all_variants_preserve_positive_training_pairs"] is True
    assert result["identity_tracking_present"] is False
    assert result["temporal_prediction_present"] is False
    for variant in ("P1", "P2", "P3"):
        variant_result = result["variants"][variant]
        assert len(variant_result["prediction_rows"]) == len(edges)
        assert variant_result["all_labelled_edges_predicted_exactly_once"] is True
        assert variant_result["source_group_leakage_count"] == 0
        assert all(
            row["positive_training_edge_count"] == row["positive_training_edge_selected_count"]
            for row in variant_result["fold_ledger"]
        )
        assert set(variant_result["metrics"]["confusion_matrix"]) == set(PAIR_CLASSES)
        assert len(variant_result["metrics"]["ledger"]) == len(edges)
        assert all(row["group_balanced_deterministic_negative_sampling"] for row in variant_result["fold_ledger"])
    assert all(row["model"]["endpoint_order_invariance_verified"] for row in result["variants"]["P2"]["fold_ledger"])
    assert all(row["model"]["duplicate_stage_is_binary"] for row in result["variants"]["P3"]["fold_ledger"])


def test_grouped_oof_evaluation_drops_no_positive_when_negatives_are_downsampled() -> None:
    edges, folds, node_features = _synthetic_population()
    canonical = canonicalize_g7a_edge_features(edges)
    result = grouped_oof_pairwise_evaluation(
        edges,
        canonical,
        folds,
        node_features,
        config=PairwiseOOFConfig(negative_ratio=0.5, p2_epochs=2, seed=92),
        variants=("P1", "P3"),
    )
    for variant in ("P1", "P3"):
        for fold in result["variants"][variant]["fold_ledger"]:
            assert fold["all_duplicate_and_merged_training_pairs_preserved"] is True
            assert fold["missing_positive_edge_ids"] == []
            assert fold["selected_training_edge_count"] < fold["training_pool_edge_count"]


def test_runtime_oof_inference_covers_unlabelled_graph_edges_without_scoring_them() -> None:
    edges, folds, node_features = _synthetic_population()
    unlabelled = _edge("runtime_only", "group_0", "runtime_left", "runtime_right", "DISTINCT_PEOPLE", 0.4)
    unlabelled["target_available"] = False
    unlabelled["target_relation"] = None
    all_edges = [*edges, unlabelled]
    node_features["runtime_left"] = (0.4, 0.0, 0.0, 0.25)
    node_features["runtime_right"] = (0.41, 0.0, 0.0, 0.75)
    result = grouped_oof_pairwise_evaluation(
        all_edges,
        canonicalize_g7a_edge_features(all_edges),
        folds,
        node_features,
        config=PairwiseOOFConfig(seed=94),
        variants=("P1",),
        runtime_edge_rows=all_edges,
    )
    variant = result["variants"]["P1"]
    assert result["labelled_edge_count"] == len(edges)
    assert result["runtime_edge_count"] == len(all_edges)
    assert len(variant["prediction_rows"]) == len(edges)
    assert len(variant["runtime_prediction_rows"]) == len(all_edges)
    runtime_row = next(row for row in variant["runtime_prediction_rows"] if row["edge_uuid"] == "runtime_only")
    assert runtime_row["target_available"] is False
    assert "target_relation" not in runtime_row


def test_outer_fold_specific_stacked_feature_payloads_are_used_and_audited() -> None:
    edges, folds, node_features = _synthetic_population()
    canonical = canonicalize_g7a_edge_features(edges)
    result = grouped_oof_pairwise_evaluation(
        edges,
        canonical,
        folds,
        node_features,
        config=PairwiseOOFConfig(p2_epochs=2, seed=93),
        variants=("P1",),
        feature_payload_by_outer_fold={0: canonical, 1: canonical},
    )
    assert result["outer_fold_specific_pair_features_used"] is True
    assert all(row["outer_fold_specific_pair_features_used"] for row in result["variants"]["P1"]["fold_ledger"])
    with pytest.raises(ValueError, match="missing outer fold 1"):
        grouped_oof_pairwise_evaluation(
            edges,
            canonical,
            folds,
            node_features,
            variants=("P1",),
            feature_payload_by_outer_fold={0: canonical},
        )
    unaudited_stacked = deepcopy(canonical)
    unaudited_stacked["audit"] = {
        "schema_version": "football_intelligence.m5_5g7b.supplied_oof_node_probability_join.v1",
        "additional_excluded_source_group_ids": [],
        "all_edge_groups_excluded_from_node_model_fits": True,
        "all_additional_groups_excluded_from_node_model_fits": True,
    }
    with pytest.raises(ValueError, match="do not exclude all outer-test groups"):
        grouped_oof_pairwise_evaluation(
            edges,
            canonical,
            folds,
            node_features,
            variants=("P1",),
            feature_payload_by_outer_fold={0: unaudited_stacked, 1: unaudited_stacked},
        )


def test_pair_screen_has_explicit_denominators_and_contract_thresholds() -> None:
    ledger = []
    for index, relation in enumerate(PAIR_CLASSES):
        ledger.append(
            {
                "edge_uuid": f"edge_{index}",
                "target_relation": relation,
                "predicted_relation": relation,
            }
        )
    result = pair_metrics_and_screen(ledger)
    assert result["pair_screen_passed"] is True
    assert [(row["relation"], row["metric"], row["threshold"]) for row in result["screens"]] == [
        ("SAME_PERSON_DUPLICATE", "recall", 0.90),
        ("SAME_PERSON_DUPLICATE", "precision", 0.90),
        ("MERGED_CONTAINS_BOTH", "recall", 0.50),
        ("MERGED_CONTAINS_BOTH", "precision", 0.80),
    ]
    assert all(row["numerator"] == 1 and row["denominator"] == 1 for row in result["screens"])
