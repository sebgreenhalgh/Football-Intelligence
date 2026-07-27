from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from football_intelligence.football_observation_reasoner.contracts import CandidateState, PairRelation
from football_intelligence.football_observation_reasoner.dataset import (
    audit_runtime_feature_pipeline,
    assign_grouped_folds,
    fold_local_pair_sampling_manifest,
    historical_relation_mapping,
    make_edge_row,
    make_node_row,
    make_scene_row,
    map_historical_candidate_relation,
    map_historical_pair_relation,
    pair_sampling_manifest,
    sample_group_balanced_edges,
)
from football_intelligence.football_observation_reasoner.features import (
    ENCODER_PROVENANCE_SCHEMA_VERSION,
    FrozenTorchvisionEncoder,
    candidate_shape_features,
    colour_kit_evidence_features,
    deterministic_candidate_crop_boxes,
    feature_specification,
    fit_robust_perspective_prior,
    frozen_torchvision_encoder_provenance,
    pairwise_candidate_features,
    perspective_residual_features,
    pitch_context_features,
    proposal_provenance_features,
)


def _node(
    identifier: str,
    *,
    source_group: str = "group-a",
    source_hash: str = "a" * 64,
    lineage: tuple[str, ...] = (),
    state: str = "CLEAN_SINGLE_INSTANCE",
):
    return make_node_row(
        example_uuid=f"example-{identifier}",
        source_group_id=source_group,
        source_frame_sha256=source_hash,
        frame_index=12,
        candidate_uuid=identifier,
        proposal_family="YOLOV8M_OFFICIAL",
        source_view="PANORAMA",
        proposal_stage="POST_NMS",
        score=0.72,
        visible_box={"x1": 20, "y1": 30, "x2": 40, "y2": 90},
        source_coordinates={"x": 30, "y": 90, "coordinate_space": "SOURCE_FRAME_PIXELS"},
        proposal_lineage=lineage,
        candidate_state_target=state,
        label_availability_mask={"candidate_state": True, "provenance": True},
        universe="STATIC",
        case_family="A_CORE_STATIC",
    )


def test_historical_relations_map_explicitly_without_turning_unsupported_pairs_into_background() -> None:
    assert map_historical_candidate_relation("CLEAN_SINGLE_INSTANCE") is CandidateState.CLEAN_INDEPENDENT_PERSON
    assert map_historical_candidate_relation("DUPLICATE_OF_INSTANCE") is CandidateState.DUPLICATE_OF_PERSON
    assert map_historical_candidate_relation("MERGED_MULTIPLE_INSTANCES") is CandidateState.MERGED_MULTIPLE_PEOPLE
    assert map_historical_pair_relation("SAME_PERSON_ALTERNATIVES") is PairRelation.SAME_PERSON_DUPLICATE
    assert map_historical_pair_relation("MERGED_OR_MULTI_PERSON") is PairRelation.MERGED_CONTAINS_BOTH
    assert map_historical_pair_relation("BACKGROUND_OR_UNSUPPORTED") is PairRelation.INSUFFICIENT_EVIDENCE
    mapping = historical_relation_mapping()
    assert mapping["background_or_unsupported_pair_policy"] == "INSUFFICIENT_EVIDENCE_NOT_BACKGROUND_NODE_TRUTH"


def test_node_edge_and_scene_rows_are_immutable_hash_bound_and_source_bound() -> None:
    node = _node("candidate-a")
    duplicate = _node("candidate-a")
    assert node["provenance_hash"] == duplicate["provenance_hash"]
    assert node["proposal_lineage"] == ()
    assert node["source_artifact_hashes"]["source_frame"] == "a" * 64
    with pytest.raises(TypeError):
        node["visible_box"]["x1"] = 99
    with pytest.raises(TypeError):
        node["new_key"] = True

    right = _node("candidate-b", state="DUPLICATE_OF_INSTANCE")
    edge = make_edge_row(
        edge_uuid="edge-a-b",
        source_group_id="group-a",
        source_frame_sha256="a" * 64,
        frame_index=12,
        left_candidate_uuid="candidate-b",
        right_candidate_uuid="candidate-a",
        left_node_provenance_hash=right["provenance_hash"],
        right_node_provenance_hash=node["provenance_hash"],
        pair_features={"bbox_iou": 0.9},
        target_relation="SAME_PERSON_ALTERNATIVES",
        same_lineage_cluster=True,
        lineage_ids=("raw-row-1",),
        universe="STATIC",
    )
    assert edge["left_candidate_uuid"] == "candidate-a"
    assert edge["target_relation"] == "SAME_PERSON_DUPLICATE"
    assert edge["positive_pair_for_sampling"] is True
    scene = make_scene_row(
        scene_uuid="scene-a",
        source_group_id="group-a",
        source_frame_sha256="a" * 64,
        frame_index=12,
        candidate_uuids=("candidate-b", "candidate-a"),
        edge_uuids=("edge-a-b",),
        pitch_polygon=(
            {"x": 0, "y": 0},
            {"x": 100, "y": 0},
            {"x": 100, "y": 100},
            {"x": 0, "y": 100},
        ),
        perspective_map={"prior_hash": "p"},
        evaluator_person_count=1,
    )
    assert scene["candidate_uuids"] == ("candidate-a", "candidate-b")
    assert scene["evaluator_targets"]["runtime_input"] is False
    assert scene["exact_22_forcing_performed"] is False


def test_runtime_feature_maps_reject_human_target_leakage() -> None:
    with pytest.raises(ValueError, match="leaks"):
        make_node_row(
            example_uuid="leaky",
            source_group_id="group-a",
            source_frame_sha256="a" * 64,
            frame_index=1,
            candidate_uuid="candidate-leaky",
            proposal_family="family",
            source_view="view",
            proposal_stage="RAW",
            score=0.1,
            visible_box={"x1": 0, "y1": 0, "x2": 5, "y2": 10},
            source_coordinates={"x": 2.5, "y": 10},
            shape_features={"role_target": "GOALKEEPER"},
        )

    with pytest.raises(ValueError, match="leaks"):
        make_node_row(
            example_uuid="nested-leaky",
            source_group_id="group-a",
            source_frame_sha256="a" * 64,
            frame_index=1,
            candidate_uuid="candidate-nested-leaky",
            proposal_family="family",
            source_view="view",
            proposal_stage="RAW",
            score=0.1,
            visible_box={"x1": 0, "y1": 0, "x2": 5, "y2": 10},
            source_coordinates={"x": 2.5, "y": 10},
            shape_features={"nested": [{"team_target": "TEAM_1"}]},
        )


def test_runtime_feature_pipeline_receipt_scans_materialized_and_pre_materialized_containers() -> None:
    node = _node("candidate-a")
    clean = audit_runtime_feature_pipeline(
        node_rows=[node],
        edge_rows=[],
        feature_rows=[{"feature_families": {"shape_features": {"aspect_ratio": 0.4}}}],
    )
    assert clean["passed"] is True
    assert clean["defect_count"] == 0
    assert clean["scanned_container_count"] > 0
    assert clean["receipt_hash"]

    leaky = audit_runtime_feature_pipeline(
        node_rows=[node],
        edge_rows=[],
        feature_rows=[{"feature_families": {"nested": {"role_target": "GOALKEEPER"}}}],
    )
    assert leaky["passed"] is False
    assert leaky["defect_count"] == 1
    assert leaky["defects"][0]["reason"] == "HUMAN_OR_EVALUATOR_TARGET_KEY"


def test_node_label_masks_must_match_target_presence_field_by_field() -> None:
    with pytest.raises(ValueError, match="label availability mask"):
        make_node_row(
            example_uuid="mask-mismatch",
            source_group_id="group-a",
            source_frame_sha256="a" * 64,
            frame_index=1,
            candidate_uuid="candidate-mask-mismatch",
            proposal_family="family",
            source_view="view",
            proposal_stage="RAW",
            score=0.1,
            visible_box={"x1": 0, "y1": 0, "x2": 5, "y2": 10},
            source_coordinates={"x": 2.5, "y": 10},
            role_target="GOALKEEPER",
            label_availability_mask={"role": False},
        )


def test_footpoint_target_is_evaluator_only_mask_bound_and_not_a_runtime_feature() -> None:
    node = make_node_row(
        example_uuid="footpoint-labelled",
        source_group_id="group-a",
        source_frame_sha256="a" * 64,
        frame_index=1,
        candidate_uuid="candidate-footpoint",
        proposal_family="family",
        source_view="view",
        proposal_stage="RAW",
        score=0.1,
        visible_box={"x1": 0, "y1": 0, "x2": 5, "y2": 10},
        source_coordinates={"x": 2.5, "y": 10},
        footpoint_estimate={"x": 2.5, "y": 10},
        footpoint_target_source_pixels={"x": 2.25, "y": 9.75},
        footpoint_target_uncertainty_pixels=1.5,
        label_availability_mask={"footpoint": True},
    )
    assert node["footpoint_target_source_pixels"] == {"x": 2.25, "y": 9.75}
    assert node["footpoint_target_uncertainty_pixels"] == 1.5
    assert node["label_availability_mask"]["footpoint"] is True

    with pytest.raises(ValueError, match="footpoint label availability"):
        make_node_row(
            example_uuid="footpoint-mask-mismatch",
            source_group_id="group-a",
            source_frame_sha256="a" * 64,
            frame_index=1,
            candidate_uuid="candidate-footpoint-mask-mismatch",
            proposal_family="family",
            source_view="view",
            proposal_stage="RAW",
            score=0.1,
            visible_box={"x1": 0, "y1": 0, "x2": 5, "y2": 10},
            source_coordinates={"x": 2.5, "y": 10},
            footpoint_target_source_pixels={"x": 2.25, "y": 9.75},
            label_availability_mask={"footpoint": False},
        )

    with pytest.raises(ValueError, match="leaks"):
        make_node_row(
            example_uuid="footpoint-runtime-leak",
            source_group_id="group-a",
            source_frame_sha256="a" * 64,
            frame_index=1,
            candidate_uuid="candidate-footpoint-runtime-leak",
            proposal_family="family",
            source_view="view",
            proposal_stage="RAW",
            score=0.1,
            visible_box={"x1": 0, "y1": 0, "x2": 5, "y2": 10},
            source_coordinates={"x": 2.5, "y": 10},
            shape_features={"footpoint_target_source_pixels": {"x": 2.25, "y": 9.75}},
        )


def test_five_fold_assignment_unions_source_hash_group_lineage_and_positive_edges() -> None:
    rows = [
        _node("a1", source_group="g1", source_hash="1" * 64),
        _node("a2", source_group="g1", source_hash="2" * 64),
        _node("a3", source_group="g2", source_hash="1" * 64),
        _node("b1", source_group="g3", source_hash="3" * 64, lineage=("shared-raw",)),
        _node("b2", source_group="g4", source_hash="4" * 64, lineage=("shared-raw",)),
        _node("c1", source_group="g5", source_hash="5" * 64),
        _node("c2", source_group="g6", source_hash="6" * 64),
        _node("d1", source_group="g7", source_hash="7" * 64),
        _node("d2", source_group="g8", source_hash="8" * 64),
        _node("e1", source_group="g9", source_hash="9" * 64),
        _node("e2", source_group="g10", source_hash="b" * 64),
    ]
    positive_edge = {
        "edge_uuid": "positive-c1-c2",
        "left_candidate_uuid": "c1",
        "right_candidate_uuid": "c2",
        "target_relation": "MERGED_OR_MULTI_PERSON",
    }
    first = assign_grouped_folds(rows, positive_edges=(positive_edge,))
    second = assign_grouped_folds(rows, positive_edges=(positive_edge,))
    assert first["manifest_hash"] == second["manifest_hash"]
    assignments = first["assignment_by_candidate_uuid"]
    assert assignments["a1"] == assignments["a2"] == assignments["a3"]
    assert assignments["b1"] == assignments["b2"]
    assert assignments["c1"] == assignments["c2"]
    assert first["fold_count"] == 5
    assert first["random_row_split_performed"] is False
    assert first["leakage_checks"]["passed"] is True


def test_grouping_unions_equivalent_lineage_tokens_across_source_fields() -> None:
    left = _node("cross-field-left", source_group="left", source_hash="c" * 64, lineage=("shared",)).to_dict()
    right = _node("cross-field-right", source_group="right", source_hash="d" * 64).to_dict()
    right["lineage_ids"] = ["shared"]
    split = assign_grouped_folds([left, right], fold_count=2)
    assert (
        split["assignment_by_candidate_uuid"]["cross-field-left"]
        == split["assignment_by_candidate_uuid"]["cross-field-right"]
    )
    assert split["leakage_checks"]["lineage_cross_fold_count"] == 0


def test_pair_sampling_preserves_all_hard_positives_and_balances_negative_groups() -> None:
    edges = [
        {"edge_uuid": "p-dup", "source_group_id": "g1", "target_relation": "SAME_PERSON_ALTERNATIVES"},
        {"edge_uuid": "p-merge", "source_group_id": "g2", "target_relation": "MERGED_OR_MULTI_PERSON"},
    ]
    for group in ("g1", "g2", "g3"):
        edges.extend(
            {
                "edge_uuid": f"n-{group}-{index}",
                "source_group_id": group,
                "target_relation": "DISTINCT_PEOPLE",
            }
            for index in range(4)
        )
    selected = sample_group_balanced_edges(edges, negative_ratio=2.0, seed="fixed")
    selected_ids = {row["edge_uuid"] for row in selected}
    assert {"p-dup", "p-merge"} <= selected_ids
    manifest = pair_sampling_manifest(edges, selected, negative_ratio=2.0, seed="fixed")
    counts = list(manifest["negative_counts_by_source_group"].values())
    assert manifest["all_duplicate_and_merged_positives_preserved"] is True
    assert max(counts) - min(counts) <= 1
    assert manifest["random_sampling_performed"] is False


def test_fold_local_pair_sampling_never_uses_held_out_labels_and_evaluates_all_labelled_edges() -> None:
    assignments: dict[str, int] = {}
    edges = []
    for fold in range(3):
        for index, relation in enumerate(
            ("SAME_PERSON_DUPLICATE", "MERGED_CONTAINS_BOTH", "DISTINCT_PEOPLE", "DISTINCT_PEOPLE")
        ):
            left = f"f{fold}-left-{index}"
            right = f"f{fold}-right-{index}"
            assignments[left] = fold
            assignments[right] = fold
            edges.append(
                {
                    "edge_uuid": f"edge-f{fold}-{index}",
                    "source_group_id": f"group-f{fold}",
                    "left_candidate_uuid": left,
                    "right_candidate_uuid": right,
                    "target_relation": relation,
                    "target_available": True,
                }
            )
    split = {
        "fold_count": 3,
        "manifest_hash": "b" * 64,
        "assignment_by_candidate_uuid": assignments,
    }
    first = fold_local_pair_sampling_manifest(edges, split, negative_ratio=1.0, seed="fold-safe")
    assert first["all_labelled_edges_evaluated_exactly_once"] is True
    assert first["held_out_labels_used_for_training_selection"] is False
    evaluated = [edge_uuid for rows in first["held_out_evaluation_edge_uuids_by_fold"].values() for edge_uuid in rows]
    assert sorted(evaluated) == sorted(edge["edge_uuid"] for edge in edges)
    assert len(evaluated) == len(set(evaluated))
    assert all(row["training_sample_audit"]["all_duplicate_and_merged_positives_preserved"] for row in first["folds"])

    # A label-only change in fold 0 cannot alter the edge sample used when fold
    # 0 is held out, because that sampling operates on folds 1 and 2 only.
    modified = [dict(edge) for edge in edges]
    modified[0]["target_relation"] = "DISTINCT_PEOPLE"
    second = fold_local_pair_sampling_manifest(modified, split, negative_ratio=1.0, seed="fold-safe")
    assert (
        first["selected_training_edge_uuids_by_held_out_fold"]["0"]
        == second["selected_training_edge_uuids_by_held_out_fold"]["0"]
    )


def _perspective_rows() -> list[dict]:
    rows = []
    for index in range(24):
        normalized_y = 0.12 + index * 0.032
        height = 12.0 + 58.0 * normalized_y
        width = 0.36 * height
        rows.append(
            {
                "candidate_uuid": f"person-{index}",
                "source_frame_sha256": f"{index + 1:064x}",
                "source_view": "PANORAMA",
                "source_coordinates": {"x": 500.0, "y": normalized_y * 500.0},
                "visible_box": {"x1": 100.0, "y1": 10.0, "x2": 100.0 + width, "y2": 10.0 + height},
                "reliable_geometry": True,
            }
        )
    rows.append(
        {
            "candidate_uuid": "reviewer-outlier",
            "source_frame_sha256": "f" * 64,
            "source_view": "PANORAMA",
            "source_coordinates": {"x": 500.0, "y": 180.0},
            "visible_box": {"x1": 10.0, "y1": 5.0, "x2": 160.0, "y2": 455.0},
            "reliable_geometry": True,
        }
    )
    return rows


def test_robust_perspective_prior_is_probabilistic_deterministic_and_outlier_tolerant() -> None:
    rows = _perspective_rows()
    first = fit_robust_perspective_prior(rows, image_width=1000, image_height=500)
    second = fit_robust_perspective_prior(rows, image_width=1000, image_height=500)
    assert first.to_dict()["prior_hash"] == second.to_dict()["prior_hash"]
    near = first.predict_distribution(source_x=500, source_y=430, source_view="PANORAMA")
    far = first.predict_distribution(source_x=500, source_y=80, source_view="PANORAMA")
    assert near["expected_log_height"] > far["expected_log_height"]
    ordinary = {
        "source_view": "PANORAMA",
        "source_coordinates": {"x": 500.0, "y": 250.0},
        "visible_box": {"x1": 100.0, "y1": 20.0, "x2": 111.0, "y2": 51.0},
    }
    implausible = {
        **ordinary,
        "visible_box": {"x1": 100.0, "y1": 20.0, "x2": 400.0, "y2": 490.0},
    }
    ordinary_features = perspective_residual_features(first, ordinary)
    implausible_features = perspective_residual_features(first, implausible)
    assert 0.0 <= ordinary_features["plausible_scale_probability"] <= 1.0
    assert ordinary_features["plausible_scale_probability"] > implausible_features["plausible_scale_probability"]
    assert ordinary_features["hard_scale_rejection"] is False


def test_geometry_crops_colour_shape_pitch_provenance_and_pair_features_are_deterministic() -> None:
    box = {"x1": 2.2, "y1": 3.4, "x2": 18.7, "y2": 38.9}
    crops = deterministic_candidate_crop_boxes(box, image_width=24, image_height=42)
    assert crops == deterministic_candidate_crop_boxes(box, image_width=24, image_height=42)
    assert crops["human_mask_used"] is False
    specification = feature_specification()
    assert specification["candidate_crops"]["visual_embedding_input_crop"] == "context"
    assert specification["candidate_crops"]["fixed_context_fraction"] == pytest.approx(0.18)
    assert all(0 <= crop["x1"] < crop["x2"] <= 24 for crop in crops["crops"].values())
    assert all(0 <= crop["y1"] < crop["y2"] <= 42 for crop in crops["crops"].values())

    rgb = np.zeros((36, 16, 3), dtype=np.uint8)
    rgb[:18, :, 0] = 220
    rgb[18:, :, 2] = 180
    colour = colour_kit_evidence_features(rgb)
    assert colour["colour_used_as_role_truth"] is False
    assert colour["warmup_colour_mismatch_maps_to_staff"] is False
    mask = np.zeros((36, 16), dtype=np.uint8)
    mask[2:24, 5:11] = 1
    mask[24:35, 4:7] = 1
    mask[24:35, 9:12] = 1
    shape = candidate_shape_features(box, candidate_rgb=rgb, visible_mask=mask)
    assert shape["visible_mask_component_count"] == 1
    assert shape["multi_peak_lower_body_count"] == 2

    polygon = ({"x": 0, "y": 0}, {"x": 24, "y": 0}, {"x": 24, "y": 40}, {"x": 0, "y": 40})
    pitch = pitch_context_features(box, polygon, frame_width=24, frame_height=42)
    assert pitch["off_pitch_is_background"] is False
    candidate = {
        "candidate_uuid": "left",
        "visible_box": box,
        "proposal_family": "family",
        "source_view": "panorama",
        "proposal_stage": "POST_NMS",
        "score": 0.8,
        "proposal_lineage": ["raw-a"],
    }
    other = {
        **candidate,
        "candidate_uuid": "right",
        "visible_box": {"x1": 3.0, "y1": 4.0, "x2": 19.0, "y2": 39.0},
    }
    provenance = proposal_provenance_features(candidate)
    pair = pairwise_candidate_features(
        candidate,
        other,
        left_embedding=[1.0, 0.0],
        right_embedding=[1.0, 0.0],
        left_colour_vector=[0.5, 0.5],
        right_colour_vector=[0.5, 0.5],
    )
    assert provenance["human_truth_used"] is False
    assert pair["visual_embedding_cosine_similarity"] == pytest.approx(1.0)
    assert pair["same_lineage_cluster"] is True
    assert pair["human_identity_feature_used"] is False


def test_arbitrary_module_cannot_be_blessed_as_an_official_encoder(tmp_path: Path) -> None:
    torch.manual_seed(13)
    encoder = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )
    checkpoint = tmp_path / "official-test-checkpoint.pth"
    checkpoint.write_bytes(b"not official weights")
    with pytest.raises(ValueError, match="pinned official"):
        frozen_torchvision_encoder_provenance(
            encoder,
            architecture="synthetic_official_test_encoder",
            weights_identifier="SyntheticWeights.IMAGENET1K_V1",
            checkpoint_path=checkpoint,
            checkpoint_url="https://download.pytorch.org/models/synthetic-official-test.pth",
            model_card_url="https://pytorch.org/vision/stable/models/generated/synthetic.html",
            preprocessing={"crop_size": [16, 16], "mean": [0.5] * 3, "std": [0.25] * 3},
            output_dimension=4,
            torchvision_version="synthetic-test-version",
        )


def test_pinned_official_frozen_encoder_provenance_gradients_and_embeddings() -> None:
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
    if not checkpoint.is_file():
        pytest.skip("official pinned ResNet-18 checkpoint is not cached")
    wrapper = FrozenTorchvisionEncoder.from_official_weights(progress=False)
    provenance = wrapper.provenance
    assert provenance["schema_version"] == ENCODER_PROVENANCE_SCHEMA_VERSION
    assert provenance["official_pretrained_weights"] is True
    assert provenance["official_source_verified"] is True
    assert provenance["checkpoint_state_loaded_and_matched"] is True
    assert provenance["official_repository_url"] == "https://github.com/pytorch/vision"
    assert provenance["code_license_identifier"] == "BSD-3-Clause"
    assert (
        provenance["checkpoint_and_training_dataset_terms"][
            "torchvision_code_license_not_asserted_as_weight_or_imagenet_dataset_license"
        ]
        is True
    )
    assert provenance["checkpoint_sha256"] == "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
    wrapper.train(True)
    crops = torch.full((2, 3, 32, 24), 0.4)
    first = wrapper(crops)
    second = wrapper(crops)
    assert torch.equal(first, second)
    assert first.requires_grad is False
    assert wrapper.training is False
    assert wrapper.encoder.training is False
    assert all(
        parameter.requires_grad is False and parameter.grad is None for parameter in wrapper.encoder.parameters()
    )
    rows = wrapper.embedding_rows(("a", "b"), ("a" * 64, "b" * 64), crops)
    assert rows[0]["gradient_attached"] is False
    assert rows[0]["encoder_provenance_hash"] == provenance["provenance_hash"]
