from __future__ import annotations

from collections import Counter

import pytest
import torch

from football_intelligence.football_observation_reasoner.g7b_training import (
    HierarchicalSoftConditioningNodeModel,
    InterpretablePairLinear,
    MultitaskNodeMLP,
    SymmetricPairMLP,
    TwoStagePairClassifier,
    abstaining_predictions,
    apply_temperature,
    class_balanced_weights,
    classification_metrics,
    fit_abstention_threshold,
    fit_head_calibration,
    fit_temperature_scaling,
    grouped_inner_fold_assignments,
    macro_f1_score,
    masked_class_balanced_multitask_loss,
    multiclass_brier_score,
    nested_grouped_oof_splits,
    pair_model_logits,
    symmetric_endpoint_features,
    train_masked_multitask_node_model,
    train_masked_pair_model,
    validate_grouped_outer_folds,
)


def grouped_rows() -> tuple[list[str], list[int]]:
    groups = [f"source-{index}" for index in range(10) for _ in range(2)]
    folds = [index % 5 for index in range(10) for _ in range(2)]
    return groups, folds


def test_exact_outer_folds_are_validated_and_nested_splits_are_group_disjoint() -> None:
    groups, folds = grouped_rows()
    validation = validate_grouped_outer_folds(groups, folds)
    assert validation["passed"] is True
    assert validation["fold_row_counts"] == {0: 4, 1: 4, 2: 4, 3: 4, 4: 4}
    assert validation["source_group_cross_fold_count"] == 0

    first = nested_grouped_oof_splits(groups, folds, inner_fold_count=2)
    second = nested_grouped_oof_splits(groups, folds, inner_fold_count=2)
    assert first == second
    assert len(first) == 10
    for outer_fold in range(5):
        outer = [split for split in first if split.outer_fold == outer_fold]
        outer_test = set(outer[0].outer_test_indices)
        calibration_counts: Counter[int] = Counter()
        for split in outer:
            training = set(split.training_indices)
            calibration = set(split.calibration_indices)
            assert not training & calibration
            assert not (training | calibration) & outer_test
            assert set(split.training_group_ids).isdisjoint(split.calibration_group_ids)
            assert set(split.outer_test_group_ids).isdisjoint(
                set(split.training_group_ids) | set(split.calibration_group_ids)
            )
            calibration_counts.update(calibration)
        expected_inner_rows = set(range(len(groups))) - outer_test
        assert set(calibration_counts) == expected_inner_rows
        assert set(calibration_counts.values()) == {1}


def test_grouped_helpers_reject_cross_fold_groups_and_never_use_labels() -> None:
    groups, folds = grouped_rows()
    folds[1] = 1
    with pytest.raises(ValueError, match="source groups span outer folds"):
        validate_grouped_outer_folds(groups, folds)

    valid_groups, valid_folds = grouped_rows()
    assignment = grouped_inner_fold_assignments(valid_groups, valid_folds, 0, inner_fold_count=2)
    assert {value for value in assignment if value >= 0} == {0, 1}
    assert all(value == -1 for value, outer in zip(assignment, valid_folds, strict=True) if outer == 0)


def test_masked_class_balancing_uses_only_available_rows() -> None:
    targets = torch.tensor([0, 0, 1, 999])
    availability = torch.tensor([True, True, True, False])
    weights, counts = class_balanced_weights(targets, availability, 3)
    assert counts == (2, 1, 0)
    assert weights[1] > weights[0]
    assert torch.isfinite(weights).all()

    logits = torch.zeros((4, 6), requires_grad=True)
    result = masked_class_balanced_multitask_loss(
        {"role": logits},
        {"role": targets},
        {"role": availability},
    )
    assert result.labelled_counts == {"role": 3}
    assert result.class_counts["role"] == (2, 1, 0, 0, 0, 0)
    result.total.backward()
    assert torch.equal(logits.grad[3], torch.zeros(6))
    with pytest.raises(ValueError, match="forbidden"):
        masked_class_balanced_multitask_loss(
            {"certainty": torch.zeros((1, 2))},
            {"certainty": torch.zeros(1, dtype=torch.long)},
            {"certainty": torch.ones(1, dtype=torch.bool)},
        )


def node_targets(row_count: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    targets = {
        "candidate_state": torch.tensor([index % 2 for index in range(row_count)]),
        "role": torch.tensor([index % 3 for index in range(row_count)]),
        "team": torch.tensor([index % 4 for index in range(row_count)]),
        "kit": torch.tensor([index % 3 for index in range(row_count)]),
        "pitch": torch.tensor([index % 3 for index in range(row_count)]),
        "participation": torch.tensor([index % 4 for index in range(row_count)]),
    }
    availability = {name: torch.ones(row_count, dtype=torch.bool) for name in targets}
    availability["team"][-1] = False
    targets["team"][-1] = 999
    return targets, availability


def test_n2_training_is_deterministic_and_cached_input_has_no_gradient() -> None:
    features = torch.arange(48, dtype=torch.float32).reshape(8, 6).div(48).requires_grad_()
    targets, availability = node_targets(8)
    first = MultitaskNodeMLP(6, hidden_dim=10, seed=11)
    second = MultitaskNodeMLP(6, hidden_dim=10, seed=11)
    first_receipt = train_masked_multitask_node_model(
        first,
        features,
        targets,
        availability,
        epochs=8,
        learning_rate=0.01,
        seed=91,
    )
    second_receipt = train_masked_multitask_node_model(
        second,
        features,
        targets,
        availability,
        epochs=8,
        learning_rate=0.01,
        seed=91,
    )
    assert first_receipt.state_dict_sha256 == second_receipt.state_dict_sha256
    assert first_receipt.final_loss == second_receipt.final_loss
    assert first_receipt.variant == "N2"
    assert first_receipt.input_features_detached is True
    assert first_receipt.visual_encoder_trained is False
    assert first_receipt.human_certainty_head_present is False
    assert features.grad is None
    assert first.specification()["visual_encoder_present"] is False
    assert first.specification()["human_certainty_head_present"] is False


def test_n3_soft_hierarchy_has_no_certainty_and_participation_does_not_use_pitch() -> None:
    model = HierarchicalSoftConditioningNodeModel(5, hidden_dim=8, seed=17).eval()
    features = torch.arange(20, dtype=torch.float32).reshape(4, 5).div(20)
    before = model(features)
    assert "certainty_logits" not in before
    assert model.specification()["hard_argmax_conditioning_used"] is False
    with torch.no_grad():
        model.pitch_head.weight.fill_(1000.0)
        model.pitch_head.bias.fill_(-1000.0)
    after = model(features)
    assert torch.equal(before["participation_logits"], after["participation_logits"])
    assert not torch.equal(before["pitch_logits"], after["pitch_logits"])


def test_symmetric_pair_models_are_order_invariant_and_p3_is_normalized() -> None:
    left = torch.tensor([[0.1, 0.3], [0.2, 0.5], [0.7, 0.9]])
    right = torch.tensor([[0.4, 0.8], [0.6, 0.1], [0.3, 0.2]])
    pair = torch.tensor([[0.2], [0.4], [0.6]])
    symmetric = symmetric_endpoint_features(left, right, pair)
    assert torch.equal(symmetric, symmetric_endpoint_features(right, left, pair))

    p1 = InterpretablePairLinear([f"feature_{index}" for index in range(symmetric.shape[1])], seed=3)
    p2 = SymmetricPairMLP(2, 1, hidden_dim=5, seed=4)
    p3 = TwoStagePairClassifier(symmetric.shape[1], hidden_dim=5, seed=5)
    assert p1(symmetric).shape == (3, 4)
    assert torch.equal(
        pair_model_logits(p2, pair, left_features=left, right_features=right),
        pair_model_logits(p2, pair, left_features=right, right_features=left),
    )
    p3_log_probabilities = p3(symmetric)
    assert p3_log_probabilities.shape == (3, 4)
    assert torch.allclose(p3_log_probabilities.exp().sum(dim=1), torch.ones(3))
    assert p1.specification()["identity_features_present"] is False
    assert p3.specification()["stage_1"] == "SAME_PERSON_DUPLICATE_VERSUS_NOT_DUPLICATE"


def test_pair_training_is_masked_class_balanced_and_deterministic() -> None:
    features = torch.arange(32, dtype=torch.float32).reshape(8, 4).div(32)
    targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 999])
    availability = torch.tensor([True, True, True, True, True, True, True, False])
    first = InterpretablePairLinear([f"f{index}" for index in range(4)], seed=7)
    second = InterpretablePairLinear([f"f{index}" for index in range(4)], seed=7)
    first_receipt = train_masked_pair_model(
        first,
        features,
        targets,
        availability,
        epochs=6,
        learning_rate=0.01,
        seed=99,
    )
    second_receipt = train_masked_pair_model(
        second,
        features,
        targets,
        availability,
        epochs=6,
        learning_rate=0.01,
        seed=99,
    )
    assert first_receipt["labelled_count"] == 7
    assert first_receipt["class_counts"] == [2, 2, 2, 1]
    assert first_receipt["state_dict_sha256"] == second_receipt["state_dict_sha256"]


def test_metrics_temperature_and_independent_abstention() -> None:
    probabilities = torch.tensor([[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]])
    targets = torch.tensor([0, 1, 1, 1])
    predictions = probabilities.argmax(dim=1)
    assert macro_f1_score(predictions, targets, 2) == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert multiclass_brier_score(probabilities, targets) == pytest.approx(0.285)
    metrics = classification_metrics(probabilities, targets, bin_count=5)
    assert metrics["denominator"] == 4
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert metrics["expected_calibration_error"] is not None
    assert metrics["selective_risk"][-1]["risk"] == pytest.approx(0.25)

    threshold = fit_abstention_threshold(probabilities, targets, maximum_risk=0.0)
    assert threshold.threshold == pytest.approx(0.8)
    assert threshold.retained_count == 2
    assert abstaining_predictions(probabilities, threshold.threshold).tolist() == [0, -1, -1, 1]

    logits = torch.tensor([[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0]])
    difficult_targets = torch.tensor([0, 1, 1, 0])
    scaling = fit_temperature_scaling(logits, difficult_targets)
    assert scaling.temperature > 1.0
    assert scaling.nll_after is not None and scaling.nll_before is not None
    assert scaling.nll_after <= scaling.nll_before
    assert torch.allclose(apply_temperature(logits, scaling.temperature).sum(dim=1), torch.ones(4))

    calibration = fit_head_calibration(
        "role",
        logits,
        difficult_targets,
        torch.ones(4, dtype=torch.bool),
        calibration_indices=(0, 1, 2, 3),
        maximum_risk=0.5,
    )
    assert calibration.head_name == "role"
    assert calibration.calibration_count == 4
    assert calibration.fitted_on_grouped_inner_holdout is True
