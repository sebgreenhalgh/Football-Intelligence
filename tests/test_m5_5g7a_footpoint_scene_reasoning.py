from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from football_intelligence.football_observation_reasoner.contracts import CandidateState
from football_intelligence.football_observation_reasoner.models import (
    LightweightGraphReasoner,
    SoftSceneEnergyRanker,
    graph_model_specification,
    masked_heteroscedastic_footpoint_loss,
    masked_scene_ranking_loss,
)


REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "scripts" / "build_m5_5g7a_football_observation_reasoner.py"


def load_builder():
    specification = importlib.util.spec_from_file_location("m5_5g7a_footpoint_scene_test", BUILDER)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_masked_heteroscedastic_footpoint_loss_ignores_unavailable_rows() -> None:
    mean = torch.tensor([[0.1, -0.1], [9.0, 9.0]], requires_grad=True)
    log_variance = torch.tensor([[-1.0, -0.5], [5.0, 5.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 0.0], [999.0, 999.0]])
    result = masked_heteroscedastic_footpoint_loss(
        mean,
        log_variance,
        targets,
        torch.tensor([True, False]),
        huber_delta=0.25,
    )
    assert result.labelled_count == 1
    assert torch.isfinite(result.total)
    result.total.backward()
    assert torch.count_nonzero(mean.grad[0]) == 2
    assert torch.equal(mean.grad[1], torch.zeros(2))
    assert torch.equal(log_variance.grad[1], torch.zeros(2))


def test_graph_reasoner_exposes_mean_and_heteroscedastic_footpoint_heads() -> None:
    model = LightweightGraphReasoner(3, 2, hidden_dim=8, seed=23)
    output = model(
        torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.3, 0.2]]),
        torch.tensor([[0], [1]], dtype=torch.long),
        torch.tensor([[0.2, 0.4]]),
    )
    assert output["footpoint_mean"].shape == (2, 2)
    assert output["footpoint_log_variance"].shape == (2, 2)
    specification = graph_model_specification(model)
    assert specification["footpoint_head"]["mean_coordinates"] == 2
    assert specification["footpoint_head"]["heteroscedastic_log_variance_coordinates"] == 2
    assert specification["hard_count_head_present"] is False


def test_soft_scene_ranking_loss_is_structured_but_does_not_make_decisions() -> None:
    preferred = torch.tensor([0.0, 1.0], requires_grad=True)
    reversed_order = torch.tensor([1.0, 0.0])
    clean = torch.tensor([True, False])
    available = torch.tensor([True, True])
    scenes = torch.tensor([0, 0])
    preferred_loss, preferred_pairs = masked_scene_ranking_loss(
        preferred,
        clean,
        available,
        scenes,
        margin=0.2,
    )
    reversed_loss, reversed_pairs = masked_scene_ranking_loss(
        reversed_order,
        clean,
        available,
        scenes,
        margin=0.2,
    )
    assert preferred_pairs == reversed_pairs == 1
    assert preferred_loss < reversed_loss
    preferred_loss.backward()
    assert preferred.grad is not None
    ranker = SoftSceneEnergyRanker(3, hidden_dim=4, seed=5)
    assert ranker(torch.ones(2, 3)).shape == (2,)
    assert not any("count" in name or "accept" in name for name, _parameter in ranker.named_parameters())


def test_builder_normalizes_footpoints_without_exposing_targets_as_features() -> None:
    builder = load_builder()
    nodes = [
        {
            "example_uuid": "node-1",
            "visible_box": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 60.0},
            "footpoint_target_source_pixels": {"x": 24.0, "y": 64.0},
            "label_availability_mask": {"footpoint": True},
        },
        {
            "example_uuid": "node-2",
            "visible_box": {"x1": 30.0, "y1": 10.0, "x2": 50.0, "y2": 50.0},
            "footpoint_target_source_pixels": None,
            "label_availability_mask": {"footpoint": False},
        },
    ]
    targets, mask = builder._footpoint_target_arrays(nodes)
    assert np.allclose(targets[0], [0.1, 0.1])
    assert mask.tolist() == [True, False]
    metrics, predictions = builder._footpoint_evaluation(
        nodes,
        np.asarray([[0.1, 0.1], [0.0, 0.0]]),
        np.zeros((2, 2)),
        targets,
        mask,
    )
    assert metrics["denominator"] == 1
    assert metrics["mean_error_source_pixels"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["target_used_as_runtime_feature"] is False
    assert predictions["node-1"]["mean_source_pixels"] == {"x": 24.0, "y": 64.0}


def test_scene_audit_applies_soft_ranking_without_changing_hard_acceptance() -> None:
    builder = load_builder()
    classes = tuple(value.value for value in CandidateState)
    clean = classes.index(CandidateState.CLEAN_INDEPENDENT_PERSON.value)
    duplicate = classes.index(CandidateState.DUPLICATE_OF_PERSON.value)
    first_vector = np.full(len(classes), 0.01)
    first_vector[clean] = 0.95
    first_vector /= first_vector.sum()
    second_vector = np.full(len(classes), 0.01)
    second_vector[duplicate] = 0.95
    second_vector /= second_vector.sum()
    nodes = [
        {"source_group_id": "group", "candidate_uuid": "a", "example_uuid": "example-a"},
        {"source_group_id": "group", "candidate_uuid": "b", "example_uuid": "example-b"},
    ]
    graph = {
        "predictions_by_head": {
            "candidate_state": {
                "example-a": CandidateState.CLEAN_INDEPENDENT_PERSON.value,
                "example-b": CandidateState.DUPLICATE_OF_PERSON.value,
            },
            "role": {"example-a": "OUTFIELD_PLAYER", "example-b": "OUTFIELD_PLAYER"},
            "pitch": {"example-a": "ON_PITCH", "example-b": "ON_PITCH"},
        },
        "candidate_probabilities": {
            "example-a": first_vector.tolist(),
            "example-b": second_vector.tolist(),
        },
    }
    soft = {
        "energy_by_example": {"example-a": -1.0, "example-b": 1.0},
        "metrics": {"pairwise_ranking_accuracy": 1.0},
        "loss_kind": "WITHIN_SCENE_CLEAN_VS_NON_CLEAN_SOFTPLUS_PAIRWISE_RANKING",
    }
    scenes = [
        {
            "scene_uuid": "scene",
            "source_group_id": "group",
            "candidate_uuids": ["a", "b"],
            "evaluator_targets": {"visible_person_count": 1},
        }
    ]
    audit = builder._scene_warning_only_audit(nodes, scenes, graph, soft)
    assert audit["hard_prediction_changes"] == 0
    assert audit["scene_cardinality_loss_used"] is False
    assert audit["soft_scene_ranking_loss_used"] is True
    assert audit["scenes"][0]["soft_structured_ranking"][0]["example_uuid"] == "example-a"
    energy = audit["scenes"][0]["energy"]
    assert energy["accepted_candidate_uuids_before"] == energy["accepted_candidate_uuids_after"]
    assert audit["count_warning_usefulness"]["evaluator_counts_used_as_runtime_inputs"] is False
