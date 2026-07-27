from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from football_intelligence.football_observation_reasoner import (
    CandidateState,
    EntityRole,
    FootballObservationAxes,
    FrozenVisualEncoder,
    KitState,
    LabelAvailabilityMask,
    LightweightGraphReasoner,
    ParticipationState,
    PitchState,
    ReasonerCandidateContract,
    SceneCandidateAssessment,
    TeamAffiliation,
    TemporalDiagnosticEvidence,
    TemporalDiagnosticName,
    TemporalDiagnosticSignal,
    assert_visual_encoder_frozen,
    goalkeeper_team_key,
    graph_model_specification,
    is_off_pitch_warmup_player,
    masked_multitask_cross_entropy,
    ontology_contract,
    warning_only_scene_energy,
)

SOURCE_HASH = "a" * 64


def axes(
    *,
    role: EntityRole = EntityRole.OUTFIELD_PLAYER,
    team: TeamAffiliation = TeamAffiliation.TEAM_1,
    kit: KitState = KitState.MATCH_OUTFIELD_KIT,
    pitch: PitchState = PitchState.ON_PITCH,
    participation: ParticipationState = ParticipationState.ACTIVE_ON_PITCH,
    candidate_state: CandidateState = CandidateState.CLEAN_INDEPENDENT_PERSON,
) -> FootballObservationAxes:
    return FootballObservationAxes(
        role=role,
        team=team,
        kit=kit,
        pitch=pitch,
        participation=participation,
        candidate_state=candidate_state,
    )


def assessment(
    identifier: str,
    semantic_axes: FootballObservationAxes,
    *,
    accepted: bool = True,
) -> SceneCandidateAssessment:
    return SceneCandidateAssessment(
        candidate_uuid=identifier,
        axes=semantic_axes,
        accepted_as_independent_person=accepted,
        confidence=0.9,
    )


def test_ontology_axes_are_exact_and_independent() -> None:
    contract = ontology_contract()
    assert contract["schema_version"] == "football_intelligence.m5_5g7a.ontology_scene_contract.v1"
    assert contract["entity_roles"] == [
        "OUTFIELD_PLAYER",
        "GOALKEEPER",
        "REFEREE",
        "OTHER_MATCH_OFFICIAL",
        "STAFF_OR_SPECTATOR",
        "UNKNOWN_ROLE",
    ]
    assert contract["team_affiliations"] == ["TEAM_1", "TEAM_2", "NO_TEAM", "UNKNOWN_TEAM"]
    assert contract["axes_must_remain_separate"] is True
    assert contract["exact_visible_person_count_forcing_forbidden"] is True
    assert {
        "role": "GOALKEEPER",
        "team": "TEAM_2",
        "kit": "WARMUP_OR_BIB",
        "pitch": "OFF_PITCH",
    } in contract["critical_valid_combinations"]
    row = axes(role=EntityRole.GOALKEEPER, team=TeamAffiliation.TEAM_2)
    assert row.model_dump()["role"] is EntityRole.GOALKEEPER
    assert row.model_dump()["team"] is TeamAffiliation.TEAM_2
    assert "team_2_goalkeeper" not in str(row.model_dump(mode="json")).lower()


def test_two_team_goalkeepers_are_distinct_without_a_generic_team_state() -> None:
    team_1 = axes(
        role=EntityRole.GOALKEEPER,
        team=TeamAffiliation.TEAM_1,
        kit=KitState.MATCH_GOALKEEPER_KIT,
    )
    team_2 = axes(
        role=EntityRole.GOALKEEPER,
        team=TeamAffiliation.TEAM_2,
        kit=KitState.MATCH_GOALKEEPER_KIT,
    )
    assert goalkeeper_team_key(team_1) == (TeamAffiliation.TEAM_1, EntityRole.GOALKEEPER)
    assert goalkeeper_team_key(team_2) == (TeamAffiliation.TEAM_2, EntityRole.GOALKEEPER)
    assert goalkeeper_team_key(team_1) != goalkeeper_team_key(team_2)
    result = warning_only_scene_energy([assessment("gk-1", team_1), assessment("gk-2", team_2)])
    assert result.goalkeeper_team_conflict_warning is False
    assert result.accepted_candidate_uuids_after == ("gk-1", "gk-2")


@pytest.mark.parametrize("role", [EntityRole.OUTFIELD_PLAYER, EntityRole.GOALKEEPER])
@pytest.mark.parametrize("team", [TeamAffiliation.TEAM_1, TeamAffiliation.TEAM_2])
def test_off_pitch_warmup_players_remain_players(role: EntityRole, team: TeamAffiliation) -> None:
    reserve = axes(
        role=role,
        team=team,
        kit=KitState.WARMUP_OR_BIB,
        pitch=PitchState.OFF_PITCH,
        participation=ParticipationState.OFF_PITCH_SUBSTITUTE_OR_WARMING,
    )
    assert is_off_pitch_warmup_player(reserve) is True
    assert reserve.role is role
    assert reserve.candidate_state is CandidateState.CLEAN_INDEPENDENT_PERSON


def test_strict_candidate_and_temporal_contracts_reject_identity_or_decisive_temporal_fields() -> None:
    diagnostic = TemporalDiagnosticEvidence(
        candidate_uuid="candidate-1",
        reference_frame_offsets=(-1, 1),
        signals=(
            TemporalDiagnosticSignal(
                name=TemporalDiagnosticName.CANDIDATE_VISIBLE_BEFORE,
                value=True,
                available=True,
            ),
        ),
    )
    candidate = ReasonerCandidateContract(
        example_uuid="example-1",
        source_group_id="source-group-1",
        source_frame_sha256=SOURCE_HASH,
        frame_index=10,
        candidate_uuid="candidate-1",
        target=axes(),
        label_availability=LabelAvailabilityMask(candidate_state=True),
        temporal_diagnostics=diagnostic,
    )
    assert candidate.temporal_diagnostics is not None
    assert candidate.temporal_diagnostics.diagnostic_only is True
    assert candidate.temporal_diagnostics.used_for_acceptance_decision is False
    with pytest.raises(ValidationError):
        TemporalDiagnosticEvidence(
            candidate_uuid="candidate-1",
            used_for_acceptance_decision=True,
        )
    with pytest.raises(ValidationError):
        ReasonerCandidateContract(
            example_uuid="example-1",
            source_group_id="source-group-1",
            source_frame_sha256=SOURCE_HASH,
            frame_index=10,
            candidate_uuid="candidate-1",
            track_id="forbidden",
        )


def test_frozen_visual_encoder_never_receives_gradients_or_training_state() -> None:
    encoder = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3), nn.ReLU())
    wrapper = FrozenVisualEncoder(encoder)
    head = nn.Linear(3, 2)
    wrapper.train(True)
    assert wrapper.training is False
    assert wrapper.encoder.training is False
    encoded = wrapper(torch.ones(2, 4))
    loss = head(encoded).sum()
    loss.backward()
    assert_visual_encoder_frozen(wrapper.encoder)
    assert all(parameter.grad is None for parameter in wrapper.encoder.parameters())
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_masked_multitask_loss_ignores_unavailable_targets() -> None:
    candidate_logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], requires_grad=True)
    role_logits = torch.tensor([[2.0, 0.0], [2.0, 0.0]], requires_grad=True)
    result = masked_multitask_cross_entropy(
        {"candidate_state": candidate_logits, "role": role_logits},
        {"candidate_state": torch.tensor([0, 1]), "role": torch.tensor([0, 999])},
        {"candidate_state": torch.tensor([True, True]), "role": torch.tensor([True, False])},
        loss_weights={"candidate_state": 2.0, "role": 1.0},
        class_weights_by_head={
            "candidate_state": torch.tensor([1.0, 3.0]),
            "role": torch.tensor([1.0, 2.0]),
        },
    )
    assert result.labelled_counts == {"candidate_state": 2, "role": 1}
    assert torch.isfinite(result.total)
    result.total.backward()
    assert candidate_logits.grad is not None
    assert role_logits.grad is not None
    assert torch.equal(role_logits.grad[1], torch.zeros_like(role_logits.grad[1]))


def test_graph_reasoner_is_deterministic_and_jointly_outputs_node_and_pair_heads() -> None:
    node_features = torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.4, 0.3], [0.9, 0.1, 0.2]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_features = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    first_model = LightweightGraphReasoner(3, 2, hidden_dim=8, seed=19)
    second_model = LightweightGraphReasoner(3, 2, hidden_dim=8, seed=19)
    first = first_model(node_features, edge_index, edge_features)
    second = second_model(node_features, edge_index, edge_features)
    assert first.keys() == second.keys()
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert first["candidate_state_logits"].shape == (3, 6)
    assert first["role_logits"].shape == (3, 6)
    assert first["team_logits"].shape == (3, 4)
    assert first["pair_relation_logits"].shape == (2, 4)
    assert not any("identity" in key or "temporal" in key for key in first)
    specification = graph_model_specification(first_model)
    assert specification["identity_head_present"] is False
    assert specification["temporal_acceptance_head_present"] is False
    assert specification["hard_count_head_present"] is False


def test_graph_node_outputs_are_invariant_to_edge_row_order() -> None:
    model = LightweightGraphReasoner(2, 1, hidden_dim=6, seed=5)
    nodes = torch.tensor([[0.2, 0.4], [0.4, 0.6], [0.6, 0.8]])
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    features = torch.tensor([[0.3], [0.7]])
    forward = model(nodes, edges, features)
    reverse = model(nodes, edges[:, [1, 0]], features[[1, 0]])
    for name in ("candidate_state_logits", "role_logits", "team_logits", "kit_logits", "pitch_logits"):
        assert torch.equal(forward[name], reverse[name])
    assert torch.equal(forward["pair_relation_logits"], reverse["pair_relation_logits"][[1, 0]])


def test_warning_only_scene_energy_never_invents_or_deletes_for_count_prior() -> None:
    candidates = [assessment(f"person-{index}", axes()) for index in range(3)]
    result = warning_only_scene_energy(candidates, expected_visible_person_count=22)
    assert result.count_under_resolution_warning is True
    assert result.accepted_candidate_uuids_before == result.accepted_candidate_uuids_after
    assert result.invented_candidate_uuids == ()
    assert result.deleted_clean_candidate_uuids == ()
    assert result.hard_cardinality_forcing_performed is False
    assert result.exact_visible_person_count_forcing_performed is False
    assert result.exact_22_forcing_performed is False


def test_goalkeeper_conflict_is_warning_only_and_reserve_goalkeeper_is_exempt() -> None:
    active = axes(
        role=EntityRole.GOALKEEPER,
        team=TeamAffiliation.TEAM_1,
        kit=KitState.MATCH_GOALKEEPER_KIT,
    )
    reserve = axes(
        role=EntityRole.GOALKEEPER,
        team=TeamAffiliation.TEAM_1,
        kit=KitState.WARMUP_OR_BIB,
        pitch=PitchState.OFF_PITCH,
        participation=ParticipationState.OFF_PITCH_SUBSTITUTE_OR_WARMING,
    )
    safe = warning_only_scene_energy([assessment("active", active), assessment("reserve", reserve)])
    assert safe.goalkeeper_team_conflict_warning is False
    conflict = warning_only_scene_energy(
        [assessment("active-a", active), assessment("active-b", active), assessment("reserve", reserve)]
    )
    assert conflict.goalkeeper_team_conflict_warning is True
    assert conflict.accepted_candidate_uuids_after == ("active-a", "active-b", "reserve")
    assert conflict.hard_goalkeeper_deletions == ()
    assert conflict.exactly_one_goalkeeper_per_team_forcing_performed is False
    assert conflict.exactly_two_goalkeeper_forcing_performed is False
