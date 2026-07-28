from __future__ import annotations

import torch
import pytest

from football_intelligence.football_observation_reasoner.contracts import (
    CandidateState,
    EntityRole,
    KitState,
    ParticipationState,
    PitchState,
)
from football_intelligence.football_observation_reasoner.hierarchical_selection import (
    HierarchicalSoftConditioningNodeModel,
    MultitaskNodeMLP,
    PrimaryPopulationRoute,
    SymmetricPairMLP,
    assess_component_merge_risk,
    classify_pitch_from_confirmed_polygon,
    deterministic_complete_link_clusters,
    deterministic_correlation_clusters,
    deterministic_duplicate_connected_components,
    deterministic_hierarchical_selection,
    route_primary_population,
)


POLYGON = (
    {"x": 0.0, "y": 0.0},
    {"x": 10.0, "y": 0.0},
    {"x": 10.0, "y": 10.0},
    {"x": 0.0, "y": 10.0},
)
EXPANDED = (
    {"x": -10.0, "y": -10.0},
    {"x": 20.0, "y": -10.0},
    {"x": 20.0, "y": 20.0},
    {"x": -10.0, "y": 20.0},
)


def _candidate(identifier: str, score: float, x1: float, *, state: str | None = None, **extra: object) -> dict:
    return {
        "candidate_uuid": identifier,
        "candidate_state": state or CandidateState.CLEAN_INDEPENDENT_PERSON.value,
        "independent_person_probability": score,
        "localization_quality": score,
        "footpoint_quality": score,
        "perspective_plausibility": score,
        "provenance_quality": score,
        "role_confidence": score,
        "team_confidence": score,
        "kit_confidence": score,
        "merge_probability": 0.0,
        "truncation_risk": 0.0,
        "blur_risk": 0.0,
        "source_coordinates": {"x1": x1, "y1": 0.0, "x2": x1 + 2.0, "y2": 5.0},
        **extra,
    }


def _pair(left: str, right: str, duplicate: float, distinct: float, merged: float = 0.0) -> dict:
    return {
        "left_candidate_uuid": left,
        "right_candidate_uuid": right,
        "same_person_duplicate_probability": duplicate,
        "distinct_people_probability": distinct,
        "merged_contains_both_probability": merged,
    }


def test_original_confirmed_polygon_and_footpoint_uncertainty_are_authoritative() -> None:
    inside = classify_pitch_from_confirmed_polygon(
        POLYGON,
        {"x": 5.0, "y": 5.0},
        1.0,
        human_confirmed=True,
        expanded_search_polygon=EXPANDED,
    )
    outside_original_but_inside_expanded = classify_pitch_from_confirmed_polygon(
        POLYGON,
        {"x": 12.0, "y": 5.0},
        1.0,
        human_confirmed=True,
        expanded_search_polygon=EXPANDED,
    )
    boundary = classify_pitch_from_confirmed_polygon(
        POLYGON,
        {"x": 9.5, "y": 5.0},
        1.0,
        human_confirmed=True,
        expanded_search_polygon=EXPANDED,
    )
    assert inside.pitch_state is PitchState.ON_PITCH
    assert outside_original_but_inside_expanded.pitch_state is PitchState.OFF_PITCH
    assert boundary.pitch_state is PitchState.BOUNDARY_UNCERTAIN
    assert outside_original_but_inside_expanded.expanded_polygon_used_for_person_search is True
    assert outside_original_but_inside_expanded.expanded_polygon_used_for_classification is False
    assert outside_original_but_inside_expanded.learned_pitch_head_authoritative is False
    with pytest.raises(ValueError, match="human-confirmed"):
        classify_pitch_from_confirmed_polygon(POLYGON, {"x": 5.0, "y": 5.0}, 1.0, human_confirmed=False)


def test_primary_population_routing_preserves_off_pitch_active_and_never_infers_participation() -> None:
    active_throw_in = route_primary_population(
        pitch_state=PitchState.OFF_PITCH,
        role=EntityRole.OUTFIELD_PLAYER,
        participation=ParticipationState.ACTIVE_ON_PITCH,
        kit=KitState.MATCH_OUTFIELD_KIT,
    )
    warmup = route_primary_population(
        pitch_state=PitchState.OFF_PITCH,
        role=EntityRole.OUTFIELD_PLAYER,
        participation=ParticipationState.OFF_PITCH_SUBSTITUTE_OR_WARMING,
        kit=KitState.WARMUP_OR_BIB,
    )
    official = route_primary_population(
        pitch_state=PitchState.ON_PITCH,
        role=EntityRole.REFEREE,
        participation=ParticipationState.ACTIVE_ON_PITCH,
        kit=KitState.OFFICIAL_KIT,
    )
    unknown = route_primary_population(
        pitch_state=PitchState.ON_PITCH,
        role=EntityRole.OUTFIELD_PLAYER,
        participation=ParticipationState.UNKNOWN_PARTICIPATION,
    )
    assert active_throw_in.route is PrimaryPopulationRoute.BOUNDARY_OR_PARTICIPATION_UNRESOLVED
    assert warmup.route is PrimaryPopulationRoute.OUT_OF_SCOPE_PERSON
    assert official.route is PrimaryPopulationRoute.ACTIVE_OBSERVATION
    assert unknown.route is PrimaryPopulationRoute.BOUNDARY_OR_PARTICIPATION_UNRESOLVED
    assert all(not row.participation_inferred_from_polygon for row in (active_throw_in, warmup, official, unknown))
    assert active_throw_in.temporal_continuity_used is False


def test_n2_and_n3_have_separate_axes_without_certainty_identity_temporal_or_count_heads() -> None:
    features = torch.randn(4, 12)
    n2 = MultitaskNodeMLP(12, hidden_dim=16, seed=11)
    n3 = HierarchicalSoftConditioningNodeModel(12, hidden_dim=16, seed=12)
    for model in (n2, n3):
        output = model(features)
        assert output["candidate_state_logits"].shape == (4, 6)
        assert output["role_logits"].shape == (4, 6)
        assert output["team_logits"].shape == (4, 4)
        assert output["kit_logits"].shape == (4, 6)
        assert output["pitch_logits"].shape == (4, 4)
        assert output["participation_logits"].shape == (4, 4)
        assert output["footpoint_mean"].shape == (4, 2)
        assert not any("certainty" in name or "identity" in name or "temporal" in name for name in output)
        specification = model.specification()
        assert specification["human_certainty_head_present"] is False
        assert specification["identity_head_present"] is False
        assert specification["temporal_head_present"] is False
        assert specification["count_head_present"] is False
        assert specification["visual_encoder_present"] is False
        assert specification["learned_pitch_head_authoritative"] is False
        assert specification["participation_inferred_from_polygon"] is False

    loss = n3(features)["participation_logits"].square().mean()
    loss.backward()
    assert n3.role_head.weight.grad is not None and torch.count_nonzero(n3.role_head.weight.grad)
    assert n3.team_head.weight.grad is not None and torch.count_nonzero(n3.team_head.weight.grad)
    assert n3.kit_head.weight.grad is not None and torch.count_nonzero(n3.kit_head.weight.grad)
    assert n3.specification()["hard_argmax_conditioning_used"] is False


def test_symmetric_pair_mlp_is_exactly_invariant_to_pair_order() -> None:
    model = SymmetricPairMLP(5, 3, hidden_dim=11, seed=19)
    left = torch.randn(7, 5)
    right = torch.randn(7, 5)
    pair = torch.randn(7, 3)
    assert torch.equal(model(left, right, pair), model(right, left, pair))
    assert model.specification()["pair_order_invariant_by_construction"] is True


def test_complete_link_prevents_duplicate_chain_collapse() -> None:
    pairs = [
        _pair("a", "b", 0.95, 0.02),
        _pair("b", "c", 0.96, 0.02),
        _pair("a", "c", 0.05, 0.95),
    ]
    assert deterministic_complete_link_clusters(["c", "b", "a"], pairs) == (("a", "b"), ("c",))


def test_clustering_comparison_exposes_connected_chain_and_two_chain_safe_variants() -> None:
    pairs = [
        _pair("a", "b", 0.95, 0.02),
        _pair("b", "c", 0.96, 0.02),
        _pair("a", "c", 0.05, 0.95),
    ]
    assert deterministic_duplicate_connected_components(["a", "b", "c"], pairs) == (("a", "b", "c"),)
    assert deterministic_complete_link_clusters(["a", "b", "c"], pairs) == (("a", "b"), ("c",))
    # The transparent correlation objective chooses the slightly stronger
    # b--c edge first; the explicit a--c distinct edge then vetoes a chain.
    assert deterministic_correlation_clusters(["a", "b", "c"], pairs) == (("a",), ("b", "c"))


def test_h2_never_accepts_both_ends_of_a_duplicate_chain_edge() -> None:
    candidates = [_candidate("a", 0.8, 0.0), _candidate("b", 0.9, 10.0), _candidate("c", 0.7, 20.0)]
    pairs = [
        _pair("a", "b", 0.95, 0.02),
        _pair("b", "c", 0.96, 0.02),
        _pair("a", "c", 0.05, 0.95),
    ]
    result = deterministic_hierarchical_selection(candidates, pairs, variant="H2")
    assert result["accepted_candidate_uuids"] == ["b"]
    assert result["duplicate_pair_both_accepted_count"] == 0


def test_merge_routing_covers_candidate_pair_footpoint_appearance_and_scale_evidence() -> None:
    candidates = {
        "a": _candidate(
            "a",
            0.9,
            0.0,
            footpoint_hypothesis_count=2,
            appearance_incompatibility=0.8,
            abnormal_scale_probability=0.7,
        )
    }
    decision = assess_component_merge_risk(["a"], candidates, [_pair("a", "b", 0.0, 0.2, 0.9)])
    assert decision.routed is True
    assert set(decision.reasons) == {
        "ABNORMAL_SCALE_EVIDENCE",
        "HIGH_PAIRWISE_MERGED_RELATION_PROBABILITY",
        "INCOMPATIBLE_APPEARANCE_EVIDENCE",
        "MULTIPLE_FOOTPOINT_HYPOTHESES",
    }


def test_h0_h3_selection_is_deterministic_real_member_only_and_count_free() -> None:
    candidates = [
        _candidate("a", 0.9, 0.0),
        _candidate("b", 0.7, 20.0),
        _candidate(
            "g1",
            0.8,
            40.0,
            role="GOALKEEPER",
            team="TEAM_1",
            kit="MATCH_GOALKEEPER_KIT",
            participation="ACTIVE_ON_PITCH",
        ),
        _candidate(
            "g2",
            0.75,
            60.0,
            role="GOALKEEPER",
            team="TEAM_1",
            kit="MATCH_OUTFIELD_KIT",
            participation="ACTIVE_ON_PITCH",
        ),
        _candidate("merged", 1.0, 80.0, state=CandidateState.MERGED_MULTIPLE_PEOPLE.value),
    ]
    pairs = [
        _pair("a", "b", 0.95, 0.01),
        _pair("a", "g1", 0.0, 0.99),
        _pair("a", "g2", 0.0, 0.99),
        _pair("b", "g1", 0.0, 0.99),
        _pair("b", "g2", 0.0, 0.99),
        _pair("g1", "g2", 0.0, 0.99),
        _pair("merged", "a", 0.0, 0.3),
        _pair("merged", "b", 0.0, 0.3),
        _pair("merged", "g1", 0.0, 0.3),
        _pair("merged", "g2", 0.0, 0.3),
    ]
    h2 = deterministic_hierarchical_selection(candidates, pairs, variant="H2")
    repeated = deterministic_hierarchical_selection(list(reversed(candidates)), list(reversed(pairs)), variant="H2")
    h3 = deterministic_hierarchical_selection(candidates, pairs, variant="H3")
    assert h2 == repeated
    assert h2["accepted_candidate_uuids"] == ["a", "g1", "g2"]
    assert "merged" in h2["routed_candidate_uuids"]
    assert h3["accepted_candidate_uuids"] == h2["accepted_candidate_uuids"]
    assert h3["semantic_warnings_changed_selection"] is False
    assert {row["warning"] for row in h3["semantic_warnings"]} == {
        "GOALKEEPER_WITH_OUTFIELD_KIT",
        "MULTIPLE_ACTIVE_GOALKEEPERS_FOR_TEAM_WARNING_ONLY",
    }
    selected_coordinates = {
        row["candidate_uuid"]: row["source_coordinates"]
        for row in h2["decision_ledger"]
        if row["candidate_uuid"] in h2["accepted_candidate_uuids"]
    }
    source_coordinates = {row["candidate_uuid"]: row["source_coordinates"] for row in candidates}
    assert selected_coordinates == {identifier: source_coordinates[identifier] for identifier in selected_coordinates}
    assert h2["coordinate_averaging_performed"] is False
    assert h2["merged_candidate_clean_acceptance_count"] == 0
    assert h2["duplicate_pair_both_accepted_count"] == 0
    assert h2["scene_count_prior_used"] is False
    assert h2["exact_22_forcing_performed"] is False
    assert h2["goalkeeper_count_forcing_performed"] is False
    assert h2["identity_tracking_performed"] is False
    assert h2["temporal_predictions_created"] is False


def test_h1_routes_merge_risk_while_h0_remains_an_auditable_baseline() -> None:
    candidates = [_candidate("risk", 0.9, 0.0, merge_probability=0.8)]
    h0 = deterministic_hierarchical_selection(candidates, [], variant="H0")
    h1 = deterministic_hierarchical_selection(candidates, [], variant="H1")
    assert h0["accepted_candidate_uuids"] == ["risk"]
    assert h1["accepted_candidate_uuids"] == []
    assert h1["routed_candidate_uuids"] == ["risk"]
