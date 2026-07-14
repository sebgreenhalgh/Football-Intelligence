from __future__ import annotations

from football_intelligence.replay.anonymous_occlusion_state import ObservationNodeType
from football_intelligence.replay.short_window_candidate_graph import (
    CandidateObservation,
    ImageBBox,
    approach_to_occlusion_signals,
    detect_reciprocal_conflict,
    k_best_hypotheses,
    mine_local_candidates,
    one_to_one_assign,
)


def _obs(name: str, x: float) -> CandidateObservation:
    return CandidateObservation(name, 1, ImageBBox(x, 0, x + 10, 30), 0.8, appearance_similarity=0.9)


def test_candidate_generation_preserves_null_and_omits_appearance() -> None:
    rows = mine_local_candidates(_obs("source", 0), [_obs("target", 2)])

    assert any(row["candidate_node_type"] == ObservationNodeType.OCCLUDED_NULL.value for row in rows)
    assert all(row["candidate_generation_uses_appearance"] is False for row in rows)


def test_one_to_one_assignment_prevents_duplicate_detection_assignment() -> None:
    result = one_to_one_assign([_obs("s1", 0), _obs("s2", 50)], [_obs("t1", 1), _obs("t2", 51)])

    targets = [row["target_observation_id"] for row in result["rows"]]
    assert len(targets) == len(set(targets))
    assert all(row["one_to_one_enforced"] is True for row in result["rows"])


def test_reciprocal_conflict_and_approach_detection() -> None:
    conflict = detect_reciprocal_conflict(
        [
            {"source_observation_id": "s1", "target_observation_id": "t", "total_cost": 0.1},
            {"source_observation_id": "s2", "target_observation_id": "t", "total_cost": 0.12},
        ]
    )
    approach = approach_to_occlusion_signals(
        [_obs("s1", 0), _obs("s2", 5)],
        [_obs("t1", 3)],
        challenge_category_present=True,
    )

    assert conflict["conflict_detected"] is True
    assert approach["approaching_occlusion"] is True


def test_k_best_hypotheses_are_bounded_and_ordered() -> None:
    rows = k_best_hypotheses(_obs("source", 0), [_obs("t1", 1), _obs("t2", 4), _obs("t3", 20), _obs("t4", 40)], k=3)

    assert len(rows) == 3
    assert [row["hypothesis_rank"] for row in rows] == [1, 2, 3]
