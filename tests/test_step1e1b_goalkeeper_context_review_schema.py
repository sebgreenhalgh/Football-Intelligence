from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_schema import (  # noqa: E402
    corrected_belief_for_decision,
    reviewed_decision_row,
    validate_reviewed_decision_payload,
)


def candidate() -> dict:
    return {
        "step1e1_review_candidate_id": "e1_review_1",
        "visible_person_base_id": "base_1",
        "frame_sequence": 59,
        "e1_goalkeeper_context_belief": "unknown_goalkeeper_context",
        "review_reason_tags": ["gold8_goalkeeper_proxy_match"],
    }


def test_allowed_decisions_map_to_corrected_goalkeeper_context_beliefs() -> None:
    row = reviewed_decision_row(candidate(), "correct_to_goalkeeper_like_team_1_context")
    assert row["human_corrected_goalkeeper_context_belief"] == "goalkeeper_like_team_1_context"
    assert row["approve_any_goalkeeper_slot_use"] is False
    assert row["approve_any_identity_tracking"] is False
    assert row["approve_any_metric_use"] is False
    assert row["production_ready"] is False
    corrected = corrected_belief_for_decision("accept_e1_belief", "outfield_player_like_not_goalkeeper")
    assert corrected == "outfield_player_like_not_goalkeeper"


def test_invalid_decision_rejected() -> None:
    with pytest.raises(ValueError):
        reviewed_decision_row(candidate(), "assign_goalkeeper_slot")


def test_reviewed_decision_requires_ids() -> None:
    bad = candidate()
    bad["visible_person_base_id"] = ""
    with pytest.raises(ValueError):
        reviewed_decision_row(bad, "accept_e1_belief")


def test_validation_blocks_slot_identity_metric_approval() -> None:
    review = reviewed_decision_row(candidate(), "accept_e1_belief")
    review["approve_any_goalkeeper_slot_use"] = True
    validation, usable = validate_reviewed_decision_payload(
        {"rows": [candidate()]},
        {"rows": [review]},
        reviewed_decisions_loaded=True,
    )
    assert usable == []
    assert validation["reviewed_decisions_valid"] is False
    assert any(error["error"] == "forbidden_approval_flag_true_or_missing" for error in validation["validation_errors"])
