from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_review_schema import (  # noqa: E402
    corrected_belief_for_decision,
    reviewed_decision_row,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def candidate() -> dict:
    return {
        "c2b_review_candidate_id": "candidate_1",
        "visible_person_base_id": "base_1",
        "frame_sequence": 59,
        "c1c_seed_team_colour_belief": "unknown_ambiguous_colour",
        "c2_stable_colour_belief": "team_1_outfield_colour_like",
    }


def test_allowed_decisions_map_to_corrected_beliefs() -> None:
    assert (
        corrected_belief_for_decision("accept_c2_stable_colour", "team_1_outfield_colour_like")
        == "team_1_outfield_colour_like"
    )
    assert (
        corrected_belief_for_decision("reject_to_unknown_ambiguous_colour", "team_1_outfield_colour_like")
        == "unknown_ambiguous_colour"
    )
    assert (
        corrected_belief_for_decision("reject_to_team_2_outfield_colour_like", "team_1_outfield_colour_like")
        == "team_2_outfield_colour_like"
    )


def test_invalid_decision_is_rejected() -> None:
    with pytest.raises(ValueError):
        corrected_belief_for_decision("made_up_decision", "team_1_outfield_colour_like")


def test_reviewed_decision_row_preserves_visual_only_fields() -> None:
    row = reviewed_decision_row(candidate(), "accept_c2_stable_colour")
    assert row["human_confirmed"] is True
    assert row["visual_only_warning"] == VISUAL_ONLY_WARNING
    assert row["do_not_use_for_metrics"] is True
    assert row["production_ready"] is False
