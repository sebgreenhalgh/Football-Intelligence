from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.official_context_beliefs import (  # noqa: E402
    ALLOWED_OFFICIAL_CONTEXT_BELIEFS,
)
from football_intelligence.step1_visual_reconstruction.official_context_review_schema import (  # noqa: E402
    ALLOWED_HUMAN_OFFICIAL_CONTEXT_DECISIONS,
    corrected_belief_for_decision,
    reviewed_decision_row,
    validate_reviewed_decision_payload,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def candidate() -> dict:
    return {
        "step1d1_review_candidate_id": "d1_review_1",
        "visible_person_base_id": "base_1",
        "frame_sequence": 59,
        "official_context_belief": "official_referee_like",
        "official_context_belief_confidence": 0.82,
        "official_context_belief_state": "review_required",
        "review_reason_tags": ["gold8_official_proxy_match"],
    }


def test_allowed_decisions_map_to_allowed_corrected_beliefs() -> None:
    for decision in ALLOWED_HUMAN_OFFICIAL_CONTEXT_DECISIONS:
        corrected = corrected_belief_for_decision(decision, "official_referee_like")
        assert corrected in ALLOWED_OFFICIAL_CONTEXT_BELIEFS
    assert corrected_belief_for_decision("accept_d1_belief", "official_referee_like") == "official_referee_like"
    assert (
        corrected_belief_for_decision("correct_to_assistant_or_line_official_like", "official_referee_like")
        == "assistant_or_line_official_like"
    )
    assert (
        corrected_belief_for_decision("unsure_needs_later_review", "official_referee_like")
        == "unknown_official_context"
    )


def test_invalid_decision_is_rejected() -> None:
    with pytest.raises(ValueError):
        corrected_belief_for_decision("not_a_d1b_decision", "official_referee_like")


def test_reviewed_decision_row_preserves_visual_only_fields() -> None:
    row = reviewed_decision_row(candidate(), "accept_d1_belief")
    assert row["human_confirmed"] is True
    assert row["visual_only_warning"] == VISUAL_ONLY_WARNING
    assert row["do_not_use_for_metrics"] is True
    assert row["production_ready"] is False


def test_payload_validation_rejects_disallowed_decisions_and_production_ready_true() -> None:
    row = reviewed_decision_row(candidate(), "accept_d1_belief")
    row["human_review_decision"] = "not_allowed"
    row["production_ready"] = True
    validation, usable = validate_reviewed_decision_payload(
        {"rows": [candidate()]},
        {"rows": [row]},
        reviewed_decisions_loaded=True,
    )
    assert validation["reviewed_decisions_valid"] is False
    assert usable == []
    errors = {error["error"] for error in validation["validation_errors"]}
    assert "human_review_decision_not_allowed" in errors
    assert "production_ready_true_rejected" in errors
