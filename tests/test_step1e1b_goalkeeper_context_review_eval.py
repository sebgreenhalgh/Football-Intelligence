from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_eval import (  # noqa: E402
    GATE_FAIL_ACTION,
    GATE_PASS_HIGH_CORRECTION_ACTION,
    GATE_PASS_LOW_CORRECTION_ACTION,
    review_decision_summary_payload,
    systematic_team_inversion_diagnostic,
)
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_schema import (  # noqa: E402
    reviewed_decision_row,
)


def candidate(index: int, tags: list[str], belief: str = "outfield_player_like_not_goalkeeper") -> dict:
    return {
        "step1e1_review_candidate_id": f"e1_review_{index}",
        "visible_person_base_id": f"base_{index}",
        "frame_sequence": index,
        "review_priority": 90,
        "review_reason_tags": tags,
        "review_reason": ";".join(tags),
        "e1_goalkeeper_context_belief": belief,
        "production_ready": False,
    }


def complete_payload() -> dict:
    rows = []
    index = 0
    specs = [
        ("gold8_goalkeeper_proxy_match", 2, "unknown_goalkeeper_context"),
        ("goalkeeper_like_belief", 2, "goalkeeper_like_unknown_team_context"),
        ("bad_detection_with_goalkeeper_like_hint", 2, "bad_detection_or_not_person"),
        ("contradictory_official_context_goalkeeper_hints", 2, "official_or_context_not_goalkeeper"),
        ("unknown_goalkeeper_context_with_non_outfield_colour_hint", 10, "unknown_goalkeeper_context"),
        ("balanced_sample_outfield_player_like_not_goalkeeper", 60, "outfield_player_like_not_goalkeeper"),
        ("balanced_sample_official_or_context_not_goalkeeper", 60, "official_or_context_not_goalkeeper"),
        ("review_required", 212, "unknown_goalkeeper_context"),
    ]
    for tag, count, belief in specs:
        for _ in range(count):
            rows.append(candidate(index, [tag], belief))
            index += 1
    return {"rows": rows}


def safe_e1_summary() -> dict:
    return {"d1c_safe_for_step1e_candidate": True, "e1_safe_for_human_review_candidate": True}


def reviewed_all(payload: dict, *, correction_count: int = 0) -> dict[str, dict]:
    reviewed = {}
    for index, row in enumerate(payload["rows"]):
        decision = "accept_e1_belief"
        if index < correction_count:
            decision = "correct_to_unknown_goalkeeper_context"
        reviewed[row["step1e1_review_candidate_id"]] = reviewed_decision_row(row, decision)
    return reviewed


def test_approval_gate_false_when_required_buckets_incomplete() -> None:
    payload = complete_payload()
    summary = review_decision_summary_payload(payload, {}, e1_summary=safe_e1_summary())
    assert summary["e1b_approve_e1_for_next_stage_candidate"] is False
    assert summary["recommended_next_action"] == GATE_FAIL_ACTION
    assert "gold_goalkeeper_proxy_review_incomplete" in summary["e1b_safety_missing_reasons"]


def test_approval_gate_true_when_required_buckets_and_minimum_counts_complete() -> None:
    payload = complete_payload()
    summary = review_decision_summary_payload(payload, reviewed_all(payload), e1_summary=safe_e1_summary())
    assert summary["reviewed_candidates"] == 350
    assert summary["e1b_approve_e1_for_next_stage_candidate"] is True
    assert summary["recommended_next_action"] == GATE_PASS_LOW_CORRECTION_ACTION


def test_high_correction_rate_recommends_e1c_but_does_not_fail_gate() -> None:
    payload = complete_payload()
    summary = review_decision_summary_payload(
        payload,
        reviewed_all(payload, correction_count=100),
        e1_summary=safe_e1_summary(),
    )
    assert summary["e1b_approve_e1_for_next_stage_candidate"] is True
    assert summary["high_correction_rate_recommends_e1c"] is True
    assert summary["recommended_next_action"] == GATE_PASS_HIGH_CORRECTION_ACTION


def test_systematic_inversion_only_blocks_when_evaluable_and_detected() -> None:
    candidates = {
        f"e1_review_{index}": candidate(index, ["goalkeeper_like_belief"], "goalkeeper_like_team_1_context")
        for index in range(10)
    }
    reviewed = {
        review_id: reviewed_decision_row(row, "correct_to_goalkeeper_like_team_2_context")
        for review_id, row in candidates.items()
    }
    diagnostic = systematic_team_inversion_diagnostic(candidates, reviewed)
    assert diagnostic["systematic_team_inversion_evaluable"] is True
    assert diagnostic["systematic_team_inversion_detected"] is True

    few_candidates = dict(list(candidates.items())[:2])
    few_reviewed = {review_id: reviewed[review_id] for review_id in few_candidates}
    few_diagnostic = systematic_team_inversion_diagnostic(few_candidates, few_reviewed)
    assert few_diagnostic["systematic_team_inversion_evaluable"] is False
    assert few_diagnostic["systematic_team_inversion_detected"] is False
