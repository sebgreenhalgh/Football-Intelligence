from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import football_intelligence.step1_visual_reconstruction.official_context_review_eval as d1b_eval  # noqa: E402
from football_intelligence.step1_visual_reconstruction.official_context_review_eval import (  # noqa: E402
    progress_summary_payload,
    review_decision_summary_payload,
    save_single_review_decision,
)
from football_intelligence.step1_visual_reconstruction.official_context_review_schema import (  # noqa: E402
    reviewed_decision_row,
    reviewed_rows_from_payload,
)


def candidate(index: int, **overrides: object) -> dict:
    row = {
        "step1d1_review_candidate_id": f"d1b_eval_{index:04d}",
        "visible_person_base_id": f"base_{index:04d}",
        "frame_sequence": index,
        "review_reason_tags": [],
        "official_context_belief": "player_like_not_official_context",
        "official_context_belief_confidence": 0.6,
        "official_context_belief_state": "candidate",
        "c2c_final_colour_belief": "team_1_outfield_colour_like",
        "c2c_context_or_offroi_human_team_override": False,
        "source_official_candidate_flag": False,
    }
    row.update(overrides)
    return row


def reviewed(rows: list[dict]) -> dict[str, dict]:
    return {row["step1d1_review_candidate_id"]: reviewed_decision_row(row, "accept_d1_belief") for row in rows}


def gate_pass_candidates() -> list[dict]:
    rows: list[dict] = []
    for index in range(120):
        belief = "official_referee_like"
        tags = []
        if index < 16:
            tags = ["gold8_official_proxy_match"]
        elif index < 19:
            belief = "unknown_official_context"
            tags = ["gold8_official_proxy_match"]
        elif index == 19:
            belief = "player_like_not_official_context"
            tags = ["gold8_official_proxy_match"]
        rows.append(
            candidate(
                index,
                review_reason_tags=tags,
                official_context_belief=belief,
                c2c_final_colour_belief="other_distinct_colour_like",
                source_official_candidate_flag=True,
            )
        )
    rows.extend(
        candidate(
            200 + index,
            official_context_belief="off_pitch_context_person_like",
            c2c_context_or_offroi_human_team_override=True,
            review_reason_tags=["c2c_context_offroi_human_team_override"],
        )
        for index in range(8)
    )
    rows.extend(candidate(400 + index, official_context_belief="bad_detection_or_not_person") for index in range(104))
    rows.extend(
        candidate(
            600 + index,
            official_context_belief="official_referee_like",
            c2c_final_colour_belief="team_2_outfield_colour_like",
        )
        for index in range(5)
    )
    return rows


def test_gate_false_when_no_decisions() -> None:
    rows = gate_pass_candidates()
    progress = progress_summary_payload(rows, {})
    summary = review_decision_summary_payload(rows, {})
    assert progress["reviewed_candidates"] == 0
    assert summary["d1b_approve_d1_for_next_stage_candidate"] is False
    assert "review_minimum_not_satisfied" in summary["d1b_safety_missing_reasons"]
    assert summary["approve_any_official_exclusion"] is False
    assert summary["approve_any_player_slot_use"] is False


def test_required_buckets_are_tracked() -> None:
    rows = gate_pass_candidates()
    partial = reviewed(rows[:20])
    progress = progress_summary_payload(rows, partial)
    assert progress["gold8_official_proxy_reviewed_count"] == 20
    assert progress["bad_detection_reviewed_count"] == 0
    assert progress["c2c_context_offroi_override_reviewed_count"] == 0
    assert progress["team_colour_with_context_like_reviewed_count"] == 0


def test_gate_can_pass_after_required_buckets_and_review_minimum_are_satisfied() -> None:
    rows = gate_pass_candidates()
    summary = review_decision_summary_payload(rows, reviewed(rows))
    assert summary["reviewed_candidates"] == len(rows)
    assert summary["gold8_official_proxy_review_complete"] is True
    assert summary["official_proxy_missed_or_unknown_review_complete"] is True
    assert summary["c2c_context_offroi_override_review_complete"] is True
    assert summary["bad_detection_review_complete"] is True
    assert summary["team_colour_with_context_like_review_complete"] is True
    assert summary["official_source_reviewed_count"] >= 120
    assert summary["official_source_reviewed_sufficient"] is True
    assert summary["d1b_approve_d1_for_next_stage_candidate"] is True
    assert summary["approve_any_official_exclusion"] is False
    assert summary["approve_any_player_slot_use"] is False
    assert summary["production_ready"] is False


def test_save_single_review_decision_writes_autosave_payload(monkeypatch, tmp_path: Path) -> None:
    candidates = [candidate(1)]
    monkeypatch.setattr(d1b_eval, "ordered_review_candidates", lambda: candidates)
    monkeypatch.setattr(d1b_eval, "load_reviewed_decisions", lambda: {})
    first = candidates[0]
    output_path = tmp_path / "step1d1b_reviewed_official_context_decisions.json"
    payload = save_single_review_decision(
        str(first["step1d1_review_candidate_id"]),
        "accept_d1_belief",
        output_path=output_path,
    )
    rows = reviewed_rows_from_payload(payload)
    assert output_path.exists()
    assert len(rows) == 1
    assert rows[0]["step1d1_review_candidate_id"] == first["step1d1_review_candidate_id"]
    assert rows[0]["human_review_decision"] == "accept_d1_belief"
    assert rows[0]["production_ready"] is False
