from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.official_context_human_corrections import (  # noqa: E402
    build_human_corrected_official_context_payloads,
)
from football_intelligence.step1_visual_reconstruction.official_context_review_schema import (  # noqa: E402
    reviewed_decision_row,
)


def d1_row(index: int, belief: str, *, review_required: bool = False) -> dict:
    return {
        "visible_person_base_id": f"base_{index}",
        "frame_id": f"frame_{index}",
        "frame_sequence": index,
        "timestamp_seconds": float(index),
        "detection_id": f"det_{index}",
        "source_detection_id": f"source_{index}",
        "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 70.0},
        "footpoint": {"x": 20.0, "y": 70.0, "method": "bbox", "confidence": 0.9},
        "state": "observed_clear",
        "roi_status": "inside_or_unverified_visual_roi",
        "candidate_type": "player_candidate_source",
        "original_role_source": "player",
        "c2c_final_colour_belief": "team_1_outfield_colour_like",
        "c2c_colour_source": "c2c_human_corrected",
        "c2c_human_reviewed": True,
        "c2c_context_or_offroi_human_team_override": False,
        "official_context_belief": belief,
        "official_context_belief_state": "review_required" if review_required else "high_confidence_visual_context",
        "official_context_belief_confidence": 0.82,
        "official_context_review_required": review_required,
        "retained_for_future_player_team_review": True,
        "production_ready": False,
    }


def candidate(row: dict, index: int) -> dict:
    return {
        **row,
        "step1d1_review_candidate_id": f"d1_review_{index}",
        "review_reason_tags": ["test"],
    }


def test_human_decisions_apply_row_by_row_and_preserve_retention() -> None:
    d1_rows = [
        d1_row(1, "official_referee_like"),
        d1_row(2, "official_referee_like"),
        d1_row(3, "player_like_not_official_context"),
        d1_row(4, "off_pitch_context_person_like", review_required=True),
    ]
    candidates = [candidate(d1_rows[0], 1), candidate(d1_rows[1], 2), candidate(d1_rows[2], 3)]
    reviewed = [
        reviewed_decision_row(candidates[0], "accept_d1_belief"),
        reviewed_decision_row(candidates[1], "correct_to_assistant_or_line_official_like"),
        reviewed_decision_row(candidates[2], "unsure_needs_later_review"),
    ]
    corrected, audit = build_human_corrected_official_context_payloads(
        {"rows": d1_rows},
        {"rows": candidates},
        {"rows": reviewed},
    )
    by_id = {row["visible_person_base_id"]: row for row in corrected["rows"]}
    assert corrected["summary"]["d1c_row_count"] == 4
    assert corrected["summary"]["one_row_per_d1_belief_row"] is True
    assert len(audit["rows"]) == 3
    assert by_id["base_1"]["d1c_final_official_context_belief"] == "official_referee_like"
    assert by_id["base_1"]["d1c_context_source"] == "d1b_human_accepted"
    assert by_id["base_2"]["d1c_final_official_context_belief"] == "assistant_or_line_official_like"
    assert by_id["base_2"]["d1c_context_source"] == "d1b_human_corrected"
    assert by_id["base_2"]["d1c_assistant_or_line_official_like_visual_context"] is True
    assert by_id["base_3"]["d1c_final_official_context_belief"] == "unknown_official_context"
    assert by_id["base_3"]["d1c_review_required"] is True
    assert by_id["base_4"]["d1c_context_source"] == "d1_not_reviewed_retained"
    assert all(row["retained_for_future_player_team_review"] is True for row in corrected["rows"])
    assert all("official_exclusion" not in row for row in corrected["rows"])
