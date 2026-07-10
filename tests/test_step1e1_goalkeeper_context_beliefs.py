from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_beliefs import (  # noqa: E402
    ALLOWED_E1_GOALKEEPER_CONTEXT_BELIEFS,
    GOALKEEPER_LIKE_BELIEFS,
    build_goalkeeper_context_payloads,
)


def d1c_row(
    index: int,
    *,
    belief: str = "player_like_not_official_context",
    original_role_source: str = "player",
    x1: float = 900.0,
    colour: str = "team_1_outfield_colour_like",
) -> dict:
    return {
        "visible_person_base_id": f"base_{index}",
        "frame_id": f"frame_{index}",
        "frame_sequence": index,
        "timestamp_seconds": float(index),
        "detection_id": f"det_{index}",
        "source_detection_id": f"source_{index}",
        "bbox": {"x1": x1, "y1": 90.0, "x2": x1 + 34.0, "y2": 170.0},
        "footpoint": {"x": x1 + 17.0, "y": 170.0, "method": "bbox", "confidence": 0.9},
        "state": "observed_clear",
        "roi_status": "inside_or_unverified_visual_roi",
        "candidate_type": "player_candidate_source",
        "original_role_source": original_role_source,
        "c2c_final_colour_belief": colour,
        "c2c_colour_source": "c2c_human_corrected",
        "c2c_human_reviewed": True,
        "d1c_final_official_context_belief": belief,
        "d1c_context_source": "d1c_test",
        "d1c_human_reviewed": False,
        "d1c_bad_detection_or_not_person": belief == "bad_detection_or_not_person",
        "d1c_official_like_visual_context": belief == "official_referee_like",
        "d1c_assistant_or_line_official_like_visual_context": False,
        "retained_for_future_player_team_review": True,
        "production_ready": False,
    }


def c2c_row(row: dict, colour: str, *, crop_quality: str = "clear") -> dict:
    return {
        "visible_person_base_id": row["visible_person_base_id"],
        "c2c_final_colour_belief": colour,
        "torso_crop_bbox": row["bbox"],
        "crop_quality": crop_quality,
    }


def payloads_for(rows: list[dict]) -> tuple[dict, dict, dict]:
    c2c_rows = [c2c_row(row, row["c2c_final_colour_belief"]) for row in rows]
    return build_goalkeeper_context_payloads({"rows": rows}, {"rows": c2c_rows})


def test_e1_beliefs_are_conservative_visual_context_and_preserve_rows() -> None:
    rows = [
        d1c_row(1, colour="team_1_outfield_colour_like"),
        d1c_row(2, x1=100.0, colour="other_distinct_colour_like"),
        d1c_row(3, belief="official_referee_like", x1=100.0, colour="other_distinct_colour_like"),
        d1c_row(4, belief="bad_detection_or_not_person", x1=100.0, colour="other_distinct_colour_like"),
        d1c_row(5, original_role_source="goalkeeper", colour="team_2_outfield_colour_like"),
    ]
    feature_payload, belief_payload, review_payload = payloads_for(rows)
    beliefs = {row["visible_person_base_id"]: row for row in belief_payload["rows"]}

    assert len(feature_payload["rows"]) == len(rows)
    assert len(belief_payload["rows"]) == len(rows)
    assert belief_payload["summary"]["one_belief_row_per_d1c_row"] is True
    emitted_beliefs = {row["e1_goalkeeper_context_belief"] for row in belief_payload["rows"]}
    assert emitted_beliefs <= ALLOWED_E1_GOALKEEPER_CONTEXT_BELIEFS
    assert beliefs["base_1"]["e1_goalkeeper_context_belief"] == "outfield_player_like_not_goalkeeper"
    assert beliefs["base_2"]["e1_goalkeeper_context_belief"] == "goalkeeper_like_unknown_team_context"
    assert beliefs["base_3"]["e1_goalkeeper_context_belief"] == "official_or_context_not_goalkeeper"
    assert beliefs["base_3"]["e1_goalkeeper_like_visual_context"] is False
    assert beliefs["base_4"]["e1_goalkeeper_context_belief"] == "bad_detection_or_not_person"
    assert beliefs["base_4"]["e1_goalkeeper_like_visual_context"] is False
    assert beliefs["base_5"]["e1_goalkeeper_context_belief"] == "goalkeeper_like_team_2_context"
    assert all(row["retained_for_future_player_team_review"] is True for row in belief_payload["rows"])
    assert all(row["eligible_for_identity_tracking"] is False for row in belief_payload["rows"])
    assert all(row["eligible_for_player_slot_assignment"] is False for row in belief_payload["rows"])
    assert belief_payload["summary"]["exact_two_goalkeeper_forcing_performed"] is False
    assert len(review_payload["rows"]) > 0


def test_e1_preserves_10418_rows_and_does_not_force_exactly_two_goalkeepers() -> None:
    rows = [d1c_row(index) for index in range(10418)]
    rows[0] = d1c_row(0, x1=100.0, colour="other_distinct_colour_like")
    rows[1] = d1c_row(1, original_role_source="goalkeeper", colour="team_2_outfield_colour_like")
    rows[2] = d1c_row(2, original_role_source="goalkeeper", colour="team_1_outfield_colour_like")
    feature_payload, belief_payload, _review_payload = payloads_for(rows)

    assert feature_payload["summary"]["e1_feature_row_count"] == 10418
    assert belief_payload["summary"]["e1_belief_row_count"] == 10418
    assert belief_payload["summary"]["one_belief_row_per_d1c_row"] is True
    assert belief_payload["summary"]["exact_two_goalkeeper_forcing_performed"] is False
    assert belief_payload["summary"]["e1_goalkeeper_like_visual_context_count"] > 2
    assert all(
        row["e1_goalkeeper_context_belief"] in GOALKEEPER_LIKE_BELIEFS
        for row in belief_payload["rows"]
        if row["e1_goalkeeper_like_visual_context"]
    )
