from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.official_context_beliefs import (  # noqa: E402
    ALLOWED_OFFICIAL_CONTEXT_BELIEFS,
    build_official_context_belief_payload,
)


def feature(index: int, **overrides: object) -> dict:
    row = {
        "official_context_feature_id": f"feature_{index}",
        "visible_person_base_id": f"base_{index}",
        "frame_id": f"frame_{index}",
        "frame_sequence": index,
        "timestamp_seconds": float(index),
        "detection_id": f"det_{index}",
        "source_detection_id": f"source_{index}",
        "bbox": {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
        "footpoint": {"x": 2.0, "y": 4.0, "method": "bbox", "confidence": 0.9},
        "state": "observed_clear",
        "roi_status": "inside_or_unverified_visual_roi",
        "candidate_type": "player_candidate_source",
        "original_role_source": "player",
        "c2c_final_colour_belief": "team_1_outfield_colour_like",
        "c2c_colour_source": "c2_not_reviewed_retained",
        "c2c_human_reviewed": False,
        "c2c_context_or_offroi_human_team_override": False,
        "source_official_candidate_flag": False,
        "source_unknown_context_candidate_flag": False,
        "source_player_candidate_flag": True,
        "offroi_or_recovery_context_flag": False,
        "image_space_lower_frame_band_flag": False,
        "image_space_near_touchline_context_flag": False,
        "team_colour_like_flag": True,
        "non_team_colour_like_flag": False,
        "dark_or_black_like_visual_flag": False,
        "bright_referee_colour_like_visual_flag": False,
        "red_or_pink_like_visual_flag": False,
        "yellow_or_orange_like_visual_flag": False,
        "mixed_colour_or_overlap_warning": False,
        "bad_detection_candidate_flag": False,
    }
    row.update(overrides)
    return row


def test_beliefs_are_allowed_one_row_per_feature_and_no_exclusion() -> None:
    payload = build_official_context_belief_payload(
        {"rows": [feature(1), feature(2, source_official_candidate_flag=True)]}
    )
    assert len(payload["rows"]) == 2
    assert payload["summary"]["one_belief_row_per_feature_row"] is True
    assert all(row["official_context_belief"] in ALLOWED_OFFICIAL_CONTEXT_BELIEFS for row in payload["rows"])
    assert all(row["retained_for_future_player_team_review"] is True for row in payload["rows"])
    assert all(row["eligible_for_player_slot_assignment"] is False for row in payload["rows"])


def test_context_override_remains_flagged_and_bad_detection_review_required() -> None:
    payload = build_official_context_belief_payload(
        {
            "rows": [
                feature(
                    1,
                    c2c_context_or_offroi_human_team_override=True,
                    offroi_or_recovery_context_flag=True,
                ),
                feature(2, bad_detection_candidate_flag=True),
            ]
        }
    )
    by_id = {row["visible_person_base_id"]: row for row in payload["rows"]}
    assert by_id["base_1"]["c2c_context_or_offroi_human_team_override"] is True
    assert (
        "c2c_context_offroi_human_team_override_not_player_evidence"
        in by_id["base_1"]["official_context_warning_flags"]
    )
    assert by_id["base_2"]["official_context_belief"] == "bad_detection_or_not_person"
    assert by_id["base_2"]["official_context_review_required"] is True
