from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.official_context_review_state import (  # noqa: E402
    ordered_review_candidates,
    review_bucket,
)


def candidate(index: int, **overrides: object) -> dict:
    row = {
        "step1d1_review_candidate_id": f"d1_review_{index}",
        "visible_person_base_id": f"base_{index}",
        "frame_sequence": index,
        "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 70.0},
        "review_priority": 50,
        "review_reason": "test",
        "review_reason_tags": [],
        "official_context_belief": "player_like_not_official_context",
        "official_context_belief_confidence": 0.7,
        "official_context_belief_state": "candidate",
        "official_context_warning_flags": [],
        "c2c_final_colour_belief": "team_1_outfield_colour_like",
        "c2c_colour_source": "c2c_human_corrected",
        "c2c_human_reviewed": True,
        "c2c_context_or_offroi_human_team_override": False,
        "candidate_type": "player_candidate_source",
        "original_role_source": "player",
        "roi_status": "inside_or_unverified_visual_roi",
        "source_official_candidate_flag": False,
        "source_player_candidate_flag": True,
        "source_unknown_context_candidate_flag": False,
    }
    row.update(overrides)
    return row


def feature(base_id: str) -> dict:
    return {
        "visible_person_base_id": base_id,
        "crop_quality": "medium",
        "crop_quality_reason": "usable",
        "torso_crop_bbox": {"x1": 11.0, "y1": 21.0, "x2": 29.0, "y2": 48.0},
    }


def test_loads_expected_d1_review_candidate_count_from_artifact() -> None:
    assert len(ordered_review_candidates()) == 1846


def test_ordering_prioritises_required_review_buckets_and_preserves_fields() -> None:
    rows = [
        candidate(6, official_context_belief="bad_detection_or_not_person"),
        candidate(4, source_official_candidate_flag=True),
        candidate(
            1,
            review_reason_tags=["gold8_official_proxy_match"],
            official_context_belief="official_referee_like",
        ),
        candidate(3, official_context_belief="official_referee_like"),
        candidate(5, c2c_context_or_offroi_human_team_override=True),
    ]
    ordered = ordered_review_candidates(
        {"rows": rows},
        {"rows": [feature(row["visible_person_base_id"]) for row in rows]},
    )
    assert [review_bucket(row) for row in ordered] == [0, 2, 3, 4, 5]
    assert ordered[0]["c2c_final_colour_belief"] == "team_1_outfield_colour_like"
    assert ordered[0]["c2c_colour_source"] == "c2c_human_corrected"
    assert ordered[0]["torso_crop_bbox"]["x1"] == 11.0
    assert ordered[0]["production_ready"] is False
