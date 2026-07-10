from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_state import (  # noqa: E402
    ordered_review_candidates,
    review_state_payload,
)


def candidate(index: int, tags: list[str]) -> dict:
    return {
        "step1e1_review_candidate_id": f"e1_review_{index}",
        "visible_person_base_id": f"base_{index}",
        "frame_sequence": index,
        "review_priority": 90,
        "review_reason_tags": tags,
        "review_reason": ";".join(tags),
        "e1_goalkeeper_context_belief": "unknown_goalkeeper_context",
        "production_ready": False,
    }


def test_required_buckets_are_selected_and_prioritized() -> None:
    payload = {
        "rows": [
            candidate(6, ["balanced_sample_official_or_context_not_goalkeeper"]),
            candidate(2, ["goalkeeper_like_belief"]),
            candidate(5, ["balanced_sample_outfield_player_like_not_goalkeeper"]),
            candidate(4, ["contradictory_official_context_goalkeeper_hints"]),
            candidate(0, ["gold8_goalkeeper_proxy_match"]),
            candidate(3, ["bad_detection_with_goalkeeper_like_hint"]),
            candidate(1, ["unknown_goalkeeper_context_with_non_outfield_colour_hint"]),
            candidate(7, ["review_required"]),
        ]
    }
    rows = ordered_review_candidates(payload)
    assert [row["e1b_review_bucket_priority"] for row in rows] == [0, 5, 10, 15, 20, 70, 75, 90]
    state = review_state_payload(payload)
    counts = state["required_bucket_counts"]
    assert counts["gold8_goalkeeper_proxy_match"] == 1
    assert counts["goalkeeper_like_belief"] == 1
    assert counts["unknown_goalkeeper_context_with_non_outfield_colour_hint"] == 1
    assert counts["bad_detection_with_goalkeeper_like_hint"] == 1
    assert counts["contradictory_official_context_goalkeeper_hints"] == 1
    assert counts["balanced_sample_outfield_player_like_not_goalkeeper"] == 1
    assert counts["balanced_sample_official_or_context_not_goalkeeper"] == 1
    assert state["total_review_candidates"] == 8
