from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.manual_seed_review_state import (  # noqa: E402
    filter_candidates,
    is_reviewed,
    load_seed_candidates,
    merged_review_state,
    next_unreviewed_index,
)


def seed_candidate(seed_id: str, category: str, frame_sequence: int = 1) -> dict:
    return {
        "seed_candidate_id": seed_id,
        "visible_person_base_id": f"base_{seed_id}",
        "frame_sequence": frame_sequence,
        "seed_candidate_category": category,
        "review_priority": 1,
    }


def test_loads_184_seed_candidates() -> None:
    rows = load_seed_candidates()
    assert len(rows) == 184
    assert rows[0]["seed_candidate_category"] == "likely_team_1_colour_seed_prefill"


def test_preserves_existing_labels_and_finds_next_unreviewed() -> None:
    candidates = [
        seed_candidate("seed_a", "likely_team_1_colour_seed_prefill"),
        seed_candidate("seed_b", "likely_team_2_colour_seed_prefill"),
    ]
    reviewed = {
        "seed_a": {
            "seed_candidate_id": "seed_a",
            "manual_colour_label": "team_1_outfield_colour_seed",
            "manual_label_confidence": "high",
            "reviewer_notes": "clear",
            "reviewer_name": "reviewer",
            "reviewed_at": "2026-07-05T00:00:00+00:00",
            "human_confirmed": True,
        }
    }
    rows = merged_review_state(candidates, reviewed)
    assert rows[0]["saved_manual_colour_label"] == "team_1_outfield_colour_seed"
    assert rows[0]["saved_reviewer_notes"] == "clear"
    assert is_reviewed(rows[0]) is True
    assert is_reviewed(rows[1]) is False
    assert next_unreviewed_index(rows) == 1


def test_category_and_review_filters_work() -> None:
    rows = merged_review_state(
        [
            seed_candidate("seed_a", "likely_team_1_colour_seed_prefill", 10),
            seed_candidate("seed_b", "dark_context_seed_review", 11),
        ],
        {
            "seed_b": {
                "seed_candidate_id": "seed_b",
                "manual_colour_label": "dark_context_colour",
                "human_confirmed": True,
            }
        },
    )
    dark_rows = filter_candidates(rows, category="dark_context_seed_review")
    assert [row["seed_candidate_id"] for row in dark_rows] == ["seed_b"]
    assert [row["seed_candidate_id"] for row in filter_candidates(rows, reviewed_state="reviewed")] == ["seed_b"]
    assert [row["seed_candidate_id"] for row in filter_candidates(rows, reviewed_state="unreviewed")] == ["seed_a"]
    assert [row["seed_candidate_id"] for row in filter_candidates(rows, frame_sequence=10)] == ["seed_a"]
    assert [row["seed_candidate_id"] for row in filter_candidates(rows, manual_label="dark_context_colour")] == [
        "seed_b"
    ]
