from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_beliefs import (  # noqa: E402
    build_goalkeeper_context_payloads,
)
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_eval import (  # noqa: E402
    build_goalkeeper_context_eval_summary,
)


def d1c_row(index: int, *, x1: float, colour: str, source: str = "player") -> dict:
    return {
        "visible_person_base_id": f"base_{index}",
        "frame_id": "frame_59",
        "frame_sequence": 59,
        "timestamp_seconds": 59.0,
        "detection_id": f"det_{index}",
        "source_detection_id": f"source_{index}",
        "bbox": {"x1": x1, "y1": 100.0, "x2": x1 + 34.0, "y2": 178.0},
        "footpoint": {"x": x1 + 17.0, "y": 178.0, "method": "bbox", "confidence": 0.9},
        "state": "observed_clear",
        "roi_status": "inside_or_unverified_visual_roi",
        "candidate_type": "player_candidate_source",
        "original_role_source": source,
        "c2c_final_colour_belief": colour,
        "c2c_colour_source": "c2c_human_corrected",
        "c2c_human_reviewed": True,
        "d1c_final_official_context_belief": "player_like_not_official_context",
        "d1c_context_source": "d1c_test",
        "d1c_human_reviewed": False,
        "d1c_bad_detection_or_not_person": False,
        "d1c_official_like_visual_context": False,
        "d1c_assistant_or_line_official_like_visual_context": False,
        "retained_for_future_player_team_review": True,
        "production_ready": False,
    }


def c2c_row(row: dict) -> dict:
    return {
        "visible_person_base_id": row["visible_person_base_id"],
        "c2c_final_colour_belief": row["c2c_final_colour_belief"],
        "torso_crop_bbox": row["bbox"],
        "crop_quality": "clear",
    }


def labels_payload() -> dict:
    return {
        "frames": [
            {
                "frame_id": "frame_59",
                "frame_sequence": 59,
                "timestamp_seconds": 59.0,
                "labels_complete": True,
                "persons": [
                    {
                        "gold_person_id": "gold_gk",
                        "visible_person_type_gold": "gk_team_1",
                        "occlusion_state_gold": "observed_clear",
                        "bbox": {"x1": 100.0, "y1": 100.0, "x2": 134.0, "y2": 178.0},
                    },
                    {
                        "gold_person_id": "gold_player",
                        "visible_person_type_gold": "team_1_player",
                        "occlusion_state_gold": "observed_clear",
                        "bbox": {"x1": 200.0, "y1": 100.0, "x2": 234.0, "y2": 178.0},
                    },
                ],
            }
        ]
    }


def test_gold_proxy_eval_is_visual_only_and_can_pass_when_row_preservation_holds() -> None:
    rows = [
        d1c_row(0, x1=100.0, colour="other_distinct_colour_like", source="goalkeeper"),
        d1c_row(1, x1=200.0, colour="other_distinct_colour_like"),
    ]
    rows.extend(
        d1c_row(index, x1=900.0 + (index % 50), colour="team_1_outfield_colour_like")
        for index in range(2, 10418)
    )
    feature_payload, belief_payload, review_payload = build_goalkeeper_context_payloads(
        {"rows": rows},
        {"rows": [c2c_row(row) for row in rows]},
    )
    summary = build_goalkeeper_context_eval_summary(
        {"d1c_row_count": 10418, "d1c_safe_for_step1e_candidate": True},
        feature_payload,
        belief_payload,
        review_payload,
        labels_payload=labels_payload(),
    )

    assert summary["gold_proxy_note"].startswith("Gold visible_person_type_gold is used only")
    assert summary["gold_goalkeeper_proxy_rows"] == 1
    assert summary["gold_goalkeeper_proxy_matched_rows"] == 1
    assert "goalkeeper_like_unknown_team_context" in summary["e1_goalkeeper_proxy_belief_distribution"]
    assert summary["e1_goalkeeper_like_false_positive_proxy_count"] == 1
    assert summary["e1_safe_for_human_review_candidate"] is True
    assert summary["production_ready"] is False


def test_safe_for_human_review_requires_d1c_safe_flag_and_10418_rows() -> None:
    rows = [d1c_row(0, x1=100.0, colour="other_distinct_colour_like", source="goalkeeper")]
    feature_payload, belief_payload, review_payload = build_goalkeeper_context_payloads(
        {"rows": rows},
        {"rows": [c2c_row(row) for row in rows]},
    )
    summary = build_goalkeeper_context_eval_summary(
        {"d1c_row_count": 1, "d1c_safe_for_step1e_candidate": False},
        feature_payload,
        belief_payload,
        review_payload,
        labels_payload=labels_payload(),
    )

    assert summary["e1_safe_for_human_review_candidate"] is False
    assert "d1c_not_safe_for_step1e_candidate" in summary["e1_safety_missing_reasons"]
    assert "e1_row_counts_not_10418" in summary["e1_safety_missing_reasons"]
    assert summary["gold_goalkeeper_proxy_matched_rows"] == 1
