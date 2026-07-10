from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state import (  # noqa: E402
    build_fused_visual_role_state_payloads,
)
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_eval import (  # noqa: E402
    build_f1_eval_summary,
)
from test_step1f1_fused_visual_role_state import payload, row  # noqa: E402


def labels_payload() -> dict:
    return {
        "frames": [
            {
                "frame_id": "frame_0",
                "frame_sequence": 0,
                "timestamp_seconds": 0.0,
                "labels_complete": True,
                "persons": [
                    {
                        "gold_person_id": "gold_gk",
                        "visible_person_type_gold": "gk_team_1",
                        "occlusion_state_gold": "observed_clear",
                        "bbox": {"x1": 100.0, "y1": 100.0, "x2": 130.0, "y2": 180.0},
                    },
                    {
                        "gold_person_id": "gold_player",
                        "visible_person_type_gold": "team_1_player",
                        "occlusion_state_gold": "observed_clear",
                        "bbox": {"x1": 103.0, "y1": 100.0, "x2": 133.0, "y2": 180.0},
                    },
                ],
            }
        ]
    }


def test_gold_proxy_eval_is_visual_only_and_distributions_are_emitted() -> None:
    rows = [
        row(0, e1c="goalkeeper_like_team_1_context"),
        row(1, c2c="team_1_outfield_colour_like", e1c="outfield_player_like_not_goalkeeper"),
    ]
    fused, conflict = build_fused_visual_role_state_payloads(payload(rows), payload(rows), payload(rows), payload(rows))
    summary = build_f1_eval_summary(
        fused,
        conflict,
        {"e1c_safe_for_step1f_candidate": True},
        labels_payload=labels_payload(),
    )
    assert summary["gold_proxy_note"].startswith("Gold visible_person_type_gold is used only")
    assert "goalkeeper_proxy_distribution" in summary
    assert "outfield_player_proxy_distribution" in summary
    assert "missed_goalkeeper_proxy_count" in summary
    assert "official_context_proxy_counts" in summary
    assert summary["production_ready"] is False
    assert summary["f1_safe_for_f2_human_review_candidate"] is False
    assert "f1_row_count_not_10418" in summary["f1_safety_missing_reasons"]


def test_f1_safe_for_f2_requires_e1c_safe_and_row_preservation() -> None:
    rows = [row(0, e1c="goalkeeper_like_team_1_context")]
    rows.extend(
        row(index, c2c="team_1_outfield_colour_like", e1c="outfield_player_like_not_goalkeeper")
        for index in range(1, 10418)
    )
    fused, conflict = build_fused_visual_role_state_payloads(payload(rows), payload(rows), payload(rows), payload(rows))
    safe_summary = build_f1_eval_summary(
        fused,
        conflict,
        {"e1c_safe_for_step1f_candidate": True},
        labels_payload=labels_payload(),
    )
    assert safe_summary["f1_row_count"] == 10418
    assert safe_summary["one_row_per_e1c_row"] is True
    assert safe_summary["f1_safe_for_f2_human_review_candidate"] is True

    blocked_summary = build_f1_eval_summary(
        fused,
        conflict,
        {"e1c_safe_for_step1f_candidate": False},
        labels_payload=labels_payload(),
    )
    assert blocked_summary["f1_safe_for_f2_human_review_candidate"] is False
    assert "e1c_not_safe_for_step1f_candidate" in blocked_summary["f1_safety_missing_reasons"]
