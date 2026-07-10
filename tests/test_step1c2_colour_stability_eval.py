from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_eval import (  # noqa: E402
    build_colour_stability_eval_summary,
)


def labels_payload() -> dict:
    return {
        "frames": [
            {
                "frame_id": "frame_001",
                "frame_sequence": 1,
                "timestamp_seconds": 1.0,
                "labels_complete": True,
                "persons": [
                    {
                        "candidate_row_id": "gold_1",
                        "gold_person_id": "gold_1",
                        "visible_person_type_gold": "team_1_player",
                        "occlusion_state_gold": "observed_clear",
                        "bbox": {"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 40.0},
                    }
                ],
            }
        ]
    }


def stability_payload() -> dict:
    return {
        "rows": [
            {
                "frame_id": "frame_001",
                "frame_sequence": 1,
                "timestamp_seconds": 1.0,
                "visible_person_base_id": "base_1",
                "detection_id": "det_1",
                "source_detection_id": "source_1",
                "bbox": {"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 40.0},
                "footpoint": {"x": 15.0, "y": 40.0, "method": "bbox_bottom_center", "confidence": 0.9},
                "candidate_type": "player_candidate_source",
                "roi_status": "inside_or_unverified_visual_roi",
                "c1c_seed_team_colour_belief": "team_1_outfield_colour_like",
                "c2_stable_colour_belief": "team_1_outfield_colour_like",
                "c2_stable_colour_belief_confidence": 0.9,
                "c2_stable_colour_belief_state": "high_confidence_stable_visual_colour",
                "c2_stability_action": "retained_seeded_belief",
                "short_burst_colour_group_id": "step1c2_sbcg_f000001_000",
                "c2_review_required": False,
                "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
                "production_ready": False,
            }
        ],
        "summary": {
            "one_row_per_c1c_seeded_belief_row": True,
            "context_offroi_forced_to_team_count": 0,
            "c2_stability_action_counts": {"retained_seeded_belief": 1},
            "c2_stable_belief_counts": {"team_1_outfield_colour_like": 1},
        },
    }


def test_gold_visible_type_is_used_only_as_visual_qa_proxy() -> None:
    summary = build_colour_stability_eval_summary(
        {
            "seeded_belief_row_count": 1,
            "seeded_team_1_team_2_gold_proxy_distribution": {},
            "seeded_team_1_team_2_separation_score": 0.0,
            "unknown_on_gold_player_proxy_count": 0,
            "dark_context_on_gold_player_proxy_count": 0,
            "frames_needing_manual_followup": [],
        },
        {"rows": [{"short_burst_colour_group_id": "step1c2_sbcg_f000001_000"}]},
        stability_payload(),
        {"rows": [], "summary": {"flip_type_counts": {}}},
        labels_payload=labels_payload(),
    )
    assert summary["gold_proxy_note"].startswith("Gold visible_person_type_gold")
    assert summary["c2_proxy_distribution"]["team_1_player"]["team_1_outfield_colour_like"] == 1
    assert summary["identity_tracking_performed"] is False


def test_row_count_and_c1c_c2_comparison_fields_are_emitted() -> None:
    summary = build_colour_stability_eval_summary(
        {
            "seeded_belief_row_count": 1,
            "seeded_team_1_team_2_gold_proxy_distribution": {"team_1_player": {"team_1_outfield_colour_like": 1}},
            "seeded_team_1_team_2_separation_score": 0.5,
            "unknown_on_gold_player_proxy_count": 0,
            "dark_context_on_gold_player_proxy_count": 0,
            "frames_needing_manual_followup": [59, 60, 61, 62],
        },
        {"rows": [{"short_burst_colour_group_id": "step1c2_sbcg_f000001_000"}]},
        stability_payload(),
        {"rows": [], "summary": {"flip_type_counts": {}}},
        labels_payload=labels_payload(),
    )
    assert summary["c1c_seeded_belief_row_count"] == 1
    assert summary["c2_stability_row_count"] == 1
    assert "c1c_separation_score" in summary
    assert "c2_separation_score" in summary
    assert set([59, 60, 61, 62]).issubset(set(summary["frames_needing_manual_followup"]))
