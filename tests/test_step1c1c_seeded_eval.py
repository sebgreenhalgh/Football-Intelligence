from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.seeded_colour_eval import build_seeded_eval_summary  # noqa: E402


def test_seeded_eval_safety_false_without_reviewed_labels(monkeypatch) -> None:
    monkeypatch.setattr(
        "football_intelligence.step1_visual_reconstruction.seeded_colour_eval.read_json",
        lambda _path: {
            "gold8_colour_eval_summary": {"gold8_colour_proxy_distribution": {}},
            "c1b_best_profile_name": "torso_upper_only",
            "c1b_safe_for_team_colour_separation_review": False,
        },
    )
    summary = build_seeded_eval_summary(
        {"reviewed_seed_labels_loaded": False, "reviewed_seed_labels_valid": False},
        {"rows": [], "summary": {"context_offroi_forced_to_team_count": 0}},
        labels_payload={"frames": []},
    )
    assert summary["c1c_safe_for_c2_smoothing_review"] is False
    assert "reviewed_seed_labels_missing" in summary["c1c_safety_missing_reasons"]
    assert summary["gold_proxy_note"].startswith("Gold visible_person_type_gold")


def test_seeded_eval_does_not_require_identity_or_role_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        "football_intelligence.step1_visual_reconstruction.seeded_colour_eval.read_json",
        lambda _path: {
            "gold8_colour_eval_summary": {"gold8_colour_proxy_distribution": {}},
            "c1b_best_profile_name": "torso_upper_only",
            "c1b_safe_for_team_colour_separation_review": False,
        },
    )
    labels_payload = {
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
                        "role_gold": "outfield",
                        "stable_visual_identity_id_gold": "id_1",
                        "occlusion_state_gold": "observed_clear",
                        "bbox": {"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 40.0},
                    }
                ],
            }
        ]
    }
    belief_payload = {
        "rows": [
            {
                "frame_id": "frame_001",
                "frame_sequence": 1,
                "timestamp_seconds": 1.0,
                "visible_person_base_id": "base_1",
                "detection_id": "det_1",
                "bbox": {"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 40.0},
                "footpoint": {"x": 15.0, "y": 40.0, "method": "bbox_bottom_center", "confidence": 0.9},
                "seed_team_colour_belief": "team_1_outfield_colour_like",
                "seed_team_colour_belief_confidence": 0.9,
            }
        ],
        "summary": {"context_offroi_forced_to_team_count": 0},
    }
    summary = build_seeded_eval_summary(
        {
            "reviewed_seed_labels_loaded": True,
            "reviewed_seed_labels_valid": True,
            "human_confirmed_team_1_seed_count": 8,
            "human_confirmed_team_2_seed_count": 8,
            "human_confirmed_negative_seed_count": 4,
        },
        belief_payload,
        labels_payload=labels_payload,
    )
    assert summary["seeded_team_1_team_2_gold_proxy_distribution"]["team_1_player"]["team_1_outfield_colour_like"] == 1
    assert summary["identity_tracking_performed"] is False
    assert summary["player_slots_assigned"] is False
