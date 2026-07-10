from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_cluster_diagnostics import evaluate_gold8_proxy_clusters  # noqa: E402


def gold_person(row_id: str, visible_type: str, x1: float) -> dict:
    return {
        "candidate_row_id": row_id,
        "gold_person_id": row_id,
        "visible_person_type_gold": visible_type,
        "occlusion_state_gold": "observed_clear",
        "bbox": {"x1": x1, "y1": 10.0, "x2": x1 + 10.0, "y2": 40.0},
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
    }


def belief_row(base_id: str, x1: float, belief: str) -> dict:
    return {
        "frame_id": "frame_001",
        "frame_sequence": 1,
        "timestamp_seconds": 1.0,
        "visible_person_base_id": base_id,
        "detection_id": f"det_{base_id}",
        "source_detection_id": f"src_{base_id}",
        "bbox": {"x1": x1, "y1": 10.0, "x2": x1 + 10.0, "y2": 40.0},
        "footpoint": {"x": x1 + 5.0, "y": 40.0, "method": "bbox_bottom_center", "confidence": 0.9},
        "team_colour_belief": belief,
        "team_colour_belief_state": "high_confidence_visual_colour",
        "colour_cluster_candidate": belief,
        "team_colour_belief_confidence": 0.86,
        "crop_quality": "high",
        "crop_quality_reason": "usable_low_background_torso_crop",
    }


def test_gold8_proxy_diagnostics_report_same_cluster_collapse_without_role_evaluation() -> None:
    labels_payload = {
        "frames": [
            {
                "frame_id": "frame_001",
                "frame_sequence": 1,
                "timestamp_seconds": 1.0,
                "labels_complete": True,
                "persons": [
                    gold_person("g1", "team_1_player", 10.0),
                    gold_person("g2", "team_2_player", 40.0),
                ],
            }
        ]
    }
    belief_payload = {
        "rows": [
            belief_row("a", 10.0, "outfield_colour_cluster_a"),
            belief_row("b", 40.0, "outfield_colour_cluster_a"),
        ],
        "sandbox_only": True,
    }
    summary, rows = evaluate_gold8_proxy_clusters(
        belief_payload,
        labels_payload=labels_payload,
        profile_name="test_profile",
        prototype_strategy="test_strategy",
    )
    assert summary["gold8_colour_eval_fields_used"] == ["visible_person_type_gold"]
    assert summary["team_1_team_2_same_cluster_proxy"] is True
    assert summary["one_cluster_dominates_both_teams"] is True
    assert summary["profile_visually_promising"] is False
    assert "stable_visual_identity_id_gold" not in rows[0]
    assert "role_gold" not in rows[0]
