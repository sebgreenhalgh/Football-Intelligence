from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_crop_audit import build_crop_audit_payloads  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def feature_row(base_id: str, reason: str, green: float = 0.0) -> dict:
    return {
        "frame_id": "frame_001",
        "frame_sequence": 62,
        "timestamp_seconds": 1.0,
        "visible_person_base_id": base_id,
        "detection_id": f"det_{base_id}",
        "source_detection_id": f"src_{base_id}",
        "bbox": {"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 40.0},
        "torso_crop_bbox": {"x1": 12.0, "y1": 15.0, "x2": 18.0, "y2": 30.0},
        "crop_width": 6,
        "crop_height": 15,
        "crop_quality": "low",
        "crop_quality_reason": reason,
        "feature_extraction_warning": reason,
        "green_background_fraction": green,
        "blue_like_fraction": 0.2,
        "red_or_orange_like_fraction": 0.0,
        "white_like_fraction": 0.0,
        "black_or_dark_like_fraction": 0.0,
    }


def belief_row(base_id: str, belief: str = "outfield_colour_cluster_a") -> dict:
    return {
        "frame_id": "frame_001",
        "frame_sequence": 62,
        "timestamp_seconds": 1.0,
        "visible_person_base_id": base_id,
        "detection_id": f"det_{base_id}",
        "source_detection_id": f"src_{base_id}",
        "bbox": {"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 40.0},
        "candidate_type": "player_candidate_source",
        "roi_status": "inside_or_unverified_visual_roi",
        "team_colour_belief": belief,
        "team_colour_belief_state": "high_confidence_visual_colour",
        "colour_cluster_candidate": belief,
        "team_colour_belief_confidence": 0.84,
    }


def test_crop_audit_has_one_row_per_c1_feature_and_expected_flags() -> None:
    feature_payload = {"rows": [feature_row("a", "small_torso_crop"), feature_row("b", "mostly_green_background", 0.9)]}
    belief_payload = {"rows": [belief_row("a"), belief_row("b", "unknown_ambiguous_colour")]}
    c1_eval_summary = {
        "gold8_colour_eval_summary": {
            "gold8_colour_proxy_distribution": {
                "team_1_player": {"outfield_colour_cluster_a": 3},
                "team_2_player": {"outfield_colour_cluster_a": 2},
            },
            "frames_needing_manual_followup": [62],
        }
    }
    gold_proxy_rows = [
        {"visible_person_base_id": "a", "visible_person_type_gold": "team_1_player"},
        {"visible_person_base_id": "b", "visible_person_type_gold": "team_2_player"},
    ]
    audit_payload, summary_payload = build_crop_audit_payloads(
        feature_payload,
        belief_payload,
        c1_eval_summary,
        gold_proxy_rows=gold_proxy_rows,
    )
    assert len(audit_payload["rows"]) == len(feature_payload["rows"])
    flags = {row["visible_person_base_id"]: set(row["audit_issue_flags"]) for row in audit_payload["rows"]}
    assert {"small_torso_crop", "team_1_team_2_same_cluster_proxy", "frame_colour_contradiction"} <= flags["a"]
    assert {"mostly_green_background", "unknown_on_gold_player_proxy"} <= flags["b"]
    assert summary_payload["audit_issue_flag_counts"]["needs_manual_crop_review"] == 2
    assert all(row["visual_only_warning"] == VISUAL_ONLY_WARNING for row in audit_payload["rows"])
    assert audit_payload["production_ready"] is False
