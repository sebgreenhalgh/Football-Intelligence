from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.official_context_eval import (  # noqa: E402
    build_official_context_eval_summary,
)


def belief_row(index: int, belief: str = "player_like_not_official_context") -> dict:
    return {
        "visible_person_base_id": f"base_{index}",
        "frame_id": f"frame_{index}",
        "frame_sequence": index,
        "timestamp_seconds": float(index),
        "detection_id": f"det_{index}",
        "source_detection_id": f"source_{index}",
        "bbox": {"x1": 100.0, "y1": 100.0, "x2": 150.0, "y2": 190.0},
        "footpoint": {"x": 125.0, "y": 190.0, "method": "bbox", "confidence": 0.9},
        "official_context_belief": belief,
        "official_context_review_required": belief != "player_like_not_official_context",
        "c2c_context_or_offroi_human_team_override": False,
        "retained_for_future_player_team_review": True,
        "production_ready": False,
    }


def labels_payload() -> dict:
    return {
        "frames": [
            {
                "frame_id": "frame_1",
                "frame_sequence": 1,
                "timestamp_seconds": 1.0,
                "labels_complete": True,
                "persons": [
                    {
                        "gold_person_id": "gold_1",
                        "visible_person_type_gold": "official_referee",
                        "occlusion_state_gold": "observed_clear",
                        "bbox": {"x1": 100.0, "y1": 100.0, "x2": 150.0, "y2": 190.0},
                    }
                ],
            }
        ]
    }


def base_payloads(row_count: int) -> tuple[dict, dict, dict, dict, dict]:
    rows = [belief_row(index) for index in range(row_count)]
    c2c = {"rows": rows}
    feature = {"rows": rows}
    belief = {
        "rows": rows,
        "summary": {
            "official_context_belief_counts": {"player_like_not_official_context": row_count},
            "review_required_count": 0,
            "source_official_candidate_count": 0,
            "c2c_context_offroi_human_team_override_count": 0,
        },
        "production_ready": False,
        "project_wide_defaults_changed": False,
        "stage3d_registries_changed": False,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
    }
    review = {"rows": []}
    c2c_summary = {"context_offroi_human_team_override_count": 0}
    return c2c, feature, belief, review, c2c_summary


def test_gold_proxy_distributions_and_safety_fields_are_emitted() -> None:
    c2c, feature, belief, review, c2c_summary = base_payloads(10418)
    belief["rows"][1]["official_context_belief"] = "official_referee_like"
    review["rows"] = [belief["rows"][1]]
    summary = build_official_context_eval_summary(
        c2c,
        feature,
        belief,
        review,
        c2c_summary,
        labels_payload=labels_payload(),
    )
    assert (
        summary["gold_proxy_note"]
        == "Gold visible_person_type_gold is used only as visual official/context QA proxy context."
    )
    assert "official_proxy_d1_belief_distribution" in summary
    assert "non_official_player_proxy_d1_belief_distribution" in summary
    assert "d1_safe_for_human_review_candidate" in summary
    assert summary["production_ready"] is False


def test_d1_safe_for_review_requires_row_preservation() -> None:
    c2c, feature, belief, review, c2c_summary = base_payloads(10)
    summary = build_official_context_eval_summary(
        c2c,
        feature,
        belief,
        review,
        c2c_summary,
        labels_payload={"frames": []},
    )
    assert summary["d1_safe_for_human_review_candidate"] is False
    assert "d1_row_counts_not_10418" in summary["d1_safety_missing_reasons"]
