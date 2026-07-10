from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.official_context_correction_eval import (  # noqa: E402
    build_d1c_eval_summary,
)
from football_intelligence.step1_visual_reconstruction.official_context_human_corrections import (  # noqa: E402
    build_human_corrected_official_context_payloads,
)
from football_intelligence.step1_visual_reconstruction.official_context_review_schema import (  # noqa: E402
    reviewed_decision_row,
)


def d1_row(index: int, belief: str = "player_like_not_official_context") -> dict:
    x1 = 100.0 + index * 3.0
    return {
        "visible_person_base_id": f"base_{index}",
        "frame_id": "frame_59",
        "frame_sequence": 59,
        "timestamp_seconds": 59.0,
        "detection_id": f"det_{index}",
        "source_detection_id": f"source_{index}",
        "bbox": {"x1": x1, "y1": 100.0, "x2": x1 + 30.0, "y2": 170.0},
        "footpoint": {"x": x1 + 15.0, "y": 170.0, "method": "bbox", "confidence": 0.9},
        "state": "observed_clear",
        "roi_status": "inside_or_unverified_visual_roi",
        "candidate_type": "player_candidate_source",
        "original_role_source": "player",
        "c2c_final_colour_belief": "team_1_outfield_colour_like",
        "c2c_colour_source": "c2c_human_corrected",
        "c2c_human_reviewed": True,
        "c2c_context_or_offroi_human_team_override": False,
        "official_context_belief": belief,
        "official_context_belief_state": "high_confidence_visual_context",
        "official_context_belief_confidence": 0.82,
        "official_context_review_required": False,
        "retained_for_future_player_team_review": True,
        "production_ready": False,
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
                        "gold_person_id": "gold_official",
                        "visible_person_type_gold": "official_referee",
                        "occlusion_state_gold": "observed_clear",
                        "bbox": {"x1": 100.0, "y1": 100.0, "x2": 130.0, "y2": 170.0},
                    },
                    {
                        "gold_person_id": "gold_player",
                        "visible_person_type_gold": "team_1_player",
                        "occlusion_state_gold": "observed_clear",
                        "bbox": {"x1": 103.0, "y1": 100.0, "x2": 133.0, "y2": 170.0},
                    },
                ],
            }
        ]
    }


def test_gold_proxy_eval_reports_d1_and_d1c_distributions() -> None:
    d1_rows = [d1_row(0, "official_referee_like"), d1_row(1, "player_like_not_official_context")]
    candidate = {
        **d1_rows[0],
        "step1d1_review_candidate_id": "d1_review_0",
        "review_reason_tags": ["gold8_official_proxy_match"],
    }
    review = reviewed_decision_row(candidate, "accept_d1_belief")
    d1c_payload, audit_payload = build_human_corrected_official_context_payloads(
        {"rows": d1_rows},
        {"rows": [candidate]},
        {"rows": [review]},
    )
    summary = build_d1c_eval_summary(
        {"rows": d1_rows},
        d1c_payload,
        audit_payload,
        {"rows": [candidate]},
        {"rows": [review]},
        {"d1b_approve_d1_for_next_stage_candidate": True},
        labels_payload=labels_payload(),
    )
    assert summary["gold_proxy_note"].startswith("Gold visible_person_type_gold is used only")
    assert "d1_baseline_official_proxy_distribution" in summary
    assert "d1c_corrected_official_proxy_distribution" in summary
    assert summary["d1c_safe_for_step1e_candidate"] is False
    assert "d1c_row_count_not_10418" in summary["d1c_safety_missing_reasons"]


def test_d1c_safety_can_pass_with_valid_reviewed_decisions_and_row_preservation() -> None:
    rows = [d1_row(0, "official_referee_like")]
    rows.extend(d1_row(index) for index in range(1, 10418))
    candidate = {
        **rows[0],
        "step1d1_review_candidate_id": "d1_review_0",
        "review_reason_tags": ["gold8_official_proxy_match"],
    }
    review = reviewed_decision_row(candidate, "accept_d1_belief")
    d1c_payload, audit_payload = build_human_corrected_official_context_payloads(
        {"rows": rows},
        {"rows": [candidate]},
        {"rows": [review]},
    )
    summary = build_d1c_eval_summary(
        {"rows": rows},
        d1c_payload,
        audit_payload,
        {"rows": [candidate]},
        {"rows": [review]},
        {"d1b_approve_d1_for_next_stage_candidate": True},
        labels_payload=labels_payload(),
    )
    assert summary["d1c_row_count"] == 10418
    assert summary["one_row_per_d1_belief_row"] is True
    assert summary["d1b_reviewed_decisions_valid"] is True
    assert summary["d1c_safe_for_step1e_candidate"] is True
    assert summary["production_ready"] is False
