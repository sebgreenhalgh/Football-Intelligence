from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_correction_eval import (  # noqa: E402
    build_e1c_eval_summary,
)
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_human_corrections import (  # noqa: E402
    build_human_corrected_goalkeeper_context_payloads,
)
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_schema import (  # noqa: E402
    reviewed_decision_row,
)


def e1_row(index: int, belief: str = "unknown_goalkeeper_context") -> dict:
    x1 = 100.0 + float(index % 50) * 3.0
    return {
        "visible_person_base_id": f"base_{index}",
        "frame_id": f"frame_{index}",
        "frame_sequence": index,
        "timestamp_seconds": float(index),
        "detection_id": f"det_{index}",
        "source_detection_id": f"source_{index}",
        "bbox": {"x1": x1, "y1": 100.0, "x2": x1 + 30.0, "y2": 180.0},
        "footpoint": {"x": x1 + 15.0, "y": 180.0, "method": "bbox", "confidence": 0.9},
        "state": "observed_clear",
        "roi_status": "inside_or_unverified_visual_roi",
        "candidate_type": "player_candidate_source",
        "original_role_source": "player",
        "c2c_final_colour_belief": "team_1_outfield_colour_like",
        "d1c_final_official_context_belief": "player_like_not_official_context",
        "e1_goalkeeper_context_belief": belief,
        "e1_goalkeeper_context_belief_state": "high_confidence_visual_context",
        "e1_goalkeeper_context_belief_confidence": 0.78,
        "e1_goalkeeper_context_review_required": False,
        "retained_for_future_player_team_review": True,
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
        "production_ready": False,
    }


def candidate_for(row: dict, index: int) -> dict:
    return {
        **row,
        "step1e1_review_candidate_id": f"e1_review_{index}",
        "review_reason_tags": ["gold8_goalkeeper_proxy_match"],
    }


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
                    }
                ],
            }
        ]
    }


def corrected_payloads(rows: list[dict], decision: str = "accept_e1_belief") -> tuple[dict, dict, dict, dict, dict]:
    candidate = candidate_for(rows[0], 0)
    review = reviewed_decision_row(candidate, decision)
    e1_payload = {"rows": rows}
    candidate_payload = {"rows": [candidate]}
    reviewed_payload = {"rows": [review]}
    e1c_payload, audit_payload = build_human_corrected_goalkeeper_context_payloads(
        e1_payload,
        candidate_payload,
        reviewed_payload,
    )
    return e1_payload, e1c_payload, audit_payload, candidate_payload, reviewed_payload


def test_gold_proxy_eval_reports_e1_and_e1c_distributions() -> None:
    rows = [e1_row(0, "unknown_goalkeeper_context")]
    e1_payload, e1c_payload, audit_payload, candidate_payload, reviewed_payload = corrected_payloads(
        rows,
        "correct_to_goalkeeper_like_team_1_context",
    )
    summary = build_e1c_eval_summary(
        e1_payload,
        e1c_payload,
        audit_payload,
        candidate_payload,
        reviewed_payload,
        {"e1b_approve_e1_for_next_stage_candidate": True},
        labels_payload=labels_payload(),
    )
    assert summary["gold_proxy_note"].startswith("Gold visible_person_type_gold is used only")
    assert summary["e1_baseline_goalkeeper_proxy_distribution"] == {"unknown_goalkeeper_context": 1}
    assert summary["e1c_corrected_goalkeeper_proxy_distribution"] == {"goalkeeper_like_team_1_context": 1}
    assert summary["e1c_missed_goalkeeper_proxy_count"] == 0
    assert summary["e1c_safe_for_step1f_candidate"] is False
    assert "e1c_row_count_not_10418" in summary["e1c_safety_missing_reasons"]


def test_e1c_safety_can_pass_with_valid_reviewed_decisions_and_row_preservation() -> None:
    rows = [e1_row(0, "goalkeeper_like_unknown_team_context")]
    rows.extend(e1_row(index, "outfield_player_like_not_goalkeeper") for index in range(1, 10418))
    e1_payload, e1c_payload, audit_payload, candidate_payload, reviewed_payload = corrected_payloads(rows)
    summary = build_e1c_eval_summary(
        e1_payload,
        e1c_payload,
        audit_payload,
        candidate_payload,
        reviewed_payload,
        {"e1b_approve_e1_for_next_stage_candidate": True},
        labels_payload=labels_payload(),
    )
    assert summary["e1_row_count"] == 10418
    assert summary["e1c_row_count"] == 10418
    assert summary["one_row_per_e1_belief_row"] is True
    assert summary["e1b_reviewed_decisions_valid"] is True
    assert summary["e1c_safe_for_step1f_candidate"] is True
    assert summary["production_ready"] is False
    assert summary["forbidden_keys_present"] == []


def test_e1c_safety_fails_when_e1b_gate_is_false() -> None:
    rows = [e1_row(0, "goalkeeper_like_unknown_team_context")]
    rows.extend(e1_row(index, "outfield_player_like_not_goalkeeper") for index in range(1, 10418))
    e1_payload, e1c_payload, audit_payload, candidate_payload, reviewed_payload = corrected_payloads(rows)
    summary = build_e1c_eval_summary(
        e1_payload,
        e1c_payload,
        audit_payload,
        candidate_payload,
        reviewed_payload,
        {"e1b_approve_e1_for_next_stage_candidate": False},
        labels_payload=labels_payload(),
    )
    assert summary["e1c_safe_for_step1f_candidate"] is False
    assert "e1b_review_gate_not_passed" in summary["e1c_safety_missing_reasons"]
