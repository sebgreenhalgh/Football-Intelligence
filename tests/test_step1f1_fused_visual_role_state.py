from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state import (  # noqa: E402
    build_f1_row,
    build_fused_visual_role_state_payloads,
)


def row(
    index: int,
    *,
    c2c: str = "unknown_ambiguous_colour",
    d1c: str = "unknown_official_context",
    e1c: str = "unknown_goalkeeper_context",
    c2c_bad: bool = False,
    d1c_bad: bool = False,
    e1c_bad: bool = False,
    review_required: bool = False,
) -> dict:
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
        "c2c_final_colour_belief": c2c,
        "c2c_final_colour_belief_confidence": 0.8,
        "c2c_colour_source": "c2c_human_corrected",
        "c2c_human_reviewed": True,
        "c2c_bad_detection_or_not_person": c2c_bad,
        "c2c_review_required": review_required,
        "d1c_final_official_context_belief": d1c,
        "d1c_final_official_context_belief_confidence": 0.82,
        "d1c_context_source": "d1c_human_corrected",
        "d1c_human_reviewed": True,
        "d1c_bad_detection_or_not_person": d1c_bad,
        "d1c_official_like_visual_context": d1c == "official_referee_like",
        "d1c_assistant_or_line_official_like_visual_context": d1c == "assistant_or_line_official_like",
        "d1c_review_required": review_required,
        "e1c_final_goalkeeper_context_belief": e1c,
        "e1c_final_goalkeeper_context_belief_confidence": 0.84,
        "e1c_context_source": "e1c_human_corrected",
        "e1c_human_reviewed": True,
        "e1c_bad_detection_or_not_person": e1c_bad,
        "e1c_review_required": review_required,
        "retained_for_future_player_team_review": True,
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
        "production_ready": False,
    }


def payload(rows: list[dict]) -> dict:
    return {"rows": rows}


def test_output_row_count_preserved_at_10418_and_retention() -> None:
    rows = [
        row(index, c2c="team_1_outfield_colour_like", e1c="outfield_player_like_not_goalkeeper")
        for index in range(10418)
    ]
    fused, conflict = build_fused_visual_role_state_payloads(payload(rows), payload(rows), payload(rows), payload(rows))
    assert len(fused["rows"]) == 10418
    assert fused["summary"]["one_row_per_e1c_row"] is True
    assert fused["summary"]["input_visible_person_base_ids_aligned"] is True
    assert all(out["retained_for_future_player_team_review"] is True for out in fused["rows"])
    assert isinstance(conflict["rows"], list)


def test_fusion_priority_bad_detection_goalkeeper_official_and_outfield() -> None:
    cases = [
        (
            row(1, c2c_bad=True, c2c="team_1_outfield_colour_like", e1c="goalkeeper_like_team_1_context"),
            "bad_detection_or_not_person",
        ),
        (row(2, e1c="goalkeeper_like_team_1_context"), "team_1_goalkeeper_visual_context"),
        (row(3, e1c="goalkeeper_like_team_2_context"), "team_2_goalkeeper_visual_context"),
        (row(4, e1c="goalkeeper_like_unknown_team_context"), "goalkeeper_unknown_team_visual_context"),
        (
            row(5, d1c="official_referee_like", e1c="outfield_player_like_not_goalkeeper"),
            "official_referee_visual_context",
        ),
        (
            row(6, d1c="assistant_or_line_official_like", e1c="outfield_player_like_not_goalkeeper"),
            "assistant_or_line_official_visual_context",
        ),
        (
            row(7, d1c="off_pitch_context_person_like", e1c="outfield_player_like_not_goalkeeper"),
            "off_pitch_context_person_visual_context",
        ),
        (
            row(8, c2c="team_1_outfield_colour_like", e1c="outfield_player_like_not_goalkeeper"),
            "team_1_outfield_visual_context",
        ),
        (
            row(9, c2c="team_2_outfield_colour_like", e1c="outfield_player_like_not_goalkeeper"),
            "team_2_outfield_visual_context",
        ),
        (
            row(10, c2c="other_distinct_colour_like", e1c="outfield_player_like_not_goalkeeper"),
            "team_unknown_outfield_visual_context",
        ),
        (row(11), "unknown_visible_person_visual_context"),
    ]
    for source, expected in cases:
        out = build_f1_row(source)
        assert out["step1f1_fused_visual_role_state"] == expected
        assert out["eligible_for_identity_tracking"] is False
        assert out["eligible_for_player_slot_assignment"] is False
        assert out["eligible_for_goalkeeper_slot_assignment"] is False
        assert out["eligible_for_metric_use"] is False


def test_conflict_audit_flags_are_warnings_only() -> None:
    cases = [
        row(20, c2c="team_1_outfield_colour_like", e1c="goalkeeper_like_team_2_context"),
        row(21, d1c="official_referee_like", e1c="goalkeeper_like_team_1_context"),
        row(
            22,
            d1c="official_referee_like",
            c2c="team_1_outfield_colour_like",
            e1c="outfield_player_like_not_goalkeeper",
        ),
        row(23, d1c_bad=True, c2c="team_2_outfield_colour_like", e1c="outfield_player_like_not_goalkeeper"),
        row(24, c2c="non_outfield_context_colour", e1c="goalkeeper_like_team_1_context"),
        row(25, e1c="unknown_goalkeeper_context", c2c="ambiguous_outfield_colour", d1c="unknown_official_context"),
        row(26, c2c="team_1_outfield_colour_like", e1c="outfield_player_like_not_goalkeeper", review_required=True),
    ]
    fused, conflict = build_fused_visual_role_state_payloads(
        payload(cases),
        payload(cases),
        payload(cases),
        payload(cases),
    )
    assert len(conflict["rows"]) == len(cases)
    assert all(audit["conflict_is_visual_qa_warning_only"] is True for audit in conflict["rows"])
    assert all(audit["retained_for_future_player_team_review"] is True for audit in conflict["rows"])
    assert all(row["step1f1_review_required"] is True for row in fused["rows"])
