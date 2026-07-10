# ruff: noqa: E501

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.nodes import build_node_payload  # noqa: E402


def f3_row(
    index: int,
    *,
    frame_sequence: int | None = None,
    role: str = "team_1_outfield_visual_context",
    colour: str = "team_1_outfield_colour_like",
    candidate_type: str = "player_candidate_source",
    warnings: list[str] | None = None,
    crop_quality: str = "good",
) -> dict:
    frame = index if frame_sequence is None else frame_sequence
    x1 = 100.0 + (index % 5) * 8.0
    return {
        "visible_person_base_id": f"base_{index:03d}",
        "frame_id": f"frame_{frame}",
        "frame_sequence": frame,
        "timestamp_seconds": frame / 10.0,
        "detection_id": f"det_{index:03d}",
        "source_detection_id": f"source_det_{index:03d}",
        "bbox": {"x1": x1, "y1": 100.0, "x2": x1 + 22.0, "y2": 170.0},
        "footpoint": {"x": x1 + 11.0, "y": 170.0, "method": "bbox_bottom_center", "confidence": 0.9},
        "state": "observed_clear",
        "crop_quality": crop_quality,
        "candidate_type": candidate_type,
        "roi_status": "inside_or_unverified_visual_roi",
        "c2c_final_colour_belief": colour,
        "c2c_colour_source": "c2c_synthetic",
        "d1c_final_official_context_belief": "player_like_not_official_context",
        "d1c_context_source": "d1c_synthetic",
        "e1c_final_goalkeeper_context_belief": "outfield_player_like_not_goalkeeper",
        "e1c_context_source": "e1c_synthetic",
        "step1f1_fused_visual_role_state": role,
        "step1f1_warning_flags": [],
        "step1f1_conflict_flags": [],
        "step1f3_final_visual_role_state": role,
        "step1f3_final_visual_role_group": "player_outfield_visual_context",
        "step1f3_role_team_context": (
            "team_1_visual_context"
            if "team_1" in role
            else "team_2_visual_context"
            if "team_2" in role
            else "unknown_visual_context"
        ),
        "step1f3_warning_flags": warnings or [],
        "step1f3_review_required": bool(warnings),
        "retained_for_future_player_team_review": True,
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
        "production_ready": False,
    }


def synthetic_f3_payload() -> dict:
    rows = [
        f3_row(0, frame_sequence=0),
        f3_row(1, frame_sequence=1, role="team_unknown_outfield_visual_context", colour="unknown_ambiguous_colour_like", warnings=["ambiguous_team_colour"]),
        f3_row(2, frame_sequence=2, role="official_referee_visual_context", candidate_type="official_candidate_source", warnings=["official_context_warning"]),
        f3_row(3, frame_sequence=3, role="bad_detection_or_not_person", candidate_type="false_positive_candidate", warnings=["bad_detection_proxy"]),
    ]
    return {"artifact": "step1f3_human_corrected_fused_visual_role_state_rows", "rows": rows}


def test_node_builder_preserves_row_count_alignment_and_retains_ambiguous_official_bad_rows() -> None:
    f3_payload = synthetic_f3_payload()
    node_payload = build_node_payload(f3_payload, {"step1g1_safe_for_step2_visual_continuity_candidate": True})
    assert len(node_payload["rows"]) == len(f3_payload["rows"])
    assert [row["visible_person_base_id"] for row in node_payload["rows"]] == [
        row["visible_person_base_id"] for row in f3_payload["rows"]
    ]
    roles = {row["step1f3_final_visual_role_state"] for row in node_payload["rows"]}
    assert "team_unknown_outfield_visual_context" in roles
    assert "official_referee_visual_context" in roles
    assert "bad_detection_or_not_person" in roles
    assert all(row["retained_for_future_player_team_review"] is True for row in node_payload["rows"])
    assert node_payload["one_node_per_f3_row"] is True
    assert node_payload["production_ready"] is False
