from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.team_colour_eval import build_eval_summary  # noqa: E402


def test_colour_eval_reports_unavailable_when_gold_colour_fields_are_missing() -> None:
    labels_payload = {
        "frames": [
            {
                "frame_id": "frame_001",
                "frame_sequence": 1,
                "timestamp_seconds": 1.0,
                "labels_complete": True,
                "persons": [{"bbox": {"x1": 1, "y1": 1, "x2": 2, "y2": 3}}],
            }
        ]
    }
    summary, _issues = build_eval_summary(
        {"rows": []},
        {"rows": [], "summary": {"b4_visible_person_base_rows": 0}},
        b4_summary={"b4_total_visible_person_base_rows": 0},
        labels_payload=labels_payload,
    )
    assert summary["gold8_colour_eval_available"] is False
    assert summary["identity_tracking_performed"] is False
    assert summary["player_slots_assigned"] is False
    assert summary["expected_22_role_states_created"] is False
    assert summary["goalkeeper_classification_performed"] is False
    assert summary["official_specialist_exclusion_performed"] is False
