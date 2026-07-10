from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_groups import (  # noqa: E402
    MAX_GROUP_SPAN_FRAMES,
    build_short_burst_colour_group_payload,
)


def row(base_id: str, frame_sequence: int, x: float = 100.0, belief: str = "team_1_outfield_colour_like") -> dict:
    return {
        "visible_person_base_id": base_id,
        "frame_id": f"frame_{frame_sequence}",
        "frame_sequence": frame_sequence,
        "timestamp_seconds": float(frame_sequence),
        "detection_id": f"det_{base_id}",
        "source_detection_id": f"source_{base_id}",
        "bbox": {"x1": x, "y1": 100.0, "x2": x + 20.0, "y2": 160.0},
        "footpoint": {"x": x + 10.0, "y": 160.0, "method": "bbox_bottom_center", "confidence": 0.9},
        "seed_team_colour_belief": belief,
        "seed_team_colour_belief_confidence": 0.9,
        "seed_review_required": False,
        "review_required": False,
    }


def test_groups_use_only_local_adjacent_visual_evidence() -> None:
    payload = build_short_burst_colour_group_payload(
        {
            "rows": [
                row("a", 1),
                row("b", 2, x=104.0),
                row("far", 2, x=500.0),
                row("late", 5, x=108.0),
            ]
        }
    )
    groups = payload["rows"]
    linked = [group for group in groups if set(group["visible_person_base_ids"]) == {"a", "b"}]
    assert linked
    assert all("pitch_x_metric" not in group for group in groups)
    assert all(group["local_group_not_identity"] is True for group in groups)


def test_max_group_span_is_respected_and_ids_are_temporary() -> None:
    payload = build_short_burst_colour_group_payload({"rows": [row(f"r{i}", i, x=100.0 + i) for i in range(1, 10)]})
    assert max(group["group_frame_count"] for group in payload["rows"]) <= MAX_GROUP_SPAN_FRAMES
    assert all(group["short_burst_colour_group_id"].startswith("step1c2_sbcg_f") for group in payload["rows"])
    assert payload["identity_tracking_performed"] is False
    assert payload["player_slots_assigned"] is False
