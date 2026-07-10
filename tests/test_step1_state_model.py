from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING, visual_stamp  # noqa: E402
from football_intelligence.step1_visual_reconstruction.state_model import (  # noqa: E402
    build_person_states_payload,
    renderable_visible_rows,
)


def candidate_row(det_id: str, candidate_type: str, quality: float, x1: float = 10.0) -> dict[str, Any]:
    return visual_stamp(
        {
            "frame_id": "frame_001",
            "frame_sequence": 1,
            "timestamp_seconds": 1.0,
            "frame_file": "frame_001.jpg",
            "detection_id": det_id,
            "source_detection_id": f"source_{det_id}",
            "bbox": {"x1": x1, "y1": 20.0, "x2": x1 + 20.0, "y2": 80.0},
            "footpoint": {"x": x1 + 10.0, "y": 80.0, "method": "bbox_bottom_center", "confidence": 0.85},
            "candidate_type": candidate_type,
            "bbox_confidence": 0.8,
            "bbox_quality_score": quality,
            "bbox_quality_reason": "small_bbox" if quality < 0.66 else "bbox_plausible",
            "crop_quality": None,
            "roi_status": "inside_or_unverified_visual_roi",
            "duplicate_group_id": "group",
            "duplicate_action": "unique",
            "state": "unknown",
            "confidence": 0.0,
            "reason": "fixture",
        }
    )


def candidate_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "artifact": "step1_person_candidates",
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "project_wide_defaults_changed": False,
        "stage3d_registries_changed": False,
        "rows": rows,
        "frames": [
            {
                "frame_id": "frame_001",
                "frame_sequence": 1,
                "timestamp_seconds": 1.0,
                "frame_file": "frame_001.jpg",
            }
        ],
    }


def test_partial_candidates_are_not_dropped() -> None:
    payload = build_person_states_payload(candidate_payload([candidate_row("partial", "player_candidate_source", 0.5)]))
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["state"] == "observed_partial"
    assert payload["rows"][0]["observed_visible_candidate"] is True


def test_unknown_candidates_are_not_forced_to_player_or_team_roles() -> None:
    payload = build_person_states_payload(
        candidate_payload([candidate_row("unknown", "unknown_candidate_source", 0.82)])
    )
    item = payload["rows"][0]
    assert item["candidate_type"] == "unknown_candidate_source"
    assert item["state"] == "observed_partial"
    assert "player_slot_id" not in item
    assert "slot_id" not in item
    assert "team_id" not in item


def test_only_observed_clear_and_partial_render_as_visible_candidates() -> None:
    payload = build_person_states_payload(
        candidate_payload(
            [
                candidate_row("clear", "player_candidate_source", 0.9, 10.0),
                candidate_row("partial", "player_candidate_source", 0.5, 100.0),
                candidate_row("unknown", "false_positive_candidate", 0.9, 200.0),
            ]
        )
    )
    visible = renderable_visible_rows(payload["rows"])
    assert {row["detection_id"] for row in visible} == {"clear", "partial"}
