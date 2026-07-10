from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.render_tiers import (  # noqa: E402
    build_render_tier_payload,
    qa_render_tier,
)
from football_intelligence.step1_visual_reconstruction.schema import visual_stamp  # noqa: E402


def row(det_id: str, state: str, quality: float, candidate_type: str = "player_candidate_source") -> dict[str, Any]:
    return visual_stamp(
        {
            "frame_id": "frame_001",
            "frame_sequence": 1,
            "timestamp_seconds": 1.0,
            "frame_file": "frame_001.jpg",
            "detection_id": det_id,
            "source_detection_id": f"source_{det_id}",
            "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 80.0},
            "footpoint": {"x": 20.0, "y": 80.0, "method": "bbox_bottom_center", "confidence": 0.85},
            "candidate_type": candidate_type,
            "bbox_confidence": 0.8,
            "bbox_quality_score": quality,
            "bbox_quality_reason": "bbox_plausible",
            "crop_quality": None,
            "roi_status": "inside_or_unverified_visual_roi",
            "duplicate_group_id": "group",
            "duplicate_action": "unique",
            "state": state,
            "confidence": 0.8,
            "reason": "fixture",
            "observed_visible_candidate": state in {"observed_clear", "observed_partial"},
        }
    )


def state_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"rows": rows}


def test_qa_render_tier_does_not_alter_state_or_observed_visibility() -> None:
    source = row("det_1", "observed_clear", 0.9)
    tier = qa_render_tier(source)
    payload = build_render_tier_payload(state_payload([source]), [])
    out = payload["rows"][0]
    assert tier == "primary_observed"
    assert out["state"] == source["state"]
    assert out["observed_visible_candidate"] is True
    assert out["qa_render_tier_presentation_only"] is True


def test_unknown_hidden_by_default_is_not_counted_as_observed_visible() -> None:
    payload = build_render_tier_payload(state_payload([row("det_unknown", "unknown", 0.1)]), [])
    out = payload["rows"][0]
    assert out["qa_render_tier"] == "unknown_hidden_by_default"
    assert out["observed_visible_candidate"] is False


def test_observed_clear_and_partial_remain_observed_visible() -> None:
    payload = build_render_tier_payload(
        state_payload([row("det_clear", "observed_clear", 0.8), row("det_partial", "observed_partial", 0.5)]),
        [],
    )
    assert {item["state"] for item in payload["rows"]} == {"observed_clear", "observed_partial"}
    assert all(item["observed_visible_candidate"] is True for item in payload["rows"])
