from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.count_policy import (  # noqa: E402
    build_count_policy_payload,
    count_policy_for_row,
)
from football_intelligence.step1_visual_reconstruction.schema import visual_stamp  # noqa: E402


def row(
    det_id: str,
    action: str,
    *,
    candidate_type: str = "player_candidate_source",
    state: str = "observed_clear",
) -> dict[str, Any]:
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
            "bbox_quality_score": 0.8,
            "bbox_quality_reason": "bbox_plausible",
            "crop_quality": None,
            "roi_status": "inside_or_unverified_visual_roi",
            "duplicate_group_id": "group",
            "duplicate_action": "unique",
            "state": state,
            "confidence": 0.8,
            "reason": "fixture",
            "qa_render_tier": "primary_observed",
            "issue_flags": [],
            "reconciliation_action": action,
            "review_required": False,
        }
    )


def test_duplicate_and_source_overlap_shadows_are_not_counted() -> None:
    duplicate_counted, duplicate_reason = count_policy_for_row(row("det_dup", "duplicate_shadow_candidate"))
    source_counted, source_reason = count_policy_for_row(row("det_source", "source_overlap_shadow_candidate"))
    assert duplicate_counted is False
    assert duplicate_reason == "duplicate_shadow_candidate_not_counted"
    assert source_counted is False
    assert source_reason == "source_overlap_shadow_candidate_not_counted"


def test_retained_overlaps_remain_counted() -> None:
    counted, reason = count_policy_for_row(row("det_overlap", "retained_overlap_candidate"))
    assert counted is True
    assert reason == "retained_overlap_candidate_counted"


def test_context_candidates_are_retained_and_not_forced_to_player_or_team_roles() -> None:
    payload = build_count_policy_payload(
        {
            "rows": [
                row("det_ref", "context_observation_candidate", candidate_type="referee_candidate_source"),
                row("det_staff", "context_observation_candidate", candidate_type="staff_context_candidate_source"),
                row("det_unknown", "context_observation_candidate", candidate_type="unknown_candidate_source"),
            ]
        }
    )
    assert len(payload["rows"]) == 3
    assert {item["candidate_type"] for item in payload["rows"]} == {
        "referee_candidate_source",
        "staff_context_candidate_source",
        "unknown_candidate_source",
    }
    assert all("team" not in item for item in payload["rows"])
    assert all(item["count_policy_presentation_only_or_sandbox"] is True for item in payload["rows"])


def test_unknown_state_is_not_counted_even_when_action_is_primary() -> None:
    counted, reason = count_policy_for_row(row("det_unknown_state", "primary_observation_candidate", state="unknown"))
    assert counted is False
    assert reason == "state_not_observed_visible"
