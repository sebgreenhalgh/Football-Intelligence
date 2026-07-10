from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_review_candidates import (  # noqa: E402
    build_colour_stability_review_candidate_payload,
)


def stability_row(base_id: str, frame: int, c1c: str, c2: str, *, review_required: bool = False) -> dict:
    return {
        "visible_person_base_id": base_id,
        "frame_sequence": frame,
        "timestamp_seconds": float(frame),
        "detection_id": f"det_{base_id}",
        "source_detection_id": f"source_{base_id}",
        "bbox": {"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 40.0},
        "footpoint": {"x": 15.0, "y": 40.0, "method": "bbox_bottom_center", "confidence": 0.9},
        "state": "observed_clear",
        "roi_status": "inside_or_unverified_visual_roi",
        "candidate_type": "player_candidate_source",
        "original_role_source": "player",
        "crop_quality": "medium",
        "crop_quality_reason": "usable",
        "torso_crop_bbox": {"x1": 11.0, "y1": 12.0, "x2": 19.0, "y2": 30.0},
        "c1c_seed_team_colour_belief": c1c,
        "c1c_seed_team_colour_belief_confidence": 0.7,
        "c2_stable_colour_belief": c2,
        "c2_stable_colour_belief_confidence": 0.8,
        "c2_stability_action": "retained_seeded_belief",
        "c2_stability_reason": "test",
        "c2_review_required": review_required,
        "short_burst_colour_group_id": "step1c2_sbcg_f000001_000",
        "group_belief_counts": {c2: 2},
    }


def flip_row(base_id: str, frame: int, flip_type: str) -> dict:
    return {
        "visible_person_base_id": base_id,
        "frame_sequence": frame,
        "flip_type": flip_type,
        "flip_reason": "test",
    }


def test_candidate_selection_includes_required_sets_and_deduplicates() -> None:
    rows = [
        stability_row("changed", 10, "unknown_ambiguous_colour", "team_1_outfield_colour_like"),
        stability_row("review", 11, "team_1_outfield_colour_like", "team_1_outfield_colour_like", review_required=True),
        stability_row("frame59", 59, "team_1_outfield_colour_like", "team_1_outfield_colour_like"),
        stability_row("unknown_to_team", 12, "unknown_ambiguous_colour", "team_2_outfield_colour_like"),
        stability_row("dedupe", 60, "unknown_ambiguous_colour", "team_1_outfield_colour_like", review_required=True),
    ]
    flips = [
        flip_row("changed", 10, "unknown_to_team_colour"),
        flip_row("review", 11, "review_required_conflict"),
        flip_row("frame59", 59, "retained_no_flip"),
        flip_row("unknown_to_team", 12, "unknown_to_team_colour"),
        flip_row("dedupe", 60, "unknown_to_team_colour"),
    ]
    payload = build_colour_stability_review_candidate_payload(
        {"rows": rows},
        {"rows": flips},
        eval_summary={"c2_safe_for_human_review": True},
    )
    selected = {row["visible_person_base_id"]: row for row in payload["rows"]}
    assert {"changed", "review", "frame59", "unknown_to_team", "dedupe"}.issubset(selected)
    assert len(selected) == len(payload["rows"])
    assert "changed_by_c2" in selected["changed"]["review_reason_tags"]
    assert "c2_review_required" in selected["review"]["review_reason_tags"]
    assert "frame_59_62_manual_followup" in selected["frame59"]["review_reason_tags"]
    assert "unknown_to_team_colour" in selected["unknown_to_team"]["review_reason_tags"]
