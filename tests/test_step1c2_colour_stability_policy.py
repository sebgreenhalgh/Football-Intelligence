from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_policy import (  # noqa: E402
    build_colour_stability_payloads,
)


def seed_row(
    base_id: str,
    belief: str,
    *,
    candidate_type: str = "player_candidate_source",
    roi_status: str = "inside_or_unverified_visual_roi",
) -> dict:
    return {
        "visible_person_base_id": base_id,
        "frame_id": "frame_001",
        "frame_sequence": 1,
        "timestamp_seconds": 1.0,
        "detection_id": f"det_{base_id}",
        "source_detection_id": f"source_{base_id}",
        "bbox": {"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 40.0},
        "footpoint": {"x": 15.0, "y": 40.0, "method": "bbox_bottom_center", "confidence": 0.9},
        "state": "observed_clear",
        "roi_status": roi_status,
        "candidate_type": candidate_type,
        "original_role_source": "player",
        "crop_quality": "medium",
        "crop_quality_reason": "usable_torso_crop",
        "seed_team_colour_belief": belief,
        "seed_team_colour_belief_confidence": 0.7,
        "seed_team_colour_belief_state": "ambiguous_seeded_colour",
        "seed_team_colour_belief_reason": "test",
    }


def group(base_ids: list[str], counts: dict[str, int], dominant: str = "team_1_outfield_colour_like") -> dict:
    return {
        "short_burst_colour_group_id": "step1c2_sbcg_f000001_000",
        "visible_person_base_ids": base_ids,
        "dominant_seed_team_colour_belief": dominant,
        "dominant_belief_count": counts.get(dominant, 0),
        "group_belief_counts": counts,
        "group_frame_count": 2,
        "group_row_count": sum(counts.values()),
        "group_confidence": 0.8,
    }


def test_one_stability_row_per_seeded_belief_row() -> None:
    stability, flip = build_colour_stability_payloads(
        {"rows": [seed_row("a", "unknown_ambiguous_colour"), seed_row("b", "team_1_outfield_colour_like")]},
        {"rows": [group(["a", "b"], {"team_1_outfield_colour_like": 2})]},
    )
    assert stability["summary"]["c2_stability_row_count"] == 2
    assert stability["summary"]["one_row_per_c1c_seeded_belief_row"] is True
    assert flip["summary"]["flip_audit_row_count"] == 2


def test_context_and_other_distinct_are_not_forced_to_team_labels() -> None:
    stability, _flip = build_colour_stability_payloads(
        {
            "rows": [
                seed_row(
                    "context",
                    "non_outfield_context_colour",
                    candidate_type="unknown_candidate_source",
                    roi_status="outside_playing_roi",
                ),
                seed_row("other", "other_distinct_colour_like"),
            ]
        },
        {"rows": [group(["context", "other"], {"team_1_outfield_colour_like": 2})]},
    )
    rows = {row["visible_person_base_id"]: row for row in stability["rows"]}
    assert rows["context"]["c2_stable_colour_belief"] == "non_outfield_context_colour"
    assert rows["context"]["c2_stability_action"] == "retained_context_or_other_distinct"
    assert rows["other"]["c2_stable_colour_belief"] == "other_distinct_colour_like"
    assert stability["summary"]["context_offroi_forced_to_team_count"] == 0


def test_team_colour_conflicts_become_review_required_without_flip() -> None:
    stability, _flip = build_colour_stability_payloads(
        {"rows": [seed_row("a", "team_1_outfield_colour_like"), seed_row("b", "team_2_outfield_colour_like")]},
        {
            "rows": [
                group(
                    ["a", "b"],
                    {"team_1_outfield_colour_like": 1, "team_2_outfield_colour_like": 1},
                    dominant="team_1_outfield_colour_like",
                )
            ]
        },
    )
    rows = {row["visible_person_base_id"]: row for row in stability["rows"]}
    assert rows["a"]["c2_stable_colour_belief"] == "team_1_outfield_colour_like"
    assert rows["b"]["c2_stable_colour_belief"] == "team_2_outfield_colour_like"
    assert rows["a"]["c2_review_required"] is True
    assert rows["b"]["c2_review_required"] is True
